import ast

from functools import lru_cache
from typing import TypedDict, Literal

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.agent.llm import planner
from app.agent.plan import PlanError, validate
from app.services import tools as tool_service
from app.tools import UnknownTool, label_of

_SYSTEM = """你是电商图片修图助手，通过调用工具完成用户的修图请求。

规则：
- 只能使用已提供的工具。可以一次安排多步，按完成先后调用；后一步默认依赖前一步。
  用户一句话里出现的每个动作，都必须在本次回复中一次性输出全部工具调用。
- 能一步完成的不要拆成多步，最多 8 步。
  禁止只输出第一个动作然后等用户补充，
  也禁止用「先做第一步，后面再说」的方式分轮完成同一句话。
- 缺失参数用画布信息与常识补齐，可推断的参数不要反问用户。
- 图层只是内部实现，不要求用户理解。layer_id 只能从当前画布摘要明确列出的 id 或名称中选择；
  禁止把「人物」「主体」「衣服」「背景」等画面语义编成图层名。未明确点名已存在图层时省略 layer_id。
- 画布摘要标明已有选区，且用户要求严格只改选区时，用 replace_region 或 erase_region，
  不要填写 layer_id，由服务端根据选区确定目标。不要再让用户重选，也不要猜测选区是否点准。
- 改颜色、换材质、换衣服、换人物、换物体、改图中文字或其他生成式修改，默认用 smart_edit，
  把用户的完整意思写进 instruction，不要拆层，不要填写任何图层参数。
- 已有选区且必须保留选区外像素时才优先 replace_region；只改整张背景可用 replace_background。
  adjust_image 只做明暗冷暖，不做改色。
- 翻转画面用 flip_layer；说了角度用 rotate_layer。
- 用户要求拆层、把物体独立成层时，使用 split_layers 或 promote_object_to_layer。
- 拆层默认不拆文字，只有用户明确要求时才传 include_text。
- 只有画布摘要已经列出文字图层时，才用 set_layer_text 修改其文案、字号或颜色；
  普通图片里烘焙的文字没有文字图层，直接用 smart_edit，不要先拆层。
- 翻转、缩放、裁剪、旋转、移动这类简单操作与 smart_edit 同时出现时，两者都要进计划，
  顺序按用户表述的先后。smart_edit、replace_background、expand_canvas、upscale_image
  会拍平画布并重建为单层底图，这类步骤排在最后，其后不要再依赖具体图层。
- 出营销图、主图、场景图、模特图、海报用 generate_marketing，
  kind 分别为 product / scene / model / poster。结果只进图片墙。
- 改尺寸、出投放物料、出 1:1 / 4:5 / 9:16 用 prepare_delivery_sizes，
  不要裁切主体，也不要用 crop_canvas。
- 对多张图做同一套去背景、换背景、调色、超分、扩图或改尺寸时用 batch_process。
  单张精修、营销图、局部编辑和拆层不要用它。
- 调用工具时不要输出解释或工具名。只有指令与修图无关、或现有工具确实做不到时，
  才用一句中文说明，不要提内部参数。

当前画布：{context}"""

_FALLBACK_REPLY = "没太理解这条指令，换个说法或说得更具体一些。"
_REFUSAL_LIMIT = 60

_CONNECTORS = ("再", "然后", "接着", "并把", "同时", "最后", "之后", "顺便")
_MAX_PLAN_TRIES = 2

class AgentState(TypedDict):
    goal: str
    context: str
    plan: list[dict]
    reply: str
    retries: int

def _calls_from_text(text: str) -> list[dict]:
    """兼容模型偶尔写出的 tool(arg=...) 文本；只解析字面量，绝不 eval。"""
    source = text.strip()

    # 兼容模型用 Markdown 代码块包住工具调用的情况
    if source.startswith("```") and source.endswith("```"):
        lines = source.splitlines()
        source = "\n".join(lines[1:-1]).strip()

    if not source:
        return []

    try:
        tree = ast.parse(source, mode="exec")
    except SyntaxError:
        return []

    calls: list[dict] = []

    for statement in tree.body:
        if (
            not isinstance(statement, ast.Expr)
            or not isinstance(statement.value, ast.Call)
        ):
            return []

        call = statement.value

        # 只接受 smart_edit(...) 这种简单形式
        if not isinstance(call.func, ast.Name) or call.args:
            return []

        params = {}

        for keyword in call.keywords:
            if keyword.arg is None:
                return []

            try:
                # 只允许字符串、数字、列表、字典等字面量
                params[keyword.arg] = ast.literal_eval(keyword.value)
            except (ValueError, TypeError):
                return []

        calls.append(
            {
                "tool": call.func.id,
                "params": params,
            }
        )

    return calls

async def _plan(state: AgentState) -> AgentState:
    messages = [
        SystemMessage(_SYSTEM.format(context=state["context"])),
        HumanMessage(state["goal"]),
    ]
    # temperature=0，重试必须换个输入，否则得到一模一样的单个调用
    if state["retries"]:
        messages.append(AIMessage(state["reply"] or "（上一轮只给出了一个步骤）"))
        messages.append(
            HumanMessage(
                "上一轮漏掉了后续动作。这句话包含多个动作，"
                "请在同一次回复中输出全部工具调用，不要只给第一步。"
            )
        )

    message = await planner().ainvoke(messages)

    reply = _text_of(message)

    # 正常情况：模型返回结构化工具调用
    plan = [
        {
            "tool": call["name"],
            "params": call["args"],
        }
        for call in message.tool_calls
    ]

    # 兜底：模型把工具调用写成了普通文本
    if not plan:
        plan = _calls_from_text(reply)
        if plan:
            # 不再把 smart_edit(...) 原始文本展示给用户
            reply = ""

    return {
        "plan": plan,
        "reply": reply,
        "retries": state["retries"] + 1,
    }



def _verify(state: AgentState) -> AgentState:
    """模型给出的计划一律经服务端校验，不可直接执行。"""
    try:
        return {"plan": validate(state["plan"])}
    except (UnknownTool, tool_service.InvalidParams, PlanError) as exc:
        return {"plan": [], "reply": f"这一步暂时执行不了：{exc}"}

def _route(state: AgentState) -> Literal["plan", "__end__"]:
    """只规划出 1 步、但原话明显是多步 → 回到 plan 重来一次。"""
    under_covered = bool(state["plan"]) and len(state["plan"]) < 2
    if under_covered and state["retries"] < _MAX_PLAN_TRIES:
        return "plan"
    return END


@lru_cache
def _graph():
    builder = StateGraph(AgentState)
    builder.add_node("plan", _plan)
    builder.add_node("verify", _verify)

    builder.add_edge(START, "plan")
    builder.add_edge("plan", "verify")
    builder.add_conditional_edges("verify", _route, {"plan": "plan", END: END})
    return builder.compile()


async def run(goal: str, context: str) -> tuple[str, list[dict]]:
    """规划并校验一轮指令，返回答复与尚未下发的计划。"""
    state = await _graph().ainvoke(
        {"goal": goal, "context": context, "plan": [], "reply": "", "retries": 0}
    )
    return spoken(state["reply"], state["plan"]), state["plan"]


def spoken(reply: str, plan: list[dict]) -> str:
    """有计划就按步骤说话；没计划才用兜底或截成一句拒绝。"""
    text = (reply or "").strip()
    if plan:
        if text and text != _FALLBACK_REPLY:
            return text
        labels = "、".join(label_of(step["tool"], step.get("params")) for step in plan)
        if len(plan) > 1:
            return f"将按以下步骤执行：{labels}。确认后开始。"
        return f"好，正在{labels}。"
    return _one_sentence(text) if text else _FALLBACK_REPLY


def _one_sentence(text: str) -> str:
    for sep in ("。", "！", "？", "\n"):
        head, found, _ = text.partition(sep)
        if found and head.strip():
            return head.strip() + (sep if sep != "\n" else "。")
    if len(text) > _REFUSAL_LIMIT:
        return text[:_REFUSAL_LIMIT].rstrip() + "…"
    return text


def _text_of(message: AIMessage) -> str:
    if isinstance(message.content, str):
        return message.content.strip()
    return "".join(
        block.get("text", "") for block in message.content if isinstance(block, dict)
    ).strip()
