"""指令分流与修图规划图。无关提问走 chat，修图请求才进 plan。"""

import ast
import logging
import re
from collections.abc import Awaitable, Callable
from functools import lru_cache
from typing import Literal, TypedDict

from langchain_core.messages import AIMessage, AIMessageChunk, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.agent.chat import answer
from app.agent.llm import planner
from app.agent.plan import PlanError, validate
from app.agent.triage import CHAT, EDIT, classify
from app.agent.words import plain, text_of
from app.services import tools as tool_service
from app.tools import SPECS, UnknownTool, label_of

logger = logging.getLogger(__name__)

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
- 只能通过 function call 表达要做什么。回复文字里禁止出现任何工具名或参数名，
  禁止用「smart_edit 指令为…」这类句子描述你打算怎么调用。
- 需要说话时只说一句给用户听的中文，不解释、不复述、不确认参数。
  与修图无关的提问已经在上游分流到闲聊通道，送到你这里的都是修图请求；
  现有工具确实做不到时，用一句中文说明做不到，不要推荐别的软件，也不要扯开话题。

当前画布：{context}"""

_FALLBACK_REPLY = "没太理解这条指令，换个说法或说得更具体一些。"

_CONNECTORS = ("再", "然后", "接着", "并把", "同时", "最后", "之后", "顺便")
_MAX_PLAN_TRIES = 2

# 散文式调用里最可能承载指令的参数名，用于兜底抢救
_PROSE_PARAM = {
    "smart_edit": "instruction",
    "replace_region": "prompt",
    "erase_region": "prompt",
    "replace_background": "prompt",
}
# 元话语开头，剥掉后剩下的才是要执行的指令
_META_HEAD = re.compile(r"^\s*(?:指令|参数|内容|要求|说明)\s*(?:为|是|如下)?\s*[:：]?\s*")
# 整段被引号包住时只脱一层，内部引号原样保留
_WRAPPED = (
    ("\u300c", "\u300d"),
    ("\u201c", "\u201d"),
    ('"', '"'),
    ("\u2018", "\u2019"),
    ("'", "'"),
)

# 画布摘要要查库，闲聊通道用不上，所以传取数函数而不是现成的字符串
GetContext = Callable[[], Awaitable[str]]

# 对话事件逐条推给前端：阶段、通道、步骤名、正文增量
Emit = Callable[[dict], Awaitable[None]]

# 伪思考：只说已经确定会发生的事，不额外花模型 token
_STAGES = {
    "triage": "在看你这句话的意思",
    EDIT: "这是个修图需求，正在安排步骤",
    CHAT: "这句和修图无关，直接回答",
    "replan": "这句话里有好几个动作，重新安排",
}


async def _silent(event: dict) -> None:
    """不关心过程时的空回调。"""


class AgentState(TypedDict):
    goal: str
    context: GetContext
    emit: Emit
    kind: str
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


def _mentioned_tools(text: str) -> list[str]:
    """散文里提到了哪几个已注册工具名。"""
    return [spec.name for spec in SPECS if spec.name in text]


def _instruction_of(text: str, name: str) -> str:
    """去掉工具名与「指令为」这类元话语开头，再脱掉最外层引号。"""
    cleaned = text.replace(name, " ").strip()
    cleaned = _META_HEAD.sub("", cleaned).strip()
    cleaned = cleaned.strip("。.!！，,、 \t\r\n")
    for left, right in _WRAPPED:
        if len(cleaned) > 2 and cleaned.startswith(left) and cleaned.endswith(right):
            cleaned = cleaned[len(left) : -len(right)].strip()
            break
    return cleaned


def _calls_from_prose(text: str) -> list[dict]:
    """兜底：模型只用散文描述了一次调用，把这句话还原成真正的参数。

    提到多于一个工具名时不抢救，那是对整条计划的描述，硬塞成单步会做错事。
    """
    mentioned = _mentioned_tools(text)
    if len(mentioned) != 1:
        return []
    name = mentioned[0]
    field = _PROSE_PARAM.get(name)
    # 代码形式归 _calls_from_text 管，这里只处理散文
    if field is None or f"{name}(" in text:
        return []
    instruction = _instruction_of(text, name)
    if len(instruction) < 2:
        return []
    return [{"tool": name, "params": {field: instruction}}]


async def _triage(state: AgentState) -> AgentState:
    """先分通道：无关提问不进规划模型，也就不占工具上下文。"""
    emit = state["emit"]
    await emit({"type": "stage", "text": _STAGES["triage"]})
    kind = await classify(state["goal"])
    # 手动测试时靠这行确认走的是哪条通道
    logger.info("指令分流 kind=%s goal=%s", kind, state["goal"][:40])
    # 通道一确定就推出去，闲聊气泡能早点换成虚线灰底
    await emit({"type": "kind", "value": kind})
    await emit({"type": "stage", "text": _STAGES[kind]})
    return {"kind": kind}


async def _plan(state: AgentState) -> AgentState:
    messages = [
        SystemMessage(_SYSTEM.format(context=await state["context"]())),
        HumanMessage(state["goal"]),
    ]
    # temperature=0，重试必须换个输入，否则得到一模一样的单个调用
    if state["retries"]:
        # 只放占位，绝不把上一轮的散文回填，否则等于示范「用文字描述调用」这种格式
        messages.append(AIMessage("（上一轮只给出了一个步骤）"))
        messages.append(
            HumanMessage(
                "上一轮漏掉了后续动作。这句话包含多个动作，"
                "请把全部动作一次性给出，不要只给第一步。"
            )
        )

    message = AIMessageChunk(content="")
    announced = 0
    emit = state["emit"]
    if state["retries"]:
        await emit({"type": "stage", "text": _STAGES["replan"]})

    async for chunk in planner().astream(messages):
        message = message + chunk
        # 工具名比参数先到，第几步做什么此刻就能报，不用等整条计划成形
        for part in getattr(chunk, "tool_call_chunks", None) or []:
            index = (part.get("index") or 0) + 1
            name = part.get("name")
            if name and index > announced:
                announced = index
                await emit({"type": "plan", "index": index, "label": label_of(name, None)})

    reply = text_of(message)

    # 正常情况：模型返回结构化工具调用
    plan = [
        {
            "tool": call["name"],
            "params": call["args"],
        }
        for call in message.tool_calls
    ]

    # 兜底一：模型把工具调用写成了 tool(arg=...) 代码形式
    if not plan:
        plan = _calls_from_text(reply)
        if plan:
            # 不再把 smart_edit(...) 原始文本展示给用户
            reply = ""

    # 兜底二：模型只写了散文（提到工具名却没真调用）
    if not plan:
        plan = _calls_from_prose(reply)
        if plan:
            reply = ""

    return {
        "plan": plan,
        "reply": reply,
        "retries": state["retries"] + 1,
    }


async def _chat(state: AgentState) -> AgentState:
    """只出一句话，不产计划：这一轮没有步骤，也就没有确认、执行与编辑历史。"""
    body = ""
    async for piece in answer(state["goal"]):
        body += piece
        await state["emit"]({"type": "token", "text": piece})
    return {"kind": CHAT, "reply": body, "plan": []}


def _verify(state: AgentState) -> AgentState:
    """模型给出的计划一律经服务端校验，不可直接执行。"""
    try:
        return {"plan": validate(state["plan"])}
    except (UnknownTool, tool_service.InvalidParams, PlanError) as exc:
        return {"plan": [], "reply": f"这一步暂时执行不了：{exc}"}


def _channel(state: AgentState) -> Literal["plan", "chat"]:
    """分流结果决定这一轮进规划节点还是闲聊节点。"""
    return "chat" if state["kind"] == CHAT else "plan"


def _route(state: AgentState) -> Literal["plan", "__end__"]:
    """只规划出 1 步、但原话明显是多步 → 回到 plan 重来一次。

    必须拿连接词做判据，否则每个单步请求都会白跑两遍模型。
    """
    multi = bool(state["plan"]) and len(state["plan"]) < 2
    if multi and any(word in state["goal"] for word in _CONNECTORS):
        if state["retries"] < _MAX_PLAN_TRIES:
            return "plan"
    return END


@lru_cache
def _graph():
    builder = StateGraph(AgentState)
    builder.add_node("triage", _triage)
    builder.add_node("plan", _plan)
    builder.add_node("verify", _verify)
    builder.add_node("chat", _chat)

    builder.add_edge(START, "triage")
    builder.add_conditional_edges("triage", _channel, {"plan": "plan", "chat": "chat"})
    builder.add_edge("plan", "verify")
    builder.add_conditional_edges("verify", _route, {"plan": "plan", END: END})
    builder.add_edge("chat", END)
    return builder.compile()


async def run(
    goal: str, context: GetContext, emit: Emit | None = None
) -> tuple[str, list[dict], str]:
    """规划一轮指令，返回（答复，尚未下发的计划，走的通道）。事件不接就不推。"""
    state = await _graph().ainvoke(
        {
            "goal": goal,
            "context": context,
            "emit": emit or _silent,
            "kind": EDIT,
            "plan": [],
            "reply": "",
            "retries": 0,
        }
    )
    if state["kind"] == CHAT:
        # 闲聊的话术已经组织好了，不再走步骤标签兜底
        return state["reply"], [], CHAT
    return spoken(state["reply"], state["plan"]), state["plan"], EDIT


def spoken(reply: str, plan: list[dict]) -> str:
    """有计划就按步骤说话；没计划才用兜底或截成一句拒绝。"""
    text = plain(reply)
    if plan:
        labels = "、".join(label_of(step["tool"], step.get("params")) for step in plan)
        if text and text != _FALLBACK_REPLY:
            return text
        if len(plan) > 1:
            return f"将按以下步骤执行：{labels}。确认后开始。"
        return f"好，正在{labels}。"
    return text or _FALLBACK_REPLY