import asyncio
import json
import uuid

import httpx
import pytest
from langchain_core.messages import AIMessage, AIMessageChunk

from app import events
from app.agent import chat as chat_mod
from app.agent import graph, triage
from app.agent.llm import planner
from app.agent.triage import CHAT, EDIT
from app.agent.words import brief
from app.config import get_settings
from app.db import SessionFactory
from app.models import AgentRun, EditSession, ToolRun
from app.models.tool_run import RunStatus
from app.services import agent as agent_service
from app.services import runs as runs_service
from app.tasks.tools import run_tool
from app.tools import label_of
from tests.canvas import apply, drain, error_of, layers, scene, select, settle
from tests.test_sessions import open_session

PROMPT = {"prompt": "浅木色桌面上的白色马克杯", "ratio": "1:1", "count": 2}


class FakePlanner:
    def __init__(self, message: AIMessage) -> None:
        self._message = message
        self.calls = 0
        self.messages = None

    async def ainvoke(self, messages):
        self.calls += 1
        self.messages = messages
        return self._message

    async def astream(self, messages):
        """真实模型是一段一段给的，测试桩也按这个形状产出。"""
        self.calls += 1
        self.messages = messages
        for index, call in enumerate(self._message.tool_calls):
            yield AIMessageChunk(
                content="",
                tool_call_chunks=[
                    {
                        "name": call["name"],
                        "args": json.dumps(call["args"], ensure_ascii=False),
                        "index": index,
                        "id": call["id"],
                    }
                ],
            )
        if self._message.content:
            yield AIMessageChunk(content=self._message.content)


@pytest.fixture
def fake_planner(monkeypatch):
    def install(message: AIMessage) -> FakePlanner:
        fake = FakePlanner(message)
        monkeypatch.setattr(graph, "planner", lambda: fake)
        return fake

    return install


def channel(value: str):
    """把分流固定成某个通道。triage 节点会 await 它，所以必须是异步函数。"""

    async def classify(goal: str) -> str:
        return value

    return classify


@pytest.fixture(autouse=True)
def edit_channel(monkeypatch):
    """分流默认走修图通道，不让测试花真实模型调用。闲聊通道由需要它的用例自己改。"""
    monkeypatch.setattr(graph, "classify", channel(EDIT))


def tool_call(name: str, args: dict) -> AIMessage:
    return tool_calls((name, args))


def tool_calls(*pairs: tuple[str, dict]) -> AIMessage:
    return AIMessage(
        content="",
        tool_calls=[
            {"name": name, "args": args, "id": f"call_{index}"}
            for index, (name, args) in enumerate(pairs, start=1)
        ],
    )


@pytest.fixture
async def signed_in(client: httpx.AsyncClient, credentials):
    await client.post("/api/auth/register", json=credentials)
    return client


async def send(client: httpx.AsyncClient, session_id: str, text: str) -> dict:
    """发一条指令并等后台把这一轮跑完。

    POST 现在只保证轮次已落库，答复与计划都在后台补，断言前必须先收口。
    """
    response = await client.post(f"/api/sessions/{session_id}/messages", json={"text": text})
    assert response.status_code == 201, response.text
    turn = response.json()
    await drain()
    stored = (await client.get(f"/api/sessions/{session_id}/messages")).json()
    return next(item for item in stored if item["id"] == turn["id"])


async def test_tool_call_is_planned_and_dispatched(signed_in: httpx.AsyncClient, fake_planner):
    fake_planner(tool_call("generate_image", PROMPT))
    session_id = (await open_session(signed_in))["id"]

    turn = await send(signed_in, session_id, "换一张白底马克杯")

    assert turn["status"] == "running"
    assert [step["tool"] for step in turn["steps"]] == ["generate_image"]
    assert turn["steps"][0]["label"] == "生成图片"
    assert turn["steps"][0]["run_id"]
    assert "没太理解" not in turn["reply"]
    assert "生成图片" in turn["reply"]


async def test_marketing_tool_is_planned_and_dispatched(signed_in: httpx.AsyncClient, fake_planner):
    fake_planner(tool_call("generate_marketing", {"kind": "product"}))
    session_id = (await open_session(signed_in))["id"]

    turn = await send(signed_in, session_id, "生成一张纯净背景图")

    assert turn["status"] == "running"
    assert [step["tool"] for step in turn["steps"]] == ["generate_marketing"]
    assert turn["steps"][0]["label"] == "衍生图"
    assert "衍生图" in turn["reply"]


async def test_plain_answer_dispatches_nothing(signed_in: httpx.AsyncClient, fake_planner):
    """修图通道里模型只回一句话、没给工具调用：本轮不得产生任何步骤。"""
    fake_planner(AIMessage(content="现有工具做不到这个。"))
    session_id = (await open_session(signed_in))["id"]

    turn = await send(signed_in, session_id, "把这匹马改成会飞的")

    assert turn["kind"] == "edit"
    assert turn["steps"] == []
    assert turn["reply"] == "现有工具做不到这个。"


async def test_long_refusal_keeps_only_the_first_sentence(
    signed_in: httpx.AsyncClient, fake_planner
):
    fake_planner(
        AIMessage(
            content="选区不一定包含老鼠。"
            "replace_region 需要完整提示词，adjust_image 也不能局部改色。"
        )
    )
    session_id = (await open_session(signed_in))["id"]

    turn = await send(signed_in, session_id, "老鼠换成红色")

    assert turn["steps"] == []
    assert turn["reply"] == "选区不一定包含老鼠。"
    assert "replace_region" not in turn["reply"]


def test_spoken_ignores_fallback_when_plan_exists():
    from app.agent.graph import _FALLBACK_REPLY, spoken

    assert "没太理解" not in spoken("", [{"tool": "flip_layer"}])
    assert spoken(_FALLBACK_REPLY, [{"tool": "flip_layer"}, {"tool": "rotate_layer"}]).startswith(
        "将按以下步骤执行"
    )


def test_spoken_never_shows_internal_names_but_keeps_the_explanation():
    from app.agent.graph import spoken

    plan = [{"tool": "smart_edit", "params": {"instruction": "把老马改成老徐"}}]
    prose = 'smart_edit 指令为"把老马改成老徐"。'
    assert "smart_edit" not in spoken(prose, plan)
    assert "没太理解" in spoken(prose, [])
    refusal = "选区不一定包含老鼠。replace_region 需要完整提示词。"
    assert spoken(refusal, []) == "选区不一定包含老鼠。"


def test_prose_call_is_rescued_only_when_unambiguous():
    from app.agent.graph import _calls_from_prose

    rescued = _calls_from_prose("smart_edit 指令为“把衣服改成红色，保持背景不变”。")
    assert rescued == [
        {"tool": "smart_edit", "params": {"instruction": "把衣服改成红色，保持背景不变"}}
    ]
    # 提到两个工具名是对整条计划的描述，不能硬塞成单步
    assert _calls_from_prose("我先用 smart_edit 改人物，再用 flip_layer 翻转。") == []


def test_text_call_parsing_only_accepts_literals():
    from app.agent.graph import _calls_from_text

    assert _calls_from_text('smart_edit(instruction="把衣服改成红色")') == [
        {"tool": "smart_edit", "params": {"instruction": "把衣服改成红色"}}
    ]
    assert _calls_from_text("__import__('os').system('calc')") == []
    assert _calls_from_text('smart_edit("位置参数")') == []


async def test_single_action_goal_is_planned_once(signed_in: httpx.AsyncClient, fake_planner):
    fake = fake_planner(tool_call("flip_layer", {"direction": "vertical"}))
    session_id = (await open_session(signed_in))["id"]

    turn = await send(signed_in, session_id, "垂直翻转画面")

    assert fake.calls == 1
    assert [step["tool"] for step in turn["steps"]] == ["flip_layer"]


async def test_dropped_action_triggers_one_replan(signed_in: httpx.AsyncClient, fake_planner):
    fake = fake_planner(tool_call("flip_layer", {"direction": "vertical"}))
    session_id = (await open_session(signed_in))["id"]

    turn = await send(signed_in, session_id, "垂直翻转画面，然后把背景改成夏天")

    assert fake.calls == 2
    assert "漏掉了后续动作" in fake.messages[-1].content
    assert [step["tool"] for step in turn["steps"]] == ["flip_layer"]


async def test_canvas_facts_are_given_to_the_planner(signed_in: httpx.AsyncClient, fake_planner):
    fake = fake_planner(AIMessage(content="好的。"))
    session_id = (await open_session(signed_in))["id"]

    await send(signed_in, session_id, "看看这张图")

    system = fake.messages[0].content
    assert "画幅 320×240" in system
    assert "修订号 1" in system
    assert "底图" in system
    assert "id=base" in system
    assert "当前无选区" in system


async def test_existing_selection_is_given_to_the_planner(
    signed_in: httpx.AsyncClient, fake_planner
):
    fake = fake_planner(tool_call("replace_region", {"prompt": "改成黑色"}))
    session = await open_session(signed_in)
    selected = await signed_in.post(
        f"/api/sessions/{session['id']}/selection",
        json={"revision": session["revision"], "points": [{"x": 0.5, "y": 0.5}]},
    )
    assert selected.status_code == 200

    turn = await send(signed_in, session["id"], "骨头改成黑色")

    assert "已有选区" in fake.messages[0].content
    assert [step["tool"] for step in turn["steps"]] == ["replace_region"]


async def test_chat_channel_skips_the_canvas_and_the_planner(
    signed_in: httpx.AsyncClient, fake_planner, monkeypatch
):
    """闲聊通道：不读画布、不进规划模型、不出步骤，也不写编辑历史。"""
    read = 0

    async def spy(session, record):
        nonlocal read
        read += 1
        return "画幅 1×1"

    async def chatter(goal: str):
        yield "写诗我不会，图片相关的需求直接说。"

    fake = fake_planner(AIMessage(content="不该被调用"))
    monkeypatch.setattr(graph, "classify", channel(CHAT))
    monkeypatch.setattr(graph, "answer", chatter)
    monkeypatch.setattr(agent_service, "describe", spy)

    session = await open_session(signed_in)
    turn = await send(signed_in, session["id"], "帮我写一首诗")

    assert turn["kind"] == "chat"
    assert turn["status"] == "succeeded"
    assert turn["steps"] == []
    assert turn["reply"] == "写诗我不会，图片相关的需求直接说。"
    assert (fake.calls, read) == (0, 0)

    stored = (await signed_in.get(f"/api/sessions/{session['id']}/messages")).json()
    assert [item["kind"] for item in stored] == ["chat"]
    detail = (await signed_in.get(f"/api/sessions/{session['id']}")).json()
    assert detail["revision"] == session["revision"]
    history = (await signed_in.get(f"/api/sessions/{session['id']}/history")).json()
    assert [entry["action"] for entry in history] == ["create_session"]


async def test_chat_turn_does_not_disturb_the_next_edit_turn(
    signed_in: httpx.AsyncClient, fake_planner, monkeypatch
):
    """两条通道交替：“你是谁”只说一句，下一句修图照常执行。"""
    fake_planner(tool_call("flip_layer", {"direction": "vertical"}))
    session_id = (await open_session(signed_in))["id"]

    async def chatter(goal: str):
        yield "在的。"

    async def route(goal: str) -> str:
        return CHAT if goal == "你是谁" else EDIT

    monkeypatch.setattr(graph, "classify", route)
    monkeypatch.setattr(graph, "answer", chatter)

    chat = await send(signed_in, session_id, "你是谁")
    edited = await send(signed_in, session_id, "垂直翻转画面")

    assert (chat["kind"], chat["steps"]) == ("chat", [])
    assert edited["kind"] == "edit"
    assert [step["tool"] for step in edited["steps"]] == ["flip_layer"]
    assert edited["status"] == "succeeded"

    turns = (await signed_in.get(f"/api/sessions/{session_id}/messages")).json()
    assert [turn["kind"] for turn in turns] == ["chat", "edit"]


class FakeTriage:
    def __init__(self) -> None:
        self.calls = 0

    async def ainvoke(self, messages):
        self.calls += 1
        return AIMessage(content="chat")


async def test_triage_spends_no_model_call_on_obvious_edits(monkeypatch):
    """命中关键词就直接判定为修图，模棱两可才问分流模型。"""
    fake = FakeTriage()
    monkeypatch.setattr(triage, "triager", lambda: fake)

    assert await triage.classify("把背景换成白色") == EDIT
    assert fake.calls == 0

    assert await triage.classify("你是谁") == CHAT
    assert fake.calls == 1


class TriageGone:
    """分流模型下线时的样子：一调就抛。"""

    async def ainvoke(self, messages):
        raise RuntimeError("Model not exist")


async def test_triage_falls_back_to_edit_when_the_model_is_gone(monkeypatch):
    """分流模型不可用不能拖垮整轮指令：退回修图通道，那才是改动前的唯一行为。"""
    monkeypatch.setattr(triage, "triager", lambda: TriageGone())

    assert await triage.classify("今天天气怎么样") == EDIT


async def test_split_refuses_multi_step_edits():
    """两段都是修图就不拆：多步编排要留给规划模型一次输出，拆开反而把一步做成两轮。"""
    judge = channel(EDIT)

    assert await triage.split("主体改红色，背景改蓝色", judge) is None
    assert await triage.split("垂直翻转画面", judge) is None


async def test_split_refuses_a_question_that_is_not_really_a_question():
    """后半句长得像修图就不拆，否则「保持背景不变」这种会被当成闲聊拖走。"""

    async def judge(text: str) -> str:
        return EDIT if "背景" in text else CHAT

    assert await triage.split("把衣服改成红色，保持背景不变", judge) is None
    # 拆不出两段就不该拆
    assert await triage.split("你是谁", channel(CHAT)) is None


async def test_split_recognises_imperative_questions():
    """「讲一下X」是祈使句式的提问：没问号也没「是谁」，但确实在要一个答案。

    认不出它时整句会原样进规划模型，模型在「用散文回答」与「调用工具」之间摇摆，
    同一种句式就会有时只答、有时只修。用户原话回归。
    """

    async def judge(text: str) -> str:
        return EDIT if triage.looks_like_edit(text) else CHAT

    edit_goal, chat_goal = await triage.split(
        "讲一下热力学第一定律，然后把画面里的20岁年轻人换成女生", judge
    )
    assert chat_goal == "讲一下热力学第一定律"
    assert edit_goal == "把画面里的20岁年轻人换成女生"

    edit_goal, chat_goal = await triage.split("介绍一下牛顿，顺便把背景换成纯白", judge)
    assert chat_goal == "介绍一下牛顿"
    assert edit_goal == "把背景换成纯白"

    # 新词表不能把纯修图句误拆：两段都被判成修图时依旧不拆
    assert await triage.split("把衣服改成红色，保持背景不变", judge) is None
    assert await triage.split("把介绍文字改成三行，然后调亮一点", judge) is None


async def test_split_keeps_the_original_order_of_the_pieces():
    """同通道的片段拼回一句，且按原句先后，不重排。"""

    async def judge(text: str) -> str:
        return CHAT if "公式" in text else EDIT

    edit_goal, chat_goal = await triage.split(
        "画面旋转90度然后告诉我二次函数的公式，再把背景换成蓝色", judge
    )

    assert edit_goal == "画面旋转90度，再把背景换成蓝色"
    assert chat_goal == "告诉我二次函数的公式"


async def test_mixed_turn_answers_first_and_plans_the_edit_after(
    signed_in: httpx.AsyncClient, fake_planner, monkeypatch
):
    """先改后问：文字答案要答出来，修图步骤也要照常安排，不能只旋转不回答。"""
    fake = fake_planner(tool_call("rotate_layer", {"angle": 90}))

    async def chatter(goal: str):
        yield "二次函数一般写成 y=ax²+bx+c。"

    async def route(text: str) -> str:
        return CHAT if "公式" in text else EDIT

    monkeypatch.setattr(graph, "classify", route)
    monkeypatch.setattr(graph, "answer", chatter)
    session_id = (await open_session(signed_in))["id"]

    turn = await send(signed_in, session_id, "画面旋转90度然后告诉我二次函数的公式")

    assert turn["kind"] == "mixed"
    assert [step["tool"] for step in turn["steps"]] == ["rotate_layer"]
    assert turn["reply"].startswith("二次函数一般写成 y=ax²+bx+c。")
    assert "旋转" in turn["reply"]
    assert turn["error"] is None
    # 规划模型只拿到修图那半句，无关提问不会再被当成要执行的动作整体拒绝
    assert fake.messages[1].content == "画面旋转90度"


async def test_mixed_turn_does_not_refuse_a_question_put_first(
    signed_in: httpx.AsyncClient, fake_planner, monkeypatch
):
    """先问后改：换个语序不能把「二次函数」当成修图请求回一句无法完成。"""
    fake = fake_planner(tool_call("rotate_layer", {"angle": 90}))

    async def chatter(goal: str):
        yield "二次函数一般写成 y=ax²+bx+c。"

    async def route(text: str) -> str:
        return CHAT if "公式" in text else EDIT

    monkeypatch.setattr(graph, "classify", route)
    monkeypatch.setattr(graph, "answer", chatter)
    session_id = (await open_session(signed_in))["id"]

    turn = await send(signed_in, session_id, "告诉我二次函数的公式然后画面旋转90度")

    assert turn["kind"] == "mixed"
    assert [step["tool"] for step in turn["steps"]] == ["rotate_layer"]
    assert "无法完成" not in turn["reply"]
    assert fake.messages[1].content == "画面旋转90度"


async def test_mixed_turn_splits_an_imperative_question(
    signed_in: httpx.AsyncClient, fake_planner, monkeypatch
):
    """用户原话回归：先「讲一下X」再修图，两段都要触发，不能只答或只修。"""
    fake = fake_planner(tool_call("rotate_layer", {"angle": 90}))

    async def chatter(goal: str):
        yield "热力学第一定律是能量守恒定律在热现象中的体现。"

    async def route(text: str) -> str:
        return CHAT if "定律" in text else EDIT

    monkeypatch.setattr(graph, "classify", route)
    monkeypatch.setattr(graph, "answer", chatter)
    session_id = (await open_session(signed_in))["id"]

    turn = await send(
        signed_in, session_id, "讲一下热力学第一定律，然后把画面里人物换成女生"
    )

    assert turn["kind"] == "mixed"
    assert [step["tool"] for step in turn["steps"]] == ["rotate_layer"]
    assert turn["reply"].startswith("热力学第一定律是能量守恒定律在热现象中的体现。")
    assert "旋转" in turn["reply"]
    # 规划模型只拿到修图那半句，提问不会进去干扰它
    assert fake.messages[1].content == "把画面里人物换成女生"


async def test_mixed_edit_part_still_asks_for_confirmation(
    signed_in: httpx.AsyncClient, fake_planner, monkeypatch
):
    """混合轮里修图那段是多步时，依旧先列步骤等确认，文字答案不受影响。"""
    fake_planner(
        tool_calls(
            ("rotate_layer", {"angle": 90}),
            ("flip_layer", {"direction": "horizontal"}),
        )
    )

    async def chatter(goal: str):
        yield "牛顿第二定律是 F=ma。"

    async def route(text: str) -> str:
        return CHAT if "定律" in text else EDIT

    monkeypatch.setattr(graph, "classify", route)
    monkeypatch.setattr(graph, "answer", chatter)
    session_id = (await open_session(signed_in))["id"]

    turn = await send(signed_in, session_id, "旋转90度然后告诉我牛顿第二定律，再水平翻转")

    assert turn["status"] == "queued"
    assert [step["tool"] for step in turn["steps"]] == ["rotate_layer", "flip_layer"]
    assert turn["reply"].startswith("牛顿第二定律是 F=ma。")
    assert "确认后开始" in turn["reply"]


class PlannerGone:
    """规划模型下线时的样子：一进流就抛。"""

    async def astream(self, messages):
        raise RuntimeError("Model not exist")
        yield AIMessageChunk(content="")


async def test_mixed_turn_keeps_the_answer_when_the_edit_part_fails(
    signed_in: httpx.AsyncClient, monkeypatch
):
    """两段结果独立：修图那段挂了只报这一段，已经答出来的文字照样落库展示。

    这一轮不能记 error——前端一旦拿到 error 就把正文藏掉，用户会看不见答案。
    """

    async def chatter(goal: str):
        yield "二次函数一般写成 y=ax²+bx+c。"

    async def route(text: str) -> str:
        return CHAT if "公式" in text else EDIT

    monkeypatch.setattr(graph, "classify", route)
    monkeypatch.setattr(graph, "answer", chatter)
    monkeypatch.setattr(graph, "planner", lambda: PlannerGone())
    session_id = (await open_session(signed_in))["id"]

    turn = await send(signed_in, session_id, "画面旋转90度然后告诉我二次函数的公式")

    assert turn["kind"] == "mixed"
    assert turn["error"] is None
    assert turn["steps"] == []
    assert "二次函数一般写成 y=ax²+bx+c。" in turn["reply"]
    assert "没安排出来" in turn["reply"]


class ChatterGone:
    """闲聊模型下线时的样子。"""

    def astream(self, messages):
        raise RuntimeError("Model not exist")


async def test_mixed_turn_keeps_the_plan_when_the_answer_fails(
    signed_in: httpx.AsyncClient, fake_planner, monkeypatch
):
    """反方向同样独立：文字段挂了不能把修图步骤一并拖掉。"""
    fake_planner(tool_call("rotate_layer", {"angle": 90}))

    async def route(text: str) -> str:
        return CHAT if "公式" in text else EDIT

    monkeypatch.setattr(graph, "classify", route)
    monkeypatch.setattr(chat_mod, "chatter", lambda: ChatterGone())
    session_id = (await open_session(signed_in))["id"]

    turn = await send(signed_in, session_id, "画面旋转90度然后告诉我二次函数的公式")

    assert turn["kind"] == "mixed"
    assert turn["error"] is None
    assert [step["tool"] for step in turn["steps"]] == ["rotate_layer"]
    assert "答不上来" in turn["reply"]


async def test_chat_turn_reports_the_real_reason_when_the_model_fails(
    signed_in: httpx.AsyncClient, monkeypatch
):
    """闲聊模型挂了得说出是模型不可用，不能只丢一句「规划失败，请重试」。"""

    def no_chatter():
        raise RuntimeError("Model not exist")

    monkeypatch.setattr(graph, "classify", channel(CHAT))
    monkeypatch.setattr(chat_mod, "chatter", no_chatter)
    session_id = (await open_session(signed_in))["id"]

    turn = await send(signed_in, session_id, "计算 666+87")

    assert turn["status"] == "failed"
    assert turn["steps"] == []
    assert "闲聊模型" in turn["error"]


async def test_turn_is_saved_before_the_model_answers(
    signed_in: httpx.AsyncClient, fake_planner, monkeypatch
):
    """POST 不等模型：先落一条进行中轮次，答复与计划在后台补。"""
    entered = asyncio.Event()
    release = asyncio.Event()

    async def slow(goal: str) -> str:
        entered.set()
        await release.wait()
        return EDIT

    fake = fake_planner(tool_call("flip_layer", {"direction": "vertical"}))
    monkeypatch.setattr(graph, "classify", slow)
    session_id = (await open_session(signed_in))["id"]

    response = await signed_in.post(
        f"/api/sessions/{session_id}/messages", json={"text": "垂直翻转画面"}
    )
    turn = response.json()

    assert response.status_code == 201
    assert (turn["status"], turn["reply"], turn["steps"]) == ("running", "", [])

    # 后台确实已经卡在模型上：轮次已落库，规划模型一次都还没跑
    await asyncio.wait_for(entered.wait(), 5)
    assert fake.calls == 0
    stored = (await signed_in.get(f"/api/sessions/{session_id}/messages")).json()
    assert stored[-1]["status"] == "running"

    release.set()
    await drain()

    stored = (await signed_in.get(f"/api/sessions/{session_id}/messages")).json()
    assert stored[-1]["status"] == "succeeded"
    assert [step["tool"] for step in stored[-1]["steps"]] == ["flip_layer"]


def _chunk(name: str | None, args: str, index: int, call_id: str | None = None):
    """一段工具调用增量。真实模型就是这样一段一段给的。"""
    return AIMessageChunk(
        content="",
        tool_call_chunks=[{"name": name, "args": args, "index": index, "id": call_id}],
    )


class StreamPlanner:
    def __init__(self, pieces: list[AIMessageChunk]) -> None:
        self._pieces = pieces

    async def astream(self, messages):
        for piece in self._pieces:
            yield piece


async def test_steps_are_announced_while_the_model_is_still_talking(monkeypatch):
    """步骤名在计划成形前就推出去，同时增量必须能拼回完整 tool_calls。"""
    pieces = [
        _chunk("flip_layer", '{"direction"', 0, "call_1"),
        _chunk(None, ': "horizontal"}', 0),
        _chunk("rotate_layer", '{"angle"', 1, "call_2"),
        _chunk(None, ": 15}", 1),
    ]
    events: list[dict] = []

    async def collect(event: dict) -> None:
        events.append(event)

    async def context() -> str:
        return "画幅 1×1，修订号 1。\n图层：底图（id=base，图像 1×1，锁定）\n当前无选区。"

    monkeypatch.setattr(graph, "classify", channel(EDIT))
    monkeypatch.setattr(graph, "planner", lambda: StreamPlanner(pieces))

    reply, steps, kind = await graph.run("水平翻转再转 15 度", context, collect)

    assert kind == EDIT
    assert [step["tool"] for step in steps] == ["flip_layer", "rotate_layer"]
    # 校验过的参数会补齐字段默认值，这里只认模型给的那个角度
    assert steps[1]["params"]["angle"] == 15
    assert [event["index"] for event in events if event["type"] == "plan"] == [1, 2]
    assert [event["label"] for event in events if event["type"] == "plan"] == [
        label_of("flip_layer"),
        label_of("rotate_layer"),
    ]
    assert [event["type"] for event in events] == ["stage", "kind", "stage", "plan", "plan"]
    assert label_of("flip_layer") in reply


class FakeChatter:
    def __init__(self, pieces: list[str]) -> None:
        self._pieces = pieces

    async def astream(self, messages):
        for piece in self._pieces:
            yield AIMessageChunk(content=piece)


async def test_chat_pieces_add_up_to_the_stored_reply(monkeypatch):
    """逐句推出去的正文拼起来必须与落库那句完全一致，否则收尾会跳字。"""
    pieces = ["666 加 87 等于 753", "。", "牛顿第一定律", "也叫惯性定律", "。"]
    monkeypatch.setattr(chat_mod, "chatter", lambda: FakeChatter(pieces))

    streamed = [piece async for piece in chat_mod.answer("计算 666+87")]

    assert streamed == ["666 加 87 等于 753。", "牛顿第一定律也叫惯性定律。"]
    assert "".join(streamed) == chat_mod.compose(brief("".join(pieces)))


async def test_tool_names_never_reach_the_stream(monkeypatch):
    """含工具名的句子整句按住，用户看到的与落库的都不许泄漏。"""
    pieces = ["replace_region 需要完整提示词。", "这张图我改不了。"]
    monkeypatch.setattr(chat_mod, "chatter", lambda: FakeChatter(pieces))

    streamed = [piece async for piece in chat_mod.answer("把老鼠换成红色")]

    assert "".join(streamed) == "这张图我改不了。"


def test_latex_is_written_as_plain_math():
    """界面不渲染 LaTeX，公式得换成能直接读的形式，带层级的分数也不能读错。"""
    assert brief(r"质能方程 $E=mc^2$，其中 $m$ 是质量。") == "质能方程 E=mc²，其中 m 是质量。"
    assert brief(r"由 $\frac{-b \pm \sqrt{b^2-4ac}}{2a}$ 可解。") == "由 (-b ± √(b²-4ac))/2a 可解。"
    acid = r"水的 pH 约 $7$，氢氧化钠溶液 $pH \approx 14$。"
    assert brief(acid) == "水的 pH 约 7，氢氧化钠溶液 pH ≈ 14。"
    # 落单的一个 $ 是货币符号，不是公式开头，不能跟着被拆
    assert brief("这件衣服价格 $100。") == "这件衣服价格 $100。"


async def test_math_pieces_add_up_to_the_stored_reply(monkeypatch):
    """公式是在流里改写的，拼起来必须与落库那句一致，不能收尾时整段重排。"""
    pieces = [r"勾股定理 $a^2+b^2=", r"c^2$。", r"斜边 $c = ", r"\sqrt{a^2+b^2}", r"。"]
    monkeypatch.setattr(chat_mod, "chatter", lambda: FakeChatter(pieces))

    streamed = [piece async for piece in chat_mod.answer("勾股定理")]
    text = "".join(streamed)

    assert text == chat_mod.compose(brief("".join(pieces)))
    assert text == "勾股定理 a²+b²=c²。斜边 c = √(a²+b²)。"


async def test_turn_events_are_private_to_the_owner(
    client: httpx.AsyncClient, credentials, other_credentials, fake_planner
):
    """事件流只拿到 turn id，归属要顺着会话查，不能让别人偷看对话。"""
    fake_planner(tool_call("flip_layer", {"direction": "vertical"}))
    await client.post("/api/auth/register", json=credentials)
    session_id = (await open_session(client))["id"]
    turn = await send(client, session_id, "垂直翻转画面")
    path = f"/events/turns/{turn['id']}"

    await client.post("/api/auth/logout")
    assert (await client.get(path)).status_code == 401

    await client.post("/api/auth/register", json=other_credentials)
    assert (await client.get(path)).status_code == 404


async def test_turn_events_are_replayed_to_late_subscribers():
    """晚一步订阅也要拿到全部过程：阶段话术往往在浏览器连上之前就推完了。"""
    turn_id = uuid.uuid4()

    await events.publish_turn(turn_id, {"type": "stage", "text": "在看你这句话的意思"})
    await events.publish_turn(uuid.uuid4(), {"type": "token", "text": "别的轮次"})
    await events.publish_turn(turn_id, {"type": "plan", "index": 1, "label": "换背景"})
    await events.publish_turn(turn_id, {"type": "done", "status": "running"})

    received: list[dict] = []
    async with events.subscribe_turn(turn_id) as messages:
        async for payload in messages:
            # 空闲超时产出 None，事件流只拿它发心跳，不算内容
            if payload is None:
                continue
            received.append(payload)
            if payload["type"] == "done":
                break

    assert [item["type"] for item in received] == ["stage", "plan", "done"]


async def test_turn_stream_returns_snapshot_for_finished_turn(
    signed_in: httpx.AsyncClient, fake_planner
):
    """跑完的一轮再连事件流，给完终态快照就断，刷新页面不会停在「正在安排…」。"""
    fake_planner(tool_call("flip_layer", {"direction": "vertical"}))
    session_id = (await open_session(signed_in))["id"]
    turn = await send(signed_in, session_id, "垂直翻转画面")

    frames: list[dict] = []
    async with signed_in.stream("GET", f"/events/turns/{turn['id']}") as stream:
        assert stream.headers["content-type"].startswith("text/event-stream")
        async for line in stream.aiter_lines():
            if line.startswith("data: "):
                frames.append(json.loads(line[6:]))

    assert [frame["type"] for frame in frames] == ["snapshot"]
    assert frames[0]["turn"]["id"] == turn["id"]
    assert frames[0]["turn"]["status"] == "succeeded"


async def test_unregistered_tool_is_refused(signed_in: httpx.AsyncClient, fake_planner):
    fake_planner(tool_call("teleport_subject", {}))
    session_id = (await open_session(signed_in))["id"]

    turn = await send(signed_in, session_id, "把主体传送走")

    assert turn["steps"] == []
    assert "执行不了" in turn["reply"]


async def test_illegal_params_are_refused(signed_in: httpx.AsyncClient, fake_planner):
    fake_planner(tool_call("generate_image", {"prompt": "  ", "count": 99}))
    session_id = (await open_session(signed_in))["id"]

    turn = await send(signed_in, session_id, "随便来一张")

    assert turn["steps"] == []
    assert "执行不了" in turn["reply"]


async def test_results_join_the_wall_without_switching_current(
    signed_in: httpx.AsyncClient, fake_planner
):
    fake_planner(tool_call("generate_image", PROMPT))
    session = await open_session(signed_in)
    turn = await send(signed_in, session["id"], "换一张")

    await run_tool({}, uuid.UUID(turn["steps"][0]["run_id"]))

    updated = (await signed_in.get(f"/api/sessions/{session['id']}")).json()
    assert len(updated["assets"]) == 3
    assert updated["current_asset_id"] == session["current_asset_id"]
    assert updated["revision"] == 1

    turns = (await signed_in.get(f"/api/sessions/{session['id']}/messages")).json()
    assert turns[-1]["status"] == "succeeded"
    assert turns[-1]["steps"][0]["status"] == "succeeded"


async def test_conversation_is_returned_in_order(signed_in: httpx.AsyncClient, fake_planner):
    fake_planner(AIMessage(content="好的。"))
    session_id = (await open_session(signed_in))["id"]

    await send(signed_in, session_id, "第一句")
    await send(signed_in, session_id, "第二句")

    turns = (await signed_in.get(f"/api/sessions/{session_id}/messages")).json()
    assert [turn["goal"] for turn in turns] == ["第一句", "第二句"]


async def test_message_list_keeps_the_latest_turns(signed_in: httpx.AsyncClient, fake_planner):
    """超过一页时留下的必须是最近的几轮，否则刚发的指令会从对话框里消失。"""
    fake_planner(AIMessage(content="好的。"))
    session_id = (await open_session(signed_in))["id"]
    for goal in ("第一句", "第二句", "第三句"):
        await send(signed_in, session_id, goal)

    async with SessionFactory() as session:
        record = await session.get(EditSession, uuid.UUID(session_id))
        recent = await agent_service.turns_of(session, record, limit=2)

    assert [turn.goal for turn in recent] == ["第二句", "第三句"]


async def test_deleting_the_session_drops_its_turns(
    signed_in: httpx.AsyncClient, fake_planner
):
    """会话整条删掉，名下的轮次行也跟着级联，不给对话列表留孤儿。"""
    fake_planner(tool_call("flip_layer", {"direction": "vertical"}))
    session_id = (await open_session(signed_in))["id"]
    turn = await send(signed_in, session_id, "垂直翻转画面")

    assert (await signed_in.delete(f"/api/sessions/{session_id}")).status_code == 204

    async with SessionFactory() as session:
        assert await session.get(AgentRun, uuid.UUID(turn["id"])) is None


async def test_missing_api_key_fails_the_turn(signed_in: httpx.AsyncClient, monkeypatch):
    """模型不可用时必须明确失败，不能伪造成功结果。"""
    settings = get_settings()
    monkeypatch.setattr(settings, "dashscope_api_key", "")
    planner.cache_clear()
    session_id = (await open_session(signed_in))["id"]

    turn = await send(signed_in, session_id, "换个背景")

    planner.cache_clear()
    assert turn["status"] == "failed"
    assert "DASHSCOPE_API_KEY" in turn["error"]


async def test_blank_message_is_rejected(signed_in: httpx.AsyncClient):
    session_id = (await open_session(signed_in))["id"]

    response = await signed_in.post(f"/api/sessions/{session_id}/messages", json={"text": "   "})

    assert response.status_code == 422


async def test_messages_require_authentication(client: httpx.AsyncClient):
    path = f"/api/sessions/{uuid.uuid4()}/messages"

    assert (await client.get(path)).status_code == 401
    assert (await client.post(path, json={"text": "你好"})).status_code == 401


def test_cyclic_dependencies_are_rejected():
    from app.agent.plan import PlanError, assemble

    with pytest.raises(PlanError, match="循环"):
        assemble(
            [
                {"id": "a", "tool": "flip_layer", "params": {}, "depends_on": ["b"]},
                {"id": "b", "tool": "flip_layer", "params": {}, "depends_on": ["a"]},
            ]
        )


def test_plan_longer_than_limit_is_rejected():
    from app.agent.plan import MAX_STEPS, PlanError, assemble

    with pytest.raises(PlanError, match="超过"):
        assemble([{"tool": "flip_layer", "params": {}}] * (MAX_STEPS + 1))


async def test_multi_step_plan_waits_for_confirm(signed_in: httpx.AsyncClient, fake_planner):
    fake_planner(
        tool_calls(
            ("flip_layer", {"direction": "horizontal"}),
            ("rotate_layer", {"angle": 15}),
        )
    )
    session_id = (await open_session(signed_in))["id"]

    turn = await send(signed_in, session_id, "水平翻转再转 15 度")

    assert turn["status"] == "queued"
    assert "没太理解" not in turn["reply"]
    assert "确认后开始" in turn["reply"]
    assert [step["tool"] for step in turn["steps"]] == ["flip_layer", "rotate_layer"]
    assert turn["steps"][1]["depends_on"] == ["s1"]
    assert all(step["run_id"] is None for step in turn["steps"])

    session = (await signed_in.get(f"/api/sessions/{session_id}")).json()
    assert session["document"]["layers"][0]["transform"]["scale_x"] == 1


async def test_flip_steps_show_their_direction(signed_in: httpx.AsyncClient, fake_planner):
    fake_planner(
        tool_calls(
            ("flip_layer", {"direction": "horizontal"}),
            ("flip_layer", {"direction": "vertical"}),
        )
    )
    session_id = (await open_session(signed_in))["id"]

    turn = await send(signed_in, session_id, "水平翻转再垂直翻转")

    assert [step["label"] for step in turn["steps"]] == ["水平翻转", "垂直翻转"]
    assert "水平翻转、垂直翻转" in turn["reply"]


async def test_confirm_runs_dependent_steps_in_order(signed_in: httpx.AsyncClient, fake_planner):
    fake_planner(
        tool_calls(
            ("flip_layer", {"direction": "horizontal"}),
            ("rotate_layer", {"angle": 15}),
        )
    )
    session_id = (await open_session(signed_in))["id"]
    turn = await send(signed_in, session_id, "翻转并旋转")

    confirmed = (
        await signed_in.post(f"/api/sessions/{session_id}/messages/{turn['id']}/confirm")
    ).json()
    layer = (
        await signed_in.get(f"/api/sessions/{session_id}")
    ).json()["document"]["layers"][0]

    assert confirmed["status"] == "succeeded"
    assert [step["status"] for step in confirmed["steps"]] == ["succeeded", "succeeded"]
    assert layer["transform"]["scale_x"] == -1
    assert layer["transform"]["rotation"] == 15


async def test_queued_step_unblocks_the_next(signed_in: httpx.AsyncClient, fake_planner):
    fake_planner(
        tool_calls(
            ("remove_background", {}),
            ("flip_layer", {"direction": "horizontal"}),
        )
    )
    session = await open_session(signed_in)
    turn = await send(signed_in, session["id"], "去背景再水平翻转")
    started = (
        await signed_in.post(f"/api/sessions/{session['id']}/messages/{turn['id']}/confirm")
    ).json()

    assert started["status"] == "running"
    assert started["steps"][0]["run_id"]
    assert started["steps"][1]["run_id"] is None

    await run_tool({}, uuid.UUID(started["steps"][0]["run_id"]))
    finished = (await signed_in.get(f"/api/sessions/{session['id']}/messages")).json()[-1]
    updated = (await signed_in.get(f"/api/sessions/{session['id']}")).json()

    assert finished["status"] == "succeeded"
    assert [step["status"] for step in finished["steps"]] == ["succeeded", "succeeded"]
    assert updated["document"]["layers"][0]["transform"]["scale_x"] == -1


async def test_two_layers_in_one_turn_share_the_same_selection(
    signed_in: httpx.AsyncClient, fake_planner
):
    """选区被钉进每一步，第一步清除选区、修订号递增都不影响第二步。"""
    fake_planner(
        tool_calls(
            ("replace_region", {"prompt": "改成红色", "layer_id": "subject"}),
            ("replace_region", {"prompt": "改成蓝色", "layer_id": "background"}),
        )
    )
    session = await open_session(signed_in, image=scene())
    before = await apply(signed_in, session["id"], "split_layers")
    await select(signed_in, session["id"], before["revision"], points=[{"x": 0.5, "y": 0.5}])

    turn = await send(signed_in, session["id"], "主体改红色，背景改蓝色")
    await signed_in.post(f"/api/sessions/{session['id']}/messages/{turn['id']}/confirm")
    finished = await settle(signed_in, session["id"])
    after = (await signed_in.get(f"/api/sessions/{session['id']}")).json()

    assert finished["status"] == "succeeded"
    for layer_id in ("subject", "background"):
        assert layers(after)[layer_id]["asset_id"] != layers(before)[layer_id]["asset_id"]


async def test_selection_expires_when_the_plan_changes_the_canvas(
    signed_in: httpx.AsyncClient, fake_planner
):
    fake_planner(
        tool_calls(
            ("crop_canvas", {"ratio": "1:1"}),
            ("replace_region", {"prompt": "改成红色"}),
        )
    )
    session = await open_session(signed_in)
    await select(signed_in, session["id"], session["revision"], points=[{"x": 0.5, "y": 0.5}])

    turn = await send(signed_in, session["id"], "先裁成正方形再把选区改成红色")
    await signed_in.post(f"/api/sessions/{session['id']}/messages/{turn['id']}/confirm")
    finished = await settle(signed_in, session["id"])
    failed = next(step for step in finished["steps"] if step["tool"] == "replace_region")

    assert finished["status"] == "failed"
    assert "选区失效" in await error_of(signed_in, failed["run_id"])


async def test_cancel_drops_unstarted_steps(signed_in: httpx.AsyncClient, fake_planner):
    fake_planner(
        tool_calls(
            ("flip_layer", {"direction": "horizontal"}),
            ("rotate_layer", {"angle": 10}),
        )
    )
    session_id = (await open_session(signed_in))["id"]
    turn = await send(signed_in, session_id, "翻转再旋转")

    canceled = (
        await signed_in.post(f"/api/sessions/{session_id}/messages/{turn['id']}/cancel")
    ).json()
    session = (await signed_in.get(f"/api/sessions/{session_id}")).json()

    assert canceled["status"] == "canceled"
    assert [step["status"] for step in canceled["steps"]] == ["canceled", "canceled"]
    assert session["document"]["layers"][0]["transform"]["scale_x"] == 1


async def test_cancel_also_stops_a_running_step(signed_in: httpx.AsyncClient, fake_planner):
    """取消不再只划掉没开跑的：进行中的任务也要落定，并给 worker 递中止信号。"""
    fake_planner(tool_call("generate_image", PROMPT))
    session_id = (await open_session(signed_in))["id"]
    turn = await send(signed_in, session_id, "换一张白底马克杯")
    run_id = uuid.UUID(turn["steps"][0]["run_id"])

    # 模拟 worker 已把任务领走开跑
    async with SessionFactory() as db:
        assert await runs_service.start(db, await db.get(ToolRun, run_id))

    canceled = (
        await signed_in.post(f"/api/sessions/{session_id}/messages/{turn['id']}/cancel")
    ).json()
    assert canceled["status"] == "canceled"
    assert [step["status"] for step in canceled["steps"]] == ["canceled"]
    assert (await signed_in.get(f"/api/runs/{run_id}")).json()["status"] == "canceled"

    # 中止信号已写进 arq 的集合，worker 轮询到就会掐断正在跑的那个任务
    from arq.constants import abort_jobs_ss

    members = await events.redis_client().zrange(abort_jobs_ss, 0, -1)
    assert str(run_id) in members

    # 抢在掐断前跑完的迟到结果不能复活已取消的任务，也不许写回画布
    async with SessionFactory() as db:
        late = await db.get(ToolRun, run_id)
        assert await runs_service.finish(db, late, status=RunStatus.SUCCEEDED, result={}) is False
        await db.refresh(late)
        assert late.status is RunStatus.CANCELED

    # worker 之后才收到这个任务时直接跳过，不会二次执行
    await run_tool({}, run_id)
    assert (await signed_in.get(f"/api/runs/{run_id}")).json()["status"] == "canceled"


async def test_delete_message_removes_only_that_turn(signed_in: httpx.AsyncClient, fake_planner):
    """删除单条对话：只少这一轮，别的轮次原样留着。"""
    fake_planner(AIMessage(content="现有工具做不到这个。"))
    session_id = (await open_session(signed_in))["id"]
    first = await send(signed_in, session_id, "第一条")
    second = await send(signed_in, session_id, "第二条")

    response = await signed_in.delete(f"/api/sessions/{session_id}/messages/{first['id']}")
    assert response.status_code == 204

    remaining = (await signed_in.get(f"/api/sessions/{session_id}/messages")).json()
    assert [item["id"] for item in remaining] == [second["id"]]

    missing = await signed_in.delete(f"/api/sessions/{session_id}/messages/{first['id']}")
    assert missing.status_code == 404


async def test_delete_message_cancels_its_open_step(signed_in: httpx.AsyncClient, fake_planner):
    """删掉还在跑的那一轮：排队中的任务一并取消，不能删完消息图还在偷偷生成。"""
    fake_planner(tool_call("generate_image", PROMPT))
    session_id = (await open_session(signed_in))["id"]
    turn = await send(signed_in, session_id, "换一张白底马克杯")
    run_id = uuid.UUID(turn["steps"][0]["run_id"])

    response = await signed_in.delete(f"/api/sessions/{session_id}/messages/{turn['id']}")
    assert response.status_code == 204
    assert (await signed_in.get(f"/api/sessions/{session_id}/messages")).json() == []

    fetched = (await signed_in.get(f"/api/runs/{run_id}")).json()
    assert fetched["status"] == "canceled"

    from arq.constants import abort_jobs_ss

    members = await events.redis_client().zrange(abort_jobs_ss, 0, -1)
    assert str(run_id) in members


async def test_failed_step_can_be_retried(signed_in: httpx.AsyncClient, fake_planner):
    fake_planner(tool_call("adjust_image", {"layer_id": "不存在的层", "brightness": 0.3}))
    session_id = (await open_session(signed_in))["id"]
    turn = await send(signed_in, session_id, "把不存在的层调亮")

    await run_tool({}, uuid.UUID(turn["steps"][0]["run_id"]))
    failed = (await signed_in.get(f"/api/sessions/{session_id}/messages")).json()[-1]
    assert failed["status"] == "failed"

    retried = (
        await signed_in.post(f"/api/sessions/{session_id}/messages/{failed['id']}/retry")
    ).json()
    assert retried["status"] == "running"
    assert retried["steps"][0]["run_id"] != failed["steps"][0]["run_id"]


async def test_made_up_layer_name_falls_back_to_the_only_image(
    signed_in: httpx.AsyncClient, fake_planner
):
    """局部修改允许图层回退：模型编造图层名时改唯一图像层，而不是报错。"""
    fake_planner(tool_call("replace_region", {"prompt": "改成黑色", "layer_id": "人物"}))
    session = await open_session(signed_in)
    turn = await send(signed_in, session["id"], "把人物换成黑色")

    await run_tool({}, uuid.UUID(turn["steps"][0]["run_id"]))
    finished = (await signed_in.get(f"/api/sessions/{session['id']}/messages")).json()[-1]
    after = (await signed_in.get(f"/api/sessions/{session['id']}")).json()

    assert finished["status"] == "succeeded"
    before_layer = session["document"]["layers"][0]["asset_id"]
    assert after["document"]["layers"][0]["asset_id"] != before_layer
