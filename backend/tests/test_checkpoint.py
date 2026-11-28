"""LangGraph 检查点与启动恢复的回归测试。"""

import uuid

import pytest
from langchain_core.messages import AIMessageChunk

from app.agent import checkpoint as checkpoint_store
from app.agent import graph
from app.agent.triage import EDIT
from app.db import SessionFactory
from app.models import AgentRun, EditSession, ToolRun
from app.models.tool_run import RunStatus
from app.services import agent as agent_service
from app.services import runs as runs_service
from tests.canvas import drain
from tests.test_agent import PROMPT, FakePlanner, channel, send, tool_call
from tests.test_sessions import open_session


class BrokenPlanner:
    """规划模型下线：进流即抛，模拟节点执行到一半被中断。"""

    async def astream(self, messages):
        raise RuntimeError("模型暂不可用")
        yield AIMessageChunk(content="")


async def context() -> str:
    return "画幅 320×240，修订号 1。\n图层：底图（id=base，图像 320×240）\n当前无选区。"


@pytest.fixture
async def signed_in(client, credentials):
    await client.post("/api/auth/register", json=credentials)
    return client


async def running_turn(signed_in, goal: str = "垂直翻转画面") -> tuple[dict, uuid.UUID]:
    """建一条已落库、规划还没跑完的轮次，模拟进程在规划途中被重启。"""
    session = await open_session(signed_in)
    async with SessionFactory() as db:
        record = await db.get(EditSession, uuid.UUID(session["id"]))
        turn = AgentRun(
            user_id=record.user_id,
            session_id=record.id,
            revision=record.revision,
            goal=goal,
            status=RunStatus.RUNNING,
        )
        db.add(turn)
        await db.commit()
        await db.refresh(turn)
    return session, turn.id


async def stored_turn(signed_in, session_id: str, turn_id: uuid.UUID) -> dict:
    turns = (await signed_in.get(f"/api/sessions/{session_id}/messages")).json()
    return next(item for item in turns if item["id"] == str(turn_id))


async def test_graph_checkpoint_survives_failure_and_resumes(monkeypatch):
    """规划节点失败后检查点仍停在分流之后，恢复时不再重跑已经完成的节点。"""
    thread = uuid.uuid4()
    classified = 0

    async def classify(goal: str) -> str:
        nonlocal classified
        classified += 1
        return EDIT

    monkeypatch.setattr(graph, "classify", classify)
    monkeypatch.setattr(graph, "planner", lambda: BrokenPlanner())

    with pytest.raises(RuntimeError, match="模型暂不可用"):
        await graph.run("垂直翻转画面", context, thread_id=thread)

    snapshot = await graph.snapshot(thread)
    assert snapshot is not None
    assert snapshot.next == ("plan",)
    assert classified == 1

    monkeypatch.setattr(
        graph,
        "planner",
        lambda: FakePlanner(tool_call("flip_layer", {"direction": "vertical"})),
    )
    result = await graph.resume(thread, context)

    assert result is not None
    reply, steps, kind = result
    assert kind == EDIT
    assert [step["tool"] for step in steps] == ["flip_layer"]
    assert "翻转" in reply
    # 分流没有重跑：恢复从失败的规划节点接着执行
    assert classified == 1
    # 每个节点都留了快照，可以直接回溯这一轮是怎么规划出来的
    assert len(await graph.history(thread)) >= 3
    await checkpoint_store.delete(thread)


async def test_service_resumes_an_interrupted_planning_turn(signed_in, monkeypatch):
    """服务重启后，规划到一半的轮次从检查点续跑并正常落库。"""
    session, turn_id = await running_turn(signed_in)
    monkeypatch.setattr(graph, "classify", channel(EDIT))
    monkeypatch.setattr(graph, "planner", lambda: BrokenPlanner())

    with pytest.raises(RuntimeError, match="模型暂不可用"):
        await graph.run("垂直翻转画面", context, thread_id=turn_id)
    assert (await graph.snapshot(turn_id)).next == ("plan",)

    monkeypatch.setattr(
        graph,
        "planner",
        lambda: FakePlanner(tool_call("flip_layer", {"direction": "vertical"})),
    )
    await agent_service.recover()
    await drain()

    turn = await stored_turn(signed_in, session["id"], turn_id)
    assert turn["status"] == "succeeded"
    assert [step["tool"] for step in turn["steps"]] == ["flip_layer"]
    assert "翻转" in turn["reply"]
    await checkpoint_store.delete(turn_id)


async def test_service_fails_a_turn_without_any_checkpoint(signed_in):
    """没有检查点的老轮次无从续跑，必须标失败，不能永远停在「正在理解」。"""
    session, turn_id = await running_turn(signed_in, "无法续跑的旧指令")

    await agent_service.recover()
    await drain()

    turn = await stored_turn(signed_in, session["id"], turn_id)
    assert turn["status"] == "failed"
    assert "中断" in turn["error"]


async def test_service_reconciles_finished_tool_runs(signed_in, monkeypatch):
    """工具已经跑完但没来得及推进计划时，启动恢复要把步骤状态补齐并结算。"""
    monkeypatch.setattr(graph, "classify", channel(EDIT))
    monkeypatch.setattr(
        graph, "planner", lambda: FakePlanner(tool_call("generate_image", PROMPT))
    )
    session = await open_session(signed_in)
    turn = await send(signed_in, session["id"], "换一张")
    assert turn["status"] == "running"
    run_id = uuid.UUID(turn["steps"][0]["run_id"])

    # 模拟 worker 完成落库后进程退出：continue_plan 没来得及把计划推进下去
    async with SessionFactory() as db:
        run = await db.get(ToolRun, run_id)
        await runs_service.finish(db, run, status=RunStatus.SUCCEEDED, result={})

    await agent_service.recover()
    await drain()

    settled = await stored_turn(signed_in, session["id"], uuid.UUID(turn["id"]))
    assert settled["status"] == "succeeded"
    assert [step["status"] for step in settled["steps"]] == ["succeeded"]
    await checkpoint_store.delete(turn["id"])