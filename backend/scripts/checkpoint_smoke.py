"""LangGraph 检查点冒烟：验证快照落库、失败后续跑与历史回溯。

不调用真实模型，分流与规划都用本地桩，只连真实 PostgreSQL。
运行：backend/.venv/bin/python scripts/checkpoint_smoke.py
"""

import asyncio
import sys
import uuid
from pathlib import Path

from langchain_core.messages import AIMessageChunk
from sqlalchemy import text

# 直接 python scripts/checkpoint_smoke.py 时把 backend 根目录放进 import 路径
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.agent import (
    checkpoint,  # noqa: E402
    graph,  # noqa: E402
)
from app.agent.triage import EDIT  # noqa: E402
from app.db import SessionFactory  # noqa: E402


class FakePlanner:
    """只产出一个翻转调用的规划桩。"""

    async def astream(self, messages):
        yield AIMessageChunk(
            content="",
            tool_call_chunks=[
                {
                    "name": "flip_layer",
                    "args": '{"direction": "vertical"}',
                    "index": 0,
                    "id": "call_smoke",
                }
            ],
        )


class BrokenPlanner:
    """进流即抛，模拟规划节点执行到一半进程退出。"""

    async def astream(self, messages):
        raise RuntimeError("smoke: 规划中断")
        yield AIMessageChunk(content="")


async def context() -> str:
    return "画幅 320×240，修订号 1。\n图层：底图（id=base，图像 320×240）\n当前无选区。"


async def main() -> None:
    original = (graph.classify, graph.planner)
    thread = uuid.uuid4()
    classified = 0

    async def classify(goal: str) -> str:
        nonlocal classified
        classified += 1
        return EDIT

    try:
        graph.classify = classify
        graph.planner = lambda: BrokenPlanner()
        try:
            await graph.run("垂直翻转画面", context, thread_id=thread)
        except RuntimeError:
            pass
        else:
            raise AssertionError("中断没有向上抛出")

        snapshot = await graph.snapshot(thread)
        assert snapshot is not None, "没有写入检查点"
        assert snapshot.next == ("plan",), f"检查点位置不对：{snapshot.next}"
        assert classified == 1, "分流次数不对"
        print("[1/4] 规划中断后检查点停在 plan 之前  OK")

        graph.planner = lambda: FakePlanner()
        result = await graph.resume(thread, context)
        assert result is not None, "检查点无法恢复"
        reply, steps, kind = result
        assert [step["tool"] for step in steps] == ["flip_layer"], f"恢复结果不对：{steps}"
        assert kind == EDIT and "翻转" in reply, "恢复后的答复不对"
        assert classified == 1, "恢复时把分流节点重跑了一遍"
        print("[2/4] 从检查点续跑，已完成的分流没有重跑  OK")

        async with SessionFactory() as db:
            rows = await db.scalar(
                text("select count(*) from checkpoints where thread_id = :thread"),
                {"thread": str(thread)},
            )
        assert rows and rows >= 3, f"checkpoints 表只有 {rows} 行"
        print(f"[3/4] checkpoints 表持久化 {rows} 个快照  OK")

        history = await graph.history(thread)
        assert len(history) >= 3, f"历史只有 {len(history)} 份"
        print(f"[4/4] 历史快照可回溯，共 {len(history)} 份  OK")
        print("SMOKE OK")
    finally:
        graph.classify, graph.planner = original
        await checkpoint.delete(thread)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as exc:
        print(f"SMOKE FAIL: {exc}")
        raise SystemExit(1) from exc