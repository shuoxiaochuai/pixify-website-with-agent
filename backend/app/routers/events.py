import asyncio
import json
import uuid
from collections.abc import AsyncIterator, Callable

from fastapi import APIRouter, HTTPException, Response, status
from fastapi.responses import StreamingResponse

from app import events
from app.db import SessionDep
from app.deps import CurrentUser
from app.models.tool_run import RunStatus
from app.schemas.agent import TurnOut
from app.services import agent as agent_service
from app.services import runs
from app.services.agent import TurnNotFound
from app.services.runs import RunNotFound

router = APIRouter(prefix="/events", tags=["events"])

_HEADERS = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}

# 超时后主动断开，客户端重连时会先收到快照，不会丢状态
_MAX_STREAM_SECONDS = 600.0


def _frame(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def _tail(
    messages: AsyncIterator[dict | None], done: Callable[[dict], bool]
) -> AsyncIterator[str]:
    """心跳、超时与终止。两个事件流只差「什么算结束」。"""
    deadline = asyncio.get_running_loop().time() + _MAX_STREAM_SECONDS
    async for payload in messages:
        if payload is None:
            if asyncio.get_running_loop().time() > deadline:
                return
            yield ": ping\n\n"
            continue
        yield _frame(payload)
        if done(payload):
            return


def _run_done(payload: dict) -> bool:
    return RunStatus(payload["status"]).is_terminal


def _turn_done(payload: dict) -> bool:
    # 后台一轮跑完或跑砸都算结束，前端接着去拉落库结果
    return payload.get("type") in {"done", "failed"}


@router.get("/runs/{run_id}")
async def stream_run(run_id: uuid.UUID, user: CurrentUser, session: SessionDep) -> Response:
    try:
        await runs.get(session, run_id, user.id)
    except RunNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "任务不存在") from exc

    async def stream() -> AsyncIterator[str]:
        # 先订阅再读快照，否则任务在两步之间结束会让连接一直空等
        async with events.subscribe(run_id) as messages:
            run = await runs.get(session, run_id, user.id)
            yield _frame(runs.snapshot(run))
            if run.status.is_terminal:
                return

            async for frame in _tail(messages, _run_done):
                yield frame

    return StreamingResponse(stream(), media_type="text/event-stream", headers=_HEADERS)


@router.get("/turns/{turn_id}")
async def stream_turn(turn_id: uuid.UUID, user: CurrentUser, session: SessionDep) -> Response:
    """一轮指令的过程事件：阶段、通道、步骤名、正文增量。落库结果仍以快照为准。"""
    try:
        await agent_service.turn_for_user(session, turn_id, user.id)
    except TurnNotFound as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "对话不存在") from exc

    async def stream() -> AsyncIterator[str]:
        # 同样先订阅再读快照：这一轮可能已经跑完了
        async with events.subscribe_turn(turn_id) as messages:
            turn = await agent_service.turn_for_user(session, turn_id, user.id)
            yield _frame(
                {"type": "snapshot", "turn": TurnOut.of(turn).model_dump(mode="json")}
            )
            if turn.status.is_terminal:
                return

            async for frame in _tail(messages, _turn_done):
                yield frame

    return StreamingResponse(stream(), media_type="text/event-stream", headers=_HEADERS)
