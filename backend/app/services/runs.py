import uuid
from datetime import UTC, datetime

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app import events
from app.models.tool_run import RunStatus, ToolRun
from app.queue import abort_job


class RunNotFound(Exception):
    pass


def snapshot(run: ToolRun) -> dict:
    return {
        "id": str(run.id),
        "tool": run.tool,
        "status": run.status,
        "progress": run.progress,
        "stage": run.stage,
        "error": run.error,
        "result": run.result or {},
    }


async def create(
    session: AsyncSession,
    user_id: uuid.UUID,
    tool: str,
    params: dict,
    session_id: uuid.UUID | None = None,
) -> ToolRun:
    run = ToolRun(
        user_id=user_id, session_id=session_id, tool=tool, params=params, stage="等待开始"
    )
    session.add(run)
    await session.commit()
    await session.refresh(run)
    return run


async def get(session: AsyncSession, run_id: uuid.UUID, user_id: uuid.UUID) -> ToolRun:
    run = await session.scalar(
        select(ToolRun).where(ToolRun.id == run_id, ToolRun.user_id == user_id)
    )
    if run is None:
        raise RunNotFound
    return run


async def load(session: AsyncSession, run_id: uuid.UUID) -> ToolRun:
    run = await session.get(ToolRun, run_id)
    if run is None:
        raise RunNotFound
    return run


async def _commit(session: AsyncSession, run: ToolRun) -> None:
    await session.commit()
    await events.publish(run.id, snapshot(run))


async def start(session: AsyncSession, run: ToolRun) -> bool:
    """把任务置为进行中。返回 False 表示排队期间已被取消，调用方不要再执行。"""
    await session.refresh(run)
    if run.status is RunStatus.CANCELED:
        return False
    run.status = RunStatus.RUNNING
    run.started_at = datetime.now(UTC)
    run.progress = 5
    run.stage = "已开始"
    await _commit(session, run)
    return True


async def report(
    session: AsyncSession,
    run: ToolRun,
    progress: int,
    stage: str,
    result: dict | None = None,
) -> None:
    run.progress = max(run.progress, progress)
    run.stage = stage
    if result is not None:
        run.result = result
    await _commit(session, run)


async def canceled(session: AsyncSession, run: ToolRun) -> bool:
    """任务是否在运行途中被取消。取消请求来自另一个进程，必须重读数据库才看得到。"""
    await session.refresh(run)
    return run.status is RunStatus.CANCELED


async def cancel(session: AsyncSession, run: ToolRun) -> bool:
    """取消排队或正在跑的任务：先把「已取消」落库，再给 worker 递中止信号。

    正在跑的那个真正被掐断靠 worker 轮询中止信号（约半秒一次）；
    即使它抢在掐断前跑完，start/finish 的守卫也不让迟到的结果复活这一行。
    已是终态时返回 False。
    """
    await session.refresh(run)
    if run.status.is_terminal:
        return False
    run.status = RunStatus.CANCELED
    run.error = "已取消"
    run.stage = "已结束"
    run.finished_at = datetime.now(UTC)
    await _commit(session, run)
    await abort_job(run.id)
    return True


async def finish(
    session: AsyncSession,
    run: ToolRun,
    *,
    status: RunStatus,
    result: dict | None = None,
    error: str | None = None,
) -> bool:
    """收尾。返回 False 表示任务已被别处取消，迟到的成功/失败结果不写回。"""
    await session.refresh(run)
    if run.status is RunStatus.CANCELED and status is not RunStatus.CANCELED:
        return False
    run.status = status
    run.result = result or {}
    run.error = error
    run.progress = 100 if status is RunStatus.SUCCEEDED else run.progress
    run.stage = "已完成" if status is RunStatus.SUCCEEDED else "已结束"
    run.finished_at = datetime.now(UTC)
    await _commit(session, run)
    return True
