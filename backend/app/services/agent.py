import logging
import uuid

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app import agent
from app.agent import plan as plan_mod
from app.layers import Layer, LayerDocument, LayerKind
from app.models import AgentRun, EditSession, ToolRun, TurnKind
from app.models.tool_run import RunStatus
from app.services import runs, selections, tools
from app.tools import spec_of

logger = logging.getLogger(__name__)


class TurnNotFound(Exception):
    pass


class CannotConfirm(Exception):
    pass


class CannotCancel(Exception):
    pass


class CannotRetry(Exception):
    pass


async def respond(session: AsyncSession, record: EditSession, goal: str) -> AgentRun:
    """规划一轮指令并落库。多步计划先停住等确认，单步直接开跑，无关提问只出一句话。"""
    # rollback 会把 ORM 对象全部过期，先取成局部变量，失败轮次才不会因同步懒加载掩盖真实异常
    user_id, session_id, revision = record.user_id, record.id, record.revision
    reply, steps, kind, error = "", [], TurnKind.EDIT, None

    try:
        # 画布摘要传的是函数：闲聊通道不会调用它，也就完全不读画布与选区
        reply, steps, channel = await agent.run(goal, lambda: describe(session, record))
        kind = TurnKind(channel)
        if steps:
            steps = await _pin_selection(session, record, steps)
    except agent.PlannerUnavailable as exc:
        error = str(exc)
    except Exception:
        logger.exception("指令规划异常 session_id=%s", session_id)
        await session.rollback()
        error = "规划失败，请重试"

    if error:
        status = RunStatus.FAILED
    elif plan_mod.needs_confirm(steps):
        status = RunStatus.QUEUED
    elif steps:
        status = RunStatus.RUNNING
    else:
        status = RunStatus.SUCCEEDED

    turn = AgentRun(
        user_id=user_id,
        session_id=session_id,
        revision=revision,
        goal=goal,
        reply=reply,
        plan=steps,
        kind=kind,
        status=status,
        error=error,
    )
    session.add(turn)
    await session.commit()
    await session.refresh(turn)

    if turn.status is RunStatus.RUNNING:
        await _advance(session, turn)
    return turn


async def describe(session: AsyncSession, record: EditSession) -> str:
    """给规划模型看的画布摘要：画幅、修订号、图层清单与选区状态。

    模型只能从这里抄 layer_id，也就只能改摘要里明确列出的图层，编不出「人物」「主体」这种名字。
    """
    document = (
        LayerDocument.model_validate(record.document)
        if record.document
        else LayerDocument(width=0, height=0)
    )
    lines = [
        f"画幅 {document.width}×{document.height}，修订号 {record.revision}。",
        "图层：" + ("；".join(_line_of(layer) for layer in document.layers) or "无"),
    ]
    stored = await selections.get(record.id, record.revision)
    if stored and stored.get("mask_asset_id"):
        markers = len(stored.get("markers") or [])
        lines.append(f"已有选区（标记点 {markers} 个）。" if markers else "已有选区。")
    else:
        lines.append("当前无选区。")
    return "\n".join(lines)


_KIND_LABEL = {"image": "图像", "text": "文字", "shape": "形状"}


def _line_of(layer: Layer) -> str:
    """一条图层描述。id、尺寸、文案与可见性都要列出，模型据此选目标。"""
    kind = _KIND_LABEL.get(layer.kind.value, layer.kind.value)
    bits = [f"{layer.name}（id={layer.id}", f"{kind} {layer.width}×{layer.height}"]
    if layer.kind is LayerKind.TEXT and layer.text:
        bits.append(f"文案「{layer.text}」")
    if not layer.visible:
        bits.append("隐藏")
    if layer.locked:
        bits.append("锁定")
    return "，".join(bits) + "）"


async def _pin_selection(
    session: AsyncSession, record: EditSession, steps: list[dict]
) -> list[dict]:
    """把当轮选区钉进每个需要遮罩的步骤。

    清除选区与修订号递增都发生在第一步之后，显式 id 才能让后续步骤共用同一块区域。
    """
    stored = await selections.get(record.id, record.revision)
    mask_asset_id = (stored or {}).get("mask_asset_id")
    if not mask_asset_id:
        return steps
    for step in steps:
        params = step["params"]
        if _wants_mask(step["tool"]) and not params.get("mask_asset_id"):
            step["params"] = {**params, "mask_asset_id": mask_asset_id}
    return steps


def _wants_mask(tool: str) -> bool:
    return "mask_asset_id" in spec_of(tool).params.model_fields


async def confirm(session: AsyncSession, record: EditSession, turn_id: uuid.UUID) -> AgentRun:
    turn = await _get(session, record, turn_id)
    steps = _copy(turn.plan)
    waiting = [step for step in steps if step["status"] == plan_mod.WAITING]
    if turn.status is not RunStatus.QUEUED and not waiting:
        raise CannotConfirm
    for step in waiting:
        step["status"] = plan_mod.PENDING
        step["approved"] = True
    turn.plan = steps
    turn.status = RunStatus.RUNNING
    _touch(turn)
    await session.commit()
    return await _advance(session, turn)


async def cancel(session: AsyncSession, record: EditSession, turn_id: uuid.UUID) -> AgentRun:
    turn = await _get(session, record, turn_id)
    if turn.status.is_terminal:
        raise CannotCancel
    turn.plan = plan_mod.cancel_remaining(_copy(turn.plan))
    await _cancel_queued_runs(session, turn.plan)
    _touch(turn)
    turn.status = RunStatus(plan_mod.settle(turn.plan))
    await session.commit()
    await session.refresh(turn)
    return turn


async def retry(session: AsyncSession, record: EditSession, turn_id: uuid.UUID) -> AgentRun:
    turn = await _get(session, record, turn_id)
    steps = _copy(turn.plan)
    failed = next((step for step in reversed(steps) if step["status"] == plan_mod.FAILED), None)
    if failed is None:
        raise CannotRetry
    failed["run_id"] = None
    failed["status"] = plan_mod.PENDING
    turn.plan = steps
    turn.status = RunStatus.RUNNING
    turn.error = None
    _touch(turn)
    await session.commit()
    return await _advance(session, turn)


async def continue_plan(session: AsyncSession, run: ToolRun) -> None:
    """某一步结束后推进后续依赖，刷新后也能从已落库的计划接着跑。"""
    if run.session_id is None:
        return
    turn = await _turn_of_run(session, run)
    if turn is None:
        return

    steps = _copy(turn.plan)
    for step in steps:
        if step.get("run_id") == str(run.id):
            step["status"] = run.status.value
    turn.plan = steps
    _touch(turn)
    # 先不落库：同步的下一步（如翻转）跟这次一起提交，前端不会捞到「上一步完了、下一步还没开始」
    await _advance(session, turn)


async def turns_of(session: AsyncSession, record: EditSession, limit: int = 50) -> list[AgentRun]:
    result = await session.scalars(
        select(AgentRun)
        .where(AgentRun.session_id == record.id)
        .order_by(AgentRun.created_at)
        .limit(limit)
    )
    return list(result)


async def _advance(session: AsyncSession, turn: AgentRun) -> AgentRun:
    steps = _copy(turn.plan)
    progressed = True
    while progressed:
        progressed = False
        for step in plan_mod.ready(steps):
            if spec_of(step["tool"]).needs_approval and not step.get("approved"):
                step["status"] = plan_mod.WAITING
                continue
            run = await tools.submit(
                session, turn.user_id, step["tool"], step["params"], turn.session_id
            )
            step["run_id"] = str(run.id)
            step["status"] = run.status.value
            turn.plan = steps
            turn.status = RunStatus.RUNNING
            _touch(turn)
            await session.commit()
            progressed = run.status is RunStatus.SUCCEEDED

    turn.plan = steps
    turn.status = RunStatus(plan_mod.settle(steps))
    waiting = any(step["status"] == plan_mod.WAITING for step in steps)
    if turn.status is RunStatus.QUEUED and waiting:
        turn.reply = turn.reply or "下一步需要确认后再执行。"
    _touch(turn)
    await session.commit()
    await session.refresh(turn)
    return turn


async def _cancel_queued_runs(session: AsyncSession, steps: list[dict]) -> None:
    for step in steps:
        if not step.get("run_id") or step["status"] != plan_mod.QUEUED:
            continue
        run = await session.get(ToolRun, uuid.UUID(step["run_id"]))
        if run is None or run.status is not RunStatus.QUEUED:
            continue
        await runs.finish(session, run, status=RunStatus.CANCELED, error="已取消")
        step["status"] = plan_mod.CANCELED


async def _get(session: AsyncSession, record: EditSession, turn_id: uuid.UUID) -> AgentRun:
    turn = await session.scalar(
        select(AgentRun).where(AgentRun.id == turn_id, AgentRun.session_id == record.id)
    )
    if turn is None:
        raise TurnNotFound
    return turn


async def _turn_of_run(session: AsyncSession, run: ToolRun) -> AgentRun | None:
    turns = await session.scalars(
        select(AgentRun)
        .where(AgentRun.session_id == run.session_id)
        .order_by(AgentRun.created_at.desc())
        .limit(20)
    )
    run_id = str(run.id)
    for turn in turns:
        if any(step.get("run_id") == run_id for step in turn.plan):
            return turn
    return None


def _copy(steps: list) -> list[dict]:
    return [dict(step) for step in steps]


def _touch(turn: AgentRun) -> None:
    flag_modified(turn, "plan")
