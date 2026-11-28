import asyncio
import logging
import uuid
from collections.abc import Coroutine
from functools import partial

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm.attributes import flag_modified

from app import agent, events
from app.agent import checkpoint as checkpoint_store
from app.agent import plan as plan_mod
from app.db import SessionFactory
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


# 后台在跑的规划任务。测试等它们跑完再断言，不需要额外轮询
pending_tasks: set[asyncio.Task] = set()

# turn id → 还在规划阶段的任务：删除轮次时顺手掐断，别让它规划完再把步骤提交出去
planning_tasks: dict[uuid.UUID, asyncio.Task] = {}


async def respond(session: AsyncSession, record: EditSession, goal: str) -> AgentRun:
    """先落一条进行中的空轮次立刻返回，规划与执行交给后台逐段推事件。

    模型跑完才有答复与计划，把这好几秒留在请求里，前端看起来就是卡住了。
    """
    turn = AgentRun(
        user_id=record.user_id,
        session_id=record.id,
        revision=record.revision,
        goal=goal,
        status=RunStatus.RUNNING,
    )
    session.add(turn)
    await session.commit()
    await session.refresh(turn)

    # 任务里另开一个 session：请求作用域的连接在响应返回时就关了
    _spawn(_guard(_plan_turn(turn.id, record.id, goal)), turn.id)
    return turn


def _spawn(coro: Coroutine, turn_id: uuid.UUID | None = None) -> None:
    task = asyncio.create_task(coro)
    pending_tasks.add(task)
    task.add_done_callback(pending_tasks.discard)
    if turn_id is not None:
        planning_tasks[turn_id] = task
        task.add_done_callback(lambda _done, key=turn_id: planning_tasks.pop(key, None))


async def _guard(coro: Coroutine) -> None:
    """后台任务自己收尾，不让异常掉到 loop 里只留一行没人看的日志。"""
    try:
        await coro
    except asyncio.CancelledError:
        raise
    except Exception:
        logger.exception("后台规划任务异常", exc_info=True)


async def _plan_turn(turn_id: uuid.UUID, session_id: uuid.UUID, goal: str) -> None:
    try:
        await _run_turn(turn_id, session_id, goal)
    except asyncio.CancelledError:
        # 开发时 --reload 会掐断在跑的任务，标成失败而不是永远停在「正在理解…」
        await _interrupt(turn_id)
        raise


async def _run_turn(
    turn_id: uuid.UUID, session_id: uuid.UUID, goal: str, *, resume: bool = False
) -> None:
    """跑一轮指令：边跑边推事件，跑完一次落库。

    resume=True 时从 LangGraph 检查点接着跑，用于服务重启后的规划恢复。
    """
    reply, steps, kind, error = "", [], TurnKind.EDIT, None

    async with SessionFactory() as session:
        record = await session.get(EditSession, session_id)
        if record is None:
            return
        # 先提交释放连接：模型一跑就是好几秒，不能占着连接池等下游
        await session.commit()

        try:
            # 画布摘要传的是函数：闲聊通道不会调用它，也就完全不读画布与选区
            context = partial(describe, session, record)
            emit = partial(_emit, turn_id)
            outcome = (
                await agent.resume(turn_id, context, emit)
                if resume
                else await agent.run(goal, context, emit, thread_id=turn_id)
            )
            if outcome is None:
                error = "服务重启，本轮规划已中断，请重新发送指令"
            else:
                reply, steps, channel = outcome
                kind = TurnKind(channel)
                if steps:
                    steps = await _pin_selection(session, record, steps)
        except agent.PlannerUnavailable as exc:
            error = str(exc)
        except Exception:
            logger.exception("指令规划异常 session_id=%s", session_id)
            await session.rollback()
            error = "规划失败，请重试"

        # rollback 会把对象全部过期，重新取而不是直接改脏属性
        turn = await session.get(AgentRun, turn_id)
        if turn is None:
            return
        turn.reply, turn.plan, turn.kind, turn.error = reply, steps, kind, error
        turn.status = _status_of(error, steps)
        _touch(turn)
        await session.commit()

        if turn.status is RunStatus.RUNNING:
            await _advance(session, turn)

        await _emit(
            turn_id,
            {"type": "done", "status": turn.status.value, "kind": turn.kind.value},
        )


def _status_of(error: str | None, steps: list[dict]) -> RunStatus:
    """多步计划先停住等确认，单步直接开跑，无关提问与失败都没有步骤。"""
    if error:
        return RunStatus.FAILED
    if plan_mod.needs_confirm(steps):
        return RunStatus.QUEUED
    return RunStatus.RUNNING if steps else RunStatus.SUCCEEDED


async def _emit(turn_id: uuid.UUID, event: dict) -> None:
    """推一条过程事件。Redis 抖动不能连带把整轮指令弄失败。"""
    try:
        await events.publish_turn(turn_id, event)
    except Exception:
        logger.warning(
            "对话事件推送失败 turn_id=%s type=%s", turn_id, event.get("type"), exc_info=True
        )


async def _interrupt(turn_id: uuid.UUID) -> None:
    """任务被提前掐断时把轮次标成失败，前端才不会一直等一个不会来的回答。"""
    try:
        async with SessionFactory() as session:
            turn = await session.get(AgentRun, turn_id)
            if turn is None or turn.status.is_terminal:
                return
            turn.status = RunStatus.FAILED
            turn.error = "本轮已中断，请重试"
            await session.commit()
        await _emit(turn_id, {"type": "failed", "error": "本轮已中断，请重试"})
    except Exception:
        logger.warning("中断轮次落库失败 turn_id=%s", turn_id, exc_info=True)


async def recover() -> None:
    """服务重启后接管上一个进程留下的进行中轮次。

    已经出了计划的按落库步骤校准后继续推进；规划到一半的从 LangGraph
    检查点续跑，没有检查点的老轮次直接标失败，前端不会一直停在「正在理解」。
    """
    try:
        async with SessionFactory() as session:
            turns = list(
                await session.scalars(
                    select(AgentRun).where(AgentRun.status == RunStatus.RUNNING)
                )
            )
            pending = [(turn.id, turn.session_id, bool(turn.plan)) for turn in turns]
    except Exception:
        logger.warning("读取未完成轮次失败，跳过启动恢复", exc_info=True)
        return

    for turn_id, session_id, has_plan in pending:
        if has_plan:
            _spawn(_guard(_resume_execution(turn_id)))
        else:
            _spawn(_guard(_resume_planning(turn_id, session_id)))


async def _resume_planning(turn_id: uuid.UUID, session_id: uuid.UUID) -> None:
    """规划中断的轮次：上下文与事件按轮次重新构造，检查点里的状态接着用。"""
    async with SessionFactory() as session:
        turn = await session.get(AgentRun, turn_id)
        if turn is None or turn.status.is_terminal:
            return
        goal = turn.goal
        await session.commit()

    await _emit(turn_id, {"type": "stage", "text": "正在恢复上次未完成的规划"})
    await _run_turn(turn_id, session_id, goal, resume=True)


async def _resume_execution(turn_id: uuid.UUID) -> None:
    """执行中断的轮次：先把步骤状态与工具任务对齐，再推进没跑完的部分。"""
    async with SessionFactory() as session:
        turn = await session.get(AgentRun, turn_id)
        if turn is None or turn.status.is_terminal:
            return

        steps = _copy(turn.plan)
        for step in steps:
            run_id = step.get("run_id")
            if not run_id:
                continue
            run = await session.get(ToolRun, uuid.UUID(run_id))
            if run is not None:
                step["status"] = run.status.value
        turn.plan = steps
        _touch(turn)

        # 队列里还有任务时交给 worker 重投后自然推进，别在这里重复提交
        if any(step["status"] in {plan_mod.QUEUED, plan_mod.RUNNING} for step in steps):
            await session.commit()
            return

        turn.status = RunStatus.RUNNING
        await session.commit()
        await _emit(turn_id, {"type": "stage", "text": "正在恢复上次未完成的执行"})
        await _advance(session, turn)


async def turn_for_user(
    session: AsyncSession, turn_id: uuid.UUID, user_id: uuid.UUID
) -> AgentRun:
    """事件流只拿到 turn id，归属要顺着会话查。"""
    turn = await session.scalar(
        select(AgentRun)
        .join(EditSession, EditSession.id == AgentRun.session_id)
        .where(AgentRun.id == turn_id, EditSession.user_id == user_id)
    )
    if turn is None:
        raise TurnNotFound
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
    """取消这一轮：排队与正在跑的任务都停掉，没开跑的步骤直接划掉。"""
    turn = await _get(session, record, turn_id)
    if turn.status.is_terminal:
        raise CannotCancel
    steps = plan_mod.cancel_remaining(_copy(turn.plan))
    await _cancel_open_runs(session, steps)
    turn.plan = steps
    _touch(turn)
    turn.status = RunStatus(plan_mod.settle(steps))
    await session.commit()
    await session.refresh(turn)
    return turn


async def remove_turn(session: AsyncSession, record: EditSession, turn_id: uuid.UUID) -> None:
    """删掉一条对话。

    还在跑的轮次先停掉：规划中的任务直接掐断，排队与执行中的工具任务走取消，
    然后删行。本轮已落定的画布改动不回退——要回退用撤销。
    """
    turn = await _get(session, record, turn_id)
    task = planning_tasks.pop(turn.id, None)
    if task is not None:
        task.cancel()
        # 等它真正停下来再读计划：掐断前它可能已经把步骤提交出去了
        await asyncio.gather(task, return_exceptions=True)
        await session.refresh(turn)
    await _cancel_open_runs(session, _copy(turn.plan))
    turn_id = turn.id
    await session.delete(turn)
    await session.commit()
    # 检查点不跟外键走，轮次删掉后主动清一遍
    await checkpoint_store.delete(turn_id)


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
    """取最近 limit 轮，仍按时间正序返回。

    倒着取再翻回来：顺着 created_at 取前 N 条拿到的是最早的 N 轮，
    对话多了以后刚发的指令根本就不会出现在返回里。
    """
    result = await session.scalars(
        select(AgentRun)
        .where(AgentRun.session_id == record.id)
        .order_by(AgentRun.created_at.desc())
        .limit(limit)
    )
    return list(result)[::-1]


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


async def _cancel_open_runs(session: AsyncSession, steps: list[dict]) -> None:
    """把排队与正在跑的工具任务都取消掉，步骤状态跟着落定。

    正在跑的那个（如读到一半的智能修图）先落库已取消、再给 worker 递中止信号；
    worker 侧有配套守卫，迟到的结果不会写回画布。
    """
    for step in steps:
        if not step.get("run_id") or step["status"] not in {plan_mod.QUEUED, plan_mod.RUNNING}:
            continue
        run = await session.get(ToolRun, uuid.UUID(step["run_id"]))
        if run is None or run.status.is_terminal:
            continue
        if await runs.cancel(session, run):
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
