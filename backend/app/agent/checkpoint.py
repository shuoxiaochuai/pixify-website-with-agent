"""LangGraph 检查点：把规划图的节点状态落到 PostgreSQL。

一轮指令对应一个 thread_id，每个节点执行完都会存一份快照。
服务重启后可以从最后一个快照接着跑，已经跑完的分流与闲聊节点不再重复执行；
检查点同时保留了每个节点的历史快照，便于排查一轮指令是怎么被规划出来的。
"""

import asyncio
import logging
import sys
import uuid
import warnings
from collections.abc import AsyncIterator
from contextlib import AsyncExitStack, asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from sqlalchemy.engine import make_url

from app.config import get_settings

logger = logging.getLogger(__name__)

if sys.platform == "win32":
    # psycopg 的异步连接不支持 Windows 默认的 ProactorEventLoop。
    # 本地开发与测试统一换成 Selector；线上 Linux 不会执行到这里。
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

_setup_lock = asyncio.Lock()
_setup_done = False


def dsn() -> str:
    """psycopg 不认 SQLAlchemy 的 +asyncpg 方言，连接串去掉驱动后缀再交给它。"""
    url = make_url(get_settings().database_url)
    if url.drivername.startswith("postgresql+"):
        url = url.set(drivername="postgresql")
    return url.render_as_string(hide_password=False)


@asynccontextmanager
async def saver() -> AsyncIterator[AsyncPostgresSaver | None]:
    """借一条检查点连接。

    数据库连不上时返回 None，本轮退回无持久化执行——检查点是可靠性增强，
    不能反过来把规划本身弄失败。
    """
    stack = AsyncExitStack()
    try:
        store = await stack.enter_async_context(
            AsyncPostgresSaver.from_conn_string(dsn())
        )
    except Exception:
        logger.warning("LangGraph 检查点不可用，本轮按无持久化执行", exc_info=True)
        yield None
        return

    try:
        try:
            await _setup(store)
        except Exception:
            logger.warning("检查点表初始化失败，本轮按无持久化执行", exc_info=True)
            yield None
            return
        yield store
    finally:
        await stack.aclose()


async def ensure_ready() -> None:
    """启动时把快照表准备好；失败只记日志，不影响主服务启动。"""
    async with saver() as store:
        _ = store


async def delete(thread_id: uuid.UUID | str) -> None:
    """轮次或会话删除后清掉对应快照，避免检查点表留下孤儿。"""
    try:
        async with saver() as store:
            if store is not None:
                await store.adelete_thread(str(thread_id))
    except Exception:
        logger.warning("清理检查点失败 thread_id=%s", thread_id, exc_info=True)


async def _setup(store: AsyncPostgresSaver) -> None:
    global _setup_done
    if _setup_done:
        return
    async with _setup_lock:
        if _setup_done:
            return
        await store.setup()
        _setup_done = True