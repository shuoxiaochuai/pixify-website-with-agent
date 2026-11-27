import uuid

from arq import create_pool
from arq.connections import ArqRedis, RedisSettings
from arq.constants import abort_jobs_ss
from arq.utils import timestamp_ms

from app.config import get_settings

_pool: ArqRedis | None = None


async def queue() -> ArqRedis:
    global _pool
    if _pool is None:
        _pool = await create_pool(RedisSettings.from_dsn(get_settings().redis_url))
    return _pool


async def close_queue() -> None:
    global _pool
    if _pool is not None:
        await _pool.aclose()
        _pool = None


async def enqueue(task: str, run_id: uuid.UUID) -> None:
    """以 run id 作为任务 ID，重复投递同一 run 不会产生第二次执行。"""
    pool = await queue()
    await pool.enqueue_job(task, run_id, _job_id=str(run_id))


async def abort_job(run_id: uuid.UUID) -> None:
    """给 worker 递中止信号：还在排队的会被跳过，正在跑的会被掐断。

    worker 每半秒左右轮询一次这个集合（需开 allow_abort_jobs）。
    不用 arq.jobs.Job.abort()——那个会阻塞轮询到任务真正结束，不能放进 HTTP 请求里。
    """
    pool = await queue()
    await pool.zadd(abort_jobs_ss, {str(run_id): timestamp_ms()})
