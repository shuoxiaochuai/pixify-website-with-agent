from arq.connections import RedisSettings

from app.config import get_settings
from app.tasks import TASKS

settings = get_settings()


class WorkerSettings:
    """ARQ worker 入口。新增异步任务需在 app.tasks.TASKS 中注册。"""

    redis_settings = RedisSettings.from_dsn(settings.redis_url)
    functions = TASKS
    max_jobs = 4
    job_timeout = 300
    keep_result = 3600
    # 用户取消任务时 API 会往 arq:abort 集合里写信号，worker 轮询到就掐断对应任务
    allow_abort_jobs = True
