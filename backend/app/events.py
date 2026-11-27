import json
import uuid
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from functools import lru_cache

from redis.asyncio import Redis

from app.config import get_settings

IDLE_TICK = 15.0

# 一轮的过程事件留多久、最多留几条。留着的意义是让晚一步的订阅者能补上开头
_TURN_TTL = 900
_TURN_KEEP = 200


@lru_cache
def redis_client() -> Redis:
    return Redis.from_url(get_settings().redis_url, decode_responses=True)


def _channel(kind: str, target: uuid.UUID) -> str:
    return f"{kind}:{target}"


async def _publish(channel: str, payload: dict) -> None:
    await redis_client().publish(channel, json.dumps(payload))


async def publish(run_id: uuid.UUID, payload: dict) -> None:
    await _publish(_channel("run", run_id), payload)


async def publish_turn(turn_id: uuid.UUID, payload: dict) -> None:
    """把对话过程事件写进 Stream，而不是只发 pub/sub。

    分流与阶段事件在浏览器连上事件流之前就已经推完了，纯 pub/sub 会整段丢掉开头。
    """
    redis = redis_client()
    key = _channel("turn", turn_id)
    await redis.xadd(key, {"data": json.dumps(payload)}, maxlen=_TURN_KEEP, approximate=True)
    await redis.expire(key, _TURN_TTL)


@asynccontextmanager
async def _subscribe(channel: str) -> AsyncIterator[AsyncIterator[dict | None]]:
    """订阅进度。空闲超过 IDLE_TICK 秒时产出 None，供调用方发送心跳。"""
    pubsub = redis_client().pubsub()
    await pubsub.subscribe(channel)

    async def messages() -> AsyncIterator[dict | None]:
        while True:
            message = await pubsub.get_message(ignore_subscribe_messages=True, timeout=IDLE_TICK)
            yield json.loads(message["data"]) if message else None

    try:
        yield messages()
    finally:
        await pubsub.aclose()


@asynccontextmanager
async def subscribe(run_id: uuid.UUID) -> AsyncIterator[AsyncIterator[dict | None]]:
    async with _subscribe(_channel("run", run_id)) as messages:
        yield messages


@asynccontextmanager
async def subscribe_turn(turn_id: uuid.UUID) -> AsyncIterator[AsyncIterator[dict | None]]:
    """从第一条开始回放，跟着写的人走。block 到期的空手而归当作心跳。"""
    key = _channel("turn", turn_id)

    async def entries() -> AsyncIterator[dict | None]:
        cursor = "0"
        while True:
            batch = await redis_client().xread(
                {key: cursor}, count=64, block=int(IDLE_TICK * 1000)
            )
            if not batch:
                yield None
                continue
            for _, items in batch:
                for entry_id, fields in items:
                    cursor = entry_id
                    yield json.loads(fields["data"])

    yield entries()
