"""Shared asynchronous Redis connection pool."""

from redis.asyncio import Redis
from typing import Optional
from app.config import settings

_redis_client : Optional[Redis] = None

async def get_redis() -> Redis:
    """
    返回单例 Redis 客户端（连接池）。
    在 FastAPI 的 startup/shutdown 事件中初始化与关闭。
    """
    global _redis_client
    if _redis_client is None:
        _redis_client = Redis.from_url(
            settings.REDIS_URL,
            decode_responses=True,
            socket_connect_timeout=settings.REDIS_SOCKET_TIMEOUT_SECONDS,
            socket_timeout=settings.REDIS_SOCKET_TIMEOUT_SECONDS,
            health_check_interval=30,
        )

    return _redis_client


async def redis_health() -> dict:
    """Return Redis availability without leaking connection credentials."""
    if not settings.LLM_CACHE_ENABLED:
        return {"enabled": False, "connected": False}
    try:
        client = await get_redis()
        await client.ping()
        return {"enabled": True, "connected": True}
    except Exception as error:
        return {
            "enabled": True,
            "connected": False,
            "error": f"{type(error).__name__}: {error}",
        }


async def close_redis() -> None:
    global _redis_client
    if _redis_client is not None:
        close = getattr(_redis_client, "aclose", None)
        if close is None:
            close = _redis_client.close
        await close()
        _redis_client = None

