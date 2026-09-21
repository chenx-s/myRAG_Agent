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
        _redis_client = Redis.from_url(settings.REDIS_URL , decode_response = True)

    return _redis_client


async def close_redis() ->None:
    global _redis_client
    if _redis_client is not None:
        await _redis_client.close()
        _redis_client = None

