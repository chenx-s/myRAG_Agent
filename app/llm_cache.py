"""Redis-backed cache for LangChain LLM/chat-model generations.

The async methods are the primary path used by this project.  Redis failures are
treated as cache misses so a cache outage never becomes an LLM outage.
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass
from typing import Any, Optional

from langchain_core.caches import BaseCache, RETURN_VAL_TYPE
from langchain_core.load import dumps, loads
from langchain_core.messages import AIMessage, AIMessageChunk
from langchain_core.outputs import (
    ChatGeneration,
    ChatGenerationChunk,
    Generation,
    GenerationChunk,
)
from redis import Redis as SyncRedis
from redis.asyncio import Redis

from app.config import settings
from app.redis_client import get_redis

logger = logging.getLogger(__name__)

_ALLOWED_CACHE_OBJECTS = [
    Generation,
    GenerationChunk,
    ChatGeneration,
    ChatGenerationChunk,
    AIMessage,
    AIMessageChunk,
]


@dataclass
class CacheMetrics:
    hits: int = 0
    misses: int = 0
    writes: int = 0
    errors: int = 0
    bypasses: int = 0


class RedisLLMCache(BaseCache):
    """LangChain cache with native async Redis I/O and TTL support."""

    def __init__(
        self,
        *,
        redis_url: str,
        prefix: str,
        ttl_seconds: int,
        async_client: Optional[Redis] = None,
        sync_client: Optional[SyncRedis] = None,
    ) -> None:
        self.redis_url = redis_url
        self.prefix = prefix.rstrip(":")
        self.ttl_seconds = max(0, ttl_seconds)
        self._async_client = async_client
        self._sync_client = sync_client
        self.metrics = CacheMetrics()
        self._warned = False
        self._disabled_until = 0.0

    def _key(self, prompt: str, llm_string: str) -> str:
        digest = hashlib.sha256(
            f"{llm_string}\0{prompt}".encode("utf-8")
        ).hexdigest()
        return f"{self.prefix}:{digest}"

    def _sync_redis(self) -> SyncRedis:
        if self._sync_client is None:
            self._sync_client = SyncRedis.from_url(
                self.redis_url,
                decode_responses=True,
                socket_connect_timeout=settings.REDIS_SOCKET_TIMEOUT_SECONDS,
                socket_timeout=settings.REDIS_SOCKET_TIMEOUT_SECONDS,
            )
        return self._sync_client

    async def _async_redis(self) -> Redis:
        # 注入的客户端仅用于测试；生产环境每次从生命周期管理器取得当前连接池，
        # 避免同步兼容入口创建新 event loop 后复用旧 loop 的连接。
        if self._async_client is not None:
            return self._async_client
        return await get_redis()

    def _record_error(self, operation: str, error: BaseException) -> None:
        self.metrics.errors += 1
        self._disabled_until = time.monotonic() + max(
            0.0, settings.LLM_CACHE_FAILURE_COOLDOWN_SECONDS
        )
        if not self._warned:
            logger.warning(
                "Redis LLM 缓存%s失败，已自动旁路：%s: %s",
                operation,
                type(error).__name__,
                error,
            )
            self._warned = True

    def _is_circuit_open(self) -> bool:
        return time.monotonic() < self._disabled_until

    def _circuit_open(self) -> bool:
        if self._is_circuit_open():
            self.metrics.bypasses += 1
            return True
        return False

    def _record_success(self) -> None:
        self._disabled_until = 0.0
        self._warned = False

    @staticmethod
    def _deserialize(payload: str) -> RETURN_VAL_TYPE:
        value = loads(payload, allowed_objects=_ALLOWED_CACHE_OBJECTS)
        if not isinstance(value, list):
            raise ValueError("缓存内容不是 Generation 列表")
        return value

    # Synchronous methods remain available for scripts that use ``invoke``.
    # The FastAPI and Agent paths use the native async methods below.
    def lookup(self, prompt: str, llm_string: str) -> RETURN_VAL_TYPE | None:
        if self._circuit_open():
            self.metrics.misses += 1
            return None
        try:
            payload = self._sync_redis().get(self._key(prompt, llm_string))
            self._record_success()
            if payload is None:
                self.metrics.misses += 1
                return None
            self.metrics.hits += 1
            return self._deserialize(payload)
        except Exception as error:  # cache must fail open
            self._record_error("读取", error)
            self.metrics.misses += 1
            return None

    def update(
        self,
        prompt: str,
        llm_string: str,
        return_val: RETURN_VAL_TYPE,
    ) -> None:
        if self._circuit_open():
            return
        try:
            kwargs = {"ex": self.ttl_seconds} if self.ttl_seconds else {}
            self._sync_redis().set(
                self._key(prompt, llm_string), dumps(return_val), **kwargs
            )
            self._record_success()
            self.metrics.writes += 1
        except Exception as error:  # cache must fail open
            self._record_error("写入", error)

    def clear(self, **kwargs: Any) -> None:
        if self._circuit_open():
            return
        try:
            client = self._sync_redis()
            keys = list(client.scan_iter(match=f"{self.prefix}:*", count=500))
            if keys:
                client.unlink(*keys)
        except Exception as error:
            self._record_error("清理", error)

    async def alookup(
        self, prompt: str, llm_string: str
    ) -> RETURN_VAL_TYPE | None:
        if self._circuit_open():
            self.metrics.misses += 1
            return None
        try:
            client = await self._async_redis()
            payload = await client.get(self._key(prompt, llm_string))
            self._record_success()
            if payload is None:
                self.metrics.misses += 1
                return None
            self.metrics.hits += 1
            return self._deserialize(payload)
        except Exception as error:  # cache must fail open
            self._record_error("读取", error)
            self.metrics.misses += 1
            return None

    async def aupdate(
        self,
        prompt: str,
        llm_string: str,
        return_val: RETURN_VAL_TYPE,
    ) -> None:
        if self._circuit_open():
            return
        try:
            client = await self._async_redis()
            kwargs = {"ex": self.ttl_seconds} if self.ttl_seconds else {}
            await client.set(
                self._key(prompt, llm_string), dumps(return_val), **kwargs
            )
            self._record_success()
            self.metrics.writes += 1
        except Exception as error:  # cache must fail open
            self._record_error("写入", error)

    async def aclear(self, **kwargs: Any) -> None:
        if self._circuit_open():
            return
        try:
            client = await self._async_redis()
            batch = []
            async for key in client.scan_iter(match=f"{self.prefix}:*", count=500):
                batch.append(key)
                if len(batch) >= 500:
                    await client.unlink(*batch)
                    batch.clear()
            if batch:
                await client.unlink(*batch)
        except Exception as error:
            self._record_error("清理", error)

    def info(self) -> dict[str, Any]:
        return {
            "enabled": True,
            "prefix": self.prefix,
            "ttl_seconds": self.ttl_seconds,
            "hits": self.metrics.hits,
            "misses": self.metrics.misses,
            "writes": self.metrics.writes,
            "errors": self.metrics.errors,
            "bypasses": self.metrics.bypasses,
            "circuit_open": self._is_circuit_open(),
        }


_llm_cache: Optional[RedisLLMCache] = None


def get_llm_cache() -> RedisLLMCache | bool:
    """Return the shared cache, or ``False`` when caching is disabled."""
    global _llm_cache
    if not settings.LLM_CACHE_ENABLED:
        return False
    if _llm_cache is None:
        _llm_cache = RedisLLMCache(
            redis_url=settings.REDIS_URL,
            prefix=settings.LLM_CACHE_PREFIX,
            ttl_seconds=settings.LLM_CACHE_TTL_SECONDS,
        )
    return _llm_cache


def cache_info() -> dict[str, Any]:
    cache = get_llm_cache()
    if cache is False:
        return {"enabled": False}
    return cache.info()
