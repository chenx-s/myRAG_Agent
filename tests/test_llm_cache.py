import unittest

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration

from app.llm_cache import RedisLLMCache


class FakeAsyncRedis:
    def __init__(self):
        self.data = {}
        self.last_ttl = None

    async def get(self, key):
        return self.data.get(key)

    async def set(self, key, value, **kwargs):
        self.data[key] = value
        self.last_ttl = kwargs.get("ex")

    async def scan_iter(self, match, count=10):
        prefix = match.removesuffix("*")
        for key in list(self.data):
            if key.startswith(prefix):
                yield key

    async def unlink(self, *keys):
        for key in keys:
            self.data.pop(key, None)


class BrokenAsyncRedis:
    def __init__(self):
        self.calls = 0

    async def get(self, key):
        self.calls += 1
        raise ConnectionError("redis unavailable")


class RedisLLMCacheTests(unittest.IsolatedAsyncioTestCase):
    def make_cache(self, client):
        return RedisLLMCache(
            redis_url="redis://unused/0",
            prefix="test:llm",
            ttl_seconds=60,
            async_client=client,
        )

    async def test_async_round_trip_and_ttl(self):
        client = FakeAsyncRedis()
        cache = self.make_cache(client)
        generations = [ChatGeneration(message=AIMessage(content="cached answer"))]

        self.assertIsNone(await cache.alookup("prompt", "model"))
        await cache.aupdate("prompt", "model", generations)
        cached = await cache.alookup("prompt", "model")

        self.assertEqual(cached[0].message.content, "cached answer")
        self.assertEqual(client.last_ttl, 60)
        self.assertEqual(cache.metrics.hits, 1)
        self.assertEqual(cache.metrics.misses, 1)
        self.assertEqual(cache.metrics.writes, 1)

    async def test_key_includes_model_configuration(self):
        client = FakeAsyncRedis()
        cache = self.make_cache(client)
        generations = [ChatGeneration(message=AIMessage(content="one"))]

        await cache.aupdate("same prompt", "model-a", generations)

        self.assertIsNotNone(await cache.alookup("same prompt", "model-a"))
        self.assertIsNone(await cache.alookup("same prompt", "model-b"))

    async def test_agent_tool_call_generation_round_trip(self):
        client = FakeAsyncRedis()
        cache = self.make_cache(client)
        generations = [
            ChatGeneration(
                message=AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "search_web",
                            "args": {"query": "async redis"},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                )
            )
        ]

        await cache.aupdate("agent prompt", "agent model", generations)
        cached = await cache.alookup("agent prompt", "agent model")

        self.assertEqual(cached[0].message.tool_calls[0]["name"], "search_web")

    async def test_redis_failure_is_a_cache_miss(self):
        client = BrokenAsyncRedis()
        cache = self.make_cache(client)

        self.assertIsNone(await cache.alookup("prompt", "model"))
        self.assertIsNone(await cache.alookup("another prompt", "model"))
        self.assertEqual(cache.metrics.errors, 1)
        self.assertEqual(cache.metrics.bypasses, 1)
        self.assertEqual(client.calls, 1)

    async def test_clear_only_removes_cache_prefix(self):
        client = FakeAsyncRedis()
        cache = self.make_cache(client)
        client.data = {
            "test:llm:a": "1",
            "test:llm:b": "2",
            "other:key": "3",
        }

        await cache.aclear()

        self.assertEqual(client.data, {"other:key": "3"})


if __name__ == "__main__":
    unittest.main()
