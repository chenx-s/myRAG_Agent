import os
import unittest
from unittest.mock import AsyncMock, patch

# Tests must not emit external LangSmith traces from the developer's .env.
os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from app.rag_chain import RAGChain


class AsyncRAGTests(unittest.IsolatedAsyncioTestCase):
    async def test_empty_retrieval_runs_through_async_graph(self):
        rag = RAGChain(warm=False)

        with (
            patch("app.rag_chain.settings.QUERY_TRANSFORM", "none"),
            patch("app.rag_chain.settings.MAX_RETRIES", 0),
            patch(
                "app.rag_chain.amulti_query_search",
                new=AsyncMock(return_value=[]),
            ),
        ):
            result = await rag.aanswer("没有命中的问题")

        self.assertEqual(result["num_documents"], 0)
        self.assertEqual(result["groundedness"], "skipped")
        self.assertEqual(result["queries"], ["没有命中的问题"])


if __name__ == "__main__":
    unittest.main()
