import asyncio
import os
import unittest

os.environ["LANGSMITH_TRACING"] = "false"
os.environ["LANGCHAIN_TRACING_V2"] = "false"

from langchain_core.messages import AIMessage

from research_agent.agent import ResearchAgent


class DummyAgent:
    def __init__(self):
        self.loop_ids = []

    async def ainvoke(self, payload, config):
        self.loop_ids.append(id(asyncio.get_running_loop()))
        return {"messages": [AIMessage(content="ok")]}


class ResearchAgentAsyncTests(unittest.TestCase):
    def test_sync_chat_reuses_one_event_loop(self):
        agent = ResearchAgent.__new__(ResearchAgent)
        agent.agent = DummyAgent()
        agent.thread_id = "test"
        agent._sync_runner = None

        try:
            self.assertEqual(agent.chat("one", verbose=False)["answer"], "ok")
            self.assertEqual(agent.chat("two", verbose=False)["answer"], "ok")
            self.assertEqual(len(set(agent.agent.loop_ids)), 1)
        finally:
            agent.close()


if __name__ == "__main__":
    unittest.main()
