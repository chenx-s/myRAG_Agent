"""研究助手 Agent —— 集成 RAG（本地知识库）与 Web 搜索。

模块结构：
    settings.py    配置（LLM、Tavily、重试参数、循环上限）
    tools.py       两个工具：search_local_knowledge / search_web
    resilience.py  工具容错：故障分类 → 重试 → 降级
    agent.py       Agent 组装（ReAct 循环 + Memory）与命令行交互

快速开始：
    # 1. 在项目根目录 .env 里加一行（免费申请 https://tavily.com）
    #    TAVILY_API_KEY=tvly-xxxxxxxx
    # 2. 运行
    python -m research_agent.agent
"""

from research_agent.agent import ResearchAgent, build_agent
from research_agent.tools import ALL_TOOLS, search_local_knowledge, search_web

__all__ = [
    "ResearchAgent",
    "build_agent",
    "ALL_TOOLS",
    "search_local_knowledge",
    "search_web",
]
