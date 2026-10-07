"""研究助手 Agent 的配置。

【为什么单独一个文件，而不是改 app/config.py】
    RAG 系统（app/）和 Agent（research_agent/）是两层东西：
    RAG 负责"从知识库里找资料"，Agent 负责"调度工具、编排流程"。
    分层配置，改 Agent 不会影响 RAG，反之亦然。

    但**大模型配置是共用的** —— Agent 和 RAG 都得调 GLM，
    所以这里读的还是同一批环境变量（LLM_API_KEY / LLM_BASE_URL / LLM_MODEL）。
    这样你在 .env 里改一次模型，两边同时生效，不会出现
    "RAG 用 glm-4.5-air、Agent 用 glm-4 却没人发现"的错配。

"""

import os
from pathlib import Path

from dotenv import load_dotenv

# .env 在项目根目录（本文件在 research_agent/ 下，所以要往上走一层）
_PROJECT_ROOT = Path(__file__).resolve().parent.parent

# load_dotenv 默认不覆盖已存在的环境变量，
# 所以如果系统里已经设了同名变量，以系统环境为准（方便临时覆盖调试）
load_dotenv(_PROJECT_ROOT / ".env")


class AgentSettings:
    """Agent 层配置。读取方式：`from research_agent.settings import agent_settings`"""

    # ============================================================ 大模型
    # 与 RAG 共用同一套变量，保证两边模型一致
    LLM_API_KEY: str = os.getenv("LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    LLM_BASE_URL: str = os.getenv(
        "LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"
    )
    LLM_MODEL: str = os.getenv("LLM_MODEL", "glm-4.5-air")
    # Agent 场景建议给一点点温度，纯 0 会让它过于机械、不爱多轮思考；
    # 但也别太高，否则工具调用的参数会不稳定（比如乱编搜索词）
    LLM_TEMPERATURE: float = float(os.getenv("AGENT_TEMPERATURE", 0.1))

    # ============================================================ Web 搜索（Tavily）
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    # 单次搜索返回几条结果。**别设太大** —— 每条结果几百字，
    # 5 条就能吃掉 2000+ token 上下文，Agent 还要留着空间做多轮推理
    WEB_SEARCH_MAX_RESULTS: int = int(os.getenv("WEB_SEARCH_MAX_RESULTS", 3))
    # Tavily 的搜索深度：
    #   basic   —— 快、便宜，适合一般问题
    #   advanced —— 慢一些，会做更深的抓取和摘要，适合需要细节的问题
    WEB_SEARCH_DEPTH: str = os.getenv("WEB_SEARCH_DEPTH", "basic")
    # 每条搜索结果在喂给 LLM 前，正文截断到多少字符
    WEB_SNIPPET_CHARS: int = int(os.getenv("WEB_SNIPPET_CHARS", 800))

    # ============================================================ 工具容错
    # 网络类错误（超时/限流/5xx）最多重试几次
    TOOL_MAX_ATTEMPTS: int = int(os.getenv("TOOL_MAX_ATTEMPTS", 3))
    # 指数退避：第 1 次 1s，第 2 次 2s，第 3 次 4s（封顶 10s）
    TOOL_RETRY_BASE_DELAY: float = float(os.getenv("TOOL_RETRY_BASE_DELAY", 1.0))
    TOOL_RETRY_MAX_DELAY: float = float(os.getenv("TOOL_RETRY_MAX_DELAY", 10.0))
    # 单次工具调用的硬超时（秒）。Tavily 卡住时不能把整个 Agent 拖死
    TOOL_TIMEOUT: float = float(os.getenv("TOOL_TIMEOUT", 30.0))

    # ============================================================ Agent 行为
    # LangGraph 单次 invoke 的最大步数。
    # Agent 的 ReAct 循环是「思考→调工具→观察→再思考」，
    # 每转一圈至少消耗 2 步（模型 1 步 + 工具 1 步）。
    # 设太小会导致复杂问题被硬截断；设太大会让跑偏的 Agent 空转烧 token。
    # 25 大致允许 8~10 轮工具调用，够研究类任务用。
    RECURSION_LIMIT: int = int(os.getenv("AGENT_RECURSION_LIMIT", 25))
    # 单轮对话最多让 Agent 循环几次（仅用于 CLI 展示提示）
    MAX_ITERATIONS_HINT: int = int(os.getenv("AGENT_MAX_ITERATIONS_HINT", 10))
    # 启动时是否预热本地知识库。
    #   实测：首次加载 BGE 模型要 30+ 秒，之后每次调用只要 1.6 秒。
    #   预热 = 把这 30 秒从"用户提问后干等"挪到"启动时多等一会"。
    #   如果你这轮只想用联网搜索，设成 false 可以秒启动。
    WARMUP_ON_START: bool = os.getenv(
        "AGENT_WARMUP_ON_START", "true"
    ).strip().lower() in ("true", "1", "yes", "on")

    # ============================================================ 派生属性
    @property
    def web_search_ready(self) -> bool:
        """Web 搜索是否真的可用（配了 key 才算）。"""
        return bool(self.TAVILY_API_KEY.strip())


agent_settings = AgentSettings()
