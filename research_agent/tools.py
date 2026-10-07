"""研究助手的工具集。

================================ 这一层在干什么 ================================

把「能力」包装成「Agent 能调用的工具」。整个转换过程是这样的：

    你写的 Python 函数
        ↓  @tool 装饰器读取【函数签名 + 类型注解 + docstring】
    一份 JSON Schema（OpenAI Function Calling 格式）
        ↓  随每次请求一起发给大模型
    LLM 看到："有个叫 search_web 的工具，接受 query 字符串参数，
              描述是『联网搜索最新信息』"
        ↓  LLM 判断该用它，输出结构化调用请求
    {"name": "search_web", "arguments": {"query": "2026年诺贝尔物理学奖"}}
        ↓  LangChain 解析这个 JSON，找到对应函数并执行
    函数返回值（字符串）
        ↓  作为"观察结果"塞回上下文
    LLM 基于这个结果继续推理

【最重要的一条经验】
    **docstring 不是写给人看的注释，它是写给 LLM 看的说明书。**
    LLM 决定调不调、调哪个工具，**完全依赖这段文字**。
    两个功能相近的工具（比如这里的"本地检索"和"联网搜索"），
    如果在 docstring 里没说清各自的使用场景，LLM 就会乱选 ——
    该查本地的时候它去联网，该联网的时候它翻本地库。

    所以下面每个工具的 docstring 都刻意写了三段：
        ① 这个工具是干什么的
        ② **什么时候该用它**
        ③ **什么时候不该用它**（← 这一段最容易被忽略，但最重要）
    并明确写出参数该怎么填。
"""

import asyncio
import logging
import sys
import threading
from pathlib import Path
from typing import List

from langchain.tools import tool

from research_agent.resilience import ToolPermanentError, resilient
from research_agent.settings import agent_settings

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# 路径处理：让 `python research_agent/agent.py` 和
#              `python -m research_agent.agent`
# 两种启动方式都能正常 import 到项目里的 app 包。
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))


# ===========================================================================
# 工具一：本地知识库检索（复用你已经做好的 RAG 系统）
# ===========================================================================


class _RagHolder:
    """RAG 链的懒加载单例。

    【为什么要懒加载而不是模块导入时就初始化？】
        RAGChain(warm=True) 会加载 BGE 模型、打开 Milvus、重建 BM25 倒排索引，
        首次运行还要从 HuggingFace 下载权重 —— 慢则几十秒。

        如果放在模块顶层，那么**只要 import 这个文件就会卡住**，
        哪怕你这一轮只是想联网搜个天气。
        更糟的是：如果向量库有问题，整个 Agent 连启动都启动不了。

        懒加载把这份代价推迟到"第一次真的用到本地检索时"，
        而且失败被隔离在工具内部 —— Agent 本身照常工作，
        只是这一个工具会返回"初始化失败"。
    """

    _instance = None
    _error: Exception | None = None
    _lock = threading.Lock()

    @classmethod
    def get(cls):
        if cls._instance is None and cls._error is None:
            with cls._lock:
                if cls._instance is None and cls._error is None:
                    try:
                        # 延迟 import：避免 Agent 启动时就拉起一大堆 RAG 依赖
                        from app.rag_chain import RAGChain

                        logger.info("正在初始化本地知识库（首次较慢，需加载 Embedding 模型）...")
                        cls._instance = RAGChain(warm=True)
                        logger.info("本地知识库就绪。")
                    except Exception as error:  # noqa: BLE001
                        cls._error = error
                        logger.error("本地知识库初始化失败：%s", error)

        if cls._error is not None:
            # 初始化失败属于永久性故障 —— 重试不会让模型突然下载成功，
            # 所以抛 Permanent，让容错层直接走降级分支，不做无谓重试
            raise ToolPermanentError(
                f"本地知识库初始化失败：{type(cls._error).__name__}: {cls._error}"
            )
        return cls._instance

    @classmethod
    async def aget(cls):
        # 模型加载和 Milvus 初始化没有异步 API，移入工作线程。
        return await asyncio.to_thread(cls.get)


def _format_rag_result(result: dict, max_sources: int = 3) -> str:
    """把 RAGChain.answer() 的返回字典整理成一段适合给 LLM 读的文本。

    【为什么要把结构化数据"拍平"成文本？】
        工具的返回值最终会变成 LLM 上下文里的一段字符。
        直接把整个 dict 用 json.dumps 丢过去也能跑，但：
          - 字段名（num_documents / transform_mode / rrf_score）对 LLM 是噪音；
          - LLM 真正需要的是「答案」+「这段话出自哪」。

        所以这里做一次"信息提纯"：只留下有用的部分，
        并且**明确标注这是本地知识库的结果**，方便 LLM 在后续推理中区分来源。
    """
    answer = (result.get("answer") or "").strip()
    sources = result.get("sources") or []

    lines = ["【本地知识库检索结果】"]

    if answer:
        lines.append(f"答案：{answer}")
    else:
        lines.append("答案：（本地知识库中没有找到相关内容）")

    if sources:
        lines.append("")
        lines.append("出处：")
        for i, item in enumerate(sources[:max_sources], 1):
            filename = item.get("filename") or item.get("source") or "未知文件"
            page = item.get("page")
            location = f"{filename} 第 {page} 页" if page else filename
            # 精排分数能直观反映"这条有多相关"，LLM 看到低分就知道别太依赖它
            score = item.get("relevance_score")
            score_text = f"（相关度 {score:.2f}）" if isinstance(score, (int, float)) else ""
            lines.append(f"  [{i}] {location}{score_text}")
            snippet = (item.get("snippet") or "").strip().replace("\n", " ")
            if snippet:
                lines.append(f"      「{snippet}」")
    else:
        lines.append("")
        lines.append("出处：（无 — 说明知识库里没有匹配的文档，或知识库还是空的）")

    # 忠实性自检结果也告诉 LLM，让它知道这个答案的可靠程度
    groundedness = result.get("groundedness")
    if groundedness and groundedness not in ("yes", "skipped"):
        lines.append("")
        lines.append(
            f"注意：该答案未通过忠实性自检（{groundedness}），"
            "可能存在推测成分，建议结合联网搜索交叉验证。"
        )

    return "\n".join(lines)


@tool
@resilient(
    "search_local_knowledge",
    fallback_hint="本地知识库不可用时，改用 search_web 联网搜索。",
)
async def search_local_knowledge(query: str) -> str:
    """在【本地知识库】中检索用户自己上传的文档资料。

    知识库内容：用户事先上传并索引过的文件（学术论文、技术文档、个人笔记等）。
    这些资料外网搜不到，只能通过本工具获取。

    什么时候用（优先考虑）：
    - 问题涉及专业知识、学术概念、论文内容、技术细节
    - 问题里出现了具体的人名/术语/方法名，像是"在某份资料里看到过"
    - 用户说"根据我的文档""我的资料里""知识库里"之类的话
    - 需要给出**准确出处**（本工具会返回文件名和页码）

    什么时候不要用：
    - 问的是实时信息：今天的新闻、当前股价、最新版本号、天气 —— 这些本地库不会有
    - 上一轮已经搜过同一个问题，且结果为空 —— 别重复搜，换个工具或换个问法
    - 只是闲聊、打招呼、算术 —— 直接回答即可，不必调工具

    参数 query：
        填一句**陈述性的、包含关键词**的检索语句，而不是疑问句。
        好：「SAR 图像相干斑抑制的深度学习方法」
        差：「SAR图像相干斑抑制有哪些深度学习方法呀？」（口语化会干扰向量检索）
    """
    rag = await _RagHolder.aget()
    logger.info("[本地检索] query=%r", query)

    result = await rag.aanswer(query, include_documents=False)
    num_docs = result.get("num_documents", 0)
    logger.info("[本地检索] 命中 %d 篇文档", num_docs)

    return _format_rag_result(result)


# ===========================================================================
# 工具二：联网搜索（Tavily）
# ===========================================================================


class _TavilyHolder:
    """Tavily 客户端的懒加载单例（理由同 _RagHolder：避免 import 时就联网）。"""

    _client = None

    @classmethod
    def get(cls):
        if cls._client is None:
            if not agent_settings.web_search_ready:
                # 这是**配置问题**，不是临时故障 —— 重试一万次也没用，
                # 必须明确告诉用户去配 key
                raise ToolPermanentError(
                    "未配置 TAVILY_API_KEY。请在项目根目录的 .env 里加一行："
                    "TAVILY_API_KEY=tvly-你的key（免费申请：https://tavily.com）"
                )

            from tavily import AsyncTavilyClient

            cls._client = AsyncTavilyClient(api_key=agent_settings.TAVILY_API_KEY)
        return cls._client

    @classmethod
    async def close(cls) -> None:
        if cls._client is not None:
            await cls._client.close()
            cls._client = None


def _format_web_results(query: str, payload: dict) -> str:
    """把 Tavily 的返回整理成给 LLM 读的文本。

    Tavily 的返回结构（简化）：
        {
          "query": "...",
          "answer": "  可直接使用的简短答案（可选，取决于 include_answer）",
          "results": [
             {"title": "...", "url": "...", "content": "...", "score": 0.93},
             ...
          ]
        }

    【这里最重要的设计是"截断"】
        每条搜索结果的 content 可能有一两千字，3 条就是 5000+ 字。
        Agent 还要留着上下文做多轮推理，全塞进去很快就会撑爆窗口，
        而且大量无关文字会稀释关键信息。
        所以每条只保留前 WEB_SNIPPET_CHARS 个字符（默认 800），
        并明确标注"已截断"—— 让 LLM 知道后面还有内容，
        必要时可以换个更精确的查询词重新搜。
    """
    lines = [f"【联网搜索结果】查询：{query}"]

    # Tavily 自带的 AI 摘要（include_answer=True 时才有），质量通常不错，放最前面
    answer = (payload.get("answer") or "").strip()
    if answer:
        lines.append(f"摘要：{answer}")

    results: List[dict] = payload.get("results") or []
    if not results:
        lines.append("（没有搜到结果）")
        return "\n".join(lines)

    limit = agent_settings.WEB_SNIPPET_CHARS
    for i, item in enumerate(results, 1):
        title = (item.get("title") or "无标题").strip()
        url = (item.get("url") or "").strip()
        content = (item.get("content") or "").strip().replace("\n", " ")
        truncated = len(content) > limit
        if truncated:
            content = content[:limit] + " …（已截断）"

        lines.append("")
        lines.append(f"[{i}] {title}")
        if url:
            lines.append(f"    来源：{url}")
        if content:
            lines.append(f"    内容：{content}")

    return "\n".join(lines)


@tool
@resilient(
    "search_web",
    fallback_hint="联网搜索不可用时，改用 search_local_knowledge 查本地知识库。",
)
async def search_web(query: str, max_results: int = 3) -> str:
    """通过【互联网搜索】获取最新、公开的信息。

    数据来源：Tavily 搜索 API，结果是实时抓取的网页内容。

    什么时候用：
    - 实时信息：最新新闻、当前价格、最近的发布/事件
    - 本地知识库明显没有的内容（通用常识、公开资料、行业动态）
    - 需要**交叉验证**本地知识库里的说法是否过时
    - 上一轮 search_local_knowledge 返回"没有找到相关内容"

    什么时候不要用：
    - 问题完全能在本地知识库找到（先用本地，更快更准且有出处）
    - 需要精确出处页码的学术问题（联网结果没有稳定页码）
    - 纯粹的推理、计算、写作任务（不需要外部信息）

    参数 query：
        填**搜索引擎友好的关键词组合**，不要用完整问句。
        好：「2026 诺贝尔物理学奖 获奖者」
        差：「请问2026年的诺贝尔物理学奖颁给了谁？」
    参数 max_results：
        返回几条结果，默认 3。简单问题 2~3 条足够；
        需要多角度对比时最多给到 5 —— **别贪多**，
        每条都要占用上下文，条数越多留给推理的空间越少。
    """
    client = _TavilyHolder.get()

    # 把 max_results 夹在合理区间内。
    # 【为什么必须做这件事】参数是 LLM 填的，它完全可能填 50 或 -1。
    # 不夹住的话，要么上下文被撑爆，要么 API 直接报参数错误。
    safe_max = max(1, min(int(max_results), 5))

    logger.info("[联网搜索] query=%r max_results=%d", query, safe_max)

    payload = await client.search(
        query=query,
        max_results=safe_max,
        search_depth=agent_settings.WEB_SEARCH_DEPTH,
        # include_answer=True 让 Tavily 顺带返回一段 AI 摘要，
        # 这段摘要往往比原始片段更适合直接喂给 LLM
        include_answer=True,
        timeout=agent_settings.TOOL_TIMEOUT,
    )

    results = payload.get("results") or []
    logger.info("[联网搜索] 返回 %d 条结果", len(results))

    return _format_web_results(query, payload)


# ---------------------------------------------------------------------------
# 工具清单：Agent 组装时直接用这个列表
# ---------------------------------------------------------------------------
ALL_TOOLS = [search_local_knowledge, search_web]


def warmup_local_knowledge(verbose: bool = True) -> tuple[bool, str]:
    """提前初始化本地知识库，把"首次加载模型"的代价挪到启动阶段。

    【为什么需要这个函数】
        实测数据：
            第一次调用本地检索 —— 30+ 秒（要把 BGE 模型读进内存、
                                    打开 Milvus、重建 BM25 倒排索引）
            之后每次调用     —— 1.6 秒

        这 30 秒如果发生在**用户提完问、正在等答案**的时候，
        体验是灾难性的 —— 用户会以为程序卡死了。
        所以把它提前到启动阶段：启动时多等 30 秒（用户有心理预期），
        之后每次提问都是秒回。

    【为什么不在模块 import 时就做】
        那会让"只是想联网搜个天气"的场景也要先等 30 秒，
        而且一旦 RAG 出问题，连 Agent 都启动不了。
        所以做成显式调用 —— 要不要预热，由使用者决定。

    返回：(是否成功, 说明文字)
    """
    try:
        if verbose:
            print("  ⏳ 正在预热本地知识库（加载 Embedding 模型 + 打开 Milvus）...")
            print("     首次运行需要从 HuggingFace 下载模型权重，可能耗时较久。")
        _RagHolder.get()
        if verbose:
            print("  ✅ 本地知识库就绪。")
        return True, "本地知识库已就绪"
    except Exception as error:  # noqa: BLE001
        message = f"{type(error).__name__}: {error}"
        if verbose:
            print(f"  ⚠️ 本地知识库预热失败：{message}")
            print("     不影响联网搜索工具，Agent 仍可正常工作。")
        return False, message


async def awarmup_local_knowledge(verbose: bool = True) -> tuple[bool, str]:
    """异步预热入口，避免模型加载阻塞 Agent 事件循环。"""
    return await asyncio.to_thread(warmup_local_knowledge, verbose)


async def aclose_tool_clients() -> None:
    """Close async HTTP clients owned by Agent tools."""
    await _TavilyHolder.close()


def describe_tools() -> str:
    """打印工具清单（调试用）：名字 + 参数 + 描述首行。

    这个函数能帮你直观看到"LLM 眼里的工具长什么样"。
    当你怀疑 Agent 选错工具时，先运行它看看描述是不是写得不清楚。
    """
    lines = []
    for t in ALL_TOOLS:
        schema = t.args_schema.model_json_schema() if t.args_schema else {}
        params = list((schema.get("properties") or {}).keys())
        first_line = (t.description or "").strip().splitlines()[0] if t.description else ""
        lines.append(f"  · {t.name}({', '.join(params)})")
        lines.append(f"      {first_line}")
    return "\n".join(lines)
