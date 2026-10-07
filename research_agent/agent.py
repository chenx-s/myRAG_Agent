"""研究助手 Agent 

================================ 这个 Agent 有什么 ================================

    用户提问
        ↓
    ┌──────────────────────────────────────────────────────────┐
    │  create_agent 内置的 ReAct 循环                            │
    │                                                          │
    │   ① 思考（Reason）                                        │
    │      LLM 读问题 + 历史 + 工具清单，决定下一步做什么         │
    │          ↓                                                │
    │   ② 行动（Act）                                           │
    │      如果要查资料 → 输出 function calling 结构化请求        │
    │      如果能直接回答 → 输出最终答案，循环结束                │
    │          ↓                                                │
    │   ③ 观察（Observe）                                       │
    │      执行工具，把返回结果塞回上下文                         │
    │          ↓                                                │
    │   ④ 回到 ①，带着新信息继续思考                             │
    │      （循环直到 LLM 给出最终答案，或达到 recursion_limit）  │
    └──────────────────────────────────────────────────────────┘
        ↓
    最终答案（带来源标注）

    工具：
      · search_local_knowledge —— 查你的本地知识库（你那个 RAG 系统）
      · search_web            —— 联网搜索（Tavily）

    记忆：
      · InMemorySaver + thread_id，同一个 thread 内的多轮对话自动带上下文

    容错：
      · 每个工具都套了 resilient 装饰器，失败会重试、最终降级为"失败说明"，
        而非抛异常让对话中断

【重要】ReAct 是 create_agent 内置的。
    三件事：
      ① 把工具设计好（tools.py）—— 尤其把 docstring 写清楚
      ② 把 system_prompt 写好（本文件）—— 告诉 Agent 该怎么工作
      ③ 把外部依赖兜住（resilience.py）—— 让工具失败不至于炸掉整个流程
"""

import logging
import sys
from typing import List, Optional
import asyncio

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

from app.llm_cache import get_llm_cache
from research_agent.settings import agent_settings
from research_agent.tools import (
    ALL_TOOLS,
    aclose_tool_clients,
    awarmup_local_knowledge,
    describe_tools,
)

# Windows 终端默认可能是 GBK，打印中文/emoji 会 UnicodeEncodeError。
# 这里把标准输出强制成 UTF-8（拿不到 reconfigure 的老环境就跳过）。
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]
except Exception:  # noqa: BLE001
    pass

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("research_agent")



RESEARCH_SYSTEM_PROMPT = """你是一个专业的研究助手，任务是帮用户做资料调研并给出有据可查的回答。

## 你可以使用的工具

1. `search_local_knowledge` —— 检索【本地知识库】
   内容：用户自己上传并索引过的文档（学术论文、技术资料、个人笔记）。
   特点：准确、有页码出处，但只覆盖已上传的资料。

2. `search_web` —— 【联网搜索】互联网
   内容：实时抓取的公开网页。
   特点：信息新、覆盖面广，但没有稳定页码，质量参差不齐。

## 工具选择策略（重要）

- 问题涉及专业知识、学术概念、论文内容 → **先查本地知识库**
- 问题涉及实时信息（新闻、股价、最新动态）→ **直接联网搜索**
- 拿不准用哪个 → 先查本地，本地没有结果再联网
- 本地查不到时**不要放弃**，改用联网搜索再试一次；反之亦然
- 同一个问题不要用同一个工具重复搜索，除非你换了明显不同的关键词
- 简单的闲聊、算术、常识问题 → 直接回答，不必调用工具

## 回答要求

1. **必须标注来源**：区分哪些信息来自「本地知识库」（要注明文件名，
   有页码更好），哪些来自「互联网」（要注明来源网址）。
2. **诚实优先**：如果两个工具都没找到相关信息，直接说明
   "我查了本地知识库和互联网，暂时没有找到相关信息"，
   **绝对不要编造内容，也不要用常识去补全没有依据的细节。**
3. **交叉验证**：当本地知识和联网结果互相矛盾时，指出这个矛盾，
   并说明哪个更可能可靠（比如本地论文可能较旧，联网信息可能更权威）。
4. **先结论后细节**：开头用一两句话给出核心答案，再展开说明。
5. 用中文回答，语言简洁，不要堆砌无关信息。
"""


# ===========================================================================
# 二、组装 Agent
# ===========================================================================


def build_llm() -> ChatOpenAI:
    """创建大模型客户端（智谱 GLM，走 OpenAI 兼容协议）。"""
    if not agent_settings.LLM_API_KEY:
        raise RuntimeError(
            "未找到 LLM_API_KEY。请检查项目根目录 .env 里的 LLM_API_KEY 配置。"
        )

    return ChatOpenAI(
        model=agent_settings.LLM_MODEL,
        api_key=agent_settings.LLM_API_KEY,
        base_url=agent_settings.LLM_BASE_URL,
        temperature=agent_settings.LLM_TEMPERATURE,
        cache=get_llm_cache(),
        # GLM 的思考模式在 Agent 场景下会拖慢工具调用，
        # 且它输出的"思考过程"会混进消息流干扰解析，这里显式关掉
        extra_body={"thinking": {"type": "disabled"}},
    )


def build_agent(checkpointer: Optional[InMemorySaver] = None):
    """组装并返回 Agent（本质是一个编译好的 LangGraph 图）。

    参数：
        checkpointer: 记忆存储器。传同一个实例的多个 Agent 之间不共享记忆，
                      记忆是靠 **thread_id** 隔离的（见 ResearchAgent.chat）。

    返回：
        可调用的 Agent 对象，用 `.invoke({"messages": [...]}, config=...)` 调用。
    """
    return create_agent(
        model=build_llm(),
        tools=ALL_TOOLS,
        system_prompt=RESEARCH_SYSTEM_PROMPT,
        checkpointer=checkpointer or InMemorySaver(),
    )


# ===========================================================================
# 三、对外的封装类
# ===========================================================================


class ResearchAgent:
    """研究助手 Agent 的易用封装。

    相对裸 Agent，它多做了两件事：
      ① 自动管理 thread_id（多轮对话记忆）
      ② 把 LangGraph 返回的原始消息列表，解析成「工具调用轨迹 + 最终答案」
         这种人类能看懂的形式 —— 这对理解 ReAct 循环特别有用。
    """

    def __init__(self, thread_id: str = "research-default"):
        self._checkpointer = InMemorySaver()
        self.agent = build_agent(self._checkpointer)
        self.thread_id = thread_id
        self._sync_runner: asyncio.Runner | None = None

    # ------------------------------------------------------------------ 内部
    def _invoke_config(self) -> dict:
        """构造 invoke 的 config。

        两个参数都很关键：
          thread_id       —— 记忆的隔离键，决定 Agent 记不记得上文
          recursion_limit —— ReAct 循环的步数上限。超出会抛
                             GraphRecursionError。
        """
        return {
            "configurable": {"thread_id": self.thread_id},
            "recursion_limit": agent_settings.RECURSION_LIMIT,
        }

    @staticmethod
    def _parse_trace(messages: List) -> tuple[List[dict], str]:
        """把消息列表拆成 (工具调用轨迹, 最终答案)。

        LangGraph 返回的 messages 是一条时间线，元素类型有三种：
            HumanMessage —— 用户说的话
            AIMessage    —— LLM 的输出。**注意它有两种形态**：
                            · 带 .tool_calls  → 它决定调工具（还没回答）
                            · 不带 tool_calls → 这是最终答案
            ToolMessage  —— 工具执行的结果

        这个"同一个 AIMessage 类承担两种角色"的设计，
        正是 Function Calling 协议的核心：
        模型不是"返回工具调用"或"返回文本"，而是统一输出一条消息，
        消息里要么有 tool_calls、要么有 content。
        """
        trace: List[dict] = []
        final_answer = ""

        # 从后往前找最后一条"没有工具调用"的 AI 消息，那就是最终答案。
        # 为什么不从头找？因为中间轮的 AI 消息都在调工具，
        # 只有最后一条才是真正给用户的回答。
        for msg in reversed(messages):
            if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
                content = msg.content
                # content 有时是字符串，有时是分段结构（带图片等多模态块）
                if isinstance(content, str):
                    final_answer = content
                elif isinstance(content, list):
                    final_answer = "".join(
                        part.get("text", "") if isinstance(part, dict) else str(part)
                        for part in content
                    )
                if final_answer.strip():
                    break

        # 按时间顺序收集工具调用与返回
        pending: dict = {}
        for msg in messages:
            if isinstance(msg, AIMessage) and getattr(msg, "tool_calls", None):
                for call in msg.tool_calls:
                    # call 的结构：{"name": "search_web", "args": {...}, "id": "call_xxx"}
                    pending[call.get("id")] = {
                        "tool": call.get("name"),
                        "args": call.get("args"),
                        "result": None,
                    }
                    trace.append(pending[call.get("id")])
            elif isinstance(msg, ToolMessage):
                # ToolMessage 用 tool_call_id 关联到具体那次调用
                target = pending.get(getattr(msg, "tool_call_id", None))
                if target is not None:
                    target["result"] = msg.content

        return trace, final_answer.strip()

    # ------------------------------------------------------------------ 对外
    def chat(self, question: str, verbose: bool = True) -> dict:
        """同步兼容入口；异步应用应使用 :meth:`achat`."""
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            # Runner 持有一个长期 event loop，使 Redis/httpx 连接池可以跨多轮
            # 同步 chat() 调用安全复用。
            if self._sync_runner is None:
                self._sync_runner = asyncio.Runner()
            return self._sync_runner.run(self.achat(question, verbose))
        raise RuntimeError("事件循环中不能调用 chat()，请改用 await achat()")

    async def achat(self, question: str, verbose: bool = True) -> dict:
        """问一个问题，返回 {"answer": 答案, "trace": 工具调用轨迹}。

        verbose=True 时会把 Agent 的"思考-行动-观察"过程打印到终端，
        这是理解 ReAct 循环最直观的方式。
        """
        try:
            result = await self.agent.ainvoke(
                {"messages": [HumanMessage(content=question)]},
                config=self._invoke_config(),
            )
        except Exception as error:  # noqa: BLE001
            # 兜底：Agent 层（不是工具层）的异常这里捕获。
            # 最常见的是 GraphRecursionError —— Agent 转太多圈被强制停下，
            # 通常意味着工具一直返回无效结果，Agent 在反复重试同一个动作。
            logger.error("Agent 执行失败：%s: %s", type(error).__name__, error)
            return {
                "answer": (
                    f"抱歉，处理这个问题时出错了：{type(error).__name__}: {error}\n"
                    "可能原因：Agent 循环次数超出上限（recursion_limit）。"
                    "可以尝试把问题拆得更具体一些。"
                ),
                "trace": [],
                "error": error,
            }

        messages = result.get("messages", [])
        trace, answer = self._parse_trace(messages)

        if verbose:
            self._print_trace(trace)

        if not answer:
            answer = "（Agent 没有返回文本答案，可能循环次数用尽，请换个问法再试）"

        return {"answer": answer, "trace": trace}

    @staticmethod
    def _print_trace(trace: List[dict]) -> None:
        """把工具调用轨迹打印出来 —— 这就是 ReAct 里的 Act + Observe。"""
        if not trace:
            print("  （本轮没有调用工具，Agent 直接回答）")
            return

        for i, step in enumerate(trace, 1):
            print(f"  🔧 行动 {i}：调用 `{step['tool']}`")
            args = step.get("args") or {}
            for key, value in args.items():
                text = str(value)
                if len(text) > 100:
                    text = text[:100] + " …"
                print(f"        {key} = {text}")

            result = step.get("result")
            if result is None:
                print("        ⚠️ 没有拿到返回结果")
                continue

            # 观察结果通常很长，只显示前几行，让终端保持可读
            lines = str(result).strip().splitlines()
            preview = lines[:6]
            print("  👁 观察：")
            for line in preview:
                print(f"        {line[:120]}")
            if len(lines) > len(preview):
                print(f"        …（还有 {len(lines) - len(preview)} 行）")

    def new_session(self, thread_id: Optional[str] = None) -> None:
        """开启新会话（换一个 thread_id，Agent 就不再记得之前聊过什么）。"""
        import uuid

        self.thread_id = thread_id or f"research-{uuid.uuid4().hex[:8]}"
        logger.info("已切换到新会话：%s", self.thread_id)

    async def aclose(self) -> None:
        """释放 Redis 与工具 HTTP 连接池。"""
        from app.redis_client import close_redis

        try:
            await aclose_tool_clients()
        finally:
            await close_redis()

    def close(self) -> None:
        """释放同步 ``chat`` 创建的长期事件循环。"""
        if self._sync_runner is not None:
            self._sync_runner.run(self.aclose())
            self._sync_runner.close()
            self._sync_runner = None


# ===========================================================================
# 四、命令行交互
# ===========================================================================


HELP_TEXT = """
可用命令：
  /new      开启新会话（清空对话记忆）
  /tools    查看工具清单（LLM 眼里的工具长什么样）
  /warmup   预热本地知识库（首次提问前跑一次，避免干等 30 秒）
  /quiet    切换是否显示工具调用过程
  /help     显示本帮助
  /quit     退出
"""


async def amain() -> int:
    """命令行入口：交互式研究助手。"""
    print("=" * 72)
    print(" 研究助手 Agent  ——  本地知识库 + 联网搜索")
    print("=" * 72)

    if not agent_settings.LLM_API_KEY:
        print("❌ 未配置 LLM_API_KEY，无法启动。请检查项目根目录的 .env。")
        return 1

    print(f"  模型          : {agent_settings.LLM_MODEL}")
    print(f"  联网搜索      : {'✅ 已配置' if agent_settings.web_search_ready else '❌ 未配置 TAVILY_API_KEY'}")
    print(f"  循环上限      : {agent_settings.RECURSION_LIMIT} 步")
    print(f"  工具          : {', '.join(t.name for t in ALL_TOOLS)}")
    print()
    print(HELP_TEXT)

    try:
        agent = ResearchAgent()
    except Exception as error:  # noqa: BLE001
        print(f"❌ Agent 初始化失败：{error}")
        return 1

    # 启动预热：把"首次加载 BGE 模型"的 30 秒挪到这里，
    # 而不是让用户在第一次提问后干等。
    # 想跳过（比如只打算用联网搜索）就把 .env 里的 AGENT_WARMUP_ON_START 设成 false。
    if agent_settings.WARMUP_ON_START:
        await awarmup_local_knowledge(verbose=True)

    verbose = True

    while True:
        try:
            question = (await asyncio.to_thread(input, "\n你 > ")).strip()
        except (EOFError, KeyboardInterrupt):
            print("\n再见。")
            await agent.aclose()
            return 0

        if not question:
            continue

        # ---- 内置命令 ----
        if question in ("/quit", "/exit", "q", "exit"):
            print("再见。")
            await agent.aclose()
            return 0
        if question == "/new":
            agent.new_session()
            print("✅ 已开启新会话，之前的对话记忆已清空。")
            continue
        if question == "/tools":
            print("\n工具清单：")
            print(describe_tools())
            continue
        if question == "/warmup":
            print()
            await awarmup_local_knowledge(verbose=True)
            continue
        if question == "/quiet":
            verbose = not verbose
            print(f"✅ 工具调用过程显示：{'开' if verbose else '关'}")
            continue
        if question == "/help":
            print(HELP_TEXT)
            continue

        # ---- 正常提问 ----
        print("\n🤔 Agent 思考中 ...")
        outcome = await agent.achat(question, verbose=verbose)

        print("\n" + "─" * 72)
        print("助手 >")
        print(outcome["answer"])
        print("─" * 72)


def main() -> int:
    """同步命令行入口；内部始终复用同一个事件循环。"""
    return asyncio.run(amain())


if __name__ == "__main__":
    raise SystemExit(main())
