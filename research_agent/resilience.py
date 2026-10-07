"""工具容错：故障分类 → 重试 → 降级。

================================ 核心思想 ================================

Agent 里的工具会失败，而且**必然会失败**：网络抖一下、API 限流、key 过期、
搜索词太怪导致对方返回 400……如果每次失败都让整个 Agent 崩掉，
这个 Agent 就没法用在真实场景里。

【关键认知】工具失败时，正确的做法**不是抛异常，而是把失败当成一种"结果"返回给 LLM。**

    抛异常的后果：
        异常冒泡到 create_agent → 整轮对话中断 → 用户看到红色报错。
        而且 LLM 完全不知道发生了什么，因为它的回合已经结束了。

    返回失败说明的后果：
        LLM 看到工具返回「搜索失败：网络超时，已重试 3 次」，
        它可以自己决定下一步 —— 换个搜索词重试、改用本地知识库、
        或者直接告诉用户"联网搜索暂时不可用"。
        **决策权交还给 Agent，这才是 Agentic 系统该有的样子。**

【第二个关键认知】不是所有错误都值得重试。

    值得重试（transient 临时性）：
        超时、连接被重置、429 限流、502/503/504
        —— 等一会儿再试很可能就成功了。

    重试也没用（permanent 永久性）：
        API key 无效、没配 key、配额耗尽、参数不合法、403 无权限
        —— 重试 100 次也是同样的结果，只会白白浪费时间、烧配额。

    分不清这两者，就会出现两种典型事故：
        · 全都重试 → 一个 key 过期的配置，白白卡 3 个 30 秒超时
        · 全都不重试 → 网络抖一下就直接给用户报错

【第三个关键认知】重试要有退避（backoff），不能立刻重试。

    对方限流时，你立刻重试等于继续加压，只会被继续拒绝。
    指数退避（1s → 2s → 4s → …）给对方喘息时间，
    而且最后加一点随机抖动，避免多个请求"整齐划一"地同时重试造成惊群。
"""

import asyncio
import functools
import inspect
import logging
import random
import time
from typing import Any, Callable, Tuple

from tenacity import (
    Retrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from research_agent.settings import agent_settings

logger = logging.getLogger(__name__)


# ========================================================================
# 一、异常分类：把所有可能的故障归成两类
# ========================================================================


class ToolTransientError(Exception):
    """临时性故障 —— **值得重试**。

    网络抖动、限流、服务端偶发 5xx 都归到这里。
    """


class ToolPermanentError(Exception):
    """永久性故障 —— **重试没用**。

    key 无效、配额耗尽、参数错误归到这里。重试只会浪费时间。
    """


# 把常见第三方异常映射到上面两类。
# 用「异常类名 + 关键词」双保险：既能识别已知类型，
# 也能兜住那些没被单独定义、只能靠消息文本判断的情况。
_TRANSIENT_HINTS = (
    "timeout", "timed out", "connection", "connect", "temporarily",
    "unavailable", "rate limit", "too many requests", "429",
    "502", "503", "504", "overloaded", "reset by peer", "ssl",
)

_PERMANENT_HINTS = (
    "invalid api key", "unauthorized", "authentication", "api key",
    "quota", "usage limit", "exceeded", "forbidden", "not permitted",
    "400", "401", "403", "bad request", "invalid parameter",
)


def classify_exception(exc: BaseException) -> Tuple[bool, str]:
    """判断异常是否值得重试，并生成一句给 LLM 看的中文说明。

    返回：(是否可重试, 原因说明)

    判断顺序很重要 —— 先看**明确的异常类型**，类型判断不出来时，
    再退化为**关键词匹配**。因为类型判断可靠，关键词判断只是兜底。
    """
    # ---- 第一层：我们自己抛的两类，直接下结论 ----
    if isinstance(exc, ToolPermanentError):
        return False, str(exc) or "工具调用失败（不可重试）"
    if isinstance(exc, ToolTransientError):
        return True, str(exc) or "工具调用暂时失败"

    # ---- 第二层：第三方库的明确异常类型 ----
    # 这些是 tavily-python 自己定义的，见 tavily/errors.py
    exc_type = type(exc).__name__
    try:
        from tavily.errors import (
            BadRequestError,
            ForbiddenError,
            InvalidAPIKeyError,
            MissingAPIKeyError,
            TimeoutError as TavilyTimeoutError,
            UsageLimitExceededError,
        )

        if isinstance(exc, TavilyTimeoutError):
            return True, "联网搜索超时（Tavily 响应太慢）"
        if isinstance(exc, (InvalidAPIKeyError, MissingAPIKeyError)):
            return False, "Tavily API key 无效或未配置（重试无用，请检查 .env 里的 TAVILY_API_KEY）"
        if isinstance(exc, UsageLimitExceededError):
            return False, "Tavily 本月免费额度已用完（重试无用，下月重置或升级套餐）"
        if isinstance(exc, ForbiddenError):
            return False, "Tavily 拒绝访问（可能是 key 权限问题或所在地区限制）"
        if isinstance(exc, BadRequestError):
            return False, f"搜索请求被拒绝（参数不合法）：{exc}"
    except ImportError:
        # tavily 没装也不该让这里炸 —— 会走到下面的关键词兜底
        pass

    # ---- 第三层：标准库的网络异常 ----
    if isinstance(exc, (TimeoutError, ConnectionError)):
        return True, f"网络连接异常：{exc_type}"
    try:
        import requests

        if isinstance(exc, (requests.Timeout, requests.ConnectionError)):
            return True, f"网络请求失败：{exc_type}"
        if isinstance(exc, requests.HTTPError):
            status = getattr(getattr(exc, "response", None), "status_code", None)
            if status and status >= 500:
                return True, f"服务端错误 HTTP {status}"
            if status == 429:
                return True, "请求过于频繁被限流（HTTP 429）"
            return False, f"请求被拒绝 HTTP {status}"
    except ImportError:
        pass

    # ---- 第四层：关键词兜底 ----
    message = f"{exc_type}: {exc}".lower()
    if any(hint in message for hint in _PERMANENT_HINTS):
        return False, f"工具调用失败（{exc_type}）：{exc}"
    if any(hint in message for hint in _TRANSIENT_HINTS):
        return True, f"工具调用暂时失败（{exc_type}）：{exc}"

    # 认不出来的一律当"不重试"处理 —— 保守选择。
    # 理由：不认识 = 不可预期，贸然重试可能反复执行有副作用的操作
    # （比如重复下单、重复写库）。工具设计上要保证「重试安全」，
    # 做不到就应该选择不重试。
    return False, f"工具调用失败（{exc_type}）：{exc}"


# ========================================================================
# 二、失败说明的格式化
# ========================================================================


def format_failure(
    tool_name: str,
    reason: str,
    attempts: int,
    retryable: bool,
    hint: str = "",
) -> str:
    """把失败包装成一段**给 LLM 看**的文本。

    为什么要写得这么啰嗦？
        因为这段文字会直接进入 LLM 的上下文，成为它的"观察结果"。
        写得含糊（比如只返回 "Error"），LLM 只能瞎猜，可能反复调同一个工具；
        写清楚"失败原因 + 建议动作"，LLM 才有依据做下一步决策。

        这其实就是在用自然语言给 LLM 写"错误处理逻辑"。
    """
    lines = [
        f"【工具 `{tool_name}` 执行失败】",
        f"原因：{reason}",
        f"已尝试：{attempts} 次",
    ]
    if hint:
        lines.append(f"建议：{hint}")
    if retryable:
        lines.append("提示：这是临时性故障，你可以稍后换个说法再试一次。")
    else:
        lines.append("提示：重试同样会失败，请改用其他工具或直接告知用户。")
    return "\n".join(lines)


# ========================================================================
# 三、容错装饰器：把上面两块组合起来
# ========================================================================


def resilient(tool_name: str, fallback_hint: str = "") -> Callable:
    """给工具函数加上「重试 + 降级」能力。

    用法（注意装饰器顺序 —— `@tool` 必须在最外层）：

        @tool
        @resilient("web_search", fallback_hint="可以改用 search_local_knowledge")
        def search_web(query: str) -> str:
            ...

    为什么 `@tool` 要在最外层？
        `@tool` 需要读函数的**签名和 docstring** 来生成给 LLM 看的 JSON Schema。
        只有它在最外层，才能拿到真实的参数列表和文档字符串。
        反过来写的话，`@tool` 看到的是一个 `(*args, **kwargs)` 的包装函数，
        生成的 Schema 就是空的，LLM 根本不知道该传什么参数。

    装饰后保证：**无论内部发生什么，这个函数都不会抛异常。**
    成功 → 返回正常结果；失败 → 返回一段说明文本。
    """

    def decorator(func: Callable) -> Callable:
        if inspect.iscoroutinefunction(func):
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> str:
                max_attempts = max(1, agent_settings.TOOL_MAX_ATTEMPTS)

                for attempts in range(1, max_attempts + 1):
                    try:
                        return await func(*args, **kwargs)
                    except asyncio.CancelledError:
                        # 上游取消（客户端断开/服务关闭）必须继续传播。
                        raise
                    except Exception as error:  # noqa: BLE001
                        retryable, reason = classify_exception(error)
                        if not retryable or attempts >= max_attempts:
                            logger.warning(
                                "[%s] 第 %d/%d 次失败（不再重试）：%s",
                                tool_name, attempts, max_attempts, reason,
                            )
                            return format_failure(
                                tool_name,
                                reason,
                                attempts,
                                retryable,
                                fallback_hint,
                            )

                        logger.warning(
                            "[%s] 第 %d/%d 次失败（将重试）：%s",
                            tool_name, attempts, max_attempts, reason,
                        )
                        delay = min(
                            agent_settings.TOOL_RETRY_MAX_DELAY,
                            agent_settings.TOOL_RETRY_BASE_DELAY
                            * (2 ** (attempts - 1)),
                        )
                        await asyncio.sleep(delay + random.uniform(0, 0.3))

                return format_failure(
                    tool_name, "未知错误", max_attempts, False, fallback_hint
                )

            return async_wrapper

        @functools.wraps(func)  # 保留 __name__ / __doc__ / 签名，@tool 才能正确解析
        def wrapper(*args: Any, **kwargs: Any) -> str:
            max_attempts = max(1, agent_settings.TOOL_MAX_ATTEMPTS)
            attempts = 0

            # 用 tenacity 的 Retrying 对象（而不是 @retry 装饰器），
            # 是为了能在运行时读取配置。装饰器写法要求参数在 import 时就确定，
            # 那样改 .env 就得改代码。
            retryer = Retrying(
                # 重试条件的判定交给 classify_exception —— 我们在下面手动控制，
                # 这里只让 tenacity 负责"尝试几次、等多久"
                stop=stop_after_attempt(max_attempts),
                wait=wait_exponential(
                    multiplier=agent_settings.TOOL_RETRY_BASE_DELAY,
                    max=agent_settings.TOOL_RETRY_MAX_DELAY,
                ),
                retry=retry_if_exception_type(
                    (ToolTransientError, TimeoutError, ConnectionError)
                ),
                reraise=True,  # 重试耗尽后把原异常抛出来，交给下面统一处理
            )

            last_error: BaseException | None = None

            for attempt in retryer:
                with attempt:
                    attempts += 1
                    try:
                        # 每次尝试前加一点随机抖动。
                        # 为什么不只靠 tenacity 的指数退避？
                        #   指数退避解决的是"越等越久"，但如果你同时跑多个请求，
                        #   它们的重试时刻依然容易撞在一起。抖动打散这一点。
                        if attempts > 1:
                            jitter = random.uniform(0, 0.3)
                            time.sleep(jitter)
                        return func(*args, **kwargs)

                    except BaseException as error:  # noqa: BLE001 —— 这里就是要兜住一切
                        last_error = error
                        retryable, reason = classify_exception(error)

                        if retryable and attempts < max_attempts:
                            # 可重试且还有额度 —— 抛给 tenacity，让它决定等多久再试
                            logger.warning(
                                "[%s] 第 %d/%d 次失败（将重试）：%s",
                                tool_name, attempts, max_attempts, reason,
                            )
                            raise

                        # 不可重试，或已经用完额度 —— 记录后跳出
                        logger.warning(
                            "[%s] 第 %d/%d 次失败（不再重试）：%s",
                            tool_name, attempts, max_attempts, reason,
                        )
                        return format_failure(
                            tool_name, reason, attempts, retryable, fallback_hint
                        )

            # 理论上走不到这里（上面 return 覆盖了所有分支），
            # 但保留一个兜底，避免将来改动导致静默返回 None
            reason = str(last_error) if last_error else "未知错误"
            return format_failure(tool_name, reason, attempts, False, fallback_hint)

        return wrapper

    return decorator
