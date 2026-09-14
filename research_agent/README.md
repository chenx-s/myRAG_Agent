# 研究助手 Agent

集成 **RAG（本地知识库）** 与 **Web 搜索** 的 Agent，对应学习计划 **Day 21 周度项目构建**。

---

## 快速开始（3 步）

### 1. 填一个 API key

打开项目根目录的 `.env`，找到这一行，把 key 填进去：

```ini
TAVILY_API_KEY=tvly-你的key
```

免费申请：<https://tavily.com>（注册后在 Dashboard 复制，每月 1000 次免费额度，学习够用）

> **不填也能跑。** 联网搜索工具会返回"未配置"的提示，本地知识库照常工作。
> 这也正好能让你直观看到 Agent 的降级行为（后面有实测）。

### 2. 确认本地知识库有数据

本地检索工具复用你现有的 RAG 系统。**如果向量库是空的，它会返回"没有找到相关内容"** —— 这是正常的，不是 bug。

想让它有内容，先把文档索引进去：

```bash
# 启动你的 RAG 服务
uvicorn main:app --reload --port 8000

# 索引一个文件（推荐先从小文件开始，别直接跑 /ingest/directory）
curl -X POST http://127.0.0.1:8000/ingest/file -F "file=@data/paul_graham_essay.txt"
```

### 3. 运行

```bash
cd d:/project/llm/LangChainLearn
python -m research_agent.agent
```

> 启动时会自动预热本地知识库。首次要 30 秒左右（要从 HuggingFace 加载 BGE 模型），
> 之后每次启动就快了。不想等就在 `.env` 里设 `AGENT_WARMUP_ON_START=false`。

---

## 它长什么样

```
你 > SAR图像相干斑抑制有哪些深度学习方法？

🤔 Agent 思考中 ...
  🔧 行动 1：调用 `search_local_knowledge`
        query = SAR图像相干斑抑制 深度学习方法
  👁 观察：
        【本地知识库检索结果】
        答案：主要有以下几类方法……
        出处：
          [1] 基于深度学习的SAR图像相干斑抑制网络研究_姚同钰.pdf 第 3 页（相关度 0.87）
          [2] 复杂环境下雷达图像相干斑噪声抑制...pdf 第 12 页（相关度 0.80）

────────────────────────────────────────────────────────────────
助手 >
主要有三类方法：……（答案正文）

────────────────────────────────────────────────────────────────
```

那几行 `🔧 行动` 和 `👁 观察`，就是 **ReAct 循环**的可视化 —— 这是理解 Agent 工作原理最直观的方式。

---

## 架构全景

```
                         用户提问
                            │
                            ▼
        ┌───────────────────────────────────────────────┐
        │   create_agent 内置的 ReAct 循环               │
        │                                               │
        │   ① Reason  读问题+历史+工具清单，决定下一步    │
        │        │                                      │
        │        ▼                                      │
        │   ② Act     输出 function calling 结构化请求    │
        │        │      {"name":"search_web",           │
        │        │       "arguments":{"query":"..."}}   │
        │        ▼                                      │
        │   ③ Observe 执行工具，结果塞回上下文            │
        │        │                                      │
        │        └────► 回到 ①，带着新信息继续思考        │
        │               （直到给出最终答案或步数用尽）    │
        └───────────────────────────────────────────────┘
                            │
              ┌─────────────┴─────────────┐
              ▼                           ▼
   ┌─────────────────────┐    ┌─────────────────────┐
   │ search_local_       │    │ search_web          │
   │ knowledge           │    │                     │
   │                     │    │                     │
   │ → app/rag_chain.py  │    │ → Tavily Search API │
   │   （你原有的 RAG）   │    │   （实时网页）       │
   │                     │    │                     │
   │ 准确、有页码出处     │    │ 信息新、覆盖广       │
   │ 只覆盖已索引的资料   │    │ 无稳定页码、质量参差 │
   └─────────────────────┘    └─────────────────────┘
              │                           │
              └─────────────┬─────────────┘
                            ▼
                  两个工具都套了容错层
                  （重试 → 降级 → 返回失败说明）
```

---

## 文件结构

```
research_agent/
├── __init__.py     导出主要接口
├── settings.py     配置：LLM / Tavily / 重试参数 / 循环上限
├── tools.py        ★ 两个工具的定义（这里是重点）
├── resilience.py   ★ 容错：故障分类 → 重试 → 降级
└── agent.py        ★ Agent 组装（ReAct + Memory）+ 命令行交互
```

**代码没动你现有的任何文件**，只是在 `tools.py` 里 `from app.rag_chain import RAGChain` 复用了你的 RAG。

---

## 逐段讲解

### 一、ReAct 循环（Day 15）

ReAct = **Reason + Act**，出自你链接里的那篇论文 [arXiv:2210.03629](https://arxiv.org/abs/2210.03629)。

**核心思想**：让 LLM 不要一次把答案憋出来，而是"**先想一步、做一步、看一眼结果、再想下一步**"。
这样它可以根据中间结果动态调整策略 —— 查不到就换个词再查，查到矛盾就去交叉验证。

对比一下就清楚了：

| | 传统 RAG（你的 `app/`） | Agent（本项目） |
|---|---|---|
| 流程 | 固定：检索 → 生成 | 动态：LLM 自己决定走几步、走哪条路 |
| 检索次数 | 恒定 1 次（+改写重试） | 0 到 N 次，由 LLM 判断 |
| 工具选择 | 写死在代码里 | LLM 根据问题自己选 |
| 遇到空结果 | 走预设的兜底分支 | LLM 自己决定换工具还是换问法 |

**关键认知：ReAct 循环本身不需要你写。**

`create_agent` 已经内置了这个 `while` 循环。你要做的是三件事：

1. **把工具设计好** → `tools.py`（尤其把 docstring 写清楚）
2. **把工作方法讲清楚** → `agent.py` 里的 `RESEARCH_SYSTEM_PROMPT`
3. **把外部依赖兜住** → `resilience.py`

这三件事是 Agent 开发的实际工作量所在。循环、消息管理、状态传递，框架都替你做了。

### 二、工具定义（Day 16）

**最简单也最被低估的一点：`@tool` 装饰器 + docstring = 给 LLM 的工具说明书。**

```python
@tool
def search_web(query: str, max_results: int = 3) -> str:
    """通过【互联网搜索】获取最新、公开的信息。
    ...
    """
```

装饰器做的事情：
1. 读函数的**类型注解** → 生成参数 schema（`query: str` 变成 JSON Schema 里的 string 类型）
2. 读 **docstring** → 作为工具描述
3. 把两者打包成 OpenAI Function Calling 格式，随每次请求发给 LLM

于是 LLM 每次都能看到这样一份清单：

```json
{
  "name": "search_web",
  "description": "通过【互联网搜索】获取最新、公开的信息。\n\n什么时候用：...",
  "parameters": {
    "query": {"type": "string"},
    "max_results": {"type": "integer", "default": 3}
  }
}
```

#### ★ 最重要的经验：两个"搜索"工具，全靠 docstring 区分

本项目有两个工具，功能上都是"搜索"。LLM 怎么知道该用哪个？**只能靠 docstring。**

所以这两个工具的文档都刻意写成三段式：

```
① 干什么          在【本地知识库】中检索用户自己上传的文档资料。
② 什么时候该用     问题涉及专业知识、学术概念、论文内容 → 先查本地
③ 什么时候不该用   实时信息（新闻、股价、最新版本号）→ 本地库不会有
```

**第 ③ 段最容易被忽略，但最关键。** 只写"我能查资料"，LLM 就会在你问"今天天气"的时候
去翻你的论文库；明确写出"不适用于实时信息"，它才会去联网。

你可以在运行时验证这一点：

```
你 > /tools

工具清单：
  · search_local_knowledge(query)
      在【本地知识库】中检索用户自己上传的文档资料。
  · search_web(query, max_results)
      通过【互联网搜索】获取最新、公开的信息。
```

**这就是 LLM 看到的全部信息。** 如果 Agent 老是选错工具，先回来看看这里的描述是不是写得不够清楚。

#### 装饰器顺序（踩坑点）

```python
@tool                                  # ← 必须在外层
@resilient("web_search", ...)          # ← 容错在内层
def search_web(query: str, ...) -> str:
```

`@tool` 需要读**真实的函数签名**才能生成正确的 schema。只有它在最外层才能拿到。
反过来写的话，`@tool` 看到的是容错层包装出来的 `(*args, **kwargs)`，
生成的 schema 里参数列表就是空的，LLM 会完全不知道该传什么参数。

> 本项目已用 `functools.wraps` 保住了签名，并实测验证过生成的 schema 是正确的：
> `search_web` 拿到的是 `{query: string, max_results: integer}`。

### 三、Function Calling 原理（Day 18）

这是上面"LLM 输出结构化调用请求"那一步的底层协议。理解它，Agent 就不再是黑盒。

**整个往返过程：**

```
① 请求（你发给 LLM）
   {
     messages: [{"role":"user","content":"今天天气怎么样"}],
     tools: [{"type":"function","function":{"name":"search_web", ...}}]
   }

② LLM 的响应（注意：它没有直接回答，而是"要求调用工具"）
   {
     "tool_calls": [{
       "id": "call_abc123",
       "function": {"name": "search_web",
                    "arguments": "{\"query\":\"北京今天天气\"}"}
     }]
   }

③ 你执行函数，把结果发回去
   {
     messages: [
       ...上面的往返...,
       {"role":"tool","tool_call_id":"call_abc123","content":"北京今天晴，25度"}
     ]
   }

④ LLM 这次给出最终答案
   {"content":"北京今天晴天，气温 25 度。"}
```

**三个关键点：**

1. **`arguments` 是 JSON 字符串**，不是对象。所以模型可能生成语法错误的 JSON —— 这就是
   你链接里那个 `handle_parsing_errors` 要解决的问题。LangChain 的 `@tool` 帮你做了这层解析。

2. **同一个 `AIMessage` 类承担两种角色**：带 `tool_calls` 表示"我要调工具"，
   不带就是"这是我的最终回答"。`agent.py` 的 `_parse_trace()` 就是靠这个区分来还原整个思考过程的。

3. **`tool_call_id` 是把请求和响应配对的关键**。一轮对话里模型可能同时要求调用多个工具，
   靠这个 ID 才能把结果对应回正确的调用。

> 补充：本项目用的是智谱 GLM，走的是 **OpenAI 兼容协议**，所以这套 Function Calling
> 机制完全适用。`create_agent` 会自动把 LangChain 工具转成这个格式。

### 四、Memory 对话记忆（Day 19）

```python
from langgraph.checkpoint.memory import InMemorySaver

agent = create_agent(..., checkpointer=InMemorySaver())

agent.invoke(
    {"messages": [{"role": "user", "content": "..."}]},
    config={"configurable": {"thread_id": "research-default"}},   # ← 记忆的隔离键
)
```

**`thread_id` 是理解记忆的关键**：

- **同一个 `thread_id`** → 连续对话。Agent 记得你上一句说了什么，可以直接说"再详细点"
- **换一个 `thread_id`** → 全新会话，之前聊的全部失忆

所以记忆是靠 ID 隔离的，不是靠"一个 Agent 一个记忆"。
一个 `InMemorySaver` 实例可以同时服务多个互相独立的会话。

本项目里 CLI 的 `/new` 命令就是换一个 `thread_id`。

**当前用的是内存存储，进程退出就没了。** 想持久化，把 `InMemorySaver` 换成：

| 存储 | 适用场景 |
|---|---|
| `InMemorySaver` | 开发调试、单次会话（本项目默认） |
| `SqliteSaver` | 单机持久化，重启不丢（**推荐你下一步换这个**） |
| `PostgresSaver` | 多实例部署、生产环境 |

接口完全一样，只是换个类名。

> 另外你架构图里提到的 `Memory0` / `MemoryScope`（对应图片里 Day 19 的"工具"一栏）
> 属于**长期记忆**方案：它们会从对话里自动抽取事实存进向量库，
> 比 `ConversationBufferMemory` 这种"原样存消息"更聪明，但复杂度也更高。
> 学完基础记忆之后再看它们会更顺。

### 五、错误处理与降级（Day 20）

代码在 `resilience.py`，这是本项目里"看起来最简单、实际上最见功力"的部分。

#### 核心认知一：工具失败时，不要抛异常，而要返回失败说明

```python
# ❌ 错误做法
if not api_key:
    raise ValueError("没配 key")
# 后果：异常冒泡到 create_agent → 整轮对话中断 → 用户看到红色报错
#       而且 LLM 完全不知道发生了什么，它的回合已经结束了

# ✅ 正确做法
return "【工具 search_web 执行失败】原因：未配置 API key...建议：改用 search_local_knowledge"
# 后果：LLM 看到这条"观察结果"，自己决定下一步
```

**这是 Agentic 系统和普通程序最大的思维差异**：
普通程序里，错误要抛出去让人处理；Agent 里，错误是**给 LLM 的一条信息**，让它自己决策。

#### 核心认知二：不是所有错误都值得重试

| 类型 | 例子 | 策略 |
|---|---|---|
| **临时性**（transient） | 超时、连接重置、429 限流、502/503/504 | ✅ 重试 |
| **永久性**（permanent） | key 无效、未配 key、配额耗尽、400 参数错 | ⛔ 不重试，直接降级 |

不分清楚会出两种事故：
- **全都重试** → 一个 key 过期的配置，白白卡满 3 次 × 30 秒超时
- **全都不重试** → 网络抖一下就直接给用户报错

`classify_exception()` 用四层判断：先看我们自己抛的异常类型 → 再看第三方库的明确异常类型
→ 再看标准库网络异常 → 最后才退化为关键词匹配。**越可靠的手段放越前面。**

#### 核心认知三：重试要退避 + 抖动

```python
wait=wait_exponential(multiplier=1, max=10)   # 1s → 2s → 4s ... 封顶 10s
jitter = random.uniform(0, 0.3)               # 每次再加 0~0.3s 随机抖动
```

- **退避**：对方限流时，你立刻重试等于继续加压，只会被继续拒绝。指数退避给它喘息时间。
- **抖动**：避免多个请求"整齐划一"地同时重试，造成惊群。

#### 为什么失败说明要写得那么"啰嗦"

看实际生成的这段文本：

```
【工具 `search_web` 执行失败】
原因：未配置 TAVILY_API_KEY。请在项目根目录的 .env 里加一行：TAVILY_API_KEY=tvly-你的key
已尝试：1 次
建议：联网搜索不可用时，改用 search_local_knowledge 查本地知识库。
提示：重试同样会失败，请改用其他工具或直接告知用户。
```

**这段文字会直接进入 LLM 的上下文。** 写得含糊（比如只返回 `"Error"`），LLM 只能瞎猜，
很可能反复调同一个必定失败的工具，直到耗尽 `recursion_limit`。

**你其实是在用自然语言给 LLM 写错误处理逻辑**——这就是 Agent 开发的独特之处。

---

## 实测：降级链路真的有效

这是本项目做的一次真实测试。**场景**：故意不配 `TAVILY_API_KEY`，然后问一个必须联网的问题。

```
你 > 最近人工智能领域有什么重要新闻？

🤔 Agent 思考中 ...
  🔧 行动 1：调用 `search_web`
        query = 人工智能 AI 重要新闻 最新进展
  👁 观察：
        【工具 `search_web` 执行失败】
        原因：未配置 TAVILY_API_KEY。...
        建议：联网搜索不可用时，改用 search_local_knowledge 查本地知识库。

  🔧 行动 2：调用 `search_local_knowledge`      ← ★ Agent 读懂了建议，真的换工具了
        query = 人工智能 AI 重要新闻 最新进展
  👁 观察：
        【本地知识库检索结果】
        答案：抱歉，在当前知识库中没有找到与该问题相关的信息。

助手 >
很抱歉，我无法为您提供最近人工智能领域的重要新闻。

由于联网搜索功能需要配置API密钥而暂时不可用，同时您的本地知识库中
也没有相关的新闻资料，我无法获取最新的AI领域动态。

建议您：
1. 可以直接访问主流科技媒体网站（如 TechCrunch、36氪等）查看最新AI新闻
2. 关注AI领域的官方账号和研究机构发布的信息
3. 如果您有相关的AI新闻文档，可以上传到本地知识库
```

**这次测试同时验证了三件事：**

1. ✅ **容错层生效** —— 工具没崩，返回了可读的失败说明
2. ✅ **LLM 读懂了失败说明** —— 它根据"建议：改用 search_local_knowledge"
   **主动切换了工具**（这是 Agent 自主决策的直接证据）
3. ✅ **system_prompt 的"诚实优先"生效** —— 两个工具都没结果时，
   它明确说"我无法提供"，**没有编造任何新闻**

第 3 点特别重要。如果不写那句"绝对不要编造内容"，LLM 在工具返回空结果时
**几乎一定会开始编造**——这是 Agent 系统最常见的翻车方式。

---

## 怎么再加一个工具

照抄 `tools.py` 的模板即可，三步：

```python
@tool
@resilient("calculator", fallback_hint="计算失败时，请直接说明无法计算。")
def calculator(expression: str) -> str:
    """计算数学表达式。

    什么时候用：需要精确数值计算时（LLM 自己做算术容易出错）。
    什么时候不要用：简单的加减法（直接回答即可）。

    参数 expression：合法的 Python 数学表达式，如 "(3+5)*2"。
    """
    return str(eval(expression, {"__builtins__": {}}, {}))   # 注意：真要用得做安全校验
```

然后加到清单里：

```python
ALL_TOOLS = [search_local_knowledge, search_web, calculator]
```

**就这样，不用改 Agent、不用改 prompt**——`create_agent` 会自动把它纳入工具清单。

> ⚠️ 但要注意：**工具越多，LLM 选错的概率越大**。每加一个工具，
> 所有工具的 docstring 都要足够清晰才能互相区分。经验法则：
> 超过 10 个工具时，考虑用"分类 + 二级 Agent"的方式分层组织。

---

## 与学习计划的对应关系

| 学习计划 | 对应本项目的位置 |
|---|---|
| Day 15 Agent 核心概念 / ReAct | `agent.py` 的 `create_agent` 组装 + CLI 里可视化的思考轨迹 |
| Day 16 自定义工具开发 | `tools.py`，重点是 docstring 的三段式写法 |
| Day 17 SQL & 数据库工具 | 思路完全相同（把数据库查询包装成工具），本项目用本地检索代替 |
| Day 18 Function Calling 实战 | 「逐段讲解 · 三」，以及 `_parse_trace()` 对消息流的解析 |
| Day 19 Agent Memory | `checkpointer=InMemorySaver()` + `thread_id` 隔离 |
| Day 20 Agent 错误处理 | `resilience.py` 全部内容 |
| **Day 21 周度项目构建** | **本项目整体** |

**手撕清单（图片里的三项）对照：**

- ✅ 「实现 3 个自定义工具」→ 本项目给了 2 个完整实现 + 1 个扩展模板（见上一节）
- ✅ 「基于 LangChain 构建可以链式调用工具的 Agent」→ 就是本项目
- ⬜ 「使用 OpenAI Function Calling 实现结构化数据提取」→ **这个还没做**，
  它是另一个方向的应用（不是 Agent 而是**结构化输出**）。
  可以用 `with_structured_output()` 或 `create_agent(..., response_format=...)` 实现，
  建议作为下一步练习。

---

## 常见问题

**Q: 启动很慢，卡在"正在预热本地知识库"**

首次运行要从 HuggingFace 下载 BGE 模型权重（约 130MB），国内网络可能较慢。
设置镜像可以加速：

```bash
set HF_ENDPOINT=https://hf-mirror.com     # Windows CMD
export HF_ENDPOINT=https://hf-mirror.com  # Git Bash
```

不想等就在 `.env` 里设 `AGENT_WARMUP_ON_START=false`。

---

**Q: 本地检索总是返回"没有找到相关内容"**

两种可能：

1. **向量库是空的** —— 还没索引文档。先去 RAG 服务里跑 `/ingest/file`。
   可以查一下当前有多少块：`curl http://127.0.0.1:8000/health`
2. **索引了但检索不到** —— 你现在用的 `BAAI/bge-small-en-v1.5` 是**纯英文模型**，
   中文提问时排序会偏。建议换成 `BAAI/bge-m3`（详见 `docs/项目结构与学习指南.md` 第 7.2 节）

---

**Q: 它老是调用错的工具**

99% 是 docstring 的问题，不是模型的锅。按这个顺序排查：

1. 运行 `/tools` 看看 LLM 眼里这些工具的描述是什么
2. 检查每个工具的 docstring 有没有写清「**什么时候不要用**」
3. 两个工具的适用场景如果重叠，把它们分开写，不要留模糊地带

---

**Q: 想把它接到 FastAPI 里，和现有的 RAG 服务合并**

```python
from fastapi import FastAPI
from pydantic import BaseModel
from research_agent.agent import ResearchAgent

app = FastAPI()
_agent = ResearchAgent()

class AskRequest(BaseModel):
    question: str
    session_id: str = "default"

@app.post("/research")
async def research(req: AskRequest):
    # 注意：Agent 是同步阻塞的，生产环境建议用 run_in_threadpool 包一层
    from starlette.concurrency import run_in_threadpool
    _agent.thread_id = req.session_id
    result = await run_in_threadpool(_agent.chat, req.question, False)
    return {"answer": result["answer"],
            "tool_calls": [s["tool"] for s in result["trace"]]}
```

> 一个注意点：Agent 一轮可能要几十秒（多次 LLM 调用 + 多次工具调用），
> **同步 HTTP 请求容易超时**。生产环境更适合用流式输出（`astream`）
> 或改成任务队列 + 轮询。

---

**Q: 和 `LearnAgent.py` 里我写的那个 Agent 有什么区别？**

`LearnAgent.py` 里是两个返回固定字符串的假工具（`get_horoscope`、`get_weather`），
用来理解 `create_agent` 的用法 —— 那是 Day 15 的正确做法。

本项目是在那之上加了**真实可用的工具**（一个接你现有的 RAG 系统，一个接真实搜索 API），
以及**真实环境必须处理的问题**：网络失败、key 没配、上下文长度控制、
结果格式化、延迟优化。这才是"从 demo 到能用"之间的那段距离。

---

## 关键文件速查

| 想改什么 | 去哪 |
|---|---|
| 改 Agent 的工作方法 / 工具选择策略 | `agent.py` 的 `RESEARCH_SYSTEM_PROMPT` |
| 改工具的适用场景描述 | `tools.py` 各工具的 docstring（**改这个能直接改变 Agent 行为**） |
| 加新工具 | `tools.py`，然后加进 `ALL_TOOLS` |
| 改重试次数 / 退避时长 | `.env` 的 `TOOL_MAX_ATTEMPTS` 等 |
| 改循环上限 | `.env` 的 `AGENT_RECURSION_LIMIT` |
| 换记忆存储（改成持久化） | `agent.py` 的 `build_agent()`，把 `InMemorySaver` 换成 `SqliteSaver` |
| 改联网搜索的返回条数 | `.env` 的 `WEB_SEARCH_MAX_RESULTS` |
