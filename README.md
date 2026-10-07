# RAG_Agent

端到端文档问答 API。**FastAPI + LangChain + LangGraph** 搭骨架，
检索层用 **Milvus 稠密检索 ⊕ BM25 稀疏检索的混合检索（RRF 融合）**＋**Cohere Rerank 精排**，
生成层是带自我反思的 **Agentic RAG 状态图**，评估用 **RAGAs**。

- LLM：智谱 GLM（`glm-4.5-air`，走 OpenAI 兼容协议）
- Embedding：本地 `BAAI/bge-small-en-v1.5`（384 维，不依赖外部 API）
- 向量库：Milvus（默认 Milvus Lite 本地文件模式，零部署）
- 精排：Cohere Rerank `rerank-v3.5`
- 缓存：Redis（LangChain LLM 级缓存，默认 TTL 1 小时，故障自动旁路）
- 可观测性：Prometheus（API QPS、延迟、状态码/错误率、进行中请求）

---

## 目录结构

```
RAG_mine/
├── .env                      # 环境配置（含密钥，已在 .gitignore 中）
├── .env.example              # 脱敏模板
├── .gitignore
├── README.md
├── requirements.txt
├── main.py                   # FastAPI 入口
├── eval_ragas.py             # RAGAs 评估脚本
├── eval_set.sample.json      # 评估集样例（包含有无标准答案两种写法）
├── app/
│   ├── __init__.py
│   ├── config.py             # 全局配置（模型 / 路径 / 超参数）
│   ├── document_processer.py # 文档加载与分块
│   ├── redis_client.py       # 异步 Redis 连接池与健康检查
│   ├── llm_cache.py          # Redis LLM 响应缓存
│   ├── observability.py      # Prometheus FastAPI 指标配置
│   ├── vector_store.py       # Milvus + BM25 混合检索 + RRF 融合
│   └── rag_chain.py          # LangGraph Agentic RAG 状态图
├── data/                     # 知识库源文档
├── vector_db/                # Milvus 本地库 + BM25 语料副本（自动生成）
└── docs/
    └──.md            
```

---

## 快速开始

```bash
conda activate LearnLC311
cd D:\project\llm\RAG_mine

# 1. 装依赖（ragas 若已装可跳过；详见 requirements.txt 末尾的兼容性说明）
pip install -r requirements.txt

# 2. 检查 .env（已配好；只需确认 API Key 有效）
#    注意：Milvus 地址的变量名是 RAG_MILVUS_URI，不是 MILVUS_URI！

# 3. 启动 Redis（已有 Redis 服务可跳过）
docker run -d --name rag-redis -p 6379:6379 redis:7-alpine

# 4. 启动 API
python main.py
# 或 uvicorn main:app --reload --port 8000
```

打开 http://127.0.0.1:8000/docs 可以直接在 Swagger UI 里试所有接口。

首次启动会下载 Embedding 模型（约 130MB），并构建 BM25 倒排索引。

LLM 缓存作用于 RAG 和 ResearchAgent 的全部模型调用；缓存键同时包含完整提示词与
模型配置，因此切换模型、温度或上下文不会误命中。可在 `.env` 中通过
`LLM_CACHE_ENABLED`、`LLM_CACHE_TTL_SECONDS`、`LLM_CACHE_PREFIX` 调整。
Redis 暂时不可用时请求会直接调用 LLM，不会因缓存故障而失败；`/health` 的
`llm_cache` 字段可查看连接状态与进程内命中/未命中计数。

### 使用 Docker Compose 启动

项目根目录已经提供 `Dockerfile` 和 `compose.yaml`。Docker Desktop 启动后执行：

```bash
# 构建应用镜像，并启动 RAG API + Redis
docker compose up --build -d

# 查看启动日志（首次会下载 Embedding 模型，耗时较长）
docker compose logs -f app
```

Docker 镜像使用 `requirements.runtime.txt`，只安装 API/Agent 运行依赖；
离线评估使用的 `ragas` 仍保留在完整的 `requirements.txt` 中，不会增大生产镜像或触发无关的依赖回溯。

启动完成后访问 <http://127.0.0.1:8000/docs>。检查容器状态：

```bash
docker compose ps
curl http://127.0.0.1:8000/health
```

Compose 会自动完成以下容器内配置：

- Redis 地址改为 `redis://redis:6379/0`；
- `data/` 和 `vector_db/` 挂载到宿主机，文档与 Milvus Lite 数据不会随容器删除；
- Redis 数据和 HuggingFace 模型缓存使用 Docker Volume 持久化；
- 应用代码不会把 `.env` 和其中的 API Key 打进镜像，密钥只在启动时注入。

常用管理命令：

```bash
docker compose restart app       # 重启应用
docker compose down              # 停止并删除容器，保留数据卷
docker compose down -v           # 同时删除 Redis/模型缓存卷（谨慎）
docker compose build --no-cache  # 不使用构建缓存重新打包
```

如果只想生成应用镜像而不启动：

```bash
docker build -t rag-mine:latest .
```

---

## 接口一览

| 方法 | 路径 | 说明 |
|---|---|---|
| GET | `/health` | 健康检查，回显**全部生效参数**（调参先看它） |
| GET | `/metrics` | Prometheus 文本格式指标（不显示在 Swagger 中） |
| POST | `/ingest/file` | 上传单个文档入库（pdf/txt/md/docx/html/htm/csv） |
| POST | `/ingest/directory` | 把 `data/` 下所有文档批量入库 |
| POST | `/retrieve` | **只检索不生成**，调参神器 |
| POST | `/query` | 完整 Agentic RAG 问答 |
| DELETE | `/index` | 清空向量库 |

### Prometheus API 指标

服务默认在 <http://127.0.0.1:8000/metrics> 暴露指标。可通过 `.env` 配置：

```dotenv
METRICS_ENABLED=true
METRICS_PATH=/metrics
```

核心指标如下：

| 指标 | 标签 | 用途 |
|---|---|---|
| `http_requests_total` | `handler`、`method`、`status` | 请求总数；状态码按 `2xx/4xx/5xx` 聚合 |
| `http_request_duration_seconds` | `handler`、`method` | 请求延迟直方图，桶覆盖 50ms～300s |
| `http_requests_inprogress` | `handler`、`method` | 当前正在处理的请求数 |

`/metrics` 本身不会计入上述业务指标。常用 PromQL：

```promql
# 总 QPS
sum(rate(http_requests_total[5m]))

# 按接口统计 QPS
sum by (handler, method) (rate(http_requests_total[5m]))

# 每个接口的 P95 延迟
histogram_quantile(
  0.95,
  sum by (le, handler, method) (rate(http_request_duration_seconds_bucket[5m]))
)

# 5xx 错误率（0～1）
sum(rate(http_requests_total{status="5xx"}[5m]))
/
sum(rate(http_requests_total[5m]))
```

Prometheus 抓取配置示例；Prometheus 在同一个 Compose 网络中时使用 `app:8000`，
从宿主机运行时改成 `host.docker.internal:8000`：

```yaml
scrape_configs:
  - job_name: rag-api
    scrape_interval: 15s
    static_configs:
      - targets: ["app:8000"]
```

生产环境建议只允许 Prometheus 从内网访问 `/metrics`，不要直接暴露到公网。

### 典型流程

```bash
# 1) 检查配置和向量库状态
curl http://127.0.0.1:8000/health

# 2) 批量导入 data/ 下的文档
curl -X POST http://127.0.0.1:8000/ingest/directory

# 3) 上传单个文档
curl -X POST http://127.0.0.1:8000/ingest/file -F "file=@data/paul_graham_essay.txt"

# 4) 提问
curl -X POST http://127.0.0.1:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question":"作者为什么要学习画画？"}'

# 5) 只调检索、不烧 token
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question":"混合检索为什么用 RRF？","top_k":5}'

# 6) 对比精排前后的差异
curl -X POST http://127.0.0.1:8000/retrieve \
  -H "Content-Type: application/json" \
  -d '{"question":"混合检索为什么用 RRF？","skip_rerank":true}'

# 7) 清库重来
curl -X DELETE http://127.0.0.1:8000/index
```

`/query` 的响应除答案外还带这些观测字段：

```json
{
  "answer": "...",
  "question": "...",
  "rewritten": false,
  "num_documents": 5,
  "sources": [
    {"filename": "paul_graham_essay.txt", "page": null,
     "snippet": "...", "relevance_score": 0.87, "rrf_score": 0.0164}
  ],
  "queries": ["原问题", "变换出的查询2", "变换出的查询3"],
  "transform_mode": "multi_query(3)",
  "candidates": 20,
  "reranked": true
}
```

`sources` 里的 `relevance_score` 是 Cohere 精排分，`rrf_score` 是融合分，
能直接看出"这条为什么被选中"。

---

## 架构

### RAG 流程图

```
transform_query -> retrieve -> rerank_documents -> grade_document
       ^                                                |
       |                                                v
  rewrite_query <-----(无相关文档且可重试)---------------+
       |
       +--(改写无效)--> generate --> check_groundedness --> END
```

各节点职责：

| 节点 | 干什么 | 解决什么问题 |
|---|---|---|
| `transform_query` | 查询变换：multi_query(RAG-Fusion) / HyDE | 问法不好，后面再强也救不回来 |
| `retrieve` | 稠密(Milvus) ⊕ 稀疏(BM25)，RRF 融合 | 语义匹配 + 关键词精确命中，两者互补 |
| `rerank_documents` | Cohere 交叉编码器精排 | 把粗排的 20 条砍到真正相关的 5 条 |
| `grade_document` | LLM 逐条评估相关性 | 过滤漏网的无关文档 |
| `rewrite_query` | 改写查询后重试 | 首次检索失败时的自我修正 |
| `generate` | 按上下文生成答案 | — |
| `check_groundedness` | 忠实性自检 | 拦截幻觉，不对就换成拒答 |

### 检索层：为什么是「混合检索 + 精排」

```
                 ┌─ 稠密：Milvus 向量检索 (Top 20) ─┐
   查询 ─────────┤                                  ├─ RRF 融合 ─> Top 20 ─> Rerank ─> Top 5
                 └─ 稀疏：BM25 关键词检索 (Top 20) ─┘
```

- **稠密检索**负责"语义相近"：问"如何提升检索质量"，能命中讲"recall 优化"的段落。
- **稀疏检索**负责"关键词精确命中"：专有名词、型号、错误码、人名，语义模型容易漂移，BM25 往往一把命中。
- **RRF 融合**：稠密相似度是余弦值（0~1），BM25 分数无上界（可能 0.3 也可能 27），
  两者量纲完全不同，直接加权毫无意义。RRF 只看排名：

  ```
  score(doc) = Σ weight_i × 1/(60 + rank_i(doc))
  ```

- **精排**：向量检索是双塔结构，查询和文档从未"见面"；交叉编码器把 `[查询, 文档]`
  拼在一起送进模型，每个词都能和对方的词做交互注意力。所以标准分工是
  "向量检索从百万里捞出一百，精排从一百里挑出五"。

> 精排刻意放在 `grade_document` **之前**：评分要给每个文档调一次 LLM，
> 先精排把 20 条砍到 5 条，LLM 调用次数直接降 4 倍。

### 查询变换三模式

在 `.env` 里改 `QUERY_TRANSFORM`：

| 模式 | 做法 | 适合场景 |
|---|---|---|
| `none` | 直接用原问题 | 调试基线 |
| `multi_query` | **RAG-Fusion**：裂变成 3~4 条不同角度的查询，各自检索后 RRF 融合 | 默认推荐 |
| `hyde` | 让 LLM 先"编"一段假想答案，用假想答案去检索 | 问题短、术语少 |

HyDE 的原理：问题→文档是**跨分布匹配**，答案→文档是**同分布匹配**。
用一段"长得像答案"的文本去检索，语义距离天然更近。

---

## 评估

```bash
# 用样例评估集先跑 3 条试水（省 token）
python eval_ragas.py --dataset eval_set.sample.json --limit 3

# 全量评估并导出报告
python eval_ragas.py --dataset eval_set.sample.json --out eval_report.csv
```

评估集格式（`eval_set.json`）—— `reference` 可选，但不写就只能算两个指标：

```json
[
  {"question": "问题1", "reference": "标准答案1"},
  {"question": "问题2"}
]
```

指标解读：

| 指标低 | 说明什么 | 去哪儿调 |
|---|---|---|
| `context_recall` | 该找到的证据没找全 | chunk 大小、提高 `FUSION_TOP_K`、开 `multi_query` |
| `context_precision` | 捞了一堆没用的 | 调 `RERANK_TOP_N`、调 `DENSE/SPARSE_WEIGHT` |
| `faithfulness` | 模型在编 | 收紧 `GENERATE_PROMPT`、加强忠实性检查 |
| `answer_relevancy` | 答案跑题 | 改生成模板 |

---

## 调参速查

`.env` 里最常动的几个：

```ini
HYBRID_ENABLED=true
DENSE_WEIGHT=0.5          # 知识库多专有名词/型号 -> 调低，多给 BM25
SPARSE_WEIGHT=0.5
FUSION_TOP_K=20           # 给 Reranker 的候选数，越大越全但精排越慢
RERANK_TOP_N=5            # 最终喂给 LLM 的条数，同时决定评分阶段的 LLM 调用次数
QUERY_TRANSFORM=multi_query
NUM_QUERIES=3             # 2~3 性价比最高，5 以上收益递减
```

一次 `/query` 的 LLM 调用次数大致是：
`1 次查询变换 + 可能 1 次改写 + N 次文档评分 + 1 次生成 + 1 次忠实性检查`。
想省钱就调小 `NUM_QUERIES` 和 `RERANK_TOP_N`。

---

## 常见故障排查

### `Open local milvus failed` / 启动时报 Milvus 打不开

**最常见的原因：你已经有一个服务在跑了。**

Milvus Lite 对本地库文件加**独占锁**，同一时刻只允许一个进程打开。
典型场景：`python main.py` 还开着，又开一个终端执行了一次，第二个必然失败。

排查：

```bash
netstat -ano | findstr :8000          # 看 8000 端口被哪个 PID 占用
taskkill /PID <上面查到的进程号> /F    # 停掉旧进程后再启动
```

> pymilvus 会把真实原因吞掉，只抛一句没营养的 `Open local milvus failed`。
> 本项目的 `app/vector_store.py` 里加了主动探测，会直接把真实原因和上面的处理步骤打印出来。

如果你确实需要多个实例共用同一个库（比如同时跑开发和生产），改用远程 Milvus：

```bash
docker run -d --name milvus -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
```

然后把 `.env` 里 `RAG_MILVUS_URI` 改成 `http://localhost:19530`。远程模式下多进程共用不会冲突。

### `ModuleNotFoundError: No module named 'langchain_milvus'`

依赖没装全，执行：

```bash
conda activate LearnLC311
pip install -r requirements.txt
```

### `ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'`

这是 ragas 0.4.x 与新版本 langchain-community 的兼容问题，`eval_ragas.py` 里已经内置补丁，
不需要额外处理。若在别处也遇到，参考 `eval_ragas.py` 顶部的 `_patch_missing_vertexai()`。

### 精排分数普遍很低（0.0x）

先确认知识库里**到底有没有**相关内容。`/health` 看 `index_count`，
`/retrieve` 看返回的 `filename` 分布。如果库里压根没有相关文档，
Cohere 给出的分数的确会是很低的 0.0x 量级，此时 `/query` 返回拒答是**正确行为**，不是 bug。

### 检索结果里全是同一个文件

说明这个文件被切成了很多块，占满了 Top-K。要么它确实最相关，
要么知识库内容太单一，导入其它文档即可（`POST /ingest/directory`）。

---

## 注意事项

1. **换 Embedding 模型 = 必须重建库。** 向量维度一变，Milvus collection 结构就对不上。
   把 `.env` 里 `MILVUS_DROP_OLD` 临时改成 `true`，重启一次，**然后立刻改回 `false`**，
   否则每次重启都会清空知识库。

2. **Milvus 地址的环境变量是 `RAG_MILVUS_URI`，不是 `MILVUS_URI`。**
   pymilvus 3.x 会在 import 阶段读取环境变量 `MILVUS_URI` 并强制按 `http[s]://` 解析，
   值若是本地文件路径，`import pymilvus` 会直接抛 `ConnectionConfigException`。
   `app/config.py` 里加了防御，但仍建议不要使用这个保留名。

3. **BM25 语料副本不要单独删。** `vector_db/bm25_corpus.jsonl` 是稀疏检索的数据源。
   删了之后 BM25 会**静默失效**（代码兜底为退化成纯向量检索，不报错），很难察觉。

4. **`/ingest/directory` 是追加写入，不去重。** 同一批文件重复调用会塞重复块。
   要重来时先 `DELETE /index`，或直接删掉 `vector_db/` 目录。

5. **精排没配 Key 会自动跳过**，混合检索仍正常工作，只是 `/query` 的 `reranked` 会是 `false`。
   Cohere Rerank 有免费额度：https://dashboard.cohere.com/api-keys

6. **上生产时**把 `RAG_MILVUS_URI` 改成 `http://localhost:19530`（Docker 起的 Milvus）
   或 Zilliz Cloud 地址，代码一行都不用动。
