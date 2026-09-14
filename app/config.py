# 配置（模型， 路径， 超参数）
#
# 升级版：在原配置基础上新增四组配置
#   1) Milvus 向量库（替代 Chroma/FAISS）
#   2) 混合检索 Hybrid Search（稠密向量 + BM25 稀疏，RRF 融合）
#   3) 查询变换 Query Transformations（multi-query / HyDE）
#   4) Reranker（Cohere Rerank 精排）
# 另外补上了原文件缺失的 RECURSION_LIMIT —— 没有它 rag_chain.answer() 会直接 AttributeError。

import os
from dotenv import load_dotenv

load_dotenv()


# ================================================================
# 【踩坑警告】不要把 Milvus 地址写成 MILVUS_URI
# ================================================================
# pymilvus 3.x 的 settings.Config 里有这么一行：
#     MILVUS_URI = str(os.getenv("MILVUS_URI", ""))
# 而 pymilvus/orm/connections.py 在**模块导入时**就会拿这个值去解析：
#     address, parsed_uri = self.__parse_address_from_uri(Config.MILVUS_URI)
# 解析失败直接抛 ConnectionConfigException —— 注意，是 import pymilvus 就炸，
# 还没等你调用任何 Milvus 接口。
#
# 也就是说：只要环境变量 MILVUS_URI 是个本地文件路径（./vector_db/x.db），
# 下面这行就会报 "Illegal uri: [...], expected form 'http[s]://...'"：
#     import pymilvus
#
# 所以本项目的配置项叫 RAG_MILVUS_URI，绕开这个保留名。
# 下面这段是防御：万一你的 .env 或系统环境里已经有 MILVUS_URI，
# 且它不是合法的 http(s) 地址，就先摘掉，避免污染 pymilvus。
_reserved_uri = os.environ.get("MILVUS_URI", "")
if _reserved_uri and not _reserved_uri.startswith(("http://", "https://")):
    os.environ.pop("MILVUS_URI", None)
    print(
        "[config] 检测到环境变量 MILVUS_URI 不是 http(s) 地址，已临时移除。"
        "它会让 import pymilvus 直接失败；请改用 RAG_MILVUS_URI。"
    )


# ================================================================
# 降低第三方库的日志噪音
# ================================================================
# Milvus Lite 内嵌了一个 gRPC 服务，默认会往终端刷大量
#   WARNING too_many_pings ... GOAWAY
# 这类 INFO 级别的心跳告警 —— 它不影响功能，但会把有用的日志淹没。
# 下面几个环境变量必须在 grpc 初始化之前设置才生效，
# 所以放在 config.py 顶层（它会被最早导入）。
os.environ.setdefault("GRPC_VERBOSITY", "ERROR")
os.environ.setdefault("GRPC_GO_LOG_SEVERITY_LEVEL", "ERROR")
os.environ.setdefault("GRPC_GO_LOG_VERBOSITY_LEVEL", "ERROR")


def _env_bool(key: str, default: str = "false") -> bool:
    """把 .env 里的字符串解析成布尔值。"""
    return os.getenv(key, default).strip().lower() in ("true", "1", "yes", "on")


class Settings:
    """全局配置的唯一来源。

    【怎么用】
        整个项目里任何地方要读配置，都是 `from app.config import settings`，
        然后用 `settings.XXX`。不要去别处再写一遍 `os.getenv(...)` ——
        那样一旦配置项改名或加默认值，就会漏改。

    【怎么改配置】
        改 `.env` 文件，不要改这个文件。
        下面每个属性都是 `os.getenv("环境变量名", "默认值")` 的形式：
        `.env` 里有 → 用 `.env` 的值；`.env` 里没有 → 用括号里的默认值。
        改完 `.env` 需要重启服务才生效（因为类属性在 import 时就求值完了）。

    【属性分组】
        LLM            —— 大模型（智谱 GLM，走 OpenAI 兼容协议）
        Embedding      —— 本地向量模型（HuggingFace，离线跑）
        向量库          —— Milvus 连接与 collection 配置
        RAG 参数        —— 分块大小、召回条数等基础参数
        混合检索        —— 稠密/稀疏双路检索与 RRF 融合
        查询变换        —— multi_query / hyde
        Reranker       —— Cohere 精排
        忠实性检查      —— 防幻觉（Self-RAG 反思闭环）
        Agentic RAG    —— 循环上限、图步数上限
        LangSmith      —— 链路追踪
        派生属性        —— 由上面计算得出，不需要配置
    """

    # ------------------------------------------------------------ LLM
    OPENAI_API_KEY: str = os.getenv("LLM_API_KEY")
    OPENAI_BASE_URL: str = os.getenv(
        "LLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4/"
    )
    LLM_MODEL_NAME: str = os.getenv("LLM_MODEL", "glm-4.5-air")
    LLM_TEMPERATURE: float = float(os.getenv("LLM_TEMPERATURE", 0.0))

    # ------------------------------------------------------------ Embedding
    # 本次 embedding 模型（本地 HuggingFace，384 维）
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-small-en-v1.5")

    # ------------------------------------------------------------ 向量库（Milvus）
    # 注意环境变量名叫 RAG_MILVUS_URI 而不是 MILVUS_URI，原因见文件开头的踩坑警告。
    # 三种写法都支持：
    #   ./vector_db/milvus_rag.db        -> Milvus Lite 本地文件（学习/单机，零部署）
    #   http://localhost:19530           -> Milvus 单机服务
    #   https://xxx.api.zilliz.com:443   -> Zilliz Cloud
    MILVUS_URI: str = os.getenv("RAG_MILVUS_URI", "./vector_db/milvus_rag.db")
    # 有鉴权时填 "用户名:密码" 或 Zilliz 的 API Key；本地 Lite 留空
    MILVUS_TOKEN: str = os.getenv("MILVUS_TOKEN", "")
    # 本地数据总目录（上传文件、BM25 语料副本都放这里）
    VECTOR_DB_PATH: str = os.getenv("VECTOR_DB_PATH", "./vector_db")
    # Milvus collection 名
    INDEX_NAME: str = os.getenv("INDEX_NAME", "rag_index")
    # 启动时是否重建 collection（结构变更后需要，平时保持 false）
    MILVUS_DROP_OLD: bool = _env_bool("MILVUS_DROP_OLD", "false")

    # ------------------------------------------------------------ RAG 参数
    TOP_K: int = int(os.getenv("TOP_K", 5))
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 50))

    # ------------------------------------------------------------ 混合检索
    HYBRID_ENABLED: bool = _env_bool("HYBRID_ENABLED", "true")
    # RRF 融合时稠密/稀疏两路各自的权重（0.5/0.5 表示等权）
    DENSE_WEIGHT: float = float(os.getenv("DENSE_WEIGHT", 0.5))
    SPARSE_WEIGHT: float = float(os.getenv("SPARSE_WEIGHT", 0.5))
    # 每一路先各召回多少条（粗排），再融合 + 精排
    DENSE_TOP_K: int = int(os.getenv("DENSE_TOP_K", 20))
    SPARSE_TOP_K: int = int(os.getenv("SPARSE_TOP_K", 20))
    # 融合后保留的候选数（送给 Reranker 的条数）
    FUSION_TOP_K: int = int(os.getenv("FUSION_TOP_K", 20))
    # BM25 语料副本文件名（Milvus 只存向量，BM25 需要原始文本，重启后从这里恢复）
    BM25_CORPUS_FILE: str = os.getenv("BM25_CORPUS_FILE", "bm25_corpus.jsonl")

    # ------------------------------------------------------------ 查询变换
    # none        关闭，直接用原问题检索
    # multi_query RAG-Fusion：一个问题裂变成多条查询，各自检索后 RRF 融合
    # hyde        让 LLM 先写一段"假想答案"，用这段假想答案去检索
    QUERY_TRANSFORM: str = os.getenv("QUERY_TRANSFORM", "multi_query").strip().lower()
    # multi_query 模式下额外生成几条查询（不含原问题）
    NUM_QUERIES: int = int(os.getenv("NUM_QUERIES", 3))

    # ------------------------------------------------------------ Reranker（精排）
    RERANK_ENABLED: bool = _env_bool("RERANK_ENABLED", "true")
    RERANK_PROVIDER: str = os.getenv("RERANK_PROVIDER", "cohere").strip().lower()
    COHERE_API_KEY: str = os.getenv("COHERE_API_KEY", "")
    # rerank-v3.5（多语言，推荐）/ rerank-multilingual-v3.0
    COHERE_RERANK_MODEL: str = os.getenv("COHERE_RERANK_MODEL", "rerank-v3.5")
    # 精排后最终喂给 LLM 的文档块数
    RERANK_TOP_N: int = int(os.getenv("RERANK_TOP_N", 5))

    # ------------------------------------------------------------ 忠实性检查（防幻觉）
    # 生成完答案后，会再让 LLM 判断"答案是否由检索到的文档支撑"。
    # 这个判断不总是可靠，最典型的是**跨语言数字换算**：
    #   英文事实 "7 billion to 70 billion" 对应中文 "70亿到700亿"，
    #   模型有时会把 70 billion 误读成 70亿，从而把正确答案判成幻觉。
    # 所以失败后不要立刻拒答，而是先用更严格的提示词重生成一次。
    #
    # retry  失败 -> 严格模式重生成 -> 仍失败才拒答【默认，推荐】
    # warn   失败 -> 只打印警告，保留原答案（宁可多给答案，也不要误拒）
    # off    完全不检查（最省 token，但失去防幻觉能力）
    GROUNDEDNESS_ACTION: str = os.getenv("GROUNDEDNESS_ACTION", "retry").strip().lower()
    # retry 模式下最多重新生成几次（每次多一次 LLM 调用）
    MAX_GROUNDEDNESS_RETRIES: int = int(os.getenv("MAX_GROUNDEDNESS_RETRIES", 1))

    # ------------------------------------------------------------ Agentic RAG 参数
    # 最大"检索-评分-改写"循环轮数（防止死循环）
    MAX_RETRIES: int = int(os.getenv("MAX_RETRIES", 2))
    # LangGraph 单次调用的最大步数上限（图节点+条件边执行次数），
    # 必须大于 2*(MAX_RETRIES+1)+3，否则会抛 GraphRecursionError
    RECURSION_LIMIT: int = int(os.getenv("RECURSION_LIMIT", 25))
    DATA_DIR: str = os.getenv("DATA_DIR", "./data")

    # ------------------------------------------------------------ LangSmith
    LANGSMITH_TRACING: bool = _env_bool("LANGSMITH_TRACING", "true")
    LANGSMITH_ENDPOINT: str = os.getenv(
        "LANGSMITH_ENDPOINT", "https://api.smith.langchain.com"
    )
    LANGSMITH_API_KEY: str = os.getenv("LANGSMITH_API_KEY")
    LANGSMITH_PROJECT: str = os.getenv("LANGSMITH_PROJECT", "MyFirstProject")

    # ------------------------------------------------------------ 派生属性
    @property
    def bm25_corpus_path(self) -> str:
        """BM25 语料副本的完整路径（随 VECTOR_DB_PATH 走）。"""
        return os.path.join(self.VECTOR_DB_PATH, self.BM25_CORPUS_FILE)

    @property
    def rerank_ready(self) -> bool:
        """精排是否真的可用（开关打开 + 有 Key）。"""
        if not self.RERANK_ENABLED:
            return False
        if self.RERANK_PROVIDER == "cohere":
            return bool(self.COHERE_API_KEY)
        return False


settings = Settings()
