# FastAPI 入口文件

"""FastAPI 入口：端到端文档问答 API（混合检索 + Reranker + Milvus 升级版）。

接口一览：
    GET    /health              健康检查（向量库状态、混合检索与精排配置）
    GET    /metrics             Prometheus API 指标（QPS、延迟、错误率）
    POST   /ingest/file         上传单个文档并入库（multipart/form-data）
    POST   /ingest/directory    将 data/ 目录下所有文档批量入库
    POST   /retrieve            只做检索，不调 LLM（调参神器，见下方说明）
    POST   /query               提问，执行 Agentic RAG，返回答案+来源
    DELETE /index               清空向量库

启动：
    uvicorn main:app --reload --port 8000
    或 python main.py

调参提示：
    /retrieve 只走"查询变换 -> 混合检索 -> 精排"，一次 LLM 调用都不产生
    （multi_query/HyDE 模式除外），可以反复快速试。
    先用它把 DENSE_WEIGHT / SPARSE_WEIGHT / RERANK_TOP_N 调顺，
    再去看 /query 的最终效果，能省下大量 token 和时间。
"""

import logging
import shutil
from contextlib import asynccontextmanager
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field

from redis_fastapi import FastAPIRedis , AsyncRedisDep

logging.getLogger("pypdf").setLevel(logging.ERROR)
logging.getLogger("pypdf._page").setLevel(logging.ERROR)

from app.config import settings
from app.document_processer import (
    SUPPORTED_EXTENSIONS,
    load_document,
    load_directory,
    split_documents,
)
from app.rag_chain import RAGChain
from app.observability import setup_metrics
from app.vector_store import add_documents, clear, count, stats, warmup



class QueryRequest(BaseModel):
    """`POST /query` 的请求体。"""

    question: str = Field(..., min_length=1, description="用户问题")


class SourceInfo(BaseModel):
    """单条引用来源。

    就是从向量库里检索出来的那个文本块，加上它的溯源信息。
    前端展示"参考文献"用的就是这一串。
    """

    source: str
    filename: str
    page: Optional[int] = None
    snippet: str
    # 精排分数（未启用精排时为 None）；RRF 融合分（单路检索时为 None）
    relevance_score: Optional[float] = None
    rrf_score: Optional[float] = None


class QueryResponse(BaseModel):
    """`POST /query` 的响应体。

    `answer` / `sources` 是给用户看的；
    其余字段是**观测字段** —— 这个 RAG 是四级流水线（变换→检索→精排→生成），
    当答案不对时，靠这些字段才能定位"是哪一级出了问题"：
        检索没找到      → queries 里的问法不好，或 candidates 太少
        检索到但被筛掉  → candidates 有值但 sources 为空（精排或评分把文档筛掉了）
        有文档但答错    → sources 有值，看 groundedness 是不是 failed
    """

    answer: str
    question: str
    rewritten: bool
    num_documents: int
    sources: List[SourceInfo]

    # ---- 升级后新增的观测字段，方便前端/调试看"这一轮到底做了什么" ----
    queries: List[str] = Field(default_factory=list, description="本轮实际用于检索的查询")
    transform_mode: str = Field("none", description="查询变换策略与产出条数")
    candidates: int = Field(0, description="混合检索融合后的候选数（精排前）")
    reranked: bool = Field(False, description="本轮是否执行了精排")

    groundedness: str = Field("unknown", description="忠实性自检结果")
    groundedness_retries: int = Field(0, description="触发严格模式重生成的次数")


class RetrieveRequest(BaseModel):
    """`POST /retrieve` 的请求体（调试接口，只检索不生成）。"""

    question: str = Field(..., min_length=1)
    top_k: Optional[int] = Field(None, ge=1, le=50, description="返回条数，默认取 FUSION_TOP_K")
    skip_rerank: bool = Field(False, description="跳过精排，用于对比精排前后的差异")


class RetrieveResponse(BaseModel):
    """`POST /retrieve` 的响应体。

    和 QueryResponse 的区别是：这里**只有检索结果，没有答案**。
    用途是单独检验检索层 —— 当初定位"忠实性误判"那个 bug 时，
    就是靠它一眼看出"检索到的文档其实是对的，问题出在评分环节"。
    """

    question: str
    queries: List[str]
    candidates: int
    reranked: bool
    results: List[SourceInfo]


class IngestResponse(BaseModel):
    """索引接口（`/ingest/file`、`/ingest/directory`、`DELETE /index`）的响应体。

    chunk_indexed 是本次新写入的块数，total_chunks 是写入后库里的总块数。
    """

    status: str
    chunk_indexed: int
    total_chunks: int
    detail: str = ""


# ---------------------------------------------------------------- Lifespan

rag_chain: Optional[RAGChain] = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """启动时初始化向量库与 RAG 链（Embedding 模型仅加载一次）。"""
    global rag_chain

    Path(settings.DATA_DIR).mkdir(parents=True, exist_ok=True)
    Path(settings.VECTOR_DB_PATH).mkdir(parents=True, exist_ok=True)

    # 预热：加载 Embedding 模型 -> 打开 Milvus -> 构建 BM25 倒排索引
    warmup()
    rag_chain = RAGChain()

    print(f"[startup] 向量库就绪，当前索引块数：{count()}")
    print(
        f"[startup] 混合检索={settings.HYBRID_ENABLED} "
        f"(dense={settings.DENSE_WEIGHT}, sparse={settings.SPARSE_WEIGHT}) "
        f"| 查询变换={settings.QUERY_TRANSFORM} "
        f"| 精排={settings.RERANK_ENABLED}/{settings.rerank_ready}"
    )

    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        print("[warn] 未配置 LLM_API_KEY，无法调用 LLM")
    if settings.RERANK_ENABLED and not settings.COHERE_API_KEY:
        print("[warn] 未配置 COHERE_API_KEY，精排将自动跳过（混合检索仍正常工作）")

    yield

    print("[shutdown] FastAPI 退出，清理资源")


app = FastAPI(
    title="RAG 文档问答 API",
    description=(
        "FastAPI + LangChain + LangGraph 实现的端到端 Agentic RAG 服务。"
        "LLM 为智谱 GLM（OpenAI 兼容协议），Embedding 为本地 BGE，"
        "向量库为 Milvus，检索为 稠密向量 ⊕ BM25 混合检索（RRF 融合），"
        "并使用 Cohere Rerank 精排。"
    ),
    version="2.0.0",
    lifespan=lifespan,
)

setup_metrics(
    app,
    enabled=settings.METRICS_ENABLED,
    endpoint=settings.METRICS_PATH,
)

from fastapi import FastAPI , Depends
from app.redis_client import get_redis ,close_redis
from app.config import settings


FastAPIRedis(app).lifespan()

# ---------------------------------------------------------------- 工具函数


async def _ingest_docs(docs) -> IngestResponse:
    """将文档列表分块后写入向量库，返回写入结果。"""
    if not docs:
        return IngestResponse(
            status="empty", chunk_indexed=0, total_chunks=count(), detail="没有可入库的文档"
        )

    chunks = split_documents(docs)
    num_indexed = add_documents(chunks)
    return IngestResponse(
        status="success" if num_indexed > 0 else "failed",
        chunk_indexed=num_indexed,
        total_chunks=count(),
        detail=f"已入库 {num_indexed}/{len(chunks)} 个文档块",
    )


def _to_source(document) -> SourceInfo:
    """把 Document 转成接口返回的 SourceInfo。"""
    page = document.metadata.get("page")
    return SourceInfo(
        source=document.metadata.get("source", ""),
        filename=document.metadata.get("filename", ""),
        page=page + 1 if page is not None else None,
        snippet=document.page_content[:200],
        relevance_score=document.metadata.get("relevance_score"),
        rrf_score=document.metadata.get("rrf_score"),
    )


# ---------------------------------------------------------------- 接口

@app.get("/items")
async def get_item(redis: AsyncRedisDep):
    return {"items" : await redis.get("items")}



@app.get("/health")
async def health():
    """健康检查：暴露运行时配置，方便排查"到底生效的是哪套参数"。"""
    return {
        "status": "ok",
        "version": "2.0.0",
        "llm_model": settings.LLM_MODEL_NAME,
        "llm_base_url": settings.OPENAI_BASE_URL,
        "vector_store": "milvus",
        "retrieval": {
            "hybrid_enabled": settings.HYBRID_ENABLED,
            "dense_weight": settings.DENSE_WEIGHT,
            "sparse_weight": settings.SPARSE_WEIGHT,
            "dense_top_k": settings.DENSE_TOP_K,
            "sparse_top_k": settings.SPARSE_TOP_K,
            "fusion_top_k": settings.FUSION_TOP_K,
        },
        "query_transform": {
            "mode": settings.QUERY_TRANSFORM,
            "num_queries": settings.NUM_QUERIES,
        },
        "rerank": {
            "enabled": settings.RERANK_ENABLED,
            "provider": settings.RERANK_PROVIDER,
            "model": settings.COHERE_RERANK_MODEL,
            "top_n": settings.RERANK_TOP_N,
            "ready": settings.rerank_ready,
        },
        "vector_store_stats": stats(),
        "metrics": {
            "enabled": settings.METRICS_ENABLED,
            "path": settings.METRICS_PATH,
        },
        "api_key_configured": bool(
            settings.OPENAI_API_KEY and settings.OPENAI_API_KEY != "YOUR_OPENAI_API_KEY"
        ),
    }


@app.post("/ingest/file", response_model=IngestResponse)
async def ingest_file(file: UploadFile = File(...)):
    """上传文档（pdf/txt/md/docx/html/csv）并分块入库。"""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件类型：{suffix}，支持：{sorted(SUPPORTED_EXTENSIONS)}",
        )

    upload_dir = Path(settings.DATA_DIR) / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    dest = upload_dir / Path(file.filename).name

    with open(dest, "wb") as handle:
        shutil.copyfileobj(file.file, handle)

    try:
        docs = load_document(str(dest))
    except Exception as error:
        # 原代码这里写的是 HTTPException(status=...)，status 不是合法参数，
        # 会变成 TypeError 而不是 422，顺手改掉。
        raise HTTPException(status_code=422, detail=f"文档解析失败：{error}")

    resp = _ingest_docs(docs)
    if resp.status != "success":
        raise HTTPException(status_code=500, detail=f"文档入库失败：{resp.detail}")

    return resp


@app.post("/ingest/directory", response_model=IngestResponse)
async def ingest_directory():
    """把 data/ 目录下所有支持的文档批量入库。

    注意：这是**追加**写入，不去重。同一批文件重复调用会往库里塞重复块。
    需要重来时先 DELETE /index。
    """
    docs = load_directory(settings.DATA_DIR)
    return _ingest_docs(docs)


@app.post("/retrieve", response_model=RetrieveResponse)
async def retrieve_debug(request: RetrieveRequest):
    """只做检索调试：查询变换 -> 混合检索 -> 精排，不生成答案。

    用来对比不同配置下的召回质量，比直接看最终答案直观得多：
    - 同一问题分别开/关 skip_rerank，看精排把哪几条提上来了
    - 调 DENSE_WEIGHT / SPARSE_WEIGHT，看专有名词类问题的命中变化
    """
    if count() == 0:
        raise HTTPException(
            status_code=400, detail="向量库为空，请先上传文档入库"
        )

    # 延迟导入，避免与 rag_chain 的模块级依赖形成循环
    from app.rag_chain import rerank_documents, retrieve, transform_query

    question = request.question.strip()
    state = {
        "question": question,
        "original_question": question,
        "queries": [],
        "documents": [],
        "generation": "",
        "rewritten": False,
        "retries": 0,
        "rewrite_unchanged": False,
        "candidates": 0,
        "reranked": False,
        "transform_mode": "none",
        "skip_check": False,
    }

    state.update(transform_query(state))
    state.update(retrieve(state))

    candidates = state["candidates"]

    if not request.skip_rerank:
        state.update(rerank_documents(state))

    documents = state["documents"]
    if request.top_k:
        documents = documents[: request.top_k]

    return RetrieveResponse(
        question=question,
        queries=state["queries"],
        candidates=candidates,
        reranked=state["reranked"],
        results=[_to_source(doc) for doc in documents],
    )


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """Agentic RAG 问答完整流程：

    查询变换 -> 混合检索 -> 精排 -> 文档评分
        -> (无相关文档时 改写查询 / 直接兜底)
        -> 生成答案
        -> 忠实性自检 -> (未通过且还有额度时 严格模式重生成) -> 返回

    返回值里的 `groundedness` 字段说明忠实性自检的结果，
    `queries` / `candidates` / `reranked` 说明本轮检索各阶段实际做了什么。
    """
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="RAG 链未初始化")
    if not settings.OPENAI_API_KEY or settings.OPENAI_API_KEY == "YOUR_OPENAI_API_KEY":
        raise HTTPException(status_code=503, detail="未配置 LLM_API_KEY，无法调用 LLM")
    if count() == 0:
        raise HTTPException(
            status_code=400,
            detail="向量库为空，请先调用 /ingest/file 或 /ingest/directory 上传文档入库",
        )

    try:
        result = rag_chain.answer(request.question.strip())
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"RAG 链执行失败：{error}")

    return QueryResponse(
        answer=result["answer"],
        question=result["question"],
        rewritten=result["rewritten"],
        num_documents=result["num_documents"],
        sources=[SourceInfo(**source) for source in result["sources"]],
        queries=result.get("queries", []),
        transform_mode=result.get("transform_mode", "none"),
        candidates=result.get("candidates", 0),
        reranked=result.get("reranked", False),
        groundedness=result.get("groundedness", "unknown"),
        groundedness_retries=result.get("groundedness_retries", 0),
    )


@app.delete("/index")
async def delete_index():
    """清空向量库（删除 Milvus collection + BM25 语料副本，并重置单例）。"""
    clear()  # clear() 内部已重置并重建单例，无需 cache_clear
    return {"status": "ok", "detail": "向量库已清空"}


if __name__ == "__main__":
    import os

    import uvicorn

    os.environ["LANGSMITH_TRACING"] = "true"
    os.environ["LANGSMITH_API_KEY"] = settings.LANGSMITH_API_KEY or ""
    os.environ["LANGSMITH_PROJECT"] = settings.LANGSMITH_PROJECT
    os.environ["LANGSMITH_ENDPOINT"] = settings.LANGSMITH_ENDPOINT

    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
