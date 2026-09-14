"""向量库 + 混合检索（Milvus 稠密检索 ⊕ BM25 稀疏检索，RRF 融合）。

设计说明
--------
1. 稠密检索：Milvus 存 BGE 向量，负责"语义相近"——问"如何提升检索质量"能命中
   讲"recall 优化"的段落。
2. 稀疏检索：BM25 负责"关键词精确命中"——专有名词、型号、错误码、人名这类
   语义模型容易漂移的东西，BM25 往往一把命中。
3. 融合：Reciprocal Rank Fusion（RRF）。不用加权求和而用 RRF，是因为稠密相似度
   （余弦，0~1）和 BM25 分数（无上界，可能是 0.3 也可能是 27）**量纲完全不同**，
   直接加权毫无意义；RRF 只看"排名第几"，天然免疫量纲问题：

        score(doc) = Σ  weight_i * 1 / (c + rank_i(doc))      c 取 60

   这正是 Milvus 原生 hybrid search 的默认融合策略，也是 LlamaIndex
   QueryFusionRetriever 的默认融合策略。

为什么自己实现 RRF，不用 LangChain 的 EnsembleRetriever？
    langchain 1.x 已经把 langchain.retrievers 模块移除了（实测 ImportError），
    EnsembleRetriever 挪到了 langchain_classic，且只支持"两路"。
    自己写 20 行，既不依赖版本，又能顺带复用去实现跨查询融合（RAG-Fusion）。

关于 BM25 语料持久化
    Milvus 只存向量和文本，BM25 需要重新构建倒排索引。为了让服务重启后
    BM25 依然可用，写入时把文本块同步落一份 JSONL 副本，启动时读回重建。

参考：
- LlamaIndex Advanced Retrieval / Query Transformations：QueryFusionRetriever 融合多路召回
- Milvus 官方 hybrid search（RRF / weighted ranker）
- llm-cookbook《Advanced Retrieval for AI》：混合检索 + 精排的组合拳
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import sys
import threading
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from langchain_core.documents import Document

from app.config import settings

# RRF 公式里的常数 c。c 越大，排名靠后的文档被"拉平"得越厉害；
# 60 是 Cormack 等人提出 RRF 时给出的经验值，也是 Milvus / LlamaIndex 的默认值。
RRF_C = 60

_BM25_LOCK = threading.Lock()
_BM25_RETRIEVER = None  # 模块级缓存，避免每次检索都重建倒排索引


# ================================================================ Embedding


def _detect_device() -> str:
    """自动选择推理设备。

    注意：原代码写的是 {"device": " cpu"}——引号里有空格，
    sentence-transformers 拿到 " cpu" 这种带空格的字符串会直接报错。
    """
    try:
        import torch

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


@lru_cache(maxsize=1)
def get_embeddings():
    """本地 HuggingFace Embedding（单例，避免重复加载模型权重）。

    - BAAI/bge-small-en-v1.5: 384 维，轻量，CPU 即可跑
    - normalize_embeddings=True: 归一化后内积等价于余弦相似度，检索更稳
    """
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=settings.EMBEDDING_MODEL,
        model_kwargs={"device": _detect_device()},
        encode_kwargs={"normalize_embeddings": True},
    )


def embedding_dim() -> int:
    """探测当前 Embedding 模型的向量维度（建表用）。"""
    return len(get_embeddings().embed_query("dimension probe"))


# ================================================================ Milvus


def _connection_args() -> Dict[str, str]:
    """组装 Milvus 连接参数。

    - 本地文件（Milvus Lite）：./vector_db/milvus_rag.db
    - 单机服务：http://localhost:19530
    - Zilliz Cloud：https://xxx.zillizcloud.com   + token
    """
    args: Dict[str, str] = {"uri": settings.MILVUS_URI}

    # 本地文件模式要先保证父目录存在，否则 Milvus Lite 打不开库
    if not settings.MILVUS_URI.startswith(("http://", "https://")):
        parent = Path(settings.MILVUS_URI).expanduser().resolve().parent
        parent.mkdir(parents=True, exist_ok=True)

    if settings.MILVUS_TOKEN:
        args["token"] = settings.MILVUS_TOKEN

    return args


def _exception_chain(error: BaseException) -> str:
    """把整条异常链（__cause__ / __context__）拼成一个字符串，用于关键字判断。

    第三方库里真正的原因常常被包装了一层，只看最外层异常会漏掉关键信息。
    """
    parts: List[str] = []
    seen = set()
    current: Optional[BaseException] = error

    while current is not None and id(current) not in seen:
        seen.add(id(current))
        parts.append(f"{type(current).__name__}: {current}")
        current = current.__cause__ or current.__context__

    return " | ".join(parts)


def _local_db_locked(db_path: str) -> bool:
    """探测本地 Milvus 库的独占锁是否已被别的进程持有。

    为什么要自己探：milvus_lite 的 `start_and_get_uri()` 把
    `DataDirLockedError` **吞掉并只打到 stderr**，然后返回 None，
    pymilvus 再统一抛出没营养的 `Open local milvus failed`。
    异常链里根本拿不到真实原因，只能按它的加锁方式自己试一次。

    milvus_lite 对 <库目录>/LOCK 加独占锁：Windows 用 msvcrt.locking，
    Unix 用 fcntl.flock，都是非阻塞模式。
    - 加得上   -> 没人占用（立刻解锁，无副作用）
    - 加不上   -> 确实有别的进程在用这个库
    """
    if not os.path.isdir(db_path):
        return False  # 目录都还没建，不是锁的问题

    try:
        fd = os.open(os.path.join(db_path, "LOCK"), os.O_CREAT | os.O_RDWR, 0o644)
    except OSError:
        return False

    try:
        if sys.platform == "win32":
            import msvcrt

            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        else:
            import fcntl

            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            fcntl.flock(fd, fcntl.LOCK_UN)
        return False  # 我们加锁成功了，说明没人占用
    except OSError:
        return True  # 加锁失败，说明别的进程占着
    finally:
        try:
            os.close(fd)
        except OSError:
            pass


def _explain_milvus_open_failure(error: BaseException) -> None:
    """把 pymilvus 那句含糊的报错翻译成能直接照做的提示，然后抛出 RuntimeError。

    pymilvus 打不开本地库时，无论什么原因，统一抛：
        ConnectionConfigException: Open local milvus failed
    真实原因（目录不存在 / 被别的进程锁住 / 没装 milvus-lite）全被吞在下层，
    排查起来非常费劲。这里逐一还原。

    为什么会踩到这个坑：
    上线时服务已经在 8000 端口跑着（占着库的独占锁），
    再开一个终端执行 `python main.py`，就会看到这句毫无信息量的
    "Open local milvus failed"，完全猜不到是"库被占用了"。
    """
    text = _exception_chain(error)

    # 情况一：没装 milvus-lite，只有远程客户端
    if "milvus-lite is required" in text or "No module named 'milvus_lite'" in text:
        raise RuntimeError(
            "本地文件模式需要 milvus-lite，当前环境没装：\n"
            "    pip install milvus-lite\n"
            "或改用远程 Milvus：把 .env 的 RAG_MILVUS_URI 设为 http://localhost:19530"
        ) from error

    # 情况二：地址格式不对（例如误用了保留变量名 MILVUS_URI 被提前解析）
    if "Illegal uri" in text:
        raise RuntimeError(
            f"Milvus 地址格式不对：{settings.MILVUS_URI}\n"
            "本地文件模式应形如 ./vector_db/milvus_rag.db，"
            "远程服务应形如 http://localhost:19530。\n"
            "注意：配置项必须叫 RAG_MILVUS_URI。使用 MILVUS_URI 这个变量名会让 "
            "pymilvus 在 import 阶段就去解析它，从而直接失败。"
        ) from error

    # 情况三：本地文件模式打开失败。pymilvus 的报错里没有有效信息，
    # 所以用 _local_db_locked() 主动探测一把，把真正的原因找出来。
    is_local_file = (
        settings.MILVUS_URI.endswith(".db")
        and not settings.MILVUS_URI.startswith(("http://", "https://"))
    )

    if is_local_file and "Open local milvus failed" in text:
        db_path = os.path.abspath(os.path.expanduser(settings.MILVUS_URI))
        parent = os.path.dirname(db_path)

        if not os.path.isdir(parent):
            raise RuntimeError(
                f"Milvus 本地库的上级目录不存在：{parent}\n"
                f"请先创建该目录，或确认 .env 里 RAG_MILVUS_URI（当前值 {settings.MILVUS_URI}）"
                "配置正确。"
            ) from error

        if _local_db_locked(db_path):
            raise RuntimeError(
                f"Milvus Lite 本地库已被另一个进程占用，无法打开：\n    {db_path}\n\n"
                "Milvus Lite 对同一个库文件加【独占锁】，同一时刻只允许一个进程打开。\n"
                "最常见的场景：已经有一个 `python main.py` 在跑，又启动了一次。\n\n"
                "排查与处理：\n"
                "    netstat -ano | findstr :8000          查看 8000 端口被哪个进程占用\n"
                "    taskkill /PID <上面查到的进程号> /F    停掉那个进程后再启动\n\n"
                "如果你确实需要多个实例共用同一个库，请改用远程 Milvus：\n"
                "    docker run -d --name milvus -p 19530:19530 milvusdb/milvus:latest\n"
                "然后把 .env 里 RAG_MILVUS_URI 改成 http://localhost:19530\n"
                "（远程模式下多进程共用同一个库不会冲突）"
            ) from error

        raise RuntimeError(
            f"Milvus 本地库打开失败：{db_path}\n"
            "锁没有被占用、上级目录也存在，可能是库文件损坏或权限不足。\n"
            "可以试着删掉该目录让它重建（注意会丢失已入库的数据）：\n"
            f"    rmdir /s /q \"{db_path}\""
        ) from error

    # 其他情况：保留原始异常，但把异常链打印出来方便定位
    print(f"    [warn] Milvus 打开失败，异常链：{text}")


@lru_cache(maxsize=1)
def get_vector_store():
    """Milvus 向量库（collection 名为 INDEX_NAME）。


    【混合 metadata 踩坑 —— enable_dynamic_field 必须为 True】
    ----------------------------------------------------------
    langchain-milvus 默认 `enable_dynamic_field=False`，此时它会拿**第一次入库
    时扫到的全部 metadata key** 去建固定字段（见 `_add_metadata_fields`）。

    问题在于我们的 metadata 是**异构**的：
        - PyPDFLoader 会给每页附加 producer / creator / creationdate / total_pages …
        - TextLoader 只有 source / filename / start_index
        - CSVLoader 会把 CSV 的列名带上
        - 我们自己在 document_processer 里加的 filename

    而 `/ingest/directory` 是把**所有文件合并成一个列表**一次性 add_documents 的
    （见 main.py 的 _ingest_docs）。于是：
        1. 建表时扫到 llama2.pdf 的 `producer`，把它建成固定字段；
        2. 接着插入 Data.csv 的块 —— 它没有 `producer` —— 直接报错：
           DataNotMatchException: Insert missed an field `producer` to collection
           without set nullable==true or set default_value
        3. 整批插入失败，用户看到 500。

    开启动态字段后，所有 metadata 统一进 Milvus 的 `$meta` JSON 字段，
    任意形状的 metadata 都能写入，检索时也会自动回填。
    """
    from langchain_milvus import Milvus

    try:
        return Milvus(
            embedding_function=get_embeddings(),
            collection_name=settings.INDEX_NAME,
            connection_args=_connection_args(),
            # 主键交给 Milvus 自动生成，避免重复插入时主键冲突
            auto_id=True,
            # True 时每次启动重建 collection（改了向量维度/切了 embedding 模型才需要）
            drop_old=settings.MILVUS_DROP_OLD,
            # 统一用文本字段存原文，检索结果回填 metadata
            text_field="text",
            vector_field="vector",
            # 【必须开】见下方"混合 metadata 踩坑"说明
            enable_dynamic_field=True,
        )
    except Exception as error:
        _explain_milvus_open_failure(error)
        raise


# ================================================================ BM25 语料副本


def _append_corpus(docs: Sequence[Document]) -> None:
    """把文本块追加写入 JSONL，供服务重启后重建 BM25 使用。"""
    if not docs:
        return

    path = Path(settings.bm25_corpus_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "a", encoding="utf-8") as handle:
        for doc in docs:
            handle.write(
                json.dumps(
                    {
                        "page_content": doc.page_content,
                        "metadata": doc.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def load_corpus() -> List[Document]:
    """读回 BM25 语料副本。文件不存在或损坏时返回空列表。"""
    path = Path(settings.bm25_corpus_path)
    if not path.exists():
        return []

    documents: List[Document] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            content = record.get("page_content", "")
            if content:
                documents.append(
                    Document(page_content=content, metadata=record.get("metadata") or {})
                )

    return documents


def _clear_corpus() -> None:
    """删除 BM25 语料副本。"""
    path = Path(settings.bm25_corpus_path)
    if path.exists():
        try:
            path.unlink()
        except OSError as error:
            print(f"    [warn] BM25 语料副本删除失败：{error}")


# ================================================================ 分词


_CJK_PATTERN = re.compile(r"[\u4e00-\u9fff]")
_TOKEN_PATTERN = re.compile(r"[a-zA-Z0-9_]+|[\u4e00-\u9fff]")


def _tokenize(text: str) -> List[str]:
    """BM25 分词器。

    BM25Retriever 默认用 `text.split()`，对中文等于整句一个 token，完全失效。
    这里优先用 jieba；没装 jieba 就退化为"英文按词 + 中文按字 + 相邻字二元组"，
    二元组能把"向量数据库"拆成 向量/量数/数据/据库，召回效果接近分词。
    """
    text = (text or "").lower()

    try:
        import jieba  # type: ignore

        return [token for token in jieba.lcut(text) if token.strip()]
    except Exception:
        pass

    tokens = _TOKEN_PATTERN.findall(text)

    # 给中文补相邻二元组，弥补无分词器时的召回损失
    bigrams: List[str] = []
    for left, right in zip(tokens, tokens[1:]):
        if _CJK_PATTERN.match(left) and _CJK_PATTERN.match(right):
            bigrams.append(left + right)

    return tokens + bigrams


# ================================================================ BM25 检索器


def _invalidate_bm25() -> None:
    """清空 BM25 缓存（写入/清库后调用）。"""
    global _BM25_RETRIEVER
    with _BM25_LOCK:
        _BM25_RETRIEVER = None


def get_bm25_retriever():
    """构建（并缓存）BM25 检索器；语料为空时返回 None。"""
    global _BM25_RETRIEVER
    with _BM25_LOCK:
        if _BM25_RETRIEVER is not None:
            return _BM25_RETRIEVER

        documents = load_corpus()
        if not documents:
            return None

        try:
            from langchain_community.retrievers import BM25Retriever

            retriever = BM25Retriever.from_documents(
                documents,
                preprocess_func=_tokenize,
                k=settings.SPARSE_TOP_K,
            )
        except Exception as error:
            print(f"    [warn] BM25 检索器构建失败，将退化为纯向量检索：{error}")
            return None

        _BM25_RETRIEVER = retriever
        return _BM25_RETRIEVER


# ================================================================ RRF 融合


def _doc_key(doc: Document) -> str:
    """文档去重用的稳定 key。

    【踩坑记录】这里一开始写的是"优先用 Milvus 主键 pk"，跑测试才发现是错的：

        稠密检索走 Milvus，返回的 metadata 里带 pk 字段；
        BM25 走的是本地语料副本，那份 metadata 里没有 pk。
        于是同一块文档，两路算出来的 key 不一样，RRF 融合时被当成两个不同结果，
        最终返回的 Top-K 里出现了重复文档。

    现在统一用「来源 + 内容哈希」：
    - 与检索通路无关，稠密/稀疏/多查询三条路径算出的 key 一定一致
    - 用完整内容的 md5 而不是前 N 个字符，避免"长尾相同前缀"的块被误判为同一块
    - 顺带把重复入库产生的完全相同的块也合并掉
    """
    digest = hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()[:16]
    return f"{doc.metadata.get('source', '')}|{digest}"


def rrf_fuse(
    rank_lists: Sequence[Sequence[Document]],
    top_k: int,
    weights: Optional[Sequence[float]] = None,
) -> List[Document]:
    """Reciprocal Rank Fusion：把多路召回结果按排名融合成一路。

    参数
    ----
    rank_lists : 每一路的召回结果，已按相关性从高到低排序
    top_k      : 融合后保留多少条
    weights    : 每一路的权重，缺省等权

    返回
    ----
    融合后的文档列表（降序），每条的 metadata 里写入了 rrf_score 与命中路数。
    """
    if not rank_lists:
        return []

    scores: Dict[str, float] = {}
    hits: Dict[str, int] = {}
    payload: Dict[str, Document] = {}

    for index, docs in enumerate(rank_lists):
        weight = 1.0 if not weights else float(weights[index])
        for rank, doc in enumerate(docs):
            key = _doc_key(doc)
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (RRF_C + rank + 1))
            hits[key] = hits.get(key, 0) + 1
            payload.setdefault(key, doc)

    ordered = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:top_k]

    fused: List[Document] = []
    for key, score in ordered:
        doc = payload[key]
        # 复制 metadata，避免 lru_cache 缓存的 Document 对象被反复覆写
        doc.metadata = {
            **doc.metadata,
            "rrf_score": round(score, 6),
            "rrf_hits": hits[key],
        }
        fused.append(doc)

    return fused


# ================================================================ 检索


def dense_search(query: str, k: Optional[int] = None) -> List[Document]:
    """稠密（向量）检索。"""
    store = get_vector_store()
    return store.similarity_search(query, k=k or settings.DENSE_TOP_K)


def sparse_search(query: str, k: Optional[int] = None) -> List[Document]:
    """稀疏（BM25 关键词）检索；BM25 不可用时返回空列表。"""
    retriever = get_bm25_retriever()
    if retriever is None:
        return []

    retriever.k = k or settings.SPARSE_TOP_K
    try:
        return retriever.invoke(query)
    except Exception as error:
        print(f"    [warn] BM25 检索失败：{error}")
        return []


def search(query: str, k: Optional[int] = None) -> List[Document]:
    """单查询混合检索：稠密 ⊕ 稀疏，RRF 融合。

    这是 Hybrid Search 的入口。HYBRID_ENABLED=false 时退化为纯向量检索。
    """
    limit = k or settings.FUSION_TOP_K

    if not settings.HYBRID_ENABLED:
        return dense_search(query, k=limit)

    dense_docs = dense_search(query, k=settings.DENSE_TOP_K)
    sparse_docs = sparse_search(query, k=settings.SPARSE_TOP_K)

    # BM25 不可用（没装/语料为空）时直接返回稠密结果，不让整条链路挂掉
    if not sparse_docs:
        return dense_docs[:limit]

    return rrf_fuse(
        [dense_docs, sparse_docs],
        top_k=limit,
        weights=[settings.DENSE_WEIGHT, settings.SPARSE_WEIGHT],
    )


def multi_query_search(
    queries: Sequence[str],
    k: Optional[int] = None,
    strategy: str = "rrf",
) -> List[Document]:
    """多查询融合检索（RAG-Fusion）。

    对每条查询各做一次混合检索，再用 RRF 跨查询融合。
    对应 LlamaIndex QueryFusionRetriever 的默认行为（mode="reciprocal_rerank"）。

    为什么有效：一个问题的多种问法会命中不同的文档块，
    融合后召回面明显变宽；RRF 又能把"多路都排前面"的块提上来，
    等于用排名投票做了一次无需训练的过滤。后面还有 Reranker 兜底精排，
    所以这里召回宁宽勿窄。
    """
    effective = [q.strip() for q in queries if q and q.strip()]
    if not effective:
        return []

    # 单条查询没必要走融合
    if len(effective) == 1:
        return search(effective[0], k=k)

    if strategy == "rrf":
        per_query = [search(q, k=settings.FUSION_TOP_K) for q in effective]
        return rrf_fuse(per_query, top_k=k or settings.FUSION_TOP_K)

    # 备用策略：按"被多少条查询命中"投票，命中数相同再看最好名次
    pooled: Dict[str, Document] = {}
    best_rank: Dict[str, int] = {}
    for query in effective:
        for rank, doc in enumerate(search(query, k=settings.FUSION_TOP_K)):
            key = _doc_key(doc)
            pooled.setdefault(key, doc)
            best_rank[key] = min(best_rank.get(key, 10**6), rank)

    ordered = sorted(pooled.keys(), key=lambda key: best_rank[key])
    return [pooled[key] for key in ordered[: (k or settings.FUSION_TOP_K)]]


# ================================================================ 写入 / 统计


def add_documents(docs: List[Document]) -> int:
    """把文本块写入 Milvus，同时落一份 BM25 语料副本。返回写入数量。"""
    if not docs:
        return 0

    store = get_vector_store()

    try:
        store.add_documents(list(docs))
    except Exception as error:
        # 最常见的失败是"旧 collection 的表结构和当前数据对不上"：
        # 早期用 enable_dynamic_field=False 建的库，会把某个文件的 metadata
        # 字段（比如 PDF 的 producer）固化成字段，之后再插别的文件就报
        # DataNotMatchException。换过 Embedding 模型导致向量维度变化也是类似症状。
        text = _exception_chain(error)
        if "DataNotMatchException" in text or "missed an field" in text:
            raise RuntimeError(
                f"写入 Milvus 失败：数据字段和现有 collection 的表结构对不上。\n\n"
                f"这通常说明 collection「{settings.INDEX_NAME}」是旧版本建的"
                "（早期关闭了动态字段，或换过 Embedding 模型导致向量维度变化）。\n\n"
                "处理办法（二选一，都会清空现有知识库）：\n"
                "  1) 调用 DELETE /index 清库，然后重新导入：\n"
                "         curl -X DELETE http://127.0.0.1:8000/index\n"
                "         curl -X POST   http://127.0.0.1:8000/ingest/directory\n"
                "  2) 或把 .env 里 MILVUS_DROP_OLD 临时改成 true，重启服务一次，\n"
                "     然后立刻改回 false（否则每次重启都会清空知识库）。\n\n"
                f"底层报错：{text}"
            ) from error
        raise

    # 向量写完再写语料副本，并让 BM25 下次检索时重建
    _append_corpus(docs)
    _invalidate_bm25()

    return len(docs)


def count() -> int:
    """当前向量库中的实体数量。

    用 count(*) 查询而不是 get_collection_stats：后者的 row_count 是 Milvus 的
    统计缓存，刚写入的数据可能还没算进去（实测会滞后），清库后也不一定立刻归零。
    这个数字会直接展示给用户看"入库了多少块"，不能是近似值。
    """
    try:
        client = get_vector_store().client  # langchain-milvus 0.4 起是 @property
    except Exception:
        return 0

    # 必须先判断 collection 是否存在。
    # langchain-milvus 是**惰性建表**的：没有文档时 collection 根本不会被创建，
    # 此时直接 query 会抛 MilvusException(code=100)，而 pymilvus 会顺手把
    # 一整段 ERROR 堆栈打到终端 —— 服务刚启动、库还是空的时候，
    # 每次调 /health 都会刷两屏没用的报错。提前判断就能完全避免。
    try:
        if not client.has_collection(settings.INDEX_NAME):
            return 0
    except Exception:
        return 0

    try:
        rows = client.query(
            collection_name=settings.INDEX_NAME,
            filter="",
            output_fields=["count(*)"],
        )
        if rows:
            return int(rows[0].get("count(*)", 0))
    except Exception:
        pass

    # 兜底：统计缓存（可能滞后，但聊胜于无）
    try:
        stats = client.get_collection_stats(collection_name=settings.INDEX_NAME)
        return int(stats.get("row_count", 0))
    except Exception:
        return 0


def clear() -> None:
    """清空向量库：删除 collection、BM25 语料副本，并重置全部单例。"""
    try:
        store = get_vector_store()
        client = getattr(store, "client", None)
        if client is not None and client.has_collection(settings.INDEX_NAME):
            client.drop_collection(collection_name=settings.INDEX_NAME)
    except Exception as error:
        print(f"    [warn] 通过 langchain-milvus 删除 collection 失败，改用原生客户端：{error}")
        try:
            from pymilvus import MilvusClient

            client = MilvusClient(**_connection_args())
            if client.has_collection(settings.INDEX_NAME):
                client.drop_collection(settings.INDEX_NAME)
        except Exception as inner:
            print(f"    [warn] 原生客户端删除失败：{inner}")

    # 单例必须先失效，否则下次拿到的是指向旧 collection 的空壳对象
    get_vector_store.cache_clear()
    _invalidate_bm25()
    _clear_corpus()
    print("[vector_store] 向量库已清空")


def warmup() -> None:
    """预热：加载 Embedding 模型、打开 Milvus、构建 BM25 索引。

    放在 FastAPI lifespan 里调用，避免第一个请求承担全部冷启动开销。
    """
    get_vector_store()
    get_bm25_retriever()


def stats() -> Dict[str, object]:
    """给 /health 用的运行时概览。"""
    bm25_docs = len(load_corpus())
    return {
        "milvus_uri": settings.MILVUS_URI,
        "collection": settings.INDEX_NAME,
        "index_count": count(),
        "embedding_model": settings.EMBEDDING_MODEL,
        "embedding_device": _detect_device(),
        "hybrid_enabled": settings.HYBRID_ENABLED,
        "hybrid_weights": {
            "dense": settings.DENSE_WEIGHT,
            "sparse": settings.SPARSE_WEIGHT,
        },
        "bm25_corpus_chunks": bm25_docs,
        "bm25_ready": bm25_docs > 0,
    }
