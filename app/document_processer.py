# 文档加载与文档分块
# ============================================================================
# 这是整条 RAG 链路的「入口」—— 负责把磁盘上的原始文件变成一串 Document 对象。
#
# 它处在流水线的第一环：
#   磁盘文件 --> [本文件] load_document / load_directory
#            --> [本文件] split_documents（切块）
#            --> vector_store.add_documents（BGE 向量化 + 写入 Milvus）
#            --> rag_chain（检索 + 生成）

import asyncio

from langchain_community.document_loaders import PyPDFLoader , TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import os
from app.config import settings
from pathlib import Path

SUPPORTED_EXTENSIONS = {".pdf" , ".txt" , ".md" , ".docx" , ".html" , ".htm" , ".csv"}


def load_document(file_path: str) -> List[Document]:
    """按照文件类型调度对应的 LangChain Loader，把「一个文件」读成「一串 Document」。

    参数：
        file_path: 文件的绝对或相对路径。

    返回：
        List[Document]。注意是**列表**而不是单个对象 ——
        因为 Loader 通常会按"页/行/段"拆成多条 Document，
        例如一个 30 页的 PDF 会返回 30 个 Document（每页一个）。

    写入的 metadata（后续检索溯源全靠它们）：
        - source   ：文件路径，LangChain 的 Loader 自动写入；
        - page     ：页码，PDF 独有，用于回答里标注"出自第几页"；
        - filename ：本函数统一补充的纯文件名（不含目录），
                     因为 source 可能是一长串绝对路径，展示给用户不好看。

    异常：
        FileNotFoundError      文件不存在；
        ValueError             后缀不在 SUPPORTED_EXTENSIONS 白名单里。
    """
    path = Path(file_path)

    # 先做存在性校验。如果不校验，Loader 内部会抛出一堆更难懂的底层异常
    # （比如 pypdf 的 "stream ends early"），不利于排查。
    if not path.exists():
        raise FileNotFoundError(f"文件不存在：{file_path}")

    # .lower() 是为了兼容 .PDF / .Pdf 这类大小写写法
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        # PDF：按页解析，每页一个 Document，metadata 里带 page 字段
        from langchain_community.document_loaders import PyPDFLoader
        loader = PyPDFLoader(str(path))

    elif suffix in (".txt" , ".md"):
        # Markdown 用纯文本方式加载，避免引入重量级的 unstructured 依赖。
        # 指定 encoding="utf-8" 是必须的：Windows 下默认编码是 GBK，
        # 读 UTF-8 的中文文件会直接 UnicodeDecodeError。
        from langchain_community.document_loaders import TextLoader
        loader = TextLoader(str(path) , encoding="utf-8")

    elif suffix == ".docx":
        # docx2txt 只提取正文文字，不保留表格/样式。
        # 对 RAG 场景够用（我们只需要可检索的文本）
        from langchain_community.document_loaders import Docx2txtLoader
        loader = Docx2txtLoader(str(path))

    elif suffix in (".html" , ".htm"):
        # BSHTMLLoader 会剥掉 HTML 标签，只留可见文本
        from langchain_community.document_loaders import BSHTMLLoader
        loader = BSHTMLLoader(str(path))

    elif suffix == ".csv":
        # CSVLoader 默认**每行生成一个 Document**，并把列名+列值拼成
        # "列名: 值\n列名: 值" 的文本 —— 这样每行都能被独立检索到。
        from langchain_community.document_loaders import CSVLoader
        loader = CSVLoader(str(path))

    else:
        # 白名单之外的格式直接拒绝，而不是静默返回空列表，
        # 免得用户以为"文件加载成功了"却检索不到内容
        raise ValueError(f"不支持的文件类型：{suffix}，支持：{sorted(SUPPORTED_EXTENSIONS)}")

    docs = loader.load()

    # 统一补充元数据：文件名（不含路径），方便 API 返回时溯源展示。
    # 这里用 for 循环而不是列表推导，是因为要**原地修改**每个 Document
    # （Document 是可变对象，直接改 metadata 即可，不会触发深拷贝）。
    for doc in docs:
        doc.metadata["filename"] = path.name

    return docs


def load_directory (dir_path: str = None) -> List[Document]:
    """遍历目录（含子目录），加载所有支持类型的文档，合并成一个列表。

    参数：
        dir_path: 目录路径。传 None 时回退到 settings.DATA_DIR（默认 ./data）。

    返回：
        List[Document]，目录下所有文件解析结果的并集。

    设计要点：
        1) rglob("*") 会**递归**进入子目录，所以 ./data/2024/report.pdf 也能被扫到；
        2) sorted() 保证遍历顺序稳定 —— 这点很重要，
           因为 Milvus 建表时会参考"第一批文档的 metadata 结构"，
           顺序稳定能让每次重建的结果一致，方便复现和排查；
        3) 单个文件解析失败**只跳过、不中断**：一个坏 PDF 不应该让整次索引全盘失败。
           失败信息会打印到终端（带文件名），方便你事后单独处理。
    """

    dir_path = Path(dir_path or settings.DATA_DIR)


    if not dir_path.exists():
        return []

    docs: List[Document] = []

    for f in sorted(dir_path.rglob("*")):
        # 只处理"文件"，跳过目录本身；
        # 后缀白名单过滤保证不会去读 .DS_Store / .gitkeep 这类文件
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                # extend（而不是 append）—— load_document 返回的是列表，
                # 我们要把里面的 Document 逐个摊平进 docs
                docs.extend(load_document(str(f)))
            except Exception as e:
                # 兜底：任何解析异常都只记录、不抛出。
                # 常见的失败场景：加密 PDF、损坏的 docx、CSV 编码不是 utf-8。
                print(f"[document_processor] 跳过文件{f}:{e}")

    return docs


def clean_text(text: str) -> str:
    """轻度文本清洗：去掉每行首尾空白 + 删掉空行。

    """
    # splitlines() 会正确处理 \r\n / \n / \r 三种换行符（Windows 文件很常见）
    lines = [line.strip() for line in text.splitlines()]
    # `if line` 过滤掉清洗后变成空字符串的行
    return "\n".join(line for line in lines if line)


def split_documents(docs: List[Document]) -> List[Document]:
    """把长文档递归切分成检索粒度的小块（chunk）。

    参数：
        docs: load_document / load_directory 产出的 Document 列表。

    返回：
        切分并清洗后的 chunk 列表，可直接交给 vector_store.add_documents。

    切分策略（RecursiveCharacterTextSplitter）：
        它按 ["\n\n", "\n", "。", "；", " ", ""] 这个**从粗到细**的分隔符列表递归尝试：
        先用空行切（段落边界），段落还是太长就用换行切（句子边界），
        再长就用句号、分号切……直到每块都不超过 chunk_size。
        这样能最大程度保证"在语义边界断开"，而不是拦腰截断一句话
        （llm-universe C3 的切片最佳实践，也是 rag-from-scratch 反复强调的点）。

    """
    if not docs:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.CHUNK_SIZE,
        chunk_overlap=settings.CHUNK_OVERLAP,
        # add_start_index=True 会在 metadata 里写入 "start_index"，
        # 即该块在原文中的起始字符位置。
        # 用途：升级后 BM25 语料副本靠 (source + 内容hash) 做去重键，
        #       加上 start_index 能更精确定位是原文的哪一段，排查"检索到奇怪内容"时很有用。
        add_start_index=True,
    )

    chunks = splitter.split_documents(docs)

    # 清洗每个 chunk 的文本（去掉多余空行）。
    # 注意：split_documents 返回的 Document 是**新建对象**，
    # 原地改它们不会影响传入的 docs，可以放心修改。
    for c in chunks:
        c.page_content = clean_text(c.page_content)

    # 过滤掉清洗后变空的块。
    # 场景：PDF 里的纯图片页、只有页眉页码的页面，切完就是空的。
    # 空块入库后会成为"永远匹配不上任何问题的噪声向量"，必须剔除。
    return [c for c in chunks if c.page_content.strip()]


# LangChain 的多数文档 Loader 仍是同步接口；这些异步门面把磁盘读取和
# PDF/docx 解析移到工作线程，避免阻塞 FastAPI 的事件循环。
async def aload_document(file_path: str) -> List[Document]:
    return await asyncio.to_thread(load_document, file_path)


async def aload_directory(dir_path: str = None) -> List[Document]:
    return await asyncio.to_thread(load_directory, dir_path)


async def asplit_documents(docs: List[Document]) -> List[Document]:
    return await asyncio.to_thread(split_documents, docs)
