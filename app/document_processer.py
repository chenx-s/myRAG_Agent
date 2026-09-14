# 文档加载与文档分块
# ============================================================================
# 这是整条 RAG 链路的「入口」—— 负责把磁盘上的原始文件变成一串 Document 对象。
#
# 它处在流水线的第一环：
#   磁盘文件 --> [本文件] load_document / load_directory
#            --> [本文件] split_documents（切块）
#            --> vector_store.add_documents（BGE 向量化 + 写入 Milvus）
#            --> rag_chain（检索 + 生成）
#
# 【本次升级说明】
# 本文件逻辑与你原来的版本**完全一致**，没有做任何功能改动。
# 之所以保留原样，是因为它在升级后的链路里依然正确：
#   - chunk_size / chunk_overlap 依然从 settings 读取，改 .env 就能生效；
#   - metadata 里的 source / filename / page / start_index 恰好是
#     升级后「BM25 语料副本」和「引用溯源」所需要的字段（见 vector_store._doc_key）。
# 唯一的改动是：补全了注释。逻辑一行未动。
#
# 参考来源：
# - llm-universe C3（搭建知识库）：读取、清洗与切片的标准做法
# - llm-cookbook《LangChain Chat With Your Data》：DocumentLoader + TextSplitter 标准用法
# - rag-from-scratch Part 1（Indexing）：分块大小/重叠对检索质量的影响
# ============================================================================

from langchain_community.document_loaders import PyPDFLoader , TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from typing import List
import os
from app.config import settings
from pathlib import Path

# 支持的文件后缀集合。
# 用途有二：
#   1) load_document 里做后缀分派（决定用哪个 Loader）；
#   2) load_directory 里做白名单过滤（非白名单文件直接跳过，不报错）。
# 想新增格式（比如 .pptx），除了往这里加后缀，还要在下面对应位置加一个 elif 分支。
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

    # --------------------------------------------------------------
    # 按后缀分派 Loader。
    # 注意这里用的是**函数内局部 import**，而不是文件顶部统一 import。
    # 原因：这些 Loader 各自依赖不同的第三方库（pypdf / docx2txt / beautifulsoup4 …），
    # 局部导入意味着"只有真的处理到这种格式时才需要装那个库"，
    # 否则你只想跑个 txt，却因为没装 docx2txt 而整个模块导入失败。
    # --------------------------------------------------------------
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
        # 【踩坑提示】CSV 的 metadata 里会带 source 等字段，且行与行之间
        # metadata 结构可能不完全一致；升级后用 enable_dynamic_field=True
        # 把全部 metadata 塞进动态字段，正是为了兼容这种异构情况
        # （详见 vector_store.get_vector_store 和本文档末尾的 DataNotMatchException 说明）。
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
    # `dir_path or settings.DATA_DIR` —— 注意这里不能用 `or` 之外的方式，
    # 因为 Path("") 在 Python 里会被当成当前目录，而空字符串是 falsy 的，
    # 所以传 "" 也会正确回退到配置里的 DATA_DIR。
    dir_path = Path(dir_path or settings.DATA_DIR)

    # 目录不存在时返回空列表。不抛异常的原因：
    # 服务启动时会调用这个函数做预热，此时 ./data 目录可能还没建，
    # 抛异常会导致服务起不来 —— 空目录是完全合法的初始状态。
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

    为什么不做得更"干净"（比如去掉页眉页脚、合并断行）？
        因为清洗越激进，误删正文内容的风险越大。
        这里只做**零风险**的两件事，保证不会吃掉任何有效字符。

    对检索的实际好处：
        切块时，一个块里如果有大量连续空行，会挤占 chunk_size 的字符额度，
        导致有效信息变少、向量表示变"稀"。去掉空行能让同样的
        chunk_size 装下更多有用内容。
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

    为什么必须切块？
        - LLM 上下文窗口有限，不可能把整本书塞进去；
        - 检索时"块"是匹配单位，块太大 → 夹带无关内容，稀释相关性；
                       块太小 → 语义不完整，答非所问。
        所以 chunk_size / chunk_overlap 是 RAG 里最值得调的参数之一。

    切分策略（RecursiveCharacterTextSplitter）：
        它按 ["\n\n", "\n", "。", "；", " ", ""] 这个**从粗到细**的分隔符列表递归尝试：
        先用空行切（段落边界），段落还是太长就用换行切（句子边界），
        再长就用句号、分号切……直到每块都不超过 chunk_size。
        这样能最大程度保证"在语义边界断开"，而不是拦腰截断一句话
        （llm-universe C3 的切片最佳实践，也是 rag-from-scratch 反复强调的点）。

    两个关键配置（都在 .env 里，改完重启即可生效）：
        CHUNK_SIZE    ：每块最大字符数，默认 500；
        CHUNK_OVERLAP ：相邻块的重叠字符数，默认 50。
                        重叠是为了防止"答案刚好被切在两块的交界处"，
                        导致两块都只包含半句话、谁都匹配不上。
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
