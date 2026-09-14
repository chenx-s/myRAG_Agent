"""合并后的整体验收脚本（可随时重跑，不产生副作用）。

检查项：
  1. 四个核心模块能否正常导入
  2. config 的所有配置项能否正确读取
  3. LangGraph 图能否编译，节点/边是否齐全
  4. vector_store 的关键函数是否都在，且是升级后的实现
  5. document_processer 的注释版逻辑是否与原版等价
"""
import ast
import sys
import io
import tokenize

PASS, FAIL = "  ✅", "  ❌"
errors = []


def check(label, cond, detail=""):
    print(f"{PASS if cond else FAIL} {label}{'  ' + detail if detail else ''}")
    if not cond:
        errors.append(label)


print("=" * 72)
print(" RAG 系统合并后验收")
print("=" * 72)

# ---------------------------------------------------------------- 1. 导入
print("\n【1】模块导入")
try:
    from app.config import settings
    check("app.config 导入成功", True)
except Exception as e:
    check("app.config 导入成功", False, repr(e))
    sys.exit(1)

try:
    from app import vector_store
    check("app.vector_store 导入成功", True)
except Exception as e:
    check("app.vector_store 导入成功", False, repr(e))

try:
    from app import rag_chain
    check("app.rag_chain 导入成功", True)
except Exception as e:
    check("app.rag_chain 导入成功", False, repr(e))

try:
    from app import document_processer
    check("app.document_processer 导入成功", True)
except Exception as e:
    check("app.document_processer 导入成功", False, repr(e))

# ---------------------------------------------------------------- 2. 配置
print("\n【2】配置项读取")
cfg_checks = [
    ("MILVUS_URI", settings.MILVUS_URI),
    ("INDEX_NAME", settings.INDEX_NAME),
    ("HYBRID_ENABLED", settings.HYBRID_ENABLED),
    ("DENSE_WEIGHT", settings.DENSE_WEIGHT),
    ("SPARSE_WEIGHT", settings.SPARSE_WEIGHT),
    ("DENSE_TOP_K", settings.DENSE_TOP_K),
    ("SPARSE_TOP_K", settings.SPARSE_TOP_K),
    ("FUSION_TOP_K", settings.FUSION_TOP_K),
    ("QUERY_TRANSFORM", settings.QUERY_TRANSFORM),
    ("NUM_QUERIES", settings.NUM_QUERIES),
    ("RERANK_ENABLED", settings.RERANK_ENABLED),
    ("COHERE_RERANK_MODEL", settings.COHERE_RERANK_MODEL),
    ("RERANK_TOP_N", settings.RERANK_TOP_N),
    ("GROUNDEDNESS_ACTION", settings.GROUNDEDNESS_ACTION),
    ("MAX_GROUNDEDNESS_RETRIES", settings.MAX_GROUNDEDNESS_RETRIES),
    ("RECURSION_LIMIT", settings.RECURSION_LIMIT),
    ("MAX_RETRIES", settings.MAX_RETRIES),
    ("TOP_K", settings.TOP_K),
    ("CHUNK_SIZE", settings.CHUNK_SIZE),
    ("CHUNK_OVERLAP", settings.CHUNK_OVERLAP),
    ("EMBEDDING_MODEL", settings.EMBEDDING_MODEL),
]
for name, val in cfg_checks:
    check(f"{name} = {val}", val is not None)

print(f"\n  · rerank_ready          = {settings.rerank_ready}")
print(f"  · COHERE_API_KEY 已配置 = {bool(settings.COHERE_API_KEY)}")
print(f"  · BM25 语料副本路径     = {settings.bm25_corpus_path}")

# 递归上限必须足够大，否则会抛 GraphRecursionError
need = 2 * (settings.MAX_RETRIES + 1) + 3
check(
    f"RECURSION_LIMIT({settings.RECURSION_LIMIT}) > 2*(MAX_RETRIES+1)+3 = {need}",
    settings.RECURSION_LIMIT > need,
)

# ---------------------------------------------------------------- 3. 向量库
print("\n【3】vector_store 升级后的能力")
vs_funcs = [
    ("rrf_fuse", "RRF 融合"),
    ("multi_query_search", "多查询融合检索（RAG-Fusion）"),
    ("_tokenize", "BM25 分词器"),
    ("_doc_key", "去重键（source + 内容hash）"),
    ("_local_db_locked", "Milvus 文件锁探测"),
    ("_explain_milvus_open_failure", "锁冲突中文指引"),
    ("count", "文档计数（已修空库报错）"),
    ("add_documents", "写入（已包装 DataNotMatchException）"),
    ("warmup", "预热"),
]
for fn, desc in vs_funcs:
    check(f"{fn:<32} {desc}", hasattr(vector_store, fn))

check("RRF_C 常量为 60", getattr(vector_store, "RRF_C", None) == 60)

# 去重键必须不依赖 pk —— 只检查函数体的可执行代码，
# 排除 docstring 里的"踩坑记录"文字（那里提到 pk 是在解释为什么不能用）
import inspect


def _body_code_without_docstring(fn) -> str:
    tree = ast.parse(inspect.getsource(fn))
    fnode = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef))
    # 剥掉文档字符串
    if (fnode.body and isinstance(fnode.body[0], ast.Expr)
            and isinstance(fnode.body[0].value, ast.Constant)
            and isinstance(fnode.body[0].value.value, str)):
        fnode.body = fnode.body[1:]
    # 再剥掉注释以外的所有字符串常量内容（避免字符串里出现 pk）
    for n in ast.walk(fnode):
        if isinstance(n, ast.Constant) and isinstance(n.value, str):
            n.value = ""
    return ast.dump(fnode, include_attributes=False)


body_src = _body_code_without_docstring(vector_store._doc_key)
check("_doc_key 的函数体不依赖 Milvus 的 pk 字段", "pk" not in body_src.lower())
check("_doc_key 使用 source 作为键的一部分",
      "'source'" in inspect.getsource(vector_store._doc_key))
check("_doc_key 使用内容哈希（md5）",
      "md5" in inspect.getsource(vector_store._doc_key))

ver = settings.MILVUS_URI
print(f"\n  · 向量库模式: {'Milvus Lite 本地文件' if not ver.startswith('http') else 'Milvus 服务'} -> {ver}")

# ---------------------------------------------------------------- 4. 图结构
print("\n【4】LangGraph 图结构")
try:
    graph = rag_chain.build_graph()
    nodes = set(graph.get_graph().nodes.keys())
    edges = graph.get_graph().edges

    required_nodes = {
        "transform_query", "retrieve", "rerank_documents",
        "grade_document", "rewrite_query", "generate", "check_groundedness",
    }
    for n in sorted(required_nodes):
        check(f"节点 {n}", n in nodes)

    check("条件边 decide_after_grading", hasattr(rag_chain, "decide_after_grading"))
    check("条件边 decide_after_rewrite", hasattr(rag_chain, "decide_after_rewrite"))
    check("条件边 decide_after_groundedness", hasattr(rag_chain, "decide_after_groundedness"))

    # 新增的反思闭环
    has_regen = any(
        e.source == "check_groundedness" and e.target == "generate" for e in edges
    )
    check("Self-RAG 闭环: check_groundedness -> generate", has_regen)

    has_rerank = any(
        e.source == "retrieve" and e.target == "rerank_documents" for e in edges
    )
    check("混合检索 -> 精排链路", has_rerank)

    print(f"\n  · 节点数 {len([n for n in nodes if not n.startswith('__')])}，边数 {len([e for e in edges if not e.conditional])} (不含条件边分支)")
except Exception as e:
    check("图编译成功", False, repr(e))

# 提示词齐备
print("\n【5】提示词模板")
for pname, desc in [
    ("DOC_GRADE_PROMPT", "文档相关度评分"),
    ("REWRITE_PROMPT", "查询改写"),
    ("MULTI_QUERY_PROMPT", "多查询裂变"),
    ("HYDE_PROMPT", "HyDE 假想答案"),
    ("GENERATE_PROMPT", "答案生成"),
    ("GENERATE_STRICT_PROMPT", "严格模式重生成（新增）"),
    ("GROUNDEDNESS_PROMPT", "忠实性检查（含数字换算对照表）"),
]:
    check(f"{pname:<24} {desc}", hasattr(rag_chain, pname))

# 忠实性提示词里必须有换算规则，否则会重现误判 bug
g_text = str(rag_chain.GROUNDEDNESS_PROMPT.messages[0].prompt)
check("GROUNDEDNESS_PROMPT 含 billion 换算规则", "billion" in g_text.lower())
g_strict = str(rag_chain.GENERATE_STRICT_PROMPT.messages[0].prompt)
check("GENERATE_STRICT_PROMPT 非空", len(g_strict) > 50)

# ---------------------------------------------------------------- 6. 文档处理
print("\n【6】document_processer")
for fn, desc in [
    ("load_document", "单文件加载（按后缀调度 Loader）"),
    ("load_directory", "目录递归加载"),
    ("clean_text", "文本清洗"),
    ("split_documents", "递归字符分块"),
]:
    check(f"{fn:<20} {desc}", hasattr(document_processer, fn))

check("SUPPORTED_EXTENSIONS 含 7 种格式",
      len(document_processer.SUPPORTED_EXTENSIONS) == 7,
      str(sorted(document_processer.SUPPORTED_EXTENSIONS)))

# 与原版逐节点对比 AST
try:
    import os
    old_p = "_backup_before_merge_20260912/app/document_processer.py"
    if os.path.exists(old_p):
        def _strip_doc(node):
            for nn in ast.walk(node):
                if isinstance(nn, (ast.Module, ast.ClassDef, ast.FunctionDef)):
                    if (nn.body and isinstance(nn.body[0], ast.Expr)
                            and isinstance(nn.body[0].value, ast.Constant)
                            and isinstance(nn.body[0].value.value, str)):
                        nn.body = nn.body[1:] or [ast.Pass()]
            return node

        old_tree = _strip_doc(ast.parse(open(old_p, encoding="utf-8").read()))
        new_tree = _strip_doc(ast.parse(open("app/document_processer.py", encoding="utf-8").read()))
        same = ast.dump(old_tree, include_attributes=False) == ast.dump(new_tree, include_attributes=False)
        check("与备份原版 AST 逐节点等价（仅加注释）", same)
    else:
        print("  · 未找到备份，跳过对比")
except Exception as e:
    check("AST 对比", False, repr(e))

# ---------------------------------------------------------------- 7. 注释覆盖率
print("\n【7】注释覆盖率")
files = [
    "app/config.py", "app/vector_store.py", "app/rag_chain.py",
    "app/document_processer.py", "main.py", "eval_ragas.py",
]
import os
for f in files:
    if not os.path.exists(f):
        continue
    src = open(f, encoding="utf-8").read()
    lines = src.splitlines()
    total = len(lines)
    empty = sum(1 for l in lines if not l.strip())
    cl, sl = set(), set()
    for tok in tokenize.generate_tokens(io.StringIO(src).readline):
        if tok.type == tokenize.COMMENT:
            cl.update(range(tok.start[0], tok.end[0] + 1))
        elif tok.type == tokenize.STRING:
            sl.update(range(tok.start[0], tok.end[0] + 1))
    sl -= cl
    rate = (len(cl) + len(sl)) * 100 / max(1, total - empty)
    print(f"  · {f:<32} {total:>4} 行，注释率 {rate:5.1f}%")
    check(f"  {f} 注释覆盖 >= 45%", rate >= 45, f"(实际 {rate:.1f}%)")

# ---------------------------------------------------------------- 汇总
print("\n" + "=" * 72)
if errors:
    print(f" ❌ 未通过 {len(errors)} 项：")
    for e in errors:
        print(f"    - {e}")
    sys.exit(1)
else:
    print(" ✅ 全部通过 —— 项目处于可运行状态")
print("=" * 72)
