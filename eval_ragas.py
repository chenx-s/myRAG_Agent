"""RAGAs 评估脚本：用 LLM 当裁判，量化 RAG 系统的检索与生成质量。

用法
----
1) 准备评估集（JSON 数组），放到项目根目录的 eval_set.json：

    [
      {"question": "LangGraph 是什么？", "reference": "（可选）标准答案"},
      {"question": "混合检索为什么用 RRF 融合？"}
    ]

   reference 可以不写，但写了才能算 Context Recall / Precision / 事实正确性。
   没有标准答案时，可以用 RAGAs 的 testset generator 自动造题（见文末注释）。

2) 跑评估：

    python eval_ragas.py                          # 用默认 eval_set.json
    python eval_ragas.py --dataset my_set.json --out report.csv
    python eval_ragas.py --limit 5                # 先跑 5 条试水，省 token

四个指标分别在量什么
--------------------
    faithfulness                   生成答案里有多少话是检索内容支持的（幻觉率反面）
    answer_relevancy               答案有没有正面回答用户的问题（跑题检测）
    llm_context_precision_with_reference  检索到的块里，有多少是真正有用的（排序质量）
    llm_context_recall             该找到的证据，找到了多少（召回完整性）

    经验上：
    - context_recall 低 -> 检索没找全，去调 chunk 大小 / 提高召回数 / 开多查询
    - context_precision 低 -> 捞了一堆没用的，去调 Reranker 的 top_n
    - faithfulness 低 -> 模型在编，去收紧 GENERATE_PROMPT 或加强忠实性检查
    - answer_relevancy 低 -> 提示词跑偏，去改生成模板

参考：https://docs.ragas.io/en/stable/getstarted/rag_eval/
"""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from app.config import settings
from app.rag_chain import RAGChain
from app.vector_store import get_embeddings, stats as vector_stats

DEFAULT_DATASET = "eval_set.json"
DEFAULT_OUTPUT = "eval_report.csv"


# ================================================================ 兼容性补丁
# 【踩坑记录】ragas 0.4.x 在 import 阶段会执行：
#     from langchain_community.chat_models.vertexai import ChatVertexAI
# 但 langchain-community 0.4.x 已经把这个模块删掉了（Vertex 集成迁到了独立的
# langchain-google-vertexai 包）。于是 `import ragas` 直接抛：
#     ModuleNotFoundError: No module named 'langchain_community.chat_models.vertexai'
# 注意这是 import 阶段就炸，你连第一行评估代码都执行不到。
#
# 好消息是 ChatVertexAI 在 ragas 里只有一个用处：
#     MULTIPLE_COMPLETION_SUPPORTED = [OpenAI, ChatOpenAI, ..., ChatVertexAI, VertexAI]
#     def is_multiple_completion_supported(llm): 判断 isinstance(llm, 上面这些类)
# 纯粹是个"这个模型支不支持一次返回多个候选"的类型清单，我们用的是 GLM，
# 永远走不到这个分支。所以塞一个空壳类进去，行为完全不受影响。
#
# 这个补丁不修改 site-packages 里的任何文件，只在当前进程内生效。
def _patch_missing_vertexai() -> None:
    """给 langchain-community 0.4.x 补上被删除的 vertexai 聊天模型模块。"""
    module_name = "langchain_community.chat_models.vertexai"

    if module_name in sys.modules:
        return

    try:
        importlib.import_module(module_name)
        return  # 环境里本来就有（langchain-community <= 0.3.x），什么都不用做
    except Exception:
        pass

    try:
        chat_models = importlib.import_module("langchain_community.chat_models")
    except Exception:
        return  # 连父包都没有，那就不是这个补丁能解决的问题了

    stub = types.ModuleType(module_name)

    class ChatVertexAI:  # noqa: N801  —— 占位类，永远不会被实例化
        """Vertex AI 聊天模型的占位符，仅为满足 ragas 的 isinstance 类型清单。"""

        pass

    stub.ChatVertexAI = ChatVertexAI
    sys.modules[module_name] = stub
    setattr(chat_models, "vertexai", stub)


_patch_missing_vertexai()


# ================================================================ 评估集


def load_eval_set(path: str) -> List[Dict[str, str]]:
    """读取评估集。支持两种格式：

    A. [{"question": "...", "reference": "..."}, ...]
    B. ["问题1", "问题2", ...]        # 纯问题，无标准答案
    """
    file_path = Path(path)
    if not file_path.exists():
        raise FileNotFoundError(
            f"找不到评估集 {path}。请新建该文件，格式见本脚本开头的说明。"
        )

    raw = json.loads(file_path.read_text(encoding="utf-8"))
    if not isinstance(raw, list):
        raise ValueError("评估集必须是 JSON 数组")

    items: List[Dict[str, str]] = []
    for entry in raw:
        if isinstance(entry, str):
            items.append({"question": entry.strip()})
        elif isinstance(entry, dict) and entry.get("question"):
            items.append({
                "question": str(entry["question"]).strip(),
                "reference": str(entry.get("reference") or "").strip(),
            })

    if not items:
        raise ValueError("评估集里没有有效问题")

    return items


def collect_samples(
    items: List[Dict[str, str]],
    rag: RAGChain,
    verbose: bool = True,
) -> Tuple[List[Dict[str, Any]], bool]:
    """跑一遍 RAG，收集 RAGAs 需要的四元组。

    返回 (样本列表, 是否含标准答案)

    RAGAs 的字段名是固定的，不要改：
        user_input          用户问题
        retrieved_contexts  检索到的上下文（列表，这里是精排后的最终上下文）
        response            RAG 生成的答案
        reference           标准答案（可选）
    """
    samples: List[Dict[str, Any]] = []
    has_reference = any(item.get("reference") for item in items)

    for index, item in enumerate(items, start=1):
        question = item["question"]
        if verbose:
            print(f"[{index}/{len(items)}] 提问：{question}")

        # include_documents=True 才会带出完整上下文，否则只有 200 字摘要
        result = rag.answer(question, include_documents=True)

        contexts = result.get("documents") or []
        if verbose:
            print(
                f"    候选 {result.get('candidates', 0)} 条 -> 最终 {len(contexts)} 条"
                f" | 精排={'开' if result.get('reranked') else '关'}"
                f" | 变换={result.get('transform_mode')}"
            )

        sample: Dict[str, Any] = {
            "user_input": question,
            "retrieved_contexts": contexts,
            "response": result["answer"],
        }
        if item.get("reference"):
            sample["reference"] = item["reference"]
        samples.append(sample)

    return samples, has_reference


# ================================================================ 指标与模型包装


def _import_metric(*candidates: Tuple[str, str]):
    """按候选顺序尝试导入指标类，兼容 RAGAs 不同版本的类名变更。

    返回第一个导入成功的类；全都失败返回 None。
    """
    import importlib

    for module_name, class_name in candidates:
        try:
            module = importlib.import_module(module_name)
            metric_cls = getattr(module, class_name, None)
            if metric_cls is not None:
                return metric_cls
        except Exception:
            continue
    return None


def build_metrics(has_reference: bool, verbose: bool = True) -> List[Any]:
    """挑选评估指标。没有标准答案时不启用任何 reference 类指标。"""
    metrics: List[Any] = []

    faithfulness = _import_metric(("ragas.metrics", "Faithfulness"))
    if faithfulness is not None:
        metrics.append(faithfulness())

    # 0.2+ 叫 ResponseRelevancy，更早的版本叫 AnswerRelevancy
    relevancy = _import_metric(
        ("ragas.metrics", "ResponseRelevancy"),
        ("ragas.metrics", "AnswerRelevancy"),
        ("ragas.metrics", "answer_relevancy"),
    )
    if relevancy is not None:
        metrics.append(relevancy())

    if has_reference:
        precision = _import_metric(
            ("ragas.metrics", "LLMContextPrecisionWithReference"),
            ("ragas.metrics", "ContextPrecision"),
            ("ragas.metrics", "context_precision"),
        )
        if precision is not None:
            metrics.append(precision())

        recall = _import_metric(
            ("ragas.metrics", "LLMContextRecall"),
            ("ragas.metrics", "ContextRecall"),
            ("ragas.metrics", "context_recall"),
        )
        if recall is not None:
            metrics.append(recall())

        correctness = _import_metric(
            ("ragas.metrics", "FactualCorrectness"),
            ("ragas.metrics", "factual_correctness"),
        )
        if correctness is not None:
            metrics.append(correctness())

    if verbose:
        names = [getattr(m, "name", type(m).__name__) for m in metrics]
        print(f"评估指标：{names}")

    return metrics


def build_ragas_models():
    """构建 RAGAs 的裁判模型。

    注意两点：
    1. 裁判 LLM 用的是你项目里的 GLM，和被测系统同源。严格来说"自己评自己"
       有偏向性，生产环境建议换成不同厂商的模型（比如裁判用 GPT-4o / Claude）。
       但在学习阶段这样最省钱、也最容易跑通。
    2. 裁判 LLM 必须把 temperature 设成 0，否则同一份数据两次评估结果不一致。
    """
    from ragas.llms import LangchainLLMWrapper

    judge_llm = ChatOpenAI(
        model=settings.LLM_MODEL_NAME,
        api_key=settings.OPENAI_API_KEY,
        base_url=settings.OPENAI_BASE_URL,
        temperature=0.0,
        extra_body={"thinking": {"type": "disabled"}},
    )
    evaluator_llm = LangchainLLMWrapper(judge_llm)

    evaluator_embeddings = None
    try:
        from ragas.embeddings import LangchainEmbeddingsWrapper

        # answer_relevancy 需要用 Embedding 算"答案和问题的语义相似度"，
        # 所以必须把 embedding 也交给 RAGAs
        evaluator_embeddings = LangchainEmbeddingsWrapper(get_embeddings())
    except Exception as error:
        print(f"[warn] 构建 RAGAs Embedding 包装失败，answer_relevancy 可能不可用：{error}")

    return evaluator_llm, evaluator_embeddings


# ================================================================ 主流程


def build_dataset(samples: List[Dict[str, Any]]):
    """把样本列表转成 RAGAs 的 EvaluationDataset。"""
    from ragas import EvaluationDataset

    return EvaluationDataset.from_list(samples)


def run_evaluation(
    dataset_path: str = DEFAULT_DATASET,
    output_path: str = DEFAULT_OUTPUT,
    limit: Optional[int] = None,
) -> Any:
    """端到端：跑 RAG -> 组装数据集 -> RAGAs 评估 -> 落盘报告。"""
    # ---------------- 环境自检 ----------------
    print("=" * 68)
    print("RAGAs 评估")
    print("=" * 68)

    info = vector_stats()
    print(f"向量库：{info['milvus_uri']} / collection={info['collection']} / {info['index_count']} 条")
    print(
        f"检索：hybrid={info['hybrid_enabled']} "
        f"权重={info['hybrid_weights']} BM25语料={info['bm25_corpus_chunks']} 条"
    )
    print(f"精排：enabled={settings.RERANK_ENABLED} ready={settings.rerank_ready} "
          f"top_n={settings.RERANK_TOP_N}")
    print(f"查询变换：{settings.QUERY_TRANSFORM}")

    if info["index_count"] == 0:
        print("\n[中止] 向量库为空，请先上传文档再评估。")
        return None

    if not settings.OPENAI_API_KEY:
        print("\n[中止] 未配置 LLM_API_KEY，无法调用裁判模型。")
        return None

    # ---------------- 1. 采集样本 ----------------
    items = load_eval_set(dataset_path)
    if limit:
        items = items[:limit]
    print(f"\n评估集：{dataset_path}，共 {len(items)} 条\n")

    rag = RAGChain()
    samples, has_reference = collect_samples(items, rag)

    if not has_reference:
        print(
            "\n[warn] 评估集里没有 reference 字段，"
            "将只评估 faithfulness / answer_relevancy 两个指标。"
        )

    # ---------------- 2. 构建评估对象 ----------------
    from ragas import evaluate

    dataset = build_dataset(samples)
    evaluator_llm, evaluator_embeddings = build_ragas_models()
    metrics = build_metrics(has_reference)

    if not metrics:
        print("\n[中止] 没有可用指标，请检查 ragas 安装。")
        return None

    # GLM 比 GPT 慢，且 RAGAs 会并发调用，超时要放宽，否则大量样本被判失败
    try:
        from ragas import RunConfig

        run_config = RunConfig(
            timeout=180,
            max_workers=4,
            max_retries=3,
        )
    except Exception:
        run_config = None

    # ---------------- 3. 跑评估 ----------------
    print("\n开始评估（LLM 裁判调用较多，请耐心等待）...\n")

    kwargs: Dict[str, Any] = {
        "dataset": dataset,
        "metrics": metrics,
        "llm": evaluator_llm,
    }
    if evaluator_embeddings is not None:
        kwargs["embeddings"] = evaluator_embeddings
    if run_config is not None:
        kwargs["run_config"] = run_config

    result = evaluate(**kwargs)

    # ---------------- 4. 报告 ----------------
    print("\n" + "=" * 68)
    print("总分")
    print("=" * 68)
    try:
        scores = result.to_pandas().mean(numeric_only=True)
        for name, value in scores.items():
            if name in ("user_input", "response", "reference", "retrieved_contexts"):
                continue
            print(f"  {name:<36} {value:.4f}")
    except Exception as error:
        print(f"[warn] 汇总失败：{error}")

    try:
        frame = result.to_pandas()
        frame.to_csv(output_path, index=False, encoding="utf-8-sig")
        print(f"\n逐条明细已写入：{output_path}")
    except Exception as error:
        print(f"[warn] 报告落盘失败：{error}")

    return result


# ================================================================ CLI


def main() -> int:
    """命令行入口：读评估集 → 跑 RAGAs 三个指标 → 输出 CSV 报告。

    用法：
        python eval_ragas.py                                  # 用默认评估集
        python eval_ragas.py --dataset eval_set.json          # 指定评估集
        python eval_ragas.py --limit 5                        # 只跑前 5 条（省 token）
        python eval_ragas.py --out report.csv                 # 指定输出路径

    返回 0 表示成功，非 0 表示失败（方便脚本/CI 判断）。
    """
    # 让 RAGAs 内部的 ChatOpenAI 能读到智谱的地址。
    # RAGAs 有时会自己 new 一个 ChatOpenAI，显式设环境变量最保险。
    import os

    os.environ.setdefault("OPENAI_API_KEY", settings.OPENAI_API_KEY or "")
    os.environ.setdefault("OPENAI_BASE_URL", settings.OPENAI_BASE_URL)
    if settings.COHERE_API_KEY:
        os.environ.setdefault("COHERE_API_KEY", settings.COHERE_API_KEY)

    parser = argparse.ArgumentParser(description="RAGAs 评估 RAG 系统")
    parser.add_argument("--dataset", default=r"eval_set.sample.json", help="评估集 JSON 路径")
    parser.add_argument("--out", default=r"eval_rag", help="报告 CSV 输出路径")
    parser.add_argument("--limit", type=int, default=None, help="只评估前 N 条")
    args = parser.parse_args()

    try:
        run_evaluation(args.dataset, args.out, args.limit)
    except Exception as error:
        print(f"\n[失败] {type(error).__name__}: {error}")
        return 1
    return 0


# ChatOpenAI 在 build_ragas_models 里用到，放在文件尾部导入不影响可读性
from langchain_openai import ChatOpenAI  # noqa: E402


if __name__ == "__main__":
    sys.exit(main())


# ---------------------------------------------------------------------------
# 附：没有标准答案怎么办？用 RAGAs 自动造题
#
#     from ragas.testset import TestsetGenerator
#     from langchain_community.document_loaders import DirectoryLoader
#     from ragas.llms import LangchainLLMWrapper
#     from ragas.embeddings import LangchainEmbeddingsWrapper
#
#     loader = DirectoryLoader("data", glob="**/*.md")
#     documents = loader.load()
#
#     generator = TestsetGenerator(
#         llm=LangchainLLMWrapper(judge_llm),
#         embedding_model=LangchainEmbeddingsWrapper(get_embeddings()),
#     )
#     testset = generator.generate_with_langchain_docs(documents, testset_size=20)
#     testset.to_pandas().to_json("eval_set.json", orient="records", force_ascii=False)
#
# 生成的 TestsetSample 里带 user_input / reference_contexts / reference，
# 把 user_input 和 reference 抽出来就是本脚本要的评估集格式。
# ---------------------------------------------------------------------------
