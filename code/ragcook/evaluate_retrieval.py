"""Small retrieval evaluation for RAG project.

Computes Hit@K and MRR for retrieval strategies and supports
BM25-before/after Chinese segmentation comparison.
Also reports per-intent bucket metrics and Hybrid Δ by intent.
"""

from __future__ import annotations

import argparse
import json
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Callable, Dict, List

from config import DEFAULT_CONFIG
from rag_modules import DataPreparationModule, IndexConstructionModule, RetrievalOptimizationModule
from langchain_community.retrievers import BM25Retriever

BASE_DIR = Path(__file__).resolve().parent

try:
    import jieba  # type: ignore
    HAS_JIEBA = True
except Exception:
    HAS_JIEBA = False


def _resolve_path(path_str: str) -> Path:
    p = Path(path_str)
    if p.is_absolute():
        return p
    return (BASE_DIR / p).resolve()


def load_eval_set(path: Path) -> List[Dict]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        obj = json.loads(line)
        if "query" not in obj or "targets" not in obj:
            raise ValueError(f"Invalid row in eval set: {obj}")
        if "intent" not in obj:
            obj["intent"] = "unknown"
        rows.append(obj)
    if not rows:
        raise ValueError("Eval set is empty")
    return rows


def rank_of_first_hit(dish_names: List[str], targets: List[str]) -> int:
    target_set = set(targets)
    for idx, name in enumerate(dish_names, start=1):
        if name in target_set:
            return idx
    return 0


def compute_metrics(ranks: List[int], k: int) -> Dict[str, float]:
    if not ranks:
        return {
            "count": 0,
            "hit@1": 0.0,
            f"hit@{k}": 0.0,
            f"mrr@{k}": 0.0,
        }
    n = len(ranks)
    hit1 = sum(1 for r in ranks if r == 1) / n
    hitk = sum(1 for r in ranks if 1 <= r <= k) / n
    mrr = sum((1 / r) if r > 0 else 0 for r in ranks) / n
    return {
        "count": n,
        "hit@1": hit1,
        f"hit@{k}": hitk,
        f"mrr@{k}": mrr,
    }


def tokenize_zh(text: str) -> List[str]:
    text = text.strip()
    if not text:
        return []

    if HAS_JIEBA:
        return [tok for tok in jieba.lcut(text) if tok.strip()]

    tokens: List[str] = []
    for part in re.findall(r"[A-Za-z0-9_]+|[\u4e00-\u9fff]", text):
        tokens.append(part)
    return tokens


def build_bm25_retriever(chunks, k: int, preprocess_func: Callable[[str], List[str]] | None = None):
    if preprocess_func is None:
        return BM25Retriever.from_documents(chunks, k=k)
    return BM25Retriever.from_documents(chunks, k=k, preprocess_func=preprocess_func)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval hit rate")
    parser.add_argument("--eval-set", type=str, default="eval/retrieval_eval_small.jsonl", help="Path to eval JSONL set")
    parser.add_argument("--k", type=int, default=3, help="Top-K for ranking metrics")
    parser.add_argument("--data-path", type=str, default=DEFAULT_CONFIG.data_path, help="Path to markdown dataset")
    parser.add_argument("--index-path", type=str, default="./eval_vector_index", help="Index path for evaluation")
    parser.add_argument("--embedding-model", type=str, default=DEFAULT_CONFIG.embedding_model, help="Embedding model name")
    parser.add_argument("--output", type=str, default="retrieval_eval_results.json", help="Output JSON path")
    parser.add_argument("--validate-only", action="store_true", help="Only validate eval set against dataset dish names")
    parser.add_argument("--bm25-only", action="store_true", help="Only evaluate BM25 before/after Chinese segmentation")
    args = parser.parse_args()

    eval_set_path = _resolve_path(args.eval_set)
    data_path = _resolve_path(args.data_path)
    index_path = _resolve_path(args.index_path)
    output_path = _resolve_path(args.output)

    eval_rows = load_eval_set(eval_set_path)

    dish_names_all = {p.stem for p in data_path.rglob("*.md")}
    missing_targets = []
    for i, row in enumerate(eval_rows, start=1):
        for target in row["targets"]:
            if target not in dish_names_all:
                missing_targets.append({"row": i, "target": target})

    if missing_targets:
        raise ValueError(f"Missing targets in dataset: {missing_targets}")

    if args.validate_only:
        intents = defaultdict(int)
        for row in eval_rows:
            intents[row.get("intent", "unknown")] += 1
        print(
            f"Eval set validated: rows={len(eval_rows)}, unique_dataset_dishes={len(dish_names_all)}, missing=0, intents={dict(intents)}"
        )
        return

    data_module = DataPreparationModule(str(data_path))
    data_module.load_documents()
    chunks = data_module.chunk_documents()

    bm25_default = build_bm25_retriever(chunks, k=args.k)
    bm25_zhseg = build_bm25_retriever(chunks, k=args.k, preprocess_func=tokenize_zh)

    retrieval_module = None
    vector_retriever = None

    if not args.bm25_only:
        index_module = IndexConstructionModule(model_name=args.embedding_model, index_save_path=str(index_path))
        vectorstore = index_module.load_index()
        if vectorstore is None:
            vectorstore = index_module.build_vector_index(chunks)
            index_module.save_index()

        retrieval_module = RetrievalOptimizationModule(vectorstore, chunks)
        vector_retriever = retrieval_module.vector_retriever

    methods: Dict[str, Callable[[str], List]] = {
        "bm25_default": lambda q: bm25_default.invoke(q)[: args.k],
        "bm25_zhseg": lambda q: bm25_zhseg.invoke(q)[: args.k],
    }

    if vector_retriever is not None and retrieval_module is not None:
        methods["vector"] = lambda q: vector_retriever.invoke(q)[: args.k]
        methods["hybrid_default"] = lambda q: retrieval_module.hybrid_search(q, top_k=args.k)

        def _hybrid_zhseg(q: str):
            vector_docs = vector_retriever.invoke(q)
            bm25_docs = bm25_zhseg.invoke(q)
            reranked = retrieval_module._rrf_rerank(vector_docs, bm25_docs)
            return reranked[: args.k]

        methods["hybrid_zhseg"] = _hybrid_zhseg

    method_ranks: Dict[str, List[int]] = {m: [] for m in methods}
    # per-intent ranks: intent -> method -> ranks
    per_intent_ranks: Dict[str, Dict[str, List[int]]] = defaultdict(lambda: {m: [] for m in methods})
    details = []

    t0 = time.perf_counter()
    for row in eval_rows:
        query = row["query"]
        targets = row["targets"]
        intent = row.get("intent", "unknown")
        per_query = {"query": query, "targets": targets, "intent": intent, "results": {}}

        for method, fn in methods.items():
            docs = fn(query)
            names = [d.metadata.get("dish_name", "未知菜品") for d in docs]
            rank = rank_of_first_hit(names, targets)
            method_ranks[method].append(rank)
            per_intent_ranks[intent][method].append(rank)
            per_query["results"][method] = {"retrieved": names, "rank": rank, "hit": rank > 0}

        details.append(per_query)

    elapsed = time.perf_counter() - t0

    summary = {method: compute_metrics(ranks, args.k) for method, ranks in method_ranks.items()}

    summary_by_intent: Dict[str, Dict[str, Dict[str, float]]] = {}
    for intent, methods_map in per_intent_ranks.items():
        summary_by_intent[intent] = {
            method: compute_metrics(ranks, args.k)
            for method, ranks in methods_map.items()
        }

    payload = {
        "config": {
            "eval_set": str(eval_set_path),
            "k": args.k,
            "data_path": str(data_path),
            "index_path": str(index_path),
            "embedding_model": args.embedding_model,
            "bm25_only": args.bm25_only,
            "zh_tokenizer": "jieba" if HAS_JIEBA else "regex_char_fallback",
        },
        "timing": {
            "eval_elapsed_s": elapsed,
            "queries": len(eval_rows),
            "avg_query_s": elapsed / len(eval_rows),
        },
        "summary": summary,
        "summary_by_intent": summary_by_intent,
        "details": details,
    }

    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("=== Retrieval Evaluation Summary ===")
    for method, metrics in summary.items():
        print(
            f"{method:>14} | hit@1={metrics['hit@1']:.3f} | hit@{args.k}={metrics[f'hit@{args.k}']:.3f} | "
            f"mrr@{args.k}={metrics[f'mrr@{args.k}']:.3f}"
        )

    if "bm25_default" in summary and "bm25_zhseg" in summary:
        d1 = summary["bm25_zhseg"]["hit@1"] - summary["bm25_default"]["hit@1"]
        dm = summary["bm25_zhseg"][f"mrr@{args.k}"] - summary["bm25_default"][f"mrr@{args.k}"]
        print(f"BM25 Δ: hit@1={d1:+.3f}, mrr@{args.k}={dm:+.3f}")

    if "hybrid_default" in summary and "hybrid_zhseg" in summary:
        d1 = summary["hybrid_zhseg"]["hit@1"] - summary["hybrid_default"]["hit@1"]
        dm = summary["hybrid_zhseg"][f"mrr@{args.k}"] - summary["hybrid_default"][f"mrr@{args.k}"]
        print(f"Hybrid Δ: hit@1={d1:+.3f}, mrr@{args.k}={dm:+.3f}")

    if "hybrid_default" in methods and "hybrid_zhseg" in methods:
        print("=== Hybrid Δ by intent ===")
        for intent, m in summary_by_intent.items():
            hd = m.get("hybrid_default")
            hz = m.get("hybrid_zhseg")
            if not hd or not hz or hd.get("count", 0) == 0:
                continue
            d1 = hz["hit@1"] - hd["hit@1"]
            dm = hz[f"mrr@{args.k}"] - hd[f"mrr@{args.k}"]
            print(f"{intent:>8} | count={int(hd['count'])} | Δhit@1={d1:+.3f} | Δmrr@{args.k}={dm:+.3f}")

    print(f"queries={len(eval_rows)}, total={elapsed:.2f}s, avg_query={payload['timing']['avg_query_s']:.4f}s")
    print(f"Saved to: {output_path}")


if __name__ == "__main__":
    main()