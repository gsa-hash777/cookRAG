"""Benchmark script for quantifying index cache benefits in RAG project.

Measures:
1) document loading time
2) chunking time
3) index build+save time (cold start)
4) index load time (warm start)
5) speedup ratio of warm load over cold build

Usage:
    python benchmark_index_cache.py --runs 10 --output benchmark_results.json
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path
from typing import Dict, List

from config import DEFAULT_CONFIG
from rag_modules import DataPreparationModule, IndexConstructionModule

BASE_DIR = Path(__file__).resolve().parent


def _resolve_path(path_str: str) -> str:
    p = Path(path_str)
    if p.is_absolute():
        return str(p)
    return str((BASE_DIR / p).resolve())


def _time_it(fn):
    start = time.perf_counter()
    result = fn()
    elapsed = time.perf_counter() - start
    return result, elapsed


def single_run(data_path: str, index_path: str, embedding_model: str) -> Dict[str, float]:
    data_module = DataPreparationModule(_resolve_path(data_path))

    # load + chunk timing
    _, load_time = _time_it(data_module.load_documents)
    chunks, chunk_time = _time_it(data_module.chunk_documents)

    # cold build timing
    resolved_index_path = _resolve_path(index_path)
    build_module = IndexConstructionModule(model_name=embedding_model, index_save_path=resolved_index_path)
    _, build_time = _time_it(lambda: build_module.build_vector_index(chunks))
    _, save_time = _time_it(build_module.save_index)

    # warm load timing
    load_module = IndexConstructionModule(model_name=embedding_model, index_save_path=resolved_index_path)
    _, warm_load_time = _time_it(load_module.load_index)

    cold_total = load_time + chunk_time + build_time + save_time
    warm_total = load_time + chunk_time + warm_load_time

    return {
        "documents": len(data_module.documents),
        "chunks": len(chunks),
        "load_time_s": load_time,
        "chunk_time_s": chunk_time,
        "build_time_s": build_time,
        "save_time_s": save_time,
        "warm_load_time_s": warm_load_time,
        "cold_total_s": cold_total,
        "warm_total_s": warm_total,
        "speedup_x": (cold_total / warm_total) if warm_total > 0 else 0.0,
    }


def summarize(results: List[Dict[str, float]]) -> Dict[str, float]:
    keys = [
        "load_time_s",
        "chunk_time_s",
        "build_time_s",
        "save_time_s",
        "warm_load_time_s",
        "cold_total_s",
        "warm_total_s",
        "speedup_x",
    ]

    summary = {
        "runs": len(results),
        "documents": int(results[0]["documents"]) if results else 0,
        "chunks": int(results[0]["chunks"]) if results else 0,
    }

    for key in keys:
        values = [r[key] for r in results]
        summary[f"avg_{key}"] = statistics.mean(values)
        summary[f"min_{key}"] = min(values)
        summary[f"max_{key}"] = max(values)

    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark C8 index cache speedup")
    parser.add_argument("--runs", type=int, default=3, help="Number of benchmark runs")
    parser.add_argument(
        "--data-path",
        type=str,
        default=DEFAULT_CONFIG.data_path,
        help="Path to C8 recipe markdown dataset",
    )
    parser.add_argument(
        "--index-path",
        type=str,
        default="./benchmark_vector_index",
        help="Temporary index path for benchmark",
    )
    parser.add_argument(
        "--embedding-model",
        type=str,
        default=DEFAULT_CONFIG.embedding_model,
        help="Embedding model name",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_results.json",
        help="Output JSON file path",
    )
    args = parser.parse_args()

    index_dir = Path(_resolve_path(args.index_path))
    all_results: List[Dict[str, float]] = []

    for i in range(1, args.runs + 1):
        if index_dir.exists():
            # clean previous index to ensure fair cold build each run
            for p in index_dir.rglob("*"):
                if p.is_file():
                    p.unlink()
            for p in sorted(index_dir.glob("**/*"), reverse=True):
                if p.is_dir():
                    p.rmdir()
            index_dir.rmdir()

        result = single_run(
            data_path=args.data_path,
            index_path=args.index_path,
            embedding_model=args.embedding_model,
        )
        all_results.append(result)

        print(
            f"Run {i}: cold_total={result['cold_total_s']:.2f}s, "
            f"warm_total={result['warm_total_s']:.2f}s, speedup={result['speedup_x']:.2f}x"
        )

    summary = summarize(all_results)
    payload = {
        "config": {
            "runs": args.runs,
            "data_path": args.data_path,
            "index_path": args.index_path,
            "embedding_model": args.embedding_model,
        },
        "results": all_results,
        "summary": summary,
    }

    Path(args.output).write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n=== Benchmark Summary ===")
    print(f"docs={summary['documents']}, chunks={summary['chunks']}, runs={summary['runs']}")
    print(f"avg cold_total: {summary['avg_cold_total_s']:.2f}s")
    print(f"avg warm_total: {summary['avg_warm_total_s']:.2f}s")
    print(f"avg speedup:    {summary['avg_speedup_x']:.2f}x")
    print(f"Saved to: {args.output}")


if __name__ == "__main__":
    main()