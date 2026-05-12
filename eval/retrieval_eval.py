from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from src.kb.chain import build_retriever
from src.settings import load_settings

from eval.common import DATASETS_DIR, ensure_results_dir, first_match_rank, fmt_ms, mean, print_table, read_jsonl, write_json


def evaluate(dataset_path: Path, top_k: int | None = None) -> dict:
    settings = load_settings()
    if top_k is not None:
        settings = replace(settings, top_k=top_k)
    retriever = build_retriever(settings)

    examples = read_jsonl(dataset_path)
    per_example: list[dict] = []
    recall_scores: list[float] = []
    hit_scores: list[float] = []
    mrr_scores: list[float] = []
    latencies: list[float] = []

    for example in examples:
        question = example["question"]
        expected_sources = example["relevant_sources"]
        start = time.perf_counter()
        documents = retriever.invoke(question)
        elapsed = time.perf_counter() - start

        retrieved_sources = [str(doc.metadata.get("source", "")) for doc in documents]
        rank = first_match_rank(retrieved_sources, expected_sources)
        relevant_hits = len(set(retrieved_sources) & set(expected_sources))
        recall = relevant_hits / max(len(set(expected_sources)), 1)
        hit = 1.0 if rank is not None else 0.0
        mrr = 1.0 / rank if rank else 0.0

        recall_scores.append(recall)
        hit_scores.append(hit)
        mrr_scores.append(mrr)
        latencies.append(elapsed)

        per_example.append(
            {
                "id": example.get("id", ""),
                "question": question,
                "expected": ",".join(expected_sources),
                "retrieved": ",".join(retrieved_sources[: top_k or settings.top_k]),
                "hit@k": hit,
                "recall": round(recall, 3),
                "mrr": round(mrr, 3),
                "latency_ms": fmt_ms(elapsed),
            }
        )

    summary = {
        "dataset": str(dataset_path),
        "num_examples": len(examples),
        "top_k": top_k or settings.top_k,
        "hit_at_k": round(mean(hit_scores), 4),
        "recall_at_k": round(mean(recall_scores), 4),
        "mrr": round(mean(mrr_scores), 4),
        "avg_latency_ms": round(mean(fmt_ms(x) for x in latencies), 2),
    }
    return {"summary": summary, "examples": per_example}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality")
    parser.add_argument("--dataset", type=Path, default=DATASETS_DIR / "retrieval_gold.jsonl", help="Retrieval gold set")
    parser.add_argument("--top-k", type=int, default=None, help="Override retriever top-k")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    result = evaluate(args.dataset, top_k=args.top_k)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print_table(result["examples"], ["id", "question", "expected", "retrieved", "hit@k", "recall", "mrr", "latency_ms"])
    output = args.output or ensure_results_dir() / "retrieval_eval.json"
    write_json(output, result)


if __name__ == "__main__":
    main()

