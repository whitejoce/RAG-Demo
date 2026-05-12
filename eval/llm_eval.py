from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from langchain_core.messages import HumanMessage, SystemMessage

from src.kb.chain import build_llm
from src.settings import load_settings

from eval.common import DATASETS_DIR, ensure_results_dir, fmt_ms, keyword_hit, mean, print_table, read_jsonl, token_f1, write_json


SYSTEM_MESSAGE = (
    "你是一个企业知识问答模型。"
    "请严格根据给定上下文作答。"
    "如果上下文中没有答案，就直接说明无法从资料中确定。"
)


def evaluate(dataset_path: Path) -> dict:
    settings = load_settings()
    llm = build_llm(settings)
    examples = read_jsonl(dataset_path)

    per_example: list[dict] = []
    answer_scores: list[float] = []
    keyword_hits: list[float] = []
    latencies: list[float] = []

    for example in examples:
        question = example["question"]
        context = example["context"]
        reference_answer = example.get("reference_answer", "")
        must_contain = example.get("must_contain", [])

        messages = [
            SystemMessage(content=SYSTEM_MESSAGE),
            HumanMessage(content=f"问题：{question}\n\n上下文：\n{context}\n\n请给出简洁准确的回答。"),
        ]
        start = time.perf_counter()
        response = llm.invoke(messages)
        elapsed = time.perf_counter() - start

        answer = response.content if isinstance(response.content, str) else str(response.content)
        answer_score = token_f1(answer, reference_answer) if reference_answer else 0.0
        keyword_ok = 1.0 if keyword_hit(answer, must_contain, require_all=True) else 0.0

        answer_scores.append(answer_score)
        keyword_hits.append(keyword_ok)
        latencies.append(elapsed)

        per_example.append(
            {
                "id": example.get("id", ""),
                "question": question,
                "answer": answer[:180].replace("\n", " "),
                "keyword_ok": keyword_ok,
                "answer_f1": round(answer_score, 3),
                "latency_ms": fmt_ms(elapsed),
            }
        )

    summary = {
        "dataset": str(dataset_path),
        "num_examples": len(examples),
        "avg_answer_f1": round(mean(answer_scores), 4),
        "keyword_hit_rate": round(mean(keyword_hits), 4),
        "avg_latency_ms": round(mean(fmt_ms(x) for x in latencies), 2),
    }
    return {"summary": summary, "examples": per_example}


def main() -> None:
    parser = argparse.ArgumentParser(description="Benchmark the raw LLM generation quality")
    parser.add_argument("--dataset", type=Path, default=DATASETS_DIR / "llm_gold.jsonl", help="LLM benchmark set")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    result = evaluate(args.dataset)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print_table(result["examples"], ["id", "question", "keyword_ok", "answer_f1", "latency_ms"])
    output = args.output or ensure_results_dir() / "llm_eval.json"
    write_json(output, result)


if __name__ == "__main__":
    main()

