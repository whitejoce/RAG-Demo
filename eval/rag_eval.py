from __future__ import annotations

import argparse
import json
import time
from dataclasses import replace
from pathlib import Path

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate

from src.kb.chain import SYSTEM_PROMPT, build_llm, build_retriever
from src.settings import load_settings

from eval.common import DATASETS_DIR, ensure_results_dir, fmt_ms, keyword_hit, mean, print_table, read_jsonl, token_f1, write_json


def evaluate(dataset_path: Path, top_k: int | None = None) -> dict:
    settings = load_settings()
    if top_k is not None:
        settings = replace(settings, top_k=top_k)
    retriever = build_retriever(settings)
    llm = build_llm(settings)

    document_prompt = PromptTemplate.from_template("来源: {source} 页码: {page}\n内容: {page_content}")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", SYSTEM_PROMPT),
            ("human", "问题：{input}\n\n请仅依据以下资料回答：\n{context}"),
        ]
    )
    chain = create_retrieval_chain(
        retriever,
        create_stuff_documents_chain(llm, prompt, document_prompt=document_prompt),
    )

    examples = read_jsonl(dataset_path)
    per_example: list[dict] = []
    answer_scores: list[float] = []
    source_hits: list[float] = []
    keyword_hits: list[float] = []
    latencies: list[float] = []

    for example in examples:
        question = example["question"]
        expected_sources = example.get("expected_sources", [])
        reference_answer = example.get("reference_answer", "")
        must_contain = example.get("must_contain", [])

        start = time.perf_counter()
        result = chain.invoke({"input": question})
        elapsed = time.perf_counter() - start

        answer = result["answer"]
        context = result.get("context", [])
        retrieved_sources = [str(doc.metadata.get("source", "")) for doc in context]
        source_hit = 1.0 if set(retrieved_sources) & set(expected_sources) else 0.0
        keyword_ok = 1.0 if keyword_hit(answer, must_contain, require_all=True) else 0.0
        answer_score = token_f1(answer, reference_answer) if reference_answer else 0.0

        answer_scores.append(answer_score)
        source_hits.append(source_hit)
        keyword_hits.append(keyword_ok)
        latencies.append(elapsed)

        per_example.append(
            {
                "id": example.get("id", ""),
                "question": question,
                "answer": answer[:180].replace("\n", " "),
                "sources": ",".join(retrieved_sources[: settings.top_k]),
                "source_hit": source_hit,
                "keyword_ok": keyword_ok,
                "answer_f1": round(answer_score, 3),
                "latency_ms": fmt_ms(elapsed),
            }
        )

    summary = {
        "dataset": str(dataset_path),
        "num_examples": len(examples),
        "top_k": top_k or settings.top_k,
        "source_hit_rate": round(mean(source_hits), 4),
        "keyword_hit_rate": round(mean(keyword_hits), 4),
        "avg_answer_f1": round(mean(answer_scores), 4),
        "avg_latency_ms": round(mean(fmt_ms(x) for x in latencies), 2),
    }
    return {"summary": summary, "examples": per_example}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate end-to-end RAG quality")
    parser.add_argument("--dataset", type=Path, default=DATASETS_DIR / "rag_gold.jsonl", help="RAG gold set")
    parser.add_argument("--top-k", type=int, default=None, help="Override retriever top-k")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    result = evaluate(args.dataset, top_k=args.top_k)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print_table(result["examples"], ["id", "question", "source_hit", "keyword_ok", "answer_f1", "latency_ms"])
    output = args.output or ensure_results_dir() / "rag_eval.json"
    write_json(output, result)


if __name__ == "__main__":
    main()

