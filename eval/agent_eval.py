from __future__ import annotations

import argparse
import asyncio
import importlib
import json
import time
from pathlib import Path
from typing import Any

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from src.kb.chain import build_llm, build_qa_chain, build_retriever
from src.settings import load_settings

from eval.common import DATASETS_DIR, ensure_results_dir, fmt_ms, keyword_hit, mean, print_table, read_jsonl, write_json


async def call_mcp_tool(server_url: str, question: str) -> dict[str, Any]:
    async with streamable_http_client(server_url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool("ask_kb", {"question": question})
            structured = getattr(result, "structuredContent", None)
            if structured:
                return structured
            texts = []
            for content in result.content:
                text = getattr(content, "text", None)
                if text:
                    texts.append(text)
            if not texts:
                return {"question": question, "answer": ""}
            try:
                return json.loads("\n".join(texts))
            except json.JSONDecodeError:
                return {"question": question, "answer": "\n".join(texts)}


def load_runner(path: str):
    module_name, _, attr = path.partition(":")
    if not module_name or not attr:
        raise ValueError("runner must be in module:function format")
    module = importlib.import_module(module_name)
    return getattr(module, attr)


def evaluate(dataset_path: Path, mode: str, server_url: str | None, runner_path: str | None) -> dict:
    examples = read_jsonl(dataset_path)
    per_example: list[dict] = []
    success_rates: list[float] = []
    keyword_hits: list[float] = []
    latencies: list[float] = []

    runner = load_runner(runner_path) if runner_path else None
    direct_chain = None
    if runner is None and mode == "direct":
        settings = load_settings()
        retriever = build_retriever(settings)
        llm = build_llm(settings)
        direct_chain = build_qa_chain(retriever=retriever, llm=llm, settings=settings)

    for example in examples:
        task = example["task"]
        question = example.get("question", task)
        expected_keywords = example.get("expected_keywords", [])
        expected_tool = example.get("expected_tool", "ask_kb")

        start = time.perf_counter()
        if runner is not None:
            output = runner(task=task, question=question)
        elif mode == "mcp":
            if not server_url:
                raise ValueError("--server-url is required in mcp mode")
            output = asyncio.run(call_mcp_tool(server_url, question))
        else:
            result = direct_chain.invoke({"input": question})
            output = {
                "question": question,
                "answer": result["answer"],
                "sources": result.get("context", []),
                "tool_name": "ask_kb",
                "tool_calls": 1,
            }
        elapsed = time.perf_counter() - start

        answer = output.get("answer", "") if isinstance(output, dict) else str(output)
        tool_calls = output.get("tool_calls", 1) if isinstance(output, dict) else 1
        tool_name = output.get("tool_name", expected_tool) if isinstance(output, dict) else expected_tool
        success = 1.0 if keyword_hit(answer, expected_keywords, require_all=True) else 0.0
        tool_ok = 1.0 if tool_name == expected_tool else 0.0

        success_rates.append(success)
        keyword_hits.append(tool_ok)
        latencies.append(elapsed)

        per_example.append(
            {
                "id": example.get("id", ""),
                "task": task,
                "tool_name": tool_name,
                "tool_calls": tool_calls,
                "success": success,
                "tool_ok": tool_ok,
                "latency_ms": fmt_ms(elapsed),
            }
        )

    summary = {
        "dataset": str(dataset_path),
        "mode": mode,
        "num_examples": len(examples),
        "task_success_rate": round(mean(success_rates), 4),
        "tool_match_rate": round(mean(keyword_hits), 4),
        "avg_latency_ms": round(mean(fmt_ms(x) for x in latencies), 2),
    }
    return {"summary": summary, "examples": per_example}


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the KB tool / agent handoff")
    parser.add_argument("--dataset", type=Path, default=DATASETS_DIR / "agent_tasks.jsonl", help="Agent task set")
    parser.add_argument("--mode", choices=["direct", "mcp"], default="direct", help="Evaluation mode")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/mcp", help="MCP server URL")
    parser.add_argument("--runner", default=None, help="Optional custom runner module:function")
    parser.add_argument("--output", type=Path, default=None, help="Optional JSON output path")
    args = parser.parse_args()

    result = evaluate(args.dataset, mode=args.mode, server_url=args.server_url, runner_path=args.runner)
    print(json.dumps(result["summary"], ensure_ascii=False, indent=2))
    print_table(result["examples"], ["id", "task", "tool_name", "tool_calls", "success", "tool_ok", "latency_ms"])
    output = args.output or ensure_results_dir() / "agent_eval.json"
    write_json(output, result)


if __name__ == "__main__":
    main()

