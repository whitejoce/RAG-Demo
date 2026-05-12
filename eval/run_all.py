from __future__ import annotations

import argparse
import json
from pathlib import Path

from eval.agent_eval import evaluate as evaluate_agent
from eval.common import DATASETS_DIR, ensure_results_dir, write_json
from eval.llm_eval import evaluate as evaluate_llm
from eval.rag_eval import evaluate as evaluate_rag
from eval.retrieval_eval import evaluate as evaluate_retrieval


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all KB evaluation templates")
    parser.add_argument("--output-dir", type=Path, default=None, help="Output directory")
    parser.add_argument("--mode", choices=["direct", "mcp"], default="direct", help="Agent evaluation mode")
    parser.add_argument("--server-url", default="http://127.0.0.1:8000/mcp", help="MCP server URL")
    args = parser.parse_args()

    results = {
        "retrieval": evaluate_retrieval(DATASETS_DIR / "retrieval_gold.jsonl"),
        "rag": evaluate_rag(DATASETS_DIR / "rag_gold.jsonl"),
        "llm": evaluate_llm(DATASETS_DIR / "llm_gold.jsonl"),
        "agent": evaluate_agent(DATASETS_DIR / "agent_tasks.jsonl", mode=args.mode, server_url=args.server_url, runner_path=None),
    }
    output_dir = args.output_dir or ensure_results_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    write_json(output_dir / "all_eval.json", results)
    print(json.dumps({name: payload["summary"] for name, payload in results.items()}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

