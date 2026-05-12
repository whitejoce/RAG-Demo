from __future__ import annotations

import argparse
import json
import sys

from src.kb.chain import answer_question
from src.kb.ingest import build_embeddings, build_or_load_vectorstore
from src.settings import load_settings


def cmd_build_index(_: argparse.Namespace) -> None:
    settings = load_settings()
    embeddings = build_embeddings(settings)
    vectorstore = build_or_load_vectorstore(
        settings.raw_dir,
        settings.index_dir,
        embeddings,
        rebuild=True,
    )
    print(f"Index built at: {settings.index_dir}")
    print(f"Vector count: {vectorstore.index.ntotal}")


def cmd_ask(args: argparse.Namespace) -> None:
    result = answer_question(args.question)
    print(json.dumps(result, ensure_ascii=False, indent=2))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="enterprise-kb")
    subparsers = parser.add_subparsers(dest="command", required=True)

    build_index = subparsers.add_parser("build-index", help="Build or rebuild FAISS index")
    build_index.set_defaults(func=cmd_build_index)

    ask = subparsers.add_parser("ask", help="Ask a question")
    ask.add_argument("question", help="Question to ask against the KB")
    ask.set_defaults(func=cmd_ask)

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    try:
        args.func(args)
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"Error: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc


if __name__ == "__main__":
    main()

