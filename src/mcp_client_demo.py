from __future__ import annotations

import argparse
import asyncio
import json

from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client
from mcp.types import TextContent


async def run_demo(server_url: str, question: str) -> None:
    async with streamable_http_client(server_url) as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()
            result = await session.call_tool("ask_kb", {"question": question})
            if getattr(result, "structuredContent", None):
                print(json.dumps(result.structuredContent, ensure_ascii=False, indent=2))
                return
            texts = [
                content.text
                for content in result.content
                if isinstance(content, TextContent)
            ]
            print("\n".join(texts))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--server-url",
        default="http://127.0.0.1:8000/mcp",
        help="MCP streamable HTTP URL",
    )
    parser.add_argument(
        "--question",
        default="企业宽带开通需要多久？",
        help="Question to send through the MCP tool",
    )
    args = parser.parse_args()
    asyncio.run(run_demo(args.server_url, args.question))


if __name__ == "__main__":
    main()
