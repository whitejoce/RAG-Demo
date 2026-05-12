from __future__ import annotations

import asyncio
import json
import os

from mcp.server.fastmcp import FastMCP
from mcp.types import CallToolResult, TextContent

from src.kb.chain import answer_question


mcp = FastMCP(
    "enterprise-kb",
    host=os.getenv("MCP_HOST", "127.0.0.1"),
    port=int(os.getenv("MCP_PORT", "8000")),
    json_response=True,
)


@mcp.tool()
async def ask_kb(question: str) -> CallToolResult:
    if not question.strip():
        raise ValueError("question cannot be empty")
    result = await asyncio.to_thread(answer_question, question)
    answer_block = json.dumps(result, ensure_ascii=False, indent=2)
    return CallToolResult(
        content=[TextContent(type="text", text=answer_block)],
        structuredContent=result,
    )


def main() -> None:
    mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()

