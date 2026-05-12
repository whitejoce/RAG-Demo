from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
load_dotenv(ROOT / ".env")


def env(name: str, default: str | None = None) -> str:
    value = os.getenv(name, default)
    if value is None:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    return int(value) if value else default


def env_float(name: str, default: float) -> float:
    value = os.getenv(name)
    return float(value) if value else default


@dataclass(frozen=True)
class Settings:
    api_key: str
    base_url: str
    chat_model: str
    embed_model: str
    embed_dimensions: int | None
    raw_dir: Path
    index_dir: Path
    top_k: int
    mcp_host: str
    mcp_port: int


def load_settings() -> Settings:
    dimensions_raw = os.getenv("ZHIPU_EMBED_DIMENSIONS", "").strip()
    return Settings(
        api_key=env("ZHIPU_API_KEY"),
        base_url=os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4"),
        chat_model=os.getenv("ZHIPU_CHAT_MODEL", "glm-4-flash"),
        embed_model=os.getenv("ZHIPU_EMBED_MODEL", "embedding-3"),
        embed_dimensions=int(dimensions_raw) if dimensions_raw else None,
        raw_dir=Path(os.getenv("KB_RAW_DIR", ROOT / "data" / "raw")),
        index_dir=Path(os.getenv("KB_INDEX_DIR", ROOT / "data" / "index" / "faiss")),
        top_k=env_int("KB_TOP_K", 4),
        mcp_host=os.getenv("MCP_HOST", "127.0.0.1"),
        mcp_port=env_int("MCP_PORT", 8000),
    )

