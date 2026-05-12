from __future__ import annotations

import json
import re
import statistics
from collections import Counter
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Iterable


ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = ROOT / "eval" / "datasets"
RESULTS_DIR = ROOT / "eval" / "results"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def ensure_results_dir() -> Path:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    return RESULTS_DIR


def write_json(path: Path, payload: Any) -> None:
    if is_dataclass(payload):
        payload = asdict(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def normalize_text(text: str) -> str:
    return "".join(re.findall(r"[\u4e00-\u9fffA-Za-z0-9]+", text.lower()))


def token_f1(prediction: str, reference: str) -> float:
    pred = list(normalize_text(prediction))
    ref = list(normalize_text(reference))
    if not pred or not ref:
        return 0.0
    pred_counts = Counter(pred)
    ref_counts = Counter(ref)
    overlap = sum((pred_counts & ref_counts).values())
    precision = overlap / len(pred)
    recall = overlap / len(ref)
    if precision + recall == 0:
        return 0.0
    return 2 * precision * recall / (precision + recall)


def keyword_hit(text: str, keywords: Iterable[str], require_all: bool = True) -> bool:
    normalized = text.lower()
    checks = [keyword.lower() in normalized for keyword in keywords]
    if not checks:
        return True
    return all(checks) if require_all else any(checks)


def first_match_rank(retrieved: list[str], targets: Iterable[str]) -> int | None:
    target_set = {target for target in targets}
    for index, item in enumerate(retrieved, start=1):
        if item in target_set:
            return index
    return None


def mean(values: Iterable[float]) -> float:
    values = list(values)
    return statistics.mean(values) if values else 0.0


def fmt_ms(seconds: float) -> float:
    return round(seconds * 1000, 2)


def print_table(rows: list[dict[str, Any]], headers: list[str]) -> None:
    widths = {header: len(header) for header in headers}
    for row in rows:
        for header in headers:
            widths[header] = max(widths[header], len(str(row.get(header, ""))))

    def render_row(row: dict[str, Any]) -> str:
        return " | ".join(str(row.get(header, "")).ljust(widths[header]) for header in headers)

    print(render_row({header: header for header in headers}))
    print("-+-".join("-" * widths[header] for header in headers))
    for row in rows:
        print(render_row(row))

