from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any

import yaml


def atomic_write_text(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(content, encoding="utf-8")
    os.replace(temporary, path)


def write_json(path: Path, value: Any) -> None:
    atomic_write_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def write_yaml(path: Path, value: Any) -> None:
    atomic_write_text(path, yaml.safe_dump(value, sort_keys=False, allow_unicode=True))


def normalize_usage(
    call: str, category: str, model: str, usage: dict, attempts: int
) -> dict:
    prompt_details = usage.get("prompt_tokens_details") or {}
    completion_details = usage.get("completion_tokens_details") or {}
    cost = usage.get("cost")
    try:
        normalized_cost = float(cost) if cost is not None else None
    except (TypeError, ValueError):
        normalized_cost = None
    return {
        "call": call,
        "category": category,
        "model": model,
        "attempts": attempts,
        "prompt_tokens": _number(usage.get("prompt_tokens")),
        "completion_tokens": _number(usage.get("completion_tokens")),
        "reasoning_tokens": _number(completion_details.get("reasoning_tokens")),
        "cached_tokens": _number(prompt_details.get("cached_tokens")),
        "cache_write_tokens": _number(prompt_details.get("cache_write_tokens")),
        "total_tokens": _number(usage.get("total_tokens")),
        "cost": normalized_cost,
    }


def aggregate_usage(entries: list[dict]) -> dict:
    def subtotal(rows: list[dict]) -> dict:
        known_costs = [row["cost"] for row in rows if row["cost"] is not None]
        return {
            "calls": len(rows),
            "prompt_tokens": sum(row["prompt_tokens"] for row in rows),
            "completion_tokens": sum(row["completion_tokens"] for row in rows),
            "reasoning_tokens": sum(row["reasoning_tokens"] for row in rows),
            "cached_tokens": sum(row["cached_tokens"] for row in rows),
            "cache_write_tokens": sum(row["cache_write_tokens"] for row in rows),
            "total_tokens": sum(row["total_tokens"] for row in rows),
            "observed_cost": sum(known_costs),
            "cost_complete": len(known_costs) == len(rows),
            "had_retries": any(row["attempts"] > 1 for row in rows),
        }

    reviewer_rows = [entry for entry in entries if entry["category"] == "reviewer"]
    combiner_rows = [entry for entry in entries if entry["category"] == "combiner"]
    return {
        "calls": entries,
        "reviewers": subtotal(reviewer_rows),
        "combiner": subtotal(combiner_rows),
        "total": subtotal(entries),
    }


def write_usage_csv(path: Path, entries: list[dict], aggregate: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    fields = [
        "call",
        "category",
        "model",
        "attempts",
        "prompt_tokens",
        "completion_tokens",
        "reasoning_tokens",
        "cached_tokens",
        "cache_write_tokens",
        "total_tokens",
        "cost",
    ]
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(entries)
        total = aggregate["total"]
        writer.writerow(
            {
                "call": "TOTAL",
                "category": "all",
                "attempts": "",
                "prompt_tokens": total["prompt_tokens"],
                "completion_tokens": total["completion_tokens"],
                "reasoning_tokens": total["reasoning_tokens"],
                "cached_tokens": total["cached_tokens"],
                "cache_write_tokens": total["cache_write_tokens"],
                "total_tokens": total["total_tokens"],
                "cost": total["observed_cost"],
            }
        )
    os.replace(temporary, path)


def assessment_header(
    *, mode: str, succeeded: int, requested: int, aggregate: dict
) -> str:
    total = aggregate["total"]
    cost_label = "OpenCode-estimated cost"
    warning = ""
    if not total["cost_complete"] or total["had_retries"]:
        warning = (
            "\n> **Accounting note:** At least one OpenCode step lacked cost data or "
            "required a retry; provider billing may be higher.\n"
        )
    return (
        f"> **Mode:** `{mode}`  \n"
        f"> **Reviewers:** {succeeded}/{requested}  \n"
        f"> **Tokens:** {total['total_tokens']:,} total "
        f"({total['prompt_tokens']:,} input; {total['completion_tokens']:,} output; "
        f"{total['reasoning_tokens']:,} reasoning; {total['cached_tokens']:,} cached)  \n"
        f"> **{cost_label}:** ${total['observed_cost']:.8f}\n" + warning + "\n"
    )


def _number(value: Any) -> int:
    return int(value) if isinstance(value, (int, float)) else 0
