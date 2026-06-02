#!/usr/bin/env python3
"""Locked and cross-source evaluation for frontier v13 specialists."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from v5_pipeline_utils import load_json, load_jsonl, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--boundary", default="reports/vlm/boundary_expert_v13_eval.json")
    parser.add_argument("--room", default="reports/vlm/room_space_expert_v13_eval.json")
    parser.add_argument("--symbol", default="reports/vlm/symbol_fixture_expert_v13_eval.json")
    parser.add_argument("--text", default="reports/vlm/text_dimension_expert_v13_eval.json")
    parser.add_argument("--output", default="reports/vlm/frontier_specialist_locked_comparison_v13.json")
    parser.add_argument("--gallery", default="reports/vlm/frontier_specialist_locked_gallery_v13.html")
    args = parser.parse_args()

    reports = {name: load_json(path, {}) for name, path in {"boundary": args.boundary, "room": args.room, "symbol": args.symbol, "text": args.text}.items()}
    comparison = {
        "version": "frontier_specialist_locked_comparison_v13",
        "boundary": summarize(reports["boundary"]),
        "room_space": summarize(reports["room"]),
        "symbol_fixture": summarize(reports["symbol"]),
        "text_dimension": summarize(reports["text"]),
        "cross_source_summary": cross_source_summary(reports),
        "claim_boundary": "This comparison only reflects locked report outputs already produced by specialist runs.",
    }
    Path(args.gallery).write_text(render_gallery(comparison), encoding="utf-8")
    write_json(args.output, comparison)
    print(json.dumps(comparison, ensure_ascii=False, indent=2))


def summarize(report: dict[str, Any]) -> dict[str, Any]:
    for key in ["locked_metrics", "locked_symbol_metrics", "locked_metrics", "train_metrics"]:
        value = report.get(key)
        if isinstance(value, dict):
            return value
    return {"available": False}


def cross_source_summary(reports: dict[str, dict[str, Any]]) -> dict[str, Any]:
    return {
        "boundary_has_locked_metrics": bool(reports["boundary"].get("locked_metrics")),
        "room_has_locked_metrics": bool(reports["room"].get("locked_metrics")),
        "symbol_has_locked_metrics": bool(reports["symbol"].get("locked_metrics")),
        "text_has_locked_metrics": bool(reports["text"].get("locked_metrics")),
        "at_least_two_specialists_improved": sum(bool(report.get("adopted")) for report in reports.values()) >= 2,
    }


def render_gallery(comparison: dict[str, Any]) -> str:
    body = json.dumps(comparison, ensure_ascii=False, indent=2)
    return f"<!doctype html><meta charset='utf-8'><title>frontier specialist comparison v13</title><pre>{body}</pre>"


if __name__ == "__main__":
    main()
