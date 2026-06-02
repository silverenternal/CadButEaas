#!/usr/bin/env python3
"""Audit the relation-graph reconstruction policy against the current baseline."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from relation_graph_reconstruction_v18 import DEFAULT_ADAPTER, DEFAULT_AUDIT, DEFAULT_EVAL, DEFAULT_INPUT, DEFAULT_MODEL, evaluate_selection, load_by_id, load_jsonl, render_pages, summarize_delta, write_json


def load_model(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--baseline-eval", default=str(DEFAULT_EVAL))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--review-pack", default="reports/vlm/relation_graph_reconstruction_v18_review_pack.json")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    limit = 5 if args.smoke else args.limit
    pages = load_jsonl(Path(args.input), limit=limit)
    adapter_rows = load_jsonl(Path(args.adapter), limit=limit)
    adapter_by_id = {str(row.get("id")): row for row in adapter_rows}
    model = load_model(Path(args.model))
    policy = model.get("policy") or {}
    page_rows, feature_rows, audit = render_pages(pages, adapter_by_id, model, policy)
    eval_report = evaluate_selection(page_rows, adapter_rows)
    baseline = json.loads(Path(args.baseline_eval).read_text(encoding="utf-8")) if Path(args.baseline_eval).exists() else {"relation_metrics": {}}
    delta = summarize_delta(baseline, eval_report)
    report = {
        "task": "IMG-MOE-V18-REBUILD-005",
        "rows": len(page_rows),
        "features": len(feature_rows),
        "policy": policy,
        "eval": eval_report,
        "delta": delta,
        "source_integrity": eval_report.get("source_integrity"),
    }
    write_json(Path(args.audit_output), report)
    write_json(
        Path(args.review_pack),
        {
            "task": "IMG-MOE-V18-REBUILD-005",
            "summary": {
                "rows": len(page_rows),
                "features": len(feature_rows),
                "policy": policy,
                "delta": delta,
            },
            "page_audits": audit.get("page_stats", [])[:50],
            "warning_counts": audit.get("warning_counts") or {},
            "source_integrity": eval_report.get("source_integrity"),
        },
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
