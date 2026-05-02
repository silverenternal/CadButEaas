#!/usr/bin/env python3
"""Summarize audit fields from an existing VLM evaluation report."""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

from eval_metrics import count_warnings, safe_rate


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("report")
    parser.add_argument("--output")
    args = parser.parse_args()

    report = json.loads(Path(args.report).read_text(encoding="utf-8"))
    rows = report.get("rows") if isinstance(report.get("rows"), list) else []
    warning_counts = count_warnings(rows)
    semantic_counts = [int(row.get("semantic_count", 0) or 0) for row in rows if row.get("ok")]
    audit = {
        "report": args.report,
        "total": len(rows),
        "ok": sum(1 for row in rows if row.get("ok")),
        "semantic_hit_rate": report.get("semantic_hit_rate"),
        "semantic_exact_f1_mean": report.get("semantic_exact_f1_mean"),
        "geometry_consistency_mean": report.get("geometry_consistency_mean"),
        "empty_semantic_rate": safe_rate(
            sum(1 for row in rows if row.get("ok") and int(row.get("semantic_count", 0) or 0) == 0),
            len(rows),
        ),
        "semantic_count_mean": round(statistics.mean(semantic_counts), 3) if semantic_counts else 0.0,
        "partial_recovery_count": warning_counts.get("partial_json_recovered", 0),
        "warning_counts": warning_counts,
    }
    text = json.dumps(audit, ensure_ascii=False, indent=2)
    print(text)
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(text + "\n", encoding="utf-8")

if __name__ == "__main__":
    main()
