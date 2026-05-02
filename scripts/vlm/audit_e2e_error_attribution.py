#!/usr/bin/env python3
"""Bucket end-to-end residuals by primary failure cause."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


CAUSE_PRIORITY = [
    ("proposal_miss", {"expert_or_proposal_node_miss", "proposal_or_topology_support_miss"}),
    ("expert_misclass", {"expert_extra_node"}),
    ("router_wrong_expert", {"router_wrong_expert"}),
    ("fusion_constraint", {"fusion_relation_miss", "fusion_extra_relation", "fusion_constraint_invalid_graph"}),
    ("ocr_miss", {"OCR_or_dimension_link_miss"}),
    ("quality_degradation", {"quality_degradation"}),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="reports/vlm/e2e_scene_graph_v1_cases.jsonl")
    parser.add_argument("--output", default="reports/vlm/e2e_error_attribution_v1.json")
    parser.add_argument("--cases-output", default="reports/vlm/e2e_error_attribution_v1_cases.jsonl")
    args = parser.parse_args()

    cases = load_jsonl(Path(args.cases))
    attributed: list[dict[str, Any]] = []
    primary_counts: Counter[str] = Counter()
    multi_cause_count = 0
    attributed_count = 0

    for case in cases:
        tags = set(str(tag) for tag in case.get("failure_tags") or [])
        causes = causes_for(tags, case)
        if len(causes) > 1:
            primary = "multi_cause"
            multi_cause_count += 1
        elif causes:
            primary = causes[0]
        else:
            primary = "unknown"
        if primary != "unknown":
            attributed_count += 1
        primary_counts[primary] += 1
        attributed.append(
            {
                "image": case.get("image"),
                "annotation": case.get("annotation"),
                "source_dataset": case.get("source_dataset"),
                "primary_cause": primary,
                "causes": causes,
                "failure_tags": sorted(tags),
                "missing_node_count": len(case.get("missing_nodes") or []),
                "extra_node_count": len(case.get("extra_nodes") or []),
                "missing_edge_count": len(case.get("missing_edges") or []),
                "extra_edge_count": len(case.get("extra_edges") or []),
                "warnings": case.get("warnings") or [],
            }
        )

    residual_count = len(cases)
    attribution_rate = round(attributed_count / max(residual_count, 1), 6)
    report = {
        "version": "e2e_error_attribution_v1",
        "cases": args.cases,
        "residual_count": residual_count,
        "attributed_count": attributed_count,
        "attribution_rate": attribution_rate,
        "multi_cause_count": multi_cause_count,
        "primary_cause_counts": dict(primary_counts.most_common()),
        "done_when_checks": {
            "attribution_rate_at_least_0_95": attribution_rate >= 0.95 or residual_count == 0,
            "all_residuals_have_unique_or_multi_cause": all(
                item["primary_cause"] != "unknown" for item in attributed
            ),
        },
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(Path(args.cases_output), attributed)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def causes_for(tags: set[str], case: dict[str, Any]) -> list[str]:
    causes = [cause for cause, cause_tags in CAUSE_PRIORITY if tags & cause_tags]
    if not causes and (case.get("missing_edges") or case.get("extra_edges")):
        causes.append("fusion_constraint")
    if not causes and (case.get("missing_nodes") or case.get("extra_nodes")):
        causes.append("expert_misclass")
    warnings = " ".join(str(item) for item in case.get("warnings") or [])
    if "degraded" in warnings and "quality_degradation" not in causes:
        causes.append("quality_degradation")
    return causes


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()

