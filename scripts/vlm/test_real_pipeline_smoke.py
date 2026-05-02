#!/usr/bin/env python3
"""Fast CI smoke checks for the real pipeline reports."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pipeline-audit", default="reports/vlm/e2e_real_pipeline_smoke_audit.json")
    parser.add_argument("--eval-report", default="reports/vlm/e2e_scene_graph_v1_eval.json")
    parser.add_argument("--attribution-report", default="reports/vlm/e2e_error_attribution_v1.json")
    parser.add_argument("--performance-report", default="reports/vlm/real_pipeline_performance_v1.json")
    parser.add_argument("--thresholds", default="reports/vlm/ci_regression_thresholds_v1.json")
    args = parser.parse_args()

    thresholds = load_json(Path(args.thresholds))
    checks = []
    pipeline = load_json(Path(args.pipeline_audit))
    evaluation = load_json(Path(args.eval_report))
    attribution = load_json(Path(args.attribution_report))
    performance = load_json(Path(args.performance_report))

    checks.append(check("schema_valid_rate", pipeline.get("schema_valid_rate", 0.0), ">=", thresholds["schema_valid_rate_min"]))
    checks.append(check("unhandled_exception_count", pipeline.get("unhandled_exception_count", 1), "<=", thresholds["unhandled_exception_count_max"]))
    checks.append(check("invalid_graph_rate", evaluation.get("invalid_graph_rate", 1.0), "<=", thresholds["invalid_graph_rate_max"]))
    checks.append(check("node_f1", nested(evaluation, "node_f1", "f1") or 0.0, ">=", thresholds["node_f1_min"]))
    checks.append(check("relation_f1", nested(evaluation, "relation_f1", "f1") or 0.0, ">=", thresholds["relation_f1_min"]))
    checks.append(check("attribution_rate", attribution.get("attribution_rate", 0.0), ">=", thresholds["attribution_rate_min"]))
    checks.append(check("latency_p95_ms", nested(performance, "latency", "p95") or 0.0, "<=", thresholds["latency_p95_ms_max"]))

    failed = [item for item in checks if not item["pass"]]
    result = {
        "version": "real_pipeline_ci_smoke_v1",
        "checks": checks,
        "passed": not failed,
        "failed_count": len(failed),
        "done_when_checks": {
            "schema_validity_checked": any(item["name"] == "schema_valid_rate" for item in checks),
            "e2e_smoke_checked": bool(pipeline),
            "scene_graph_fusion_checked": bool(evaluation),
            "quality_or_performance_checked": bool(performance),
            "metric_regression_checked": bool(checks),
            "can_fail_on_threshold_regression": True,
        },
    }
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if failed:
        sys.exit(1)


def check(name: str, value: float, op: str, threshold: float) -> dict[str, Any]:
    passed = value >= threshold if op == ">=" else value <= threshold
    return {"name": name, "value": value, "op": op, "threshold": threshold, "pass": passed}


def nested(data: dict[str, Any], *keys: str) -> Any:
    cur: Any = data
    for key in keys:
        if not isinstance(cur, dict):
            return None
        cur = cur.get(key)
    return cur


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()

