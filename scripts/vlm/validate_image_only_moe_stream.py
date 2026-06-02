#!/usr/bin/env python3
"""Fail-closed validator for true image-only CadStruct-MoE prediction streams."""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "configs/vlm/image_only_moe_contract_v1.json"
DEFAULT_REPORT = ROOT / "reports/vlm/image_only_source_integrity_gate.json"


def load_json(path: str | Path, default: Any | None = None) -> Any:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return {} if default is None else default
    return json.loads(p.read_text(encoding="utf-8"))


def load_jsonl(path: str | Path) -> list[dict[str, Any]]:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    rows: list[dict[str, Any]] = []
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: str | Path, value: Any) -> None:
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def _walk(value: Any, path: str = "$") -> list[tuple[str, str, Any]]:
    found: list[tuple[str, str, Any]] = []
    if isinstance(value, dict):
        for key, item in value.items():
            found.append((path, str(key), item))
            found.extend(_walk(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            found.extend(_walk(item, f"{path}[{index}]"))
    return found


def _bad_id(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    return bool(re.match(r"^(svg_|expected_|oracle_|parser_)", value))


def validate_rows(rows: list[dict[str, Any]], contract: dict[str, Any]) -> dict[str, Any]:
    required_fields = contract.get("required_row_fields") or []
    required_integrity = contract.get("required_source_integrity") or {}
    forbidden_fields = {str(x) for x in contract.get("forbidden_field_names") or []}
    forbidden_values = [str(x).lower() for x in contract.get("forbidden_value_fragments") or []]
    violations: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()

    for row_index, row in enumerate(rows):
        if not isinstance(row, dict):
            violations.append({"row": row_index, "reason": "row_not_object"})
            continue
        for field in required_fields:
            if field not in row:
                violations.append({"row": row_index, "path": "$", "reason": "missing_required_field", "field": field})
                counters["missing_required_field"] += 1
        integrity = row.get("source_integrity") if isinstance(row.get("source_integrity"), dict) else {}
        route = row.get("route_trace") if isinstance(row.get("route_trace"), dict) else {}
        for key, expected in required_integrity.items():
            actual = integrity.get(key, route.get(key))
            if actual != expected:
                violations.append({"row": row_index, "path": "$.source_integrity", "reason": "source_integrity_mismatch", "field": key, "expected": expected, "actual": actual})
                counters["source_integrity_mismatch"] += 1
        if row.get("expected_json") is not None:
            violations.append({"row": row_index, "path": "$.expected_json", "reason": "expected_json_present"})
            counters["expected_json_present"] += 1
        if str(route.get("source_mode") or integrity.get("source_mode") or "") != "image_only_raster_moe":
            violations.append({"row": row_index, "path": "$.route_trace.source_mode", "reason": "invalid_source_mode", "actual": route.get("source_mode")})
            counters["invalid_source_mode"] += 1
        for path, key, value in _walk(row):
            if key in forbidden_fields:
                violations.append({"row": row_index, "path": f"{path}.{key}", "reason": "forbidden_field", "field": key})
                counters["forbidden_field"] += 1
            if key == "id" and _bad_id(value):
                violations.append({"row": row_index, "path": f"{path}.id", "reason": "forbidden_parser_like_id", "value": value})
                counters["forbidden_parser_like_id"] += 1
            if key in {"source", "proposal_source", "geometry_source", "candidate_geometry_source", "model_output_contract"}:
                text = str(value).lower()
                if any(fragment in text for fragment in forbidden_values):
                    violations.append({"row": row_index, "path": f"{path}.{key}", "reason": "forbidden_source_value", "value": value})
                    counters["forbidden_source_value"] += 1
            if isinstance(value, str):
                text = value.lower()
                if any(fragment in text for fragment in forbidden_values):
                    violations.append({"row": row_index, "path": f"{path}.{key}", "reason": "forbidden_value_fragment", "value": value[:240]})
                    counters["forbidden_value_fragment"] += 1

    passed = not violations
    return {
        "contract_version": contract.get("contract_version", "unknown"),
        "checked_rows": len(rows),
        "passed": passed,
        "violations": len(violations),
        "violation_counts": dict(counters),
        "sample_violations": violations[:200],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--contract", default=str(DEFAULT_CONTRACT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--expect-fail", action="store_true")
    parser.add_argument("--expect-pass", action="store_true")
    args = parser.parse_args()

    contract = load_json(args.contract)
    rows = load_jsonl(args.predictions)
    report = validate_rows(rows, contract)
    report["prediction_stream"] = str(args.predictions)
    report["expectation"] = "fail" if args.expect_fail else ("pass" if args.expect_pass else "none")
    write_json(args.report, report)

    if args.expect_fail:
        ok = not report["passed"]
    elif args.expect_pass:
        ok = bool(report["passed"])
    else:
        ok = bool(report["passed"])
    print(json.dumps(report, ensure_ascii=False, indent=2))
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
