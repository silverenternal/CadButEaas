#!/usr/bin/env python3
"""Audit the v18 high-recall detector candidate stream end to end."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
SCHEMA = ROOT / "configs/vlm/detector_output_schema_v18.json"

if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.vlm.candidate_contract import build_candidate_audit, integrity, normalize_candidate
from scripts.vlm.merge_detector_outputs_v18 import DEFAULT_INPUTS, load_gold, load_jsonl, write_json

EXPECTED_ROUTE = {
    "boundary": "wall_opening",
    "space": "room_space",
    "text": "text_dimension",
    "symbol": "symbol_fixture",
}


def load_raw_family(path: Path, family: str) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, Any]], dict[str, Any]]:
    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    page_meta: dict[str, dict[str, Any]] = {}
    audit: Counter[str] = Counter()
    route_mismatch: Counter[str] = Counter()
    provenance_missing = 0
    audit_trace_missing = 0
    candidate_type_counts: Counter[str] = Counter()
    raw_payload_key_counts: Counter[str] = Counter()
    confidence_values: list[float] = []

    for row in load_jsonl(path):
        if "candidate_stream" in row:
            row_id = str(row.get("id"))
            image = row.get("image")
            page_meta[row_id] = {
                "id": row_id,
                "image": image,
                "image_size": row.get("image_size") or [512, 512],
            }
            raw_items = list(row.get("candidate_stream") or [])
        else:
            row_id = str(row.get("row_id") or row.get("id", "").split("_text_")[0].split("_symbol_")[0])
            image = row.get("image")
            raw_items = [row]

        for raw in raw_items:
            if not isinstance(raw, dict):
                audit["non_dict_rows"] += 1
                continue
            if not raw.get("candidate_id") and not raw.get("id"):
                audit["missing_candidate_id"] += 1
            if not isinstance(raw.get("bbox"), list) or len(raw.get("bbox") or []) != 4:
                audit["missing_or_invalid_bbox"] += 1
            if raw.get("payload") is None:
                audit["missing_payload"] += 1
            normalized = normalize_candidate(raw, family, row_id, image=image)
            if normalized is None:
                audit["rejected_by_normalizer"] += 1
                continue

            by_row[row_id].append(normalized)
            confidence_values.append(float(normalized.get("confidence") or 0.0))
            candidate_type_counts[str(normalized.get("candidate_type") or "")] += 1
            for key in (normalized.get("payload") or {}).keys():
                raw_payload_key_counts[str(key)] += 1
            if not normalized.get("provenance"):
                provenance_missing += 1
            if not normalized.get("audit_trace"):
                audit_trace_missing += 1
            expected_route = EXPECTED_ROUTE.get(family)
            if expected_route and str(normalized.get("route") or "") != expected_route:
                route_mismatch["route_mismatch"] += 1

    family_audit = {
        "raw_row_rejections": dict(audit),
        "route_mismatch_count": int(route_mismatch["route_mismatch"]),
        "provenance_missing_count": provenance_missing,
        "audit_trace_missing_count": audit_trace_missing,
        "candidate_type_counts": dict(candidate_type_counts),
        "payload_key_counts": dict(raw_payload_key_counts),
        "confidence_summary": _summarize_values(confidence_values),
        "raw_candidate_count": sum(len(items) for items in by_row.values()),
    }
    return by_row, page_meta, family_audit


def _summarize_values(values: list[float]) -> dict[str, Any]:
    if not values:
        return {"count": 0, "min": None, "mean": None, "max": None}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "min": round(ordered[0], 6),
        "mean": round(sum(ordered) / len(ordered), 6),
        "max": round(ordered[-1], 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--schema", default=str(SCHEMA))
    parser.add_argument("--boundary-input", default=str(DEFAULT_INPUTS["boundary"]))
    parser.add_argument("--space-input", default=str(DEFAULT_INPUTS["space"]))
    parser.add_argument("--text-input", default=str(DEFAULT_INPUTS["text"]))
    parser.add_argument("--symbol-input", default=str(DEFAULT_INPUTS["symbol"]))
    parser.add_argument("--output", default=str(REPORT / "detector_candidate_audit_v18.json"))
    args = parser.parse_args()

    schema = json.loads(Path(args.schema).read_text(encoding="utf-8"))
    caps = {
        family: int((schema.get("families") or {}).get(family, {}).get("default_cap_per_page") or 0)
        for family in EXPECTED_ROUTE
    }

    input_paths = {
        "boundary": Path(args.boundary_input),
        "space": Path(args.space_input),
        "text": Path(args.text_input),
        "symbol": Path(args.symbol_input),
    }

    family_rows: dict[str, dict[str, list[dict[str, Any]]]] = {}
    page_meta: dict[str, dict[str, Any]] = {}
    family_input_audit: dict[str, dict[str, Any]] = {}
    for family, path in input_paths.items():
        rows, meta, audit = load_raw_family(path, family)
        family_rows[family] = rows
        page_meta.update({k: v for k, v in meta.items() if k not in page_meta})
        family_input_audit[family] = audit

    gold = load_gold()
    selected_rows = {
        family: {
            row_id: sorted(items, key=lambda item: item.get("confidence", 0.0), reverse=True)[: caps[family]]
            if caps[family]
            else list(items)
            for row_id, items in rows.items()
        }
        for family, rows in family_rows.items()
    }
    source_violations: list[dict[str, Any]] = []
    for family, rows in selected_rows.items():
        for row_id, candidates in rows.items():
            for cand in candidates:
                if cand.get("source_integrity") != integrity():
                    source_violations.append({"row_id": row_id, "candidate_id": cand.get("candidate_id"), "family": family})

    audit = build_candidate_audit(
        family_rows=family_rows,
        gold=gold,
        caps=caps,
        selected_rows=selected_rows,
        source_violations=source_violations,
    )
    audit.update(
        {
            "task": "IMG-MOE-V18-CANDIDATE-AUDIT",
            "schema": str(args.schema),
            "input_paths": {family: str(path) for family, path in input_paths.items()},
            "page_meta_count": len(page_meta),
            "family_input_audit": family_input_audit,
            "findings": build_findings(family_input_audit, audit),
        }
    )

    output = Path(args.output)
    write_json(output, audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def build_findings(family_input_audit: dict[str, dict[str, Any]], audit: dict[str, Any]) -> list[str]:
    findings: list[str] = []
    for family, stats in family_input_audit.items():
        raw_rejections = stats.get("raw_row_rejections") or {}
        rejected = int(raw_rejections.get("rejected_by_normalizer", 0) or 0)
        route_mismatch = int(stats.get("route_mismatch_count", 0) or 0)
        provenance_missing = int(stats.get("provenance_missing_count", 0) or 0)
        trace_missing = int(stats.get("audit_trace_missing_count", 0) or 0)
        raw_count = int(stats.get("raw_candidate_count", 0) or 0)
        recall_loss = ((audit.get("recall_loss_accounting") or {}).get(family) or {}).get("absolute_recall_loss", 0.0)
        findings.append(
            f"{family}: raw={raw_count}, rejected={rejected}, route_mismatch={route_mismatch}, provenance_missing={provenance_missing}, audit_trace_missing={trace_missing}, cap_recall_loss={recall_loss}"
        )
    return findings


if __name__ == "__main__":
    main()
