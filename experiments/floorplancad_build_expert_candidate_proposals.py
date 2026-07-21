#!/usr/bin/env python3
"""Convert cached expert primitive-set outputs into GT-free candidate proposals.

The training model consumes ``candidate_proposals`` as runtime-safe query priors:
each candidate carries primitive ids plus descriptor features computed only from
the current primitive cache. Audited expert files may contain GT fields such as
``gt_match`` for reporting; this adapter drops those fields and never reads them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from experiments.floorplancad_topology_component_proposals import (
    DESCRIPTOR_NAMES,
    assert_no_gt_leak,
    runtime_rows,
)


DEFAULT_CACHE = Path("reports/vlm/floorplancad_line_json_primitive_cache_windowed_2048_s1536_v3_r2/train_windowed_primitive_cache.jsonl")
DEFAULT_EXPERT = Path("reports/vlm/floorplancad_vecformer_moe_adapter/vecformer_test_expert_predictions.jsonl")
DEFAULT_OUTPUT = Path("reports/vlm/floorplancad_expert_candidate_proposals/train_expert_candidates.jsonl")

AUDIT_ONLY_FIELDS = {
    "gt_match",
    "match",
    "target",
    "targets",
    "semantic_id",
    "instance_id",
    "page_instance_id",
    "y_true",
    "ground_truth",
}


def iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                row = json.loads(line)
                if isinstance(row, dict):
                    yield row


def parse_int(value: Any, default: int = -1) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def parse_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def page_key(record_id: Any) -> str:
    return str(record_id).split("::", 1)[0]


def safe_float(values: list[Any], index: int, default: float = 0.0) -> float:
    try:
        return float(values[index])
    except (IndexError, TypeError, ValueError):
        return default


def endpoint_key(x: float, y: float, radius: float = 0.0025) -> tuple[int, int]:
    scale = max(float(radius), 1e-8)
    return (int(round(float(x) / scale)), int(round(float(y) / scale)))


def precompute_runtime_descriptors(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    descriptors = []
    for row in rows:
        features = row["features"]
        x1 = safe_float(features, 0)
        y1 = safe_float(features, 1)
        x2 = safe_float(features, 2)
        y2 = safe_float(features, 3)
        cx = safe_float(features, 4)
        cy = safe_float(features, 5)
        descriptors.append(
            {
                "primitive_id": int(row["primitive_id"]),
                "x_min": min(x1, x2),
                "x_max": max(x1, x2),
                "y_min": min(y1, y2),
                "y_max": max(y1, y2),
                "cx": cx,
                "cy": cy,
                "length": safe_float(features, 6),
                "orientation": safe_float(features, 8),
                "horizontal": safe_float(features, 9),
                "vertical": safe_float(features, 10),
                "stroke": safe_float(features, 11),
                "layer": safe_float(features, 13),
                "closed": safe_float(features, 31),
                "same_layer": safe_float(features, 36),
                "start_key": endpoint_key(x1, y1),
                "end_key": endpoint_key(x2, y2),
            }
        )
    return descriptors


def describe_candidate_fast(precomputed: list[dict[str, Any]], indices: list[int]) -> list[float]:
    unique_indices = sorted(set(indices))
    selected = [precomputed[index] for index in unique_indices]
    count = max(len(selected), 1)
    bbox_x1 = min(item["x_min"] for item in selected)
    bbox_x2 = max(item["x_max"] for item in selected)
    bbox_y1 = min(item["y_min"] for item in selected)
    bbox_y2 = max(item["y_max"] for item in selected)
    bbox_area = max((bbox_x2 - bbox_x1) * (bbox_y2 - bbox_y1), 0.0)
    endpoint_counts: Counter[tuple[int, int]] = Counter()
    for item in selected:
        endpoint_counts[item["start_key"]] += 1
        endpoint_counts[item["end_key"]] += 1
    degrees = list(endpoint_counts.values()) or [0]
    length_sum = sum(float(item["length"]) for item in selected)
    return [
        float(count),
        bbox_x1,
        bbox_y1,
        bbox_x2,
        bbox_y2,
        sum(float(item["cx"]) for item in selected) / count,
        sum(float(item["cy"]) for item in selected) / count,
        bbox_area,
        length_sum / count,
        length_sum,
        sum(float(item["orientation"]) for item in selected) / count,
        sum(float(item["horizontal"]) for item in selected) / count,
        sum(float(item["vertical"]) for item in selected) / count,
        sum(float(item["stroke"]) for item in selected) / count,
        sum(float(item["layer"]) for item in selected) / count,
        sum(float(item["same_layer"]) for item in selected) / count,
        sum(float(item["closed"]) for item in selected) / count,
        sum(degrees) / len(degrees),
        float(max(degrees)),
        float(count) / max(bbox_area, 1e-6),
    ]


def expert_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    for key in ("pred_instances", "teacher_proposals", "candidates"):
        value = row.get(key)
        if isinstance(value, list):
            return [item for item in value if isinstance(item, dict)]
    return []


def has_audit_only_field(value: Any) -> bool:
    if isinstance(value, dict):
        return any(key in AUDIT_ONLY_FIELDS or has_audit_only_field(child) for key, child in value.items())
    if isinstance(value, list):
        return any(has_audit_only_field(child) for child in value)
    return False


def load_expert_primitive_sets(path: Path, *, max_per_record: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    by_record: dict[str, list[dict[str, Any]]] = {}
    counters: Counter[str] = Counter()
    for row in iter_jsonl(path):
        record_id = page_key(row.get("record_id"))
        items = []
        for item in expert_items(row):
            primitive_ids = sorted({parse_int(value) for value in item.get("primitive_ids", [])})
            primitive_ids = [value for value in primitive_ids if value >= 0]
            if not primitive_ids:
                counters["empty_primitive_sets"] += 1
                continue
            if has_audit_only_field(item):
                counters["items_with_dropped_audit_fields"] += 1
            items.append(
                {
                    "primitive_ids": primitive_ids,
                    "score": parse_float(item.get("score"), parse_float(item.get("confidence"), 1.0)),
                    "label": parse_int(item.get("label"), -1),
                    "proposal_source": str(item.get("proposal_source") or item.get("source_expert") or row.get("source_prediction") or "expert_primitive_set"),
                    "expert_owner": str(item.get("expert_owner") or item.get("moe_route") or item.get("source_expert") or "reused_moe_expert"),
                }
            )
            counters["primitive_sets"] += 1
            if len(items) >= int(max_per_record):
                break
        if items:
            by_record[record_id] = items
            counters["records_with_expert_sets"] += 1
        counters["records"] += 1
    return by_record, dict(counters)


def build_record_candidates(
    record: dict[str, Any],
    expert_sets: list[dict[str, Any]],
    *,
    max_candidates_per_record: int,
    min_intersection_primitives: int,
) -> dict[str, Any]:
    rows = runtime_rows(record)
    precomputed = precompute_runtime_descriptors(rows)
    local_by_id = {int(row["primitive_id"]): index for index, row in enumerate(rows)}
    candidates = []
    seen: set[tuple[int, ...]] = set()
    for item in sorted(expert_sets, key=lambda value: (-parse_float(value.get("score"), 0.0), -len(value.get("primitive_ids") or []))):
        local_indices = [local_by_id[primitive_id] for primitive_id in item["primitive_ids"] if primitive_id in local_by_id]
        primitive_ids = tuple(sorted({int(rows[index]["primitive_id"]) for index in local_indices}))
        if len(primitive_ids) < int(min_intersection_primitives) or primitive_ids in seen:
            continue
        seen.add(primitive_ids)
        descriptor = describe_candidate_fast(precomputed, local_indices)
        candidates.append(
            {
                "candidate_id": f"{record.get('record_id', 'record')}::expert::{len(candidates):04d}",
                "record_id": record.get("record_id"),
                "original_record_id": record.get("original_record_id", page_key(record.get("record_id"))),
                "primitive_ids": list(primitive_ids),
                "candidate_features": descriptor,
                "candidate_feature_names": list(DESCRIPTOR_NAMES),
                "proposal_source": item["proposal_source"],
                "expert_owner": item["expert_owner"],
                "runtime_allowed": True,
                "dropped_audit_fields": sorted(AUDIT_ONLY_FIELDS),
                "expert_prior": {
                    "score": item["score"],
                    "predicted_label": item["label"],
                    "used_as": "query_initialization_and_optional_soft_mask_prior",
                },
            }
        )
        if len(candidates) >= int(max_candidates_per_record):
            break
    payload = {
        "schema_version": "floorplancad_expert_candidate_proposals_v1",
        "record_id": record.get("record_id"),
        "original_record_id": record.get("original_record_id", page_key(record.get("record_id"))),
        "candidate_count": len(candidates),
        "candidates": candidates,
        "runtime_contract": {
            "gt_free": True,
            "audit_fields_dropped": sorted(AUDIT_ONLY_FIELDS),
            "feature_source": "primitive_rows.features plus expert-predicted primitive_ids only",
            "matching_policy": "soft_prior_only_not_hungarian_constraint",
        },
    }
    assert_no_gt_leak(payload)
    return payload


def build_candidate_file(
    cache_path: Path,
    expert_path: Path,
    output_path: Path,
    *,
    limit_records: int,
    max_candidates_per_record: int,
    min_intersection_primitives: int,
) -> dict[str, Any]:
    expert_by_record, expert_report = load_expert_primitive_sets(
        expert_path,
        max_per_record=max_candidates_per_record * 4,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    counters: Counter[str] = Counter()
    with output_path.open("w", encoding="utf-8") as handle:
        for index, record in enumerate(iter_jsonl(cache_path), start=1):
            if limit_records and index > limit_records:
                break
            keys = [
                str(record.get("record_id")),
                str(record.get("original_record_id") or ""),
                page_key(record.get("record_id")),
            ]
            source_sets = []
            for key in dict.fromkeys(key for key in keys if key):
                source_sets = expert_by_record.get(key, [])
                if source_sets:
                    break
            payload = build_record_candidates(
                record,
                source_sets,
                max_candidates_per_record=max_candidates_per_record,
                min_intersection_primitives=min_intersection_primitives,
            )
            handle.write(json.dumps(payload, ensure_ascii=False, separators=(",", ":")) + "\n")
            counters["records"] += 1
            counters["records_with_candidates"] += int(payload["candidate_count"] > 0)
            counters["candidates"] += int(payload["candidate_count"])
    return {
        "schema_version": "floorplancad_expert_candidate_proposals_manifest_v1",
        "status": "ready" if counters["candidates"] > 0 else "blocked_no_candidates",
        "cache": str(cache_path),
        "expert_source": str(expert_path),
        "output": str(output_path),
        "candidate_feature_names": list(DESCRIPTOR_NAMES),
        "candidate_feature_dim": len(DESCRIPTOR_NAMES),
        "expert_source_report": expert_report,
        "counters": dict(counters),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--expert", type=Path, default=DEFAULT_EXPERT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--manifest", type=Path, default=None)
    parser.add_argument("--limit-records", type=int, default=0)
    parser.add_argument("--max-candidates-per-record", type=int, default=256)
    parser.add_argument("--min-intersection-primitives", type=int, default=1)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest = build_candidate_file(
        args.cache,
        args.expert,
        args.output,
        limit_records=args.limit_records,
        max_candidates_per_record=args.max_candidates_per_record,
        min_intersection_primitives=args.min_intersection_primitives,
    )
    manifest_path = args.manifest or args.output.with_suffix(args.output.suffix + ".manifest.json")
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": manifest["status"], "output": str(args.output), "manifest": str(manifest_path)}, ensure_ascii=False))
    return 0 if manifest["status"] == "ready" else 2


if __name__ == "__main__":
    raise SystemExit(main())
