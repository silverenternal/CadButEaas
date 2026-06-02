#!/usr/bin/env python3
"""Audit bounded_by wall-segment cap failures for relation graph v18."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from relation_graph_reconstruction_v18 import (
    bounded_by_room_side,
    bounded_by_wall_segment_key,
    bbox_from_candidate_id,
    load_jsonl,
    safe_float,
    score_rows,
    select_rows,
    write_json,
)
from train_relation_graph_listwise_policy_v18 import (
    DEFAULT_HARD_CASES,
    DEFAULT_LISTWISE_MODEL,
    hard_case_lookup,
    mark_training_targets,
    representative_eval,
)
from train_relation_graph_policy_v18 import DEFAULT_DATASET, load_dataset


DEFAULT_OUTPUT = Path("reports/vlm/bounded_by_wall_segment_cap_failure_v18.json")
DEFAULT_EXAMPLES = Path("reports/vlm/bounded_by_wall_segment_cap_failure_examples_v18.jsonl")


SAFE_BOUNDED_POLICY = {
    "threshold": 0.01,
    "max_per_source": 999999,
    "max_per_target": 999999,
    "max_per_component": 999999,
    "max_per_pair": 999999,
    "max_pair_tail_slots": 0,
    "max_pair_alignment_slots": 0,
    "max_per_source_side": 48,
    "max_per_source_side_segment": 999999,
    "max_wall_segment_tail_slots": 0,
    "wall_segment_bin_size": 16,
    "listwise_policy": True,
}


SEGMENT_DIAGNOSTIC_POLICY = {
    "threshold": 0.01,
    "max_per_source": 999999,
    "max_per_target": 999999,
    "max_per_component": 999999,
    "max_per_pair": 999999,
    "max_pair_tail_slots": 0,
    "max_pair_alignment_slots": 0,
    "max_per_source_side": 48,
    "max_per_source_side_segment": 12,
    "max_wall_segment_tail_slots": 0,
    "wall_segment_bin_size": 16,
    "listwise_policy": True,
}


def load_model(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def bucket_int(value: float, cuts: list[int]) -> str:
    for cut in cuts:
        if value <= cut:
            return f"<= {cut}"
    return f"> {cuts[-1]}"


def relation_gold_key(row: dict[str, Any]) -> str:
    labels = row.get("listwise_labels") if isinstance(row.get("listwise_labels"), dict) else {}
    return str(labels.get("gold_key") or "")


def compact_features(row: dict[str, Any]) -> dict[str, Any]:
    features = row.get("listwise_features") if isinstance(row.get("listwise_features"), dict) else {}
    keys = [
        "base_relation_score",
        "source_score_rank",
        "bounded_wall_segment_score_rank",
        "bounded_wall_segment_tail_rank",
        "bounded_wall_segment_score_percentile",
        "bounded_wall_segment_tail_percentile",
        "source_cluster_edge_count",
        "bounded_wall_segment_edge_count",
        "bounded_wall_side_alignment_score",
        "bounded_distance_score",
        "bounded_axis_overlap_score",
        "bounded_target_area_ratio",
        "detector_confidence_product",
        "component_bridge_count",
        "duplicate_relation_count",
    ]
    return {key: features.get(key) for key in keys if key in features}


def missed_row_record(row: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    features = row.get("listwise_features") if isinstance(row.get("listwise_features"), dict) else {}
    bbox = bbox_from_candidate_id(row.get("target_candidate_id"))
    side = bounded_by_room_side(row)
    segment_key = bounded_by_wall_segment_key(row, int(policy.get("wall_segment_bin_size") or 16))
    segment_rank = safe_float(features.get("bounded_wall_segment_score_rank"))
    segment_tail_rank = safe_float(features.get("bounded_wall_segment_tail_rank"))
    segment_count = safe_float(features.get("bounded_wall_segment_edge_count"))
    labels = row.get("listwise_labels") if isinstance(row.get("listwise_labels"), dict) else {}
    return {
        "row_id": row.get("row_id"),
        "relation_id": row.get("relation_id"),
        "gold_key": relation_gold_key(row) or None,
        "gold_representative": bool(labels.get("gold_representative")),
        "bridge_positive": bool(labels.get("bridge_positive")),
        "duplicate_support": bool(labels.get("duplicate_support")),
        "source_candidate_id": row.get("source_candidate_id"),
        "target_candidate_id": row.get("target_candidate_id"),
        "source_cluster_id": row.get("source_cluster_id"),
        "target_cluster_id": row.get("target_cluster_id"),
        "room_side": side,
        "wall_segment_key": segment_key,
        "target_bbox": bbox,
        "relation_score": safe_float(row.get("relation_score")),
        "base_relation_score": row.get("base_relation_score"),
        "segment_score_rank": segment_rank,
        "segment_tail_rank": segment_tail_rank,
        "segment_edge_count": segment_count,
        "drop_reason": "segment_cap_overflow"
        if segment_rank > safe_float(policy.get("max_per_source_side_segment"))
        else "selected_order_interaction_or_threshold",
        "listwise_features": compact_features(row),
    }


def collect_missing_gold_keys(rows: list[dict[str, Any]], selected: list[dict[str, Any]]) -> dict[str, list[str]]:
    selected_ids = {str(row.get("relation_id")) for row in selected}
    gold_by_key: dict[str, set[str]] = defaultdict(set)
    matched_by_key: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        labels = row.get("listwise_labels") if isinstance(row.get("listwise_labels"), dict) else {}
        gold_key = relation_gold_key(row)
        if not gold_key or not labels.get("gold_representative"):
            continue
        gold_by_key[gold_key].add(str(row.get("relation_id")))
        if str(row.get("relation_id")) in selected_ids:
            matched_by_key[gold_key].add(str(row.get("relation_id")))
    return {
        "gold_keys": sorted(gold_by_key),
        "matched_gold_keys": sorted(key for key in gold_by_key if matched_by_key.get(key)),
        "missed_gold_keys": sorted(key for key in gold_by_key if not matched_by_key.get(key)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model", default=str(DEFAULT_LISTWISE_MODEL))
    parser.add_argument("--hard-cases", default=str(DEFAULT_HARD_CASES))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--examples-output", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-examples", type=int, default=200)
    args = parser.parse_args()

    model = load_model(Path(args.model))
    rows = load_dataset(Path(args.dataset), limit=args.limit)
    scored = score_rows(rows, model)
    marked = mark_training_targets(scored, hard_case_lookup(Path(args.hard_cases)))
    bounded = [row for row in marked if row.get("relation") == "bounded_by"]

    safe_selected = select_rows(bounded, {"bounded_by": SAFE_BOUNDED_POLICY})
    segment_selected = select_rows(bounded, {"bounded_by": SEGMENT_DIAGNOSTIC_POLICY})
    safe_ids = {str(row.get("relation_id")) for row in safe_selected}
    segment_ids = {str(row.get("relation_id")) for row in segment_selected}
    missed = [row for row in bounded if str(row.get("relation_id")) in safe_ids and str(row.get("relation_id")) not in segment_ids]
    missed_records = [missed_row_record(row, SEGMENT_DIAGNOSTIC_POLICY) for row in missed]

    critical_records = [
        row
        for row in missed_records
        if row.get("gold_representative") or row.get("bridge_positive")
    ]
    examples = sorted(
        critical_records or missed_records,
        key=lambda row: (
            not bool(row.get("gold_representative")),
            not bool(row.get("bridge_positive")),
            -safe_float(row.get("segment_score_rank")),
            str(row.get("row_id")),
        ),
    )[: max(0, args.max_examples)]

    side_counts = Counter(str(row.get("room_side")) for row in critical_records)
    segment_rank_counts = Counter(bucket_int(safe_float(row.get("segment_score_rank")), [12, 16, 24, 32, 48, 96]) for row in critical_records)
    segment_tail_counts = Counter(bucket_int(safe_float(row.get("segment_tail_rank")), [1, 2, 4, 8, 12, 16, 24]) for row in critical_records)
    segment_edge_counts = Counter(bucket_int(safe_float(row.get("segment_edge_count")), [12, 16, 24, 32, 48, 96, 192]) for row in critical_records)
    drop_reasons = Counter(str(row.get("drop_reason")) for row in missed_records)

    safe_rep = representative_eval(bounded, safe_selected).get("bounded_by") or {}
    segment_rep = representative_eval(bounded, segment_selected).get("bounded_by") or {}
    segment_gold = collect_missing_gold_keys(bounded, segment_selected)
    source_integrity = marked[0].get("source_integrity") if marked else None
    summary = {
        "task": "IMG-MOE-V18-REBUILD-005",
        "audit": "bounded_by_wall_segment_cap_failure_v18",
        "dataset": args.dataset,
        "model": args.model,
        "hard_cases": args.hard_cases,
        "bounded_rows": len(bounded),
        "safe_policy": SAFE_BOUNDED_POLICY,
        "segment_diagnostic_policy": SEGMENT_DIAGNOSTIC_POLICY,
        "safe_selected": len(safe_selected),
        "segment_selected": len(segment_selected),
        "safe_feature_reduction": round(1.0 - len(safe_selected) / max(len(bounded), 1), 6),
        "segment_feature_reduction": round(1.0 - len(segment_selected) / max(len(bounded), 1), 6),
        "additional_feature_reduction_vs_safe": round((len(safe_selected) - len(segment_selected)) / max(len(bounded), 1), 6),
        "safe_representative_eval": safe_rep,
        "segment_representative_eval": segment_rep,
        "segment_missing_gold_key_count": len(segment_gold["missed_gold_keys"]),
        "segment_missing_gold_keys_sample": segment_gold["missed_gold_keys"][:50],
        "missed_safe_selected_edges": len(missed_records),
        "missed_critical_edges": len(critical_records),
        "missed_gold_representatives": sum(1 for row in critical_records if row.get("gold_representative")),
        "missed_bridge_positives": sum(1 for row in critical_records if row.get("bridge_positive")),
        "critical_by_room_side": dict(sorted(side_counts.items())),
        "critical_by_segment_score_rank": dict(sorted(segment_rank_counts.items())),
        "critical_by_segment_tail_rank": dict(sorted(segment_tail_counts.items())),
        "critical_by_segment_edge_count": dict(sorted(segment_edge_counts.items())),
        "missed_drop_reasons": dict(sorted(drop_reasons.items())),
        "examples_output": args.examples_output,
        "examples_written": len(examples),
        "diagnosis": (
            "The best hard wall-segment cap mainly fails by dropping recall-critical bounded_by representatives "
            "inside crowded room-side segment buckets. The next policy should reserve predicted wall-support "
            "representatives before segment collapse instead of widening the hard cap."
        ),
        "next_policy_requirements": [
            "Predict support-critical bounded_by edges with inference-available features before applying segment caps.",
            "Reserve at least one representative per predicted room-side wall support set before duplicate collapse.",
            "Keep locked no-regression gates for original recall, gold_key recall, and bridge recall at 1.0.",
            "Emit per-edge drop reasons that distinguish segment_overflow from support_reserved and duplicate_support_pruned.",
        ],
        "source_integrity": source_integrity,
        "gold_loaded_after_inference_for_audit_only": True,
        "gold_used_for_inference": False,
    }
    write_json(Path(args.output), summary)
    write_jsonl(Path(args.examples_output), examples)
    print(json.dumps({"output": args.output, "examples_output": args.examples_output, "missed_critical_edges": len(critical_records)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
