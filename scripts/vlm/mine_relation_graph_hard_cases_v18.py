#!/usr/bin/env python3
"""Mine hard cases for the v18 relation-graph reconstruction policy."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from relation_graph_reconstruction_v18 import DEFAULT_DATASET, DEFAULT_MODEL, RELATIONS, load_jsonl, safe_float, score_rows, write_json, write_jsonl


DEFAULT_OUTPUT = Path("reports/vlm/relation_graph_hard_cases_v18.jsonl")
DEFAULT_AUDIT = Path("reports/vlm/relation_graph_hard_cases_v18_audit.json")


def load_model(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def quantile(values: list[float], q: float, fallback: float) -> float:
    if not values:
        return fallback
    return float(np.quantile(np.asarray(values, dtype=np.float32), q))


def compact_features(features: dict[str, Any]) -> dict[str, float]:
    keep = [
        "relation_confidence",
        "detector_confidence_product",
        "duplicate_relation_count",
        "component_edge_count",
        "component_node_count",
        "component_bridge_count",
        "source_graph_degree",
        "target_graph_degree",
        "pair_support_ratio",
        "target_objectness_score",
        "target_local_dark_density",
        "bbox_distance",
        "side_overlap_ratio",
        "orientation_compatible",
    ]
    return {name: round(safe_float(features.get(name)), 6) for name in keep if name in features}


def hard_case_record(row: dict[str, Any], case_type: str, reason: str, score_threshold: float | None = None) -> dict[str, Any]:
    labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
    return {
        "case_type": case_type,
        "reason": reason,
        "row_id": row.get("row_id"),
        "relation_id": row.get("relation_id"),
        "relation": row.get("relation"),
        "split": row.get("split"),
        "relation_score": round(safe_float(row.get("relation_score")), 6),
        "score_threshold": round(score_threshold, 6) if score_threshold is not None else None,
        "label_positive": bool(labels.get("label_positive")),
        "graph_role": row.get("graph_role") or labels.get("graph_role"),
        "gold_key": labels.get("gold_key"),
        "source_candidate_id": row.get("source_candidate_id"),
        "target_candidate_id": row.get("target_candidate_id"),
        "source_cluster_id": row.get("source_cluster_id"),
        "target_cluster_id": row.get("target_cluster_id"),
        "component_id": row.get("component_id"),
        "duplicate_relation_count": row.get("duplicate_relation_count"),
        "edge_features": compact_features(row.get("edge_features") if isinstance(row.get("edge_features"), dict) else {}),
        "source_integrity": row.get("source_integrity"),
        "gold_loaded_after_inference_for_training_only": True,
        "gold_used_for_inference": False,
    }


def stable_split(rows: list[dict[str, Any]]) -> None:
    row_ids = sorted({str(row.get("row_id") or "") for row in rows})
    train_cut = max(1, int(len(row_ids) * 0.70))
    dev_cut = min(len(row_ids), max(train_cut + 1, int(len(row_ids) * 0.80))) if len(row_ids) >= 3 else train_cut
    row_split: dict[str, str] = {}
    for index, row_id in enumerate(row_ids):
        if index < train_cut:
            row_split[row_id] = "train"
        elif index < dev_cut:
            row_split[row_id] = "dev"
        else:
            row_split[row_id] = "test"
    for row in rows:
        row["split"] = row_split.get(str(row.get("row_id") or ""), "unknown")


def mine(rows: list[dict[str, Any]], max_per_bucket: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_relation: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_relation[str(row.get("relation"))].append(row)

    hard_cases: list[dict[str, Any]] = []
    thresholds: dict[str, Any] = {}
    for relation in RELATIONS:
        bucket = by_relation.get(relation, [])
        positives = [row for row in bucket if row.get("labels", {}).get("label_positive")]
        negatives = [row for row in bucket if not row.get("labels", {}).get("label_positive")]
        pos_scores = [safe_float(row.get("relation_score")) for row in positives]
        neg_scores = [safe_float(row.get("relation_score")) for row in negatives]
        low_pos_threshold = quantile(pos_scores, 0.10, 0.0)
        high_neg_threshold = quantile(neg_scores, 0.99, 1.0)
        thresholds[relation] = {
            "positive_p10_score": round(low_pos_threshold, 6),
            "negative_p99_score": round(high_neg_threshold, 6),
            "positive_count": len(positives),
            "negative_count": len(negatives),
        }

        low_positives = sorted(
            [row for row in positives if safe_float(row.get("relation_score")) <= low_pos_threshold],
            key=lambda row: (safe_float(row.get("relation_score")), str(row.get("row_id")), str(row.get("relation_id"))),
        )[:max_per_bucket]
        for row in low_positives:
            hard_cases.append(hard_case_record(row, "low_score_positive", "positive edge scored in bottom decile for its relation family", low_pos_threshold))

        high_negatives = sorted(
            [row for row in negatives if safe_float(row.get("relation_score")) >= high_neg_threshold],
            key=lambda row: (-safe_float(row.get("relation_score")), str(row.get("row_id")), str(row.get("relation_id"))),
        )[:max_per_bucket]
        for row in high_negatives:
            hard_cases.append(hard_case_record(row, "high_score_negative", "negative edge scored in top one percent for its relation family", high_neg_threshold))

        bridge_lows = sorted(
            [
                row
                for row in positives
                if str(row.get("graph_role") or row.get("labels", {}).get("graph_role")) == "bridge"
                and safe_float(row.get("relation_score")) <= quantile(pos_scores, 0.25, 0.0)
            ],
            key=lambda row: (safe_float(row.get("relation_score")), -safe_float((row.get("edge_features") or {}).get("component_edge_count"))),
        )[:max_per_bucket]
        for row in bridge_lows:
            hard_cases.append(hard_case_record(row, "low_score_bridge_positive", "bridge positive is vulnerable to relation compression", thresholds[relation]["positive_p10_score"]))

    gold_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
        gold_key = labels.get("gold_key")
        if labels.get("label_positive") and gold_key:
            gold_groups[str(gold_key)].append(row)
    duplicate_groups = {key: value for key, value in gold_groups.items() if len(value) > 1}
    for gold_key, group in sorted(duplicate_groups.items(), key=lambda item: (-len(item[1]), item[0]))[: max_per_bucket * 4]:
        ordered = sorted(group, key=lambda row: safe_float(row.get("relation_score")), reverse=True)
        for row in ordered[1:3]:
            hard_cases.append(hard_case_record(row, "duplicate_positive_support", f"secondary support edge for duplicate positive group size={len(group)}"))

    page_relation_counts: Counter[str] = Counter(str(row.get("row_id")) for row in rows)
    component_edges = [safe_float((row.get("edge_features") or {}).get("component_edge_count")) for row in rows]
    component_p99 = quantile(component_edges, 0.99, 0.0)
    overloaded = sorted(
        [row for row in rows if safe_float((row.get("edge_features") or {}).get("component_edge_count")) >= component_p99],
        key=lambda row: (-safe_float((row.get("edge_features") or {}).get("component_edge_count")), str(row.get("row_id")), str(row.get("relation_id"))),
    )[:max_per_bucket]
    for row in overloaded:
        hard_cases.append(hard_case_record(row, "component_overflow_edge", "edge belongs to a top-one-percent dense relation component", component_p99))

    seen: set[tuple[str, str]] = set()
    deduped: list[dict[str, Any]] = []
    for case in hard_cases:
        key = (str(case.get("case_type")), str(case.get("relation_id")))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(case)

    audit = {
        "task": "IMG-MOE-V18-REBUILD-005",
        "rows": len({row.get("row_id") for row in rows}),
        "edge_rows": len(rows),
        "hard_cases": len(deduped),
        "case_counts": dict(Counter(str(row.get("case_type")) for row in deduped)),
        "relation_counts": dict(Counter(str(row.get("relation")) for row in deduped)),
        "split_counts": dict(Counter(str(row.get("split")) for row in rows)),
        "page_relation_counts_top10": page_relation_counts.most_common(10),
        "score_thresholds": thresholds,
        "component_edge_p99": round(component_p99, 6),
        "duplicate_positive_groups": len(duplicate_groups),
        "source_integrity": rows[0].get("source_integrity") if rows else None,
        "gold_loaded_after_inference_for_training_only": True,
        "gold_used_for_inference": False,
    }
    return deduped, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--max-per-bucket", type=int, default=256)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.dataset), limit=args.limit)
    stable_split(rows)
    model = load_model(Path(args.model))
    scored = score_rows(rows, model)
    hard_cases, audit = mine(scored, max(1, args.max_per_bucket))
    write_jsonl(Path(args.output), hard_cases)
    write_json(Path(args.audit), audit)
    print(json.dumps({"hard_cases": len(hard_cases), "output": args.output, "audit": args.audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
