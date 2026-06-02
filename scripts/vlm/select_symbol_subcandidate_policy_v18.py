#!/usr/bin/env python3
"""Select a conservative symbol subcandidate gate from an augmented dataset.

This is an offline policy-selection diagnostic. It uses labels only to choose
and audit the gate. Candidate features used by the selected gate are available
at inference time in edge_features / candidate payloads.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_topology_relations_v18 import integrity, write_json
from nms_topology_relations_v18 import load_jsonl

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"

DEFAULT_BASELINE = REPORT / "contains_symbol_bipartite_dataset.jsonl"
DEFAULT_VARIANT = REPORT / "contains_symbol_bipartite_dataset_symbol_subcandidate_augmented.jsonl"
DEFAULT_OUTPUT = REPORT / "symbol_subcandidate_policy_sweep.json"
DEFAULT_MODEL = ROOT / "checkpoints/symbol_subcandidate_policy_v18/model.json"


def safe_float(value: Any) -> float:
    try:
        if value is None:
            return 0.0
        out = float(value)
        if math.isnan(out) or math.isinf(out):
            return 0.0
        return out
    except (TypeError, ValueError):
        return 0.0


def gold_key(row: dict[str, Any]) -> str | None:
    labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
    value = labels.get("gold_key")
    return str(value) if value else None


def is_subcandidate(row: dict[str, Any]) -> bool:
    return "_subcc_" in str(row.get("target_candidate_id") or "")


def feature(row: dict[str, Any], name: str) -> float:
    feats = row.get("edge_features") if isinstance(row.get("edge_features"), dict) else {}
    return safe_float(feats.get(name))


def recovered_keys(rows: list[dict[str, Any]]) -> set[str]:
    return {key for row in rows if (key := gold_key(row))}


def parent_candidate_id(target_id: str) -> str:
    return target_id.split("_subcc_", 1)[0] if "_subcc_" in target_id else target_id


def subcandidate_score(row: dict[str, Any]) -> float:
    """Inference-only heuristic score for selecting subcandidate relation edges."""
    density = feature(row, "target_local_dark_density")
    area_ratio = feature(row, "target_bbox_to_anchor_area_ratio")
    confidence = feature(row, "target_confidence")
    relation_conf = feature(row, "relation_confidence")
    margin_ratio = feature(row, "target_room_min_edge_distance_ratio")
    symbol_rank = feature(row, "edge_rank_within_symbol")
    room_degree = feature(row, "room_graph_degree")
    component_edges = feature(row, "component_edge_count")
    compact_area_bonus = max(0.0, 1.0 - abs(area_ratio - 0.18) / 0.30)
    rank_bonus = 1.0 / (1.0 + max(symbol_rank - 1.0, 0.0))
    crowd_penalty = min(1.0, math.log1p(max(room_degree, component_edges / 20.0)) / 7.0)
    return (
        0.28 * confidence
        + 0.20 * relation_conf
        + 0.18 * min(density, 1.0)
        + 0.16 * compact_area_bonus
        + 0.10 * min(max(margin_ratio, 0.0), 1.0)
        + 0.08 * rank_bonus
        - 0.10 * crowd_penalty
    )


def policy_grid() -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for score_threshold in [0.46, 0.52, 0.58, 0.64]:
        for max_per_parent in [1, 2]:
            for max_per_room in [8, 24, 64]:
                for min_density in [0.35, 0.50, 0.65]:
                    for max_area_ratio in [0.28, 0.45, 0.70]:
                        policies.append(
                            {
                                "score_threshold": score_threshold,
                                "max_edges_per_parent_candidate": max_per_parent,
                                "max_subcandidate_edges_per_room": max_per_room,
                                "min_target_local_dark_density": min_density,
                                "max_target_bbox_to_anchor_area_ratio": max_area_ratio,
                            }
                        )
    return policies


def subcandidate_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        if not is_subcandidate(row):
            continue
        records.append(
            {
                "row_id": str(row.get("row_id")),
                "target_candidate_id": str(row.get("target_candidate_id") or ""),
                "parent_candidate_id": parent_candidate_id(str(row.get("target_candidate_id") or "")),
                "room_instance_cluster_id": str(row.get("room_instance_cluster_id")),
                "gold_key": gold_key(row),
                "score": subcandidate_score(row),
                "target_local_dark_density": feature(row, "target_local_dark_density"),
                "target_bbox_to_anchor_area_ratio": feature(row, "target_bbox_to_anchor_area_ratio"),
            }
        )
    return records


def select_subcandidate_records(records: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if safe_float(row["target_local_dark_density"]) < safe_float(policy["min_target_local_dark_density"]):
            continue
        if safe_float(row["target_bbox_to_anchor_area_ratio"]) > safe_float(policy["max_target_bbox_to_anchor_area_ratio"]):
            continue
        score = safe_float(row["score"])
        if score < safe_float(policy["score_threshold"]):
            continue
        buckets[(str(row["row_id"]), str(row["parent_candidate_id"]))].append(row)

    selected: list[dict[str, Any]] = []
    room_counts: Counter[tuple[str, str]] = Counter()
    max_parent = int(policy["max_edges_per_parent_candidate"])
    max_room = int(policy["max_subcandidate_edges_per_room"])
    for (row_id, parent_id), bucket in buckets.items():
        ordered = sorted(bucket, key=lambda item: safe_float(item.get("score")), reverse=True)
        kept_for_parent = 0
        for row in ordered:
            room_id = str(row["room_instance_cluster_id"])
            if kept_for_parent >= max_parent:
                break
            if room_counts[(row_id, room_id)] >= max_room:
                continue
            selected.append(row)
            kept_for_parent += 1
            room_counts[(row_id, room_id)] += 1
    return selected


def evaluate_policy(
    baseline_keys: set[str],
    records: list[dict[str, Any]],
    policy: dict[str, Any],
) -> dict[str, Any]:
    selected_sub = select_subcandidate_records(records, policy)
    selected_sub_keys = {str(row["gold_key"]) for row in selected_sub if row.get("gold_key")}
    gained = selected_sub_keys - baseline_keys
    return {
        "policy": policy,
        "selected_subcandidate_edges": len(selected_sub),
        "selected_subcandidate_positive_edges": sum(1 for row in selected_sub if row.get("gold_key")),
        "selected_subcandidate_gold_keys": len(selected_sub_keys),
        "gained_gold_keys_vs_baseline": len(gained),
        "estimated_row_delta_vs_baseline": len(selected_sub),
        "estimated_recoverable_gold": len(baseline_keys | selected_sub_keys),
        "estimated_recoverable_gold_delta": len((baseline_keys | selected_sub_keys)) - len(baseline_keys),
        "estimated_added_rows_per_gained_gold": round(len(selected_sub) / max(len(gained), 1), 6),
        "subcandidate_edge_precision": round(sum(1 for row in selected_sub if row.get("gold_key")) / max(len(selected_sub), 1), 6),
        "source_integrity": integrity(),
    }


def choose_policy(results: list[dict[str, Any]]) -> dict[str, Any]:
    viable = [
        row for row in results
        if row["gained_gold_keys_vs_baseline"] >= 20
        and row["estimated_added_rows_per_gained_gold"] <= 250
    ]
    pool = viable or results
    return max(
        pool,
        key=lambda row: (
            row["estimated_recoverable_gold_delta"],
            -row["estimated_added_rows_per_gained_gold"],
            -row["selected_subcandidate_edges"],
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--variant", default=str(DEFAULT_VARIANT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--model-output", default=str(DEFAULT_MODEL))
    args = parser.parse_args()

    baseline_rows = load_jsonl(Path(args.baseline))
    variant_rows = load_jsonl(Path(args.variant))
    baseline_keys = recovered_keys(baseline_rows)
    records = subcandidate_records(variant_rows)
    results = [evaluate_policy(baseline_keys, records, policy) for policy in policy_grid()]
    results.sort(key=lambda row: (-row["estimated_recoverable_gold_delta"], row["estimated_added_rows_per_gained_gold"], row["selected_subcandidate_edges"]))
    selected = choose_policy(results)
    report = {
        "task": "IMG-MOE-V18-REBUILD-001.step_g2_select_symbol_subcandidate_policy",
        "baseline": str(args.baseline),
        "variant": str(args.variant),
        "baseline_rows": len(baseline_rows),
        "variant_rows": len(variant_rows),
        "baseline_recoverable_gold": len(baseline_keys),
        "variant_recoverable_gold": len(recovered_keys(variant_rows)),
        "subcandidate_record_count": len(records),
        "selected_policy": selected,
        "top_policies": results[:50],
        "policy_count": len(results),
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_policy_selection_only": True,
        "gold_used_for_inference": False,
    }
    write_json(Path(args.output), report)
    model = {
        "model_type": "symbol_subcandidate_heuristic_gate_v18",
        "policy": selected["policy"],
        "selection_metrics": selected,
        "features_used": [
            "target_local_dark_density",
            "target_bbox_to_anchor_area_ratio",
            "target_confidence",
            "relation_confidence",
            "target_room_min_edge_distance_ratio",
            "edge_rank_within_symbol",
            "room_graph_degree",
            "component_edge_count",
        ],
        "source_integrity": integrity(),
        "gold_used_for_inference": False,
    }
    write_json(Path(args.model_output), model)
    print(json.dumps(report["selected_policy"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
