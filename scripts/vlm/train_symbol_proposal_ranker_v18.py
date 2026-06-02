#!/usr/bin/env python3
"""Select a key-level gate over symbol subcandidate and center-anchor proposals.

This is an offline policy-selection diagnostic. Gold labels are used only for
choosing and auditing the policy. The exported policy uses only inference-time
edge features and candidate ids/provenance.
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
DEFAULT_VARIANT = REPORT / "contains_symbol_bipartite_dataset_symbol_proposal_combined.jsonl"
DEFAULT_OUTPUT = REPORT / "symbol_proposal_ranker_v18_eval.json"
DEFAULT_MODEL = ROOT / "checkpoints/symbol_proposal_ranker_v18/model.json"


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


def recovered_keys(rows: list[dict[str, Any]]) -> set[str]:
    return {key for row in rows if (key := gold_key(row))}


def proposal_kind(row: dict[str, Any]) -> str:
    target_id = str(row.get("target_candidate_id") or "")
    if "_subcc_" in target_id:
        return "connected_component"
    if "_center_anchor_" in target_id:
        return "center_anchor"
    feats = row.get("edge_features") if isinstance(row.get("edge_features"), dict) else {}
    if safe_float(feats.get("target_kind_other")) and "center_anchor" in target_id:
        return "center_anchor"
    return "baseline"


def is_new_symbol_proposal(row: dict[str, Any]) -> bool:
    return proposal_kind(row) in {"connected_component", "center_anchor"}


def parent_id(target_id: str) -> str:
    if "_subcc_" in target_id:
        return target_id.split("_subcc_", 1)[0]
    if "_center_anchor_" in target_id:
        return target_id.split("_center_anchor_", 1)[0]
    return target_id


def feature(row: dict[str, Any], name: str) -> float:
    feats = row.get("edge_features") if isinstance(row.get("edge_features"), dict) else {}
    return safe_float(feats.get(name))


def proposal_score(row: dict[str, Any], weights: dict[str, float]) -> float:
    kind = proposal_kind(row)
    density = min(max(feature(row, "target_local_dark_density"), 0.0), 1.0)
    area_ratio = feature(row, "target_bbox_to_anchor_area_ratio")
    confidence = feature(row, "target_confidence")
    relation_conf = feature(row, "relation_confidence")
    edge_margin = min(max(feature(row, "target_room_min_edge_distance_ratio"), 0.0), 1.0)
    symbol_rank = feature(row, "edge_rank_within_symbol")
    room_degree = feature(row, "room_graph_degree")
    component_edges = feature(row, "component_edge_count")
    compact_bonus = max(0.0, 1.0 - abs(area_ratio - safe_float(weights["preferred_area_ratio"])) / max(safe_float(weights["area_tolerance"]), 1e-9))
    rank_bonus = 1.0 / (1.0 + max(symbol_rank - 1.0, 0.0))
    crowd_penalty = min(1.0, math.log1p(max(room_degree, component_edges / 20.0)) / 7.0)
    kind_bonus = safe_float(weights["subcc_bonus"]) if kind == "connected_component" else safe_float(weights["center_anchor_bonus"])
    return (
        safe_float(weights["confidence"]) * confidence
        + safe_float(weights["relation"]) * relation_conf
        + safe_float(weights["density"]) * density
        + safe_float(weights["compact"]) * compact_bonus
        + safe_float(weights["edge_margin"]) * edge_margin
        + safe_float(weights["rank"]) * rank_bonus
        + kind_bonus
        - safe_float(weights["crowd"]) * crowd_penalty
    )


def default_weight_grid() -> list[dict[str, float]]:
    grids: list[dict[str, float]] = []
    for preferred_area_ratio in [0.16, 0.28]:
        for center_anchor_bonus, subcc_bonus in [(0.04, 0.04), (0.08, 0.02)]:
            grids.append(
                {
                    "confidence": 0.26,
                    "relation": 0.18,
                    "density": 0.16,
                    "compact": 0.16,
                    "edge_margin": 0.10,
                    "rank": 0.08,
                    "crowd": 0.10,
                    "preferred_area_ratio": preferred_area_ratio,
                    "area_tolerance": 0.34,
                    "center_anchor_bonus": center_anchor_bonus,
                    "subcc_bonus": subcc_bonus,
                }
            )
    return grids


def build_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        if not is_new_symbol_proposal(row):
            continue
        target_id = str(row.get("target_candidate_id") or "")
        records.append(
            {
                "row_id": str(row.get("row_id")),
                "edge_id": str(row.get("edge_id") or row.get("relation_id") or ""),
                "target_candidate_id": target_id,
                "parent_candidate_id": parent_id(target_id),
                "source_candidate_id": str(row.get("source_candidate_id") or ""),
                "room_instance_cluster_id": str(row.get("room_instance_cluster_id") or ""),
                "kind": proposal_kind(row),
                "gold_key": gold_key(row),
                "target_local_dark_density": feature(row, "target_local_dark_density"),
                "target_bbox_to_anchor_area_ratio": feature(row, "target_bbox_to_anchor_area_ratio"),
                "row": row,
            }
        )
    return records


def add_scores(records: list[dict[str, Any]], weight_grid: list[dict[str, float]]) -> None:
    for record in records:
        row = record["row"]
        record["scores"] = [proposal_score(row, weights) for weights in weight_grid]
        record.pop("row", None)


def eligible(record: dict[str, Any], policy: dict[str, Any]) -> bool:
    if safe_float(record["target_local_dark_density"]) < safe_float(policy["min_density"]):
        return False
    area_ratio = safe_float(record["target_bbox_to_anchor_area_ratio"])
    if area_ratio > safe_float(policy["max_area_ratio"]):
        return False
    if record["kind"] == "center_anchor" and not bool(policy["allow_center_anchor"]):
        return False
    if record["kind"] == "connected_component" and not bool(policy["allow_connected_component"]):
        return False
    return True


def select_records(records: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    scored: list[dict[str, Any]] = []
    weight_index = int(policy["weight_index"])
    for record in records:
        if not eligible(record, policy):
            continue
        score = safe_float((record.get("scores") or [])[weight_index])
        if score < safe_float(policy["score_threshold"]):
            continue
        scored.append({**record, "score": score})
    scored.sort(key=lambda item: safe_float(item["score"]), reverse=True)

    selected: list[dict[str, Any]] = []
    parent_counts: Counter[tuple[str, str]] = Counter()
    room_counts: Counter[tuple[str, str]] = Counter()
    page_counts: Counter[str] = Counter()
    kind_counts: Counter[tuple[str, str]] = Counter()
    for record in scored:
        row_id = str(record["row_id"])
        parent_key = (row_id, str(record["parent_candidate_id"]))
        room_key = (row_id, str(record["room_instance_cluster_id"]))
        kind_key = (row_id, str(record["kind"]))
        if page_counts[row_id] >= int(policy["max_edges_per_page"]):
            continue
        if parent_counts[parent_key] >= int(policy["max_edges_per_parent"]):
            continue
        if room_counts[room_key] >= int(policy["max_edges_per_room"]):
            continue
        if kind_counts[kind_key] >= int(policy["max_edges_per_kind_per_page"]):
            continue
        selected.append(record)
        parent_counts[parent_key] += 1
        room_counts[room_key] += 1
        page_counts[row_id] += 1
        kind_counts[kind_key] += 1
    return selected


def policy_grid() -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    weights = default_weight_grid()
    for weight_index, weight in enumerate(weights):
        for threshold in [0.54, 0.62]:
            for max_page in [32, 64]:
                for max_room in [8, 16]:
                    out.append(
                        {
                            "weights": weight,
                            "weight_index": weight_index,
                            "score_threshold": threshold,
                            "min_density": 0.0,
                            "max_area_ratio": 0.95,
                            "max_edges_per_parent": 1,
                            "max_edges_per_room": max_room,
                            "max_edges_per_page": max_page,
                            "max_edges_per_kind_per_page": max_page,
                            "allow_connected_component": True,
                            "allow_center_anchor": True,
                        }
                    )
    return out


def evaluate_policy(baseline_keys: set[str], records: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    selected = select_records(records, policy)
    selected_keys = {str(record["gold_key"]) for record in selected if record.get("gold_key")}
    gained = selected_keys - baseline_keys
    estimated_keys = baseline_keys | selected_keys
    kind_counts = Counter(str(record["kind"]) for record in selected)
    positive_kind_counts = Counter(str(record["kind"]) for record in selected if record.get("gold_key"))
    return {
        "policy": policy,
        "selected_edges": len(selected),
        "selected_positive_edges": sum(1 for record in selected if record.get("gold_key")),
        "selected_gold_keys": len(selected_keys),
        "gained_gold_keys_vs_baseline": len(gained),
        "lost_gold_keys_vs_baseline": 0,
        "estimated_recoverable_gold": len(estimated_keys),
        "estimated_recoverable_gold_delta": len(estimated_keys) - len(baseline_keys),
        "estimated_added_rows_per_gained_gold": round(len(selected) / max(len(gained), 1), 6),
        "selected_kind_counts": dict(kind_counts),
        "selected_positive_kind_counts": dict(positive_kind_counts),
        "selected_edge_precision": round(sum(1 for record in selected if record.get("gold_key")) / max(len(selected), 1), 6),
    }


def choose(results: list[dict[str, Any]]) -> dict[str, Any]:
    viable = [
        row for row in results
        if row["lost_gold_keys_vs_baseline"] == 0
        and row["gained_gold_keys_vs_baseline"] >= 40
        and row["estimated_added_rows_per_gained_gold"] <= 250
    ]
    pool = viable or [row for row in results if row["lost_gold_keys_vs_baseline"] == 0] or results
    return max(
        pool,
        key=lambda row: (
            row["estimated_recoverable_gold_delta"],
            -row["estimated_added_rows_per_gained_gold"],
            row["selected_positive_edges"],
            -row["selected_edges"],
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
    variant_keys = recovered_keys(variant_rows)
    records = build_records(variant_rows)
    add_scores(records, default_weight_grid())
    results = [evaluate_policy(baseline_keys, records, policy) for policy in policy_grid()]
    results.sort(key=lambda row: (-row["estimated_recoverable_gold_delta"], row["estimated_added_rows_per_gained_gold"], row["selected_edges"]))
    selected = choose(results)
    report = {
        "task": "IMG-MOE-V18-REBUILD-001.step_g3_train_key_level_symbol_proposal_ranker",
        "baseline": str(args.baseline),
        "variant": str(args.variant),
        "baseline_rows": len(baseline_rows),
        "variant_rows": len(variant_rows),
        "baseline_recoverable_gold": len(baseline_keys),
        "variant_recoverable_gold": len(variant_keys),
        "variant_direct_recoverable_gold_delta": len(variant_keys) - len(baseline_keys),
        "variant_direct_gained_gold_keys": len(variant_keys - baseline_keys),
        "variant_direct_lost_gold_keys": len(baseline_keys - variant_keys),
        "proposal_record_count": len(records),
        "proposal_record_kind_counts": dict(Counter(str(record["kind"]) for record in records)),
        "selected_policy": selected,
        "top_policies": results[:50],
        "policy_count": len(results),
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_policy_selection_only": True,
        "gold_used_for_inference": False,
    }
    write_json(Path(args.output), report)
    model = {
        "model_type": "symbol_proposal_ranker_v18_policy_gate",
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
            "target_candidate_id_kind",
        ],
        "source_integrity": integrity(),
        "gold_used_for_inference": False,
    }
    write_json(Path(args.model_output), model)
    print(json.dumps(report["selected_policy"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
