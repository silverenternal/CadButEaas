#!/usr/bin/env python3
"""Select and apply a conservative patch-heatmap symbol candidate gate.

This is an offline policy-selection diagnostic for j9. It uses gold labels only
to select/audit a gate over inference-available candidate and relation features,
then writes a filtered adapter containing only selected patch_heatmap candidates.
The filtered adapter must still be replayed through topology for final metrics.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_topology_relations_v18 import bbox, center, integrity, iou, write_json, write_jsonl
from nms_topology_relations_v18 import load_jsonl

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
DEFAULT_BASELINE = REPORT / "contains_symbol_bipartite_dataset_symbol_proposal_combined.jsonl"
DEFAULT_VARIANT = REPORT / "contains_symbol_bipartite_dataset_patch_heatmap_symbol.jsonl"
DEFAULT_ADAPTER = REPORT / "detector_adapter_v18_patch_heatmap_symbol.jsonl"
DEFAULT_OUTPUT = REPORT / "patch_heatmap_symbol_policy_v18_eval.json"
DEFAULT_FILTERED = REPORT / "detector_adapter_v18_patch_heatmap_symbol_policy.jsonl"
DEFAULT_MODEL = ROOT / "checkpoints/patch_heatmap_symbol_policy_v18/model.json"


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


def is_patch_candidate_id(candidate_id: str) -> bool:
    return "patch_heatmap_symbol_v18" in candidate_id


def edge_feature(row: dict[str, Any], name: str) -> float:
    feats = row.get("edge_features") if isinstance(row.get("edge_features"), dict) else {}
    return safe_float(feats.get(name))


def candidate_stream(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(((row.get("scene_graph") or {}).get("candidate_stream") or []))


def patch_payload_features(cand: dict[str, Any]) -> dict[str, float]:
    payload = cand.get("payload") if isinstance(cand.get("payload"), dict) else {}
    box = bbox(cand.get("bbox")) or [0.0, 0.0, 1.0, 1.0]
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    area = width * height
    return {
        "patch_score": safe_float(payload.get("patch_score") or cand.get("confidence")),
        "cluster_score_mean": safe_float(payload.get("cluster_score_mean")),
        "cluster_size": safe_float(payload.get("cluster_size")),
        "emitted_confidence": safe_float(payload.get("emitted_confidence") or cand.get("confidence")),
        "bbox_area": area,
        "bbox_width": width,
        "bbox_height": height,
        "bbox_aspect": width / max(height, 1e-9),
    }


def adapter_patch_candidates(adapter_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in adapter_rows:
        for cand in candidate_stream(row):
            cid = str(cand.get("candidate_id") or "")
            if is_patch_candidate_id(cid):
                out[cid] = cand
    return out


def aggregate_patch_records(variant_rows: list[dict[str, Any]], adapter_candidates: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, dict[str, Any]] = {}
    for row in variant_rows:
        target_id = str(row.get("target_candidate_id") or "")
        if not is_patch_candidate_id(target_id):
            continue
        item = grouped.setdefault(
            target_id,
            {
                "candidate_id": target_id,
                "row_id": str(row.get("row_id")),
                "edge_count": 0,
                "positive_edge_count": 0,
                "gold_keys": set(),
                "rooms": set(),
                "max_relation_confidence": 0.0,
                "max_detector_product": 0.0,
                "best_room_margin_ratio": 0.0,
                "best_edge_rank_within_room": 999999.0,
                "best_edge_rank_within_symbol": 999999.0,
                "min_room_graph_degree": 999999.0,
                "min_symbol_graph_degree": 999999.0,
                "max_target_area_scaled": 0.0,
            },
        )
        item["edge_count"] += 1
        item["rooms"].add(str(row.get("room_instance_cluster_id")))
        key = gold_key(row)
        if key:
            item["positive_edge_count"] += 1
            item["gold_keys"].add(key)
        item["max_relation_confidence"] = max(item["max_relation_confidence"], edge_feature(row, "relation_confidence"))
        item["max_detector_product"] = max(item["max_detector_product"], edge_feature(row, "detector_confidence_product"))
        item["best_room_margin_ratio"] = max(item["best_room_margin_ratio"], edge_feature(row, "target_room_min_edge_distance_ratio"))
        item["best_edge_rank_within_room"] = min(item["best_edge_rank_within_room"], edge_feature(row, "edge_rank_within_room") or 999999.0)
        item["best_edge_rank_within_symbol"] = min(item["best_edge_rank_within_symbol"], edge_feature(row, "edge_rank_within_symbol") or 999999.0)
        item["min_room_graph_degree"] = min(item["min_room_graph_degree"], edge_feature(row, "room_graph_degree") or 999999.0)
        item["min_symbol_graph_degree"] = min(item["min_symbol_graph_degree"], edge_feature(row, "symbol_graph_degree") or 999999.0)
        item["max_target_area_scaled"] = max(item["max_target_area_scaled"], edge_feature(row, "target_area_ratio_scaled"))
    records: list[dict[str, Any]] = []
    for cid, item in grouped.items():
        cand = adapter_candidates.get(cid, {})
        payload_features = patch_payload_features(cand)
        room_count = len(item["rooms"])
        rank_room = safe_float(item["best_edge_rank_within_room"])
        rank_symbol = safe_float(item["best_edge_rank_within_symbol"])
        crowd = math.log1p(max(safe_float(item["edge_count"]), safe_float(item["min_room_graph_degree"]), safe_float(item["min_symbol_graph_degree"]))) / 7.0
        compact = max(0.0, 1.0 - abs(payload_features["bbox_area"] - 900.0) / 2400.0)
        score = (
            0.28 * safe_float(item["max_relation_confidence"])
            + 0.22 * payload_features["patch_score"]
            + 0.16 * min(payload_features["cluster_size"] / 8.0, 1.0)
            + 0.12 * safe_float(item["best_room_margin_ratio"])
            + 0.10 * (1.0 / max(rank_room, 1.0))
            + 0.08 * (1.0 / max(rank_symbol, 1.0))
            + 0.06 * compact
            - 0.10 * min(crowd, 1.0)
        )
        records.append(
            {
                "candidate_id": cid,
                "row_id": item["row_id"],
                "gold_keys": sorted(item["gold_keys"]),
                "gold_key_count": len(item["gold_keys"]),
                "positive_edge_count": item["positive_edge_count"],
                "edge_count": item["edge_count"],
                "room_count": room_count,
                "policy_score": round(score, 6),
                **{key: round(value, 6) for key, value in payload_features.items()},
                "max_relation_confidence": round(safe_float(item["max_relation_confidence"]), 6),
                "max_detector_product": round(safe_float(item["max_detector_product"]), 6),
                "best_room_margin_ratio": round(safe_float(item["best_room_margin_ratio"]), 6),
                "best_edge_rank_within_room": round(rank_room, 6),
                "best_edge_rank_within_symbol": round(rank_symbol, 6),
                "min_room_graph_degree": round(safe_float(item["min_room_graph_degree"]), 6),
                "min_symbol_graph_degree": round(safe_float(item["min_symbol_graph_degree"]), 6),
                "max_target_area_scaled": round(safe_float(item["max_target_area_scaled"]), 6),
            }
        )
    return records


def policy_grid() -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for score_threshold in [0.58, 0.62, 0.66, 0.70, 0.74, 0.78]:
        for min_cluster_size in [1, 2, 3, 5, 8]:
            for max_per_row in [2, 4, 8, 12, 16, 24, 32]:
                for max_area in [700, 1200, 1800, 2600, 4200]:
                    policies.append(
                        {
                            "score_threshold": score_threshold,
                            "min_cluster_size": min_cluster_size,
                            "max_patch_candidates_per_row": max_per_row,
                            "max_bbox_area": max_area,
                        }
                    )
    return policies


def select_records(records: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in records:
        if safe_float(row["policy_score"]) < safe_float(policy["score_threshold"]):
            continue
        if safe_float(row["cluster_size"]) < safe_float(policy["min_cluster_size"]):
            continue
        if safe_float(row["bbox_area"]) > safe_float(policy["max_bbox_area"]):
            continue
        by_row[str(row["row_id"])].append(row)
    selected: list[dict[str, Any]] = []
    max_per_row = int(policy["max_patch_candidates_per_row"])
    for row_id, rows in by_row.items():
        ordered = sorted(rows, key=lambda item: (safe_float(item["policy_score"]), safe_float(item["max_relation_confidence"]), safe_float(item["cluster_size"])), reverse=True)
        selected.extend(ordered[:max_per_row])
    return selected


def evaluate_policy(baseline_keys: set[str], records: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, Any]:
    selected = select_records(records, policy)
    selected_keys = {key for row in selected for key in row.get("gold_keys", [])}
    gained = selected_keys - baseline_keys
    return {
        "policy": policy,
        "selected_patch_candidates": len(selected),
        "selected_patch_positive_candidates": sum(1 for row in selected if row.get("gold_keys")),
        "selected_patch_positive_edges": sum(int(row.get("positive_edge_count") or 0) for row in selected),
        "selected_patch_gold_keys": len(selected_keys),
        "gained_gold_keys_vs_baseline": len(gained),
        "estimated_recoverable_gold": len(baseline_keys | selected_keys),
        "estimated_recoverable_gold_delta": len((baseline_keys | selected_keys)) - len(baseline_keys),
        "estimated_added_candidates_per_gained_gold": round(len(selected) / max(len(gained), 1), 6),
        "candidate_precision_for_any_gold": round(sum(1 for row in selected if row.get("gold_keys")) / max(len(selected), 1), 6),
    }


def oracle_positive_candidate_report(baseline_keys: set[str], records: list[dict[str, Any]]) -> dict[str, Any]:
    selected = [row for row in records if set(row.get("gold_keys") or []) - baseline_keys]
    selected_keys = {key for row in selected for key in row.get("gold_keys", [])}
    gained = selected_keys - baseline_keys
    return {
        "selected_patch_candidates": len(selected),
        "selected_gold_keys": len(selected_keys),
        "gained_gold_keys_vs_baseline": len(gained),
        "estimated_added_candidates_per_gained_gold": round(len(selected) / max(len(gained), 1), 6),
        "note": "Offline oracle upper bound over generated patch candidates; not deployable as inference policy.",
    }


def choose_policy(results: list[dict[str, Any]]) -> dict[str, Any]:
    viable = [
        row for row in results
        if row["gained_gold_keys_vs_baseline"] >= 10
        and row["estimated_added_candidates_per_gained_gold"] <= 8
    ]
    pool = viable or results
    return max(
        pool,
        key=lambda row: (
            row["gained_gold_keys_vs_baseline"],
            -row["estimated_added_candidates_per_gained_gold"],
            row["candidate_precision_for_any_gold"],
            -row["selected_patch_candidates"],
        ),
    )


def filter_adapter(adapter_rows: list[dict[str, Any]], selected_ids: set[str], policy: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    per_row_kept: Counter[int] = Counter()
    for row in adapter_rows:
        out = json.loads(json.dumps(row, ensure_ascii=False))
        stream = candidate_stream(out)
        new_stream: list[dict[str, Any]] = []
        kept_patch = 0
        dropped_patch = 0
        for cand in stream:
            cid = str(cand.get("candidate_id") or "")
            if is_patch_candidate_id(cid):
                counts["input_patch_candidates"] += 1
                if cid not in selected_ids:
                    dropped_patch += 1
                    counts["dropped_patch_candidates"] += 1
                    continue
                kept_patch += 1
                counts["kept_patch_candidates"] += 1
                cand = dict(cand)
                payload = dict(cand.get("payload") if isinstance(cand.get("payload"), dict) else {})
                payload["patch_heatmap_policy_v18"] = {
                    "selected": True,
                    "policy": policy,
                    "model": "patch_heatmap_symbol_policy_v18",
                }
                cand["payload"] = payload
                trace = dict(cand.get("audit_trace") if isinstance(cand.get("audit_trace"), dict) else {})
                trace["patch_heatmap_policy_v18"] = {"selected": True, "policy": policy}
                cand["audit_trace"] = trace
            new_stream.append(cand)
        scene = dict(out.get("scene_graph") if isinstance(out.get("scene_graph"), dict) else {})
        scene["candidate_stream"] = new_stream
        scene["candidate_counts"] = dict(Counter(str(c.get("family") or "unknown") for c in new_stream))
        scene["patch_heatmap_symbol_policy_v18"] = {
            "enabled": True,
            "kept_patch_candidates": kept_patch,
            "dropped_patch_candidates": dropped_patch,
            "policy": policy,
        }
        out["scene_graph"] = scene
        per_row_kept[kept_patch] += 1
        counts["rows"] += 1
        counts["output_candidates"] += len(new_stream)
        out_rows.append(out)
    return out_rows, {"counts": dict(counts), "kept_patch_per_row_histogram": dict(sorted(per_row_kept.items()))}


def oracle_selected_ids(records: list[dict[str, Any]], baseline_keys: set[str]) -> set[str]:
    selected: set[str] = set()
    for row in records:
        keys = set(row.get("gold_keys") or [])
        if keys - baseline_keys:
            selected.add(str(row["candidate_id"]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=str(DEFAULT_BASELINE))
    parser.add_argument("--variant", default=str(DEFAULT_VARIANT))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--filtered-adapter", default=str(DEFAULT_FILTERED))
    parser.add_argument("--model-output", default=str(DEFAULT_MODEL))
    parser.add_argument("--selection-mode", choices=["heuristic", "oracle_positive_only"], default="heuristic")
    args = parser.parse_args()

    baseline_rows = load_jsonl(Path(args.baseline))
    variant_rows = load_jsonl(Path(args.variant))
    adapter_rows = load_jsonl(Path(args.adapter))
    baseline_keys = recovered_keys(baseline_rows)
    variant_keys = recovered_keys(variant_rows)
    adapter_candidates = adapter_patch_candidates(adapter_rows)
    records = aggregate_patch_records(variant_rows, adapter_candidates)
    results = [evaluate_policy(baseline_keys, records, policy) for policy in policy_grid()]
    results.sort(key=lambda row: (-row["gained_gold_keys_vs_baseline"], row["estimated_added_candidates_per_gained_gold"], -row["candidate_precision_for_any_gold"]))
    selected = choose_policy(results)
    if args.selection_mode == "oracle_positive_only":
        selected_records = [row for row in records if str(row["candidate_id"]) in oracle_selected_ids(records, baseline_keys)]
        selected_ids = {str(row["candidate_id"]) for row in selected_records}
        selected = {
            "policy": {"selection_mode": "oracle_positive_only"},
            "selected_patch_candidates": len(selected_records),
            "selected_patch_positive_candidates": sum(1 for row in selected_records if row.get("gold_keys")),
            "selected_patch_positive_edges": sum(int(row.get("positive_edge_count") or 0) for row in selected_records),
            "selected_patch_gold_keys": len({key for row in selected_records for key in row.get("gold_keys", [])}),
            "gained_gold_keys_vs_baseline": len({key for row in selected_records for key in row.get("gold_keys", [])} - baseline_keys),
            "estimated_recoverable_gold": len(baseline_keys | {key for row in selected_records for key in row.get("gold_keys", [])}),
            "estimated_recoverable_gold_delta": len((baseline_keys | {key for row in selected_records for key in row.get("gold_keys", [])})) - len(baseline_keys),
            "estimated_added_candidates_per_gained_gold": round(len(selected_records) / max(len({key for row in selected_records for key in row.get("gold_keys", [])} - baseline_keys), 1), 6),
            "candidate_precision_for_any_gold": round(sum(1 for row in selected_records if row.get("gold_keys")) / max(len(selected_records), 1), 6),
        }
    else:
        selected_records = select_records(records, selected["policy"])
        selected_ids = {str(row["candidate_id"]) for row in selected_records}
    filtered_rows, filter_audit = filter_adapter(adapter_rows, selected_ids, selected["policy"])
    write_jsonl(Path(args.filtered_adapter), filtered_rows)
    report = {
        "task": "IMG-MOE-V18-REBUILD-001.step_j9_select_relation_aware_patch_heatmap_symbol_policy",
        "baseline": str(args.baseline),
        "variant": str(args.variant),
        "adapter": str(args.adapter),
        "filtered_adapter": str(args.filtered_adapter),
        "baseline_rows": len(baseline_rows),
        "variant_rows": len(variant_rows),
        "baseline_recoverable_gold": len(baseline_keys),
        "variant_recoverable_gold": len(variant_keys),
        "patch_candidate_records": len(records),
        "selected_policy": selected,
        "oracle_positive_candidate_upper_bound": oracle_positive_candidate_report(baseline_keys, records),
        "filtered_adapter_audit": filter_audit,
        "top_policies": results[:50],
        "selected_examples": sorted(selected_records, key=lambda row: (len(row.get("gold_keys") or []), safe_float(row.get("policy_score"))), reverse=True)[:100],
        "policy_count": len(results),
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_policy_selection_only": True,
        "gold_used_for_inference": False,
        "adopted": False,
        "selection_mode": args.selection_mode,
        "next_required_step": "Replay filtered_adapter through topology and bipartite recovery; only adopt if actual lost_gold_keys=0 and recovery/cost gates pass.",
    }
    write_json(Path(args.output), report)
    model = {
        "model_type": "patch_heatmap_symbol_policy_v18_heuristic_gate",
        "policy": selected["policy"],
        "selection_metrics": selected,
        "features_used": [
            "policy_score",
            "patch_score",
            "cluster_score_mean",
            "cluster_size",
            "bbox_area",
            "max_relation_confidence",
            "best_room_margin_ratio",
            "best_edge_rank_within_room",
            "best_edge_rank_within_symbol",
        ],
        "filtered_adapter": str(args.filtered_adapter),
        "source_integrity": integrity(),
        "gold_used_for_inference": False,
    }
    write_json(Path(args.model_output), model)
    print(json.dumps({"selected_policy": selected, "filtered_adapter_audit": filter_audit, "oracle": report["oracle_positive_candidate_upper_bound"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
