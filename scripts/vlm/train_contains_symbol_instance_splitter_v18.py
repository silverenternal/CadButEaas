#!/usr/bin/env python3
"""Select a detector-only symbol instance splitter for contains_symbol topology.

Gold relation labels are used only to select/evaluate the splitter policy.
The exported model itself uses only detector candidate geometry/payload at
inference time.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))

from build_topology_relations_v18 import bbox, center, integrity, iou, load_gold, relation_label, write_json  # noqa: E402
from nms_topology_relations_v18 import cluster_candidates, load_by_id, load_jsonl, relation_pair_key, row_candidate_map  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
CHECKPOINT = ROOT / "checkpoints/contains_symbol_instance_splitter_v18"

DEFAULT_INPUT = REPORT / "topology_relations_v18_symbol_boundary_fixed_candidates.jsonl"
DEFAULT_ADAPTER = REPORT / "detector_adapter_v18_symbol_boundary_fixed_candidates.jsonl"
DEFAULT_EVAL = REPORT / "contains_symbol_instance_splitter_eval.json"
DEFAULT_AUDIT = REPORT / "contains_symbol_instance_splitter_audit.json"


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


def candidate_conf(candidate: dict[str, Any]) -> float:
    return safe_float(candidate.get("confidence"))


def box_center_distance(left: list[float], right: list[float]) -> float:
    lx, ly = center(left)
    rx, ry = center(right)
    return math.hypot(lx - rx, ly - ry)


def box_size_ratio(left: list[float], right: list[float]) -> float:
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return min(left_area, right_area) / max(left_area, right_area, 1e-9)


def payload_density(candidate: dict[str, Any]) -> float:
    payload = candidate.get("payload") if isinstance(candidate.get("payload"), dict) else {}
    return safe_float(payload.get("local_dark_density"))


def instance_similar(left: dict[str, Any], right: dict[str, Any], model: dict[str, Any]) -> bool:
    lb = bbox(left.get("bbox"))
    rb = bbox(right.get("bbox"))
    if lb is None or rb is None:
        return False
    if iou(lb, rb) >= safe_float(model.get("merge_iou")):
        return True
    if box_size_ratio(lb, rb) < safe_float(model.get("min_size_ratio")):
        return False
    return box_center_distance(lb, rb) <= safe_float(model.get("center_threshold"))


def assign_instances(
    candidates: dict[str, dict[str, Any]],
    cluster_ids: dict[str, str],
    model: dict[str, Any],
) -> tuple[dict[str, str], Counter[str]]:
    instance_ids = dict(cluster_ids)
    counts: Counter[str] = Counter()
    if not model.get("enabled"):
        return instance_ids, counts

    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cid, candidate in candidates.items():
        if candidate.get("family") != "symbol":
            continue
        cluster_id = cluster_ids.get(cid)
        if cluster_id:
            by_cluster[cluster_id].append(candidate)

    min_members = int(model.get("min_members") or 4)
    min_spread = safe_float(model.get("min_spread"))
    max_instances = max(1, int(model.get("max_instances") or 1))
    for cluster_id, members in by_cluster.items():
        boxes = [bbox(member.get("bbox")) for member in members]
        boxes = [box for box in boxes if box is not None]
        if len(boxes) < min_members:
            continue
        spread = 0.0
        for index, left in enumerate(boxes):
            for right in boxes[index + 1 :]:
                spread = max(spread, box_center_distance(left, right))
        if spread < min_spread:
            continue

        reps: list[dict[str, Any]] = []
        assignment: dict[str, int] = {}
        ordered = sorted(members, key=lambda item: (candidate_conf(item), payload_density(item)), reverse=True)
        for member in ordered:
            cid = str(member.get("candidate_id"))
            best_index: int | None = None
            best_score = -1.0
            for index, rep in enumerate(reps):
                mb = bbox(member.get("bbox"))
                rb = bbox(rep.get("bbox"))
                if mb is None or rb is None or not instance_similar(member, rep, model):
                    continue
                score = iou(mb, rb) + 1.0 / max(box_center_distance(mb, rb), 1.0)
                if score > best_score:
                    best_index = index
                    best_score = score
            if best_index is None:
                if len(reps) >= max_instances:
                    mb = bbox(member.get("bbox")) or [0.0, 0.0, 0.0, 0.0]
                    best_index = min(
                        range(len(reps)),
                        key=lambda index: box_center_distance(mb, bbox(reps[index].get("bbox")) or [0.0, 0.0, 0.0, 0.0]),
                    )
                else:
                    best_index = len(reps)
                    reps.append(member)
            assignment[cid] = best_index

        if len(reps) <= 1:
            continue
        counts["split_symbol_clusters"] += 1
        counts["split_symbol_members"] += len(members)
        counts["emitted_symbol_instances"] += len(reps)
        for cid, index in assignment.items():
            instance_ids[cid] = f"{cluster_id}_inst_{index:02d}"
    return instance_ids, counts


def split_pair_key(rel: dict[str, Any], cluster_ids: dict[str, str], instance_ids: dict[str, str]) -> tuple[str, str, str]:
    if rel.get("relation") != "contains_symbol":
        return relation_pair_key(rel, cluster_ids)
    source_cluster = cluster_ids.get(str(rel.get("source_candidate_id")), f"missing:{rel.get('source_candidate_id')}")
    target_instance = instance_ids.get(
        str(rel.get("target_candidate_id")),
        cluster_ids.get(str(rel.get("target_candidate_id")), f"missing:{rel.get('target_candidate_id')}"),
    )
    return "contains_symbol", source_cluster, target_instance


def build_page_cache(rows: list[dict[str, Any]], adapter_by_id: dict[str, dict[str, Any]], gold: dict[str, Any]) -> list[dict[str, Any]]:
    page_cache: list[dict[str, Any]] = []
    for page in rows:
        row_id = str(page.get("id"))
        adapter = adapter_by_id.get(row_id)
        if not adapter:
            continue
        candidates = row_candidate_map(adapter)
        cluster_ids, _clusters, _warnings = cluster_candidates(row_id, candidates)
        relations: list[dict[str, Any]] = []
        for rel in ((page.get("scene_graph") or {}).get("relations") or []):
            if rel.get("relation") != "contains_symbol":
                continue
            left, right, ok = relation_label({**rel, "row_id": row_id}, candidates, gold)
            relations.append(
                {
                    "rel": rel,
                    "original_key": relation_pair_key(rel, cluster_ids),
                    "gold_key": (row_id, left, right) if ok and left and right else None,
                    "is_positive": bool(ok and left and right),
                }
            )
        page_cache.append({"row_id": row_id, "candidates": candidates, "cluster_ids": cluster_ids, "relations": relations})
    return page_cache


def evaluate_policy(page_cache: list[dict[str, Any]], model: dict[str, Any]) -> dict[str, Any]:
    counters: Counter[str] = Counter()
    original_all_keys: set[tuple[str, str, str]] = set()
    split_all_keys: set[tuple[str, str, str]] = set()
    original_positive_by_pair: dict[tuple[str, str, str], set[tuple[str, str, str]]] = defaultdict(set)
    split_positive_by_pair: dict[tuple[str, str, str], set[tuple[str, str, str]]] = defaultdict(set)
    examples: list[dict[str, Any]] = []

    for page in page_cache:
        row_id = str(page["row_id"])
        candidates = page["candidates"]
        cluster_ids = page["cluster_ids"]
        instance_ids, split_counts = assign_instances(candidates, cluster_ids, model)
        counters.update(split_counts)
        for item in page["relations"]:
            rel = item["rel"]
            original_key = item["original_key"]
            split_key = split_pair_key(rel, cluster_ids, instance_ids)
            original_all_keys.add(original_key)
            split_all_keys.add(split_key)
            if not item["is_positive"]:
                continue
            counters["pre_positive_relations"] += 1
            gold_key = item["gold_key"]
            original_positive_by_pair[original_key].add(gold_key)
            split_positive_by_pair[split_key].add(gold_key)
            if original_key != split_key and len(examples) < 50:
                examples.append(
                    {
                        "row_id": row_id,
                        "gold_key": list(gold_key),
                        "original_key": list(original_key),
                        "split_key": list(split_key),
                        "source_candidate_id": rel.get("source_candidate_id"),
                        "target_candidate_id": rel.get("target_candidate_id"),
                    }
                )

    original_upper_bound = sum(1 for values in original_positive_by_pair.values() if values)
    split_upper_bound = sum(1 for values in split_positive_by_pair.values() if values)
    original_collisions = sum(max(0, len(values) - 1) for values in original_positive_by_pair.values())
    split_collisions = sum(max(0, len(values) - 1) for values in split_positive_by_pair.values())
    extra_keys = max(0, len(split_all_keys) - len(original_all_keys))
    precision_proxy_penalty = extra_keys / max(len(original_all_keys), 1)
    score = (split_upper_bound - original_upper_bound) - 0.25 * extra_keys - 50.0 * precision_proxy_penalty
    return {
        "model": model,
        "score": round(score, 6),
        "original_relation_pair_keys": len(original_all_keys),
        "split_relation_pair_keys": len(split_all_keys),
        "relation_pair_key_increase": extra_keys,
        "relation_pair_key_increase_ratio": round(precision_proxy_penalty, 6),
        "pre_positive_relations": counters["pre_positive_relations"],
        "original_positive_pair_upper_bound": original_upper_bound,
        "split_positive_pair_upper_bound": split_upper_bound,
        "positive_pair_upper_bound_gain": split_upper_bound - original_upper_bound,
        "original_positive_pair_collision_loss": original_collisions,
        "split_positive_pair_collision_loss": split_collisions,
        "positive_pair_collision_loss_reduction": original_collisions - split_collisions,
        "split_symbol_clusters": counters["split_symbol_clusters"],
        "split_symbol_members": counters["split_symbol_members"],
        "emitted_symbol_instances": counters["emitted_symbol_instances"],
        "examples": examples,
    }


def model_grid(smoke: bool) -> list[dict[str, Any]]:
    if smoke:
        return [
            {"enabled": True, "min_members": 3, "min_spread": 6.0, "center_threshold": 2.5, "merge_iou": 0.65, "min_size_ratio": 0.55, "max_instances": 3}
        ]
    models: list[dict[str, Any]] = []
    for min_members in [2, 3, 4]:
        for min_spread in [4.0, 6.0, 8.0, 10.0, 12.0]:
            for center_threshold in [2.0, 3.0, 4.0, 5.0]:
                for merge_iou in [0.55, 0.65, 0.75]:
                    for max_instances in [2, 3]:
                        models.append(
                            {
                                "enabled": True,
                                "min_members": min_members,
                                "min_spread": min_spread,
                                "center_threshold": center_threshold,
                                "merge_iou": merge_iou,
                                "min_size_ratio": 0.55,
                                "max_instances": max_instances,
                            }
                        )
    models.append({"enabled": False, "min_members": 4, "min_spread": 8.0, "center_threshold": 3.0, "merge_iou": 0.65, "min_size_ratio": 0.55, "max_instances": 3})
    return models


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--eval-output", default=str(DEFAULT_EVAL))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--checkpoint", default=str(CHECKPOINT / "model.json"))
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--locked", action="store_true")
    args = parser.parse_args()

    limit = 5 if args.smoke else None
    rows = load_jsonl(Path(args.input), limit=limit)
    adapter_by_id = load_by_id(Path(args.adapter), limit=limit)
    gold = load_gold()

    page_cache = build_page_cache(rows, adapter_by_id, gold)
    results = [evaluate_policy(page_cache, model) for model in model_grid(args.smoke)]
    results.sort(key=lambda item: (-float(item["score"]), -int(item["positive_pair_upper_bound_gain"]), int(item["relation_pair_key_increase"])))
    feasible = [
        item
        for item in results
        if safe_float(item.get("relation_pair_key_increase_ratio")) <= 0.02
        and int(item.get("positive_pair_upper_bound_gain") or 0) > 0
    ]
    best = feasible[0] if feasible else next(item for item in results if not item["model"].get("enabled"))

    checkpoint = {
        "task": "IMG-MOE-V18-REBUILD-001-CONTAINS-SYMBOL-INSTANCE-SPLITTER",
        "model": best["model"],
        "selected_score": best["score"],
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_training_and_policy_selection_only": True,
        "gold_used_for_inference": False,
    }
    eval_report = {
        "task": "IMG-MOE-V18-REBUILD-001-CONTAINS-SYMBOL-INSTANCE-SPLITTER",
        "mode": "oracle-policy-selection",
        "locked": bool(args.locked),
        "smoke": bool(args.smoke),
        "rows": len(rows),
        **{key: value for key, value in best.items() if key != "examples"},
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_training_and_policy_selection_only": True,
        "gold_used_for_inference": False,
    }
    audit = {
        "task": "IMG-MOE-V18-REBUILD-001-CONTAINS-SYMBOL-INSTANCE-SPLITTER-AUDIT",
        "selected": best,
        "sweep": [{key: value for key, value in result.items() if key != "examples"} for result in results[:50]],
        "source_integrity": integrity(),
    }
    write_json(Path(args.checkpoint), checkpoint)
    write_json(Path(args.eval_output), eval_report)
    write_json(Path(args.audit_output), audit)
    print(json.dumps(eval_report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
