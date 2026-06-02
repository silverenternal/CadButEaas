#!/usr/bin/env python3
"""Build v69 uncovered-target actions with selected-set conflict features."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

from apply_symbol_budgeted_targeted_rescue_v64 import BASE_POLICY, build_rescue_pool, candidate_id, feature_score, is_focus, label
from apply_symbol_detector_recall_preserving_policy_v47 import group_pages, safe_float, select_rows
from apply_symbol_focus_rescue_policy_v52 import candidate_area
from apply_symbol_sink_tiny_refiner_page_v49 import valid_box
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import bbox_iou, rel, write_json, write_jsonl


def iou_target(row: dict[str, Any]) -> str:
    labels = row.get("labels") or {}
    if safe_float(labels.get("best_iou")) >= 0.30:
        return str(labels.get("best_iou_target_id") or "")
    return ""


def center_targets(row: dict[str, Any]) -> set[str]:
    return {str(target) for target in (row.get("labels") or {}).get("center_target_ids") or [] if str(target)}


def gold_meta(rows: list[dict[str, Any]]) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in rows:
        for gold in (row.get("labels") or {}).get("page_gold_targets") or []:
            target_id = str(gold.get("target_id") or "")
            if target_id:
                out[target_id] = {"label": str(gold.get("label") or ""), "area": str(gold.get("area_bucket") or "")}
    return out


def cluster_ranks(rows: list[dict[str, Any]], low_score_threshold: float) -> dict[str, int]:
    by_cluster: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if safe_float(row.get("score")) >= low_score_threshold:
            by_cluster[str(row.get("cluster_key") or row.get("cluster_id") or "")].append(row)
    ranks: dict[str, int] = {}
    for items in by_cluster.values():
        items.sort(key=feature_score, reverse=True)
        for idx, row in enumerate(items, start=1):
            ranks[candidate_id(row)] = idx
    return ranks


def action_bucket(row: dict[str, Any], base_iou_targets: set[str], base_center_targets: set[str]) -> str:
    target = iou_target(row)
    if target:
        return "duplicate_iou_target" if target in base_iou_targets else "new_iou_target"
    centers = center_targets(row)
    if centers:
        return "duplicate_center_only_target" if centers <= base_center_targets else "new_center_only_target"
    return "background_or_support"


def box_center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def selected_context(row: dict[str, Any], selected: list[dict[str, Any]]) -> dict[str, float]:
    row_box = valid_box(row.get("bbox"))
    if not row_box or not selected:
        return {
            "nearest_selected_iou": 0.0,
            "nearest_same_label_iou": 0.0,
            "nearest_selected_center_dist": 9999.0,
            "same_label_score_margin": safe_float(row.get("score")),
            "same_area_score_margin": safe_float(row.get("score")),
            "same_cluster_score_margin": safe_float(row.get("score")),
            "overlap_selected_count_iou_0_10": 0.0,
            "overlap_selected_count_iou_0_30": 0.0,
        }
    row_cx, row_cy = box_center(row_box)
    nearest_iou = 0.0
    nearest_same_label_iou = 0.0
    nearest_dist = 9999.0
    max_same_label_score = 0.0
    max_same_area_score = 0.0
    max_same_cluster_score = 0.0
    overlap_010 = 0
    overlap_030 = 0
    row_label = label(row)
    row_area = candidate_area(row)
    row_cluster = str(row.get("cluster_key") or row.get("cluster_id") or "")
    for item in selected:
        item_box = valid_box(item.get("bbox"))
        if not item_box:
            continue
        iou = bbox_iou(row_box, item_box)
        nearest_iou = max(nearest_iou, iou)
        if iou >= 0.10:
            overlap_010 += 1
        if iou >= 0.30:
            overlap_030 += 1
        icx, icy = box_center(item_box)
        nearest_dist = min(nearest_dist, ((row_cx - icx) ** 2 + (row_cy - icy) ** 2) ** 0.5)
        if label(item) == row_label:
            nearest_same_label_iou = max(nearest_same_label_iou, iou)
            max_same_label_score = max(max_same_label_score, safe_float(item.get("score")))
        if candidate_area(item) == row_area:
            max_same_area_score = max(max_same_area_score, safe_float(item.get("score")))
        if str(item.get("cluster_key") or item.get("cluster_id") or "") == row_cluster:
            max_same_cluster_score = max(max_same_cluster_score, safe_float(item.get("score")))
    row_score = safe_float(row.get("score"))
    return {
        "nearest_selected_iou": nearest_iou,
        "nearest_same_label_iou": nearest_same_label_iou,
        "nearest_selected_center_dist": nearest_dist,
        "same_label_score_margin": row_score - max_same_label_score,
        "same_area_score_margin": row_score - max_same_area_score,
        "same_cluster_score_margin": row_score - max_same_cluster_score,
        "overlap_selected_count_iou_0_10": float(overlap_010),
        "overlap_selected_count_iou_0_30": float(overlap_030),
    }


def fit_label_area_prior(train_page_rows: dict[str, list[dict[str, Any]]], low_score_threshold: float, rank_cap: int) -> dict[str, float]:
    counts = Counter()
    for page_rows in train_page_rows.values():
        base = select_rows(page_rows, 0.02, 1, 4, 200)
        base_ids = {candidate_id(row) for row in base}
        base_iou_targets = {target for row in base if (target := iou_target(row))}
        base_center_targets = set().union(*(center_targets(row) for row in base)) if base else set()
        policy = {
            "low_score_threshold": low_score_threshold,
            "rescue_cluster_rank_cap": rank_cap,
            "max_swaps_per_page": 0,
            "drop_mode": "none",
            "allow_any_drop": "false",
            "protect_focus_bonus": 0.0,
            "protect_singleton_bonus": 0.0,
            "generic_large_drop_bonus": 0.0,
        }
        for row in build_rescue_pool(page_rows, base_ids, policy):
            key = f"{label(row)}|{candidate_area(row)}"
            counts[f"{key}:total"] += 1
            if action_bucket(row, base_iou_targets, base_center_targets) == "new_iou_target":
                counts[f"{key}:pos"] += 1
    global_total = sum(value for key, value in counts.items() if key.endswith(":total"))
    global_pos = sum(value for key, value in counts.items() if key.endswith(":pos"))
    prior = global_pos / max(global_total, 1)
    out = {"__global__": prior}
    keys = {key.rsplit(":", 1)[0] for key in counts if key.endswith(":total")}
    for key in keys:
        pos = counts[f"{key}:pos"]
        total = counts[f"{key}:total"]
        out[key] = (pos + 8.0 * prior) / max(total + 8.0, 1.0)
    return out


def action_features(
    row: dict[str, Any],
    page_rows: list[dict[str, Any]],
    base_selected: list[dict[str, Any]],
    rank: int,
    priors: dict[str, float],
) -> dict[str, float]:
    feats = row.get("features") or {}
    row_label = label(row)
    row_area = candidate_area(row)
    same_label_selected = sum(1 for item in base_selected if label(item) == row_label)
    same_area_selected = sum(1 for item in base_selected if candidate_area(item) == row_area)
    same_cluster_selected = sum(
        1
        for item in base_selected
        if str(item.get("cluster_key") or item.get("cluster_id") or "") == str(row.get("cluster_key") or row.get("cluster_id") or "")
    )
    out = {
        "score": safe_float(row.get("score")),
        "feature_score": feature_score(row),
        "cluster_score_max": safe_float(feats.get("cluster_score_max")),
        "cluster_score_mean": safe_float(feats.get("cluster_score_mean")),
        "cluster_size": safe_float(feats.get("cluster_size")),
        "page_candidate_count": safe_float(feats.get("page_candidate_count") or len(page_rows)),
        "width_norm": safe_float(feats.get("width_norm")),
        "height_norm": safe_float(feats.get("height_norm")),
        "area_norm": safe_float(feats.get("area_norm")),
        "aspect": safe_float(feats.get("aspect")),
        "label_id": safe_float(feats.get("label_id")),
        "cluster_rank": float(rank),
        "is_focus": 1.0 if is_focus(row) else 0.0,
        "same_label_selected": float(same_label_selected),
        "same_area_selected": float(same_area_selected),
        "same_cluster_selected": float(same_cluster_selected),
        "base_selected_count": float(len(base_selected)),
        "label_area_prior": priors.get(f"{row_label}|{row_area}", priors.get("__global__", 0.0)),
    }
    out.update(selected_context(row, base_selected))
    for name in ["sink", "equipment", "shower", "stair", "generic_symbol", "appliance", "column", "bathtub", "table"]:
        out[f"label_is_{name}"] = 1.0 if row_label == name else 0.0
    for name in ["tiny_le_64", "small_le_256", "medium_le_1024", "large_le_4096", "xlarge_gt_4096"]:
        out[f"area_is_{name}"] = 1.0 if row_area == name else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--output-dir", default="datasets/symbol_uncovered_target_actions_v69")
    parser.add_argument("--low-score-threshold", type=float, default=0.005)
    parser.add_argument("--rescue-cluster-rank-cap", type=int, default=4)
    parser.add_argument("--max-actions-per-page", type=int, default=64)
    args = parser.parse_args()

    manifest = json.loads(source_path(args.data).read_text(encoding="utf-8"))
    rows = load_jsonl(source_path(manifest["outputs"]["rows"]))
    pages = group_pages(rows, "all")
    train_pages = {pid: page_rows for pid, page_rows in pages.items() if str(page_rows[0].get("split") or "") == "train"}
    priors = fit_label_area_prior(train_pages, args.low_score_threshold, args.rescue_cluster_rank_cap)
    policy = {
        "low_score_threshold": args.low_score_threshold,
        "rescue_cluster_rank_cap": args.rescue_cluster_rank_cap,
        "max_swaps_per_page": 0,
        "drop_mode": "none",
        "allow_any_drop": "false",
        "protect_focus_bonus": 0.0,
        "protect_singleton_bonus": 0.0,
        "generic_large_drop_bonus": 0.0,
    }
    action_rows: list[dict[str, Any]] = []
    page_rows_out: list[dict[str, Any]] = []
    summary = Counter()
    for page_id, page_rows in pages.items():
        split = str(page_rows[0].get("split") or "")
        base = select_rows(page_rows, 0.02, 1, 4, 200)
        base_ids = {candidate_id(row) for row in base}
        base_iou_targets = {target for row in base if (target := iou_target(row))}
        base_center_targets = set().union(*(center_targets(row) for row in base)) if base else set()
        meta = gold_meta(page_rows)
        ranks = cluster_ranks(page_rows, args.low_score_threshold)
        candidates = build_rescue_pool(page_rows, base_ids, policy)[: args.max_actions_per_page]
        for row in candidates:
            target = iou_target(row)
            bucket = action_bucket(row, base_iou_targets, base_center_targets)
            target_info = meta.get(target, {})
            action = {
                "page_id": page_id,
                "split": split,
                "candidate_id": candidate_id(row),
                "label": label(row),
                "area": candidate_area(row),
                "features": action_features(row, page_rows, base, ranks.get(candidate_id(row), 999), priors),
                "labels": {
                    "bucket": bucket,
                    "new_iou_target": bucket == "new_iou_target",
                    "duplicate_iou_target": bucket == "duplicate_iou_target",
                    "duplicate_center_only_target": bucket == "duplicate_center_only_target",
                    "new_center_only_target": bucket == "new_center_only_target",
                    "background_or_support": bucket == "background_or_support",
                    "reward": 1.0 if bucket == "new_iou_target" else (-0.25 if bucket == "new_center_only_target" else -1.0),
                    "best_iou": (row.get("labels") or {}).get("best_iou"),
                    "target_id": target,
                    "center_target_count": len(center_targets(row)),
                    "target_label": target_info.get("label"),
                    "target_area": target_info.get("area"),
                },
                "source_integrity": {
                    "gold_used_for_inference": False,
                    "runtime_uses_svg_or_cad_geometry": False,
                    "offline_labels_used_for": ["uncovered_target_reward_training", "audit"],
                },
            }
            action_rows.append(action)
            summary["actions"] += 1
            summary[f"bucket:{bucket}"] += 1
            summary[f"bucket_label:{bucket}:{label(row)}"] += 1
            summary[f"bucket_area:{bucket}:{candidate_area(row)}"] += 1
        page_rows_out.append(
            {
                "page_id": page_id,
                "split": split,
                "base_selected_count": len(base),
                "base_iou_target_count": len(base_iou_targets),
                "base_center_target_count": len(base_center_targets),
                "action_count": len(candidates),
            }
        )
        summary["pages"] += 1
        summary["base_selected"] += len(base)
        summary["base_iou_targets"] += len(base_iou_targets)

    output_dir = source_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.jsonl"
    pages_path = output_dir / "pages.jsonl"
    manifest_path = output_dir / "manifest.json"
    write_jsonl(rows_path, action_rows)
    write_jsonl(pages_path, page_rows_out)
    report = {
        "version": "symbol_uncovered_target_actions_v69",
        "source_data": rel(source_path(args.data)),
        "base_policy": BASE_POLICY,
        "candidate_policy": policy,
        "outputs": {"rows": rel(rows_path), "pages": rel(pages_path), "label_area_prior": rel(output_dir / "label_area_prior.json")},
        "summary": dict(summary),
        "source_integrity": {
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "runtime_features": [
                "candidate score",
                "predicted label",
                "bbox area",
                "cluster/page features",
                "base selected-set context counts",
                "nearest selected bbox IoU/proximity",
                "label-area prior fitted on train split",
            ],
            "offline_labels_used_for": ["uncovered_target_reward_training", "train_split_prior_fit", "audit"],
        },
    }
    write_json(output_dir / "label_area_prior.json", priors)
    write_json(manifest_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
