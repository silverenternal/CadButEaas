#!/usr/bin/env python3
"""Build uncovered-target action data for symbol add-only rescue policy."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

from apply_symbol_budgeted_targeted_rescue_v64 import BASE_POLICY, build_rescue_pool, candidate_id, feature_score, is_focus, label
from apply_symbol_detector_recall_preserving_policy_v47 import group_pages, safe_float, select_rows
from apply_symbol_focus_rescue_policy_v52 import candidate_area
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl


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
                out[target_id] = {
                    "label": str(gold.get("label") or ""),
                    "area": str(gold.get("area_bucket") or ""),
                }
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


def action_features(row: dict[str, Any], page_rows: list[dict[str, Any]], base_selected: list[dict[str, Any]], rank: int) -> dict[str, float]:
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
    }
    for name in ["sink", "equipment", "shower", "stair", "generic_symbol", "appliance", "column", "bathtub", "table"]:
        out[f"label_is_{name}"] = 1.0 if row_label == name else 0.0
    for name in ["tiny_le_64", "small_le_256", "medium_le_1024", "large_le_4096", "xlarge_gt_4096"]:
        out[f"area_is_{name}"] = 1.0 if row_area == name else 0.0
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--output-dir", default="datasets/symbol_uncovered_target_actions_v67")
    parser.add_argument("--low-score-threshold", type=float, default=0.005)
    parser.add_argument("--rescue-cluster-rank-cap", type=int, default=4)
    parser.add_argument("--max-actions-per-page", type=int, default=64)
    args = parser.parse_args()

    manifest = json.loads(source_path(args.data).read_text(encoding="utf-8"))
    rows = load_jsonl(source_path(manifest["outputs"]["rows"]))
    pages = group_pages(rows, "all")
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
            centers = center_targets(row)
            bucket = action_bucket(row, base_iou_targets, base_center_targets)
            target_info = meta.get(target, {})
            action = {
                "page_id": page_id,
                "split": split,
                "candidate_id": candidate_id(row),
                "label": label(row),
                "area": candidate_area(row),
                "features": action_features(row, page_rows, base, ranks.get(candidate_id(row), 999)),
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
                    "center_target_count": len(centers),
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
        "version": "symbol_uncovered_target_actions_v67",
        "source_data": rel(source_path(args.data)),
        "base_policy": BASE_POLICY,
        "candidate_policy": policy,
        "outputs": {"rows": rel(rows_path), "pages": rel(pages_path)},
        "summary": dict(summary),
        "source_integrity": {
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "runtime_features": ["candidate score", "predicted label", "bbox area", "cluster/page features", "base selected-set context counts"],
            "offline_labels_used_for": ["uncovered_target_reward_training", "audit"],
        },
    }
    write_json(manifest_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
