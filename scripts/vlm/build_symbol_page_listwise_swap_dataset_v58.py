#!/usr/bin/env python3
"""Build page-level listwise swap-action data for symbol compression policy training."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from itertools import product
from typing import Any

import joblib
import numpy as np

from apply_symbol_focus_rescue_policy_v52 import candidate_area
from apply_symbol_focus_swap_policy_v54 import group_pages, is_focus_candidate, rescue_candidates
from apply_symbol_sink_tiny_refiner_page_v49 import evaluate, load_gold, score_candidates, select_page, valid_box
from train_symbol_swap_dropper_v56 import vector as dropper_vector
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import bbox_iou, center_covered, rel, write_json, write_jsonl


def best_gold_relation(row: dict[str, Any], gold_map: dict[str, dict[str, Any]]) -> dict[str, Any]:
    box = valid_box(row.get("bbox"))
    if not box:
        return {"best_iou": 0.0, "center_hit": False, "target_id": None, "target_label": None, "target_area": None}
    best_iou = 0.0
    best_gold: dict[str, Any] | None = None
    center_gold: dict[str, Any] | None = None
    for target_id, gold in gold_map.items():
        gold_box = [float(v) for v in gold["bbox"]]
        iou = bbox_iou(box, gold_box)
        if iou > best_iou:
            best_iou = iou
            best_gold = {"target_id": target_id, **gold}
        if center_gold is None and center_covered(box, gold_box):
            center_gold = {"target_id": target_id, **gold}
    target = best_gold or center_gold
    return {
        "best_iou": round(float(best_iou), 6),
        "center_hit": center_gold is not None,
        "target_id": target.get("target_id") if target else None,
        "target_label": target.get("label") if target else None,
        "target_area": target.get("area_bucket") if target else None,
    }


def action_features(
    add_row: dict[str, Any],
    drop_row: dict[str, Any],
    drop_safe_score: float,
) -> dict[str, float]:
    return {
        "add_policy_score": round(float(add_row.get("policy_score") or 0.0), 6),
        "add_rescue_score": round(float(add_row.get("rescue_score") or 0.0), 6),
        "add_detector_score": round(float(add_row.get("score") or 0.0), 6),
        "drop_policy_score": round(float(drop_row.get("policy_score") or 0.0), 6),
        "drop_detector_score": round(float(drop_row.get("score") or 0.0), 6),
        "drop_safe_score": round(float(drop_safe_score), 6),
        "policy_score_delta": round(float(add_row.get("policy_score") or 0.0) - float(drop_row.get("policy_score") or 0.0), 6),
        "same_label": 1.0 if str(add_row.get("label") or "") == str(drop_row.get("label") or "") else 0.0,
        "same_area": 1.0 if candidate_area(add_row) == candidate_area(drop_row) else 0.0,
        "add_is_focus": 1.0 if is_focus_candidate(add_row) else 0.0,
        "drop_is_focus": 1.0 if is_focus_candidate(drop_row) else 0.0,
    }


def reward_for_action(
    add_relation: dict[str, Any],
    drop_relation: dict[str, Any],
    add_label: str,
    drop_label: str,
) -> dict[str, Any]:
    add_iou_hit = float(add_relation["best_iou"]) >= 0.30
    drop_iou_hit = float(drop_relation["best_iou"]) >= 0.30
    add_center = bool(add_relation["center_hit"])
    drop_center = bool(drop_relation["center_hit"])
    add_typed = add_iou_hit and add_label == str(add_relation.get("target_label") or "")
    drop_typed = drop_iou_hit and drop_label == str(drop_relation.get("target_label") or "")
    matched_delta = int(add_iou_hit) - int(drop_iou_hit)
    center_delta = int(add_center) - int(drop_center)
    typed_delta = int(add_typed) - int(drop_typed)
    sink_tiny_gain = int(
        add_iou_hit
        and (
            str(add_relation.get("target_label") or "") in {"sink", "equipment"}
            or str(add_relation.get("target_area") or "") in {"tiny_le_64", "small_le_256"}
        )
    )
    unsafe_drop_penalty = int(drop_iou_hit) + int(drop_center)
    reward = (3.0 * matched_delta) + (0.5 * center_delta) + (0.5 * typed_delta) + (0.75 * sink_tiny_gain) - (2.0 * unsafe_drop_penalty)
    return {
        "reward": round(float(reward), 6),
        "matched_delta": matched_delta,
        "center_delta": center_delta,
        "typed_delta": typed_delta,
        "sink_tiny_gain": sink_tiny_gain,
        "unsafe_drop_penalty": unsafe_drop_penalty,
        "add_iou_hit": add_iou_hit,
        "drop_iou_hit": drop_iou_hit,
        "drop_center_hit": drop_center,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_smoke120/manifest.json")
    parser.add_argument("--smoke-rows", default="datasets/symbol_tile_detector_tiny_sahi_v21/smoke_v30.jsonl")
    parser.add_argument("--suppression-model", default="checkpoints/symbol_support_suppression_v35_p2_transfer_smoke120_t065_c1/model.joblib")
    parser.add_argument("--dropper-model", default="checkpoints/symbol_swap_dropper_v56/model.joblib")
    parser.add_argument("--rescue-model", default="checkpoints/symbol_focus_rescue_reranker_v53/model.joblib")
    parser.add_argument("--split", default="all", choices=["train", "dev", "smoke_eval", "all"])
    parser.add_argument("--selection-threshold", type=float, default=0.65)
    parser.add_argument("--cluster-topk", type=int, default=1)
    parser.add_argument("--max-per-page", type=int, default=120)
    parser.add_argument("--rescue-threshold", type=float, default=0.10)
    parser.add_argument("--max-rescue", type=int, default=5)
    parser.add_argument("--max-drop-candidates", type=int, default=16)
    parser.add_argument("--output-dir", default="datasets/symbol_page_listwise_swap_v58")
    args = parser.parse_args()

    manifest = json.loads(source_path(args.data).read_text(encoding="utf-8"))
    rows = load_jsonl(source_path(manifest["outputs"]["rows"]))
    if args.split != "all":
        rows = [row for row in rows if str(row.get("split") or "") == args.split]
    scored = score_candidates(rows, joblib.load(source_path(args.suppression_model)))
    dropper_pack = joblib.load(source_path(args.dropper_model))
    rescue_pack = joblib.load(source_path(args.rescue_model))
    gold_all = load_gold(source_path(args.smoke_rows))

    output_dir = source_path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, Any]] = []
    pages_out: list[dict[str, Any]] = []
    summary = Counter()
    for page_id, page_rows in group_pages(scored).items():
        gold_map = gold_all.get(page_id, {})
        selected = select_page(page_rows, args.selection_threshold, args.cluster_topk, args.max_per_page)
        additions = rescue_candidates(
            rescue_pack["model"],
            rescue_pack["feature_names"],
            selected,
            page_rows,
            args.rescue_threshold,
            args.max_rescue,
        )
        if not additions:
            continue
        drop_x = np.asarray([dropper_vector(row, dropper_pack["feature_names"]) for row in selected], dtype=np.float32)
        drop_probs = dropper_pack["model"].predict_proba(drop_x)[:, 1] if len(selected) else []
        drop_pool = []
        for row, prob in zip(selected, drop_probs, strict=True):
            item = dict(row)
            item["drop_safe_score"] = float(prob)
            drop_pool.append(item)
        drop_pool.sort(key=lambda row: (float(row.get("drop_safe_score") or 0.0), -float(row.get("policy_score") or 0.0)), reverse=True)
        drop_pool = drop_pool[: args.max_drop_candidates]
        base_metrics = evaluate({page_id: selected}, {page_id: gold_map})
        page_action_count = 0
        for add_row, drop_row in product(additions, drop_pool):
            add_relation = best_gold_relation(add_row, gold_map)
            drop_relation = best_gold_relation(drop_row, gold_map)
            reward = reward_for_action(
                add_relation,
                drop_relation,
                str(add_row.get("label") or ""),
                str(drop_row.get("label") or ""),
            )
            row = {
                "page_id": page_id,
                "split": str(add_row.get("split") or ""),
                "add_candidate_id": add_row.get("candidate_id"),
                "drop_candidate_id": drop_row.get("candidate_id"),
                "add_label": add_row.get("label"),
                "drop_label": drop_row.get("label"),
                "add_area": candidate_area(add_row),
                "drop_area": candidate_area(drop_row),
                "features": action_features(add_row, drop_row, float(drop_row.get("drop_safe_score") or 0.0)),
                "add_relation": add_relation,
                "drop_relation": drop_relation,
                "labels": reward,
                "source_integrity": {
                    "gold_used_for_inference": False,
                    "runtime_uses_svg_or_cad_geometry": False,
                    "offline_labels_used_for": ["action_reward_training", "audit"],
                },
            }
            rows_out.append(row)
            page_action_count += 1
            summary["actions"] += 1
            if reward["reward"] > 0:
                summary["positive_reward_actions"] += 1
            if reward["drop_iou_hit"]:
                summary["drop_iou_hit_actions"] += 1
            if reward["drop_center_hit"]:
                summary["drop_center_hit_actions"] += 1
            if reward["add_iou_hit"]:
                summary["add_iou_hit_actions"] += 1
        pages_out.append(
            {
                "page_id": page_id,
                "split": str(selected[0].get("split") or "") if selected else "",
                "selected_count": len(selected),
                "addition_count": len(additions),
                "drop_pool_count": len(drop_pool),
                "action_count": page_action_count,
                "baseline_metrics": base_metrics,
            }
        )
        summary["pages"] += 1
        summary["selected_candidates"] += len(selected)
        summary["addition_candidates"] += len(additions)
        summary["drop_pool_candidates"] += len(drop_pool)

    rows_path = output_dir / "rows.jsonl"
    pages_path = output_dir / "pages.jsonl"
    manifest_path = output_dir / "manifest.json"
    write_jsonl(rows_path, rows_out)
    write_jsonl(pages_path, pages_out)
    report = {
        "version": "symbol_page_listwise_swap_v58",
        "source_integrity": {
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "offline_labels_used_for": ["action_reward_training", "audit"],
            "final_quality_claim_allowed": False,
        },
        "source_data": rel(source_path(args.data)),
        "split": args.split,
        "outputs": {"rows": rel(rows_path), "pages": rel(pages_path)},
        "summary": dict(summary),
        "config": vars(args),
        "next_step": "Train a page-level/listwise policy on action rows, then evaluate one action set per page without using gold at inference.",
    }
    write_json(manifest_path, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
