#!/usr/bin/env python3
"""P0-31: runtime-safe shrink-box subcandidate proposal prototype."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

import joblib

from apply_symbol_detector_recall_preserving_policy_v47 import group_pages
from apply_symbol_localization_refiner_p027 import evaluate_against_gold_boxes, load_gold, valid_box
from train_symbol_expanded_action_source_policy_v74 import evaluate_policy, feature_names
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import area_bucket, write_json

FOCUS_LABELS = {"sink", "shower", "equipment", "stair"}


def box_area(box: list[float]) -> float:
    return max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)


def shrink_box(box: list[float], factor: float) -> list[float]:
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    w = max(box[2] - box[0], 1.0) * factor
    h = max(box[3] - box[1], 1.0) * factor
    return [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]


def should_route(row: dict[str, Any], gate: str) -> bool:
    box = valid_box(row.get("bbox"))
    if box is None:
        return False
    label = str(row.get("label") or "")
    score = float(row.get("score") or row.get("expanded_action_score_v74") or 0.0)
    area = box_area(box)
    if gate == "focus_medium_large_low_score":
        return label in FOCUS_LABELS and area > 256.0 and score < 0.25
    if gate == "sink_medium_large_low_score":
        return label == "sink" and area > 256.0 and score < 0.25
    if gate == "sink_low_score":
        return label == "sink" and score < 0.25
    if gate == "focus_medium_large":
        return label in FOCUS_LABELS and area > 256.0
    if gate == "all_medium_large_low_score":
        return area > 256.0 and score < 0.25
    return False


def apply_shrink(selected_by_page: dict[str, list[dict[str, Any]]], gate: str, factors: list[float], mode: str, max_add_per_page: int) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    out: dict[str, list[dict[str, Any]]] = {}
    audit = Counter()
    for page_id, rows in selected_by_page.items():
        routed = [row for row in rows if should_route(row, gate)]
        audit["routed"] += len(routed)
        new_rows = []
        if mode == "replace":
            for row in rows:
                box = valid_box(row.get("bbox"))
                if box is not None and should_route(row, gate):
                    for factor in factors[:1]:
                        item = dict(row)
                        item["original_bbox"] = box
                        item["bbox"] = shrink_box(box, factor)
                        item["proposal_source"] = f"p031_shrink_replace_{factor}"
                        audit["replaced"] += 1
                        audit[f"replaced_label:{item.get('label')}"] += 1
                        audit[f"replaced_area:{area_bucket(box)}"] += 1
                    new_rows.append(item)
                else:
                    new_rows.append(row)
        else:
            new_rows = list(rows)
            added = 0
            for row in routed:
                if added >= max_add_per_page:
                    break
                box = valid_box(row.get("bbox"))
                if box is None:
                    continue
                for factor in factors:
                    if added >= max_add_per_page:
                        break
                    item = dict(row)
                    item["candidate_id"] = f"{row.get('candidate_id')}_p031_shrink_{factor}"
                    item["original_bbox"] = box
                    item["bbox"] = shrink_box(box, factor)
                    item["proposal_source"] = f"p031_shrink_add_{factor}"
                    new_rows.append(item)
                    added += 1
                    audit["added"] += 1
                    audit[f"added_label:{item.get('label')}"] += 1
                    audit[f"added_area:{area_bucket(box)}"] += 1
            audit["pages_with_added"] += int(added > 0)
        out[page_id] = new_rows
    return out, dict(audit)


def eval_config(selected_by_page: dict[str, list[dict[str, Any]]], gold_by_page: dict[str, dict[str, dict[str, Any]]], gate: str, factors: list[float], mode: str, max_add_per_page: int, baseline: dict[str, Any]) -> dict[str, Any]:
    proposed, audit = apply_shrink(selected_by_page, gate, factors, mode, max_add_per_page)
    metrics = evaluate_against_gold_boxes(proposed, gold_by_page)
    return {
        "gate": gate,
        "factors": factors,
        "mode": mode,
        "max_add_per_page": max_add_per_page,
        "audit": audit,
        "metrics": metrics,
        "delta_matched": metrics["symbol_bbox_iou_0_30"]["matched"] - baseline["symbol_bbox_iou_0_30"]["matched"],
        "delta_recall": round(metrics["symbol_bbox_iou_0_30"]["recall"] - baseline["symbol_bbox_iou_0_30"]["recall"], 6),
        "delta_predicted": metrics["symbol_bbox_iou_0_30"]["predicted"] - baseline["symbol_bbox_iou_0_30"]["predicted"],
        "delta_inflation": round(metrics["candidate_inflation"] - baseline["candidate_inflation"], 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", default="datasets/symbol_expanded_action_source_policy_v74/actions.jsonl")
    parser.add_argument("--recovery-data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--v74-model", default="checkpoints/symbol_expanded_action_source_policy_v74/model.joblib")
    parser.add_argument("--tile-rows", default="datasets/symbol_tile_detector_tiny_sahi_v21/locked.jsonl")
    parser.add_argument("--split", default="smoke_eval")
    parser.add_argument("--gates", default="sink_medium_large_low_score,focus_medium_large_low_score,all_medium_large_low_score")
    parser.add_argument("--factor-sets", default="0.15;0.25;0.35;0.5;0.25,0.35;0.15,0.25,0.35,0.5")
    parser.add_argument("--modes", default="replace,add")
    parser.add_argument("--max-add-per-page", type=int, default=8)
    parser.add_argument("--output", default="reports/vlm/symbol_shrink_box_proposals_p031_eval.json")
    args = parser.parse_args()
    action_rows = load_jsonl(source_path(args.actions))
    recovery_manifest = json.loads(source_path(args.recovery_data).read_text(encoding="utf-8"))
    recovery_rows = load_jsonl(source_path(recovery_manifest["outputs"]["rows"]))
    bundle = joblib.load(source_path(args.v74_model))
    metrics, selected_by_page, route = evaluate_policy(action_rows, recovery_rows, bundle["model"], bundle.get("feature_names") or feature_names(), args.split, 0.04, 10, 8.0, False, 999, None)
    pages = group_pages(recovery_rows, args.split)
    gold_by_page = load_gold(args.tile_rows, set(pages.keys()))
    baseline = evaluate_against_gold_boxes(selected_by_page, gold_by_page)
    reports = []
    factor_sets = [[float(x) for x in part.split(",") if x] for part in args.factor_sets.split(";") if part]
    for mode in [m.strip() for m in args.modes.split(",") if m.strip()]:
        for gate in [g.strip() for g in args.gates.split(",") if g.strip()]:
            for factors in factor_sets:
                reports.append(eval_config(selected_by_page, gold_by_page, gate, factors, mode, args.max_add_per_page, baseline))
    best_by_recall = max(reports, key=lambda r: (r["delta_matched"], -r["delta_predicted"]), default=None)
    best_no_inflation = max([r for r in reports if r["delta_predicted"] == 0], key=lambda r: r["delta_matched"], default=None)
    output = {
        "version": "symbol_shrink_box_proposals_p031",
        "split": args.split,
        "v74_label_based_metrics": metrics,
        "v74_gold_box_baseline": baseline,
        "v74_route": route.get("route", {}),
        "reports": reports,
        "decision": {"best_by_recall": compact(best_by_recall), "best_no_inflation": compact(best_no_inflation), "recommendation": "prefer_no_inflation_replace_if_positive_else_additive_requires_budget_policy"},
        "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False, "runtime_features": ["selected candidate bbox", "candidate score", "predicted label", "candidate area from bbox"], "offline_labels_used_for": ["proposal_prototype_evaluation"], "final_quality_claim_allowed": False},
    }
    write_json(source_path(args.output), output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


def compact(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {"gate": row["gate"], "factors": row["factors"], "mode": row["mode"], "delta_matched": row["delta_matched"], "delta_predicted": row["delta_predicted"], "recall": row["metrics"]["symbol_bbox_iou_0_30"]["recall"], "precision": row["metrics"]["symbol_bbox_iou_0_30"]["precision"], "candidate_inflation": row["metrics"]["candidate_inflation"]}


if __name__ == "__main__":
    main()
