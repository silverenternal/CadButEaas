#!/usr/bin/env python3
"""P0-35: runtime-safe stair nearest-context anchor proposal prototype."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

import joblib

from apply_symbol_detector_recall_preserving_policy_v47 import group_pages
from apply_symbol_localization_refiner_p027 import evaluate_against_gold_boxes, load_gold, valid_box
from train_symbol_expanded_action_source_policy_v74 import evaluate_policy, feature_names
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, write_json


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def dims(box: list[float]) -> tuple[float, float]:
    return max(box[2] - box[0], 1.0), max(box[3] - box[1], 1.0)


def make_box(cx: float, cy: float, w: float, h: float) -> list[float]:
    return [cx - w * 0.5, cy - h * 0.5, cx + w * 0.5, cy + h * 0.5]


def dist(a: list[float], b: list[float]) -> float:
    acx, acy = center(a)
    bcx, bcy = center(b)
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def load_stair_priors(tile_rows: str) -> dict[str, Any]:
    values: dict[str, list[tuple[float, float]]] = defaultdict(list)
    for row in load_jsonl(source_path(tile_rows)):
        for target in (row.get("targets") or {}).get("boxes") or []:
            if str(target.get("label") or "") != "stair":
                continue
            box = valid_box(target.get("page_bbox") or target.get("bbox"))
            if box is None:
                continue
            values[str(target.get("area_bucket") or area_bucket(box))].append(dims(box))
    import numpy as np
    priors = {}
    all_vals = []
    for area, vals in values.items():
        all_vals.extend(vals)
        arr = np.asarray(vals, dtype=float)
        priors[area] = [float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))]
    arr = np.asarray(all_vals, dtype=float)
    priors["__global__"] = [float(np.median(arr[:, 0])), float(np.median(arr[:, 1]))]
    return priors


def candidate_sources(rows: list[dict[str, Any]], labels: set[str], score_min: float, topk: int) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        if str(row.get("label") or "") not in labels:
            continue
        if float(row.get("score") or 0.0) < score_min:
            continue
        if valid_box(row.get("bbox")) is None:
            continue
        out.append(row)
    out.sort(key=lambda r: float(r.get("score") or 0.0), reverse=True)
    return out[:topk]


def apply_anchors(
    selected_by_page: dict[str, list[dict[str, Any]]],
    recovery_rows: list[dict[str, Any]],
    split: str,
    source_labels: set[str],
    score_min: float,
    topk: int,
    priors: dict[str, Any],
    prior_areas: list[str],
    scales: list[float],
    max_add_per_page: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    pages = group_pages(recovery_rows, split)
    out = {}
    audit = Counter()
    for page_id, selected in selected_by_page.items():
        selected_ids = {str(row.get("candidate_id") or "") for row in selected}
        sources = [row for row in candidate_sources(pages.get(page_id, []), source_labels, score_min, topk) if str(row.get("candidate_id") or "") not in selected_ids]
        new_rows = list(selected)
        added = 0
        for src in sources:
            if added >= max_add_per_page:
                break
            src_box = valid_box(src.get("bbox"))
            if src_box is None:
                continue
            cx, cy = center(src_box)
            src_w, src_h = dims(src_box)
            for prior_area in prior_areas:
                if added >= max_add_per_page:
                    break
                base = priors.get(prior_area) or priors["__global__"]
                for scale in scales:
                    if added >= max_add_per_page:
                        break
                    w = base[0] * scale
                    h = base[1] * scale
                    # Also avoid pathological anchors much larger than source context unless source is stair.
                    if str(src.get("label") or "") != "stair" and (w > src_w * 4 or h > src_h * 4):
                        continue
                    item = dict(src)
                    item["candidate_id"] = f"{src.get('candidate_id')}_p035_stair_anchor_{prior_area}_{scale}_{added}"
                    item["label"] = "stair"
                    item["bbox"] = make_box(cx, cy, w, h)
                    item["proposal_source"] = "p035_stair_nearest_context_anchor"
                    new_rows.append(item)
                    added += 1
                    audit["added"] += 1
                    audit[f"source_label:{src.get('label')}"] += 1
                    audit[f"prior_area:{prior_area}"] += 1
                    audit[f"scale:{scale}"] += 1
        audit["pages_with_added"] += int(added > 0)
        out[page_id] = new_rows
    return out, dict(audit)


def compact(row: dict[str, Any] | None) -> dict[str, Any] | None:
    if row is None:
        return None
    return {"source_labels": row["source_labels"], "score_min": row["score_min"], "topk": row["topk"], "prior_areas": row["prior_areas"], "scales": row["scales"], "max_add_per_page": row["max_add_per_page"], "delta_matched": row["delta_matched"], "delta_predicted": row["delta_predicted"], "recall": row["metrics"]["symbol_bbox_iou_0_30"]["recall"], "precision": row["metrics"]["symbol_bbox_iou_0_30"]["precision"], "candidate_inflation": row["metrics"]["candidate_inflation"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", default="datasets/symbol_expanded_action_source_policy_v74/actions.jsonl")
    parser.add_argument("--recovery-data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--v74-model", default="checkpoints/symbol_expanded_action_source_policy_v74/model.joblib")
    parser.add_argument("--tile-rows", default="datasets/symbol_tile_detector_tiny_sahi_v21/locked.jsonl")
    parser.add_argument("--split", default="smoke_eval")
    parser.add_argument("--source-label-sets", default="stair;stair,generic_symbol,column;generic_symbol,column")
    parser.add_argument("--score-mins", default="0.0,0.02,0.05")
    parser.add_argument("--topks", default="1,3,5")
    parser.add_argument("--prior-area-sets", default="small_le_256;small_le_256,medium_le_1024;__global__")
    parser.add_argument("--scales", default="0.75,1.0,1.25")
    parser.add_argument("--max-adds", default="1,2,4")
    parser.add_argument("--output", default="reports/vlm/symbol_stair_anchor_p035_eval.json")
    args = parser.parse_args()
    action_rows = load_jsonl(source_path(args.actions))
    manifest = json.loads(source_path(args.recovery_data).read_text(encoding="utf-8"))
    recovery_rows = load_jsonl(source_path(manifest["outputs"]["rows"]))
    bundle = joblib.load(source_path(args.v74_model))
    label_metrics, selected_by_page, route = evaluate_policy(action_rows, recovery_rows, bundle["model"], bundle.get("feature_names") or feature_names(), args.split, 0.04, 10, 8.0, False, 999, None)
    pages = group_pages(recovery_rows, args.split)
    gold_by_page = load_gold(args.tile_rows, set(pages.keys()))
    baseline = evaluate_against_gold_boxes(selected_by_page, gold_by_page)
    priors = load_stair_priors(args.tile_rows)
    reports = []
    for labels_s in [x for x in args.source_label_sets.split(";") if x]:
        labels = {p.strip() for p in labels_s.split(",") if p.strip()}
        for score_min in [float(x) for x in args.score_mins.split(",") if x]:
            for topk in [int(x) for x in args.topks.split(",") if x]:
                for prior_s in [x for x in args.prior_area_sets.split(";") if x]:
                    prior_areas = [p.strip() for p in prior_s.split(",") if p.strip()]
                    for max_add in [int(x) for x in args.max_adds.split(",") if x]:
                        proposed, audit = apply_anchors(selected_by_page, recovery_rows, args.split, labels, score_min, topk, priors, prior_areas, [float(x) for x in args.scales.split(",") if x], max_add)
                        metrics = evaluate_against_gold_boxes(proposed, gold_by_page)
                        reports.append({"source_labels": sorted(labels), "score_min": score_min, "topk": topk, "prior_areas": prior_areas, "scales": [float(x) for x in args.scales.split(",") if x], "max_add_per_page": max_add, "audit": audit, "metrics": metrics, "delta_matched": metrics["symbol_bbox_iou_0_30"]["matched"] - baseline["symbol_bbox_iou_0_30"]["matched"], "delta_predicted": metrics["symbol_bbox_iou_0_30"]["predicted"] - baseline["symbol_bbox_iou_0_30"]["predicted"], "delta_recall": round(metrics["symbol_bbox_iou_0_30"]["recall"] - baseline["symbol_bbox_iou_0_30"]["recall"], 6), "delta_inflation": round(metrics["candidate_inflation"] - baseline["candidate_inflation"], 6)})
    best = max(reports, key=lambda r: (r["delta_matched"], -r["delta_predicted"]), default=None)
    positives = [r for r in reports if r["delta_matched"] > 0]
    best_eff = max(positives, key=lambda r: (r["delta_matched"] / max(r["delta_predicted"], 1), r["delta_matched"]), default=None)
    output = {"version": "symbol_stair_anchor_p035", "split": args.split, "stair_priors": priors, "v74_label_based_metrics": label_metrics, "v74_gold_box_baseline": baseline, "v74_route": route.get("route", {}), "reports": reports, "decision": {"best_by_recall": compact(best), "best_efficiency": compact(best_eff), "recommendation": "freeze_only_if_dev_smoke_tradeoff_beats_shower_and_no_inflation_options"}, "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False, "runtime_features": ["runtime candidate bbox", "candidate score", "predicted label", "candidate area from bbox"], "offline_labels_used_for": ["stair_anchor_prior_selection_and_evaluation"], "final_quality_claim_allowed": False}}
    write_json(source_path(args.output), output)
    print(json.dumps(output["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
