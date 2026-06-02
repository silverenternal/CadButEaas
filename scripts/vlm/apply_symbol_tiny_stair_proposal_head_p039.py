#!/usr/bin/env python3
"""P0-39: apply the P0-37/38 tiny-stair proposal head to page-level anchors."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image, ImageOps

from apply_symbol_localization_refiner_p027 import evaluate_against_gold_boxes, load_gold, valid_box
from apply_symbol_shrink_box_proposals_p031 import shrink_box
from train_symbol_expanded_action_source_policy_v74 import base_selected_by_page, candidate_id
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import bbox_iou, write_json

FOCUS_LABELS = {"sink", "shower", "equipment", "stair", "generic_symbol", "column"}
OUTPUT_LABELS = ["equipment", "shower", "sink", "stair"]


def row_score(row: dict[str, Any]) -> float:
    try:
        return float(row.get("score") or row.get("selector_score") or row.get("pre_selector_score") or 0.0)
    except Exception:
        return 0.0


def box_area(box: list[float]) -> float:
    return max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)


def page_image_map(tile_rows: str, split_pages: set[str]) -> dict[str, str]:
    out: dict[str, str] = {}
    for row in load_jsonl(source_path(tile_rows)):
        page_id = str(row.get("row_id") or "")
        if page_id in split_pages and page_id not in out and row.get("image"):
            out[page_id] = str(row.get("image"))
    return out


def open_gray(path: str, cache: dict[str, Image.Image]) -> Image.Image:
    if path not in cache:
        cache[path] = ImageOps.grayscale(Image.open(source_path(path))).copy()
    return cache[path]


def crop_features(image: Image.Image, box: list[float], crop_size: int) -> np.ndarray:
    width, height = image.size
    x1, y1, x2, y2 = box
    bw = max(x2 - x1, 1.0)
    bh = max(y2 - y1, 1.0)
    pad = 0.35 * max(bw, bh)
    crop_box = (
        max(0, int(math.floor(x1 - pad))),
        max(0, int(math.floor(y1 - pad))),
        min(width, int(math.ceil(x2 + pad))),
        min(height, int(math.ceil(y2 + pad))),
    )
    if crop_box[2] <= crop_box[0] or crop_box[3] <= crop_box[1]:
        arr = np.zeros((crop_size, crop_size), dtype=np.float32)
    else:
        arr = np.asarray(image.crop(crop_box).resize((crop_size, crop_size)), dtype=np.float32) / 255.0
    gx = np.abs(np.diff(arr, axis=1)).mean() if arr.shape[1] > 1 else 0.0
    gy = np.abs(np.diff(arr, axis=0)).mean() if arr.shape[0] > 1 else 0.0
    geom = np.array([
        bw / max(width, 1),
        bh / max(height, 1),
        (bw * bh) / max(width * height, 1),
        (x1 + x2) * 0.5 / max(width, 1),
        (y1 + y2) * 0.5 / max(height, 1),
        bw / max(bh, 1.0),
        float(arr.mean()),
        float(arr.std()),
        float(gx),
        float(gy),
    ], dtype=np.float32)
    return np.concatenate([arr.reshape(-1), geom])


def anchor_boxes(row: dict[str, Any], factors: list[float]) -> list[tuple[str, list[float]]]:
    box = valid_box(row.get("bbox"))
    if box is None:
        return []
    out = []
    for factor in factors:
        out.append((f"shrink_{factor}", shrink_box(box, factor)))
    return out


def build_scored_anchors(
    pages: dict[str, list[dict[str, Any]]],
    selected: dict[str, list[dict[str, Any]]],
    image_by_page: dict[str, str],
    bundle: dict[str, Any],
    factors: list[float],
    max_source_per_page: int,
    min_source_score: float,
    min_area: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    binary_model = bundle["binary_model"]
    type_model = bundle["type_model"]
    labels = list(bundle.get("labels") or ["background", "equipment", "shower", "sink", "stair"])
    crop_size = int(bundle.get("crop_size") or 24)
    image_cache: dict[str, Image.Image] = {}
    scored: dict[str, list[dict[str, Any]]] = defaultdict(list)
    audit = Counter()
    for page_id, rows in pages.items():
        image_path = image_by_page.get(page_id)
        if not image_path:
            audit["missing_image"] += 1
            continue
        selected_ids = {candidate_id(row) for row in selected.get(page_id, [])}
        sources = [row for row in rows if candidate_id(row) not in selected_ids and str(row.get("label") or "") in FOCUS_LABELS and row_score(row) >= min_source_score and (valid_box(row.get("bbox")) and box_area(valid_box(row.get("bbox"))) >= min_area)]
        sources.sort(key=row_score, reverse=True)
        try:
            image = open_gray(image_path, image_cache)
        except Exception:
            audit["image_open_failed"] += 1
            continue
        for row in sources[:max_source_per_page]:
            for anchor_kind, box in anchor_boxes(row, factors):
                try:
                    x = crop_features(image, box, crop_size).reshape(1, -1).astype(np.float32)
                    score = float(binary_model.predict_proba(x)[0, 1])
                    type_idx = int(type_model.predict(x)[0])
                except Exception:
                    audit["score_failed"] += 1
                    continue
                label = labels[type_idx] if 0 <= type_idx < len(labels) else str(row.get("label") or "generic_symbol")
                if label not in OUTPUT_LABELS:
                    audit[f"filtered_type:{label}"] += 1
                    continue
                item = dict(row)
                item["candidate_id"] = f"{candidate_id(row)}_p039_{anchor_kind}"
                item["bbox"] = box
                item["label"] = label
                item["score"] = score
                item["proposal_head_score_p039"] = score
                item["proposal_head_label_p039"] = label
                item["proposal_source"] = f"p039_proposal_head_{anchor_kind}"
                item["source_candidate_id"] = candidate_id(row)
                scored[page_id].append(item)
                audit["scored_anchor"] += 1
                audit[f"scored_label:{label}"] += 1
                audit[f"source_label:{row.get('label')}"] += 1
    return scored, dict(audit)


def merge_topk(
    selected: dict[str, list[dict[str, Any]]],
    scored: dict[str, list[dict[str, Any]]],
    threshold: float,
    max_add_per_page: int,
    global_extra_budget: int,
    nms_iou: float,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    proposals = []
    audit = Counter()
    for page_id, base_rows in selected.items():
        rows = []
        for item in sorted(scored.get(page_id, []), key=lambda row: float(row.get("proposal_head_score_p039") or 0.0), reverse=True):
            if float(item.get("proposal_head_score_p039") or 0.0) < threshold:
                audit["filtered_threshold"] += 1
                continue
            box = valid_box(item.get("bbox"))
            if box is None:
                continue
            if any((valid_box(prev.get("bbox")) and bbox_iou(box, valid_box(prev.get("bbox"))) >= nms_iou) for prev in base_rows + rows):
                audit["filtered_nms"] += 1
                continue
            rows.append(item)
            audit["eligible_added"] += 1
            audit[f"eligible_label:{item.get('label')}"] += 1
            if len(rows) >= max_add_per_page:
                break
        proposals.append((sum(float(row.get("proposal_head_score_p039") or 0.0) for row in rows), page_id, rows))
    out = {page_id: list(rows) for page_id, rows in selected.items()}
    used = 0
    for _score_sum, page_id, added in sorted(proposals, reverse=True):
        if used + len(added) <= global_extra_budget:
            out[page_id] = out.get(page_id, []) + added
            used += len(added)
            audit["added"] += len(added)
            audit["pages_with_added"] += int(bool(added))
        else:
            audit["skipped_global_budget"] += len(added)
    audit["used_extra_budget"] = used
    audit["global_extra_budget"] = global_extra_budget
    return out, dict(audit)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--tile-rows", default="datasets/symbol_tile_detector_tiny_sahi_v21/locked.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/symbol_tiny_stair_proposal_head_p037/model.joblib")
    parser.add_argument("--split", default="smoke_eval")
    parser.add_argument("--thresholds", default="0.80,0.90,0.95")
    parser.add_argument("--factors", default="0.25,0.35,0.50")
    parser.add_argument("--max-source-per-page", type=int, default=80)
    parser.add_argument("--max-add-per-page", type=int, default=10)
    parser.add_argument("--extra-inflation", type=float, default=0.50)
    parser.add_argument("--min-source-score", type=float, default=0.005)
    parser.add_argument("--min-area", type=float, default=64.0)
    parser.add_argument("--nms-iou", type=float, default=0.85)
    parser.add_argument("--output", default="reports/vlm/symbol_tiny_stair_proposal_head_p039_smoke_eval.json")
    args = parser.parse_args()

    recovery_manifest = json.loads(source_path(args.recovery_data).read_text())
    recovery_rows = load_jsonl(source_path(recovery_manifest["outputs"]["rows"]))
    pages: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in recovery_rows:
        if str(row.get("split") or "") == args.split:
            pages[str(row.get("page_id") or "")].append(row)
    pages = dict(pages)
    selected = base_selected_by_page(recovery_rows, args.split)
    split_pages = set(pages)
    gold_by_page = load_gold(args.tile_rows, split_pages)
    image_by_page = page_image_map(args.tile_rows, split_pages)
    baseline = evaluate_against_gold_boxes(selected, gold_by_page)
    bundle = joblib.load(source_path(args.checkpoint))
    factors = [float(x) for x in args.factors.split(",") if x.strip()]
    scored, score_audit = build_scored_anchors(pages, selected, image_by_page, bundle, factors, args.max_source_per_page, args.min_source_score, args.min_area)
    gold_total = baseline["symbol_bbox_iou_0_30"]["gold"]
    extra_budget = max(0, int(gold_total * args.extra_inflation))
    reports = []
    for threshold in [float(x) for x in args.thresholds.split(",") if x.strip()]:
        proposed, merge_audit = merge_topk(selected, scored, threshold, args.max_add_per_page, extra_budget, args.nms_iou)
        metrics = evaluate_against_gold_boxes(proposed, gold_by_page)
        reports.append({
            "threshold": threshold,
            "audit": merge_audit,
            "metrics": metrics,
            "delta_matched": metrics["symbol_bbox_iou_0_30"]["matched"] - baseline["symbol_bbox_iou_0_30"]["matched"],
            "delta_predicted": metrics["symbol_bbox_iou_0_30"]["predicted"] - baseline["symbol_bbox_iou_0_30"]["predicted"],
            "delta_recall": round(metrics["symbol_bbox_iou_0_30"]["recall"] - baseline["symbol_bbox_iou_0_30"]["recall"], 6),
            "delta_inflation": round(metrics["candidate_inflation"] - baseline["candidate_inflation"], 6),
        })
    best = max(reports, key=lambda row: (row["delta_matched"], -row["delta_predicted"]), default=None)
    out = {
        "version": "symbol_tiny_stair_proposal_head_p039_page_eval",
        "split": args.split,
        "baseline_gold_box_metrics": baseline,
        "score_audit": score_audit,
        "reports": reports,
        "decision": {
            "best_threshold": (best or {}).get("threshold"),
            "best_delta_matched": (best or {}).get("delta_matched"),
            "best_delta_predicted": (best or {}).get("delta_predicted"),
            "recommendation": "promote_to_dev_only_if_smoke_delta_matched_positive_with_reasonable_candidate_inflation",
        },
        "source_integrity": {
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "runtime_features": ["raster crop pixels", "runtime recovery candidate boxes", "runtime detector score/label"],
            "offline_labels_used_for": ["page_level_evaluation_only"],
            "metric_mode": "gold-page-bbox recomputed metric; not old label-cache metric",
            "final_quality_claim_allowed": False,
        },
    }
    write_json(source_path(args.output), out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
