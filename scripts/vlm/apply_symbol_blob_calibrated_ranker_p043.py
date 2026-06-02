#!/usr/bin/env python3
"""P0-43: calibrated per-label ranking for blob-anchor proposal head."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict, deque
from typing import Any

import joblib
import numpy as np
from PIL import Image, ImageFilter, ImageOps

from apply_symbol_localization_refiner_p027 import evaluate_against_gold_boxes, load_gold, valid_box
from train_symbol_expanded_action_source_policy_v74 import base_selected_by_page
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import bbox_iou, write_json

OUTPUT_LABELS = ["equipment", "shower", "sink", "stair"]


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


def downsample_binary(image: Image.Image, max_side: int, threshold: int) -> tuple[np.ndarray, float, float]:
    width, height = image.size
    scale = min(max_side / max(width, height), 1.0)
    sw = max(1, int(round(width * scale)))
    sh = max(1, int(round(height * scale)))
    small = image.resize((sw, sh)).filter(ImageFilter.MinFilter(3))
    arr = np.asarray(small, dtype=np.uint8)
    binary = arr < threshold
    return binary, width / sw, height / sh


def connected_components(binary: np.ndarray, sx: float, sy: float, min_area: int, max_area: int, max_components: int) -> list[list[float]]:
    height, width = binary.shape
    seen = np.zeros_like(binary, dtype=bool)
    boxes: list[list[float]] = []
    for y in range(height):
        for x in range(width):
            if seen[y, x] or not binary[y, x]:
                continue
            seen[y, x] = True
            q = deque([(x, y)])
            minx = maxx = x
            miny = maxy = y
            count = 0
            while q:
                cx, cy = q.popleft()
                count += 1
                minx = min(minx, cx); maxx = max(maxx, cx)
                miny = min(miny, cy); maxy = max(maxy, cy)
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < width and 0 <= ny < height and not seen[ny, nx] and binary[ny, nx]:
                        seen[ny, nx] = True
                        q.append((nx, ny))
            if min_area <= count <= max_area:
                bw = maxx - minx + 1
                bh = maxy - miny + 1
                aspect = bw / max(bh, 1)
                if 0.15 <= aspect <= 8.0:
                    boxes.append([minx * sx, miny * sy, (maxx + 1) * sx, (maxy + 1) * sy])
                    if len(boxes) >= max_components:
                        return boxes
    return boxes


def expand_box(box: list[float], factor: float, width: int, height: int, min_size: float) -> list[float] | None:
    cx = (box[0] + box[2]) * 0.5
    cy = (box[1] + box[3]) * 0.5
    bw = max((box[2] - box[0]) * factor, min_size)
    bh = max((box[3] - box[1]) * factor, min_size)
    out = [max(0.0, cx - bw * 0.5), max(0.0, cy - bh * 0.5), min(float(width), cx + bw * 0.5), min(float(height), cy + bh * 0.5)]
    return out if out[2] > out[0] and out[3] > out[1] else None


def raster_blob_anchors(image: Image.Image, threshold: int, max_side: int, min_area: int, max_area: int, max_components: int, factors: list[float], min_size: float) -> list[list[float]]:
    binary, sx, sy = downsample_binary(image, max_side, threshold)
    components = connected_components(binary, sx, sy, min_area, max_area, max_components)
    anchors = []
    for box in components:
        for factor in factors:
            expanded = expand_box(box, factor, image.size[0], image.size[1], min_size)
            if expanded is not None:
                anchors.append(expanded)
    return anchors


def score_anchors(
    pages: dict[str, list[dict[str, Any]]],
    selected: dict[str, list[dict[str, Any]]],
    image_by_page: dict[str, str],
    bundle: dict[str, Any],
    args: argparse.Namespace,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    binary_model = bundle["binary_model"]
    type_model = bundle["type_model"]
    labels = list(bundle.get("labels") or ["background", "equipment", "shower", "sink", "stair"])
    crop_size = int(bundle.get("crop_size") or 24)
    image_cache: dict[str, Image.Image] = {}
    scored: dict[str, list[dict[str, Any]]] = defaultdict(list)
    audit = Counter()
    factors = [float(x) for x in args.factors.split(",") if x.strip()]
    for page_id in pages:
        image_path = image_by_page.get(page_id)
        if not image_path:
            audit["missing_image"] += 1
            continue
        try:
            image = open_gray(image_path, image_cache)
        except Exception:
            audit["image_open_failed"] += 1
            continue
        anchors = raster_blob_anchors(image, args.binary_threshold, args.max_side, args.min_component_area, args.max_component_area, args.max_components_per_page, factors, args.min_anchor_size)
        if not anchors:
            continue
        base_rows = selected.get(page_id, [])
        kept = []
        for box in anchors:
            if any((valid_box(row.get("bbox")) and bbox_iou(box, valid_box(row.get("bbox"))) >= args.source_nms_iou) for row in base_rows):
                audit["filtered_base_overlap"] += 1
                continue
            kept.append(box)
        audit["generated_anchor"] += len(anchors)
        audit["kept_anchor"] += len(kept)
        pending = []
        for idx, box in enumerate(kept[: args.max_anchors_per_page]):
            try:
                pending.append((idx, box, crop_features(image, box, crop_size)))
            except Exception:
                audit["feature_failed"] += 1
        for start in range(0, len(pending), args.batch_size):
            batch = pending[start:start + args.batch_size]
            try:
                x = np.stack([item[2] for item in batch]).astype(np.float32)
                scores = binary_model.predict_proba(x)[:, 1]
                type_indices = type_model.predict(x)
            except Exception:
                audit["score_failed"] += len(batch)
                continue
            for (idx, box, _features), score, type_idx in zip(batch, scores, type_indices, strict=True):
                label = labels[int(type_idx)] if 0 <= int(type_idx) < len(labels) else "background"
                if label not in OUTPUT_LABELS:
                    audit[f"filtered_type:{label}"] += 1
                    continue
                item = {
                    "page_id": page_id,
                    "split": args.split,
                    "candidate_id": f"{page_id}_p041_blob_{idx}",
                    "bbox": box,
                    "label": label,
                    "score": float(score),
                    "proposal_head_score_p041": float(score),
                    "proposal_head_label_p041": label,
                    "proposal_source": "p041_raster_blob_anchor",
                    "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False},
                }
                scored[page_id].append(item)
                audit["scored_anchor"] += 1
                audit[f"scored_label:{label}"] += 1
    return scored, dict(audit)


def calibrated_score(row: dict[str, Any], label_boost: dict[str, float]) -> float:
    label = str(row.get("label") or "")
    return float(row.get("proposal_head_score_p041") or 0.0) + label_boost.get(label, 0.0)


def parse_label_boost(value: str) -> dict[str, float]:
    out = {}
    for part in value.split(","):
        if not part.strip() or ":" not in part:
            continue
        label, score = part.split(":", 1)
        out[label.strip()] = float(score)
    return out


def merge_topk(
    selected: dict[str, list[dict[str, Any]]],
    scored: dict[str, list[dict[str, Any]]],
    threshold: float,
    max_add_per_page: int,
    global_extra_budget: int,
    nms_iou: float,
    label_boost: dict[str, float],
    per_label_cap: int,
    min_stair_equipment_per_page: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    proposals = []
    audit = Counter()
    priority_labels = {"stair", "equipment"}
    for page_id, base_rows in selected.items():
        rows = []
        label_counts = Counter()
        candidates = sorted(scored.get(page_id, []), key=lambda row: calibrated_score(row, label_boost), reverse=True)
        for pass_name, pool in [("priority", [r for r in candidates if str(r.get("label") or "") in priority_labels]), ("general", candidates)]:
            for item in pool:
                if len(rows) >= max_add_per_page:
                    break
                label = str(item.get("label") or "")
                if pass_name == "general" and label_counts[label] >= per_label_cap:
                    audit[f"filtered_label_cap:{label}"] += 1
                    continue
                if pass_name == "priority" and label_counts[label] >= min_stair_equipment_per_page:
                    continue
                if calibrated_score(item, label_boost) < threshold:
                    audit["filtered_threshold"] += 1
                    continue
                box = valid_box(item.get("bbox"))
                if box is None:
                    continue
                if any((valid_box(prev.get("bbox")) and bbox_iou(box, valid_box(prev.get("bbox"))) >= nms_iou) for prev in base_rows + rows):
                    audit["filtered_nms"] += 1
                    continue
                if item in rows:
                    continue
                rows.append(item)
                label_counts[label] += 1
                audit["eligible_added"] += 1
                audit[f"eligible_label:{label}"] += 1
        proposals.append((sum(calibrated_score(row, label_boost) for row in rows), page_id, rows))
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
    parser.add_argument("--checkpoint", default="checkpoints/symbol_blob_anchor_head_p042/model.joblib")
    parser.add_argument("--split", default="smoke_eval")
    parser.add_argument("--thresholds", default="0.20,0.40,0.60")
    parser.add_argument("--label-boost", default="stair:0.25,equipment:0.15,shower:0.05,sink:0")
    parser.add_argument("--per-label-cap", type=int, default=2)
    parser.add_argument("--min-stair-equipment-per-page", type=int, default=1)
    parser.add_argument("--binary-threshold", type=int, default=110)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--min-component-area", type=int, default=3)
    parser.add_argument("--max-component-area", type=int, default=500)
    parser.add_argument("--max-components-per-page", type=int, default=500)
    parser.add_argument("--max-anchors-per-page", type=int, default=800)
    parser.add_argument("--factors", default="1.4,2.0,3.0")
    parser.add_argument("--min-anchor-size", type=float, default=10.0)
    parser.add_argument("--source-nms-iou", type=float, default=0.80)
    parser.add_argument("--nms-iou", type=float, default=0.70)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--max-add-per-page", type=int, default=4)
    parser.add_argument("--extra-inflation", type=float, default=0.05)
    parser.add_argument("--output", default="reports/vlm/symbol_blob_calibrated_ranker_p043_smoke_eval.json")
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
    scored, score_audit = score_anchors(pages, selected, image_by_page, bundle, args)
    label_boost = parse_label_boost(args.label_boost)
    gold_total = baseline["symbol_bbox_iou_0_30"]["gold"]
    extra_budget = max(0, int(gold_total * args.extra_inflation))
    reports = []
    for threshold in [float(x) for x in args.thresholds.split(",") if x.strip()]:
        proposed, merge_audit = merge_topk(selected, scored, threshold, args.max_add_per_page, extra_budget, args.nms_iou, label_boost, args.per_label_cap, args.min_stair_equipment_per_page)
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
        "version": "symbol_blob_calibrated_ranker_p043_page_eval",
        "split": args.split,
        "baseline_gold_box_metrics": baseline,
        "score_audit": score_audit,
        "label_boost": label_boost,
        "per_label_cap": args.per_label_cap,
        "min_stair_equipment_per_page": args.min_stair_equipment_per_page,
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
            "runtime_features": ["raster page pixels", "connected component/blob anchors", "proposal-head crop scoring", "runtime-only per-label calibration"],
            "offline_labels_used_for": ["page_level_evaluation_only"],
            "metric_mode": "gold-page-bbox recomputed metric; not old label-cache metric",
            "final_quality_claim_allowed": False,
        },
    }
    write_json(source_path(args.output), out)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
