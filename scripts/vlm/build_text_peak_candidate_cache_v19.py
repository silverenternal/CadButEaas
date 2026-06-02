#!/usr/bin/env python3
"""Build cached peak/window candidate features for the v19 raster text expert."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from train_text_heatmap_affinity_v19 import (
    DATA,
    REPORT,
    ROOT,
    bbox_iou,
    center_covered,
    fixed_peak_boxes,
    load_jsonl,
    nms,
    predict_maps,
    targets,
    write_json,
    write_jsonl,
)
from train_text_peak_reranker_v19 import FEATURE_NAMES, feature_vector, load_heatmap_model, page_image


OUT = ROOT / "datasets/text_peak_candidate_cache_v19"


def peak_candidates(score: np.ndarray, threshold: float, top_k: int, min_distance: int) -> list[dict[str, Any]]:
    kernel_size = max(3, int(min_distance) | 1)
    dilated = cv2.dilate(score, np.ones((kernel_size, kernel_size), dtype=np.float32))
    peaks = (score >= threshold) & (score >= dilated - 1e-6)
    ys, xs = np.where(peaks)
    if len(xs) == 0:
        return []
    values = score[ys, xs]
    order = np.argsort(values)[::-1][:top_k]
    height, width = score.shape
    out: list[dict[str, Any]] = []
    for rank_index, peak_index in enumerate(order):
        cx, cy = int(xs[peak_index]), int(ys[peak_index])
        confidence = float(values[peak_index])
        for window_index, box in enumerate(fixed_peak_boxes(cx, cy, width, height)):
            x1, y1, x2, y2 = box
            out.append(
                {
                    "bbox": box,
                    "confidence": round(confidence, 6),
                    "peak_rank": int(rank_index),
                    "peak_xy": [cx, cy],
                    "window_index": int(window_index),
                    "area": int((x2 - x1) * (y2 - y1)),
                    "decoder": "local_peak_fixed_multi_cache_source",
                }
            )
    return out


def component_candidates(score: np.ndarray, threshold: float, top_k: int, min_area: int, close_kernel: int, max_area_ratio: float) -> list[dict[str, Any]]:
    binary = (score >= threshold).astype("uint8")
    if close_kernel > 1:
        kernel = np.ones((close_kernel, close_kernel), dtype=np.uint8)
        binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(binary, 8)
    page_area = max(int(binary.shape[0] * binary.shape[1]), 1)
    out: list[dict[str, Any]] = []
    for idx in range(1, int(n_labels)):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < min_area or area / page_area > max_area_ratio:
            continue
        crop = score[y : y + h, x : x + w]
        out.append(
            {
                "bbox": [x, y, x + w, y + h],
                "confidence": round(float(crop.mean()) if crop.size else 0.0, 6),
                "peak_rank": int(top_k),
                "peak_xy": [int(x + w / 2), int(y + h / 2)],
                "window_index": -1,
                "area": int(area),
                "decoder": "connected_component_cache_source",
            }
        )
    return out


def label_and_enrich_candidate(candidate: dict[str, Any], golds: list[dict[str, Any]]) -> dict[str, Any]:
    bbox = [int(v) for v in candidate["bbox"]]
    best_iou = 0.0
    best_gold_index = None
    center_gold_index = None
    for gold_index, gold in enumerate(golds):
        gb = [int(v) for v in gold["bbox"]]
        iou = bbox_iou(bbox, gb)
        if iou > best_iou:
            best_iou = iou
            best_gold_index = gold_index
        if center_gold_index is None and center_covered(bbox, gb):
            center_gold_index = gold_index
    return {
        **candidate,
        "labels": {
            "center_positive": center_gold_index is not None,
            "iou_0_30_positive": best_iou >= 0.30,
            "best_iou": round(float(best_iou), 6),
            "best_gold_index": best_gold_index,
            "center_gold_index": center_gold_index,
        },
    }


def split_rows(split: str, max_rows: int) -> list[dict[str, Any]]:
    rows = load_jsonl(DATA / f"{split}.jsonl")
    rows = [row for row in rows if targets(row)]
    return rows[:max_rows] if max_rows else rows


def build_split(model: Any, split: str, args: argparse.Namespace) -> dict[str, Any]:
    rows = split_rows(split, args.max_rows)
    out_rows: list[dict[str, Any]] = []
    totals = Counter()
    for row_index, row in enumerate(rows, start=1):
        maps = predict_maps(model, row, args.size, args.device)
        score = np.maximum(maps[0], maps[1] * args.affinity_weight)
        ink = page_image(row)
        golds = targets(row)
        candidates = peak_candidates(score, args.threshold, args.peak_top_k, args.peak_min_distance)
        if args.include_components:
            candidates.extend(component_candidates(score, args.threshold, args.peak_top_k, args.min_area, args.close_kernel, args.max_area_ratio))
            candidates = nms(sorted(candidates, key=lambda item: float(item["confidence"]), reverse=True), args.nms_iou)
        enriched = []
        for candidate in candidates:
            labeled = label_and_enrich_candidate(candidate, golds)
            labeled["features"] = {
                name: round(float(value), 6)
                for name, value in zip(FEATURE_NAMES, feature_vector(row, labeled, score, ink, args.peak_top_k))
            }
            enriched.append(labeled)
        center_gold_covered = set(
            int(candidate["labels"]["center_gold_index"])
            for candidate in enriched
            if candidate["labels"]["center_gold_index"] is not None
        )
        iou_gold_covered = set(
            int(candidate["labels"]["best_gold_index"])
            for candidate in enriched
            if candidate["labels"]["iou_0_30_positive"] and candidate["labels"]["best_gold_index"] is not None
        )
        totals["pages"] += 1
        totals["gold"] += len(golds)
        totals["candidates"] += len(enriched)
        totals["center_positive_candidates"] += sum(1 for candidate in enriched if candidate["labels"]["center_positive"])
        totals["iou_positive_candidates"] += sum(1 for candidate in enriched if candidate["labels"]["iou_0_30_positive"])
        totals["center_covered_gold"] += len(center_gold_covered)
        totals["iou_covered_gold"] += len(iou_gold_covered)
        out_rows.append(
            {
                "id": row["source_row_id"],
                "split": split,
                "image": row["image"],
                "gold_text_count": len(golds),
                "candidate_count": len(enriched),
                "center_covered_gold_count": len(center_gold_covered),
                "iou_covered_gold_count": len(iou_gold_covered),
                "candidates": enriched,
                "source_integrity": {
                    "model_input": "raster_image_only",
                    "offline_labels_used_for": ["candidate_cache_labels", "training", "audit"],
                    "gold_used_for_inference": False,
                    "runtime_uses_svg_or_cad_geometry": False,
                },
            }
        )
        if args.progress_every and row_index % args.progress_every == 0:
            print(json.dumps({"split": split, "processed": row_index, "rows": len(rows)}, ensure_ascii=False), flush=True)
    write_jsonl(OUT / f"{split}.jsonl", out_rows)
    return {
        "split": split,
        "rows": int(totals["pages"]),
        "gold": int(totals["gold"]),
        "candidates": int(totals["candidates"]),
        "candidate_inflation": round(totals["candidates"] / max(totals["gold"], 1), 6),
        "center_positive_candidates": int(totals["center_positive_candidates"]),
        "iou_positive_candidates": int(totals["iou_positive_candidates"]),
        "center_candidate_recall_ceiling": round(totals["center_covered_gold"] / max(totals["gold"], 1), 6),
        "iou_0_30_candidate_recall_ceiling": round(totals["iou_covered_gold"] / max(totals["gold"], 1), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits", default="train,dev,locked")
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--affinity-weight", type=float, default=0.65)
    parser.add_argument("--peak-top-k", type=int, default=250)
    parser.add_argument("--peak-min-distance", type=int, default=7)
    parser.add_argument("--include-components", action="store_true")
    parser.add_argument("--min-area", type=int, default=2)
    parser.add_argument("--close-kernel", type=int, default=1)
    parser.add_argument("--max-area-ratio", type=float, default=0.20)
    parser.add_argument("--nms-iou", type=float, default=0.35)
    parser.add_argument("--max-rows", type=int, default=0)
    parser.add_argument("--progress-every", type=int, default=100)
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    model = load_heatmap_model(args.device)
    split_reports = [build_split(model, split.strip(), args) for split in args.splits.split(",") if split.strip()]
    manifest = {
        "version": "text_peak_candidate_cache_v19",
        "task": "P0-TEXT-001",
        "path": str(OUT.relative_to(ROOT)),
        "feature_names": FEATURE_NAMES,
        "config": vars(args),
        "source_integrity": {
            "model_input": "raster_image_only",
            "offline_labels_used_for": ["candidate_cache_labels", "training", "audit"],
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
        },
        "splits": split_reports,
    }
    write_json(OUT / "manifest.json", manifest)
    write_json(REPORT / "text_peak_candidate_cache_v19_audit.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
