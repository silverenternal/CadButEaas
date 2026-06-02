#!/usr/bin/env python3
"""Train an auditable page-level reranker for v19 raster text heatmap peaks."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from train_text_heatmap_affinity_v19 import (
    CKPT as HEATMAP_CKPT,
    DATA,
    REPORT,
    ROOT,
    HeatmapAffinityNet,
    abs_path,
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


OUT = ROOT / "checkpoints/text_peak_reranker_v19"
FEATURE_NAMES = [
    "candidate_confidence",
    "window_width",
    "window_height",
    "window_area",
    "aspect_ratio",
    "page_x_center",
    "page_y_center",
    "local_mean",
    "local_std",
    "local_max",
    "local_min",
    "local_ink_mean",
    "local_ink_std",
    "local_ink_max",
    "peak_rank_norm",
]


def load_heatmap_model(device: str) -> HeatmapAffinityNet:
    import torch

    checkpoint = torch.load(HEATMAP_CKPT / "model_best.pt", map_location=device)
    model = HeatmapAffinityNet().to(device)
    model.load_state_dict(checkpoint["state_dict"])
    model.eval()
    return model


def page_image(row: dict[str, Any]) -> np.ndarray:
    from PIL import Image

    image = Image.open(abs_path(row["image"])).convert("L")
    return 1.0 - np.asarray(image, dtype=np.float32) / 255.0


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
        for box in fixed_peak_boxes(cx, cy, width, height):
            x1, y1, x2, y2 = box
            out.append(
                {
                    "bbox": box,
                    "confidence": confidence,
                    "peak_rank": rank_index,
                    "cx": cx,
                    "cy": cy,
                    "area": int((x2 - x1) * (y2 - y1)),
                    "decoder": "local_peak_fixed_multi_rerank_source",
                }
            )
    return nms(sorted(out, key=lambda item: item["confidence"], reverse=True), 0.35)


def feature_vector(row: dict[str, Any], candidate: dict[str, Any], score: np.ndarray, ink: np.ndarray, top_k: int) -> list[float]:
    x1, y1, x2, y2 = [int(v) for v in candidate["bbox"]]
    page_h, page_w = score.shape
    crop = score[y1:y2, x1:x2]
    ink_crop = ink[y1:y2, x1:x2]
    w = max(x2 - x1, 1)
    h = max(y2 - y1, 1)
    return [
        float(candidate["confidence"]),
        float(w),
        float(h),
        float(w * h),
        float(w / max(h, 1)),
        float(((x1 + x2) / 2.0) / max(page_w, 1)),
        float(((y1 + y2) / 2.0) / max(page_h, 1)),
        float(crop.mean()) if crop.size else 0.0,
        float(crop.std()) if crop.size else 0.0,
        float(crop.max()) if crop.size else 0.0,
        float(crop.min()) if crop.size else 0.0,
        float(ink_crop.mean()) if ink_crop.size else 0.0,
        float(ink_crop.std()) if ink_crop.size else 0.0,
        float(ink_crop.max()) if ink_crop.size else 0.0,
        float(candidate["peak_rank"] / max(top_k - 1, 1)),
    ]


def label_candidate(candidate: dict[str, Any], golds: list[dict[str, Any]]) -> int:
    bbox = [int(v) for v in candidate["bbox"]]
    return int(any(center_covered(bbox, [int(v) for v in gold["bbox"]]) for gold in golds))


def build_examples(model: HeatmapAffinityNet, rows: list[dict[str, Any]], args: argparse.Namespace, split: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    xs: list[list[float]] = []
    ys: list[int] = []
    audit_rows: list[dict[str, Any]] = []
    for row in rows:
        if not targets(row):
            continue
        maps = predict_maps(model, row, args.size, args.device)
        score = np.maximum(maps[0], maps[1] * args.affinity_weight)
        ink = page_image(row)
        candidates = peak_candidates(score, args.threshold, args.peak_top_k, args.peak_min_distance)
        golds = targets(row)
        labels = []
        for candidate in candidates:
            label = label_candidate(candidate, golds)
            xs.append(feature_vector(row, candidate, score, ink, args.peak_top_k))
            ys.append(label)
            labels.append(label)
        audit_rows.append(
            {
                "id": row["source_row_id"],
                "split": split,
                "gold": len(golds),
                "candidates": len(candidates),
                "positive_candidates": int(sum(labels)),
            }
        )
    if not xs:
        return np.zeros((0, len(FEATURE_NAMES)), dtype=np.float32), np.zeros((0,), dtype=np.int64), audit_rows
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), audit_rows


def evaluate_reranker(model: HeatmapAffinityNet, clf: ExtraTreesClassifier, rows: list[dict[str, Any]], args: argparse.Namespace, split: str) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    totals = Counter()
    pred_rows: list[dict[str, Any]] = []
    for row in rows:
        if not targets(row):
            continue
        maps = predict_maps(model, row, args.size, args.device)
        score = np.maximum(maps[0], maps[1] * args.affinity_weight)
        ink = page_image(row)
        candidates = peak_candidates(score, args.threshold, args.peak_top_k, args.peak_min_distance)
        if candidates:
            x = np.asarray([feature_vector(row, candidate, score, ink, args.peak_top_k) for candidate in candidates], dtype=np.float32)
            probs = clf.predict_proba(x)[:, 1]
            for candidate, prob in zip(candidates, probs):
                candidate["rerank_confidence"] = round(float(prob), 6)
                candidate["confidence"] = round(float(prob), 6)
            candidates = sorted(candidates, key=lambda item: (item["rerank_confidence"], item["confidence"]), reverse=True)[: args.max_candidates_per_page]
        golds = targets(row)
        used: set[int] = set()
        matched_iou = 0
        matched_center = 0
        for gold in golds:
            gb = [int(v) for v in gold["bbox"]]
            best_iou = 0.0
            best_index = None
            center_index = None
            for pred_index, pred in enumerate(candidates):
                if pred_index in used:
                    continue
                iou = bbox_iou(pred["bbox"], gb)
                if iou > best_iou:
                    best_iou = iou
                    best_index = pred_index
                if center_index is None and center_covered(pred["bbox"], gb):
                    center_index = pred_index
            if best_index is not None and best_iou >= 0.30:
                used.add(best_index)
                matched_iou += 1
                matched_center += 1
            elif center_index is not None:
                used.add(center_index)
                matched_center += 1
        totals["gold"] += len(golds)
        totals["predicted"] += len(candidates)
        totals["matched_iou"] += matched_iou
        totals["matched_center"] += matched_center
        pred_rows.append(
            {
                "id": row["source_row_id"],
                "image": row["image"],
                "predicted_text": [
                    {
                        "id": f"{row['source_row_id']}_text_peak_reranker_v19_{idx}",
                        "class": "text",
                        "family": "text",
                        "semantic_type": "unknown_text",
                        "bbox": pred["bbox"],
                        "confidence": pred["confidence"],
                        "proposal_source": "raster_text_peak_reranker_v19",
                        "payload": {"ocr_status": "not_invoked", "source": "raster_text_peak_reranker_v19"},
                    }
                    for idx, pred in enumerate(candidates)
                ],
                "gold_text_count": len(golds),
                "matched_iou_0_30": matched_iou,
                "matched_center": matched_center,
                "source_integrity": {"model_input": "raster_image_only", "gold_used_for_inference": False},
            }
        )
    precision = totals["matched_iou"] / max(totals["predicted"], 1)
    recall = totals["matched_iou"] / max(totals["gold"], 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    report = {
        "split": split,
        "rows": len(pred_rows),
        "threshold": args.threshold,
        "max_candidates_per_page": args.max_candidates_per_page,
        "text_bbox_iou_0_30": {
            "matched": int(totals["matched_iou"]),
            "predicted": int(totals["predicted"]),
            "gold": int(totals["gold"]),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
        },
        "text_bbox_center_recall": round(totals["matched_center"] / max(totals["gold"], 1), 6),
        "candidate_inflation": round(totals["predicted"] / max(totals["gold"], 1), 6),
    }
    return report, pred_rows


def classifier_report(clf: ExtraTreesClassifier, x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if len(set(y.tolist())) < 2:
        return {"examples": int(len(y)), "positive_rate": round(float(y.mean()) if len(y) else 0.0, 6), "roc_auc": None, "average_precision": None}
    prob = clf.predict_proba(x)[:, 1]
    return {
        "examples": int(len(y)),
        "positive": int(y.sum()),
        "positive_rate": round(float(y.mean()), 6),
        "roc_auc": round(float(roc_auc_score(y, prob)), 6),
        "average_precision": round(float(average_precision_score(y, prob)), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--threshold", type=float, default=0.60)
    parser.add_argument("--affinity-weight", type=float, default=0.65)
    parser.add_argument("--peak-top-k", type=int, default=250)
    parser.add_argument("--peak-min-distance", type=int, default=7)
    parser.add_argument("--max-candidates-per-page", type=int, default=55)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--seed", type=int, default=20260510)
    args = parser.parse_args()

    heatmap = load_heatmap_model(args.device)
    train_rows = load_jsonl(DATA / "train.jsonl")
    if args.max_train_rows:
        train_rows = train_rows[: args.max_train_rows]
    dev_rows = load_jsonl(DATA / "dev.jsonl")
    locked_rows = load_jsonl(DATA / "locked.jsonl")

    x_train, y_train, train_audit = build_examples(heatmap, train_rows, args, "train")
    if len(set(y_train.tolist())) < 2:
        raise SystemExit("reranker training needs both positive and negative peak candidates")
    clf = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        random_state=args.seed,
        n_jobs=-1,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
    )
    clf.fit(x_train, y_train)
    dev_report, _dev_predictions = evaluate_reranker(heatmap, clf, dev_rows, args, "dev")
    locked_report, locked_predictions = evaluate_reranker(heatmap, clf, locked_rows, args, "locked")

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "feature_names": FEATURE_NAMES, "args": vars(args)}, OUT / "model.joblib")
    report = {
        "version": "text_peak_reranker_v19_eval",
        "task": "P0-TEXT-001",
        "run_mode": "learned_page_level_peak_window_reranker",
        "source_integrity": {
            "model_input": "raster_image_only",
            "offline_labels_used_for": ["candidate_labeling", "training", "dev_evaluation", "locked_evaluation"],
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
        },
        "training": {
            "checkpoint": str((OUT / "model.joblib").relative_to(ROOT)),
            "heatmap_checkpoint": str((HEATMAP_CKPT / "model_best.pt").relative_to(ROOT)),
            "train_candidate_report": classifier_report(clf, x_train, y_train),
            "train_page_audit": {
                "pages": len(train_audit),
                "mean_candidates_per_page": round(float(np.mean([row["candidates"] for row in train_audit])) if train_audit else 0.0, 6),
                "mean_positive_candidates_per_page": round(float(np.mean([row["positive_candidates"] for row in train_audit])) if train_audit else 0.0, 6),
            },
        },
        "dev": dev_report,
        "locked": locked_report,
        "adopted": locked_report["text_bbox_center_recall"] >= 0.80 and locked_report["candidate_inflation"] <= 5.0,
        "blocker": None
        if locked_report["text_bbox_center_recall"] >= 0.80 and locked_report["candidate_inflation"] <= 5.0
        else "Learned reranker still cannot preserve center recall under the 5x candidate budget; upstream heatmap/ranking supervision must be strengthened.",
    }
    write_json(REPORT / "text_peak_reranker_v19_eval.json", report)
    write_jsonl(REPORT / "text_peak_reranker_v19_locked_predictions.jsonl", locked_predictions)
    print(json.dumps({"locked": locked_report, "adopted": report["adopted"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
