#!/usr/bin/env python3
"""P0-44: train runtime-safe learned ranker for blob-anchor candidates."""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from apply_symbol_raster_blob_anchors_p041 import crop_features, open_gray
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import write_json

LABELS = ["background", "equipment", "shower", "sink", "stair"]


def valid_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    box = [float(v) for v in value]
    return box if box[2] > box[0] and box[3] > box[1] else None


def box_features(box: list[float]) -> list[float]:
    w = max(box[2] - box[0], 1.0)
    h = max(box[3] - box[1], 1.0)
    area = w * h
    return [w, h, area, w / max(h, 1.0), math.log1p(w), math.log1p(h), math.log1p(area), (box[0] + box[2]) * 0.5 / 2000.0, (box[1] + box[3]) * 0.5 / 2000.0]


def runtime_vector(box: list[float], blob_score: float, blob_label: str) -> list[float]:
    values = box_features(box)
    values.append(float(blob_score))
    values.extend([1.0 if blob_label == name else 0.0 for name in LABELS])
    return values


def score_rows(rows: list[dict[str, Any]], blob_bundle: dict[str, Any], batch_size: int) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    binary_model = blob_bundle["binary_model"]
    type_model = blob_bundle["type_model"]
    labels = list(blob_bundle.get("labels") or LABELS)
    crop_size = int(blob_bundle.get("crop_size") or 24)
    image_cache = {}
    prepared = []
    for row in rows:
        box = valid_box(row.get("page_bbox") or row.get("bbox_in_tile") or row.get("bbox"))
        image_path = str(row.get("image") or "")
        if box is None or not image_path:
            continue
        try:
            image = open_gray(image_path, image_cache)
            prepared.append((row, box, crop_features(image, box, crop_size)))
        except Exception:
            continue
    xs = []
    ys = []
    kept = []
    for start in range(0, len(prepared), batch_size):
        batch = prepared[start:start + batch_size]
        x_crop = np.stack([item[2] for item in batch]).astype(np.float32)
        scores = binary_model.predict_proba(x_crop)[:, 1]
        type_ids = type_model.predict(x_crop)
        for (row, box, _features), score, type_id in zip(batch, scores, type_ids, strict=True):
            pred_label = labels[int(type_id)] if 0 <= int(type_id) < len(labels) else "background"
            xs.append(runtime_vector(box, float(score), pred_label))
            ys.append(1 if row.get("is_positive") else 0)
            item = dict(row)
            item["blob_head_score"] = float(score)
            item["blob_head_label"] = pred_label
            kept.append(item)
    return np.asarray(xs, dtype=np.float32), np.asarray(ys, dtype=np.int64), kept


def metrics(y: np.ndarray, score: np.ndarray) -> dict[str, Any]:
    out = {"rows": int(y.size), "positives": int(y.sum()), "negatives": int((1 - y).sum())}
    if y.size and len(set(y.tolist())) > 1:
        out["average_precision"] = float(average_precision_score(y, score))
        out["roc_auc"] = float(roc_auc_score(y, score))
    order = np.argsort(-score)
    for ratio in [1.0, 2.0, 4.0, 8.0]:
        k = min(y.size, max(1, int(max(y.sum(), 1) * ratio)))
        idx = order[:k]
        out[f"recall_at_{ratio:.1f}x_pos_budget"] = float(y[idx].sum() / max(y.sum(), 1))
        out[f"precision_at_{ratio:.1f}x_pos_budget"] = float(y[idx].mean())
    return out


def load_split(dataset_dir: str, split: str) -> list[dict[str, Any]]:
    return load_jsonl(source_path(f"{dataset_dir}/{split}.jsonl"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/symbol_blob_anchor_head_p042")
    parser.add_argument("--blob-head", default="checkpoints/symbol_blob_anchor_head_p042/model.joblib")
    parser.add_argument("--checkpoint", default="checkpoints/symbol_blob_anchor_ranker_p044/model.joblib")
    parser.add_argument("--report", default="reports/vlm/symbol_blob_anchor_ranker_p044_eval.json")
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260514)
    args = parser.parse_args()

    blob_bundle = joblib.load(source_path(args.blob_head))
    train_rows = load_split(args.dataset_dir, "train")
    x_train, y_train, kept_train = score_rows(train_rows, blob_bundle, args.batch_size)
    model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(max_iter=180, learning_rate=0.05, l2_regularization=0.05, class_weight={0: 1.0, 1: 5.0}, random_state=args.seed)),
    ])
    model.fit(x_train, y_train)
    report = {
        "version": "symbol_blob_anchor_ranker_p044_runtime_safe",
        "dataset_dir": args.dataset_dir,
        "blob_head": args.blob_head,
        "train_rows": len(kept_train),
        "train_positive_rows": int(y_train.sum()),
        "feature_count": int(x_train.shape[1]),
        "splits": {},
        "source_integrity": {
            "runtime_features": ["blob anchor bbox geometry", "blob-head predicted label", "blob-head score"],
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "offline_labels_used_for": ["ranker supervised target only"],
            "forbidden_runtime_features": ["best_iou", "gold label", "gold area_bucket", "target_id"],
            "metric_mode": "anchor-level ranker diagnostic; not page-level final quality",
        },
    }
    for split in ["smoke_eval", "dev"]:
        rows = load_split(args.dataset_dir, split)
        x, y, kept = score_rows(rows, blob_bundle, args.batch_size)
        score = model.predict_proba(x)[:, 1] if len(kept) else np.asarray([])
        report["splits"][split] = metrics(y, score)
    ckpt = source_path(args.checkpoint)
    ckpt.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": LABELS, "feature_count": int(x_train.shape[1]), "source_integrity": report["source_integrity"]}, ckpt)
    report["checkpoint"] = str(ckpt.relative_to(source_path('.')))
    write_json(source_path(args.report), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
