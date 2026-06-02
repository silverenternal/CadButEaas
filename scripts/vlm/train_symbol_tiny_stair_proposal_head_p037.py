#!/usr/bin/env python3
"""Train/evaluate a minimal raster+anchor proposal-head baseline for P0-37."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_curve, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import write_json

LABELS = ["background", "equipment", "shower", "sink", "stair"]
LABEL_TO_ID = {label: idx for idx, label in enumerate(LABELS)}


def valid_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    box = [float(v) for v in value]
    if not all(math.isfinite(v) for v in box):
        return None
    return box if box[2] > box[0] and box[3] > box[1] else None


def load_rows(path: str, limit: int, seed: int) -> list[dict[str, Any]]:
    rows = [row for row in load_jsonl(source_path(path)) if valid_box(row.get("page_bbox"))]
    rng = random.Random(seed)
    rng.shuffle(rows)
    return rows[:limit] if limit and len(rows) > limit else rows


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


def featurize(rows: list[dict[str, Any]], crop_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    cache: dict[str, Image.Image] = {}
    xs: list[np.ndarray] = []
    y_binary: list[int] = []
    y_type: list[int] = []
    kept: list[dict[str, Any]] = []
    for row in rows:
        box = valid_box(row.get("page_bbox"))
        image_path = str(row.get("image") or "")
        if box is None or not image_path:
            continue
        try:
            image = open_gray(image_path, cache)
            xs.append(crop_features(image, box, crop_size))
        except Exception:
            continue
        label = str(row.get("label") or "background")
        is_positive = bool(row.get("is_positive"))
        y_binary.append(1 if is_positive else 0)
        y_type.append(LABEL_TO_ID.get(label, 0) if is_positive else 0)
        kept.append(row)
    if not xs:
        return np.empty((0, crop_size * crop_size + 10), dtype=np.float32), np.array([]), np.array([]), []
    return np.stack(xs).astype(np.float32), np.asarray(y_binary, dtype=np.int64), np.asarray(y_type, dtype=np.int64), kept


def binary_metrics(y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"rows": int(y_true.size), "positives": int(y_true.sum()), "negatives": int((1 - y_true).sum())}
    if y_true.size == 0 or len(set(y_true.tolist())) < 2:
        return out
    out["average_precision"] = float(average_precision_score(y_true, scores))
    out["roc_auc"] = float(roc_auc_score(y_true, scores))
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    for floor in [0.80, 0.90, 0.95]:
        valid = np.where(recall[:-1] >= floor)[0]
        if valid.size:
            best = valid[np.argmax(precision[:-1][valid])]
            out[f"precision_at_recall_{floor:.2f}"] = float(precision[best])
            out[f"threshold_at_recall_{floor:.2f}"] = float(thresholds[best])
    order = np.argsort(-scores)
    for ratio in [1.0, 2.0, 4.0, 8.0]:
        k = min(y_true.size, max(1, int(max(y_true.sum(), 1) * ratio)))
        selected = order[:k]
        out[f"recall_at_{ratio:.1f}x_pos_budget"] = float(y_true[selected].sum() / max(y_true.sum(), 1))
        out[f"precision_at_{ratio:.1f}x_pos_budget"] = float(y_true[selected].mean())
    return out


def grouped_metrics(rows: list[dict[str, Any]], y_true: np.ndarray, scores: np.ndarray) -> dict[str, Any]:
    groups: dict[str, list[int]] = defaultdict(list)
    for idx, row in enumerate(rows):
        if not bool(row.get("is_positive")):
            continue
        groups[f"label:{row.get('label')}"] .append(idx)
        groups[f"area:{row.get('area_bucket')}"] .append(idx)
        groups[f"role:{row.get('sample_role')}"] .append(idx)
    out = {}
    order = np.argsort(-scores)
    selected_4x = set(order[: min(y_true.size, max(1, int(max(y_true.sum(), 1) * 4)))].tolist())
    for name, idxs in sorted(groups.items()):
        if not idxs:
            continue
        hit = sum(1 for i in idxs if i in selected_4x)
        out[name] = {"positives": len(idxs), "recall_at_4x_pos_budget": hit / len(idxs)}
    return out


def type_metrics(y_true_binary: np.ndarray, y_type: np.ndarray, type_pred: np.ndarray) -> dict[str, Any]:
    pos = np.where(y_true_binary == 1)[0]
    out = {"positive_rows": int(pos.size)}
    if pos.size:
        out["positive_type_accuracy"] = float((type_pred[pos] == y_type[pos]).mean())
        counts = Counter()
        correct = Counter()
        for i in pos:
            label = LABELS[int(y_type[i])]
            counts[label] += 1
            correct[label] += int(type_pred[i] == y_type[i])
        out["per_label_accuracy"] = {label: correct[label] / counts[label] for label in sorted(counts)}
    return out


def evaluate_split(name: str, rows_path: str, binary_model: Pipeline, type_model: Pipeline, crop_size: int, limit: int, seed: int) -> dict[str, Any]:
    rows = load_rows(rows_path, limit, seed)
    x, y_binary, y_type, kept = featurize(rows, crop_size)
    if x.shape[0] == 0:
        return {"split": name, "rows": 0}
    binary_scores = binary_model.predict_proba(x)[:, 1]
    type_pred = type_model.predict(x)
    return {
        "split": name,
        "rows_loaded": len(rows),
        "rows_featurized": len(kept),
        "binary": binary_metrics(y_binary, binary_scores),
        "groups": grouped_metrics(kept, y_binary, binary_scores),
        "type_on_positive_anchors": type_metrics(y_binary, y_type, type_pred),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/symbol_tiny_stair_proposal_head_p037")
    parser.add_argument("--checkpoint", default="checkpoints/symbol_tiny_stair_proposal_head_p037/model.joblib")
    parser.add_argument("--report", default="reports/vlm/symbol_tiny_stair_proposal_head_p037_baseline_eval.json")
    parser.add_argument("--crop-size", type=int, default=24)
    parser.add_argument("--max-train-rows", type=int, default=24000)
    parser.add_argument("--max-eval-rows", type=int, default=30000)
    parser.add_argument("--seed", type=int, default=20260513)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    train_rows = load_rows(str(dataset_dir / "train.jsonl"), args.max_train_rows, args.seed)
    x_train, y_binary, y_type, kept_train = featurize(train_rows, args.crop_size)
    pos_idx = np.where(y_binary == 1)[0]
    train_type_x = x_train[pos_idx]
    train_type_y = y_type[pos_idx]

    binary_model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, l2_regularization=0.02, random_state=args.seed)),
    ])
    type_model = Pipeline([
        ("scale", StandardScaler()),
        ("clf", HistGradientBoostingClassifier(max_iter=120, learning_rate=0.08, l2_regularization=0.02, random_state=args.seed)),
    ])
    binary_model.fit(x_train, y_binary)
    type_model.fit(train_type_x, train_type_y)

    manifest = json.loads(source_path(dataset_dir / "manifest.json").read_text())
    report = {
        "version": "symbol_tiny_stair_proposal_head_p037_minimal_baseline",
        "dataset_manifest": str(dataset_dir / "manifest.json"),
        "train_rows_loaded": len(train_rows),
        "train_rows_featurized": len(kept_train),
        "train_positive_rows": int(y_binary.sum()),
        "train_negative_rows": int((1 - y_binary).sum()),
        "labels": LABELS,
        "feature_contract": {
            "runtime_model_input": "raster crop pixels plus runtime anchor geometry only",
            "offline_gold_used_as_training_label": True,
            "svg_or_cad_geometry_used_at_runtime": False,
            "metric_mode": "anchor-level dataset diagnostic; not page-level final quality",
        },
        "dataset_counts": manifest.get("counts"),
        "splits": {},
    }
    for split in ["smoke_eval", "dev"]:
        rows_path = manifest["outputs"].get(split) or str(dataset_dir / f"{split}.jsonl")
        report["splits"][split] = evaluate_split(split, rows_path, binary_model, type_model, args.crop_size, args.max_eval_rows, args.seed + 17)

    ckpt_path = source_path(args.checkpoint)
    ckpt_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"binary_model": binary_model, "type_model": type_model, "labels": LABELS, "crop_size": args.crop_size, "feature_contract": report["feature_contract"]}, ckpt_path)
    report["checkpoint"] = str(ckpt_path.relative_to(source_path('.')))
    write_json(source_path(args.report), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
