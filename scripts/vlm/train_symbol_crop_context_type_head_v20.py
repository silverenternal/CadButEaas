#!/usr/bin/env python3
"""Train a cached multi-scale crop/context symbol type head.

This is still an oracle-localization audit: gold symbol boxes select the crop.
It measures whether raster crop/context evidence is strong enough for typing
before pairing the type head with a body detector.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/symbol_crop_context_cache_v20"
CHECKPOINT = ROOT / "checkpoints/symbol_crop_context_type_head_v20"
REPORT = ROOT / "reports/vlm/symbol_crop_context_type_head_v20_eval.json"
LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]
CROP_VIEWS = ["tight", "padded", "context"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def source_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def sample_balanced(items: list[dict[str, Any]], max_per_label: int | None, seed: int) -> list[dict[str, Any]]:
    if not max_per_label:
        return list(items)
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_label[str(item.get("label"))].append(item)
    out: list[dict[str, Any]] = []
    for label in LABELS:
        rows = by_label.get(label, [])
        rng.shuffle(rows)
        out.extend(rows[:max_per_label])
    rng.shuffle(out)
    return out


def limit_items(items: list[dict[str, Any]], limit: int | None, seed: int) -> list[dict[str, Any]]:
    if not limit or len(items) <= limit:
        return list(items)
    rng = random.Random(seed)
    rows = list(items)
    rng.shuffle(rows)
    return rows[:limit]


def crop_features(crop_path: str, size: int) -> np.ndarray:
    with Image.open(source_path(crop_path)) as opened:
        crop = opened.convert("L")
    crop = ImageOps.autocontrast(crop)
    crop = crop.resize((size, size), Image.Resampling.BICUBIC)
    arr = 1.0 - (np.asarray(crop, dtype=np.float32) / 255.0)
    row_density = arr.mean(axis=1)
    col_density = arr.mean(axis=0)
    quads = np.array(
        [
            arr[: size // 2, : size // 2].mean(),
            arr[: size // 2, size // 2 :].mean(),
            arr[size // 2 :, : size // 2].mean(),
            arr[size // 2 :, size // 2 :].mean(),
            arr.mean(),
            arr.std(),
        ],
        dtype=np.float32,
    )
    return np.concatenate([arr.reshape(-1), row_density, col_density, quads]).astype(np.float32)


def geometry_features(item: dict[str, Any]) -> np.ndarray:
    geom = item.get("geometry") or {}
    bbox_norm = geom.get("bbox_norm") or [0.0, 0.0, 0.0, 0.0]
    center_norm = geom.get("center_norm") or [0.0, 0.0]
    values = [
        *bbox_norm[:4],
        *center_norm[:2],
        geom.get("width_norm", 0.0),
        geom.get("height_norm", 0.0),
        geom.get("area_norm", 0.0),
        geom.get("aspect_log", 0.0),
    ]
    return np.asarray([float(v) for v in values], dtype=np.float32)


def item_features(item: dict[str, Any], size: int) -> np.ndarray:
    crops = item.get("crops") or {}
    parts: list[np.ndarray] = []
    for view in CROP_VIEWS:
        crop = crops.get(view) or {}
        path = crop.get("path")
        if path:
            parts.append(crop_features(str(path), size))
        else:
            parts.append(np.zeros(size * size + size * 2 + 6, dtype=np.float32))
    parts.append(geometry_features(item))
    feat = np.concatenate(parts).astype(np.float32)
    norm = float(np.linalg.norm(feat))
    return feat / norm if norm > 1e-9 else feat


def tensorize(items: list[dict[str, Any]], size: int) -> tuple[np.ndarray, list[str]]:
    features: list[np.ndarray] = []
    labels: list[str] = []
    for item in items:
        label = str(item.get("label") or "")
        if label not in LABELS:
            continue
        features.append(item_features(item, size))
        labels.append(label)
    if not features:
        dim = len(CROP_VIEWS) * (size * size + size * 2 + 6) + 10
        return np.zeros((0, dim), dtype=np.float32), []
    return np.stack(features).astype(np.float32), labels


def make_model(kind: str, n_estimators: int, seed: int) -> Any:
    if kind == "rf":
        return RandomForestClassifier(
            n_estimators=n_estimators,
            min_samples_leaf=1,
            class_weight="balanced_subsample",
            random_state=seed,
            n_jobs=-1,
        )
    return ExtraTreesClassifier(
        n_estimators=n_estimators,
        min_samples_leaf=1,
        class_weight="balanced",
        random_state=seed,
        n_jobs=-1,
    )


def metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
    confusion = {label: Counter() for label in LABELS}
    correct = 0
    for g, p in zip(gold, pred, strict=True):
        confusion.setdefault(g, Counter())[p] += 1
        correct += int(g == p)
    per_label: dict[str, Any] = {}
    f1s: list[float] = []
    for label in LABELS:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in LABELS if other != label)
        fn = sum(value for key, value in confusion[label].items() if key != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_label[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": sum(confusion[label].values()),
        }
        f1s.append(f1)
    return {
        "accuracy": round(correct / max(len(gold), 1), 6),
        "macro_f1": round(sum(f1s) / max(len(f1s), 1), 6),
        "per_label": per_label,
        "confusion": {label: dict(confusion[label]) for label in LABELS},
    }


def error_buckets(gold: list[str], pred: list[str], items: list[dict[str, Any]]) -> dict[str, Any]:
    buckets: dict[str, Counter[str]] = defaultdict(Counter)
    for g, p, item in zip(gold, pred, items, strict=True):
        if g == p:
            continue
        for bucket in item.get("stress_buckets") or ["unknown"]:
            buckets[str(bucket)][f"{g}->{p}"] += 1
    return {bucket: dict(counter.most_common(20)) for bucket, counter in sorted(buckets.items())}


def evaluate(model: Any, x: np.ndarray, y: list[str], items: list[dict[str, Any]]) -> dict[str, Any]:
    pred = list(model.predict(x)) if len(y) else []
    result = metrics(y, pred)
    result["error_buckets"] = error_buckets(y, pred, items)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT))
    parser.add_argument("--eval-output", default=str(REPORT))
    parser.add_argument("--error-output", default=str(ROOT / "reports/vlm/symbol_crop_context_type_head_v20_error_buckets.json"))
    parser.add_argument("--max-train-per-label", type=int, default=4000)
    parser.add_argument("--limit-dev", type=int, default=None)
    parser.add_argument("--limit-locked", type=int, default=None)
    parser.add_argument("--feature-size", type=int, default=24)
    parser.add_argument("--n-estimators", type=int, default=360)
    parser.add_argument("--model-kind", choices=["et", "rf"], default="et")
    parser.add_argument("--seed", type=int, default=20260510)
    args = parser.parse_args()

    data_dir = Path(args.data)
    train_items = sample_balanced(load_jsonl(data_dir / "train.jsonl"), args.max_train_per_label, args.seed)
    dev_items = limit_items(load_jsonl(data_dir / "dev.jsonl"), args.limit_dev, args.seed + 1)
    locked_items = limit_items(load_jsonl(data_dir / "locked.jsonl"), args.limit_locked, args.seed + 2)

    x_train, y_train = tensorize(train_items, args.feature_size)
    x_dev, y_dev = tensorize(dev_items, args.feature_size)
    x_locked, y_locked = tensorize(locked_items, args.feature_size)

    model = make_model(args.model_kind, args.n_estimators, args.seed)
    model.fit(x_train, y_train)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "labels": LABELS,
            "feature_size": args.feature_size,
            "crop_views": CROP_VIEWS,
            "runtime_contract": {
                "allowed_model_inputs": ["crops.tight", "crops.padded", "crops.context", "geometry"],
                "forbidden_runtime_features": ["raw_label", "semantic_type", "expected_json", "annotation_path", "svg_geometry"],
            },
        },
        checkpoint_dir / "model.joblib",
    )

    dev_eval = evaluate(model, x_dev, y_dev, dev_items)
    locked_eval = evaluate(model, x_locked, y_locked, locked_items)
    report = {
        "task": "symbol_crop_context_type_head_v20",
        "claim_boundary": "Oracle gold-box type-head audit only. This is not full symbol detection performance.",
        "data": rel(data_dir),
        "checkpoint": rel(checkpoint_dir / "model.joblib"),
        "config": {
            "model_kind": args.model_kind,
            "n_estimators": args.n_estimators,
            "feature_size": args.feature_size,
            "max_train_per_label": args.max_train_per_label,
            "limit_dev": args.limit_dev,
            "limit_locked": args.limit_locked,
        },
        "counts": {
            "train": len(y_train),
            "dev": len(y_dev),
            "locked": len(y_locked),
            "train_label_counts": dict(Counter(y_train).most_common()),
            "dev_label_counts": dict(Counter(y_dev).most_common()),
            "locked_label_counts": dict(Counter(y_locked).most_common()),
        },
        "dev": dev_eval,
        "locked": locked_eval,
        "baseline_comparison": {
            "previous_gold_box_handcrafted_locked_macro_f1": 0.436258,
            "delta_locked_macro_f1": round(float(locked_eval["macro_f1"]) - 0.436258, 6),
        },
        "gate": {
            "stage_1_min_type_macro_f1_0_65": float(locked_eval["macro_f1"]) >= 0.65,
            "beats_previous_handcrafted_oracle_baseline": float(locked_eval["macro_f1"]) > 0.436258,
        },
    }
    write_json(Path(args.eval_output), report)
    write_json(
        Path(args.error_output),
        {
            "task": "symbol_crop_context_type_head_v20_error_buckets",
            "dev": dev_eval["error_buckets"],
            "locked": locked_eval["error_buckets"],
        },
    )


if __name__ == "__main__":
    main()
