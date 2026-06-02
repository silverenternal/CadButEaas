#!/usr/bin/env python3
"""Train a raster-only symbol type classifier on gold symbol crops.

This is an oracle-localization audit: gold boxes are used to crop training,
dev, and locked symbols. The resulting metrics isolate the type head from body
localization, so they must not be reported as full detector performance.
"""

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
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/symbol_expert_public_raster_v19"
REPORT = ROOT / "reports/vlm"
CHECKPOINT = ROOT / "checkpoints/symbol_public_raster_v19_gold_crop"
LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def flatten_items(rows: list[dict[str, Any]], split: str) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in rows:
        image = str(row.get("image") or "")
        width, height = [int(v) for v in row.get("image_size") or [0, 0]]
        for index, target in enumerate((row.get("targets") or {}).get("boxes") or []):
            label = str(target.get("label") or target.get("symbol_type") or target.get("semantic_type") or "")
            bbox = target.get("bbox")
            if label not in LABELS or not isinstance(bbox, list) or len(bbox) != 4:
                continue
            items.append(
                {
                    "id": f"{row.get('id')}_symbol_{index}",
                    "row_id": row.get("id"),
                    "source_dataset": row.get("source_dataset"),
                    "split": split,
                    "image": image,
                    "image_size": [width, height],
                    "bbox": [int(v) for v in bbox],
                    "label": label,
                    "area": int(target.get("area") or 0),
                    "rare_class": bool(target.get("rare_class")),
                }
            )
    return items


def sample_balanced(items: list[dict[str, Any]], max_per_label: int | None, seed: int) -> list[dict[str, Any]]:
    if not max_per_label:
        return list(items)
    rng = random.Random(seed)
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_label[str(item["label"])].append(item)
    out: list[dict[str, Any]] = []
    for label in LABELS:
        rows = by_label.get(label, [])
        rng.shuffle(rows)
        out.extend(rows[:max_per_label])
    rng.shuffle(out)
    return out


class ImageCache:
    def __init__(self, max_items: int) -> None:
        self.max_items = max_items
        self.cache: dict[str, Image.Image] = {}

    def get(self, path: str) -> Image.Image:
        if path in self.cache:
            return self.cache[path]
        p = Path(path)
        image = Image.open(p if p.is_absolute() else ROOT / p).convert("RGB")
        if len(self.cache) >= self.max_items:
            self.cache.pop(next(iter(self.cache)))
        self.cache[path] = image
        return image


def crop_feature(image: Image.Image, bbox: list[int], pad: int, size: int) -> np.ndarray:
    width, height = image.size
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        crop = Image.new("L", (size, size), 255)
        bw = bh = area = aspect = fill = 0.0
    else:
        crop = image.crop((x1, y1, x2, y2)).convert("L")
        crop = ImageOps.autocontrast(crop)
        bw = (x2 - x1) / max(width, 1)
        bh = (y2 - y1) / max(height, 1)
        area = bw * bh
        aspect = math.log((x2 - x1) / max(y2 - y1, 1))
        raw = np.asarray(crop, dtype=np.uint8)
        fill = float((raw <= 205).mean()) if raw.size else 0.0
    resized = crop.resize((size, size), Image.Resampling.BICUBIC)
    arr = 1.0 - (np.asarray(resized, dtype=np.float32) / 255.0)
    row_density = arr.mean(axis=1)
    col_density = arr.mean(axis=0)
    quads = np.array(
        [
            arr[: size // 2, : size // 2].mean(),
            arr[: size // 2, size // 2 :].mean(),
            arr[size // 2 :, : size // 2].mean(),
            arr[size // 2 :, size // 2 :].mean(),
        ],
        dtype=np.float32,
    )
    geom = np.array([bw, bh, area, aspect, fill], dtype=np.float32)
    feat = np.concatenate([arr.reshape(-1), row_density, col_density, quads, geom]).astype(np.float32)
    norm = float(np.linalg.norm(feat))
    return feat / norm if norm > 1e-9 else feat


def tensorize(items: list[dict[str, Any]], pad: int, size: int, cache_size: int) -> tuple[np.ndarray, list[str]]:
    cache = ImageCache(cache_size)
    features = []
    labels = []
    for item in items:
        features.append(crop_feature(cache.get(str(item["image"])), item["bbox"], pad, size))
        labels.append(str(item["label"]))
    if not features:
        return np.zeros((0, size * size + size * 2 + 9), dtype=np.float32), []
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
    labels = list(LABELS)
    confusion = {label: Counter() for label in labels}
    correct = 0
    for g, p in zip(gold, pred, strict=True):
        confusion.setdefault(g, Counter())[p] += 1
        correct += int(g == p)
    per_label: dict[str, Any] = {}
    f1s = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
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
        "confusion": {label: dict(confusion[label]) for label in labels},
    }


def eval_model(model: Any, x: np.ndarray, y: list[str]) -> dict[str, Any]:
    pred = list(model.predict(x)) if len(y) else []
    return metrics(y, pred)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT))
    parser.add_argument("--eval-output", default=str(REPORT / "symbol_public_raster_v19_gold_crop_type_eval.json"))
    parser.add_argument("--max-train-per-label", type=int, default=5000)
    parser.add_argument("--limit-dev", type=int, default=None)
    parser.add_argument("--limit-locked", type=int, default=None)
    parser.add_argument("--pad", type=int, default=4)
    parser.add_argument("--size", type=int, default=20)
    parser.add_argument("--n-estimators", type=int, default=320)
    parser.add_argument("--model-kind", choices=["et", "rf"], default="et")
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--image-cache-size", type=int, default=256)
    args = parser.parse_args()

    data = Path(args.data)
    train_items = flatten_items(load_jsonl(data / "train.jsonl"), "train")
    dev_items = flatten_items(load_jsonl(data / "dev.jsonl"), "dev")
    locked_items = flatten_items(load_jsonl(data / "locked.jsonl"), "locked")
    train_sample = sample_balanced(train_items, args.max_train_per_label, args.seed)
    dev_eval_items = dev_items[: args.limit_dev] if args.limit_dev else dev_items
    locked_eval_items = locked_items[: args.limit_locked] if args.limit_locked else locked_items

    x_train, y_train = tensorize(train_sample, args.pad, args.size, args.image_cache_size)
    x_dev, y_dev = tensorize(dev_eval_items, args.pad, args.size, args.image_cache_size)
    x_locked, y_locked = tensorize(locked_eval_items, args.pad, args.size, args.image_cache_size)

    model = make_model(args.model_kind, args.n_estimators, args.seed)
    model.fit(x_train, y_train)
    dev_metrics = eval_model(model, x_dev, y_dev)
    locked_metrics = eval_model(model, x_locked, y_locked)

    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "labels": LABELS,
            "feature_contract": {
                "input": "raster_crop_from_gold_symbol_bbox",
                "pad": args.pad,
                "size": args.size,
                "feature_dim": int(x_train.shape[1]) if x_train.ndim == 2 else 0,
            },
            "runtime_boundary": "oracle_localization_type_head_audit_not_full_detector",
        },
        checkpoint_dir / "model.joblib",
    )

    report = {
        "version": "symbol_public_raster_v19_gold_crop_type_eval",
        "run_mode": "oracle_gold_box_raster_crop_type_classifier",
        "source_integrity": {
            "model_input": "raster_crop_only",
            "gold_bbox_use": "crop_oracle_for_type_head_audit",
            "raw_label_or_semantic_type_used_as_runtime_feature": False,
            "locked_gold_use": "crop_oracle_and_evaluation_only",
        },
        "dataset": str(data.relative_to(ROOT) if data.is_relative_to(ROOT) else data),
        "train_sampling": {
            "raw_train_items": len(train_items),
            "sampled_train_items": len(train_sample),
            "max_train_per_label": args.max_train_per_label,
            "sampled_label_counts": dict(Counter(y_train).most_common()),
        },
        "model": {
            "kind": args.model_kind,
            "n_estimators": args.n_estimators,
            "pad": args.pad,
            "size": args.size,
            "feature_dim": int(x_train.shape[1]) if x_train.ndim == 2 else 0,
        },
        "dev_metrics": dev_metrics,
        "locked_metrics": locked_metrics,
        "adopted": False,
        "adoption_note": "This is an oracle-localization type-head audit. It can guide type modeling but cannot be adopted as the full symbol detector.",
    }
    write_json(Path(args.eval_output), report)
    print(json.dumps({"dev_macro_f1": dev_metrics["macro_f1"], "locked_macro_f1": locked_metrics["macro_f1"], "checkpoint": str(checkpoint_dir / "model.joblib")}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
