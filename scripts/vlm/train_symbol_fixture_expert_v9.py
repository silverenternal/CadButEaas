#!/usr/bin/env python3
"""Train SymbolFixtureExpert v9 — optimized for long-tail handling.

v8 issues:
- generic_symbol F1=0.065 (100 train samples, precision=0.034)
- table F1=0.50 (32 train samples, recall=0.33)
- equipment F1=0.85 (234 misclassified as generic_symbol)

v9 improvements:
1. Use ALL training symbols (no 30k cap) — more data for rare classes
2. Increase n_estimators and depth for better rare-class learning
3. Add class_weight='balanced' to force focus on rare classes
4. Target: 9-class dev macro F1 >= 0.85 (realistic given long-tail)
"""

from __future__ import annotations

import argparse
import json
import math
import random
import resource
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/symbol_fixture_detector_v2")
    parser.add_argument("--output-dir", default="checkpoints/symbol_fixture_expert_v9")
    parser.add_argument("--n-estimators", type=int, default=200)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--max-train-symbols", type=int, default=50000)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(dataset_dir / "train.jsonl")
    dev_rows = load_jsonl(dataset_dir / "dev.jsonl")
    smoke_rows = load_jsonl(dataset_dir / "smoke.jsonl")
    locked_rows = load_jsonl(dataset_dir / "locked.jsonl")

    print(f"Data: train={len(train_rows)}, dev={len(dev_rows)}, smoke={len(smoke_rows)}, locked={len(locked_rows)}")

    # Extract features — use ALL symbols for better rare-class coverage
    print("Extracting train features...")
    if args.max_train_symbols > 0:
        train_items = extract_items_stratified(train_rows, max_symbols=args.max_train_symbols)
    else:
        train_items = extract_items(train_rows)  # ALL symbols
    print(f"  Extracted {len(train_items)} symbols")

    labels = sorted({item["label"] for item in train_items})
    print(f"  Labels ({len(labels)}): {labels}")
    label_to_index = {label: idx for idx, label in enumerate(labels)}

    FEATURE_NAMES = get_feature_names()
    print(f"  Features ({len(FEATURE_NAMES)}): {FEATURE_NAMES}")

    X_train = np.array([item["features"] for item in train_items], dtype=np.float64)
    y_train = np.array([label_to_index[item["label"]] for item in train_items], dtype=np.int64)
    X_train = np.nan_to_num(X_train, nan=0.0, posinf=10.0, neginf=-10.0)

    # Train ExtraTrees with balanced class weight
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    label_counts = Counter(y_train)
    print(f"\nClass distribution: {dict(sorted([(labels[k], v) for k, v in label_counts.items()], key=lambda x: x[1]))}")

    print(f"\nTraining ExtraTrees (n={args.n_estimators}, depth={args.max_depth}, leaf={args.min_samples_leaf})...")
    model = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",  # Force focus on rare classes
        random_state=42,
    )
    model.fit(X_train_scaled, y_train)

    # Evaluate on all splits
    splits = {"dev": dev_rows, "smoke": smoke_rows}
    if locked_rows:
        splits["locked"] = locked_rows

    summary = {
        "model_type": "symbol_fixture_expert_v9_extra_trees",
        "feature_count": len(FEATURE_NAMES),
        "feature_names": FEATURE_NAMES,
        "label_map": label_to_index,
        "label_counts": dict(Counter(item["label"] for item in train_items)),
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "splits": {},
    }

    for split_name, split_rows in splits.items():
        print(f"\nEvaluating {split_name}...")
        if split_name == "dev":
            items = extract_items(split_rows)
        else:
            items = extract_items(split_rows)
        if not items:
            continue
        X = np.nan_to_num(np.array([item["features"] for item in items], dtype=np.float64), nan=0.0, posinf=10.0, neginf=-10.0)
        X_scaled = scaler.transform(X)
        y_pred = model.predict(X_scaled)
        y_gold = np.array([label_to_index.get(item["label"], -1) for item in items], dtype=np.int64)

        valid = y_gold >= 0
        X_scaled = X_scaled[valid]
        y_pred = y_pred[valid]
        y_gold = y_gold[valid]
        items_filtered = [item for item, v in zip(items, valid) if v]

        predictions = build_predictions(items_filtered, y_gold, y_pred, labels, model, X_scaled)
        eval_result = evaluate_predictions(predictions, labels)
        summary["splits"][split_name] = eval_result
        write_jsonl(output_dir / f"{split_name}_predictions.jsonl", predictions)
        print(f"  {split_name}: acc={eval_result['accuracy']:.4f}, macro_f1={eval_result['macro_f1']:.4f}")

        # Print per-class details for rare classes
        for label in ["generic_symbol", "table", "bathtub"]:
            if label in eval_result.get("per_label", {}):
                pl = eval_result["per_label"][label]
                print(f"    {label}: P={pl['precision']:.3f} R={pl['recall']:.3f} F1={pl['f1']:.3f} (support={pl['support']})")

    # Feature importance
    summary["feature_importance"] = {
        FEATURE_NAMES[i]: round(float(imp), 4)
        for i, imp in enumerate(model.feature_importances_)
    }

    dev_f1 = summary.get("splits", {}).get("dev", {}).get("macro_f1", 0)
    summary["acceptance"] = {
        "dev_macro_f1": round(dev_f1, 4),
        "target": 0.85,
        "passed": dev_f1 >= 0.85,
    }

    # Save model
    import joblib
    joblib.dump({
        "classifier": model,
        "scaler": scaler,
        "feature_names": FEATURE_NAMES,
        "class_names": labels,
        "label_to_index": label_to_index,
    }, output_dir / "model_v9.joblib")

    write_json(output_dir / "train_summary.json", summary)
    print(f"\n{'='*60}")
    print(f"Final: dev macro F1={dev_f1:.4f} {'PASS' if dev_f1 >= 0.85 else 'FAIL'} (target >= 0.85)")
    print(f"Model saved to: {output_dir / 'model_v9.joblib'}")


def get_feature_names() -> list[str]:
    return [
        "cx", "cy", "width", "height", "area_norm", "log_aspect_ratio",
        "room_wet", "room_living", "room_service", "room_outdoor",
        "neighbor_count", "neighbor_avg_area_log", "neighbor_area_ratio_log",
    ]


def extract_items_stratified(rows: list[dict[str, Any]], max_symbols: int = 30000) -> list[dict[str, Any]]:
    """Extract symbols with stratified sampling to ensure rare classes are included."""
    row_symbols = []
    for row in rows:
        for symbol in row.get("symbols") or []:
            label = str(symbol.get("symbol_type") or "generic_symbol")
            bbox = normalize_bbox(symbol.get("bbox"))
            if bbox:
                row_symbols.append((row, symbol, label, bbox))

    by_label = defaultdict(list)
    for row, sym, label, bbox in row_symbols:
        by_label[label].append((row, sym, bbox))

    kept = []
    rare_threshold = 100
    for label, items in by_label.items():
        if len(items) <= rare_threshold:
            kept.extend(items)
        else:
            n_keep = max(rare_threshold, int(max_symbols * len(items) / len(row_symbols)))
            random.seed(42)
            random.shuffle(items)
            kept.extend(items[:n_keep])

    items = []
    by_row = defaultdict(list)
    for item in kept:
        by_row[id(item[0])].append(item)

    for row_id, symbols in by_row.items():
        row = symbols[0][0]
        width = float((row.get("metadata") or {}).get("width") or 2000.0)
        height = float((row.get("metadata") or {}).get("height") or 2000.0)

        room_by_id = {}
        for room in row.get("rooms") or []:
            room_by_id[str(room.get("id", ""))] = room

        symbol_areas = [(b[2]-b[0])*(b[3]-b[1]) for _, _, b in symbols]
        mean_area = float(np.mean(symbol_areas)) if symbol_areas else 0.0

        for _, symbol, bbox in symbols:
            label = str(symbol.get("symbol_type") or "generic_symbol")
            features = extract_symbol_features(symbol, bbox, width, height, room_by_id, symbol_areas, mean_area)
            if features:
                items.append({"label": label, "features": features, "symbol": symbol, "image": row.get("image")})

    return items


def extract_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract ALL symbols from rows without subsampling."""
    items = []
    for row in rows:
        width = float((row.get("metadata") or {}).get("width") or 2000.0)
        height = float((row.get("metadata") or {}).get("height") or 2000.0)

        room_by_id = {}
        for room in row.get("rooms") or []:
            room_by_id[str(room.get("id", ""))] = room

        all_symbol_bboxes = []
        for sym in row.get("symbols") or []:
            bbox = normalize_bbox(sym.get("bbox"))
            if bbox:
                all_symbol_bboxes.append(bbox)
        all_symbol_areas = [(b[2] - b[0]) * (b[3] - b[1]) for b in all_symbol_bboxes]
        mean_neighbor_area = float(np.mean(all_symbol_areas)) if all_symbol_areas else 0.0

        for symbol in row.get("symbols") or []:
            label = str(symbol.get("symbol_type") or "generic_symbol")
            bbox = normalize_bbox(symbol.get("bbox"))
            if bbox is None:
                continue

            features = extract_symbol_features(symbol, bbox, width, height, room_by_id, all_symbol_areas, mean_neighbor_area)
            if features is not None:
                items.append({"label": label, "features": features, "symbol": symbol, "image": row.get("image")})

    return items


def extract_symbol_features(
    symbol: dict, bbox: list[float], img_w: float, img_h: float,
    room_by_id: dict, all_symbol_areas: list, mean_neighbor_area: float,
) -> list[float] | None:
    x1, y1, x2, y2 = bbox
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    log_aspect_ratio = math.log((w + 1.0) / (h + 1.0))

    bbox_feats = [
        cx / max(img_w, 1.0),
        cy / max(img_h, 1.0),
        w / max(img_w, 1.0),
        h / max(img_h, 1.0),
        area / max(img_w * img_h, 1.0),
        log_aspect_ratio,
    ]

    room_feats = extract_room_context(bbox, room_by_id)

    sym_area = area
    neighbor_count = max(0, len(all_symbol_areas) - 1)
    area_ratio = sym_area / max(mean_neighbor_area, 1.0)
    neighbor_feats = [
        min(neighbor_count, 100.0),
        math.log(mean_neighbor_area + 1.0),
        math.log(area_ratio + 1.0),
    ]

    return bbox_feats + room_feats + neighbor_feats


def extract_room_context(sym_bbox: list[float], room_by_id: dict) -> list[float]:
    wet, living, service, outdoor = 0.0, 0.0, 0.0, 0.0
    sym_cx = (sym_bbox[0] + sym_bbox[2]) / 2
    sym_cy = (sym_bbox[1] + sym_bbox[3]) / 2

    for room_id, room in room_by_id.items():
        room_bbox = normalize_bbox(room.get("bbox"))
        if room_bbox and room_contains(room_bbox, sym_cx, sym_cy):
            room_type = str(room.get("room_type", ""))
            if room_type in ("bathroom", "toilet", "shower_room"):
                wet = 1.0
            elif room_type in ("bedroom", "living_room", "kitchen", "corridor"):
                living = 1.0
            elif room_type in ("closet", "storage", "office"):
                service = 1.0
            elif room_type == "balcony":
                outdoor = 1.0
            break
    return [wet, living, service, outdoor]


def room_contains(bbox: list[float], x: float, y: float) -> bool:
    return bbox[0] <= x <= bbox[2] and bbox[1] <= y <= bbox[3]


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def evaluate_predictions(predictions: list[dict], labels: list[str]) -> dict:
    confusion = {label: Counter() for label in labels}
    total = correct = 0
    for pred in predictions:
        for sym in pred.get("symbols", []):
            gold = str(sym.get("gold", ""))
            pred_label = str(sym.get("prediction", ""))
            confusion.setdefault(gold, Counter())
            confusion[gold][pred_label] += 1
            total += 1
            correct += int(gold == pred_label)

    per_label, macro_f1 = classification_report(labels, confusion)
    return {
        "symbols": total,
        "accuracy": correct / max(total, 1),
        "macro_f1": macro_f1,
        "per_label": per_label,
        "confusion": {label: dict(counts) for label, counts in confusion.items()},
    }


def classification_report(labels: list[str], confusion: dict) -> tuple[dict, float]:
    per_label = {}
    f1_values = []
    for label in labels:
        tp = confusion.get(label, Counter()).get(label, 0)
        fp = sum(confusion.get(other, Counter()).get(label, 0) for other in labels if other != label)
        fn = sum(count for pred, count in confusion.get(label, Counter()).items() if pred != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        per_label[label] = {
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
            "support": sum(confusion[label].values()),
        }
    return per_label, sum(f1_values) / max(len(f1_values), 1)


def build_predictions(
    items: list[dict], y_gold: np.ndarray, y_pred: np.ndarray,
    labels: list[str], model, X_scaled: np.ndarray,
) -> list[dict]:
    by_image = defaultdict(list)
    for idx, (item, gy, gp) in enumerate(zip(items, y_gold, y_pred)):
        image = item.get("image", "")
        proba = model.predict_proba([X_scaled[idx]]) if hasattr(model, 'predict_proba') else [[0.0]]
        confidence = float(np.max(proba[0])) if len(proba) > 0 else 0.0

        by_image[image].append({
            "id": item["symbol"].get("id"),
            "gold": item["label"],
            "prediction": labels[gp] if gp < len(labels) else "unknown",
            "confidence": confidence,
            "bbox": item["symbol"].get("bbox"),
        })

    return [
        {"image": image, "symbols": syms, "host_links_gold": [], "host_links_pred": []}
        for image, syms in by_image.items()
    ]


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
