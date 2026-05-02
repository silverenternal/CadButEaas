#!/usr/bin/env python3
"""Train a dependency-light SymbolFixture v2 crop-context prototype model."""

from __future__ import annotations

import argparse
import json
import math
import resource
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from train_symbol_fixture_expert import evaluate_predictions, predict_host_links, write_jsonl
except ImportError:
    from scripts.vlm.train_symbol_fixture_expert import evaluate_predictions, predict_host_links, write_jsonl


BASE_FEATURES = ["cx", "cy", "width", "height", "area", "aspect", "rotation", "room_area_ratio", "inside_room"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/symbol_fixture_expert_v2")
    parser.add_argument("--output-dir", default="checkpoints/symbol_fixture_expert_v2")
    parser.add_argument("--report", default="reports/vlm/symbol_fixture_expert_v2_eval.json")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_rows = load_jsonl(dataset_dir / "train.jsonl")
    model = train_model(train_rows)
    model_path = output_dir / "model.json"
    model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary: dict[str, Any] = {
        "dataset_dir": str(dataset_dir),
        "model": str(model_path),
        "model_type": "symbol_fixture_context_standardized_centroid_v2",
        "memory_audit": memory_audit("after_training"),
        "train_item_counts": model["label_counts"],
        "feature_names": model["feature_names"],
        "splits": {},
    }
    for split in ("train", "dev", "locked_test", "smoke"):
        path = dataset_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = load_jsonl(path)
        predictions = predict_rows(rows, model)
        write_jsonl(output_dir / f"{split}_predictions.jsonl", predictions)
        metrics = evaluate_predictions(predictions)
        metrics["error_audit"] = error_audit(predictions)
        summary["splits"][split] = metrics

    summary["memory_audit"] = memory_audit("after_evaluation")
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = {
        "task_id": "P2-T3",
        "model_type": summary["model_type"],
        "dataset_dir": str(dataset_dir),
        "checkpoint_dir": str(output_dir),
        "splits": summary["splits"],
        "memory_audit": summary["memory_audit"],
        "finding": "Dependency-light context prototype for SymbolFixture v2; intended as auditable baseline when torch is unavailable.",
    }
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def train_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    room_types = sorted({str(symbol.get("room_type") or "unknown_room") for row in rows for symbol in row.get("symbols") or []})
    feature_names = [*BASE_FEATURES, *[f"room_type_{label}" for label in room_types]]
    by_label: dict[str, list[list[float]]] = defaultdict(list)
    counts: Counter[str] = Counter()
    all_features = []
    for row in rows:
        width, height = page_size(row)
        for symbol in row.get("symbols") or []:
            label = str(symbol.get("symbol_type") or "generic_symbol")
            feature = feature_vector(symbol, row.get("rooms") or [], width, height, room_types)
            if feature is None:
                continue
            by_label[label].append(feature)
            all_features.append(feature)
            counts[label] += 1
    mean, std = feature_stats(all_features)
    prototypes = {label: mean_vector([standardize(item, mean, std) for item in features]) for label, features in by_label.items() if features}
    total = sum(counts.values())
    return {
        "labels": sorted(prototypes),
        "feature_names": feature_names,
        "room_types": room_types,
        "feature_mean": mean,
        "feature_std": std,
        "prototypes": prototypes,
        "priors": {label: counts[label] / max(total, 1) for label in counts},
        "label_counts": dict(counts),
    }


def predict_rows(rows: list[dict[str, Any]], model: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = []
    for row in rows:
        width, height = page_size(row)
        symbols = []
        for symbol in row.get("symbols") or []:
            feature = feature_vector(symbol, row.get("rooms") or [], width, height, model["room_types"])
            prediction, confidence = predict_label(feature, model)
            symbols.append(
                {
                    "id": symbol.get("id"),
                    "gold": symbol.get("symbol_type"),
                    "prediction": prediction,
                    "confidence": confidence,
                    "bbox": symbol.get("bbox"),
                    "iou": 1.0,
                    "room_type": symbol.get("room_type"),
                    "symbol_type_raw": symbol.get("symbol_type_raw"),
                }
            )
        predictions.append(
            {
                "image": row.get("image"),
                "annotation": row.get("annotation"),
                "source_dataset": row.get("source_dataset"),
                "symbols": symbols,
                "host_links_gold": row.get("host_links") or [],
                "host_links_pred": predict_host_links(symbols, row.get("rooms") or []),
            }
        )
    return predictions


def predict_label(feature: list[float] | None, model: dict[str, Any]) -> tuple[str, float]:
    labels = model.get("labels") or []
    if not labels:
        return "generic_symbol", 0.0
    if feature is None:
        label = max(labels, key=lambda item: model.get("priors", {}).get(item, 0.0))
        return str(label), float(model.get("priors", {}).get(label, 0.0))
    normalized = standardize(feature, model["feature_mean"], model["feature_std"])
    scored = []
    for label in labels:
        distance = euclidean(normalized, model["prototypes"][label])
        prior = max(float(model.get("priors", {}).get(label, 1e-6)), 1e-6)
        scored.append((distance - 0.03 * math.log(prior), label))
    scored.sort()
    best_distance, best_label = scored[0]
    second = scored[1][0] if len(scored) > 1 else best_distance + 1.0
    confidence = 1.0 / (1.0 + max(best_distance, 0.0))
    if second > 0:
        confidence *= min(1.0, max(0.05, (second - best_distance) / second + 0.1))
    return str(best_label), float(confidence)


def feature_vector(symbol: dict[str, Any], rooms: list[dict[str, Any]], width: float, height: float, room_types: list[str]) -> list[float] | None:
    box = bbox(symbol.get("bbox"))
    if box is None:
        return None
    x1, y1, x2, y2 = box
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    containing = [room for room in rooms if contains(bbox(room.get("bbox")), box)]
    host = min(containing, key=lambda item: area_of(bbox(item.get("bbox"))), default=None)
    host_area = area_of(bbox(host.get("bbox"))) if host else 0.0
    symbol_room_type = str(symbol.get("room_type") or (host or {}).get("room_type") or "unknown_room")
    room_one_hot = [1.0 if symbol_room_type == label else 0.0 for label in room_types]
    return [
        ((x1 + x2) / 2.0) / max(width, 1.0),
        ((y1 + y2) / 2.0) / max(height, 1.0),
        w / max(width, 1.0),
        h / max(height, 1.0),
        area / max(width * height, 1.0),
        math.log((w + 1.0) / (h + 1.0)),
        float(symbol.get("rotation") or 0.0) / 360.0,
        area / max(host_area, 1.0),
        1.0 if host else 0.0,
        *room_one_hot,
    ]


def error_audit(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    pairs = Counter()
    by_raw = Counter()
    for row in predictions:
        for symbol in row.get("symbols") or []:
            gold = str(symbol.get("gold"))
            pred = str(symbol.get("prediction"))
            if gold != pred:
                pairs[(gold, pred)] += 1
                by_raw[(str(symbol.get("symbol_type_raw") or ""), gold, pred)] += 1
    return {
        "top_error_pairs": [{"target": a, "prediction": b, "count": c} for (a, b), c in pairs.most_common(20)],
        "top_raw_error_pairs": [{"raw": r, "target": a, "prediction": b, "count": c} for (r, a, b), c in by_raw.most_common(20)],
    }


def page_size(row: dict[str, Any]) -> tuple[float, float]:
    metadata = row.get("metadata") or {}
    return float(metadata.get("width") or 1.0), float(metadata.get("height") or 1.0)


def bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def contains(left: list[float] | None, right: list[float] | None) -> bool:
    if not left or not right:
        return False
    return left[0] <= right[0] and left[1] <= right[1] and left[2] >= right[2] and left[3] >= right[3]


def area_of(box: list[float] | None) -> float:
    if not box:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def mean_vector(rows: list[list[float]]) -> list[float]:
    if not rows:
        return []
    return [sum(row[index] for row in rows) / len(rows) for index in range(len(rows[0]))]


def feature_stats(rows: list[list[float]]) -> tuple[list[float], list[float]]:
    mean = mean_vector(rows)
    std = []
    for index, value in enumerate(mean):
        variance = sum((row[index] - value) ** 2 for row in rows) / max(len(rows), 1)
        std.append(max(math.sqrt(variance), 1e-6))
    return mean, std


def standardize(row: list[float], mean: list[float], std: list[float]) -> list[float]:
    return [(value - mean[index]) / std[index] for index, value in enumerate(row)]


def euclidean(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def memory_audit(stage: str) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"stage": stage, "max_rss_kb": int(usage.ru_maxrss)}


if __name__ == "__main__":
    main()
