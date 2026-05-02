#!/usr/bin/env python3
"""Train a grouped MoE-style sklearn baseline for RoomSpace labels."""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.append(str(SCRIPT_DIR))
from train_room_space_expert import evaluate_predictions, write_jsonl


SYMBOL_TYPES = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]
BOUNDARY_TYPES = ["door", "hard_wall", "opening", "partition_wall", "window"]
FEATURE_NAMES = [
    "cx",
    "cy",
    "width",
    "height",
    "area",
    "aspect",
    "adjacency_degree",
    "contained_symbol_count",
    "contained_symbol_density",
    "room_label_count",
    *[f"symbol_count_{label}" for label in SYMBOL_TYPES],
    *[f"symbol_area_{label}" for label in SYMBOL_TYPES],
    *[f"boundary_touch_{label}" for label in BOUNDARY_TYPES],
]
ENHANCED_FEATURE_NAMES = [
    *FEATURE_NAMES,
    "x1",
    "y1",
    "x2",
    "y2",
    "min_page_margin",
    "max_page_margin",
    "touches_left",
    "touches_top",
    "touches_right",
    "touches_bottom",
    "area_rank",
    "area_percentile",
    "rooms_per_page",
    "same_row_neighbors",
    "same_col_neighbors",
]


LABEL_GROUPS = {
    "room": "generic",
    "balcony": "outdoor",
    "closet": "service",
    "storage": "service",
    "bathroom": "sanitary",
    "toilet": "sanitary",
    "bedroom": "activity",
    "living_room": "activity",
    "kitchen": "activity",
    "corridor": "activity",
    "office": "activity",
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def page_size(row: dict[str, Any]) -> tuple[float, float]:
    metadata = row.get("metadata") or {}
    width = metadata.get("width")
    height = metadata.get("height")
    if width and height:
        return float(width), float(height)
    return 1.0, 1.0


def bbox_area(bbox: list[float]) -> float:
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def bbox_contains(left: list[float], right: list[float]) -> bool:
    return left[0] <= right[0] and left[1] <= right[1] and left[2] >= right[2] and left[3] >= right[3]


def bbox_intersects(left: list[float], right: list[float]) -> bool:
    return not (left[2] < right[0] or right[2] < left[0] or left[3] < right[1] or right[3] < left[1])


def overlap_length(left_min: float, left_max: float, right_min: float, right_max: float) -> float:
    return max(0.0, min(left_max, right_max) - max(left_min, right_min))


def adjacent(left: list[float], right: list[float]) -> bool:
    if bbox_contains(left, right) or bbox_contains(right, left):
        return False
    horizontal_gap = max(left[0] - right[2], right[0] - left[2], 0.0)
    vertical_gap = max(left[1] - right[3], right[1] - left[3], 0.0)
    if horizontal_gap > 2.0 or vertical_gap > 2.0:
        return False
    x_overlap = overlap_length(left[0], left[2], right[0], right[2])
    y_overlap = overlap_length(left[1], left[3], right[1], right[3])
    min_side = max(min(left[2] - left[0], left[3] - left[1], right[2] - right[0], right[3] - right[1]), 1.0)
    return max(x_overlap, y_overlap) / min_side >= 0.03


def row_context(row: dict[str, Any]) -> dict[str, Any]:
    expected = row.get("expected_json") or {}
    width, height = page_size(row)
    rooms = [
        {
            "id": str(item.get("id") or f"room_{index}"),
            "room_type": str(item.get("room_type") or "room"),
            "bbox": normalize_bbox(item.get("bbox")) or [0.0, 0.0, 0.0, 0.0],
            "shape_features": item.get("shape_features") if isinstance(item.get("shape_features"), dict) else {},
        }
        for index, item in enumerate(expected.get("room_candidates") or [])
        if isinstance(item, dict) and normalize_bbox(item.get("bbox")) is not None
    ]
    symbols = [
        {
            "id": str(item.get("id") or f"symbol_{index}"),
            "symbol_type": str(item.get("symbol_type") or "generic_symbol"),
            "bbox": normalize_bbox(item.get("bbox")) or [0.0, 0.0, 0.0, 0.0],
        }
        for index, item in enumerate(expected.get("symbol_candidates") or [])
        if isinstance(item, dict) and normalize_bbox(item.get("bbox")) is not None
    ]
    graph = ((row.get("request_hints") or {}).get("primitive_graph") or {})
    boundaries = [
        {
            "semantic_type": str(node.get("semantic_type") or "unknown"),
            "bbox": normalize_bbox(node.get("bbox")) or [0.0, 0.0, 0.0, 0.0],
        }
        for node in graph.get("nodes") or []
        if isinstance(node, dict) and normalize_bbox(node.get("bbox")) is not None
    ]
    adjacency = room_adjacency(rooms)
    return {
        "width": width,
        "height": height,
        "rooms": rooms,
        "symbols": symbols,
        "boundaries": boundaries,
        "adjacency": adjacency,
    }


def room_adjacency(rooms: list[dict[str, Any]]) -> dict[str, int]:
    degrees = {room["id"]: 0 for room in rooms}
    for left_index, left in enumerate(rooms):
        for right in rooms[left_index + 1 :]:
            if adjacent(left["bbox"], right["bbox"]):
                degrees[left["id"]] += 1
                degrees[right["id"]] += 1
    return degrees


def room_feature(room: dict[str, Any], context: dict[str, Any]) -> list[float] | None:
    bbox = room["bbox"]
    width = float(context["width"])
    height = float(context["height"])
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = bbox_area(bbox)
    page_area = max(width * height, 1.0)
    symbol_counts = {label: 0.0 for label in SYMBOL_TYPES}
    symbol_areas = {label: 0.0 for label in SYMBOL_TYPES}
    contained_symbol_count = 0.0
    for symbol in context["symbols"]:
        if bbox_contains(bbox, symbol["bbox"]):
            label = symbol["symbol_type"] if symbol["symbol_type"] in symbol_counts else "generic_symbol"
            contained_symbol_count += 1.0
            symbol_counts[label] += 1.0
            symbol_areas[label] += bbox_area(symbol["bbox"]) / max(area, 1.0)
    boundary_touch = {label: 0.0 for label in BOUNDARY_TYPES}
    for boundary in context["boundaries"]:
        if bbox_intersects(bbox, boundary["bbox"]):
            label = boundary["semantic_type"]
            if label in boundary_touch:
                boundary_touch[label] += 1.0
    room_label_count = 0.0
    adjacency_degree = float(context["adjacency"].get(room["id"], 0))
    return [
        ((x1 + x2) / 2.0) / max(width, 1.0),
        ((y1 + y2) / 2.0) / max(height, 1.0),
        w / max(width, 1.0),
        h / max(height, 1.0),
        area / page_area,
        math.log((w + 1.0) / (h + 1.0)),
        adjacency_degree / 16.0,
        contained_symbol_count / 32.0,
        contained_symbol_count / max(area / 10000.0, 1.0),
        room_label_count / 4.0,
        *[symbol_counts[label] / 16.0 for label in SYMBOL_TYPES],
        *[symbol_areas[label] for label in SYMBOL_TYPES],
        *[boundary_touch[label] / 32.0 for label in BOUNDARY_TYPES],
    ]


def enhanced_room_feature(room: dict[str, Any], context: dict[str, Any]) -> list[float] | None:
    base = room_feature(room, context)
    if base is None:
        return None
    bbox = room["bbox"]
    width = float(context["width"])
    height = float(context["height"])
    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = bbox_area(bbox)
    page_area = max(width * height, 1.0)
    area_rank = 0.0
    area_percentile = 0.0
    if context["rooms"]:
        areas = sorted([bbox_area(item["bbox"]) for item in context["rooms"]], reverse=True)
        area_rank = float(areas.index(area) if area in areas else len(areas)) / max(len(areas) - 1, 1)
        area_percentile = sum(1 for value in areas if value <= area) / max(len(areas), 1)
    margins = [x1 / max(width, 1.0), y1 / max(height, 1.0), (width - x2) / max(width, 1.0), (height - y2) / max(height, 1.0)]
    same_row = 0.0
    same_col = 0.0
    for other in context["rooms"]:
        if other["id"] == room["id"]:
            continue
        other_bbox = other["bbox"]
        if overlap_length(y1, y2, other_bbox[1], other_bbox[3]) / max(min(h, other_bbox[3] - other_bbox[1]), 1.0) > 0.35:
            same_row += 1.0
        if overlap_length(x1, x2, other_bbox[0], other_bbox[2]) / max(min(w, other_bbox[2] - other_bbox[0]), 1.0) > 0.35:
            same_col += 1.0

    return [
        *base,
        x1 / max(width, 1.0),
        y1 / max(height, 1.0),
        x2 / max(width, 1.0),
        y2 / max(height, 1.0),
        min(margins),
        max(margins),
        float(x1 <= max(6.0, min(w, h) * 0.03)),
        float(y1 <= max(6.0, min(w, h) * 0.03)),
        float(width - x2 <= max(6.0, min(w, h) * 0.03)),
        float(height - y2 <= max(6.0, min(w, h) * 0.03)),
        area_rank,
        area_percentile,
        len(context["rooms"]) / 128.0,
        same_row / 32.0,
        same_col / 32.0,
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/cadstruct_cubicasa5k_moe_locked")
    parser.add_argument("--output-dir", default="checkpoints/cadstruct_moe_room_space_grouped_moe_sklearn")
    parser.add_argument("--n-estimators", type=int, default=300)
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=20260430)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_items = collect_items(load_jsonl(input_dir / "train.jsonl"))
    group_encoder = LabelEncoder()
    group_y = group_encoder.fit_transform([label_group(item["label"]) for item in train_items])
    group_model = build_tree(args)
    group_model.fit([item["feature"] for item in train_items], group_y)

    expert_models = {}
    expert_encoders = {}
    grouped_items: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in train_items:
        grouped_items[label_group(item["label"])].append(item)
    for group, items in grouped_items.items():
        labels = sorted({item["label"] for item in items})
        if len(labels) <= 1:
            continue
        encoder = LabelEncoder()
        y = encoder.fit_transform([item["label"] for item in items])
        model = build_tree(args)
        model.fit([item["feature"] for item in items], y)
        expert_models[group] = model
        expert_encoders[group] = encoder

    model_path = output_dir / "model.joblib"
    joblib.dump(
        {
            "group_model": group_model,
            "group_encoder": group_encoder,
            "expert_models": expert_models,
            "expert_encoders": expert_encoders,
            "label_groups": LABEL_GROUPS,
            "feature_names": ENHANCED_FEATURE_NAMES,
        },
        model_path,
    )

    summary: dict[str, Any] = {
        "input_dir": str(input_dir),
        "model": str(model_path),
        "model_type": "room_space_grouped_moe_extra_trees",
        "label_groups": LABEL_GROUPS,
        "feature_names": ENHANCED_FEATURE_NAMES,
        "train_item_counts": dict(Counter(item["label"] for item in train_items)),
        "train_group_counts": dict(Counter(label_group(item["label"]) for item in train_items)),
        "train_group_metrics": {
            "accuracy": float(accuracy_score(group_y, group_model.predict([item["feature"] for item in train_items]))),
            "macro_f1": float(f1_score(group_y, group_model.predict([item["feature"] for item in train_items]), average="macro")),
        },
        "expert_train_counts": {group: dict(Counter(item["label"] for item in items)) for group, items in grouped_items.items()},
        "splits": {},
    }

    for split in ("dev", "locked_test", "smoke"):
        path = input_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = load_jsonl(path)
        predictions = predict_rows(rows, group_model, group_encoder, expert_models, expert_encoders)
        write_jsonl(output_dir / f"{split}_predictions.jsonl", predictions)
        metrics = evaluate_predictions(predictions)
        metrics["routing_audit"] = routing_audit(predictions)
        summary["splits"][split] = metrics

    summary["memory_audit"] = memory_audit()
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_tree(args: argparse.Namespace) -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        class_weight=None,
        random_state=args.seed,
        n_jobs=args.n_jobs,
    )


def collect_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for row in rows:
        context = row_context(row)
        for room in context["rooms"]:
            feature = enhanced_room_feature(room, context)
            if feature is not None:
                items.append({"id": room["id"], "label": room["room_type"], "feature": feature})
    return items


def predict_rows(
    rows: list[dict[str, Any]],
    group_model: ExtraTreesClassifier,
    group_encoder: LabelEncoder,
    expert_models: dict[str, ExtraTreesClassifier],
    expert_encoders: dict[str, LabelEncoder],
) -> list[dict[str, Any]]:
    predictions = []
    for row in rows:
        context = row_context(row)
        rooms = []
        features = []
        for room in context["rooms"]:
            feature = enhanced_room_feature(room, context)
            if feature is not None:
                rooms.append(room)
                features.append(feature)
        room_predictions = []
        if features:
            group_indices = group_model.predict(features)
            group_probs = group_model.predict_proba(features)
            groups = group_encoder.inverse_transform(group_indices)
            expert_batches: dict[str, list[int]] = defaultdict(list)
            for index, group in enumerate(groups):
                expert_batches[str(group)].append(index)
            predicted_labels = [singleton_label(str(group)) for group in groups]
            confidences = [float(max(prob)) for prob in group_probs]
            for group, indices in expert_batches.items():
                model = expert_models.get(group)
                encoder = expert_encoders.get(group)
                if model is None or encoder is None:
                    continue
                group_features = [features[index] for index in indices]
                label_indices = model.predict(group_features)
                label_probs = model.predict_proba(group_features)
                labels = encoder.inverse_transform(label_indices)
                for index, label, label_prob in zip(indices, labels, label_probs):
                    predicted_labels[index] = str(label)
                    confidences[index] *= float(max(label_prob))
            for room, group, label, confidence in zip(rooms, groups, predicted_labels, confidences):
                room_predictions.append(
                    {
                        "id": room["id"],
                        "gold": room["room_type"],
                        "prediction": label,
                        "confidence": confidence,
                        "route": str(group),
                        "gold_route": label_group(room["room_type"]),
                        "bbox": room["bbox"],
                        "iou": 1.0,
                    }
                )
        predictions.append(
            {
                "image": row.get("image_path"),
                "annotation": row.get("annotation_path"),
                "source_dataset": row.get("source_dataset"),
                "rooms": room_predictions,
            }
        )
    return predictions


def routing_audit(predictions: list[dict[str, Any]]) -> dict[str, Any]:
    predicted_counts: Counter[str] = Counter()
    gold_counts: Counter[str] = Counter()
    correct_by_predicted: Counter[str] = Counter()
    correct_by_gold: Counter[str] = Counter()
    for row in predictions:
        for room in row.get("rooms") or []:
            route = str(room.get("route"))
            gold_route = str(room.get("gold_route"))
            predicted_counts[route] += 1
            gold_counts[gold_route] += 1
            if route == gold_route:
                correct_by_predicted[route] += 1
                correct_by_gold[gold_route] += 1
    return {
        "predicted_route_counts": dict(predicted_counts),
        "gold_route_counts": dict(gold_counts),
        "route_precision": {route: correct_by_predicted[route] / max(predicted_counts[route], 1) for route in sorted(predicted_counts)},
        "route_recall": {route: correct_by_gold[route] / max(gold_counts[route], 1) for route in sorted(gold_counts)},
    }


def label_group(label: str) -> str:
    return LABEL_GROUPS.get(str(label), "generic")


def singleton_label(group: str) -> str:
    labels = [label for label, item_group in LABEL_GROUPS.items() if item_group == group]
    return labels[0] if len(labels) == 1 else "room"


def memory_audit() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"max_rss_kb": int(usage.ru_maxrss)}


if __name__ == "__main__":
    main()
