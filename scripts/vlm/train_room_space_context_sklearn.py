#!/usr/bin/env python3
"""Train sklearn tabular baselines for RoomSpace context features."""

from __future__ import annotations

import argparse
import json
import math
import resource
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier, VotingClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.pipeline import make_pipeline

try:
    from train_room_space_context_mlp import (
        BOUNDARY_TYPES,
        FEATURE_NAMES,
        SYMBOL_TYPES,
        adjacent,
        bbox_area,
        bbox_contains,
        bbox_intersects,
        collect_items_from_jsonl,
        load_jsonl,
        overlap_length,
        row_context,
        room_feature,
    )
    from train_room_space_expert import evaluate_predictions, write_jsonl
    from room_text_lexicon import ROOM_TEXT_LABELS, normalize_room_text, room_text_match_vector
except ImportError:
    from scripts.vlm.train_room_space_context_mlp import (
        BOUNDARY_TYPES,
        FEATURE_NAMES,
        SYMBOL_TYPES,
        adjacent,
        bbox_area,
        bbox_contains,
        bbox_intersects,
        collect_items_from_jsonl,
        load_jsonl,
        overlap_length,
        row_context,
        room_feature,
    )
    from scripts.vlm.train_room_space_expert import evaluate_predictions, write_jsonl
    from scripts.vlm.room_text_lexicon import ROOM_TEXT_LABELS, normalize_room_text, room_text_match_vector


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
    "adjacent_area_mean",
    "adjacent_area_max",
    "adjacent_width_mean",
    "adjacent_height_mean",
    "contained_room_count",
    "inside_other_room_count",
    "overlap_room_count",
    "nearest_room_gap",
    "nearest_symbol_gap",
    "room_label_center_count",
    "room_label_overlap_count",
    "dimension_text_overlap_count",
    *[f"symbol_center_count_{label}" for label in SYMBOL_TYPES],
    *[f"symbol_overlap_count_{label}" for label in SYMBOL_TYPES],
    *[f"symbol_near_count_{label}" for label in SYMBOL_TYPES],
    *[f"boundary_intersection_area_{label}" for label in BOUNDARY_TYPES],
    *[f"boundary_center_touch_{label}" for label in BOUNDARY_TYPES],
    "shape_point_count",
    "shape_polygon_area_ratio",
    "shape_polygon_perimeter_norm",
    "shape_bbox_fill_ratio",
    "shape_compactness",
    "linked_room_text_count",
    "linked_room_text_token_count",
    "linked_room_text_area_ratio",
    "linked_room_text_exact_unknown",
    *[f"linked_room_text_match_{label}" for label in ROOM_TEXT_LABELS],
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/cadstruct_cubicasa5k_moe")
    parser.add_argument("--output-dir", default="checkpoints/cadstruct_moe_room_space_context_sklearn")
    parser.add_argument("--model-kind", choices=["extra_trees", "random_forest", "hist_gbdt", "voting"], default="extra_trees")
    parser.add_argument("--feature-set", choices=["base", "enhanced"], default="base")
    parser.add_argument("--n-estimators", type=int, default=800)
    parser.add_argument("--max-depth", type=int, default=0)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--class-weight", choices=["balanced", "balanced_subsample", "none"], default="balanced")
    parser.add_argument("--seed", type=int, default=20260430)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.feature_set == "base":
        feature_names = FEATURE_NAMES
        train_items, train_stream_audit = collect_items_from_jsonl(input_dir / "train.jsonl", output_dir / "train_features.jsonl")
    else:
        feature_names = ENHANCED_FEATURE_NAMES
        train_items, train_stream_audit = collect_enhanced_items_from_jsonl(input_dir / "train.jsonl", output_dir / "train_features.jsonl")
    labels = [item["label"] for item in train_items]
    encoder = LabelEncoder()
    y = encoder.fit_transform(labels)
    x = [item["feature"] for item in train_items]

    model = build_model(args)
    model.fit(x, y)
    model_path = output_dir / "model.joblib"
    joblib.dump({"model": model, "label_encoder": encoder, "feature_names": feature_names, "feature_set": args.feature_set}, model_path)

    summary: dict[str, Any] = {
        "input_dir": str(input_dir),
        "model": str(model_path),
        "model_type": f"room_space_context_{args.model_kind}",
        "feature_set": args.feature_set,
        "feature_names": feature_names,
        "train_stream_audit": train_stream_audit,
        "train_item_counts": dict(Counter(labels)),
        "train_metrics": {
            "accuracy": float(accuracy_score(y, model.predict(x))),
            "macro_f1": float(f1_score(y, model.predict(x), average="macro")),
        },
        "splits": {},
    }
    for split in ("dev", "locked_test", "smoke"):
        path = input_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = load_jsonl(path)
        predictions = predict_rows(rows, model, encoder, args.feature_set)
        write_jsonl(output_dir / f"{split}_predictions.jsonl", predictions)
        summary["splits"][split] = evaluate_predictions(predictions)
        summary["splits"][split]["context_audit"] = context_audit(rows)

    summary["memory_audit"] = memory_audit()
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_model(args: argparse.Namespace):
    max_depth = None if args.max_depth <= 0 else args.max_depth
    class_weight = None if args.class_weight == "none" else args.class_weight
    if args.model_kind == "extra_trees":
        return ExtraTreesClassifier(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight=class_weight,
            random_state=args.seed,
            n_jobs=-1,
        )
    if args.model_kind == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.n_estimators,
            max_depth=max_depth,
            min_samples_leaf=args.min_samples_leaf,
            class_weight=class_weight,
            random_state=args.seed,
            n_jobs=-1,
        )
    if args.model_kind == "hist_gbdt":
        return make_pipeline(
            StandardScaler(),
            HistGradientBoostingClassifier(
                max_iter=max(args.n_estimators // 4, 100),
                max_leaf_nodes=63,
                l2_regularization=0.02,
                class_weight=class_weight if class_weight != "balanced_subsample" else "balanced",
                random_state=args.seed,
            ),
        )
    return VotingClassifier(
        estimators=[
            (
                "extra_trees",
                ExtraTreesClassifier(
                    n_estimators=args.n_estimators,
                    max_depth=max_depth,
                    min_samples_leaf=args.min_samples_leaf,
                    class_weight=class_weight,
                    random_state=args.seed,
                    n_jobs=-1,
                ),
            ),
            (
                "hist_gbdt",
                make_pipeline(
                    StandardScaler(),
                    HistGradientBoostingClassifier(
                        max_iter=max(args.n_estimators // 4, 100),
                        max_leaf_nodes=63,
                        l2_regularization=0.02,
                        class_weight=class_weight if class_weight != "balanced_subsample" else "balanced",
                        random_state=args.seed + 1,
                    ),
                ),
            ),
        ],
        voting="soft",
        n_jobs=-1,
    )


def collect_enhanced_items_from_jsonl(path: Path, feature_cache: Path | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    items = []
    rows = 0
    rooms = 0
    skipped_rooms = 0
    feature_handle = feature_cache.open("w", encoding="utf-8") if feature_cache else None
    try:
        for row in iter_jsonl(path):
            rows += 1
            context = row_context(row)
            for room in context["rooms"]:
                feature = enhanced_room_feature(room, context)
                if feature is None:
                    skipped_rooms += 1
                    continue
                item = {"id": room["id"], "label": room["room_type"], "feature": feature}
                items.append(item)
                rooms += 1
                if feature_handle:
                    feature_handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    finally:
        if feature_handle:
            feature_handle.close()
    return items, {
        "source": str(path),
        "rows": rows,
        "items": len(items),
        "rooms": rooms,
        "skipped_rooms": skipped_rooms,
        "feature_cache": str(feature_cache) if feature_cache else None,
    }


def predict_rows(rows: list[dict[str, Any]], model: Any, encoder: LabelEncoder, feature_set: str) -> list[dict[str, Any]]:
    predictions = []
    for row in rows:
        context = row_context(row)
        room_predictions = []
        features = []
        rooms = []
        for room in context["rooms"]:
            feature = room_feature(room, context) if feature_set == "base" else enhanced_room_feature(room, context)
            if feature is None:
                continue
            rooms.append(room)
            features.append(feature)
        if features:
            pred_indices = model.predict(features)
            probabilities = predict_probabilities(model, features)
            for room, pred_index, probs in zip(rooms, pred_indices, probabilities):
                confidence = float(max(probs)) if probs is not None else 1.0
                room_predictions.append(
                    {
                        "id": room["id"],
                        "gold": room["room_type"],
                        "prediction": str(encoder.inverse_transform([int(pred_index)])[0]),
                        "confidence": confidence,
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
    page_diag = max(math.hypot(width, height), 1.0)
    margins = [x1 / max(width, 1.0), y1 / max(height, 1.0), (width - x2) / max(width, 1.0), (height - y2) / max(height, 1.0)]
    areas = sorted([bbox_area(item["bbox"]) for item in context["rooms"]], reverse=True)
    area_rank = float(areas.index(area) if area in areas else len(areas)) / max(len(areas) - 1, 1)
    area_percentile = sum(1 for value in areas if value <= area) / max(len(areas), 1)

    same_row = 0.0
    same_col = 0.0
    adjacent_areas = []
    adjacent_widths = []
    adjacent_heights = []
    contained_room_count = 0.0
    inside_other_room_count = 0.0
    overlap_room_count = 0.0
    nearest_room_gap = page_diag
    for other in context["rooms"]:
        if other["id"] == room["id"]:
            continue
        other_bbox = other["bbox"]
        other_area = bbox_area(other_bbox)
        if bbox_contains(bbox, other_bbox):
            contained_room_count += 1.0
        if bbox_contains(other_bbox, bbox):
            inside_other_room_count += 1.0
        if bbox_intersects(bbox, other_bbox):
            inter_area = intersection_area(bbox, other_bbox)
            if inter_area > 0.0:
                overlap_room_count += 1.0
        if adjacent(bbox, other_bbox):
            adjacent_areas.append(other_area / page_area)
            adjacent_widths.append(max(0.0, other_bbox[2] - other_bbox[0]) / max(width, 1.0))
            adjacent_heights.append(max(0.0, other_bbox[3] - other_bbox[1]) / max(height, 1.0))
        if overlap_length(y1, y2, other_bbox[1], other_bbox[3]) / max(min(h, other_bbox[3] - other_bbox[1]), 1.0) > 0.35:
            same_row += 1.0
        if overlap_length(x1, x2, other_bbox[0], other_bbox[2]) / max(min(w, other_bbox[2] - other_bbox[0]), 1.0) > 0.35:
            same_col += 1.0
        nearest_room_gap = min(nearest_room_gap, bbox_gap(bbox, other_bbox))

    symbol_center_counts = {label: 0.0 for label in SYMBOL_TYPES}
    symbol_overlap_counts = {label: 0.0 for label in SYMBOL_TYPES}
    symbol_near_counts = {label: 0.0 for label in SYMBOL_TYPES}
    nearest_symbol_gap = page_diag
    near_threshold = max(math.sqrt(max(area, 1.0)) * 0.25, 24.0)
    for symbol in context["symbols"]:
        label = symbol["symbol_type"] if symbol["symbol_type"] in symbol_center_counts else "generic_symbol"
        symbol_bbox = symbol["bbox"]
        if bbox_center_inside(bbox, symbol_bbox):
            symbol_center_counts[label] += 1.0
        if intersection_area(bbox, symbol_bbox) > 0.0:
            symbol_overlap_counts[label] += 1.0
        gap = bbox_gap(bbox, symbol_bbox)
        nearest_symbol_gap = min(nearest_symbol_gap, gap)
        if gap <= near_threshold:
            symbol_near_counts[label] += 1.0

    room_label_center_count = 0.0
    room_label_overlap_count = 0.0
    dimension_text_overlap_count = 0.0
    linked_room_texts = []
    linked_room_text_area = 0.0
    for text in context["texts"]:
        text_type = text["text_type"]
        text_bbox = text["bbox"]
        if text_type == "room_label" and bbox_center_inside(bbox, text_bbox):
            room_label_center_count += 1.0
            text_value = str(text.get("text") or "").strip()
            if text_value:
                linked_room_texts.append(text_value)
                linked_room_text_area += bbox_area(text_bbox)
        if text_type == "room_label" and intersection_area(bbox, text_bbox) > 0.0:
            room_label_overlap_count += 1.0
        if text_type == "dimension_text" and intersection_area(bbox, text_bbox) > 0.0:
            dimension_text_overlap_count += 1.0

    boundary_intersection_areas = {label: 0.0 for label in BOUNDARY_TYPES}
    boundary_center_touch = {label: 0.0 for label in BOUNDARY_TYPES}
    boundary_band = max(6.0, min(w, h) * 0.03)
    expanded = [x1 - boundary_band, y1 - boundary_band, x2 + boundary_band, y2 + boundary_band]
    for boundary in context["boundaries"]:
        label = boundary["semantic_type"]
        if label not in boundary_intersection_areas:
            continue
        boundary_bbox = boundary["bbox"]
        boundary_intersection_areas[label] += intersection_area(bbox, boundary_bbox) / max(area, 1.0)
        if bbox_intersects(expanded, boundary_bbox):
            boundary_center_touch[label] += 1.0
    shape = room.get("shape_features") or {}
    text_match_scores = room_text_match_vector(linked_room_texts)
    normalized_texts = [normalize_room_text(text) for text in linked_room_texts]
    text_token_count = sum(len(text.split()) for text in normalized_texts)
    text_exact_unknown = sum(1 for text in normalized_texts if text in {"undefined", "ulko", "ulkotila", "autokatos"})

    return [
        *base,
        x1 / max(width, 1.0),
        y1 / max(height, 1.0),
        x2 / max(width, 1.0),
        y2 / max(height, 1.0),
        min(margins),
        max(margins),
        float(x1 <= boundary_band),
        float(y1 <= boundary_band),
        float(width - x2 <= boundary_band),
        float(height - y2 <= boundary_band),
        area_rank,
        area_percentile,
        len(context["rooms"]) / 128.0,
        same_row / 32.0,
        same_col / 32.0,
        mean(adjacent_areas),
        max(adjacent_areas) if adjacent_areas else 0.0,
        mean(adjacent_widths),
        mean(adjacent_heights),
        contained_room_count / 8.0,
        inside_other_room_count / 8.0,
        overlap_room_count / 16.0,
        nearest_room_gap / page_diag,
        nearest_symbol_gap / page_diag,
        room_label_center_count / 4.0,
        room_label_overlap_count / 4.0,
        dimension_text_overlap_count / 16.0,
        *[symbol_center_counts[label] / 16.0 for label in SYMBOL_TYPES],
        *[symbol_overlap_counts[label] / 16.0 for label in SYMBOL_TYPES],
        *[symbol_near_counts[label] / 16.0 for label in SYMBOL_TYPES],
        *[boundary_intersection_areas[label] for label in BOUNDARY_TYPES],
        *[boundary_center_touch[label] / 32.0 for label in BOUNDARY_TYPES],
        float(shape.get("point_count") or 0.0) / 32.0,
        float(shape.get("polygon_area") or 0.0) / page_area,
        float(shape.get("polygon_perimeter") or 0.0) / page_diag,
        float(shape.get("bbox_fill_ratio") or 0.0),
        float(shape.get("compactness") or 0.0),
        len(linked_room_texts) / 4.0,
        text_token_count / 16.0,
        linked_room_text_area / max(area, 1.0),
        text_exact_unknown / 4.0,
        *[text_match_scores[label] / 4.0 for label in ROOM_TEXT_LABELS],
    ]


def intersection_area(left: list[float], right: list[float]) -> float:
    return overlap_length(left[0], left[2], right[0], right[2]) * overlap_length(left[1], left[3], right[1], right[3])


def bbox_center_inside(left: list[float], right: list[float]) -> bool:
    cx = (right[0] + right[2]) / 2.0
    cy = (right[1] + right[3]) / 2.0
    return left[0] <= cx <= left[2] and left[1] <= cy <= left[3]


def bbox_gap(left: list[float], right: list[float]) -> float:
    dx = max(left[0] - right[2], right[0] - left[2], 0.0)
    dy = max(left[1] - right[3], right[1] - left[3], 0.0)
    return math.hypot(dx, dy)


def mean(values: list[float]) -> float:
    return sum(values) / max(len(values), 1)


def iter_jsonl(path: Path):
    if not path.exists():
        return
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                yield json.loads(line)


def predict_probabilities(model: Any, features: list[list[float]]) -> list[Any]:
    if hasattr(model, "predict_proba"):
        return list(model.predict_proba(features))
    return [None for _ in features]


def context_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    room_counts = []
    symbol_counts = []
    for row in rows:
        context = row_context(row)
        room_counts.append(len(context["rooms"]))
        symbol_counts.append(len(context["symbols"]))
    return {
        "rows": len(rows),
        "rooms": sum(room_counts),
        "symbols": sum(symbol_counts),
        "max_rooms_per_record": max(room_counts) if room_counts else 0,
        "max_symbols_per_record": max(symbol_counts) if symbol_counts else 0,
    }


def memory_audit() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"max_rss_kb": int(usage.ru_maxrss)}


if __name__ == "__main__":
    main()
