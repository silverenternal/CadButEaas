#!/usr/bin/env python3
"""Train and apply a boundary geometry refiner for v6 visual model output.

The remaining v5 visual defects are not boundary label errors. They are
line-like opening nodes rendered from bbox-only geometry. This script trains a
leakage-free geometry classifier on CubiCasa train/dev records, evaluates it on
the locked split, and applies the adopted refiner to the model stream by adding
explicit line `source_geometry` for line-like boundary nodes.
"""

from __future__ import annotations

import argparse
import json
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score

from v5_pipeline_utils import bbox_area, bbox_aspect, load_jsonl, normalize_bbox, sample_id, write_json, write_jsonl


BOUNDARY_LABELS = ["hard_wall", "door", "window", "opening"]
OPENING_LABELS = {"door", "window", "opening"}
warnings.filterwarnings("ignore", message="`sklearn.utils.parallel.delayed` should be used.*")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/train.jsonl")
    parser.add_argument("--dev", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/dev.jsonl")
    parser.add_argument("--locked", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/locked_test.jsonl")
    parser.add_argument("--model-stream", default="reports/vlm/real_upstream_model_predictions_model_v5.jsonl")
    parser.add_argument("--output-stream", default="reports/vlm/real_upstream_model_predictions_model_v6.jsonl")
    parser.add_argument("--checkpoint", default="checkpoints/boundary_geometry_refiner_v6/model.joblib")
    parser.add_argument("--eval", default="reports/vlm/boundary_geometry_refiner_v6_eval.json")
    parser.add_argument("--summary", default="checkpoints/boundary_geometry_refiner_v6/train_summary.json")
    parser.add_argument("--min-probability", type=float, default=0.55)
    parser.add_argument("--max-train-records", type=int, default=1600)
    parser.add_argument("--max-locked-records", type=int, default=0)
    args = parser.parse_args()

    train_rows = [*load_jsonl(args.train), *load_jsonl(args.dev)]
    locked_rows = load_jsonl(args.locked)
    if args.max_train_records > 0:
        train_rows = train_rows[: args.max_train_records]
    if args.max_locked_records > 0:
        locked_rows = locked_rows[: args.max_locked_records]
    train_ids = {sample_id(row) for row in train_rows if sample_id(row)}
    locked_ids = {sample_id(row) for row in locked_rows if sample_id(row)}
    overlap = sorted(train_ids & locked_ids)
    if overlap:
        raise SystemExit(f"train/locked leakage detected: {len(overlap)} overlapping ids")

    train_items = collect_boundary_items(train_rows)
    locked_items = collect_boundary_items(locked_rows)
    x_train = [item["features"] for item in train_items]
    y_train = [item["target_line_geometry"] for item in train_items]
    x_locked = [item["features"] for item in locked_items]
    y_locked = [item["target_line_geometry"] for item in locked_items]

    model = ExtraTreesClassifier(
        n_estimators=96,
        max_depth=16,
        min_samples_leaf=3,
        class_weight="balanced",
        random_state=20260507,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    locked_pred = list(model.predict(x_locked))
    locked_prob = [float(prob[1]) for prob in model.predict_proba(x_locked)]
    metrics = {
        "accuracy": accuracy_score(y_locked, locked_pred),
        "macro_f1": f1_score(y_locked, locked_pred, average="macro", zero_division=0),
        "positive_rate_gold": sum(y_locked) / max(len(y_locked), 1),
        "positive_rate_pred": sum(locked_pred) / max(len(locked_pred), 1),
        "confusion": confusion(y_locked, locked_pred),
    }
    adopted = metrics["macro_f1"] >= 0.995 and metrics["accuracy"] >= 0.995

    checkpoint = Path(args.checkpoint)
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": [0, 1], "feature_contract": feature_contract()}, checkpoint)

    events = apply_to_stream(args.model_stream, args.output_stream, model, args.min_probability) if adopted else []
    eval_report = {
        "version": "boundary_geometry_refiner_v6_eval",
        "trained": True,
        "adopted": adopted,
        "checkpoint": args.checkpoint,
        "output_stream": args.output_stream if adopted else None,
        "leakage_check": {"train_ids": len(train_ids), "locked_ids": len(locked_ids), "overlap": len(overlap), "passed": not overlap},
        "train_count": len(train_items),
        "locked_count": len(locked_items),
        "record_limits": {"max_train_records": args.max_train_records, "max_locked_records": args.max_locked_records},
        "train_target_counts": dict(Counter(y_train)),
        "locked_target_counts": dict(Counter(y_locked)),
        "locked_metrics": metrics,
        "thresholds": {
            "min_probability": args.min_probability,
            "adoption_accuracy": 0.995,
            "adoption_macro_f1": 0.995,
        },
        "application": {
            "event_count": len(events),
            "by_sample": dict(Counter(event["sample_id"] for event in events).most_common()),
            "semantic_counts": dict(Counter(event["semantic_type"] for event in events).most_common()),
        },
        "claim_boundary": "This is a geometry-output refiner. It does not change boundary semantic labels; it supplies line source_geometry for line-like boundary openings.",
        "examples": events[:20],
    }
    write_json(args.eval, eval_report)
    write_json(args.summary, {"version": "boundary_geometry_refiner_v6_train_summary", **eval_report})
    print(json.dumps({"adopted": adopted, "metrics": metrics, "events": len(events)}, ensure_ascii=False, indent=2))


def collect_boundary_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in rows:
        graph = ((row.get("request_hints") or {}).get("primitive_graph") or {})
        primitives = {
            str(node.get("id")): node
            for node in graph.get("nodes") or []
            if isinstance(node, dict) and node.get("id") is not None
        }
        expected = row.get("expected_json") if isinstance(row.get("expected_json"), dict) else {}
        for candidate in expected.get("semantic_candidates") or []:
            label = str(candidate.get("semantic_type") or "")
            if label not in BOUNDARY_LABELS:
                continue
            target_id = str(candidate.get("target_id", candidate.get("id")))
            primitive = primitives.get(target_id) or candidate
            bbox = normalize_bbox(primitive.get("bbox") or candidate.get("bbox"))
            if not bbox:
                continue
            canvas = page_bounds(primitives.values())
            aspect = bbox_aspect(bbox)
            target = int(label in OPENING_LABELS and aspect >= 45.0)
            items.append(
                {
                    "sample_id": sample_id(row),
                    "candidate_id": target_id,
                    "semantic_type": label,
                    "bbox": bbox,
                    "target_line_geometry": target,
                    "features": feature_row(label, bbox, canvas, primitive),
                }
            )
    return items


def page_bounds(nodes: Any) -> list[float]:
    bboxes = [normalize_bbox(node.get("bbox")) for node in nodes if isinstance(node, dict)]
    bboxes = [bbox for bbox in bboxes if bbox]
    if not bboxes:
        return [0.0, 0.0, 1.0, 1.0]
    return [min(b[0] for b in bboxes), min(b[1] for b in bboxes), max(b[2] for b in bboxes), max(b[3] for b in bboxes)]


def feature_row(label: str, bbox: list[float], canvas: list[float], primitive: dict[str, Any]) -> list[float]:
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    canvas_w = max(canvas[2] - canvas[0], 1e-6)
    canvas_h = max(canvas[3] - canvas[1], 1e-6)
    aspect = bbox_aspect(bbox)
    area_ratio = bbox_area(bbox) / max(canvas_w * canvas_h, 1e-6)
    orientation = str(primitive.get("orientation") or "unknown")
    return [
        width / canvas_w,
        height / canvas_h,
        area_ratio,
        aspect,
        float(aspect >= 45.0),
        float(aspect >= 50.0),
        float(label == "door"),
        float(label == "window"),
        float(label == "opening"),
        float(label == "hard_wall"),
        float(orientation == "horizontal"),
        float(orientation == "vertical"),
        float(orientation == "rectangular"),
        float(orientation == "diagonal"),
    ]


def feature_contract() -> list[str]:
    return [
        "width_over_canvas",
        "height_over_canvas",
        "area_over_canvas",
        "aspect_ratio",
        "aspect_ge_45",
        "aspect_ge_50",
        "semantic_door",
        "semantic_window",
        "semantic_opening",
        "semantic_hard_wall",
        "orientation_horizontal",
        "orientation_vertical",
        "orientation_rectangular",
        "orientation_diagonal",
    ]


def apply_to_stream(input_path: str, output_path: str, model: ExtraTreesClassifier, min_probability: float) -> list[dict[str, Any]]:
    rows = []
    events: list[dict[str, Any]] = []
    for row in load_jsonl(input_path):
        out = json.loads(json.dumps(row, ensure_ascii=False))
        canvas = row_canvas(out)
        sid = sample_id(out)
        for node in ((out.get("scene_graph") or {}).get("nodes") or []):
            if not isinstance(node, dict) or str(node.get("family")) != "boundary":
                continue
            semantic = str(node.get("semantic_type") or "")
            bbox = normalize_bbox((node.get("geometry") or {}).get("bbox") or node.get("bbox"))
            if not bbox or semantic not in BOUNDARY_LABELS:
                continue
            features = feature_row(semantic, bbox, canvas, {})
            prob = float(model.predict_proba([features])[0][1])
            if prob < min_probability:
                continue
            geometry = node.setdefault("geometry", {})
            if not isinstance(geometry, dict):
                continue
            geometry["bbox"] = bbox
            geometry["source_geometry"] = {"type": "line", "points": centerline_from_bbox(bbox)}
            geometry["render_hint"] = "line_like_boundary_centerline"
            metadata = node.setdefault("metadata", {})
            if isinstance(metadata, dict):
                metadata["boundary_geometry_refiner_v6"] = {
                    "probability": round(prob, 6),
                    "model": "boundary_geometry_refiner_v6",
                    "rule_contract": "trained classifier output; line geometry derived from candidate bbox centerline",
                }
                metadata["model_version"] = "model_v6"
            events.append({"sample_id": sid, "node_id": str(node.get("id") or ""), "semantic_type": semantic, "bbox": bbox, "probability": round(prob, 6)})
        out.setdefault("route_trace", {})["boundary_geometry_refiner_v6"] = {
            "model_version": "model_v6",
            "event_count": sum(1 for event in events if event["sample_id"] == sid),
            "claim_boundary": "Adds line source_geometry for line-like boundary nodes; semantic labels are unchanged.",
        }
        rows.append(out)
    write_jsonl(output_path, rows)
    return events


def row_canvas(row: dict[str, Any]) -> list[float]:
    bboxes = []
    for node in ((row.get("scene_graph") or {}).get("nodes") or []):
        if isinstance(node, dict):
            bbox = normalize_bbox((node.get("geometry") or {}).get("bbox") or node.get("bbox"))
            if bbox:
                bboxes.append(bbox)
    if not bboxes:
        return [0.0, 0.0, 1.0, 1.0]
    return [min(b[0] for b in bboxes), min(b[1] for b in bboxes), max(b[2] for b in bboxes), max(b[3] for b in bboxes)]


def centerline_from_bbox(bbox: list[float]) -> list[list[float]]:
    x1, y1, x2, y2 = bbox
    if (x2 - x1) >= (y2 - y1):
        y = round((y1 + y2) / 2.0, 3)
        return [[round(x1, 3), y], [round(x2, 3), y]]
    x = round((x1 + x2) / 2.0, 3)
    return [[x, round(y1, 3)], [x, round(y2, 3)]]


def confusion(gold: list[int], pred: list[int]) -> dict[str, int]:
    counter = Counter((g, p) for g, p in zip(gold, pred))
    return {f"{g}->{p}": count for (g, p), count in sorted(counter.items())}


if __name__ == "__main__":
    main()
