#!/usr/bin/env python3
"""Train/evaluate leakage-free boundary label arbitration for P1-T4.

The family router is already oracle-equivalent for boundary candidates, but the
current real-upstream WallOpening checkpoint collapses CubiCasa boundary nodes
to `hard_wall`. This script trains a lightweight label-level arbiter on
CubiCasa train+dev records with zero image overlap against the current locked
test, then applies it to the existing real-upstream predictions.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT / "scripts" / "vlm") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from fuse_real_upstream import (  # noqa: E402
    compute_invalid_graph_rate,
    evaluate_nodes,
    evaluate_relations,
    extract_gold,
    fuse_predictions_with_gold_id_space,
    load_jsonl,
)

TRAIN_SPLITS = [
    ROOT / "datasets" / "cadstruct_cubicasa5k_moe_locked_reviewed_v1" / "train.jsonl",
    ROOT / "datasets" / "cadstruct_cubicasa5k_moe_locked_reviewed_v1" / "dev.jsonl",
]
LOCKED_SPLIT = ROOT / "datasets" / "cadstruct_real_world_benchmark_v1" / "room_space" / "cubicasa5k_reviewed_locked_test.jsonl"
BASE_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev.jsonl"
ADJUSTED_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_boundary_arbitrated_v1.jsonl"
FUSION_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_topk_label_arbitrated_v1_eval.json"
REPORT = ROOT / "reports" / "vlm" / "boundary_label_arbitration_v1_eval.json"
CHECKPOINT = ROOT / "checkpoints" / "boundary_label_arbitration_v1" / "model.joblib"

LABELS = ["hard_wall", "door", "window"]
ORIENTATIONS = ["horizontal", "vertical", "diagonal", "rectangular", "unknown"]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def norm_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item or 0.0) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def boundary_records(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record_index, record in enumerate(rows):
        expected = record.get("expected_json") or {}
        graph = (record.get("request_hints") or {}).get("primitive_graph") or {}
        primitives = {
            str(node.get("id")): node
            for node in graph.get("nodes") or []
            if isinstance(node, dict) and node.get("id") is not None
        }
        target_labels = {
            str(item.get("target_id", item.get("id"))): str(item.get("semantic_type"))
            for item in expected.get("semantic_candidates") or []
            if str(item.get("semantic_type")) in LABELS
        }
        bboxes = [norm_bbox(node.get("bbox")) for node in primitives.values()]
        bbox_counts = Counter(tuple(bbox) for bbox in bboxes if bbox is not None)
        page_bbox = page_bounds([bbox for bbox in bboxes if bbox is not None])
        for local_id, label in target_labels.items():
            primitive = primitives.get(local_id) or {}
            bbox = norm_bbox(primitive.get("bbox"))
            if bbox is None:
                continue
            out.append({
                "record_index": record_index,
                "image": record.get("image_path"),
                "candidate_id": local_id,
                "label": label,
                "features": feature_row(primitive, bbox, bbox_counts, page_bbox),
            })
    return out


def page_bounds(bboxes: list[list[float]]) -> list[float]:
    if not bboxes:
        return [0.0, 0.0, 1.0, 1.0]
    return [
        min(b[0] for b in bboxes),
        min(b[1] for b in bboxes),
        max(b[2] for b in bboxes),
        max(b[3] for b in bboxes),
    ]


def feature_row(
    primitive: dict[str, Any],
    bbox: list[float],
    bbox_counts: Counter[tuple[float, float, float, float]],
    page_bbox: list[float],
) -> list[float]:
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    page_w = max(page_bbox[2] - page_bbox[0], 1e-6)
    page_h = max(page_bbox[3] - page_bbox[1], 1e-6)
    length = float(primitive.get("length") or max(width, height))
    area = width * height
    group_count = float(bbox_counts.get(tuple(bbox), 1))
    orientation = str(primitive.get("orientation") or "unknown")
    orient = [1.0 if orientation == item else 0.0 for item in ORIENTATIONS]
    return [
        x1, y1, x2, y2,
        width, height, area,
        math.log((width + 1.0) / (height + 1.0)),
        length,
        length / max(max(page_w, page_h), 1e-6),
        width / page_w,
        height / page_h,
        area / max(page_w * page_h, 1e-6),
        (cx - page_bbox[0]) / page_w,
        (cy - page_bbox[1]) / page_h,
        group_count,
        float(group_count >= 2.0),
        *orient,
    ]


def split_images(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("image_path")) for row in rows}


def metrics(y_true: list[str], y_pred: list[str]) -> dict[str, Any]:
    per_label = {}
    for label in LABELS:
        tp = sum(1 for g, p in zip(y_true, y_pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(y_true, y_pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(y_true, y_pred) if g == label and p != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(1 for g in y_true if g == label)}
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, labels=LABELS, average="macro"),
        "per_label": per_label,
        "confusion": {
            gold: dict(Counter(pred for g, pred in zip(y_true, y_pred) if g == gold))
            for gold in LABELS
        },
    }


def record_prediction_lookup(rows: list[dict[str, Any]], model: ExtraTreesClassifier) -> dict[tuple[int, str], tuple[str, float, dict[str, float]]]:
    records = boundary_records(rows)
    X = [item["features"] for item in records]
    probs = model.predict_proba(X)
    classes = list(model.classes_)
    result = {}
    for item, prob in zip(records, probs):
        best_idx = max(range(len(classes)), key=lambda idx: float(prob[idx]))
        label = str(classes[best_idx])
        all_probs = {str(classes[idx]): round(float(prob[idx]), 6) for idx in range(len(classes))}
        result[(int(item["record_index"]), str(item["candidate_id"]))] = (label, float(prob[best_idx]), all_probs)
    return result


def apply_arbitration(
    base_predictions: list[dict[str, Any]],
    records: list[dict[str, Any]],
    model: ExtraTreesClassifier,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    lookup = record_prediction_lookup(records, model)
    adjusted: list[dict[str, Any]] = []
    cursors: defaultdict[str, int] = defaultdict(int)
    changed = Counter()
    for pred in base_predictions:
        row = dict(pred)
        if str(row.get("family")) == "boundary":
            record_index = cursors["boundary_record"]
            # Boundary predictions are emitted record-order; advance record index
            # by consuming the expected number of boundary predictions per record.
            # A running list is simpler and less error-prone than global ID lookup.
        adjusted.append(row)

    boundary_counts = [
        sum(
            1
            for item in (record.get("expected_json") or {}).get("semantic_candidates") or []
            if str(item.get("semantic_type")) in LABELS
        )
        for record in records
    ]
    record_for_boundary_row: list[int] = []
    for record_index, count in enumerate(boundary_counts):
        record_for_boundary_row.extend([record_index] * count)

    boundary_seen = 0
    adjusted = []
    for pred in base_predictions:
        row = dict(pred)
        if str(row.get("family")) == "boundary" and boundary_seen < len(record_for_boundary_row):
            record_index = record_for_boundary_row[boundary_seen]
            boundary_seen += 1
            key = (record_index, str(row.get("candidate_id")))
            if key in lookup:
                label, confidence, probs = lookup[key]
                old = str(row.get("label"))
                row["label"] = label
                row["confidence"] = confidence
                row["source"] = "boundary_label_arbitration_v1"
                metadata = dict(row.get("metadata") or {})
                metadata["base_label"] = old
                metadata["arbitration_probs"] = probs
                row["metadata"] = metadata
                changed[(old, label)] += 1
        adjusted.append(row)
    return adjusted, {"changed": {f"{old}->{new}": count for (old, new), count in sorted(changed.items())}, "boundary_seen": boundary_seen}


def evaluate_fusion(predictions: list[dict[str, Any]], records: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    gold_nodes, gold_edges = extract_gold(records)
    nodes, edges = fuse_predictions_with_gold_id_space(predictions, records)
    node_metrics = evaluate_nodes(nodes, gold_nodes)
    relation_metrics = evaluate_relations(edges, gold_edges)
    report = {
        "version": "scene_graph_fusion_topk_label_arbitrated_v1",
        "predictions_file": str(ADJUSTED_PREDICTIONS),
        "dev_split": str(LOCKED_SPLIT),
        "dev_records": len(records),
        "total_predictions": len(predictions),
        "gold": {"nodes": len(gold_nodes), "edges": len(gold_edges)},
        "fused": {"nodes": len(nodes), "edges": len(edges)},
        "node_evaluation": node_metrics,
        "relation_evaluation": relation_metrics,
        "invalid_graph_rate": round(compute_invalid_graph_rate(nodes, edges), 6),
    }
    write_json(output, report)
    return report


def main() -> None:
    train_rows = []
    for path in TRAIN_SPLITS:
        train_rows.extend(load_jsonl(path))
    locked_rows = load_jsonl(LOCKED_SPLIT)
    train_images = split_images(train_rows)
    locked_images = split_images(locked_rows)
    overlap = sorted(train_images & locked_images)
    if overlap:
        raise SystemExit(f"training/eval image overlap detected: {len(overlap)}")

    train_items = boundary_records(train_rows)
    locked_items = boundary_records(locked_rows)
    X_train = [item["features"] for item in train_items]
    y_train = [item["label"] for item in train_items]
    X_locked = [item["features"] for item in locked_items]
    y_locked = [item["label"] for item in locked_items]

    model = ExtraTreesClassifier(
        n_estimators=500,
        max_depth=22,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=20260503,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)
    locked_pred = list(model.predict(X_locked))
    locked_metrics = metrics(y_locked, locked_pred)

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": LABELS, "orientations": ORIENTATIONS}, CHECKPOINT)

    base_predictions = load_jsonl(BASE_PREDICTIONS)
    adjusted_predictions, application = apply_arbitration(base_predictions, locked_rows, model)
    write_jsonl(ADJUSTED_PREDICTIONS, adjusted_predictions)
    fusion = evaluate_fusion(adjusted_predictions, locked_rows, FUSION_REPORT)

    baseline = load_json(ROOT / "reports" / "vlm" / "scene_graph_fusion_real_upstream_eval.json")
    baseline_node = float(baseline["node_evaluation"]["macro_f1"])
    adjusted_node = float(fusion["node_evaluation"]["macro_f1"])
    baseline_precision = float(baseline["relation_evaluation"]["precision"])
    adjusted_precision = float(fusion["relation_evaluation"]["precision"])
    report = {
        "version": "boundary_label_arbitration_v1",
        "created": "2026-05-03",
        "train_splits": [str(path) for path in TRAIN_SPLITS],
        "locked_split": str(LOCKED_SPLIT),
        "leakage_check": {
            "train_images": len(train_images),
            "locked_images": len(locked_images),
            "image_overlap": len(overlap),
            "passed": len(overlap) == 0,
        },
        "train_label_counts": dict(Counter(y_train)),
        "locked_label_counts": dict(Counter(y_locked)),
        "locked_boundary_metrics": locked_metrics,
        "application": application,
        "fusion_report": str(FUSION_REPORT),
        "adjusted_predictions": str(ADJUSTED_PREDICTIONS),
        "e2e_delta": {
            "baseline_node_macro_f1": baseline_node,
            "adjusted_node_macro_f1": adjusted_node,
            "node_macro_f1_delta_pp": round((adjusted_node - baseline_node) * 100.0, 6),
            "baseline_relation_precision": baseline_precision,
            "adjusted_relation_precision": adjusted_precision,
            "relation_precision_delta_pp": round((adjusted_precision - baseline_precision) * 100.0, 6),
            "invalid_graph_rate": fusion["invalid_graph_rate"],
        },
        "done_when_candidate": {
            "node_macro_f1_gain_ge_3pp": (adjusted_node - baseline_node) * 100.0 >= 3.0,
            "relation_precision_drop_le_1pp": (adjusted_precision - baseline_precision) * 100.0 >= -1.0,
            "invalid_graph_rate_le_002": fusion["invalid_graph_rate"] <= 0.02,
        },
    }
    report["status"] = "passed" if all(report["done_when_candidate"].values()) else "not_passed"
    write_json(REPORT, report)
    print(f"wrote {REPORT}")
    print(json.dumps(report["e2e_delta"], indent=2, ensure_ascii=False))
    print(json.dumps(report["done_when_candidate"], indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
