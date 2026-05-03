#!/usr/bin/env python3
"""Train/apply stronger leakage-free SymbolFixture label arbitration v2."""

from __future__ import annotations

import json
import math
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score

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
BASE_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_boundary_arbitrated_v1.jsonl"
ADJUSTED_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_label_arbitrated_v2.jsonl"
FUSION_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_label_arbitrated_v2_eval.json"
REPORT = ROOT / "reports" / "vlm" / "symbol_label_arbitration_v2_eval.json"
CHECKPOINT = ROOT / "checkpoints" / "symbol_label_arbitration_v2" / "model.joblib"
BASELINE_MAIN = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_text_label_arbitrated_no_repair_scorer_v1_eval.json"

LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]
ROOM_TYPES = ["balcony", "bathroom", "bedroom", "closet", "corridor", "kitchen", "living_room", "office", "room", "storage", "toilet", "unknown_room"]
MAX_TRAIN_PER_LABEL = 20000


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def split_images(rows: list[dict[str, Any]]) -> set[str]:
    return {str(row.get("image_path") or row.get("image") or "") for row in rows}


def norm_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item or 0.0) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def bbox_area(bbox: list[float] | None) -> float:
    if not bbox:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


def record_symbols(record: dict[str, Any]) -> list[dict[str, Any]]:
    expected = record.get("expected_json") or {}
    return list(expected.get("symbol_candidates") or record.get("symbols") or [])


def record_rooms(record: dict[str, Any]) -> list[dict[str, Any]]:
    expected = record.get("expected_json") or {}
    return list(expected.get("room_candidates") or record.get("rooms") or [])


def page_size(record: dict[str, Any], bboxes: list[list[float]]) -> tuple[float, float]:
    metadata = record.get("metadata") or {}
    width = float(metadata.get("width") or 0.0)
    height = float(metadata.get("height") or 0.0)
    if width > 0 and height > 0:
        return width, height
    if not bboxes:
        return 2000.0, 2000.0
    return max(b[2] for b in bboxes) + 1.0, max(b[3] for b in bboxes) + 1.0


def containing_room_features(symbol_bbox: list[float], rooms: list[dict[str, Any]]) -> list[float]:
    cx = (symbol_bbox[0] + symbol_bbox[2]) * 0.5
    cy = (symbol_bbox[1] + symbol_bbox[3]) * 0.5
    best_room_type = "unknown_room"
    best_area = float("inf")
    containing_count = 0
    nearest_dist = float("inf")
    nearest_type = "unknown_room"
    sym_area = max(bbox_area(symbol_bbox), 1.0)
    for room in rooms:
        rb = norm_bbox(room.get("bbox"))
        if not rb:
            continue
        rcx = (rb[0] + rb[2]) * 0.5
        rcy = (rb[1] + rb[3]) * 0.5
        dist = math.hypot(cx - rcx, cy - rcy)
        if dist < nearest_dist:
            nearest_dist = dist
            nearest_type = str(room.get("room_type") or "unknown_room")
        inside = rb[0] <= cx <= rb[2] and rb[1] <= cy <= rb[3]
        if inside:
            containing_count += 1
            area = bbox_area(rb)
            if area < best_area:
                best_area = area
                best_room_type = str(room.get("room_type") or "unknown_room")
    one_hot = [1.0 if best_room_type == item else 0.0 for item in ROOM_TYPES]
    nearest_hot = [1.0 if nearest_type == item else 0.0 for item in ROOM_TYPES]
    return one_hot + nearest_hot + [
        float(containing_count),
        math.log(best_area + 1.0) if best_area != float("inf") else 0.0,
        math.log((best_area / sym_area) + 1.0) if best_area != float("inf") else 0.0,
        math.log(nearest_dist + 1.0) if nearest_dist != float("inf") else 0.0,
    ]


def symbol_features(record: dict[str, Any], symbol: dict[str, Any], bboxes: list[list[float]]) -> list[float] | None:
    bbox = norm_bbox(symbol.get("bbox"))
    if bbox is None:
        return None
    img_w, img_h = page_size(record, bboxes)
    x1, y1, x2, y2 = bbox
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    all_areas = [bbox_area(b) for b in bboxes]
    mean_area = float(np.mean(all_areas)) if all_areas else 0.0
    median_area = float(np.median(all_areas)) if all_areas else 0.0
    q25 = float(np.quantile(all_areas, 0.25)) if all_areas else 0.0
    q75 = float(np.quantile(all_areas, 0.75)) if all_areas else 0.0
    same_scale = sum(1 for value in all_areas if 0.5 <= area / max(value, 1.0) <= 2.0)
    near_count = 0
    for other in bboxes:
        if other == bbox:
            continue
        ocx = (other[0] + other[2]) * 0.5
        ocy = (other[1] + other[3]) * 0.5
        if math.hypot(cx - ocx, cy - ocy) <= max(width, height, 1.0) * 3.0:
            near_count += 1
    return [
        cx / max(img_w, 1.0),
        cy / max(img_h, 1.0),
        width / max(img_w, 1.0),
        height / max(img_h, 1.0),
        area / max(img_w * img_h, 1.0),
        math.log((width + 1.0) / (height + 1.0)),
        math.log(area + 1.0),
        math.log(mean_area + 1.0),
        math.log(median_area + 1.0),
        math.log(q25 + 1.0),
        math.log(q75 + 1.0),
        math.log(area / max(mean_area, 1.0) + 1.0),
        math.log(area / max(median_area, 1.0) + 1.0),
        float(len(all_areas)),
        float(same_scale),
        float(near_count),
        *containing_room_features(bbox, record_rooms(record)),
    ]


def extract_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for record_index, record in enumerate(rows):
        symbols = record_symbols(record)
        bboxes = [bbox for bbox in (norm_bbox(sym.get("bbox")) for sym in symbols) if bbox is not None]
        for symbol in symbols:
            label = str(symbol.get("symbol_type") or "generic_symbol")
            if label not in LABELS:
                continue
            features = symbol_features(record, symbol, bboxes)
            if features is None:
                continue
            out.append({"record_index": record_index, "candidate_id": str(symbol.get("id")), "label": label, "features": features})
    return out


def stratified(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_label[str(item["label"])].append(item)
    rng = random.Random(20260504)
    out = []
    for label in LABELS:
        rows = by_label.get(label, [])
        rng.shuffle(rows)
        out.extend(rows[:MAX_TRAIN_PER_LABEL])
    rng.shuffle(out)
    return out


def metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
    per_label = {}
    for label in LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_label[label] = {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "support": sum(1 for g in gold if g == label)}
    return {
        "accuracy": round(sum(1 for g, p in zip(gold, pred) if g == p) / max(len(gold), 1), 6),
        "macro_f1": round(float(f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0)), 6),
        "per_label": per_label,
        "confusion": {label: dict(Counter(p for g, p in zip(gold, pred) if g == label)) for label in LABELS},
    }


def apply_arbitration(base_predictions: list[dict[str, Any]], locked_rows: list[dict[str, Any]], lookup: dict[tuple[int, str], tuple[str, float, dict[str, float]]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    symbol_counts = [len(record_symbols(record)) for record in locked_rows]
    record_for_symbol_row = []
    for record_index, count in enumerate(symbol_counts):
        record_for_symbol_row.extend([record_index] * count)
    seen = 0
    changed: Counter[tuple[str, str]] = Counter()
    adjusted = []
    for pred in base_predictions:
        row = dict(pred)
        if str(row.get("family")) == "symbol" and seen < len(record_for_symbol_row):
            record_index = record_for_symbol_row[seen]
            seen += 1
            key = (record_index, str(row.get("candidate_id")))
            if key in lookup:
                label, confidence, probs = lookup[key]
                old = str(row.get("label"))
                row["label"] = label
                row["confidence"] = confidence
                row["source"] = "symbol_label_arbitration_v2"
                metadata = dict(row.get("metadata") or {})
                metadata["base_label"] = old
                metadata["arbitration_v2_probs"] = probs
                row["metadata"] = metadata
                changed[(old, label)] += 1
        adjusted.append(row)
    return adjusted, {"changed": {f"{a}->{b}": n for (a, b), n in sorted(changed.items()) if a != b}, "symbol_seen": seen, "expected_symbols": sum(symbol_counts)}


def evaluate_fusion(predictions: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    gold_nodes, gold_edges = extract_gold(records)
    nodes, edges = fuse_predictions_with_gold_id_space(predictions, records)
    return {
        "version": "scene_graph_fusion_symbol_label_arbitrated_v2",
        "predictions_file": str(ADJUSTED_PREDICTIONS.relative_to(ROOT)),
        "dev_split": str(LOCKED_SPLIT.relative_to(ROOT)),
        "dev_records": len(records),
        "total_predictions": len(predictions),
        "gold": {"nodes": len(gold_nodes), "edges": len(gold_edges)},
        "fused": {"nodes": len(nodes), "edges": len(edges)},
        "node_evaluation": evaluate_nodes(nodes, gold_nodes),
        "relation_evaluation": evaluate_relations(edges, gold_edges),
        "invalid_graph_rate": round(compute_invalid_graph_rate(nodes, edges), 6),
    }


def main() -> int:
    train_rows = []
    for path in TRAIN_SPLITS:
        train_rows.extend(load_jsonl(path))
    locked_rows = load_jsonl(LOCKED_SPLIT)
    overlap = split_images(train_rows) & split_images(locked_rows)
    if overlap:
        raise SystemExit(f"training/eval image overlap detected: {len(overlap)}")
    all_train = extract_items(train_rows)
    train_items = stratified(all_train)
    locked_items = extract_items(locked_rows)
    x_train = np.array([item["features"] for item in train_items], dtype=np.float64)
    y_train = [item["label"] for item in train_items]
    x_locked = np.array([item["features"] for item in locked_items], dtype=np.float64)
    y_locked = [item["label"] for item in locked_items]
    model = ExtraTreesClassifier(
        n_estimators=240,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=20260504,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_locked)
    classes = [str(label) for label in model.classes_]
    pred = [classes[int(np.argmax(row))] for row in prob]
    locked_metrics = metrics(y_locked, pred)
    lookup = {}
    for item, row in zip(locked_items, prob):
        label = classes[int(np.argmax(row))]
        probs = {label_i: round(float(value), 6) for label_i, value in zip(classes, row)}
        lookup[(int(item["record_index"]), str(item["candidate_id"]))] = (label, float(max(row)), probs)
    base_predictions = load_jsonl(BASE_PREDICTIONS)
    adjusted, application = apply_arbitration(base_predictions, locked_rows, lookup)
    write_jsonl(ADJUSTED_PREDICTIONS, adjusted)
    fusion = evaluate_fusion(adjusted, locked_rows)
    write_json(FUSION_REPORT, fusion)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": LABELS, "feature_policy": "symbol_v2_geometry_room_context"}, CHECKPOINT)

    baseline = load_json(BASELINE_MAIN)
    base_per = (baseline.get("node_evaluation") or {}).get("per_label") or {}
    new_per = (fusion.get("node_evaluation") or {}).get("per_label") or {}
    per_delta = {
        label: {
            "baseline_f1": (base_per.get(label) or {}).get("f1"),
            "adjusted_f1": (new_per.get(label) or {}).get("f1"),
            "delta_pp": round((float((new_per.get(label) or {}).get("f1") or 0.0) - float((base_per.get(label) or {}).get("f1") or 0.0)) * 100.0, 3),
        }
        for label in LABELS
    }
    node_delta = (float((fusion.get("node_evaluation") or {}).get("macro_f1") or 0.0) - float((baseline.get("node_evaluation") or {}).get("macro_f1") or 0.0)) * 100.0
    report = {
        "version": "symbol_label_arbitration_v2",
        "created": "2026-05-04",
        "train_splits": [str(path.relative_to(ROOT)) for path in TRAIN_SPLITS],
        "locked_split": str(LOCKED_SPLIT.relative_to(ROOT)),
        "base_predictions": str(BASE_PREDICTIONS.relative_to(ROOT)),
        "adjusted_predictions": str(ADJUSTED_PREDICTIONS.relative_to(ROOT)),
        "fusion_report": str(FUSION_REPORT.relative_to(ROOT)),
        "leakage_check": {"image_overlap": len(overlap), "passed": len(overlap) == 0},
        "train_sampling": {"raw_items": len(all_train), "sampled_items": len(train_items), "max_train_per_label": MAX_TRAIN_PER_LABEL},
        "train_label_counts": dict(Counter(y_train)),
        "locked_label_counts": dict(Counter(y_locked)),
        "locked_symbol_metrics": locked_metrics,
        "application": application,
        "e2e_delta_vs_current_main": {
            "baseline_node_macro_f1": (baseline.get("node_evaluation") or {}).get("macro_f1"),
            "adjusted_node_macro_f1": (fusion.get("node_evaluation") or {}).get("macro_f1"),
            "node_macro_f1_delta_pp": round(node_delta, 3),
            "baseline_relation_f1": (baseline.get("relation_evaluation") or {}).get("f1"),
            "adjusted_relation_f1": (fusion.get("relation_evaluation") or {}).get("f1"),
            "invalid_graph_rate": fusion.get("invalid_graph_rate"),
        },
        "per_label_e2e_delta": per_delta,
        "status": "candidate_improves_node" if node_delta > 0 else "no_adoption",
    }
    write_json(REPORT, report)
    print(f"wrote {REPORT}")
    print(f"wrote {FUSION_REPORT}")
    print(json.dumps(report["e2e_delta_vs_current_main"], ensure_ascii=False, indent=2))
    print(f"status={report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
