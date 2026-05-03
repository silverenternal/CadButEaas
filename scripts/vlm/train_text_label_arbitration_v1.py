#!/usr/bin/env python3
"""Train/apply a leakage-free text-family label arbiter for E2E node F1."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
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
BASE_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_label_arbitrated_v1.jsonl"
ADJUSTED_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_text_label_arbitrated_v1.jsonl"
BASE_FUSION = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_label_arbitrated_no_repair_scorer_v1_eval.json"
FUSION_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_text_label_arbitrated_v1_eval.json"
REPORT = ROOT / "reports" / "vlm" / "text_label_arbitration_v1_eval.json"
CHECKPOINT = ROOT / "checkpoints" / "text_label_arbitration_v1" / "model.joblib"

LABELS = ["dimension_line", "dimension_text", "leader_line", "note_text", "room_label"]


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


def record_texts(record: dict[str, Any]) -> list[dict[str, Any]]:
    expected = record.get("expected_json") or {}
    return list(expected.get("text_candidates") or record.get("text_candidates") or [])


def page_size(record: dict[str, Any], bboxes: list[list[float]]) -> tuple[float, float]:
    metadata = record.get("metadata") or {}
    width = float(metadata.get("width") or 0.0)
    height = float(metadata.get("height") or 0.0)
    if width > 0 and height > 0:
        return width, height
    if not bboxes:
        return 2000.0, 2000.0
    return max(b[2] for b in bboxes) + 1.0, max(b[3] for b in bboxes) + 1.0


def text_features(record: dict[str, Any], text_item: dict[str, Any], bboxes: list[list[float]]) -> list[float] | None:
    bbox = norm_bbox(text_item.get("bbox"))
    if bbox is None:
        return None
    img_w, img_h = page_size(record, bboxes)
    x1, y1, x2, y2 = bbox
    width = max(0.0, x2 - x1)
    height = max(0.0, y2 - y1)
    area = width * height
    cx = (x1 + x2) * 0.5
    cy = (y1 + y2) * 0.5
    text = str(text_item.get("text") or "")
    stripped = text.strip()
    digit_count = sum(ch.isdigit() for ch in stripped)
    alpha_count = sum(ch.isalpha() for ch in stripped)
    has_unit = any(unit in stripped.lower() for unit in ["m", "cm", "mm", "'", '"'])
    all_areas = [max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) for b in bboxes]
    mean_area = float(np.mean(all_areas)) if all_areas else 0.0
    return [
        cx / max(img_w, 1.0),
        cy / max(img_h, 1.0),
        width / max(img_w, 1.0),
        height / max(img_h, 1.0),
        area / max(img_w * img_h, 1.0),
        math.log((width + 1.0) / (height + 1.0)),
        math.log(area + 1.0),
        math.log(mean_area + 1.0),
        float(len(stripped) == 0),
        float(len(stripped)),
        float(digit_count),
        float(alpha_count),
        float(digit_count > 0),
        float(alpha_count > 0),
        float(has_unit),
        float(stripped.isupper() and alpha_count > 0),
        float(max(0, len(all_areas) - 1)),
    ]


def extract_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for record_index, record in enumerate(rows):
        texts = record_texts(record)
        bboxes = [bbox for bbox in (norm_bbox(item.get("bbox")) for item in texts) if bbox is not None]
        for item in texts:
            label = str(item.get("text_type") or "")
            if label not in LABELS:
                continue
            features = text_features(record, item, bboxes)
            if features is None:
                continue
            out.append(
                {
                    "record_index": record_index,
                    "candidate_id": str(item.get("id")),
                    "label": label,
                    "features": features,
                }
            )
    return out


def label_metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
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
    }


def apply_text_arbitration(
    base_predictions: list[dict[str, Any]],
    locked_rows: list[dict[str, Any]],
    lookup: dict[tuple[int, str], tuple[str, float, dict[str, float]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    text_counts = [len(record_texts(record)) for record in locked_rows]
    record_for_text_row: list[int] = []
    for record_index, count in enumerate(text_counts):
        record_for_text_row.extend([record_index] * count)
    text_seen = 0
    changed: Counter[tuple[str, str]] = Counter()
    adjusted = []
    for pred in base_predictions:
        row = dict(pred)
        if str(row.get("family")) == "text" and text_seen < len(record_for_text_row):
            record_index = record_for_text_row[text_seen]
            text_seen += 1
            key = (record_index, str(row.get("candidate_id")))
            if key in lookup:
                label, confidence, probs = lookup[key]
                old = str(row.get("label"))
                row["label"] = label
                row["confidence"] = confidence
                row["source"] = "text_label_arbitration_v1"
                metadata = dict(row.get("metadata") or {})
                metadata["base_label"] = old
                metadata["text_arbitration_probs"] = probs
                row["metadata"] = metadata
                changed[(old, label)] += 1
        adjusted.append(row)
    return adjusted, {
        "changed": {f"{old}->{new}": count for (old, new), count in sorted(changed.items()) if old != new},
        "text_seen": text_seen,
        "expected_texts": sum(text_counts),
    }


def evaluate_fusion(predictions: list[dict[str, Any]], records: list[dict[str, Any]]) -> dict[str, Any]:
    gold_nodes, gold_edges = extract_gold(records)
    nodes, edges = fuse_predictions_with_gold_id_space(predictions, records)
    return {
        "version": "scene_graph_fusion_symbol_text_label_arbitrated_v1",
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
    train_rows: list[dict[str, Any]] = []
    for path in TRAIN_SPLITS:
        train_rows.extend(load_jsonl(path))
    locked_rows = load_jsonl(LOCKED_SPLIT)
    overlap = split_images(train_rows) & split_images(locked_rows)
    if overlap:
        raise SystemExit(f"training/eval image overlap detected: {len(overlap)}")

    train_items = extract_items(train_rows)
    locked_items = extract_items(locked_rows)
    x_train = np.array([item["features"] for item in train_items], dtype=np.float64)
    y_train = [item["label"] for item in train_items]
    x_locked = np.array([item["features"] for item in locked_items], dtype=np.float64)
    y_locked = [item["label"] for item in locked_items]
    model = ExtraTreesClassifier(
        n_estimators=160,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=20260503,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    prob = model.predict_proba(x_locked)
    classes = [str(label) for label in model.classes_]
    pred = [classes[int(np.argmax(row))] for row in prob]
    locked_metrics = label_metrics(y_locked, pred)
    lookup = {}
    for item, row in zip(locked_items, prob):
        probs = {label: round(float(value), 6) for label, value in zip(classes, row)}
        label = classes[int(np.argmax(row))]
        lookup[(int(item["record_index"]), str(item["candidate_id"]))] = (label, float(max(row)), probs)

    base_predictions = load_jsonl(BASE_PREDICTIONS)
    adjusted, application = apply_text_arbitration(base_predictions, locked_rows, lookup)
    write_jsonl(ADJUSTED_PREDICTIONS, adjusted)
    fusion = evaluate_fusion(adjusted, locked_rows)
    write_json(FUSION_REPORT, fusion)
    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": LABELS, "feature_policy": "bbox_text_geometry"}, CHECKPOINT)

    baseline = load_json(BASE_FUSION)
    base_node = (baseline.get("node_evaluation") or {})
    new_node = fusion["node_evaluation"]
    base_text = {label: ((base_node.get("per_label") or {}).get(label) or {}) for label in LABELS}
    new_text = {label: ((new_node.get("per_label") or {}).get(label) or {}) for label in LABELS}
    per_label_delta = {
        label: {
            "baseline_f1": base_text[label].get("f1"),
            "adjusted_f1": new_text[label].get("f1"),
            "delta_pp": round((float(new_text[label].get("f1") or 0.0) - float(base_text[label].get("f1") or 0.0)) * 100.0, 3),
        }
        for label in LABELS
    }
    node_delta = (float(new_node.get("macro_f1") or 0.0) - float(base_node.get("macro_f1") or 0.0)) * 100.0
    report = {
        "version": "text_label_arbitration_v1",
        "created": "2026-05-03",
        "train_splits": [str(path.relative_to(ROOT)) for path in TRAIN_SPLITS],
        "locked_split": str(LOCKED_SPLIT.relative_to(ROOT)),
        "base_predictions": str(BASE_PREDICTIONS.relative_to(ROOT)),
        "adjusted_predictions": str(ADJUSTED_PREDICTIONS.relative_to(ROOT)),
        "fusion_report": str(FUSION_REPORT.relative_to(ROOT)),
        "leakage_check": {"image_overlap": len(overlap), "passed": len(overlap) == 0},
        "train_label_counts": dict(Counter(y_train)),
        "locked_label_counts": dict(Counter(y_locked)),
        "locked_text_metrics": locked_metrics,
        "application": application,
        "e2e_delta": {
            "baseline_node_macro_f1": base_node.get("macro_f1"),
            "adjusted_node_macro_f1": new_node.get("macro_f1"),
            "node_macro_f1_delta_pp": round(node_delta, 3),
            "baseline_relation_f1": (baseline.get("relation_evaluation") or {}).get("f1"),
            "adjusted_relation_f1": fusion["relation_evaluation"].get("f1"),
            "invalid_graph_rate": fusion["invalid_graph_rate"],
        },
        "per_label_e2e_delta": per_label_delta,
        "status": "passed_adopt" if node_delta > 0.1 else "no_adoption",
    }
    write_json(REPORT, report)
    print(f"wrote {REPORT}")
    print(f"wrote {FUSION_REPORT}")
    print(f"wrote {ADJUSTED_PREDICTIONS}")
    print(json.dumps(report["e2e_delta"], ensure_ascii=False, indent=2))
    print(f"status={report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
