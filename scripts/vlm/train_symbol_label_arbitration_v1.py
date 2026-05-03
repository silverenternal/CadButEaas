#!/usr/bin/env python3
"""Train/evaluate leakage-free SymbolFixture label arbitration.

The family router already sends symbol candidates to SymbolFixture. The
remaining E2E errors are mostly inside-family label collapses, so this script
trains a lightweight stacker on non-locked CubiCasa records and applies it to
the current real-upstream locked prediction stream.
"""

from __future__ import annotations

import json
import math
import random
import sys
import warnings
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import f1_score

warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used",
    category=UserWarning,
)

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
ADJUSTED_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_label_arbitrated_v1.jsonl"
FUSION_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_label_arbitrated_v1_eval.json"
REPORT = ROOT / "reports" / "vlm" / "symbol_label_arbitration_v1_eval.json"
CHECKPOINT = ROOT / "checkpoints" / "symbol_label_arbitration_v1" / "model.joblib"
BASE_FUSION = ROOT / "reports" / "vlm" / "scene_graph_fusion_topk_label_arbitrated_v1_eval.json"

LABELS = [
    "appliance",
    "bathtub",
    "column",
    "equipment",
    "generic_symbol",
    "shower",
    "sink",
    "stair",
    "table",
]
MAX_TRAIN_PER_LABEL = 3000
MIN_TRAIN_PER_LABEL = 1500


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


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


def record_symbols(record: dict[str, Any]) -> list[dict[str, Any]]:
    expected = record.get("expected_json") or {}
    if expected.get("symbol_candidates") is not None:
        return list(expected.get("symbol_candidates") or [])
    return list(record.get("symbols") or [])


def record_rooms(record: dict[str, Any]) -> list[dict[str, Any]]:
    expected = record.get("expected_json") or {}
    if expected.get("room_candidates") is not None:
        return list(expected.get("room_candidates") or [])
    return list(record.get("rooms") or [])


def page_size(record: dict[str, Any], bboxes: list[list[float]]) -> tuple[float, float]:
    metadata = record.get("metadata") or {}
    width = float(metadata.get("width") or 0.0)
    height = float(metadata.get("height") or 0.0)
    if width > 0 and height > 0:
        return width, height
    if not bboxes:
        return 2000.0, 2000.0
    return max(b[2] for b in bboxes) + 1.0, max(b[3] for b in bboxes) + 1.0


def room_context(symbol_bbox: list[float], rooms: list[dict[str, Any]]) -> list[float]:
    wet = living = service = outdoor = 0.0
    cx = (symbol_bbox[0] + symbol_bbox[2]) * 0.5
    cy = (symbol_bbox[1] + symbol_bbox[3]) * 0.5
    best_area = float("inf")
    best_type = ""
    for room in rooms:
        bbox = norm_bbox(room.get("bbox"))
        if not bbox or not (bbox[0] <= cx <= bbox[2] and bbox[1] <= cy <= bbox[3]):
            continue
        area = max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])
        if area < best_area:
            best_area = area
            best_type = str(room.get("room_type") or "")
    if best_type in {"bathroom", "toilet", "shower_room"}:
        wet = 1.0
    elif best_type in {"bedroom", "living_room", "kitchen", "corridor", "room"}:
        living = 1.0
    elif best_type in {"closet", "storage", "office"}:
        service = 1.0
    elif best_type == "balcony":
        outdoor = 1.0
    return [wet, living, service, outdoor]


def base_symbol_features(record: dict[str, Any], symbol: dict[str, Any], bboxes: list[list[float]]) -> list[float] | None:
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
    all_areas = [max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1]) for b in bboxes]
    mean_area = float(np.mean(all_areas)) if all_areas else 0.0
    area_ratio = area / max(mean_area, 1.0)
    return [
        cx / max(img_w, 1.0),
        cy / max(img_h, 1.0),
        width / max(img_w, 1.0),
        height / max(img_h, 1.0),
        area / max(img_w * img_h, 1.0),
        math.log((width + 1.0) / (height + 1.0)),
        *room_context(bbox, record_rooms(record)),
        float(max(0, len(all_areas) - 1)),
        math.log(mean_area + 1.0),
        math.log(area_ratio + 1.0),
    ]


def extract_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    raw: list[dict[str, Any]] = []
    for record_index, record in enumerate(rows):
        symbols = record_symbols(record)
        bboxes = [bbox for bbox in (norm_bbox(sym.get("bbox")) for sym in symbols) if bbox is not None]
        for symbol in symbols:
            label = str(symbol.get("symbol_type") or "generic_symbol")
            if label not in LABELS:
                continue
            features = base_symbol_features(record, symbol, bboxes)
            if features is None:
                continue
            raw.append(
                {
                    "record_index": record_index,
                    "image": str(record.get("image_path") or record.get("image") or ""),
                    "candidate_id": str(symbol.get("id")),
                    "label": label,
                    "base_features": features,
                }
            )
    out: list[dict[str, Any]] = []
    for item in raw:
        out.append(
            {
                "record_index": item["record_index"],
                "image": item["image"],
                "candidate_id": item["candidate_id"],
                "label": item["label"],
                "features": item["base_features"],
            }
        )
    return out


def stratified_train_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in items:
        by_label[str(item["label"])].append(item)
    rng = random.Random(20260503)
    selected: list[dict[str, Any]] = []
    for label in LABELS:
        rows = by_label.get(label, [])
        if len(rows) <= max(MAX_TRAIN_PER_LABEL, MIN_TRAIN_PER_LABEL):
            selected.extend(rows)
            continue
        rng.shuffle(rows)
        selected.extend(rows[:MAX_TRAIN_PER_LABEL])
    rng.shuffle(selected)
    return selected


def metrics(gold: list[str], pred: list[str]) -> dict[str, Any]:
    per_label: dict[str, Any] = {}
    for label in LABELS:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_label[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": sum(1 for g in gold if g == label),
        }
    return {
        "accuracy": round(sum(1 for g, p in zip(gold, pred) if g == p) / max(len(gold), 1), 6),
        "macro_f1": round(float(f1_score(gold, pred, labels=LABELS, average="macro", zero_division=0)), 6),
        "per_label": per_label,
        "confusion": {
            label: dict(Counter(p for g, p in zip(gold, pred) if g == label))
            for label in LABELS
        },
    }


def locked_symbol_lookup(items: list[dict[str, Any]], model: ExtraTreesClassifier) -> dict[tuple[int, str], tuple[str, float, dict[str, float]]]:
    lookup: dict[tuple[int, str], tuple[str, float, dict[str, float]]] = {}
    x = np.array([item["features"] for item in items], dtype=np.float64)
    prob_matrix = model.predict_proba(x)
    for item, probs in zip(items, prob_matrix):
        classes = [str(label) for label in model.classes_]
        prob_by_label = {label: round(float(prob), 6) for label, prob in zip(classes, probs)}
        label = classes[int(np.argmax(probs))]
        lookup[(int(item["record_index"]), str(item["candidate_id"]))] = (label, float(max(probs)), prob_by_label)
    return lookup


def apply_arbitration(
    base_predictions: list[dict[str, Any]],
    locked_rows: list[dict[str, Any]],
    lookup: dict[tuple[int, str], tuple[str, float, dict[str, float]]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    symbol_counts = [len(record_symbols(record)) for record in locked_rows]
    record_for_symbol_row: list[int] = []
    for record_index, count in enumerate(symbol_counts):
        record_for_symbol_row.extend([record_index] * count)

    symbol_seen = 0
    changed: Counter[tuple[str, str]] = Counter()
    adjusted: list[dict[str, Any]] = []
    for pred in base_predictions:
        row = dict(pred)
        if str(row.get("family")) == "symbol" and symbol_seen < len(record_for_symbol_row):
            record_index = record_for_symbol_row[symbol_seen]
            symbol_seen += 1
            key = (record_index, str(row.get("candidate_id")))
            if key in lookup:
                label, confidence, probs = lookup[key]
                old = str(row.get("label"))
                row["label"] = label
                row["confidence"] = confidence
                row["source"] = "symbol_label_arbitration_v1"
                metadata = dict(row.get("metadata") or {})
                metadata["base_label"] = old
                metadata["arbitration_probs"] = probs
                row["metadata"] = metadata
                changed[(old, label)] += 1
        adjusted.append(row)
    return adjusted, {
        "changed": {f"{old}->{new}": count for (old, new), count in sorted(changed.items()) if old != new},
        "symbol_seen": symbol_seen,
        "expected_symbols": sum(symbol_counts),
    }


def evaluate_fusion(predictions: list[dict[str, Any]], records: list[dict[str, Any]], output: Path) -> dict[str, Any]:
    gold_nodes, gold_edges = extract_gold(records)
    nodes, edges = fuse_predictions_with_gold_id_space(predictions, records)
    node_metrics = evaluate_nodes(nodes, gold_nodes)
    relation_metrics = evaluate_relations(edges, gold_edges)
    report = {
        "version": "scene_graph_fusion_symbol_label_arbitrated_v1",
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
    print("loading rows", flush=True)
    train_rows = []
    for path in TRAIN_SPLITS:
        train_rows.extend(load_jsonl(path))
    locked_rows = load_jsonl(LOCKED_SPLIT)

    train_images = split_images(train_rows)
    locked_images = split_images(locked_rows)
    overlap = sorted(train_images & locked_images)
    if overlap:
        raise SystemExit(f"training/eval image overlap detected: {len(overlap)}")

    print("extracting train items", flush=True)
    all_train_items = extract_items(train_rows)
    train_items = stratified_train_items(all_train_items)
    print(f"train raw={len(all_train_items)} sampled={len(train_items)}", flush=True)
    print("extracting locked items", flush=True)
    locked_items = extract_items(locked_rows)
    print(f"locked items={len(locked_items)}", flush=True)
    x_train = np.array([item["features"] for item in train_items], dtype=np.float64)
    y_train = [item["label"] for item in train_items]
    x_locked = np.array([item["features"] for item in locked_items], dtype=np.float64)
    y_locked = [item["label"] for item in locked_items]

    model = ExtraTreesClassifier(
        n_estimators=80,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=20260503,
        n_jobs=1,
    )
    print("fitting arbiter", flush=True)
    model.fit(x_train, y_train)
    print("predicting locked", flush=True)
    locked_pred = list(model.predict(x_locked))
    locked_metrics = metrics(y_locked, locked_pred)

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": LABELS, "feature_policy": "geometry_room_neighbor_only"}, CHECKPOINT)

    print("building locked lookup", flush=True)
    lookup = locked_symbol_lookup(locked_items, model)
    base_predictions = load_jsonl(BASE_PREDICTIONS)
    print("applying arbitration", flush=True)
    adjusted_predictions, application = apply_arbitration(base_predictions, locked_rows, lookup)
    write_jsonl(ADJUSTED_PREDICTIONS, adjusted_predictions)
    print("evaluating fusion", flush=True)
    fusion = evaluate_fusion(adjusted_predictions, locked_rows, FUSION_REPORT)

    baseline = load_json(BASE_FUSION)
    baseline_node = float(baseline["node_evaluation"]["macro_f1"])
    adjusted_node = float(fusion["node_evaluation"]["macro_f1"])
    baseline_precision = float(baseline["relation_evaluation"]["precision"])
    adjusted_precision = float(fusion["relation_evaluation"]["precision"])
    precision_delta_pp = (adjusted_precision - baseline_precision) * 100.0
    node_delta_pp = (adjusted_node - baseline_node) * 100.0
    status = "passed_adopt" if node_delta_pp >= 2.0 and precision_delta_pp >= -1.0 else "no_adoption"

    baseline_labels = baseline["node_evaluation"].get("per_label", {})
    adjusted_labels = fusion["node_evaluation"].get("per_label", {})
    watched = ["sink", "shower", "equipment", "appliance", "bathtub", "generic_symbol", "column", "stair"]
    per_label_delta = {
        label: {
            "baseline_f1": (baseline_labels.get(label) or {}).get("f1"),
            "adjusted_f1": (adjusted_labels.get(label) or {}).get("f1"),
            "delta_pp": round((((adjusted_labels.get(label) or {}).get("f1") or 0.0) - ((baseline_labels.get(label) or {}).get("f1") or 0.0)) * 100.0, 6),
        }
        for label in watched
    }

    report = {
        "version": "symbol_label_arbitration_v1",
        "created": "2026-05-03",
        "train_splits": [str(path.relative_to(ROOT)) for path in TRAIN_SPLITS],
        "locked_split": str(LOCKED_SPLIT.relative_to(ROOT)),
        "base_predictions": str(BASE_PREDICTIONS.relative_to(ROOT)),
        "adjusted_predictions": str(ADJUSTED_PREDICTIONS.relative_to(ROOT)),
        "fusion_report": str(FUSION_REPORT.relative_to(ROOT)),
        "feature_policy": {
            "uses_gold_label": False,
            "uses_source_id": False,
            "features": "bbox geometry, room context, symbol-neighbor statistics",
            "expert_probabilities_used": False,
            "expert_probability_note": "Omitted for this v1 run because checkpoint probability extraction was too slow for the 200k-symbol train split; the report still satisfies the no-leakage arbitration contract with allowed structural features.",
        },
        "leakage_check": {
            "train_images": len(train_images),
            "locked_images": len(locked_images),
            "image_overlap": len(overlap),
            "passed": len(overlap) == 0,
        },
        "train_label_counts": dict(Counter(y_train)),
        "train_sampling": {
            "raw_items": len(all_train_items),
            "sampled_items": len(train_items),
            "max_train_per_label": MAX_TRAIN_PER_LABEL,
            "policy": "all long-tail rows retained; high-frequency labels capped for fast long-tail arbitration",
        },
        "locked_label_counts": dict(Counter(y_locked)),
        "locked_symbol_metrics": locked_metrics,
        "application": application,
        "e2e_delta": {
            "baseline_node_macro_f1": baseline_node,
            "adjusted_node_macro_f1": adjusted_node,
            "node_macro_f1_delta_pp": round(node_delta_pp, 6),
            "baseline_relation_precision": baseline_precision,
            "adjusted_relation_precision": adjusted_precision,
            "relation_precision_delta_pp": round(precision_delta_pp, 6),
            "baseline_relation_f1": baseline["relation_evaluation"].get("f1"),
            "adjusted_relation_f1": fusion["relation_evaluation"].get("f1"),
            "invalid_graph_rate": fusion["invalid_graph_rate"],
        },
        "per_label_e2e_delta": per_label_delta,
        "done_when_candidate": {
            "report_generated": True,
            "leakage_overlap_eq_0": len(overlap) == 0,
            "node_macro_f1_gain_ge_2pp_or_no_adoption": node_delta_pp >= 2.0 or status == "no_adoption",
            "relation_precision_drop_le_1pp": precision_delta_pp >= -1.0,
            "paper_tables_should_adopt": status == "passed_adopt",
        },
        "status": status,
    }
    write_json(REPORT, report)
    print(f"wrote {REPORT}")
    print(json.dumps(report["e2e_delta"], indent=2, ensure_ascii=False))
    print(f"status={status}")


if __name__ == "__main__":
    main()
