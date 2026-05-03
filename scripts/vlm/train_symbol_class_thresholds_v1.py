#!/usr/bin/env python3
"""Dev-calibrated class-threshold overrides for long-tail symbol labels."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT / "scripts" / "vlm") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from fuse_real_upstream import load_jsonl  # noqa: E402
from train_symbol_label_arbitration_v2 import (  # noqa: E402
    LABELS,
    LOCKED_SPLIT,
    apply_arbitration,
    evaluate_fusion,
    metrics,
    norm_bbox,
    record_rooms,
    record_symbols,
    split_images,
    stratified,
    write_json,
    write_jsonl,
)


TRAIN_ONLY = ROOT / "datasets" / "cadstruct_cubicasa5k_moe_locked_reviewed_v1" / "train.jsonl"
DEV_ONLY = ROOT / "datasets" / "cadstruct_cubicasa5k_moe_locked_reviewed_v1" / "dev.jsonl"
CURRENT_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_v2_text_conservative_generic_override_v1.jsonl"
CURRENT_FUSION = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_v2_text_conservative_generic_override_v1_eval.json"
ADJUSTED_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_v2_text_conservative_generic_class_threshold_v1.jsonl"
FUSION_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_v2_text_conservative_generic_class_threshold_v1_eval.json"
REPORT = ROOT / "reports" / "vlm" / "symbol_class_thresholds_v1_eval.json"
CHECKPOINT = ROOT / "checkpoints" / "symbol_class_thresholds_v1" / "model.joblib"
CACHE_DIR = ROOT / "reports" / "vlm" / "cache" / "symbol_class_thresholds_v1"
FORMAL_REPORT = ROOT / "reports" / "vlm" / "symbol_long_tail_threshold_formal_v1_eval.json"
FORMAL_FUSION_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_long_tail_threshold_formal_v1_eval.json"
FORMAL_SCORER_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_long_tail_threshold_formal_no_repair_scorer_v1_eval.json"
FORMAL_DECISION = ROOT / "reports" / "vlm" / "relation_scorer_symbol_long_tail_threshold_formal_adoption_v1.json"
METRIC_SUMMARY = ROOT / "reports" / "vlm" / "metric_improvement_summary_v5.json"

TARGET_LABELS = ["bathtub", "column", "equipment", "generic_symbol", "stair", "appliance"]
MAX_FAST_TRAIN_PER_LABEL = 2200
ROOM_TYPES = ["balcony", "bathroom", "bedroom", "closet", "corridor", "kitchen", "living_room", "office", "room", "storage", "toilet", "unknown_room"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def bbox_area(bbox: list[float] | None) -> float:
    if not bbox:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


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
        dist = float(np.hypot(cx - rcx, cy - rcy))
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
        float(np.log(best_area + 1.0)) if best_area != float("inf") else 0.0,
        float(np.log((best_area / sym_area) + 1.0)) if best_area != float("inf") else 0.0,
        float(np.log(nearest_dist + 1.0)) if nearest_dist != float("inf") else 0.0,
    ]


def fast_extract_items(rows: list[dict[str, Any]], cache_name: str) -> list[dict[str, Any]]:
    cache_path = CACHE_DIR / f"{cache_name}.json"
    if cache_path.exists():
        return json.loads(cache_path.read_text(encoding="utf-8"))
    out: list[dict[str, Any]] = []
    for record_index, record in enumerate(rows):
        symbols = record_symbols(record)
        bboxes_by_id: dict[str, list[float]] = {}
        for symbol in symbols:
            bbox = norm_bbox(symbol.get("bbox"))
            if bbox is not None:
                bboxes_by_id[str(symbol.get("id"))] = bbox
        bboxes = list(bboxes_by_id.values())
        areas = np.array([bbox_area(bbox) for bbox in bboxes], dtype=float)
        mean_area = float(np.mean(areas)) if len(areas) else 0.0
        median_area = float(np.median(areas)) if len(areas) else 0.0
        q25 = float(np.quantile(areas, 0.25)) if len(areas) else 0.0
        q75 = float(np.quantile(areas, 0.75)) if len(areas) else 0.0
        centers = np.array([[(bbox[0] + bbox[2]) * 0.5, (bbox[1] + bbox[3]) * 0.5] for bbox in bboxes], dtype=float) if bboxes else np.zeros((0, 2))
        img_w, img_h = page_size(record, bboxes)
        rooms = record_rooms(record)
        for symbol in symbols:
            label = str(symbol.get("symbol_type") or "generic_symbol")
            if label not in LABELS:
                continue
            bbox = bboxes_by_id.get(str(symbol.get("id")))
            if bbox is None:
                continue
            x1, y1, x2, y2 = bbox
            width = max(0.0, x2 - x1)
            height = max(0.0, y2 - y1)
            area = width * height
            cx = (x1 + x2) * 0.5
            cy = (y1 + y2) * 0.5
            same_scale = int(np.sum((0.5 <= area / np.maximum(areas, 1.0)) & (area / np.maximum(areas, 1.0) <= 2.0))) if len(areas) else 0
            if len(centers):
                distances = np.hypot(centers[:, 0] - cx, centers[:, 1] - cy)
                near_count = int(np.sum(distances <= max(width, height, 1.0) * 3.0) - 1)
            else:
                near_count = 0
            features = [
                cx / max(img_w, 1.0),
                cy / max(img_h, 1.0),
                width / max(img_w, 1.0),
                height / max(img_h, 1.0),
                area / max(img_w * img_h, 1.0),
                float(np.log((width + 1.0) / (height + 1.0))),
                float(np.log(area + 1.0)),
                float(np.log(mean_area + 1.0)),
                float(np.log(median_area + 1.0)),
                float(np.log(q25 + 1.0)),
                float(np.log(q75 + 1.0)),
                float(np.log(area / max(mean_area, 1.0) + 1.0)),
                float(np.log(area / max(median_area, 1.0) + 1.0)),
                float(len(areas)),
                float(same_scale),
                float(near_count),
                *containing_room_features(bbox, rooms),
            ]
            out.append({"record_index": record_index, "candidate_id": str(symbol.get("id")), "label": label, "features": features})
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(out, ensure_ascii=False) + "\n", encoding="utf-8")
    return out


def f1_for_label(gold: list[str], pred: list[str], label: str) -> float:
    tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
    fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
    fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    return 2 * precision * recall / max(precision + recall, 1e-12)


def labels_from_policy(prob: np.ndarray, classes: list[str], policy: dict[str, Any]) -> list[str]:
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    pred: list[str] = []
    rules = policy.get("rules") or {}
    for row in prob:
        order = np.argsort(row)[::-1]
        label = classes[int(order[0])]
        for target, rule in rules.items():
            target_idx = class_to_idx.get(target)
            if target_idx is None:
                continue
            best_other = max(float(row[i]) for i in range(len(classes)) if i != target_idx)
            if float(row[target_idx]) >= float(rule["threshold"]) and float(row[target_idx]) - best_other >= float(rule["margin"]):
                label = target
                break
        pred.append(label)
    return pred


def lookup_from_policy(items: list[dict[str, Any]], prob: np.ndarray, classes: list[str], policy: dict[str, Any]) -> dict[tuple[int, str], tuple[str, float, dict[str, float]]]:
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    lookup: dict[tuple[int, str], tuple[str, float, dict[str, float]]] = {}
    rules = policy.get("rules") or {}
    for item, row in zip(items, prob):
        order = np.argsort(row)[::-1]
        label = classes[int(order[0])]
        confidence = float(row[int(order[0])])
        applied_rule = None
        for target, rule in rules.items():
            target_idx = class_to_idx.get(target)
            if target_idx is None:
                continue
            best_other = max(float(row[i]) for i in range(len(classes)) if i != target_idx)
            if float(row[target_idx]) >= float(rule["threshold"]) and float(row[target_idx]) - best_other >= float(rule["margin"]):
                label = target
                confidence = float(row[target_idx])
                applied_rule = target
                break
        probs = {label_i: round(float(value), 6) for label_i, value in zip(classes, row)}
        if applied_rule:
            probs["_class_threshold_rule"] = applied_rule
        lookup[(int(item["record_index"]), str(item["candidate_id"]))] = (label, confidence, probs)
    return lookup


def candidate_rules_for_label(label: str) -> list[dict[str, float]]:
    if label == "generic_symbol":
        thresholds = np.linspace(0.06, 0.34, 8)
        margins = np.linspace(-0.22, 0.02, 7)
    elif label == "bathtub":
        thresholds = np.linspace(0.12, 0.60, 9)
        margins = np.linspace(-0.18, 0.14, 7)
    else:
        thresholds = np.linspace(0.24, 0.72, 9)
        margins = np.linspace(-0.08, 0.16, 7)
    return [{"threshold": round(float(t), 4), "margin": round(float(m), 4)} for t in thresholds for m in margins]


def select_policy(dev_prob: np.ndarray, classes: list[str], y_dev: list[str]) -> dict[str, Any]:
    base_pred = [classes[int(np.argmax(row))] for row in dev_prob]
    base_metrics = metrics(y_dev, base_pred)
    selected: dict[str, dict[str, float]] = {}
    steps = []
    current_pred = base_pred
    current_macro = float(base_metrics["macro_f1"])
    class_to_idx = {label: idx for idx, label in enumerate(classes)}
    for label in TARGET_LABELS:
        if label not in class_to_idx:
            continue
        candidates = []
        for rule in candidate_rules_for_label(label):
            trial_policy = {"rules": {**selected, label: rule}}
            pred = labels_from_policy(dev_prob, classes, trial_policy)
            trial_metrics = metrics(y_dev, pred)
            candidates.append(
                {
                    "label": label,
                    "threshold": rule["threshold"],
                    "margin": rule["margin"],
                    "dev_macro_f1": trial_metrics["macro_f1"],
                    "dev_target_f1": round(f1_for_label(y_dev, pred, label), 6),
                    "dev_pred_count": sum(1 for item in pred if item == label),
                }
            )
        best = max(candidates, key=lambda row: (row["dev_macro_f1"], row["dev_target_f1"], -abs(row["dev_pred_count"] - Counter(y_dev)[label])))
        accepted = float(best["dev_macro_f1"]) > current_macro + 1e-12
        if accepted:
            selected[label] = {"threshold": float(best["threshold"]), "margin": float(best["margin"])}
            current_pred = labels_from_policy(dev_prob, classes, {"rules": selected})
            current_macro = float(best["dev_macro_f1"])
        steps.append({**best, "accepted": accepted})
    final_metrics = metrics(y_dev, current_pred)
    return {
        "base_dev_symbol_metrics": base_metrics,
        "selected_dev_symbol_metrics": final_metrics,
        "rules": selected,
        "selection_steps": steps,
        "dev_macro_delta_pp": round((float(final_metrics["macro_f1"]) - float(base_metrics["macro_f1"])) * 100.0, 3),
    }


def main() -> int:
    train_rows = load_jsonl(TRAIN_ONLY)
    dev_rows = load_jsonl(DEV_ONLY)
    locked_rows = load_jsonl(LOCKED_SPLIT)
    if split_images(train_rows) & split_images(dev_rows) or split_images(train_rows) & split_images(locked_rows) or split_images(dev_rows) & split_images(locked_rows):
        raise SystemExit("split image overlap detected")

    train_items = stratified(fast_extract_items(train_rows, "train_items_fast_v1"))
    capped = []
    seen_by_label: Counter[str] = Counter()
    for item in train_items:
        label = str(item["label"])
        if seen_by_label[label] >= MAX_FAST_TRAIN_PER_LABEL:
            continue
        capped.append(item)
        seen_by_label[label] += 1
    train_items = capped
    dev_items = fast_extract_items(dev_rows, "dev_items_fast_v1")
    locked_items = fast_extract_items(locked_rows, "locked_items_fast_v1")
    x_train = np.array([item["features"] for item in train_items], dtype=np.float64)
    y_train = [item["label"] for item in train_items]
    x_dev = np.array([item["features"] for item in dev_items], dtype=np.float64)
    y_dev = [item["label"] for item in dev_items]
    x_locked = np.array([item["features"] for item in locked_items], dtype=np.float64)
    y_locked = [item["label"] for item in locked_items]

    model = ExtraTreesClassifier(
        n_estimators=96,
        max_depth=None,
        min_samples_leaf=1,
        class_weight="balanced_subsample",
        random_state=20260504,
        n_jobs=-1,
    )
    model.fit(x_train, y_train)
    classes = [str(label) for label in model.classes_]
    dev_prob = model.predict_proba(x_dev)
    locked_prob = model.predict_proba(x_locked)
    policy = select_policy(dev_prob, classes, y_dev)

    locked_pred = labels_from_policy(locked_prob, classes, policy)
    locked_metrics = metrics(y_locked, locked_pred)
    lookup = lookup_from_policy(locked_items, locked_prob, classes, policy)
    current_predictions = load_jsonl(CURRENT_PREDICTIONS)
    adjusted, application = apply_arbitration(current_predictions, locked_rows, lookup)
    write_jsonl(ADJUSTED_PREDICTIONS, adjusted)
    fusion = evaluate_fusion(adjusted, locked_rows)
    fusion["version"] = "scene_graph_fusion_symbol_v2_text_conservative_generic_class_threshold_v1"
    fusion["predictions_file"] = str(ADJUSTED_PREDICTIONS.relative_to(ROOT))
    write_json(FUSION_REPORT, fusion)

    current_fusion = load_json(CURRENT_FUSION)
    current_node = float(((current_fusion.get("node_evaluation") or {}).get("macro_f1")) or 0.0)
    new_node = float(((fusion.get("node_evaluation") or {}).get("macro_f1")) or 0.0)
    current_per = (current_fusion.get("node_evaluation") or {}).get("per_label") or {}
    new_per = (fusion.get("node_evaluation") or {}).get("per_label") or {}
    per_label_delta = {
        label: {
            "current_f1": (current_per.get(label) or {}).get("f1"),
            "new_f1": (new_per.get(label) or {}).get("f1"),
            "delta_pp": round((float((new_per.get(label) or {}).get("f1") or 0.0) - float((current_per.get(label) or {}).get("f1") or 0.0)) * 100.0, 3),
        }
        for label in LABELS
    }

    CHECKPOINT.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": LABELS, "selected_policy": policy}, CHECKPOINT)
    report = {
        "version": "symbol_class_thresholds_v1",
        "created": "2026-05-04",
        "protocol": "Train split trains the symbol model; dev split greedily selects class threshold/margin rules; locked split is evaluated once against the current strongest prediction stream.",
        "train_split": str(TRAIN_ONLY.relative_to(ROOT)),
        "dev_split": str(DEV_ONLY.relative_to(ROOT)),
        "locked_split": str(LOCKED_SPLIT.relative_to(ROOT)),
        "current_predictions": str(CURRENT_PREDICTIONS.relative_to(ROOT)),
        "adjusted_predictions": str(ADJUSTED_PREDICTIONS.relative_to(ROOT)),
        "fusion_report": str(FUSION_REPORT.relative_to(ROOT)),
        "train_label_counts": dict(Counter(y_train)),
        "train_sampling": {"max_fast_train_per_label": MAX_FAST_TRAIN_PER_LABEL, "sampled_items": len(train_items)},
        "dev_label_counts": dict(Counter(y_dev)),
        "locked_label_counts": dict(Counter(y_locked)),
        "selected_policy": policy,
        "locked_symbol_metrics": locked_metrics,
        "application": application,
        "locked_e2e_delta_vs_current": {
            "current_node_macro_f1": round(current_node, 6),
            "new_node_macro_f1": round(new_node, 6),
            "node_macro_f1_delta_pp": round((new_node - current_node) * 100.0, 3),
            "current_relation_f1": ((current_fusion.get("relation_evaluation") or {}).get("f1")),
            "new_relation_f1_pre_scorer": ((fusion.get("relation_evaluation") or {}).get("f1")),
            "invalid_graph_rate": fusion.get("invalid_graph_rate"),
        },
        "per_label_e2e_delta": per_label_delta,
        "adopt_as_current_best_candidate": new_node > current_node,
        "status": "candidate_improves_node" if new_node > current_node else "no_adoption",
    }
    write_json(REPORT, report)
    write_json(FORMAL_REPORT, {**report, "version": "symbol_long_tail_threshold_formal_v1_eval", "formal_outputs": {"fusion_report": str(FORMAL_FUSION_REPORT.relative_to(ROOT)), "scorer_report": str(FORMAL_SCORER_REPORT.relative_to(ROOT)), "metric_summary": str(METRIC_SUMMARY.relative_to(ROOT))}})
    write_json(FORMAL_FUSION_REPORT, fusion)
    print(f"wrote {REPORT}")
    print(f"wrote {FUSION_REPORT}")
    print(json.dumps(report["locked_e2e_delta_vs_current"], ensure_ascii=False, indent=2))
    print(f"status={report['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
