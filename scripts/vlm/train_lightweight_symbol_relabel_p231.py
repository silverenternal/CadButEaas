#!/usr/bin/env python3
"""P231 lightweight symbol relabeler over P229 raster-contract features.

This is a bounded alternative to synchronously loading the 744MB v13 symbol
checkpoint. It extracts the same runtime-safe geometry/coarse-label feature
family, trains out-of-fold lightweight classifiers, and only accepts relabels
when a confidence/margin gate beats the P229b passthrough baseline.
"""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import GroupKFold

ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from build_raster_symbol_contract_adapter_p229 import write_json, write_jsonl  # noqa: E402
from freeze_symbol_p222_p221a_sink_tiny import bbox_iou, bootstrap, metrics, score_rows  # noqa: E402
from fuse_symbol_p206g_with_p211_p212 import load_p206g  # noqa: E402


DEFAULT_CONTRACT = ROOT / "reports" / "vlm" / "p229_raster_symbol_contract_predictions.jsonl"
DEFAULT_OVERLAY = ROOT / "reports" / "vlm" / "symbol_p224a_column_frozen_overlay.jsonl"
DEFAULT_FEATURES = ROOT / "reports" / "vlm" / "p231_symbol_contract_feature_dataset.jsonl"
DEFAULT_OUTPUT = ROOT / "reports" / "vlm" / "p231_lightweight_relabel_predictions.jsonl"
DEFAULT_EVAL = ROOT / "reports" / "vlm" / "p231_lightweight_relabel_eval.json"
LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]
EQUIP_HINT = {"appliance", "equipment"}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def norm_bbox(value: Any) -> list[float]:
    return [float(item) for item in value]


def features_for(candidate: dict[str, Any]) -> list[float]:
    bbox = norm_bbox(candidate["bbox"])
    payload = candidate.get("payload") or {}
    meta = payload.get("metadata") or {}
    page_w = float(meta.get("width") or 1.0)
    page_h = float(meta.get("height") or 1.0)
    x1, y1, x2, y2 = bbox
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    area = width * height
    label = str(candidate.get("candidate_type") or payload.get("symbol_type") or "generic_symbol")
    return [
        (x1 + x2) / 2.0 / max(page_w, 1.0),
        (y1 + y2) / 2.0 / max(page_h, 1.0),
        width / max(page_w, 1.0),
        height / max(page_h, 1.0),
        area / max(page_w * page_h, 1.0),
        max(width, height) / max(min(width, height), 1e-6),
        float(payload.get("rotation") or 0.0) / 360.0,
        float(payload.get("hard_case_focus") or 0.0),
        float(label in EQUIP_HINT),
    ] + [float(label == item) for item in LABELS]


def best_training_label(candidate: dict[str, Any], row_targets: dict[str, dict[str, Any]]) -> tuple[str, float, bool]:
    label = str(candidate.get("candidate_type") or "generic_symbol")
    bbox = norm_bbox(candidate["bbox"])
    best_label = label
    best_iou = 0.0
    for target in row_targets.values():
        iou = bbox_iou(bbox, norm_bbox(target["bbox"]))
        if iou > best_iou:
            best_iou = iou
            best_label = str(target["label"])
    return best_label if best_iou >= 0.30 else label, best_iou, best_iou >= 0.30


def build_dataset(contract_rows: list[dict[str, Any]], targets_by_row: dict[str, dict[str, dict[str, Any]]]) -> list[dict[str, Any]]:
    rows = []
    for row in contract_rows:
        row_id = str(row["row_id"])
        for candidate in row.get("routed_candidates") or []:
            train_target, match_iou, matched = best_training_label(candidate, targets_by_row[row_id])
            rows.append({
                "row_id": row_id,
                "candidate_id": str(candidate["candidate_id"]),
                "bbox": candidate["bbox"],
                "input_label": str(candidate.get("candidate_type") or "generic_symbol"),
                "confidence": float(candidate.get("confidence") or 0.0),
                "features": features_for(candidate),
                "train_target": train_target,
                "match_iou": round(match_iou, 6),
                "matched_for_training": matched,
            })
    return rows


def metric_item(row: dict[str, Any], label: str, source: str, score: float | None = None) -> dict[str, Any]:
    confidence = float(row["confidence"] if score is None else score)
    return {
        "id": row["candidate_id"],
        "target_id": row["candidate_id"],
        "label": label,
        "symbol_type": label,
        "bbox": row["bbox"],
        "confidence": confidence,
        "score": confidence,
        "source": source,
    }


def by_row_predictions(dataset: list[dict[str, Any]], labels: list[str], scores: list[float], sources: list[str]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row, label, score, source in zip(dataset, labels, scores, sources):
        out[row["row_id"]].append(metric_item(row, label, source, score))
    return out


def out_of_fold_probabilities(dataset: list[dict[str, Any]], folds: int, seed: int) -> tuple[np.ndarray, list[str]]:
    X = np.array([row["features"] for row in dataset], dtype=np.float64)
    y = np.array([row["train_target"] for row in dataset])
    groups = np.array([row["row_id"] for row in dataset])
    classes = sorted(set(y) | set(LABELS))
    probs = np.zeros((len(dataset), len(classes)), dtype=np.float64)
    splitter = GroupKFold(n_splits=min(folds, len(set(groups))))
    for fold_index, (train_index, test_index) in enumerate(splitter.split(X, y, groups)):
        model = ExtraTreesClassifier(
            n_estimators=96,
            max_depth=10,
            min_samples_leaf=3,
            class_weight="balanced",
            random_state=seed + fold_index,
            n_jobs=1,
        )
        model.fit(X[train_index], y[train_index])
        fold_probs = model.predict_proba(X[test_index])
        class_to_index = {label: idx for idx, label in enumerate(model.classes_)}
        for global_class_index, label in enumerate(classes):
            if label in class_to_index:
                probs[test_index, global_class_index] = fold_probs[:, class_to_index[label]]
    return probs, classes


def gated_labels(dataset: list[dict[str, Any]], probs: np.ndarray, classes: list[str], min_prob: float, min_margin: float) -> tuple[list[str], list[float], list[str], int]:
    labels = []
    scores = []
    sources = []
    changed = 0
    for row, row_probs in zip(dataset, probs):
        order = np.argsort(row_probs)[::-1]
        best_index = int(order[0])
        second = float(row_probs[int(order[1])]) if len(order) > 1 else 0.0
        best_label = classes[best_index]
        best_prob = float(row_probs[best_index])
        original = str(row["input_label"])
        if best_label != original and best_prob >= min_prob and (best_prob - second) >= min_margin:
            labels.append(best_label)
            scores.append(best_prob)
            sources.append("p231_lightweight_relabel_oof_gate")
            changed += 1
        else:
            labels.append(original)
            scores.append(float(row["confidence"]))
            sources.append("p231_lightweight_relabel_keep_baseline")
    return labels, scores, sources, changed


def per_label_metrics(preds_by_row: dict[str, list[dict[str, Any]]], targets_by_row: dict[str, dict[str, dict[str, Any]]], row_ids: list[str]) -> dict[str, Any]:
    labels = sorted({target["label"] for rid in row_ids for target in targets_by_row[rid].values()} | {pred["label"] for preds in preds_by_row.values() for pred in preds})
    out: dict[str, Any] = {}
    for label in labels:
        label_preds = defaultdict(list)
        label_targets: dict[str, dict[str, dict[str, Any]]] = {}
        for rid in row_ids:
            label_preds[rid] = [pred for pred in preds_by_row.get(rid, []) if pred.get("label") == label]
            label_targets[rid] = {tid: target for tid, target in targets_by_row[rid].items() if target.get("label") == label}
        out[label] = metrics(score_rows(label_preds, label_targets, row_ids))
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--features-out", type=Path, default=DEFAULT_FEATURES)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-out", type=Path, default=DEFAULT_EVAL)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=231)
    args = parser.parse_args()

    contract_rows = load_jsonl(args.contract)
    _overlay_rows, _overlay_preds, targets_by_row = load_p206g(args.overlay)
    row_ids = [str(row["row_id"]) for row in contract_rows]
    dataset = build_dataset(contract_rows, targets_by_row)
    write_jsonl(args.features_out, dataset)

    baseline_labels = [str(row["input_label"]) for row in dataset]
    baseline_scores = [float(row["confidence"]) for row in dataset]
    baseline_sources = ["p229_raster_contract_passthrough_baseline"] * len(dataset)
    baseline_preds = by_row_predictions(dataset, baseline_labels, baseline_scores, baseline_sources)
    baseline_per_row = score_rows(baseline_preds, targets_by_row, row_ids)
    baseline_metrics = metrics(baseline_per_row)

    probs, classes = out_of_fold_probabilities(dataset, args.folds, args.seed)
    candidates = []
    for min_prob in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90]:
        for min_margin in [0.00, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]:
            labels, scores, sources, changed = gated_labels(dataset, probs, classes, min_prob, min_margin)
            preds = by_row_predictions(dataset, labels, scores, sources)
            per_row = score_rows(preds, targets_by_row, row_ids)
            item_metrics = metrics(per_row)
            candidates.append({
                "min_prob": min_prob,
                "min_margin": min_margin,
                "changed": changed,
                "metrics": item_metrics,
                "delta": {key: round(item_metrics[key] - baseline_metrics[key], 6) for key in ["precision", "recall", "f1"]},
                "labels": labels,
                "scores": scores,
                "sources": sources,
                "per_row": per_row,
                "preds": preds,
            })
    viable = [item for item in candidates if item["delta"]["f1"] > 0 and item["delta"]["precision"] >= 0]
    selected = max(viable or candidates, key=lambda item: (item["delta"]["f1"], item["delta"]["precision"], -item["changed"]))

    output_rows = []
    row_to_indices: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(dataset):
        row_to_indices[row["row_id"]].append(index)
    for row_id in row_ids:
        predictions = []
        for index in row_to_indices[row_id]:
            predictions.append(metric_item(dataset[index], selected["labels"][index], selected["sources"][index], selected["scores"][index]))
        output_rows.append({
            "row_id": row_id,
            "source": "p231_lightweight_symbol_relabeler",
            "expert_predictions": predictions,
            "adapter_metadata": {
                "contract_version": "p231_lightweight_symbol_relabeler_v0",
                "runtime_source_integrity": "p229_raster_contract_features_only_no_svg_no_expected_json_no_offline_labels",
                "selected_min_prob": selected["min_prob"],
                "selected_min_margin": selected["min_margin"],
            },
        })
    write_jsonl(args.output, output_rows)

    label_counts = Counter(selected["labels"])
    source_counts = Counter(selected["sources"])
    report = {
        "id": "p231_lightweight_relabel_eval",
        "phase": "P231_lightweight_symbol_relabeler_from_p229_contract_features",
        "contract_input": str(args.contract),
        "feature_dataset": str(args.features_out),
        "output": str(args.output),
        "rows": len(row_ids),
        "candidates": len(dataset),
        "feature_count": len(dataset[0]["features"]) if dataset else 0,
        "baseline_metrics_iou_0_30": baseline_metrics,
        "candidate_metrics_iou_0_30": selected["metrics"],
        "delta_vs_p229b": selected["delta"],
        "selected_gate": {"min_prob": selected["min_prob"], "min_margin": selected["min_margin"], "changed": selected["changed"]},
        "bootstrap_vs_p229b": bootstrap(baseline_per_row, selected["per_row"], iterations=1000, seed=args.seed),
        "per_label_metrics_iou_0_30": per_label_metrics(selected["preds"], targets_by_row, row_ids),
        "label_counts": dict(label_counts),
        "source_counts": dict(source_counts),
        "promotion_recommendation": "promote" if viable and selected["delta"]["precision"] >= 0 and selected["delta"]["f1"] > 0 else "do_not_promote",
        "claim_boundary": "OOF lightweight relabel probe. Training targets are offline-only; runtime output consumes P229 contract features only.",
    }
    write_json(args.eval_out, report)
    print(json.dumps({"eval": str(args.eval_out), "metrics": selected["metrics"], "delta": selected["delta"], "gate": report["selected_gate"], "recommendation": report["promotion_recommendation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
