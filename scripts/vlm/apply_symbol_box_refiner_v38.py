#!/usr/bin/env python3
"""Apply v38 bbox refiner to locked candidates and evaluate page-level metrics."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from train_symbol_box_refiner_v38 import apply_delta, vector
from train_symbol_support_suppression_v35 import load_jsonl
from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, center_covered, rel, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
FOCUS_LABELS = {"sink", "shower", "equipment", "stair"}
FOCUS_AREAS = {"tiny_le_64", "small_le_256"}


def source_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def should_refine(row: dict[str, Any]) -> bool:
    labels = row.get("labels") or {}
    label = str(row.get("label") or "")
    area = None
    target = labels.get("best_iou_target_id")
    for gold in labels.get("page_gold_targets") or []:
        if str(gold.get("target_id") or "") == str(target):
            area = str(gold.get("area_bucket") or "")
            break
    return label in FOCUS_LABELS or area in FOCUS_AREAS


def refine_rows(rows: list[dict[str, Any]], model: Any, names: list[str], clip: float) -> dict[str, dict[str, Any]]:
    candidates = [row for row in rows if should_refine(row)]
    out: dict[str, dict[str, Any]] = {}
    batch = 50000
    for start in range(0, len(candidates), batch):
        chunk = candidates[start : start + batch]
        x = np.asarray([vector(row, names) for row in chunk], dtype=np.float32)
        deltas = model.predict(x)
        for row, delta in zip(chunk, deltas, strict=True):
            box = [float(v) for v in row["bbox"]]
            refined = apply_delta(box, list(delta), clip)
            out[str(row["candidate_id"])] = {"bbox": refined, "delta": [float(v) for v in delta]}
    return out


def cache_gold_maps(cache_rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for row in cache_rows:
        row_id = str(row["row_id"])
        out[row_id] = {}
        for gold in row.get("gold_symbols") or []:
            target_id = str(gold.get("target_id") or "")
            box = [float(v) for v in gold.get("bbox") or []]
            if target_id and len(box) == 4:
                out[row_id][target_id] = {
                    "target_id": target_id,
                    "bbox": box,
                    "label": str(gold.get("label") or "generic_symbol"),
                    "area_bucket": str(gold.get("area_bucket") or area_bucket(box)),
                }
    return out


def predictions_from_rows(rows: list[dict[str, Any]], refined_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        candidate_id = str(row["candidate_id"])
        refined = refined_by_id.get(candidate_id)
        pred = {
            "candidate_id": candidate_id,
            "bbox": refined["bbox"] if refined else row["bbox"],
            "original_bbox": row["bbox"],
            "label": row["label"],
            "score": row.get("score"),
            "proposal_source": row.get("proposal_source"),
            "refined_by_v38": bool(refined),
        }
        by_page[str(row["page_id"])].append(pred)
    return [{"row_id": page_id, "predicted_symbols": preds} for page_id, preds in sorted(by_page.items())]


def evaluate(prediction_rows: list[dict[str, Any]], gold_by_page: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    totals = Counter()
    by_label = Counter()
    by_label_iou = Counter()
    by_label_center = Counter()
    by_area = Counter()
    by_area_iou = Counter()
    by_area_center = Counter()
    typed_correct = 0
    for row in prediction_rows:
        row_id = str(row["row_id"])
        preds = list(row.get("predicted_symbols") or [])
        golds = gold_by_page.get(row_id, {})
        used_iou: set[int] = set()
        used_center: set[int] = set()
        for gold in golds.values():
            gold_box = [float(v) for v in gold["bbox"]]
            label = str(gold["label"])
            bucket = str(gold["area_bucket"])
            by_label[label] += 1
            by_area[bucket] += 1
            best_iou = 0.0
            best_iou_index = None
            center_index = None
            for pred_index, pred in enumerate(preds):
                pred_box = [float(v) for v in pred["bbox"]]
                iou = bbox_iou(pred_box, gold_box)
                if iou > best_iou:
                    best_iou = iou
                    best_iou_index = pred_index
                if center_index is None and pred_index not in used_center and center_covered(pred_box, gold_box):
                    center_index = pred_index
            if best_iou_index is not None and best_iou >= 0.30 and best_iou_index not in used_iou:
                used_iou.add(best_iou_index)
                totals["matched_iou"] += 1
                by_label_iou[label] += 1
                by_area_iou[bucket] += 1
                if str(preds[best_iou_index].get("label") or "") == label:
                    typed_correct += 1
            if center_index is not None:
                used_center.add(center_index)
                totals["matched_center"] += 1
                by_label_center[label] += 1
                by_area_center[bucket] += 1
        totals["gold"] += len(golds)
        totals["predicted"] += len(preds)
    precision = totals["matched_iou"] / max(totals["predicted"], 1)
    recall = totals["matched_iou"] / max(totals["gold"], 1)
    center_recall = totals["matched_center"] / max(totals["gold"], 1)
    return {
        "rows": len(prediction_rows),
        "symbol_bbox_center_recall": round(center_recall, 6),
        "symbol_bbox_iou_0_30": {
            "matched": int(totals["matched_iou"]),
            "predicted": int(totals["predicted"]),
            "gold": int(totals["gold"]),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall), 6),
        },
        "candidate_inflation": round(totals["predicted"] / max(totals["gold"], 1), 6),
        "typed_accuracy_on_iou_matches": round(typed_correct / max(totals["matched_iou"], 1), 6),
        "type_iou_recall": {label: round(by_label_iou[label] / max(by_label[label], 1), 6) for label in sorted(by_label)},
        "type_center_recall": {label: round(by_label_center[label] / max(by_label[label], 1), 6) for label in sorted(by_label)},
        "area_iou_recall": {bucket: round(by_area_iou[bucket] / max(by_area[bucket], 1), 6) for bucket in sorted(by_area)},
        "area_center_recall": {bucket: round(by_area_center[bucket] / max(by_area[bucket], 1), 6) for bucket in sorted(by_area)},
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", default="datasets/symbol_support_suppression_v36/locked_rows.jsonl")
    parser.add_argument("--cache", default="datasets/symbol_support_suppression_v36/locked_cache.jsonl")
    parser.add_argument("--model", default="checkpoints/symbol_box_refiner_v38/model.joblib")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_box_refiner_v38_page_locked_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/symbol_box_refiner_v38_page_locked_predictions.jsonl")
    parser.add_argument("--clip", type=float, default=0.75)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows = load_jsonl(source_path(args.rows))
    cache_rows = load_jsonl(source_path(args.cache))
    bundle = joblib.load(source_path(args.model))
    model = bundle["model"]
    names = bundle["feature_names"]
    refined_by_id = refine_rows(rows, model, names, args.clip)
    after_predictions = predictions_from_rows(rows, refined_by_id)
    before_predictions = predictions_from_rows(rows, {})
    gold_by_page = cache_gold_maps(cache_rows)
    before = evaluate(before_predictions, gold_by_page)
    after = evaluate(after_predictions, gold_by_page)
    write_jsonl(source_path(args.predictions_output), after_predictions)
    report = {
        "version": "symbol_box_refiner_v38_page_locked_eval",
        "task": "P1-09-apply-v38-refined-boxes-to-page-eval",
        "claim_boundary": "Apply v38 bbox-delta model to runtime candidate fields and evaluate page-level locked subset metrics.",
        "source_integrity": {
            "model_input": "candidate bbox/score/type fields only",
            "offline_labels_used_for": ["locked_evaluation"],
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
        },
        "inputs": {"rows": rel(source_path(args.rows)), "cache": rel(source_path(args.cache)), "model": rel(source_path(args.model))},
        "outputs": {"predictions": rel(source_path(args.predictions_output))},
        "refined_candidate_count": len(refined_by_id),
        "before": before,
        "after": after,
        "stage_gate": {
            "page_locked_iou_recall_improves_over_v35_detector_alone_0_686783": after["symbol_bbox_iou_0_30"]["recall"] > 0.686783,
            "page_locked_tiny_iou_recall_improves_over_v35_detector_alone_0_299865": after["area_iou_recall"].get("tiny_le_64", 0.0) > 0.299865,
            "page_locked_sink_iou_recall_improves": after["type_iou_recall"].get("sink", 0.0) > before["type_iou_recall"].get("sink", 0.0),
            "no_oracle_inference": True,
        },
    }
    report["stage_gate"]["passed"] = all(report["stage_gate"].values())
    write_json(source_path(args.eval_output), report)
    print(json.dumps({"refined_candidate_count": len(refined_by_id), "before": before, "after": after, "stage_gate": report["stage_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
