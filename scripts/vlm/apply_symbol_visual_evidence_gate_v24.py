#!/usr/bin/env python3
"""Apply symbol visual-evidence gate to v24 page-level YOLO proposals."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image

from apply_symbol_visual_evidence_v8_to_v18 import IntegralImageStats, feature_vector, parse_thresholds
from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, center_covered, load_jsonl, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def page_gold_from_tiles(tile_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for row in tile_rows:
        for target in (row.get("targets") or {}).get("boxes") or []:
            tid = str(target.get("target_id") or "")
            if not tid or tid in out:
                continue
            box = target.get("page_bbox") or target.get("bbox")
            if isinstance(box, list) and len(box) == 4:
                out[tid] = {"bbox": [float(v) for v in box], "label": str(target.get("label") or "generic_symbol")}
    return out


def load_golds(tile_path: Path, prediction_page_ids: set[str]) -> dict[str, dict[str, dict[str, Any]]]:
    pages: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(tile_path):
        row_id = str(row.get("row_id"))
        if row_id in prediction_page_ids:
            pages[row_id].append(row)
    return {row_id: page_gold_from_tiles(rows) for row_id, rows in pages.items()}


def score_page_metrics(rows: list[dict[str, Any]], golds: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    totals = Counter()
    by_area = Counter()
    by_area_center = Counter()
    by_area_iou = Counter()
    typed_correct = 0
    for row in rows:
        row_id = str(row.get("row_id"))
        preds = list(row.get("predicted_symbols") or [])
        gold_map = golds.get(row_id, {})
        used_iou: set[int] = set()
        used_center: set[int] = set()
        for gold in gold_map.values():
            gold_box = [float(v) for v in gold["bbox"]]
            bucket = area_bucket(gold_box)
            by_area[bucket] += 1
            best_iou = 0.0
            best_iou_index = None
            center_index = None
            for index, pred in enumerate(preds):
                box = [float(v) for v in pred.get("bbox") or []]
                if len(box) != 4:
                    continue
                overlap = bbox_iou(box, gold_box)
                if overlap > best_iou:
                    best_iou = overlap
                    best_iou_index = index
                if center_index is None and index not in used_center and center_covered(box, gold_box):
                    center_index = index
            if best_iou_index is not None and best_iou >= 0.30 and best_iou_index not in used_iou:
                used_iou.add(best_iou_index)
                totals["matched_iou"] += 1
                by_area_iou[bucket] += 1
                if str(preds[best_iou_index].get("label")) == str(gold["label"]):
                    typed_correct += 1
            if center_index is not None:
                used_center.add(center_index)
                totals["matched_center"] += 1
                by_area_center[bucket] += 1
        totals["gold"] += len(gold_map)
        totals["predicted"] += len(preds)
    precision = totals["matched_iou"] / max(totals["predicted"], 1)
    recall = totals["matched_iou"] / max(totals["gold"], 1)
    return {
        "symbol_bbox_iou_0_30": {
            "matched": int(totals["matched_iou"]),
            "predicted": int(totals["predicted"]),
            "gold": int(totals["gold"]),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall), 6),
        },
        "symbol_bbox_center_recall": round(totals["matched_center"] / max(totals["gold"], 1), 6),
        "candidate_inflation": round(totals["predicted"] / max(totals["gold"], 1), 6),
        "typed_accuracy_on_iou_matches": round(typed_correct / max(totals["matched_iou"], 1), 6),
        "area_center_recall": {bucket: round(by_area_center[bucket] / max(by_area[bucket], 1), 6) for bucket in sorted(by_area)},
        "area_iou_recall": {bucket: round(by_area_iou[bucket] / max(by_area[bucket], 1), 6) for bucket in sorted(by_area)},
    }


def load_page_images(tile_path: Path, page_ids: set[str]) -> dict[str, str]:
    images = {}
    for row in load_jsonl(tile_path):
        row_id = str(row.get("row_id"))
        if row_id in page_ids and row_id not in images:
            images[row_id] = str(row.get("image"))
    return images


def score_rows(rows: list[dict[str, Any]], page_images: dict[str, str], model: Any, feature_names: list[str]) -> list[dict[str, Any]]:
    image_cache: dict[str, IntegralImageStats] = {}
    output = []
    for row in rows:
        row_id = str(row.get("row_id"))
        image_path = page_images.get(row_id)
        if not image_path:
            continue
        if image_path not in image_cache:
            image_cache[image_path] = IntegralImageStats(Image.open(resolve(image_path)).convert("L"))
        stats = image_cache[image_path]
        feature_rows = []
        scored = []
        for pred in row.get("predicted_symbols") or []:
            item = json.loads(json.dumps(pred))
            box = item.get("bbox")
            if not isinstance(box, list) or len(box) != 4:
                continue
            features = stats.crop_features([float(v) for v in box])
            feature_rows.append(feature_vector(features, feature_names))
            item.setdefault("payload", {})
            item["payload"] = dict(item["payload"])
            item["payload"]["symbol_visual_evidence_v8_features"] = {key: round(float(features.get(key) or 0.0), 6) for key in feature_names}
            scored.append(item)
        if feature_rows:
            reject_probs = model.predict_proba(np.asarray(feature_rows, dtype=float))[:, 1]
            for item, prob in zip(scored, reject_probs, strict=True):
                item["payload"]["symbol_visual_evidence_v8_reject_probability"] = round(float(prob), 6)
        out = dict(row)
        out["predicted_symbols"] = scored
        output.append(out)
    return output


def materialize(rows: list[dict[str, Any]], threshold: float) -> tuple[list[dict[str, Any]], Counter]:
    counts = Counter()
    output = []
    for row in rows:
        kept = []
        for pred in row.get("predicted_symbols") or []:
            reject_prob = float((pred.get("payload") or {}).get("symbol_visual_evidence_v8_reject_probability") or 0.0)
            counts["before"] += 1
            if reject_prob < threshold:
                item = dict(pred)
                item["visual_evidence_gate_threshold"] = threshold
                kept.append(item)
                counts["kept"] += 1
            else:
                counts["rejected"] += 1
        out = dict(row)
        out["predicted_symbols"] = kept
        out["prediction_count_before_visual_evidence_gate"] = counts["before"]
        output.append(out)
    return output, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="reports/vlm/symbol_yolov8n_pretrained_v22_dedup_hi640_probe_page_predictions.jsonl")
    parser.add_argument("--locked-tiles", default="datasets/symbol_tile_detector_tiny_sahi_v21/locked.jsonl")
    parser.add_argument("--model", default="checkpoints/symbol_visual_evidence_v8/model.joblib")
    parser.add_argument("--output", default="reports/vlm/symbol_visual_gate_yolo_v24_predictions.jsonl")
    parser.add_argument("--audit", default="reports/vlm/symbol_visual_gate_yolo_v24_eval.json")
    parser.add_argument("--threshold-sweep", default="0.5,0.7,0.8,0.9,0.95,0.98,0.99,0.995,0.999")
    parser.add_argument("--max-center-recall-drop", type=float, default=0.005)
    args = parser.parse_args()

    rows = load_jsonl(resolve(args.input))
    page_ids = {str(row.get("row_id")) for row in rows}
    golds = load_golds(resolve(args.locked_tiles), page_ids)
    page_images = load_page_images(resolve(args.locked_tiles), page_ids)
    bundle = joblib.load(resolve(args.model))
    model = bundle["model"]
    feature_names = list(bundle.get("features") or [])
    scored_rows = score_rows(rows, page_images, model, feature_names)
    baseline = score_page_metrics(scored_rows, golds)
    sweep = []
    selected = None
    for threshold in parse_thresholds(args.threshold_sweep):
        trial_rows, counts = materialize(scored_rows, threshold)
        metrics = score_page_metrics(trial_rows, golds)
        center_drop = round(baseline["symbol_bbox_center_recall"] - metrics["symbol_bbox_center_recall"], 6)
        item = {
            "threshold": threshold,
            "counts": dict(counts),
            "reduction": round(counts["rejected"] / max(counts["before"], 1), 6),
            "metrics": metrics,
            "center_recall_drop": center_drop,
            "passes_center_drop_gate": center_drop <= args.max_center_recall_drop,
        }
        sweep.append(item)
        if item["passes_center_drop_gate"] and (selected is None or item["reduction"] > selected["reduction"]):
            selected = item
    if selected is None:
        selected = min(sweep, key=lambda item: (item["center_recall_drop"], -item["reduction"]))
    output_rows, counts = materialize(scored_rows, float(selected["threshold"]))
    write_jsonl(resolve(args.output), output_rows)
    report = {
        "version": "symbol_visual_gate_yolo_v24_eval",
        "task": "P0-03-symbol-proposal-and-type-adaptation.visual_evidence_gate",
        "claim_boundary": "Applies adopted v8 raster crop evidence gate to real v24 YOLO page proposals. Gold is used only for locked evaluation and threshold selection.",
        "input": args.input,
        "output": args.output,
        "model": args.model,
        "feature_names": feature_names,
        "baseline": baseline,
        "selected": selected,
        "threshold_sweep": sweep,
        "source_integrity": {
            "model_input": "raster_page_pixels_and_detector_boxes_only",
            "offline_gold_used_at_inference": False,
            "svg_or_vector_geometry_used_at_inference": False,
        },
        "success_gate": {
            "stage_1_center_recall_min": 0.94,
            "stage_1_iou_0_30_recall_min": 0.78,
            "stage_1_precision_must_improve_over": 0.096685,
            "must_not_drop_center_recall_below": 0.911595,
            "selected_center_recall": selected["metrics"]["symbol_bbox_center_recall"],
            "selected_iou_0_30_recall": selected["metrics"]["symbol_bbox_iou_0_30"]["recall"],
            "selected_precision": selected["metrics"]["symbol_bbox_iou_0_30"]["precision"],
        },
    }
    gate = report["success_gate"]
    gate["passed"] = (
        gate["selected_center_recall"] >= gate["stage_1_center_recall_min"]
        and gate["selected_iou_0_30_recall"] >= gate["stage_1_iou_0_30_recall_min"]
        and gate["selected_precision"] > gate["stage_1_precision_must_improve_over"]
        and gate["selected_center_recall"] >= gate["must_not_drop_center_recall_below"]
    )
    write_json(resolve(args.audit), report)
    print(json.dumps({"selected": selected, "success_gate": gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
