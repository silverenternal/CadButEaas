#!/usr/bin/env python3
"""Mine page-level YOLO symbol detector errors for targeted next training."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from eval_symbol_yolo_tile_detector_v22 import filter_rows_with_exported_images, image_path_for_yolo_tile, sample_tiles_area_aware
from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, center_covered, load_jsonl, rel, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
YOLO_DIR = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_v22"
PREDICTIONS = ROOT / "reports/vlm/symbol_yolov8n_pretrained_v22_dedup_hi640_probe_page_predictions.jsonl"
REPORT = ROOT / "reports/vlm/symbol_yolov8n_pretrained_v22_dedup_hi640_probe_error_audit.json"
HARD_CASES = ROOT / "reports/vlm/symbol_yolov8n_pretrained_v22_dedup_hi640_probe_hard_cases.jsonl"


def load_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    return {str(row["row_id"]): list(row.get("predicted_symbols") or []) for row in load_jsonl(path)}


def build_page_golds(rows: list[dict[str, Any]]) -> dict[str, dict[str, dict[str, Any]]]:
    page_golds: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        row_id = str(row.get("row_id"))
        for gold in ((row.get("targets") or {}).get("boxes") or []):
            target_id = str(gold.get("target_id") or f"{row_id}_{len(page_golds[row_id])}")
            page_golds[row_id][target_id] = {
                "target_id": target_id,
                "bbox": [float(v) for v in gold.get("page_bbox") or gold.get("bbox")],
                "label": str(gold.get("label") or "generic_symbol"),
            }
    return page_golds


def tile_scale(tile_id: str | None) -> str:
    value = str(tile_id or "")
    if "_s384_" in value:
        return "s384"
    if "_s640_" in value:
        return "s640"
    return "unknown"


def best_prediction(gold_box: list[float], preds: list[dict[str, Any]]) -> tuple[int | None, float, int | None]:
    best_iou = 0.0
    best_iou_index: int | None = None
    center_index: int | None = None
    for index, pred in enumerate(preds):
        pred_box = [float(v) for v in pred["bbox"]]
        iou = bbox_iou(pred_box, gold_box)
        if iou > best_iou:
            best_iou = iou
            best_iou_index = index
        if center_index is None and center_covered(pred_box, gold_box):
            center_index = index
    return best_iou_index, best_iou, center_index


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--yolo-dir", default=str(YOLO_DIR))
    parser.add_argument("--split", default="locked")
    parser.add_argument("--limit-tiles", type=int, default=2000)
    parser.add_argument("--positive-ratio", type=float, default=0.85)
    parser.add_argument("--small-positive-ratio", type=float, default=0.65)
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument("--predictions", default=str(PREDICTIONS))
    parser.add_argument("--report-output", default=str(REPORT))
    parser.add_argument("--hard-cases-output", default=str(HARD_CASES))
    parser.add_argument("--low-iou-threshold", type=float, default=0.30)
    parser.add_argument("--top-hard-cases", type=int, default=2000)
    args = parser.parse_args()

    exported_rows = filter_rows_with_exported_images(load_jsonl(Path(args.data) / f"{args.split}.jsonl"), args.split, Path(args.yolo_dir))
    rows = sample_tiles_area_aware(
        exported_rows,
        args.limit_tiles,
        args.seed + (2 if args.split == "locked" else 1),
        args.positive_ratio,
        args.small_positive_ratio,
    )
    page_golds = build_page_golds(rows)
    page_preds = load_predictions(Path(args.predictions))

    totals = Counter()
    by_label = defaultdict(Counter)
    by_area = defaultdict(Counter)
    by_label_area = defaultdict(Counter)
    by_pred_scale = Counter()
    hard_cases: list[dict[str, Any]] = []
    matched_pred_ids: set[tuple[str, int]] = set()

    for row_id, gold_map in page_golds.items():
        preds = page_preds.get(row_id, [])
        used_iou: set[int] = set()
        used_center: set[int] = set()
        totals["rows"] += 1
        totals["gold"] += len(gold_map)
        totals["predicted"] += len(preds)
        for pred in preds:
            by_pred_scale[tile_scale(pred.get("tile_id"))] += 1
        for gold in gold_map.values():
            gold_box = [float(v) for v in gold["bbox"]]
            label = str(gold["label"])
            bucket = area_bucket(gold_box)
            best_index, best_iou, center_index = best_prediction(gold_box, preds)
            best_pred = preds[best_index] if best_index is not None else None
            matched = best_index is not None and best_iou >= args.low_iou_threshold and best_index not in used_iou
            center_matched = center_index is not None and center_index not in used_center
            typed_ok = bool(matched and best_pred and best_pred.get("label") == label)
            if matched:
                used_iou.add(int(best_index))
                matched_pred_ids.add((row_id, int(best_index)))
            if center_matched:
                used_center.add(int(center_index))
            if matched and typed_ok:
                bucket_name = "matched_typed_ok"
            elif matched:
                bucket_name = "matched_type_error"
            elif center_matched:
                bucket_name = "center_only_low_iou"
            else:
                bucket_name = "missed_no_center"
            totals[bucket_name] += 1
            by_label[label][bucket_name] += 1
            by_area[bucket][bucket_name] += 1
            by_label_area[f"{label}|{bucket}"][bucket_name] += 1
            if bucket_name != "matched_typed_ok":
                hard_cases.append(
                    {
                        "row_id": row_id,
                        "bucket": bucket_name,
                        "gold": gold,
                        "area_bucket": bucket,
                        "best_iou": round(best_iou, 6),
                        "center_matched": center_matched,
                        "best_pred": best_pred,
                    }
                )

    for row_id, preds in page_preds.items():
        for index, pred in enumerate(preds):
            if (row_id, index) not in matched_pred_ids:
                totals["unmatched_predictions"] += 1
                by_label[f"pred:{pred.get('label')}"]["unmatched_predictions"] += 1
                by_pred_scale[f"unmatched_{tile_scale(pred.get('tile_id'))}"] += 1

    hard_cases.sort(
        key=lambda item: (
            item["bucket"] != "missed_no_center",
            item["bucket"] != "center_only_low_iou",
            item["best_iou"],
        )
    )
    report = {
        "version": "symbol_yolo_page_error_audit_v22",
        "claim_boundary": "Error mining only; gold is used for offline audit and targeted training planning, not runtime inference.",
        "dataset": rel(Path(args.data)),
        "yolo_dir": rel(Path(args.yolo_dir)),
        "predictions": rel(Path(args.predictions)),
        "config": vars(args),
        "totals": dict(totals),
        "rates": {
            "matched_typed_ok_recall": round(totals["matched_typed_ok"] / max(totals["gold"], 1), 6),
            "matched_any_iou_recall": round((totals["matched_typed_ok"] + totals["matched_type_error"]) / max(totals["gold"], 1), 6),
            "center_only_low_iou_rate": round(totals["center_only_low_iou"] / max(totals["gold"], 1), 6),
            "missed_no_center_rate": round(totals["missed_no_center"] / max(totals["gold"], 1), 6),
            "unmatched_prediction_rate": round(totals["unmatched_predictions"] / max(totals["predicted"], 1), 6),
        },
        "by_label": {key: dict(value) for key, value in sorted(by_label.items())},
        "by_area": {key: dict(value) for key, value in sorted(by_area.items())},
        "by_label_area": {key: dict(value) for key, value in sorted(by_label_area.items())},
        "prediction_tile_scale": dict(by_pred_scale),
        "hard_cases_written": min(len(hard_cases), args.top_hard_cases),
    }
    write_json(Path(args.report_output), report)
    write_jsonl(Path(args.hard_cases_output), hard_cases[: args.top_hard_cases])
    print(json.dumps({"report": rel(Path(args.report_output)), "hard_cases": rel(Path(args.hard_cases_output)), "rates": report["rates"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
