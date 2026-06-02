#!/usr/bin/env python3
"""P0-51: audit frozen v28 smoke predictions for localization/merger failure modes."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, center_covered, load_jsonl, rel, write_json

ROOT = Path(__file__).resolve().parents[2]


def load_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    preds: dict[str, list[dict[str, Any]]] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        if not raw.strip():
            continue
        row = json.loads(raw)
        preds[str(row["row_id"])] = list(row.get("predicted_symbols") or [])
    return preds


def load_page_golds(tile_jsonl: Path, row_ids: set[str]) -> dict[str, dict[str, dict[str, Any]]]:
    page_golds: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in load_jsonl(tile_jsonl):
        row_id = str(row.get("row_id"))
        if row_id not in row_ids:
            continue
        for gold in ((row.get("targets") or {}).get("boxes") or []):
            target_id = str(gold.get("target_id") or f"{row_id}_{len(page_golds[row_id])}")
            page_golds[row_id][target_id] = {
                "target_id": target_id,
                "bbox": [float(v) for v in gold.get("page_bbox") or gold.get("bbox")],
                "label": str(gold.get("label") or "generic_symbol"),
            }
    return page_golds


def classify_gold(gold: dict[str, Any], preds: list[dict[str, Any]], used_iou: set[int], used_center: set[int]) -> dict[str, Any]:
    gold_box = [float(v) for v in gold["bbox"]]
    gold_label = str(gold["label"])
    best_iou = 0.0
    best_iou_index: int | None = None
    best_center_iou = 0.0
    center_indices: list[int] = []
    same_label_center_indices: list[int] = []
    same_label_best_iou = 0.0
    for index, pred in enumerate(preds):
        pred_box = [float(v) for v in pred["bbox"]]
        iou = bbox_iou(pred_box, gold_box)
        if iou > best_iou:
            best_iou = iou
            best_iou_index = index
        if str(pred.get("label")) == gold_label:
            same_label_best_iou = max(same_label_best_iou, iou)
        if center_covered(pred_box, gold_box):
            center_indices.append(index)
            best_center_iou = max(best_center_iou, iou)
            if str(pred.get("label")) == gold_label:
                same_label_center_indices.append(index)

    matched_iou = best_iou_index is not None and best_iou >= 0.30 and best_iou_index not in used_iou
    matched_center = bool([idx for idx in center_indices if idx not in used_center])
    pred_label = str(preds[best_iou_index].get("label")) if best_iou_index is not None else None
    typed_ok = bool(matched_iou and pred_label == gold_label)

    if matched_iou:
        used_iou.add(best_iou_index)  # type: ignore[arg-type]
    if matched_center:
        for idx in center_indices:
            if idx not in used_center:
                used_center.add(idx)
                break

    if matched_iou and typed_ok:
        failure = "matched_iou_typed_ok"
    elif matched_iou:
        failure = "matched_iou_wrong_type"
    elif matched_center:
        failure = "center_only_poor_box"
    elif same_label_best_iou >= 0.10:
        failure = "same_type_near_miss_no_center"
    elif best_iou >= 0.10:
        failure = "wrong_type_or_loose_near_miss"
    else:
        failure = "proposal_absent"

    return {
        "target_id": gold["target_id"],
        "label": gold_label,
        "area_bucket": area_bucket(gold_box),
        "bbox": gold_box,
        "failure_mode": failure,
        "best_iou": round(best_iou, 6),
        "best_center_iou": round(best_center_iou, 6),
        "same_label_best_iou": round(same_label_best_iou, 6),
        "center_candidates": len(center_indices),
        "same_label_center_candidates": len(same_label_center_indices),
        "best_iou_label": pred_label,
        "matched_iou": matched_iou,
        "matched_center": matched_center,
        "typed_ok": typed_ok,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tiles", default="datasets/symbol_tile_detector_tiny_sahi_v21/smoke_v30.jsonl")
    parser.add_argument("--predictions", default="reports/vlm/symbol_yolov8s_seg_rect_v28_smoke_v30_page_predictions.jsonl")
    parser.add_argument("--output", default="reports/vlm/symbol_v28_frozen_localization_audit_p051.json")
    parser.add_argument("--examples-output", default="reports/vlm/symbol_v28_frozen_localization_audit_p051_examples.jsonl")
    parser.add_argument("--max-examples-per-mode", type=int, default=20)
    args = parser.parse_args()

    predictions = load_predictions(Path(args.predictions))
    golds = load_page_golds(Path(args.tiles), set(predictions))
    mode_counts = Counter()
    mode_by_label: dict[str, Counter[str]] = defaultdict(Counter)
    mode_by_area: dict[str, Counter[str]] = defaultdict(Counter)
    best_iou_hist = Counter()
    examples: dict[str, list[dict[str, Any]]] = defaultdict(list)
    total_preds = 0

    for row_id, gold_map in golds.items():
        preds = predictions.get(row_id, [])
        total_preds += len(preds)
        used_iou: set[int] = set()
        used_center: set[int] = set()
        for gold in gold_map.values():
            item = classify_gold(gold, preds, used_iou, used_center)
            mode = item["failure_mode"]
            label = item["label"]
            bucket = item["area_bucket"]
            mode_counts[mode] += 1
            mode_by_label[label][mode] += 1
            mode_by_area[bucket][mode] += 1
            best_iou = float(item["best_iou"])
            if best_iou >= 0.30:
                best_iou_hist["gte_0_30"] += 1
            elif best_iou >= 0.20:
                best_iou_hist["0_20_to_0_30"] += 1
            elif best_iou >= 0.10:
                best_iou_hist["0_10_to_0_20"] += 1
            else:
                best_iou_hist["lt_0_10"] += 1
            if len(examples[mode]) < args.max_examples_per_mode:
                examples[mode].append({"row_id": row_id, **item})

    total_gold = sum(mode_counts.values())
    report = {
        "version": "symbol_v28_frozen_localization_audit_p051",
        "claim_boundary": "Frozen v28 smoke_v30 diagnostic over existing detector predictions; no retraining and no runtime gold usage.",
        "source_integrity": {
            "runtime_inputs_audited": ["v28 raster detector predictions"],
            "gold_use": "offline_error_audit_only",
            "uses_svg_or_cad_geometry_at_runtime": False,
        },
        "inputs": {"tiles": rel(Path(args.tiles)), "predictions": rel(Path(args.predictions))},
        "totals": {"rows": len(golds), "gold": total_gold, "predicted": total_preds},
        "failure_modes": {mode: {"count": count, "rate": round(count / max(total_gold, 1), 6)} for mode, count in sorted(mode_counts.items())},
        "best_iou_hist": dict(best_iou_hist),
        "failure_modes_by_label": {label: dict(counter) for label, counter in sorted(mode_by_label.items())},
        "failure_modes_by_area": {bucket: dict(counter) for bucket, counter in sorted(mode_by_area.items())},
        "interpretation": {
            "center_only_poor_box": "Candidate center exists but box is too loose/tight for IoU@0.30; postprocess localization may help.",
            "proposal_absent": "No useful detector proposal near the gold; postprocess cannot recover without new proposals.",
            "matched_iou_wrong_type": "Localization is adequate but type head/label arbitration fails.",
        },
    }
    write_json(Path(args.output), report)
    with Path(args.examples_output).open("w", encoding="utf-8") as handle:
        for mode in sorted(examples):
            for item in examples[mode]:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
