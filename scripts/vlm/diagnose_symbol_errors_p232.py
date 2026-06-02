#!/usr/bin/env python3
"""Diagnose P229/P232 symbol proposal errors for contract repair planning."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from freeze_symbol_p222_p221a_sink_tiny import bbox_iou  # noqa: E402
from fuse_symbol_p206g_with_p211_p212 import load_p206g  # noqa: E402


DEFAULT_OVERLAY = ROOT / "reports" / "vlm" / "symbol_p224a_column_frozen_overlay.jsonl"
DEFAULT_OUT = ROOT / "reports" / "vlm" / "p232_symbol_error_diagnostic.json"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def distance(left: list[float], right: list[float]) -> float:
    lx, ly = center(left)
    rx, ry = center(right)
    return math.hypot(lx - rx, ly - ry)


def bucket_area(value: float) -> str:
    if value <= 64:
        return "tiny_le_64"
    if value <= 256:
        return "small_le_256"
    if value <= 1024:
        return "medium_le_1024"
    if value <= 4096:
        return "large_le_4096"
    return "xlarge_gt_4096"


def match_row(preds: list[dict[str, Any]], golds: dict[str, dict[str, Any]]) -> tuple[set[int], set[str], list[dict[str, Any]], list[dict[str, Any]]]:
    candidates = []
    for pred_index, pred in enumerate(preds):
        pbox = [float(v) for v in pred["bbox"]]
        plabel = str(pred["label"])
        for gold_id, gold in golds.items():
            if plabel != str(gold["label"]):
                continue
            iou = bbox_iou(pbox, [float(v) for v in gold["bbox"]])
            if iou >= 0.30:
                candidates.append((iou, pred_index, gold_id))
    used_preds: set[int] = set()
    used_golds: set[str] = set()
    for _iou, pred_index, gold_id in sorted(candidates, reverse=True):
        if pred_index in used_preds or gold_id in used_golds:
            continue
        used_preds.add(pred_index)
        used_golds.add(gold_id)

    false_negatives = []
    false_positives = []
    for gold_id, gold in golds.items():
        if gold_id not in used_golds:
            gbox = [float(v) for v in gold["bbox"]]
            nearest_same = None
            nearest_any = None
            for pred_index, pred in enumerate(preds):
                pbox = [float(v) for v in pred["bbox"]]
                item = {
                    "pred_index": pred_index,
                    "label": str(pred["label"]),
                    "score": float(pred.get("score") or 0.0),
                    "iou": bbox_iou(pbox, gbox),
                    "center_distance": distance(pbox, gbox),
                    "area_ratio": area(pbox) / max(area(gbox), 1e-6),
                }
                if nearest_any is None or item["center_distance"] < nearest_any["center_distance"]:
                    nearest_any = item
                if item["label"] == str(gold["label"]) and (nearest_same is None or item["center_distance"] < nearest_same["center_distance"]):
                    nearest_same = item
            false_negatives.append({"gold_id": gold_id, "label": str(gold["label"]), "bbox": gbox, "area": area(gbox), "nearest_same": nearest_same, "nearest_any": nearest_any})
    for pred_index, pred in enumerate(preds):
        if pred_index not in used_preds:
            pbox = [float(v) for v in pred["bbox"]]
            false_positives.append({"pred_index": pred_index, "label": str(pred["label"]), "bbox": pbox, "score": float(pred.get("score") or 0.0), "area": area(pbox)})
    return used_preds, used_golds, false_negatives, false_positives


def summarize_false_negatives(items: list[dict[str, Any]]) -> dict[str, Any]:
    by_label = Counter(item["label"] for item in items)
    by_area = Counter(bucket_area(item["area"]) for item in items)
    nearest_same = defaultdict(list)
    nearest_any = defaultdict(list)
    for item in items:
        if item.get("nearest_same"):
            nearest_same[item["label"]].append(item["nearest_same"])
        if item.get("nearest_any"):
            nearest_any[item["label"]].append(item["nearest_any"])
    details = {}
    for label in sorted(by_label):
        same = nearest_same.get(label, [])
        any_items = nearest_any.get(label, [])
        details[label] = {
            "fn": by_label[label],
            "area_buckets": dict(Counter(bucket_area(item["area"]) for item in items if item["label"] == label)),
            "nearest_same_median_distance": round(sorted([x["center_distance"] for x in same])[len(same) // 2], 3) if same else None,
            "nearest_same_iou_ge_0_10": sum(1 for x in same if x["iou"] >= 0.10),
            "nearest_any_label_counts": dict(Counter(x["label"] for x in any_items)),
        }
    return {"by_label": dict(by_label), "by_area": dict(by_area), "details": details}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    rows, preds_by_row, golds_by_row = load_p206g(args.overlay)
    all_fn = []
    all_fp = []
    counts = Counter()
    per_row = []
    for row in rows:
        row_id = str(row.get("id") or row.get("row_id"))
        preds = preds_by_row[row_id]
        golds = golds_by_row[row_id]
        used_preds, used_golds, false_negatives, false_positives = match_row(preds, golds)
        counts.update({"rows": 1, "tp": len(used_golds), "pred": len(preds), "gold": len(golds), "fp": len(false_positives), "fn": len(false_negatives)})
        for item in false_negatives:
            item["row_id"] = row_id
        for item in false_positives:
            item["row_id"] = row_id
        all_fn.extend(false_negatives)
        all_fp.extend(false_positives)
        per_row.append({"row_id": row_id, "tp": len(used_golds), "pred": len(preds), "gold": len(golds), "fp": len(false_positives), "fn": len(false_negatives)})

    report = {
        "id": "p232_symbol_error_diagnostic",
        "phase": "P232_precision_safe_contract_proposal_repair",
        "input": str(args.overlay),
        "counts": dict(counts),
        "false_negatives": summarize_false_negatives(all_fn),
        "false_positives_by_label": dict(Counter(item["label"] for item in all_fp)),
        "false_positive_area_buckets": dict(Counter(bucket_area(item["area"]) for item in all_fp)),
        "priority_labels": ["stair", "column", "equipment", "sink"],
        "examples": {
            "false_negatives": all_fn[:80],
            "false_positives": all_fp[:80],
        },
        "per_row_worst_fn": sorted(per_row, key=lambda item: item["fn"], reverse=True)[:20],
        "claim_boundary": "Offline diagnostic for rule design. Runtime repair rules may only use candidate labels/scores/bboxes/page size/constants, not gold fields.",
    }
    write_json(args.out, report)
    print(json.dumps({"out": str(args.out), "counts": report["counts"], "fn_by_label": report["false_negatives"]["by_label"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
