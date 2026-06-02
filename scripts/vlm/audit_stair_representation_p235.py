#!/usr/bin/env python3
"""P235 stair representation taxonomy and bucket metrics audit.

This script is offline-only: it reads locked targets to understand why stair is
unstable. It does not produce a runtime prediction artifact.
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from freeze_symbol_p222_p221a_sink_tiny import bbox_iou  # noqa: E402
from fuse_symbol_p206g_with_p211_p212 import load_p206g  # noqa: E402


DEFAULT_OVERLAY = ROOT / "reports" / "vlm" / "symbol_p224a_column_frozen_overlay.jsonl"
DEFAULT_P232 = ROOT / "reports" / "vlm" / "p232_repaired_contract_predictions.jsonl"
DEFAULT_P234 = ROOT / "reports" / "vlm" / "p234_stair_fusion_predictions.jsonl"
DEFAULT_AUDIT = ROOT / "reports" / "vlm" / "p235_stair_representation_audit.json"
DEFAULT_BUCKETS = ROOT / "reports" / "vlm" / "p235_stair_bucket_metrics.json"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def dims(box: list[float]) -> tuple[float, float]:
    return max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1])


def taxonomy(box: list[float]) -> str:
    width, height = dims(box)
    box_area = area(box)
    if box == [0.0, 0.0, 10.0, 12.0] or (box[0] == 0.0 and box[1] == 0.0 and width <= 12.0 and height <= 14.0):
        return "sentinel_placeholder"
    if min(width, height) <= 3.0 and max(width, height) >= 20.0:
        return "ultra_thin_tread"
    if box_area <= 64.0:
        return "tiny_real_or_noise"
    if box_area <= 256.0:
        return "small_stair_part"
    if box_area <= 4096.0:
        return "medium_large_stair_object"
    return "xlarge_grouped_stair"


def load_prediction_file(path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(path):
        preds = []
        for item in row.get("expert_predictions") or row.get("predicted_symbols") or row.get("symbol_candidates") or []:
            label = str(item.get("label") or item.get("symbol_type") or "")
            if label != "stair":
                continue
            preds.append({
                "id": str(item.get("candidate_id") or item.get("id") or len(preds)),
                "bbox": [float(v) for v in item["bbox"]],
                "label": "stair",
                "score": float(item.get("score", item.get("confidence", 0.0)) or 0.0),
                "source": str(item.get("source") or path.name),
            })
        out[str(row.get("row_id") or row.get("id"))] = preds
    return out


def match_bucket(preds_by_row: dict[str, list[dict[str, Any]]], stair_targets_by_row: dict[str, dict[str, dict[str, Any]]], row_ids: list[str]) -> dict[str, Any]:
    totals = defaultdict(Counter)
    examples = defaultdict(list)
    for row_id in row_ids:
        preds = preds_by_row.get(row_id, [])
        golds = stair_targets_by_row[row_id]
        candidates = []
        for pred_index, pred in enumerate(preds):
            pbox = pred["bbox"]
            for gold_id, gold in golds.items():
                iou = bbox_iou(pbox, gold["bbox"])
                if iou >= 0.30:
                    candidates.append((iou, pred_index, gold_id))
        used_preds: set[int] = set()
        used_golds: set[str] = set()
        for _iou, pred_index, gold_id in sorted(candidates, reverse=True):
            if pred_index in used_preds or gold_id in used_golds:
                continue
            used_preds.add(pred_index)
            used_golds.add(gold_id)
        pred_bucket = defaultdict(int)
        for pred_index, pred in enumerate(preds):
            best_bucket = "unmatched_fp"
            best_iou = 0.0
            for gold in golds.values():
                iou = bbox_iou(pred["bbox"], gold["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best_bucket = gold["taxonomy"]
            pred_bucket[best_bucket if best_iou >= 0.10 else "unmatched_fp"] += 1
        for bucket, count in pred_bucket.items():
            totals[bucket]["pred_near_or_fp"] += count
        for gold_id, gold in golds.items():
            bucket = gold["taxonomy"]
            totals[bucket]["gold"] += 1
            if gold_id in used_golds:
                totals[bucket]["tp"] += 1
            else:
                totals[bucket]["fn"] += 1
                if len(examples[bucket]) < 20:
                    examples[bucket].append({"row_id": row_id, "target_id": gold_id, "bbox": gold["bbox"], "area": gold["area"]})
        for pred_index, pred in enumerate(preds):
            if pred_index not in used_preds:
                totals["all_stair"]["fp"] += 1
        totals["all_stair"]["pred"] += len(preds)
    results = {}
    for bucket, counter in totals.items():
        gold = counter["gold"]
        tp = counter["tp"]
        pred = counter["pred"] if bucket == "all_stair" else counter["pred_near_or_fp"]
        fp = max(pred - tp, 0)
        fn = counter["fn"]
        precision = tp / max(pred, 1)
        recall = tp / max(gold, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        results[bucket] = {
            "tp": int(tp),
            "pred_or_near": int(pred),
            "gold": int(gold),
            "fp_est": int(fp),
            "fn": int(fn),
            "precision_or_near_precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1_or_near_f1": round(f1, 6),
        }
    return {"metrics": results, "fn_examples": dict(examples)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--p232", type=Path, default=DEFAULT_P232)
    parser.add_argument("--p234", type=Path, default=DEFAULT_P234)
    parser.add_argument("--audit-out", type=Path, default=DEFAULT_AUDIT)
    parser.add_argument("--bucket-out", type=Path, default=DEFAULT_BUCKETS)
    args = parser.parse_args()

    rows, _preds_by_row, golds_by_row = load_p206g(args.overlay)
    row_ids = [str(row.get("id") or row.get("row_id")) for row in rows]
    stair_targets_by_row: dict[str, dict[str, dict[str, Any]]] = {}
    taxonomy_counts = Counter()
    raw_label_counts = Counter()
    area_stats = defaultdict(list)
    for row in rows:
        row_id = str(row.get("id") or row.get("row_id"))
        stair_targets_by_row[row_id] = {}
        for target in (row.get("targets") or {}).get("symbol") or []:
            if str(target.get("semantic_type")) != "stair":
                continue
            box = [float(v) for v in target["bbox"]]
            bucket = taxonomy(box)
            target_id = str(target.get("target_id") or f"{row_id}_{len(stair_targets_by_row[row_id])}")
            item = {"target_id": target_id, "bbox": box, "label": "stair", "taxonomy": bucket, "area": area(box), "raw_label": str(target.get("raw_label") or "")}
            stair_targets_by_row[row_id][target_id] = item
            taxonomy_counts[bucket] += 1
            raw_label_counts[item["raw_label"]] += 1
            area_stats[bucket].append(item["area"])

    audit = {
        "id": "p235_stair_representation_audit",
        "phase": "P235_stair_representation_redesign_and_dataset_audit",
        "overlay": str(args.overlay),
        "total_stair_targets": sum(taxonomy_counts.values()),
        "taxonomy_counts": dict(taxonomy_counts),
        "raw_label_counts": dict(raw_label_counts),
        "area_summary": {
            bucket: {
                "count": len(values),
                "min": round(min(values), 3),
                "median": round(sorted(values)[len(values) // 2], 3),
                "max": round(max(values), 3),
            }
            for bucket, values in area_stats.items()
        },
        "taxonomy_definition": {
            "sentinel_placeholder": "[0,0,10,12]-like placeholder boxes; should not be mixed with real visual stair objects in model training claims.",
            "ultra_thin_tread": "One dimension <=3px and long dimension >=20px; line/tread representation, not object-like bbox.",
            "tiny_real_or_noise": "Area <=64, ambiguous tiny target.",
            "small_stair_part": "Area <=256, likely part-level stair target.",
            "medium_large_stair_object": "Area <=4096, object-like stair target.",
            "xlarge_grouped_stair": "Area >4096, grouped/structure-level stair target.",
        },
        "claim_boundary": "Offline target taxonomy audit. This file may reference labels/targets and is not a runtime prediction artifact.",
    }
    p232_metrics = match_bucket(load_prediction_file(args.p232), stair_targets_by_row, row_ids)
    p234_metrics = match_bucket(load_prediction_file(args.p234), stair_targets_by_row, row_ids)
    bucket_report = {
        "id": "p235_stair_bucket_metrics",
        "phase": "P235_stair_representation_redesign_and_dataset_audit",
        "p232_source": str(args.p232),
        "p234_source": str(args.p234),
        "taxonomy_counts": dict(taxonomy_counts),
        "p232_bucket_metrics": p232_metrics["metrics"],
        "p234_bucket_metrics": p234_metrics["metrics"],
        "p232_fn_examples": p232_metrics["fn_examples"],
        "p234_fn_examples": p234_metrics["fn_examples"],
        "claim_boundary": "Offline bucket scoring for representation design; not a runtime metric replacement.",
    }
    write_json(args.audit_out, audit)
    write_json(args.bucket_out, bucket_report)
    print(json.dumps({"audit": str(args.audit_out), "bucket_metrics": str(args.bucket_out), "taxonomy_counts": dict(taxonomy_counts)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
