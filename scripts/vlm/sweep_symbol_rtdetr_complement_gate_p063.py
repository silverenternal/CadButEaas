#!/usr/bin/env python3
"""Sweep sparse gates for adding RTDETR complement predictions to frozen v28."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from train_symbol_tile_detector_v20 import ID_TO_LABEL, area_bucket, bbox_iou, center_covered, load_jsonl, rel

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
DEFAULT_V28 = ROOT / "reports/vlm/symbol_yolov8s_seg_rect_v28_smoke_v30_page_predictions_p062_refresh.jsonl"
DEFAULT_RTDETR = ROOT / "reports/vlm/symbol_rtdetr_l_bbox_p061_smoke4000_v3_smoke_v30_page_predictions.jsonl"
DEFAULT_JSON = ROOT / "reports/vlm/symbol_rtdetr_complement_gate_p063_smoke_v30.json"
DEFAULT_MD = ROOT / "reports/vlm/symbol_rtdetr_complement_gate_p063_smoke_v30.md"


def read_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    return {str(row["row_id"]): list(row.get("predicted_symbols") or []) for row in load_jsonl(path)}


def read_golds(data_dir: Path, split: str, row_ids: set[str]) -> dict[str, dict[str, dict[str, Any]]]:
    golds: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for tile_row in load_jsonl(data_dir / f"{split}.jsonl"):
        row_id = str(tile_row.get("row_id"))
        if row_id not in row_ids:
            continue
        for gold in ((tile_row.get("targets") or {}).get("boxes") or []):
            target_id = str(gold.get("target_id") or f"{row_id}_{len(golds[row_id])}")
            golds[row_id][target_id] = {
                "target_id": target_id,
                "bbox": [float(value) for value in gold.get("page_bbox") or gold.get("bbox")],
                "label": str(gold.get("label") or "generic_symbol"),
            }
    return dict(golds)


def matched_target_ids(gold_map: dict[str, dict[str, Any]], predictions: list[dict[str, Any]], mode: str) -> set[str]:
    matched: set[str] = set()
    used_predictions: set[int] = set()
    for target_id, gold in gold_map.items():
        gold_bbox = [float(value) for value in gold["bbox"]]
        selected_index: int | None = None
        if mode == "iou":
            best_iou = 0.0
            for pred_index, prediction in enumerate(predictions):
                pred_bbox = [float(value) for value in prediction["bbox"]]
                current_iou = bbox_iou(pred_bbox, gold_bbox)
                if current_iou > best_iou:
                    best_iou = current_iou
                    selected_index = pred_index
            if selected_index is not None and best_iou >= 0.30 and selected_index not in used_predictions:
                used_predictions.add(selected_index)
                matched.add(str(target_id))
        else:
            for pred_index, prediction in enumerate(predictions):
                if pred_index in used_predictions:
                    continue
                pred_bbox = [float(value) for value in prediction["bbox"]]
                if center_covered(pred_bbox, gold_bbox):
                    selected_index = pred_index
                    break
            if selected_index is not None:
                used_predictions.add(selected_index)
                matched.add(str(target_id))
    return matched


def max_overlap(prediction: dict[str, Any], references: list[dict[str, Any]]) -> float:
    pred_bbox = [float(value) for value in prediction["bbox"]]
    return max((bbox_iou(pred_bbox, [float(value) for value in reference["bbox"]]) for reference in references), default=0.0)


def pred_area_bucket(prediction: dict[str, Any]) -> str:
    return area_bucket([float(value) for value in prediction["bbox"]])


def allowed_labels(name: str) -> set[str]:
    if name == "all":
        return set(ID_TO_LABEL.values())
    if name == "sink_tiny_focus":
        return {"sink", "appliance", "column", "equipment", "generic_symbol"}
    if name == "sink_only":
        return {"sink"}
    if name == "tiny_candidate_classes":
        return {"sink", "appliance", "column", "equipment"}
    return {name}


def allowed_area_buckets(name: str) -> set[str]:
    if name == "all":
        return {"tiny_le_64", "small_le_256", "medium_le_1024", "large_le_4096", "xlarge_gt_4096"}
    if name == "tiny_small_medium":
        return {"tiny_le_64", "small_le_256", "medium_le_1024"}
    if name == "tiny_small":
        return {"tiny_le_64", "small_le_256"}
    return {name}


def select_rtdetr(predictions: list[dict[str, Any]], v28_predictions: list[dict[str, Any]], config: dict[str, Any]) -> list[dict[str, Any]]:
    labels = allowed_labels(str(config["labels"]))
    areas = allowed_area_buckets(str(config["areas"]))
    selected = []
    for prediction in predictions:
        if str(prediction.get("label")) not in labels:
            continue
        if float(prediction.get("score", 0.0)) < float(config["score_min"]):
            continue
        if pred_area_bucket(prediction) not in areas:
            continue
        if max_overlap(prediction, v28_predictions) >= float(config["max_iou_with_v28"]):
            continue
        selected.append(prediction)
    selected.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
    return selected[: int(config["max_add_per_page"])]


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def score(golds: dict[str, dict[str, dict[str, Any]]], v28_predictions: dict[str, list[dict[str, Any]]], rtdetr_predictions: dict[str, list[dict[str, Any]]], config: dict[str, Any]) -> dict[str, Any]:
    totals = Counter()
    by_label_total = Counter()
    by_label_iou = Counter()
    by_area_total = Counter()
    by_area_iou = Counter()
    v28_iou_total = 0
    unique_recovered = 0
    added_total = 0
    for row_id, gold_map in golds.items():
        v28_items = v28_predictions.get(row_id, [])
        added_items = select_rtdetr(rtdetr_predictions.get(row_id, []), v28_items, config)
        combined_items = v28_items + added_items
        v28_iou = matched_target_ids(gold_map, v28_items, "iou")
        combined_iou = matched_target_ids(gold_map, combined_items, "iou")
        combined_center = matched_target_ids(gold_map, combined_items, "center")
        v28_iou_total += len(v28_iou)
        unique_recovered += len(combined_iou - v28_iou)
        added_total += len(added_items)
        for target_id, gold in gold_map.items():
            label = str(gold["label"])
            bucket = area_bucket([float(value) for value in gold["bbox"]])
            by_label_total[label] += 1
            by_area_total[bucket] += 1
            if target_id in combined_iou:
                by_label_iou[label] += 1
                by_area_iou[bucket] += 1
        totals["gold"] += len(gold_map)
        totals["predicted"] += len(combined_items)
        totals["matched_iou"] += len(combined_iou)
        totals["matched_center"] += len(combined_center)
    precision = ratio(int(totals["matched_iou"]), int(totals["predicted"]))
    recall = ratio(int(totals["matched_iou"]), int(totals["gold"]))
    return {
        "config": config,
        "matched_iou": int(totals["matched_iou"]),
        "gold": int(totals["gold"]),
        "predicted": int(totals["predicted"]),
        "added_rtdetr": added_total,
        "precision": precision,
        "iou_0_30_recall": recall,
        "center_recall": ratio(int(totals["matched_center"]), int(totals["gold"])),
        "candidate_inflation": ratio(int(totals["predicted"]), int(totals["gold"])),
        "unique_recovered_iou": unique_recovered,
        "unique_recovered_iou_recall": ratio(unique_recovered, int(totals["gold"])),
        "per_label_iou_recall": {label: ratio(int(by_label_iou[label]), int(by_label_total[label])) for label in sorted(by_label_total)},
        "per_area_iou_recall": {bucket: ratio(int(by_area_iou[bucket]), int(by_area_total[bucket])) for bucket in sorted(by_area_total)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--split", default="smoke_v30")
    parser.add_argument("--v28-predictions", default=str(DEFAULT_V28))
    parser.add_argument("--rtdetr-predictions", default=str(DEFAULT_RTDETR))
    parser.add_argument("--output-json", default=str(DEFAULT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_MD))
    args = parser.parse_args()

    v28_predictions = read_predictions(Path(args.v28_predictions))
    rtdetr_predictions = read_predictions(Path(args.rtdetr_predictions))
    golds = read_golds(Path(args.data), args.split, set(v28_predictions) | set(rtdetr_predictions))
    configs = []
    for labels in ["all", "sink_tiny_focus", "sink_only"]:
        for areas in ["all", "tiny_le_64"]:
            for score_min in [0.20, 0.50]:
                for max_iou_with_v28 in [0.35, 1.01]:
                    for max_add_per_page in [5, 20]:
                        configs.append({
                            "labels": labels,
                            "areas": areas,
                            "score_min": score_min,
                            "max_iou_with_v28": max_iou_with_v28,
                            "max_add_per_page": max_add_per_page,
                        })
    results = [score(golds, v28_predictions, rtdetr_predictions, config) for config in configs]
    baseline = score(golds, v28_predictions, {row_id: [] for row_id in v28_predictions}, {"labels": "all", "areas": "all", "score_min": 1.1, "max_iou_with_v28": 0.0, "max_add_per_page": 0})
    budget = baseline["candidate_inflation"] + 2.0
    feasible = [item for item in results if item["candidate_inflation"] <= budget and item["iou_0_30_recall"] >= baseline["iou_0_30_recall"]]
    feasible.sort(key=lambda item: (item["iou_0_30_recall"], item["per_area_iou_recall"].get("tiny_le_64", 0.0), -item["candidate_inflation"]), reverse=True)
    all_sorted = sorted(results, key=lambda item: (item["iou_0_30_recall"], item["per_area_iou_recall"].get("tiny_le_64", 0.0), -item["candidate_inflation"]), reverse=True)
    report = {
        "version": "symbol_rtdetr_complement_gate_p063",
        "source_integrity": "offline gold is used only for smoke gate sweep; runtime features are detector predictions from raster tile pixels",
        "inputs": {"data": rel(Path(args.data)), "v28_predictions": rel(Path(args.v28_predictions)), "rtdetr_predictions": rel(Path(args.rtdetr_predictions))},
        "budget_candidate_inflation_lte": round(budget, 6),
        "baseline_v28": baseline,
        "best_feasible": feasible[0] if feasible else None,
        "top_feasible": feasible[:20],
        "top_unconstrained": all_sorted[:20],
        "sweep_count": len(results),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    best = report["best_feasible"]
    lines = ["# P0-63 sparse RTDETR complement gate sweep", "", "## Summary", ""]
    lines.append(f"- baseline v28 IoU recall / inflation: `{baseline['iou_0_30_recall']:.6f}` / `{baseline['candidate_inflation']:.6f}`")
    lines.append(f"- budget inflation <= `{budget:.6f}`")
    if best:
        lines.extend([
            f"- best feasible IoU recall / inflation: `{best['iou_0_30_recall']:.6f}` / `{best['candidate_inflation']:.6f}`",
            f"- best feasible center recall: `{best['center_recall']:.6f}`",
            f"- unique recovered IoU: `{best['unique_recovered_iou_recall']:.6f}` ({best['unique_recovered_iou']} golds)",
            f"- tiny / sink IoU recall: `{best['per_area_iou_recall'].get('tiny_le_64', 0.0):.6f}` / `{best['per_label_iou_recall'].get('sink', 0.0):.6f}`",
            f"- config: `{json.dumps(best['config'], ensure_ascii=False)}`",
        ])
    else:
        lines.append("- no feasible gate improved recall within budget")
    lines.extend(["", "## Artifacts", "", f"- `{rel(output_json)}`", f"- `{rel(Path(args.output_md))}`", ""])
    Path(args.output_md).write_text("\n".join(lines))
    print(json.dumps({"baseline": baseline, "best_feasible": best, "budget": budget}, ensure_ascii=False, indent=2)[:6000])


if __name__ == "__main__":
    main()
