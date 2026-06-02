#!/usr/bin/env python3
"""Audit complement coverage between two page-level symbol detector prediction files."""

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

from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, center_covered, load_jsonl, rel

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
DEFAULT_V28 = ROOT / "reports/vlm/symbol_yolov8s_seg_rect_v28_smoke_v30_page_predictions.jsonl"
DEFAULT_RTDETR = ROOT / "reports/vlm/symbol_rtdetr_l_bbox_p061_smoke4000_v3_smoke_v30_page_predictions.jsonl"
DEFAULT_JSON = ROOT / "reports/vlm/symbol_cross_detector_complement_p062_smoke_v30.json"
DEFAULT_MD = ROOT / "reports/vlm/symbol_cross_detector_complement_p062_smoke_v30.md"


def read_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    predictions: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(path):
        predictions[str(row["row_id"])] = list(row.get("predicted_symbols") or [])
    return predictions


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


def duplicate_against(reference: list[dict[str, Any]], candidates: list[dict[str, Any]], threshold: float) -> int:
    duplicate_count = 0
    for candidate in candidates:
        candidate_bbox = [float(value) for value in candidate["bbox"]]
        if any(bbox_iou(candidate_bbox, [float(value) for value in item["bbox"]]) >= threshold for item in reference):
            duplicate_count += 1
    return duplicate_count


def ratio(numerator: int, denominator: int) -> float:
    return round(numerator / denominator, 6) if denominator else 0.0


def bucket_report(counter_total: Counter[str], counter_v28: Counter[str], counter_rtdetr: Counter[str], counter_both: Counter[str], counter_union: Counter[str], counter_unique_rtdetr: Counter[str]) -> dict[str, Any]:
    report: dict[str, Any] = {}
    for key in sorted(counter_total):
        total = int(counter_total[key])
        report[key] = {
            "gold": total,
            "v28_recall": ratio(int(counter_v28[key]), total),
            "rtdetr_recall": ratio(int(counter_rtdetr[key]), total),
            "both_recall": ratio(int(counter_both[key]), total),
            "union_recall": ratio(int(counter_union[key]), total),
            "unique_rtdetr_recall": ratio(int(counter_unique_rtdetr[key]), total),
            "unique_rtdetr_count": int(counter_unique_rtdetr[key]),
        }
    return report


def audit(args: argparse.Namespace) -> dict[str, Any]:
    v28_predictions = read_predictions(Path(args.v28_predictions))
    rtdetr_predictions = read_predictions(Path(args.rtdetr_predictions))
    row_ids = set(v28_predictions) | set(rtdetr_predictions)
    golds = read_golds(Path(args.data), args.split, row_ids)
    modes = ["center", "iou"]
    totals = {mode: Counter() for mode in modes}
    by_label = {mode: {name: Counter() for name in ["total", "v28", "rtdetr", "both", "union", "unique_rtdetr"]} for mode in modes}
    by_area = {mode: {name: Counter() for name in ["total", "v28", "rtdetr", "both", "union", "unique_rtdetr"]} for mode in modes}
    examples: list[dict[str, Any]] = []

    candidate_counts = {
        "v28": sum(len(items) for items in v28_predictions.values()),
        "rtdetr": sum(len(items) for items in rtdetr_predictions.values()),
    }
    candidate_counts["combined_raw"] = candidate_counts["v28"] + candidate_counts["rtdetr"]
    candidate_counts["rtdetr_duplicates_against_v28_iou_0_50"] = sum(
        duplicate_against(v28_predictions.get(row_id, []), rtdetr_predictions.get(row_id, []), 0.50) for row_id in row_ids
    )

    for row_id, gold_map in golds.items():
        v28_items = v28_predictions.get(row_id, [])
        rtdetr_items = rtdetr_predictions.get(row_id, [])
        combined_items = v28_items + rtdetr_items
        matched_by_mode = {
            mode: {
                "v28": matched_target_ids(gold_map, v28_items, mode),
                "rtdetr": matched_target_ids(gold_map, rtdetr_items, mode),
                "union": matched_target_ids(gold_map, combined_items, mode),
            }
            for mode in modes
        }
        for target_id, gold in gold_map.items():
            label = str(gold["label"])
            area = area_bucket([float(value) for value in gold["bbox"]])
            for mode in modes:
                v28_hit = target_id in matched_by_mode[mode]["v28"]
                rtdetr_hit = target_id in matched_by_mode[mode]["rtdetr"]
                union_hit = target_id in matched_by_mode[mode]["union"]
                totals[mode]["gold"] += 1
                by_label[mode]["total"][label] += 1
                by_area[mode]["total"][area] += 1
                if v28_hit:
                    totals[mode]["v28"] += 1
                    by_label[mode]["v28"][label] += 1
                    by_area[mode]["v28"][area] += 1
                if rtdetr_hit:
                    totals[mode]["rtdetr"] += 1
                    by_label[mode]["rtdetr"][label] += 1
                    by_area[mode]["rtdetr"][area] += 1
                if v28_hit and rtdetr_hit:
                    totals[mode]["both"] += 1
                    by_label[mode]["both"][label] += 1
                    by_area[mode]["both"][area] += 1
                if union_hit:
                    totals[mode]["union"] += 1
                    by_label[mode]["union"][label] += 1
                    by_area[mode]["union"][area] += 1
                if (not v28_hit) and rtdetr_hit:
                    totals[mode]["unique_rtdetr"] += 1
                    by_label[mode]["unique_rtdetr"][label] += 1
                    by_area[mode]["unique_rtdetr"][area] += 1
                    if mode == "iou" and len(examples) < args.max_examples:
                        examples.append({"row_id": row_id, "target_id": gold["target_id"], "label": label, "area_bucket": area, "bbox": gold["bbox"]})

    gold_total = int(totals["iou"]["gold"])
    mode_reports = {}
    for mode in modes:
        mode_totals = totals[mode]
        mode_reports[mode] = {
            "gold": int(mode_totals["gold"]),
            "v28_matched": int(mode_totals["v28"]),
            "rtdetr_matched": int(mode_totals["rtdetr"]),
            "both_matched": int(mode_totals["both"]),
            "union_matched": int(mode_totals["union"]),
            "unique_rtdetr_matched": int(mode_totals["unique_rtdetr"]),
            "v28_recall": ratio(int(mode_totals["v28"]), int(mode_totals["gold"])),
            "rtdetr_recall": ratio(int(mode_totals["rtdetr"]), int(mode_totals["gold"])),
            "union_recall": ratio(int(mode_totals["union"]), int(mode_totals["gold"])),
            "unique_rtdetr_recall": ratio(int(mode_totals["unique_rtdetr"]), int(mode_totals["gold"])),
            "per_label": bucket_report(by_label[mode]["total"], by_label[mode]["v28"], by_label[mode]["rtdetr"], by_label[mode]["both"], by_label[mode]["union"], by_label[mode]["unique_rtdetr"]),
            "per_area": bucket_report(by_area[mode]["total"], by_area[mode]["v28"], by_area[mode]["rtdetr"], by_area[mode]["both"], by_area[mode]["union"], by_area[mode]["unique_rtdetr"]),
        }

    candidate_counts["v28_inflation"] = ratio(candidate_counts["v28"], gold_total)
    candidate_counts["rtdetr_inflation"] = ratio(candidate_counts["rtdetr"], gold_total)
    candidate_counts["combined_raw_inflation"] = ratio(candidate_counts["combined_raw"], gold_total)
    candidate_counts["rtdetr_duplicate_rate_vs_v28_iou_0_50"] = ratio(candidate_counts["rtdetr_duplicates_against_v28_iou_0_50"], candidate_counts["rtdetr"])

    return {
        "version": "symbol_cross_detector_complement_p062",
        "split": args.split,
        "source_integrity": "offline gold is used only for smoke audit; runtime detector inputs remain raster tile pixels only",
        "inputs": {
            "data": rel(Path(args.data)),
            "v28_predictions": rel(Path(args.v28_predictions)),
            "rtdetr_predictions": rel(Path(args.rtdetr_predictions)),
        },
        "rows": len(golds),
        "candidate_counts": candidate_counts,
        "center": mode_reports["center"],
        "iou_0_30": mode_reports["iou"],
        "unique_rtdetr_iou_examples": examples,
        "decision_hint": "promote complement gate only if unique RTDETR IoU gains are material relative to the added inflation",
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    iou_report = report["iou_0_30"]
    center_report = report["center"]
    candidates = report["candidate_counts"]
    lines = [
        "# P0-62 cross-detector complement audit",
        "",
        "## Summary",
        "",
        f"- rows: `{report['rows']}`",
        f"- IoU@0.30 v28 / RTDETR / union recall: `{iou_report['v28_recall']:.6f}` / `{iou_report['rtdetr_recall']:.6f}` / `{iou_report['union_recall']:.6f}`",
        f"- unique RTDETR IoU gain: `{iou_report['unique_rtdetr_recall']:.6f}` ({iou_report['unique_rtdetr_matched']} golds)",
        f"- center v28 / RTDETR / union recall: `{center_report['v28_recall']:.6f}` / `{center_report['rtdetr_recall']:.6f}` / `{center_report['union_recall']:.6f}`",
        f"- candidate inflation v28 / RTDETR / raw union: `{candidates['v28_inflation']:.6f}` / `{candidates['rtdetr_inflation']:.6f}` / `{candidates['combined_raw_inflation']:.6f}`",
        f"- RTDETR duplicate rate vs v28 at IoU>=0.50: `{candidates['rtdetr_duplicate_rate_vs_v28_iou_0_50']:.6f}`",
        "",
        "## Per-size unique RTDETR IoU gains",
        "",
    ]
    for bucket, values in iou_report["per_area"].items():
        lines.append(f"- `{bucket}`: unique `{values['unique_rtdetr_recall']:.6f}` ({values['unique_rtdetr_count']}/{values['gold']}), union `{values['union_recall']:.6f}`")
    lines.extend(["", "## Per-class unique RTDETR IoU gains", ""])
    for label, values in iou_report["per_label"].items():
        lines.append(f"- `{label}`: unique `{values['unique_rtdetr_recall']:.6f}` ({values['unique_rtdetr_count']}/{values['gold']}), union `{values['union_recall']:.6f}`")
    lines.extend(["", "## Decision Hint", "", report["decision_hint"], ""])
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--split", default="smoke_v30")
    parser.add_argument("--v28-predictions", default=str(DEFAULT_V28))
    parser.add_argument("--rtdetr-predictions", default=str(DEFAULT_RTDETR))
    parser.add_argument("--output-json", default=str(DEFAULT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_MD))
    parser.add_argument("--max-examples", type=int, default=50)
    args = parser.parse_args()

    report = audit(args)
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    write_markdown(report, output_md)
    print(json.dumps({"center": report["center"], "iou_0_30": report["iou_0_30"], "candidate_counts": report["candidate_counts"]}, ensure_ascii=False, indent=2)[:6000])


if __name__ == "__main__":
    main()
