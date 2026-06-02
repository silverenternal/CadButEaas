#!/usr/bin/env python3
"""Audit residual symbol misses after the P0-70 precision-gated policy."""

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

from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, center_covered, load_jsonl, rel, write_jsonl
from sweep_symbol_rtdetr_complement_gate_p063 import matched_target_ids, read_predictions

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
DEFAULT_YOLO_DIR = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27"
DEFAULT_POLICY = ROOT / "reports/vlm/symbol_precision_gated_policy_p070_locked_predictions.jsonl"
DEFAULT_V28 = ROOT / "reports/vlm/symbol_yolov8s_seg_rect_v28_locked_page_predictions_p064_refresh.jsonl"
DEFAULT_RTDETR = ROOT / "reports/vlm/symbol_rtdetr_l_bbox_p061_locked_page_predictions_p064.jsonl"
DEFAULT_JSON = ROOT / "reports/vlm/symbol_post_p070_residuals_p071_locked.json"
DEFAULT_MD = ROOT / "reports/vlm/symbol_post_p070_residuals_p071_locked.md"
DEFAULT_EXAMPLES = ROOT / "reports/vlm/symbol_post_p070_residual_examples_p071_locked.jsonl"


def read_exported_golds(data_dir: Path, yolo_dir: Path, split: str, row_ids: set[str]) -> dict[str, dict[str, dict[str, Any]]]:
    golds: dict[str, dict[str, dict[str, Any]]] = {}
    for tile_row in load_jsonl(data_dir / f"{split}.jsonl"):
        row_id = str(tile_row.get("row_id"))
        tile_id = str(tile_row.get("id"))
        if row_id not in row_ids:
            continue
        if not (yolo_dir / "images" / split / f"{tile_id}.jpg").exists():
            continue
        page_golds = golds.setdefault(row_id, {})
        for gold in ((tile_row.get("targets") or {}).get("boxes") or []):
            target_id = str(gold.get("target_id") or f"{row_id}_{len(page_golds)}")
            page_golds[target_id] = {
                "target_id": target_id,
                "bbox": [float(value) for value in gold.get("page_bbox") or gold.get("bbox")],
                "label": str(gold.get("label") or "generic_symbol"),
            }
    return golds


def best_iou(gold_bbox: list[float], predictions: list[dict[str, Any]]) -> tuple[float, dict[str, Any] | None]:
    best_score = 0.0
    best_prediction = None
    for prediction in predictions:
        score = bbox_iou([float(value) for value in prediction["bbox"]], gold_bbox)
        if score > best_score:
            best_score = score
            best_prediction = prediction
    return best_score, best_prediction


def has_center(gold_bbox: list[float], predictions: list[dict[str, Any]]) -> bool:
    return any(center_covered([float(value) for value in prediction["bbox"]], gold_bbox) for prediction in predictions)


def ratio(num: int, den: int) -> float:
    return round(num / den, 6) if den else 0.0


def bucket_table(total: Counter[str], missed: Counter[str], solved_by_policy: Counter[str], center_only: Counter[str], no_center: Counter[str]) -> dict[str, Any]:
    table = {}
    for key in sorted(total):
        table[key] = {
            "gold": int(total[key]),
            "missed_after_policy": int(missed[key]),
            "miss_rate_after_policy": ratio(int(missed[key]), int(total[key])),
            "solved_by_policy": int(solved_by_policy[key]),
            "solved_by_policy_rate": ratio(int(solved_by_policy[key]), int(total[key])),
            "center_only_after_policy": int(center_only[key]),
            "no_center_after_policy": int(no_center[key]),
        }
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--yolo-dir", default=str(DEFAULT_YOLO_DIR))
    parser.add_argument("--split", default="locked")
    parser.add_argument("--policy-predictions", default=str(DEFAULT_POLICY))
    parser.add_argument("--v28-predictions", default=str(DEFAULT_V28))
    parser.add_argument("--rtdetr-predictions", default=str(DEFAULT_RTDETR))
    parser.add_argument("--output-json", default=str(DEFAULT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_MD))
    parser.add_argument("--examples-output", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--max-examples", type=int, default=200)
    args = parser.parse_args()

    policy_predictions = read_predictions(Path(args.policy_predictions))
    v28_predictions = read_predictions(Path(args.v28_predictions))
    rtdetr_predictions = read_predictions(Path(args.rtdetr_predictions))
    golds = read_exported_golds(Path(args.data), Path(args.yolo_dir), args.split, set(policy_predictions) | set(v28_predictions) | set(rtdetr_predictions))

    total = Counter()
    missed = Counter()
    solved_by_policy = Counter()
    center_only = Counter()
    no_center = Counter()
    by_label = {name: Counter() for name in ["total", "missed", "solved", "center_only", "no_center"]}
    by_area = {name: Counter() for name in ["total", "missed", "solved", "center_only", "no_center"]}
    residual_modes = Counter()
    examples: list[dict[str, Any]] = []

    for row_id, gold_map in golds.items():
        policy_items = policy_predictions.get(row_id, [])
        v28_items = v28_predictions.get(row_id, [])
        rtdetr_items = rtdetr_predictions.get(row_id, [])
        v28_matched_iou = matched_target_ids(gold_map, v28_items, "iou")
        policy_matched_iou = matched_target_ids(gold_map, policy_items, "iou")
        for target_id, gold in gold_map.items():
            gold_bbox = [float(value) for value in gold["bbox"]]
            label = str(gold["label"])
            area = area_bucket(gold_bbox)
            total["gold"] += 1
            by_label["total"][label] += 1
            by_area["total"][area] += 1
            v28_iou, v28_best = best_iou(gold_bbox, v28_items)
            rtdetr_iou, rtdetr_best = best_iou(gold_bbox, rtdetr_items)
            policy_iou, policy_best = best_iou(gold_bbox, policy_items)
            v28_hit = target_id in v28_matched_iou
            policy_hit = target_id in policy_matched_iou
            if policy_hit:
                total["policy_hit"] += 1
                if not v28_hit:
                    solved_by_policy["gold"] += 1
                    solved_by_policy[label] += 1
                    solved_by_policy[area] += 1
                    by_label["solved"][label] += 1
                    by_area["solved"][area] += 1
                continue
            missed["gold"] += 1
            missed[label] += 1
            missed[area] += 1
            by_label["missed"][label] += 1
            by_area["missed"][area] += 1
            policy_center = has_center(gold_bbox, policy_items)
            v28_center = has_center(gold_bbox, v28_items)
            rtdetr_center = has_center(gold_bbox, rtdetr_items)
            if policy_center:
                mode = "center_only_poor_box"
                center_only["gold"] += 1
                center_only[label] += 1
                center_only[area] += 1
                by_label["center_only"][label] += 1
                by_area["center_only"][area] += 1
            elif v28_center or rtdetr_center:
                mode = "source_center_exists_but_not_policy"
            else:
                mode = "proposal_absent_no_center"
                no_center["gold"] += 1
                no_center[label] += 1
                no_center[area] += 1
                by_label["no_center"][label] += 1
                by_area["no_center"][area] += 1
            if policy_best is not None and policy_best.get("label") != label and policy_iou >= 0.10:
                mode = f"{mode}_wrong_type_nearby"
            residual_modes[mode] += 1
            if len(examples) < args.max_examples:
                examples.append({
                    "row_id": row_id,
                    "target_id": gold["target_id"],
                    "label": label,
                    "area_bucket": area,
                    "bbox": gold_bbox,
                    "mode": mode,
                    "v28_best_iou": round(v28_iou, 6),
                    "rtdetr_best_iou": round(rtdetr_iou, 6),
                    "policy_best_iou": round(policy_iou, 6),
                    "policy_best_label": None if policy_best is None else policy_best.get("label"),
                    "v28_center": v28_center,
                    "rtdetr_center": rtdetr_center,
                    "policy_center": policy_center,
                })

    report = {
        "version": "symbol_post_p070_residuals_p071",
        "split": args.split,
        "source_integrity": "offline gold is used only for locked residual audit; runtime policy uses raster-derived detector predictions only",
        "inputs": {"policy_predictions": rel(Path(args.policy_predictions)), "v28_predictions": rel(Path(args.v28_predictions)), "rtdetr_predictions": rel(Path(args.rtdetr_predictions))},
        "totals": {
            "gold": int(total["gold"]),
            "policy_hit_iou_0_30": int(total["policy_hit"]),
            "policy_iou_recall": ratio(int(total["policy_hit"]), int(total["gold"])),
            "missed_after_policy": int(missed["gold"]),
            "miss_rate_after_policy": ratio(int(missed["gold"]), int(total["gold"])),
            "solved_by_policy_vs_v28": int(solved_by_policy["gold"]),
            "center_only_after_policy": int(center_only["gold"]),
            "no_center_after_policy": int(no_center["gold"]),
        },
        "residual_modes": dict(residual_modes.most_common()),
        "by_label": bucket_table(by_label["total"], by_label["missed"], by_label["solved"], by_label["center_only"], by_label["no_center"]),
        "by_area": bucket_table(by_area["total"], by_area["missed"], by_area["solved"], by_area["center_only"], by_area["no_center"]),
        "decision_hint": "Prioritize the largest residual buckets with center-only poor boxes before adding broad high-inflation proposal heads.",
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    write_jsonl(Path(args.examples_output), examples)

    top_labels = sorted(report["by_label"].items(), key=lambda item: item[1]["missed_after_policy"], reverse=True)[:6]
    top_areas = sorted(report["by_area"].items(), key=lambda item: item[1]["missed_after_policy"], reverse=True)
    lines = [
        "# P0-71 post-P0-70 residual audit",
        "",
        "## Summary",
        "",
        f"- gold: `{report['totals']['gold']}`",
        f"- P0-70 IoU recall: `{report['totals']['policy_iou_recall']:.6f}`",
        f"- missed after policy: `{report['totals']['missed_after_policy']}` (`{report['totals']['miss_rate_after_policy']:.6f}`)",
        f"- solved by P0-70 vs v28: `{report['totals']['solved_by_policy_vs_v28']}`",
        f"- center-only poor box residuals: `{report['totals']['center_only_after_policy']}`",
        f"- no-center proposal-absent residuals: `{report['totals']['no_center_after_policy']}`",
        "",
        "## Residual Modes",
        "",
    ]
    for mode, count in residual_modes.most_common():
        lines.append(f"- `{mode}`: `{count}`")
    lines.extend(["", "## Top Label Residuals", ""])
    for label, values in top_labels:
        lines.append(f"- `{label}`: missed `{values['missed_after_policy']}` / gold `{values['gold']}`, center-only `{values['center_only_after_policy']}`, no-center `{values['no_center_after_policy']}`")
    lines.extend(["", "## Area Residuals", ""])
    for area, values in top_areas:
        lines.append(f"- `{area}`: missed `{values['missed_after_policy']}` / gold `{values['gold']}`, center-only `{values['center_only_after_policy']}`, no-center `{values['no_center_after_policy']}`")
    lines.extend(["", "## Artifacts", "", f"- `{rel(Path(args.output_json))}`", f"- `{rel(Path(args.examples_output))}`", ""])
    Path(args.output_md).write_text("\n".join(lines))
    print(json.dumps({"totals": report["totals"], "top_residual_modes": dict(residual_modes.most_common(5)), "top_labels": top_labels}, ensure_ascii=False, indent=2)[:6000])


if __name__ == "__main__":
    main()
