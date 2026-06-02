#!/usr/bin/env python3
"""Apply the fixed P0-67 smoke-selected geometric box repair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from apply_symbol_rtdetr_complement_policy_p065 import read_exported_golds
from sweep_symbol_center_only_box_repair_p067 import repaired_predictions, rows_from_predictions
from sweep_symbol_rtdetr_complement_gate_p063 import read_predictions, score
from train_symbol_tile_detector_v20 import rel, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
DEFAULT_YOLO_DIR = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27"
FIXED_CONFIG = {
    "labels": "sink_shower_equipment",
    "areas": "tiny_small",
    "score_min": 0.1,
    "scale_x": 0.85,
    "scale_y": 0.85,
    "max_add_per_page": 20,
}


def write_markdown(report: dict[str, Any], path: Path) -> None:
    baseline = report["baseline_policy"]
    repaired = report["repaired_policy"]
    delta = report["delta_vs_policy"]
    lines = [
        f"# P0-67 fixed center-only box repair - {report['split']}",
        "",
        "## Decision",
        "",
        f"- `{report['decision']}`",
        "",
        "## Metrics",
        "",
        f"- baseline IoU / center / inflation / precision: `{baseline['iou_0_30_recall']:.6f}` / `{baseline['center_recall']:.6f}` / `{baseline['candidate_inflation']:.6f}` / `{baseline['precision']:.6f}`",
        f"- repaired IoU / center / inflation / precision: `{repaired['iou_0_30_recall']:.6f}` / `{repaired['center_recall']:.6f}` / `{repaired['candidate_inflation']:.6f}` / `{repaired['precision']:.6f}`",
        f"- delta IoU / center / inflation / precision: `{delta['iou_0_30_recall']:+.6f}` / `{delta['center_recall']:+.6f}` / `{delta['candidate_inflation']:+.6f}` / `{delta['precision']:+.6f}`",
        f"- tiny IoU: `{baseline['per_area_iou_recall'].get('tiny_le_64', 0.0):.6f}` -> `{repaired['per_area_iou_recall'].get('tiny_le_64', 0.0):.6f}`",
        f"- sink / shower / equipment IoU: `{baseline['per_label_iou_recall'].get('sink', 0.0):.6f}` / `{baseline['per_label_iou_recall'].get('shower', 0.0):.6f}` / `{baseline['per_label_iou_recall'].get('equipment', 0.0):.6f}` -> `{repaired['per_label_iou_recall'].get('sink', 0.0):.6f}` / `{repaired['per_label_iou_recall'].get('shower', 0.0):.6f}` / `{repaired['per_label_iou_recall'].get('equipment', 0.0):.6f}`",
        "",
        "## Fixed Config",
        "",
        f"- `{json.dumps(report['repair_config'], ensure_ascii=False)}`",
        "",
        "## Artifacts",
        "",
        f"- `{report['outputs']['predictions']}`",
        f"- `{report['outputs']['json']}`",
        f"- `{report['outputs']['markdown']}`",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--yolo-dir", default=str(DEFAULT_YOLO_DIR))
    parser.add_argument("--split", required=True)
    parser.add_argument("--policy-predictions", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()

    policy_predictions = read_predictions(Path(args.policy_predictions))
    golds = read_exported_golds(Path(args.data), Path(args.yolo_dir), args.split, set(policy_predictions))
    baseline = score(golds, policy_predictions, {row_id: [] for row_id in policy_predictions}, {"labels": "all", "areas": "all", "score_min": 1.1, "max_iou_with_v28": 0.0, "max_add_per_page": 0})
    repaired_page_predictions = repaired_predictions(policy_predictions, FIXED_CONFIG)
    repaired = score(golds, repaired_page_predictions, {row_id: [] for row_id in repaired_page_predictions}, {"labels": "all", "areas": "all", "score_min": 1.1, "max_iou_with_v28": 0.0, "max_add_per_page": 0})
    delta = {key: round(float(repaired[key]) - float(baseline[key]), 6) for key in ["iou_0_30_recall", "center_recall", "candidate_inflation", "precision"]}
    decision = "positive_validate_for_packaging" if delta["iou_0_30_recall"] > 0 and delta["candidate_inflation"] <= 1.0 and delta["precision"] >= -0.015 else "negative_do_not_package"

    out_pred = Path(args.output_predictions)
    out_json = Path(args.output_json)
    out_md = Path(args.output_md)
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_pred, rows_from_predictions(repaired_page_predictions))
    report = {
        "version": "symbol_center_only_box_repair_p067_fixed",
        "split": args.split,
        "source_integrity": "offline gold is used only for validation; runtime repair uses detector boxes/scores/labels from raster-derived predictions only",
        "inputs": {"policy_predictions": rel(Path(args.policy_predictions)), "data": rel(Path(args.data)), "yolo_dir": rel(Path(args.yolo_dir))},
        "outputs": {"predictions": rel(out_pred), "json": rel(out_json), "markdown": rel(out_md)},
        "repair_config": FIXED_CONFIG,
        "baseline_policy": baseline,
        "repaired_policy": repaired,
        "delta_vs_policy": delta,
        "decision": decision,
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    write_markdown(report, out_md)
    print(json.dumps({"split": args.split, "decision": decision, "delta": delta, "repaired": {k: repaired[k] for k in ["iou_0_30_recall", "center_recall", "candidate_inflation", "precision"]}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
