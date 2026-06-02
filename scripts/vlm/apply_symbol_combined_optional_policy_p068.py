#!/usr/bin/env python3
"""Apply the combined optional P0-68 symbol policy: P0-65 complement then P0-67 repair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from apply_symbol_rtdetr_complement_policy_p065 import apply_policy as apply_p065_rows
from apply_symbol_rtdetr_complement_policy_p065 import read_exported_golds
from sweep_symbol_center_only_box_repair_p067 import repaired_predictions, rows_from_predictions
from sweep_symbol_rtdetr_complement_gate_p063 import read_predictions, score
from train_symbol_tile_detector_v20 import rel, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/vlm/symbol_combined_optional_policy_p068.json"
DEFAULT_DATA = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
DEFAULT_YOLO_DIR = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27"


def rows_to_prediction_map(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {str(row["row_id"]): list(row.get("predicted_symbols") or []) for row in rows}


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    baseline = summary["baseline_v28"]
    combined = summary["combined_policy"]
    delta = summary["delta_vs_v28"]
    lines = [
        f"# P0-68 combined optional symbol policy - {summary['split']}",
        "",
        "## Decision",
        "",
        f"- `{summary['decision']}`",
        "",
        "## Metrics",
        "",
        f"- v28 IoU / center / inflation / precision: `{baseline['iou_0_30_recall']:.6f}` / `{baseline['center_recall']:.6f}` / `{baseline['candidate_inflation']:.6f}` / `{baseline['precision']:.6f}`",
        f"- combined IoU / center / inflation / precision: `{combined['iou_0_30_recall']:.6f}` / `{combined['center_recall']:.6f}` / `{combined['candidate_inflation']:.6f}` / `{combined['precision']:.6f}`",
        f"- delta IoU / center / inflation / precision: `{delta['iou_0_30_recall']:+.6f}` / `{delta['center_recall']:+.6f}` / `{delta['candidate_inflation']:+.6f}` / `{delta['precision']:+.6f}`",
        f"- tiny IoU: `{baseline['per_area_iou_recall'].get('tiny_le_64', 0.0):.6f}` -> `{combined['per_area_iou_recall'].get('tiny_le_64', 0.0):.6f}`",
        f"- sink / shower / equipment IoU: `{baseline['per_label_iou_recall'].get('sink', 0.0):.6f}` / `{baseline['per_label_iou_recall'].get('shower', 0.0):.6f}` / `{baseline['per_label_iou_recall'].get('equipment', 0.0):.6f}` -> `{combined['per_label_iou_recall'].get('sink', 0.0):.6f}` / `{combined['per_label_iou_recall'].get('shower', 0.0):.6f}` / `{combined['per_label_iou_recall'].get('equipment', 0.0):.6f}`",
        "",
        "## Artifacts",
        "",
        f"- `{summary['outputs']['predictions']}`",
        f"- `{summary['outputs']['summary_json']}`",
        f"- `{summary['outputs']['summary_md']}`",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--yolo-dir", default=str(DEFAULT_YOLO_DIR))
    parser.add_argument("--split", required=True)
    parser.add_argument("--v28-predictions", required=True)
    parser.add_argument("--rtdetr-predictions", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-summary-json", required=True)
    parser.add_argument("--output-summary-md", required=True)
    args = parser.parse_args()

    config = json.loads(Path(args.config).read_text())
    p065_gate = config["steps"][0]["gate"]
    p067_gate = config["steps"][1]["gate"]
    v28_predictions = read_predictions(Path(args.v28_predictions))
    rtdetr_predictions = read_predictions(Path(args.rtdetr_predictions))
    p065_rows = apply_p065_rows(v28_predictions, rtdetr_predictions, p065_gate)
    p065_predictions = rows_to_prediction_map(p065_rows)
    combined_predictions = repaired_predictions(p065_predictions, p067_gate)
    row_ids = set(v28_predictions) | set(rtdetr_predictions) | set(combined_predictions)
    golds = read_exported_golds(Path(args.data), Path(args.yolo_dir), args.split, row_ids)
    baseline = score(golds, v28_predictions, {row_id: [] for row_id in v28_predictions}, {"labels": "all", "areas": "all", "score_min": 1.1, "max_iou_with_v28": 0.0, "max_add_per_page": 0})
    combined = score(golds, combined_predictions, {row_id: [] for row_id in combined_predictions}, {"labels": "all", "areas": "all", "score_min": 1.1, "max_iou_with_v28": 0.0, "max_add_per_page": 0})
    delta = {key: round(float(combined[key]) - float(baseline[key]), 6) for key in ["iou_0_30_recall", "center_recall", "candidate_inflation", "precision"]}
    output_predictions = Path(args.output_predictions)
    output_summary_json = Path(args.output_summary_json)
    output_summary_md = Path(args.output_summary_md)
    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_predictions, rows_from_predictions(combined_predictions))
    summary = {
        "version": "symbol_combined_optional_policy_p068_result",
        "split": args.split,
        "source_integrity": config["runtime_input_boundary"],
        "policy_config": rel(Path(args.config)),
        "inputs": {"v28_predictions": rel(Path(args.v28_predictions)), "rtdetr_predictions": rel(Path(args.rtdetr_predictions)), "data": rel(Path(args.data)), "yolo_dir": rel(Path(args.yolo_dir))},
        "outputs": {"predictions": rel(output_predictions), "summary_json": rel(output_summary_json), "summary_md": rel(output_summary_md)},
        "baseline_v28": baseline,
        "combined_policy": combined,
        "delta_vs_v28": delta,
        "decision": "optional_combined_policy_positive" if delta["iou_0_30_recall"] > 0 and delta["candidate_inflation"] <= 2.0 else "optional_combined_policy_not_promoted",
    }
    output_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_markdown(summary, output_summary_md)
    print(json.dumps({"split": args.split, "decision": summary["decision"], "delta": delta, "combined": {k: combined[k] for k in ["iou_0_30_recall", "center_recall", "candidate_inflation", "precision"]}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
