#!/usr/bin/env python3
"""Apply P0-70 precision-gated optional symbol policy in one command."""

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
from sweep_symbol_added_candidate_precision_gate_p069 import gated_predictions
from sweep_symbol_center_only_box_repair_p067 import repaired_predictions, rows_from_predictions
from sweep_symbol_rtdetr_complement_gate_p063 import read_predictions, score
from train_symbol_tile_detector_v20 import rel, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/vlm/symbol_precision_gated_policy_p070.json"
DEFAULT_DATA = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
DEFAULT_YOLO_DIR = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27"


def rows_to_map(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {str(row["row_id"]): list(row.get("predicted_symbols") or []) for row in rows}


def write_md(summary: dict[str, Any], path: Path) -> None:
    b = summary["baseline_v28"]
    c = summary["precision_gated_policy"]
    d = summary["delta_vs_v28"]
    lines = [
        f"# P0-70 precision-gated symbol policy - {summary['split']}",
        "",
        "## Decision",
        "",
        f"- `{summary['decision']}`",
        "",
        "## Metrics",
        "",
        f"- v28 IoU / center / inflation / precision: `{b['iou_0_30_recall']:.6f}` / `{b['center_recall']:.6f}` / `{b['candidate_inflation']:.6f}` / `{b['precision']:.6f}`",
        f"- P0-70 IoU / center / inflation / precision: `{c['iou_0_30_recall']:.6f}` / `{c['center_recall']:.6f}` / `{c['candidate_inflation']:.6f}` / `{c['precision']:.6f}`",
        f"- delta IoU / center / inflation / precision: `{d['iou_0_30_recall']:+.6f}` / `{d['center_recall']:+.6f}` / `{d['candidate_inflation']:+.6f}` / `{d['precision']:+.6f}`",
        f"- tiny IoU: `{b['per_area_iou_recall'].get('tiny_le_64', 0.0):.6f}` -> `{c['per_area_iou_recall'].get('tiny_le_64', 0.0):.6f}`",
        f"- sink IoU: `{b['per_label_iou_recall'].get('sink', 0.0):.6f}` -> `{c['per_label_iou_recall'].get('sink', 0.0):.6f}`",
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
    v28 = read_predictions(Path(args.v28_predictions))
    rtdetr = read_predictions(Path(args.rtdetr_predictions))
    p065 = rows_to_map(apply_p065_rows(v28, rtdetr, config["generation_steps"][0]["gate"]))
    p068 = repaired_predictions(p065, config["generation_steps"][1]["gate"])
    p070 = gated_predictions(v28, p068, config["precision_gate"])
    golds = read_exported_golds(Path(args.data), Path(args.yolo_dir), args.split, set(v28) | set(rtdetr) | set(p070))
    baseline = score(golds, v28, {row_id: [] for row_id in v28}, {"labels": "all", "areas": "all", "score_min": 1.1, "max_iou_with_v28": 0.0, "max_add_per_page": 0})
    metrics = score(golds, p070, {row_id: [] for row_id in p070}, {"labels": "all", "areas": "all", "score_min": 1.1, "max_iou_with_v28": 0.0, "max_add_per_page": 0})
    delta = {key: round(metrics[key] - baseline[key], 6) for key in ["iou_0_30_recall", "center_recall", "candidate_inflation", "precision"]}
    decision = "preferred_opt_in_policy_positive" if delta["iou_0_30_recall"] > 0 and metrics["precision"] >= baseline["precision"] else "policy_requires_review"
    out_pred = Path(args.output_predictions); out_json = Path(args.output_summary_json); out_md = Path(args.output_summary_md)
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_pred, rows_from_predictions(p070))
    summary = {
        "version": "symbol_precision_gated_policy_p070_result",
        "split": args.split,
        "source_integrity": config["runtime_input_boundary"],
        "policy_config": rel(Path(args.config)),
        "inputs": {"v28_predictions": rel(Path(args.v28_predictions)), "rtdetr_predictions": rel(Path(args.rtdetr_predictions)), "data": rel(Path(args.data)), "yolo_dir": rel(Path(args.yolo_dir))},
        "outputs": {"predictions": rel(out_pred), "summary_json": rel(out_json), "summary_md": rel(out_md)},
        "baseline_v28": baseline,
        "precision_gated_policy": metrics,
        "delta_vs_v28": delta,
        "decision": decision,
    }
    out_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_md(summary, out_md)
    print(json.dumps({"split": args.split, "decision": decision, "delta": delta, "metrics": {k: metrics[k] for k in ["iou_0_30_recall", "center_recall", "candidate_inflation", "precision"]}}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
