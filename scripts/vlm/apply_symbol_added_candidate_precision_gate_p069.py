#!/usr/bin/env python3
"""Apply the fixed P0-69 added-candidate precision gate."""

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
from sweep_symbol_added_candidate_precision_gate_p069 import gated_predictions, rows_from_predictions
from sweep_symbol_rtdetr_complement_gate_p063 import read_predictions, score
from train_symbol_tile_detector_v20 import rel, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
DEFAULT_YOLO_DIR = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27"
FIXED_CONFIG = {
    "sources": "p067_only",
    "labels": "sink",
    "areas": "tiny",
    "score_min": 0.1,
    "min_iou_with_v28": 0.0,
    "max_iou_with_v28": 1.01,
    "max_add_per_page": 20,
}


def write_md(report: dict[str, Any], path: Path) -> None:
    b = report["baseline_v28"]
    f = report["full_combined_p068"]
    g = report["gated_policy"]
    d = report["delta_vs_v28"]
    lines = [
        f"# P0-69 fixed added-candidate precision gate - {report['split']}",
        "",
        "## Decision",
        "",
        f"- `{report['decision']}`",
        "",
        "## Metrics",
        "",
        f"- v28 IoU / inflation / precision: `{b['iou_0_30_recall']:.6f}` / `{b['candidate_inflation']:.6f}` / `{b['precision']:.6f}`",
        f"- P0-68 IoU / inflation / precision: `{f['iou_0_30_recall']:.6f}` / `{f['candidate_inflation']:.6f}` / `{f['precision']:.6f}`",
        f"- gated IoU / inflation / precision: `{g['iou_0_30_recall']:.6f}` / `{g['candidate_inflation']:.6f}` / `{g['precision']:.6f}`",
        f"- gated delta vs v28 IoU / inflation / precision: `{d['iou_0_30_recall']:+.6f}` / `{d['candidate_inflation']:+.6f}` / `{d['precision']:+.6f}`",
        f"- retained P0-68 IoU gain fraction: `{report['retained_iou_gain_fraction']:.6f}`",
        f"- tiny IoU: `{b['per_area_iou_recall'].get('tiny_le_64', 0.0):.6f}` -> `{g['per_area_iou_recall'].get('tiny_le_64', 0.0):.6f}`",
        f"- sink IoU: `{b['per_label_iou_recall'].get('sink', 0.0):.6f}` -> `{g['per_label_iou_recall'].get('sink', 0.0):.6f}`",
        "",
        "## Fixed Config",
        "",
        f"- `{json.dumps(report['gate_config'], ensure_ascii=False)}`",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--yolo-dir", default=str(DEFAULT_YOLO_DIR))
    parser.add_argument("--split", required=True)
    parser.add_argument("--v28-predictions", required=True)
    parser.add_argument("--combined-predictions", required=True)
    parser.add_argument("--output-predictions", required=True)
    parser.add_argument("--output-json", required=True)
    parser.add_argument("--output-md", required=True)
    args = parser.parse_args()
    v28 = read_predictions(Path(args.v28_predictions))
    combined = read_predictions(Path(args.combined_predictions))
    golds = read_exported_golds(Path(args.data), Path(args.yolo_dir), args.split, set(v28) | set(combined))
    baseline = score(golds, v28, {row_id: [] for row_id in v28}, {"labels": "all", "areas": "all", "score_min": 1.1, "max_iou_with_v28": 0.0, "max_add_per_page": 0})
    full = score(golds, combined, {row_id: [] for row_id in combined}, {"labels": "all", "areas": "all", "score_min": 1.1, "max_iou_with_v28": 0.0, "max_add_per_page": 0})
    gated = gated_predictions(v28, combined, FIXED_CONFIG)
    metrics = score(golds, gated, {row_id: [] for row_id in gated}, {"labels": "all", "areas": "all", "score_min": 1.1, "max_iou_with_v28": 0.0, "max_add_per_page": 0})
    delta = {k: round(metrics[k] - baseline[k], 6) for k in ["iou_0_30_recall", "center_recall", "candidate_inflation", "precision"]}
    retained = round((metrics["iou_0_30_recall"] - baseline["iou_0_30_recall"]) / max(full["iou_0_30_recall"] - baseline["iou_0_30_recall"], 1e-9), 6)
    decision = "positive_locked_precision_gate" if retained >= 0.5 and metrics["precision"] >= baseline["precision"] and metrics["candidate_inflation"] < full["candidate_inflation"] else "negative_do_not_promote_gate"
    out_pred = Path(args.output_predictions); out_json = Path(args.output_json); out_md = Path(args.output_md)
    out_pred.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(out_pred, rows_from_predictions(gated))
    report = {
        "version": "symbol_added_candidate_precision_gate_p069_fixed",
        "split": args.split,
        "source_integrity": "runtime gate uses raster-derived prediction fields only; gold is validation-only",
        "inputs": {"v28_predictions": rel(Path(args.v28_predictions)), "combined_predictions": rel(Path(args.combined_predictions))},
        "outputs": {"predictions": rel(out_pred), "json": rel(out_json), "markdown": rel(out_md)},
        "gate_config": FIXED_CONFIG,
        "baseline_v28": baseline,
        "full_combined_p068": full,
        "gated_policy": metrics,
        "delta_vs_v28": delta,
        "retained_iou_gain_fraction": retained,
        "decision": decision,
    }
    out_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    write_md(report, out_md)
    print(json.dumps({"split": args.split, "decision": decision, "delta": delta, "retained": retained, "gated": {k: metrics[k] for k in ["iou_0_30_recall", "candidate_inflation", "precision"]}}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
