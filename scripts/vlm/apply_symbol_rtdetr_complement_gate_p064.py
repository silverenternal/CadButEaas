#!/usr/bin/env python3
"""Apply the fixed P0-63 sparse RTDETR complement gate on locked predictions."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sweep_symbol_rtdetr_complement_gate_p063 import read_predictions, score
from train_symbol_tile_detector_v20 import rel

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
DEFAULT_V28 = ROOT / "reports/vlm/symbol_yolov8s_seg_rect_v28_locked_page_predictions_p064_refresh.jsonl"
DEFAULT_RTDETR = ROOT / "reports/vlm/symbol_rtdetr_l_bbox_p061_locked_page_predictions_p064.jsonl"
DEFAULT_YOLO_DIR = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27"
DEFAULT_JSON = ROOT / "reports/vlm/symbol_rtdetr_complement_gate_p064_locked.json"
DEFAULT_MD = ROOT / "reports/vlm/symbol_rtdetr_complement_gate_p064_locked.md"


def read_exported_golds(data_dir: Path, yolo_dir: Path, split: str, row_ids: set[str]) -> dict[str, dict[str, dict[str, Any]]]:
    golds: dict[str, dict[str, dict[str, Any]]] = {}
    for tile_row in __import__("train_symbol_tile_detector_v20").load_jsonl(data_dir / f"{split}.jsonl"):
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

FIXED_CONFIG = {
    "labels": "sink_only",
    "areas": "tiny_le_64",
    "score_min": 0.2,
    "max_iou_with_v28": 1.01,
    "max_add_per_page": 20,
}


def write_markdown(report: dict[str, Any], path: Path) -> None:
    baseline = report["baseline_v28"]
    gated = report["fixed_gate"]
    delta = report["delta_vs_v28"]
    lines = [
        "# P0-64 locked validation for sparse RTDETR complement gate",
        "",
        "## Decision",
        "",
        f"- `{report['decision']}`",
        "",
        "## Fixed Gate",
        "",
        f"- config: `{json.dumps(report['gate_config'], ensure_ascii=False)}`",
        "",
        "## Locked Metrics",
        "",
        f"- v28 IoU recall / center / inflation / precision: `{baseline['iou_0_30_recall']:.6f}` / `{baseline['center_recall']:.6f}` / `{baseline['candidate_inflation']:.6f}` / `{baseline['precision']:.6f}`",
        f"- gated IoU recall / center / inflation / precision: `{gated['iou_0_30_recall']:.6f}` / `{gated['center_recall']:.6f}` / `{gated['candidate_inflation']:.6f}` / `{gated['precision']:.6f}`",
        f"- delta IoU recall / center / inflation / precision: `{delta['iou_0_30_recall']:+.6f}` / `{delta['center_recall']:+.6f}` / `{delta['candidate_inflation']:+.6f}` / `{delta['precision']:+.6f}`",
        f"- unique recovered IoU: `{gated['unique_recovered_iou_recall']:.6f}` ({gated['unique_recovered_iou']} golds)",
        f"- tiny IoU recall v28 -> gated: `{baseline['per_area_iou_recall'].get('tiny_le_64', 0.0):.6f}` -> `{gated['per_area_iou_recall'].get('tiny_le_64', 0.0):.6f}`",
        f"- sink IoU recall v28 -> gated: `{baseline['per_label_iou_recall'].get('sink', 0.0):.6f}` -> `{gated['per_label_iou_recall'].get('sink', 0.0):.6f}`",
        "",
        "## Artifacts",
        "",
        f"- `{rel(Path(report['inputs']['v28_predictions']))}`",
        f"- `{rel(Path(report['inputs']['rtdetr_predictions']))}`",
        f"- `{rel(Path(report['outputs']['json']))}`",
        f"- `{rel(Path(report['outputs']['markdown']))}`",
        "",
    ]
    path.write_text("\n".join(lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DEFAULT_DATA))
    parser.add_argument("--split", default="locked")
    parser.add_argument("--v28-predictions", default=str(DEFAULT_V28))
    parser.add_argument("--rtdetr-predictions", default=str(DEFAULT_RTDETR))
    parser.add_argument("--yolo-dir", default=str(DEFAULT_YOLO_DIR))
    parser.add_argument("--output-json", default=str(DEFAULT_JSON))
    parser.add_argument("--output-md", default=str(DEFAULT_MD))
    args = parser.parse_args()

    v28_predictions = read_predictions(Path(args.v28_predictions))
    rtdetr_predictions = read_predictions(Path(args.rtdetr_predictions))
    golds = read_exported_golds(Path(args.data), Path(args.yolo_dir), args.split, set(v28_predictions) | set(rtdetr_predictions))
    baseline = score(golds, v28_predictions, {row_id: [] for row_id in v28_predictions}, {"labels": "all", "areas": "all", "score_min": 1.1, "max_iou_with_v28": 0.0, "max_add_per_page": 0})
    gated = score(golds, v28_predictions, rtdetr_predictions, FIXED_CONFIG)
    delta = {
        key: round(float(gated[key]) - float(baseline[key]), 6)
        for key in ["iou_0_30_recall", "center_recall", "candidate_inflation", "precision"]
    }
    passes = delta["iou_0_30_recall"] > 0 and gated["per_area_iou_recall"].get("tiny_le_64", 0.0) >= baseline["per_area_iou_recall"].get("tiny_le_64", 0.0) and delta["candidate_inflation"] <= 2.0
    decision = "positive_locked_optional_complement" if passes else "negative_locked_do_not_promote"
    report = {
        "version": "symbol_rtdetr_complement_gate_p064_locked",
        "source_integrity": "offline gold is used only for locked validation; runtime features are detector predictions from raster tile pixels",
        "inputs": {"data": str(Path(args.data)), "yolo_dir": str(Path(args.yolo_dir)), "v28_predictions": str(Path(args.v28_predictions)), "rtdetr_predictions": str(Path(args.rtdetr_predictions))},
        "outputs": {"json": str(Path(args.output_json)), "markdown": str(Path(args.output_md))},
        "gate_config": FIXED_CONFIG,
        "baseline_v28": baseline,
        "fixed_gate": gated,
        "delta_vs_v28": delta,
        "decision": decision,
    }
    output_json = Path(args.output_json)
    output_md = Path(args.output_md)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    write_markdown(report, output_md)
    print(json.dumps({"decision": decision, "baseline_v28": baseline, "fixed_gate": gated, "delta_vs_v28": delta}, ensure_ascii=False, indent=2)[:6000])


if __name__ == "__main__":
    main()
