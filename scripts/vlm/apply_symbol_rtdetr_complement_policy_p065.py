#!/usr/bin/env python3
"""Apply the packaged P0-65 optional RTDETR complement policy."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from sweep_symbol_rtdetr_complement_gate_p063 import read_predictions, score, select_rtdetr
from train_symbol_tile_detector_v20 import load_jsonl, rel, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/vlm/symbol_rtdetr_complement_policy_p065.json"
DEFAULT_DATA = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
DEFAULT_YOLO_DIR = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27"


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


def apply_policy(v28_predictions: dict[str, list[dict[str, Any]]], rtdetr_predictions: dict[str, list[dict[str, Any]]], gate: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_id in sorted(set(v28_predictions) | set(rtdetr_predictions)):
        v28_items = list(v28_predictions.get(row_id, []))
        added_items = select_rtdetr(list(rtdetr_predictions.get(row_id, [])), v28_items, gate)
        merged = v28_items + [{**item, "source_policy": "p065_rtdetr_complement"} for item in added_items]
        rows.append({"row_id": row_id, "predicted_symbols": merged, "added_rtdetr_count": len(added_items)})
    return rows


def write_markdown(summary: dict[str, Any], path: Path) -> None:
    baseline = summary["baseline_v28"]
    policy = summary["policy"]
    delta = summary["delta_vs_v28"]
    lines = [
        f"# P0-65 optional RTDETR complement policy - {summary['split']}",
        "",
        "## Metrics",
        "",
        f"- v28 IoU / center / inflation / precision: `{baseline['iou_0_30_recall']:.6f}` / `{baseline['center_recall']:.6f}` / `{baseline['candidate_inflation']:.6f}` / `{baseline['precision']:.6f}`",
        f"- policy IoU / center / inflation / precision: `{policy['iou_0_30_recall']:.6f}` / `{policy['center_recall']:.6f}` / `{policy['candidate_inflation']:.6f}` / `{policy['precision']:.6f}`",
        f"- delta IoU / center / inflation / precision: `{delta['iou_0_30_recall']:+.6f}` / `{delta['center_recall']:+.6f}` / `{delta['candidate_inflation']:+.6f}` / `{delta['precision']:+.6f}`",
        f"- unique recovered IoU: `{policy['unique_recovered_iou_recall']:.6f}` ({policy['unique_recovered_iou']} golds)",
        f"- tiny IoU v28 -> policy: `{baseline['per_area_iou_recall'].get('tiny_le_64', 0.0):.6f}` -> `{policy['per_area_iou_recall'].get('tiny_le_64', 0.0):.6f}`",
        f"- sink IoU v28 -> policy: `{baseline['per_label_iou_recall'].get('sink', 0.0):.6f}` -> `{policy['per_label_iou_recall'].get('sink', 0.0):.6f}`",
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
    gate = config["gate"]
    v28_predictions = read_predictions(Path(args.v28_predictions))
    rtdetr_predictions = read_predictions(Path(args.rtdetr_predictions))
    row_ids = set(v28_predictions) | set(rtdetr_predictions)
    golds = read_exported_golds(Path(args.data), Path(args.yolo_dir), args.split, row_ids)
    baseline = score(golds, v28_predictions, {row_id: [] for row_id in v28_predictions}, {"labels": "all", "areas": "all", "score_min": 1.1, "max_iou_with_v28": 0.0, "max_add_per_page": 0})
    policy = score(golds, v28_predictions, rtdetr_predictions, gate)
    delta = {key: round(float(policy[key]) - float(baseline[key]), 6) for key in ["iou_0_30_recall", "center_recall", "candidate_inflation", "precision"]}
    merged_rows = apply_policy(v28_predictions, rtdetr_predictions, gate)
    output_predictions = Path(args.output_predictions)
    output_summary_json = Path(args.output_summary_json)
    output_summary_md = Path(args.output_summary_md)
    output_predictions.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(output_predictions, merged_rows)
    summary = {
        "version": "symbol_rtdetr_complement_policy_p065_result",
        "split": args.split,
        "source_integrity": config["runtime_input_boundary"],
        "policy_config": rel(Path(args.config)),
        "inputs": {"v28_predictions": rel(Path(args.v28_predictions)), "rtdetr_predictions": rel(Path(args.rtdetr_predictions)), "data": rel(Path(args.data)), "yolo_dir": rel(Path(args.yolo_dir))},
        "outputs": {"predictions": rel(output_predictions), "summary_json": rel(output_summary_json), "summary_md": rel(output_summary_md)},
        "baseline_v28": baseline,
        "policy": policy,
        "delta_vs_v28": delta,
        "decision": "optional_policy_positive" if delta["iou_0_30_recall"] > 0 and delta["candidate_inflation"] <= 2.0 else "optional_policy_not_promoted",
    }
    output_summary_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n")
    write_markdown(summary, output_summary_md)
    print(json.dumps({"split": args.split, "decision": summary["decision"], "delta": delta, "policy": {k: policy[k] for k in ["iou_0_30_recall", "center_recall", "candidate_inflation", "precision", "unique_recovered_iou"]}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
