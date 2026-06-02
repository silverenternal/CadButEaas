#!/usr/bin/env python3
"""Compare existing multi-scale symbol detector probes for v24."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from train_symbol_tile_detector_v20 import write_json


ROOT = Path(__file__).resolve().parents[2]

DEFAULT_REPORTS = [
    "reports/vlm/symbol_yolov8n_pretrained_v22_dedup_hi640_probe_page_eval.json",
    "reports/vlm/symbol_yolov8n_pretrained_v23_targeted_probe_page_eval.json",
    "reports/vlm/symbol_yolov8n_pretrained_v24_recall_probe_page_eval.json",
    "reports/vlm/symbol_yolo_p2_pretrained_v24_probe_page_eval.json",
    "reports/vlm/symbol_yolo_sliced_hi640_v24_page_eval.json",
    "reports/vlm/symbol_yolo_sliced_hi640_v24_smoke_page_eval.json",
]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def metric_block(data: dict[str, Any]) -> dict[str, Any]:
    return data.get("locked") or data.get("metrics") or {}


def summarize(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    metrics = metric_block(data)
    iou = metrics.get("symbol_bbox_iou_0_30") or {}
    return {
        "report": str(path.relative_to(ROOT) if path.is_relative_to(ROOT) else path),
        "weights": data.get("weights"),
        "version": data.get("version"),
        "selected_thresholds": data.get("selected_thresholds"),
        "center_recall": metrics.get("symbol_bbox_center_recall"),
        "iou_0_30_recall": iou.get("recall"),
        "precision": iou.get("precision"),
        "candidate_inflation": metrics.get("candidate_inflation"),
        "typed_accuracy_on_iou_matches": metrics.get("typed_accuracy_on_iou_matches"),
        "tiny_iou_recall": (metrics.get("area_iou_recall") or {}).get("tiny_le_64"),
        "small_iou_recall": (metrics.get("area_iou_recall") or {}).get("small_le_256"),
        "passed_stage_1": bool(
            (metrics.get("symbol_bbox_center_recall") or 0.0) >= 0.94
            and (iou.get("recall") or 0.0) >= 0.78
            and (iou.get("precision") or 0.0) > 0.096685
            and (metrics.get("symbol_bbox_center_recall") or 0.0) >= 0.911595
        ),
    }


def rank_key(row: dict[str, Any]) -> tuple[float, float, float, float]:
    return (
        1.0 if row.get("passed_stage_1") else 0.0,
        float(row.get("center_recall") or 0.0),
        float(row.get("iou_0_30_recall") or 0.0),
        float(row.get("precision") or 0.0),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reports", nargs="*", default=DEFAULT_REPORTS)
    parser.add_argument("--output", default="reports/vlm/symbol_multiscale_probe_v24_eval.json")
    args = parser.parse_args()

    rows = [item for item in (summarize(resolve(path)) for path in args.reports) if item is not None]
    rows.sort(key=rank_key, reverse=True)
    best = rows[0] if rows else {}
    report = {
        "version": "symbol_multiscale_probe_v24_eval",
        "task": "P0-03-symbol-proposal-and-type-adaptation.multiscale_probe",
        "claim_boundary": "Compares existing raster-only YOLO/P2/sliced symbol detector probes using their locked page-level reports; no oracle crop labels are used as runtime features.",
        "candidates": rows,
        "selected_best_available": best,
        "success_gate": {
            "stage_1_center_recall_min": 0.94,
            "stage_1_iou_0_30_recall_min": 0.78,
            "stage_1_precision_must_improve_over": 0.096685,
            "must_not_drop_center_recall_below": 0.911595,
            "best_center_recall": best.get("center_recall"),
            "best_iou_0_30_recall": best.get("iou_0_30_recall"),
            "best_precision": best.get("precision"),
            "passed": bool(best.get("passed_stage_1")),
        },
        "decision": "proceed_to_center_heatmap_route" if not bool(best.get("passed_stage_1")) else "use_best_multiscale_detector_then_train_type_adapter",
    }
    write_json(resolve(args.output), report)
    print(json.dumps({"best": best, "success_gate": report["success_gate"], "decision": report["decision"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
