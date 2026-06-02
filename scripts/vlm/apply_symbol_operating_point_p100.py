#!/usr/bin/env python3
"""Apply a registered symbol detector operating point to page predictions.

This is a post-processing utility: it does not run the detector or retrain. It
filters existing page-level symbol predictions by score and class-wise NMS so
precision-sensitive MoE overlays can be built without slow raster inference.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REGISTRY = ROOT / "configs/vlm/symbol_detector_operating_points_p100.json"
DEFAULT_INPUT = ROOT / "reports/vlm/symbol_yolov8s_seg_rect_v28_locked_page_predictions_p064_refresh.jsonl"
DEFAULT_OUTPUT = ROOT / "reports/vlm/symbol_yolov8s_seg_rect_v28_locked_precision_at_recall70_predictions_p100.jsonl"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def bbox_iou(a: list[float], b: list[float]) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    area_a = max(0.0, ax2 - ax1) * max(0.0, ay2 - ay1)
    area_b = max(0.0, bx2 - bx1) * max(0.0, by2 - by1)
    return inter / max(area_a + area_b - inter, 1e-9)


def pure_python_nms(preds: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    ordered = sorted(preds, key=lambda item: float(item.get("score") or 0.0), reverse=True)
    kept: list[dict[str, Any]] = []
    for pred in ordered:
        box = [float(v) for v in pred.get("bbox")[:4]]
        if all(bbox_iou(box, [float(v) for v in kept_pred.get("bbox")[:4]]) <= threshold for kept_pred in kept):
            kept.append(pred)
    return kept


def classwise_nms(preds: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    by_label: dict[str, list[dict[str, Any]]] = {}
    for pred in preds:
        label_key = str(pred.get("label_id", pred.get("label", "generic_symbol")))
        by_label.setdefault(label_key, []).append(pred)
    kept: list[dict[str, Any]] = []
    for label_preds in by_label.values():
        kept.extend(pure_python_nms(label_preds, threshold))
    kept.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
    return kept


def apply_operating_point(preds: list[dict[str, Any]], score_threshold: float, nms_threshold: float, max_per_page: int) -> list[dict[str, Any]]:
    valid: list[dict[str, Any]] = []
    for pred in preds:
        bbox = pred.get("bbox")
        if not isinstance(bbox, list) or len(bbox) < 4:
            continue
        try:
            x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
            score = float(pred.get("score") or 0.0)
        except (TypeError, ValueError):
            continue
        if x2 <= x1 or y2 <= y1 or score < score_threshold:
            continue
        item = dict(pred)
        item["bbox"] = [x1, y1, x2, y2]
        valid.append(item)
    return classwise_nms(valid, nms_threshold)[:max_per_page]


def resolve_operating_point(registry: dict[str, Any], operating_point: str) -> dict[str, Any]:
    points = registry.get("operating_points") or {}
    if operating_point not in points:
        raise SystemExit(f"unknown operating point {operating_point!r}; valid: {', '.join(sorted(points))}")
    point = points[operating_point]
    return {
        "id": operating_point,
        "score_threshold": float(point["score_threshold"]),
        "nms_threshold": float(point["nms_threshold"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--registry", default=str(DEFAULT_REGISTRY))
    parser.add_argument("--operating-point", default="precision_at_recall70_p100")
    parser.add_argument("--max-per-page", type=int, default=500)
    args = parser.parse_args()

    registry = load_json(Path(args.registry))
    point = resolve_operating_point(registry, args.operating_point)
    input_path = Path(args.input)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rows = 0
    before = 0
    after = 0
    with input_path.open("r", encoding="utf-8") as src, output_path.open("w", encoding="utf-8") as dst:
        for line in src:
            if not line.strip():
                continue
            row = json.loads(line)
            preds = list(row.get("predicted_symbols") or [])
            filtered = apply_operating_point(preds, point["score_threshold"], point["nms_threshold"], args.max_per_page)
            row["predicted_symbols"] = filtered
            row["symbol_operating_point"] = {
                "id": point["id"],
                "registry": str(Path(args.registry).relative_to(ROOT) if Path(args.registry).is_absolute() else args.registry),
                "score_threshold": point["score_threshold"],
                "nms_threshold": point["nms_threshold"],
                "source_predictions": str(input_path.relative_to(ROOT) if input_path.is_absolute() and input_path.is_relative_to(ROOT) else input_path),
                "postprocess_only": True,
            }
            rows += 1
            before += len(preds)
            after += len(filtered)
            dst.write(json.dumps(row, ensure_ascii=False) + "\n")

    summary = {
        "rows": rows,
        "input": str(input_path.relative_to(ROOT) if input_path.is_absolute() and input_path.is_relative_to(ROOT) else input_path),
        "output": str(output_path.relative_to(ROOT) if output_path.is_absolute() and output_path.is_relative_to(ROOT) else output_path),
        "operating_point": point,
        "predictions_before": before,
        "predictions_after": after,
        "mean_before": round(before / max(rows, 1), 6),
        "mean_after": round(after / max(rows, 1), 6),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
