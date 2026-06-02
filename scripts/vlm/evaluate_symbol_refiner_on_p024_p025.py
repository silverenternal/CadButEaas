#!/usr/bin/env python3
"""P0-25: evaluate existing symbol bbox refiners on P0-24 localization repair cases."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

import joblib
import numpy as np

from train_symbol_box_refiner_v38 import apply_delta
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import bbox_iou, write_json, write_jsonl

FOCUS_LABELS = {"sink", "shower", "stair", "equipment"}
FOCUS_AREAS = {"tiny_le_64", "small_le_256"}


def box_stats(box: list[float]) -> dict[str, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    width = max(x2 - x1, 1e-6)
    height = max(y2 - y1, 1e-6)
    return {"x1": x1, "y1": y1, "x2": x2, "y2": y2, "cx": (x1 + x2) * 0.5, "cy": (y1 + y2) * 0.5, "width": width, "height": height, "area": width * height, "aspect": width / height}


def feature_row(row: dict[str, Any]) -> dict[str, Any]:
    box = [float(v) for v in row["candidate_bbox"]]
    stats = box_stats(box)
    image_size = row.get("image_size") or [1000, 1000]
    image_w = max(float(image_size[0] if len(image_size) > 0 else 1000), 1.0)
    image_h = max(float(image_size[1] if len(image_size) > 1 else 1000), 1.0)
    area_bucket = str(row.get("area_bucket") or "")
    label = str(row.get("candidate_label") or row.get("label") or "")
    features = {
        **stats,
        "score": float(row.get("candidate_score") or 0.0),
        "x1_norm": stats["x1"] / image_w,
        "y1_norm": stats["y1"] / image_h,
        "x2_norm": stats["x2"] / image_w,
        "y2_norm": stats["y2"] / image_h,
        "cx_norm": stats["cx"] / image_w,
        "cy_norm": stats["cy"] / image_h,
        "width_norm": stats["width"] / image_w,
        "height_norm": stats["height"] / image_h,
        "area_norm": stats["area"] / max(image_w * image_h, 1.0),
        "label_id": float(row.get("label_id") or 0.0),
        "pre_selector_score": float(row.get("candidate_score") or 0.0),
        "page_candidate_count": 500.0,
        "cluster_id_mod_17": 0.0,
        "cluster_mask_count": 0.0,
        "cluster_score_max": float(row.get("candidate_score") or 0.0),
        "cluster_score_mean": float(row.get("candidate_score") or 0.0),
        "cluster_size": 1.0,
        "cluster_source_center_count": 0.0,
        "is_center_branch_v30": 0.0,
        "is_mask_v28": 1.0,
        "pred_area_bucket_tiny": 1.0 if stats["area"] <= 64.0 else 0.0,
        "area_is_tiny": 1.0 if area_bucket == "tiny_le_64" else 0.0,
        "area_is_small": 1.0 if area_bucket == "small_le_256" else 0.0,
        "label_is_equipment": 1.0 if label == "equipment" else 0.0,
        "label_is_shower": 1.0 if label == "shower" else 0.0,
        "label_is_sink": 1.0 if label == "sink" else 0.0,
        "label_is_stair": 1.0 if label == "stair" else 0.0,
    }
    return {"features": features, "bbox": box, "candidate_bbox": box, **row}


def vector(row: dict[str, Any], names: list[str]) -> list[float]:
    feats = (row.get("features") or {})
    return [float(feats.get(name, 0.0) or 0.0) for name in names]


def quant(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"count": 0, "mean": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.asarray(vals, dtype=np.float64)
    return {"count": int(arr.size), "mean": round(float(arr.mean()), 6), "p50": round(float(np.quantile(arr, .5)), 6), "p90": round(float(np.quantile(arr, .9)), 6)}


def evaluate_model(rows: list[dict[str, Any]], checkpoint: str, clip: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bundle = joblib.load(source_path(checkpoint))
    model = bundle.get("model") or bundle.get("refiner")
    names = list(bundle.get("feature_names") or [])
    if model is None or not names:
        return {"checkpoint": checkpoint, "supported": False, "reason": "missing model or feature_names"}, []
    prepared = [feature_row(row) for row in rows]
    preds = model.predict(np.asarray([vector(row, names) for row in prepared], dtype=np.float32))
    out_rows = []
    counters = Counter()
    by_label = defaultdict(Counter)
    by_area = defaultdict(Counter)
    input_ious = []
    refined_ious = []
    for row, delta in zip(prepared, preds, strict=True):
        box = [float(v) for v in row["candidate_bbox"]]
        target = [float(v) for v in row["target_bbox"]]
        refined = apply_delta(box, list(delta), clip)
        input_iou = bbox_iou(box, target)
        refined_iou = bbox_iou(refined, target)
        label = str(row.get("label") or "")
        area = str(row.get("area_bucket") or "")
        input_ious.append(input_iou)
        refined_ious.append(refined_iou)
        counters["rows"] += 1
        counters["input_hit"] += int(input_iou >= 0.30)
        counters["refined_hit"] += int(refined_iou >= 0.30)
        counters["improved"] += int(refined_iou > input_iou)
        counters["worse"] += int(refined_iou < input_iou)
        for group in [by_label[label], by_area[area]]:
            group["rows"] += 1
            group["input_hit"] += int(input_iou >= 0.30)
            group["refined_hit"] += int(refined_iou >= 0.30)
        out = dict(row)
        out.update({"checkpoint": checkpoint, "refined_bbox": [round(float(v), 6) for v in refined], "refined_iou": round(refined_iou, 6), "input_iou_recomputed": round(input_iou, 6), "delta_pred": [round(float(v), 6) for v in list(delta)]})
        out_rows.append(out)
    n = max(counters["rows"], 1)
    def rates(c: Counter) -> dict[str, Any]:
        m = max(c["rows"], 1)
        return {"rows": int(c["rows"]), "input_hit_rate": round(c["input_hit"] / m, 6), "refined_hit_rate": round(c["refined_hit"] / m, 6)}
    return {
        "checkpoint": checkpoint,
        "supported": True,
        "feature_count": len(names),
        "clip": clip,
        "rows": int(counters["rows"]),
        "input_hit_rate": round(counters["input_hit"] / n, 6),
        "refined_hit_rate": round(counters["refined_hit"] / n, 6),
        "hit_gain": int(counters["refined_hit"] - counters["input_hit"]),
        "improved_rate": round(counters["improved"] / n, 6),
        "worse_rate": round(counters["worse"] / n, 6),
        "input_iou": quant(input_ious),
        "refined_iou": quant(refined_ious),
        "by_label": {k: rates(v) for k, v in sorted(by_label.items())},
        "by_area": {k: rates(v) for k, v in sorted(by_area.items())},
    }, out_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/symbol_localization_repair_p024/smoke_center_low_iou.jsonl")
    parser.add_argument("--checkpoints", default="checkpoints/symbol_box_refiner_v38/model.joblib,checkpoints/symbol_box_refiner_v38_sink_tiny_v49/model.joblib")
    parser.add_argument("--clip", type=float, default=0.75)
    parser.add_argument("--output", default="reports/vlm/symbol_refiner_on_p024_p025_eval.json")
    parser.add_argument("--rows-output", default="reports/vlm/symbol_refiner_on_p024_p025_rows.jsonl")
    args = parser.parse_args()
    rows = load_jsonl(source_path(args.dataset))
    focus = [row for row in rows if str(row.get("label") or "") in FOCUS_LABELS and str(row.get("area_bucket") or "") in FOCUS_AREAS]
    reports = []
    all_out = []
    for checkpoint in [part.strip() for part in args.checkpoints.split(",") if part.strip()]:
        report_all, rows_all = evaluate_model(rows, checkpoint, args.clip)
        report_focus, rows_focus = evaluate_model(focus, checkpoint, args.clip)
        reports.append({"checkpoint": checkpoint, "all": report_all, "focus": report_focus})
        all_out.extend(rows_focus[:2000])
    best_focus = max((r for r in reports if r["focus"].get("supported")), key=lambda r: r["focus"].get("hit_gain", -10**9), default=None)
    output = {
        "version": "symbol_refiner_on_p024_p025",
        "dataset": args.dataset,
        "rows": len(rows),
        "focus_rows": len(focus),
        "reports": reports,
        "decision": {
            "best_focus_checkpoint": (best_focus or {}).get("checkpoint"),
            "best_focus_hit_gain": (best_focus or {}).get("focus", {}).get("hit_gain"),
            "recommendation": "reuse_existing_refiner_if_gain_positive_else_build_p024_specific_refiner_or_proposal_generator",
        },
        "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False, "offline_labels_used_for": ["refiner_evaluation_on_repair_cases"], "final_quality_claim_allowed": False},
    }
    write_json(source_path(args.output), output)
    write_jsonl(source_path(args.rows_output), all_out)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
