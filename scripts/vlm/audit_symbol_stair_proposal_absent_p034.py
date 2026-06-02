#!/usr/bin/env python3
"""P0-34: audit stair proposal_absent cases for runtime-safe proposal feasibility."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import bbox_iou, write_json, write_jsonl


def valid_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    box = [float(v) for v in value]
    return box if box[2] > box[0] and box[3] > box[1] else None


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def dims(box: list[float]) -> tuple[float, float, float, float]:
    w = max(box[2] - box[0], 1e-6)
    h = max(box[3] - box[1], 1e-6)
    return w, h, w * h, w / h


def dist_norm(a: list[float], b: list[float]) -> float:
    acx, acy = center(a)
    bcx, bcy = center(b)
    bw, bh, _, _ = dims(b)
    return float(np.hypot(acx - bcx, acy - bcy) / max(np.hypot(bw, bh), 1e-6))


def load_gold(tile_rows: str) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in load_jsonl(source_path(tile_rows)):
        page = str(row.get("row_id") or "")
        image_size = row.get("image_size") or [0, 0]
        for target in (row.get("targets") or {}).get("boxes") or []:
            tid = str(target.get("target_id") or "")
            box = valid_box(target.get("page_bbox") or target.get("bbox"))
            if page and tid and box and tid not in out[page]:
                out[page][tid] = {"bbox": box, "label": str(target.get("label") or ""), "area_bucket": str(target.get("area_bucket") or ""), "image_size": image_size}
    return out


def load_recovery_by_page(recovery_data: str, split: str) -> dict[str, list[dict[str, Any]]]:
    manifest = json.loads(source_path(recovery_data).read_text(encoding="utf-8"))
    rows = load_jsonl(source_path(manifest["outputs"]["rows"]))
    by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        if str(row.get("split") or "") == split:
            by_page[str(row.get("page_id") or "")].append(row)
    return by_page


def quant(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"count": 0, "mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.asarray(vals, dtype=np.float64)
    return {"count": int(arr.size), "mean": round(float(arr.mean()), 6), "p10": round(float(np.quantile(arr, .1)), 6), "p50": round(float(np.quantile(arr, .5)), 6), "p90": round(float(np.quantile(arr, .9)), 6)}


def nearest_context(target_box: list[float], rows: list[dict[str, Any]]) -> dict[str, Any]:
    best = []
    for row in rows:
        box = valid_box(row.get("bbox"))
        if box is None:
            continue
        iou = bbox_iou(box, target_box)
        d = dist_norm(box, target_box)
        best.append((d, -iou, row, box))
    best.sort(key=lambda x: (x[0], x[1]))
    out = {"candidate_count": len(best)}
    for k in [1, 3, 5, 10]:
        subset = best[:k]
        out[f"near{k}_count"] = len(subset)
        out[f"near{k}_labels"] = dict(Counter(str(item[2].get("label") or "") for item in subset).most_common())
        out[f"near{k}_score_max"] = max([float(item[2].get("score") or 0.0) for item in subset], default=0.0)
        out[f"near{k}_dist_min"] = round(float(subset[0][0]), 6) if subset else 0.0
        out[f"near{k}_iou_max"] = round(float(max([-item[1] for item in subset], default=0.0)), 6)
    if best:
        row = best[0][2]
        box = best[0][3]
        out.update({"nearest_candidate_id": row.get("candidate_id"), "nearest_label": row.get("label"), "nearest_score": float(row.get("score") or 0.0), "nearest_dist_norm": round(float(best[0][0]), 6), "nearest_iou": round(float(-best[0][1]), 6), "nearest_area_bucket": area_bucket_from_box(box)})
    return out


def area_bucket_from_box(box: list[float]) -> str:
    _, _, area, _ = dims(box)
    if area <= 64:
        return "tiny_le_64"
    if area <= 256:
        return "small_le_256"
    if area <= 1024:
        return "medium_le_1024"
    if area <= 4096:
        return "large_le_4096"
    return "xlarge_gt_4096"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "rows": len(rows),
        "by_area": dict(Counter(str(r.get("area_bucket") or "") for r in rows).most_common()),
        "by_nearest_label": dict(Counter(str(r.get("nearest_label") or "none") for r in rows).most_common()),
        "by_nearest_area": dict(Counter(str(r.get("nearest_area_bucket") or "none") for r in rows).most_common()),
        "numeric": {
            "target_w": quant([float(r.get("target_w") or 0.0) for r in rows]),
            "target_h": quant([float(r.get("target_h") or 0.0) for r in rows]),
            "target_area": quant([float(r.get("target_area") or 0.0) for r in rows]),
            "target_aspect": quant([float(r.get("target_aspect") or 0.0) for r in rows]),
            "candidate_count": quant([float(r.get("candidate_count") or 0.0) for r in rows]),
            "nearest_dist_norm": quant([float(r.get("nearest_dist_norm") or 0.0) for r in rows]),
            "nearest_score": quant([float(r.get("nearest_score") or 0.0) for r in rows]),
            "near10_score_max": quant([float(r.get("near10_score_max") or 0.0) for r in rows]),
        },
        "runtime_signal_counts": {
            "has_any_runtime_candidate_on_page": sum(1 for r in rows if int(r.get("candidate_count") or 0) > 0),
            "nearest_within_2_diag": sum(1 for r in rows if float(r.get("nearest_dist_norm") or 999.0) <= 2.0),
            "near10_has_stair": sum(1 for r in rows if "stair" in (r.get("near10_labels") or {})),
            "near10_score_ge_0_20": sum(1 for r in rows if float(r.get("near10_score_max") or 0.0) >= 0.20),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p030-cases", default="reports/vlm/symbol_proposal_generation_p030_cases.jsonl")
    parser.add_argument("--tile-rows", default="datasets/symbol_tile_detector_tiny_sahi_v21/locked.jsonl")
    parser.add_argument("--recovery-data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--split", default="smoke_eval")
    parser.add_argument("--output", default="reports/vlm/symbol_stair_proposal_absent_p034_audit.json")
    parser.add_argument("--cases-output", default="reports/vlm/symbol_stair_proposal_absent_p034_cases.jsonl")
    args = parser.parse_args()
    gold = load_gold(args.tile_rows)
    by_page = load_recovery_by_page(args.recovery_data, args.split)
    cases = []
    for row in load_jsonl(source_path(args.p030_cases)):
        if str(row.get("split") or "") != args.split or str(row.get("label") or "") != "stair" or str(row.get("category") or "") != "proposal_absent":
            continue
        page = str(row.get("page_id") or "")
        tid = str(row.get("target_id") or "")
        target = gold.get(page, {}).get(tid)
        if not target:
            continue
        box = target["bbox"]
        w, h, area, aspect = dims(box)
        image_size = target.get("image_size") or [0, 0]
        image_w = max(float(image_size[0] or 1), 1.0)
        image_h = max(float(image_size[1] or 1), 1.0)
        ctx = nearest_context(box, by_page.get(page, []))
        item = {**row, "target_bbox": box, "target_w": round(w, 6), "target_h": round(h, 6), "target_area": round(area, 6), "target_aspect": round(aspect, 6), "target_cx_norm": round(center(box)[0] / image_w, 6), "target_cy_norm": round(center(box)[1] / image_h, 6), **ctx}
        cases.append(item)
    report = {"version": "symbol_stair_proposal_absent_p034", "split": args.split, "inputs": {"p030_cases": args.p030_cases, "tile_rows": args.tile_rows, "recovery_data": args.recovery_data}, "summary": summarize(cases), "decision": {"runtime_signal_available": bool(cases and summarize(cases)["runtime_signal_counts"]["nearest_within_2_diag"] / max(len(cases),1) > 0.5), "recommendation": "prototype_nearest_context_stair_anchor_if_signal_available_else_new_detector_head"}, "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False, "offline_labels_used_for": ["stair_proposal_absent_audit"], "final_quality_claim_allowed": False}}
    write_json(source_path(args.output), report)
    write_jsonl(source_path(args.cases_output), cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
