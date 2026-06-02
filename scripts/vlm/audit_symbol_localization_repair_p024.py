#!/usr/bin/env python3
"""P0-24 audit: quantify localization repair opportunities for v74 misses."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

import numpy as np

from apply_symbol_detector_recall_preserving_policy_v47 import safe_float
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import bbox_iou, write_json, write_jsonl

FOCUS_LABELS = {"sink", "shower", "stair", "equipment"}
FOCUS_AREAS = {"tiny_le_64", "small_le_256"}


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) * 0.5, (box[1] + box[3]) * 0.5)


def dims(box: list[float]) -> tuple[float, float]:
    return (max(float(box[2]) - float(box[0]), 1e-6), max(float(box[3]) - float(box[1]), 1e-6))


def area(box: list[float]) -> float:
    w, h = dims(box)
    return w * h


def has_center(candidate_box: list[float], target_box: list[float]) -> bool:
    cx, cy = center(target_box)
    return float(candidate_box[0]) <= cx <= float(candidate_box[2]) and float(candidate_box[1]) <= cy <= float(candidate_box[3])


def load_gold_boxes(tile_path: str) -> dict[str, dict[str, Any]]:
    gold: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(source_path(tile_path)):
        for target in (row.get("targets") or {}).get("boxes") or []:
            tid = str(target.get("target_id") or "")
            box = target.get("page_bbox") or target.get("bbox")
            if not tid or not box:
                continue
            if tid not in gold:
                gold[tid] = {
                    "target_id": tid,
                    "target_bbox": [float(x) for x in box],
                    "label": str(target.get("label") or ""),
                    "area_bucket": str(target.get("area_bucket") or ""),
                    "image": row.get("image"),
                    "image_size": row.get("image_size"),
                }
    return gold


def index_rows(recovery_manifest_path: str) -> dict[str, dict[str, dict[str, Any]]]:
    manifest = json.loads(source_path(recovery_manifest_path).read_text(encoding="utf-8"))
    by_page: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in load_jsonl(source_path(manifest["outputs"]["rows"])):
        page_id = str(row.get("page_id") or "")
        cid = str(row.get("candidate_id") or "")
        if page_id and cid:
            by_page[page_id][cid] = row
    return by_page


def quant(vals: list[float]) -> dict[str, float]:
    if not vals:
        return {"count": 0, "mean": 0.0, "p10": 0.0, "p50": 0.0, "p90": 0.0}
    arr = np.asarray(vals, dtype=np.float64)
    return {"count": int(arr.size), "mean": round(float(arr.mean()), 6), "p10": round(float(np.quantile(arr, .1)), 6), "p50": round(float(np.quantile(arr, .5)), 6), "p90": round(float(np.quantile(arr, .9)), 6)}


def correction(candidate_box: list[float], target_box: list[float]) -> dict[str, float]:
    pcx, pcy = center(candidate_box)
    gcx, gcy = center(target_box)
    pw, ph = dims(candidate_box)
    gw, gh = dims(target_box)
    return {
        "dx_over_w": (gcx - pcx) / pw,
        "dy_over_h": (gcy - pcy) / ph,
        "log_w_ratio": float(np.log(gw / pw)),
        "log_h_ratio": float(np.log(gh / ph)),
        "input_iou": bbox_iou(candidate_box, target_box),
        "center_dist_over_diag": float(np.hypot(gcx - pcx, gcy - pcy) / max(np.hypot(pw, ph), 1e-6)),
        "pred_area_over_target": area(candidate_box) / max(area(target_box), 1e-6),
    }


def scale_about_center(box: list[float], sx: float, sy: float) -> list[float]:
    cx, cy = center(box)
    w, h = dims(box)
    nw, nh = w * sx, h * sy
    return [cx - nw * 0.5, cy - nh * 0.5, cx + nw * 0.5, cy + nh * 0.5]


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    keys = ["input_iou", "dx_over_w", "dy_over_h", "log_w_ratio", "log_h_ratio", "center_dist_over_diag", "pred_area_over_target", "best_oracle_scale_iou"]
    return {
        "rows": len(rows),
        "by_label": dict(Counter(str(r.get("label") or "") for r in rows).most_common()),
        "by_area": dict(Counter(str(r.get("area_bucket") or "") for r in rows).most_common()),
        "by_category": dict(Counter(str(r.get("category") or "") for r in rows).most_common()),
        "numeric": {k: quant([float(r.get(k) or 0.0) for r in rows]) for k in keys},
        "repair_upper_bounds": {
            "oracle_scale_iou_ge_030": sum(1 for r in rows if float(r.get("best_oracle_scale_iou") or 0.0) >= 0.30),
            "oracle_scale_iou_ge_050": sum(1 for r in rows if float(r.get("best_oracle_scale_iou") or 0.0) >= 0.50),
            "center_hit_input_iou_lt_030": sum(1 for r in rows if bool(r.get("input_center_hit")) and float(r.get("input_iou") or 0.0) < 0.30),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p023-cases", default="reports/vlm/symbol_proposal_localization_p023_cases.jsonl")
    parser.add_argument("--tile-rows", default="datasets/symbol_tile_detector_tiny_sahi_v21/locked.jsonl")
    parser.add_argument("--recovery-data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--split", default="smoke_eval")
    parser.add_argument("--output", default="reports/vlm/symbol_localization_repair_p024_audit.json")
    parser.add_argument("--dataset-output", default="datasets/symbol_localization_repair_p024/smoke_center_low_iou.jsonl")
    args = parser.parse_args()

    gold = load_gold_boxes(args.tile_rows)
    rows_by_page = index_rows(args.recovery_data)
    repair_rows: list[dict[str, Any]] = []
    missing_gold = 0
    missing_candidate = 0
    scale_grid = [0.5, 0.67, 0.8, 1.0, 1.25, 1.5, 2.0, 3.0]
    for case in load_jsonl(source_path(args.p023_cases)):
        if str(case.get("split") or "") != args.split:
            continue
        if str(case.get("category") or "") not in {"localization_low_iou_center_only", "localization_low_iou_no_center", "duplicate_or_center_conflict"}:
            continue
        target_id = str(case.get("target_id") or "")
        target = gold.get(target_id)
        if not target:
            missing_gold += 1
            continue
        page_id = str(case.get("page_id") or "")
        cid = str(case.get("best_iou_candidate_id") or case.get("best_score_candidate_id") or "")
        candidate = rows_by_page.get(page_id, {}).get(cid)
        if not candidate:
            missing_candidate += 1
            continue
        cand_box = [float(x) for x in candidate.get("bbox") or []]
        target_box = [float(x) for x in target.get("target_bbox") or []]
        if len(cand_box) != 4 or len(target_box) != 4:
            continue
        corr = correction(cand_box, target_box)
        best_scale_iou = 0.0
        best_scale = [1.0, 1.0]
        for sx in scale_grid:
            for sy in scale_grid:
                iou = bbox_iou(scale_about_center(cand_box, sx, sy), target_box)
                if iou > best_scale_iou:
                    best_scale_iou = iou
                    best_scale = [sx, sy]
        row = {
            "split": args.split,
            "page_id": page_id,
            "target_id": target_id,
            "candidate_id": cid,
            "category": case.get("category"),
            "label": target.get("label") or case.get("label"),
            "area_bucket": target.get("area_bucket") or case.get("area_bucket"),
            "candidate_bbox": cand_box,
            "target_bbox": target_box,
            "candidate_score": safe_float(candidate.get("score")),
            "candidate_label": candidate.get("label"),
            "target_label": target.get("label"),
            "input_center_hit": has_center(cand_box, target_box),
            "best_oracle_scale_iou": round(best_scale_iou, 6),
            "best_oracle_scale": best_scale,
            **{k: round(float(v), 6) for k, v in corr.items()},
        }
        repair_rows.append(row)
    focus = [r for r in repair_rows if r.get("label") in FOCUS_LABELS and r.get("area_bucket") in FOCUS_AREAS]
    report = {
        "version": "symbol_localization_repair_p024",
        "inputs": {"p023_cases": args.p023_cases, "tile_rows": args.tile_rows, "recovery_data": args.recovery_data},
        "split": args.split,
        "gold_targets_loaded": len(gold),
        "missing_gold_cases": missing_gold,
        "missing_candidate_cases": missing_candidate,
        "all_repair_candidates": summarize(repair_rows),
        "focus_tiny_small_sink_shower_stair_equipment": summarize(focus),
        "decision": {
            "primary_bottleneck": "center_hit_low_iou_localization",
            "focus_rows": len(focus),
            "oracle_scale_focus_iou_ge_030": sum(1 for r in focus if float(r.get("best_oracle_scale_iou") or 0.0) >= 0.30),
            "recommendation": "train_or_audit_bbox_refiner_on_center_hit_low_iou_cases_if_gold_bbox_supervision_is_allowed_offline",
        },
        "source_integrity": {
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "offline_labels_used_for": ["localization_repair_audit", "bbox_refiner_supervision_candidate"],
            "final_quality_claim_allowed": False,
        },
    }
    source_path(args.dataset_output).parent.mkdir(parents=True, exist_ok=True)
    write_json(source_path(args.output), report)
    write_jsonl(source_path(args.dataset_output), repair_rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
