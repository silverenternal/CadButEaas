#!/usr/bin/env python3
"""P0-27: page-level integration test for runtime-safe P0-26 localization refiner."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from typing import Any

import joblib
import numpy as np

from apply_symbol_detector_recall_preserving_policy_v47 import evaluate_selection, group_pages, page_gold_targets
from train_symbol_box_refiner_v38 import apply_delta
from train_symbol_expanded_action_source_policy_v74 import evaluate_policy, feature_names
from train_symbol_localization_repair_p026 import feature_dict, vector as p026_vector
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, write_json


def valid_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    box = [float(v) for v in value]
    return box if box[2] > box[0] and box[3] > box[1] else None


def should_refine(row: dict[str, Any], gate: str) -> bool:
    box = valid_box(row.get("bbox"))
    if box is None:
        return False
    label = str(row.get("label") or "")
    cand_area = (box[2] - box[0]) * (box[3] - box[1])
    cand_area_bucket = area_bucket(box)
    if gate == "all":
        return True
    if gate == "sink":
        return label == "sink"
    score = float(row.get("score") or row.get("expanded_action_score_v74") or 0.0)
    if gate == "sink_tiny_pred_area":
        return label == "sink" and cand_area <= 256.0
    if gate == "sink_medium_large_pred_area":
        return label == "sink" and cand_area > 256.0
    if gate == "sink_large_pred_area":
        return label == "sink" and cand_area > 1024.0
    if gate == "sink_low_score":
        return label == "sink" and score < 0.25
    if gate == "sink_medium_large_low_score":
        return label == "sink" and cand_area > 256.0 and score < 0.25
    if gate == "sink_or_tiny_pred_area":
        return label == "sink" or cand_area <= 256.0
    if gate == "focus_labels":
        return label in {"sink", "shower", "stair", "equipment"}
    if gate == "tiny_pred_area":
        return cand_area <= 256.0 or cand_area_bucket in {"tiny_le_64", "small_le_256"}
    return False


def row_for_refiner(row: dict[str, Any]) -> dict[str, Any]:
    item = {
        "candidate_bbox": [float(v) for v in row.get("bbox")],
        "candidate_score": float(row.get("score") or row.get("expanded_action_score_v74") or 0.0),
        "candidate_label": row.get("label"),
        "label": row.get("label"),
    }
    item["features"] = feature_dict(item)
    return item


def refine_selected(selected_by_page: dict[str, list[dict[str, Any]]], bundle: dict[str, Any], method: str, gate: str, clip: float) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    model = bundle["models"][method]
    names = list(bundle["feature_names"])
    routed: list[tuple[str, int, dict[str, Any]]] = []
    for page_id, rows in selected_by_page.items():
        for idx, row in enumerate(rows):
            if should_refine(row, gate):
                routed.append((page_id, idx, row))
    audit = Counter({"routed": len(routed)})
    delta_by_key: dict[tuple[str, int], list[float]] = {}
    if routed:
        x = np.asarray([p026_vector(row_for_refiner(row), names) for _page, _idx, row in routed], dtype=np.float32)
        for (page_id, idx, _row), delta in zip(routed, model.predict(x), strict=True):
            delta_by_key[(page_id, idx)] = list(delta)
    out: dict[str, list[dict[str, Any]]] = {}
    for page_id, rows in selected_by_page.items():
        new_rows = []
        for idx, row in enumerate(rows):
            item = dict(row)
            box = valid_box(item.get("bbox"))
            delta = delta_by_key.get((page_id, idx))
            if box is not None and delta is not None:
                item["original_bbox"] = box
                item["bbox"] = apply_delta(box, delta, clip)
                item["refined_by"] = f"symbol_localization_repair_p026:{method}:{gate}"
                audit["refined"] += 1
                audit[f"refined_label:{item.get('label')}"] += 1
                audit[f"refined_candidate_area:{area_bucket(box)}"] += 1
            new_rows.append(item)
        out[page_id] = new_rows
    return out, dict(audit)


def evaluate_against_gold_boxes(selected_by_page: dict[str, list[dict[str, Any]]], gold_by_page: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    totals = Counter()
    misses_by_label = Counter()
    misses_by_area = Counter()
    for page_id, gold_map in gold_by_page.items():
        selected = selected_by_page.get(page_id, [])
        used_pred: set[int] = set()
        iou_hits: set[str] = set()
        center_hits: set[str] = set()
        typed_total = 0
        typed_correct = 0
        for target_id, gold in gold_map.items():
            gold_box = valid_box(gold.get("bbox"))
            if gold_box is None:
                continue
            best_iou = 0.0
            best_idx = None
            for idx, pred in enumerate(selected):
                box = valid_box(pred.get("bbox"))
                if box is None:
                    continue
                iou = bbox_iou(box, gold_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = idx
                cx = (gold_box[0] + gold_box[2]) * 0.5
                cy = (gold_box[1] + gold_box[3]) * 0.5
                if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
                    center_hits.add(target_id)
            if best_idx is not None and best_iou >= 0.30 and best_idx not in used_pred:
                used_pred.add(best_idx)
                iou_hits.add(target_id)
                typed_total += 1
                if str(selected[best_idx].get("label") or "") == str(gold.get("label") or ""):
                    typed_correct += 1
            else:
                misses_by_label[str(gold.get("label") or "unknown")] += 1
                misses_by_area[str(gold.get("area_bucket") or "unknown")] += 1
        totals["gold"] += len(gold_map)
        totals["selected"] += len(selected)
        totals["iou_hit"] += len(iou_hits)
        totals["center_hit"] += len(center_hits & set(gold_map))
        totals["typed_total"] += typed_total
        totals["typed_correct"] += typed_correct
    precision = totals["iou_hit"] / max(totals["selected"], 1)
    recall = totals["iou_hit"] / max(totals["gold"], 1)
    f1 = 0.0 if precision + recall == 0.0 else 2 * precision * recall / (precision + recall)
    return {"pages": len(gold_by_page), "symbol_bbox_iou_0_30": {"matched": int(totals["iou_hit"]), "predicted": int(totals["selected"]), "gold": int(totals["gold"]), "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}, "symbol_bbox_center_recall": round(totals["center_hit"] / max(totals["gold"], 1), 6), "candidate_inflation": round(totals["selected"] / max(totals["gold"], 1), 6), "typed_accuracy_on_iou_matches": round(totals["typed_correct"] / max(totals["typed_total"], 1), 6), "misses_by_label": dict(misses_by_label.most_common()), "misses_by_area": dict(misses_by_area.most_common())}


def load_gold(tile_rows: str, split_pages: set[str]) -> dict[str, dict[str, dict[str, Any]]]:
    out: dict[str, dict[str, dict[str, Any]]] = {page_id: {} for page_id in split_pages}
    for row in load_jsonl(source_path(tile_rows)):
        page_id = str(row.get("row_id") or "")
        if page_id not in out:
            continue
        for gold in (row.get("targets") or {}).get("boxes") or []:
            tid = str(gold.get("target_id") or "")
            box = valid_box(gold.get("page_bbox") or gold.get("bbox"))
            if tid and box and tid not in out[page_id]:
                out[page_id][tid] = {"bbox": box, "label": str(gold.get("label") or ""), "area_bucket": str(gold.get("area_bucket") or area_bucket(box))}
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--actions", default="datasets/symbol_expanded_action_source_policy_v74/actions.jsonl")
    parser.add_argument("--recovery-data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--v74-model", default="checkpoints/symbol_expanded_action_source_policy_v74/model.joblib")
    parser.add_argument("--refiner", default="checkpoints/symbol_localization_repair_p026_runtime_safe/model.joblib")
    parser.add_argument("--tile-rows", default="datasets/symbol_tile_detector_tiny_sahi_v21/locked.jsonl")
    parser.add_argument("--split", default="smoke_eval")
    parser.add_argument("--method", default="extra_trees")
    parser.add_argument("--gates", default="sink,sink_tiny_pred_area,sink_or_tiny_pred_area,focus_labels,tiny_pred_area,all")
    parser.add_argument("--clip", type=float, default=0.9)
    parser.add_argument("--output", default="reports/vlm/symbol_localization_refiner_p027_page_eval.json")
    args = parser.parse_args()

    action_rows = load_jsonl(source_path(args.actions))
    recovery_manifest = json.loads(source_path(args.recovery_data).read_text(encoding="utf-8"))
    recovery_rows = load_jsonl(source_path(recovery_manifest["outputs"]["rows"]))
    v74_bundle = joblib.load(source_path(args.v74_model))
    v74_model = v74_bundle["model"]
    v74_names = v74_bundle.get("feature_names") or feature_names()
    baseline_metrics, selected_by_page, route = evaluate_policy(action_rows, recovery_rows, v74_model, v74_names, args.split, 0.04, 10, 8.0, False, 999, None)
    pages = group_pages(recovery_rows, args.split)
    gold_by_page = load_gold(args.tile_rows, set(pages.keys()))
    baseline_gold_box_metrics = evaluate_against_gold_boxes(selected_by_page, gold_by_page)
    refiner_bundle = joblib.load(source_path(args.refiner))
    gate_reports = []
    for gate in [part.strip() for part in args.gates.split(",") if part.strip()]:
        refined, audit = refine_selected(selected_by_page, refiner_bundle, args.method, gate, args.clip)
        metrics = evaluate_against_gold_boxes(refined, gold_by_page)
        gate_reports.append({"gate": gate, "audit": audit, "metrics": metrics, "delta_recall_vs_gold_box_baseline": round(metrics["symbol_bbox_iou_0_30"]["recall"] - baseline_gold_box_metrics["symbol_bbox_iou_0_30"]["recall"], 6), "delta_matched_vs_gold_box_baseline": metrics["symbol_bbox_iou_0_30"]["matched"] - baseline_gold_box_metrics["symbol_bbox_iou_0_30"]["matched"]})
    best = max(gate_reports, key=lambda r: (r["delta_matched_vs_gold_box_baseline"], r["metrics"]["symbol_bbox_iou_0_30"]["precision"]), default=None)
    output = {"version": "symbol_localization_refiner_p027", "split": args.split, "inputs": {"actions": args.actions, "recovery_data": args.recovery_data, "v74_model": args.v74_model, "refiner": args.refiner}, "v74_label_based_metrics": baseline_metrics, "v74_gold_box_metrics": baseline_gold_box_metrics, "v74_route": route.get("route", {}), "gate_reports": gate_reports, "decision": {"best_gate": (best or {}).get("gate"), "best_delta_matched": (best or {}).get("delta_matched_vs_gold_box_baseline"), "best_recall": (best or {}).get("metrics", {}).get("symbol_bbox_iou_0_30", {}).get("recall"), "recommendation": "freeze_refiner_gate_if_positive_page_level_gain_else_do_not_integrate"}, "source_integrity": {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False, "runtime_features": ["selected candidate bbox", "candidate score", "predicted label", "candidate area from bbox"], "offline_labels_used_for": ["page_level_evaluation"], "final_quality_claim_allowed": False}}
    write_json(source_path(args.output), output)
    print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
