#!/usr/bin/env python3
"""Evaluate the v35 source-aware suppression policy on a locked cache."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib

from build_symbol_proposal_selector_features_v30 import load_preds
from build_symbol_support_suppression_v35 import build_rows
from cache_symbol_proposal_eval_v35 import candidate_gold_matches, merge_sources
from train_symbol_support_suppression_v35 import evaluate, load_jsonl
from train_symbol_tile_detector_v20 import area_bucket, rel, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]


def source_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def selected_tile_ids(*pred_maps: dict[str, list[dict[str, Any]]]) -> set[str]:
    ids: set[str] = set()
    for pred_map in pred_maps:
        for preds in pred_map.values():
            for pred in preds:
                tile_id = str(pred.get("tile_id") or "")
                if tile_id:
                    ids.add(tile_id)
    return ids


def load_locked_golds(path: Path, row_ids: set[str], tile_ids: set[str]) -> dict[str, dict[str, dict[str, Any]]]:
    pages: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in load_jsonl(path):
        row_id = str(row.get("row_id") or "")
        if row_id not in row_ids:
            continue
        if tile_ids and str(row.get("id") or "") not in tile_ids:
            continue
        for gold in ((row.get("targets") or {}).get("boxes") or []):
            target_id = str(gold.get("target_id") or f"{row_id}_symbol_{len(pages[row_id])}")
            box = [float(v) for v in gold.get("page_bbox") or gold.get("bbox") or []]
            if len(box) != 4 or box[2] <= box[0] or box[3] <= box[1]:
                continue
            pages[row_id][target_id] = {
                "target_id": target_id,
                "bbox": box,
                "label": str(gold.get("label") or "generic_symbol"),
                "area_bucket": str(gold.get("area_bucket") or area_bucket(box)),
            }
    return pages


def build_cache(mask_predictions: Path, tiny_predictions: Path, locked_tiles: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    mask = load_preds(mask_predictions) if mask_predictions.exists() else {}
    tiny = load_preds(tiny_predictions)
    merged = merge_sources(("mask_v28", mask), ("pretrained_tiny_v35", tiny))
    row_ids = set(merged)
    tile_ids = selected_tile_ids(mask, tiny)
    golds = load_locked_golds(locked_tiles, row_ids, tile_ids)
    rows: list[dict[str, Any]] = []
    counts = Counter()
    for row_id, preds in sorted(merged.items()):
        gold_map = golds.get(row_id, {})
        gold_rows = [
            {
                "target_id": gold["target_id"],
                "bbox": gold["bbox"],
                "label": gold["label"],
                "area_bucket": gold["area_bucket"],
            }
            for gold in gold_map.values()
        ]
        rows.append(
            {
                "row_id": row_id,
                "predicted_symbols": preds,
                "gold_symbols": gold_rows,
                "candidate_gold_matches": candidate_gold_matches(preds, gold_map),
            }
        )
        counts["pages"] += 1
        counts["candidates"] += len(preds)
        counts["golds"] += len(gold_rows)
        for pred in preds:
            counts[f"source:{pred.get('proposal_source') or 'unknown'}"] += 1
            if pred.get("tile_id"):
                counts["predicted_tile_refs"] += 1
        for gold in gold_rows:
            counts[f"gold_label:{gold['label']}"] += 1
            counts[f"gold_area:{gold['area_bucket']}"] += 1
    counts["selected_tile_ids"] = len(tile_ids)
    return rows, dict(counts)


def hard_cases_from_rows(rows: list[dict[str, Any]], predictions: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    pred_by_page = {str(row["page_id"]): row.get("predicted_symbols") or [] for row in predictions}
    cases: list[dict[str, Any]] = []
    for cache_row in rows:
        page_id = str(cache_row["row_id"])
        selected = pred_by_page.get(page_id, [])
        selected_boxes = [item.get("bbox") for item in selected]
        selected_ids = {str(item.get("candidate_id") or "") for item in selected}
        candidate_rows, _counts = build_rows([cache_row])
        selected_rows = [row for row in candidate_rows if row["candidate_id"] in selected_ids]
        hit_targets = set()
        center_only = []
        false_positive = []
        for row in selected_rows:
            labels = row.get("labels") or {}
            best_iou = float(labels.get("best_iou", 0.0) or 0.0)
            target = labels.get("best_iou_target_id")
            if target and best_iou >= 0.30:
                hit_targets.add(str(target))
            elif labels.get("center_target_ids"):
                center_only.append(row)
            else:
                false_positive.append(row)
        for gold in cache_row.get("gold_symbols") or []:
            target = str(gold.get("target_id") or "")
            if target and target not in hit_targets:
                cases.append(
                    {
                        "page_id": page_id,
                        "case_type": "missed_iou_0_30",
                        "target_id": target,
                        "label": gold.get("label"),
                        "area_bucket": gold.get("area_bucket"),
                        "gold_bbox": gold.get("bbox"),
                        "selected_count_on_page": len(selected_boxes),
                    }
                )
        for row in center_only[:10]:
            cases.append(
                {
                    "page_id": page_id,
                    "case_type": "center_only_no_iou_selected",
                    "candidate_id": row.get("candidate_id"),
                    "label": row.get("label"),
                    "proposal_source": row.get("proposal_source"),
                    "bbox": row.get("bbox"),
                    "best_iou": (row.get("labels") or {}).get("best_iou"),
                    "center_target_ids": (row.get("labels") or {}).get("center_target_ids"),
                }
            )
        for row in false_positive[:10]:
            cases.append(
                {
                    "page_id": page_id,
                    "case_type": "selected_false_positive",
                    "candidate_id": row.get("candidate_id"),
                    "label": row.get("label"),
                    "proposal_source": row.get("proposal_source"),
                    "bbox": row.get("bbox"),
                    "suppression_reason": (row.get("labels") or {}).get("suppression_reason"),
                }
            )
    return cases[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mask-predictions", default="reports/vlm/symbol_yolov8s_seg_rect_v28_page_predictions.jsonl")
    parser.add_argument("--tiny-predictions", default="reports/vlm/symbol_pretrained_tiny_detector_v35_locked_predictions.jsonl")
    parser.add_argument("--locked-tiles", default="datasets/symbol_tile_detector_tiny_sahi_v21/locked.jsonl")
    parser.add_argument("--model", default="checkpoints/symbol_support_suppression_v35_precision/model.joblib")
    parser.add_argument("--threshold", type=float, default=0.018)
    parser.add_argument("--cluster-topk", type=int, default=0)
    parser.add_argument("--max-per-page", type=int, default=120)
    parser.add_argument("--cache-output", default="reports/vlm/symbol_proposal_eval_v35_locked_cache.jsonl")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_support_suppression_v35_locked_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/symbol_support_suppression_v35_locked_predictions.jsonl")
    parser.add_argument("--hard-cases-output", default="reports/vlm/symbol_support_suppression_v35_locked_hard_cases.jsonl")
    parser.add_argument("--max-hard-cases", type=int, default=1000)
    args = parser.parse_args()

    cache_rows, cache_counts = build_cache(source_path(args.mask_predictions), source_path(args.tiny_predictions), source_path(args.locked_tiles))
    write_jsonl(source_path(args.cache_output), cache_rows)
    listwise_rows, listwise_counts = build_rows(cache_rows)
    for row in listwise_rows:
        row["split"] = "locked"
    bundle = joblib.load(source_path(args.model))
    model = bundle["model"]
    names = bundle["feature_names"]
    metrics, predictions = evaluate(model, listwise_rows, names, "locked", args.threshold, args.cluster_topk, args.max_per_page)
    hard_cases = hard_cases_from_rows(cache_rows, predictions, args.max_hard_cases)
    by_case = Counter(str(row.get("case_type") or "unknown") for row in hard_cases)
    report = {
        "version": "symbol_support_suppression_v35_locked_eval",
        "task": "P1-05-locked-validation-and-box-quality-residuals-v35",
        "claim_boundary": "Locked subset evaluation using existing v28 predictions where available plus v35 locked predictions. The fixed P1-04 smoke policy is applied without locked threshold tuning.",
        "source_integrity": {
            "model_input": "raster-derived candidate bbox/score/source/type fields only",
            "offline_labels_used_for": ["locked_evaluation", "hard_case_audit"],
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
        },
        "inputs": {
            "mask_predictions": rel(source_path(args.mask_predictions)),
            "tiny_predictions": rel(source_path(args.tiny_predictions)),
            "locked_tiles": rel(source_path(args.locked_tiles)),
            "model": rel(source_path(args.model)),
        },
        "outputs": {
            "cache": rel(source_path(args.cache_output)),
            "predictions": rel(source_path(args.predictions_output)),
            "hard_cases": rel(source_path(args.hard_cases_output)),
        },
        "policy": {"threshold": args.threshold, "cluster_topk": args.cluster_topk, "max_per_page": args.max_per_page},
        "cache_counts": cache_counts,
        "listwise_counts": listwise_counts,
        "locked": metrics,
        "hard_case_counts": dict(by_case),
        "stage_gate": {
            "locked_center_recall_min_0_94": metrics["symbol_bbox_center_recall"] >= 0.94,
            "locked_iou_0_30_recall_min_0_72": metrics["symbol_bbox_iou_0_30"]["recall"] >= 0.72,
            "locked_precision_min_0_12": metrics["symbol_bbox_iou_0_30"]["precision"] >= 0.12,
            "locked_candidate_inflation_max_7": metrics["candidate_inflation"] <= 7.0,
            "no_oracle_inference": True,
        },
    }
    report["stage_gate"]["passed"] = all(report["stage_gate"].values())
    write_json(source_path(args.eval_output), report)
    write_jsonl(source_path(args.predictions_output), predictions)
    write_jsonl(source_path(args.hard_cases_output), hard_cases)
    print(json.dumps({"locked": metrics, "stage_gate": report["stage_gate"], "hard_case_counts": report["hard_case_counts"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
