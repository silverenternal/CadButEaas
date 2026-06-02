#!/usr/bin/env python3
"""Apply v45 quality-gated visual bbox refiner to page-level locked candidates."""

from __future__ import annotations

import argparse
import importlib
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from apply_symbol_box_refiner_v38 import cache_gold_maps, evaluate, predictions_from_rows
from train_symbol_support_suppression_v35 import load_jsonl
from train_symbol_tile_detector_v20 import area_bucket, rel, write_json, write_jsonl
from train_symbol_visual_box_refiner_v40 import apply_delta, features as default_features


ROOT = Path(__file__).resolve().parents[2]


def source_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def quality_score(model: Any, x: np.ndarray) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        proba = model.predict_proba(x)
        if len(proba) and proba.shape[1] > 1:
            return proba[:, 1]
    return model.predict(x).astype(np.float32)


def load_features(module_name: str | None):
    if not module_name:
        return default_features
    module = importlib.import_module(module_name)
    return module.features


def proposal_area(row: dict[str, Any]) -> str:
    box = [float(v) for v in (row.get("proposal") or {}).get("bbox") or [0, 0, 0, 0]]
    return area_bucket(box)


def group_route_stats(counter: Counter) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    for (key, routed), count in counter.items():
        row = out.setdefault(str(key), {"rows": 0, "routed": 0})
        row["rows"] += int(count)
        row["routed"] += int(count) if routed else 0
    return dict(sorted(out.items()))


def refine_crop_rows(
    crop_rows: list[dict[str, Any]],
    refiner: Any,
    quality: Any,
    feature_fn: Any,
    threshold: float,
    clip: float,
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    refined_by_id: dict[str, dict[str, Any]] = {}
    route_counts = Counter()
    route_by_label = Counter()
    route_by_area = Counter()
    if crop_rows:
        x = np.asarray([feature_fn(row) for row in crop_rows], dtype=np.float32)
        deltas = refiner.predict(x)
        scores = quality_score(quality, x)
        for row, delta, score in zip(crop_rows, deltas, scores, strict=True):
            label = str((row.get("proposal") or {}).get("label") or "")
            bucket = proposal_area(row)
            applied = float(score) >= threshold
            route_counts["crop_rows"] += 1
            route_counts["routed_crop_rows"] += int(applied)
            route_by_label[(label, applied)] += 1
            route_by_area[(bucket, applied)] += 1
            if not applied:
                continue
            box = [float(v) for v in row["proposal"]["bbox"]]
            refined_by_id[str(row["id"])] = {
                "bbox": apply_delta(box, list(delta), clip),
                "delta": [float(v) for v in delta],
                "quality_score": float(score),
            }

    stats = {
        "crop_rows": int(route_counts["crop_rows"]),
        "routed_crop_rows": int(route_counts["routed_crop_rows"]),
        "refined_candidate_count": len(refined_by_id),
        "quality_threshold": threshold,
        "by_runtime_label": group_route_stats(route_by_label),
        "by_runtime_area": group_route_stats(route_by_area),
    }
    return refined_by_id, stats


def compact_prediction_rows(rows: list[dict[str, Any]], refined_by_id: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        candidate_id = str(row["candidate_id"])
        refined = refined_by_id.get(candidate_id)
        if not refined:
            continue
        out.append(
            {
                "candidate_id": candidate_id,
                "page_id": str(row["page_id"]),
                "bbox": refined["bbox"],
                "delta": refined["delta"],
                "quality_score": refined["quality_score"],
                "label": row.get("label"),
                "proposal_source": row.get("proposal_source"),
            }
        )
    return out


def parse_thresholds(spec: str) -> list[float]:
    out = sorted({float(part.strip()) for part in (spec or "").split(",") if part.strip()})
    return out


def compare_eval_metrics(baseline: dict[str, Any], current: dict[str, Any], baseline_source: str, current_source: str) -> dict[str, Any]:
    metric_paths = [
        ("symbol_bbox_iou_0_30", "matched"),
        ("symbol_bbox_iou_0_30", "predicted"),
        ("symbol_bbox_iou_0_30", "gold"),
        ("symbol_bbox_iou_0_30", "precision"),
        ("symbol_bbox_iou_0_30", "recall"),
        ("symbol_bbox_iou_0_30", "f1"),
        ("candidate_inflation", None),
        ("typed_accuracy_on_iou_matches", None),
    ]
    area_keys = sorted(set(baseline.get("area_iou_recall", {})) | set(current.get("area_iou_recall", {})))
    type_keys = sorted(set(baseline.get("type_iou_recall", {})) | set(current.get("type_iou_recall", {})))
    summary: dict[str, Any] = {
        "baseline_source": baseline_source,
        "current_source": current_source,
        "baseline": {
            "matched": int(baseline["symbol_bbox_iou_0_30"]["matched"]),
            "precision": float(baseline["symbol_bbox_iou_0_30"]["precision"]),
            "recall": float(baseline["symbol_bbox_iou_0_30"]["recall"]),
            "f1": float(baseline["symbol_bbox_iou_0_30"]["f1"]),
            "candidate_inflation": float(baseline["candidate_inflation"]),
            "typed_accuracy_on_iou_matches": float(baseline.get("typed_accuracy_on_iou_matches", 0.0)),
        },
        "current": {
            "matched": int(current["symbol_bbox_iou_0_30"]["matched"]),
            "precision": float(current["symbol_bbox_iou_0_30"]["precision"]),
            "recall": float(current["symbol_bbox_iou_0_30"]["recall"]),
            "f1": float(current["symbol_bbox_iou_0_30"]["f1"]),
            "candidate_inflation": float(current["candidate_inflation"]),
            "typed_accuracy_on_iou_matches": float(current.get("typed_accuracy_on_iou_matches", 0.0)),
        },
        "delta": {
            "matched": int(current["symbol_bbox_iou_0_30"]["matched"]) - int(baseline["symbol_bbox_iou_0_30"]["matched"]),
            "precision": round(float(current["symbol_bbox_iou_0_30"]["precision"]) - float(baseline["symbol_bbox_iou_0_30"]["precision"]), 6),
            "recall": round(float(current["symbol_bbox_iou_0_30"]["recall"]) - float(baseline["symbol_bbox_iou_0_30"]["recall"]), 6),
            "f1": round(float(current["symbol_bbox_iou_0_30"]["f1"]) - float(baseline["symbol_bbox_iou_0_30"]["f1"]), 6),
            "candidate_inflation": round(float(current["candidate_inflation"]) - float(baseline["candidate_inflation"]), 6),
            "typed_accuracy_on_iou_matches": round(float(current.get("typed_accuracy_on_iou_matches", 0.0)) - float(baseline.get("typed_accuracy_on_iou_matches", 0.0)), 6),
        },
        "area_iou_recall_delta": {key: round(float(current.get("area_iou_recall", {}).get(key, 0.0)) - float(baseline.get("area_iou_recall", {}).get(key, 0.0)), 6) for key in area_keys},
        "type_iou_recall_delta": {key: round(float(current.get("type_iou_recall", {}).get(key, 0.0)) - float(baseline.get("type_iou_recall", {}).get(key, 0.0)), 6) for key in type_keys},
    }
    summary["delta"]["small_iou_recall"] = round(float(current.get("area_iou_recall", {}).get("small_le_256", 0.0)) - float(baseline.get("area_iou_recall", {}).get("small_le_256", 0.0)), 6)
    summary["delta"]["tiny_iou_recall"] = round(float(current.get("area_iou_recall", {}).get("tiny_le_64", 0.0)) - float(baseline.get("area_iou_recall", {}).get("tiny_le_64", 0.0)), 6)
    summary["delta"]["sink_iou_recall"] = round(float(current.get("type_iou_recall", {}).get("sink", 0.0)) - float(baseline.get("type_iou_recall", {}).get("sink", 0.0)), 6)
    summary["adopt"] = current["symbol_bbox_iou_0_30"]["recall"] >= baseline["symbol_bbox_iou_0_30"]["recall"] and current["symbol_bbox_iou_0_30"]["precision"] >= baseline["symbol_bbox_iou_0_30"]["precision"] * 0.95
    return summary


def page_coverage(rows: list[dict[str, Any]], refined_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pages = {str(row["page_id"]) for row in rows}
    affected = {str(row["page_id"]) for row in rows if str(row["candidate_id"]) in refined_by_id}
    return {
        "page_count": len(pages),
        "affected_page_count": len(affected),
        "candidate_count": len(rows),
        "refined_candidate_count": len(refined_by_id),
        "refined_candidate_fraction": round(len(refined_by_id) / max(len(rows), 1), 6),
        "affected_page_fraction": round(len(affected) / max(len(pages), 1), 6),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--crop-records", default="datasets/symbol_visual_box_refiner_v44_fulltarget/locked.jsonl")
    parser.add_argument("--rows", default="datasets/symbol_support_suppression_v36/locked_rows.jsonl")
    parser.add_argument("--cache", default="datasets/symbol_support_suppression_v36/locked_cache.jsonl")
    parser.add_argument("--model", default="checkpoints/symbol_visual_box_refiner_v45_quality_policy/model.joblib")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_visual_box_refiner_v45_quality_policy_page_locked_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/symbol_visual_box_refiner_v45_quality_policy_page_locked_predictions.jsonl")
    parser.add_argument("--changed-predictions-output", default=None)
    parser.add_argument("--baseline-report", default=None)
    parser.add_argument("--quality-threshold", type=float, default=None)
    parser.add_argument("--threshold-sweep", default="0.15,0.25,0.35,0.45,0.55")
    parser.add_argument("--feature-module", default=None)
    parser.add_argument("--clip", type=float, default=0.75)
    parser.add_argument("--write-full-predictions", action="store_true", default=False)
    return parser.parse_args()


def evaluate_threshold_sweep(
    crop_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    gold_by_page: dict[str, dict[str, dict[str, Any]]],
    refiner: Any,
    quality: Any,
    feature_fn: Any,
    thresholds: list[float],
    clip: float,
) -> list[dict[str, Any]]:
    if not crop_rows or not thresholds:
        return []
    x = np.asarray([feature_fn(row) for row in crop_rows], dtype=np.float32)
    deltas = refiner.predict(x)
    scores = quality_score(quality, x)
    sweep: list[dict[str, Any]] = []
    for threshold in thresholds:
        refined_by_id: dict[str, dict[str, Any]] = {}
        for row, delta, score in zip(crop_rows, deltas, scores, strict=True):
            if float(score) < threshold:
                continue
            box = [float(v) for v in row["proposal"]["bbox"]]
            refined_by_id[str(row["id"])] = {
                "bbox": apply_delta(box, list(delta), clip),
                "delta": [float(v) for v in delta],
                "quality_score": float(score),
            }
        after_predictions = predictions_from_rows(rows, refined_by_id)
        after = evaluate(after_predictions, gold_by_page)
        sweep.append(
            {
                "threshold": round(float(threshold), 6),
                "refined_candidate_count": len(refined_by_id),
                "routing_fraction": round(len(refined_by_id) / max(len(crop_rows), 1), 6),
                "after": after,
            }
        )
    return sweep


def main() -> None:
    args = parse_args()
    crop_rows = load_jsonl(source_path(args.crop_records))
    rows = load_jsonl(source_path(args.rows))
    cache_rows = load_jsonl(source_path(args.cache))
    bundle = joblib.load(source_path(args.model))
    threshold = float(args.quality_threshold if args.quality_threshold is not None else bundle.get("quality_threshold", 0.35))
    feature_module = args.feature_module or bundle.get("feature_module")
    feature_fn = load_features(feature_module)
    refined_by_id, routing = refine_crop_rows(crop_rows, bundle["refiner"], bundle["quality_model"], feature_fn, threshold, args.clip)
    before_predictions = predictions_from_rows(rows, {})
    after_predictions = predictions_from_rows(rows, refined_by_id)
    gold_by_page = cache_gold_maps(cache_rows)
    before = evaluate(before_predictions, gold_by_page)
    after = evaluate(after_predictions, gold_by_page)
    coverage = page_coverage(rows, refined_by_id)
    baseline_report = json.loads(source_path(args.baseline_report).read_text(encoding="utf-8")) if args.baseline_report else None
    baseline_after = (baseline_report or {}).get("after") or before
    baseline_source = (baseline_report or {}).get("version") or "before"
    comparison = compare_eval_metrics(baseline_after, after, baseline_source, "current_after")
    sweep_thresholds = parse_thresholds(args.threshold_sweep)
    threshold_sweep = evaluate_threshold_sweep(crop_rows, rows, gold_by_page, bundle["refiner"], bundle["quality_model"], feature_fn, sweep_thresholds, args.clip)
    sweep_best = {
        "by_recall": max(threshold_sweep, key=lambda row: (row["after"]["symbol_bbox_iou_0_30"]["recall"], row["after"]["symbol_bbox_iou_0_30"]["precision"], -row["after"]["candidate_inflation"]), default=None),
        "by_precision": max(threshold_sweep, key=lambda row: (row["after"]["symbol_bbox_iou_0_30"]["precision"], row["after"]["symbol_bbox_iou_0_30"]["recall"], -row["after"]["candidate_inflation"]), default=None),
        "by_f1": max(threshold_sweep, key=lambda row: (row["after"]["symbol_bbox_iou_0_30"]["f1"], row["after"]["symbol_bbox_iou_0_30"]["recall"], -row["after"]["candidate_inflation"]), default=None),
    }
    predictions_payload = predictions_from_rows(rows, refined_by_id) if args.write_full_predictions else compact_prediction_rows(rows, refined_by_id)
    write_jsonl(source_path(args.predictions_output), predictions_payload)
    changed_output = None
    if args.changed_predictions_output:
        changed_output = source_path(args.changed_predictions_output)
        write_jsonl(changed_output, compact_prediction_rows(rows, refined_by_id))
    report = {
        "version": "symbol_visual_box_refiner_v45_quality_policy_page_locked_eval",
        "task": "P1-17-box-quality-policy-v45",
        "claim_boundary": "Apply v45 quality-gated visual bbox refiner to v44 covered locked page candidates. Runtime routing uses model quality score from crop pixels/proposal fields only.",
        "source_integrity": {
            "model_input": "raster crop pixels plus proposal bbox/score/type",
            "routing_input": "model quality score only",
            "offline_labels_used_for": ["training", "locked_evaluation"],
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
        },
        "inputs": {"crop_records": rel(source_path(args.crop_records)), "rows": rel(source_path(args.rows)), "cache": rel(source_path(args.cache)), "model": rel(source_path(args.model))},
        "outputs": {
            "predictions": rel(source_path(args.predictions_output)),
            "predictions_format": "full_page_predictions" if args.write_full_predictions else "compact_changed_candidates",
            **({"changed_predictions": rel(changed_output)} if changed_output else {}),
        },
        "routing": routing,
        "feature_module": feature_module or "train_symbol_visual_box_refiner_v40",
        "coverage": coverage,
        "before": before,
        "after": after,
        "comparison": comparison,
        "threshold_sweep": threshold_sweep,
        "threshold_sweep_summary": {
            "requested_thresholds": sweep_thresholds,
            "best_by_recall": sweep_best["by_recall"],
            "best_by_precision": sweep_best["by_precision"],
            "best_by_f1": sweep_best["by_f1"],
        },
        "stage_gate": {
            "page_locked_iou_recall_improves_over_baseline": after["symbol_bbox_iou_0_30"]["recall"] > baseline_after["symbol_bbox_iou_0_30"]["recall"],
            "page_locked_matched_increase_over_baseline_min_20": after["symbol_bbox_iou_0_30"]["matched"] >= baseline_after["symbol_bbox_iou_0_30"]["matched"] + 20,
            "page_locked_tiny_iou_recall_not_drop_over_baseline": after["area_iou_recall"].get("tiny_le_64", 0.0) >= baseline_after["area_iou_recall"].get("tiny_le_64", 0.0),
            "no_oracle_inference": True,
        },
    }
    report["stage_gate"]["passed"] = all(report["stage_gate"].values())
    write_json(source_path(args.eval_output), report)
    print(json.dumps({"routing": routing, "coverage": coverage, "comparison": comparison, "before": before["symbol_bbox_iou_0_30"], "after": after["symbol_bbox_iou_0_30"], "tiny": {"before": before["area_iou_recall"].get("tiny_le_64", 0.0), "after": after["area_iou_recall"].get("tiny_le_64", 0.0)}, "small": {"before": before["area_iou_recall"].get("small_le_256", 0.0), "after": after["area_iou_recall"].get("small_le_256", 0.0)}, "stage_gate": report["stage_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
