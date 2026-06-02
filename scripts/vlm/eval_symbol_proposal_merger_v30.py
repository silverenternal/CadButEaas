#!/usr/bin/env python3
"""Evaluate a v30 proposal merger over mask and center-branch proposals."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from train_symbol_center_heatmap_probe_v24 import merge_predictions, score_predictions
from train_symbol_tile_detector_v20 import bbox_iou, load_jsonl, rel, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]


def load_preds(path: Path) -> dict[str, list[dict[str, Any]]]:
    return {str(row.get("row_id")): list(row.get("predicted_symbols") or []) for row in load_jsonl(path)}


def build_golds_from_center_targets(path: Path) -> dict[str, dict[str, dict[str, Any]]]:
    golds: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in load_jsonl(path):
        row_id = str(row.get("row_id"))
        target_id = str(row.get("target_id") or f"{row_id}_{len(golds[row_id])}")
        golds[row_id][target_id] = {
            "target_id": target_id,
            "bbox": [float(v) for v in row.get("page_bbox") or []],
            "label": str(row.get("label") or "generic_symbol"),
        }
    return golds


def rel_from_manifest(manifest_path: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def scale_and_tag(preds: list[dict[str, Any]], source: str, score_offset: float, score_scale: float) -> list[dict[str, Any]]:
    out = []
    for pred in preds:
        item = dict(pred)
        item["original_score"] = float(item.get("score", 0.0))
        item["score"] = score_offset + float(item.get("score", 0.0)) * score_scale
        item["proposal_source"] = item.get("proposal_source") or source
        out.append(item)
    return out


def merge_sources(
    mask: dict[str, list[dict[str, Any]]],
    center: dict[str, list[dict[str, Any]]],
    mask_score_offset: float,
    center_score_scale: float,
    center_min_score: float,
) -> dict[str, list[dict[str, Any]]]:
    keys = set(mask) | set(center)
    out: dict[str, list[dict[str, Any]]] = {}
    for key in keys:
        mask_items = scale_and_tag(mask.get(key, []), "mask_v28", mask_score_offset, 1.0)
        center_items = [
            item
            for item in scale_and_tag(center.get(key, []), "center_branch_v30", 0.0, center_score_scale)
            if float(item.get("original_score", item.get("score", 0.0))) >= center_min_score
        ]
        out[key] = mask_items + center_items
    return out


def source_counts(predictions: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter()
    for row in predictions:
        for pred in row.get("predicted_symbols") or []:
            counts[str(pred.get("proposal_source") or "unknown")] += 1
    return dict(counts)


def metric_summary(row: dict[str, Any]) -> dict[str, float]:
    metrics = row["metrics"]
    return {
        "center_recall": float(metrics["symbol_bbox_center_recall"]),
        "iou_0_30_recall": float(metrics["symbol_bbox_iou_0_30"]["recall"]),
        "precision": float(metrics["symbol_bbox_iou_0_30"]["precision"]),
        "f1": float(metrics["symbol_bbox_iou_0_30"]["f1"]),
        "candidate_inflation": float(metrics["candidate_inflation"]),
    }


def choose_views(grid: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    recall_first = sorted(
        grid,
        key=lambda row: (
            metric_summary(row)["center_recall"],
            metric_summary(row)["iou_0_30_recall"],
            metric_summary(row)["precision"],
            -metric_summary(row)["candidate_inflation"],
        ),
        reverse=True,
    )[0]
    balanced = sorted(
        grid,
        key=lambda row: (
            metric_summary(row)["candidate_inflation"] <= 7.0,
            metric_summary(row)["precision"] >= 0.12,
            metric_summary(row)["f1"],
            metric_summary(row)["iou_0_30_recall"],
            -metric_summary(row)["candidate_inflation"],
        ),
        reverse=True,
    )[0]
    compression_first = sorted(
        grid,
        key=lambda row: (
            metric_summary(row)["precision"],
            -metric_summary(row)["candidate_inflation"],
            metric_summary(row)["iou_0_30_recall"],
        ),
        reverse=True,
    )[0]
    return {
        "recall_first": recall_first,
        "balanced_compression": balanced,
        "precision_first": compression_first,
    }


def center_distance(left: list[float], right: list[float]) -> float:
    lcx = (left[0] + left[2]) / 2.0
    lcy = (left[1] + left[3]) / 2.0
    rcx = (right[0] + right[2]) / 2.0
    rcy = (right[1] + right[3]) / 2.0
    return ((lcx - rcx) ** 2 + (lcy - rcy) ** 2) ** 0.5


def candidate_features(index: int, preds: list[dict[str, Any]]) -> dict[str, float]:
    pred = preds[index]
    box = [float(v) for v in pred.get("bbox") or [0, 0, 0, 0]]
    width = max(0.0, box[2] - box[0])
    height = max(0.0, box[3] - box[1])
    max_iou = 0.0
    overlap_count = 0
    near_center_count = 0
    for other_index, other in enumerate(preds):
        if other_index == index:
            continue
        other_box = [float(v) for v in other.get("bbox") or [0, 0, 0, 0]]
        iou = bbox_iou(box, other_box)
        max_iou = max(max_iou, iou)
        if iou >= 0.30:
            overlap_count += 1
        if center_distance(box, other_box) <= 12.0:
            near_center_count += 1
    source = str(pred.get("proposal_source") or "unknown")
    return {
        "area": width * height,
        "aspect": width / max(height, 1e-6),
        "height": height,
        "is_center_branch_v30": float(source == "center_branch_v30"),
        "is_mask_v28": float(source == "mask_v28"),
        "label_id": float(pred.get("label_id") or 5),
        "max_peer_iou": max_iou,
        "near_center_peer_count": float(near_center_count),
        "overlap_peer_count": float(overlap_count),
        "score": float(pred.get("original_score", pred.get("score", 0.0))),
        "width": width,
    }


def apply_selector(page_preds: dict[str, list[dict[str, Any]]], selector_path: Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    bundle = joblib.load(selector_path)
    model = bundle["model"]
    feature_names = list(bundle["feature_names"])
    out: dict[str, list[dict[str, Any]]] = {}
    for row_id, preds in page_preds.items():
        if not preds:
            out[row_id] = []
            continue
        matrix = np.asarray([[candidate_features(index, preds).get(name, 0.0) for name in feature_names] for index in range(len(preds))], dtype=np.float32)
        probs = model.predict_proba(matrix)[:, 1]
        rescored = []
        for pred, prob in zip(preds, probs.tolist(), strict=True):
            item = dict(pred)
            item["pre_selector_score"] = float(item.get("score", 0.0))
            item["selector_score"] = float(prob)
            item["score"] = float(prob)
            rescored.append(item)
        out[row_id] = rescored
    return out, {"selector": rel(selector_path), "selected_threshold": float(bundle.get("selected_threshold", 0.1)), "feature_names": feature_names}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/symbol_center_branch_v30/manifest.json")
    parser.add_argument("--center-predictions", default="reports/vlm/symbol_center_branch_v30_smoke_predictions.jsonl")
    parser.add_argument("--mask-predictions", default="reports/vlm/symbol_yolov8s_seg_rect_v28_smoke_v30_page_predictions.jsonl")
    parser.add_argument("--split", default="smoke_v30")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_proposal_merger_v30_smoke_page_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/symbol_proposal_merger_v30_smoke_page_predictions.jsonl")
    parser.add_argument("--audit-output", default="reports/vlm/symbol_proposal_merger_v30_smoke_error_audit.json")
    parser.add_argument("--policy-output", default="checkpoints/symbol_proposal_merger_v30/model.joblib")
    parser.add_argument("--selector", default="")
    parser.add_argument("--selector-threshold-grid", default="")
    parser.add_argument("--mask-score-offset-grid", default="1.0")
    parser.add_argument("--center-score-scale-grid", default="0.05,0.1,0.2")
    parser.add_argument("--center-min-score-grid", default="0.05,0.1,0.2,0.4")
    parser.add_argument("--score-threshold-grid", default="0.001,0.005,0.01")
    parser.add_argument("--nms-threshold-grid", default="0.35,0.45,0.55,0.65")
    parser.add_argument("--max-per-page", type=int, default=1200)
    args = parser.parse_args()

    manifest_path = Path(args.dataset)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if args.split != "smoke_v30":
        raise SystemExit("v30 merger currently supports smoke_v30 because locked center targets are intentionally not materialized yet.")
    golds = build_golds_from_center_targets(rel_from_manifest(manifest_path, manifest["outputs"]["smoke_center_targets"]))
    mask = load_preds(Path(args.mask_predictions))
    center = load_preds(Path(args.center_predictions))

    grid = []
    selector_info: dict[str, Any] | None = None
    if args.selector:
        merged_sources = merge_sources(mask, center, 0.0, 1.0, 0.0)
        merged_sources, selector_info = apply_selector(merged_sources, Path(args.selector))
        thresholds = [float(v) for v in (args.selector_threshold_grid or args.score_threshold_grid).split(",") if v.strip()]
        for score_threshold in thresholds:
            for nms_threshold in [float(v) for v in args.nms_threshold_grid.split(",") if v.strip()]:
                metrics, predictions, errors = score_predictions(merged_sources, golds, score_threshold, nms_threshold, args.max_per_page, 200)
                grid.append(
                    {
                        "policy": {
                            "selector": rel(Path(args.selector)),
                            "selector_score_threshold": score_threshold,
                            "nms_threshold": nms_threshold,
                            "max_per_page": args.max_per_page,
                        },
                        "metrics": metrics,
                        "source_counts": source_counts(predictions),
                        "error_count": len(errors),
                        "predictions": predictions,
                        "errors": errors,
                    }
                )
    else:
        for mask_offset in [float(v) for v in args.mask_score_offset_grid.split(",") if v.strip()]:
            for center_scale in [float(v) for v in args.center_score_scale_grid.split(",") if v.strip()]:
                for center_min in [float(v) for v in args.center_min_score_grid.split(",") if v.strip()]:
                    merged_sources = merge_sources(mask, center, mask_offset, center_scale, center_min)
                    for score_threshold in [float(v) for v in args.score_threshold_grid.split(",") if v.strip()]:
                        for nms_threshold in [float(v) for v in args.nms_threshold_grid.split(",") if v.strip()]:
                            metrics, predictions, errors = score_predictions(merged_sources, golds, score_threshold, nms_threshold, args.max_per_page, 200)
                            grid.append(
                                {
                                    "policy": {
                                        "mask_score_offset": mask_offset,
                                        "center_score_scale": center_scale,
                                        "center_min_score": center_min,
                                        "score_threshold": score_threshold,
                                        "nms_threshold": nms_threshold,
                                        "max_per_page": args.max_per_page,
                                    },
                                    "metrics": metrics,
                                    "source_counts": source_counts(predictions),
                                    "error_count": len(errors),
                                    "predictions": predictions,
                                    "errors": errors,
                                }
                            )
    selected_views = choose_views(grid)
    selected = selected_views["balanced_compression"] if args.selector else selected_views["recall_first"]
    report = {
        "version": "symbol_proposal_merger_v30_smoke_eval",
        "metric_mode": "smoke",
        "claim_boundary": "Offline proposal-source merger evaluation. Center branch is class-agnostic support; v28 mask proposals remain tight-box source.",
        "dataset": rel(manifest_path),
        "inputs": {"mask_predictions": rel(Path(args.mask_predictions)), "center_predictions": rel(Path(args.center_predictions))},
        "selector_info": selector_info,
        "selected_policy": selected["policy"],
        "selected_metrics": selected["metrics"],
        "selected_source_counts": selected["source_counts"],
        "selection_views": {
            name: {
                "policy": row["policy"],
                "metrics": row["metrics"],
                "source_counts": row["source_counts"],
            }
            for name, row in selected_views.items()
        },
        "threshold_grid": [{k: v for k, v in row.items() if k not in {"predictions", "errors"}} for row in grid],
        "stage_gate": {
            "center_recall_min_0_94": selected["metrics"]["symbol_bbox_center_recall"] >= 0.94,
            "iou_0_30_recall_min_0_82": selected["metrics"]["symbol_bbox_iou_0_30"]["recall"] >= 0.82,
            "precision_min_0_12": selected["metrics"]["symbol_bbox_iou_0_30"]["precision"] >= 0.12,
            "candidate_inflation_max_7": selected["metrics"]["candidate_inflation"] <= 7.0,
        },
    }
    report["stage_gate"]["passed"] = all(report["stage_gate"].values())
    write_json(Path(args.eval_output), report)
    write_jsonl(Path(args.predictions_output), selected["predictions"])
    write_json(Path(args.audit_output), {"errors": selected["errors"][:2000], "source_counts": selected["source_counts"]})
    write_json(Path(args.policy_output), {"format": "json_policy_saved_with_joblib_extension", "policy": selected["policy"], "metrics": selected["metrics"]})
    print(json.dumps({"selected_metrics": selected["metrics"], "source_counts": selected["source_counts"], "stage_gate": report["stage_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
