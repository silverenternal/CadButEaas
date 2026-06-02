#!/usr/bin/env python3
"""Audit a segmentation/graph-style boundary proposal stream for v24."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from apply_boundary_proposals_with_graph_node_gnn_v24 import BOUNDARY_TO_GRAPH_LABEL, bbox, center, center_covered, iou, load_jsonl, write_json, write_jsonl
from train_boundary_segmenter_v18 import POLICY_GRID, detect_candidates, fit_threshold, select_policy


ROOT = Path(__file__).resolve().parents[2]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_gray(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(resolve(path)).convert("L"), dtype=np.uint8)


def mask_from_boxes(row: dict[str, Any], labels: set[str] | None = None) -> np.ndarray:
    image = load_gray(row["image"])
    mask = np.zeros(image.shape, dtype=bool)
    height, width = mask.shape
    for target in (row.get("targets") or {}).get("boxes") or []:
        label = BOUNDARY_TO_GRAPH_LABEL.get(str(target.get("label")), str(target.get("label")))
        if labels is not None and label not in labels:
            continue
        box = bbox(target.get("bbox"))
        if box is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return mask


def mask_from_predictions(row: dict[str, Any], preds: list[dict[str, Any]], labels: set[str] | None = None) -> np.ndarray:
    image = load_gray(row["image"])
    mask = np.zeros(image.shape, dtype=bool)
    height, width = mask.shape
    for pred in preds:
        semantic = "door" if str(pred.get("semantic_type")) == "opening_or_window" else "hard_wall"
        if labels is not None and semantic not in labels:
            continue
        box = bbox(pred.get("bbox"))
        if box is None:
            continue
        x1, y1, x2, y2 = [int(round(v)) for v in box]
        x1 = max(0, min(width, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height, y1))
        y2 = max(0, min(height, y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return mask


def mask_iou(pred: np.ndarray, gold: np.ndarray) -> float:
    inter = int(np.logical_and(pred, gold).sum())
    union = int(np.logical_or(pred, gold).sum())
    return inter / max(union, 1)


def gold_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for target in (row.get("targets") or {}).get("boxes") or []:
        box = bbox(target.get("bbox"))
        label = BOUNDARY_TO_GRAPH_LABEL.get(str(target.get("label")), str(target.get("label")))
        if box is not None and label in {"hard_wall", "door", "window"}:
            out.append({"bbox": box, "label": label, "target_id": str(target.get("target_id") or "")})
    return out


def proposal_hits(preds: list[dict[str, Any]], golds: list[dict[str, Any]]) -> tuple[Counter[str], list[dict[str, Any]]]:
    hits: Counter[str] = Counter()
    misses = []
    for gold in golds:
        matches = [
            pred
            for pred in preds
            if bbox(pred.get("bbox")) is not None
            and (iou(bbox(pred.get("bbox")), gold["bbox"]) >= 0.20 or center_covered(bbox(pred.get("bbox")), gold["bbox"]))
        ]
        hits[f"{gold['label']}_gold"] += 1
        if matches:
            hits[f"{gold['label']}_hit"] += 1
            hits["overall_hit"] += 1
        else:
            misses.append(gold)
        hits["overall_gold"] += 1
    return hits, misses


def endpoint_points_for_box(box: list[float]) -> list[tuple[float, float]]:
    x1, y1, x2, y2 = box
    cx, cy = center(box)
    if (x2 - x1) >= (y2 - y1):
        return [(x1, cy), (x2, cy)]
    return [(cx, y1), (cx, y2)]


def junction_points(items: list[dict[str, Any]]) -> list[tuple[float, float]]:
    counts: Counter[tuple[int, int]] = Counter()
    for item in items:
        box = bbox(item.get("bbox"))
        if box is None:
            continue
        for x, y in endpoint_points_for_box(box):
            counts[(int(round(x / 8.0)), int(round(y / 8.0)))] += 1
    return [(key[0] * 8.0, key[1] * 8.0) for key, count in counts.items() if count >= 2]


def junction_recall(preds: list[dict[str, Any]], golds: list[dict[str, Any]], tolerance: float = 12.0) -> dict[str, Any]:
    pred_points = junction_points(preds)
    gold_points = junction_points(golds)
    matched = 0
    for gx, gy in gold_points:
        if any(math.hypot(gx - px, gy - py) <= tolerance for px, py in pred_points):
            matched += 1
    return {
        "matched": matched,
        "gold": len(gold_points),
        "predicted": len(pred_points),
        "recall": round(matched / max(len(gold_points), 1), 6),
        "precision_proxy": round(matched / max(len(pred_points), 1), 6),
        "tolerance_px": tolerance,
    }


def yolo_pred_map(path: Path, limit: int | None) -> dict[str, list[dict[str, Any]]]:
    return {str(row.get("id")): list(row.get("candidate_stream") or []) for row in load_jsonl(path, limit)}


def metric_from_counts(counts: Counter[str]) -> dict[str, Any]:
    per_label = {}
    for label in ["hard_wall", "door", "window"]:
        gold = counts[f"{label}_gold"]
        hit = counts[f"{label}_hit"]
        per_label[label] = {"gold": gold, "matched": hit, "recall": round(hit / max(gold, 1), 6)}
    return {
        "gold": counts["overall_gold"],
        "matched": counts["overall_hit"],
        "recall": round(counts["overall_hit"] / max(counts["overall_gold"], 1), 6),
        "per_label": per_label,
    }


def evaluate(rows: list[dict[str, Any]], policy: dict[str, Any], yolo_by_id: dict[str, list[dict[str, Any]]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    segment_counts: Counter[str] = Counter()
    union_counts: Counter[str] = Counter()
    mask_ious = []
    opening_mask_ious = []
    junction_total = Counter()
    prediction_rows = []
    miss_examples = []
    for row in rows:
        preds = detect_candidates(row, policy)
        golds = gold_items(row)
        hits, misses = proposal_hits(preds, golds)
        segment_counts.update(hits)
        union_hits, _ = proposal_hits(list(yolo_by_id.get(str(row.get("id")), [])) + preds, golds)
        union_counts.update(union_hits)
        mask_ious.append(mask_iou(mask_from_predictions(row, preds), mask_from_boxes(row)))
        opening_mask_ious.append(mask_iou(mask_from_predictions(row, preds, {"door", "window"}), mask_from_boxes(row, {"door", "window"})))
        j = junction_recall(preds, [{"bbox": item["bbox"]} for item in golds])
        junction_total.update({"matched": j["matched"], "gold": j["gold"], "predicted": j["predicted"]})
        miss_examples.extend({"row_id": row["id"], **miss} for miss in misses[:3])
        prediction_rows.append(
            {
                "id": row["id"],
                "image": row["image"],
                "image_size": row["image_size"],
                "source_integrity": {
                    "source_mode": "image_only_raster_moe",
                    "vector_candidate_ids_used": False,
                    "annotation_geometry_used_at_inference": False,
                    "model_input": "raster_image_only",
                },
                "candidate_stream": [
                    {
                        "candidate_id": pred["id"],
                        "bbox": pred["bbox"],
                        "prediction": "door" if pred.get("semantic_type") == "opening_or_window" else "hard_wall",
                        "label_hint": "door" if pred.get("semantic_type") == "opening_or_window" else "hard_wall",
                        "proposal_source": "boundary_segmentation_graph_probe_v24",
                        "proposal_confidence": pred.get("confidence", 0.5),
                        "confidence": pred.get("confidence", 0.5),
                        "payload": {key: pred.get(key) for key in ["p1", "p2", "orientation", "primitive_type", "length"]},
                    }
                    for pred in preds
                ],
            }
        )
    junction_metric = {
        "matched": int(junction_total["matched"]),
        "gold": int(junction_total["gold"]),
        "predicted": int(junction_total["predicted"]),
        "recall": round(junction_total["matched"] / max(junction_total["gold"], 1), 6),
        "precision_proxy": round(junction_total["matched"] / max(junction_total["predicted"], 1), 6),
    }
    return {
        "mask_iou_mean": round(float(np.mean(mask_ious)) if mask_ious else 0.0, 6),
        "opening_mask_iou_mean": round(float(np.mean(opening_mask_ious)) if opening_mask_ious else 0.0, 6),
        "segment_recall": metric_from_counts(segment_counts),
        "junction_recall": junction_metric,
        "yolo_union_recall": metric_from_counts(union_counts),
        "candidate_inflation": round(sum(len(row["candidate_stream"]) for row in prediction_rows) / max(segment_counts["overall_gold"], 1), 6),
        "miss_examples": miss_examples[:200],
    }, prediction_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="datasets/boundary_expert_public_raster_v19")
    parser.add_argument("--yolo-locked", default="reports/vlm/boundary_public_raster_v24_yolo_full_locked50_candidate_stream.jsonl")
    parser.add_argument("--report-output", default="reports/vlm/boundary_segmentation_graph_probe_v24_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/boundary_segmentation_graph_probe_v24_locked_predictions.jsonl")
    parser.add_argument("--checkpoint-output", default="checkpoints/boundary_segmentation_graph_probe_v24/policy.json")
    parser.add_argument("--limit-train", type=int, default=160)
    parser.add_argument("--limit-dev", type=int, default=80)
    parser.add_argument("--limit-locked", type=int, default=50)
    parser.add_argument("--policy-index", type=int, default=-1)
    parser.add_argument("--cap", type=int, default=0)
    parser.add_argument("--max-detect-side", type=int, default=0)
    args = parser.parse_args()

    dataset = resolve(args.dataset)
    train_rows = load_jsonl(dataset / "train.jsonl", args.limit_train)
    dev_rows = load_jsonl(dataset / "dev.jsonl", args.limit_dev)
    locked_rows = load_jsonl(dataset / "locked.jsonl", args.limit_locked)
    threshold_fit = fit_threshold(train_rows, sample_limit=max(1, args.limit_train))
    if args.policy_index >= 0:
        policy = dict(POLICY_GRID[args.policy_index])
        policy["threshold"] = int(policy["threshold"] or threshold_fit["selected_threshold"])
    else:
        policy, _ = select_policy(dev_rows, int(threshold_fit["selected_threshold"]))
    if args.cap:
        policy["cap"] = int(args.cap)
    if args.max_detect_side:
        policy["max_detect_side"] = int(args.max_detect_side)
    yolo_by_id = yolo_pred_map(resolve(args.yolo_locked), args.limit_locked)
    locked_eval, predictions = evaluate(locked_rows, policy, yolo_by_id)
    checkpoint = {
        "version": "boundary_segmentation_graph_probe_v24",
        "selected_policy": policy,
        "threshold_fit": threshold_fit,
        "runtime_input": "raster_image_only",
        "offline_gold_used_for": ["threshold_fit", "dev_policy_selection", "locked_evaluation"],
    }
    write_json(resolve(args.checkpoint_output), checkpoint)
    report = {
        "version": "boundary_segmentation_graph_probe_v24_eval",
        "task": "P0-02B-boundary-segmentation-graph-probe",
        "checkpoint": str(resolve(args.checkpoint_output).relative_to(ROOT)),
        "claim_boundary": "Raster-only mask/run vectorizer used as segmentation/graph proposal probe. Gold labels are used only for threshold fit, dev selection, and locked evaluation.",
        "locked_eval": locked_eval,
        "success_gate": {
            "reports_mask_iou": True,
            "reports_segment_recall": True,
            "reports_junction_recall": True,
            "reports_yolo_union_no_drop": True,
            "yolo_union_recall_min": 0.980474,
            "yolo_union_recall": locked_eval["yolo_union_recall"]["recall"],
            "opening_union_recall": {
                "door": locked_eval["yolo_union_recall"]["per_label"]["door"]["recall"],
                "window": locked_eval["yolo_union_recall"]["per_label"]["window"]["recall"],
            },
        },
        "source_integrity": {
            "model_input": "raster_image_only",
            "offline_gold_used_at_inference": False,
            "svg_or_vector_geometry_used_at_inference": False,
        },
    }
    gate = report["success_gate"]
    gate["passed"] = gate["yolo_union_recall"] >= gate["yolo_union_recall_min"]
    write_json(resolve(args.report_output), report)
    write_jsonl(resolve(args.predictions_output), predictions)
    print(json.dumps({"policy": policy, "locked_eval": locked_eval, "success_gate": gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
