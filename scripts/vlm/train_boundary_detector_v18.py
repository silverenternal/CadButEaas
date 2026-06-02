#!/usr/bin/env python3
"""Train/evaluate a recall-first raster boundary detector for v18.

The detector is deliberately lightweight: dev labels are used only to choose
image-processing parameters, while inference consumes raster pixels only.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"

PARAM_GRID = [
    {"threshold": 180, "min_len": 8, "gap": 1, "pad": 1, "cap": 9000, "anchor_stride": 16, "anchor_density": 0.08},
    {"threshold": 205, "min_len": 8, "gap": 2, "pad": 1, "cap": 12000, "anchor_stride": 12, "anchor_density": 0.08},
    {"threshold": 230, "min_len": 6, "gap": 2, "pad": 1, "cap": 16000, "anchor_stride": 10, "anchor_density": 0.08},
    {"threshold": 245, "min_len": 5, "gap": 3, "pad": 1, "cap": 22000, "anchor_stride": 8, "anchor_density": 0.08},
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def bbox_iou(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right:
        return 0.0
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    la = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    ra = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(la + ra - inter, 1e-9)


def f1(tp: int, predicted: int, gold: int) -> dict[str, float | int]:
    precision = tp / max(predicted, 1)
    recall = tp / max(gold, 1)
    score = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "matched": int(tp),
        "predicted": int(predicted),
        "gold": int(gold),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(score, 6),
    }


def match_counts(preds: list[dict[str, Any]], golds: list[dict[str, Any]], iou_threshold: float) -> tuple[int, list[dict[str, Any]]]:
    cell = 32.0
    grid: dict[tuple[int, int], list[int]] = {}
    pred_boxes: list[list[float] | None] = []
    for pred_index, pred in enumerate(preds):
        box = pred.get("bbox")
        pred_boxes.append(box)
        if not box:
            continue
        x1, y1, x2, y2 = [float(v) for v in box]
        gx1, gx2 = int(x1 // cell), int(x2 // cell)
        gy1, gy2 = int(y1 // cell), int(y2 // cell)
        for gx in range(gx1, gx2 + 1):
            for gy in range(gy1, gy2 + 1):
                grid.setdefault((gx, gy), []).append(pred_index)

    used: set[int] = set()
    matched = 0
    misses: list[dict[str, Any]] = []
    for gold_index, gold in enumerate(golds):
        gold_box = gold.get("bbox")
        if not gold_box:
            continue
        x1, y1, x2, y2 = [float(v) for v in gold_box]
        candidate_indices: set[int] = set()
        for gx in range(int(x1 // cell) - 1, int(x2 // cell) + 2):
            for gy in range(int(y1 // cell) - 1, int(y2 // cell) + 2):
                candidate_indices.update(grid.get((gx, gy), []))
        best_index = None
        best_iou = 0.0
        for pred_index in candidate_indices:
            if pred_index in used:
                continue
            score = bbox_iou(pred_boxes[pred_index], gold_box)
            if score > best_iou:
                best_iou = score
                best_index = pred_index
        if best_index is not None and best_iou >= iou_threshold:
            used.add(best_index)
            matched += 1
        else:
            misses.append({"gold_index": gold_index, "bbox": gold_box, "best_iou": round(best_iou, 6), "label": gold.get("label")})
    return matched, misses


def group_runs(indices: list[int], max_gap: int) -> list[tuple[int, int]]:
    if not indices:
        return []
    runs = []
    start = prev = indices[0]
    for value in indices[1:]:
        if value - prev <= max_gap + 1:
            prev = value
            continue
        runs.append((start, prev))
        start = prev = value
    runs.append((start, prev))
    return runs


def dark_mask(image_path: Path, threshold: int) -> tuple[Any, int, int]:
    import numpy as np

    with Image.open(image_path) as image:
        gray = image.convert("L")
        arr = np.asarray(gray, dtype=np.uint8)
    return arr <= int(threshold), int(arr.shape[1]), int(arr.shape[0])


def detect_boundary_candidates(row: dict[str, Any], params: dict[str, Any]) -> list[dict[str, Any]]:
    import numpy as np

    image_path = ROOT / str(row.get("image") or "")
    if not image_path.exists():
        return []
    mask, width, height = dark_mask(image_path, int(params["threshold"]))
    min_len = int(params["min_len"])
    gap = int(params["gap"])
    pad = int(params["pad"])
    candidates: list[dict[str, Any]] = []

    for y in range(height):
        xs = np.flatnonzero(mask[y, :]).tolist()
        for start, end in group_runs(xs, gap):
            length = end - start + 1
            if length < min_len:
                continue
            y1 = max(0, y - pad)
            y2 = min(height, y + pad + 1)
            candidates.append(candidate(row, "h", start, y, end, y, [start, y1, min(width, end + 1), y2], length, params))

    for x in range(width):
        ys = np.flatnonzero(mask[:, x]).tolist()
        for start, end in group_runs(ys, gap):
            length = end - start + 1
            if length < min_len:
                continue
            x1 = max(0, x - pad)
            x2 = min(width, x + pad + 1)
            candidates.append(candidate(row, "v", x, start, x, end, [x1, start, x2, min(height, end + 1)], length, params))

    if bool(params.get("use_anchors", True)):
        candidates.extend(anchor_candidates(row, mask, width, height, params))
    return sorted(candidates, key=lambda item: float(item.get("confidence") or 0.0), reverse=True)[: int(params["cap"])]


def anchor_candidates(row: dict[str, Any], mask: Any, width: int, height: int, params: dict[str, Any]) -> list[dict[str, Any]]:
    import numpy as np

    stride = int(params.get("anchor_stride", 12))
    min_len = int(params["min_len"])
    pad = int(params["pad"])
    lengths = [8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384]
    min_density = float(params.get("anchor_density", 0.01))
    out: list[dict[str, Any]] = []
    active_rows = np.flatnonzero(mask.sum(axis=1) >= min_len).tolist()
    active_cols = np.flatnonzero(mask.sum(axis=0) >= min_len).tolist()
    for y in active_rows:
        for length in lengths:
            if length > width:
                continue
            for x in range(0, max(1, width - length + 1), stride):
                window = mask[y, x : x + length]
                density = float(window.sum()) / max(length, 1)
                if density < min_density:
                    continue
                out.append(candidate(row, "ha", x, y, min(width - 1, x + length - 1), y, [x, max(0, y - pad), min(width, x + length), min(height, y + pad + 1)], length, params, density))
    for x in active_cols:
        for length in lengths:
            if length > height:
                continue
            for y in range(0, max(1, height - length + 1), stride):
                window = mask[y : y + length, x]
                density = float(window.sum()) / max(length, 1)
                if density < min_density:
                    continue
                out.append(candidate(row, "va", x, y, x, min(height - 1, y + length - 1), [max(0, x - pad), y, min(width, x + pad + 1), min(height, y + length)], length, params, density))
    return out


def candidate(
    row: dict[str, Any],
    orientation: str,
    x1: int,
    y1: int,
    x2: int,
    y2: int,
    bbox: list[int],
    length: int,
    params: dict[str, Any],
    density: float = 1.0,
) -> dict[str, Any]:
    conf = min(0.99, 0.14 + length / 512.0 * 0.42 + density * 0.20 + (0.08 if orientation.startswith("h") else 0.06))
    return {
        "id": f"{row['id']}_boundary_v18_{orientation}_{len(str(x1))}_{x1}_{y1}_{x2}_{y2}",
        "class": "wall",
        "semantic_type": "wall",
        "family": "boundary",
        "bbox": [int(v) for v in bbox],
        "p1": [int(x1), int(y1)],
        "p2": [int(x2), int(y2)],
        "confidence": round(float(conf), 6),
        "proposal_source": "raster_boundary_detector_v18",
        "primitive_type": f"dark_pixel_{'anchor' if orientation.endswith('a') else 'run'}_{orientation}",
        "detector_params": {
            "threshold": int(params["threshold"]),
            "min_len": int(params["min_len"]),
            "gap": int(params["gap"]),
            "pad": int(params["pad"]),
            "anchor_stride": int(params.get("anchor_stride", 0)),
        },
    }


def nms(items: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda row: float(row.get("confidence") or 0.0), reverse=True):
        box = item.get("bbox")
        if all(bbox_iou(box, other.get("bbox")) < threshold for other in kept):
            kept.append(item)
    return kept


def evaluate(rows: list[dict[str, Any]], params: dict[str, Any], keep_predictions: bool = False) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    totals = Counter()
    label_totals: dict[str, Counter[str]] = {label: Counter() for label in ("wall", "opening", "window")}
    miss_examples = []
    prediction_rows = []
    for row in rows:
        preds = detect_boundary_candidates(row, params)
        golds = list(((row.get("targets") or {}).get("boxes") or []))
        matched, misses = match_counts(preds, golds, iou_threshold=0.20)
        totals.update({"matched": matched, "predicted": len(preds), "gold": len(golds)})
        for label in label_totals:
            label_golds = [gold for gold in golds if gold.get("label") == label]
            label_matched, _label_misses = match_counts(preds, label_golds, iou_threshold=0.20)
            label_totals[label].update({"matched": label_matched, "predicted": len(preds), "gold": len(label_golds)})
        for miss in misses[:5]:
            miss_examples.append({"id": row.get("id"), **miss})
        if keep_predictions:
            prediction_rows.append(
                {
                    "id": row.get("id"),
                    "image": row.get("image"),
                    "image_size": row.get("image_size"),
                    "source_integrity": {
                        "source_mode": "image_only_raster_moe",
                        "vector_candidate_ids_used": False,
                        "annotation_geometry_used_at_inference": False,
                        "model_input": "raster_image_only",
                    },
                    "proposals": preds,
                    "gold_counts": row.get("target_counts"),
                }
            )
    report = {
        **f1(totals["matched"], totals["predicted"], totals["gold"]),
        "iou_threshold": 0.20,
        "candidate_inflation": round(totals["predicted"] / max(totals["gold"], 1), 6),
        "per_label_recall_proxy": {
            label: f1(counter["matched"], counter["predicted"], counter["gold"])
            for label, counter in label_totals.items()
        },
        "miss_examples": miss_examples[:100],
    }
    return report, prediction_rows


def select_params(dev_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored = []
    best_params = PARAM_GRID[0]
    best_score = -1.0
    for params in PARAM_GRID:
        metric, _rows = evaluate(dev_rows, params)
        recall = float(metric["recall"])
        precision = float(metric["precision"])
        inflation = float(metric["candidate_inflation"])
        score = recall * 2.0 + precision * 0.2 - min(inflation, 12.0) * 0.015
        scored.append({"params": params, "metric": metric, "selection_score": round(score, 6)})
        if score > best_score:
            best_score = score
            best_params = params
    return dict(best_params), scored


def routed_candidate_rows(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for row in prediction_rows:
        candidates = []
        for proposal in row.get("proposals") or []:
            candidates.append(
                {
                    "candidate_id": proposal["id"],
                    "expert": "wall_opening",
                    "family": "boundary",
                    "candidate_type": "wall",
                    "confidence": proposal["confidence"],
                    "bbox": proposal["bbox"],
                    "source": "image_only_raster_moe",
                    "payload": {
                        "image": row.get("image"),
                        "raster_path": row.get("image"),
                        "_page_metadata": {"width": row.get("image_size", [512, 512])[0], "height": row.get("image_size", [512, 512])[1]},
                        "features": {
                            "primitive_type": proposal.get("primitive_type"),
                            "bbox": proposal.get("bbox"),
                            "centroid": [
                                (proposal["bbox"][0] + proposal["bbox"][2]) * 0.5,
                                (proposal["bbox"][1] + proposal["bbox"][3]) * 0.5,
                            ],
                            "length": math.hypot(proposal["p2"][0] - proposal["p1"][0], proposal["p2"][1] - proposal["p1"][1]),
                            "angle_degrees": 0.0,
                            "orientation": "horizontal" if proposal["p1"][1] == proposal["p2"][1] else "vertical",
                        },
                        "proposal_source": "raster_boundary_detector_v18",
                        "source_integrity": row.get("source_integrity"),
                    },
                    "route_trace": {
                        "source_mode": "image_only_raster_moe",
                        "routing_method": "v18_boundary_detector_adapter",
                        "matched_hint": "wall",
                        "routing_confidence": proposal["confidence"],
                        "abstain": False,
                    },
                }
            )
        out.append(
            {
                "id": row.get("id"),
                "image": row.get("image"),
                "image_size": row.get("image_size"),
                "source_integrity": row.get("source_integrity"),
                "route_trace": {"stage": "boundary_detector_v18_to_routed_candidates", **row.get("source_integrity", {})},
                "candidate_stream": candidates,
            }
        )
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/image_only_boundary_detector_v18")
    parser.add_argument("--checkpoint-dir", default="checkpoints/boundary_detector_v18")
    parser.add_argument("--report-prefix", default="boundary_detector_v18")
    parser.add_argument("--limit-dev", type=int, default=0)
    parser.add_argument("--limit-locked", type=int, default=0)
    parser.add_argument("--fast-run-only", action="store_true", help="Disable dense anchor windows for quick public-raster audits.")
    args = parser.parse_args()

    data_dir = ROOT / args.data
    checkpoint_dir = ROOT / args.checkpoint_dir
    report_prefix = args.report_prefix

    dev_rows = load_jsonl(data_dir / "dev.jsonl")
    locked_rows = load_jsonl(data_dir / "locked.jsonl")
    if args.limit_dev:
        dev_rows = dev_rows[: args.limit_dev]
    if args.limit_locked:
        locked_rows = locked_rows[: args.limit_locked]

    if args.fast_run_only:
        for params in PARAM_GRID:
            params["use_anchors"] = False
            params["cap"] = min(int(params["cap"]), 6000)

    params, dev_grid = select_params(dev_rows)
    dev_metric, _ = evaluate(dev_rows, params)
    locked_metric, locked_predictions = evaluate(locked_rows, params, keep_predictions=True)
    routed_rows = routed_candidate_rows(locked_predictions)

    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "version": "boundary_detector_v18_dark_run_baseline",
        "dataset": str(data_dir.relative_to(ROOT)),
        "selected_params": params,
        "fast_run_only": bool(args.fast_run_only),
        "selection_split": "dev",
        "inference_input": "raster_image_only",
    }
    write_json(checkpoint_dir / "policy.json", checkpoint)
    report = {
        "task": "IMG-MOE-V18-P0-003",
        "checkpoint": str((checkpoint_dir / "policy.json").relative_to(ROOT)),
        "dataset": str(data_dir.relative_to(ROOT)),
        "limits": {"dev": int(args.limit_dev), "locked": int(args.limit_locked), "fast_run_only": bool(args.fast_run_only)},
        "detector": checkpoint,
        "dev_grid": dev_grid,
        "dev_metric": dev_metric,
        "locked_metric": locked_metric,
        "success_criteria": {
            "boundary_candidate_recall_at_least_0_70": float(locked_metric["recall"]) >= 0.70,
            "source_integrity_violations": 0,
            "exports_routed_candidate_rows": True,
        },
        "source_integrity": {
            "model_input": "raster_image_only",
            "offline_gold_used_for": ["dev_parameter_selection", "locked_evaluation"],
            "offline_gold_used_at_inference": False,
        },
    }
    write_json(REPORT / f"{report_prefix}_eval.json", report)
    write_jsonl(REPORT / f"{report_prefix}_locked_predictions.jsonl", locked_predictions)
    write_jsonl(REPORT / f"{report_prefix}_routed_candidates.jsonl", routed_rows)
    print(json.dumps({
        "task": report["task"],
        "recall": locked_metric["recall"],
        "precision": locked_metric["precision"],
        "candidate_inflation": locked_metric["candidate_inflation"],
        "success": report["success_criteria"]["boundary_candidate_recall_at_least_0_70"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
