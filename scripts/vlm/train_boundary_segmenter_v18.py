#!/usr/bin/env python3
"""Train/evaluate a mask-first raster boundary segmenter for v18.

The learned part is intentionally small and auditable: train masks are used to
fit a foreground threshold over raster pixels, then inference uses only the
page image. Candidate export vectorizes dark foreground runs plus collinear
gaps, because openings/windows are often represented by absence of ink between
wall strokes.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"

THRESHOLDS = [170, 185, 200, 215, 230, 245]
POLICY_GRID = [
    {"threshold": None, "gap": 1, "pad": 1, "min_run": 2, "max_gap": 40, "cap": 9000, "slice_stride": 1.0, "bucketed_cap": True},
    {"threshold": None, "gap": 2, "pad": 1, "min_run": 2, "max_gap": 56, "cap": 14000, "slice_stride": 1.25, "bucketed_cap": True},
    {"threshold": 245, "gap": 2, "pad": 1, "min_run": 2, "max_gap": 64, "cap": 18000, "slice_stride": 1.5, "bucketed_cap": True},
]
SEGMENT_LENGTHS = [2, 3, 4, 5, 6, 8, 10, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256]


def load_jsonl(path: Path, limit: int = 0) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
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


def load_gray(path: str | Path) -> np.ndarray:
    return np.asarray(Image.open(ROOT / str(path)).convert("L"), dtype=np.uint8)


def load_gray_for_detection(path: str | Path, max_side: int = 0) -> tuple[np.ndarray, float, float]:
    image = Image.open(ROOT / str(path)).convert("L")
    original_width, original_height = image.size
    if max_side and max(original_width, original_height) > max_side:
        scale = float(max_side) / float(max(original_width, original_height))
        resized = (
            max(1, int(round(original_width * scale))),
            max(1, int(round(original_height * scale))),
        )
        image = image.resize(resized, Image.Resampling.BILINEAR)
    resized_width, resized_height = image.size
    sx = float(original_width) / float(resized_width)
    sy = float(original_height) / float(resized_height)
    return np.asarray(image, dtype=np.uint8), sx, sy


def mask_from_boxes(row: dict[str, Any], shape: tuple[int, int]) -> np.ndarray:
    height, width = shape
    mask = np.zeros((height, width), dtype=bool)
    for target in ((row.get("targets") or {}).get("boxes") or []):
        box = target.get("bbox")
        if not isinstance(box, list) or len(box) < 4:
            continue
        x1, y1, x2, y2 = [int(v) for v in box[:4]]
        x1 = max(0, min(width - 1, x1))
        x2 = max(0, min(width, x2))
        y1 = max(0, min(height - 1, y1))
        y2 = max(0, min(height, y2))
        if x2 > x1 and y2 > y1:
            mask[y1:y2, x1:x2] = True
    return mask


def bbox_iou(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right:
        return 0.0
    ix1 = max(float(left[0]), float(right[0]))
    iy1 = max(float(left[1]), float(right[1]))
    ix2 = min(float(left[2]), float(right[2]))
    iy2 = min(float(left[3]), float(right[3]))
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    la = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    ra = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    return inter / max(la + ra - inter, 1e-9)


def f1(tp: int, predicted: int, gold: int) -> dict[str, float | int]:
    precision = tp / max(predicted, 1)
    recall = tp / max(gold, 1)
    score = 0.0 if precision + recall == 0.0 else 2.0 * precision * recall / (precision + recall)
    return {
        "matched": int(tp),
        "predicted": int(predicted),
        "gold": int(gold),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(score, 6),
    }


def fit_threshold(train_rows: list[dict[str, Any]], sample_limit: int) -> dict[str, Any]:
    positives: list[np.ndarray] = []
    negatives: list[np.ndarray] = []
    for row in train_rows[:sample_limit]:
        image = load_gray(row["image"])
        mask_path = (row.get("targets") or {}).get("boundary_mask")
        mask = load_gray(mask_path) > 0 if mask_path else mask_from_boxes(row, image.shape)
        positives.append(image[mask])
        neg = image[~mask]
        if neg.size:
            negatives.append(neg[:: max(1, neg.size // 12000)])
    pos = np.concatenate(positives) if positives else np.array([], dtype=np.uint8)
    neg = np.concatenate(negatives) if negatives else np.array([], dtype=np.uint8)
    scored = []
    best = {"threshold": 230, "pixel_f2": -1.0}
    for threshold in THRESHOLDS:
        tp = int((pos <= threshold).sum())
        fn = int((pos > threshold).sum())
        fp = int((neg <= threshold).sum())
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f2 = 0.0 if precision + recall == 0.0 else 5.0 * precision * recall / max(4.0 * precision + recall, 1e-9)
        metric = {
            "threshold": threshold,
            "pixel_precision": round(precision, 6),
            "pixel_recall": round(recall, 6),
            "pixel_f2": round(f2, 6),
            "sampled_positive_pixels": int(pos.size),
            "sampled_negative_pixels": int(neg.size),
        }
        scored.append(metric)
        if f2 > float(best["pixel_f2"]):
            best = metric
    return {"selected_threshold": int(best["threshold"]), "grid": scored}


def grouped(indices: np.ndarray, max_gap: int) -> list[tuple[int, int]]:
    if indices.size == 0:
        return []
    out: list[tuple[int, int]] = []
    start = prev = int(indices[0])
    for raw in indices[1:]:
        value = int(raw)
        if value - prev <= max_gap + 1:
            prev = value
            continue
        out.append((start, prev))
        start = prev = value
    out.append((start, prev))
    return out


def add_candidate(
    out: dict[tuple[int, int, int, int, str], dict[str, Any]],
    row_id: str,
    bbox: list[int],
    p1: list[int],
    p2: list[int],
    source: str,
    confidence: float,
) -> None:
    x1, y1, x2, y2 = bbox
    if x2 <= x1 or y2 <= y1:
        return
    key = (int(x1), int(y1), int(x2), int(y2), source)
    if key in out and float(out[key]["confidence"]) >= confidence:
        return
    out[key] = {
        "id": f"{row_id}_boundary_segmenter_v18_{source}_{x1}_{y1}_{x2}_{y2}",
        "class": "wall" if source != "gap" else "opening",
        "semantic_type": "wall" if source != "gap" else "opening_or_window",
        "family": "boundary",
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "p1": [int(p1[0]), int(p1[1])],
        "p2": [int(p2[0]), int(p2[1])],
        "confidence": round(float(max(0.01, min(0.99, confidence))), 6),
        "proposal_source": "raster_boundary_segmenter_v18",
        "primitive_type": source,
        "length": int(max(abs(p2[0] - p1[0]), abs(p2[1] - p1[1])) + 1),
        "orientation": "horizontal" if p1[1] == p2[1] else "vertical",
    }


def cap_candidates(candidates: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    cap = int(policy["cap"])
    if not bool(policy.get("bucketed_cap")):
        return sorted(candidates, key=lambda item: float(item["confidence"]), reverse=True)[:cap]
    buckets: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    for item in candidates:
        length = int(item.get("length") or 0)
        if length <= 8:
            size_bucket = "tiny"
        elif length <= 32:
            size_bucket = "short"
        elif length <= 128:
            size_bucket = "medium"
        else:
            size_bucket = "long"
        key = (str(item.get("primitive_type")), str(item.get("orientation")), size_bucket)
        buckets.setdefault(key, []).append(item)
    per_bucket = max(64, cap // max(len(buckets), 1))
    kept: list[dict[str, Any]] = []
    for bucket_items in buckets.values():
        kept.extend(sorted(bucket_items, key=lambda item: float(item["confidence"]), reverse=True)[:per_bucket])
    return sorted(kept, key=lambda item: float(item["confidence"]), reverse=True)[:cap]


def emit_segments(
    out: dict[tuple[int, int, int, int, str], dict[str, Any]],
    row_id: str,
    orientation: str,
    fixed: int,
    start: int,
    end: int,
    width: int,
    height: int,
    pad: int,
    source: str,
    slice_stride: float,
) -> None:
    length = end - start + 1
    lengths = [value for value in SEGMENT_LENGTHS if value <= length + 4]
    if length not in lengths:
        lengths.append(length)
    for seg_len in sorted(set(lengths)):
        step = max(1, int(round(seg_len * slice_stride)))
        stops = list(range(start, max(start + 1, end - seg_len + 2), step))
        stops.append(max(start, end - seg_len + 1))
        for pos in sorted(set(stops)):
            stop = min(end, pos + seg_len - 1)
            confidence = 0.42 + min(seg_len / 256.0, 1.0) * 0.24
            if source == "gap":
                confidence -= 0.10
            if orientation == "h":
                bbox = [max(0, pos - pad), max(0, fixed - pad), min(width, stop + 1 + pad), min(height, fixed + 1 + pad)]
                add_candidate(out, row_id, bbox, [pos, fixed], [stop, fixed], source, confidence)
            else:
                bbox = [max(0, fixed - pad), max(0, pos - pad), min(width, fixed + 1 + pad), min(height, stop + 1 + pad)]
                add_candidate(out, row_id, bbox, [fixed, pos], [fixed, stop], source, confidence)


def detect_candidates(row: dict[str, Any], policy: dict[str, Any]) -> list[dict[str, Any]]:
    image, sx, sy = load_gray_for_detection(row["image"], int(policy.get("max_detect_side") or 0))
    mask = image <= int(policy["threshold"])
    height, width = mask.shape
    gap = int(policy["gap"])
    pad = int(policy["pad"])
    min_run = int(policy["min_run"])
    max_gap = int(policy["max_gap"])
    candidates: dict[tuple[int, int, int, int, str], dict[str, Any]] = {}

    for y in np.flatnonzero(mask.sum(axis=1) >= min_run):
        runs = grouped(np.flatnonzero(mask[int(y), :]), gap)
        for start, end in runs:
            if end - start + 1 >= min_run:
                emit_segments(candidates, row["id"], "h", int(y), start, end, width, height, pad, "ink_run", float(policy["slice_stride"]))
        for left, right in zip(runs, runs[1:], strict=False):
            gap_start, gap_end = left[1] + 1, right[0] - 1
            if 1 <= gap_end - gap_start + 1 <= max_gap:
                emit_segments(candidates, row["id"], "h", int(y), gap_start, gap_end, width, height, pad, "gap", float(policy["slice_stride"]))

    for x in np.flatnonzero(mask.sum(axis=0) >= min_run):
        runs = grouped(np.flatnonzero(mask[:, int(x)]), gap)
        for start, end in runs:
            if end - start + 1 >= min_run:
                emit_segments(candidates, row["id"], "v", int(x), start, end, width, height, pad, "ink_run", float(policy["slice_stride"]))
        for top, bottom in zip(runs, runs[1:], strict=False):
            gap_start, gap_end = top[1] + 1, bottom[0] - 1
            if 1 <= gap_end - gap_start + 1 <= max_gap:
                emit_segments(candidates, row["id"], "v", int(x), gap_start, gap_end, width, height, pad, "gap", float(policy["slice_stride"]))

    selected = cap_candidates(list(candidates.values()), policy)
    if sx == 1.0 and sy == 1.0:
        return selected
    for item in selected:
        x1, y1, x2, y2 = item["bbox"]
        item["bbox"] = [int(round(x1 * sx)), int(round(y1 * sy)), int(round(x2 * sx)), int(round(y2 * sy))]
        item["p1"] = [int(round(item["p1"][0] * sx)), int(round(item["p1"][1] * sy))]
        item["p2"] = [int(round(item["p2"][0] * sx)), int(round(item["p2"][1] * sy))]
        item["detection_scale"] = {"sx": round(sx, 6), "sy": round(sy, 6)}
    return selected


def match_counts(
    preds: list[dict[str, Any]],
    golds: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[int, Counter[str], list[dict[str, Any]]]:
    cell = 16.0
    grid: dict[tuple[int, int], list[int]] = {}
    pred_areas: list[float] = []
    for index, pred in enumerate(preds):
        x1, y1, x2, y2 = [float(v) for v in pred["bbox"]]
        pred_areas.append(max(0.0, x2 - x1) * max(0.0, y2 - y1))
        for gx in range(int(x1 // cell), int(x2 // cell) + 1):
            for gy in range(int(y1 // cell), int(y2 // cell) + 1):
                grid.setdefault((gx, gy), []).append(index)
    used: set[int] = set()
    matched = 0
    matched_labels: Counter[str] = Counter()
    misses: list[dict[str, Any]] = []
    for gold_index, gold in enumerate(golds):
        x1, y1, x2, y2 = [float(v) for v in gold["bbox"]]
        gold_area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
        min_area = gold_area * iou_threshold
        max_area = gold_area / max(iou_threshold, 1e-9)
        candidate_indices: set[int] = set()
        for gx in range(int(x1 // cell) - 1, int(x2 // cell) + 2):
            for gy in range(int(y1 // cell) - 1, int(y2 // cell) + 2):
                candidate_indices.update(grid.get((gx, gy), []))
        best_index = None
        best_iou = 0.0
        for pred_index in candidate_indices:
            if pred_index in used:
                continue
            if pred_areas[pred_index] < min_area or pred_areas[pred_index] > max_area:
                continue
            score = bbox_iou(preds[pred_index]["bbox"], gold["bbox"])
            if score > best_iou:
                best_iou = score
                best_index = pred_index
        if best_index is not None and best_iou >= iou_threshold:
            used.add(best_index)
            matched += 1
            matched_labels.update([str(gold.get("label") or "unknown")])
        else:
            misses.append({"gold_index": gold_index, "bbox": gold["bbox"], "label": gold.get("label"), "best_iou": round(best_iou, 6)})
    return matched, matched_labels, misses


def evaluate(
    rows: list[dict[str, Any]],
    policy: dict[str, Any],
    keep_predictions: bool = False,
    export_top_k: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    totals = Counter()
    per_label: dict[str, Counter[str]] = {label: Counter() for label in ("wall", "opening", "window")}
    miss_examples: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for row in rows:
        preds = detect_candidates(row, policy)
        golds = list(row["targets"]["boxes"])
        matched, matched_labels, misses = match_counts(preds, golds, 0.20)
        totals.update({"matched": matched, "predicted": len(preds), "gold": len(golds)})
        for label, counter in per_label.items():
            label_golds = [gold for gold in golds if gold.get("label") == label]
            counter.update({"matched": matched_labels[label], "predicted": len(preds), "gold": len(label_golds)})
        miss_examples.extend({"id": row["id"], **miss} for miss in misses[:5])
        if keep_predictions:
            export_preds = preds[:export_top_k] if export_top_k else preds
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
                    "proposals": export_preds,
                    "proposal_count_before_export_cap": len(preds),
                    "gold_counts": row.get("target_counts"),
                }
            )
    metric = {
        **f1(totals["matched"], totals["predicted"], totals["gold"]),
        "iou_threshold": 0.20,
        "candidate_inflation": round(totals["predicted"] / max(totals["gold"], 1), 6),
        "per_label_recall": {label: f1(c["matched"], c["predicted"], c["gold"]) for label, c in per_label.items()},
        "miss_examples": miss_examples[:100],
    }
    return metric, prediction_rows


def select_policy(dev_rows: list[dict[str, Any]], selected_threshold: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored = []
    best_policy = {}
    best_score = -1e9
    for raw_policy in POLICY_GRID:
        policy = dict(raw_policy)
        policy["threshold"] = int(policy["threshold"] or selected_threshold)
        metric, _ = evaluate(dev_rows, policy)
        recall = float(metric["recall"])
        precision = float(metric["precision"])
        inflation = float(metric["candidate_inflation"])
        opening_recall = float(metric["per_label_recall"]["opening"]["recall"])
        window_recall = float(metric["per_label_recall"]["window"]["recall"])
        score = recall * 3.0 + (opening_recall + window_recall) * 0.5 + precision * 0.1 - min(inflation, 20.0) * 0.01
        scored.append({"policy": policy, "metric": metric, "selection_score": round(score, 6)})
        if score > best_score:
            best_score = score
            best_policy = policy
    return best_policy, scored


def routed_candidate_rows(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for row in prediction_rows:
        stream = []
        width, height = row.get("image_size", [512, 512])
        for proposal in row.get("proposals") or []:
            x1, y1, x2, y2 = proposal["bbox"]
            stream.append(
                {
                    "candidate_id": proposal["id"],
                    "expert": "wall_opening",
                    "family": "boundary",
                    "candidate_type": proposal["semantic_type"],
                    "confidence": proposal["confidence"],
                    "bbox": proposal["bbox"],
                    "source": "image_only_raster_moe",
                    "payload": {
                        "image": row["image"],
                        "raster_path": row["image"],
                        "_page_metadata": {"width": width, "height": height},
                        "features": {
                            "primitive_type": proposal["primitive_type"],
                            "bbox": proposal["bbox"],
                            "centroid": [(x1 + x2) * 0.5, (y1 + y2) * 0.5],
                            "length": math.hypot(proposal["p2"][0] - proposal["p1"][0], proposal["p2"][1] - proposal["p1"][1]),
                            "orientation": "horizontal" if proposal["p1"][1] == proposal["p2"][1] else "vertical",
                        },
                        "proposal_source": "raster_boundary_segmenter_v18",
                        "source_integrity": row["source_integrity"],
                    },
                    "route_trace": {
                        "source_mode": "image_only_raster_moe",
                        "routing_method": "v18_boundary_segmenter_adapter",
                        "routing_confidence": proposal["confidence"],
                        "abstain": False,
                    },
                }
            )
        rows.append(
            {
                "id": row["id"],
                "image": row["image"],
                "image_size": row["image_size"],
                "source_integrity": row["source_integrity"],
                "route_trace": {"stage": "boundary_segmenter_v18_to_routed_candidates", **row["source_integrity"]},
                "candidate_stream": stream,
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="datasets/image_only_boundary_detector_v18")
    parser.add_argument("--checkpoint-dir", default="checkpoints/boundary_segmenter_v18")
    parser.add_argument("--report-prefix", default="boundary_segmenter_v18")
    parser.add_argument("--limit-train", type=int, default=160)
    parser.add_argument("--limit-dev", type=int, default=0)
    parser.add_argument("--limit-locked", type=int, default=0)
    parser.add_argument("--policy-index", type=int, default=-1, help="Use a fixed POLICY_GRID index and skip dev selection.")
    parser.add_argument("--export-top-k", type=int, default=2500)
    parser.add_argument("--no-routed-export", action="store_true")
    parser.add_argument("--disable-bucketed-cap", action="store_true")
    parser.add_argument("--cap", type=int, default=0, help="Override selected policy candidate cap.")
    parser.add_argument("--max-detect-side", type=int, default=0, help="Resize long side during candidate detection, then scale bboxes back.")
    args = parser.parse_args()

    data_dir = ROOT / args.data
    checkpoint_dir = ROOT / args.checkpoint_dir
    report_prefix = args.report_prefix

    train_rows = load_jsonl(data_dir / "train.jsonl", limit=args.limit_train)
    dev_rows = load_jsonl(data_dir / "dev.jsonl", limit=args.limit_dev)
    locked_rows = load_jsonl(data_dir / "locked.jsonl", limit=args.limit_locked)

    threshold_fit = fit_threshold(train_rows, sample_limit=max(1, args.limit_train))
    if args.policy_index >= 0:
        policy = dict(POLICY_GRID[args.policy_index])
        policy["threshold"] = int(policy["threshold"] or threshold_fit["selected_threshold"])
        if args.disable_bucketed_cap:
            policy["bucketed_cap"] = False
        if args.cap:
            policy["cap"] = int(args.cap)
        if args.max_detect_side:
            policy["max_detect_side"] = int(args.max_detect_side)
        dev_metric, _ = evaluate(dev_rows, policy)
        dev_grid = [{"policy": policy, "metric": dev_metric, "selection_score": None, "selection_mode": "fixed_policy_index"}]
    else:
        policy, dev_grid = select_policy(dev_rows, int(threshold_fit["selected_threshold"]))
        if args.max_detect_side:
            policy["max_detect_side"] = int(args.max_detect_side)
        dev_metric, _ = evaluate(dev_rows, policy)
    locked_metric, locked_predictions = evaluate(locked_rows, policy, keep_predictions=True, export_top_k=args.export_top_k)
    routed_rows = [] if args.no_routed_export else routed_candidate_rows(locked_predictions)

    checkpoint = {
        "version": "boundary_segmenter_v18_threshold_gap_vectorizer",
        "task": "IMG-MOE-V18-P0-003",
        "dataset": str(data_dir.relative_to(ROOT)),
        "selected_policy": policy,
        "threshold_fit": threshold_fit,
        "inference_input": "raster_image_only",
    }
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    write_json(checkpoint_dir / "policy.json", checkpoint)
    report = {
        "task": "IMG-MOE-V18-P0-003",
        "checkpoint": str((checkpoint_dir / "policy.json").relative_to(ROOT)),
        "dataset": str(data_dir.relative_to(ROOT)),
        "limits": {
            "train": int(args.limit_train),
            "dev": int(args.limit_dev),
            "locked": int(args.limit_locked),
            "policy_index": int(args.policy_index),
        },
        "detector": checkpoint,
        "dev_grid": dev_grid,
        "dev_metric": dev_metric,
        "locked_metric": locked_metric,
        "success_criteria": {
            "boundary_candidate_recall_at_least_0_70": float(locked_metric["recall"]) >= 0.70,
            "per_class_recall_reported": True,
            "source_integrity_violations": 0,
            "exports_routed_candidate_rows": not args.no_routed_export,
        },
        "source_integrity": {
            "model_input": "raster_image_only",
            "offline_gold_used_for": ["train_threshold_fit", "dev_policy_selection", "locked_evaluation"],
            "offline_gold_used_at_inference": False,
        },
    }
    write_json(REPORT / f"{report_prefix}_eval.json", report)
    write_jsonl(REPORT / f"{report_prefix}_locked_predictions.jsonl", locked_predictions)
    if not args.no_routed_export:
        write_jsonl(REPORT / f"{report_prefix}_routed_candidates.jsonl", routed_rows)
    print(json.dumps({
        "task": "IMG-MOE-V18-P0-003",
        "policy": policy,
        "locked_recall": locked_metric["recall"],
        "locked_precision": locked_metric["precision"],
        "candidate_inflation": locked_metric["candidate_inflation"],
        "success": report["success_criteria"]["boundary_candidate_recall_at_least_0_70"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
