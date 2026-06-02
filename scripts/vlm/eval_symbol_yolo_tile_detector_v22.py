#!/usr/bin/env python3
"""Evaluate a YOLO tile detector with the existing page-level symbol metrics."""

from __future__ import annotations

import argparse
import json
import random
import resource
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import torch
from ultralytics import RTDETR, YOLO

from train_symbol_tile_detector_v20 import (
    FORBIDDEN_RUNTIME_FIELDS,
    ID_TO_LABEL,
    LABELS,
    area_bucket,
    bbox_iou,
    center_covered,
    load_jsonl,
    nwd_similarity,
    rel,
    target_area_buckets,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
WEIGHTS = ROOT / "runs/detect/runs/vlm/symbol_yolo_p2_v22_smoke/weights/best.pt"
REPORT = ROOT / "reports/vlm/symbol_yolo_p2_v22_smoke_page_eval.json"
BASELINE_CENTER_RECALL = 0.851394
BASELINE_TINY_IOU_RECALL = 0.393013
BASELINE_CANDIDATE_INFLATION = 7.919152


def sample_tiles_area_aware(rows: list[dict[str, Any]], limit: int | None, seed: int, positive_ratio: float, small_positive_ratio: float) -> list[dict[str, Any]]:
    if not limit or len(rows) <= limit:
        return list(rows)
    rng = random.Random(seed)
    positives = [row for row in rows if int((row.get("target_counts") or {}).get("symbols") or 0) > 0]
    empties = [row for row in rows if int((row.get("target_counts") or {}).get("symbols") or 0) == 0]
    small = [row for row in positives if target_area_buckets(row) & {"tiny_le_64", "small_le_256"}]
    small_ids = {id(row) for row in small}
    other = [row for row in positives if id(row) not in small_ids]
    for group in (small, other, empties):
        rng.shuffle(group)
    pos_n = min(len(positives), int(limit * positive_ratio))
    small_n = min(len(small), int(pos_n * small_positive_ratio))
    selected = small[:small_n] + other[: max(0, pos_n - small_n)]
    if len(selected) < pos_n:
        selected.extend(small[small_n : small_n + (pos_n - len(selected))])
    selected.extend(empties[: max(0, limit - len(selected))])
    if len(selected) < limit:
        used = {id(row) for row in selected}
        leftovers = [row for row in rows if id(row) not in used]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: limit - len(selected)])
    rng.shuffle(selected)
    return selected[:limit]


def image_path_for_yolo_tile(yolo_dir: Path, split: str, row: dict[str, Any]) -> Path:
    if split == "dev":
        candidate_splits = ["val"]
    elif split.startswith("smoke"):
        candidate_splits = [split, "smoke", "smoke_v30", "locked"]
    else:
        candidate_splits = [split]
    for yolo_split in candidate_splits:
        path = yolo_dir / "images" / yolo_split / f"{row['id']}.jpg"
        if path.exists():
            return path
    return yolo_dir / "images" / candidate_splits[0] / f"{row['id']}.jpg"


def filter_rows_with_exported_images(rows: list[dict[str, Any]], split: str, yolo_dir: Path) -> list[dict[str, Any]]:
    return [row for row in rows if image_path_for_yolo_tile(yolo_dir, split, row).exists()]


def collect_yolo_predictions(model: YOLO, rows: list[dict[str, Any]], split: str, yolo_dir: Path, args: argparse.Namespace) -> tuple[dict[str, list[dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    page_preds: dict[str, list[dict[str, Any]]] = defaultdict(list)
    page_golds: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    paths = [str(image_path_for_yolo_tile(yolo_dir, split, row)) for row in rows]
    for row in rows:
        row_id = str(row.get("row_id"))
        for gold in ((row.get("targets") or {}).get("boxes") or []):
            target_id = str(gold.get("target_id") or f"{row_id}_{len(page_golds[row_id])}")
            page_golds[row_id][target_id] = {
                "target_id": target_id,
                "bbox": [float(v) for v in gold.get("page_bbox") or gold.get("bbox")],
                "label": str(gold.get("label") or "generic_symbol"),
            }
    batch_size = max(1, int(args.predict_batch))
    for start in range(0, len(rows), batch_size):
        chunk_rows = rows[start : start + batch_size]
        chunk_paths = paths[start : start + batch_size]
        results = model.predict(
            source=chunk_paths,
            imgsz=args.imgsz,
            conf=args.decode_conf,
            iou=args.decode_iou,
            max_det=args.max_det_per_tile,
            device=args.device,
            batch=batch_size,
            stream=False,
            verbose=False,
        )
        for row, result in zip(chunk_rows, results, strict=True):
            left, top, _right, _bottom = [float(v) for v in (row.get("tile") or {}).get("bbox")]
            row_id = str(row.get("row_id"))
            if result.boxes is None:
                continue
            xyxy = result.boxes.xyxy.detach().cpu().tolist()
            confs = result.boxes.conf.detach().cpu().tolist()
            classes = result.boxes.cls.detach().cpu().tolist()
            for box, conf, cls in zip(xyxy, confs, classes, strict=True):
                label_id = int(cls) + 1
                if label_id not in ID_TO_LABEL:
                    continue
                page_preds[row_id].append(
                    {
                        "bbox": [float(box[0] + left), float(box[1] + top), float(box[2] + left), float(box[3] + top)],
                        "label_id": label_id,
                        "label": ID_TO_LABEL[label_id],
                        "score": float(conf),
                        "tile_id": row.get("id"),
                    }
                )
    return page_preds, page_golds


def merge_predictions(preds: list[dict[str, Any]], score_threshold: float, nms_threshold: float, max_per_page: int) -> list[dict[str, Any]]:
    filtered = [pred for pred in preds if float(pred["score"]) >= score_threshold]
    if not filtered:
        return []
    boxes = torch.tensor([pred["bbox"] for pred in filtered], dtype=torch.float32)
    scores = torch.tensor([float(pred["score"]) for pred in filtered], dtype=torch.float32)
    labels = [int(pred["label_id"]) for pred in filtered]
    keep_indices: list[int] = []
    from torchvision.ops import nms

    for label in sorted(set(labels)):
        idx = torch.tensor([i for i, current in enumerate(labels) if current == label], dtype=torch.long)
        keep = nms(boxes[idx], scores[idx], nms_threshold)
        keep_indices.extend(int(idx[int(i)]) for i in keep.tolist())
    keep_indices.sort(key=lambda i: float(filtered[i]["score"]), reverse=True)
    return [filtered[i] for i in keep_indices[:max_per_page]]


def passes_baseline_gate(metrics: dict[str, Any]) -> bool:
    return (
        float(metrics["symbol_bbox_center_recall"]) > BASELINE_CENTER_RECALL
        and float(metrics["area_iou_recall"].get("tiny_le_64", 0.0)) > BASELINE_TINY_IOU_RECALL
        and float(metrics["candidate_inflation"]) <= BASELINE_CANDIDATE_INFLATION
    )


def selection_key(row: dict[str, Any], mode: str = "recall_gate") -> tuple[float, ...]:
    metrics = row["metrics"]
    iou_metrics = metrics["symbol_bbox_iou_0_30"]
    precision = float(iou_metrics["precision"])
    recall = float(iou_metrics["recall"])
    f1 = float(iou_metrics["f1"])
    center_recall = float(metrics["symbol_bbox_center_recall"])
    tiny_iou = float(metrics["area_iou_recall"].get("tiny_le_64", 0.0))
    inflation = float(metrics["candidate_inflation"])
    gate = 1.0 if passes_baseline_gate(metrics) else 0.0
    if mode == "balanced_f1":
        return (f1, precision, recall, center_recall, -inflation)
    if mode == "precision_at_recall60":
        return (1.0 if recall >= 0.60 else 0.0, precision, f1, recall, -inflation)
    if mode == "low_inflation_at_recall60":
        return (1.0 if recall >= 0.60 else 0.0, -inflation, f1, precision, recall)
    return (gate, tiny_iou, recall, center_recall, -inflation)


def score_predictions(
    page_preds: dict[str, list[dict[str, Any]]],
    page_golds: dict[str, dict[str, dict[str, Any]]],
    score_threshold: float,
    nms_threshold: float,
    max_per_page: int,
    tile_count: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    predictions: list[dict[str, Any]] = []
    totals = Counter()
    by_label = Counter()
    by_label_center = Counter()
    by_label_iou = Counter()
    by_area = Counter()
    by_area_center = Counter()
    by_area_iou = Counter()
    by_area_nwd_070 = Counter()
    typed_correct = 0
    for row_id, gold_map in page_golds.items():
        merged = merge_predictions(page_preds.get(row_id, []), score_threshold, nms_threshold, max_per_page)
        used_iou: set[int] = set()
        used_center: set[int] = set()
        for gold in gold_map.values():
            gold_box = [float(v) for v in gold["bbox"]]
            label = str(gold["label"])
            bucket = area_bucket(gold_box)
            by_label[label] += 1
            by_area[bucket] += 1
            best_iou = 0.0
            best_iou_index: int | None = None
            best_nwd = 0.0
            center_index: int | None = None
            for pred_index, pred in enumerate(merged):
                pred_box = [float(v) for v in pred["bbox"]]
                iou = bbox_iou(pred_box, gold_box)
                if iou > best_iou:
                    best_iou = iou
                    best_iou_index = pred_index
                best_nwd = max(best_nwd, nwd_similarity(pred_box, gold_box))
                if center_index is None and pred_index not in used_center and center_covered(pred_box, gold_box):
                    center_index = pred_index
            if best_nwd >= 0.70:
                by_area_nwd_070[bucket] += 1
            if best_iou_index is not None and best_iou >= 0.30 and best_iou_index not in used_iou:
                used_iou.add(best_iou_index)
                totals["matched_iou"] += 1
                by_label_iou[label] += 1
                by_area_iou[bucket] += 1
                if merged[best_iou_index]["label"] == label:
                    typed_correct += 1
            if center_index is not None:
                used_center.add(center_index)
                totals["matched_center"] += 1
                by_label_center[label] += 1
                by_area_center[bucket] += 1
        totals["gold"] += len(gold_map)
        totals["predicted"] += len(merged)
        predictions.append({"row_id": row_id, "predicted_symbols": merged, "gold_symbol_count": len(gold_map)})
    precision = totals["matched_iou"] / max(totals["predicted"], 1)
    recall = totals["matched_iou"] / max(totals["gold"], 1)
    return {
        "rows": len(page_golds),
        "tiles": tile_count,
        "symbol_bbox_iou_0_30": {
            "matched": int(totals["matched_iou"]),
            "predicted": int(totals["predicted"]),
            "gold": int(totals["gold"]),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall), 6),
        },
        "symbol_bbox_center_recall": round(totals["matched_center"] / max(totals["gold"], 1), 6),
        "candidate_inflation": round(totals["predicted"] / max(totals["gold"], 1), 6),
        "typed_accuracy_on_iou_matches": round(typed_correct / max(totals["matched_iou"], 1), 6),
        "type_center_recall": {label: round(by_label_center[label] / max(by_label[label], 1), 6) for label in sorted(by_label)},
        "type_iou_recall": {label: round(by_label_iou[label] / max(by_label[label], 1), 6) for label in sorted(by_label)},
        "area_center_recall": {bucket: round(by_area_center[bucket] / max(by_area[bucket], 1), 6) for bucket in sorted(by_area)},
        "area_iou_recall": {bucket: round(by_area_iou[bucket] / max(by_area[bucket], 1), 6) for bucket in sorted(by_area)},
        "nwd_tiny_box_audit": {"area_recall_at_0_70": {bucket: round(by_area_nwd_070[bucket] / max(by_area[bucket], 1), 6) for bucket in sorted(by_area)}},
    }, predictions


def memory_audit() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"max_rss_kb": int(usage.ru_maxrss)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--yolo-dir", default=str(ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_v22"))
    parser.add_argument("--weights", default=str(WEIGHTS))
    parser.add_argument("--split", default="locked")
    parser.add_argument("--eval-output", default=str(REPORT))
    parser.add_argument("--predictions-output", default=str(ROOT / "reports/vlm/symbol_yolo_p2_v22_smoke_page_predictions.jsonl"))
    parser.add_argument("--limit-tiles", type=int, default=2000)
    parser.add_argument("--positive-ratio", type=float, default=0.85)
    parser.add_argument("--small-positive-ratio", type=float, default=0.75)
    parser.add_argument("--imgsz", type=int, default=384)
    parser.add_argument("--decode-conf", type=float, default=0.001)
    parser.add_argument("--decode-iou", type=float, default=0.7)
    parser.add_argument("--max-det-per-tile", type=int, default=300)
    parser.add_argument("--predict-batch", type=int, default=16)
    parser.add_argument("--score-threshold-grid", default="0.001,0.005,0.01,0.02,0.05")
    parser.add_argument("--nms-threshold-grid", default="0.45,0.55,0.65,0.75")
    parser.add_argument("--max-per-page", type=int, default=500)
    parser.add_argument("--device", default="0")
    parser.add_argument("--seed", type=int, default=20260510)
    parser.add_argument(
        "--selection-mode",
        choices=["recall_gate", "balanced_f1", "precision_at_recall60", "low_inflation_at_recall60"],
        default="recall_gate",
        help="Operating-point selection policy. recall_gate preserves the historical high-recall default.",
    )
    args = parser.parse_args()

    if args.split not in {"dev", "locked"} and not args.split.startswith("smoke"):
        raise ValueError(f"unsupported split={args.split}; expected dev, locked, or smoke*")
    split_rows = load_jsonl(Path(args.data) / f"{args.split}.jsonl")
    exported_rows = filter_rows_with_exported_images(split_rows, args.split, Path(args.yolo_dir))
    rows = sample_tiles_area_aware(
        exported_rows,
        args.limit_tiles,
        args.seed + (2 if args.split == "locked" else 3 if args.split.startswith("smoke") else 1),
        args.positive_ratio,
        args.small_positive_ratio,
    )
    if not rows:
        raise RuntimeError(f"no exported YOLO images found for split={args.split} under {args.yolo_dir}")
    model = RTDETR(args.weights) if "rtdetr" in str(args.weights).lower() else YOLO(args.weights)
    page_preds, page_golds = collect_yolo_predictions(model, rows, args.split, Path(args.yolo_dir), args)
    score_grid = [float(item) for item in args.score_threshold_grid.split(",") if item.strip()]
    nms_grid = [float(item) for item in args.nms_threshold_grid.split(",") if item.strip()]
    grid_reports: list[dict[str, Any]] = []
    for score_threshold in score_grid:
        for nms_threshold in nms_grid:
            metrics, _ = score_predictions(page_preds, page_golds, score_threshold, nms_threshold, args.max_per_page, len(rows))
            grid_reports.append({"score_threshold": score_threshold, "nms_threshold": nms_threshold, "metrics": metrics})
    grid_reports.sort(key=lambda row: selection_key(row, args.selection_mode), reverse=True)
    selected = grid_reports[0]
    locked_eval, predictions = score_predictions(
        page_preds,
        page_golds,
        float(selected["score_threshold"]),
        float(selected["nms_threshold"]),
        args.max_per_page,
        len(rows),
    )
    report = {
        "version": "symbol_yolo_tile_detector_v22_page_eval",
        "claim_boundary": "YOLO/P2 tile detector evaluated with existing page-level symbol metrics.",
        "source_integrity": {
            "model_input": "raster_tile_pixels_only",
            "raw_label_or_semantic_type_used_as_runtime_feature": False,
            "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS,
            "locked_gold_use": "evaluation_only",
        },
        "baseline_to_beat": {
            "scaled_faster_rcnn_center_recall": BASELINE_CENTER_RECALL,
            "scaled_faster_rcnn_tiny_iou_recall": BASELINE_TINY_IOU_RECALL,
            "scaled_faster_rcnn_candidate_inflation": BASELINE_CANDIDATE_INFLATION,
        },
        "dataset": rel(Path(args.data)),
        "yolo_dir": rel(Path(args.yolo_dir)),
        "weights": rel(Path(args.weights)),
        "evaluation_mode": "smoke_fast_subset" if args.split.startswith("smoke") else "locked_or_dev_full_subset",
        "metric_claim_boundary": "Smoke metrics are for fast regression checks only and must not be reported as final model quality." if args.split.startswith("smoke") else "Non-smoke metrics may be used for locked/dev comparison according to split and limit-tiles.",
        "config": vars(args),
        "threshold_grid": grid_reports,
        "selection_mode": args.selection_mode,
        "selected_thresholds": {"score_threshold": float(selected["score_threshold"]), "nms_threshold": float(selected["nms_threshold"])},
        args.split: locked_eval,
        "gate": {
            "beats_scaled_faster_rcnn_center_0_851394": locked_eval["symbol_bbox_center_recall"] > BASELINE_CENTER_RECALL,
            "beats_scaled_faster_rcnn_tiny_iou_0_393013": locked_eval["area_iou_recall"].get("tiny_le_64", 0.0) > BASELINE_TINY_IOU_RECALL,
            "candidate_inflation_lte_7_919152": locked_eval["candidate_inflation"] <= BASELINE_CANDIDATE_INFLATION,
        },
        "memory_audit": memory_audit(),
    }
    report["gate"]["passed"] = all(bool(value) for value in report["gate"].values())
    write_json(Path(args.eval_output), report)
    write_jsonl(Path(args.predictions_output), predictions)
    print(json.dumps({args.split: locked_eval, "gate": report["gate"], "selected": report["selected_thresholds"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
