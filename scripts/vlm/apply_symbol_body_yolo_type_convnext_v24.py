#!/usr/bin/env python3
"""Apply the ConvNeXt symbol type head to YOLO body proposals for audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps

from eval_symbol_yolo_tile_detector_v22 import (
    filter_rows_with_exported_images,
    sample_tiles_area_aware,
    score_predictions,
)
from train_symbol_crop_context_pretrained_v20 import (
    CROP_VIEWS,
    IMAGENET_MEAN,
    IMAGENET_STD,
    LABELS,
    SharedPretrainedTypeHead,
)
from train_symbol_tile_detector_v20 import (
    FORBIDDEN_RUNTIME_FIELDS,
    bbox_iou,
    load_jsonl,
    rel,
    write_json,
    write_jsonl,
)


ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
YOLO_DIR = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_v22"
BODY_PREDICTIONS = ROOT / "reports/vlm/symbol_yolov8n_pretrained_v22_dedup_hi640_probe_page_predictions.jsonl"
TYPE_CHECKPOINT = ROOT / "checkpoints/symbol_crop_context_pretrained_v20_convnext_tiny_finetune/model.pt"
REPORT = ROOT / "reports/vlm/symbol_yolo_convnext_two_stage_v24_eval.json"
PREDICTIONS = ROOT / "reports/vlm/symbol_yolo_convnext_two_stage_v24_predictions.jsonl"


def clamp_box(box: list[float], width: int, height: int) -> tuple[int, int, int, int]:
    left = max(0, min(width - 1, int(round(float(box[0])))))
    top = max(0, min(height - 1, int(round(float(box[1])))))
    right = max(left + 1, min(width, int(round(float(box[2])))))
    bottom = max(top + 1, min(height, int(round(float(box[3])))))
    return left, top, right, bottom


def expand_box(box: list[float], width: int, height: int, scale: float) -> tuple[int, int, int, int]:
    cx = (float(box[0]) + float(box[2])) / 2.0
    cy = (float(box[1]) + float(box[3])) / 2.0
    bw = max(1.0, float(box[2]) - float(box[0]))
    bh = max(1.0, float(box[3]) - float(box[1]))
    expanded = [cx - bw * scale / 2.0, cy - bh * scale / 2.0, cx + bw * scale / 2.0, cy + bh * scale / 2.0]
    return clamp_box(expanded, width, height)


def preprocess_crop(image: Image.Image, box: tuple[int, int, int, int], size: int) -> np.ndarray:
    crop = ImageOps.autocontrast(image.crop(box).convert("RGB"))
    crop = crop.resize((size, size), Image.Resampling.BICUBIC)
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return arr.transpose(2, 0, 1)


def geom_tensor(box: list[float], width: int, height: int) -> list[float]:
    left, top, right, bottom = [float(v) for v in box]
    bw = max(1.0, right - left)
    bh = max(1.0, bottom - top)
    area = bw * bh
    return [
        left / max(width, 1),
        top / max(height, 1),
        right / max(width, 1),
        bottom / max(height, 1),
        ((left + right) / 2.0) / max(width, 1),
        ((top + bottom) / 2.0) / max(height, 1),
        bw / max(width, 1),
        bh / max(height, 1),
        area / max(width * height, 1),
        float(np.log(max(bw / max(bh, 1.0), 1e-6))),
    ]


def build_eval_contract(rows: list[dict[str, Any]]) -> tuple[dict[str, Path], dict[str, dict[str, dict[str, Any]]]]:
    page_images: dict[str, Path] = {}
    page_golds: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in rows:
        row_id = str(row.get("row_id"))
        page_images[row_id] = Path(str(row.get("image")))
        for gold in ((row.get("targets") or {}).get("boxes") or []):
            target_id = str(gold.get("target_id") or f"{row_id}_{len(page_golds[row_id])}")
            page_golds[row_id][target_id] = {
                "target_id": target_id,
                "bbox": [float(v) for v in gold.get("page_bbox") or gold.get("bbox")],
                "label": str(gold.get("label") or "generic_symbol"),
            }
    return page_images, page_golds


def load_body_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in load_jsonl(path):
        by_page[str(row.get("row_id"))].extend(row.get("predicted_symbols") or [])
    return by_page


def classify_predictions(
    model: torch.nn.Module,
    page_images: dict[str, Path],
    page_preds: dict[str, list[dict[str, Any]]],
    size: int,
    batch_size: int,
    device: torch.device,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    typed: dict[str, list[dict[str, Any]]] = defaultdict(list)
    pending_crops: list[np.ndarray] = []
    pending_geom: list[list[float]] = []
    pending_meta: list[tuple[str, dict[str, Any]]] = []
    counts = Counter()

    def flush() -> None:
        nonlocal pending_crops, pending_geom, pending_meta
        if not pending_crops:
            return
        crops = torch.from_numpy(np.stack(pending_crops).astype(np.float32)).to(device)
        geom = torch.tensor(pending_geom, dtype=torch.float32, device=device)
        with torch.no_grad():
            probs = torch.softmax(model(crops, geom), dim=1)
            conf, indices = probs.max(dim=1)
        for (row_id, pred), score, index in zip(pending_meta, conf.detach().cpu().tolist(), indices.detach().cpu().tolist(), strict=True):
            new_pred = dict(pred)
            new_pred["body_label_before_type_head"] = pred.get("label")
            new_pred["body_label_id_before_type_head"] = pred.get("label_id")
            new_pred["type_head_label"] = LABELS[int(index)]
            new_pred["type_head_label_id"] = int(index) + 1
            new_pred["type_head_confidence"] = float(score)
            new_pred["label"] = LABELS[int(index)]
            new_pred["label_id"] = int(index) + 1
            typed[row_id].append(new_pred)
            counts["typed_predictions"] += 1
        pending_crops = []
        pending_geom = []
        pending_meta = []

    model.eval()
    for row_id, preds in sorted(page_preds.items()):
        image_path = page_images.get(row_id)
        if image_path is None or not image_path.exists():
            counts["missing_page_image"] += len(preds)
            continue
        with Image.open(image_path) as image:
            image = image.convert("RGB")
            width, height = image.size
            for pred in preds:
                box = [float(v) for v in pred.get("bbox") or []]
                if len(box) != 4:
                    counts["bad_box"] += 1
                    continue
                views = [
                    preprocess_crop(image, clamp_box(box, width, height), size),
                    preprocess_crop(image, expand_box(box, width, height, 1.6), size),
                    preprocess_crop(image, expand_box(box, width, height, 3.0), size),
                ]
                if len(views) != len(CROP_VIEWS):
                    raise RuntimeError("crop view contract mismatch")
                pending_crops.append(np.stack(views))
                pending_geom.append(geom_tensor(box, width, height))
                pending_meta.append((row_id, pred))
                if len(pending_crops) >= batch_size:
                    flush()
    flush()
    return typed, dict(counts)


def matched_type_audit(page_preds: dict[str, list[dict[str, Any]]], page_golds: dict[str, dict[str, dict[str, Any]]], thresholds: list[float]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for threshold in thresholds:
        totals = Counter()
        per_label = {label: Counter() for label in LABELS}
        for row_id, gold_map in page_golds.items():
            preds = page_preds.get(row_id, [])
            used: set[int] = set()
            for gold in gold_map.values():
                gold_box = [float(v) for v in gold["bbox"]]
                label = str(gold["label"])
                best_index = None
                best_iou = 0.0
                for index, pred in enumerate(preds):
                    if index in used:
                        continue
                    iou = bbox_iou([float(v) for v in pred["bbox"]], gold_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_index = index
                totals["gold"] += 1
                per_label[label]["gold"] += 1
                if best_index is None or best_iou < 0.30:
                    continue
                used.add(best_index)
                pred = preds[best_index]
                totals["body_matched"] += 1
                per_label[label]["body_matched"] += 1
                if float(pred.get("type_head_confidence") or 0.0) >= threshold:
                    totals["typed_kept"] += 1
                    per_label[label]["typed_kept"] += 1
                    if pred.get("type_head_label") == label:
                        totals["typed_correct"] += 1
                        per_label[label]["typed_correct"] += 1
        rows.append(
            {
                "threshold": threshold,
                "body_matched": int(totals["body_matched"]),
                "typed_kept": int(totals["typed_kept"]),
                "typed_correct": int(totals["typed_correct"]),
                "coverage_on_body_matches": round(totals["typed_kept"] / max(totals["body_matched"], 1), 6),
                "typed_accuracy_on_kept": round(totals["typed_correct"] / max(totals["typed_kept"], 1), 6),
                "typed_recall_on_gold": round(totals["typed_correct"] / max(totals["gold"], 1), 6),
                "per_label": {
                    label: {
                        "coverage_on_body_matches": round(per_label[label]["typed_kept"] / max(per_label[label]["body_matched"], 1), 6),
                        "typed_accuracy_on_kept": round(per_label[label]["typed_correct"] / max(per_label[label]["typed_kept"], 1), 6),
                        "typed_recall_on_gold": round(per_label[label]["typed_correct"] / max(per_label[label]["gold"], 1), 6),
                        "gold": int(per_label[label]["gold"]),
                    }
                    for label in LABELS
                },
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--yolo-dir", default=str(YOLO_DIR))
    parser.add_argument("--body-predictions", default=str(BODY_PREDICTIONS))
    parser.add_argument("--type-checkpoint", default=str(TYPE_CHECKPOINT))
    parser.add_argument("--eval-output", default=str(REPORT))
    parser.add_argument("--predictions-output", default=str(PREDICTIONS))
    parser.add_argument("--split", default="locked", choices=["dev", "locked"])
    parser.add_argument("--limit-tiles", type=int, default=2000)
    parser.add_argument("--positive-ratio", type=float, default=0.85)
    parser.add_argument("--small-positive-ratio", type=float, default=0.75)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--seed", type=int, default=20260510)
    args = parser.parse_args()

    exported_rows = filter_rows_with_exported_images(load_jsonl(Path(args.data) / f"{args.split}.jsonl"), args.split, Path(args.yolo_dir))
    rows = sample_tiles_area_aware(exported_rows, args.limit_tiles, args.seed + (2 if args.split == "locked" else 1), args.positive_ratio, args.small_positive_ratio)
    page_images, page_golds = build_eval_contract(rows)
    body_preds = load_body_predictions(Path(args.body_predictions))
    body_metrics, _ = score_predictions(body_preds, page_golds, 0.0, 1.0, 500, len(rows))

    device = torch.device(args.device if args.device == "cpu" or torch.cuda.is_available() else "cpu")
    model = SharedPretrainedTypeHead("convnext_tiny", "none", len(LABELS), freeze_encoder=False).to(device)
    model.load_state_dict(torch.load(Path(args.type_checkpoint), map_location=device))
    typed_preds, classify_audit = classify_predictions(model, page_images, body_preds, args.size, args.batch_size, device)
    typed_metrics, typed_prediction_rows = score_predictions(typed_preds, page_golds, 0.0, 1.0, 500, len(rows))
    fail_closed = matched_type_audit(typed_preds, page_golds, [0.5, 0.7, 0.85, 0.95])

    report = {
        "version": "symbol_yolo_convnext_two_stage_v24_eval",
        "claim_boundary": "Two-stage audit only: YOLO body proposals are preserved; ConvNeXt crop/context type head retypes proposals and reports fail-closed typed metrics without suppressing body candidates.",
        "source_integrity": {
            "model_input": "raster_page_pixels_cropped_by_body_proposals_plus_bbox_geometry",
            "raw_label_or_semantic_type_used_as_runtime_feature": False,
            "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS,
            "locked_gold_use": "evaluation_only",
        },
        "dataset": rel(Path(args.data)),
        "body_predictions": rel(Path(args.body_predictions)),
        "type_checkpoint": rel(Path(args.type_checkpoint)),
        "config": vars(args),
        "classify_audit": classify_audit,
        "body_only_metrics_from_input_predictions": body_metrics,
        "two_stage_all_retyped_metrics": typed_metrics,
        "fail_closed_type_audit": fail_closed,
        "gate": {
            "body_recall_preserved": abs(typed_metrics["symbol_bbox_iou_0_30"]["recall"] - body_metrics["symbol_bbox_iou_0_30"]["recall"]) <= 1e-9,
            "typed_accuracy_on_iou_matches_gte_0_96": typed_metrics["typed_accuracy_on_iou_matches"] >= 0.96,
            "fail_closed_any_threshold_accuracy_gte_0_98_coverage_gte_0_75": any(
                row["typed_accuracy_on_kept"] >= 0.98 and row["coverage_on_body_matches"] >= 0.75 for row in fail_closed
            ),
        },
    }
    report["gate"]["passed"] = all(bool(value) for value in report["gate"].values())
    write_json(Path(args.eval_output), report)
    write_jsonl(Path(args.predictions_output), typed_prediction_rows)
    print(json.dumps({"body": body_metrics, "two_stage": typed_metrics, "fail_closed": fail_closed, "gate": report["gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
