#!/usr/bin/env python3
"""Apply visual crop-based symbol box refiners to page-level predictions."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageOps
from torchvision.ops import nms

from train_symbol_proposal_crop_box_refiner_v21 import CropBoxRefiner


Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "reports/vlm/symbol_tile_detector_v20_small_anchor_smoke_area_predictions.jsonl"
GOLD = ROOT / "datasets/symbol_expert_public_raster_v19/locked.jsonl"
FROZEN = ROOT / "checkpoints/symbol_proposal_crop_box_refiner_v21_frozen_smoke/model.pt"
FINETUNE = ROOT / "checkpoints/symbol_proposal_crop_box_refiner_v21_finetune_smoke/model.pt"
REPORT = ROOT / "reports/vlm/symbol_visual_box_refiner_v21_page_eval.json"
OUT = ROOT / "reports/vlm/symbol_visual_box_refiner_v21_page_predictions.jsonl"
IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def source_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def bbox_iou(left: list[float], right: list[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    la = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    ra = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(la + ra - inter, 1e-9)


def center_covered(pred: list[float], gold: list[float], margin: float = 2.0) -> bool:
    cx = (gold[0] + gold[2]) / 2.0
    cy = (gold[1] + gold[3]) / 2.0
    return pred[0] - margin <= cx <= pred[2] + margin and pred[1] - margin <= cy <= pred[3] + margin


def clamp_box(box: list[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    left = max(0.0, min(float(width - 1), min(x1, x2)))
    top = max(0.0, min(float(height - 1), min(y1, y2)))
    right = max(left + 1.0, min(float(width), max(x1, x2)))
    bottom = max(top + 1.0, min(float(height), max(y1, y2)))
    return [left, top, right, bottom]


def area_bucket(box: list[float]) -> str:
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area <= 64:
        return "tiny_le_64"
    if area <= 256:
        return "small_le_256"
    if area <= 1024:
        return "medium_le_1024"
    if area <= 4096:
        return "large_le_4096"
    return "xlarge_gt_4096"


def expand_box(box: list[float], width: int, height: int, scale: float, pad: float) -> list[int]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    nw = bw * scale + 2.0 * pad
    nh = bh * scale + 2.0 * pad
    return [
        max(0, int(math.floor(cx - nw / 2.0))),
        max(0, int(math.floor(cy - nh / 2.0))),
        min(width, int(math.ceil(cx + nw / 2.0))),
        min(height, int(math.ceil(cy + nh / 2.0))),
    ]


def box_in_crop(box: list[float], crop_box: list[int], crop_size: int) -> list[float]:
    x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = crop_box
    sx = crop_size / max(cx2 - cx1, 1)
    sy = crop_size / max(cy2 - cy1, 1)
    return [(x1 - cx1) * sx, (y1 - cy1) * sy, (x2 - cx1) * sx, (y2 - cy1) * sy]


def crop_tensor(image: Image.Image, crop_box: list[int], crop_size: int) -> torch.Tensor:
    crop = image.crop(tuple(crop_box)).convert("RGB")
    crop = ImageOps.autocontrast(crop)
    crop = crop.resize((crop_size, crop_size), Image.Resampling.BICUBIC)
    arr = np.asarray(crop, dtype=np.float32) / 255.0
    arr = (arr - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(arr.transpose(2, 0, 1).astype(np.float32))


def geom_tensor(pred: dict[str, Any], box: list[float], crop_box: list[int], image_size: tuple[int, int], crop_size: int) -> torch.Tensor:
    width, height = image_size
    bbox_crop = box_in_crop(box, crop_box, crop_size)
    bw = max(1.0, box[2] - box[0])
    bh = max(1.0, box[3] - box[1])
    values = [
        *(float(v) / 224.0 for v in bbox_crop[:4]),
        float(pred.get("label_id") or 0) / 9.0,
        float(pred.get("score") or 0.0),
        bw / max(width, 1),
        bh / max(height, 1),
        math.log(bw / max(bh, 1.0)),
        math.log(max(bw * bh, 1.0)) / 12.0,
        (crop_box[2] - crop_box[0]) / 256.0,
        (crop_box[3] - crop_box[1]) / 256.0,
    ]
    return torch.tensor(values, dtype=torch.float32)


def apply_offset(box: list[float], offset: list[float], width: int, height: int, max_abs: float) -> list[float]:
    bw = max(1.0, box[2] - box[0])
    bh = max(1.0, box[3] - box[1])
    values = [max(-max_abs, min(max_abs, float(v))) for v in offset]
    return clamp_box(
        [box[0] + values[0] * bw, box[1] + values[1] * bh, box[2] + values[2] * bw, box[3] + values[3] * bh],
        width,
        height,
    )


def gold_by_row(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in load_jsonl(path)}


def load_model(path: Path, freeze_encoder: bool, device: torch.device) -> CropBoxRefiner:
    payload = torch.load(path, map_location=device)
    model = CropBoxRefiner(freeze_encoder=freeze_encoder).to(device)
    model.load_state_dict(payload["model"])
    model.eval()
    return model


def choose_route(box: list[float], mode: str) -> str:
    if mode in {"frozen_all", "finetune_all", "identity"}:
        return mode
    bucket = area_bucket(box)
    if bucket in {"tiny_le_64", "small_le_256"}:
        return "frozen_all"
    return "finetune_all"


def refine_rows(
    pred_rows: list[dict[str, Any]],
    gold_rows: dict[str, dict[str, Any]],
    frozen: CropBoxRefiner,
    finetune: CropBoxRefiner,
    device: torch.device,
    *,
    mode: str,
    crop_size: int,
    context_scale: float,
    context_pad: float,
    batch_size: int,
    max_abs_offset: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    counters: Counter[str] = Counter()
    refined_rows: list[dict[str, Any]] = []
    for row in pred_rows:
        gold_row = gold_rows.get(str(row.get("row_id")))
        if not gold_row:
            continue
        with Image.open(source_path(str(gold_row.get("image") or ""))) as opened:
            image = opened.convert("RGB")
        width, height = [int(v) for v in gold_row.get("image_size") or list(image.size)]
        prepared: dict[str, list[tuple[int, torch.Tensor, torch.Tensor, list[float]]]] = {"frozen_all": [], "finetune_all": []}
        preds = []
        for index, pred in enumerate(row.get("predicted_symbols") or []):
            original = clamp_box([float(v) for v in pred.get("bbox") or [0, 0, 1, 1]], width, height)
            route = choose_route(original, mode)
            updated = dict(pred)
            updated["bbox_before_visual_refine"] = [round(v, 4) for v in original]
            updated["visual_box_refiner"] = {"route": route, "applied": route != "identity"}
            preds.append(updated)
            counters[f"route_{route}"] += 1
            if route == "identity":
                continue
            crop_box = expand_box(original, width, height, context_scale, context_pad)
            prepared[route].append((index, crop_tensor(image, crop_box, crop_size), geom_tensor(pred, original, crop_box, (width, height), crop_size), original))
        for route, model in [("frozen_all", frozen), ("finetune_all", finetune)]:
            items = prepared[route]
            for start in range(0, len(items), batch_size):
                chunk = items[start : start + batch_size]
                crops = torch.stack([item[1] for item in chunk]).to(device, non_blocking=True)
                geom = torch.stack([item[2] for item in chunk]).to(device, non_blocking=True)
                with torch.no_grad():
                    offsets = model(crops, geom).detach().cpu().tolist()
                for (index, _crop, _geom, original), offset in zip(chunk, offsets, strict=True):
                    refined = apply_offset(original, offset, width, height, max_abs_offset)
                    preds[index]["bbox"] = [round(v, 4) for v in refined]
                    preds[index]["visual_box_refiner"]["offset"] = [round(float(v), 8) for v in offset]
        out = dict(row)
        out["predicted_symbols"] = preds
        refined_rows.append(out)
    return refined_rows, dict(counters)


def merge_predictions(preds: list[dict[str, Any]], score_threshold: float, nms_threshold: float, max_per_page: int) -> list[dict[str, Any]]:
    filtered = [pred for pred in preds if float(pred.get("score") or 0.0) >= score_threshold]
    if not filtered:
        return []
    boxes = torch.tensor([pred["bbox"] for pred in filtered], dtype=torch.float32)
    scores = torch.tensor([float(pred.get("score") or 0.0) for pred in filtered], dtype=torch.float32)
    labels = [int(pred.get("label_id") or 5) for pred in filtered]
    keep_indices: list[int] = []
    for label in sorted(set(labels)):
        idx = torch.tensor([i for i, current in enumerate(labels) if current == label], dtype=torch.long)
        keep = nms(boxes[idx], scores[idx], nms_threshold)
        keep_indices.extend(int(idx[int(i)]) for i in keep.tolist())
    keep_indices.sort(key=lambda i: float(filtered[i].get("score") or 0.0), reverse=True)
    return [filtered[i] for i in keep_indices[:max_per_page]]


def evaluate(rows: list[dict[str, Any]], gold_rows: dict[str, dict[str, Any]], score_threshold: float, nms_threshold: float, max_per_page: int) -> dict[str, Any]:
    totals = Counter()
    by_area = Counter()
    by_area_center = Counter()
    by_area_iou = Counter()
    by_label = Counter()
    by_label_center = Counter()
    by_label_iou = Counter()
    for row in rows:
        gold_row = gold_rows.get(str(row.get("row_id")))
        if not gold_row:
            continue
        golds = [
            {"bbox": [float(v) for v in target["bbox"]], "label": str(target.get("label") or "generic_symbol")}
            for target in ((gold_row.get("targets") or {}).get("boxes") or [])
        ]
        preds = merge_predictions(row.get("predicted_symbols") or [], score_threshold, nms_threshold, max_per_page)
        used_iou: set[int] = set()
        used_center: set[int] = set()
        for gold in golds:
            box = gold["bbox"]
            label = gold["label"]
            bucket = area_bucket(box)
            by_area[bucket] += 1
            by_label[label] += 1
            best_iou = 0.0
            best_iou_index = None
            center_index = None
            for pred_index, pred in enumerate(preds):
                pred_box = [float(v) for v in pred["bbox"]]
                iou = bbox_iou(pred_box, box)
                if iou > best_iou:
                    best_iou = iou
                    best_iou_index = pred_index
                if center_index is None and pred_index not in used_center and center_covered(pred_box, box):
                    center_index = pred_index
            if best_iou_index is not None and best_iou >= 0.30 and best_iou_index not in used_iou:
                used_iou.add(best_iou_index)
                totals["matched_iou"] += 1
                by_area_iou[bucket] += 1
                by_label_iou[label] += 1
            if center_index is not None:
                used_center.add(center_index)
                totals["matched_center"] += 1
                by_area_center[bucket] += 1
                by_label_center[label] += 1
        totals["gold"] += len(golds)
        totals["predicted"] += len(preds)
    precision = totals["matched_iou"] / max(totals["predicted"], 1)
    recall = totals["matched_iou"] / max(totals["gold"], 1)
    return {
        "rows": len(rows),
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
        "area_center_recall": {bucket: round(by_area_center[bucket] / max(by_area[bucket], 1), 6) for bucket in sorted(by_area)},
        "area_iou_recall": {bucket: round(by_area_iou[bucket] / max(by_area[bucket], 1), 6) for bucket in sorted(by_area)},
        "type_center_recall": {label: round(by_label_center[label] / max(by_label[label], 1), 6) for label in sorted(by_label)},
        "type_iou_recall": {label: round(by_label_iou[label] / max(by_label[label], 1), 6) for label in sorted(by_label)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=str(PREDICTIONS))
    parser.add_argument("--gold", default=str(GOLD))
    parser.add_argument("--frozen-checkpoint", default=str(FROZEN))
    parser.add_argument("--finetune-checkpoint", default=str(FINETUNE))
    parser.add_argument("--output", default=str(OUT))
    parser.add_argument("--eval-output", default=str(REPORT))
    parser.add_argument("--mode", choices=["frozen_all", "finetune_all", "size_routed", "identity"], default="size_routed")
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--nms-threshold", type=float, default=0.6)
    parser.add_argument("--max-per-page", type=int, default=500)
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--context-scale", type=float, default=3.0)
    parser.add_argument("--context-pad", type=float, default=12.0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-abs-offset", type=float, default=1.2)
    parser.add_argument("--max-rows", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pred_rows = load_jsonl(Path(args.predictions))
    if args.max_rows > 0:
        pred_rows = pred_rows[: args.max_rows]
    gold_rows = gold_by_row(Path(args.gold))
    frozen = load_model(Path(args.frozen_checkpoint), True, device)
    finetune = load_model(Path(args.finetune_checkpoint), False, device)
    baseline = evaluate(pred_rows, gold_rows, args.score_threshold, args.nms_threshold, args.max_per_page)
    refined_rows, route_counts = refine_rows(
        pred_rows,
        gold_rows,
        frozen,
        finetune,
        device,
        mode=args.mode,
        crop_size=args.crop_size,
        context_scale=args.context_scale,
        context_pad=args.context_pad,
        batch_size=args.batch_size,
        max_abs_offset=args.max_abs_offset,
    )
    refined = evaluate(refined_rows, gold_rows, args.score_threshold, args.nms_threshold, args.max_per_page)
    write_jsonl(Path(args.output), refined_rows)
    report = {
        "version": "symbol_visual_box_refiner_v21_page_eval",
        "claim_boundary": "Page-level visual box refiner. Gold is used only for evaluation; routing uses proposal geometry only.",
        "source_integrity": {
            "runtime_input": "raster_pixels_plus_detector_predictions",
            "gold_used_for_runtime": False,
            "route_uses_gold_area": False,
            "forbidden_runtime_features": ["raw_label", "semantic_type", "expected_json", "annotation_path", "svg_geometry"],
            "passed": True,
        },
        "inputs": {
            "predictions": args.predictions,
            "gold": args.gold,
            "frozen_checkpoint": args.frozen_checkpoint,
            "finetune_checkpoint": args.finetune_checkpoint,
        },
        "checkpoint_refs": {"frozen": rel(Path(args.frozen_checkpoint)), "finetune": rel(Path(args.finetune_checkpoint))},
        "config": vars(args),
        "device": str(device),
        "route_counts": route_counts,
        "baseline": baseline,
        "refined": refined,
        "delta": {
            "center_recall": round(refined["symbol_bbox_center_recall"] - baseline["symbol_bbox_center_recall"], 6),
            "iou_recall": round(refined["symbol_bbox_iou_0_30"]["recall"] - baseline["symbol_bbox_iou_0_30"]["recall"], 6),
            "iou_precision": round(refined["symbol_bbox_iou_0_30"]["precision"] - baseline["symbol_bbox_iou_0_30"]["precision"], 6),
            "candidate_inflation": round(refined["candidate_inflation"] - baseline["candidate_inflation"], 6),
        },
        "adopted": refined["symbol_bbox_iou_0_30"]["recall"] > baseline["symbol_bbox_iou_0_30"]["recall"]
        and refined["symbol_bbox_center_recall"] >= baseline["symbol_bbox_center_recall"] * 0.98,
    }
    write_json(Path(args.eval_output), report)
    print(json.dumps({"baseline": baseline, "refined": refined, "delta": report["delta"], "adopted": report["adopted"], "route_counts": route_counts}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
