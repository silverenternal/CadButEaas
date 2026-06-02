#!/usr/bin/env python3
"""Build high-resolution proposal crops for symbol box refinement.

Detector proposals are runtime candidates. Gold boxes are used only to create
offline supervised offset targets and audit metrics.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PRED = ROOT / "reports/vlm/symbol_tile_detector_v20_small_anchor_train_proposals.jsonl"
LOCKED_PRED = ROOT / "reports/vlm/symbol_tile_detector_v20_small_anchor_smoke_area_predictions.jsonl"
TRAIN_GOLD = ROOT / "datasets/symbol_expert_public_raster_v19/train.jsonl"
LOCKED_GOLD = ROOT / "datasets/symbol_expert_public_raster_v19/locked.jsonl"
OUT = ROOT / "datasets/symbol_proposal_crop_refiner_v21"
REPORT = ROOT / "reports/vlm/symbol_proposal_crop_refiner_v21_dataset_audit.json"
LABELS = ("appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table")
FORBIDDEN_RUNTIME_FIELDS = ("raw_label", "semantic_type", "expected_json", "annotation_path", "svg_geometry")


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


def safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "item")).strip("_")
    return text[:96] or "item"


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


def center_covered(pred: list[float], gold: list[float], margin: float) -> bool:
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


def encode_offset(pred_box: list[float], gold_box: list[float]) -> list[float]:
    pw = max(1.0, pred_box[2] - pred_box[0])
    ph = max(1.0, pred_box[3] - pred_box[1])
    return [
        (gold_box[0] - pred_box[0]) / pw,
        (gold_box[1] - pred_box[1]) / ph,
        (gold_box[2] - pred_box[2]) / pw,
        (gold_box[3] - pred_box[3]) / ph,
    ]


def box_in_crop(box: list[float], crop_box: list[int], crop_size: int) -> list[float]:
    x1, y1, x2, y2 = box
    cx1, cy1, cx2, cy2 = crop_box
    sx = crop_size / max(cx2 - cx1, 1)
    sy = crop_size / max(cy2 - cy1, 1)
    return [
        round((x1 - cx1) * sx, 4),
        round((y1 - cy1) * sy, 4),
        round((x2 - cx1) * sx, 4),
        round((y2 - cy1) * sy, 4),
    ]


def crop_save(image: Image.Image, crop_box: list[int], output: Path, crop_size: int) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    crop = image.crop(tuple(crop_box)).convert("RGB")
    crop = ImageOps.autocontrast(crop)
    crop = crop.resize((crop_size, crop_size), Image.Resampling.BICUBIC)
    crop.save(output)
    return {
        "path": rel(output),
        "crop_box": [int(v) for v in crop_box],
        "size": [crop_size, crop_size],
        "source_size": [int(crop_box[2] - crop_box[0]), int(crop_box[3] - crop_box[1])],
    }


def gold_by_row(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in load_jsonl(path)}


def best_match(pred_box: list[float], golds: list[dict[str, Any]], min_iou: float, center_margin: float) -> tuple[dict[str, Any] | None, float, bool]:
    best: dict[str, Any] | None = None
    best_iou = 0.0
    best_center = False
    for gold in golds:
        box = gold["bbox"]
        iou = bbox_iou(pred_box, box)
        center_hit = center_covered(pred_box, box, center_margin)
        if iou > best_iou or (best is None and center_hit):
            best = gold
            best_iou = iou
            best_center = center_hit
    if best is None:
        return None, 0.0, False
    if best_iou >= min_iou or best_center:
        return best, best_iou, best_center
    return None, best_iou, best_center


def geometry(pred_box: list[float], gold_box: list[float], image_size: tuple[int, int], score: float) -> dict[str, Any]:
    width, height = image_size
    pw = max(1.0, pred_box[2] - pred_box[0])
    ph = max(1.0, pred_box[3] - pred_box[1])
    return {
        "proposal_bbox_norm": [round(pred_box[0] / width, 8), round(pred_box[1] / height, 8), round(pred_box[2] / width, 8), round(pred_box[3] / height, 8)],
        "proposal_size_norm": [round(pw / width, 8), round(ph / height, 8)],
        "proposal_aspect_log": round(math.log(pw / max(ph, 1.0)), 8),
        "proposal_area_log": round(math.log(max(pw * ph, 1.0)), 8),
        "detector_score": round(float(score), 8),
        "target_offset": [round(v, 8) for v in encode_offset(pred_box, gold_box)],
    }


def build_split(
    *,
    split: str,
    predictions_path: Path,
    gold_rows: dict[str, dict[str, Any]],
    out_dir: Path,
    crop_size: int,
    context_scale: float,
    context_pad: float,
    min_iou: float,
    center_margin: float,
    score_threshold: float,
    max_rows: int,
    max_records: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    pred_rows = load_jsonl(predictions_path)
    if max_rows > 0:
        pred_rows = pred_rows[:max_rows]
    records: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    by_area: Counter[str] = Counter()
    by_label: Counter[str] = Counter()
    matched_gold_ids: set[str] = set()
    image_cache: dict[str, Image.Image] = {}
    for pred_row in pred_rows:
        row_id = str(pred_row.get("row_id"))
        gold_row = gold_rows.get(row_id)
        if not gold_row:
            counters["missing_gold_row"] += 1
            continue
        image_path = str(gold_row.get("image") or "")
        if image_path not in image_cache:
            image_cache[image_path] = Image.open(source_path(image_path)).convert("RGB")
        image = image_cache[image_path]
        width, height = [int(v) for v in gold_row.get("image_size") or list(image.size)]
        golds = []
        for target in (gold_row.get("targets") or {}).get("boxes") or []:
            golds.append(
                {
                    "target_id": str(target.get("target_id") or ""),
                    "bbox": clamp_box([float(v) for v in target.get("bbox") or [0, 0, 1, 1]], width, height),
                    "label": str(target.get("label") or "generic_symbol"),
                    "label_id": int(target.get("label_id") or 5),
                }
            )
        for proposal_index, pred in enumerate(pred_row.get("predicted_symbols") or []):
            score = float(pred.get("score") or 0.0)
            if score < score_threshold:
                counters["below_score_threshold"] += 1
                continue
            pred_box = clamp_box([float(v) for v in pred.get("bbox") or [0, 0, 1, 1]], width, height)
            gold, iou, center_hit = best_match(pred_box, golds, min_iou, center_margin)
            if gold is None:
                counters["unmatched"] += 1
                continue
            target_id = str(gold.get("target_id") or f"gold_{proposal_index}")
            matched_gold_ids.add(f"{row_id}::{target_id}")
            bucket = area_bucket(gold["bbox"])
            label = str(gold["label"])
            by_area[bucket] += 1
            by_label[label] += 1
            crop_box = expand_box(pred_box, width, height, context_scale, context_pad)
            digest = hashlib.sha1(f"{row_id}::{proposal_index}::{pred_box}::{target_id}".encode("utf-8")).hexdigest()[:12]
            stem = f"{safe_name(row_id)}_{proposal_index:04d}_{digest}"
            crop = crop_save(image, crop_box, out_dir / "crops" / split / f"{stem}.png", crop_size)
            record = {
                "id": f"{row_id}::proposal_{proposal_index}::{digest}",
                "row_id": row_id,
                "split": split,
                "source_dataset": gold_row.get("source_dataset"),
                "original_image": gold_row.get("image"),
                "image_size": [width, height],
                "proposal": {
                    "bbox": [round(v, 4) for v in pred_box],
                    "label": str(pred.get("label") or ""),
                    "label_id": int(pred.get("label_id") or 0),
                    "score": round(score, 8),
                    "tile_id": pred.get("tile_id"),
                    "bbox_in_crop": box_in_crop(pred_box, crop_box, crop_size),
                },
                "target": {
                    "bbox": [round(v, 4) for v in gold["bbox"]],
                    "label": label,
                    "label_id": int(gold["label_id"]),
                    "bbox_in_crop": box_in_crop(gold["bbox"], crop_box, crop_size),
                    "offset": [round(v, 8) for v in encode_offset(pred_box, gold["bbox"])],
                },
                "crop": crop,
                "geometry": geometry(pred_box, gold["bbox"], (width, height), score),
                "match_audit": {
                    "matched_gold_target_id": target_id,
                    "proposal_gold_iou": round(iou, 8),
                    "center_match": bool(center_hit),
                    "gold_area_bucket": bucket,
                },
                "runtime_contract": {
                    "model_input_features": ["crop.path", "proposal.bbox_in_crop", "proposal.label_id", "proposal.score", "geometry.proposal_*"],
                    "forbidden_runtime_features": list(FORBIDDEN_RUNTIME_FIELDS),
                    "target_use": "offline_supervised_training_and_evaluation_only",
                    "gold_used_for_runtime": False,
                },
            }
            records.append(record)
            counters["matched_records"] += 1
            if max_records > 0 and len(records) >= max_records:
                break
        if max_records > 0 and len(records) >= max_records:
            break
    total_gold = sum(len((row.get("targets") or {}).get("boxes") or []) for row in gold_rows.values() if str(row.get("id")) in {str(item.get("row_id")) for item in pred_rows})
    audit = {
        "prediction_rows": len(pred_rows),
        "records": len(records),
        "counters": dict(counters),
        "matched_unique_gold_targets": len(matched_gold_ids),
        "gold_targets_in_prediction_rows": total_gold,
        "matched_gold_coverage": round(len(matched_gold_ids) / max(total_gold, 1), 8),
        "by_area": dict(sorted(by_area.items())),
        "by_label": dict(sorted(by_label.items())),
        "image_cache": len(image_cache),
    }
    return records, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-predictions", default=str(TRAIN_PRED))
    parser.add_argument("--locked-predictions", default=str(LOCKED_PRED))
    parser.add_argument("--train-gold", default=str(TRAIN_GOLD))
    parser.add_argument("--locked-gold", default=str(LOCKED_GOLD))
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--crop-size", type=int, default=224)
    parser.add_argument("--context-scale", type=float, default=3.0)
    parser.add_argument("--context-pad", type=float, default=12.0)
    parser.add_argument("--min-iou", type=float, default=0.05)
    parser.add_argument("--center-margin", type=float, default=2.0)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--max-train-rows", type=int, default=0)
    parser.add_argument("--max-locked-rows", type=int, default=64)
    parser.add_argument("--max-train-records", type=int, default=20000)
    parser.add_argument("--max-locked-records", type=int, default=5000)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    train_records, train_audit = build_split(
        split="train",
        predictions_path=Path(args.train_predictions),
        gold_rows=gold_by_row(Path(args.train_gold)),
        out_dir=out_dir,
        crop_size=args.crop_size,
        context_scale=args.context_scale,
        context_pad=args.context_pad,
        min_iou=args.min_iou,
        center_margin=args.center_margin,
        score_threshold=args.score_threshold,
        max_rows=args.max_train_rows,
        max_records=args.max_train_records,
    )
    locked_records, locked_audit = build_split(
        split="locked_smoke",
        predictions_path=Path(args.locked_predictions),
        gold_rows=gold_by_row(Path(args.locked_gold)),
        out_dir=out_dir,
        crop_size=args.crop_size,
        context_scale=args.context_scale,
        context_pad=args.context_pad,
        min_iou=args.min_iou,
        center_margin=args.center_margin,
        score_threshold=args.score_threshold,
        max_rows=args.max_locked_rows,
        max_records=args.max_locked_records,
    )
    write_jsonl(out_dir / "train.jsonl", train_records)
    write_jsonl(out_dir / "locked_smoke.jsonl", locked_records)
    manifest = {
        "version": "symbol_proposal_crop_refiner_v21",
        "splits": {"train": "train.jsonl", "locked_smoke": "locked_smoke.jsonl"},
        "crop_size": args.crop_size,
        "runtime_contract": {
            "model_input": "raster proposal crop plus detector proposal geometry",
            "gold_used_for_runtime": False,
            "forbidden_runtime_features": list(FORBIDDEN_RUNTIME_FIELDS),
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    report = {
        "version": "symbol_proposal_crop_refiner_v21_dataset_audit",
        "claim_boundary": "Dataset for high-resolution visual proposal box refinement; gold is used only for offline target offsets and audit.",
        "inputs": {
            "train_predictions": args.train_predictions,
            "locked_predictions": args.locked_predictions,
            "train_gold": args.train_gold,
            "locked_gold": args.locked_gold,
        },
        "output_dir": rel(out_dir),
        "config": vars(args),
        "splits": {"train": train_audit, "locked_smoke": locked_audit},
        "source_integrity": {
            "runtime_input": "raster_crop_and_detector_proposal_geometry",
            "gold_used_for_runtime": False,
            "forbidden_runtime_features": list(FORBIDDEN_RUNTIME_FIELDS),
            "passed": True,
        },
    }
    write_json(Path(args.report), report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
