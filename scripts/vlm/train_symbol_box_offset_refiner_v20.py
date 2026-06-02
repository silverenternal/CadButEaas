#!/usr/bin/env python3
"""Train a supervised bbox-offset refiner for symbol detector proposals."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image, ImageOps
from sklearn.ensemble import ExtraTreesRegressor, RandomForestRegressor
from sklearn.multioutput import MultiOutputRegressor
import torch
from torchvision.ops import nms


Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
TRAIN_PRED = ROOT / "reports/vlm/symbol_tile_detector_v20_small_anchor_train_proposals.jsonl"
LOCKED_PRED = ROOT / "reports/vlm/symbol_tile_detector_v20_small_anchor_smoke_area_predictions.jsonl"
TRAIN_GOLD = ROOT / "datasets/symbol_expert_public_raster_v19/train.jsonl"
LOCKED_GOLD = ROOT / "datasets/symbol_expert_public_raster_v19/locked.jsonl"
CHECKPOINT = ROOT / "checkpoints/symbol_box_offset_refiner_v20"
REPORT = ROOT / "reports/vlm/symbol_box_offset_refiner_v20_eval.json"
PRED_OUT = ROOT / "reports/vlm/symbol_box_offset_refiner_v20_predictions.jsonl"
LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]
LABEL_TO_ID = {label: index + 1 for index, label in enumerate(LABELS)}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def limit_rows(rows: list[dict[str, Any]], max_rows: int | None) -> list[dict[str, Any]]:
    if not max_rows or max_rows <= 0:
        return rows
    return rows[:max_rows]


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


def gold_by_row(path: Path) -> dict[str, dict[str, Any]]:
    return {str(row.get("id")): row for row in load_jsonl(path)}


def load_gray_cache(path: str, cache: dict[str, np.ndarray]) -> np.ndarray:
    if path not in cache:
        with Image.open(source_path(path)) as opened:
            cache[path] = np.asarray(opened.convert("L"), dtype=np.uint8)
    return cache[path]


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


def crop_stats(gray: np.ndarray, box: list[float]) -> list[float]:
    height, width = gray.shape
    region_box = expand_box(clamp_box(box, width, height), width, height, 1.4, 3.0)
    x1, y1, x2, y2 = region_box
    crop = gray[y1:y2, x1:x2]
    if crop.size == 0:
        return [0.0] * 14
    auto = ImageOps.autocontrast(Image.fromarray(crop))
    arr = np.asarray(auto, dtype=np.float32) / 255.0
    dark = arr <= 0.80
    ys, xs = np.nonzero(dark)
    if len(xs) > 0:
        dark_bbox = [
            xs.min() / max(arr.shape[1], 1),
            ys.min() / max(arr.shape[0], 1),
            (xs.max() + 1) / max(arr.shape[1], 1),
            (ys.max() + 1) / max(arr.shape[0], 1),
        ]
    else:
        dark_bbox = [0.0, 0.0, 1.0, 1.0]
    halves = [
        float(dark[: arr.shape[0] // 2, :].mean()) if arr.shape[0] >= 2 else 0.0,
        float(dark[arr.shape[0] // 2 :, :].mean()) if arr.shape[0] >= 2 else 0.0,
        float(dark[:, : arr.shape[1] // 2].mean()) if arr.shape[1] >= 2 else 0.0,
        float(dark[:, arr.shape[1] // 2 :].mean()) if arr.shape[1] >= 2 else 0.0,
    ]
    return [
        float(arr.mean()),
        float(arr.std()),
        float(dark.mean()),
        *dark_bbox,
        *halves,
        float(np.quantile(arr, 0.10)),
        float(np.quantile(arr, 0.50)),
        float(np.quantile(arr, 0.90)),
    ]


def proposal_features(pred: dict[str, Any], row: dict[str, Any], gray: np.ndarray) -> list[float]:
    width, height = [int(v) for v in row.get("image_size") or [gray.shape[1], gray.shape[0]]]
    box = clamp_box([float(v) for v in pred.get("bbox") or [0, 0, 1, 1]], width, height)
    bw = max(1.0, box[2] - box[0])
    bh = max(1.0, box[3] - box[1])
    label_id = int(pred.get("label_id") or LABEL_TO_ID.get(str(pred.get("label") or ""), 5))
    one_hot = [1.0 if label_id == idx else 0.0 for idx in range(1, len(LABELS) + 1)]
    geom = [
        box[0] / max(width, 1),
        box[1] / max(height, 1),
        box[2] / max(width, 1),
        box[3] / max(height, 1),
        (box[0] + box[2]) / 2.0 / max(width, 1),
        (box[1] + box[3]) / 2.0 / max(height, 1),
        bw / max(width, 1),
        bh / max(height, 1),
        math.log(bw / max(bh, 1.0)),
        math.log(max(bw * bh, 1.0)),
        float(pred.get("score") or 0.0),
    ]
    return geom + one_hot + crop_stats(gray, box)


def encode_offset(pred_box: list[float], gold_box: list[float]) -> list[float]:
    pw = max(1.0, pred_box[2] - pred_box[0])
    ph = max(1.0, pred_box[3] - pred_box[1])
    return [
        (gold_box[0] - pred_box[0]) / pw,
        (gold_box[1] - pred_box[1]) / ph,
        (gold_box[2] - pred_box[2]) / pw,
        (gold_box[3] - pred_box[3]) / ph,
    ]


def apply_offset(pred_box: list[float], offset: list[float], width: int, height: int, max_abs: float) -> list[float]:
    pw = max(1.0, pred_box[2] - pred_box[0])
    ph = max(1.0, pred_box[3] - pred_box[1])
    clipped = [max(-max_abs, min(max_abs, float(v))) for v in offset]
    return clamp_box(
        [
            pred_box[0] + clipped[0] * pw,
            pred_box[1] + clipped[1] * ph,
            pred_box[2] + clipped[2] * pw,
            pred_box[3] + clipped[3] * ph,
        ],
        width,
        height,
    )


def matched_training_examples(
    pred_rows: list[dict[str, Any]],
    gold_rows: dict[str, dict[str, Any]],
    min_iou: float,
    center_match: bool,
    max_examples: int | None,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    features: list[list[float]] = []
    targets: list[list[float]] = []
    counters: Counter[str] = Counter()
    cache: dict[str, np.ndarray] = {}
    for pred_row in pred_rows:
        row_id = str(pred_row.get("row_id"))
        gold_row = gold_rows.get(row_id)
        if not gold_row:
            continue
        gray = load_gray_cache(str(gold_row.get("image") or ""), cache)
        width, height = [int(v) for v in gold_row.get("image_size") or [gray.shape[1], gray.shape[0]]]
        golds = [
            {"bbox": [float(v) for v in target["bbox"]], "label": str(target.get("label") or "generic_symbol")}
            for target in ((gold_row.get("targets") or {}).get("boxes") or [])
        ]
        for pred in pred_row.get("predicted_symbols") or []:
            pred_box = clamp_box([float(v) for v in pred.get("bbox") or [0, 0, 1, 1]], width, height)
            best: dict[str, Any] | None = None
            best_iou = 0.0
            for gold in golds:
                iou = bbox_iou(pred_box, gold["bbox"])
                if iou > best_iou:
                    best_iou = iou
                    best = gold
            if best is None:
                counters["unmatched_no_gold"] += 1
                continue
            if best_iou < min_iou and not (center_match and center_covered(pred_box, best["bbox"])):
                counters["unmatched_below_gate"] += 1
                continue
            features.append(proposal_features(pred, gold_row, gray))
            targets.append(encode_offset(pred_box, best["bbox"]))
            counters[f"matched_{area_bucket(best['bbox'])}"] += 1
            if max_examples and len(features) >= max_examples:
                return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32), {"counts": dict(counters), "image_cache": len(cache)}
    return np.asarray(features, dtype=np.float32), np.asarray(targets, dtype=np.float32), {"counts": dict(counters), "image_cache": len(cache)}


def make_model(kind: str, estimators: int, seed: int) -> Any:
    if kind == "rf":
        base = RandomForestRegressor(n_estimators=estimators, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    else:
        base = ExtraTreesRegressor(n_estimators=estimators, min_samples_leaf=2, random_state=seed, n_jobs=-1)
    return MultiOutputRegressor(base)


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


def evaluate(pred_rows: list[dict[str, Any]], gold_rows: dict[str, dict[str, Any]], score_threshold: float, nms_threshold: float, max_per_page: int) -> dict[str, Any]:
    totals = Counter()
    by_area = Counter()
    by_area_center = Counter()
    by_area_iou = Counter()
    by_label = Counter()
    by_label_center = Counter()
    by_label_iou = Counter()
    for pred_row in pred_rows:
        row_id = str(pred_row.get("row_id"))
        gold_row = gold_rows.get(row_id)
        if not gold_row:
            continue
        golds = [
            {"bbox": [float(v) for v in target["bbox"]], "label": str(target.get("label") or "generic_symbol")}
            for target in ((gold_row.get("targets") or {}).get("boxes") or [])
        ]
        preds = merge_predictions(pred_row.get("predicted_symbols") or [], score_threshold, nms_threshold, max_per_page)
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
        "rows": len(pred_rows),
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


def refine_rows(model: Any, pred_rows: list[dict[str, Any]], gold_rows: dict[str, dict[str, Any]], max_abs_offset: float) -> list[dict[str, Any]]:
    cache: dict[str, np.ndarray] = {}
    refined_rows: list[dict[str, Any]] = []
    for pred_row in pred_rows:
        row_id = str(pred_row.get("row_id"))
        gold_row = gold_rows.get(row_id)
        if not gold_row:
            continue
        gray = load_gray_cache(str(gold_row.get("image") or ""), cache)
        width, height = [int(v) for v in gold_row.get("image_size") or [gray.shape[1], gray.shape[0]]]
        source_preds = pred_row.get("predicted_symbols") or []
        pred_boxes = [
            clamp_box([float(v) for v in pred.get("bbox") or [0, 0, 1, 1]], width, height)
            for pred in source_preds
        ]
        if pred_boxes:
            features = np.asarray(
                [proposal_features(pred, gold_row, gray) for pred in source_preds],
                dtype=np.float32,
            )
            offsets = model.predict(features).tolist()
        else:
            offsets = []
        preds = []
        for pred, pred_box, offset in zip(source_preds, pred_boxes, offsets, strict=True):
            pred_box = clamp_box([float(v) for v in pred.get("bbox") or [0, 0, 1, 1]], width, height)
            updated = dict(pred)
            updated["bbox_before_refine"] = [round(v, 4) for v in pred_box]
            updated["bbox"] = [round(v, 4) for v in apply_offset(pred_box, offset, width, height, max_abs_offset)]
            updated["box_refiner"] = {"method": "supervised_offset_extra_trees_v20", "offset": [round(float(v), 6) for v in offset]}
            preds.append(updated)
        row = dict(pred_row)
        row["predicted_symbols"] = preds
        refined_rows.append(row)
    return refined_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-predictions", default=str(TRAIN_PRED))
    parser.add_argument("--locked-predictions", default=str(LOCKED_PRED))
    parser.add_argument("--train-gold", default=str(TRAIN_GOLD))
    parser.add_argument("--locked-gold", default=str(LOCKED_GOLD))
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT))
    parser.add_argument("--eval-output", default=str(REPORT))
    parser.add_argument("--predictions-output", default=str(PRED_OUT))
    parser.add_argument("--model-kind", choices=["et", "rf"], default="et")
    parser.add_argument("--estimators", type=int, default=180)
    parser.add_argument("--max-train-examples", type=int, default=60000)
    parser.add_argument("--max-locked-rows", type=int, default=0)
    parser.add_argument("--match-min-iou", type=float, default=0.05)
    parser.add_argument("--center-match", action="store_true")
    parser.add_argument("--max-abs-offset", type=float, default=1.2)
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--nms-threshold", type=float, default=0.6)
    parser.add_argument("--max-per-page", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260510)
    args = parser.parse_args()

    train_predictions = load_jsonl(Path(args.train_predictions))
    locked_predictions = limit_rows(load_jsonl(Path(args.locked_predictions)), args.max_locked_rows)
    train_gold = gold_by_row(Path(args.train_gold))
    locked_gold = gold_by_row(Path(args.locked_gold))
    x_train, y_train, match_audit = matched_training_examples(
        train_predictions,
        train_gold,
        args.match_min_iou,
        args.center_match,
        args.max_train_examples,
    )
    if len(y_train) == 0:
        raise SystemExit("no matched training examples for box refiner")
    model = make_model(args.model_kind, args.estimators, args.seed)
    model.fit(x_train, y_train)
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model": model,
            "feature_dim": int(x_train.shape[1]),
            "labels": LABELS,
            "max_abs_offset": args.max_abs_offset,
            "runtime_contract": {
                "model_input": "raster_crop_stats_plus_detector_proposal_geometry",
                "gold_used_for_runtime": False,
            },
        },
        checkpoint_dir / "model.joblib",
    )
    baseline = evaluate(locked_predictions, locked_gold, args.score_threshold, args.nms_threshold, args.max_per_page)
    refined_rows = refine_rows(model, locked_predictions, locked_gold, args.max_abs_offset)
    refined = evaluate(refined_rows, locked_gold, args.score_threshold, args.nms_threshold, args.max_per_page)
    write_jsonl(Path(args.predictions_output), refined_rows)
    report = {
        "version": "symbol_box_offset_refiner_v20_eval",
        "claim_boundary": "Supervised proposal box-offset refiner. Gold is used for training/evaluation only; runtime features are raster stats and detector proposal geometry.",
        "source_integrity": {
            "model_input": "raster_crop_stats_plus_detector_proposal_geometry",
            "train_gold_use": "proposal_to_gold_offset_supervision_only",
            "locked_gold_use": "evaluation_only",
            "raw_label_or_semantic_type_used_as_runtime_feature": False,
        },
        "inputs": {
            "train_predictions": args.train_predictions,
            "locked_predictions": args.locked_predictions,
            "train_gold": args.train_gold,
            "locked_gold": args.locked_gold,
        },
        "checkpoint": rel(checkpoint_dir / "model.joblib"),
        "config": vars(args),
        "training": {
            "examples": int(len(y_train)),
            "feature_dim": int(x_train.shape[1]),
            "target_mean_abs_offset": [round(float(v), 6) for v in np.mean(np.abs(y_train), axis=0).tolist()],
            "match_audit": match_audit,
        },
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
    print(json.dumps({"baseline": baseline, "refined": refined, "delta": report["delta"], "adopted": report["adopted"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
