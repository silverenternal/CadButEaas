#!/usr/bin/env python3
"""Apply a raster-only box refiner to symbol detector predictions."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
from torchvision.ops import nms
import torch


Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
PREDICTIONS = ROOT / "reports/vlm/symbol_tile_detector_v20_small_anchor_smoke_area_predictions.jsonl"
GOLD = ROOT / "datasets/symbol_expert_public_raster_v19/locked.jsonl"
REPORT = ROOT / "reports/vlm/symbol_box_refiner_v20_eval.json"
OUT = ROOT / "reports/vlm/symbol_box_refiner_v20_predictions.jsonl"
LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]


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


def overlap_area(a: list[float], b: list[float]) -> float:
    return max(0.0, min(a[2], b[2]) - max(a[0], b[0])) * max(0.0, min(a[3], b[3]) - max(a[1], b[1]))


def dark_component_bbox(gray: np.ndarray, pred_box: list[float], threshold: int, scale: float, pad: float, component_pad: float) -> list[float] | None:
    height, width = gray.shape
    pred_box = clamp_box(pred_box, width, height)
    region_box = expand_box(pred_box, width, height, scale, pad)
    rx1, ry1, rx2, ry2 = region_box
    if rx2 <= rx1 or ry2 <= ry1:
        return None
    region = gray[ry1:ry2, rx1:rx2]
    dark = region <= threshold
    if int(dark.sum()) < 2:
        return None
    try:
        import cv2

        n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(dark.astype("uint8"), 8)
        selected: list[list[float]] = []
        pred_area = max(1.0, (pred_box[2] - pred_box[0]) * (pred_box[3] - pred_box[1]))
        for idx in range(1, int(n_labels)):
            x, y, w, h, area = [int(v) for v in stats[idx]]
            if area < 2:
                continue
            comp = [float(rx1 + x), float(ry1 + y), float(rx1 + x + w), float(ry1 + y + h)]
            comp_area = max(1.0, (comp[2] - comp[0]) * (comp[3] - comp[1]))
            aspect = (comp[2] - comp[0]) / max(comp[3] - comp[1], 1.0)
            if aspect > 25.0 or aspect < 0.04:
                continue
            if comp_area > pred_area * 5.0 and comp_area > 4096:
                continue
            inter = overlap_area(comp, pred_box)
            cx = (comp[0] + comp[2]) / 2.0
            cy = (comp[1] + comp[3]) / 2.0
            center_inside = pred_box[0] - 2 <= cx <= pred_box[2] + 2 and pred_box[1] - 2 <= cy <= pred_box[3] + 2
            if inter > 0 or center_inside:
                selected.append(comp)
        if not selected:
            return None
        xs1 = [box[0] for box in selected]
        ys1 = [box[1] for box in selected]
        xs2 = [box[2] for box in selected]
        ys2 = [box[3] for box in selected]
    except Exception:
        ys, xs = np.nonzero(dark)
        if len(xs) < 2:
            return None
        xs1, ys1, xs2, ys2 = [float(rx1 + xs.min())], [float(ry1 + ys.min())], [float(rx1 + xs.max() + 1)], [float(ry1 + ys.max() + 1)]
    refined = [
        max(0.0, min(xs1) - component_pad),
        max(0.0, min(ys1) - component_pad),
        min(float(width), max(xs2) + component_pad),
        min(float(height), max(ys2) + component_pad),
    ]
    if refined[2] <= refined[0] or refined[3] <= refined[1]:
        return None
    return refined


def refine_predictions_for_row(row: dict[str, Any], image_path: str, params: dict[str, Any]) -> tuple[dict[str, Any], Counter[str]]:
    with Image.open(source_path(image_path)) as opened:
        gray = np.asarray(opened.convert("L"), dtype=np.uint8)
    height, width = gray.shape
    counters: Counter[str] = Counter()
    refined_preds: list[dict[str, Any]] = []
    for pred in row.get("predicted_symbols") or []:
        original = clamp_box([float(v) for v in pred.get("bbox") or [0, 0, 1, 1]], width, height)
        refined_box = dark_component_bbox(
            gray,
            original,
            int(params["threshold"]),
            float(params["scale"]),
            float(params["pad"]),
            float(params["component_pad"]),
        )
        updated = dict(pred)
        updated["bbox_before_refine"] = [round(v, 4) for v in original]
        if refined_box is None:
            updated["bbox"] = original
            updated["box_refiner"] = {"applied": False, "reason": "no_local_dark_component"}
            counters["unchanged"] += 1
        else:
            updated["bbox"] = [round(v, 4) for v in refined_box]
            updated["box_refiner"] = {"applied": True, "method": "local_dark_component_bbox", "params": params}
            counters["refined"] += 1
        refined_preds.append(updated)
    out = dict(row)
    out["predicted_symbols"] = refined_preds
    return out, counters


def gold_by_row(path: Path) -> dict[str, dict[str, Any]]:
    rows = {}
    for row in load_jsonl(path):
        rows[str(row.get("id"))] = row
    return rows


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
    by_label = Counter()
    by_label_center = Counter()
    by_label_iou = Counter()
    by_area = Counter()
    by_area_center = Counter()
    by_area_iou = Counter()
    typed_correct = 0
    for row in rows:
        row_id = str(row.get("row_id"))
        gold_row = gold_rows.get(row_id)
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
            gold_box = gold["bbox"]
            label = gold["label"]
            bucket = area_bucket(gold_box)
            by_label[label] += 1
            by_area[bucket] += 1
            best_iou = 0.0
            best_iou_index = None
            center_index = None
            for pred_index, pred in enumerate(preds):
                pred_box = [float(v) for v in pred["bbox"]]
                iou = bbox_iou(pred_box, gold_box)
                if iou > best_iou:
                    best_iou = iou
                    best_iou_index = pred_index
                if center_index is None and pred_index not in used_center and center_covered(pred_box, gold_box):
                    center_index = pred_index
            if best_iou_index is not None and best_iou >= 0.30 and best_iou_index not in used_iou:
                used_iou.add(best_iou_index)
                totals["matched_iou"] += 1
                by_label_iou[label] += 1
                by_area_iou[bucket] += 1
                if str(preds[best_iou_index].get("label")) == label:
                    typed_correct += 1
            if center_index is not None:
                used_center.add(center_index)
                totals["matched_center"] += 1
                by_label_center[label] += 1
                by_area_center[bucket] += 1
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
        "typed_accuracy_on_iou_matches": round(typed_correct / max(totals["matched_iou"], 1), 6),
        "type_center_recall": {label: round(by_label_center[label] / max(by_label[label], 1), 6) for label in sorted(by_label)},
        "type_iou_recall": {label: round(by_label_iou[label] / max(by_label[label], 1), 6) for label in sorted(by_label)},
        "area_center_recall": {bucket: round(by_area_center[bucket] / max(by_area[bucket], 1), 6) for bucket in sorted(by_area)},
        "area_iou_recall": {bucket: round(by_area_iou[bucket] / max(by_area[bucket], 1), 6) for bucket in sorted(by_area)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default=str(PREDICTIONS))
    parser.add_argument("--gold", default=str(GOLD))
    parser.add_argument("--output", default=str(OUT))
    parser.add_argument("--eval-output", default=str(REPORT))
    parser.add_argument("--score-threshold", type=float, default=0.05)
    parser.add_argument("--nms-threshold", type=float, default=0.6)
    parser.add_argument("--max-per-page", type=int, default=500)
    parser.add_argument("--dark-threshold", type=int, default=215)
    parser.add_argument("--scale", type=float, default=1.15)
    parser.add_argument("--pad", type=float, default=3.0)
    parser.add_argument("--component-pad", type=float, default=1.5)
    args = parser.parse_args()

    pred_rows = load_jsonl(Path(args.predictions))
    gold_rows = gold_by_row(Path(args.gold))
    baseline = evaluate(pred_rows, gold_rows, args.score_threshold, args.nms_threshold, args.max_per_page)
    params = {
        "threshold": args.dark_threshold,
        "scale": args.scale,
        "pad": args.pad,
        "component_pad": args.component_pad,
    }
    refined_rows: list[dict[str, Any]] = []
    counters: Counter[str] = Counter()
    for row in pred_rows:
        gold_row = gold_rows.get(str(row.get("row_id")))
        if not gold_row:
            continue
        refined, row_counts = refine_predictions_for_row(row, str(gold_row.get("image") or ""), params)
        refined_rows.append(refined)
        counters.update(row_counts)
    refined_eval = evaluate(refined_rows, gold_rows, args.score_threshold, args.nms_threshold, args.max_per_page)
    write_jsonl(Path(args.output), refined_rows)
    report = {
        "version": "symbol_box_refiner_v20_eval",
        "claim_boundary": "Raster-only postprocess refiner. Gold is used for evaluation only, not for refinement.",
        "source_integrity": {
            "model_input": "raster_pixels_plus_detector_predictions",
            "raw_label_or_semantic_type_used_as_runtime_feature": False,
            "locked_gold_use": "evaluation_only",
        },
        "inputs": {"predictions": args.predictions, "gold": args.gold},
        "params": params,
        "postprocess": {"score_threshold": args.score_threshold, "nms_threshold": args.nms_threshold, "max_per_page": args.max_per_page},
        "refiner_counts": dict(counters.most_common()),
        "baseline": baseline,
        "refined": refined_eval,
        "delta": {
            "center_recall": round(refined_eval["symbol_bbox_center_recall"] - baseline["symbol_bbox_center_recall"], 6),
            "iou_recall": round(refined_eval["symbol_bbox_iou_0_30"]["recall"] - baseline["symbol_bbox_iou_0_30"]["recall"], 6),
            "iou_precision": round(refined_eval["symbol_bbox_iou_0_30"]["precision"] - baseline["symbol_bbox_iou_0_30"]["precision"], 6),
            "candidate_inflation": round(refined_eval["candidate_inflation"] - baseline["candidate_inflation"], 6),
        },
        "adopted": refined_eval["symbol_bbox_iou_0_30"]["recall"] > baseline["symbol_bbox_iou_0_30"]["recall"]
        and refined_eval["symbol_bbox_center_recall"] >= baseline["symbol_bbox_center_recall"] * 0.98,
    }
    write_json(Path(args.eval_output), report)
    print(json.dumps({"baseline": baseline, "refined": refined_eval, "delta": report["delta"], "adopted": report["adopted"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
