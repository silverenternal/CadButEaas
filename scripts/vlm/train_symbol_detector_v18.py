#!/usr/bin/env python3
"""Evaluate a recall-first raster-only symbol candidate generator for v18."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/image_only_symbol_detector_v18"
REPORT = ROOT / "reports/vlm"
CHECKPOINT = ROOT / "checkpoints/symbol_detector_v18"

PARAM_GRID = [
    {"threshold": 185, "component_cap": 1000, "anchor_stride": 18, "anchor_cap": 900},
    {"threshold": 205, "component_cap": 1600, "anchor_stride": 14, "anchor_cap": 1400},
    {"threshold": 225, "component_cap": 2600, "anchor_stride": 10, "anchor_cap": 2400},
    {"threshold": 245, "component_cap": 3600, "anchor_stride": 8, "anchor_cap": 3800},
]

ANCHOR_SIZES = [
    (3, 3), (4, 4), (6, 4), (4, 8), (8, 8), (12, 8), (8, 14),
    (16, 12), (20, 16), (28, 16), (24, 28), (36, 24), (48, 32),
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
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


def bbox_iou(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right:
        return 0.0
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    la = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    ra = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(la + ra - inter, 1e-9)


def center_covered(pred: list[int], gold: list[int], margin: int = 2) -> bool:
    cx = (gold[0] + gold[2]) / 2.0
    cy = (gold[1] + gold[3]) / 2.0
    return pred[0] - margin <= cx <= pred[2] + margin and pred[1] - margin <= cy <= pred[3] + margin


def dark_mask(image_path: str | Path, threshold: int) -> np.ndarray:
    path = Path(image_path)
    with Image.open(path if path.is_absolute() else ROOT / path) as image:
        arr = np.asarray(image.convert("L"), dtype=np.uint8)
    return arr <= int(threshold)


def gold_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    targets = row.get("targets") or {}
    symbols = targets.get("symbols")
    if symbols is None:
        symbols = targets.get("boxes")
    return [
        item for item in symbols or []
        if item.get("bbox") and len(item["bbox"]) == 4
    ]


def candidate(row: dict[str, Any], bbox: list[int], source: str, confidence: float, extra: dict[str, Any]) -> dict[str, Any]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return {
        "id": f"{row['id']}_symbol_v18_{source}_{x1}_{y1}_{x2}_{y2}",
        "class": "symbol",
        "family": "symbol",
        "symbol_type": "generic_symbol",
        "semantic_type": "generic_symbol",
        "bbox": [x1, y1, x2, y2],
        "confidence": round(float(max(0.01, min(0.99, confidence))), 6),
        "proposal_source": "raster_symbol_detector_v18",
        "payload": {
            "symbol_type": "generic_symbol",
            "rotation": 0.0,
            "source": "raster_symbol_detector_v18",
            **extra,
        },
    }


def component_candidates(row: dict[str, Any], mask: np.ndarray, params: dict[str, Any]) -> list[dict[str, Any]]:
    import cv2

    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype("uint8"), 8)
    comps: list[tuple[float, list[int], dict[str, Any]]] = []
    height, width = mask.shape
    for idx in range(1, int(n_labels)):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < 1 or area > 2500 or w < 1 or h < 1 or w > 96 or h > 96:
            continue
        aspect = w / max(h, 1)
        if aspect > 35.0 or aspect < 0.03:
            continue
        fill = area / max(w * h, 1)
        if fill < 0.02:
            continue
        for pad in (1, 3, 6):
            bbox = [max(0, x - pad), max(0, y - pad), min(width, x + w + pad), min(height, y + h + pad)]
            score = 0.22 + min(area / 180.0, 1.0) * 0.28 + min(fill, 1.0) * 0.24 - pad * 0.01
            comps.append((score, bbox, {"candidate_kind": "dark_connected_component", "area": area, "fill": round(fill, 6), "pad": pad}))
    comps.sort(key=lambda item: item[0], reverse=True)
    return [
        candidate(row, bbox, "component", score, extra)
        for score, bbox, extra in comps[: int(params["component_cap"])]
    ]


def anchor_candidates(row: dict[str, Any], mask: np.ndarray, params: dict[str, Any]) -> list[dict[str, Any]]:
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    stride = int(params["anchor_stride"])
    buckets = Counter((int(x // stride) * stride + stride // 2, int(y // stride) * stride + stride // 2) for x, y in zip(xs, ys, strict=True))
    points = [point for point, _count in buckets.most_common(int(params["anchor_cap"]))]
    out: list[dict[str, Any]] = []
    for cx, cy in points:
        local = mask[max(0, cy - 5) : min(height, cy + 6), max(0, cx - 5) : min(width, cx + 6)]
        density = float(local.mean()) if local.size else 0.0
        for aw, ah in ANCHOR_SIZES:
            x1 = max(0, int(round(cx - aw / 2)))
            y1 = max(0, int(round(cy - ah / 2)))
            x2 = min(width, x1 + aw)
            y2 = min(height, y1 + ah)
            out.append(candidate(
                row,
                [x1, y1, x2, y2],
                "dark_anchor",
                0.10 + min(density, 1.0) * 0.20,
                {"candidate_kind": "dark_pixel_anchor", "anchor_size": [aw, ah], "local_dark_density": round(density, 6)},
            ))
            if len(out) >= int(params["anchor_cap"]):
                return out
    return out


def dedupe(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for cand in sorted(candidates, key=lambda c: c.get("confidence", 0.0), reverse=True):
        key = tuple(int(v) for v in cand["bbox"])
        if key in seen:
            continue
        if any(bbox_iou(cand["bbox"], prev["bbox"]) > 0.94 for prev in kept[-400:]):
            continue
        seen.add(key)
        kept.append(cand)
        if len(kept) >= limit:
            break
    return kept


def predict_row(row: dict[str, Any], params: dict[str, Any]) -> list[dict[str, Any]]:
    mask = dark_mask(row["image"], int(params["threshold"]))
    comps = component_candidates(row, mask, params)
    anchors = anchor_candidates(row, mask, params)
    return dedupe(comps + anchors, int(params["component_cap"]) + int(params["anchor_cap"]))


def evaluate_rows(rows: list[dict[str, Any]], params: dict[str, Any], limit: int | None = None) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    selected = rows[:limit] if limit else rows
    totals = Counter()
    by_type = Counter()
    by_type_hit = Counter()
    misses: list[dict[str, Any]] = []
    predictions: list[dict[str, Any]] = []
    routed: list[dict[str, Any]] = []
    for row in selected:
        preds = predict_row(row, params)
        golds = gold_symbols(row)
        used: set[int] = set()
        row_iou = row_center = 0
        for gold_index, gold in enumerate(golds):
            gb = [int(v) for v in gold["bbox"]]
            label = str(gold.get("symbol_type") or gold.get("label") or gold.get("semantic_type") or "generic_symbol")
            by_type[label] += 1
            best_iou, best_index = 0.0, None
            center_index = None
            for pred_index, pred in enumerate(preds):
                score = bbox_iou(pred["bbox"], gb)
                if score > best_iou:
                    best_iou, best_index = score, pred_index
                if center_index is None and pred_index not in used and center_covered(pred["bbox"], gb):
                    center_index = pred_index
            if best_index is not None and best_iou >= 0.30 and best_index not in used:
                used.add(best_index)
                row_iou += 1
                row_center += 1
                by_type_hit[label] += 1
            elif center_index is not None:
                used.add(center_index)
                row_center += 1
                by_type_hit[label] += 1
            else:
                misses.append({"row_id": row["id"], "gold_index": gold_index, "bbox": gb, "symbol_type": label, "best_iou": round(best_iou, 6)})
        totals["gold"] += len(golds)
        totals["predicted"] += len(preds)
        totals["matched_iou_0_30"] += row_iou
        totals["matched_center"] += row_center
        predictions.append({
            "id": row["id"],
            "image": row["image"],
            "predicted_symbols": preds,
            "gold_symbol_count": len(golds),
            "matched_iou_0_30": row_iou,
            "matched_center": row_center,
            "source_integrity": {"model_input": "raster_image_only", "gold_used_for_inference": False},
        })
        for pred in preds[:2500]:
            routed.append({
                "candidate_id": pred["id"],
                "row_id": row["id"],
                "family": "symbol",
                "route": "symbol_fixture",
                "bbox": pred["bbox"],
                "confidence": pred["confidence"],
                "payload": pred["payload"],
                "source_integrity": {"model_input": "raster_image_only", "gold_used_for_inference": False},
            })

    precision = totals["matched_iou_0_30"] / max(totals["predicted"], 1)
    recall = totals["matched_iou_0_30"] / max(totals["gold"], 1)
    report = {
        "rows": len(selected),
        "params": params,
        "symbol_bbox_iou_0_30": {
            "matched": int(totals["matched_iou_0_30"]),
            "predicted": int(totals["predicted"]),
            "gold": int(totals["gold"]),
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall), 6),
        },
        "symbol_bbox_center_recall": round(totals["matched_center"] / max(totals["gold"], 1), 6),
        "candidate_inflation": round(totals["predicted"] / max(totals["gold"], 1), 6),
        "type_center_recall": {key: round(by_type_hit[key] / max(by_type[key], 1), 6) for key in sorted(by_type)},
        "typed_label_f1": 0.0,
        "typed_label_note": "Detector emits generic_symbol only; SymbolFixtureExpert labeling is not credited in this baseline.",
        "miss_examples": misses[:25],
    }
    return report, predictions, routed


def choose_policy(dev_rows: list[dict[str, Any]], limit: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reports = [evaluate_rows(dev_rows, params, limit)[0] for params in PARAM_GRID]
    reports.sort(
        key=lambda r: (r["symbol_bbox_center_recall"], r["symbol_bbox_iou_0_30"]["recall"], -r["candidate_inflation"]),
        reverse=True,
    )
    return dict(reports[0]["params"]), reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--task", default="IMG-MOE-V18-P1-007")
    parser.add_argument("--checkpoint-dir", default=str(CHECKPOINT))
    parser.add_argument("--eval-output", default=str(REPORT / "symbol_detector_v18_eval.json"))
    parser.add_argument("--predictions-output", default=str(REPORT / "symbol_detector_v18_locked_predictions.jsonl"))
    parser.add_argument("--routed-output", default=str(REPORT / "symbol_detector_v18_routed_candidates.jsonl"))
    parser.add_argument("--limit-dev", type=int, default=None)
    parser.add_argument("--limit-locked", type=int, default=None)
    args = parser.parse_args()

    data = Path(args.data)
    dev_rows = load_jsonl(data / "dev.jsonl")
    locked_rows = load_jsonl(data / "locked.jsonl")
    params, dev_grid = choose_policy(dev_rows, args.limit_dev)
    locked_report, locked_predictions, routed = evaluate_rows(locked_rows, params, args.limit_locked)
    success = locked_report["symbol_bbox_center_recall"] >= 0.60 and locked_report["typed_label_f1"] > 0.0
    report = {
        "task": args.task,
        "run_mode": "raster_only_symbol_candidate_generator",
        "source_integrity": {
            "model_input": "raster_image_only",
            "dev_gold_use": "parameter_selection_only",
            "locked_gold_use": "evaluation_only",
        },
        "selected_policy": params,
        "dev_grid": dev_grid,
        "locked": locked_report,
        "success_criteria": {
            "symbol_candidate_center_recall_at_least_0_60": locked_report["symbol_bbox_center_recall"] >= 0.60,
            "typed_label_f1_nonzero": locked_report["typed_label_f1"] > 0.0,
            "long_tail_buckets_visible": bool(locked_report["type_center_recall"]),
        },
        "adopted": success,
        "blocker": None if success else "Candidate localization and typed symbol labeling are reported separately; this baseline does not meet all P1-007 gates.",
    }
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(checkpoint_dir / "policy.json", {"task": args.task, "policy": params, "adopted": success})
    write_json(Path(args.eval_output), report)
    write_jsonl(Path(args.predictions_output), locked_predictions)
    write_jsonl(Path(args.routed_output), routed)
    print("task", args.task)
    print("policy", json.dumps(params, sort_keys=True))
    print("locked_center_recall", locked_report["symbol_bbox_center_recall"])
    print("locked_iou_recall", locked_report["symbol_bbox_iou_0_30"]["recall"])
    print("candidate_inflation", locked_report["candidate_inflation"])
    print("adopted", success)


if __name__ == "__main__":
    main()
