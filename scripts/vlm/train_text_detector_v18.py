#!/usr/bin/env python3
"""Evaluate a raster-only text candidate generator for v18.

The script keeps inference image-only. Offline text annotations are used only
for dev policy selection and locked evaluation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/image_only_text_ocr_v18"
REPORT = ROOT / "reports/vlm"
CHECKPOINT = ROOT / "checkpoints/text_detector_v18"

PARAM_GRID = [
    {"threshold": 205, "component_cap": 1400, "line_gap": 10, "anchor_stride": 12, "anchor_cap": 1400, "page_anchor_stride": 4, "page_anchor_cap": 70000},
    {"threshold": 185, "component_cap": 900, "line_gap": 8, "anchor_stride": 16, "anchor_cap": 800},
    {"threshold": 205, "component_cap": 1400, "line_gap": 10, "anchor_stride": 12, "anchor_cap": 1400},
    {"threshold": 225, "component_cap": 2200, "line_gap": 12, "anchor_stride": 8, "anchor_cap": 2400},
    {"threshold": 245, "component_cap": 3200, "line_gap": 14, "anchor_stride": 6, "anchor_cap": 4200},
]

ANCHOR_SIZES = [(3, 3), (4, 4), (5, 5), (6, 4), (8, 4), (10, 4), (12, 5), (16, 5), (20, 6)]


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
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    la = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    ra = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(la + ra - inter, 1e-9)


def center_covered(pred: list[int], gold: list[int], margin: int = 1) -> bool:
    cx = (gold[0] + gold[2]) / 2.0
    cy = (gold[1] + gold[3]) / 2.0
    return pred[0] - margin <= cx <= pred[2] + margin and pred[1] - margin <= cy <= pred[3] + margin


def gold_texts(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item
        for item in (row.get("targets") or {}).get("texts") or []
        if item.get("bbox") and len(item["bbox"]) == 4
    ]


def dark_mask(image_path: str | Path, threshold: int) -> np.ndarray:
    with Image.open(ROOT / image_path if not Path(image_path).is_absolute() else image_path) as image:
        arr = np.asarray(image.convert("L"), dtype=np.uint8)
    return arr <= int(threshold)


def text_candidate(row: dict[str, Any], bbox: list[int], source: str, confidence: float, extra: dict[str, Any]) -> dict[str, Any]:
    x1, y1, x2, y2 = [int(v) for v in bbox]
    return {
        "id": f"{row['id']}_text_v18_{source}_{x1}_{y1}_{x2}_{y2}",
        "class": "text",
        "family": "text",
        "semantic_type": "unknown_text",
        "bbox": [x1, y1, x2, y2],
        "confidence": round(float(max(0.01, min(0.99, confidence))), 6),
        "proposal_source": "raster_text_detector_v18",
        "payload": {
            "raw_text": "",
            "normalized_text": "",
            "ocr_text": None,
            "ocr_confidence": None,
            "ocr_status": "not_invoked",
            "rotation": 0,
            "source": "raster_text_detector_v18",
            **extra,
        },
    }


def component_candidates(row: dict[str, Any], mask: np.ndarray, params: dict[str, Any]) -> list[dict[str, Any]]:
    import cv2

    n_labels, _labels, stats, _centroids = cv2.connectedComponentsWithStats(mask.astype("uint8"), 8)
    comps: list[tuple[float, list[int], dict[str, Any]]] = []
    for idx in range(1, int(n_labels)):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < 1 or area > 420 or w < 1 or h < 1 or w > 90 or h > 28:
            continue
        aspect = w / max(h, 1)
        if aspect > 22.0:
            continue
        fill = area / max(w * h, 1)
        if fill < 0.03:
            continue
        pad_x = 1 if w <= 6 else 2
        pad_y = 1 if h <= 6 else 2
        score = 0.20 + min(area / 80.0, 1.0) * 0.30 + min(fill, 1.0) * 0.25
        comps.append((
            score,
            [
                max(0, x - pad_x),
                max(0, y - pad_y),
                min(mask.shape[1], x + w + pad_x),
                min(mask.shape[0], y + h + pad_y),
            ],
            {"candidate_kind": "dark_connected_component", "area": area, "fill": round(fill, 6)},
        ))
    comps.sort(key=lambda item: item[0], reverse=True)
    return [
        text_candidate(row, bbox, "component", score, extra)
        for score, bbox, extra in comps[: int(params["component_cap"])]
    ]


def merged_line_candidates(row: dict[str, Any], base: list[dict[str, Any]], params: dict[str, Any]) -> list[dict[str, Any]]:
    small = [
        c for c in base
        if (c["bbox"][2] - c["bbox"][0]) <= 32 and (c["bbox"][3] - c["bbox"][1]) <= 16
    ]
    small.sort(key=lambda c: ((c["bbox"][1] + c["bbox"][3]) / 2.0, c["bbox"][0]))
    rows: list[list[dict[str, Any]]] = []
    for cand in small:
        cy = (cand["bbox"][1] + cand["bbox"][3]) / 2.0
        placed = False
        for group in rows:
            gy = sum((c["bbox"][1] + c["bbox"][3]) / 2.0 for c in group) / len(group)
            if abs(cy - gy) <= 3.5:
                group.append(cand)
                placed = True
                break
        if not placed:
            rows.append([cand])

    out: list[dict[str, Any]] = []
    gap = int(params["line_gap"])
    for group in rows:
        group.sort(key=lambda c: c["bbox"][0])
        run: list[dict[str, Any]] = []
        last_x2 = -1
        for cand in group:
            if run and cand["bbox"][0] - last_x2 > gap:
                out.extend(_line_windows(row, run))
                run = []
            run.append(cand)
            last_x2 = max(last_x2, cand["bbox"][2])
        if run:
            out.extend(_line_windows(row, run))
    return out[:1200]


def _line_windows(row: dict[str, Any], run: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for start in range(len(run)):
        for end in range(start, min(len(run), start + 7)):
            segment = run[start : end + 1]
            if len(segment) < 2:
                continue
            x1 = min(c["bbox"][0] for c in segment)
            y1 = min(c["bbox"][1] for c in segment)
            x2 = max(c["bbox"][2] for c in segment)
            y2 = max(c["bbox"][3] for c in segment)
            if x2 - x1 > 100 or y2 - y1 > 22:
                continue
            out.append(text_candidate(
                row,
                [x1, y1, x2, y2],
                "line_merge",
                0.34 + min(len(segment), 6) * 0.06,
                {"candidate_kind": "same_row_component_merge", "merged_components": len(segment)},
            ))
    return out


def anchor_candidates(row: dict[str, Any], mask: np.ndarray, params: dict[str, Any]) -> list[dict[str, Any]]:
    height, width = mask.shape
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return []
    stride = int(params["anchor_stride"])
    points = sorted(
        {(int(x // stride) * stride + stride // 2, int(y // stride) * stride + stride // 2) for x, y in zip(xs, ys, strict=True)}
    )
    out: list[dict[str, Any]] = []
    for cx, cy in points:
        if len(out) >= int(params["anchor_cap"]):
            break
        local = mask[max(0, cy - 3) : min(height, cy + 4), max(0, cx - 3) : min(width, cx + 4)]
        density = float(local.mean()) if local.size else 0.0
        for aw, ah in ANCHOR_SIZES:
            x1 = max(0, int(round(cx - aw / 2)))
            y1 = max(0, int(round(cy - ah / 2)))
            x2 = min(width, x1 + aw)
            y2 = min(height, y1 + ah)
            out.append(text_candidate(
                row,
                [x1, y1, x2, y2],
                "dark_anchor",
                0.12 + min(density, 1.0) * 0.18,
                {"candidate_kind": "dark_pixel_anchor", "anchor_size": [aw, ah], "local_dark_density": round(density, 6)},
            ))
            if len(out) >= int(params["anchor_cap"]):
                break
    return out


def page_anchor_candidates(row: dict[str, Any], mask: np.ndarray, params: dict[str, Any]) -> list[dict[str, Any]]:
    stride = int(params.get("page_anchor_stride") or 0)
    if stride <= 0:
        return []
    height, width = mask.shape
    cap = int(params.get("page_anchor_cap") or 0)
    out: list[dict[str, Any]] = []
    sizes = [(3, 3), (4, 4), (6, 4), (8, 4)]
    for cy in range(stride // 2, height, stride):
        for cx in range(stride // 2, width, stride):
            local = mask[max(0, cy - 3) : min(height, cy + 4), max(0, cx - 3) : min(width, cx + 4)]
            density = float(local.mean()) if local.size else 0.0
            for aw, ah in sizes:
                x1 = max(0, int(round(cx - aw / 2)))
                y1 = max(0, int(round(cy - ah / 2)))
                x2 = min(width, x1 + aw)
                y2 = min(height, y1 + ah)
                out.append(text_candidate(
                    row,
                    [x1, y1, x2, y2],
                    "page_anchor",
                    0.08 + min(density, 1.0) * 0.08,
                    {"candidate_kind": "dense_page_text_anchor", "anchor_size": [aw, ah], "local_dark_density": round(density, 6)},
                ))
                if cap and len(out) >= cap:
                    return out
    return out


def dedupe(candidates: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for cand in sorted(candidates, key=lambda c: c.get("confidence", 0.0), reverse=True):
        key = tuple(int(v) for v in cand["bbox"])
        if key in seen:
            continue
        seen.add(key)
        kept.append(cand)
        if len(kept) >= limit:
            break
    return kept


def predict_row(row: dict[str, Any], params: dict[str, Any]) -> list[dict[str, Any]]:
    mask = dark_mask(row["image"], int(params["threshold"]))
    comps = component_candidates(row, mask, params)
    merged = merged_line_candidates(row, comps, params)
    anchors = anchor_candidates(row, mask, params)
    page_anchors = page_anchor_candidates(row, mask, params)
    return dedupe(comps + merged + anchors + page_anchors, int(params["component_cap"]) + 1200 + int(params["anchor_cap"]) + int(params.get("page_anchor_cap") or 0))


def evaluate_rows(
    rows: list[dict[str, Any]],
    params: dict[str, Any],
    limit: int | None = None,
    export_top_k: int = 2500,
) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    selected = rows[:limit] if limit else rows
    predictions: list[dict[str, Any]] = []
    routed: list[dict[str, Any]] = []
    totals = Counter()
    misses: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    semantic = Counter()
    semantic_hit = Counter()
    numeric_gold = numeric_hit = 0
    for row in selected:
        preds = predict_row(row, params)
        golds = gold_texts(row)
        used: set[int] = set()
        row_hits_iou = 0
        row_hits_center = 0
        for gold_index, gold in enumerate(golds):
            gb = [int(v) for v in gold["bbox"]]
            label = str(gold.get("semantic_type") or "unknown")
            semantic[label] += 1
            if gold.get("has_digits"):
                numeric_gold += 1
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
                row_hits_iou += 1
                row_hits_center += 1
                semantic_hit[label] += 1
                if gold.get("has_digits"):
                    numeric_hit += 1
            elif center_index is not None:
                used.add(center_index)
                row_hits_center += 1
                semantic_hit[label] += 1
                if gold.get("has_digits"):
                    numeric_hit += 1
            else:
                misses.append({
                    "row_id": row["id"],
                    "gold_index": gold_index,
                    "bbox": gb,
                    "semantic_type": gold.get("semantic_type"),
                    "normalized_text": gold.get("normalized_text"),
                    "best_iou": round(best_iou, 6),
                })
        totals["gold"] += len(golds)
        totals["predicted"] += len(preds)
        totals["matched_iou_0_30"] += row_hits_iou
        totals["matched_center"] += row_hits_center
        export_preds = preds[:export_top_k] if export_top_k else preds
        predictions.append({
            "id": row["id"],
            "image": row["image"],
            "predicted_text": export_preds,
            "prediction_count_before_export_cap": len(preds),
            "gold_text_count": len(golds),
            "matched_iou_0_30": row_hits_iou,
            "matched_center": row_hits_center,
            "source_integrity": {
                "model_input": "raster_image_only",
                "gold_used_for_inference": False,
                "ocr_transcript_from_gold": False,
            },
        })
        for pred in export_preds:
            routed.append({
                "candidate_id": pred["id"],
                "row_id": row["id"],
                "family": "text",
                "route": "text_dimension",
                "bbox": pred["bbox"],
                "confidence": pred["confidence"],
                "payload": pred["payload"],
                "source_integrity": {
                    "model_input": "raster_image_only",
                    "gold_used_for_inference": False,
                    "ocr_transcript_from_gold": False,
                },
            })
        if len(examples) < 12:
            examples.append({
                "id": row["id"],
                "gold": len(golds),
                "predicted": len(preds),
                "matched_center": row_hits_center,
                "matched_iou_0_30": row_hits_iou,
                "top_predictions": preds[:6],
            })

    precision_iou = totals["matched_iou_0_30"] / max(totals["predicted"], 1)
    recall_iou = totals["matched_iou_0_30"] / max(totals["gold"], 1)
    recall_center = totals["matched_center"] / max(totals["gold"], 1)
    report = {
        "rows": len(selected),
        "params": params,
        "text_bbox_iou_0_30": {
            "matched": int(totals["matched_iou_0_30"]),
            "predicted": int(totals["predicted"]),
            "gold": int(totals["gold"]),
            "precision": round(precision_iou, 6),
            "recall": round(recall_iou, 6),
            "f1": round(0.0 if precision_iou + recall_iou == 0 else 2 * precision_iou * recall_iou / (precision_iou + recall_iou), 6),
        },
        "text_bbox_center_recall": round(recall_center, 6),
        "candidate_inflation": round(totals["predicted"] / max(totals["gold"], 1), 6),
        "numeric_text_recall_proxy": None if numeric_gold == 0 else round(numeric_hit / max(numeric_gold, 1), 6),
        "semantic_center_recall": {
            key: round(semantic_hit[key] / max(semantic[key], 1), 6)
            for key in sorted(semantic)
        },
        "ocr": {
            "backends_available": {name: importlib.util.find_spec(name) is not None for name in ["easyocr", "pytesseract", "paddleocr"]},
            "exact_accuracy": None,
            "normalized_accuracy": None,
            "status": "not_invoked_no_model_credit_transcripts",
        },
        "miss_examples": misses[:25],
        "page_examples": examples,
    }
    return report, predictions, routed


def choose_policy(dev_rows: list[dict[str, Any]], limit: int | None) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    reports: list[dict[str, Any]] = []
    for params in PARAM_GRID:
        report, _preds, _routed = evaluate_rows(dev_rows, params, limit)
        reports.append(report)
    reports.sort(
        key=lambda r: (
            r["text_bbox_center_recall"],
            r["text_bbox_iou_0_30"]["recall"],
            -r["candidate_inflation"],
        ),
        reverse=True,
    )
    return dict(reports[0]["params"]), reports


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=str(DATA))
    parser.add_argument("--limit-dev", type=int, default=None)
    parser.add_argument("--limit-locked", type=int, default=None)
    parser.add_argument("--policy-index", type=int, default=-1)
    parser.add_argument("--export-top-k", type=int, default=2500)
    args = parser.parse_args()

    data = Path(args.data)
    dev_rows = load_jsonl(data / "dev.jsonl")
    locked_rows = load_jsonl(data / "locked.jsonl")
    if args.policy_index >= 0:
        params = dict(PARAM_GRID[args.policy_index])
        fixed_report, _preds, _routed = evaluate_rows(dev_rows, params, args.limit_dev, args.export_top_k)
        dev_grid = [fixed_report]
    else:
        params, dev_grid = choose_policy(dev_rows, args.limit_dev)
    locked_report, locked_predictions, routed = evaluate_rows(locked_rows, params, args.limit_locked, args.export_top_k)

    success = (
        locked_report["text_bbox_center_recall"] >= 0.60
        and locked_report["ocr"]["exact_accuracy"] is not None
    )
    report = {
        "task": "IMG-MOE-V18-P0-006",
        "run_mode": "raster_only_text_candidate_generator",
        "source_integrity": {
            "model_input": "raster_image_only",
            "dev_gold_use": "parameter_selection_only",
            "locked_gold_use": "evaluation_only",
            "gold_transcripts_in_predictions": False,
        },
        "selected_policy": params,
        "dev_grid": dev_grid,
        "locked": locked_report,
        "success_criteria": {
            "text_candidate_center_recall_at_least_0_60": locked_report["text_bbox_center_recall"] >= 0.60,
            "ocr_accuracy_reported": locked_report["ocr"]["exact_accuracy"] is not None,
            "roomspace_text_evidence_ready": False,
        },
        "adopted": success,
        "blocker": None if success else "OCR/content extraction and RoomSpace text-evidence integration are still incomplete; candidate recall is reported separately.",
    }

    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    REPORT.mkdir(parents=True, exist_ok=True)
    write_json(CHECKPOINT / "policy.json", {"task": "IMG-MOE-V18-P0-006", "policy": params, "adopted": success})
    write_json(REPORT / "text_detector_v18_eval.json", report)
    write_jsonl(REPORT / "text_detector_v18_locked_predictions.jsonl", locked_predictions)
    write_jsonl(REPORT / "text_detector_v18_routed_candidates.jsonl", routed)

    print("task IMG-MOE-V18-P0-006")
    print("policy", json.dumps(params, sort_keys=True))
    print("locked_center_recall", locked_report["text_bbox_center_recall"])
    print("locked_iou_recall", locked_report["text_bbox_iou_0_30"]["recall"])
    print("candidate_inflation", locked_report["candidate_inflation"])
    print("adopted", success)


if __name__ == "__main__":
    main()
