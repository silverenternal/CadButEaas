#!/usr/bin/env python3
"""Build a runtime-feature selector dataset from P119 symbol candidates."""

from __future__ import annotations

import argparse
import json
import math
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from analyze_symbol_subset_failure_p117 import bbox_iou, center_covered, nwd_similarity, reconstruct_page_golds


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = ROOT / "reports/vlm/full_public_raster_symbol_eval_subset_p119_expanded_grid.json"
DEFAULT_PREDICTIONS = ROOT / "reports/vlm/full_public_raster_symbol_eval_subset_p119_expanded_grid_predictions.jsonl"
DEFAULT_OUT_DIR = ROOT / "datasets/symbol_selector_subset_p122"
DEFAULT_CONFIG = ROOT / "configs/vlm/symbol_selector_dataset_p122.json"
DEFAULT_REPORT = ROOT / "reports/vlm/symbol_selector_dataset_p122.md"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def bbox_features(box: list[float]) -> dict[str, float]:
    x1, y1, x2, y2 = [float(v) for v in box]
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    area = w * h
    return {
        "x1": x1,
        "y1": y1,
        "x2": x2,
        "y2": y2,
        "cx": (x1 + x2) / 2.0,
        "cy": (y1 + y2) / 2.0,
        "width": w,
        "height": h,
        "area": area,
        "log_area": math.log(area + 1.0),
        "aspect": w / max(h, 1e-6),
        "log_aspect": math.log((w + 1.0) / max(h + 1.0, 1e-6)),
    }


def tile_features(tile_id: str) -> dict[str, float]:
    match = re.search(r"_s(?P<scale>\d+)_tile_(?P<x1>-?\d+)_(?P<y1>-?\d+)_(?P<x2>-?\d+)_(?P<y2>-?\d+)", tile_id or "")
    if not match:
        return {"tile_scale": 0.0, "tile_x1": 0.0, "tile_y1": 0.0, "tile_width": 0.0, "tile_height": 0.0}
    x1 = float(match.group("x1"))
    y1 = float(match.group("y1"))
    x2 = float(match.group("x2"))
    y2 = float(match.group("y2"))
    return {
        "tile_scale": float(match.group("scale")),
        "tile_x1": x1,
        "tile_y1": y1,
        "tile_width": max(0.0, x2 - x1),
        "tile_height": max(0.0, y2 - y1),
    }


def page_extent(preds: list[dict[str, Any]]) -> tuple[float, float]:
    max_x = max((float(pred["bbox"][2]) for pred in preds if pred.get("bbox")), default=1.0)
    max_y = max((float(pred["bbox"][3]) for pred in preds if pred.get("bbox")), default=1.0)
    return max(max_x, 1.0), max(max_y, 1.0)


def overlap_features(index: int, preds: list[dict[str, Any]]) -> dict[str, float]:
    current = preds[index]
    box = [float(v) for v in current["bbox"]]
    score = float(current.get("score") or 0.0)
    label = str(current.get("label") or "")
    same_overlap = any_overlap = same_higher = any_higher = 0
    max_same_higher = 0.0
    max_any_higher = 0.0
    max_same = 0.0
    max_any = 0.0
    for other_index, other in enumerate(preds):
        if other_index == index:
            continue
        other_box = [float(v) for v in other["bbox"]]
        iou = bbox_iou(box, other_box)
        if iou <= 0.0:
            continue
        other_score = float(other.get("score") or 0.0)
        other_label = str(other.get("label") or "")
        max_any = max(max_any, iou)
        if iou >= 0.10:
            any_overlap += 1
        if other_score > score:
            any_higher += 1
            max_any_higher = max(max_any_higher, iou)
        if other_label == label:
            max_same = max(max_same, iou)
            if iou >= 0.10:
                same_overlap += 1
            if other_score > score:
                same_higher += 1
                max_same_higher = max(max_same_higher, iou)
    return {
        "overlap_count_iou10_any": float(any_overlap),
        "overlap_count_iou10_same_label": float(same_overlap),
        "higher_score_overlap_any_count": float(any_higher),
        "higher_score_overlap_same_label_count": float(same_higher),
        "max_iou_any": max_any,
        "max_iou_same_label": max_same,
        "max_iou_higher_score_any": max_any_higher,
        "max_iou_higher_score_same_label": max_same_higher,
    }


def supervision(pred: dict[str, Any], golds: dict[str, dict[str, Any]]) -> dict[str, Any]:
    box = [float(v) for v in pred["bbox"]]
    best_iou = 0.0
    best_nwd = 0.0
    best_label = ""
    center_hit = False
    iou_label_match = False
    tiny_nwd_hit = False
    for gold in golds.values():
        gold_box = [float(v) for v in gold["bbox"]]
        iou = bbox_iou(box, gold_box)
        nwd = nwd_similarity(box, gold_box)
        if iou > best_iou:
            best_iou = iou
            best_label = str(gold.get("label") or "")
            iou_label_match = str(pred.get("label") or "") == best_label
        best_nwd = max(best_nwd, nwd)
        center_hit = center_hit or center_covered(box, gold_box)
        if bbox_area(gold_box) <= 64.0 and nwd >= 0.70:
            tiny_nwd_hit = True
    keep_iou = best_iou >= 0.30
    keep_center = center_hit
    keep_tiny_nwd = tiny_nwd_hit
    keep_any = keep_iou or keep_center or keep_tiny_nwd
    if keep_iou and iou_label_match:
        reason = "keep_iou_type_match"
    elif keep_iou:
        reason = "keep_iou_type_mismatch"
    elif keep_center:
        reason = "keep_center"
    elif keep_tiny_nwd:
        reason = "keep_tiny_nwd"
    else:
        reason = "negative"
    return {
        "keep_any": keep_any,
        "keep_iou": keep_iou,
        "keep_center": keep_center,
        "keep_tiny_nwd": keep_tiny_nwd,
        "label_reason": reason,
        "best_gold_iou": round(best_iou, 6),
        "best_gold_nwd": round(best_nwd, 6),
        "best_gold_label": best_label,
        "best_iou_label_match": bool(iou_label_match),
    }


def bbox_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def build_rows(eval_path: Path, predictions_path: Path) -> list[dict[str, Any]]:
    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    page_golds = reconstruct_page_golds(eval_data)
    prediction_rows = load_jsonl(predictions_path)
    output: list[dict[str, Any]] = []
    for row in prediction_rows:
        row_id = str(row.get("row_id"))
        preds = list(row.get("predicted_symbols") or [])
        preds.sort(key=lambda item: float(item.get("score") or 0.0), reverse=True)
        page_w, page_h = page_extent(preds)
        label_counts = Counter(str(pred.get("label") or "") for pred in preds)
        label_seen = Counter()
        for index, pred in enumerate(preds):
            label = str(pred.get("label") or "")
            label_seen[label] += 1
            box = [float(v) for v in pred["bbox"]]
            b = bbox_features(box)
            t = tile_features(str(pred.get("tile_id") or ""))
            overlaps = overlap_features(index, preds)
            features = {
                "score": float(pred.get("score") or 0.0),
                "label_id": float(pred.get("label_id") or 0),
                "bbox_width": b["width"],
                "bbox_height": b["height"],
                "bbox_area": b["area"],
                "bbox_log_area": b["log_area"],
                "bbox_aspect": b["aspect"],
                "bbox_log_aspect": b["log_aspect"],
                "page_norm_cx": b["cx"] / page_w,
                "page_norm_cy": b["cy"] / page_h,
                "page_norm_width": b["width"] / page_w,
                "page_norm_height": b["height"] / page_h,
                "page_norm_area": b["area"] / max(page_w * page_h, 1.0),
                "tile_scale": t["tile_scale"],
                "tile_norm_x1": t["tile_x1"] / page_w,
                "tile_norm_y1": t["tile_y1"] / page_h,
                "tile_norm_width": t["tile_width"] / page_w,
                "tile_norm_height": t["tile_height"] / page_h,
                "page_candidate_count": float(len(preds)),
                "label_candidate_count": float(label_counts[label]),
                "score_rank_page": float(index + 1),
                "score_rank_page_norm": float(index + 1) / max(len(preds), 1),
                "score_rank_label": float(label_seen[label]),
                "score_rank_label_norm": float(label_seen[label]) / max(label_counts[label], 1),
                **overlaps,
            }
            sup = supervision(pred, page_golds.get(row_id, {}))
            output.append({
                "row_id": row_id,
                "candidate_id": f"{row_id}::symbol_candidate::{index:05d}",
                "runtime_features": features,
                "predicted": {
                    "bbox": box,
                    "label": label,
                    "label_id": int(pred.get("label_id") or 0),
                    "score": float(pred.get("score") or 0.0),
                    "tile_id": pred.get("tile_id"),
                },
                "offline_label": sup,
                "source_trace": {
                    "features_runtime_available": True,
                    "gold_used_for_label_only": True,
                    "source_predictions": "reports/vlm/full_public_raster_symbol_eval_subset_p119_expanded_grid_predictions.jsonl",
                },
            })
    return output


def split_rows(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    row_ids = sorted({row["row_id"] for row in rows})
    rng = random.Random(seed)
    rng.shuffle(row_ids)
    dev_ids = set(row_ids[: max(1, int(len(row_ids) * 0.2))])
    train = [row for row in rows if row["row_id"] not in dev_ids]
    dev = [row for row in rows if row["row_id"] in dev_ids]
    return train, dev


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter(row["offline_label"]["label_reason"] for row in rows)
    pred_labels = Counter(row["predicted"]["label"] for row in rows)
    return {
        "candidates": len(rows),
        "pages": len({row["row_id"] for row in rows}),
        "keep_any": sum(1 for row in rows if row["offline_label"]["keep_any"]),
        "label_reasons": dict(labels.most_common()),
        "predicted_labels": dict(pred_labels.most_common()),
    }


def write_report(path: Path, summary: dict[str, Any]) -> None:
    lines = [
        "# P2-122 Symbol Selector Dataset",
        "",
        "## Decision",
        "",
        "- Built selector dataset with runtime-only candidate features and offline gold-derived labels.",
        "- Claim boundary: dataset construction only; no selector is adopted yet.",
        "",
        "## Summary",
        "",
        "| Split | Candidates | Pages | Keep Any |",
        "|---|---:|---:|---:|",
    ]
    for split in ["train", "dev", "all"]:
        item = summary[split]
        lines.append(f"| `{split}` | {item['candidates']} | {item['pages']} | {item['keep_any']} |")
    lines.extend(["", "## Label Reasons", "", "| Reason | Count |", "|---|---:|"])
    for reason, count in summary["all"]["label_reasons"].items():
        lines.append(f"| `{reason}` | {count} |")
    lines.extend(["", "## Source Integrity", "", "- Runtime features exclude gold bbox, gold label, annotation path, SVG geometry, and expected_json-derived fields.", "- Gold is used only to create offline training/evaluation labels."])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default=str(DEFAULT_EVAL))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--seed", type=int, default=20260516)
    args = parser.parse_args()

    rows = build_rows(Path(args.eval), Path(args.predictions))
    train, dev = split_rows(rows, args.seed)
    out_dir = Path(args.out_dir)
    write_jsonl(out_dir / "train.jsonl", train)
    write_jsonl(out_dir / "dev.jsonl", dev)
    write_jsonl(out_dir / "all.jsonl", rows)
    manifest = {
        "id": "SCI-P2-122-symbol-selector-dataset-build",
        "claim_boundary": "Offline selector dataset only; labels use gold for supervision/evaluation and are forbidden at runtime.",
        "inputs": {"eval": rel(Path(args.eval)), "predictions": rel(Path(args.predictions))},
        "outputs": {"train": rel(out_dir / "train.jsonl"), "dev": rel(out_dir / "dev.jsonl"), "all": rel(out_dir / "all.jsonl")},
        "feature_contract": {
            "runtime_available": True,
            "forbidden_runtime_features": ["gold_bbox", "gold_label", "annotation_path", "svg_geometry", "expected_json"],
            "offline_label_only": ["keep_any", "keep_iou", "keep_center", "keep_tiny_nwd", "best_gold_iou", "best_gold_nwd", "best_gold_label"],
        },
        "summary": {"train": summarize(train), "dev": summarize(dev), "all": summarize(rows)},
        "split_policy": {"unit": "row_id", "dev_fraction": 0.2, "seed": args.seed},
    }
    Path(args.config).parent.mkdir(parents=True, exist_ok=True)
    Path(args.config).write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(Path(args.report), manifest["summary"])
    print(json.dumps({"config": rel(Path(args.config)), "report": rel(Path(args.report)), "summary": manifest["summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
