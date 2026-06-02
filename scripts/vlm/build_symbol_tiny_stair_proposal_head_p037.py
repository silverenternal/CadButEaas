#!/usr/bin/env python3
"""Build P0-37 raster-only tiny/stair proposal-head dataset manifest."""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import write_json, write_jsonl

FOCUS_LABELS = {"stair", "sink", "shower", "equipment"}
FOCUS_AREAS = {"tiny_le_64", "small_le_256"}
RUNTIME_CONTRACT = {
    "model_input_features": ["image_tile_pixels", "tile_bbox", "anchor_geometry_optional"],
    "label_use": "offline_supervised_training_and_evaluation_only",
    "forbidden_runtime_features": ["svg_geometry", "cad_geometry", "gold_bbox", "target_id"],
}


def page_num(row_id: str) -> int:
    try:
        return int(str(row_id).split("_")[-1])
    except Exception:
        return -1


def split_for_page(row_id: str) -> str:
    n = page_num(row_id)
    if n < 0:
        return "train"
    if 0 <= n <= 199:
        return "smoke_eval"
    if 200 <= n <= 282:
        return "dev"
    return "train"


def valid_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    box = [float(v) for v in value]
    return box if box[2] > box[0] and box[3] > box[1] else None


def area(box: list[float]) -> float:
    return max(box[2] - box[0], 0.0) * max(box[3] - box[1], 0.0)


def clamp_box(box: list[float], width: float, height: float) -> list[float] | None:
    out = [max(0.0, min(float(box[0]), width)), max(0.0, min(float(box[1]), height)), max(0.0, min(float(box[2]), width)), max(0.0, min(float(box[3]), height))]
    return out if out[2] > out[0] and out[3] > out[1] else None


def iou(a: list[float], b: list[float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    inter = max(x2 - x1, 0.0) * max(y2 - y1, 0.0)
    denom = area(a) + area(b) - inter
    return inter / denom if denom > 0 else 0.0


def max_iou(box: list[float], boxes: list[list[float]]) -> float:
    return max((iou(box, other) for other in boxes), default=0.0)


def page_to_tile_box(page_box: list[float], tile_box: list[float]) -> list[float]:
    return [page_box[0] - tile_box[0], page_box[1] - tile_box[1], page_box[2] - tile_box[0], page_box[3] - tile_box[1]]


def make_negative(tile: dict[str, Any], box: list[float], role: str, idx: int) -> dict[str, Any]:
    tile_box = (tile.get("tile") or {}).get("bbox") or [0, 0, 384, 384]
    return {
        "id": f"{tile['id']}|neg|{role}|{idx}",
        "split": split_for_page(str(tile.get("row_id") or "")),
        "row_id": tile.get("row_id"),
        "tile_id": tile.get("id"),
        "image": tile.get("image"),
        "image_size": tile.get("image_size"),
        "tile_bbox": tile_box,
        "label": "background",
        "label_id": -1,
        "area_bucket": "background_anchor",
        "target_id": "",
        "bbox_in_tile": box,
        "page_bbox": [box[0] + tile_box[0], box[1] + tile_box[1], box[2] + tile_box[0], box[3] + tile_box[1]],
        "is_positive": False,
        "sample_role": role,
        "hard_miss_seed": False,
        "runtime_contract": RUNTIME_CONTRACT,
    }


def hard_miss_ids(paths: list[str]) -> set[str]:
    out = set()
    for path in paths:
        p = source_path(path)
        if not p.exists():
            continue
        for row in load_jsonl(p):
            tid = str(row.get("target_id") or "")
            if tid:
                out.add(tid)
    return out


def make_positive(tile: dict[str, Any], target: dict[str, Any], hard_ids: set[str]) -> dict[str, Any] | None:
    label = str(target.get("label") or "")
    area_bucket = str(target.get("area_bucket") or "")
    tid = str(target.get("target_id") or "")
    box = valid_box(target.get("bbox"))
    page_box = valid_box(target.get("page_bbox") or target.get("bbox"))
    if label not in FOCUS_LABELS or (area_bucket not in FOCUS_AREAS and label != "stair" and tid not in hard_ids):
        return None
    if box is None or page_box is None:
        return None
    return {
        "id": f"{tile['id']}|pos|{tid}",
        "split": split_for_page(str(tile.get("row_id") or "")),
        "row_id": tile.get("row_id"),
        "tile_id": tile.get("id"),
        "image": tile.get("image"),
        "image_size": tile.get("image_size"),
        "tile_bbox": (tile.get("tile") or {}).get("bbox"),
        "label": label,
        "label_id": target.get("label_id"),
        "area_bucket": area_bucket,
        "target_id": tid,
        "bbox_in_tile": box,
        "page_bbox": page_box,
        "is_positive": True,
        "sample_role": "positive_hard_miss" if tid in hard_ids else "positive_focus_target",
        "hard_miss_seed": tid in hard_ids,
        "runtime_contract": RUNTIME_CONTRACT,
    }


def target_boxes(tile: dict[str, Any]) -> list[list[float]]:
    boxes = []
    for target in (tile.get("targets") or {}).get("boxes") or []:
        box = valid_box(target.get("bbox"))
        if box is not None:
            boxes.append(box)
    return boxes


def negative_anchors(tile: dict[str, Any], positives: list[dict[str, Any]], random_count: int, rng: random.Random, hard_count: int) -> list[dict[str, Any]]:
    tile_box = (tile.get("tile") or {}).get("bbox") or [0, 0, 384, 384]
    width = max(float(tile_box[2] - tile_box[0]), 1.0)
    height = max(float(tile_box[3] - tile_box[1]), 1.0)
    all_target_boxes = target_boxes(tile)
    positive_boxes = [p["bbox_in_tile"] for p in positives]
    anchors = []
    idx = 0

    for target in (tile.get("targets") or {}).get("boxes") or []:
        if len(anchors) >= hard_count:
            break
        label = str(target.get("label") or "")
        box = valid_box(target.get("bbox"))
        if box is None:
            continue
        if label not in FOCUS_LABELS and max_iou(box, positive_boxes) < 0.10:
            anchors.append(make_negative(tile, box, "negative_nonfocus_symbol", idx))
            idx += 1

    for pos in positives:
        if len(anchors) >= hard_count * 3:
            break
        box = pos["bbox_in_tile"]
        w = max(box[2] - box[0], 1.0)
        h = max(box[3] - box[1], 1.0)
        cx = (box[0] + box[2]) * 0.5
        cy = (box[1] + box[3]) * 0.5
        proposals = [
            [cx - w * 1.2, cy - h * 0.5, cx - w * 0.2, cy + h * 0.5],
            [cx + w * 0.2, cy - h * 0.5, cx + w * 1.2, cy + h * 0.5],
            [cx - w * 0.5, cy - h * 1.2, cx + w * 0.5, cy - h * 0.2],
            [cx - w * 0.5, cy + h * 0.2, cx + w * 0.5, cy + h * 1.2],
            [cx - w * 1.1, cy - h * 1.1, cx + w * 1.1, cy + h * 1.1],
            [cx - w * 1.8, cy - h * 0.8, cx + w * 1.8, cy + h * 0.8],
        ]
        for raw in proposals:
            clamped = clamp_box(raw, width, height)
            if clamped is None:
                continue
            role = "negative_overmerged_context" if max_iou(clamped, [box]) >= 0.20 else "negative_near_miss_jitter"
            if role == "negative_near_miss_jitter" and max_iou(clamped, all_target_boxes) >= 0.15:
                continue
            if role == "negative_overmerged_context" and area(clamped) <= area(box) * 1.5:
                continue
            anchors.append(make_negative(tile, clamped, role, idx))
            idx += 1
            if len(anchors) >= hard_count * 3:
                break

    for _ in range(random_count):
        size = rng.choice([12.0, 24.0, 48.0, 96.0])
        cx = rng.uniform(size * 0.5, max(size * 0.5, width - size * 0.5))
        cy = rng.uniform(size * 0.5, max(size * 0.5, height - size * 0.5))
        box = [cx - size * 0.5, cy - size * 0.5, cx + size * 0.5, cy + size * 0.5]
        if max_iou(box, all_target_boxes) < 0.10:
            anchors.append(make_negative(tile, box, "negative_random_anchor", idx))
            idx += 1
    return anchors


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--tile-rows", default="datasets/symbol_tile_detector_tiny_sahi_v21/locked.jsonl")
    parser.add_argument("--hard-cases", default="reports/vlm/symbol_proposal_generation_p030_cases.jsonl,reports/vlm/symbol_proposal_generation_p030_dev_cases.jsonl,reports/vlm/symbol_stair_proposal_absent_p034_smoke_eval_cases.jsonl,reports/vlm/symbol_stair_proposal_absent_p034_dev_cases.jsonl")
    parser.add_argument("--output-dir", default="datasets/symbol_tiny_stair_proposal_head_p037")
    parser.add_argument("--max-positives-per-split", type=int, default=30000)
    parser.add_argument("--negatives-per-positive-tile", type=int, default=1)
    parser.add_argument("--hard-negatives-per-positive-tile", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260513)
    args = parser.parse_args()
    out_dir = source_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    hard_ids = hard_miss_ids([x.strip() for x in args.hard_cases.split(",") if x.strip()])
    rng = random.Random(args.seed)
    positives_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    negative_tiles_by_split: dict[str, list[tuple[dict[str, Any], list[dict[str, Any]]]]] = defaultdict(list)
    counts = Counter()
    for tile in load_jsonl(source_path(args.tile_rows)):
        tile_positives = []
        for target in (tile.get("targets") or {}).get("boxes") or []:
            pos = make_positive(tile, target, hard_ids)
            if pos is None:
                continue
            split = pos["split"]
            if len(positives_by_split[split]) < args.max_positives_per_split:
                positives_by_split[split].append(pos)
                tile_positives.append(pos)
                counts["positive"] += 1
                counts[f"positive_split:{split}"] += 1
                counts[f"positive_label:{pos['label']}"] += 1
                counts[f"positive_area:{pos['area_bucket']}"] += 1
                counts[f"positive_role:{pos['sample_role']}"] += 1
        if tile_positives:
            negative_tiles_by_split[split_for_page(str(tile.get("row_id") or ""))].append((tile, tile_positives))
    outputs = {}
    all_rows = []
    for split, positives in positives_by_split.items():
        rows = list(positives)
        tiles = negative_tiles_by_split.get(split, [])
        for tile, tile_positives in tiles[: len(positives)]:
            negs = negative_anchors(tile, tile_positives, args.negatives_per_positive_tile, rng, args.hard_negatives_per_positive_tile)
            rows.extend(negs)
            counts["negative"] += len(negs)
            counts[f"negative_split:{split}"] += len(negs)
            for neg in negs:
                counts[f"negative_role:{neg['sample_role']}"] += 1
        rng.shuffle(rows)
        path = out_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        outputs[split] = str(path.relative_to(source_path('.')))
        all_rows.extend(rows)
    write_jsonl(out_dir / "all.jsonl", all_rows)
    outputs["all"] = str((out_dir / "all.jsonl").relative_to(source_path('.')))
    manifest = {
        "version": "symbol_tiny_stair_proposal_head_p037_dataset",
        "rows": len(all_rows),
        "positive_rows": sum(len(v) for v in positives_by_split.values()),
        "negative_rows": counts["negative"],
        "hard_miss_target_ids": len(hard_ids),
        "counts": dict(counts),
        "outputs": outputs,
        "inputs": {"tile_rows": args.tile_rows, "hard_cases": [x.strip() for x in args.hard_cases.split(',') if x.strip()]},
        "source_integrity": {"runtime_model_input": "raster crop pixels plus runtime anchor geometry only", "offline_gold_used_as_training_label": True, "svg_or_cad_geometry_used_at_runtime": False, "final_quality_claim_allowed": False},
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
