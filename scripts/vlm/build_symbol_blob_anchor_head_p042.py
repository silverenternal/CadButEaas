#!/usr/bin/env python3
"""P0-42: build supervised blob-anchor dataset from raster connected components."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from apply_symbol_localization_refiner_p027 import load_gold, valid_box
from apply_symbol_raster_blob_anchors_p041 import open_gray, raster_blob_anchors
from train_symbol_expanded_action_source_policy_v74 import base_selected_by_page
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, write_json, write_jsonl

FOCUS_LABELS = {"stair", "sink", "shower", "equipment"}


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


def page_image_map(tile_rows: str) -> dict[str, str]:
    out = {}
    for row in load_jsonl(source_path(tile_rows)):
        page_id = str(row.get("row_id") or "")
        if page_id and page_id not in out and row.get("image"):
            out[page_id] = str(row.get("image"))
    return out


def best_gold_match(box: list[float], gold_map: dict[str, dict[str, Any]], iou_threshold: float) -> tuple[str, dict[str, Any] | None, float]:
    best_tid = ""
    best_gold = None
    best_iou = 0.0
    for tid, gold in gold_map.items():
        if str(gold.get("label") or "") not in FOCUS_LABELS:
            continue
        gbox = valid_box(gold.get("bbox"))
        if gbox is None:
            continue
        score = bbox_iou(box, gbox)
        if score > best_iou:
            best_tid = tid
            best_gold = gold
            best_iou = score
    if best_iou >= iou_threshold:
        return best_tid, best_gold, best_iou
    return "", None, best_iou


def overlaps_selected(box: list[float], selected_rows: list[dict[str, Any]], threshold: float) -> bool:
    for row in selected_rows:
        other = valid_box(row.get("bbox"))
        if other is not None and bbox_iou(box, other) >= threshold:
            return True
    return False


def make_row(page_id: str, image: str, split: str, idx: int, box: list[float], target_id: str, gold: dict[str, Any] | None, best_iou: float, role: str) -> dict[str, Any]:
    label = str((gold or {}).get("label") or "background")
    return {
        "id": f"{page_id}|blob|{idx}",
        "split": split,
        "row_id": page_id,
        "tile_id": "page_raster_blob",
        "image": image,
        "image_size": None,
        "tile_bbox": None,
        "label": label,
        "label_id": -1,
        "area_bucket": str((gold or {}).get("area_bucket") or "background_anchor"),
        "target_id": target_id,
        "bbox_in_tile": box,
        "page_bbox": box,
        "best_iou": round(best_iou, 6),
        "is_positive": bool(gold),
        "sample_role": role,
        "hard_miss_seed": False,
        "runtime_contract": {
            "model_input_features": ["page_raster_pixels", "blob_anchor_geometry_optional"],
            "label_use": "offline_supervised_training_and_evaluation_only",
            "forbidden_runtime_features": ["svg_geometry", "cad_geometry", "gold_bbox", "target_id"],
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--recovery-data", default="datasets/symbol_detector_listwise_recovery_v47_p2_transfer_locked_full_v2/manifest.json")
    parser.add_argument("--tile-rows", default="datasets/symbol_tile_detector_tiny_sahi_v21/locked.jsonl")
    parser.add_argument("--output-dir", default="datasets/symbol_blob_anchor_head_p042")
    parser.add_argument("--iou-positive", type=float, default=0.30)
    parser.add_argument("--selected-overlap-filter", type=float, default=0.80)
    parser.add_argument("--binary-threshold", type=int, default=110)
    parser.add_argument("--max-side", type=int, default=512)
    parser.add_argument("--min-component-area", type=int, default=3)
    parser.add_argument("--max-component-area", type=int, default=500)
    parser.add_argument("--max-components-per-page", type=int, default=500)
    parser.add_argument("--max-anchors-per-page", type=int, default=900)
    parser.add_argument("--factors", default="1.4,2.0,3.0")
    parser.add_argument("--min-anchor-size", type=float, default=10.0)
    parser.add_argument("--negative-ratio", type=int, default=4)
    args = parser.parse_args()

    out_dir = source_path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    recovery_manifest = json.loads(source_path(args.recovery_data).read_text())
    recovery_rows = load_jsonl(source_path(recovery_manifest["outputs"]["rows"]))
    selected_by_page = base_selected_by_page(recovery_rows, "all")
    image_by_page = page_image_map(args.tile_rows)
    split_pages = set(image_by_page)
    gold_by_page = load_gold(args.tile_rows, split_pages)
    image_cache = {}
    factors = [float(x) for x in args.factors.split(",") if x.strip()]
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    neg_buffer_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    counts = Counter()
    for page_id, image_path in image_by_page.items():
        split = split_for_page(page_id)
        try:
            image = open_gray(image_path, image_cache)
        except Exception:
            counts["image_open_failed"] += 1
            continue
        anchors = raster_blob_anchors(image, args.binary_threshold, args.max_side, args.min_component_area, args.max_component_area, args.max_components_per_page, factors, args.min_anchor_size)[: args.max_anchors_per_page]
        counts["generated_anchor"] += len(anchors)
        positive_count = 0
        for idx, box in enumerate(anchors):
            if overlaps_selected(box, selected_by_page.get(page_id, []), args.selected_overlap_filter):
                counts["filtered_selected_overlap"] += 1
                continue
            target_id, gold, best_iou = best_gold_match(box, gold_by_page.get(page_id, {}), args.iou_positive)
            if gold:
                row = make_row(page_id, image_path, split, idx, box, target_id, gold, best_iou, "positive_blob_iou")
                rows_by_split[split].append(row)
                positive_count += 1
                counts["positive"] += 1
                counts[f"positive_split:{split}"] += 1
                counts[f"positive_label:{row['label']}"] += 1
                counts[f"positive_area:{row['area_bucket']}"] += 1
            else:
                row = make_row(page_id, image_path, split, idx, box, "", None, best_iou, "negative_blob_background")
                neg_buffer_by_split[split].append(row)
        counts[f"pages_split:{split}"] += 1
        counts[f"page_positive_count:{min(positive_count, 5)}"] += 1
    all_rows = []
    outputs = {}
    for split, positives in rows_by_split.items():
        negatives = neg_buffer_by_split.get(split, [])[: max(len(positives) * args.negative_ratio, 1000)]
        rows = positives + negatives
        counts["negative"] += len(negatives)
        counts[f"negative_split:{split}"] += len(negatives)
        path = out_dir / f"{split}.jsonl"
        write_jsonl(path, rows)
        outputs[split] = str(path.relative_to(source_path('.')))
        all_rows.extend(rows)
    write_jsonl(out_dir / "all.jsonl", all_rows)
    outputs["all"] = str((out_dir / "all.jsonl").relative_to(source_path('.')))
    manifest = {
        "version": "symbol_blob_anchor_head_p042_dataset",
        "rows": len(all_rows),
        "positive_rows": counts["positive"],
        "negative_rows": counts["negative"],
        "counts": dict(counts),
        "outputs": outputs,
        "inputs": {"tile_rows": args.tile_rows, "recovery_data": args.recovery_data},
        "source_integrity": {
            "runtime_model_input": "page raster pixels plus generated blob anchor geometry only",
            "offline_gold_used_as_training_label": True,
            "svg_or_cad_geometry_used_at_runtime": False,
            "final_quality_claim_allowed": False,
        },
    }
    write_json(out_dir / "manifest.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
