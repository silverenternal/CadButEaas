#!/usr/bin/env python3
"""Build the v32 symbol box-quality/refiner dataset from the v31 eval cache."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat

from train_symbol_tile_detector_v20 import area_bucket, bbox_iou, rel, write_json, write_jsonl


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CACHE = ROOT / "reports/vlm/symbol_proposal_eval_v31_smoke_cache.jsonl"
DEFAULT_CENTER_TARGETS = ROOT / "datasets/symbol_center_branch_v30/smoke_center_targets.jsonl"
DEFAULT_OUTPUT_DIR = ROOT / "datasets/symbol_box_quality_refiner_v32"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def source_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def valid_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        box = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def clamp_bbox(box: list[float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    x1 = max(0, min(width, int(box[0])))
    y1 = max(0, min(height, int(box[1])))
    x2 = max(0, min(width, int(box[2])))
    y2 = max(0, min(height, int(box[3])))
    if x2 <= x1:
        x2 = min(width, x1 + 1)
    if y2 <= y1:
        y2 = min(height, y1 + 1)
    return x1, y1, x2, y2


def expand_bbox(box: list[float], scale: float, size: tuple[int, int]) -> tuple[int, int, int, int]:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    w = max(1.0, (x2 - x1) * scale)
    h = max(1.0, (y2 - y1) * scale)
    expanded = [cx - w / 2.0, cy - h / 2.0, cx + w / 2.0, cy + h / 2.0]
    return clamp_bbox(expanded, size)


def crop_stats(image: Image.Image, box: tuple[int, int, int, int]) -> dict[str, float]:
    crop = image.crop(box)
    if crop.width <= 0 or crop.height <= 0:
        return {
            "mean": 1.0,
            "std": 0.0,
            "dark_ratio": 0.0,
            "very_dark_ratio": 0.0,
            "area": 0.0,
            "width": 0.0,
            "height": 0.0,
            "aspect": 0.0,
        }
    stat = ImageStat.Stat(crop)
    pixels = list(crop.getdata())
    total = max(len(pixels), 1)
    dark = sum(1 for value in pixels if value < 210)
    very_dark = sum(1 for value in pixels if value < 80)
    return {
        "mean": stat.mean[0] / 255.0,
        "std": stat.stddev[0] / 255.0,
        "dark_ratio": dark / total,
        "very_dark_ratio": very_dark / total,
        "area": float(crop.width * crop.height),
        "width": float(crop.width),
        "height": float(crop.height),
        "aspect": max(crop.width, crop.height) / max(min(crop.width, crop.height), 1),
    }


def box_features(box: list[float], page_size: list[float] | None) -> dict[str, float]:
    x1, y1, x2, y2 = box
    width = max(1e-6, x2 - x1)
    height = max(1e-6, y2 - y1)
    page_w = float(page_size[0]) if page_size else 1.0
    page_h = float(page_size[1]) if page_size else 1.0
    return {
        "x1_norm": x1 / max(page_w, 1.0),
        "y1_norm": y1 / max(page_h, 1.0),
        "x2_norm": x2 / max(page_w, 1.0),
        "y2_norm": y2 / max(page_h, 1.0),
        "cx_norm": ((x1 + x2) / 2.0) / max(page_w, 1.0),
        "cy_norm": ((y1 + y2) / 2.0) / max(page_h, 1.0),
        "width_norm": width / max(page_w, 1.0),
        "height_norm": height / max(page_h, 1.0),
        "area_norm": (width * height) / max(page_w * page_h, 1.0),
        "aspect": width / height,
    }


def delta_target(pred: list[float], gold: list[float]) -> list[float]:
    px1, py1, px2, py2 = pred
    gx1, gy1, gx2, gy2 = gold
    pw = max(px2 - px1, 1e-6)
    ph = max(py2 - py1, 1e-6)
    pcx = (px1 + px2) / 2.0
    pcy = (py1 + py2) / 2.0
    gw = max(gx2 - gx1, 1e-6)
    gh = max(gy2 - gy1, 1e-6)
    gcx = (gx1 + gx2) / 2.0
    gcy = (gy1 + gy2) / 2.0
    return [
        (gcx - pcx) / pw,
        (gcy - pcy) / ph,
        (gw - pw) / pw,
        (gh - ph) / ph,
    ]


def split_for_row(row_id: str) -> str:
    suffix = sum(ord(ch) for ch in row_id) % 10
    if suffix < 7:
        return "train"
    if suffix < 8:
        return "dev"
    return "smoke_eval"


def load_page_metadata(center_targets: Path) -> dict[str, dict[str, Any]]:
    pages: dict[str, dict[str, Any]] = {}
    for row in load_jsonl(center_targets):
        row_id = str(row.get("row_id"))
        if row_id and row_id not in pages:
            pages[row_id] = {
                "image": row.get("image"),
                "image_size": row.get("image_size"),
            }
    return pages


def choose_refine_target(match: dict[str, Any], golds: dict[str, dict[str, Any]]) -> tuple[str | None, dict[str, Any] | None]:
    best_id = match.get("best_iou_target_id")
    if best_id in golds:
        return str(best_id), golds[str(best_id)]
    for target_id in match.get("center_target_ids") or []:
        if target_id in golds:
            return str(target_id), golds[str(target_id)]
    return None, None


def build_rows(cache_rows: list[dict[str, Any]], page_meta: dict[str, dict[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    counts = Counter()
    by_split = Counter()
    by_bucket = Counter()
    by_label = Counter()
    by_source = Counter()
    skipped = Counter()
    for cache_row in cache_rows:
        row_id = str(cache_row.get("row_id"))
        split = split_for_row(row_id)
        meta = page_meta.get(row_id, {})
        page_size = meta.get("image_size")
        golds = {
            str(gold.get("target_id")): gold
            for gold in cache_row.get("gold_symbols") or []
            if gold.get("target_id") and valid_box(gold.get("bbox"))
        }
        matches = {
            int(match.get("candidate_index", -1)): match
            for match in cache_row.get("candidate_gold_matches") or []
        }
        preds = list(cache_row.get("predicted_symbols") or [])
        for index, pred in enumerate(preds):
            pred_box = valid_box(pred.get("bbox"))
            if pred_box is None:
                skipped["invalid_pred_box"] += 1
                continue
            match = matches.get(index, {})
            target_id, gold = choose_refine_target(match, golds)
            if gold is None:
                skipped["no_refine_target"] += 1
                continue
            gold_box = valid_box(gold.get("bbox"))
            if gold_box is None:
                skipped["invalid_gold_box"] += 1
                continue
            source = str(pred.get("proposal_source") or "unknown")
            label = str(pred.get("label") or "generic_symbol")
            best_iou = float(match.get("best_iou", 0.0) or 0.0)
            center_hit = bool(match.get("center_target_ids"))
            sample_type = "positive_iou" if best_iou >= 0.30 else "center_only_or_loose"
            image_path = meta.get("image")
            image_features: dict[str, float] = {}
            context_features: dict[str, float] = {}
            if image_path:
                image = Image.open(source_path(image_path)).convert("L")
                image_features = {f"cand_{key}": value for key, value in crop_stats(image, clamp_bbox(pred_box, image.size)).items()}
                context_features = {
                    f"context_{key}": value
                    for key, value in crop_stats(image, expand_bbox(pred_box, 1.8, image.size)).items()
                }
            features = {
                **box_features(pred_box, page_size),
                "score": float(pred.get("score", 0.0) or 0.0),
                "selector_score": float(pred.get("selector_score", pred.get("score", 0.0)) or 0.0),
                "pre_selector_score": float(pred.get("pre_selector_score", pred.get("score", 0.0)) or 0.0),
                "label_id": float(pred.get("label_id") or 5),
                "is_mask_v28": float(source == "mask_v28"),
                "is_center_branch_v30": float(source == "center_branch_v30"),
                "cluster_id_mod_17": float(int(pred.get("cluster_id") or 0) % 17) / 17.0,
                **image_features,
                **context_features,
            }
            out = {
                "row_id": row_id,
                "split": split,
                "image": meta.get("image"),
                "image_size": page_size,
                "candidate_index": index,
                "proposal_source": source,
                "label": label,
                "label_id": int(pred.get("label_id") or 5),
                "candidate_bbox": pred_box,
                "target_id": target_id,
                "target_bbox": gold_box,
                "target_label": str(gold.get("label") or "generic_symbol"),
                "area_bucket": area_bucket(gold_box),
                "input_iou": best_iou,
                "input_center_hit": center_hit,
                "sample_type": sample_type,
                "delta_target": delta_target(pred_box, gold_box),
                "features": features,
                "image_path": image_path,
            }
            rows.append(out)
            counts["rows"] += 1
            by_split[split] += 1
            by_bucket[out["area_bucket"]] += 1
            by_label[out["target_label"]] += 1
            by_source[source] += 1
            counts[f"sample_type:{sample_type}"] += 1
    manifest = {
        "version": "symbol_box_quality_refiner_v32_dataset",
        "metric_mode": "smoke",
        "claim_boundary": "Offline supervised box-refiner dataset from v31 cache. Gold boxes are labels only; runtime refiner inputs are raster-derived candidate fields.",
        "counts": dict(counts),
        "by_split": dict(by_split),
        "by_area_bucket": dict(by_bucket),
        "by_label": dict(by_label),
        "by_source": dict(by_source),
        "skipped": dict(skipped),
    }
    return rows, manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, default=DEFAULT_CACHE)
    parser.add_argument("--center-targets", type=Path, default=DEFAULT_CENTER_TARGETS)
    parser.add_argument("--support-negative-audit", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_path = source_path(args.cache)
    center_targets_path = source_path(args.center_targets)
    output_dir = source_path(args.output_dir)
    rows, manifest = build_rows(load_jsonl(cache_path), load_page_metadata(center_targets_path))
    output_dir.mkdir(parents=True, exist_ok=True)
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row["split"])].append(row)
    outputs = {}
    for split, split_rows in sorted(by_split.items()):
        path = output_dir / f"{split}.jsonl"
        write_jsonl(path, split_rows)
        outputs[split] = rel(path)
    all_path = output_dir / "all.jsonl"
    write_jsonl(all_path, rows)
    manifest.update(
        {
            "inputs": {
                "cache": rel(cache_path),
                "center_targets": rel(center_targets_path),
                "support_negative_audit": rel(source_path(args.support_negative_audit)) if args.support_negative_audit else None,
            },
            "outputs": {
                "all": rel(all_path),
                **outputs,
            },
            "source_integrity": {
                "runtime_model_input": "raster-derived candidate bbox/score/source/type fields",
                "offline_gold_used_as_training_label": True,
                "svg_or_parser_geometry_used_at_runtime": False,
                "expected_json_used_at_runtime": False,
            },
        }
    )
    manifest_path = output_dir / "manifest.json"
    write_json(manifest_path, manifest)
    print(json.dumps({"manifest": rel(manifest_path), "counts": manifest["counts"], "by_split": manifest["by_split"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
