#!/usr/bin/env python3
"""Attach raster-local evidence to contains_symbol support-criticality rows.

The current contains_symbol support-criticality policy only sees relation
scores, ranks, and geometry. This builder keeps the raster-only inference
contract while adding deterministic image evidence that can later be consumed by
the support-set policy: symbol crop ink/edge density, room crop context, and the
union region between room and symbol candidates.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageFilter, ImageStat

from relation_graph_reconstruction_v18 import load_jsonl, safe_float, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"

DEFAULT_DATASET = REPORT / "contains_symbol_support_criticality_v18_dataset.jsonl"
DEFAULT_ADAPTER = REPORT / "detector_adapter_v18_symbol_boundary_fixed_candidates.jsonl"
DEFAULT_OUTPUT = REPORT / "contains_symbol_visual_support_features_v18_dataset.jsonl"
DEFAULT_AUDIT = REPORT / "contains_symbol_visual_support_features_v18_audit.json"
DEFAULT_EXAMPLES = REPORT / "contains_symbol_visual_support_features_v18_examples.jsonl"


def candidate_stream(row: dict[str, Any]) -> list[dict[str, Any]]:
    graph = row.get("scene_graph") if isinstance(row.get("scene_graph"), dict) else {}
    stream = graph.get("candidate_stream")
    return stream if isinstance(stream, list) else []


def bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def load_candidate_index(adapter_path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, str]]:
    candidates: dict[str, dict[str, Any]] = {}
    images: dict[str, str] = {}
    for row in load_jsonl(adapter_path):
        row_id = str(row.get("id"))
        image = str(row.get("image") or "")
        if image:
            images[row_id] = image
        for cand in candidate_stream(row):
            candidate_id = str(cand.get("candidate_id") or "")
            if not candidate_id:
                continue
            candidates[f"{row_id}|{candidate_id}"] = cand
    return candidates, images


def open_raster(path: str, cache: dict[str, tuple[Image.Image, Image.Image] | None]) -> tuple[Image.Image, Image.Image] | None:
    if path in cache:
        return cache[path]
    try:
        image = Image.open(ROOT / path if not Path(path).is_absolute() else Path(path)).convert("L")
        cache[path] = (image, image.filter(ImageFilter.FIND_EDGES))
    except (FileNotFoundError, OSError):
        cache[path] = None
    return cache[path]


def crop(image: Image.Image, box: list[float], pad: float) -> Image.Image | None:
    x1 = max(0, int(math.floor(box[0] - pad)))
    y1 = max(0, int(math.floor(box[1] - pad)))
    x2 = min(image.width, int(math.ceil(box[2] + pad)))
    y2 = min(image.height, int(math.ceil(box[3] + pad)))
    if x2 <= x1 or y2 <= y1:
        return None
    return image.crop((x1, y1, x2, y2))


def hist_frac(image: Image.Image, start: int, end: int) -> float:
    hist = image.histogram()
    total = sum(hist)
    if total <= 0:
        return 0.0
    return float(sum(hist[start:end]) / total)


def layout_features(patch: Image.Image) -> dict[str, float]:
    width, height = patch.size
    if width <= 2 or height <= 2:
        dark = hist_frac(patch, 0, 128)
        return {
            "dark_center_ratio": 1.0,
            "dark_border_ratio": 1.0,
            "dark_horizontal_balance": 0.0,
            "dark_vertical_balance": 0.0,
            "aspect": float(width / max(height, 1)),
            "area": float(width * height),
        }
    center = patch.crop((width // 4, height // 4, max(width // 4 + 1, width * 3 // 4), max(height // 4 + 1, height * 3 // 4)))
    top = patch.crop((0, 0, width, max(1, height // 4)))
    bottom = patch.crop((0, max(0, height * 3 // 4), width, height))
    left = patch.crop((0, 0, max(1, width // 4), height))
    right = patch.crop((max(0, width * 3 // 4), 0, width, height))
    dark = max(hist_frac(patch, 0, 128), 1e-6)
    left_dark = hist_frac(left, 0, 128)
    right_dark = hist_frac(right, 0, 128)
    top_dark = hist_frac(top, 0, 128)
    bottom_dark = hist_frac(bottom, 0, 128)
    return {
        "dark_center_ratio": hist_frac(center, 0, 128) / dark,
        "dark_border_ratio": ((left_dark + right_dark + top_dark + bottom_dark) / 4.0) / dark,
        "dark_horizontal_balance": (left_dark - right_dark) / max(left_dark + right_dark, 1e-6),
        "dark_vertical_balance": (top_dark - bottom_dark) / max(top_dark + bottom_dark, 1e-6),
        "aspect": float(width / max(height, 1)),
        "area": float(width * height),
    }


def patch_features(
    raster: tuple[Image.Image, Image.Image] | None,
    box: list[float] | None,
    *,
    prefix: str,
    pad: float,
) -> dict[str, float]:
    defaults = {
        f"{prefix}_mean": 0.0,
        f"{prefix}_std": 0.0,
        f"{prefix}_dark_density": 0.0,
        f"{prefix}_very_dark_density": 0.0,
        f"{prefix}_mid_dark_density": 0.0,
        f"{prefix}_edge_density": 0.0,
        f"{prefix}_edge_strong_density": 0.0,
        f"{prefix}_context_dark_density": 0.0,
        f"{prefix}_context_edge_density": 0.0,
        f"{prefix}_dark_ratio": 0.0,
        f"{prefix}_edge_ratio": 0.0,
        f"{prefix}_dark_center_ratio": 0.0,
        f"{prefix}_dark_border_ratio": 0.0,
        f"{prefix}_dark_horizontal_balance": 0.0,
        f"{prefix}_dark_vertical_balance": 0.0,
        f"{prefix}_aspect": 0.0,
        f"{prefix}_area": 0.0,
    }
    if raster is None or box is None:
        return defaults
    image, edge = raster
    patch = crop(image, box, pad)
    edge_patch = crop(edge, box, pad)
    context_pad = max(8.0, max(box[2] - box[0], box[3] - box[1]) * 0.75)
    context = crop(image, box, context_pad)
    edge_context = crop(edge, box, context_pad)
    if patch is None or edge_patch is None or context is None or edge_context is None:
        return defaults
    stat = ImageStat.Stat(patch)
    dark = hist_frac(patch, 0, 128)
    edge_density = hist_frac(edge_patch, 33, 256)
    context_dark = hist_frac(context, 0, 128)
    context_edge = hist_frac(edge_context, 33, 256)
    layout = layout_features(patch)
    out = {
        f"{prefix}_mean": float(stat.mean[0]) / 255.0,
        f"{prefix}_std": float(stat.stddev[0]) / 255.0,
        f"{prefix}_dark_density": dark,
        f"{prefix}_very_dark_density": hist_frac(patch, 0, 64),
        f"{prefix}_mid_dark_density": hist_frac(patch, 64, 160),
        f"{prefix}_edge_density": edge_density,
        f"{prefix}_edge_strong_density": hist_frac(edge_patch, 96, 256),
        f"{prefix}_context_dark_density": context_dark,
        f"{prefix}_context_edge_density": context_edge,
        f"{prefix}_dark_ratio": dark / max(context_dark, 1e-6),
        f"{prefix}_edge_ratio": edge_density / max(context_edge, 1e-6),
    }
    for name, value in layout.items():
        out[f"{prefix}_{name}"] = value
    return out


def union_bbox(left: list[float] | None, right: list[float] | None) -> list[float] | None:
    if left is None or right is None:
        return None
    return [min(left[0], right[0]), min(left[1], right[1]), max(left[2], right[2]), max(left[3], right[3])]


def relative_features(source_box: list[float] | None, target_box: list[float] | None) -> dict[str, float]:
    if source_box is None or target_box is None:
        return {
            "visual_target_center_x_in_source": 0.0,
            "visual_target_center_y_in_source": 0.0,
            "visual_target_area_to_source": 0.0,
            "visual_target_inside_source": 0.0,
        }
    sw = max(source_box[2] - source_box[0], 1e-6)
    sh = max(source_box[3] - source_box[1], 1e-6)
    tw = max(target_box[2] - target_box[0], 1e-6)
    th = max(target_box[3] - target_box[1], 1e-6)
    tcx = (target_box[0] + target_box[2]) / 2.0
    tcy = (target_box[1] + target_box[3]) / 2.0
    ix1 = max(source_box[0], target_box[0])
    iy1 = max(source_box[1], target_box[1])
    ix2 = min(source_box[2], target_box[2])
    iy2 = min(source_box[3], target_box[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return {
        "visual_target_center_x_in_source": (tcx - source_box[0]) / sw,
        "visual_target_center_y_in_source": (tcy - source_box[1]) / sh,
        "visual_target_area_to_source": (tw * th) / max(sw * sh, 1e-6),
        "visual_target_inside_source": inter / max(tw * th, 1e-6),
    }


def quantiles(values: list[float]) -> dict[str, float]:
    if not values:
        return {}
    ordered = sorted(values)
    out: dict[str, float] = {}
    for name, q in [("p10", 0.10), ("p25", 0.25), ("p50", 0.50), ("p75", 0.75), ("p90", 0.90)]:
        out[name] = round(ordered[min(len(ordered) - 1, int(q * (len(ordered) - 1)))], 6)
    return out


def feature_separability(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_feature: dict[str, dict[str, list[float]]] = defaultdict(lambda: defaultdict(list))
    for row in rows:
        labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
        feats = row.get("features") if isinstance(row.get("features"), dict) else {}
        group = "critical" if labels.get("support_critical") else "duplicate" if labels.get("duplicate_support") else "other"
        for name, value in feats.items():
            if not str(name).startswith("visual_"):
                continue
            by_feature[str(name)][group].append(safe_float(value))
    ranked: list[dict[str, Any]] = []
    for name, groups in by_feature.items():
        critical = groups.get("critical") or []
        duplicate = groups.get("duplicate") or []
        other = groups.get("other") or []
        if not critical or not duplicate:
            continue
        crit_med = quantiles(critical).get("p50", 0.0)
        dup_med = quantiles(duplicate).get("p50", 0.0)
        ranked.append(
            {
                "feature": name,
                "critical": quantiles(critical),
                "duplicate": quantiles(duplicate),
                "other": quantiles(other),
                "abs_median_gap_critical_vs_duplicate": round(abs(crit_med - dup_med), 6),
            }
        )
    ranked.sort(key=lambda item: item["abs_median_gap_critical_vs_duplicate"], reverse=True)
    return ranked


def enrich_rows(
    rows: list[dict[str, Any]],
    candidates: dict[str, dict[str, Any]],
    images: dict[str, str],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raster_cache: dict[str, tuple[Image.Image, Image.Image] | None] = {}
    patch_cache: dict[tuple[str, str, tuple[float, float, float, float] | None, float], dict[str, float]] = {}
    out: list[dict[str, Any]] = []
    audit = Counter()

    def cached_patch(
        raster: tuple[Image.Image, Image.Image] | None,
        box: list[float] | None,
        *,
        prefix: str,
        pad: float,
    ) -> dict[str, float]:
        box_key = tuple(round(float(value), 3) for value in box) if box is not None else None
        cache_key = (str(id(raster)), prefix, box_key, float(pad))
        if cache_key not in patch_cache:
            patch_cache[cache_key] = patch_features(raster, box, prefix=prefix, pad=pad)
        return patch_cache[cache_key]

    for row in rows:
        row_id = str(row.get("row_id"))
        source = candidates.get(f"{row_id}|{row.get('source_candidate_id')}")
        target = candidates.get(f"{row_id}|{row.get('target_candidate_id')}")
        source_box = bbox(source.get("bbox")) if isinstance(source, dict) else None
        target_box = bbox(target.get("bbox")) if isinstance(target, dict) else None
        image_path = images.get(row_id, "")
        raster = open_raster(image_path, raster_cache) if image_path else None
        if source_box is None:
            audit["missing_source_bbox"] += 1
        if target_box is None:
            audit["missing_target_bbox"] += 1
        if raster is None:
            audit["missing_raster"] += 1
        visual = {}
        visual.update(cached_patch(raster, target_box, prefix="visual_symbol", pad=2.0))
        visual.update(cached_patch(raster, source_box, prefix="visual_room", pad=2.0))
        visual.update(cached_patch(raster, union_bbox(source_box, target_box), prefix="visual_union", pad=4.0))
        visual.update(relative_features(source_box, target_box))
        item = dict(row)
        item["features"] = {**(row.get("features") or {}), **visual}
        item["visual_evidence"] = {
            "image": image_path,
            "source_bbox": source_box,
            "target_bbox": target_box,
            "feature_count": len(visual),
            "source_integrity": "raster_pixels_and_detector_bboxes_only",
        }
        out.append(item)
    audit["rows"] = len(out)
    audit["images_opened"] = sum(1 for value in raster_cache.values() if value is not None)
    audit["images_missing"] = sum(1 for value in raster_cache.values() if value is None)
    audit["patch_feature_cache_entries"] = len(patch_cache)
    return out, dict(audit)


def build_examples(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
        if not (labels.get("support_critical") or labels.get("duplicate_support") or labels.get("high_score_negative")):
            continue
        feats = row.get("features") if isinstance(row.get("features"), dict) else {}
        examples.append(
            {
                "row_id": row.get("row_id"),
                "relation_id": row.get("relation_id"),
                "support_set_key": row.get("contains_support_set_key"),
                "support_rank": row.get("contains_support_rank"),
                "labels": labels,
                "visual_evidence": row.get("visual_evidence"),
                "visual_features": {name: round(safe_float(value), 6) for name, value in feats.items() if str(name).startswith("visual_")},
            }
        )
        if len(examples) >= limit:
            break
    return examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--examples-output", default=str(DEFAULT_EXAMPLES))
    parser.add_argument("--example-limit", type=int, default=120)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.dataset))
    candidates, images = load_candidate_index(Path(args.adapter))
    enriched, coverage = enrich_rows(rows, candidates, images)
    separability = feature_separability(enriched)
    label_counts = Counter()
    for row in enriched:
        labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
        for name in ["support_critical", "gold_representative", "bridge_positive", "duplicate_support", "high_score_negative"]:
            label_counts[name] += int(bool(labels.get(name)))
    audit = {
        "task": "IMG-MOE-V18-REBUILD-005.step_contains_symbol_visual_support_features",
        "dataset": args.dataset,
        "adapter": args.adapter,
        "output": args.output,
        "rows": len(enriched),
        "candidate_index_size": len(candidates),
        "image_rows": len(images),
        "coverage": coverage,
        "label_counts": dict(label_counts),
        "top_visual_feature_separability": separability[:30],
        "feature_contract": "inference_available_raster_pixels_and_detector_bboxes_only",
        "gold_loaded_after_inference_for_audit_only": True,
        "gold_used_for_inference": False,
    }
    write_jsonl(Path(args.output), enriched)
    write_json(Path(args.audit_output), audit)
    write_jsonl(Path(args.examples_output), build_examples(enriched, args.example_limit))
    print(json.dumps({"rows": len(enriched), "coverage": coverage, "top_features": separability[:5]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
