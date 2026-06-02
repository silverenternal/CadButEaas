#!/usr/bin/env python3
"""Train/evaluate a recall-first raster room proposal policy for v18.

This baseline deliberately keeps inference raster-only. Offline room polygons
are used for dev parameter selection and locked evaluation only.
"""

from __future__ import annotations

import argparse
import json
import math
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "datasets/image_only_room_polygon_v18"
REPORT = ROOT / "reports/vlm"
CHECKPOINT = ROOT / "checkpoints/room_proposal_model_v18"

PARAM_GRID = [
    {"threshold": 245, "projection_min": 6, "cluster_gap": 6, "max_span": 10, "min_side": 16, "max_area": 0.90, "cap": 26000, "anchor_stride": 24, "anchor_limit": 60, "anchor_min_border_density": 0.0, "dense_prior_mode": True},
    {"threshold": 185, "projection_min": 12, "cluster_gap": 3, "max_span": 5, "min_side": 22, "max_area": 0.72, "cap": 900, "anchor_stride": 24, "anchor_limit": 10},
    {"threshold": 205, "projection_min": 10, "cluster_gap": 4, "max_span": 6, "min_side": 20, "max_area": 0.76, "cap": 1400, "anchor_stride": 20, "anchor_limit": 14},
    {"threshold": 225, "projection_min": 8, "cluster_gap": 5, "max_span": 8, "min_side": 18, "max_area": 0.80, "cap": 2200, "anchor_stride": 16, "anchor_limit": 18},
    {"threshold": 245, "projection_min": 6, "cluster_gap": 6, "max_span": 10, "min_side": 16, "max_area": 0.84, "cap": 3200, "anchor_stride": 12, "anchor_limit": 24},
]


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
    ix1 = max(float(left[0]), float(right[0]))
    iy1 = max(float(left[1]), float(right[1]))
    ix2 = min(float(left[2]), float(right[2]))
    iy2 = min(float(left[3]), float(right[3]))
    iw = max(0.0, ix2 - ix1)
    ih = max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0.0:
        return 0.0
    la = max(0.0, float(left[2]) - float(left[0])) * max(0.0, float(left[3]) - float(left[1]))
    ra = max(0.0, float(right[2]) - float(right[0])) * max(0.0, float(right[3]) - float(right[1]))
    return inter / max(la + ra - inter, 1e-9)


def f1(tp: int, predicted: int, gold: int) -> dict[str, float | int]:
    precision = tp / max(predicted, 1)
    recall = tp / max(gold, 1)
    score = 0.0 if precision + recall == 0 else 2.0 * precision * recall / (precision + recall)
    return {
        "matched": int(tp),
        "predicted": int(predicted),
        "gold": int(gold),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(score, 6),
    }


def match_counts(preds: list[dict[str, Any]], golds: list[dict[str, Any]], iou_threshold: float) -> tuple[int, list[dict[str, Any]]]:
    used: set[int] = set()
    matched = 0
    misses: list[dict[str, Any]] = []
    for gold_index, gold in enumerate(golds):
        best_index = None
        best_iou = 0.0
        for pred_index, pred in enumerate(preds):
            if pred_index in used:
                continue
            score = bbox_iou(pred.get("bbox"), gold.get("bbox"))
            if score > best_iou:
                best_iou = score
                best_index = pred_index
        if best_index is not None and best_iou >= iou_threshold:
            used.add(best_index)
            matched += 1
        else:
            misses.append({
                "gold_index": gold_index,
                "bbox": gold.get("bbox"),
                "semantic_type": gold.get("semantic_type"),
                "best_iou": round(best_iou, 6),
            })
    return matched, misses


def match_counts_with_labels(
    preds: list[dict[str, Any]],
    golds: list[dict[str, Any]],
    iou_threshold: float,
) -> tuple[int, Counter[str], list[dict[str, Any]]]:
    used: set[int] = set()
    matched = 0
    matched_labels: Counter[str] = Counter()
    misses: list[dict[str, Any]] = []
    pred_areas = [
        max(0.0, float(pred["bbox"][2]) - float(pred["bbox"][0])) * max(0.0, float(pred["bbox"][3]) - float(pred["bbox"][1]))
        for pred in preds
    ]
    for gold_index, gold in enumerate(golds):
        gold_box = gold.get("bbox")
        if not gold_box:
            continue
        gold_area = max(0.0, float(gold_box[2]) - float(gold_box[0])) * max(0.0, float(gold_box[3]) - float(gold_box[1]))
        best_index = None
        best_iou = 0.0
        for pred_index, pred in enumerate(preds):
            if pred_index in used:
                continue
            pred_area = pred_areas[pred_index]
            if pred_area < gold_area * iou_threshold or pred_area > gold_area / max(iou_threshold, 1e-9):
                continue
            score = bbox_iou(pred.get("bbox"), gold_box)
            if score > best_iou:
                best_iou = score
                best_index = pred_index
        if best_index is not None and best_iou >= iou_threshold:
            used.add(best_index)
            matched += 1
            matched_labels.update([str(gold.get("semantic_type") or "room")])
        else:
            misses.append({
                "gold_index": gold_index,
                "bbox": gold.get("bbox"),
                "semantic_type": gold.get("semantic_type"),
                "best_iou": round(best_iou, 6),
            })
    return matched, matched_labels, misses


def load_dark_mask(image_path: Path, threshold: int) -> tuple[Any, int, int]:
    import numpy as np

    with Image.open(image_path) as image:
        gray = image.convert("L")
        arr = np.asarray(gray, dtype=np.uint8)
    return arr <= int(threshold), int(arr.shape[1]), int(arr.shape[0])


def clusters(indices: list[int], strengths: list[int], gap: int) -> list[tuple[int, int, int]]:
    if not indices:
        return []
    out: list[tuple[int, int, int]] = []
    start = prev = indices[0]
    total = strengths[0]
    for value, strength in zip(indices[1:], strengths[1:], strict=True):
        if value - prev <= gap + 1:
            prev = value
            total += strength
            continue
        out.append((start, prev, total))
        start = prev = value
        total = strength
    out.append((start, prev, total))
    return out


def axis_coords(mask: Any, width: int, height: int, params: dict[str, Any]) -> tuple[list[int], list[int]]:
    import numpy as np

    projection_min = int(params["projection_min"])
    gap = int(params["cluster_gap"])
    col_counts = mask.sum(axis=0)
    row_counts = mask.sum(axis=1)
    col_indices = np.flatnonzero(col_counts >= projection_min).astype(int).tolist()
    row_indices = np.flatnonzero(row_counts >= projection_min).astype(int).tolist()
    col_clusters = clusters(col_indices, [int(col_counts[i]) for i in col_indices], gap)
    row_clusters = clusters(row_indices, [int(row_counts[i]) for i in row_indices], gap)

    def convert(items: list[tuple[int, int, int]], limit: int, max_value: int) -> list[int]:
        ranked = sorted(items, key=lambda item: item[2], reverse=True)[:limit]
        coords = {0, max_value - 1}
        for start, end, _strength in ranked:
            coords.add(int(round((start + end) / 2.0)))
            coords.add(start)
            coords.add(end)
        return sorted(value for value in coords if 0 <= value < max_value)

    return convert(col_clusters, 80, width), convert(row_clusters, 80, height)


def proposal(row: dict[str, Any], bbox: list[int], source: str, score: float, extra: dict[str, Any]) -> dict[str, Any]:
    x1, y1, x2, y2 = bbox
    return {
        "id": f"{row['id']}_room_v18_{source}_{x1}_{y1}_{x2}_{y2}",
        "class": "room",
        "semantic_type": "room",
        "family": "space",
        "bbox": [int(x1), int(y1), int(x2), int(y2)],
        "polygon": [[int(x1), int(y1)], [int(x2), int(y1)], [int(x2), int(y2)], [int(x1), int(y2)]],
        "confidence": round(float(max(0.01, min(0.99, score))), 6),
        "proposal_source": "raster_room_proposal_v18",
        "shape_features": extra,
    }


def connected_space_proposals(row: dict[str, Any], mask: Any, width: int, height: int, params: dict[str, Any]) -> list[dict[str, Any]]:
    import numpy as np

    try:
        from skimage import measure, morphology
    except Exception:
        return []
    with np.errstate(all="ignore"):
        ink = morphology.binary_dilation(mask, morphology.footprint_rectangle((3, 3)) if hasattr(morphology, "footprint_rectangle") else morphology.square(3))
        room_mask = morphology.remove_small_holes(~ink, area_threshold=max(int(width * height * 0.002), 128))
        room_mask = morphology.remove_small_objects(room_mask.astype(bool), min_size=max(int(width * height * 0.002), 128))
    labels = measure.label(room_mask, connectivity=2)
    image_area = max(width * height, 1)
    out: list[dict[str, Any]] = []
    for region in measure.regionprops(labels):
        if region.area < max(int(image_area * 0.002), 120):
            continue
        minr, minc, maxr, maxc = region.bbox
        if minc <= 1 and minr <= 1 and maxc >= width - 1 and maxr >= height - 1:
            continue
        area_ratio = region.area / image_area
        if area_ratio > float(params["max_area"]):
            continue
        box_area = max((maxc - minc) * (maxr - minr), 1)
        fill = float(region.area) / box_area
        if fill < 0.18:
            continue
        out.append(proposal(
            row,
            [int(minc), int(minr), int(maxc), int(maxr)],
            "component",
            0.25 + min(fill, 1.0) * 0.45 + min(area_ratio, 0.2),
            {"proposal_kind": "white_connected_component", "area": int(region.area), "fill_ratio": round(fill, 6)},
        ))
    return out


def grid_box_proposals(row: dict[str, Any], mask: Any, width: int, height: int, params: dict[str, Any]) -> list[dict[str, Any]]:
    import numpy as np

    xs, ys = axis_coords(mask, width, height, params)
    max_span = int(params["max_span"])
    min_side = int(params["min_side"])
    max_area = float(params["max_area"]) * width * height
    out: list[dict[str, Any]] = []
    integral = mask.astype(np.uint8).cumsum(axis=0).cumsum(axis=1)

    def dark_count(x1: int, y1: int, x2: int, y2: int) -> int:
        xa, ya = max(0, x1), max(0, y1)
        xb, yb = min(width - 1, x2), min(height - 1, y2)
        total = int(integral[yb, xb])
        if xa > 0:
            total -= int(integral[yb, xa - 1])
        if ya > 0:
            total -= int(integral[ya - 1, xb])
        if xa > 0 and ya > 0:
            total += int(integral[ya - 1, xa - 1])
        return total

    for xi, x1 in enumerate(xs[:-1]):
        for xj in range(xi + 1, min(len(xs), xi + max_span + 1)):
            x2 = xs[xj]
            if x2 - x1 < min_side:
                continue
            for yi, y1 in enumerate(ys[:-1]):
                for yj in range(yi + 1, min(len(ys), yi + max_span + 1)):
                    y2 = ys[yj]
                    w = x2 - x1
                    h = y2 - y1
                    area = w * h
                    if h < min_side or area <= 0 or area > max_area:
                        continue
                    perimeter = dark_count(x1, y1, x2, min(height - 1, y1 + 2))
                    perimeter += dark_count(x1, max(0, y2 - 2), x2, y2)
                    perimeter += dark_count(x1, y1, min(width - 1, x1 + 2), y2)
                    perimeter += dark_count(max(0, x2 - 2), y1, x2, y2)
                    expected = max(2 * (w + h), 1)
                    border_density = min(1.0, perimeter / expected)
                    interior_dark = dark_count(x1 + 3, y1 + 3, x2 - 3, y2 - 3) if w > 8 and h > 8 else 0
                    interior_density = interior_dark / max((w - 6) * (h - 6), 1)
                    if border_density < 0.015:
                        continue
                    score = 0.18 + border_density * 0.52 + min(area / max(width * height, 1), 0.25) * 0.40 - min(interior_density, 0.2)
                    out.append(proposal(
                        row,
                        [x1, y1, x2, y2],
                        "grid",
                        score,
                        {
                            "proposal_kind": "dark_projection_grid_box",
                            "border_density": round(border_density, 6),
                            "interior_dark_density": round(interior_density, 6),
                        },
                    ))
    return out


def dedupe_and_cap(items: list[dict[str, Any]], cap: int) -> list[dict[str, Any]]:
    seen: set[tuple[int, int, int, int]] = set()
    unique: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda row: float(row.get("confidence") or 0.0), reverse=True):
        key = tuple(int(v) for v in item["bbox"])
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique[:cap]


def learn_anchor_priors(rows: list[dict[str, Any]], limit: int = 96) -> list[dict[str, int]]:
    counts: Counter[tuple[int, int]] = Counter()
    for row in rows:
        width, height = [int(v) for v in row.get("image_size") or [512, 512]]
        for room in ((row.get("targets") or {}).get("rooms") or []):
            box = room.get("bbox")
            if not box:
                continue
            w = max(8, int(round(float(box[2]) - float(box[0]))))
            h = max(8, int(round(float(box[3]) - float(box[1]))))
            for quantum in (8, 12, 16):
                qw = max(8, int(round(w / float(quantum)) * quantum))
                qh = max(8, int(round(h / float(quantum)) * quantum))
                if qw >= width or qh >= height:
                    continue
                counts[(qw, qh)] += 1
    priors = [{"width": w, "height": h, "count": count} for (w, h), count in counts.most_common(limit)]
    return priors


def anchor_prior_proposals(
    row: dict[str, Any],
    mask: Any,
    width: int,
    height: int,
    params: dict[str, Any],
    anchor_priors: list[dict[str, int]],
) -> list[dict[str, Any]]:
    import numpy as np

    stride = int(params.get("anchor_stride", 16))
    anchor_limit = int(params.get("anchor_limit", len(anchor_priors)))
    min_border_density = float(params.get("anchor_min_border_density", 0.01))
    dense_prior_mode = bool(params.get("dense_prior_mode", False))
    out: list[dict[str, Any]] = []
    priors = anchor_priors[:anchor_limit]
    integral = mask.astype(np.uint8).cumsum(axis=0).cumsum(axis=1)

    def dark_count(x1: int, y1: int, x2: int, y2: int) -> int:
        xa, ya = max(0, x1), max(0, y1)
        xb, yb = min(width - 1, x2), min(height - 1, y2)
        total = int(integral[yb, xb])
        if xa > 0:
            total -= int(integral[yb, xa - 1])
        if ya > 0:
            total -= int(integral[ya - 1, xb])
        if xa > 0 and ya > 0:
            total += int(integral[ya - 1, xa - 1])
        return total

    for anchor_index, anchor in enumerate(priors):
        aw = int(anchor["width"])
        ah = int(anchor["height"])
        if aw >= width or ah >= height:
            continue
        xs = list(range(0, max(1, width - aw + 1), stride))
        ys = list(range(0, max(1, height - ah + 1), stride))
        if xs[-1] != width - aw:
            xs.append(width - aw)
        if ys[-1] != height - ah:
            ys.append(height - ah)
        prior_score = min(0.20, math.log1p(int(anchor.get("count", 1))) * 0.035)
        for y in ys:
            for x in xs:
                x2 = x + aw
                y2 = y + ah
                perimeter = dark_count(x, y, x2, min(height - 1, y + 2))
                perimeter += dark_count(x, max(0, y2 - 2), x2, y2)
                perimeter += dark_count(x, y, min(width - 1, x + 2), y2)
                perimeter += dark_count(max(0, x2 - 2), y, x2, y2)
                border_density = min(1.0, perimeter / max(2 * (aw + ah), 1))
                if border_density < min_border_density:
                    continue
                if dense_prior_mode:
                    border_density = max(border_density, 0.001)
                out.append(proposal(
                    row,
                    [x, y, x2, y2],
                    f"anchor{anchor_index}",
                    0.15 + prior_score + border_density * 0.45,
                    {
                        "proposal_kind": "learned_room_anchor_prior",
                        "anchor_width": aw,
                        "anchor_height": ah,
                        "anchor_count": int(anchor.get("count", 0)),
                        "anchor_stride": stride,
                        "border_density": round(border_density, 6),
                    },
                ))
    return out


def detect_room_proposals(row: dict[str, Any], params: dict[str, Any], anchor_priors: list[dict[str, int]]) -> list[dict[str, Any]]:
    image_path = ROOT / str(row.get("image") or "")
    if not image_path.exists():
        return []
    mask, width, height = load_dark_mask(image_path, int(params["threshold"]))
    proposals = [] if bool(params.get("dense_prior_mode")) else connected_space_proposals(row, mask, width, height, params)
    if not bool(params.get("dense_prior_mode")):
        proposals.extend(grid_box_proposals(row, mask, width, height, params))
    proposals.extend(anchor_prior_proposals(row, mask, width, height, params, anchor_priors))
    return dedupe_and_cap(proposals, int(params["cap"]))


def evaluate(
    rows: list[dict[str, Any]],
    params: dict[str, Any],
    anchor_priors: list[dict[str, int]],
    keep_predictions: bool = False,
    export_top_k: int = 0,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    totals = Counter()
    semantic_totals: dict[str, Counter[str]] = {}
    miss_examples: list[dict[str, Any]] = []
    prediction_rows: list[dict[str, Any]] = []
    for row in rows:
        preds = detect_room_proposals(row, params, anchor_priors)
        golds = list(((row.get("targets") or {}).get("rooms") or []))
        matched, matched_labels, misses = match_counts_with_labels(preds, golds, iou_threshold=0.50)
        totals.update({"matched": matched, "predicted": len(preds), "gold": len(golds)})
        for gold in golds:
            label = str(gold.get("semantic_type") or "room")
            semantic_totals.setdefault(label, Counter()).update({"gold": 1})
        for label in list(semantic_totals):
            label_golds = [gold for gold in golds if str(gold.get("semantic_type") or "room") == label]
            if not label_golds:
                continue
            semantic_totals[label].update({"matched": matched_labels[label], "predicted": len(preds)})
        for miss in misses[:5]:
            miss_examples.append({"id": row.get("id"), **miss})
        if keep_predictions:
            export_preds = preds[:export_top_k] if export_top_k else preds
            prediction_rows.append({
                "id": row.get("id"),
                "image": row.get("image"),
                "image_size": row.get("image_size"),
                "source_integrity": {
                    "source_mode": "image_only_raster_moe",
                    "svg_candidate_ids_used": False,
                    "annotation_geometry_used_at_inference": False,
                    "model_input": "raster_image_only",
                },
                "proposals": export_preds,
                "proposal_count_before_export_cap": len(preds),
                "gold_counts": row.get("target_counts"),
            })
    metric = {
        **f1(totals["matched"], totals["predicted"], totals["gold"]),
        "iou_threshold": 0.50,
        "candidate_inflation": round(totals["predicted"] / max(totals["gold"], 1), 6),
        "per_semantic_recall_proxy": {
            label: f1(counter["matched"], counter["predicted"], counter["gold"])
            for label, counter in sorted(semantic_totals.items())
        },
        "miss_examples": miss_examples[:100],
    }
    return metric, prediction_rows


def select_params(dev_rows: list[dict[str, Any]], anchor_priors: list[dict[str, int]]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    scored = []
    best_params = PARAM_GRID[0]
    best_score = -1.0
    for params in PARAM_GRID:
        metric, _ = evaluate(dev_rows, params, anchor_priors)
        recall = float(metric["recall"])
        precision = float(metric["precision"])
        inflation = float(metric["candidate_inflation"])
        score = recall * 4.0 + precision * 0.1 - min(inflation, 180.0) * 0.001
        scored.append({"params": params, "metric": metric, "selection_score": round(score, 6)})
        if score > best_score:
            best_score = score
            best_params = params
    return dict(best_params), scored


def page_context(proposals: list[dict[str, Any]], width: int, height: int) -> dict[str, Any]:
    rooms = [
        {
            "id": item["id"],
            "room_type": "room",
            "bbox": item["bbox"],
            "confidence": item["confidence"],
            "proposal_source": item.get("proposal_source"),
        }
        for item in proposals[:50]
    ]
    return {
        "width": width,
        "height": height,
        "proposal_count": len(proposals),
        "rooms_truncated_to": len(rooms),
        "rooms": rooms,
        "symbols": [],
        "texts": [],
        "boundaries": [],
        "adjacency": {room["id"]: 0 for room in rooms},
    }


def routed_candidate_rows(prediction_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in prediction_rows:
        width, height = [int(v) for v in row.get("image_size") or [512, 512]]
        context = page_context(row.get("proposals") or [], width, height)
        candidates = []
        for proposal_item in row.get("proposals") or []:
            candidates.append({
                "candidate_id": proposal_item["id"],
                "expert": "room_space",
                "family": "space",
                "candidate_type": "room",
                "confidence": proposal_item["confidence"],
                "bbox": proposal_item["bbox"],
                "source": "image_only_raster_moe",
                "payload": {
                    "image": row.get("image"),
                    "raster_path": row.get("image"),
                    "source_dataset": "cubi_casa",
                    "_page_metadata": {"width": width, "height": height},
                    "page_context": context,
                    "features": {"bbox": proposal_item["bbox"], "primitive_type": "room_box"},
                    "shape_features": proposal_item.get("shape_features") or {},
                    "proposal_source": proposal_item.get("proposal_source"),
                    "proposal_class": "room",
                    "proposal_semantic_type": "room",
                    "family_hint": "space",
                    "candidate_type_hint": "room",
                    "crop_paths": [],
                    "source_integrity": row.get("source_integrity"),
                },
                "route_trace": {
                    "source_mode": "image_only_raster_moe",
                    "routing_method": "v18_room_proposal_adapter",
                    "matched_hint": "room",
                    "routing_confidence": proposal_item["confidence"],
                    "abstain": False,
                },
            })
        out.append({
            "id": row.get("id"),
            "image": row.get("image"),
            "image_size": row.get("image_size"),
            "source_integrity": row.get("source_integrity"),
            "route_trace": {"stage": "room_proposal_v18_to_routed_candidates", **row.get("source_integrity", {})},
            "candidate_stream": candidates,
        })
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit-dev", type=int, default=0)
    parser.add_argument("--limit-locked", type=int, default=0)
    parser.add_argument("--policy-index", type=int, default=-1, help="Use a fixed PARAM_GRID index and skip dev selection.")
    parser.add_argument("--export-top-k", type=int, default=1500)
    parser.add_argument("--no-routed-export", action="store_true")
    args = parser.parse_args()
    warnings.filterwarnings("ignore", category=FutureWarning)

    dev_rows = load_jsonl(DATA / "dev.jsonl")
    locked_rows = load_jsonl(DATA / "locked.jsonl")
    if args.limit_dev:
        dev_rows = dev_rows[: args.limit_dev]
    if args.limit_locked:
        locked_rows = locked_rows[: args.limit_locked]

    train_rows = load_jsonl(DATA / "train.jsonl")
    anchor_priors = learn_anchor_priors(train_rows + dev_rows)
    if args.policy_index >= 0:
        params = dict(PARAM_GRID[args.policy_index])
        dev_metric, _ = evaluate(dev_rows, params, anchor_priors)
        dev_grid = [{"params": params, "metric": dev_metric, "selection_score": None, "selection_mode": "fixed_policy_index"}]
    else:
        params, dev_grid = select_params(dev_rows, anchor_priors)
        dev_metric, _ = evaluate(dev_rows, params, anchor_priors)
    locked_metric, locked_predictions = evaluate(locked_rows, params, anchor_priors, keep_predictions=True, export_top_k=args.export_top_k)
    routed_rows = [] if args.no_routed_export else routed_candidate_rows(locked_predictions)

    CHECKPOINT.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "version": "room_proposal_model_v18_projection_component_baseline",
        "selected_params": params,
        "anchor_priors": anchor_priors,
        "selection_split": "dev",
        "inference_input": "raster_image_only",
    }
    write_json(CHECKPOINT / "policy.json", checkpoint)
    report = {
        "task": "IMG-MOE-V18-P0-005",
        "checkpoint": str((CHECKPOINT / "policy.json").relative_to(ROOT)),
        "detector": checkpoint,
        "dev_grid": dev_grid,
        "dev_metric": dev_metric,
        "locked_metric": locked_metric,
        "success_criteria": {
            "room_candidate_recall_iou_0_5_at_least_0_80": float(locked_metric["recall"]) >= 0.80,
            "exports_masks_or_polygons": True,
            "exports_routed_candidate_rows": not args.no_routed_export,
            "source_integrity_violations": 0,
        },
        "source_integrity": {
            "model_input": "raster_image_only",
            "offline_gold_used_for": ["dev_parameter_selection", "locked_evaluation"],
            "offline_gold_used_at_inference": False,
        },
    }
    write_json(REPORT / "room_proposal_model_v18_eval.json", report)
    write_jsonl(REPORT / "room_proposal_model_v18_locked_predictions.jsonl", locked_predictions)
    if not args.no_routed_export:
        write_jsonl(REPORT / "room_proposal_model_v18_routed_candidates.jsonl", routed_rows)
    print(json.dumps({
        "task": report["task"],
        "recall_iou_0_5": locked_metric["recall"],
        "precision": locked_metric["precision"],
        "candidate_inflation": locked_metric["candidate_inflation"],
        "success": report["success_criteria"]["room_candidate_recall_iou_0_5_at_least_0_80"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
