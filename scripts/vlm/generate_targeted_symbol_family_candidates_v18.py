#!/usr/bin/env python3
"""Generate room-constrained targeted symbol-family candidates from raster images."""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
DEFAULT_INPUT = REPORT / "detector_adapter_v18_symbol_proposal_combined.jsonl"
DEFAULT_MODEL = ROOT / "checkpoints/targeted_symbol_family_proposal_v18/model.json"
DEFAULT_OUTPUT = REPORT / "detector_adapter_v18_targeted_symbol_family.jsonl"
DEFAULT_AUDIT = REPORT / "detector_adapter_v18_targeted_symbol_family_audit.json"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_topology_relations_v18 import bbox, integrity, iou, write_json, write_jsonl  # noqa: E402
from generate_symbol_recall_candidates_v18 import (  # noqa: E402
    box_area,
    clip_box,
    crop_stats,
    image_array,
    parse_csv_ints,
    stream_counts,
)
from nms_topology_relations_v18 import load_jsonl  # noqa: E402
from train_targeted_symbol_family_proposal_expert_v18 import score_row  # noqa: E402


def candidate_stream(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(((row.get("scene_graph") or {}).get("candidate_stream") or []))


def family_boxes(stream: list[dict[str, Any]], family: str) -> list[list[float]]:
    boxes: list[list[float]] = []
    for cand in stream:
        if cand.get("family") != family:
            continue
        box = bbox(cand.get("bbox"))
        if box is not None:
            boxes.append(box)
    return boxes


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def contains_point(box: list[float], x: float, y: float, margin: float = 0.0) -> bool:
    return box[0] - margin <= x <= box[2] + margin and box[1] - margin <= y <= box[3] + margin


def in_any_room(box: list[float], room_boxes: list[list[float]], margin: float) -> bool:
    cx, cy = center(box)
    return any(contains_point(room, cx, cy, margin) for room in room_boxes)


def nearest_room_distance(box: list[float], room_boxes: list[list[float]]) -> float:
    cx, cy = center(box)
    best = 1e9
    for room in room_boxes:
        dx = max(room[0] - cx, 0.0, cx - room[2])
        dy = max(room[1] - cy, 0.0, cy - room[3])
        best = min(best, math.hypot(dx, dy))
    return 0.0 if best == 1e9 else best


def prior_sizes(model: dict[str, Any]) -> list[tuple[str, float, float]]:
    out: list[tuple[str, float, float]] = []
    for family, spec in (model.get("families") or {}).items():
        priors = spec.get("shape_priors") or {}
        width = (priors.get("width") or {}).get("p50")
        height = (priors.get("height") or {}).get("p50")
        if width and height:
            out.append((str(family), float(width), float(height)))
        width90 = (priors.get("width") or {}).get("p90")
        height90 = (priors.get("height") or {}).get("p90")
        if width90 and height90 and (abs(float(width90) - float(width or 0.0)) > 2.0 or abs(float(height90) - float(height or 0.0)) > 2.0):
            out.append((str(family), float(width90), float(height90)))
    return out


def component_boxes(arr: np.ndarray, threshold: int, args: argparse.Namespace) -> list[dict[str, Any]]:
    import cv2

    mask = arr <= threshold
    n_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask.astype("uint8"), 8)
    out: list[dict[str, Any]] = []
    for idx in range(1, int(n_labels)):
        x, y, w, h, area = [int(v) for v in stats[idx]]
        if area < args.min_component_area or area > args.max_component_area:
            continue
        if w < args.min_side or h < args.min_side or w > args.max_side or h > args.max_side:
            continue
        aspect = w / max(h, 1)
        if aspect < args.min_aspect or aspect > args.max_aspect:
            continue
        fill = area / max(w * h, 1)
        if fill < args.min_fill:
            continue
        cx, cy = [float(v) for v in centroids[idx]]
        out.append(
            {
                "component_bbox": [x, y, x + w, y + h],
                "component_center": [round(cx, 3), round(cy, 3)],
                "component_area": int(area),
                "component_fill": round(float(fill), 6),
                "threshold": int(threshold),
            }
        )
    return out


def raw_windows_for_component(
    component: dict[str, Any],
    model: dict[str, Any],
    width: int,
    height: int,
    args: argparse.Namespace,
) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    cbox = component["component_bbox"]
    cx, cy = component["component_center"]
    for pad in args.component_pads:
        box = clip_box([cbox[0] - pad, cbox[1] - pad, cbox[2] + pad, cbox[3] + pad], width, height)
        if box is not None:
            out.append({**component, "bbox": box, "proposal_kind": f"targeted_component_pad_{pad}"})
    for family, prior_w, prior_h in prior_sizes(model):
        scale = args.prior_scale
        box = clip_box([cx - prior_w * scale / 2.0, cy - prior_h * scale / 2.0, cx + prior_w * scale / 2.0, cy + prior_h * scale / 2.0], width, height)
        if box is not None:
            out.append({**component, "bbox": box, "proposal_kind": f"targeted_{family}_prior_p50_or_p90", "prior_family": family})
    return out


def proposal_record(
    row: dict[str, Any],
    arr: np.ndarray,
    stream: list[dict[str, Any]],
    raw: dict[str, Any],
    room_boxes: list[list[float]],
) -> dict[str, Any]:
    box = raw["bbox"]
    counts = stream_counts(stream)
    return {
        "id": f"{row.get('id')}|targeted_symbol_family_window|{box[0]}_{box[1]}_{box[2]}_{box[3]}",
        "row_id": row.get("id"),
        "image": row.get("image"),
        "image_size": row.get("image_size") or [arr.shape[1], arr.shape[0]],
        "bbox": box,
        "features": {
            "existing_symbol_candidate_count": counts.get("symbol", 0),
            "space_candidate_count": counts.get("space", 0),
            "boundary_candidate_count": counts.get("boundary", 0),
            "text_candidate_count": counts.get("text", 0),
            **crop_stats(arr, box),
            "component_area": float(raw.get("component_area") or 0.0),
            "component_fill": float(raw.get("component_fill") or 0.0),
            "room_center_distance": nearest_room_distance(box, room_boxes),
        },
    }


def score_proposal(
    row: dict[str, Any],
    arr: np.ndarray,
    stream: list[dict[str, Any]],
    raw: dict[str, Any],
    room_boxes: list[list[float]],
    model: dict[str, Any],
    image_cache: dict[str, Image.Image],
) -> dict[str, Any]:
    record = proposal_record(row, arr, stream, raw, room_boxes)
    scores = score_row(record, model, image_cache)
    family_scores = scores.get("family_scores") or {}
    return {
        **raw,
        "objectness_score": round(float(scores["objectness_score"]), 6),
        "predicted_family": scores.get("predicted_family"),
        "family_scores": family_scores,
    }


def prefilter_score(item: dict[str, Any]) -> float:
    area = float(item.get("component_area") or 0.0)
    fill = float(item.get("component_fill") or 0.0)
    box = item.get("bbox") if isinstance(item.get("bbox"), list) else [0.0, 0.0, 1.0, 1.0]
    window_area = box_area(box)
    compact_bonus = min(area, 80.0) / 80.0
    large_penalty = max(window_area - 1600.0, 0.0) / 3200.0
    return fill + 0.5 * compact_bonus - large_penalty


def prefilter_windows(raw_windows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(raw_windows) <= limit:
        return raw_windows
    ranked = sorted(raw_windows, key=prefilter_score, reverse=True)
    return ranked[:limit]


def dedupe_ranked(items: list[dict[str, Any]], existing_boxes: list[list[float]], args: argparse.Namespace) -> list[dict[str, Any]]:
    ranked = sorted(
        items,
        key=lambda item: (float(item.get("objectness_score") or 0.0), -nearest_room_distance(item["bbox"], []), -box_area(item["bbox"])),
        reverse=True,
    )
    kept: list[dict[str, Any]] = []
    for item in ranked:
        if float(item.get("objectness_score") or 0.0) < args.score_threshold:
            continue
        if any(iou(item["bbox"], existing) > args.existing_duplicate_iou for existing in existing_boxes):
            continue
        if any(iou(item["bbox"], prev["bbox"]) > args.new_duplicate_iou for prev in kept):
            continue
        kept.append(item)
        if len(kept) >= args.max_added_per_row:
            break
    return kept


def make_candidate(row: dict[str, Any], item: dict[str, Any], index: int) -> dict[str, Any]:
    row_id = str(row.get("id"))
    box = item["bbox"]
    family = str(item.get("predicted_family") or "symbol")
    candidate_id = f"{row_id}_targeted_symbol_family_proposal_v18_{index:03d}_{box[0]}_{box[1]}_{box[2]}_{box[3]}"
    payload = {
        "candidate_kind": "targeted_symbol_family_proposal_v18",
        "proposal_kind": item.get("proposal_kind"),
        "source": "targeted_symbol_family_proposal_v18",
        "symbol_type": "symbol",
        "typed_symbol_type": family,
        "type_label_adopted": False,
        "predicted_symbol_family": family,
        "objectness_score": item["objectness_score"],
        "family_scores": item.get("family_scores"),
        "raster_threshold": item.get("threshold"),
        "component_bbox": item.get("component_bbox"),
        "component_area": item.get("component_area"),
        "component_fill": item.get("component_fill"),
        "prior_family": item.get("prior_family"),
    }
    return {
        "candidate_id": candidate_id,
        "candidate_contract_version": "detector_candidate_contract_v1",
        "row_id": row_id,
        "family": "symbol",
        "route": "symbol_fixture",
        "candidate_type": "symbol",
        "bbox": box,
        "confidence": item["objectness_score"],
        "payload": payload,
        "source_integrity": integrity(),
        "provenance": {
            "input_source": "targeted_symbol_family_proposal_v18",
            "raw_candidate_id": candidate_id,
            "row_id": row_id,
            "family": "symbol",
            "route": "symbol_fixture",
            "raster_only": True,
            "image": row.get("image"),
        },
        "audit_trace": {
            "stage": "targeted_symbol_family_proposal_v18",
            "proposal_kind": item.get("proposal_kind"),
            "bbox": box,
            "objectness_score": item["objectness_score"],
            "predicted_symbol_family": family,
            "source_integrity": integrity(),
        },
    }


def generate_for_row(
    row: dict[str, Any],
    model: dict[str, Any],
    image_np_cache: dict[str, np.ndarray],
    image_pil_cache: dict[str, Image.Image],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Counter[str], list[dict[str, Any]]]:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    arr = image_array(out, image_np_cache)
    stream = candidate_stream(out)
    counts: Counter[str] = Counter({"rows": 1, "input_candidates": len(stream)})
    if arr is None:
        counts["missing_image_rows"] += 1
        return out, counts, []

    room_boxes = sorted(family_boxes(stream, "space"), key=box_area, reverse=True)[: args.max_room_boxes]
    existing_symbols = family_boxes(stream, "symbol")
    raw_components: list[dict[str, Any]] = []
    for threshold in args.thresholds:
        raw_components.extend(component_boxes(arr, int(threshold), args))
    counts["raw_components"] = len(raw_components)
    raw_windows: list[dict[str, Any]] = []
    for component in raw_components:
        raw_windows.extend(raw_windows_for_component(component, model, arr.shape[1], arr.shape[0], args))
    counts["raw_windows_before_room_filter"] = len(raw_windows)
    raw_windows = [item for item in raw_windows if in_any_room(item["bbox"], room_boxes, args.room_margin)]
    counts["raw_windows_after_room_filter"] = len(raw_windows)
    raw_windows = prefilter_windows(raw_windows, args.max_scored_windows_per_row)
    counts["raw_windows_after_prefilter"] = len(raw_windows)
    scored = [score_proposal(out, arr, stream, item, room_boxes, model, image_pil_cache) for item in raw_windows]
    selected = dedupe_ranked(scored, existing_symbols, args)
    added = [make_candidate(out, item, index) for index, item in enumerate(selected)]
    new_stream = stream + added
    scene = dict(out.get("scene_graph") if isinstance(out.get("scene_graph"), dict) else {})
    scene["candidate_stream"] = new_stream
    scene["candidate_counts"] = stream_counts(new_stream)
    scene["targeted_symbol_family_proposal_v18"] = {
        "enabled": True,
        "added_candidates": len(added),
        "raw_windows_after_room_filter": len(raw_windows),
        "score_threshold": args.score_threshold,
        "max_added_per_row": args.max_added_per_row,
        "room_margin": args.room_margin,
    }
    out["scene_graph"] = scene
    counts["added_targeted_symbol_family_candidates"] = len(added)
    counts["output_candidates"] = len(new_stream)
    examples = [
        {
            "row_id": out.get("id"),
            "candidate_id": cand.get("candidate_id"),
            "bbox": cand.get("bbox"),
            "score": cand.get("confidence"),
            "predicted_symbol_family": (cand.get("payload") or {}).get("predicted_symbol_family"),
            "proposal_kind": (cand.get("payload") or {}).get("proposal_kind"),
        }
        for cand in added[:10]
    ]
    return out, counts, examples


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    rows = load_jsonl(Path(args.input))
    if args.limit:
        rows = rows[: args.limit]
    image_np_cache: dict[str, np.ndarray] = {}
    image_pil_cache: dict[str, Image.Image] = {}
    out_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    per_row_added: Counter[int] = Counter()
    examples: list[dict[str, Any]] = []
    for row in rows:
        out, row_counts, row_examples = generate_for_row(row, model, image_np_cache, image_pil_cache, args)
        counts.update(row_counts)
        per_row_added[int(row_counts.get("added_targeted_symbol_family_candidates", 0))] += 1
        if len(examples) < 100:
            examples.extend(row_examples[: max(0, 100 - len(examples))])
        out_rows.append(out)
    audit = {
        "task": "IMG-MOE-V18-REBUILD-001.step_j4_generate_targeted_symbol_family_candidates",
        "input": str(args.input),
        "output": str(args.output),
        "model": str(args.model),
        "rows": len(out_rows),
        "counts": dict(counts),
        "added_per_row_histogram": dict(sorted(per_row_added.items())),
        "examples": examples,
        "params": {
            "thresholds": args.thresholds,
            "score_threshold": args.score_threshold,
            "max_added_per_row": args.max_added_per_row,
            "room_margin": args.room_margin,
            "max_room_boxes": args.max_room_boxes,
            "prior_scale": args.prior_scale,
            "component_pads": args.component_pads,
            "max_scored_windows_per_row": args.max_scored_windows_per_row,
        },
        "source_integrity": integrity(),
        "gold_loaded": False,
        "gold_used_for_inference": False,
        "adopted": False,
    }
    return out_rows, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--thresholds", type=parse_csv_ints, default=parse_csv_ints("185,205,225"))
    parser.add_argument("--component-pads", type=parse_csv_ints, default=parse_csv_ints("1,3,5"))
    parser.add_argument("--min-component-area", type=int, default=1)
    parser.add_argument("--max-component-area", type=int, default=260)
    parser.add_argument("--min-side", type=int, default=1)
    parser.add_argument("--max-side", type=int, default=44)
    parser.add_argument("--min-aspect", type=float, default=0.08)
    parser.add_argument("--max-aspect", type=float, default=12.0)
    parser.add_argument("--min-fill", type=float, default=0.02)
    parser.add_argument("--room-margin", type=float, default=4.0)
    parser.add_argument("--max-room-boxes", type=int, default=80)
    parser.add_argument("--prior-scale", type=float, default=1.15)
    parser.add_argument("--score-threshold", type=float, default=0.8)
    parser.add_argument("--max-added-per-row", type=int, default=16)
    parser.add_argument("--max-scored-windows-per-row", type=int, default=1800)
    parser.add_argument("--existing-duplicate-iou", type=float, default=0.82)
    parser.add_argument("--new-duplicate-iou", type=float, default=0.72)
    args = parser.parse_args()
    rows, audit = run(args)
    write_jsonl(Path(args.output), rows)
    write_json(Path(args.audit), audit)
    print(json.dumps({"rows": audit["rows"], "counts": audit["counts"], "added_per_row_histogram": audit["added_per_row_histogram"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
