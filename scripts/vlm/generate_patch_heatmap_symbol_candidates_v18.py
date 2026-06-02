#!/usr/bin/env python3
"""Generate auditable symbol candidates from learned raster patch heatmaps.

This is an inference-time candidate generator. It loads a patch scorer trained
offline, scores dense raster patches, clusters high responses into symbol-body
windows, and appends generic symbol candidates without loading gold labels.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
DEFAULT_INPUT = REPORT / "detector_adapter_v18_symbol_proposal_combined.jsonl"
DEFAULT_MODEL = ROOT / "checkpoints/patch_symbol_body_segmenter_v18/model.json"
DEFAULT_OUTPUT = REPORT / "detector_adapter_v18_patch_heatmap_symbol.jsonl"
DEFAULT_AUDIT = REPORT / "detector_adapter_v18_patch_heatmap_symbol_audit.json"

if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from build_topology_relations_v18 import bbox, center, contains_point, integrity, iou, write_json, write_jsonl  # noqa: E402
from generate_symbol_recall_candidates_v18 import image_array, parse_csv_ints, stream_counts  # noqa: E402
from nms_topology_relations_v18 import load_jsonl  # noqa: E402
from train_patch_symbol_body_segmenter_v18 import dense_patch_boxes, evidence_arrays, patch_features, score  # noqa: E402


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


def box_area(box: list[float] | None) -> float:
    if box is None:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def clip_rect(box: list[float], width: int, height: int) -> list[float] | None:
    x1 = max(0.0, min(float(width - 1), float(box[0])))
    y1 = max(0.0, min(float(height - 1), float(box[1])))
    x2 = max(x1 + 1.0, min(float(width), float(box[2])))
    y2 = max(y1 + 1.0, min(float(height), float(box[3])))
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]


def center_distance(left: list[float], right: list[float]) -> float:
    lx, ly = center(left)
    rx, ry = center(right)
    return math.hypot(lx - rx, ly - ry)


def in_any_room(box: list[float], room_boxes: list[list[float]], margin: float) -> bool:
    if not room_boxes:
        return True
    cx, cy = center(box)
    return any(contains_point(room, cx, cy, margin) for room in room_boxes)


def score_patch(row: dict[str, Any], arr: np.ndarray, evidence: dict[str, np.ndarray], patch_box: list[float], model: dict[str, Any]) -> dict[str, Any]:
    record = {
        "row_id": row.get("id"),
        "image": row.get("image"),
        "image_size": row.get("image_size"),
        "bbox": patch_box,
        "features": patch_features(row, arr, evidence, patch_box),
    }
    return {
        "bbox": patch_box,
        "patch_score": round(score(record, model), 6),
        "features": record["features"],
    }


def patch_rank_key(item: dict[str, Any]) -> tuple[float, float, float]:
    features = item.get("features") if isinstance(item.get("features"), dict) else {}
    dark = float(features.get("dark_density_205") or 0.0)
    edge_touch = float(features.get("edge_touch_dark_ratio") or 0.0)
    return (float(item.get("patch_score") or 0.0), dark, -edge_touch)


def cluster_patches(scored: list[dict[str, Any]], args: argparse.Namespace, width: int, height: int) -> list[dict[str, Any]]:
    clusters: list[dict[str, Any]] = []
    for item in sorted(scored, key=patch_rank_key, reverse=True):
        box = item["bbox"]
        best_index = None
        best_dist = 1e9
        for index, cluster in enumerate(clusters):
            dist = center_distance(box, cluster["center_box"])
            if dist <= args.cluster_radius and dist < best_dist:
                best_index = index
                best_dist = dist
        if best_index is None:
            clusters.append(
                {
                    "patches": [item],
                    "center_box": box,
                    "max_score": float(item["patch_score"]),
                    "score_sum": float(item["patch_score"]),
                    "bbox_union": list(box),
                }
            )
        else:
            cluster = clusters[best_index]
            cluster["patches"].append(item)
            cluster["score_sum"] += float(item["patch_score"])
            if float(item["patch_score"]) > float(cluster["max_score"]):
                cluster["max_score"] = float(item["patch_score"])
                cluster["center_box"] = box
            union = cluster["bbox_union"]
            cluster["bbox_union"] = [min(union[0], box[0]), min(union[1], box[1]), max(union[2], box[2]), max(union[3], box[3])]

    out: list[dict[str, Any]] = []
    for cluster in clusters:
        patches = cluster["patches"]
        max_box = cluster["center_box"]
        cx, cy = center(max_box)
        side = max(max_box[2] - max_box[0], max_box[3] - max_box[1]) * args.window_scale
        union = cluster["bbox_union"]
        half = side / 2.0
        candidate_box = [
            min(union[0], cx - half) - args.window_pad,
            min(union[1], cy - half) - args.window_pad,
            max(union[2], cx + half) + args.window_pad,
            max(union[3], cy + half) + args.window_pad,
        ]
        clipped = clip_rect(candidate_box, width, height)
        if clipped is None:
            continue
        out.append(
            {
                "bbox": clipped,
                "patch_score": round(float(cluster["max_score"]), 6),
                "cluster_score_mean": round(float(cluster["score_sum"]) / max(len(patches), 1), 6),
                "cluster_size": len(patches),
                "peak_patch_bbox": [round(float(v), 3) for v in max_box],
            }
        )
    out.sort(key=lambda item: (float(item["patch_score"]), int(item["cluster_size"]), -box_area(item["bbox"])), reverse=True)
    return out


def dedupe_clusters(clusters: list[dict[str, Any]], existing_boxes: list[list[float]], args: argparse.Namespace) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in clusters:
        if float(item.get("patch_score") or 0.0) < args.score_threshold:
            continue
        box = item["bbox"]
        if any(iou(box, existing) > args.existing_duplicate_iou for existing in existing_boxes):
            continue
        if any(iou(box, prev["bbox"]) > args.new_duplicate_iou or center_distance(box, prev["bbox"]) <= args.new_duplicate_center_distance for prev in kept):
            continue
        kept.append(item)
        if len(kept) >= args.max_added_per_row:
            break
    return kept


def emitted_confidence(item: dict[str, Any], args: argparse.Namespace) -> float:
    value = float(item["patch_score"]) * float(args.confidence_scale)
    value = min(value, float(args.confidence_cap))
    return round(max(0.0, value), 6)


def make_candidate(row: dict[str, Any], item: dict[str, Any], index: int, args: argparse.Namespace) -> dict[str, Any]:
    row_id = str(row.get("id"))
    box = item["bbox"]
    kind = str(args.candidate_kind)
    candidate_id = f"{row_id}_{kind}_{index:03d}_{box[0]}_{box[1]}_{box[2]}_{box[3]}"
    confidence = emitted_confidence(item, args)
    payload = {
        "candidate_kind": kind,
        "source": kind,
        "symbol_type": "symbol",
        "typed_symbol_type": "symbol",
        "type_label_adopted": False,
        "objectness_score": item["patch_score"],
        "patch_score": item["patch_score"],
        "emitted_confidence": confidence,
        "cluster_score_mean": item["cluster_score_mean"],
        "cluster_size": item["cluster_size"],
        "peak_patch_bbox": item["peak_patch_bbox"],
        "model": args.model_tag,
    }
    return {
        "candidate_id": candidate_id,
        "candidate_contract_version": "detector_candidate_contract_v1",
        "row_id": row_id,
        "family": "symbol",
        "route": "symbol_fixture",
        "candidate_type": "symbol",
        "bbox": box,
        "confidence": confidence,
        "payload": payload,
        "source_integrity": integrity(),
        "provenance": {
            "input_source": kind,
            "raw_candidate_id": candidate_id,
            "row_id": row_id,
            "family": "symbol",
            "route": "symbol_fixture",
            "raster_only": True,
            "image": row.get("image"),
        },
        "audit_trace": {
            "stage": kind,
            "bbox": box,
            "patch_score": item["patch_score"],
            "emitted_confidence": confidence,
            "cluster_score_mean": item["cluster_score_mean"],
            "cluster_size": item["cluster_size"],
            "peak_patch_bbox": item["peak_patch_bbox"],
            "source_integrity": integrity(),
        },
    }


def generate_for_row(
    row: dict[str, Any],
    model: dict[str, Any],
    image_cache: dict[str, np.ndarray],
    args: argparse.Namespace,
) -> tuple[dict[str, Any], Counter[str], list[dict[str, Any]]]:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    arr = image_array(out, image_cache)
    stream = candidate_stream(out)
    counts: Counter[str] = Counter({"rows": 1, "input_candidates": len(stream)})
    if arr is None:
        counts["missing_image_rows"] += 1
        return out, counts, []
    height, width = arr.shape
    evidence = evidence_arrays(arr)
    room_boxes = sorted(family_boxes(stream, "space"), key=box_area, reverse=True)[: args.max_room_boxes]
    existing_symbols = family_boxes(stream, "symbol")
    patch_boxes = dense_patch_boxes(width, height, args.stride, args.patch_sizes)
    counts["raw_patch_windows"] = len(patch_boxes)
    if not args.disable_room_filter:
        patch_boxes = [box for box in patch_boxes if in_any_room(box, room_boxes, args.room_margin)]
    counts["patch_windows_after_room_filter"] = len(patch_boxes)
    scored = [score_patch(out, arr, evidence, patch_box, model) for patch_box in patch_boxes]
    if args.max_selected_patches_per_row > 0:
        scored = sorted(scored, key=patch_rank_key, reverse=True)[: args.max_selected_patches_per_row]
    counts["selected_patch_responses"] = len(scored)
    clusters = cluster_patches(scored, args, width, height)
    counts["clusters_before_dedupe"] = len(clusters)
    selected = dedupe_clusters(clusters, existing_symbols, args)
    added = [make_candidate(out, item, index, args) for index, item in enumerate(selected)]
    new_stream = stream + added
    scene = dict(out.get("scene_graph") if isinstance(out.get("scene_graph"), dict) else {})
    scene["candidate_stream"] = new_stream
    scene["candidate_counts"] = stream_counts(new_stream)
    scene[str(args.candidate_kind)] = {
        "enabled": True,
        "added_candidates": len(added),
        "patch_sizes": args.patch_sizes,
        "stride": args.stride,
        "max_selected_patches_per_row": args.max_selected_patches_per_row,
        "max_added_per_row": args.max_added_per_row,
            "score_threshold": args.score_threshold,
            "confidence_scale": args.confidence_scale,
            "confidence_cap": args.confidence_cap,
            "cluster_radius": args.cluster_radius,
        "room_filter_enabled": not args.disable_room_filter,
    }
    out["scene_graph"] = scene
    counts["added_patch_heatmap_symbol_candidates"] = len(added)
    counts["output_candidates"] = len(new_stream)
    examples = [
        {
            "row_id": out.get("id"),
            "candidate_id": cand.get("candidate_id"),
            "bbox": cand.get("bbox"),
            "score": cand.get("confidence"),
            "cluster_size": (cand.get("payload") or {}).get("cluster_size"),
        }
        for cand in added[:10]
    ]
    return out, counts, examples


def run(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = json.loads(Path(args.model).read_text(encoding="utf-8"))
    if not args.model_tag:
        args.model_tag = str(model.get("model_type") or Path(args.model).stem)
    rows = load_jsonl(Path(args.input))
    if args.limit:
        rows = rows[: args.limit]
    image_cache: dict[str, np.ndarray] = {}
    out_rows: list[dict[str, Any]] = []
    counts: Counter[str] = Counter()
    per_row_added: Counter[int] = Counter()
    examples: list[dict[str, Any]] = []
    for row in rows:
        out, row_counts, row_examples = generate_for_row(row, model, image_cache, args)
        counts.update(row_counts)
        per_row_added[int(row_counts.get("added_patch_heatmap_symbol_candidates", 0))] += 1
        if len(examples) < 100:
            examples.extend(row_examples[: max(0, 100 - len(examples))])
        out_rows.append(out)
    audit = {
        "task": "IMG-MOE-V18-REBUILD-001.step_j8_generate_patch_heatmap_symbol_candidates",
        "input": str(args.input),
        "output": str(args.output),
        "model": str(args.model),
        "rows": len(out_rows),
        "counts": dict(counts),
        "added_per_row_histogram": dict(sorted(per_row_added.items())),
        "examples": examples,
        "params": {
            "candidate_kind": args.candidate_kind,
            "model_tag": args.model_tag,
            "patch_sizes": args.patch_sizes,
            "stride": args.stride,
            "max_selected_patches_per_row": args.max_selected_patches_per_row,
            "max_added_per_row": args.max_added_per_row,
            "cluster_radius": args.cluster_radius,
            "window_scale": args.window_scale,
            "window_pad": args.window_pad,
        "score_threshold": args.score_threshold,
        "confidence_scale": args.confidence_scale,
        "confidence_cap": args.confidence_cap,
        "room_filter_enabled": not args.disable_room_filter,
            "room_margin": args.room_margin,
            "max_room_boxes": args.max_room_boxes,
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
    parser.add_argument("--patch-sizes", type=parse_csv_ints, default=parse_csv_ints("9,17"))
    parser.add_argument("--stride", type=int, default=12)
    parser.add_argument("--max-selected-patches-per-row", type=int, default=500)
    parser.add_argument("--max-added-per-row", type=int, default=32)
    parser.add_argument("--cluster-radius", type=float, default=12.0)
    parser.add_argument("--window-scale", type=float, default=1.25)
    parser.add_argument("--window-pad", type=float, default=2.0)
    parser.add_argument("--score-threshold", type=float, default=0.0)
    parser.add_argument("--confidence-scale", type=float, default=1.0)
    parser.add_argument("--confidence-cap", type=float, default=1.0)
    parser.add_argument("--candidate-kind", default="patch_heatmap_symbol_v18")
    parser.add_argument("--model-tag", default="")
    parser.add_argument("--existing-duplicate-iou", type=float, default=0.82)
    parser.add_argument("--new-duplicate-iou", type=float, default=0.72)
    parser.add_argument("--new-duplicate-center-distance", type=float, default=4.0)
    parser.add_argument("--room-margin", type=float, default=4.0)
    parser.add_argument("--max-room-boxes", type=int, default=80)
    parser.add_argument("--disable-room-filter", action="store_true")
    args = parser.parse_args()
    rows, audit = run(args)
    write_jsonl(Path(args.output), rows)
    write_json(Path(args.audit), audit)
    print(json.dumps({"rows": audit["rows"], "counts": audit["counts"], "added_per_row_histogram": audit["added_per_row_histogram"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
