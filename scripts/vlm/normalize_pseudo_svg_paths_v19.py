#!/usr/bin/env python3
"""Normalize pseudo-SVG paths into grouped boundary and room candidates."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"


def abs_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


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


def source_integrity() -> dict[str, Any]:
    return {
        "source_mode": "image_only_raster_pseudo_svg_normalized",
        "svg_candidate_ids_used": False,
        "annotation_geometry_used_at_inference": False,
        "model_input": "raster_image_only",
        "runtime_uses_svg_or_cad_geometry": False,
    }


def bbox_iou(left: list[float], right: list[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    la = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    ra = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(la + ra - inter, 1e-9)


def center_covered(pred: list[float], gold: list[float], margin: float = 2.0) -> bool:
    cx = (gold[0] + gold[2]) / 2.0
    cy = (gold[1] + gold[3]) / 2.0
    return pred[0] - margin <= cx <= pred[2] + margin and pred[1] - margin <= cy <= pred[3] + margin


def line_orientation(path: dict[str, Any], max_slope: float) -> tuple[str, float, float, float] | None:
    points = path.get("points") or []
    if len(points) < 2:
        return None
    x1, y1 = [float(v) for v in points[0]]
    x2, y2 = [float(v) for v in points[-1]]
    dx, dy = x2 - x1, y2 - y1
    length = math.hypot(dx, dy)
    if length <= 0:
        return None
    if abs(dy) <= max_slope * max(abs(dx), 1.0):
        return "horizontal", (y1 + y2) / 2.0, min(x1, x2), max(x1, x2)
    if abs(dx) <= max_slope * max(abs(dy), 1.0):
        return "vertical", (x1 + x2) / 2.0, min(y1, y2), max(y1, y2)
    return None


def merge_intervals(items: list[dict[str, Any]], gap: float) -> list[dict[str, Any]]:
    if not items:
        return []
    items = sorted(items, key=lambda item: (item["start"], item["end"]))
    merged: list[dict[str, Any]] = []
    current = dict(items[0])
    current["members"] = list(current["members"])
    for item in items[1:]:
        if item["start"] <= current["end"] + gap:
            current["end"] = max(current["end"], item["end"])
            current["coord_sum"] += item["coord"] * item["weight"]
            current["weight"] += item["weight"]
            current["confidence"] = max(current["confidence"], item["confidence"])
            current["members"].extend(item["members"])
        else:
            current["coord"] = current["coord_sum"] / max(current["weight"], 1e-9)
            merged.append(current)
            current = dict(item)
            current["members"] = list(current["members"])
    current["coord"] = current["coord_sum"] / max(current["weight"], 1e-9)
    merged.append(current)
    return merged


def normalize_lines(paths: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    buckets: dict[tuple[str, int], list[dict[str, Any]]] = defaultdict(list)
    for path in paths:
        if path.get("kind") != "line":
            continue
        parsed = line_orientation(path, args.max_axis_slope)
        if parsed is None:
            continue
        orientation, coord, start, end = parsed
        length = end - start
        if length < args.min_line_length:
            continue
        bucket = int(round(coord / args.coord_bucket))
        confidence = float(path.get("confidence") or min(1.0, length / 120.0))
        buckets[(orientation, bucket)].append(
            {
                "orientation": orientation,
                "coord": coord,
                "coord_sum": coord * max(length, 1.0),
                "start": start,
                "end": end,
                "weight": max(length, 1.0),
                "confidence": confidence,
                "members": [path["id"]],
            }
        )
    normalized: list[dict[str, Any]] = []
    for (_orientation, _bucket), items in buckets.items():
        normalized.extend(merge_intervals(items, args.merge_gap))
    return [item for item in normalized if item["end"] - item["start"] >= args.min_grouped_length]


def boundary_candidate(row_id: str, image_path: str, line: dict[str, Any], index: int, pad: int) -> dict[str, Any]:
    coord = float(line["coord"])
    start = float(line["start"])
    end = float(line["end"])
    if line["orientation"] == "horizontal":
        bbox = [int(start - pad), int(coord - pad), int(end + pad), int(coord + pad + 1)]
        points = [[round(start, 3), round(coord, 3)], [round(end, 3), round(coord, 3)]]
    else:
        bbox = [int(coord - pad), int(start - pad), int(coord + pad + 1), int(end + pad)]
        points = [[round(coord, 3), round(start, 3)], [round(coord, 3), round(end, 3)]]
    length = end - start
    return {
        "candidate_id": f"{row_id}_pseudo_svg_norm_boundary_{index:04d}",
        "row_id": row_id,
        "family": "boundary",
        "route": "wall_opening",
        "candidate_type": "wall_or_vector_line_group",
        "bbox": [max(0, value) for value in bbox],
        "confidence": round(min(1.0, 0.35 + length / 240.0 + min(len(line["members"]), 10) * 0.03), 6),
        "payload": {
            "image": image_path,
            "raster_path": image_path,
            "proposal_source": "pseudo_svg_normalizer_v19",
            "features": {
                "primitive_type": "merged_axis_line",
                "orientation": line["orientation"],
                "length": round(length, 3),
                "coord": round(coord, 3),
                "member_count": len(line["members"]),
                "points": points,
            },
            "member_path_ids": line["members"][:40],
            "source_integrity": source_integrity(),
        },
        "source_integrity": source_integrity(),
    }


def overlap_exists(lines: list[dict[str, Any]], coord: float, start: float, end: float, orientation: str, tol: float) -> bool:
    for line in lines:
        if line["orientation"] != orientation:
            continue
        if abs(float(line["coord"]) - coord) > tol:
            continue
        if max(float(line["start"]), start) <= min(float(line["end"]), end) + tol:
            return True
    return False


def room_candidates(row_id: str, image_path: str, lines: list[dict[str, Any]], args: argparse.Namespace) -> list[dict[str, Any]]:
    horizontals = [line for line in lines if line["orientation"] == "horizontal" and line["end"] - line["start"] >= args.min_room_edge]
    verticals = [line for line in lines if line["orientation"] == "vertical" and line["end"] - line["start"] >= args.min_room_edge]
    xs = sorted({round(float(line["coord"])) for line in verticals})
    ys = sorted({round(float(line["coord"])) for line in horizontals})
    out: list[dict[str, Any]] = []
    for xi in range(len(xs) - 1):
        x1, x2 = xs[xi], xs[xi + 1]
        if x2 - x1 < args.min_room_size or x2 - x1 > args.max_room_size:
            continue
        for yi in range(len(ys) - 1):
            y1, y2 = ys[yi], ys[yi + 1]
            if y2 - y1 < args.min_room_size or y2 - y1 > args.max_room_size:
                continue
            top = overlap_exists(horizontals, y1, x1, x2, "horizontal", args.room_edge_tolerance)
            bottom = overlap_exists(horizontals, y2, x1, x2, "horizontal", args.room_edge_tolerance)
            left = overlap_exists(verticals, x1, y1, y2, "vertical", args.room_edge_tolerance)
            right = overlap_exists(verticals, x2, y1, y2, "vertical", args.room_edge_tolerance)
            support = sum([top, bottom, left, right])
            if support < args.min_room_edge_support:
                continue
            bbox = [int(x1), int(y1), int(x2), int(y2)]
            area = (x2 - x1) * (y2 - y1)
            out.append(
                {
                    "candidate_id": f"{row_id}_pseudo_svg_norm_room_{len(out):04d}",
                    "row_id": row_id,
                    "family": "space",
                    "route": "room_space",
                    "candidate_type": "room_region_from_vector_grid",
                    "bbox": bbox,
                    "confidence": round(0.45 + 0.12 * support, 6),
                    "payload": {
                        "image": image_path,
                        "raster_path": image_path,
                        "proposal_source": "pseudo_svg_normalizer_v19",
                        "proposal_class": "room",
                        "proposal_semantic_type": "unknown_room",
                        "shape_features": {"area": float(area), "edge_support": support, "width": x2 - x1, "height": y2 - y1},
                        "source_integrity": source_integrity(),
                    },
                    "source_integrity": source_integrity(),
                }
            )
            if len(out) >= args.max_room_candidates:
                return out
    return out


def gold_boundary(path: Path) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for row in load_jsonl(path):
        out[str(row["id"])] = [
            item for item in (row.get("targets") or {}).get("boxes") or []
            if item.get("bbox") and len(item["bbox"]) == 4
        ]
    return out


def gold_rooms(path: Path) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for row in load_jsonl(path):
        out[str(row["id"])] = [
            item for item in (row.get("targets") or {}).get("rooms") or []
            if item.get("bbox") and len(item["bbox"]) == 4
        ]
    return out


def recall(rows: list[dict[str, Any]], gold: dict[str, list[dict[str, Any]]], family: str) -> dict[str, Any]:
    matched = 0
    total = 0
    predicted = 0
    for row in rows:
        preds = [item for item in row["scene_graph"]["candidate_stream"] if item["family"] == family]
        predicted += len(preds)
        used: set[int] = set()
        for gold_index, gold_item in enumerate(gold.get(str(row["id"]), [])):
            total += 1
            gb = [float(v) for v in gold_item["bbox"]]
            best = None
            best_iou = 0.0
            for pred_index, pred in enumerate(preds):
                if pred_index in used:
                    continue
                pb = [float(v) for v in pred["bbox"]]
                iou = bbox_iou(pb, gb)
                if center_covered(pb, gb) or iou >= 0.30:
                    if iou >= best_iou:
                        best = pred_index
                        best_iou = iou
            if best is not None:
                used.add(best)
                matched += 1
    return {
        "gold": total,
        "predicted": predicted,
        "matched": matched,
        "center_or_iou_recall": round(matched / max(total, 1), 6),
        "candidate_inflation": round(predicted / max(total, 1), 6),
    }


def process_row(row: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], dict[str, Any]]:
    row_id = str(row["id"])
    image_path = str(row["image"])
    lines = normalize_lines(row.get("paths") or [], args)
    boundary = [boundary_candidate(row_id, image_path, line, idx, args.boundary_pad) for idx, line in enumerate(lines)]
    rooms = room_candidates(row_id, image_path, lines, args)
    candidates = boundary[: args.max_boundary_candidates] + rooms
    out_row = {
        "id": row_id,
        "image": image_path,
        "image_size": row.get("image_size"),
        "pseudo_svg": row.get("pseudo_svg"),
        "source_integrity": source_integrity(),
        "route_trace": {"stage": "pseudo_svg_normalizer_v19", **source_integrity()},
        "scene_graph": {"nodes": [], "relations": [], "candidate_stream": candidates},
    }
    audit = {
        "id": row_id,
        "raw_path_count": len(row.get("paths") or []),
        "grouped_line_count": len(lines),
        "boundary_candidates": len(boundary),
        "room_candidates": len(rooms),
        "candidate_count": len(candidates),
    }
    return out_row, audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/pseudo_svg_vectorizer_v19_locked/pseudo_svg_paths.jsonl")
    parser.add_argument("--output", default="datasets/pseudo_svg_vectorizer_v19_locked/pseudo_svg_normalized_candidates.jsonl")
    parser.add_argument("--audit", default="reports/vlm/pseudo_svg_normalizer_v19_audit.json")
    parser.add_argument("--boundary-gold", default="datasets/image_only_boundary_detector_v18/locked.jsonl")
    parser.add_argument("--room-gold", default="datasets/image_only_room_polygon_v18/locked.jsonl")
    parser.add_argument("--max-axis-slope", type=float, default=0.08)
    parser.add_argument("--coord-bucket", type=float, default=3.0)
    parser.add_argument("--merge-gap", type=float, default=8.0)
    parser.add_argument("--min-line-length", type=float, default=8.0)
    parser.add_argument("--min-grouped-length", type=float, default=12.0)
    parser.add_argument("--boundary-pad", type=int, default=3)
    parser.add_argument("--max-boundary-candidates", type=int, default=1200)
    parser.add_argument("--min-room-edge", type=float, default=35.0)
    parser.add_argument("--min-room-size", type=float, default=28.0)
    parser.add_argument("--max-room-size", type=float, default=420.0)
    parser.add_argument("--room-edge-tolerance", type=float, default=8.0)
    parser.add_argument("--min-room-edge-support", type=int, default=3)
    parser.add_argument("--max-room-candidates", type=int, default=120)
    args = parser.parse_args()

    rows = load_jsonl(abs_path(args.input))
    out_rows = []
    row_audit = []
    totals = Counter()
    for row in rows:
        out_row, audit = process_row(row, args)
        out_rows.append(out_row)
        row_audit.append(audit)
        totals["raw_paths"] += audit["raw_path_count"]
        totals["grouped_lines"] += audit["grouped_line_count"]
        totals["boundary_candidates"] += audit["boundary_candidates"]
        totals["room_candidates"] += audit["room_candidates"]
    write_jsonl(abs_path(args.output), out_rows)
    boundary_recall = recall(out_rows, gold_boundary(abs_path(args.boundary_gold)), "boundary")
    room_recall = recall(out_rows, gold_rooms(abs_path(args.room_gold)), "space")
    report = {
        "version": "pseudo_svg_normalizer_v19",
        "task": "P0-PSEUDO-SVG-001",
        "input": args.input,
        "output": args.output,
        "source_integrity": source_integrity(),
        "config": vars(args),
        "summary": {
            "rows": len(rows),
            "raw_paths": int(totals["raw_paths"]),
            "grouped_lines": int(totals["grouped_lines"]),
            "boundary_candidates": int(totals["boundary_candidates"]),
            "room_candidates": int(totals["room_candidates"]),
            "boundary_recall": boundary_recall,
            "room_recall": room_recall,
        },
        "row_audit": row_audit,
    }
    write_json(abs_path(args.audit), report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
