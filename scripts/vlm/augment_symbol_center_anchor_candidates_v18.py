#!/usr/bin/env python3
"""Add strict center-anchor symbol candidates for drifted symbol boxes.

This inference-only pass creates small raster-derived symbol candidates near
room boundaries when an existing symbol candidate overlaps a room but its center
falls just outside that room. It targets center-drift failures without expanding
room geometry or blindly splitting every large symbol anchor.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from build_topology_relations_v18 import area, bbox, center, confidence, contains_point, expanded_box, integrity, iou, write_json, write_jsonl
from nms_topology_relations_v18 import load_jsonl

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"

DEFAULT_INPUT = REPORT / "detector_adapter_v18_symbol_boundary_fixed_candidates.jsonl"
DEFAULT_OUTPUT = REPORT / "detector_adapter_v18_symbol_center_anchor.jsonl"
DEFAULT_AUDIT = REPORT / "detector_adapter_v18_symbol_center_anchor_audit.json"


def candidate_stream(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(((row.get("scene_graph") or {}).get("candidate_stream") or []))


def box_size(box: list[float]) -> tuple[float, float]:
    return max(0.0, box[2] - box[0]), max(0.0, box[3] - box[1])


def nearest_inside_point(room_box: list[float], symbol_box: list[float], margin: float) -> tuple[float, float]:
    sx, sy = center(symbol_box)
    return (
        min(max(sx, room_box[0] + margin), room_box[2] - margin),
        min(max(sy, room_box[1] + margin), room_box[3] - margin),
    )


def clip_box(box: list[float], image_size: list[float]) -> list[float]:
    width, height = float(image_size[0]), float(image_size[1])
    return [
        max(0.0, min(width, box[0])),
        max(0.0, min(height, box[1])),
        max(0.0, min(width, box[2])),
        max(0.0, min(height, box[3])),
    ]


def center_distance_to_room(room_box: list[float], symbol_box: list[float]) -> float:
    sx, sy = center(symbol_box)
    dx = max(room_box[0] - sx, 0.0, sx - room_box[2])
    dy = max(room_box[1] - sy, 0.0, sy - room_box[3])
    return (dx * dx + dy * dy) ** 0.5


def make_anchor(
    row_id: str,
    room: dict[str, Any],
    symbol: dict[str, Any],
    room_box: list[float],
    symbol_box: list[float],
    image_size: list[float],
    index: int,
    *,
    inside_margin: float,
    min_side: float,
    max_side: float,
    scale: float,
) -> dict[str, Any]:
    width, height = box_size(symbol_box)
    side = min(max(min(width, height) * scale, min_side), max_side)
    ax, ay = nearest_inside_point(room_box, symbol_box, inside_margin)
    child_box = clip_box([ax - side / 2.0, ay - side / 2.0, ax + side / 2.0, ay + side / 2.0], image_size)
    parent_payload = symbol.get("payload") if isinstance(symbol.get("payload"), dict) else {}
    parent_id = str(symbol.get("candidate_id"))
    room_id = str(room.get("candidate_id"))
    payload = {
        **parent_payload,
        "candidate_kind": "symbol_center_anchor",
        "source": "symbol_center_anchor_v18",
        "parent_candidate_id": parent_id,
        "anchor_room_candidate_id": room_id,
        "type_label_adopted": False,
        "objectness_score": round(min(0.99, confidence(symbol) * 0.86), 6),
        "symbol_center_anchor_v18": {
            "version": "symbol_center_anchor_v18",
            "strategy": "project_symbol_center_inside_overlapping_room",
            "parent_bbox": symbol_box,
            "anchor_bbox": child_box,
            "anchor_center": [round(ax, 3), round(ay, 3)],
            "room_bbox": room_box,
            "parent_center_distance_to_room": round(center_distance_to_room(room_box, symbol_box), 6),
            "anchor_area_ratio_to_parent": round(area(child_box) / max(area(symbol_box), 1e-9), 6),
        },
    }
    return {
        "candidate_id": f"{parent_id}_center_anchor_{index:02d}_{int(round(child_box[0]))}_{int(round(child_box[1]))}_{int(round(child_box[2]))}_{int(round(child_box[3]))}",
        "candidate_contract_version": symbol.get("candidate_contract_version") or "detector_candidate_contract_v1",
        "row_id": row_id,
        "family": "symbol",
        "route": symbol.get("route") or "symbol_fixture",
        "candidate_type": symbol.get("candidate_type") or "symbol",
        "bbox": [round(value, 3) for value in child_box],
        "confidence": round(min(0.99, confidence(symbol) * 0.84 + confidence(room) * 0.04), 6),
        "payload": payload,
        "source_integrity": integrity(),
        "provenance": {
            "input_source": "symbol_center_anchor_v18",
            "raw_candidate_id": f"{parent_id}_center_anchor_{index:02d}",
            "parent_candidate_id": parent_id,
            "anchor_room_candidate_id": room_id,
            "row_id": row_id,
            "family": "symbol",
            "route": symbol.get("route") or "symbol_fixture",
            "raster_only": True,
        },
        "audit_trace": {
            "stage": "symbol_center_anchor_v18",
            "parent_candidate_id": parent_id,
            "anchor_room_candidate_id": room_id,
            "parent_bbox": symbol.get("bbox"),
            "anchor_bbox": [round(value, 3) for value in child_box],
            "source_integrity": integrity(),
        },
    }


def eligible_symbol(symbol: dict[str, Any], min_parent_area: float, max_parent_area: float) -> bool:
    box = bbox(symbol.get("bbox"))
    if box is None or symbol.get("family") != "symbol":
        return False
    symbol_area = area(box)
    if symbol_area < min_parent_area or symbol_area > max_parent_area:
        return False
    payload = symbol.get("payload") if isinstance(symbol.get("payload"), dict) else {}
    return str(payload.get("candidate_kind") or "") in {"dark_pixel_anchor", "dark_connected_component"}


def augment_row(row: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], Counter[str], list[dict[str, Any]]]:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    stream = candidate_stream(out)
    image_size = out.get("image_size") or [512, 512]
    rooms = [cand for cand in stream if cand.get("family") == "space" and bbox(cand.get("bbox")) is not None]
    symbols = [cand for cand in stream if eligible_symbol(cand, args.min_parent_area, args.max_parent_area)]
    counts: Counter[str] = Counter({"rows": 1, "input_rooms": len(rooms), "eligible_symbols": len(symbols)})
    added: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []

    for symbol in symbols:
        symbol_box = bbox(symbol.get("bbox"))
        if symbol_box is None:
            continue
        sx, sy = center(symbol_box)
        local: list[tuple[float, dict[str, Any], dict[str, Any]]] = []
        for room in rooms:
            room_box = bbox(room.get("bbox"))
            if room_box is None:
                continue
            if contains_point(room_box, sx, sy, margin=2.0):
                continue
            if iou(room_box, symbol_box) <= 0.0:
                continue
            if not contains_point(expanded_box(room_box, args.max_center_drift), sx, sy, margin=0.0):
                continue
            ax, ay = nearest_inside_point(room_box, symbol_box, args.inside_margin)
            if not (symbol_box[0] - args.anchor_slack <= ax <= symbol_box[2] + args.anchor_slack and symbol_box[1] - args.anchor_slack <= ay <= symbol_box[3] + args.anchor_slack):
                counts["rejected_anchor_outside_parent"] += 1
                continue
            drift = center_distance_to_room(room_box, symbol_box)
            score = confidence(symbol) * 0.55 + confidence(room) * 0.20 + max(0.0, 1.0 - drift / max(args.max_center_drift, 1e-9)) * 0.25
            child = make_anchor(
                str(out.get("id")),
                room,
                symbol,
                room_box,
                symbol_box,
                image_size,
                len(added) + len(local),
                inside_margin=args.inside_margin,
                min_side=args.min_side,
                max_side=args.max_side,
                scale=args.scale,
            )
            if any(iou(child["bbox"], existing.get("bbox") or []) > args.duplicate_iou for existing in stream + added):
                counts["rejected_duplicate"] += 1
                continue
            local.append((score, child, room))
        local.sort(key=lambda item: item[0], reverse=True)
        for _, child, room in local[: args.max_per_parent]:
            added.append(child)
            if len(examples) < 50:
                examples.append(
                    {
                        "row_id": out.get("id"),
                        "parent_candidate_id": symbol.get("candidate_id"),
                        "anchor_room_candidate_id": room.get("candidate_id"),
                        "child_candidate_id": child.get("candidate_id"),
                        "parent_bbox": symbol.get("bbox"),
                        "child_bbox": child.get("bbox"),
                    }
                )
            if len(added) >= args.max_added_per_row:
                break
        if len(added) >= args.max_added_per_row:
            break

    stream.extend(added)
    graph = out.setdefault("scene_graph", {})
    graph["candidate_stream"] = stream
    graph["candidate_counts"] = dict(Counter(str(cand.get("family") or "unknown") for cand in stream))
    graph["symbol_center_anchor_v18"] = {
        "enabled": True,
        "added_symbol_center_anchors": len(added),
        "max_center_drift": args.max_center_drift,
        "max_per_parent": args.max_per_parent,
        "max_added_per_row": args.max_added_per_row,
    }
    counts["added_symbol_center_anchors"] = len(added)
    counts["candidate_stream_out"] = len(stream)
    return out, counts, examples


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--max-center-drift", type=float, default=18.0)
    parser.add_argument("--inside-margin", type=float, default=4.0)
    parser.add_argument("--anchor-slack", type=float, default=2.0)
    parser.add_argument("--min-side", type=float, default=10.0)
    parser.add_argument("--max-side", type=float, default=28.0)
    parser.add_argument("--scale", type=float, default=0.72)
    parser.add_argument("--min-parent-area", type=float, default=300.0)
    parser.add_argument("--max-parent-area", type=float, default=2400.0)
    parser.add_argument("--max-per-parent", type=int, default=1)
    parser.add_argument("--max-added-per-row", type=int, default=96)
    parser.add_argument("--duplicate-iou", type=float, default=0.78)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    out_rows: list[dict[str, Any]] = []
    total: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []
    per_row_added: Counter[int] = Counter()
    for row in rows:
        out, counts, row_examples = augment_row(row, args)
        out_rows.append(out)
        total.update(counts)
        per_row_added[int(counts["added_symbol_center_anchors"])] += 1
        examples.extend(row_examples[: max(0, 50 - len(examples))])

    audit = {
        "task": "IMG-MOE-V18-REBUILD-001.step_i_symbol_center_anchor_repair",
        "input": args.input,
        "output": args.output,
        "params": {
            "max_center_drift": args.max_center_drift,
            "inside_margin": args.inside_margin,
            "anchor_slack": args.anchor_slack,
            "min_side": args.min_side,
            "max_side": args.max_side,
            "scale": args.scale,
            "min_parent_area": args.min_parent_area,
            "max_parent_area": args.max_parent_area,
            "max_per_parent": args.max_per_parent,
            "max_added_per_row": args.max_added_per_row,
            "duplicate_iou": args.duplicate_iou,
        },
        "counts": dict(total),
        "added_per_row_histogram": {str(key): value for key, value in sorted(per_row_added.items())},
        "examples": examples,
        "source_integrity": integrity(),
        "gold_loaded": False,
        "gold_used_for_inference": False,
    }
    write_jsonl(Path(args.output), out_rows)
    write_json(Path(args.audit), audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
