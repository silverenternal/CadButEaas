#!/usr/bin/env python3
"""Add auditable room coverage recovery candidates from raster detector output.

The augmentation is inference-only: it reads existing raster-derived room and
symbol candidates, then creates separate room candidates that minimally expand a
room bbox toward nearby symbol centers. Offline labels are not loaded here.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

from build_topology_relations_v18 import area, bbox, center, confidence, contains_point, expanded_box, integrity, write_json, write_jsonl
from nms_topology_relations_v18 import load_jsonl

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"

DEFAULT_INPUT = REPORT / "detector_adapter_v18_symbol_boundary_fixed_candidates.jsonl"
DEFAULT_OUTPUT = REPORT / "detector_adapter_v18_room_coverage_augmented.jsonl"
DEFAULT_AUDIT = REPORT / "detector_adapter_v18_room_coverage_augmented_audit.json"


def candidate_stream(row: dict[str, Any]) -> list[dict[str, Any]]:
    return list(((row.get("scene_graph") or {}).get("candidate_stream") or []))


def clipped_box(box: list[float], width: float, height: float) -> list[float]:
    return [
        max(0.0, min(width, box[0])),
        max(0.0, min(height, box[1])),
        max(0.0, min(width, box[2])),
        max(0.0, min(height, box[3])),
    ]


def outside_distance(room_box: list[float], x: float, y: float) -> float:
    dx = max(room_box[0] - x, 0.0, x - room_box[2])
    dy = max(room_box[1] - y, 0.0, y - room_box[3])
    return math.hypot(dx, dy)


def expansion_side(room_box: list[float], x: float, y: float) -> str:
    distances = {
        "left": room_box[0] - x,
        "right": x - room_box[2],
        "top": room_box[1] - y,
        "bottom": y - room_box[3],
    }
    outside = {key: value for key, value in distances.items() if value > 0.0}
    if not outside:
        return "inside"
    return max(outside, key=outside.get)


def make_recovered_room(
    row_id: str,
    room: dict[str, Any],
    symbol: dict[str, Any],
    room_box: list[float],
    symbol_box: list[float],
    image_size: list[float],
    index: int,
    pad: float,
) -> dict[str, Any]:
    sx, sy = center(symbol_box)
    expanded = [
        min(room_box[0], sx - pad),
        min(room_box[1], sy - pad),
        max(room_box[2], sx + pad),
        max(room_box[3], sy + pad),
    ]
    width, height = float(image_size[0]), float(image_size[1])
    expanded = clipped_box(expanded, width, height)
    room_payload = room.get("payload") if isinstance(room.get("payload"), dict) else {}
    payload = {
        **room_payload,
        "proposal_source": "room_coverage_recovery_v18",
        "parent_candidate_id": room.get("candidate_id"),
        "anchor_symbol_candidate_id": symbol.get("candidate_id"),
        "anchor_symbol_bbox": symbol_box,
        "source_integrity": integrity(),
        "room_coverage_recovery_v18": {
            "version": "room_coverage_recovery_v18",
            "strategy": "minimal_expand_to_nearby_symbol_center",
            "expansion_side": expansion_side(room_box, sx, sy),
            "parent_bbox": room_box,
            "recovered_bbox": expanded,
            "anchor_symbol_center": [round(sx, 3), round(sy, 3)],
            "bbox_area_ratio_to_parent": round(area(expanded) / max(area(room_box), 1e-9), 6),
        },
    }
    parent_id = str(room.get("candidate_id"))
    symbol_id = str(symbol.get("candidate_id"))
    return {
        "candidate_id": f"{row_id}_room_coverage_v18_{index:04d}_{parent_id[-24:]}_{symbol_id[-24:]}",
        "row_id": row_id,
        "family": "space",
        "route": room.get("route") or "room_space",
        "candidate_type": room.get("candidate_type") or "room",
        "bbox": [round(value, 3) for value in expanded],
        "confidence": round(max(0.05, confidence(room) * 0.82 + confidence(symbol) * 0.08), 6),
        "payload": payload,
        "provenance": {
            "source": "room_coverage_recovery_v18",
            "parent_candidate_id": parent_id,
            "anchor_symbol_candidate_id": symbol_id,
            "raster_only": True,
        },
        "audit_trace": {
            "room_coverage_recovery_v18": payload["room_coverage_recovery_v18"],
        },
        "source_integrity": integrity(),
    }


def augment_row(
    row: dict[str, Any],
    *,
    margin: float,
    pad: float,
    max_area_ratio: float,
    max_per_room: int,
    max_per_page: int,
) -> tuple[dict[str, Any], Counter[str]]:
    out = json.loads(json.dumps(row, ensure_ascii=False))
    stream = candidate_stream(out)
    image_size = out.get("image_size") or [512, 512]
    rooms = [cand for cand in stream if str(cand.get("family")) == "space" and bbox(cand.get("bbox")) is not None]
    symbols = [cand for cand in stream if str(cand.get("family")) == "symbol" and bbox(cand.get("bbox")) is not None]
    row_id = str(out.get("id"))
    counts: Counter[str] = Counter({"rows": 1, "input_rooms": len(rooms), "input_symbols": len(symbols)})
    proposals: list[tuple[float, dict[str, Any]]] = []

    for room in rooms:
        room_box = bbox(room.get("bbox"))
        if room_box is None:
            continue
        local: list[tuple[float, dict[str, Any]]] = []
        search = expanded_box(room_box, margin)
        for symbol in symbols:
            symbol_box = bbox(symbol.get("bbox"))
            if symbol_box is None:
                continue
            sx, sy = center(symbol_box)
            if contains_point(room_box, sx, sy, margin=2.0):
                continue
            if not contains_point(search, sx, sy, margin=0.0):
                continue
            expanded_area = area(
                [
                    min(room_box[0], sx - pad),
                    min(room_box[1], sy - pad),
                    max(room_box[2], sx + pad),
                    max(room_box[3], sy + pad),
                ]
            )
            area_ratio = expanded_area / max(area(room_box), 1e-9)
            if area_ratio > max_area_ratio:
                counts["rejected_area_ratio"] += 1
                continue
            dist = outside_distance(room_box, sx, sy)
            score = confidence(room) * 0.55 + confidence(symbol) * 0.25 + max(0.0, 1.0 - dist / max(margin, 1e-9)) * 0.20
            candidate = make_recovered_room(row_id, room, symbol, room_box, symbol_box, image_size, len(proposals) + len(local), pad)
            local.append((score, candidate))
        local.sort(key=lambda item: item[0], reverse=True)
        proposals.extend(local[:max_per_room])
        counts["candidate_room_anchors_with_local_proposals"] += 1 if local else 0

    proposals.sort(key=lambda item: item[0], reverse=True)
    selected = [candidate for _, candidate in proposals[:max_per_page]]
    stream.extend(selected)
    graph = out.setdefault("scene_graph", {})
    graph["candidate_stream"] = stream
    graph["candidate_counts"] = dict(Counter(str(cand.get("family") or "unknown") for cand in stream))
    route_trace = out.setdefault("route_trace", {})
    route_trace["room_coverage_recovery_v18"] = {
        "enabled": True,
        "margin": margin,
        "pad": pad,
        "max_area_ratio": max_area_ratio,
        "max_per_room": max_per_room,
        "max_per_page": max_per_page,
        "raster_only": True,
    }
    counts["added_room_coverage_candidates"] = len(selected)
    counts["candidate_stream_out"] = len(stream)
    return out, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--margin", type=float, default=18.0)
    parser.add_argument("--pad", type=float, default=4.0)
    parser.add_argument("--max-area-ratio", type=float, default=1.35)
    parser.add_argument("--max-per-room", type=int, default=2)
    parser.add_argument("--max-per-page", type=int, default=64)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    out_rows: list[dict[str, Any]] = []
    total: Counter[str] = Counter()
    row_examples: list[dict[str, Any]] = []
    for row in rows:
        out, counts = augment_row(
            row,
            margin=args.margin,
            pad=args.pad,
            max_area_ratio=args.max_area_ratio,
            max_per_room=args.max_per_room,
            max_per_page=args.max_per_page,
        )
        out_rows.append(out)
        total.update(counts)
        if counts["added_room_coverage_candidates"] and len(row_examples) < 50:
            row_examples.append({"row_id": out.get("id"), "added": counts["added_room_coverage_candidates"]})

    audit = {
        "task": "IMG-MOE-V18-REBUILD-001.step_h_repair_room_proposal_coverage",
        "input": args.input,
        "output": args.output,
        "params": {
            "margin": args.margin,
            "pad": args.pad,
            "max_area_ratio": args.max_area_ratio,
            "max_per_room": args.max_per_room,
            "max_per_page": args.max_per_page,
        },
        "counts": dict(total),
        "row_examples": row_examples,
        "source_integrity": integrity(),
        "gold_loaded": False,
    }
    write_jsonl(Path(args.output), out_rows)
    write_json(Path(args.audit), audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
