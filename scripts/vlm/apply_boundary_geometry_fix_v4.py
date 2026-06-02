#!/usr/bin/env python3
"""Apply conservative boundary geometry rendering fixes for v4 visual demos."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


BOUNDARY_LABELS = {"door", "window", "opening", "hard_wall", "partition_wall"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/vlm/real_upstream_model_postprocessed_predictions_textfix_v4.jsonl")
    parser.add_argument("--ledger", default="reports/vlm/visual_residual_ledger_v4.json")
    parser.add_argument("--output", default="reports/vlm/real_upstream_model_postprocessed_predictions_boundaryfix_v4.jsonl")
    parser.add_argument("--report", default="reports/vlm/boundary_geometry_preservation_v4.json")
    parser.add_argument("--aspect-threshold", type=float, default=50.0)
    args = parser.parse_args()

    ledger = json.loads(Path(args.ledger).read_text(encoding="utf-8"))
    residual_ids = {
        (str(item.get("sample_id")), str(item.get("node_id")))
        for item in ledger.get("cases") or []
        if item.get("defect_type") == "unsupported_wall"
    }
    rows = []
    events: list[dict[str, Any]] = []
    for row in load_jsonl(Path(args.predictions)):
        row = json.loads(json.dumps(row, ensure_ascii=False))
        sid = sample_id(row)
        for node in ((row.get("scene_graph") or {}).get("nodes") or []):
            if not isinstance(node, dict) or str(node.get("family")) != "boundary":
                continue
            event = maybe_fix_boundary_node(sid, node, residual_ids, args.aspect_threshold)
            if event:
                events.append(event)
                row.setdefault("warnings", []).append(f"boundary_geometry_fix_v4:{event['node_id']}:{event['decision']}")
        row.setdefault("route_trace", {})["boundary_geometry_fix_v4"] = {
            "events": sum(1 for event in events if event["sample_id"] == sid),
            "claim_boundary": "Adds source/derived centerline geometry for line-like bbox-only boundary nodes; labels are not treated as retrained model improvements.",
        }
        rows.append(row)

    write_jsonl(Path(args.output), rows)
    report = {
        "version": "boundary_geometry_preservation_v4",
        "inputs": {"predictions": args.predictions, "ledger": args.ledger},
        "output": args.output,
        "summary": {
            "event_count": len(events),
            "decision_counts": dict(Counter(event["decision"] for event in events).most_common()),
            "residual_unsupported_wall_ids": len(residual_ids),
            "residual_ids_fixed": sum(1 for event in events if event["is_residual_case"]),
            "by_sample": by_sample(events),
        },
        "events": events,
    }
    write_json(Path(args.report), report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


def maybe_fix_boundary_node(
    sid: str,
    node: dict[str, Any],
    residual_ids: set[tuple[str, str]],
    aspect_threshold: float,
) -> dict[str, Any] | None:
    node_id = str(node.get("id") or "")
    semantic = str(node.get("semantic_type") or "")
    if semantic not in BOUNDARY_LABELS:
        return None
    geometry = node.get("geometry") if isinstance(node.get("geometry"), dict) else {}
    existing_type = str(geometry.get("type") or "")
    bbox = normalize_bbox(geometry.get("bbox") or node.get("bbox"))
    if bbox is None:
        return None
    width = max(0.0, bbox[2] - bbox[0])
    height = max(0.0, bbox[3] - bbox[1])
    aspect = max(width, height) / max(min(width, height), 1e-6)
    is_residual = (sid, node_id) in residual_ids
    if existing_type in {"line", "polyline", "polygon"} and not is_residual:
        return None
    if aspect < aspect_threshold and not is_residual:
        return None
    line = centerline_from_bbox(bbox)
    geometry["source_geometry"] = {"type": "line", "points": line}
    geometry["render_hint"] = "line_like_boundary_centerline"
    geometry["bbox"] = bbox
    node["geometry"] = geometry
    flags = node.setdefault("quality_flags", [])
    if isinstance(flags, list):
        flags.append("needs_review_boundary_geometry")
    metadata = node.setdefault("metadata", {})
    if isinstance(metadata, dict):
        metadata["boundary_geometry_fix_v4"] = {
            "source": "derived_from_line_like_bbox",
            "aspect_ratio": round(aspect, 6),
            "original_geometry_type": existing_type or "bbox",
        }
    return {
        "sample_id": sid,
        "node_id": node_id,
        "semantic_type": semantic,
        "raw_label": metadata.get("raw_label") if isinstance(metadata, dict) else None,
        "bbox": bbox,
        "aspect_ratio": round(aspect, 6) if math.isfinite(aspect) else "inf",
        "line": line,
        "decision": "derive_centerline_from_bbox",
        "is_residual_case": is_residual,
    }


def centerline_from_bbox(bbox: list[float]) -> list[list[float]]:
    x1, y1, x2, y2 = bbox
    if (x2 - x1) >= (y2 - y1):
        y = round((y1 + y2) / 2.0, 3)
        return [[round(x1, 3), y], [round(x2, 3), y]]
    x = round((x1 + x2) / 2.0, 3)
    return [[x, round(y1, 3)], [x, round(y2, 3)]]


def by_sample(events: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    grouped: dict[str, Counter[str]] = defaultdict(Counter)
    for event in events:
        grouped[str(event["sample_id"])][str(event["decision"])] += 1
    return {sample: dict(counts.most_common()) for sample, counts in sorted(grouped.items())}


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, (list, tuple)) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(value[index]) for index in range(4)]
    except (TypeError, ValueError):
        return None
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def sample_id(row: dict[str, Any]) -> str:
    for key in ["sample_id", "image", "image_path"]:
        value = str(row.get(key) or "")
        if value:
            parts = Path(value).parts
            if len(parts) >= 2 and parts[-1].lower().endswith(".png"):
                return parts[-2]
            return Path(value).stem
    return ""


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
