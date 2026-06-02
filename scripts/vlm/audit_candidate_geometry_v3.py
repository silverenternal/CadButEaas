#!/usr/bin/env python3
"""Audit candidate geometry clipping and canvas consistency for visual predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/vlm/real_upstream_model_postprocessed_predictions_v3.jsonl")
    parser.add_argument("--quality-gate-report", default="reports/vlm/node_quality_gate_sweep_v3.json")
    parser.add_argument("--defect-summary", default="reports/vlm/visual_demo/model_defect_summary_postprocessed_v3.json")
    parser.add_argument("--output", default="reports/vlm/candidate_geometry_audit_v3.json")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.predictions))
    gate_report = load_json(Path(args.quality_gate_report))
    defect_summary = load_json(Path(args.defect_summary))
    report = build_report(args, rows, gate_report, defect_summary)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


def build_report(
    args: argparse.Namespace,
    rows: list[dict[str, Any]],
    gate_report: dict[str, Any],
    defect_summary: dict[str, Any],
) -> dict[str, Any]:
    family_counts: Counter[str] = Counter()
    clipped_counts: Counter[str] = Counter()
    outside_after_clip: Counter[str] = Counter()
    canvas_counts: Counter[str] = Counter()
    examples = []
    for row in rows:
        sample = sample_id(row)
        for node in (row.get("scene_graph") or {}).get("nodes") or []:
            if not isinstance(node, dict):
                continue
            family = str(node.get("family") or "unknown")
            family_counts[family] += 1
            metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
            canvas = normalize_bbox(metadata.get("source_canvas_bbox"))
            bbox = normalize_bbox((node.get("geometry") or {}).get("bbox") or node.get("bbox"))
            if canvas is not None:
                canvas_counts[family] += 1
            if bool(metadata.get("was_clipped_to_canvas")):
                clipped_counts[family] += 1
                if len(examples) < 30:
                    examples.append(
                        {
                            "sample_id": sample,
                            "node_id": node.get("id"),
                            "family": family,
                            "semantic_type": node.get("semantic_type"),
                            "unclipped_bbox": metadata.get("unclipped_bbox"),
                            "clipped_bbox": bbox,
                            "source_canvas_bbox": canvas,
                            "source_expert": node.get("source_expert"),
                        }
                    )
            if bbox is not None and canvas is not None and bbox_outside_canvas(bbox, canvas):
                outside_after_clip[family] += 1
    gate_summary = gate_report.get("summary") if isinstance(gate_report.get("summary"), dict) else {}
    event_clipped = Counter(
        str(event.get("family") or "unknown")
        for event in gate_report.get("events") or []
        if isinstance(event, dict) and event.get("was_clipped_to_canvas")
    )
    defect_counts = defect_summary.get("defect_counts") or {}
    if "summary" in defect_summary and isinstance(defect_summary["summary"], dict):
        defect_counts = defect_summary["summary"].get("defect_counts") or defect_counts
    return {
        "version": "cadstruct_candidate_geometry_audit_v3",
        "inputs": {
            "predictions": args.predictions,
            "quality_gate_report": args.quality_gate_report,
            "defect_summary": args.defect_summary,
        },
        "summary": {
            "records": len(rows),
            "node_counts_by_family": dict(family_counts.most_common()),
            "nodes_with_source_canvas_bbox_by_family": dict(canvas_counts.most_common()),
            "clipped_nodes_by_family": dict(clipped_counts.most_common()),
            "quality_gate_clipped_events_by_family": dict(event_clipped.most_common()),
            "outside_after_clip_by_family": dict(outside_after_clip.most_common()),
            "visual_defect_bbox_outside_canvas": int(defect_counts.get("bbox_outside_canvas", 0)),
            "quality_gate_bbox_outside_events": int((gate_summary.get("reason_counts") or {}).get("bbox_outside_canvas_drop", 0))
            + int((gate_summary.get("reason_counts") or {}).get("bbox_outside_canvas_review", 0)),
            "done_when": {
                "bbox_outside_canvas_le_5": int(defect_counts.get("bbox_outside_canvas", 0)) <= 5,
                "has_clipping_metadata": (sum(clipped_counts.values()) + sum(event_clipped.values())) > 0 and sum(canvas_counts.values()) > 0,
                "all_remaining_nodes_inside_canvas": sum(outside_after_clip.values()) == 0,
            },
        },
        "clipped_examples": examples,
    }


def sample_id(row: dict[str, Any]) -> str:
    image = Path(str(row.get("image") or row.get("image_path") or "sample"))
    return image.parent.name or image.stem


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def bbox_outside_canvas(bbox: list[float], canvas: list[float]) -> bool:
    return bbox[0] < canvas[0] or bbox[1] < canvas[1] or bbox[2] > canvas[2] or bbox[3] > canvas[3]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
