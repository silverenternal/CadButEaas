#!/usr/bin/env python3
"""Audit and apply v4 text-candidate visibility handling."""

from __future__ import annotations

import argparse
import json
import xml.etree.ElementTree as ET
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/vlm/real_upstream_model_postprocessed_predictions_roomlink_v3.jsonl")
    parser.add_argument("--converted", default="datasets/cadstruct_real_world_benchmark_v1/room_space/cubicasa5k_reviewed_locked_test.jsonl")
    parser.add_argument("--ledger", default="reports/vlm/visual_residual_ledger_v4.json")
    parser.add_argument("--output", default="reports/vlm/real_upstream_model_postprocessed_predictions_textfix_v4.jsonl")
    parser.add_argument("--report", default="reports/vlm/text_candidate_recovery_v7.json")
    parser.add_argument("--manifest", default="datasets/text_dimension_expert_v7_hard_cases/manifest.jsonl")
    args = parser.parse_args()

    predictions = load_jsonl(Path(args.predictions))
    converted_rows = load_jsonl(Path(args.converted))
    converted_by_sample = {sample_id(row): row for row in converted_rows}
    ledger = json.loads(Path(args.ledger).read_text(encoding="utf-8"))
    text_cases = [case for case in ledger.get("cases") or [] if case.get("defect_type") == "missing_visible_text"]
    text_cases_by_sample: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in text_cases:
        text_cases_by_sample[str(case.get("sample_id"))].append(case)

    recovery_cases: list[dict[str, Any]] = []
    for case in text_cases:
        sid = str(case.get("sample_id") or "")
        converted = converted_by_sample.get(sid, {})
        candidate = find_text_candidate(converted, str(case.get("node_id") or ""))
        canvas = svg_viewbox_canvas_bbox(converted) or metadata_canvas_bbox(converted)
        bbox = normalize_bbox(candidate.get("bbox") or case.get("bbox"))
        text = str(candidate.get("text") or "").strip()
        outside_ratio = bbox_outside_area_ratio(bbox, canvas) if canvas and bbox else 0.0
        outside = outside_ratio >= 0.15
        if text and not outside:
            decision = "recoverable_text_missing_from_scene_graph"
        elif text and outside:
            decision = "suppress_outside_canvas_text"
        elif not text:
            decision = "suppress_empty_text_candidate"
        else:
            decision = "manual_review"
        recovery_cases.append(
            {
                "sample_id": sid,
                "node_id": case.get("node_id"),
                "text_type": candidate.get("text_type") or case.get("semantic_type"),
                "bbox": bbox,
                "text": text,
                "canvas": canvas,
                "outside_canvas": outside,
                "outside_canvas_ratio": round(outside_ratio, 6),
                "decision": decision,
                "fix_scope": "audit_contract" if decision.startswith("suppress_") else "export_or_quality_gate",
            }
        )

    output_rows = []
    suppressed_by_sample: dict[str, Counter[str]] = defaultdict(Counter)
    for row in predictions:
        sid = sample_id(row)
        row = json.loads(json.dumps(row, ensure_ascii=False))
        for case in text_cases_by_sample.get(sid, []):
            recovery = next((item for item in recovery_cases if item["sample_id"] == sid and item["node_id"] == case.get("node_id")), None)
            if not recovery:
                continue
            if recovery["decision"].startswith("suppress_"):
                suppressed_by_sample[sid][recovery["decision"]] += 1
                row.setdefault("warnings", []).append(f"text_candidate_fix_v4:{case.get('node_id')}:{recovery['decision']}")
        trace = row.setdefault("route_trace", {})
        trace["text_candidate_fix_v4"] = {
            "suppressed_candidates": sum(suppressed_by_sample[sid].values()),
            "claim_boundary": "Suppresses empty or outside-canvas converted text candidates from visual missing-text claims; does not insert oracle text.",
        }
        output_rows.append(row)

    write_jsonl(Path(args.output), output_rows)
    write_jsonl(Path(args.manifest), recovery_cases)
    report = {
        "version": "text_candidate_recovery_v7",
        "inputs": {"predictions": args.predictions, "converted": args.converted, "ledger": args.ledger},
        "output": args.output,
        "manifest": args.manifest,
        "summary": {
            "case_count": len(recovery_cases),
            "decision_counts": dict(Counter(item["decision"] for item in recovery_cases).most_common()),
            "recoverable_text_nodes": sum(1 for item in recovery_cases if item["decision"] == "recoverable_text_missing_from_scene_graph"),
            "suppressed_non_visible_or_empty": sum(1 for item in recovery_cases if str(item["decision"]).startswith("suppress_")),
            "by_sample": {sample: dict(counts.most_common()) for sample, counts in sorted(suppressed_by_sample.items())},
        },
        "cases": recovery_cases,
    }
    write_json(Path(args.report), report)
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


def find_text_candidate(row: dict[str, Any], node_id: str) -> dict[str, Any]:
    expected = row.get("expected_json") if isinstance(row.get("expected_json"), dict) else {}
    for item in expected.get("text_candidates") or []:
        if isinstance(item, dict) and str(item.get("id")) == node_id:
            return item
    return {}


def svg_viewbox_canvas_bbox(row: dict[str, Any]) -> list[float] | None:
    annotation = row.get("annotation_path") or row.get("annotation")
    if not annotation:
        return None
    path = Path(str(annotation))
    if not path.exists():
        return None
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError:
        return None
    view_box = root.attrib.get("viewBox")
    if view_box:
        values = [safe_float(item, 0.0) for item in view_box.replace(",", " ").split()]
        if len(values) == 4 and values[2] > 0 and values[3] > 0:
            return [values[0], values[1], values[0] + values[2], values[1] + values[3]]
    return None


def metadata_canvas_bbox(row: dict[str, Any]) -> list[float] | None:
    metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
    try:
        return [0.0, 0.0, float(metadata.get("width")), float(metadata.get("height"))]
    except (TypeError, ValueError):
        return None


def bbox_outside_area_ratio(bbox: list[float], canvas: list[float]) -> float:
    intersection = [max(bbox[0], canvas[0]), max(bbox[1], canvas[1]), min(bbox[2], canvas[2]), min(bbox[3], canvas[3])]
    outside = bbox_area(bbox) - bbox_area(intersection)
    return max(0.0, outside) / max(bbox_area(bbox), 1.0)


def bbox_area(bbox: list[float] | None) -> float:
    if bbox is None:
        return 0.0
    return max(0.0, bbox[2] - bbox[0]) * max(0.0, bbox[3] - bbox[1])


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


def safe_float(value: Any, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


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
