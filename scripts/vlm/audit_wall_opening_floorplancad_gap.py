#!/usr/bin/env python3
"""Classify residual FloorPlanCAD wall/opening errors into repairable buckets."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hard-cases", default="reports/vlm/paper_v2_floor_target_h768doorw150_residual_fine_maxf1_hard_cases.jsonl")
    parser.add_argument("--summary", default="reports/vlm/paper_v2_floor_target_h768doorw150_residual_fine_maxf1_hard_cases_summary.json")
    parser.add_argument("--output", default="reports/vlm/wall_opening_floorplancad_gap_audit_v1.json")
    parser.add_argument("--cases-output", default="reports/vlm/wall_opening_floorplancad_gap_cases_v1.jsonl")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.hard_cases))
    summary = load_json(Path(args.summary))
    cases = []
    bucket_counts: Counter[str] = Counter()
    pair_counts: Counter[str] = Counter()
    fix_counts: Counter[str] = Counter()
    by_image: dict[str, Counter[str]] = defaultdict(Counter)

    for row in rows:
        if row.get("correct") is True:
            continue
        bucket = classify_case(row)
        fix = repair_for(bucket)
        pair = f"{row.get('label')}->{row.get('prediction')}"
        enriched = {
            **row,
            "error_pair": pair,
            "error_bucket": bucket,
            "repair_candidate": fix,
            "audit_version": "wall_opening_floorplancad_gap_audit_v1",
        }
        cases.append(enriched)
        bucket_counts[bucket] += 1
        pair_counts[pair] += 1
        fix_counts[fix] += 1
        by_image[str(row.get("image"))][bucket] += 1

    report = {
        "version": "wall_opening_floorplancad_gap_audit_v1",
        "input_hard_cases": args.hard_cases,
        "input_summary": args.summary,
        "source": "floorplancad",
        "baseline_summary": summary,
        "error_records": len(cases),
        "error_pairs": dict(pair_counts),
        "error_buckets": dict(bucket_counts),
        "repair_candidates": dict(fix_counts),
        "top_images": top_images(by_image),
        "fix_plan": {
            "thin_wall_false_door": "Add thin elongated wall hard negatives and a rule/residual feature for aspect plus relation_contains=0.",
            "isolated_door_as_wall": "Train a FloorPlanCAD residual branch on isolated rectangular door crops and keep source-specific calibration separate from the frozen main model.",
            "door_window_subtype_ambiguous": "Add opening subtype calibration with crop context and abstain when door/window posterior margin is small.",
            "raster_noise_or_symbol_overlap": "Route low dark-density or high edge-ratio crops to abstain/residual review; add symbol-overlap negatives.",
            "low_margin_opening": "Use confidence calibration and self-router instead of changing the base WallOpeningExpert.",
        },
        "done_when_check": "Each residual class has a data/calibration/residual/rule/abstain repair candidate.",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_jsonl(Path(args.cases_output), cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def classify_case(row: dict[str, Any]) -> str:
    gold = str(row.get("label"))
    pred = str(row.get("prediction"))
    bbox = row.get("bbox") if isinstance(row.get("bbox"), list) else [0, 0, 0, 0]
    width = abs(float(bbox[2]) - float(bbox[0])) if len(bbox) == 4 else 0.0
    height = abs(float(bbox[3]) - float(bbox[1])) if len(bbox) == 4 else 0.0
    aspect = max(width, height) / max(min(width, height), 1e-6)
    dark = float(row.get("raster_dark_density") or 0.0)
    edge = float(row.get("raster_edge_density") or 0.0)
    degree = float(row.get("graph_degree") or 0.0)

    if {gold, pred} == {"door", "window"}:
        return "door_window_subtype_ambiguous"
    if gold == "hard_wall" and pred == "door" and aspect >= 8:
        return "thin_wall_false_door"
    if gold == "door" and pred == "hard_wall" and degree <= 1:
        return "isolated_door_as_wall"
    if dark < 0.9 or edge > 0.28:
        return "raster_noise_or_symbol_overlap"
    return "low_margin_opening"


def repair_for(bucket: str) -> str:
    return {
        "thin_wall_false_door": "rule+hard_negative_data",
        "isolated_door_as_wall": "source_residual_branch",
        "door_window_subtype_ambiguous": "calibration+abstain",
        "raster_noise_or_symbol_overlap": "abstain+symbol_overlap_data",
        "low_margin_opening": "calibration+self_router",
    }[bucket]


def top_images(by_image: dict[str, Counter[str]]) -> list[dict[str, Any]]:
    items = []
    for image, counts in by_image.items():
        items.append({"image": image, "errors": sum(counts.values()), "buckets": dict(counts)})
    return sorted(items, key=lambda item: (-item["errors"], item["image"]))[:20]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
