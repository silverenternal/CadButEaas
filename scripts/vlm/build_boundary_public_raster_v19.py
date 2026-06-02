#!/usr/bin/env python3
"""Build boundary-only supervision from public raster MoE records.

This is a training view, not a new source of labels. It keeps only raster image
paths and wall/opening/window boxes from public_raster_moe_supervision_v19.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageStat


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets/public_raster_moe_supervision_v19"
OUT = ROOT / "datasets/boundary_expert_public_raster_v19"
REPORT = ROOT / "reports/vlm"
SPLITS = ("train", "dev", "locked", "smoke")
LABELS = ("wall", "opening", "window")
LABEL_TO_ID = {"wall": 1, "opening": 2, "window": 3}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
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


def norm_bbox(value: Any, width: int, height: int) -> list[int] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    left = max(0, min(width - 1, int(math.floor(min(x1, x2)))))
    top = max(0, min(height - 1, int(math.floor(min(y1, y2)))))
    right = max(0, min(width - 1, int(math.ceil(max(x1, x2)))))
    bottom = max(0, min(height - 1, int(math.ceil(max(y1, y2)))))
    if right <= left or bottom <= top:
        return None
    return [left, top, right, bottom]


def image_stats(path: str) -> dict[str, Any]:
    try:
        with Image.open(ROOT / path) as img:
            gray = img.convert("L")
            stat = ImageStat.Stat(gray)
            return {
                "mean": round(float(stat.mean[0]), 4),
                "stddev": round(float(stat.stddev[0]), 4),
            }
    except Exception as exc:
        return {"mean": None, "stddev": None, "error": f"{type(exc).__name__}: {exc}"}


def centerline_from_bbox(box: list[int]) -> list[list[int]]:
    x1, y1, x2, y2 = box
    if (x2 - x1) >= (y2 - y1):
        cy = int(round((y1 + y2) * 0.5))
        return [[x1, cy], [x2, cy]]
    cx = int(round((x1 + x2) * 0.5))
    return [[cx, y1], [cx, y2]]


def target_from_public(item: dict[str, Any], width: int, height: int) -> dict[str, Any] | None:
    label = str(item.get("semantic_type") or "").strip()
    if label not in LABELS:
        return None
    box = norm_bbox(item.get("bbox"), width, height)
    if box is None:
        return None
    line = centerline_from_bbox(box)
    length = math.hypot(line[1][0] - line[0][0], line[1][1] - line[0][1])
    return {
        "target_id": str(item.get("target_id") or ""),
        "label": label,
        "label_id": LABEL_TO_ID[label],
        "bbox": box,
        "centerline": line,
        "length": round(length, 4),
        "thin_extent": min(max(1, box[2] - box[0] + 1), max(1, box[3] - box[1] + 1)),
        "label_source": "offline_public_annotation_converted_to_raster_target",
        "raw_labels": [str(item.get("raw_label") or label)],
    }


def dedupe_targets(targets: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    merged: dict[tuple[str, tuple[int, int, int, int]], dict[str, Any]] = {}
    duplicates = 0
    for item in targets:
        key = (str(item["label"]), tuple(int(v) for v in item["bbox"]))
        if key not in merged:
            merged[key] = dict(item)
            continue
        duplicates += 1
        existing = merged[key]
        raw = set(existing.get("raw_labels") or [])
        raw.update(item.get("raw_labels") or [])
        existing["raw_labels"] = sorted(raw)
        ids = existing.setdefault("merged_target_ids", [existing.get("target_id")])
        ids.append(item.get("target_id"))
    return list(merged.values()), duplicates


def stress_buckets(targets: list[dict[str, Any]], stats: dict[str, Any], width: int, height: int) -> list[str]:
    buckets: set[str] = set()
    counts = Counter(item["label"] for item in targets)
    if any(int(item.get("thin_extent") or 99) <= 3 for item in targets):
        buckets.add("thin_lines")
    if len(targets) >= 150:
        buckets.add("dense_boundary_graph")
    if counts.get("opening", 0) >= 30:
        buckets.add("many_openings")
    if counts.get("window", 0) >= 20:
        buckets.add("many_windows")
    if width * height >= 8_000_000:
        buckets.add("large_raster")
    stddev = stats.get("stddev")
    if isinstance(stddev, (int, float)) and stddev < 35.0:
        buckets.add("low_contrast")
    if not buckets:
        buckets.add("standard")
    return sorted(buckets)


def convert_row(row: dict[str, Any], compute_image_stats: bool, include_sources: set[str]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    if include_sources and str(row.get("source_dataset") or "") not in include_sources:
        return None, {
            "skipped": True,
            "skip_reason": "source_dataset_not_boundary_structural",
            "source_dataset": row.get("source_dataset"),
        }
    width, height = [int(v) for v in row.get("image_size") or [0, 0]]
    raw_targets = []
    for item in ((row.get("targets") or {}).get("boundary") or []):
        target = target_from_public(item, width, height)
        if target is not None:
            raw_targets.append(target)
    targets, duplicate_count = dedupe_targets(raw_targets)
    label_counts = Counter(item["label"] for item in targets)
    stats = image_stats(str(row.get("image") or "")) if compute_image_stats else {"mean": None, "stddev": None, "skipped": True}
    converted = {
        "id": row.get("id"),
        "source_key": row.get("source_row_ref") or row.get("id"),
        "source_dataset": row.get("source_dataset"),
        "split": row.get("split"),
        "image": row.get("image"),
        "image_size": [width, height],
        "targets": {
            "boxes": targets,
            "boundary_mask": None,
            "centerline_mask": None,
        },
        "target_counts": {
            "total": len(targets),
            **{label: int(label_counts.get(label, 0)) for label in LABELS},
        },
        "stress_buckets": stress_buckets(targets, stats, width, height),
        "image_stats": stats,
        "source_integrity": {
            "source_mode": "offline_public_training_gold",
            "model_input": "raster_image_only",
            "label_use": "training_or_locked_evaluation_only",
            "for_model_credit_inference": False,
            "annotation_path_exposed": False,
            "vector_geometry_exposed": False,
        },
    }
    audit = {
        "raw_boundary_targets": len(raw_targets),
        "deduped_boundary_targets": len(targets),
        "duplicate_targets_removed": duplicate_count,
        "labels": dict(label_counts),
    }
    return converted, audit


def summarize(rows_by_split: dict[str, list[dict[str, Any]]], per_row_audit: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    labels = Counter()
    sources = Counter()
    stress = Counter()
    duplicate_targets_removed = 0
    skipped_rows = 0
    large_images = 0
    for split, rows in rows_by_split.items():
        for row in rows:
            sources[str(row.get("source_dataset"))] += 1
            labels.update({label: int((row.get("target_counts") or {}).get(label) or 0) for label in LABELS})
            stress.update(row.get("stress_buckets") or [])
            width, height = [int(v) for v in row.get("image_size") or [0, 0]]
            if width * height >= 8_000_000:
                large_images += 1
        for audit in per_row_audit.get(split, []):
            duplicate_targets_removed += int(audit.get("duplicate_targets_removed") or 0)
            if audit.get("skipped"):
                skipped_rows += 1
    return {
        "splits": {split: len(rows) for split, rows in rows_by_split.items()},
        "rows": sum(len(rows) for rows in rows_by_split.values()),
        "sources": dict(sources),
        "labels": dict(labels),
        "targets": int(sum(labels.values())),
        "stress_buckets": dict(sorted(stress.items())),
        "large_images": large_images,
        "duplicate_targets_removed": duplicate_targets_removed,
        "skipped_rows": skipped_rows,
    }


def audit_runtime_contract(rows_by_split: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    forbidden_keys = {"annotation_path", "expected_json", "geometry", "points", "polygon", "polygons", "svg_id", "svg_path", "source_id"}
    violations = []

    def visit(value: Any, path: str, row_id: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                lower = key.lower()
                child_path = f"{path}.{key}" if path else key
                if lower in forbidden_keys or lower.startswith("svg_"):
                    violations.append({"row_id": row_id, "path": child_path})
                visit(child, child_path, row_id)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]", row_id)

    count = 0
    for rows in rows_by_split.values():
        for row in rows:
            count += 1
            visit(row, "", str(row.get("id") or ""))
    return {
        "records_checked": count,
        "forbidden_runtime_field_violation_count": len(violations),
        "violations": violations[:50],
        "passed": not violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(SOURCE.relative_to(ROOT)))
    parser.add_argument("--out-dir", default=str(OUT.relative_to(ROOT)))
    parser.add_argument("--audit", default="reports/vlm/boundary_public_raster_v19_dataset_audit.json")
    parser.add_argument("--compute-image-stats", action="store_true", help="Open every source image and compute grayscale stats. Slow on large public images.")
    parser.add_argument("--include-source", action="append", default=["cubicasa5k"], help="Source dataset to include for structural boundary training. Repeatable.")
    args = parser.parse_args()

    source_dir = ROOT / args.source_dir
    out_dir = ROOT / args.out_dir
    rows_by_split: dict[str, list[dict[str, Any]]] = {}
    per_row_audit: dict[str, list[dict[str, Any]]] = defaultdict(list)
    include_sources = {str(item) for item in args.include_source if item}
    for split in SPLITS:
        converted_rows = []
        for row in load_jsonl(source_dir / f"{split}.jsonl"):
            converted, row_audit = convert_row(row, args.compute_image_stats, include_sources)
            if converted is not None:
                converted_rows.append(converted)
            per_row_audit[split].append(row_audit)
        rows_by_split[split] = converted_rows
        write_jsonl(out_dir / f"{split}.jsonl", converted_rows)

    contract = audit_runtime_contract(rows_by_split)
    manifest = {
        "version": "boundary_expert_public_raster_v19",
        "task": "P0-BOUNDARY-001.step_1_dataset_view",
        "created": "2026-05-10",
        "dataset": str(out_dir.relative_to(ROOT)),
        "source_dataset": str(source_dir.relative_to(ROOT)),
        "included_source_datasets": sorted(include_sources),
        "label_map": LABEL_TO_ID,
        "target_schema": {
            "boxes": ["target_id", "label", "label_id", "bbox", "centerline", "length", "thin_extent", "raw_labels"],
            "masks": ["boundary_mask", "centerline_mask"],
            "stress_buckets": ["thin_lines", "dense_boundary_graph", "many_openings", "many_windows", "large_raster", "low_contrast", "standard"],
        },
        "summary": summarize(rows_by_split, per_row_audit),
        "split_summaries": {
            split: summarize({split: rows}, {split: per_row_audit.get(split, [])})
            for split, rows in rows_by_split.items()
        },
        "runtime_record_contract_audit": contract,
        "source_integrity_policy": {
            "runtime_input": "raster_image_only",
            "offline_public_labels_used_for": ["training_targets", "dev_selection", "locked_evaluation", "audit"],
            "offline_public_labels_forbidden_for": ["model_credit_inference", "runtime_features"],
        },
        "audit_notes": [
            "Duplicate same-label same-bbox public annotations are merged per page so Door/Threshold and Window/Glass duplicates do not overweight training.",
            "FloorPlanCAD is excluded by default from structural boundary training because its wall/door/window boxes are object-level or coarse extent annotations, not CubiCasa-style wall/opening/window structural segments.",
            "No vector geometry or annotation paths are exposed in records. Centerlines are inferred from raster target boxes for compatibility with existing boundary tooling.",
        ],
    }
    if not contract["passed"]:
        raise SystemExit(json.dumps(contract, ensure_ascii=False, indent=2))
    write_json(out_dir / "manifest.json", manifest)
    write_json(ROOT / args.audit, manifest)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
