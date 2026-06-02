#!/usr/bin/env python3
"""Audit external symbol datasets for the P0-21 registry and class-map plan."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from train_symbol_tile_detector_v20 import area_bucket, rel, write_json

ROOT = Path(__file__).resolve().parents[2]

CURRENT_SYMBOL_SPACE = ["sink", "shower", "stair", "equipment", "appliance", "generic_symbol", "table", "bathtub", "column"]

FLOORPLANCAD_CLASS_MAP = {
    "sink": "sink",
    "shower": "shower",
    "stair": "stair",
    "stairs": "stair",
    "escalator": "stair",
    "table": "table",
    "bathtub": "bathtub",
    "bath_tub": "bathtub",
    "bath": "bathtub",
    "column": "column",
    "chair": "equipment",
    "bed": "equipment",
    "sofa": "equipment",
    "wardrobe": "equipment",
    "half_height_cabinet": "equipment",
    "high_cabinet": "equipment",
    "tv_cabinet": "equipment",
    "bedside_cupboard": "equipment",
    "parking": "generic_symbol",
    "toilet": "equipment",
    "squat_toilet": "equipment",
    "urinal": "equipment",
    "single_door": "generic_symbol",
    "double_door": "generic_symbol",
    "sliding_door": "generic_symbol",
    "window": "generic_symbol",
    "bay_window": "generic_symbol",
    "blind_window": "generic_symbol",
    "opening_symbol": "generic_symbol",
    "wall": None,
    "class_31": "generic_symbol",
    "class_32": "generic_symbol",
    "class_34": "generic_symbol",
    "class_35": "generic_symbol",
}


def source_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    out: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                out.append(json.loads(line))
                if limit is not None and len(out) >= limit:
                    break
    return out


def file_count(root: Path, suffixes: set[str]) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def valid_box(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        box = [float(v) for v in value]
    except (TypeError, ValueError):
        return None
    if box[2] <= box[0] or box[3] <= box[1]:
        return None
    return box


def summarize_symbols(symbols: list[dict[str, Any]], map_fn) -> dict[str, Any]:
    raw = Counter()
    mapped = Counter()
    buckets = Counter()
    invalid = 0
    for sym in symbols:
        raw_label = str(sym.get("semantic_type") or sym.get("label") or "")
        raw[raw_label] += 1
        box = valid_box(sym.get("bounding_box") or sym.get("bbox"))
        if box is None:
            invalid += 1
            continue
        mapped_label = map_fn(raw_label)
        if mapped_label is None:
            continue
        mapped[mapped_label] += 1
        buckets[area_bucket(box)] += 1
    total = sum(mapped.values())
    covered = sum(mapped[label] for label in CURRENT_SYMBOL_SPACE if label in mapped)
    return {
        "raw_label_counts_top30": dict(raw.most_common(30)),
        "mapped_label_counts": dict(mapped.most_common()),
        "mapped_total": total,
        "mapped_current_symbol_space": covered,
        "mapped_current_symbol_ratio": round(covered / max(total, 1), 6),
        "invalid_targets": invalid,
        "area_bucket_counts": dict(buckets.most_common()),
    }


def audit_floorplancad(root: Path) -> dict[str, Any]:
    samples_path = root / "samples.json"
    metadata_path = root / "metadata.json"
    readme_path = root / "README.md"
    samples = load_json(samples_path).get("samples") or []
    per_split = Counter()
    total_symbols: list[dict[str, Any]] = []
    for sample in samples:
        filepath = str(sample.get("filepath") or "")
        split = filepath.split("/", 1)[0] if "/" in filepath else "unknown"
        per_split[split] += 1
        total_symbols.extend(((sample.get("ground_truth") or {}).get("detections") or []))
    mapped = summarize_symbols(total_symbols, lambda raw: FLOORPLANCAD_CLASS_MAP.get(raw, "generic_symbol"))
    return {
        "name": "FloorPlanCAD",
        "path": rel(root),
        "available": root.exists() and bool(samples),
        "trainable": bool(total_symbols),
        "license": "CC BY-NC-SA 4.0",
        "source": "https://huggingface.co/datasets/Voxel51/FloorPlanCAD",
        "split_counts": dict(per_split),
        "sample_count": len(samples),
        "image_file_count": file_count(root / "data", {".png", ".jpg", ".jpeg"}),
        "metadata_file_present": metadata_path.exists(),
        "readme_present": readme_path.exists(),
        "symbol_summary": mapped,
        "class_map": FLOORPLANCAD_CLASS_MAP,
        "source_integrity": {
            "runtime_input": "raster image only",
            "offline_supervision": "FiftyOne detections from local export",
            "svg_or_cad_geometry_at_runtime": False,
        },
        "decision": "use_for_symbol_bbox_pretrain_and_type_pretrain",
    }


def audit_floorplancad_pretrain(root: Path) -> dict[str, Any]:
    train = load_jsonl(root / "train.jsonl")
    label_counts = Counter()
    split_image_ids = set()
    symbol_counts = Counter()
    invalid = 0
    for row in train:
        split_image_ids.add(str(row.get("image") or row.get("id") or ""))
        for sym in ((row.get("structured") or {}).get("symbols") or []):
            raw = str(sym.get("semantic_type") or sym.get("label") or "")
            label = FLOORPLANCAD_CLASS_MAP.get(raw, "generic_symbol")
            box = valid_box(sym.get("bbox"))
            if box is None:
                invalid += 1
                continue
            if label is None:
                continue
            label_counts[label] += 1
            symbol_counts["symbols"] += 1
    return {
        "name": "floorplancad_symbol_pretrain_v16",
        "path": rel(root),
        "available": root.exists(),
        "trainable": bool(train),
        "license": "CC BY-NC-SA 4.0 / derived local raster export",
        "source": "datasets/external/floorplancad + local converted raster export",
        "rows": len(train),
        "image_file_count": file_count(root / "images", {".png", ".jpg", ".jpeg"}),
        "class_map": FLOORPLANCAD_CLASS_MAP,
        "mapped_label_counts": dict(label_counts.most_common()),
        "invalid_targets": invalid,
        "source_integrity": {
            "runtime_input": "raster image only",
            "offline_supervision": "local converted bbox JSONL",
            "svg_or_cad_geometry_at_runtime": False,
        },
        "decision": "use_for_symbol_bbox_pretrain_and_type_pretrain",
    }


def audit_simple_external(name: str, path: Path, source_url: str, license_name: str, role: str) -> dict[str, Any]:
    files = list(path.rglob("*")) if path.exists() else []
    images = sum(1 for p in files if p.is_file() and p.suffix.lower() in {".png", ".jpg", ".jpeg"})
    jsonl = sum(1 for p in files if p.is_file() and p.suffix.lower() == ".jsonl")
    json_files = sum(1 for p in files if p.is_file() and p.suffix.lower() == ".json")
    return {
        "name": name,
        "path": rel(path),
        "available": path.exists(),
        "trainable": False,
        "source": source_url,
        "license": license_name,
        "counts": {"image_files": images, "json_files": json_files, "jsonl_files": jsonl},
        "role": role,
        "decision": "research_only_until_bbox_labels_are_verified" if not jsonl else "needs_schema_audit_before_training",
    }


def build_registry() -> dict[str, Any]:
    external_datasets = {
        "FloorPlanCAD": audit_floorplancad(ROOT / "datasets/external/floorplancad"),
        "floorplancad_symbol_pretrain_v16": audit_floorplancad_pretrain(ROOT / "datasets/floorplancad_symbol_pretrain_v16"),
        "CubiCasa5K": audit_simple_external(
            "CubiCasa5K",
            ROOT / "datasets/external/cubicasa5k",
            "https://github.com/CubiCasa/CubiCasa5k",
            "Apache-2.0",
            "raster_symbol_and_room_source_only; offline SVG-derived supervision only",
        ),
        "CVC-FP": audit_simple_external(
            "CVC-FP",
            ROOT / "datasets/external/cvc_fp",
            "https://dag.cvc.uab.es/dataset/cvc-fp-database-for-structural-floor-plan-analysis/",
            "CC BY-NC 4.0",
            "structural floor-plan auxiliary source",
        ),
        "ResPlan": audit_simple_external(
            "ResPlan",
            ROOT / "datasets/external/resplan",
            "local package / README",
            "license in repo",
            "layout/topology auxiliary source",
        ),
        "MLStructFP": audit_simple_external(
            "MLStructFP",
            ROOT / "datasets/external/mlstructfp",
            "repo/README.rst",
            "repo license",
            "blocked until public download link is resolved",
        ),
        "CubiCasa5K_hf": audit_simple_external(
            "CubiCasa5K_hf",
            ROOT / "datasets/external/cubicasa5k_hf",
            "huggingface mirror metadata",
            "Apache-2.0 mirror metadata",
            "mirror metadata only",
        ),
    }
    return {
        "version": "symbol_external_dataset_audit_v47",
        "current_symbol_space": CURRENT_SYMBOL_SPACE,
        "external_datasets": external_datasets,
        "summary": {
            "available_symbol_bbox_sources": [name for name, item in external_datasets.items() if item.get("decision") == "use_for_symbol_bbox_pretrain_and_type_pretrain"],
            "research_only_sources": [name for name, item in external_datasets.items() if "research_only" in str(item.get("decision"))],
            "blocked_sources": [name for name, item in external_datasets.items() if "blocked" in str(item.get("role", "")) or "blocked" in str(item.get("decision", ""))],
        },
        "source_integrity": {
            "runtime_input": "raster image only",
            "offline_supervision": "all labels are offline and for audit/training only",
            "svg_or_cad_geometry_at_runtime": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="reports/vlm/symbol_external_dataset_audit_v47.json")
    parser.add_argument("--registry-output", default="configs/vlm/symbol_external_dataset_registry_v47.json")
    parser.add_argument("--class-map-output", default="configs/vlm/symbol_cross_dataset_class_map_v47.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report = build_registry()
    write_json(source_path(args.output), report)
    registry = {
        "version": "symbol_external_dataset_registry_v47",
        "datasets": report["external_datasets"],
        "summary": report["summary"],
        "source_integrity": report["source_integrity"],
    }
    write_json(source_path(args.registry_output), registry)
    class_map = {
        "version": "symbol_cross_dataset_class_map_v47",
        "current_symbol_space": CURRENT_SYMBOL_SPACE,
        "floorplancad_map": FLOORPLANCAD_CLASS_MAP,
        "notes": {
            "wall": "excluded from symbol detector claims; belongs to boundary/structure tasks",
            "parking": "treated as generic_symbol for symbol detector pretraining only",
            "class_31": "unknown semantic bucket in local export; map to generic_symbol until clarified",
            "class_32": "unknown semantic bucket in local export; map to generic_symbol until clarified",
            "class_34": "unknown semantic bucket in local export; map to generic_symbol until clarified",
            "class_35": "unknown semantic bucket in local export; map to generic_symbol until clarified",
        },
        "source_integrity": report["source_integrity"],
    }
    write_json(source_path(args.class_map_output), class_map)
    print(json.dumps({"summary": report["summary"], "registry_output": rel(source_path(args.registry_output)), "class_map_output": rel(source_path(args.class_map_output))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
