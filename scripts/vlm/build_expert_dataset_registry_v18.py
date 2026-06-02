#!/usr/bin/env python3
"""Build an auditable dataset registry for the weak raster MoE experts."""

from __future__ import annotations

import argparse
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]

EXPERT_DATASETS: dict[str, list[dict[str, Any]]] = {
    "text_detection_ocr": [
        {
            "dataset_id": "cadstruct_text_dimensions_v1",
            "path": "datasets/cadstruct_text_dimensions_v1",
            "manifest": "datasets/cadstruct_text_dimensions_v1/manifest.json",
            "role": "primary_train",
            "source_modality": "raster_images_with_offline_structured_text_labels",
            "locked_safe_use": "train_dev_only; no locked split in this converted source",
        },
        {
            "dataset_id": "image_only_text_ocr_v18",
            "path": "datasets/image_only_text_ocr_v18",
            "manifest": "datasets/image_only_text_ocr_v18/manifest.json",
            "role": "calibration_and_locked_eval",
            "source_modality": "raster_images_with_offline_text_masks_and_transcripts",
            "locked_safe_use": "train on train/dev only; locked for final audit",
        },
        {
            "dataset_id": "cubicasa5k_external",
            "path": "datasets/external/cubicasa5k",
            "role": "external_pretrain_or_validation",
            "source_modality": "raster_images_plus_offline_svg_annotations",
            "locked_safe_use": "offline conversion only; never runtime geometry",
        },
    ],
    "symbol_body_type": [
        {
            "dataset_id": "cadstruct_symbols_v1",
            "path": "datasets/cadstruct_symbols_v1",
            "manifest": "datasets/cadstruct_symbols_v1/manifest.json",
            "role": "primary_train",
            "source_modality": "raster_images_with_offline_symbol_labels",
            "locked_safe_use": "train_dev_only; no current v18 locked split",
        },
        {
            "dataset_id": "floorplancad_symbol_pretrain_v16",
            "path": "datasets/floorplancad_symbol_pretrain_v16",
            "manifest": "datasets/floorplancad_symbol_pretrain_v16/manifest.json",
            "role": "external_pretrain",
            "source_modality": "rasterized_floorplancad_symbol_labels",
            "locked_safe_use": "pretrain only; remap taxonomy before adoption",
        },
        {
            "dataset_id": "image_only_symbol_detector_v18",
            "path": "datasets/image_only_symbol_detector_v18",
            "manifest": "datasets/image_only_symbol_detector_v18/manifest.json",
            "role": "domain_calibration_and_locked_eval",
            "source_modality": "raster_images_with_offline_symbol_masks",
            "locked_safe_use": "locked split for audit only",
        },
        {
            "dataset_id": "image_only_symbol_crops_v18",
            "path": "datasets/image_only_symbol_crops_v18",
            "manifest": "datasets/image_only_symbol_crops_v18/manifest.json",
            "role": "type_crop_train_and_locked_eval",
            "source_modality": "raster_crops_with_offline_type_labels",
            "locked_safe_use": "locked split for audit only",
        },
        {
            "dataset_id": "image_only_symbol_objectness_v18",
            "path": "datasets/image_only_symbol_objectness_v18",
            "manifest": "datasets/image_only_symbol_objectness_v18/manifest.json",
            "role": "objectness_calibration",
            "source_modality": "raster_crops_with offline positive/negative labels",
            "locked_safe_use": "locked split for audit only",
        },
        {
            "dataset_id": "floorplancad_external",
            "path": "datasets/external/floorplancad",
            "role": "external_pretrain_source",
            "source_modality": "CAD-derived labels rendered to raster for training",
            "locked_safe_use": "offline conversion only; never runtime CAD geometry",
        },
    ],
    "space_room_detection": [
        {
            "dataset_id": "cadstruct_rooms_v1",
            "path": "datasets/cadstruct_rooms_v1",
            "manifest": "datasets/cadstruct_rooms_v1/manifest.json",
            "role": "primary_train",
            "source_modality": "raster_images_with_offline_room_polygons",
            "locked_safe_use": "train_dev_only; no current v18 locked split",
        },
        {
            "dataset_id": "image_only_room_polygon_v18",
            "path": "datasets/image_only_room_polygon_v18",
            "manifest": "datasets/image_only_room_polygon_v18/manifest.json",
            "role": "domain_calibration_and_locked_eval",
            "source_modality": "raster_images_with_offline_room_masks_polygons",
            "locked_safe_use": "locked split for audit only",
        },
        {
            "dataset_id": "room_space_expert_v12_mixed",
            "path": "datasets/room_space_expert_v12_mixed",
            "manifest": "datasets/room_space_expert_v12_mixed/manifest.json",
            "role": "historical_auxiliary_train",
            "source_modality": "mixed raster-derived room supervision",
            "locked_safe_use": "audit before reuse",
        },
        {
            "dataset_id": "cubicasa5k_external",
            "path": "datasets/external/cubicasa5k",
            "role": "external_pretrain_or_validation",
            "source_modality": "raster_images_plus_offline_svg_room_labels",
            "locked_safe_use": "offline conversion only; never runtime geometry",
        },
        {
            "dataset_id": "resplan_external",
            "path": "datasets/external/resplan",
            "role": "layout_topology_auxiliary",
            "source_modality": "layout/topology supervision",
            "locked_safe_use": "auxiliary prior only after source audit",
        },
    ],
    "boundary_opening_window": [
        {
            "dataset_id": "image_only_boundary_detector_v18",
            "path": "datasets/image_only_boundary_detector_v18",
            "manifest": "datasets/image_only_boundary_detector_v18/manifest.json",
            "role": "primary_train_and_locked_eval",
            "source_modality": "raster_images_with_offline_boundary_masks",
            "locked_safe_use": "train on train/dev only; locked for final audit",
        },
        {
            "dataset_id": "cadstruct_rooms_v1",
            "path": "datasets/cadstruct_rooms_v1",
            "manifest": "datasets/cadstruct_rooms_v1/manifest.json",
            "role": "wall_supported_auxiliary",
            "source_modality": "raster_images_with_offline_room_polygons",
            "locked_safe_use": "derive training labels only; audit no room gold at runtime",
        },
        {
            "dataset_id": "cvc_fp_figshare_external",
            "path": "datasets/external/cvc_fp_figshare",
            "role": "external_validation_or_auxiliary",
            "source_modality": "floorplan images plus shape annotations",
            "locked_safe_use": "offline conversion only",
        },
        {
            "dataset_id": "floorplancad_external",
            "path": "datasets/external/floorplancad",
            "role": "external_pretrain_source",
            "source_modality": "CAD-derived labels rendered to raster for training",
            "locked_safe_use": "offline conversion only; never runtime CAD geometry",
        },
    ],
    "relation_topology": [
        {
            "dataset_id": "cubicasa_contains_symbol_assignment_v18",
            "path": "datasets/external_supervision/cubicasa_contains_symbol_assignment_v18",
            "manifest": "datasets/external_supervision/cubicasa_contains_symbol_assignment_v18/manifest.json",
            "role": "deferred_relation_train",
            "source_modality": "offline assignment labels over raster-compatible nodes",
            "locked_safe_use": "deferred until P0 experts improve",
        },
        {
            "dataset_id": "image_only_scene_graph_refiner_v18",
            "path": "datasets/image_only_scene_graph_refiner_v18",
            "manifest": "datasets/image_only_scene_graph_refiner_v18/manifest.json",
            "role": "deferred_relation_train_eval",
            "source_modality": "raster-derived scene graph candidates with offline labels",
            "locked_safe_use": "deferred until P0 experts improve",
        },
    ],
}

RARE_SYMBOL_CLASSES = ["bathtub", "generic_symbol", "table", "stair", "appliance"]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"_load_error": str(exc)}


def count_suffixes(root: Path, suffixes: set[str]) -> int:
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*") if path.is_file() and path.suffix.lower() in suffixes)


def summarize_manifest(manifest: dict[str, Any]) -> dict[str, Any]:
    if not manifest:
        return {}
    summary: dict[str, Any] = {}
    for key in ("version", "task", "dataset", "source", "source_dataset", "inference_contract"):
        if key in manifest:
            summary[key] = manifest[key]
    if "splits" in manifest:
        summary["splits"] = manifest["splits"]
    if "labels" in manifest:
        summary["labels"] = manifest["labels"]
    if "aggregate_counts" in manifest:
        summary["aggregate_counts"] = manifest["aggregate_counts"]
    if "label_map" in manifest:
        summary["label_map"] = manifest["label_map"]
    if "positive_rows" in manifest:
        summary["positive_rows"] = manifest["positive_rows"]
    if "negative_rows" in manifest:
        summary["negative_rows"] = manifest["negative_rows"]
    if "top_labels" in manifest:
        summary["top_labels"] = manifest["top_labels"]
    if "source_integrity_policy" in manifest:
        summary["source_integrity_policy"] = manifest["source_integrity_policy"]
    if "validation" in manifest:
        summary["validation"] = manifest["validation"]
    if "_load_error" in manifest:
        summary["load_error"] = manifest["_load_error"]
    return summary


def flatten_numeric_counts(obj: Any, prefix: str = "") -> dict[str, float]:
    counts: dict[str, float] = {}
    if isinstance(obj, Mapping):
        for key, value in obj.items():
            child = f"{prefix}.{key}" if prefix else str(key)
            counts.update(flatten_numeric_counts(value, child))
    elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
        counts[prefix] = float(obj)
    return counts


def dataset_entry(config: dict[str, Any]) -> dict[str, Any]:
    root = ROOT / config["path"]
    manifest_path = ROOT / config["manifest"] if config.get("manifest") else None
    manifest = load_json(manifest_path) if manifest_path else {}
    summary = summarize_manifest(manifest)
    entry = {
        "dataset_id": config["dataset_id"],
        "path": config["path"],
        "exists": root.exists(),
        "manifest": str(manifest_path.relative_to(ROOT)) if manifest_path and manifest_path.exists() else config.get("manifest"),
        "manifest_exists": bool(manifest_path and manifest_path.exists()),
        "role": config["role"],
        "source_modality": config["source_modality"],
        "offline_label_status": "allowed_for_training_eval_audit_only",
        "runtime_safety": {
            "runtime_input": "raster_image_only",
            "forbidden_runtime_features": [
                "SVG geometry",
                "CAD parser geometry",
                "expected_json",
                "gold labels",
                "offline semantic annotations",
            ],
            "locked_safe_use": config["locked_safe_use"],
        },
        "manifest_summary": summary,
        "file_counts": {
            "jsonl": count_suffixes(root, {".jsonl"}),
            "json": count_suffixes(root, {".json"}),
            "images": count_suffixes(root, {".png", ".jpg", ".jpeg", ".tif", ".tiff"}),
            "masks": count_suffixes(root / "masks", {".png", ".jpg", ".jpeg", ".tif", ".tiff"}) if root.exists() else 0,
        },
    }
    entry["numeric_count_index"] = flatten_numeric_counts(summary)
    return entry


def build_registry() -> dict[str, Any]:
    experts: dict[str, Any] = {}
    for expert, datasets in EXPERT_DATASETS.items():
        entries = [dataset_entry(config) for config in datasets]
        experts[expert] = {
            "datasets": entries,
            "has_primary_training_source": any("primary" in item["role"] and item["exists"] for item in entries),
            "has_locked_compatible_eval": any(
                ("locked" in item["role"] or "locked" in item["runtime_safety"]["locked_safe_use"]) and item["exists"]
                for item in entries
            ),
        }
    return {
        "schema_version": "cadstruct_expert_dataset_registry_v18",
        "created": "2026-05-10",
        "purpose": "Map weak raster-only MoE experts to auditable training, calibration, locked-eval, and external-validation datasets.",
        "hard_contract": {
            "target_model": "MoE model for non-vector, raster-only drawing recognition.",
            "runtime_input": "Raster image only.",
            "offline_label_use": "Offline SVG/CAD/structured labels are allowed only for conversion, supervised training, dev/locked evaluation, error mining, and upper-bound audits.",
            "forbidden_runtime_features": [
                "SVG geometry",
                "CAD parser geometry",
                "expected_json",
                "gold labels",
                "offline semantic annotations",
            ],
        },
        "experts": experts,
        "data_gaps": build_data_gaps(experts),
        "next_training_order": [
            "text_detection_ocr",
            "symbol_body_type",
            "space_room_detection",
            "boundary_opening_window",
            "relation_topology_deferred",
        ],
    }


def build_data_gaps(experts: dict[str, Any]) -> list[dict[str, Any]]:
    gaps: list[dict[str, Any]] = []
    symbol_labels: dict[str, float] = {}
    for item in experts["symbol_body_type"]["datasets"]:
        labels = item.get("manifest_summary", {}).get("labels") or {}
        if isinstance(labels, dict):
            for label, count in labels.items():
                if isinstance(count, (int, float)):
                    symbol_labels[label] = max(symbol_labels.get(label, 0.0), float(count))
    for label in RARE_SYMBOL_CLASSES:
        gaps.append(
            {
                "expert": "symbol_body_type",
                "class": label,
                "largest_local_label_count": int(symbol_labels.get(label, 0)),
                "risk": "rare_class_or_taxonomy_gap",
                "required_action": "oversample, map external labels, and report per-class recall before adoption",
            }
        )
    text_v18 = experts["text_detection_ocr"]["datasets"][1]["manifest_summary"].get("aggregate_counts", {})
    semantic_types = text_v18.get("semantic_types", {}) if isinstance(text_v18, dict) else {}
    for label in ("dimension_text", "note_text", "dimension_line"):
        gaps.append(
            {
                "expert": "text_detection_ocr",
                "class": label,
                "v18_label_count": int(semantic_types.get(label, 0)) if isinstance(semantic_types, dict) else 0,
                "risk": "v18_domain_has_too_few_examples",
                "required_action": "pretrain from cadstruct_text_dimensions_v1 and calibrate on v18",
            }
        )
    return gaps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/vlm/expert_dataset_registry_v18.json")
    args = parser.parse_args()

    registry = build_registry()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(output.relative_to(ROOT)),
                "experts": list(registry["experts"].keys()),
                "data_gap_count": len(registry["data_gaps"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
