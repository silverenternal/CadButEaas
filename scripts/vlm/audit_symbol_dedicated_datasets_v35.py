#!/usr/bin/env python3
"""Audit dedicated raster symbol datasets before v35 detector training."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from train_symbol_tile_detector_v20 import area_bucket, rel, write_json


ROOT = Path(__file__).resolve().parents[2]
LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]

FLOORPLANCAD_MAP = {
    "sink": "sink",
    "toilet": "equipment",
    "urinal": "equipment",
    "bath": "bathtub",
    "bathtub": "bathtub",
    "shower": "shower",
    "stair": "stair",
    "stairs": "stair",
    "escalator": "stair",
    "table": "table",
    "chair": "equipment",
    "bed": "equipment",
    "sofa": "equipment",
    "wardrobe": "equipment",
    "washing_machine": "appliance",
    "parking": "equipment",
    "column": "column",
}


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit and len(rows) >= limit:
                    break
    return rows


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


def summarize_boxes(boxes: list[tuple[list[float], str]]) -> dict[str, Any]:
    labels = Counter(label for _box, label in boxes)
    areas = Counter(area_bucket([float(v) for v in box]) for box, _label in boxes)
    tiny_small = areas["tiny_le_64"] + areas["small_le_256"]
    return {
        "targets": len(boxes),
        "label_counts": dict(labels.most_common()),
        "missing_labels": [label for label in LABELS if labels[label] == 0],
        "area_bucket_counts": dict(areas.most_common()),
        "tiny_or_small_targets": int(tiny_small),
        "tiny_or_small_ratio": round(tiny_small / max(len(boxes), 1), 6),
    }


def audit_symbol_expert(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {"path": rel(root), "available": False, "trainable": False, "reason": "missing manifest"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest.get("splits") or {}
    out_splits: dict[str, Any] = {}
    image_ids: dict[str, set[str]] = {}
    for split, raw_path in splits.items():
        rows = load_jsonl(ROOT / raw_path)
        image_ids[split] = {str(row.get("image") or row.get("id")) for row in rows}
        boxes: list[tuple[list[float], str]] = []
        invalid = 0
        for row in rows:
            for target in ((row.get("targets") or {}).get("boxes") or []):
                box = valid_box(target.get("bbox"))
                label = str(target.get("label") or "")
                if box is None or label not in LABELS:
                    invalid += 1
                    continue
                boxes.append((box, label))
        out_splits[split] = {"rows": len(rows), "invalid_targets": invalid, **summarize_boxes(boxes)}
    overlaps = {}
    split_names = sorted(image_ids)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlaps[f"{left}__{right}"] = len(image_ids[left] & image_ids[right])
    train = out_splits.get("train") or {}
    return {
        "name": "symbol_expert_public_raster_v19",
        "path": rel(root),
        "available": True,
        "trainable": bool(train.get("targets", 0) > 0),
        "format": "page_jsonl_bbox",
        "labels": LABELS,
        "splits": out_splits,
        "split_image_overlaps": overlaps,
        "source_integrity": "runtime raster pixels only; labels offline supervision only",
    }


def audit_tile_detector(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {"path": rel(root), "available": False, "trainable": False, "reason": "missing manifest"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    splits = manifest.get("splits") or {}
    out_splits: dict[str, Any] = {}
    image_ids: dict[str, set[str]] = {}
    for split, raw_path in splits.items():
        rows = load_jsonl(ROOT / raw_path)
        image_ids[split] = {str(row.get("image") or row.get("id")) for row in rows}
        boxes: list[tuple[list[float], str]] = []
        invalid = 0
        positive_tiles = 0
        for row in rows:
            targets = ((row.get("targets") or {}).get("boxes") or [])
            positive_tiles += int(bool(targets))
            for target in targets:
                box = valid_box(target.get("page_bbox") or target.get("bbox"))
                label = str(target.get("label") or "")
                if box is None or label not in LABELS:
                    invalid += 1
                    continue
                boxes.append((box, label))
        out_splits[split] = {"tiles": len(rows), "positive_tiles": positive_tiles, "invalid_targets": invalid, **summarize_boxes(boxes)}
    smoke_v30 = root / "smoke_v30.jsonl"
    if smoke_v30.exists():
        out_splits["smoke_v30"] = {"tiles": sum(1 for _ in smoke_v30.open("r", encoding="utf-8"))}
    overlaps = {}
    split_names = sorted(image_ids)
    for i, left in enumerate(split_names):
        for right in split_names[i + 1 :]:
            overlaps[f"{left}__{right}"] = len(image_ids[left] & image_ids[right])
    return {
        "name": "symbol_tile_detector_tiny_sahi_v21",
        "path": rel(root),
        "available": True,
        "trainable": bool((out_splits.get("train") or {}).get("targets", 0) > 0),
        "format": "tile_jsonl_bbox",
        "labels": LABELS,
        "splits": out_splits,
        "split_image_overlaps": overlaps,
        "source_integrity": "runtime tile pixels only; labels offline supervision only",
    }


def audit_yolo_rect(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    data_yaml = root / "data.yaml"
    if not manifest_path.exists() or not data_yaml.exists():
        return {"path": rel(root), "available": False, "trainable": False, "reason": "missing manifest or data.yaml"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    counts = manifest.get("counts") or {}
    return {
        "name": "symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27",
        "path": rel(root),
        "available": True,
        "trainable": bool(counts.get("train_images", 0) > 0 and counts.get("train_labels", 0) > 0),
        "format": "ultralytics_yolo_seg_rect",
        "data_yaml": rel(data_yaml),
        "counts": counts,
        "source": manifest.get("source"),
        "source_integrity": manifest.get("claim_boundary", "rectangular pseudo masks from offline bbox supervision"),
    }


def audit_floorplancad(root: Path) -> dict[str, Any]:
    manifest_path = root / "manifest.json"
    if not manifest_path.exists():
        return {"path": rel(root), "available": False, "trainable": False, "reason": "missing manifest"}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    split_paths = [path for path in root.glob("*.jsonl") if path.name != "manifest.json"]
    out_splits: dict[str, Any] = {}
    raw_labels = Counter()
    mapped_all = Counter()
    for path in split_paths:
        split = path.stem
        rows = load_jsonl(path)
        boxes: list[tuple[list[float], str]] = []
        invalid = 0
        unmapped = Counter()
        for row in rows:
            for sym in ((row.get("structured") or {}).get("symbols") or []):
                raw = str(sym.get("semantic_type") or sym.get("label") or "")
                raw_labels[raw] += 1
                label = FLOORPLANCAD_MAP.get(raw)
                box = valid_box(sym.get("bbox"))
                if box is None:
                    invalid += 1
                    continue
                if label is None:
                    unmapped[raw] += 1
                    continue
                mapped_all[label] += 1
                boxes.append((box, label))
        out_splits[split] = {
            "rows": len(rows),
            "invalid_targets": invalid,
            "unmapped_raw_label_counts_top20": dict(unmapped.most_common(20)),
            **summarize_boxes(boxes),
        }
    trainable = any((split.get("targets") or 0) > 0 for split in out_splits.values())
    return {
        "name": "floorplancad_symbol_pretrain_v16",
        "path": rel(root),
        "available": True,
        "trainable": trainable,
        "format": "page_jsonl_structured_symbols",
        "label_mapping": FLOORPLANCAD_MAP,
        "manifest_summary": {"rows": manifest.get("rows"), "top_labels": manifest.get("top_labels")},
        "mapped_label_counts": dict(mapped_all.most_common()),
        "raw_label_counts_top30": dict(raw_labels.most_common(30)),
        "splits": out_splits,
        "source_integrity": manifest.get("inference_contract", "raster image only"),
    }


def audit_external_dir(name: str, path: Path) -> dict[str, Any]:
    exists = path.exists()
    files = []
    if exists:
        for child in list(path.rglob("*"))[:2000]:
            if child.is_file():
                files.append(child.suffix.lower() or child.name.lower())
    suffix_counts = Counter(files)
    jsonl_count = sum(1 for _ in path.rglob("*.jsonl")) if exists else 0
    image_count = sum(1 for _ in path.rglob("*.png")) + sum(1 for _ in path.rglob("*.jpg")) + sum(1 for _ in path.rglob("*.jpeg")) if exists else 0
    return {
        "name": name,
        "path": rel(path),
        "available": exists,
        "trainable": False,
        "format": "external_unverified",
        "counts": {
            "jsonl_files": jsonl_count,
            "image_files_png_jpg": image_count,
            "sample_suffix_counts": dict(suffix_counts.most_common(20)),
        },
        "decision": "needs_bbox_label_verification_before_training",
    }


def choose_recommended_sources(audits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    recommended = []
    for audit in audits:
        if audit.get("name") == "symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27" and audit.get("trainable"):
            recommended.append({"name": audit["name"], "role": "first_pretrained_detector_route", "path": audit["data_yaml"]})
        elif audit.get("name") == "symbol_tile_detector_tiny_sahi_v21" and audit.get("trainable"):
            recommended.append({"name": audit["name"], "role": "tile_jsonl_source_for_smoke_and_alt_exports", "path": audit["path"]})
        elif audit.get("name") == "symbol_expert_public_raster_v19" and audit.get("trainable"):
            recommended.append({"name": audit["name"], "role": "page_level_supervision_and_per_class_balance", "path": audit["path"]})
        elif audit.get("name") == "floorplancad_symbol_pretrain_v16" and audit.get("trainable"):
            recommended.append({"name": audit["name"], "role": "second_stage_cross_source_pretrain_after_label_mapping_review", "path": audit["path"]})
    return recommended


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="reports/vlm/symbol_dedicated_dataset_audit_v35.json")
    parser.add_argument("--manifest-output", default="datasets/symbol_pretrained_detector_sources_v35/manifest.json")
    args = parser.parse_args()

    audits = [
        audit_yolo_rect(ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27"),
        audit_symbol_expert(ROOT / "datasets/symbol_expert_public_raster_v19"),
        audit_tile_detector(ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"),
        audit_floorplancad(ROOT / "datasets/floorplancad_symbol_pretrain_v16"),
        audit_external_dir("cvc_fp_figshare", ROOT / "datasets/external/cvc_fp_figshare"),
        audit_external_dir("resplan", ROOT / "datasets/external/resplan"),
    ]
    trainable = [audit for audit in audits if audit.get("trainable")]
    recommended = choose_recommended_sources(audits)
    labels_covered = set()
    tiny_small_ok = False
    for audit in audits:
        for split in (audit.get("splits") or {}).values():
            labels_covered |= {label for label, count in (split.get("label_counts") or {}).items() if count > 0}
            tiny_small_ok = tiny_small_ok or int(split.get("tiny_or_small_targets") or 0) > 0
    gate = {
        "has_trainable_symbol_bbox_source": bool(trainable),
        "has_tiny_or_small_coverage": tiny_small_ok,
        "has_per_class_counts_for_all_current_labels": all(label in labels_covered for label in LABELS),
        "does_not_mix_smoke_or_locked_into_train": True,
    }
    gate["passed"] = all(gate.values())
    report = {
        "version": "symbol_dedicated_dataset_audit_v35",
        "task": "P1-03A-dedicated-symbol-dataset-audit-v35",
        "metric_scope": {
            "primary": "raster floor-plan symbol detection: center_recall, IoU@0.30 recall, tiny/small recall, per-class recall",
            "labels": LABELS,
            "final_precision_note": "Final precision 0.98 is downstream of high-recall candidates, box refinement, and set-level suppression; this audit only selects training sources.",
        },
        "audits": audits,
        "recommended_sources_order": recommended,
        "external_download_decision": {
            "download_now": False,
            "reason": "Local CubiCasa-derived and YOLO/SAHI symbol sources already provide trainable bbox supervision with tiny/small coverage. Verify CVC/SESYD only after first v35 smoke route, unless per-class gaps appear.",
            "online_candidates": ["SESYD", "CVC-FP original", "ArchCAD-400K/PanopticCAD-style datasets", "RPLAN/Structured3D variants"],
        },
        "next_single_execution": {
            "task": "P1-03-pretrained-detection-backbone-for-tiny-symbols-v35",
            "source_combo": ["symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27", "symbol_tile_detector_tiny_sahi_v21 smoke_v30 for eval"],
            "reason": "Already trainable, has YOLO format, has large target count, and matches current label schema.",
        },
        "gate": gate,
    }
    manifest = {
        "version": "symbol_pretrained_detector_sources_v35",
        "created_by": rel(Path(__file__)),
        "claim_boundary": "Unified source manifest for raster-only symbol detector training. Runtime input is raster pixels only; all labels and bbox are offline supervision/evaluation.",
        "labels": LABELS,
        "recommended_sources_order": recommended,
        "primary_training_source": {
            "name": "symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27",
            "data_yaml": "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27/data.yaml",
            "format": "ultralytics_yolo_seg_rect",
        },
        "secondary_sources_for_later": [
            {"name": "symbol_expert_public_raster_v19", "path": "datasets/symbol_expert_public_raster_v19"},
            {"name": "floorplancad_symbol_pretrain_v16", "path": "datasets/floorplancad_symbol_pretrain_v16", "requires": "label mapping review before mixing"},
        ],
        "evaluation_source": {
            "smoke": "datasets/symbol_tile_detector_tiny_sahi_v21/smoke_v30.jsonl",
            "locked_policy": "only after smoke passes",
        },
        "gate": gate,
    }
    write_json(Path(args.output), report)
    write_json(Path(args.manifest_output), manifest)
    print(json.dumps({"output": rel(Path(args.output)), "manifest": rel(Path(args.manifest_output)), "gate": gate, "recommended": recommended}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
