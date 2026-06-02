#!/usr/bin/env python3
"""Build raster-only public supervision from CubiCasa5K and FloorPlanCAD.

The output is intentionally not an SVG dataset. SVG/CAD/vector annotations are
used only offline to create labels; runtime-visible records contain raster image
paths plus target boxes/classes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "datasets/public_raster_moe_supervision_v19"
REPORT = ROOT / "reports/vlm"


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


def abs_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def image_size(path: str) -> list[int]:
    with Image.open(abs_path(path)) as img:
        return [int(img.width), int(img.height)]


def source_integrity(dataset: str) -> dict[str, Any]:
    return {
        "dataset": dataset,
        "runtime_input": "raster_image_only",
        "record_fields_expose_svg_or_cad_geometry": False,
        "annotation_path_exposed": False,
        "vector_candidate_ids_exposed": False,
        "offline_labels_used_for": ["supervised_training", "dev_selection", "locked_evaluation", "audit"],
        "offline_labels_forbidden_for": ["model_credit_inference", "runtime_features"],
    }


def bbox4(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [round(x1, 3), round(y1, 3), round(x2, 3), round(y2, 3)]


def target(target_id: str, family: str, label: str, box: list[float], raw_label: str | None = None, text: str | None = None) -> dict[str, Any]:
    item = {
        "target_id": target_id,
        "family": family,
        "semantic_type": label,
        "bbox": box,
        "label_source": "offline_public_annotation_converted_to_raster_target",
    }
    if raw_label:
        item["raw_label"] = raw_label
    if text is not None:
        item["text"] = text
        item["normalized_text"] = " ".join(text.strip().lower().split())
    return item


def convert_cubicasa_row(row: dict[str, Any], split: str, index: int) -> dict[str, Any] | None:
    image = str(row.get("image_path") or "")
    if not image or not abs_path(image).exists():
        return None
    expected = row.get("expected_json") or {}
    targets = {"boundary": [], "space": [], "symbol": [], "text": []}
    for idx, item in enumerate(expected.get("semantic_candidates") or []):
        label = cubicasa_boundary_label(str(item.get("semantic_type") or ""))
        if label is None:
            continue
        box = bbox4(item.get("bbox") or bbox_from_geometry(item.get("geometry")))
        if box:
            targets["boundary"].append(target(f"cubicasa_{split}_{index}_boundary_{idx}", "boundary", label, box, item.get("raw_label")))
    for idx, item in enumerate(expected.get("room_candidates") or []):
        box = bbox4(item.get("bbox"))
        if box:
            label = str(item.get("room_type") or item.get("semantic_type") or "room")
            targets["space"].append(target(f"cubicasa_{split}_{index}_room_{idx}", "space", label, box, item.get("raw_label")))
    for idx, item in enumerate(expected.get("symbol_candidates") or []):
        box = bbox4(item.get("bbox"))
        if box:
            label = cubicasa_symbol_label(str(item.get("symbol_type") or item.get("semantic_type") or "generic_symbol"))
            targets["symbol"].append(target(f"cubicasa_{split}_{index}_symbol_{idx}", "symbol", label, box, item.get("raw_label")))
    for idx, item in enumerate(expected.get("text_candidates") or []):
        box = bbox4(item.get("bbox"))
        if box:
            label = str(item.get("text_type") or item.get("semantic_type") or "room_label")
            targets["text"].append(target(f"cubicasa_{split}_{index}_text_{idx}", "text", label, box, item.get("raw_label"), str(item.get("text") or "")))

    return raster_record(
        row_id=f"cubicasa5k_{split}_{index:05d}",
        split=split,
        source_dataset="cubicasa5k",
        image=image,
        targets=targets,
        source_row_ref=hashlib.sha1(str(row.get("annotation_path") or image).encode("utf-8")).hexdigest()[:16],
    )


def bbox_from_geometry(geometry: Any) -> list[float] | None:
    if not isinstance(geometry, dict):
        return None
    points = geometry.get("points")
    if not isinstance(points, list) or not points:
        return None
    try:
        xs = [float(p[0]) for p in points if isinstance(p, list) and len(p) >= 2]
        ys = [float(p[1]) for p in points if isinstance(p, list) and len(p) >= 2]
    except (TypeError, ValueError):
        return None
    if not xs or not ys:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def cubicasa_boundary_label(label: str) -> str | None:
    value = label.strip()
    if value in {"hard_wall", "partition_wall", "wall"}:
        return "wall"
    if value in {"door", "opening"}:
        return "opening"
    if value == "window":
        return "window"
    return None


def cubicasa_symbol_label(label: str) -> str:
    value = label.strip()
    if value in {"appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"}:
        return value
    return "generic_symbol"


def convert_floorplancad_sample(sample: dict[str, Any], index: int, split: str, root: str) -> dict[str, Any] | None:
    rel_image = str(sample.get("filepath") or "")
    image = str((Path(root) / rel_image).as_posix())
    if not rel_image or not abs_path(image).exists():
        return None
    meta = sample.get("metadata") or {}
    width = int(meta.get("width") or image_size(image)[0])
    height = int(meta.get("height") or image_size(image)[1])
    targets = {"boundary": [], "space": [], "symbol": [], "text": []}
    for det_index, det in enumerate(((sample.get("ground_truth") or {}).get("detections") or [])):
        raw_label = str(det.get("label") or "")
        mapped = floorplancad_label(raw_label)
        if mapped is None:
            continue
        family, label = mapped
        box = floorplancad_bbox(det.get("bounding_box"), width, height)
        if box:
            targets[family].append(target(f"floorplancad_{index:05d}_{family}_{det_index}", family, label, box, raw_label))
    return raster_record(
        row_id=f"floorplancad_{index:05d}",
        split=split,
        source_dataset="floorplancad",
        image=image,
        targets=targets,
        source_row_ref=str((sample.get("_id") or {}).get("$oid") or index),
        size=[width, height],
    )


def floorplancad_bbox(value: Any, width: int, height: int) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x, y, w, h = [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    box = [x * width, y * height, (x + w) * width, (y + h) * height]
    return bbox4(box)


def floorplancad_label(label: str) -> tuple[str, str] | None:
    if label in {"wall", "railing"}:
        return "boundary", "wall"
    if label in {"window", "bay_window", "blind_window"}:
        return "boundary", "window"
    if label in {"single_door", "double_door", "sliding_door", "opening_symbol"}:
        return "boundary", "opening"
    if label in {"bath", "bath_tub"}:
        return "symbol", "bathtub"
    if label == "sink":
        return "symbol", "sink"
    if label in {"stair", "escalator"}:
        return "symbol", "stair"
    if label == "table":
        return "symbol", "table"
    if label in {"refrigerator", "washing_machine", "gas_stove"}:
        return "symbol", "appliance"
    if label in {"toilet", "squat_toilet", "urinal", "elevator"}:
        return "symbol", "equipment"
    if label in {
        "chair",
        "parking",
        "wardrobe",
        "bed",
        "tv_cabinet",
        "half_height_cabinet",
        "high_cabinet",
        "sofa",
        "bedside_cupboard",
        "class_31",
        "class_32",
        "class_34",
        "class_35",
    }:
        return "symbol", "generic_symbol"
    return None


def raster_record(
    row_id: str,
    split: str,
    source_dataset: str,
    image: str,
    targets: dict[str, list[dict[str, Any]]],
    source_row_ref: str,
    size: list[int] | None = None,
) -> dict[str, Any]:
    size = size or image_size(image)
    counts = {family: len(items) for family, items in targets.items()}
    return {
        "id": row_id,
        "split": split,
        "source_dataset": source_dataset,
        "image": image,
        "image_size": size,
        "targets": targets,
        "target_counts": counts,
        "source_row_ref": source_row_ref,
        "source_integrity": source_integrity(source_dataset),
    }


def split_floorplancad(index: int, sample: dict[str, Any]) -> str:
    key = str((sample.get("_id") or {}).get("$oid") or sample.get("filepath") or index)
    value = int(hashlib.sha1(key.encode("utf-8")).hexdigest()[:8], 16) % 100
    if value < 80:
        return "train"
    if value < 90:
        return "dev"
    return "locked"


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_source = Counter(row["source_dataset"] for row in rows)
    by_split = Counter(row["split"] for row in rows)
    family_counts = Counter()
    label_counts = Counter()
    for row in rows:
        for family, items in (row.get("targets") or {}).items():
            family_counts[family] += len(items)
            for item in items:
                label_counts[f"{family}:{item['semantic_type']}"] += 1
    return {
        "rows": len(rows),
        "by_source": dict(by_source),
        "by_split": dict(by_split),
        "family_counts": dict(family_counts),
        "label_counts": dict(label_counts.most_common()),
    }


def audit_runtime_record_contract(rows: list[dict[str, Any]]) -> dict[str, Any]:
    forbidden_exact_keys = {
        "annotation_path",
        "expected_json",
        "geometry",
        "points",
        "polygon",
        "polygons",
        "svg_path",
        "svg_id",
        "source_id",
        "cad_entity",
        "cad_geometry",
    }
    allowed_top_keys = {
        "id",
        "split",
        "source_dataset",
        "image",
        "image_size",
        "targets",
        "target_counts",
        "source_row_ref",
        "source_integrity",
    }
    violations: list[dict[str, Any]] = []
    top_key_violations: list[dict[str, Any]] = []

    def visit(value: Any, path: str, row_id: str) -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_path = f"{path}.{key}" if path else key
                lower_key = key.lower()
                if lower_key in forbidden_exact_keys or lower_key.startswith("svg_"):
                    violations.append({"row_id": row_id, "path": child_path, "reason": "forbidden_runtime_key"})
                visit(child, child_path, row_id)
        elif isinstance(value, list):
            for index, child in enumerate(value):
                visit(child, f"{path}[{index}]", row_id)

    for row in rows:
        row_id = str(row.get("id") or "")
        extra_keys = sorted(set(row) - allowed_top_keys)
        if extra_keys:
            top_key_violations.append({"row_id": row_id, "extra_keys": extra_keys})
        visit(row, "", row_id)

    return {
        "records_checked": len(rows),
        "forbidden_exact_keys": sorted(forbidden_exact_keys),
        "allowed_top_keys": sorted(allowed_top_keys),
        "top_key_violations": top_key_violations[:50],
        "top_key_violation_count": len(top_key_violations),
        "forbidden_runtime_field_violations": violations[:50],
        "forbidden_runtime_field_violation_count": len(violations),
        "passed": not top_key_violations and not violations,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cubicasa-dir", default="datasets/cadstruct_cubicasa5k_moe")
    parser.add_argument("--cubicasa-locked-dir", default="datasets/cadstruct_cubicasa5k_moe_locked")
    parser.add_argument("--floorplancad-samples", default="datasets/external/floorplancad/samples.json")
    parser.add_argument("--floorplancad-root", default="datasets/external/floorplancad")
    parser.add_argument("--out-dir", default=str(OUT.relative_to(ROOT)))
    parser.add_argument("--audit", default="reports/vlm/public_raster_moe_supervision_v19_audit.json")
    args = parser.parse_args()

    out_dir = abs_path(args.out_dir)
    rows_by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    source_input_counts = Counter()
    skipped = Counter()

    cubicasa_locked_dir = abs_path(args.cubicasa_locked_dir)
    cubicasa_dir = cubicasa_locked_dir if cubicasa_locked_dir.exists() else abs_path(args.cubicasa_dir)
    cubicasa_splits = [("train", "train"), ("dev", "dev"), ("locked_test", "locked"), ("smoke", "smoke")] if cubicasa_locked_dir.exists() else [("train", "train"), ("dev", "dev"), ("smoke", "smoke")]
    for file_split, out_split in cubicasa_splits:
        for index, row in enumerate(load_jsonl(cubicasa_dir / f"{file_split}.jsonl")):
            source_input_counts[f"cubicasa_{file_split}"] += 1
            converted = convert_cubicasa_row(row, out_split, index)
            if converted is None:
                skipped[f"cubicasa_{file_split}"] += 1
                continue
            rows_by_split[out_split].append(converted)

    floor_path = abs_path(args.floorplancad_samples)
    floor_samples = json.loads(floor_path.read_text(encoding="utf-8")).get("samples") or []
    for index, sample in enumerate(floor_samples):
        source_input_counts["floorplancad"] += 1
        split = split_floorplancad(index, sample)
        converted = convert_floorplancad_sample(sample, index, split, args.floorplancad_root)
        if converted is None:
            skipped["floorplancad"] += 1
            continue
        rows_by_split[split].append(converted)

    all_rows: list[dict[str, Any]] = []
    for split, rows in sorted(rows_by_split.items()):
        write_jsonl(out_dir / f"{split}.jsonl", rows)
        all_rows.extend(rows)

    manifest = {
        "schema_version": "public_raster_moe_supervision_v19",
        "purpose": "Raster-only public supervision for non-SVG drawing recognition MoE experts.",
        "hard_contract": {
            "runtime_input": "raster image only",
            "annotation_path_exposed": False,
            "svg_or_cad_geometry_exposed_as_runtime_feature": False,
            "offline_svg_or_cad_labels_allowed_only_for": ["training", "dev_selection", "locked_evaluation", "audit"],
        },
        "inputs": {
            "cubicasa": args.cubicasa_dir,
            "cubicasa_locked": args.cubicasa_locked_dir if cubicasa_locked_dir.exists() else None,
            "floorplancad_samples": args.floorplancad_samples,
            "floorplancad_root": args.floorplancad_root,
        },
        "outputs": {split: str((out_dir / f"{split}.jsonl").relative_to(ROOT)) for split in sorted(rows_by_split)},
        "source_input_counts": dict(source_input_counts),
        "skipped": dict(skipped),
        "summary": summarize(all_rows),
        "split_summaries": {split: summarize(rows) for split, rows in sorted(rows_by_split.items())},
        "runtime_record_contract_audit": audit_runtime_record_contract(all_rows),
        "field_policy": {
            "removed": ["annotation_path", "source_id", "svg_* ids", "geometry.points", "polygon runtime features"],
            "kept": ["image", "image_size", "targets.family", "targets.semantic_type", "targets.bbox", "targets.text when available"],
        },
    }
    if not manifest["runtime_record_contract_audit"]["passed"]:
        raise SystemExit(json.dumps(manifest["runtime_record_contract_audit"], ensure_ascii=False, indent=2))
    write_json(out_dir / "manifest.json", manifest)
    write_json(abs_path(args.audit), manifest)
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
