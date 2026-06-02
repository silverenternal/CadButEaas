#!/usr/bin/env python3
"""Build cached multi-scale raster crops for the symbol type expert.

The source labels are used only as supervised targets. Runtime features are the
cached raster crops plus geometry derived from the candidate box.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from PIL import Image, ImageOps


Image.MAX_IMAGE_PIXELS = None

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets/symbol_expert_public_raster_v19"
OUT = ROOT / "datasets/symbol_crop_context_cache_v20"
REPORT = ROOT / "reports/vlm/symbol_crop_context_cache_v20_audit.json"
SPLITS = ("train", "dev", "locked", "smoke")
LABELS = ("appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table")
FORBIDDEN_RUNTIME_FIELDS = ("raw_label", "semantic_type", "expected_json", "annotation_path", "svg_geometry")


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


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)


def safe_name(value: Any) -> str:
    text = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(value or "item")).strip("_")
    return text[:96] or "item"


def box_area(box: list[int]) -> int:
    return max(0, int(box[2]) - int(box[0])) * max(0, int(box[3]) - int(box[1]))


def expand_box(box: list[int], image_size: tuple[int, int], *, scale: float = 1.0, pad: int = 0) -> list[int]:
    width, height = image_size
    x1, y1, x2, y2 = [float(v) for v in box]
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    nw = bw * scale + 2 * pad
    nh = bh * scale + 2 * pad
    left = max(0, int(math.floor(cx - nw / 2.0)))
    top = max(0, int(math.floor(cy - nh / 2.0)))
    right = min(width, int(math.ceil(cx + nw / 2.0)))
    bottom = min(height, int(math.ceil(cy + nh / 2.0)))
    if right <= left:
        right = min(width, left + 1)
    if bottom <= top:
        bottom = min(height, top + 1)
    return [left, top, right, bottom]


def crop_and_save(image: Image.Image, box: list[int], output: Path, size: int) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    crop = image.crop(tuple(box)).convert("RGB")
    crop = ImageOps.autocontrast(crop)
    crop = crop.resize((size, size), Image.Resampling.BICUBIC)
    crop.save(output)
    return {
        "path": rel(output),
        "box": [int(v) for v in box],
        "size": [size, size],
        "source_size": [int(box[2] - box[0]), int(box[3] - box[1])],
    }


def area_bucket(area: int) -> str:
    if area <= 64:
        return "tiny_le_64"
    if area <= 256:
        return "small_le_256"
    if area <= 1024:
        return "medium_le_1024"
    if area <= 4096:
        return "large_le_4096"
    return "xlarge_gt_4096"


def geometry_features(box: list[int], image_size: tuple[int, int]) -> dict[str, Any]:
    width, height = image_size
    x1, y1, x2, y2 = [int(v) for v in box]
    bw = max(1, x2 - x1)
    bh = max(1, y2 - y1)
    area = bw * bh
    return {
        "bbox_norm": [
            round(x1 / max(width, 1), 8),
            round(y1 / max(height, 1), 8),
            round(x2 / max(width, 1), 8),
            round(y2 / max(height, 1), 8),
        ],
        "center_norm": [
            round((x1 + x2) / 2.0 / max(width, 1), 8),
            round((y1 + y2) / 2.0 / max(height, 1), 8),
        ],
        "width_norm": round(bw / max(width, 1), 8),
        "height_norm": round(bh / max(height, 1), 8),
        "area_norm": round(area / max(width * height, 1), 10),
        "aspect_log": round(math.log(bw / max(bh, 1)), 8),
        "area_bucket": area_bucket(area),
    }


def stress_buckets(target: dict[str, Any], image_size: tuple[int, int]) -> list[str]:
    width, height = image_size
    box = [int(v) for v in target.get("bbox") or [0, 0, 0, 0]]
    area = box_area(box)
    buckets: set[str] = {area_bucket(area)}
    if target.get("rare_class"):
        buckets.add("rare_class")
    if width * height >= 8_000_000:
        buckets.add("large_page")
    if min(max(1, box[2] - box[0]), max(1, box[3] - box[1])) <= 8:
        buckets.add("thin_or_micro_symbol")
    return sorted(buckets)


def source_image_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def build_target_record(
    *,
    row: dict[str, Any],
    target: dict[str, Any],
    index: int,
    image: Image.Image,
    output_dir: Path,
    crop_size: int,
    tight_pad: int,
    padded_scale: float,
    context_scale: float,
) -> dict[str, Any]:
    image_size = image.size
    bbox = [int(v) for v in target["bbox"]]
    row_id = safe_name(row.get("id"))
    target_id = safe_name(target.get("target_id") or f"target_{index}")
    digest = hashlib.sha1(f"{row.get('id')}::{target.get('target_id')}::{index}::{bbox}".encode("utf-8")).hexdigest()[:10]
    stem = f"{target_id}_{digest}"
    base = output_dir / "crops" / str(row.get("split")) / row_id

    tight_box = expand_box(bbox, image_size, scale=1.0, pad=tight_pad)
    padded_box = expand_box(bbox, image_size, scale=padded_scale, pad=tight_pad)
    context_box = expand_box(bbox, image_size, scale=context_scale, pad=tight_pad)

    crops = {
        "tight": crop_and_save(image, tight_box, base / f"{stem}_tight.png", crop_size),
        "padded": crop_and_save(image, padded_box, base / f"{stem}_padded.png", crop_size),
        "context": crop_and_save(image, context_box, base / f"{stem}_context.png", crop_size),
    }
    label = str(target.get("label") or "")
    return {
        "id": f"{row.get('id')}::{target.get('target_id') or index}",
        "row_id": row.get("id"),
        "source_dataset": row.get("source_dataset"),
        "split": row.get("split"),
        "original_image": row.get("image"),
        "image_size": list(image_size),
        "bbox": bbox,
        "label": label,
        "label_id": int(target.get("label_id") or (LABELS.index(label) + 1 if label in LABELS else 0)),
        "rare_class": bool(target.get("rare_class")),
        "crops": crops,
        "geometry": geometry_features(bbox, image_size),
        "stress_buckets": stress_buckets(target, image_size),
        "runtime_contract": {
            "model_input_features": ["crops.tight", "crops.padded", "crops.context", "geometry"],
            "forbidden_runtime_features": list(FORBIDDEN_RUNTIME_FIELDS),
            "label_use": "offline_supervised_training_and_evaluation_only",
        },
        "audit_only": {
            "source_target_id": target.get("target_id"),
            "label_source": target.get("label_source"),
            "raw_label": target.get("raw_label"),
        },
    }


def validate_runtime_contract(record: dict[str, Any]) -> list[str]:
    violations: list[str] = []
    runtime = record.get("runtime_contract") or {}
    feature_text = json.dumps(runtime.get("model_input_features") or [], ensure_ascii=False)
    crop_text = json.dumps(record.get("crops") or {}, ensure_ascii=False)
    geometry_text = json.dumps(record.get("geometry") or {}, ensure_ascii=False)
    searchable = f"{feature_text} {crop_text} {geometry_text}"
    for field in FORBIDDEN_RUNTIME_FIELDS:
        if field in searchable:
            violations.append(field)
    return violations


def process_split(
    *,
    split: str,
    source_dir: Path,
    output_dir: Path,
    crop_size: int,
    tight_pad: int,
    padded_scale: float,
    context_scale: float,
    limit_rows: int | None,
    limit_targets: int | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows = load_jsonl(source_dir / f"{split}.jsonl")
    if limit_rows is not None:
        rows = rows[:limit_rows]

    records: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    stress_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    decode_failures: list[dict[str, str]] = []
    crop_failures = 0
    runtime_violations: Counter[str] = Counter()
    rows_with_targets = 0

    for row in rows:
        targets = list(((row.get("targets") or {}).get("boxes") or []))
        if limit_targets is not None:
            remaining = max(0, limit_targets - len(records))
            if remaining <= 0:
                break
            targets = targets[:remaining]
        if not targets:
            continue
        path = source_image_path(str(row.get("image") or ""))
        try:
            with Image.open(path) as opened:
                image = opened.convert("RGB")
        except Exception as exc:  # noqa: BLE001 - audit should keep processing.
            decode_failures.append({"row_id": str(row.get("id")), "image": str(row.get("image")), "error": repr(exc)})
            continue

        rows_with_targets += 1
        for index, target in enumerate(targets):
            label = str(target.get("label") or "")
            if label not in LABELS:
                continue
            try:
                record = build_target_record(
                    row=row,
                    target=target,
                    index=index,
                    image=image,
                    output_dir=output_dir,
                    crop_size=crop_size,
                    tight_pad=tight_pad,
                    padded_scale=padded_scale,
                    context_scale=context_scale,
                )
            except Exception:  # noqa: BLE001 - record failed crop count in audit.
                crop_failures += 1
                continue
            for violation in validate_runtime_contract(record):
                runtime_violations[violation] += 1
            records.append(record)
            label_counts[label] += 1
            source_counts[str(row.get("source_dataset") or "")] += 1
            for bucket in record.get("stress_buckets") or []:
                stress_counts[str(bucket)] += 1
            if limit_targets is not None and len(records) >= limit_targets:
                break

    write_jsonl(output_dir / f"{split}.jsonl", records)
    audit = {
        "rows_read": len(rows),
        "rows_with_targets_cached": rows_with_targets,
        "records": len(records),
        "crop_images": len(records) * 3,
        "label_counts": dict(label_counts.most_common()),
        "source_counts": dict(source_counts.most_common()),
        "stress_bucket_counts": dict(stress_counts.most_common()),
        "decode_failure_count": len(decode_failures),
        "decode_failures_sample": decode_failures[:20],
        "crop_failure_count": crop_failures,
        "forbidden_runtime_feature_violations": dict(runtime_violations.most_common()),
    }
    return records, audit


def sample_readability_checks(records: list[dict[str, Any]], max_checks: int) -> dict[str, Any]:
    checked = 0
    failures: list[dict[str, str]] = []
    for record in records[:max_checks]:
        for kind, crop in (record.get("crops") or {}).items():
            path = source_image_path(str(crop.get("path") or ""))
            try:
                with Image.open(path) as image:
                    image.verify()
            except Exception as exc:  # noqa: BLE001
                failures.append({"record_id": str(record.get("id")), "kind": str(kind), "path": str(path), "error": repr(exc)})
            checked += 1
    return {"checked_crop_images": checked, "failure_count": len(failures), "failures_sample": failures[:20]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(SOURCE))
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--audit", default=str(REPORT))
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--crop-size", type=int, default=96)
    parser.add_argument("--tight-pad", type=int, default=2)
    parser.add_argument("--padded-scale", type=float, default=1.8)
    parser.add_argument("--context-scale", type=float, default=3.5)
    parser.add_argument("--limit-rows-per-split", type=int, default=None)
    parser.add_argument("--limit-targets-per-split", type=int, default=None)
    parser.add_argument("--readability-checks", type=int, default=300)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    split_audits: dict[str, Any] = {}
    all_records_for_checks: list[dict[str, Any]] = []
    total_records = 0
    total_crop_images = 0
    all_labels: Counter[str] = Counter()
    all_sources: Counter[str] = Counter()

    for split in args.splits:
        records, audit = process_split(
            split=split,
            source_dir=source_dir,
            output_dir=output_dir,
            crop_size=args.crop_size,
            tight_pad=args.tight_pad,
            padded_scale=args.padded_scale,
            context_scale=args.context_scale,
            limit_rows=args.limit_rows_per_split,
            limit_targets=args.limit_targets_per_split,
        )
        split_audits[split] = audit
        all_records_for_checks.extend(records[: max(0, args.readability_checks - len(all_records_for_checks))])
        total_records += len(records)
        total_crop_images += len(records) * 3
        all_labels.update(audit["label_counts"])
        all_sources.update(audit["source_counts"])

    manifest = {
        "version": "symbol_crop_context_cache_v20",
        "source": rel(source_dir),
        "splits": {split: rel(output_dir / f"{split}.jsonl") for split in args.splits},
        "crop_root": rel(output_dir / "crops"),
        "labels": list(LABELS),
        "crop_views": {
            "tight": {"scale": 1.0, "pad": args.tight_pad, "size": args.crop_size},
            "padded": {"scale": args.padded_scale, "pad": args.tight_pad, "size": args.crop_size},
            "context": {"scale": args.context_scale, "pad": args.tight_pad, "size": args.crop_size},
        },
        "runtime_contract": {
            "allowed_model_inputs": ["crops.tight", "crops.padded", "crops.context", "geometry"],
            "forbidden_runtime_features": list(FORBIDDEN_RUNTIME_FIELDS),
            "label_use": "offline_supervised_training_and_evaluation_only",
        },
        "counts": {
            "records": total_records,
            "crop_images": total_crop_images,
            "label_counts": dict(all_labels.most_common()),
            "source_counts": dict(all_sources.most_common()),
        },
    }
    write_json(output_dir / "manifest.json", manifest)

    readability = sample_readability_checks(all_records_for_checks, args.readability_checks)
    forbidden_total = sum(
        sum((split_audit.get("forbidden_runtime_feature_violations") or {}).values()) for split_audit in split_audits.values()
    )
    locked_records = int((split_audits.get("locked") or {}).get("records") or 0)
    audit = {
        "version": "symbol_crop_context_cache_v20_audit",
        "outputs": {
            "manifest": rel(output_dir / "manifest.json"),
            **{split: rel(output_dir / f"{split}.jsonl") for split in args.splits},
        },
        "source": rel(source_dir),
        "splits": split_audits,
        "totals": {
            "records": total_records,
            "crop_images": total_crop_images,
            "label_counts": dict(all_labels.most_common()),
            "source_counts": dict(all_sources.most_common()),
            "forbidden_runtime_feature_violation_count": forbidden_total,
            "locked_targets_cached": locked_records,
        },
        "readability": readability,
        "gate": {
            "forbidden_runtime_feature_violations_zero": forbidden_total == 0,
            "locked_targets_cached_min_18000": locked_records >= 18_000 if "locked" in args.splits else None,
            "sample_crops_readable": readability["failure_count"] == 0,
            "passed": forbidden_total == 0 and readability["failure_count"] == 0 and ("locked" not in args.splits or locked_records >= 18_000),
        },
    }
    write_json(Path(args.audit), audit)


if __name__ == "__main__":
    main()
