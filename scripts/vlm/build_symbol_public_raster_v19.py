#!/usr/bin/env python3
"""Build a raster-only symbol expert view from public MoE supervision."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets/public_raster_moe_supervision_v19"
OUT = ROOT / "datasets/symbol_expert_public_raster_v19"
REPORT = ROOT / "reports/vlm"
SPLITS = ("train", "dev", "locked", "smoke")
LABELS = ("appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table")
LABEL_TO_ID = {label: index + 1 for index, label in enumerate(LABELS)}
RARE_LABELS = {"bathtub", "table", "column"}


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


def area(box: list[int]) -> int:
    return max(0, box[2] - box[0]) * max(0, box[3] - box[1])


def target_from_public(item: dict[str, Any], width: int, height: int) -> dict[str, Any] | None:
    label = str(item.get("semantic_type") or "").strip()
    if label not in LABEL_TO_ID:
        return None
    box = norm_bbox(item.get("bbox"), width, height)
    if box is None:
        return None
    return {
        "target_id": str(item.get("target_id") or ""),
        "label": label,
        "label_id": LABEL_TO_ID[label],
        "bbox": box,
        "area": area(box),
        "label_source": "offline_public_annotation_converted_to_raster_target",
        "raw_label": str(item.get("raw_label") or label),
        "rare_class": label in RARE_LABELS,
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
        raw = set(existing.get("raw_labels") or [existing.get("raw_label")])
        raw.add(str(item.get("raw_label") or item["label"]))
        existing["raw_labels"] = sorted(str(v) for v in raw if v)
        ids = existing.setdefault("merged_target_ids", [existing.get("target_id")])
        ids.append(item.get("target_id"))
    return list(merged.values()), duplicates


def stress_buckets(targets: list[dict[str, Any]], width: int, height: int) -> list[str]:
    buckets: set[str] = set()
    counts = Counter(item["label"] for item in targets)
    if len(targets) >= 80:
        buckets.add("dense_symbols")
    if sum(1 for item in targets if item["area"] <= 64) >= 5:
        buckets.add("many_tiny_symbols")
    if any(counts.get(label, 0) > 0 for label in RARE_LABELS):
        buckets.add("rare_symbol_class_present")
    if width * height >= 8_000_000:
        buckets.add("large_raster")
    if not buckets:
        buckets.add("standard")
    return sorted(buckets)


def convert_row(row: dict[str, Any]) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    width, height = [int(v) for v in row.get("image_size") or [0, 0]]
    raw_targets = []
    for item in ((row.get("targets") or {}).get("symbol") or []):
        target = target_from_public(item, width, height)
        if target is not None:
            raw_targets.append(target)
    targets, duplicate_count = dedupe_targets(raw_targets)
    if not targets:
        return None, {"skipped": True, "skip_reason": "no_symbol_targets", "source_dataset": row.get("source_dataset")}
    label_counts = Counter(item["label"] for item in targets)
    converted = {
        "id": row.get("id"),
        "source_key": row.get("source_row_ref") or row.get("id"),
        "source_dataset": row.get("source_dataset"),
        "split": row.get("split"),
        "image": row.get("image"),
        "image_size": [width, height],
        "targets": {
            "boxes": targets,
            "symbol_mask": None,
            "type_mask": None,
        },
        "target_counts": {
            "total": len(targets),
            **{label: int(label_counts.get(label, 0)) for label in LABELS},
        },
        "stress_buckets": stress_buckets(targets, width, height),
        "dedupe": {"duplicate_targets_merged": duplicate_count},
        "runtime_contract": {
            "input_features_allowed": ["image", "image_size"],
            "forbidden_runtime_features": ["raw_label", "semantic_type", "expected_json", "annotation_path", "svg_geometry"],
            "label_use": "offline_supervised_training_and_evaluation_only",
        },
    }
    return converted, {"skipped": False, "duplicate_targets_merged": duplicate_count}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(SOURCE))
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--audit", default=str(REPORT / "symbol_public_raster_v19_dataset_audit.json"))
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    all_counts: dict[str, Any] = {}
    all_rows: dict[str, int] = {}
    all_labels: dict[str, Counter[str]] = {}
    skip_reasons: Counter[str] = Counter()
    duplicate_targets = 0
    stress_counts: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()

    for split in SPLITS:
        converted_rows: list[dict[str, Any]] = []
        labels: Counter[str] = Counter()
        for row in load_jsonl(source_dir / f"{split}.jsonl"):
            converted, audit = convert_row(row)
            if converted is None:
                skip_reasons[str(audit.get("skip_reason") or "unknown")] += 1
                continue
            converted_rows.append(converted)
            source_counts[str(converted.get("source_dataset") or "")] += 1
            duplicate_targets += int(audit.get("duplicate_targets_merged") or 0)
            for target in converted["targets"]["boxes"]:
                labels[str(target["label"])] += 1
            for bucket in converted.get("stress_buckets") or []:
                stress_counts[str(bucket)] += 1
        write_jsonl(output_dir / f"{split}.jsonl", converted_rows)
        all_rows[split] = len(converted_rows)
        all_labels[split] = labels
        all_counts[split] = {
            "rows": len(converted_rows),
            "targets": sum(labels.values()),
            "label_counts": dict(labels.most_common()),
        }

    manifest = {
        "version": "symbol_expert_public_raster_v19",
        "source": str(source_dir.relative_to(ROOT) if source_dir.is_relative_to(ROOT) else source_dir),
        "splits": {split: str((output_dir / f"{split}.jsonl").relative_to(ROOT)) for split in SPLITS},
        "labels": list(LABELS),
        "label_to_id": LABEL_TO_ID,
        "runtime_contract": "Raster image only. raw_label/semantic_type are target annotations and must not be used as runtime features.",
        "counts": all_counts,
    }
    write_json(output_dir / "manifest.json", manifest)

    audit = {
        "version": "symbol_public_raster_v19_dataset_audit",
        "outputs": {
            "manifest": str((output_dir / "manifest.json").relative_to(ROOT)),
            **{split: str((output_dir / f"{split}.jsonl").relative_to(ROOT)) for split in SPLITS},
        },
        "row_counts": all_rows,
        "target_counts": all_counts,
        "source_row_counts": dict(source_counts.most_common()),
        "skip_reasons": dict(skip_reasons.most_common()),
        "duplicate_targets_merged": duplicate_targets,
        "stress_bucket_counts": dict(stress_counts.most_common()),
        "rare_labels": sorted(RARE_LABELS),
        "claim_boundary": "This is a raster-only supervised training/evaluation view. Label fields are offline targets only and are forbidden as model runtime features.",
    }
    write_json(Path(args.audit), audit)
    print(json.dumps({"manifest": manifest["splits"], "row_counts": all_rows, "targets": {k: v["targets"] for k, v in all_counts.items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
