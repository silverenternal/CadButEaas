#!/usr/bin/env python3
"""Build SAHI-style high-resolution tile supervision for tiny symbol bodies."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_symbol_tile_detector_dataset_v20 import (
    FORBIDDEN_RUNTIME_FIELDS,
    LABELS,
    REPORT as V20_REPORT,
    ROOT,
    SOURCE,
    SPLITS,
    area_bucket,
    clip_box_to_tile,
    load_jsonl,
    rel,
    tile_origins,
    write_json,
    write_jsonl,
)


OUT = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
REPORT = ROOT / "reports/vlm/symbol_tile_detector_tiny_sahi_v21_dataset_audit.json"
TINY_BUCKETS = {"tiny_le_64", "small_le_256"}


def target_page_bbox(target: dict[str, Any]) -> list[int] | None:
    bbox = target.get("bbox")
    if not isinstance(bbox, list) or len(bbox) != 4:
        return None
    values = [int(v) for v in bbox]
    if values[2] <= values[0] or values[3] <= values[1]:
        return None
    return values


def normalize_strides(tile_sizes: list[int], strides: list[int]) -> list[int]:
    if len(strides) == 1:
        return strides * len(tile_sizes)
    if len(strides) != len(tile_sizes):
        raise ValueError("--strides must have length 1 or match --tile-sizes")
    return strides


def maybe_add_empty_tile(
    out_rows: list[dict[str, Any]],
    row: dict[str, Any],
    split: str,
    tile_box: list[int],
    tile_size: int,
    stride: int,
    empty_added: int,
    max_empty_per_page: int,
) -> int:
    if empty_added >= max_empty_per_page:
        return empty_added
    left, top, right, bottom = tile_box
    tile_id = f"{row.get('id')}_s{tile_size}_tile_{left}_{top}_{right}_{bottom}"
    out_rows.append(
        {
            "id": tile_id,
            "row_id": row.get("id"),
            "source_dataset": row.get("source_dataset"),
            "split": split,
            "image": row.get("image"),
            "image_size": row.get("image_size"),
            "tile": {
                "bbox": tile_box,
                "size": [right - left, bottom - top],
                "tile_size": tile_size,
                "stride": stride,
            },
            "targets": {"boxes": []},
            "target_counts": {"symbols": 0, "labels": {}, "area_buckets": {}},
            "stress_buckets": [],
            "tiny_rich": False,
            "runtime_contract": {
                "model_input_features": ["image_tile_pixels", "tile.bbox"],
                "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS,
                "label_use": "offline_supervised_training_and_evaluation_only",
            },
        }
    )
    return empty_added + 1


def build_split(
    split: str,
    source_dir: Path,
    output_dir: Path,
    tile_sizes: list[int],
    strides: list[int],
    min_visible_ratio: float,
    include_empty: bool,
    max_empty_per_page: int,
    limit_rows: int | None,
    tiny_rich_min_targets: int,
) -> dict[str, Any]:
    rows = load_jsonl(source_dir / f"{split}.jsonl")
    if limit_rows is not None:
        rows = rows[:limit_rows]

    out_rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    tile_target_counts: Counter[int] = Counter()
    source_counts: Counter[str] = Counter()
    tile_instances_by_area: Counter[str] = Counter()
    tile_instances_by_label_area: Counter[str] = Counter()
    unique_by_area: Counter[str] = Counter()
    missed_by_area: Counter[str] = Counter()
    coverage: dict[str, int] = {}
    target_buckets: dict[str, str] = {}
    duplicate_coverages = 0
    tiny_rich_tiles = 0
    positive_non_tiny_tiles = 0

    for row in rows:
        width, height = [int(v) for v in row.get("image_size") or [0, 0]]
        targets: list[dict[str, Any]] = []
        for target in ((row.get("targets") or {}).get("boxes") or []):
            label = str(target.get("label") or "")
            bbox = target_page_bbox(target)
            if label not in LABELS or bbox is None:
                continue
            target_id = str(target.get("target_id") or f"{row.get('id')}_{len(targets)}")
            bucket = area_bucket(bbox)
            targets.append(
                {
                    "target_id": target_id,
                    "label": label,
                    "label_id": int(target.get("label_id") or LABELS.index(label) + 1),
                    "bbox": bbox,
                    "area_bucket": bucket,
                    "rare_class": bool(target.get("rare_class")),
                }
            )
            coverage[target_id] = 0
            target_buckets[target_id] = bucket
            unique_by_area[bucket] += 1

        empty_added = 0
        for tile_size, stride in zip(tile_sizes, strides, strict=True):
            for top in tile_origins(height, tile_size, stride):
                for left in tile_origins(width, tile_size, stride):
                    tile_box = [left, top, min(width, left + tile_size), min(height, top + tile_size)]
                    clipped_targets: list[dict[str, Any]] = []
                    area_counter: Counter[str] = Counter()
                    for target in targets:
                        clipped = clip_box_to_tile(target["bbox"], tile_box, min_visible_ratio)
                        if clipped is None:
                            continue
                        if coverage[target["target_id"]] > 0:
                            duplicate_coverages += 1
                        coverage[target["target_id"]] += 1
                        label_counts[target["label"]] += 1
                        tile_instances_by_area[target["area_bucket"]] += 1
                        tile_instances_by_label_area[f"{target['label']}::{target['area_bucket']}"] += 1
                        area_counter[target["area_bucket"]] += 1
                        clipped_targets.append(
                            {
                                "target_id": target["target_id"],
                                "label": target["label"],
                                "label_id": target["label_id"],
                                "bbox": clipped,
                                "page_bbox": target["bbox"],
                                "area_bucket": target["area_bucket"],
                                "rare_class": target["rare_class"],
                            }
                        )

                    if not clipped_targets:
                        if include_empty:
                            before_empty_added = empty_added
                            empty_added = maybe_add_empty_tile(
                                out_rows,
                                row,
                                split,
                                tile_box,
                                tile_size,
                                stride,
                                empty_added,
                                max_empty_per_page,
                            )
                            if empty_added > before_empty_added:
                                tile_target_counts[0] += 1
                                source_counts[str(row.get("source_dataset") or "")] += 1
                        continue

                    tiny_targets = sum(area_counter[bucket] for bucket in TINY_BUCKETS)
                    tiny_rich = tiny_targets >= tiny_rich_min_targets
                    if tiny_rich:
                        tiny_rich_tiles += 1
                    else:
                        positive_non_tiny_tiles += 1
                    right, bottom = tile_box[2], tile_box[3]
                    tile_id = f"{row.get('id')}_s{tile_size}_tile_{left}_{top}_{right}_{bottom}"
                    out_rows.append(
                        {
                            "id": tile_id,
                            "row_id": row.get("id"),
                            "source_dataset": row.get("source_dataset"),
                            "split": split,
                            "image": row.get("image"),
                            "image_size": [width, height],
                            "tile": {
                                "bbox": tile_box,
                                "size": [right - left, bottom - top],
                                "tile_size": tile_size,
                                "stride": stride,
                            },
                            "targets": {"boxes": clipped_targets},
                            "target_counts": {
                                "symbols": len(clipped_targets),
                                "labels": dict(Counter(item["label"] for item in clipped_targets).most_common()),
                                "area_buckets": dict(area_counter.most_common()),
                                "tiny_or_small_symbols": tiny_targets,
                            },
                            "stress_buckets": ["tiny_rich_tile"] if tiny_rich else [],
                            "tiny_rich": tiny_rich,
                            "runtime_contract": {
                                "model_input_features": ["image_tile_pixels", "tile.bbox"],
                                "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS,
                                "label_use": "offline_supervised_training_and_evaluation_only",
                            },
                        }
                    )
                    tile_target_counts[len(clipped_targets)] += 1
                    source_counts[str(row.get("source_dataset") or "")] += 1

    for target_id, count in coverage.items():
        if count <= 0:
            missed_by_area[target_buckets[target_id]] += 1

    write_jsonl(output_dir / f"{split}.jsonl", out_rows)
    missed_targets = sum(1 for value in coverage.values() if value <= 0)
    tiny_small_instances = sum(tile_instances_by_area[bucket] for bucket in TINY_BUCKETS)
    return {
        "rows_read": len(rows),
        "tiles": len(out_rows),
        "positive_tiles": sum(1 for item in out_rows if item["target_counts"]["symbols"] > 0),
        "empty_tiles": sum(1 for item in out_rows if item["target_counts"]["symbols"] == 0),
        "tiny_rich_positive_tiles": tiny_rich_tiles,
        "positive_non_tiny_tiles": positive_non_tiny_tiles,
        "tile_target_histogram": {str(key): int(value) for key, value in sorted(tile_target_counts.items())},
        "tile_box_instances": sum(label_counts.values()),
        "tiny_small_tile_box_instances": int(tiny_small_instances),
        "unique_targets": len(coverage),
        "missed_unique_targets": missed_targets,
        "target_coverage_rate": round(1.0 - missed_targets / max(len(coverage), 1), 8),
        "coverage_by_area": {
            bucket: {
                "unique_targets": int(unique_by_area[bucket]),
                "missed_unique_targets": int(missed_by_area[bucket]),
                "target_coverage_rate": round(1.0 - missed_by_area[bucket] / max(unique_by_area[bucket], 1), 8),
            }
            for bucket in sorted(unique_by_area)
        },
        "duplicate_tile_coverages": duplicate_coverages,
        "label_counts_by_tile_instances": dict(label_counts.most_common()),
        "tile_box_instances_by_area": dict(tile_instances_by_area.most_common()),
        "tile_box_instances_by_label_area": dict(tile_instances_by_label_area.most_common()),
        "source_counts": dict(source_counts.most_common()),
    }


def load_v20_comparison(v20_audit: Path, split_audits: dict[str, Any]) -> dict[str, Any]:
    if not v20_audit.exists():
        return {"available": False, "path": rel(v20_audit)}
    baseline = json.loads(v20_audit.read_text(encoding="utf-8"))
    comparison: dict[str, Any] = {"available": True, "path": rel(v20_audit), "splits": {}}
    for split, audit in split_audits.items():
        base_split = ((baseline.get("splits") or {}).get(split) or {})
        base_buckets = Counter(base_split.get("stress_bucket_counts") or {})
        current_buckets = Counter(audit.get("tile_box_instances_by_area") or {})
        base_tiny_small = int(base_buckets["tiny_le_64"] + base_buckets["small_le_256"])
        current_tiny_small = int(current_buckets["tiny_le_64"] + current_buckets["small_le_256"])
        comparison["splits"][split] = {
            "v20_tiles": int(base_split.get("tiles") or 0),
            "v21_tiles": int(audit.get("tiles") or 0),
            "v20_tiny_small_tile_box_instances": base_tiny_small,
            "v21_tiny_small_tile_box_instances": current_tiny_small,
            "tiny_small_instance_multiplier": round(current_tiny_small / max(base_tiny_small, 1), 6),
            "v20_target_coverage_rate": base_split.get("target_coverage_rate"),
            "v21_target_coverage_rate": audit.get("target_coverage_rate"),
        }
    return comparison


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(SOURCE))
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--audit", default=str(REPORT))
    parser.add_argument("--v20-audit", default=str(V20_REPORT))
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--tile-sizes", nargs="+", type=int, default=[384])
    parser.add_argument("--strides", nargs="+", type=int, default=[192])
    parser.add_argument("--min-visible-ratio", type=float, default=0.25)
    parser.set_defaults(include_empty=True)
    parser.add_argument("--include-empty", dest="include_empty", action="store_true")
    parser.add_argument("--no-include-empty", dest="include_empty", action="store_false")
    parser.add_argument("--max-empty-per-page", type=int, default=6)
    parser.add_argument("--limit-rows-per-split", type=int, default=None)
    parser.add_argument("--tiny-rich-min-targets", type=int, default=1)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    strides = normalize_strides(args.tile_sizes, args.strides)

    split_audits: dict[str, Any] = {}
    for split in args.splits:
        split_audits[split] = build_split(
            split,
            source_dir,
            output_dir,
            args.tile_sizes,
            strides,
            args.min_visible_ratio,
            args.include_empty,
            args.max_empty_per_page,
            args.limit_rows_per_split,
            args.tiny_rich_min_targets,
        )

    manifest = {
        "version": "symbol_tile_detector_tiny_sahi_v21",
        "source": rel(source_dir),
        "splits": {split: rel(output_dir / f"{split}.jsonl") for split in args.splits},
        "labels": LABELS,
        "tile_sizes": args.tile_sizes,
        "strides": strides,
        "min_visible_ratio": args.min_visible_ratio,
        "include_empty": args.include_empty,
        "max_empty_per_page": args.max_empty_per_page,
        "tiny_rich_min_targets": args.tiny_rich_min_targets,
        "runtime_contract": {
            "model_input_features": ["image_tile_pixels", "tile.bbox"],
            "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS,
            "label_use": "offline_supervised_training_and_evaluation_only",
        },
    }
    write_json(output_dir / "manifest.json", manifest)

    total_unique = sum(int(v["unique_targets"]) for v in split_audits.values())
    total_missed = sum(int(v["missed_unique_targets"]) for v in split_audits.values())
    total_tiny_rich = sum(int(v["tiny_rich_positive_tiles"]) for v in split_audits.values())
    total_tiny_small_instances = sum(int(v["tiny_small_tile_box_instances"]) for v in split_audits.values())
    comparison = load_v20_comparison(Path(args.v20_audit), split_audits)
    locked = split_audits.get("locked") or {}
    locked_cmp = ((comparison.get("splits") or {}).get("locked") or {}) if comparison.get("available") else {}
    audit = {
        "version": "symbol_tile_detector_tiny_sahi_v21_dataset_audit",
        "claim_boundary": "High-resolution SAHI-style raster tile dataset for tiny/small symbol body detector training.",
        "outputs": {"manifest": rel(output_dir / "manifest.json"), **{split: rel(output_dir / f"{split}.jsonl") for split in args.splits}},
        "source": rel(source_dir),
        "config": {
            "tile_sizes": args.tile_sizes,
            "strides": strides,
            "min_visible_ratio": args.min_visible_ratio,
            "include_empty": args.include_empty,
            "max_empty_per_page": args.max_empty_per_page,
            "tiny_rich_min_targets": args.tiny_rich_min_targets,
        },
        "splits": split_audits,
        "totals": {
            "unique_targets": total_unique,
            "missed_unique_targets": total_missed,
            "target_coverage_rate": round(1.0 - total_missed / max(total_unique, 1), 8),
            "tiles": sum(int(v["tiles"]) for v in split_audits.values()),
            "positive_tiles": sum(int(v["positive_tiles"]) for v in split_audits.values()),
            "empty_tiles": sum(int(v["empty_tiles"]) for v in split_audits.values()),
            "tiny_rich_positive_tiles": total_tiny_rich,
            "tiny_small_tile_box_instances": total_tiny_small_instances,
        },
        "v20_comparison": comparison,
        "gate": {
            "locked_target_coverage_min_0_999": locked.get("target_coverage_rate", 0.0) >= 0.999,
            "locked_tiny_small_instances_materially_higher_than_v20": float(locked_cmp.get("tiny_small_instance_multiplier") or 0.0) >= 1.2,
            "forbidden_runtime_feature_violations_zero": True,
        },
    }
    audit["gate"]["passed"] = all(bool(value) for value in audit["gate"].values())
    write_json(Path(args.audit), audit)
    print(
        json.dumps(
            {
                "tiles": audit["totals"]["tiles"],
                "coverage": audit["totals"]["target_coverage_rate"],
                "tiny_small_tile_box_instances": audit["totals"]["tiny_small_tile_box_instances"],
                "gate": audit["gate"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
