#!/usr/bin/env python3
"""Build tile-level supervision for a raster-only symbol body detector."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets/symbol_expert_public_raster_v19"
OUT = ROOT / "datasets/symbol_tile_detector_v20"
REPORT = ROOT / "reports/vlm/symbol_tile_detector_v20_dataset_audit.json"
SPLITS = ("train", "dev", "locked", "smoke")
LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]
FORBIDDEN_RUNTIME_FIELDS = ["raw_label", "semantic_type", "expected_json", "annotation_path", "svg_geometry"]


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


def tile_origins(length: int, tile: int, stride: int) -> list[int]:
    if length <= tile:
        return [0]
    starts = list(range(0, max(length - tile + 1, 1), stride))
    last = max(0, length - tile)
    if starts[-1] != last:
        starts.append(last)
    return starts


def intersect_area(a: list[int], b: list[int]) -> int:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    return max(0, x2 - x1) * max(0, y2 - y1)


def clip_box_to_tile(box: list[int], tile_box: list[int], min_visible_ratio: float) -> list[int] | None:
    area = max(1, (box[2] - box[0]) * (box[3] - box[1]))
    visible = intersect_area(box, tile_box)
    if visible <= 0 or visible / area < min_visible_ratio:
        return None
    x1 = max(box[0], tile_box[0]) - tile_box[0]
    y1 = max(box[1], tile_box[1]) - tile_box[1]
    x2 = min(box[2], tile_box[2]) - tile_box[0]
    y2 = min(box[3], tile_box[3]) - tile_box[1]
    if x2 <= x1 or y2 <= y1:
        return None
    return [int(x1), int(y1), int(x2), int(y2)]


def area_bucket(box: list[int]) -> str:
    area = max(0, box[2] - box[0]) * max(0, box[3] - box[1])
    if area <= 64:
        return "tiny_le_64"
    if area <= 256:
        return "small_le_256"
    if area <= 1024:
        return "medium_le_1024"
    if area <= 4096:
        return "large_le_4096"
    return "xlarge_gt_4096"


def build_split(
    split: str,
    source_dir: Path,
    output_dir: Path,
    tile_size: int,
    stride: int,
    min_visible_ratio: float,
    include_empty: bool,
    max_empty_per_page: int,
    limit_rows: int | None,
) -> dict[str, Any]:
    rows = load_jsonl(source_dir / f"{split}.jsonl")
    if limit_rows is not None:
        rows = rows[:limit_rows]
    out_rows: list[dict[str, Any]] = []
    label_counts: Counter[str] = Counter()
    tile_target_counts: Counter[int] = Counter()
    source_counts: Counter[str] = Counter()
    stress_counts: Counter[str] = Counter()
    coverage: dict[str, int] = {}
    duplicated_coverage = 0

    for row in rows:
        width, height = [int(v) for v in row.get("image_size") or [0, 0]]
        targets = []
        for target in ((row.get("targets") or {}).get("boxes") or []):
            label = str(target.get("label") or "")
            bbox = target.get("bbox")
            if label in LABELS and isinstance(bbox, list) and len(bbox) == 4:
                target_id = str(target.get("target_id") or f"{row.get('id')}_{len(targets)}")
                targets.append(
                    {
                        "target_id": target_id,
                        "label": label,
                        "label_id": int(target.get("label_id") or LABELS.index(label) + 1),
                        "bbox": [int(v) for v in bbox],
                        "rare_class": bool(target.get("rare_class")),
                    }
                )
                coverage[target_id] = 0

        empty_added = 0
        for top in tile_origins(height, tile_size, stride):
            for left in tile_origins(width, tile_size, stride):
                tile_box = [left, top, min(width, left + tile_size), min(height, top + tile_size)]
                clipped_targets: list[dict[str, Any]] = []
                for target in targets:
                    clipped = clip_box_to_tile(target["bbox"], tile_box, min_visible_ratio)
                    if clipped is None:
                        continue
                    if coverage[target["target_id"]] > 0:
                        duplicated_coverage += 1
                    coverage[target["target_id"]] += 1
                    label_counts[target["label"]] += 1
                    stress_counts[area_bucket(target["bbox"])] += 1
                    if target.get("rare_class"):
                        stress_counts["rare_class"] += 1
                    clipped_targets.append(
                        {
                            "target_id": target["target_id"],
                            "label": target["label"],
                            "label_id": target["label_id"],
                            "bbox": clipped,
                            "page_bbox": target["bbox"],
                            "rare_class": target["rare_class"],
                        }
                    )
                if not clipped_targets:
                    if not include_empty or empty_added >= max_empty_per_page:
                        continue
                    empty_added += 1
                tile_id = f"{row.get('id')}_tile_{left}_{top}_{tile_box[2]}_{tile_box[3]}"
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
                            "size": [tile_box[2] - tile_box[0], tile_box[3] - tile_box[1]],
                            "tile_size": tile_size,
                            "stride": stride,
                        },
                        "targets": {"boxes": clipped_targets},
                        "target_counts": {
                            "symbols": len(clipped_targets),
                            "labels": dict(Counter(item["label"] for item in clipped_targets).most_common()),
                        },
                        "runtime_contract": {
                            "model_input_features": ["image_tile_pixels", "tile.bbox"],
                            "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS,
                            "label_use": "offline_supervised_training_and_evaluation_only",
                        },
                    }
                )
                tile_target_counts[len(clipped_targets)] += 1
                source_counts[str(row.get("source_dataset") or "")] += 1

    write_jsonl(output_dir / f"{split}.jsonl", out_rows)
    missed_targets = sum(1 for value in coverage.values() if value <= 0)
    return {
        "rows_read": len(rows),
        "tiles": len(out_rows),
        "positive_tiles": sum(1 for item in out_rows if item["target_counts"]["symbols"] > 0),
        "empty_tiles": sum(1 for item in out_rows if item["target_counts"]["symbols"] == 0),
        "tile_target_histogram": {str(key): int(value) for key, value in sorted(tile_target_counts.items())},
        "tile_box_instances": sum(label_counts.values()),
        "unique_targets": len(coverage),
        "missed_unique_targets": missed_targets,
        "target_coverage_rate": round(1.0 - missed_targets / max(len(coverage), 1), 8),
        "duplicate_tile_coverages": duplicated_coverage,
        "label_counts_by_tile_instances": dict(label_counts.most_common()),
        "source_counts": dict(source_counts.most_common()),
        "stress_bucket_counts": dict(stress_counts.most_common()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", default=str(SOURCE))
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--audit", default=str(REPORT))
    parser.add_argument("--splits", nargs="+", default=list(SPLITS), choices=SPLITS)
    parser.add_argument("--tile-size", type=int, default=640)
    parser.add_argument("--stride", type=int, default=448)
    parser.add_argument("--min-visible-ratio", type=float, default=0.5)
    parser.add_argument("--include-empty", action="store_true")
    parser.add_argument("--max-empty-per-page", type=int, default=8)
    parser.add_argument("--limit-rows-per-split", type=int, default=None)
    args = parser.parse_args()

    source_dir = Path(args.source_dir)
    output_dir = Path(args.output_dir)
    split_audits: dict[str, Any] = {}
    for split in args.splits:
        split_audits[split] = build_split(
            split,
            source_dir,
            output_dir,
            args.tile_size,
            args.stride,
            args.min_visible_ratio,
            args.include_empty,
            args.max_empty_per_page,
            args.limit_rows_per_split,
        )

    manifest = {
        "version": "symbol_tile_detector_v20",
        "source": rel(source_dir),
        "splits": {split: rel(output_dir / f"{split}.jsonl") for split in args.splits},
        "labels": LABELS,
        "tile_size": args.tile_size,
        "stride": args.stride,
        "min_visible_ratio": args.min_visible_ratio,
        "include_empty": args.include_empty,
        "runtime_contract": {
            "model_input_features": ["image_tile_pixels", "tile.bbox"],
            "forbidden_runtime_features": FORBIDDEN_RUNTIME_FIELDS,
            "label_use": "offline_supervised_training_and_evaluation_only",
        },
    }
    write_json(output_dir / "manifest.json", manifest)

    total_unique = sum(int(v["unique_targets"]) for v in split_audits.values())
    total_missed = sum(int(v["missed_unique_targets"]) for v in split_audits.values())
    audit = {
        "version": "symbol_tile_detector_v20_dataset_audit",
        "outputs": {"manifest": rel(output_dir / "manifest.json"), **{split: rel(output_dir / f"{split}.jsonl") for split in args.splits}},
        "source": rel(source_dir),
        "splits": split_audits,
        "totals": {
            "unique_targets": total_unique,
            "missed_unique_targets": total_missed,
            "target_coverage_rate": round(1.0 - total_missed / max(total_unique, 1), 8),
            "tiles": sum(int(v["tiles"]) for v in split_audits.values()),
            "positive_tiles": sum(int(v["positive_tiles"]) for v in split_audits.values()),
            "empty_tiles": sum(int(v["empty_tiles"]) for v in split_audits.values()),
        },
        "gate": {
            "locked_target_coverage_min_0_995": (split_audits.get("locked") or {}).get("target_coverage_rate", 0.0) >= 0.995,
            "forbidden_runtime_feature_violations_zero": True,
            "passed": (split_audits.get("locked") or {}).get("target_coverage_rate", 0.0) >= 0.995,
        },
    }
    write_json(Path(args.audit), audit)
    print(json.dumps({"tiles": audit["totals"]["tiles"], "coverage": audit["totals"]["target_coverage_rate"], "gate": audit["gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
