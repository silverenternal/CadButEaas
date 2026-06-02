#!/usr/bin/env python3
"""Build a frontier detector training/eval view for symbol detection and suppression."""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from train_symbol_tile_detector_v20 import rel, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]


def source_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def copy_or_link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        dst.symlink_to(src.resolve())
    except OSError:
        shutil.copy2(src, dst)


def build_smoke_split(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    if limit <= 0 or len(rows) <= limit:
        return list(rows)
    selected: list[dict[str, Any]] = []
    by_page: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_page[str(row.get("page_id") or row.get("row_id") or row.get("id") or "")].append(row)
    for page_id in sorted(by_page):
        if len(selected) >= limit:
            break
        selected.append(by_page[page_id][0])
    if len(selected) < limit:
        for row in rows:
            if len(selected) >= limit:
                break
            if row not in selected:
                selected.append(row)
    return selected[:limit]


def infer_source(rows: list[dict[str, Any]]) -> str:
    if not rows:
        return "unknown"
    for row in rows:
        path = str(row.get("image") or row.get("filepath") or "")
        if "floorplancad" in path.lower():
            return "floorplancad"
        if "cubicasa" in path.lower():
            return "cubicasa"
    return "unknown"


def build_manifest(output: Path, source: str, split_counts: dict[str, int], files: dict[str, Any]) -> dict[str, Any]:
    return {
        "version": "symbol_detector_frontier_view_v47",
        "output": rel(output),
        "source": source,
        "split_counts": split_counts,
        "files": files,
        "source_integrity": {
            "runtime_input": "raster image only",
            "offline_supervision": "bbox/type supervision from existing converted datasets only",
            "svg_or_cad_geometry_at_runtime": False,
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--floorplancad", default="datasets/floorplancad_symbol_pretrain_v16/train.jsonl")
    parser.add_argument("--cubi-train", default="datasets/cadstruct_symbols_v1/train.jsonl")
    parser.add_argument("--cubi-dev", default="datasets/cadstruct_symbols_v1/dev.jsonl")
    parser.add_argument("--cubi-locked", default="datasets/cadstruct_symbols_v1/locked.jsonl")
    parser.add_argument("--yolo-dir", default="datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27")
    parser.add_argument("--output-dir", default="datasets/symbol_detector_frontier_v47")
    parser.add_argument("--smoke-limit", type=int, default=500)
    parser.add_argument("--smoke-output", default="datasets/symbol_detector_frontier_v47_smoke")
    parser.add_argument("--manifest-output", default="datasets/symbol_detector_frontier_v47/manifest.json")
    parser.add_argument("--registry-output", default="configs/vlm/symbol_detector_frontier_view_v47.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = source_path(args.output_dir)
    smoke_out = source_path(args.smoke_output)
    out_dir.mkdir(parents=True, exist_ok=True)
    smoke_out.mkdir(parents=True, exist_ok=True)

    floor_rows = load_jsonl(source_path(args.floorplancad))
    cubi_train = load_jsonl(source_path(args.cubi_train))
    cubi_dev = load_jsonl(source_path(args.cubi_dev))
    cubi_locked = load_jsonl(source_path(args.cubi_locked))
    source = infer_source(floor_rows + cubi_train + cubi_dev + cubi_locked)

    split_counts = {
        "floorplancad_train": len(floor_rows),
        "cubi_train": len(cubi_train),
        "cubi_dev": len(cubi_dev),
        "cubi_locked": len(cubi_locked),
    }
    smoke_rows = build_smoke_split(cubi_train + cubi_dev + cubi_locked, args.smoke_limit)

    def dump_rows(path: Path, rows: list[dict[str, Any]]) -> None:
        write_jsonl(path, rows)

    dump_rows(out_dir / "floorplancad_train.jsonl", floor_rows)
    dump_rows(out_dir / "cubi_train.jsonl", cubi_train)
    dump_rows(out_dir / "cubi_dev.jsonl", cubi_dev)
    dump_rows(out_dir / "cubi_locked.jsonl", cubi_locked)
    dump_rows(smoke_out / "smoke.jsonl", smoke_rows)

    files = {
        "floorplancad_train": rel(out_dir / "floorplancad_train.jsonl"),
        "cubi_train": rel(out_dir / "cubi_train.jsonl"),
        "cubi_dev": rel(out_dir / "cubi_dev.jsonl"),
        "cubi_locked": rel(out_dir / "cubi_locked.jsonl"),
        "smoke": rel(smoke_out / "smoke.jsonl"),
    }
    manifest = build_manifest(out_dir, source, split_counts, files)
    write_json(source_path(args.manifest_output), manifest)
    registry = {
        "version": "symbol_detector_frontier_view_v47",
        "output_dir": rel(out_dir),
        "smoke_output": rel(smoke_out),
        "datasets": {
            "FloorPlanCAD": rel(source_path(args.floorplancad)),
            "CubiCasa_train": rel(source_path(args.cubi_train)),
            "CubiCasa_dev": rel(source_path(args.cubi_dev)),
            "CubiCasa_locked": rel(source_path(args.cubi_locked)),
        },
        "split_counts": split_counts,
        "smoke_limit": args.smoke_limit,
        "source_integrity": manifest["source_integrity"],
    }
    write_json(source_path(args.registry_output), registry)
    print(json.dumps({"manifest": manifest, "registry": registry}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
