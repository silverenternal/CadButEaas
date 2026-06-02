#!/usr/bin/env python3
"""Build a YOLO-ready frontier training view for the symbol detector.

This view mixes the existing CubiCasa-derived symbol expert supervision with
FloorPlanCAD raster symbol pretrain rows. It keeps validation/locked isolated
from any train-list duplication logic and emits a smoke split for fast checks.
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from train_symbol_tile_detector_v20 import rel, write_json

ROOT = Path(__file__).resolve().parents[2]
LABELS = {
    0: "appliance",
    1: "bathtub",
    2: "column",
    3: "equipment",
    4: "generic_symbol",
    5: "shower",
    6: "sink",
    7: "stair",
    8: "table",
}
SYMBOL_SOURCE = ROOT / "datasets/symbol_expert_public_raster_v19"
FLOORPLANCAD_SOURCE = ROOT / "datasets/floorplancad_symbol_pretrain_v16"
OUT = ROOT / "datasets/symbol_detector_frontier_yolo_v47"
REPORT = ROOT / "reports/vlm/symbol_detector_frontier_yolo_v47_audit.json"


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


def box_to_yolo(box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    cx = ((x1 + x2) / 2.0) / max(width, 1)
    cy = ((y1 + y2) / 2.0) / max(height, 1)
    bw = (x2 - x1) / max(width, 1)
    bh = (y2 - y1) / max(height, 1)
    return cx, cy, bw, bh


def yolo_line(label_id: int, box: list[float], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    if width <= 0 or height <= 0:
        return ""
    left = max(0.0, min(1.0, x1 / width))
    top = max(0.0, min(1.0, y1 / height))
    right = max(0.0, min(1.0, x2 / width))
    bottom = max(0.0, min(1.0, y2 / height))
    if right <= left or bottom <= top:
        return ""
    values = [
        label_id,
        left,
        top,
        right,
        top,
        right,
        bottom,
        left,
        bottom,
    ]
    return " ".join([str(values[0]), *[f"{float(v):.8f}" for v in values[1:]]])


def floorplancad_rows(limit: int | None = None) -> list[dict[str, Any]]:
    rows = load_jsonl(FLOORPLANCAD_SOURCE / "train.jsonl")
    filtered = [row for row in rows if floorplancad_targets(row)]
    return filtered[:limit] if limit else filtered


def symbol_rows(split: str, include_floorplancad: bool = False, include_colorful: bool = True) -> list[dict[str, Any]]:
    rows = load_jsonl(SYMBOL_SOURCE / f"{split}.jsonl")
    filtered: list[dict[str, Any]] = []
    for row in rows:
        source_dataset = str(row.get("source_dataset") or "")
        image = str(row.get("image") or "")
        if source_dataset == "floorplancad" and not include_floorplancad:
            continue
        if "colorful/" in image and not include_colorful:
            continue
        filtered.append(row)
    return filtered


def image_size_from_row(row: dict[str, Any]) -> tuple[int, int]:
    size = row.get("image_size") or [0, 0]
    return int(size[0] or 0), int(size[1] or 0)


def floorplancad_targets(row: dict[str, Any]) -> list[dict[str, Any]]:
    targets: list[dict[str, Any]] = []
    for sym in ((row.get("structured") or {}).get("symbols") or []):
        label = str(sym.get("semantic_type") or sym.get("label") or "")
        label_id = next((idx for idx, name in LABELS.items() if name == label), None)
        if label_id is None:
            continue
        box = valid_box(sym.get("bbox"))
        if box is None:
            continue
        targets.append({"label_id": label_id, "label": label, "bbox": box})
    return targets


def write_image_and_labels(row: dict[str, Any], image_root: Path, label_root: Path, prefix: str) -> tuple[str, int, int]:
    src = Path(str(row.get("image") or ""))
    if not src.is_absolute():
        src = ROOT / src
    out_name = f"{prefix}_{row.get('id')}.png" if src.suffix.lower() != ".png" else f"{prefix}_{row.get('id')}.png"
    dst = image_root / out_name
    copy_or_link(src, dst)
    width, height = image_size_from_row(row)
    if width <= 0 or height <= 0:
        with Image.open(src) as im:
            width, height = im.size
    if prefix == "floorplancad":
        targets = floorplancad_targets(row)
    else:
        targets = []
        for target in ((row.get("targets") or {}).get("boxes") or []):
            label = str(target.get("label") or "")
            label_id = next((idx for idx, name in LABELS.items() if name == label), None)
            box = valid_box(target.get("bbox"))
            if label_id is None or box is None:
                continue
            targets.append({"label_id": label_id, "label": label, "bbox": box})
    lines = [yolo_line(int(target["label_id"]), target["bbox"], width, height) for target in targets]
    label_path = label_root / Path(out_name).with_suffix(".txt").name
    label_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return out_name, width, height


def export_split(split: str, rows: list[dict[str, Any]], out_dir: Path, prefix: str, limit: int | None = None) -> dict[str, Any]:
    if limit is not None:
        rows = rows[:limit]
    image_root = out_dir / "images" / split
    label_root = out_dir / "labels" / split
    image_root.mkdir(parents=True, exist_ok=True)
    label_root.mkdir(parents=True, exist_ok=True)
    stats = Counter()
    label_counts = Counter()
    for row in rows:
        out_name, width, height = write_image_and_labels(row, image_root, label_root, prefix)
        label_path = label_root / Path(out_name).with_suffix(".txt").name
        lines = [line for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
        stats["images"] += 1
        stats["targets"] += len(lines)
        if lines:
            stats["positive_images"] += 1
        else:
            stats["empty_images"] += 1
        for line in lines:
            label_id = int(line.split()[0])
            label_counts[LABELS.get(label_id, "unknown")] += 1
    return {
        "images": int(stats["images"]),
        "positive_images": int(stats["positive_images"]),
        "empty_images": int(stats["empty_images"]),
        "targets": int(stats["targets"]),
        "label_counts": dict(label_counts.most_common()),
    }


def build_train_list(source_dir: Path, split: str) -> list[str]:
    path = source_dir / "images" / split
    if not path.exists():
        return []
    # Keep paths under the exported YOLO view. If symlinks are resolved here,
    # Ultralytics derives labels from the original source image directories and
    # silently treats every training image as background.
    return [str(item.absolute()) for item in sorted(path.glob("*")) if item.suffix.lower() in {".jpg", ".png"}]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default=str(OUT))
    parser.add_argument("--report-output", default=str(REPORT))
    parser.add_argument("--symbol-train-limit", type=int, default=5000)
    parser.add_argument("--symbol-dev-limit", type=int, default=800)
    parser.add_argument("--symbol-locked-limit", type=int, default=800)
    parser.add_argument("--floorplancad-limit", type=int, default=1200)
    parser.add_argument("--include-symbol-floorplancad", action="store_true")
    parser.add_argument("--include-colorful-cubicasa", action="store_true")
    parser.add_argument("--include-floorplancad-pretrain", action="store_true")
    parser.add_argument("--smoke-limit", type=int, default=200)
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    symbol_train = symbol_rows("train", args.include_symbol_floorplancad, args.include_colorful_cubicasa)
    symbol_dev = symbol_rows("dev", False, args.include_colorful_cubicasa)
    symbol_locked = symbol_rows("locked", False, args.include_colorful_cubicasa)
    symbol_smoke = symbol_rows("smoke", False, args.include_colorful_cubicasa)
    floor_rows = floorplancad_rows(args.floorplancad_limit) if args.include_floorplancad_pretrain else []

    split_stats: dict[str, Any] = {}
    split_stats["train"] = export_split("train", symbol_train[: args.symbol_train_limit], out_dir, "cubi")
    split_stats["val"] = export_split("val", symbol_dev[: args.symbol_dev_limit], out_dir, "cubi")
    split_stats["locked"] = export_split("locked", symbol_locked[: args.symbol_locked_limit], out_dir, "cubi")
    split_stats["smoke"] = export_split("smoke", symbol_smoke[: args.smoke_limit], out_dir, "cubi")
    split_stats["floorplancad_train"] = export_split("train", floor_rows, out_dir, "floorplancad")

    train_list = build_train_list(out_dir, "train")
    train_txt = out_dir / "train.txt"
    train_txt.write_text("\n".join(train_list) + ("\n" if train_list else ""), encoding="utf-8")

    data_yaml = out_dir / "data.yaml"
    data_yaml.write_text(
        "\n".join(
            [
                f"path: {out_dir.resolve()}",
                f"train: {train_txt.resolve()}",
                "val: images/val",
                "test: images/locked",
                "names:",
                *[f"  {idx}: {name}" for idx, name in sorted(LABELS.items())],
                "",
            ]
        ),
        encoding="utf-8",
    )
    manifest = {
        "version": "symbol_detector_frontier_yolo_v47",
        "output": rel(out_dir),
        "data_yaml": rel(data_yaml),
        "train_txt": rel(train_txt),
        "split_stats": split_stats,
        "source_integrity": {
            "runtime_input": "raster pixels only",
            "offline_supervision": "CubiCasa-derived raster symbol supervision by default; optional FloorPlanCAD raster pretrain is disabled unless requested because it is black-background/domain-shifted.",
            "svg_or_cad_geometry_at_runtime": False,
        },
        "filters": {
            "include_symbol_floorplancad": bool(args.include_symbol_floorplancad),
            "include_colorful_cubicasa": bool(args.include_colorful_cubicasa),
            "include_floorplancad_pretrain": bool(args.include_floorplancad_pretrain),
        },
    }
    write_json(Path(args.report_output), manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
