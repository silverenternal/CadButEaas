#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27'
OUT = ROOT / 'datasets/symbol_tile_detector_tiny_sahi_v21_rtdetr_bbox_p061'


def reset_dir(path: Path) -> None:
    if path.is_symlink() or path.is_file():
        path.unlink()
    elif path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def link_file(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def materialize_from_list(split: str, list_path: Path, limit: int | None = None) -> int:
    out_dir = OUT / 'images' / split
    reset_dir(out_dir)
    rows = [line.strip() for line in list_path.read_text(encoding='utf-8').splitlines() if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    for row in rows:
        src = Path(row)
        if not src.is_absolute():
            src = ROOT / row
        link_file(src, out_dir / src.name)
    return len(rows)


def materialize_dir(split: str) -> int:
    out_dir = OUT / 'images' / split
    reset_dir(out_dir)
    count = 0
    for src in (SRC / 'images' / split).glob('*.jpg'):
        link_file(src, out_dir / src.name)
        count += 1
    return count


def main() -> None:
    stats = {
        'train_smoke4000_images': materialize_from_list('train', OUT / 'train_smoke4000_p061.txt'),
        'val_images': materialize_dir('val'),
        'locked_images': materialize_dir('locked'),
        'smoke_v30_images': materialize_dir('smoke_v30'),
    }
    (OUT / 'materialized_images_p061.json').write_text(json.dumps(stats, indent=2) + '\n')
    print(json.dumps(stats, indent=2))

if __name__ == '__main__':
    main()
