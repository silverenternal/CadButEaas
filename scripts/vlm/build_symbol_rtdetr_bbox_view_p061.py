#!/usr/bin/env python3
"""P0-61: build bbox-only detector view from YOLO segmentation rect labels."""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / 'datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27'
OUT = ROOT / 'datasets/symbol_tile_detector_tiny_sahi_v21_rtdetr_bbox_p061'
NAMES = ['appliance','bathtub','column','equipment','generic_symbol','shower','sink','stair','table']


def convert_line(line: str) -> str | None:
    parts = line.strip().split()
    if not parts:
        return None
    cls = parts[0]
    nums = [float(x) for x in parts[1:]]
    if len(nums) == 4:
        x, y, w, h = nums
    elif len(nums) >= 8 and len(nums) % 2 == 0:
        xs = nums[0::2]
        ys = nums[1::2]
        x1, x2 = max(0.0, min(xs)), min(1.0, max(xs))
        y1, y2 = max(0.0, min(ys)), min(1.0, max(ys))
        w, h = max(0.0, x2 - x1), max(0.0, y2 - y1)
        if w <= 0 or h <= 0:
            return None
        x, y = x1 + w / 2.0, y1 + h / 2.0
    else:
        return None
    return f"{cls} {x:.8f} {y:.8f} {w:.8f} {h:.8f}"


def convert_split(split: str) -> dict[str, int]:
    src_dir = SRC / 'labels' / split
    out_dir = OUT / 'labels' / split
    out_dir.mkdir(parents=True, exist_ok=True)
    stats = {'files': 0, 'objects': 0, 'empty_files': 0}
    if not src_dir.exists():
        return stats
    for src in src_dir.glob('*.txt'):
        rows = []
        for raw in src.read_text(encoding='utf-8').splitlines():
            converted = convert_line(raw)
            if converted is not None:
                rows.append(converted)
        (out_dir / src.name).write_text('\n'.join(rows) + ('\n' if rows else ''), encoding='utf-8')
        stats['files'] += 1
        stats['objects'] += len(rows)
        if not rows:
            stats['empty_files'] += 1
    return stats


def link_dir(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        return
    dst.symlink_to(src, target_is_directory=True)


def remap_list(src_txt: Path, out_txt: Path, split: str, limit: int | None = None) -> int:
    rows = [line.strip() for line in src_txt.read_text(encoding='utf-8').splitlines() if line.strip()]
    if limit is not None:
        rows = rows[:limit]
    mapped = [str((OUT / 'images' / split / Path(row).name).resolve()) for row in rows]
    out_txt.write_text('\n'.join(mapped) + ('\n' if mapped else ''), encoding='utf-8')
    return len(mapped)


def write_yaml(path: Path, train: Path) -> None:
    lines = [
        f"path: {OUT}",
        f"train: {train}",
        "val: images/val",
        "test: images/locked",
        "names:",
    ]
    for i, name in enumerate(NAMES):
        lines.append(f"  {i}: {name}")
    path.write_text('\n'.join(lines) + '\n', encoding='utf-8')


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for split in ['train', 'val', 'locked', 'smoke_v30']:
        link_dir(SRC / 'images' / split, OUT / 'images' / split)
    stats = {split: convert_split(split) for split in ['train', 'val', 'locked', 'smoke_v30']}
    full_train = OUT / 'train_full_p061.txt'
    smoke_train = OUT / 'train_smoke4000_p061.txt'
    full_n = remap_list(SRC / 'train_recall_v27.txt', full_train, 'train')
    smoke_n = remap_list(SRC / 'train_recall_v27.txt', smoke_train, 'train', limit=4000)
    write_yaml(OUT / 'data_full_p061.yaml', full_train)
    write_yaml(OUT / 'data_smoke4000_p061.yaml', smoke_train)
    manifest = {
        'version': 'symbol_rtdetr_bbox_view_p061',
        'source': str(SRC.relative_to(ROOT)),
        'output': str(OUT.relative_to(ROOT)),
        'label_conversion': 'normalized polygon or xywh labels converted to YOLO bbox xywh labels; class ids unchanged',
        'stats': stats,
        'train_full_rows': full_n,
        'train_smoke4000_rows': smoke_n,
        'data_full': str((OUT / 'data_full_p061.yaml').relative_to(ROOT)),
        'data_smoke4000': str((OUT / 'data_smoke4000_p061.yaml').relative_to(ROOT)),
        'source_integrity': 'raster images only at runtime; segmentation labels converted offline for detector training/eval only',
    }
    (OUT / 'manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
