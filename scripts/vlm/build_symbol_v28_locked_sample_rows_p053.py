#!/usr/bin/env python3
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from eval_symbol_yolo_tile_detector_v22 import filter_rows_with_exported_images, sample_tiles_area_aware
from train_symbol_tile_detector_v20 import load_jsonl

cfg = json.loads(Path('reports/vlm/symbol_yolov8s_seg_rect_v28_page_eval.json').read_text())['config']
rows = load_jsonl(Path(cfg['data']) / f"{cfg['split']}.jsonl")
exported = filter_rows_with_exported_images(rows, cfg['split'], Path(cfg['yolo_dir']))
sampled = sample_tiles_area_aware(
    exported,
    int(cfg['limit_tiles']),
    int(cfg['seed']) + 2,
    float(cfg['positive_ratio']),
    float(cfg['small_positive_ratio']),
)
out = Path('reports/vlm/symbol_yolov8s_seg_rect_v28_locked_sample_rows_p053.jsonl')
out.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in sampled), encoding='utf-8')
page_golds = defaultdict(dict)
for row in sampled:
    row_id = str(row.get('row_id'))
    for gold in ((row.get('targets') or {}).get('boxes') or []):
        target_id = str(gold.get('target_id') or f"{row_id}_{len(page_golds[row_id])}")
        page_golds[row_id][target_id] = gold
print(json.dumps({'exported': len(exported), 'sampled': len(sampled), 'pages': len(page_golds), 'gold': sum(len(x) for x in page_golds.values()), 'out': str(out)}, indent=2))
