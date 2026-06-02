#!/usr/bin/env python3
"""Build P211 high-recall YOLO tile data focused on residual symbol FNs."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
OUT = ROOT / "datasets/symbol_recall_detector_p211_yolo"
LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]
TARGET_LABELS = {"sink", "shower", "equipment", "stair"}
TARGET_BUCKETS = {"tiny_le_64", "small_le_256", "medium_le_1024"}
WORST_ROWS = {
    "cubicasa5k_locked_00003", "cubicasa5k_locked_00009", "cubicasa5k_locked_00034", "cubicasa5k_locked_00066",
    "cubicasa5k_locked_00073", "cubicasa5k_locked_00029", "cubicasa5k_locked_00045", "cubicasa5k_locked_00040",
    "cubicasa5k_locked_00041", "cubicasa5k_locked_00065", "cubicasa5k_locked_00042", "cubicasa5k_locked_00046",
}


def load_jsonl(path: Path):
    with path.open(encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def row_prefix(tile_id: str) -> str:
    marker = "_s384_tile_"
    return tile_id.split(marker)[0] if marker in tile_id else tile_id


def target_score(row: dict[str, Any]) -> int:
    counts = row.get("target_counts") or {}
    labels = counts.get("labels") or {}
    buckets = counts.get("area_buckets") or {}
    score = 0
    for label, count in labels.items():
        if label in TARGET_LABELS:
            score += int(count) * 8
    for bucket, count in buckets.items():
        if bucket in TARGET_BUCKETS:
            score += int(count) * (8 if bucket == "tiny_le_64" else 5)
    if row_prefix(str(row.get("id") or "")) in WORST_ROWS:
        score += 10
    return score


def sample_rows_stream(path: Path, limit: int, normal_cap: int, seed: int, scan_limit: int | None = None) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    weighted: list[dict[str, Any]] = []
    normal: list[dict[str, Any]] = []
    target_limit = max(0, limit - normal_cap)
    seen = 0
    for row in load_jsonl(path):
        seen += 1
        score = target_score(row)
        if score <= 0:
            if len(normal) < normal_cap:
                normal.append(row)
            elif rng.random() < normal_cap / max(seen, 1):
                normal[rng.randrange(normal_cap)] = row
        else:
            weight = min(40, max(2, score))
            for _ in range(weight):
                if len(weighted) < target_limit:
                    weighted.append(row)
                elif rng.random() < target_limit / max(seen, 1):
                    weighted[rng.randrange(target_limit)] = row
        if scan_limit and seen >= scan_limit and len(weighted) >= target_limit and len(normal) >= normal_cap:
            break
    selected = weighted[:target_limit] + normal[:normal_cap]
    rng.shuffle(selected)
    return selected[:limit]


def crop_tile(row: dict[str, Any], image_out: Path, copy_mode: str = "crop") -> tuple[int, int]:
    tile = row.get("tile") or {}
    x1, y1, x2, y2 = [int(v) for v in tile.get("bbox") or [0, 0, 1, 1]]
    src = ROOT / str(row.get("image"))
    image_out.parent.mkdir(parents=True, exist_ok=True)
    if copy_mode == "page_symlink":
        if image_out.exists() or image_out.is_symlink():
            image_out.unlink()
        image_out.symlink_to(src.resolve())
        with Image.open(src) as opened:
            return opened.size
    with Image.open(src) as opened:
        crop = opened.convert("RGB").crop((x1, y1, x2, y2))
        crop.save(image_out, quality=95)
        return crop.size


def yolo_lines(row: dict[str, Any], w: int, h: int, copy_mode: str = "crop") -> list[str]:
    lines=[]; seen=set()
    for t in ((row.get("targets") or {}).get("boxes") or []):
        label=str(t.get("label") or "")
        if label not in LABELS: continue
        box=t.get("page_bbox") if copy_mode == "page_symlink" else t.get("bbox")
        if not isinstance(box,list) or len(box)!=4: continue
        x1,y1,x2,y2=[float(v) for v in box]
        if x2<=x1 or y2<=y1: continue
        vals=[LABELS.index(label), (x1+x2)/2/max(w,1), (y1+y2)/2/max(h,1), (x2-x1)/max(w,1), (y2-y1)/max(h,1)]
        line=" ".join([str(vals[0]), *[f"{v:.8f}" for v in vals[1:]]])
        if line not in seen:
            seen.add(line); lines.append(line)
    return lines


def export(split: str, rows: list[dict[str, Any]], out: Path, copy_mode: str) -> dict[str, Any]:
    stats=Counter(); label_counts=Counter(); bucket_counts=Counter()
    list_lines=[]
    for idx,row in enumerate(rows):
        suffix = ".png" if copy_mode == "page_symlink" else ".jpg"
        image_path=out/"images"/split/f"{row['id']}_{idx:06d}{suffix}"
        label_path=out/"labels"/split/f"{row['id']}_{idx:06d}.txt"
        w,h=crop_tile(row,image_path,copy_mode)
        lines=yolo_lines(row,w,h,copy_mode)
        label_path.parent.mkdir(parents=True, exist_ok=True)
        label_path.write_text("\n".join(lines)+("\n" if lines else ""), encoding="utf-8")
        list_lines.append(str(image_path.resolve()))
        stats['images']+=1; stats['targets']+=len(lines)
        stats['positive_images']+=1 if lines else 0
        for t in ((row.get('targets') or {}).get('boxes') or []):
            label_counts[str(t.get('label'))]+=1
            bucket_counts[str(t.get('area_bucket'))]+=1
    list_path=out/f"{split}.txt"
    list_path.write_text("\n".join(list_lines)+"\n", encoding="utf-8")
    return {"images":stats['images'],"positive_images":stats['positive_images'],"targets":stats['targets'],"list":str(list_path),"label_counts":dict(label_counts),"bucket_counts":dict(bucket_counts)}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--src', default=str(SRC)); ap.add_argument('--out', default=str(OUT))
    ap.add_argument('--train-limit', type=int, default=60000); ap.add_argument('--dev-limit', type=int, default=6000); ap.add_argument('--locked-limit', type=int, default=6000)
    ap.add_argument('--normal-cap', type=int, default=5000); ap.add_argument('--seed', type=int, default=211)
    ap.add_argument('--train-scan-limit', type=int, default=180000)
    ap.add_argument('--eval-scan-limit', type=int, default=30000)
    ap.add_argument('--copy-mode', choices=['crop','page_symlink'], default='crop')
    ap.add_argument('--overwrite', action='store_true')
    args=ap.parse_args()
    out=Path(args.out)
    if args.overwrite and out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    src_path = Path(args.src)
    train=sample_rows_stream(src_path/'train.jsonl',args.train_limit,args.normal_cap,args.seed,args.train_scan_limit)
    dev=sample_rows_stream(src_path/'dev.jsonl',args.dev_limit,max(500,args.normal_cap//5),args.seed+1,args.eval_scan_limit)
    locked=sample_rows_stream(src_path/'locked.jsonl',args.locked_limit,max(500,args.normal_cap//5),args.seed+2,args.eval_scan_limit)
    reports={
        'train':export('train',train,out,args.copy_mode),
        'dev':export('dev',dev,out,args.copy_mode),
        'locked':export('locked',locked,out,args.copy_mode),
    }
    data_yaml=out/'data.yaml'
    data_yaml.write_text('path: '+str(out.resolve())+'\ntrain: '+str((out/'train.txt').resolve())+'\nval: '+str((out/'dev.txt').resolve())+'\ntest: '+str((out/'locked.txt').resolve())+'\nnames:\n'+''.join(f'  {i}: {n}\n' for i,n in enumerate(LABELS)), encoding='utf-8')
    report={'id':'P211_symbol_recall_detector_yolo_data','source':str(SRC),'copy_mode':args.copy_mode,'target_labels':sorted(TARGET_LABELS),'target_buckets':sorted(TARGET_BUCKETS),'worst_rows':sorted(WORST_ROWS),'splits':reports,'data_yaml':str(data_yaml),'claim_boundary':'Offline supervised raster tile data. Labels are training/evaluation only, not runtime inputs.'}
    (out/'build_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'data_yaml':str(data_yaml),'splits':{k:v['images'] for k,v in reports.items()},'targets':{k:v['targets'] for k,v in reports.items()}},indent=2))
if __name__=='__main__': main()
