#!/usr/bin/env python3
"""Build P213b residual specialist data from P212 remaining FNs."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from fuse_symbol_p206g_with_p211_p212 import LABELS, LABEL_TO_ID, area_bucket

ROOT=Path(__file__).resolve().parents[2]
AUDIT=ROOT/"reports/vlm/symbol_p212_residuals_p213b.json"
OVERLAY=ROOT/"reports/vlm/symbol_p206g_p212_specialist_precision_repair_overlay.jsonl"
OUT=ROOT/"datasets/symbol_residual_specialist_p213b_yolo"
TARGET_LABELS={"sink","stair","equipment","shower"}
TARGET_BUCKETS={"tiny_le_64","small_le_256"}


def load_overlay_rows(path: Path) -> dict[str, dict[str, Any]]:
    out={}
    with path.open(encoding='utf-8') as f:
        for line in f:
            if line.strip():
                r=json.loads(line); out[str(r.get('id') or r.get('row_id'))]=r
    return out


def crop_around(box: list[float], size: tuple[int,int], crop_size: int, jitter: int, rng: random.Random):
    w,h=size; cx=(box[0]+box[2])/2+rng.randint(-jitter,jitter); cy=(box[1]+box[3])/2+rng.randint(-jitter,jitter)
    left=max(0,min(int(round(cx-crop_size/2)),max(w-crop_size,0))); top=max(0,min(int(round(cy-crop_size/2)),max(h-crop_size,0)))
    return left,top,min(left+crop_size,w),min(top+crop_size,h)


def yolo_lines(targets: list[dict[str,Any]], crop):
    left,top,right,bottom=crop; cw=max(1,right-left); ch=max(1,bottom-top); lines=[]; seen=set()
    for t in targets:
        label=str(t.get('semantic_type') or t.get('label') or '')
        if label not in LABEL_TO_ID: continue
        x1,y1,x2,y2=[float(v) for v in t.get('bbox')]
        ix1,iy1=max(x1,left),max(y1,top); ix2,iy2=min(x2,right),min(y2,bottom)
        if ix2<=ix1 or iy2<=iy1: continue
        visible=(ix2-ix1)*(iy2-iy1)/max((x2-x1)*(y2-y1),1e-9)
        if visible<0.45: continue
        vals=[LABEL_TO_ID[label]-1, ((ix1+ix2)/2-left)/cw, ((iy1+iy2)/2-top)/ch, (ix2-ix1)/cw, (iy2-iy1)/ch]
        line=' '.join([str(vals[0]), *[f'{v:.8f}' for v in vals[1:]]])
        if line not in seen: seen.add(line); lines.append(line)
    return lines


def export(split, items, rows, out, crop_size, aug, seed):
    rng=random.Random(seed); img_dir=out/'images'/split; lbl_dir=out/'labels'/split; img_dir.mkdir(parents=True,exist_ok=True); lbl_dir.mkdir(parents=True,exist_ok=True)
    stats=Counter(); labels=Counter(); buckets=Counter(); list_lines=[]
    cache={}
    for idx,item in enumerate(items):
        row=rows[item['row_id']]
        image_path=Path(str(row.get('image_path') or row.get('image')))
        if not image_path.is_absolute(): image_path=ROOT/image_path
        if image_path not in cache:
            cache[image_path]=Image.open(image_path).convert('RGB')
        image=cache[image_path]
        all_targets=(row.get('targets') or {}).get('symbol') or []
        for a in range(aug):
            crop=crop_around(item['bbox'], image.size, crop_size, max(6,crop_size//6), rng)
            stem=f"{item['row_id']}_{item['target_id']}_{idx:05d}_{a:02d}"
            out_img=img_dir/f'{stem}.jpg'; out_lbl=lbl_dir/f'{stem}.txt'
            image.crop(crop).save(out_img, quality=95)
            lines=yolo_lines(all_targets,crop)
            out_lbl.write_text('\n'.join(lines)+('\n' if lines else ''),encoding='utf-8')
            list_lines.append(str(out_img.resolve())); stats['images']+=1; stats['targets']+=len(lines); stats['positive_images']+=1 if lines else 0
            labels[item['label']]+=1; buckets[item['bucket']]+=1
    for im in cache.values(): im.close()
    list_path=out/f'{split}.txt'; list_path.write_text('\n'.join(list_lines)+('\n' if list_lines else ''),encoding='utf-8')
    return {'images':stats['images'],'positive_images':stats['positive_images'],'targets':stats['targets'],'label_counts':dict(labels),'bucket_counts':dict(buckets),'list':str(list_path)}


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--audit',default=str(AUDIT)); ap.add_argument('--overlay',default=str(OVERLAY)); ap.add_argument('--out',default=str(OUT)); ap.add_argument('--crop-size',type=int,default=160); ap.add_argument('--aug',type=int,default=12); ap.add_argument('--seed',type=int,default=213); ap.add_argument('--overwrite',action='store_true')
    args=ap.parse_args(); out=Path(args.out)
    if args.overwrite and out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True,exist_ok=True)
    audit=json.loads(Path(args.audit).read_text()); rows=load_overlay_rows(Path(args.overlay))
    items=[]
    for r in audit['residuals']:
        if r['label'] in TARGET_LABELS or r['bucket'] in TARGET_BUCKETS:
            weight=4 if r['label']=='stair' else 3 if r['bucket']=='tiny_le_64' else 2 if r['label']=='sink' else 1
            for _ in range(weight): items.append(r)
    rng=random.Random(args.seed); rng.shuffle(items); n=len(items)
    train=items[:int(n*.7)]; dev=items[int(n*.7):int(n*.85)]; locked=items[int(n*.85):]
    report={'id':'P213b_residual_specialist_data','source':str(Path(args.audit)),'crop_size':args.crop_size,'aug':args.aug,'weighted_items':n,'target_labels':sorted(TARGET_LABELS),'target_buckets':sorted(TARGET_BUCKETS),'splits':{
        'train':export('train',train,rows,out,args.crop_size,args.aug,args.seed+1),
        'dev':export('dev',dev,rows,out,args.crop_size,args.aug,args.seed+2),
        'locked':export('locked',locked,rows,out,args.crop_size,args.aug,args.seed+3)}}
    (out/'data.yaml').write_text('path: '+str(out.resolve())+'\ntrain: '+str((out/'train.txt').resolve())+'\nval: '+str((out/'dev.txt').resolve())+'\ntest: '+str((out/'locked.txt').resolve())+'\nnames:\n'+''.join(f'  {i}: {name}\n' for i,name in enumerate(LABELS)),encoding='utf-8')
    (out/'build_report.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,ensure_ascii=False,indent=2))

if __name__=='__main__': main()
