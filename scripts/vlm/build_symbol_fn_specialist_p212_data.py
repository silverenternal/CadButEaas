#!/usr/bin/env python3
"""Build P212 specialist YOLO data around P206g false-negative symbol regions."""
from __future__ import annotations

import argparse
import json
import random
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image

from fuse_symbol_p206g_with_p211_p212 import LABELS, LABEL_TO_ID, bbox_iou, area_bucket, load_p206g

ROOT = Path(__file__).resolve().parents[2]
P206G = ROOT / "reports/vlm/symbol_p206f_precision_repair_p206g_overlay.jsonl"
OUT = ROOT / "datasets/symbol_fn_specialist_p212_yolo"
TARGET_LABELS = {"sink", "shower", "equipment", "stair"}
TARGET_BUCKETS = {"tiny_le_64", "small_le_256", "medium_le_1024"}


def is_matched_by_core(gold_box: list[float], core_preds: list[dict[str, Any]]) -> bool:
    return any(bbox_iou([float(v) for v in pred["bbox"]], gold_box) >= 0.30 for pred in core_preds)


def crop_box_around(box: list[float], image_size: tuple[int, int], crop_size: int, jitter: int, rng: random.Random) -> tuple[int, int, int, int]:
    width, height = image_size
    cx = (box[0] + box[2]) / 2.0 + rng.randint(-jitter, jitter)
    cy = (box[1] + box[3]) / 2.0 + rng.randint(-jitter, jitter)
    left = int(round(cx - crop_size / 2))
    top = int(round(cy - crop_size / 2))
    left = max(0, min(left, max(width - crop_size, 0)))
    top = max(0, min(top, max(height - crop_size, 0)))
    return left, top, min(left + crop_size, width), min(top + crop_size, height)


def yolo_lines(targets: list[dict[str, Any]], crop: tuple[int, int, int, int]) -> list[str]:
    left, top, right, bottom = crop
    w = max(1, right - left); h = max(1, bottom - top)
    lines=[]; seen=set()
    for target in targets:
        label=str(target.get("label") or target.get("semantic_type") or "")
        if label not in LABEL_TO_ID:
            continue
        x1,y1,x2,y2=[float(v) for v in target.get("bbox")]
        ix1,iy1=max(x1,left),max(y1,top)
        ix2,iy2=min(x2,right),min(y2,bottom)
        if ix2 <= ix1 or iy2 <= iy1:
            continue
        visible=(ix2-ix1)*(iy2-iy1)/max((x2-x1)*(y2-y1),1e-9)
        if visible < 0.55:
            continue
        vals=[LABEL_TO_ID[label]-1, ((ix1+ix2)/2-left)/w, ((iy1+iy2)/2-top)/h, (ix2-ix1)/w, (iy2-iy1)/h]
        line=" ".join([str(vals[0]), *[f"{v:.8f}" for v in vals[1:]]])
        if line not in seen:
            seen.add(line); lines.append(line)
    return lines


def export_split(name: str, items: list[dict[str, Any]], out: Path, crop_size: int, aug: int, seed: int) -> dict[str, Any]:
    rng=random.Random(seed)
    image_dir=out/"images"/name; label_dir=out/"labels"/name
    image_dir.mkdir(parents=True, exist_ok=True); label_dir.mkdir(parents=True, exist_ok=True)
    list_lines=[]; stats=Counter(); label_counts=Counter(); bucket_counts=Counter()
    for item_index,item in enumerate(items):
        row=item["row"]; target=item["target"]
        image_path=Path(str(row.get("image_path") or row.get("image")))
        if not image_path.is_absolute(): image_path=ROOT/image_path
        with Image.open(image_path) as opened:
            image=opened.convert("RGB")
            for aug_index in range(aug):
                crop=crop_box_around(target["bbox"], image.size, crop_size, max(4,crop_size//8), rng)
                crop_image=image.crop(crop)
                stem=f"{row['id']}_{target['target_id']}_{item_index:05d}_{aug_index:02d}"
                out_img=image_dir/f"{stem}.jpg"; out_lbl=label_dir/f"{stem}.txt"
                crop_image.save(out_img, quality=95)
                lines=yolo_lines(item["all_targets"], crop)
                out_lbl.write_text("\n".join(lines)+("\n" if lines else ""), encoding="utf-8")
                list_lines.append(str(out_img.resolve()))
                stats["images"]+=1; stats["targets"]+=len(lines); stats["positive_images"]+=1 if lines else 0
                label_counts[target["label"]]+=1; bucket_counts[target["bucket"]]+=1
    list_path=out/f"{name}.txt"
    list_path.write_text("\n".join(list_lines)+("\n" if list_lines else ""), encoding="utf-8")
    return {"images":stats["images"],"positive_images":stats["positive_images"],"targets":stats["targets"],"label_counts":dict(label_counts),"bucket_counts":dict(bucket_counts),"list":str(list_path)}


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--p206g", default=str(P206G))
    parser.add_argument("--out", default=str(OUT))
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--aug", type=int, default=8)
    parser.add_argument("--seed", type=int, default=212)
    parser.add_argument("--overwrite", action="store_true")
    args=parser.parse_args()
    out=Path(args.out)
    if args.overwrite and out.exists(): shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    rows, core, _golds=load_p206g(Path(args.p206g))
    items=[]
    for row in rows:
        row_id=str(row.get("id") or row.get("row_id"))
        all_targets=[]
        for target in (row.get("targets") or {}).get("symbol") or []:
            label=str(target.get("semantic_type") or "generic_symbol")
            box=[float(v) for v in target.get("bbox")]
            all_targets.append({"target_id":target.get("target_id"),"label":label,"bbox":box})
        for target in all_targets:
            bucket=area_bucket(target["bbox"])
            if target["label"] not in TARGET_LABELS and bucket not in TARGET_BUCKETS:
                continue
            if is_matched_by_core(target["bbox"], core.get(row_id, [])):
                continue
            item_target=dict(target); item_target["bucket"]=bucket
            items.append({"row":row,"target":item_target,"all_targets":all_targets})
    rng=random.Random(args.seed); rng.shuffle(items)
    n=len(items); train=items[:int(n*0.7)]; dev=items[int(n*0.7):int(n*0.85)]; locked=items[int(n*0.85):]
    report={
        "id":"P212_symbol_fn_specialist_data",
        "source":"P206g false negatives only; offline gold used for supervised training target construction",
        "target_labels":sorted(TARGET_LABELS),"target_buckets":sorted(TARGET_BUCKETS),"crop_size":args.crop_size,"aug":args.aug,"total_fn_items":n,
        "splits":{
            "train":export_split("train",train,out,args.crop_size,args.aug,args.seed+1),
            "dev":export_split("dev",dev,out,args.crop_size,args.aug,args.seed+2),
            "locked":export_split("locked",locked,out,args.crop_size,args.aug,args.seed+3),
        },
    }
    (out/"data.yaml").write_text("path: "+str(out.resolve())+"\ntrain: "+str((out/"train.txt").resolve())+"\nval: "+str((out/"dev.txt").resolve())+"\ntest: "+str((out/"locked.txt").resolve())+"\nnames:\n"+"".join(f"  {i}: {name}\n" for i,name in enumerate(LABELS)), encoding="utf-8")
    (out/"build_report.json").write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n", encoding="utf-8")
    print(json.dumps(report,ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
