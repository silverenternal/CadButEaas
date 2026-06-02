#!/usr/bin/env python3
"""Probe sink-tiny shrink/refit rules for P221a."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fuse_symbol_p206g_with_p211_p212 import area_bucket, bbox_iou, load_p206g

ROOT = Path(__file__).resolve().parents[2]
P217 = ROOT / "reports/vlm/symbol_p218_p217_frozen_overlay.jsonl"
OUT_JSON = ROOT / "reports/vlm/symbol_p221a_sink_tiny_shrink_refiner_probe.json"
OUT_MD = ROOT / "reports/vlm/symbol_p221a_sink_tiny_shrink_refiner_probe.md"


def pred_label(pred):
    return str(pred.get("label", pred.get("symbol_type", "unknown")))


def box_area(box):
    return max(0.0, float(box[2])-float(box[0])) * max(0.0, float(box[3])-float(box[1]))


def center(box):
    return ((float(box[0])+float(box[2]))/2.0, (float(box[1])+float(box[3]))/2.0)


def area_bucket_value(area):
    if area <= 64: return "tiny_le_64"
    if area <= 256: return "small_le_256"
    if area <= 1024: return "medium_le_1024"
    if area <= 4096: return "large_le_4096"
    return "xlarge_gt_4096"


def score(preds_by_row, golds_by_row):
    tp = fp = fn = 0
    tp_label = Counter(); fp_label = Counter(); fn_label = Counter(); fn_bucket = Counter()
    for row_id, gold_map in golds_by_row.items():
        preds = preds_by_row.get(row_id, [])
        golds = list(gold_map.values())
        candidates = []
        for pi, pred in enumerate(preds):
            pbox = [float(v) for v in pred["bbox"]]
            plabel = pred_label(pred)
            for gi, gold in enumerate(golds):
                if plabel != str(gold["label"]): continue
                iou = bbox_iou(pbox, [float(v) for v in gold["bbox"]])
                if iou >= 0.30:
                    candidates.append((iou, pi, gi))
        used_p, used_g = set(), set()
        for iou, pi, gi in sorted(candidates, reverse=True):
            if pi in used_p or gi in used_g: continue
            used_p.add(pi); used_g.add(gi)
            label = str(golds[gi]["label"])
            tp += 1; tp_label[label] += 1
        for pi, pred in enumerate(preds):
            if pi in used_p: continue
            fp += 1; fp_label[pred_label(pred)] += 1
        for gi, gold in enumerate(golds):
            if gi in used_g: continue
            label = str(gold["label"])
            fn += 1; fn_label[label] += 1; fn_bucket[area_bucket([float(v) for v in gold["bbox"]])] += 1
    p = tp / max(tp+fp, 1); r = tp / max(tp+fn, 1); f1 = 2*p*r/max(p+r, 1e-9)
    return {"tp":tp,"fp":fp,"fn":fn,"precision":p,"recall":r,"f1":f1,"tp_label":dict(tp_label),"fp_label":dict(fp_label),"fn_label":dict(fn_label),"fn_bucket":dict(fn_bucket)}


def refit_box(box, mode):
    cx, cy = center(box)
    w = max(1e-6, float(box[2])-float(box[0]))
    h = max(1e-6, float(box[3])-float(box[1]))
    if "scale" in mode:
        w *= mode["scale"]; h *= mode["scale"]
    if "size" in mode:
        w = h = mode["size"]
    if "w" in mode:
        w = mode["w"]
    if "h" in mode:
        h = mode["h"]
    return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]


def transform(preds_by_row, mode):
    out = {}
    for row_id, preds in preds_by_row.items():
        new = []
        for pred in preds:
            p = dict(pred)
            label = pred_label(pred)
            box = [float(v) for v in pred["bbox"]]
            area = box_area(box)
            score_value = float(pred.get("score", pred.get("confidence", 0.0)) or 0.0)
            if label == "sink" and area <= mode.get("max_area", 4096) and score_value >= mode.get("min_score", 0.0):
                p["bbox"] = refit_box(box, mode)
                meta = dict(p.get("metadata") or {})
                meta["p221a_probe"] = mode["name"]
                p["metadata"] = meta
            new.append(p)
        out[row_id] = new
    return out


def summarize_offsets(golds_by_row, preds_by_row):
    widths=[]; heights=[]; pred_widths=[]; pred_heights=[]; ratios=[]; ious=[]
    for row_id, gold_map in golds_by_row.items():
        preds = preds_by_row.get(row_id, [])
        for gold in gold_map.values():
            if str(gold["label"]) != "sink": continue
            gbox = [float(v) for v in gold["bbox"]]
            if area_bucket(gbox) != "tiny_le_64": continue
            nearest = None
            for pred in preds:
                if pred_label(pred) != "sink": continue
                pbox = [float(v) for v in pred["bbox"]]
                dist = ((center(gbox)[0]-center(pbox)[0])**2 + (center(gbox)[1]-center(pbox)[1])**2)**0.5
                if nearest is None or dist < nearest[0]: nearest = (dist, pbox)
            if nearest and nearest[0] <= 8:
                pbox = nearest[1]
                gw, gh = gbox[2]-gbox[0], gbox[3]-gbox[1]
                pw, ph = pbox[2]-pbox[0], pbox[3]-pbox[1]
                widths.append(gw); heights.append(gh); pred_widths.append(pw); pred_heights.append(ph)
                ratios.append((pw*ph)/max(gw*gh,1e-9)); ious.append(bbox_iou(pbox,gbox))
    def stats(values):
        values=sorted(values)
        if not values: return {}
        return {"n":len(values),"min":values[0],"p25":values[len(values)//4],"median":values[len(values)//2],"p75":values[(3*len(values))//4],"max":values[-1],"avg":sum(values)/len(values)}
    return {"gold_w":stats(widths),"gold_h":stats(heights),"pred_w":stats(pred_widths),"pred_h":stats(pred_heights),"area_ratio_pred_over_gold":stats(ratios),"nearest_iou":stats(ious)}


def main():
    _rows, preds_by_row, golds_by_row = load_p206g(P217)
    base = score(preds_by_row, golds_by_row)
    modes=[]
    for scale in [0.08,0.10,0.12,0.15,0.18,0.20,0.25,0.30,0.40,0.50,0.70]:
        modes.append({"name":f"sink_shrink_scale_{scale}","scale":scale,"max_area":4096})
    for size in [3,4,5,6,7,8,9,10,12,14,16]:
        modes.append({"name":f"sink_fixed_square_{size}","size":size,"max_area":4096})
    for w,h in [(4,4),(5,4),(6,4),(8,4),(10,4),(12,4),(6,5),(8,5),(10,5),(12,6),(16,8)]:
        modes.append({"name":f"sink_fixed_{w}x{h}","w":w,"h":h,"max_area":4096})
    for max_area in [64,128,256,512,1024,4096]:
        for size in [4,5,6,8]:
            modes.append({"name":f"sink_area_lte_{max_area}_fixed_{size}","size":size,"max_area":max_area})
    results=[]
    for mode in modes:
        metrics=score(transform(preds_by_row, mode), golds_by_row)
        results.append({"mode":mode,"metrics":metrics,"delta_f1":metrics['f1']-base['f1'],"delta_precision":metrics['precision']-base['precision'],"delta_recall":metrics['recall']-base['recall'],"sink_fn":metrics['fn_label'].get('sink',0),"tiny_fn":metrics['fn_bucket'].get('tiny_le_64',0)})
    results.sort(key=lambda r:(r['metrics']['f1'],r['metrics']['precision']), reverse=True)
    offset_summary=summarize_offsets(golds_by_row, preds_by_row)
    payload={"id":"P221a_sink_tiny_shrink_refiner_probe","baseline":base,"offset_summary":offset_summary,"top_results":results[:50],"claim_boundary":"Probe only; uses gold only for offline evaluation and rule selection."}
    OUT_JSON.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+"\n",encoding='utf-8')
    lines=["# P221a Sink-Tiny Shrink/Refit Probe","","## Geometry Diagnosis",f"- Gold width stats: `{json.dumps(offset_summary['gold_w'], ensure_ascii=False)}`",f"- Gold height stats: `{json.dumps(offset_summary['gold_h'], ensure_ascii=False)}`",f"- Nearest p206g sink width stats: `{json.dumps(offset_summary['pred_w'], ensure_ascii=False)}`",f"- Nearest p206g sink height stats: `{json.dumps(offset_summary['pred_h'], ensure_ascii=False)}`",f"- Pred/gold area ratio stats: `{json.dumps(offset_summary['area_ratio_pred_over_gold'], ensure_ascii=False)}`",f"- Nearest IoU stats: `{json.dumps(offset_summary['nearest_iou'], ensure_ascii=False)}`","","## Baseline",f"- F1/P/R: {base['f1']:.6f}/{base['precision']:.6f}/{base['recall']:.6f}",f"- TP/FP/FN: {base['tp']}/{base['fp']}/{base['fn']}","","## Top Refit Probes","| Mode | F1 | P | R | ΔF1 | ΔP | ΔR | Sink FN | Tiny FN |","|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results[:20]:
        m=r['metrics']
        lines.append(f"| {r['mode']['name']} | {m['f1']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {r['delta_f1']:.6f} | {r['delta_precision']:.6f} | {r['delta_recall']:.6f} | {r['sink_fn']} | {r['tiny_fn']} |")
    lines += ["","## Interpretation","- P206g nearest sink boxes are often much larger than tiny gold boxes; shrink/refit is the right first probe, not expansion.","- This probe still changes all qualifying sink boxes globally, so any positive result needs tighter verifier/rule gating before promotion."]
    OUT_MD.write_text("\n".join(lines)+"\n",encoding='utf-8')
    print(json.dumps({"report":str(OUT_MD),"best":results[0],"offset_summary":offset_summary},ensure_ascii=False,indent=2)[:4000])

if __name__=='__main__':
    main()
