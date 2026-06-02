#!/usr/bin/env python3
"""Oracle upper-bound for adding residual sink-tiny candidates instead of replacing nearby boxes."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fuse_symbol_p206g_with_p211_p212 import area_bucket, bbox_iou, load_p206g

ROOT=Path(__file__).resolve().parents[2]
P217=ROOT/'reports/vlm/symbol_p218_p217_frozen_overlay.jsonl'
CASES=ROOT/'reports/vlm/symbol_p221a_sink_tiny_residual_cases.jsonl'
OUT_JSON=ROOT/'reports/vlm/symbol_p221a_sink_tiny_add_oracle.json'
OUT_MD=ROOT/'reports/vlm/symbol_p221a_sink_tiny_add_oracle.md'

def pred_label(p): return str(p.get('label', p.get('symbol_type','unknown')))
def center(b): return ((b[0]+b[2])/2,(b[1]+b[3])/2)
def fixed(cx,cy,w,h): return [cx-w/2,cy-h/2,cx+w/2,cy+h/2]

def score(preds_by_row,golds_by_row):
    tp=fp=fn=0; fn_label=Counter(); fp_label=Counter(); tp_label=Counter(); fn_bucket=Counter()
    for rid,gold_map in golds_by_row.items():
        preds=preds_by_row.get(rid,[]); golds=list(gold_map.values()); cand=[]
        for pi,p in enumerate(preds):
            pb=[float(v) for v in p['bbox']]; pl=pred_label(p)
            for gi,g in enumerate(golds):
                if pl!=str(g['label']): continue
                iou=bbox_iou(pb,[float(v) for v in g['bbox']])
                if iou>=0.30: cand.append((iou,pi,gi))
        up,ug=set(),set()
        for iou,pi,gi in sorted(cand, reverse=True):
            if pi in up or gi in ug: continue
            up.add(pi); ug.add(gi); tp+=1; tp_label[str(golds[gi]['label'])]+=1
        for pi,p in enumerate(preds):
            if pi not in up: fp+=1; fp_label[pred_label(p)]+=1
        for gi,g in enumerate(golds):
            if gi not in ug:
                fn+=1; fn_label[str(g['label'])]+=1; fn_bucket[area_bucket([float(v) for v in g['bbox']])]+=1
    pr=tp/max(tp+fp,1); re=tp/max(tp+fn,1); f1=2*pr*re/max(pr+re,1e-9)
    return {'tp':tp,'fp':fp,'fn':fn,'precision':pr,'recall':re,'f1':f1,'fn_label':dict(fn_label),'fp_label':dict(fp_label),'tp_label':dict(tp_label),'fn_bucket':dict(fn_bucket)}

def add_oracle(preds_by_row,cases,mode):
    out={rid:[dict(p) for p in preds] for rid,preds in preds_by_row.items()}
    for idx,c in enumerate(cases):
        rid=c['row_id']; gbox=[float(v) for v in c['bbox']]; cx,cy=center(gbox)
        if mode['kind']=='gold_box': bbox=gbox
        else: bbox=fixed(cx,cy,mode['w'],mode['h'])
        out.setdefault(rid,[]).append({'id':f'p221a_oracle_add_{idx:05d}','target_id':f'p221a_oracle_add_{idx:05d}','label':'sink','symbol_type':'sink','bbox':bbox,'score':0.999,'confidence':0.999,'source':'p221a_oracle_add','metadata':{'p221a_oracle_add':mode['name']}})
    return out

def main():
    _rows,preds_by_row,golds_by_row=load_p206g(P217)
    cases=[json.loads(l) for l in CASES.read_text().splitlines() if l.strip()]
    base=score(preds_by_row,golds_by_row)
    modes=[{'name':'add_gold_box','kind':'gold_box'}]
    for size in [3,4,5,6,7,8,10]: modes.append({'name':f'add_gold_center_fixed_{size}','kind':'fixed','w':size,'h':size})
    for w,h in [(5,4),(6,4),(7,5),(8,5),(10,6)]: modes.append({'name':f'add_gold_center_fixed_{w}x{h}','kind':'fixed','w':w,'h':h})
    results=[]
    for mode in modes:
        pp=add_oracle(preds_by_row,cases,mode); m=score(pp,golds_by_row)
        results.append({'mode':mode,'added':len(cases),'metrics':m,'delta_f1':m['f1']-base['f1'],'delta_precision':m['precision']-base['precision'],'delta_recall':m['recall']-base['recall'],'sink_fn':m['fn_label'].get('sink',0),'tiny_fn':m['fn_bucket'].get('tiny_le_64',0)})
    results.sort(key=lambda r:(r['metrics']['f1'],r['metrics']['precision']), reverse=True)
    OUT_JSON.write_text(json.dumps({'id':'P221a_sink_tiny_add_oracle','baseline':base,'case_count':len(cases),'top_results':results,'claim_boundary':'Oracle add upper bound only; gold centers/boxes forbidden at runtime.'},ensure_ascii=False,indent=2)+'\n')
    lines=['# P221a Sink-Tiny Add Oracle','','## Baseline',f"- F1/P/R: {base['f1']:.6f}/{base['precision']:.6f}/{base['recall']:.6f}",f"- Sink FN/Tiny FN: {base['fn_label'].get('sink',0)}/{base['fn_bucket'].get('tiny_le_64',0)}",'', '## Top Add Oracle Probes','| Mode | Added | F1 | P | R | ΔF1 | ΔP | ΔR | Sink FN | Tiny FN |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results:
        m=r['metrics']; lines.append(f"| {r['mode']['name']} | {r['added']} | {m['f1']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {r['delta_f1']:.6f} | {r['delta_precision']:.6f} | {r['delta_recall']:.6f} | {r['sink_fn']} | {r['tiny_fn']} |")
    lines += ['', '## Interpretation','- Replacement oracle only gained a few matches; add oracle tests whether multiple tiny sinks are hidden inside/near existing larger sink boxes.', '- If add oracle is much higher, P221a should generate additional tiny sink sub-candidates around existing sink proposals rather than resizing existing boxes.']
    OUT_MD.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(OUT_MD),'best':results[0]},ensure_ascii=False,indent=2)[:4000])
if __name__=='__main__': main()
