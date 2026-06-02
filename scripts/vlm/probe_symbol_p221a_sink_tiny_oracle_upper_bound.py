#!/usr/bin/env python3
"""Oracle upper-bound for fixing only residual sink-tiny nearest predictions."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fuse_symbol_p206g_with_p211_p212 import area_bucket, bbox_iou, load_p206g

ROOT = Path(__file__).resolve().parents[2]
P217 = ROOT / "reports/vlm/symbol_p218_p217_frozen_overlay.jsonl"
CASES = ROOT / "reports/vlm/symbol_p221a_sink_tiny_residual_cases.jsonl"
OUT_JSON = ROOT / "reports/vlm/symbol_p221a_sink_tiny_oracle_upper_bound.json"
OUT_MD = ROOT / "reports/vlm/symbol_p221a_sink_tiny_oracle_upper_bound.md"


def pred_label(pred): return str(pred.get("label", pred.get("symbol_type", "unknown")))
def center(box): return ((box[0]+box[2])/2, (box[1]+box[3])/2)
def dist(a,b):
    ax,ay=center(a); bx,by=center(b); return ((ax-bx)**2+(ay-by)**2)**0.5

def score(preds_by_row, golds_by_row):
    tp=fp=fn=0; fn_label=Counter(); fp_label=Counter(); tp_label=Counter(); fn_bucket=Counter()
    for row_id,gold_map in golds_by_row.items():
        preds=preds_by_row.get(row_id,[]); golds=list(gold_map.values()); cand=[]
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
    prec=tp/max(tp+fp,1); rec=tp/max(tp+fn,1); f1=2*prec*rec/max(prec+rec,1e-9)
    return {'tp':tp,'fp':fp,'fn':fn,'precision':prec,'recall':rec,'f1':f1,'fn_label':dict(fn_label),'fp_label':dict(fp_label),'tp_label':dict(tp_label),'fn_bucket':dict(fn_bucket)}

def fixed_box_at_center(box, w, h):
    cx,cy=center(box); return [cx-w/2,cy-h/2,cx+w/2,cy+h/2]

def apply_oracle(preds_by_row, cases, mode):
    out={rid:[dict(p) for p in preds] for rid,preds in preds_by_row.items()}
    changed=0; missed=0
    for c in cases:
        rid=c['row_id']; gbox=[float(v) for v in c['bbox']]
        preds=out.get(rid, [])
        best_i=None; best_d=1e9
        for i,p in enumerate(preds):
            if pred_label(p)!='sink': continue
            d=dist([float(v) for v in p['bbox']], gbox)
            if d<best_d:
                best_d=d; best_i=i
        if best_i is None or best_d>mode.get('max_dist',8):
            missed+=1; continue
        newp=dict(preds[best_i]);
        if mode['kind']=='gold_center_fixed':
            # oracle center, not deployable; upper bound only
            newp['bbox']=fixed_box_at_center(gbox, mode['w'], mode['h'])
        elif mode['kind']=='pred_center_fixed':
            newp['bbox']=fixed_box_at_center([float(v) for v in newp['bbox']], mode['w'], mode['h'])
        elif mode['kind']=='gold_box':
            newp['bbox']=gbox
        meta=dict(newp.get('metadata') or {}); meta['p221a_oracle']=mode['name']; newp['metadata']=meta
        preds[best_i]=newp; changed+=1
    return out, changed, missed

def main():
    _rows,preds_by_row,golds_by_row=load_p206g(P217)
    cases=[json.loads(l) for l in CASES.read_text().splitlines() if l.strip()]
    base=score(preds_by_row,golds_by_row)
    modes=[]
    for size in [4,5,6,7,8,10,12]:
        modes.append({'name':f'oracle_gold_center_fixed_{size}', 'kind':'gold_center_fixed','w':size,'h':size,'max_dist':8})
        modes.append({'name':f'oracle_pred_center_fixed_{size}', 'kind':'pred_center_fixed','w':size,'h':size,'max_dist':8})
    for w,h in [(5,4),(6,4),(7,5),(8,5),(10,6)]:
        modes.append({'name':f'oracle_gold_center_fixed_{w}x{h}', 'kind':'gold_center_fixed','w':w,'h':h,'max_dist':8})
        modes.append({'name':f'oracle_pred_center_fixed_{w}x{h}', 'kind':'pred_center_fixed','w':w,'h':h,'max_dist':8})
    modes.append({'name':'oracle_gold_box','kind':'gold_box','max_dist':8})
    results=[]
    for mode in modes:
        pp,changed,missed=apply_oracle(preds_by_row,cases,mode)
        m=score(pp,golds_by_row)
        results.append({'mode':mode,'changed':changed,'missed':missed,'metrics':m,'delta_f1':m['f1']-base['f1'],'delta_precision':m['precision']-base['precision'],'delta_recall':m['recall']-base['recall'],'sink_fn':m['fn_label'].get('sink',0),'tiny_fn':m['fn_bucket'].get('tiny_le_64',0)})
    results.sort(key=lambda r:(r['metrics']['f1'],r['metrics']['precision']), reverse=True)
    OUT_JSON.write_text(json.dumps({'id':'P221a_sink_tiny_oracle_upper_bound','baseline':base,'case_count':len(cases),'top_results':results,'claim_boundary':'Oracle upper bound only; gold center/box is forbidden at runtime.'},ensure_ascii=False,indent=2)+"\n")
    lines=['# P221a Sink-Tiny Oracle Upper Bound','','## Baseline',f"- F1/P/R: {base['f1']:.6f}/{base['precision']:.6f}/{base['recall']:.6f}",f"- Sink FN/Tiny FN: {base['fn_label'].get('sink',0)}/{base['fn_bucket'].get('tiny_le_64',0)}",'', '## Top Oracle Probes','| Mode | Changed | F1 | P | R | ΔF1 | ΔP | ΔR | Sink FN | Tiny FN |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results[:20]:
        m=r['metrics']; lines.append(f"| {r['mode']['name']} | {r['changed']} | {m['f1']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {r['delta_f1']:.6f} | {r['delta_precision']:.6f} | {r['delta_recall']:.6f} | {r['sink_fn']} | {r['tiny_fn']} |")
    lines += ['', '## Interpretation', '- If oracle-gold-center improves strongly but pred-center does not, the blocker is sub-pixel/center localization rather than box size alone.', '- If pred-center fixed improves, a runtime-safe shrink/refit gate may be enough.', '- Gold-box/gold-center rows are upper bound only and cannot be used at runtime.']
    OUT_MD.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(OUT_MD),'best':results[0]},ensure_ascii=False,indent=2)[:4000])
if __name__=='__main__': main()
