#!/usr/bin/env python3
"""Oracle probes for P221b stair/equipment residuals."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fuse_symbol_p206g_with_p211_p212 import area_bucket, bbox_iou, load_p206g

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'reports/vlm/symbol_p222_p221a_frozen_overlay.jsonl'
CASES=ROOT/'reports/vlm/symbol_p221b_stair_equipment_residual_cases.jsonl'
OUT_JSON=ROOT/'reports/vlm/symbol_p221b_stair_equipment_oracle_probe.json'
OUT_MD=ROOT/'reports/vlm/symbol_p221b_stair_equipment_oracle_probe.md'

def pred_label(p): return str(p.get('label', p.get('symbol_type','unknown')))
def center(b): return ((float(b[0])+float(b[2]))/2,(float(b[1])+float(b[3]))/2)
def dist(a,b):
    ax,ay=center(a); bx,by=center(b); return ((ax-bx)**2+(ay-by)**2)**0.5

def fixed_at(box,w,h):
    cx,cy=center(box); return [cx-w/2,cy-h/2,cx+w/2,cy+h/2]

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

def apply(preds_by_row,cases,mode):
    out={rid:[dict(p) for p in preds] for rid,preds in preds_by_row.items()}; changed=0; added=0; skipped=0
    for idx,c in enumerate(cases):
        if mode.get('labels') and c['label'] not in mode['labels']: continue
        if mode.get('buckets') and c['bucket'] not in mode['buckets']: continue
        rid=c['row_id']; gbox=[float(v) for v in c['bbox']]
        if mode['op']=='replace_nearest_same':
            best_i=None; best_d=1e18
            for i,p in enumerate(out.get(rid,[])):
                if pred_label(p)!=c['label']: continue
                d=dist([float(v) for v in p['bbox']],gbox)
                if d<best_d: best_d=d; best_i=i
            if best_i is None or best_d>mode.get('max_dist',1e18): skipped+=1; continue
            p=dict(out[rid][best_i]); p['bbox']=gbox; p['metadata']=dict(p.get('metadata') or {}, p221b_oracle=mode['name']); out[rid][best_i]=p; changed+=1
        elif mode['op']=='add_gold':
            out.setdefault(rid,[]).append({'id':f'p221b_oracle_add_{idx}','target_id':f'p221b_oracle_add_{idx}','label':c['label'],'symbol_type':c['label'],'bbox':gbox,'score':0.999,'confidence':0.999,'source':'p221b_oracle_add','metadata':{'p221b_oracle':mode['name']}}); added+=1
    return out, changed, added, skipped

def main():
    _rows,preds_by_row,golds_by_row=load_p206g(BASE)
    cases=[json.loads(l) for l in CASES.read_text().splitlines() if l.strip()]
    base=score(preds_by_row,golds_by_row)
    modes=[]
    for labelset in [{'stair'},{'equipment'},{'stair','equipment'}]:
        lname='_'.join(sorted(labelset))
        modes.append({'name':f'add_gold_{lname}','op':'add_gold','labels':labelset})
        for d in [8,16,32,64,128,99999]:
            modes.append({'name':f'replace_gold_{lname}_d{d}','op':'replace_nearest_same','labels':labelset,'max_dist':d})
    for bucket in ['small_le_256','large_le_4096','xlarge_gt_4096','tiny_le_64']:
        modes.append({'name':f'add_gold_bucket_{bucket}','op':'add_gold','buckets':{bucket}})
    results=[]
    for mode in modes:
        pp,changed,added,skipped=apply(preds_by_row,cases,mode); m=score(pp,golds_by_row)
        results.append({'mode':{k:(sorted(v) if isinstance(v,set) else v) for k,v in mode.items()},'changed':changed,'added':added,'skipped':skipped,'metrics':m,'delta_f1':m['f1']-base['f1'],'delta_precision':m['precision']-base['precision'],'delta_recall':m['recall']-base['recall'],'stair_fn':m['fn_label'].get('stair',0),'equipment_fn':m['fn_label'].get('equipment',0)})
    results.sort(key=lambda r:(r['metrics']['f1'],r['metrics']['precision']), reverse=True)
    OUT_JSON.write_text(json.dumps({'id':'P221b_stair_equipment_oracle_probe','baseline':base,'case_count':len(cases),'top_results':results,'claim_boundary':'Oracle upper bound only; gold boxes forbidden at runtime.'},ensure_ascii=False,indent=2)+'\n')
    lines=['# P221b Stair/Equipment Oracle Probe','','## Baseline',f"- F1/P/R: {base['f1']:.6f}/{base['precision']:.6f}/{base['recall']:.6f}",f"- Stair FN / Equipment FN: {base['fn_label'].get('stair',0)} / {base['fn_label'].get('equipment',0)}",'', '## Top Oracle Probes','| Mode | Changed | Added | F1 | P | R | ΔF1 | ΔP | ΔR | Stair FN | Equipment FN |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results[:25]:
        m=r['metrics']; lines.append(f"| {r['mode']['name']} | {r['changed']} | {r['added']} | {m['f1']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {r['delta_f1']:.6f} | {r['delta_precision']:.6f} | {r['delta_recall']:.6f} | {r['stair_fn']} | {r['equipment_fn']} |")
    lines += ['', '## Interpretation','- Add oracle estimates proposal-generation ceiling.','- Replacement oracle estimates whether nearby same-label boxes only need refinement.','- Large gap between add and replace implies missing proposals or multiple instances around same parent.']
    OUT_MD.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(OUT_MD),'best':results[0]},ensure_ascii=False,indent=2)[:4000])
if __name__=='__main__': main()
