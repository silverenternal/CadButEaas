#!/usr/bin/env python3
"""Runtime-safe subcandidate probe for sink-tiny residuals."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fuse_symbol_p206g_with_p211_p212 import area_bucket, bbox_iou, load_p206g, simple_nms

ROOT=Path(__file__).resolve().parents[2]
P217=ROOT/'reports/vlm/symbol_p218_p217_frozen_overlay.jsonl'
OUT_JSON=ROOT/'reports/vlm/symbol_p221a_sink_tiny_runtime_subcandidate_probe.json'
OUT_MD=ROOT/'reports/vlm/symbol_p221a_sink_tiny_runtime_subcandidate_probe.md'

def pred_label(p): return str(p.get('label', p.get('symbol_type','unknown')))
def center(b): return ((b[0]+b[2])/2,(b[1]+b[3])/2)
def fixed(cx,cy,w,h): return [cx-w/2,cy-h/2,cx+w/2,cy+h/2]
def area(b): return max(0,float(b[2])-float(b[0]))*max(0,float(b[3])-float(b[1]))

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

def add_subcandidates(preds_by_row, mode):
    out={}; added=0
    for rid,preds in preds_by_row.items():
        new=[dict(p) for p in preds]
        sink_preds=[p for p in preds if pred_label(p)=='sink']
        for idx,p in enumerate(sink_preds):
            box=[float(v) for v in p['bbox']]; a=area(box)
            score_value=float(p.get('score',p.get('confidence',0.0)) or 0.0)
            if a < mode.get('min_area',0) or a > mode.get('max_area',4096): continue
            if score_value < mode.get('min_score',0.0): continue
            cx,cy=center(box)
            w,h=mode['w'],mode['h']
            offsets=mode['offsets']
            for ox,oy in offsets:
                nb=fixed(cx+ox,cy+oy,w,h)
                if mode.get('skip_if_iou_to_any_sink_ge') is not None:
                    if any(bbox_iou(nb,[float(v) for v in q['bbox']])>=mode['skip_if_iou_to_any_sink_ge'] for q in new if pred_label(q)=='sink'):
                        continue
                new.append({'id':f'p221a_sub_{rid}_{idx}_{added}','target_id':f'p221a_sub_{rid}_{idx}_{added}','label':'sink','symbol_type':'sink','bbox':nb,'score':score_value*mode.get('score_scale',0.5),'confidence':score_value*mode.get('score_scale',0.5),'source':'p221a_runtime_subcandidate','metadata':{'p221a_mode':mode['name'],'parent_score':score_value}})
                added+=1
        if mode.get('nms') is not None:
            new=simple_nms(new, mode['nms'])
        out[rid]=new
    return out, added

def make_offsets(radius, include_center=True):
    offs=[]
    if include_center: offs.append((0,0))
    for r in radius:
        offs += [(r,0),(-r,0),(0,r),(0,-r),(r,r),(r,-r),(-r,r),(-r,-r)]
    # dedup
    seen=set(); out=[]
    for o in offs:
        if o not in seen: seen.add(o); out.append(o)
    return out

def main():
    _rows,preds_by_row,golds_by_row=load_p206g(P217)
    base=score(preds_by_row,golds_by_row)
    modes=[]
    offset_sets={'center':[(0,0)],'r2':make_offsets([2]),'r4':make_offsets([4]),'r2_4':make_offsets([2,4]),'r4_8':make_offsets([4,8])}
    for max_area in [64,128,256,512,1024,2048,4096]:
        for min_score in [0.0,0.5,0.8,0.9]:
            for size in [4,5,6]:
                for oname,offs in offset_sets.items():
                    modes.append({'name':f'a{max_area}_s{min_score}_box{size}_{oname}','max_area':max_area,'min_score':min_score,'w':size,'h':size,'offsets':offs,'score_scale':0.5})
    results=[]
    for mode in modes:
        pp,added=add_subcandidates(preds_by_row,mode); m=score(pp,golds_by_row)
        results.append({'mode':{k:v for k,v in mode.items() if k!='offsets'}|{'offset_count':len(mode['offsets'])},'added':added,'metrics':m,'delta_f1':m['f1']-base['f1'],'delta_precision':m['precision']-base['precision'],'delta_recall':m['recall']-base['recall'],'sink_fn':m['fn_label'].get('sink',0),'tiny_fn':m['fn_bucket'].get('tiny_le_64',0)})
    results.sort(key=lambda r:(r['metrics']['f1'], r['metrics']['precision']), reverse=True)
    OUT_JSON.write_text(json.dumps({'id':'P221a_sink_tiny_runtime_subcandidate_probe','baseline':base,'top_results':results[:100],'claim_boundary':'Runtime-safe geometry-only subcandidate probe; selected on P101 so needs bootstrap/freeze before promotion.'},ensure_ascii=False,indent=2)+'\n')
    lines=['# P221a Runtime-Safe Sink-Tiny Subcandidate Probe','','## Baseline',f"- F1/P/R: {base['f1']:.6f}/{base['precision']:.6f}/{base['recall']:.6f}",f"- Sink FN/Tiny FN: {base['fn_label'].get('sink',0)}/{base['fn_bucket'].get('tiny_le_64',0)}",'', '## Top Runtime Subcandidate Rules','| Mode | Added | F1 | P | R | ΔF1 | ΔP | ΔR | Sink FN | Tiny FN |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for r in results[:25]:
        m=r['metrics']; lines.append(f"| {r['mode']['name']} | {r['added']} | {m['f1']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {r['delta_f1']:.6f} | {r['delta_precision']:.6f} | {r['delta_recall']:.6f} | {r['sink_fn']} | {r['tiny_fn']} |")
    lines += ['', '## Interpretation','- Add-oracle shows the ceiling is high, but naive runtime subcandidates may add many FPs.', '- Promote only if a rule improves F1 while precision loss is acceptable/non-negative under bootstrap.', '- If no geometry-only rule works, train a crop verifier to select subcandidate centers.']
    OUT_MD.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(OUT_MD),'best':results[0]},ensure_ascii=False,indent=2)[:4000])
if __name__=='__main__': main()
