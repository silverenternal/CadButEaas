#!/usr/bin/env python3
"""Apply selected P221a runtime-safe sink-tiny subcandidate rule and bootstrap vs P217."""
from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from fuse_symbol_p206g_with_p211_p212 import bbox_iou, load_p206g

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'reports/vlm/symbol_p218_p217_frozen_overlay.jsonl'
OUT_OVERLAY=ROOT/'reports/vlm/symbol_p221a_sink_tiny_subcandidate_overlay.jsonl'
OUT_JSON=ROOT/'reports/vlm/symbol_p221a_sink_tiny_subcandidate_eval.json'
OUT_BOOT_JSON=ROOT/'reports/vlm/symbol_p221a_vs_p217_bootstrap_validation.json'
OUT_BOOT_MD=ROOT/'reports/vlm/symbol_p221a_vs_p217_bootstrap_validation.md'
OUT_SUMMARY=ROOT/'reports/vlm/symbol_p221a_sink_tiny_subcandidate_summary.md'
RULE={'name':'p221a_sink_area_lte64_score_ge05_box4_center','max_area':64.0,'min_score':0.5,'w':4.0,'h':4.0,'score_scale':0.5}

def load_rows(path:Path): return [json.loads(l) for l in path.read_text().splitlines() if l.strip()]
def row_id(row): return str(row.get('id') or row.get('row_id'))
def pred_label(p): return str(p.get('label', p.get('symbol_type','unknown')))
def pred_score(p): return float(p.get('score',p.get('confidence',0.0)) or 0.0)
def box_area(b): return max(0,float(b[2])-float(b[0]))*max(0,float(b[3])-float(b[1]))
def center(b): return ((float(b[0])+float(b[2]))/2,(float(b[1])+float(b[3]))/2)
def fixed(cx,cy,w,h): return [cx-w/2,cy-h/2,cx+w/2,cy+h/2]

def symbol_candidates(row):
    out=[]
    for p in row.get('symbol_candidates') or []:
        q=dict(p)
        if 'label' not in q and 'symbol_type' in q: q['label']=q['symbol_type']
        if 'symbol_type' not in q and 'label' in q: q['symbol_type']=q['label']
        if 'score' not in q and 'confidence' in q: q['score']=q['confidence']
        if 'confidence' not in q and 'score' in q: q['confidence']=q['score']
        out.append(q)
    return out

def target_symbols(row):
    # load_p206g handles gold extraction; this is only for row writing not eval.
    return row.get('target_symbols') or row.get('symbols') or []

def add_subs_to_candidates(row):
    rid=row_id(row); preds=symbol_candidates(row); new=list(preds); added=0
    for idx,p in enumerate(preds):
        if pred_label(p)!='sink': continue
        box=[float(v) for v in p['bbox']]
        if box_area(box)>RULE['max_area']: continue
        if pred_score(p)<RULE['min_score']: continue
        cx,cy=center(box); nb=fixed(cx,cy,RULE['w'],RULE['h'])
        score=pred_score(p)*RULE['score_scale']
        new.append({'id':f'{rid}_p221a_sinktiny_{idx:04d}','target_id':f'{rid}_p221a_sinktiny_{idx:04d}','symbol_type':'sink','label':'sink','bbox':nb,'confidence':score,'score':score,'source':'p221a_sink_tiny_subcandidate','metadata':{'fusion_policy':RULE['name'],'parent_id':p.get('id') or p.get('target_id'),'parent_score':pred_score(p),'runtime_features':'parent sink label/score/bbox geometry only'}})
        added+=1
    out=dict(row); out['symbol_candidates']=new
    meta=dict(out.get('metadata') or {}); meta['p221a_added_sink_tiny_subcandidates']=added; meta['p221a_rule']=RULE['name']; out['metadata']=meta
    return out, added

def score_rows(preds_by_row,golds_by_row,row_ids):
    per=[]
    for rid in row_ids:
        preds=preds_by_row.get(rid,[]); golds=list(golds_by_row[rid].values()); cand=[]
        for pi,p in enumerate(preds):
            pb=[float(v) for v in p['bbox']]; pl=pred_label(p)
            for gi,g in enumerate(golds):
                if pl!=str(g['label']): continue
                iou=bbox_iou(pb,[float(v) for v in g['bbox']])
                if iou>=0.30: cand.append((iou,pi,gi))
        up,ug=set(),set(); tp=0
        for iou,pi,gi in sorted(cand, reverse=True):
            if pi in up or gi in ug: continue
            up.add(pi); ug.add(gi); tp+=1
        per.append({'row_id':rid,'counts':{'tp':tp,'pred':len(preds),'gold':len(golds),'fp':len(preds)-len(up),'fn':len(golds)-len(ug)}})
    return per

def metrics(per):
    c=Counter()
    for r in per: c.update(r['counts'])
    p=c['tp']/max(c['pred'],1); r=c['tp']/max(c['gold'],1); f1=2*p*r/max(p+r,1e-9)
    return {'tp':int(c['tp']),'predicted':int(c['pred']),'gold':int(c['gold']),'fp':int(c['fp']),'fn':int(c['fn']),'precision':round(p,6),'recall':round(r,6),'f1':round(f1,6)}

def percentile(vals,q):
    vals=sorted(vals); pos=(len(vals)-1)*q; lo=int(pos); hi=min(lo+1,len(vals)-1); frac=pos-lo
    return vals[lo]*(1-frac)+vals[hi]*frac

def bootstrap(base_per,cand_per,iterations=1000,seed=221):
    rng=random.Random(seed); vals={'f1':[],'precision':[],'recall':[]}
    n=len(base_per)
    for _ in range(iterations):
        idx=[rng.randrange(n) for _ in range(n)]
        bm=metrics([base_per[i] for i in idx]); cm=metrics([cand_per[i] for i in idx])
        vals['f1'].append(cm['f1']-bm['f1']); vals['precision'].append(cm['precision']-bm['precision']); vals['recall'].append(cm['recall']-bm['recall'])
    out={}
    for k,v in vals.items():
        out[k+'_delta']={'mean':round(sum(v)/len(v),6),'ci95':[round(percentile(v,0.025),6),round(percentile(v,0.975),6)],'prob_positive':round(sum(x>0 for x in v)/len(v),6)}
    return out

def main():
    rows=load_rows(BASE); out=[]; total_added=0
    for row in rows:
        nr,added=add_subs_to_candidates(row); out.append(nr); total_added+=added
    OUT_OVERLAY.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in out)+'\n')
    _,base_preds,golds=load_p206g(BASE)
    _,cand_preds,_=load_p206g(OUT_OVERLAY)
    ids=[row_id(r) for r in rows]
    base_per=score_rows(base_preds,golds,ids); cand_per=score_rows(cand_preds,golds,ids)
    bm=metrics(base_per); cm=metrics(cand_per); boot=bootstrap(base_per,cand_per)
    report={'id':'P221a_sink_tiny_subcandidate','rule':RULE,'added':total_added,'baseline_metrics':bm,'candidate_metrics':cm,'bootstrap':boot,'claim_boundary':'P101-selected runtime-safe geometry rule; must be frozen/source-audited before paper promotion; independent validation still pending.','artifacts':{'overlay':str(OUT_OVERLAY.relative_to(ROOT)),'eval':str(OUT_JSON.relative_to(ROOT)),'bootstrap_md':str(OUT_BOOT_MD.relative_to(ROOT))}}
    OUT_JSON.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    OUT_BOOT_JSON.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    fd, pd, rd=boot['f1_delta'], boot['precision_delta'], boot['recall_delta']
    md=['# P221a Sink-Tiny Subcandidate Bootstrap vs P217','', '## Metrics', '| Variant | F1 | Precision | Recall | TP | Pred | Gold |','|---|---:|---:|---:|---:|---:|---:|', f"| P217/P218 baseline | {bm['f1']:.6f} | {bm['precision']:.6f} | {bm['recall']:.6f} | {bm['tp']} | {bm['predicted']} | {bm['gold']} |", f"| P221a subcandidate | {cm['f1']:.6f} | {cm['precision']:.6f} | {cm['recall']:.6f} | {cm['tp']} | {cm['predicted']} | {cm['gold']} |",'', '## Rule', f"- Rule: `{RULE['name']}`", f"- Added subcandidates: `{total_added}`", '- Runtime features: parent sink label/score/bbox geometry only; no row_id/gold/SVG/parser geometry at runtime.', '', '## Paired Bootstrap', f"- ΔF1 mean/CI/P>0: `{fd['mean']:.6f}` / `[{fd['ci95'][0]:.6f}, {fd['ci95'][1]:.6f}]` / `{fd['prob_positive']:.3f}`", f"- ΔPrecision mean/CI/P>0: `{pd['mean']:.6f}` / `[{pd['ci95'][0]:.6f}, {pd['ci95'][1]:.6f}]` / `{pd['prob_positive']:.3f}`", f"- ΔRecall mean/CI/P>0: `{rd['mean']:.6f}` / `[{rd['ci95'][0]:.6f}, {rd['ci95'][1]:.6f}]` / `{rd['prob_positive']:.3f}`", '', '## Claim Boundary', report['claim_boundary']]
    OUT_BOOT_MD.write_text('\n'.join(md)+'\n')
    OUT_SUMMARY.write_text('\n'.join(md)+'\n')
    print(json.dumps({'baseline':bm,'candidate':cm,'bootstrap':boot,'added':total_added,'overlay':str(OUT_OVERLAY)},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
