#!/usr/bin/env python3
"""Probe adding existing P213b/P213c stair/equipment candidates over P222."""
from __future__ import annotations

import json, random
from collections import Counter
from pathlib import Path
from typing import Any

from fuse_symbol_p206g_with_p211_p212 import bbox_iou, load_p206g

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'reports/vlm/symbol_p222_p221a_frozen_overlay.jsonl'
PROPOSAL=ROOT/'reports/vlm/symbol_p212_p213b_residual_fusion_overlay.jsonl'
OUT_JSON=ROOT/'reports/vlm/symbol_p221b_add_existing_candidates_probe.json'
OUT_MD=ROOT/'reports/vlm/symbol_p221b_add_existing_candidates_probe.md'

def pred_label(p): return str(p.get('label', p.get('symbol_type','unknown')))
def pred_score(p): return float(p.get('score', p.get('confidence',0.0)) or 0.0)
def row_id(r): return str(r.get('id') or r.get('row_id'))
def load_rows(p): return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def norm_preds(row):
    out=[]
    for p in row.get('symbol_candidates') or []:
        q=dict(p)
        if 'label' not in q and 'symbol_type' in q: q['label']=q['symbol_type']
        if 'symbol_type' not in q and 'label' in q: q['symbol_type']=q['label']
        if 'score' not in q and 'confidence' in q: q['score']=q['confidence']
        if 'confidence' not in q and 'score' in q: q['confidence']=q['score']
        out.append(q)
    return out

def score_rows(preds_by_row,golds_by_row,ids):
    per=[]
    for rid in ids:
        preds=preds_by_row.get(rid,[]); golds=list(golds_by_row[rid].values()); cand=[]
        for pi,p in enumerate(preds):
            pb=[float(v) for v in p['bbox']]; pl=pred_label(p)
            for gi,g in enumerate(golds):
                if pl!=str(g['label']): continue
                iou=bbox_iou(pb,[float(v) for v in g['bbox']])
                if iou>=0.30: cand.append((iou,pi,gi))
        up,ug=set(),set()
        for iou,pi,gi in sorted(cand, reverse=True):
            if pi in up or gi in ug: continue
            up.add(pi); ug.add(gi)
        per.append({'row_id':rid,'counts':{'tp':len(ug),'pred':len(preds),'gold':len(golds),'fp':len(preds)-len(up),'fn':len(golds)-len(ug)}})
    return per

def metrics(per):
    c=Counter();
    for r in per: c.update(r['counts'])
    p=c['tp']/max(c['pred'],1); r=c['tp']/max(c['gold'],1); f1=2*p*r/max(p+r,1e-9)
    return {'tp':int(c['tp']),'predicted':int(c['pred']),'gold':int(c['gold']),'fp':int(c['fp']),'fn':int(c['fn']),'precision':p,'recall':r,'f1':f1}

def pct(v,q):
    v=sorted(v); pos=(len(v)-1)*q; lo=int(pos); hi=min(lo+1,len(v)-1); f=pos-lo; return v[lo]*(1-f)+v[hi]*f

def bootstrap(base,cand,it=1000,seed=2212):
    rng=random.Random(seed); vals={'f1':[],'precision':[],'recall':[]}; n=len(base)
    for _ in range(it):
        idx=[rng.randrange(n) for _ in range(n)]; bm=metrics([base[i] for i in idx]); cm=metrics([cand[i] for i in idx])
        vals['f1'].append(cm['f1']-bm['f1']); vals['precision'].append(cm['precision']-bm['precision']); vals['recall'].append(cm['recall']-bm['recall'])
    return {k+'_delta':{'mean':round(sum(v)/len(v),6),'ci95':[round(pct(v,0.025),6),round(pct(v,0.975),6)],'prob_positive':round(sum(x>0 for x in v)/len(v),6)} for k,v in vals.items()}

def build_candidate(base_rows, prop_rows, mode):
    prop_by={row_id(r):norm_preds(r) for r in prop_rows}
    out=[]; added=0
    for row in base_rows:
        rid=row_id(row); base=norm_preds(row); new=list(base)
        for i,p in enumerate(prop_by.get(rid,[])):
            label=pred_label(p); score=pred_score(p)
            if label not in mode['labels']: continue
            if score<mode['min_score']: continue
            pbox=[float(v) for v in p['bbox']]
            if any(pred_label(q)==label and bbox_iou(pbox,[float(v) for v in q['bbox']])>=mode['max_iou_to_base'] for q in base): continue
            q=dict(p); q['id']=f'{rid}_p221b_existing_{added:05d}'; q['target_id']=q['id']; q['source']='p221b_existing_p213b_add'; q['metadata']=dict(q.get('metadata') or {}, p221b_mode=mode['name'])
            new.append(q); added+=1
        nr=dict(row); nr['symbol_candidates']=new; out.append(nr)
    return out,added

def main():
    base_rows=load_rows(BASE); prop_rows=load_rows(PROPOSAL); ids=[row_id(r) for r in base_rows]
    _r,base_preds,golds=load_p206g(BASE)
    base_per=score_rows(base_preds,golds,ids); bm=metrics(base_per)
    modes=[]
    for labels in [{'equipment'},{'stair'},{'equipment','stair'}]:
        lname='_'.join(sorted(labels))
        for min_score in [0.0,0.2,0.4,0.6,0.75,0.85,0.9,0.95]:
            for max_iou in [0.05,0.1,0.2,0.3,0.5,0.8,1.1]:
                modes.append({'name':f'{lname}_s{min_score}_iou{max_iou}','labels':labels,'min_score':min_score,'max_iou_to_base':max_iou})
    results=[]
    for mode in modes:
        rows,added=build_candidate(base_rows,prop_rows,mode)
        tmp=ROOT/'reports/vlm/.tmp_p221b_probe_overlay.jsonl'; tmp.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
        _rr,preds,_g=load_p206g(tmp); per=score_rows(preds,golds,ids); m=metrics(per)
        results.append({'mode':{k:(sorted(v) if isinstance(v,set) else v) for k,v in mode.items()},'added':added,'metrics':m,'delta_f1':m['f1']-bm['f1'],'delta_precision':m['precision']-bm['precision'],'delta_recall':m['recall']-bm['recall']})
    results.sort(key=lambda r:(r['metrics']['f1'],r['metrics']['precision']), reverse=True)
    # bootstrap top 5
    for r in results[:5]:
        mode={k:(set(v) if k=='labels' else v) for k,v in r['mode'].items() if k!='name'}; mode['name']=r['mode']['name']
        rows,added=build_candidate(base_rows,prop_rows,mode); tmp=ROOT/'reports/vlm/.tmp_p221b_probe_overlay.jsonl'; tmp.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n')
        _rr,preds,_g=load_p206g(tmp); per=score_rows(preds,golds,ids); r['bootstrap_vs_p222']=bootstrap(base_per,per)
    OUT_JSON.write_text(json.dumps({'id':'P221b_add_existing_candidates_probe','base':str(BASE.relative_to(ROOT)),'proposal':str(PROPOSAL.relative_to(ROOT)),'baseline':bm,'top_results':results[:50]},ensure_ascii=False,indent=2)+'\n')
    lines=['# P221b Add Existing P213b Candidates Probe','',f"- Baseline P222 F1/P/R: {bm['f1']:.6f}/{bm['precision']:.6f}/{bm['recall']:.6f}",'','| Mode | Added | F1 | P | R | ΔF1 | ΔP | ΔR | Bootstrap ΔF1 CI | Bootstrap ΔP CI |','|---|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for r in results[:25]:
        m=r['metrics']; b=r.get('bootstrap_vs_p222',{}); fd=b.get('f1_delta',{}).get('ci95',''); pd=b.get('precision_delta',{}).get('ci95','')
        lines.append(f"| {r['mode']['name']} | {r['added']} | {m['f1']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {r['delta_f1']:.6f} | {r['delta_precision']:.6f} | {r['delta_recall']:.6f} | `{fd}` | `{pd}` |")
    lines += ['', '## Interpretation','- Existing P213b proposals can be reused only if precision CI stays non-negative.', '- If best rules have negative precision CI, P221b needs a verifier/training branch before promotion.']
    OUT_MD.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(OUT_MD),'best':results[0]},ensure_ascii=False,indent=2)[:5000])
if __name__=='__main__': main()
