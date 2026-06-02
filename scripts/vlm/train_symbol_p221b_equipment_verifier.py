#!/usr/bin/env python3
"""Train/probe runtime-safe verifier for P221b equipment candidates from P213b over P222."""
from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

from fuse_symbol_p206g_with_p211_p212 import area_bucket, bbox_iou, load_p206g

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'reports/vlm/symbol_p222_p221a_frozen_overlay.jsonl'
PROP=ROOT/'reports/vlm/symbol_p212_p213b_residual_fusion_overlay.jsonl'
OUT_JSON=ROOT/'reports/vlm/symbol_p221b_equipment_verifier_eval.json'
OUT_MD=ROOT/'reports/vlm/symbol_p221b_equipment_verifier_summary.md'
OUT_OVERLAY=ROOT/'reports/vlm/symbol_p221b_equipment_verifier_overlay.jsonl'

def load_rows(p): return [json.loads(l) for l in p.read_text().splitlines() if l.strip()]
def row_id(r): return str(r.get('id') or r.get('row_id'))
def pred_label(p): return str(p.get('label', p.get('symbol_type','unknown')))
def pred_score(p): return float(p.get('score', p.get('confidence',0.0)) or 0.0)
def area(b): return max(0,float(b[2])-float(b[0]))*max(0,float(b[3])-float(b[1]))
def center(b): return ((float(b[0])+float(b[2]))/2,(float(b[1])+float(b[3]))/2)
def dist(a,b):
    ax,ay=center(a); bx,by=center(b); return ((ax-bx)**2+(ay-by)**2)**0.5

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
    c=Counter()
    for r in per: c.update(r['counts'])
    p=c['tp']/max(c['pred'],1); r=c['tp']/max(c['gold'],1); f1=2*p*r/max(p+r,1e-9)
    return {'tp':int(c['tp']),'predicted':int(c['pred']),'gold':int(c['gold']),'fp':int(c['fp']),'fn':int(c['fn']),'precision':p,'recall':r,'f1':f1}

def pct(v,q):
    v=sorted(v); pos=(len(v)-1)*q; lo=int(pos); hi=min(lo+1,len(v)-1); f=pos-lo; return v[lo]*(1-f)+v[hi]*f

def bootstrap(base,cand,it=1000,seed=2213):
    rng=random.Random(seed); vals={'f1':[],'precision':[],'recall':[]}; n=len(base)
    for _ in range(it):
        idx=[rng.randrange(n) for _ in range(n)]; bm=metrics([base[i] for i in idx]); cm=metrics([cand[i] for i in idx])
        vals['f1'].append(cm['f1']-bm['f1']); vals['precision'].append(cm['precision']-bm['precision']); vals['recall'].append(cm['recall']-bm['recall'])
    return {k+'_delta':{'mean':round(sum(v)/len(v),6),'ci95':[round(pct(v,0.025),6),round(pct(v,0.975),6)],'prob_positive':round(sum(x>0 for x in v)/len(v),6)} for k,v in vals.items()}

def build_table(base_rows, prop_rows, base_preds, golds):
    base_by={row_id(r):norm_preds(r) for r in base_rows}; prop_by={row_id(r):norm_preds(r) for r in prop_rows}
    table=[]
    for rid, props in prop_by.items():
        if rid not in base_by or rid not in golds: continue
        base=base_by[rid]
        equipment_golds=[g for g in golds[rid].values() if str(g['label'])=='equipment']
        for idx,p in enumerate(props):
            if pred_label(p)!='equipment': continue
            pbox=[float(v) for v in p['bbox']]
            score=pred_score(p)
            overlaps=[bbox_iou(pbox,[float(v) for v in q['bbox']]) for q in base if pred_label(q)=='equipment']
            any_overlaps=[bbox_iou(pbox,[float(v) for v in q['bbox']]) for q in base]
            dists=[dist(pbox,[float(v) for v in q['bbox']]) for q in base if pred_label(q)=='equipment']
            best_gold=max([bbox_iou(pbox,[float(v) for v in g['bbox']]) for g in equipment_golds] or [0.0])
            y=1 if best_gold>=0.30 else 0
            w=max(0.0,pbox[2]-pbox[0]); h=max(0.0,pbox[3]-pbox[1]); a=area(pbox)
            table.append({'row_id':rid,'prop_index':idx,'candidate':p,'y':y,'best_gold_iou':best_gold,'features':{
                'score':score,'area':a,'sqrt_area':math.sqrt(a),'w':w,'h':h,'aspect':w/max(h,1e-6),
                'bucket':area_bucket(pbox),'max_iou_to_base_equipment':max(overlaps or [0.0]),'max_iou_to_base_any':max(any_overlaps or [0.0]),
                'min_dist_to_base_equipment':min(dists or [9999.0]),'base_equipment_count':sum(1 for q in base if pred_label(q)=='equipment'),
                'base_any_count':len(base)
            }})
    return table

def passes(row, rule):
    f=row['features']
    if f['score']<rule.get('min_score',0): return False
    if f['area']<rule.get('min_area',0): return False
    if f['area']>rule.get('max_area',1e18): return False
    if f['max_iou_to_base_equipment']<rule.get('min_iou_to_base_equipment',0): return False
    if f['max_iou_to_base_equipment']>rule.get('max_iou_to_base_equipment',1e18): return False
    if f['min_dist_to_base_equipment']<rule.get('min_dist_to_base_equipment',-1): return False
    if f['min_dist_to_base_equipment']>rule.get('max_dist_to_base_equipment',1e18): return False
    buckets=rule.get('buckets')
    if buckets and f['bucket'] not in buckets: return False
    return True

def build_overlay(base_rows, selected):
    selected_by=Counter(); selected_map={}
    for row in selected:
        selected_map.setdefault(row['row_id'],[]).append(row)
    out=[]; added=0
    for r in base_rows:
        rid=row_id(r); preds=norm_preds(r); new=list(preds)
        for item in selected_map.get(rid,[]):
            q=dict(item['candidate']); q['id']=f'{rid}_p221b_eqver_{added:05d}'; q['target_id']=q['id']; q['source']='p221b_equipment_verifier'; q['metadata']=dict(q.get('metadata') or {}, p221b_verifier_rule=item.get('rule_name','unknown'))
            new.append(q); added+=1
        nr=dict(r); nr['symbol_candidates']=new; out.append(nr)
    return out,added

def main():
    base_rows=load_rows(BASE); prop_rows=load_rows(PROP); ids=[row_id(r) for r in base_rows]
    _r,base_preds,golds=load_p206g(BASE)
    base_per=score_rows(base_preds,golds,ids); bm=metrics(base_per)
    table=build_table(base_rows,prop_rows,base_preds,golds)
    pos=sum(r['y'] for r in table)
    rules=[]
    for min_score in [0.94,0.95,0.96,0.97,0.98]:
        for max_iou in [1.1,0.8,0.4,0.1]:
            for min_iou in [0.0,0.1,0.3]:
                for buckets in [None, {'large_le_4096','xlarge_gt_4096'}, {'large_le_4096'}, {'xlarge_gt_4096'}]:
                    rules.append({'name':f's{min_score}_miniou{min_iou}_maxiou{max_iou}_b{("all" if buckets is None else "+".join(sorted(buckets)))}','min_score':min_score,'min_iou_to_base_equipment':min_iou,'max_iou_to_base_equipment':max_iou,'buckets':buckets})
    results=[]
    for rule in rules:
        selected=[]
        for row in table:
            if passes(row,rule):
                x=dict(row); x['rule_name']=rule['name']; selected.append(x)
        if not selected: continue
        rows,added=build_overlay(base_rows,selected); tmp=ROOT/'reports/vlm/.tmp_p221b_eqver_overlay.jsonl'; tmp.write_text('\n'.join(json.dumps(r,ensure_ascii=False) for r in rows)+'\n')
        _rr,preds,_g=load_p206g(tmp); per=score_rows(preds,golds,ids); m=metrics(per)
        selected_pos=sum(r['y'] for r in selected)
        results.append({'rule':{k:(sorted(v) if isinstance(v,set) else v) for k,v in rule.items()},'selected':len(selected),'selected_positive':selected_pos,'selected_precision_label':selected_pos/max(len(selected),1),'metrics':m,'delta_f1':m['f1']-bm['f1'],'delta_precision':m['precision']-bm['precision'],'delta_recall':m['recall']-bm['recall']})
    results.sort(key=lambda r:(r['metrics']['f1'],r['metrics']['precision']), reverse=True)
    for r in results[:5]:
        rule={k:(set(v) if k=='buckets' and isinstance(v,list) else v) for k,v in r['rule'].items()}
        selected=[]
        for row in table:
            if passes(row,rule):
                x=dict(row); x['rule_name']=rule['name']; selected.append(x)
        rows,added=build_overlay(base_rows,selected); tmp=ROOT/'reports/vlm/.tmp_p221b_eqver_overlay.jsonl'; tmp.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in rows)+'\n')
        _rr,preds,_g=load_p206g(tmp); per=score_rows(preds,golds,ids); r['bootstrap_vs_p222']=bootstrap(base_per,per)
    best=results[0] if results else None
    if best:
        rule={k:(set(v) if k=='buckets' and isinstance(v,list) else v) for k,v in best['rule'].items()}
        selected=[]
        for row in table:
            if passes(row,rule):
                x=dict(row); x['rule_name']=rule['name']; selected.append(x)
        overlay,added=build_overlay(base_rows,selected); OUT_OVERLAY.write_text('\n'.join(json.dumps(x,ensure_ascii=False) for x in overlay)+'\n')
    payload={'id':'P221b_equipment_verifier_probe','baseline':bm,'candidate_table':{'rows':len(table),'positives':pos,'positive_rate':pos/max(len(table),1)},'top_results':results[:50],'best_overlay':str(OUT_OVERLAY.relative_to(ROOT)) if best else None,'claim_boundary':'P101 verifier/rule search evidence; promote only if bootstrap precision CI non-negative.'}
    OUT_JSON.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    lines=['# P221b Equipment Verifier Probe','',f"- Candidate table rows/positives: {len(table)}/{pos}",f"- Baseline P222 F1/P/R: {bm['f1']:.6f}/{bm['precision']:.6f}/{bm['recall']:.6f}",'','| Rule | Selected | Label precision | F1 | P | R | ΔF1 | ΔP | ΔR | ΔF1 CI | ΔP CI |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for r in results[:25]:
        m=r['metrics']; b=r.get('bootstrap_vs_p222',{}); fd=b.get('f1_delta',{}).get('ci95',''); pd=b.get('precision_delta',{}).get('ci95','')
        lines.append(f"| {r['rule']['name']} | {r['selected']} | {r['selected_precision_label']:.3f} | {m['f1']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {r['delta_f1']:.6f} | {r['delta_precision']:.6f} | {r['delta_recall']:.6f} | `{fd}` | `{pd}` |")
    lines += ['', '## Interpretation','- This is a runtime-safe rule/verifier probe over P213b equipment candidates.', '- If top precision CI crosses negative, keep P222 baseline and train a stronger verifier rather than promoting.']
    OUT_MD.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(OUT_MD),'table':payload['candidate_table'],'best':best},ensure_ascii=False,indent=2)[:5000])
if __name__=='__main__': main()
