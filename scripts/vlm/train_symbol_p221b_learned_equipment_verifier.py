#!/usr/bin/env python3
"""Learned runtime-safe equipment verifier for P221b P213b candidates."""
from __future__ import annotations

import json, math, random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier, ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from fuse_symbol_p206g_with_p211_p212 import area_bucket, bbox_iou, load_p206g

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'reports/vlm/symbol_p222_p221a_frozen_overlay.jsonl'
PROP=ROOT/'reports/vlm/symbol_p212_p213b_residual_fusion_overlay.jsonl'
OUT_JSON=ROOT/'reports/vlm/symbol_p221b_learned_equipment_verifier_eval.json'
OUT_MD=ROOT/'reports/vlm/symbol_p221b_learned_equipment_verifier_summary.md'
OUT_OVERLAY=ROOT/'reports/vlm/symbol_p221b_learned_equipment_verifier_overlay.jsonl'

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

def bootstrap(base,cand,it=1000,seed=2214):
    rng=random.Random(seed); vals={'f1':[],'precision':[],'recall':[]}; n=len(base)
    for _ in range(it):
        idx=[rng.randrange(n) for _ in range(n)]; bm=metrics([base[i] for i in idx]); cm=metrics([cand[i] for i in idx])
        vals['f1'].append(cm['f1']-bm['f1']); vals['precision'].append(cm['precision']-bm['precision']); vals['recall'].append(cm['recall']-bm['recall'])
    return {k+'_delta':{'mean':round(sum(v)/len(v),6),'ci95':[round(pct(v,0.025),6),round(pct(v,0.975),6)],'prob_positive':round(sum(x>0 for x in v)/len(v),6)} for k,v in vals.items()}

def onehot_bucket(bucket):
    names=['tiny_le_64','small_le_256','medium_le_1024','large_le_4096','xlarge_gt_4096']
    return [1.0 if bucket==n else 0.0 for n in names]

def build_table(base_rows, prop_rows, golds):
    base_by={row_id(r):norm_preds(r) for r in base_rows}; prop_by={row_id(r):norm_preds(r) for r in prop_rows}
    table=[]
    for rid, props in prop_by.items():
        if rid not in base_by or rid not in golds: continue
        base=base_by[rid]
        eq_base=[q for q in base if pred_label(q)=='equipment']
        any_base=list(base)
        eq_golds=[g for g in golds[rid].values() if str(g['label'])=='equipment']
        for idx,p in enumerate(props):
            if pred_label(p)!='equipment': continue
            pbox=[float(v) for v in p['bbox']]
            score=pred_score(p); a=area(pbox); w=max(0,pbox[2]-pbox[0]); h=max(0,pbox[3]-pbox[1]); bucket=area_bucket(pbox)
            eq_ious=[bbox_iou(pbox,[float(v) for v in q['bbox']]) for q in eq_base]
            any_ious=[bbox_iou(pbox,[float(v) for v in q['bbox']]) for q in any_base]
            eq_dists=[dist(pbox,[float(v) for v in q['bbox']]) for q in eq_base]
            best_gold=max([bbox_iou(pbox,[float(v) for v in g['bbox']]) for g in eq_golds] or [0.0])
            feats=[score,a,math.sqrt(a),w,h,w/max(h,1e-6),max(eq_ious or [0]),max(any_ious or [0]),min(eq_dists or [9999]),len(eq_base),len(any_base)] + onehot_bucket(bucket)
            table.append({'row_id':rid,'prop_index':idx,'candidate':p,'y':1 if best_gold>=0.30 else 0,'best_gold_iou':best_gold,'features':feats,'feature_names':['score','area','sqrt_area','w','h','aspect','max_iou_to_base_equipment','max_iou_to_base_any','min_dist_to_base_equipment','base_equipment_count','base_any_count','bucket_tiny','bucket_small','bucket_medium','bucket_large','bucket_xlarge']})
    return table

def split_rows(rows, seed=2214):
    unique=sorted(set(rows)); rng=random.Random(seed); rng.shuffle(unique)
    n=len(unique); return set(unique[:max(1,int(n*0.6))]), set(unique[max(1,int(n*0.6)):max(2,int(n*0.8))]), set(unique[max(2,int(n*0.8)):])

def build_overlay(base_rows, selected, tag):
    by=defaultdict(list)
    for s in selected: by[s['row_id']].append(s)
    out=[]; added=0
    for r in base_rows:
        rid=row_id(r); preds=norm_preds(r); new=list(preds)
        for item in by.get(rid,[]):
            q=dict(item['candidate']); q['id']=f'{rid}_p221b_lver_{added:05d}'; q['target_id']=q['id']; q['source']='p221b_learned_equipment_verifier'; q['metadata']=dict(q.get('metadata') or {}, p221b_verifier=tag, verifier_score=float(item['verifier_score']))
            new.append(q); added+=1
        nr=dict(r); nr['symbol_candidates']=new; out.append(nr)
    return out,added

def main():
    base_rows=load_rows(BASE); prop_rows=load_rows(PROP); ids=[row_id(r) for r in base_rows]
    _r,base_preds,golds=load_p206g(BASE)
    base_per=score_rows(base_preds,golds,ids); bm=metrics(base_per)
    table=build_table(base_rows,prop_rows,golds)
    train_rows,val_rows,test_rows=split_rows([r['row_id'] for r in table])
    models={
        'hgb_l2_0.0':HistGradientBoostingClassifier(max_iter=80,learning_rate=0.05,l2_regularization=0.0,max_leaf_nodes=8,random_state=2214),
        'hgb_l2_0.1':HistGradientBoostingClassifier(max_iter=80,learning_rate=0.05,l2_regularization=0.1,max_leaf_nodes=8,random_state=2214),
        'rf_depth3':RandomForestClassifier(n_estimators=200,max_depth=3,min_samples_leaf=5,random_state=2214,class_weight='balanced'),
        'extra_depth3':ExtraTreesClassifier(n_estimators=300,max_depth=3,min_samples_leaf=5,random_state=2214,class_weight='balanced'),
        'logreg':make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000,class_weight='balanced',random_state=2214)),
    }
    X=np.array([r['features'] for r in table],dtype=float); y=np.array([r['y'] for r in table],dtype=int)
    train_idx=[i for i,r in enumerate(table) if r['row_id'] in train_rows]
    val_idx=[i for i,r in enumerate(table) if r['row_id'] in val_rows]
    test_idx=[i for i,r in enumerate(table) if r['row_id'] in test_rows]
    results=[]
    for name,model in models.items():
        model.fit(X[train_idx],y[train_idx])
        if hasattr(model,'predict_proba'):
            scores=model.predict_proba(X)[:,1]
        else:
            scores=model.decision_function(X)
        val_auc=roc_auc_score(y[val_idx],scores[val_idx]) if len(set(y[val_idx]))>1 else None
        for th in [0.5,0.6,0.7,0.8,0.85,0.9,0.93,0.95,0.97,0.99]:
            selected=[]
            for i,r in enumerate(table):
                if scores[i] >= th:
                    item=dict(r); item['verifier_score']=float(scores[i]); selected.append(item)
            if not selected: continue
            overlay,added=build_overlay(base_rows,selected,f'{name}_th{th}')
            tmp=ROOT/'reports/vlm/.tmp_p221b_learned_eq_overlay.jsonl'; tmp.write_text('\n'.join(json.dumps(o,ensure_ascii=False) for o in overlay)+'\n')
            _rr,preds,_g=load_p206g(tmp); per=score_rows(preds,golds,ids); m=metrics(per)
            sel_pos=sum(s['y'] for s in selected)
            results.append({'model':name,'threshold':th,'selected':len(selected),'selected_positive':sel_pos,'selected_precision_label':sel_pos/max(len(selected),1),'val_auc':val_auc,'metrics':m,'delta_f1':m['f1']-bm['f1'],'delta_precision':m['precision']-bm['precision'],'delta_recall':m['recall']-bm['recall']})
    results.sort(key=lambda r:(r['metrics']['f1'],r['metrics']['precision']),reverse=True)
    for r in results[:5]:
        # rebuild for bootstrap
        model=models[r['model']]
        scores=model.predict_proba(X)[:,1] if hasattr(model,'predict_proba') else model.decision_function(X)
        selected=[]
        for i,row in enumerate(table):
            if scores[i]>=r['threshold']:
                item=dict(row); item['verifier_score']=float(scores[i]); selected.append(item)
        overlay,added=build_overlay(base_rows,selected,f"{r['model']}_th{r['threshold']}")
        tmp=ROOT/'reports/vlm/.tmp_p221b_learned_eq_overlay.jsonl'; tmp.write_text('\n'.join(json.dumps(o,ensure_ascii=False) for o in overlay)+'\n')
        _rr,preds,_g=load_p206g(tmp); per=score_rows(preds,golds,ids); r['bootstrap_vs_p222']=bootstrap(base_per,per)
    best=results[0]
    model=models[best['model']]; scores=model.predict_proba(X)[:,1] if hasattr(model,'predict_proba') else model.decision_function(X)
    selected=[]
    for i,row in enumerate(table):
        if scores[i]>=best['threshold']:
            item=dict(row); item['verifier_score']=float(scores[i]); selected.append(item)
    overlay,added=build_overlay(base_rows,selected,f"{best['model']}_th{best['threshold']}"); OUT_OVERLAY.write_text('\n'.join(json.dumps(o,ensure_ascii=False) for o in overlay)+'\n')
    payload={'id':'P221b_learned_equipment_verifier','baseline':bm,'table':{'rows':len(table),'positives':int(sum(y)),'train_rows':len(train_rows),'val_rows':len(val_rows),'test_rows':len(test_rows)},'feature_names':table[0]['feature_names'] if table else [],'top_results':results[:50],'best_overlay':str(OUT_OVERLAY.relative_to(ROOT)),'claim_boundary':'P101 learned verifier probe with row-safe split for diagnostics; promote only if bootstrap precision CI non-negative and source audit/freeze pass.'}
    OUT_JSON.write_text(json.dumps(payload,ensure_ascii=False,indent=2)+'\n')
    lines=['# P221b Learned Equipment Verifier','',f"- Candidate rows/positives: {len(table)}/{int(sum(y))}",f"- Row split train/val/test: {len(train_rows)}/{len(val_rows)}/{len(test_rows)}",f"- Baseline P222 F1/P/R: {bm['f1']:.6f}/{bm['precision']:.6f}/{bm['recall']:.6f}",'','| Model | Th | Selected | Label P | F1 | P | R | ΔF1 | ΔP | ΔR | ΔF1 CI | ΔP CI |','|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|---|']
    for r in results[:25]:
        m=r['metrics']; b=r.get('bootstrap_vs_p222',{}); fd=b.get('f1_delta',{}).get('ci95',''); pd=b.get('precision_delta',{}).get('ci95','')
        lines.append(f"| {r['model']} | {r['threshold']} | {r['selected']} | {r['selected_precision_label']:.3f} | {m['f1']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {r['delta_f1']:.6f} | {r['delta_precision']:.6f} | {r['delta_recall']:.6f} | `{fd}` | `{pd}` |")
    lines += ['','## Interpretation','- Features are runtime-safe candidate score/bbox/overlap/density-style counts only; row_id/gold are labels/splits only, not runtime features.', '- If precision CI remains negative, do not promote; use the verifier result as evidence that equipment rescue needs visual/crop evidence.']
    OUT_MD.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(OUT_MD),'best':best,'table':payload['table']},ensure_ascii=False,indent=2)[:5000])
if __name__=='__main__': main()
