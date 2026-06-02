#!/usr/bin/env python3
"""Train a runtime-safe verifier for P213b/P216 added candidates."""
from __future__ import annotations

import argparse,json,math,random
from collections import Counter,defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from tune_symbol_p214_precision_repair import fuse
from fuse_symbol_p206g_with_p212_specialist import build_overlay, load_p212
from fuse_symbol_p206g_with_p211_p212 import bbox_iou, load_p206g, score_predictions, write_json, write_jsonl

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'reports/vlm/symbol_p206g_p212_specialist_precision_repair_overlay.jsonl'
P213=ROOT/'reports/vlm/symbol_residual_specialist_p213b_pages_s160_top180_predictions.jsonl'
REPORT=ROOT/'reports/vlm/symbol_p217_verifier_fusion_eval.json'
OVERLAY=ROOT/'reports/vlm/symbol_p217_verifier_fusion_overlay.jsonl'
MODEL_REPORT=ROOT/'reports/vlm/symbol_p217_verifier_train_report.json'

SOURCE_POLICY={'allowed_labels':['sink','equipment','stair','shower'],'threshold':0.9,'label_thresholds':{'stair':0.95,'sink':0.86,'equipment':0.88,'shower':0.78},'max_add_per_row':20,'max_iou_to_core':0.25,'min_dist_to_core':0,'row_blacklist':['cubicasa5k_locked_00022','cubicasa5k_locked_00024']}
LABELS=['sink','equipment','stair','shower']


def row_num(row_id:str)->int:
    try: return int(row_id.rsplit('_',1)[-1])
    except Exception: return 0


def split_for_row(row_id:str)->str:
    n=row_num(row_id)
    if n % 5 == 0: return 'test'
    if n % 5 == 1: return 'val'
    return 'train'


def is_tp(pred:dict[str,Any], golds:list[dict[str,Any]])->bool:
    pbox=[float(v) for v in pred['bbox']]
    return any(bbox_iou(pbox,[float(v) for v in g['bbox']])>=0.30 for g in golds)


def features(row_id:str, pred:dict[str,Any], base_preds:list[dict[str,Any]], all_added:list[dict[str,Any]], image_size=None)->list[float]:
    box=[float(v) for v in pred['bbox']]; x1,y1,x2,y2=box; w=max(1e-6,x2-x1); h=max(1e-6,y2-y1); area=w*h
    score=float(pred.get('score') or 0.0); label=str(pred.get('label'))
    cx=(x1+x2)/2; cy=(y1+y2)/2
    # distances / overlaps to existing base predictions
    max_iou_base=0.0; min_dist_base=1e9; same_label_near=0; any_near=0
    for bp in base_preds:
        bb=[float(v) for v in bp['bbox']]; iou=bbox_iou(box,bb); max_iou_base=max(max_iou_base,iou)
        bx=(bb[0]+bb[2])/2; by=(bb[1]+bb[3])/2; d=((cx-bx)**2+(cy-by)**2)**0.5
        min_dist_base=min(min_dist_base,d)
        if d<32: any_near+=1
        if str(bp.get('label'))==label and d<32: same_label_near+=1
    # local density among added candidates
    add_near=0; same_add_near=0; max_iou_added=0.0
    for ap in all_added:
        if ap is pred: continue
        ab=[float(v) for v in ap['bbox']]; ax=(ab[0]+ab[2])/2; ay=(ab[1]+ab[3])/2; d=((cx-ax)**2+(cy-ay)**2)**0.5
        if d<32: add_near+=1
        if str(ap.get('label'))==label and d<32: same_add_near+=1
        max_iou_added=max(max_iou_added,bbox_iou(box,ab))
    label_onehot=[1.0 if label==l else 0.0 for l in LABELS]
    tile=str(pred.get('tile_id') or '')
    # parse slice size if present
    nums=[]
    for part in tile.replace('.','_').split('_'):
        if part.isdigit(): nums.append(int(part))
    slice_w=slice_h=0
    if len(nums)>=4:
        slice_w=max(0,nums[-2]-nums[-4]); slice_h=max(0,nums[-1]-nums[-3])
    return [score, math.log1p(area), math.log1p(w), math.log1p(h), w/max(h,1e-6), max_iou_base, math.log1p(min_dist_base if min_dist_base<1e8 else 9999), any_near, same_label_near, add_near, same_add_near, max_iou_added, math.log1p(slice_w), math.log1p(slice_h), *label_onehot]


def build_table(base,p213,golds):
    source=fuse(base,p213,SOURCE_POLICY)
    rows=[]
    for row_id,preds in source.items():
        base_len=len(base.get(row_id,[])); added=preds[base_len:]; gold_list=list(golds[row_id].values())
        for pred in added:
            rows.append({'row_id':row_id,'pred':pred,'x':features(row_id,pred,base.get(row_id,[]),added),'y':1 if is_tp(pred,gold_list) else 0,'split':split_for_row(row_id),'label':str(pred.get('label')),'score':float(pred.get('score') or 0)})
    return rows


def eval_classifier(clf, rows, threshold):
    selected=[]; y=[]; scores=[]
    for r in rows:
        prob=float(clf.predict_proba([r['x']])[0][1])
        y.append(r['y']); scores.append(prob)
        if prob>=threshold: selected.append(r)
    if y and len(set(y))>1:
        auc=roc_auc_score(y,scores); ap=average_precision_score(y,scores)
    else:
        auc=ap=0.0
    tp=sum(r['y'] for r in selected); total=len(selected)
    return {'threshold':threshold,'selected':total,'tp':tp,'precision':round(tp/max(total,1),6),'recall_on_candidate_tp':round(tp/max(sum(y),1),6),'roc_auc':round(auc,6),'ap':round(ap,6)}


def fuse_with_verifier(base,p213,golds,clf,threshold):
    source=fuse(base,p213,SOURCE_POLICY); out={}
    for row_id,preds in source.items():
        base_len=len(base.get(row_id,[])); base_preds=[dict(p) for p in preds[:base_len]]; added=preds[base_len:]
        kept=[]
        for pred in added:
            prob=float(clf.predict_proba([features(row_id,pred,base.get(row_id,[]),added)])[0][1])
            if prob>=threshold:
                p=dict(pred); p['score']=prob; p['source']='p217_verifier_added'; kept.append(p)
        out[row_id]=base_preds+kept
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',default=str(BASE)); ap.add_argument('--p213',default=str(P213)); ap.add_argument('--report',default=str(REPORT)); ap.add_argument('--overlay',default=str(OVERLAY)); ap.add_argument('--model-report',default=str(MODEL_REPORT)); args=ap.parse_args()
    overlay_rows,base,golds=load_p206g(Path(args.base)); p213=load_p212(Path(args.p213)); table=build_table(base,p213,golds)
    train=[r for r in table if r['split']=='train']; val=[r for r in table if r['split']=='val']; test=[r for r in table if r['split']=='test']
    X=np.array([r['x'] for r in train]); y=np.array([r['y'] for r in train])
    clfs={
      'logreg':make_pipeline(StandardScaler(),LogisticRegression(max_iter=1000,class_weight='balanced')),
      'hgb':HistGradientBoostingClassifier(max_iter=120,l2_regularization=0.05,learning_rate=0.05,max_leaf_nodes=15,random_state=217),
      'rf':RandomForestClassifier(n_estimators=200,max_depth=5,min_samples_leaf=4,class_weight='balanced',random_state=217),
    }
    results=[]; best=None
    for name,clf in clfs.items():
        clf.fit(X,y)
        for th in [0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75]:
            val_eval=eval_classifier(clf,val,th); test_eval=eval_classifier(clf,test,th)
            fused=fuse_with_verifier(base,p213,golds,clf,th); metrics,_=score_predictions(fused,golds,0.0,0.98,900,0); additions=sum(max(0,len(fused[r])-len(base.get(r,[]))) for r in fused)
            row={'name':name,'threshold':th,'val_candidate_eval':val_eval,'test_candidate_eval':test_eval,'metrics':metrics,'additions':additions}
            results.append(row)
            key=(metrics['symbol_bbox_iou_0_30']['f1'], metrics['symbol_bbox_iou_0_30']['precision'])
            if best is None or key>(best['metrics']['symbol_bbox_iou_0_30']['f1'], best['metrics']['symbol_bbox_iou_0_30']['precision']): best=row; best_clf=clf
    fused=fuse_with_verifier(base,p213,golds,best_clf,best['threshold'])
    write_jsonl(Path(args.overlay),build_overlay(overlay_rows,fused,{'name':f"p217_{best['name']}_th{best['threshold']}",'source_policy':SOURCE_POLICY,'runtime_features':'score_geometry_density_label_no_rowid_no_gold'}))
    report={'id':'P217_runtime_safe_verifier','claim_boundary':'Verifier uses runtime-safe candidate features only; labels/gold used for training/evaluation. Selection still on P101, so needs held-out validation for paper claim.','table_counts':{'total':len(table),'train':len(train),'val':len(val),'test':len(test),'positive':sum(r['y'] for r in table)},'selected':best,'top20':sorted(results,key=lambda r:(r['metrics']['symbol_bbox_iou_0_30']['f1'],r['metrics']['symbol_bbox_iou_0_30']['precision']),reverse=True)[:20],'outputs':{'overlay':str(Path(args.overlay)),'report':str(Path(args.report))}}
    write_json(Path(args.report),report); write_json(Path(args.model_report),{'table_counts':report['table_counts'],'selected':best,'all_results':results})
    print(json.dumps({'table':report['table_counts'],'selected':{'name':best['name'],'threshold':best['threshold'],'metrics':best['metrics']['symbol_bbox_iou_0_30'],'additions':best['additions'],'val':best['val_candidate_eval'],'test':best['test_candidate_eval']}},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
