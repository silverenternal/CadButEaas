#!/usr/bin/env python3
"""Smoke metadata-only selector over P0-76 added candidates."""
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path
from typing import Any
SCRIPT_DIR=Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0,str(SCRIPT_DIR))
from apply_symbol_rtdetr_complement_policy_p065 import read_exported_golds
from train_symbol_p076_added_candidate_selector_p078 import apply_feature_gate, build_rows
from sweep_symbol_added_candidate_reranker_p075 import rows_from_predictions
from sweep_symbol_rtdetr_complement_gate_p063 import read_predictions, score
from train_symbol_tile_detector_v20 import rel, write_jsonl
ROOT=Path(__file__).resolve().parents[2]
DEFAULT_DATA=ROOT/'datasets/symbol_tile_detector_tiny_sahi_v21'
DEFAULT_YOLO_DIR=ROOT/'datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27'
DEFAULT_V28=ROOT/'reports/vlm/symbol_yolov8s_seg_rect_v28_smoke_v30_page_predictions_p062_refresh.jsonl'
DEFAULT_P076=ROOT/'reports/vlm/symbol_balanced_policy_p076_smoke_v30_predictions.jsonl'
DEFAULT_JSON=ROOT/'reports/vlm/symbol_metadata_selector_p081_smoke_v30.json'
DEFAULT_MD=ROOT/'reports/vlm/symbol_metadata_selector_p081_smoke_v30.md'
DEFAULT_PRED=ROOT/'reports/vlm/symbol_metadata_selector_p081_smoke_v30_predictions.jsonl'


def box_features(pred):
    l,t,r,b=[float(v) for v in pred['bbox']]
    w=max(1e-6,r-l); h=max(1e-6,b-t); area=w*h
    return w,h,area,w/h

def simple_logistic_train(rows, steps=200, lr=0.05):
    feats=[]; ys=[]
    for r in rows:
        x=[1.0, r['score'], r['overlap_v28'], math.log1p(r['bbox_area']), r['bbox_aspect']]
        feats.append(x); ys.append(float(r['target']))
    w=[0.0]*5
    pos=sum(ys); neg=len(ys)-pos; pos_weight=neg/max(pos,1.0)
    for _ in range(steps):
        grad=[0.0]*5
        for x,y in zip(feats,ys):
            z=sum(a*b for a,b in zip(w,x)); p=1/(1+math.exp(-max(-30,min(30,z))))
            wt=pos_weight if y>0 else 1.0
            for i in range(5): grad[i]+=wt*(p-y)*x[i]
        n=max(len(feats),1)
        for i in range(5): w[i]-=lr*grad[i]/n
    return w

def predict_keep(pred, base, w, threshold):
    from sweep_symbol_added_candidate_reranker_p075 import max_overlap
    score=float(pred.get('score',0)); ov=max_overlap(pred,base); _w,_h,area,aspect=box_features(pred)
    x=[1.0,score,ov,math.log1p(area),aspect]
    z=sum(a*b for a,b in zip(w,x)); p=1/(1+math.exp(-max(-30,min(30,z))))
    return p>=threshold

def apply_model(v28,p076,w,threshold,max_add):
    out={}
    for row_id in sorted(set(v28)|set(p076)):
        base=list(v28.get(row_id,[])); kept=[]
        for pred in p076.get(row_id,[]):
            if pred.get('source_policy') is None: continue
            if predict_keep(pred,base,w,threshold): kept.append(pred)
        kept.sort(key=lambda p: float(p.get('score',0)), reverse=True)
        out[row_id]=base+kept[:max_add]
    return out

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--data',default=str(DEFAULT_DATA)); ap.add_argument('--yolo-dir',default=str(DEFAULT_YOLO_DIR)); ap.add_argument('--split',default='smoke_v30'); ap.add_argument('--v28-predictions',default=str(DEFAULT_V28)); ap.add_argument('--p076-predictions',default=str(DEFAULT_P076)); ap.add_argument('--output-json',default=str(DEFAULT_JSON)); ap.add_argument('--output-md',default=str(DEFAULT_MD)); ap.add_argument('--output-predictions',default=str(DEFAULT_PRED)); args=ap.parse_args()
    v28=read_predictions(Path(args.v28_predictions)); p076=read_predictions(Path(args.p076_predictions)); golds=read_exported_golds(Path(args.data),Path(args.yolo_dir),args.split,set(v28)|set(p076))
    rows=build_rows(v28,p076,golds)
    baseline=score(golds,v28,{r:[] for r in v28},{'labels':'all','areas':'all','score_min':1.1,'max_iou_with_v28':0.0,'max_add_per_page':0})
    full=score(golds,p076,{r:[] for r in p076},{'labels':'all','areas':'all','score_min':1.1,'max_iou_with_v28':0.0,'max_add_per_page':0})
    w=simple_logistic_train(rows)
    results=[]; gain=full['iou_0_30_recall']-baseline['iou_0_30_recall']
    for thr in [0.10,0.15,0.20,0.25,0.30,0.35,0.40,0.50,0.60]:
        for max_add in [3,5,10,20]:
            pred=apply_model(v28,p076,w,thr,max_add)
            m=score(golds,pred,{r:[] for r in pred},{'labels':'all','areas':'all','score_min':1.1,'max_iou_with_v28':0.0,'max_add_per_page':0})
            m['threshold']=thr; m['max_add_per_page']=max_add
            m['retained_gain_fraction']=round((m['iou_0_30_recall']-baseline['iou_0_30_recall'])/max(gain,1e-9),6)
            m['delta_vs_p076']={k:round(m[k]-full[k],6) for k in ['iou_0_30_recall','center_recall','candidate_inflation','precision']}
            results.append(m)
    feasible=[m for m in results if m['retained_gain_fraction']>=0.9 and m['precision']>full['precision'] and m['candidate_inflation']<=full['candidate_inflation']]
    feasible.sort(key=lambda m:(m['precision'],m['iou_0_30_recall'],-m['candidate_inflation']),reverse=True)
    best=feasible[0] if feasible else None
    if best: write_jsonl(Path(args.output_predictions), rows_from_predictions(apply_model(v28,p076,w,best['threshold'],best['max_add_per_page'])))
    report={'version':'symbol_metadata_selector_p081_smoke_v30','source_integrity':'metadata-only model uses runtime-safe features; offline labels for smoke training/eval only','dataset':{'rows':len(rows),'positive':sum(r['target'] for r in rows),'positive_rate':round(sum(r['target'] for r in rows)/max(len(rows),1),6)},'weights':w,'baseline_v28':baseline,'full_p076':full,'best_feasible':best,'all_results':results,'decision':'positive_smoke_candidate_validate_locked' if best else 'negative_metadata_only_no_clear_gain'}
    Path(args.output_json).parent.mkdir(parents=True,exist_ok=True); Path(args.output_json).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# P0-81 metadata-only selector smoke','',f"- dataset rows: `{report['dataset']['rows']}`, positives: `{report['dataset']['positive']}`, rate: `{report['dataset']['positive_rate']:.6f}`",f"- v28 IoU / inflation / precision: `{baseline['iou_0_30_recall']:.6f}` / `{baseline['candidate_inflation']:.6f}` / `{baseline['precision']:.6f}`",f"- P0-76 IoU / inflation / precision: `{full['iou_0_30_recall']:.6f}` / `{full['candidate_inflation']:.6f}` / `{full['precision']:.6f}`",f"- decision: `{report['decision']}`"]
    if best: lines += [f"- best IoU / inflation / precision: `{best['iou_0_30_recall']:.6f}` / `{best['candidate_inflation']:.6f}` / `{best['precision']:.6f}`",f"- retained gain: `{best['retained_gain_fraction']:.6f}`",f"- threshold/max_add: `{best['threshold']}` / `{best['max_add_per_page']}`"]
    Path(args.output_md).write_text('\n'.join(lines)+'\n')
    print(json.dumps({'decision':report['decision'],'dataset':report['dataset'],'best':None if best is None else {'iou':best['iou_0_30_recall'],'inflation':best['candidate_inflation'],'precision':best['precision'],'retained':best['retained_gain_fraction'],'threshold':best['threshold'],'max_add':best['max_add_per_page']}},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
