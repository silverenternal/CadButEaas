#!/usr/bin/env python3
"""Fuse P213b residual specialist proposals over P212 current best overlay."""
from __future__ import annotations

import argparse, json
from pathlib import Path
from typing import Any

from fuse_symbol_p206g_with_p212_specialist import build_overlay, conflict, fuse as generic_fuse, load_p212, metric_key
from fuse_symbol_p206g_with_p211_p212 import load_p206g, score_predictions, write_json, write_jsonl

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'reports/vlm/symbol_p206g_p212_specialist_precision_repair_overlay.jsonl'
P213=ROOT/'reports/vlm/symbol_residual_specialist_p213b_pages_s160_top180_predictions.jsonl'
REPORT=ROOT/'reports/vlm/symbol_p212_p213b_residual_fusion_eval.json'
OVERLAY=ROOT/'reports/vlm/symbol_p212_p213b_residual_fusion_overlay.jsonl'


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',default=str(BASE)); ap.add_argument('--p213',default=str(P213)); ap.add_argument('--report',default=str(REPORT)); ap.add_argument('--overlay',default=str(OVERLAY))
    args=ap.parse_args()
    rows,base,golds=load_p206g(Path(args.base)); p213=load_p212(Path(args.p213))
    baseline,_=score_predictions(base,golds,0.0,0.98,900,0)
    policies=[]
    label_sets=[['stair'],['sink'],['sink','stair'],['sink','stair','equipment'],['sink','stair','equipment','shower'],['stair','equipment']]
    for labels in label_sets:
      for threshold in [0.55,0.6,0.65,0.7,0.75,0.8,0.85]:
        for max_add in [3,5,8,12,20,30]:
          for dist in [0,2,4,8,12]:
            policies.append({'name':f"p213b_{'-'.join(labels)}_t{threshold}_a{max_add}_d{dist}",'allowed_labels':labels,'threshold':threshold,'label_thresholds':{},'max_add_per_row':max_add,'max_iou_to_core':0.25,'min_dist_to_core':dist,'same_label_only':False})
    reports=[]
    for i,policy in enumerate(policies,1):
        fused=generic_fuse(base,p213,policy)
        metrics,_=score_predictions(fused,golds,0.0,0.98,900,0)
        additions=sum(max(0,len(fused[r])-len(base.get(r,[]))) for r in fused)
        reports.append({'policy':policy,'metrics':metrics,'additions':additions})
        if i%200==0:
            best=max(reports,key=metric_key); print(json.dumps({'done':i,'total':len(policies),'best_f1':best['metrics']['symbol_bbox_iou_0_30']['f1'],'add':best['additions']}),flush=True)
    reports.sort(key=metric_key,reverse=True); best=reports[0]
    fused=generic_fuse(base,p213,best['policy'])
    write_jsonl(Path(args.overlay), build_overlay(rows,fused,best['policy']))
    result={'id':'P213b_residual_fusion_grid','claim_boundary':'P101 policy-search fusion over P212 current best; requires bootstrap before promotion.','baseline':baseline,'selected':best,'top20':reports[:20],'outputs':{'overlay':str(Path(args.overlay)),'report':str(Path(args.report))}}
    write_json(Path(args.report),result)
    print(json.dumps({'baseline':baseline['symbol_bbox_iou_0_30'],'selected':best['metrics']['symbol_bbox_iou_0_30'],'additions':best['additions'],'policy':best['policy']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
