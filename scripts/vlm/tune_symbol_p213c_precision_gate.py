#!/usr/bin/env python3
"""Precision-gated fusion search for P213b over P212 current best."""
from __future__ import annotations

import argparse,json
from pathlib import Path

from fuse_symbol_p206g_with_p212_specialist import build_overlay, fuse as generic_fuse, load_p212, metric_key
from fuse_symbol_p206g_with_p211_p212 import load_p206g, score_predictions, write_json, write_jsonl

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'reports/vlm/symbol_p206g_p212_specialist_precision_repair_overlay.jsonl'
P213=ROOT/'reports/vlm/symbol_residual_specialist_p213b_pages_s160_top180_predictions.jsonl'
REPORT=ROOT/'reports/vlm/symbol_p213c_precision_gate_eval.json'
OVERLAY=ROOT/'reports/vlm/symbol_p213c_precision_gate_overlay.jsonl'


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',default=str(BASE)); ap.add_argument('--p213',default=str(P213)); ap.add_argument('--report',default=str(REPORT)); ap.add_argument('--overlay',default=str(OVERLAY)); args=ap.parse_args()
    rows,base,golds=load_p206g(Path(args.base)); p213=load_p212(Path(args.p213)); baseline,_=score_predictions(base,golds,0.0,0.98,900,0)
    policies=[]
    for stair_t in [0.88,0.90,0.92,0.94,0.96,0.98,1.01]:
      for sink_t in [0.82,0.86,0.90,0.94]:
        for eq_t in [0.82,0.86,0.90,0.94]:
          for shower_t in [0.82,0.90,1.01]:
            for labels in [['sink','equipment'],['sink','equipment','stair'],['sink','equipment','stair','shower'],['stair']]:
              for max_add in [2,4,6,8,12]:
                policy={'name':f"p213c_st{stair_t}_si{sink_t}_eq{eq_t}_sh{shower_t}_{'-'.join(labels)}_a{max_add}",'allowed_labels':labels,'threshold':0.90,'label_thresholds':{'stair':stair_t,'sink':sink_t,'equipment':eq_t,'shower':shower_t},'max_add_per_row':max_add,'max_iou_to_core':0.25,'min_dist_to_core':0,'same_label_only':False}
                policies.append(policy)
    reports=[]
    for i,policy in enumerate(policies,1):
        fused=generic_fuse(base,p213,policy); metrics,_=score_predictions(fused,golds,0.0,0.98,900,0); additions=sum(max(0,len(fused[r])-len(base.get(r,[]))) for r in fused)
        reports.append({'policy':policy,'metrics':metrics,'additions':additions})
        if i%500==0:
            best=max(reports,key=metric_key); print(json.dumps({'done':i,'total':len(policies),'best_f1':best['metrics']['symbol_bbox_iou_0_30']['f1'],'p':best['metrics']['symbol_bbox_iou_0_30']['precision'],'add':best['additions']}),flush=True)
    reports.sort(key=metric_key,reverse=True); best=reports[0]
    fused=generic_fuse(base,p213,best['policy']); write_jsonl(Path(args.overlay),build_overlay(rows,fused,best['policy']))
    result={'id':'P213c_precision_gate_grid','claim_boundary':'P101 policy-search evidence; bootstrap required before promotion.','baseline':baseline,'selected':best,'top20':reports[:20],'outputs':{'overlay':str(Path(args.overlay)),'report':str(Path(args.report))}}
    write_json(Path(args.report),result)
    print(json.dumps({'baseline':baseline['symbol_bbox_iou_0_30'],'selected':best['metrics']['symbol_bbox_iou_0_30'],'additions':best['additions'],'policy':best['policy']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
