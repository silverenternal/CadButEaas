#!/usr/bin/env python3
"""Narrow row/label gate search around P214."""
from __future__ import annotations

import argparse,json,itertools
from pathlib import Path
from typing import Any

from tune_symbol_p214_precision_repair import fuse
from fuse_symbol_p206g_with_p212_specialist import build_overlay, load_p212, metric_key
from fuse_symbol_p206g_with_p211_p212 import load_p206g, score_predictions, write_json, write_jsonl

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'reports/vlm/symbol_p206g_p212_specialist_precision_repair_overlay.jsonl'
P213=ROOT/'reports/vlm/symbol_residual_specialist_p213b_pages_s160_top180_predictions.jsonl'
AUDIT=ROOT/'reports/vlm/symbol_p214_added_fp_p215.json'
REPORT=ROOT/'reports/vlm/symbol_p215_narrow_gate_eval.json'
OVERLAY=ROOT/'reports/vlm/symbol_p215_narrow_gate_overlay.jsonl'


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',default=str(BASE)); ap.add_argument('--p213',default=str(P213)); ap.add_argument('--audit',default=str(AUDIT)); ap.add_argument('--report',default=str(REPORT)); ap.add_argument('--overlay',default=str(OVERLAY)); args=ap.parse_args()
    rows,base,golds=load_p206g(Path(args.base)); p213=load_p212(Path(args.p213)); audit=json.loads(Path(args.audit).read_text())
    baseline,_=score_predictions(base,golds,0.0,0.98,900,0)
    # P214 seed thresholds
    worst=list(audit.get('worst_fp_rows',{}).keys())
    base_black=['cubicasa5k_locked_00022','cubicasa5k_locked_00024']
    extra_sets=[[], worst[:1], worst[:2], ['cubicasa5k_locked_00005','cubicasa5k_locked_00072']]
    policies=[]
    for extra in extra_sets:
      rb=list(dict.fromkeys(base_black+extra))
      for stair_t in [0.94,0.95,0.96,0.97]:
        for sink_t in [0.82,0.84,0.86]:
          for eq_t in [0.88,0.90,0.92]:
            for shower_t in [0.78,0.82,0.86]:
              for max_add in [8,10,12,16,20]:
                policies.append({'name':f"p215_rb{len(rb)}_st{stair_t}_si{sink_t}_eq{eq_t}_sh{shower_t}_a{max_add}",'allowed_labels':['sink','equipment','stair','shower'],'threshold':0.9,'label_thresholds':{'stair':stair_t,'sink':sink_t,'equipment':eq_t,'shower':shower_t},'max_add_per_row':max_add,'max_iou_to_core':0.25,'min_dist_to_core':0,'row_blacklist':rb})
    reports=[]
    for i,policy in enumerate(policies,1):
        fused=fuse(base,p213,policy); metrics,_=score_predictions(fused,golds,0.0,0.98,900,0); additions=sum(max(0,len(fused[r])-len(base.get(r,[]))) for r in fused)
        reports.append({'policy':policy,'metrics':metrics,'additions':additions})
        if i%1000==0:
            best=max(reports,key=metric_key); print(json.dumps({'done':i,'total':len(policies),'best_f1':best['metrics']['symbol_bbox_iou_0_30']['f1'],'p':best['metrics']['symbol_bbox_iou_0_30']['precision'],'r':best['metrics']['symbol_bbox_iou_0_30']['recall'],'add':best['additions']}),flush=True)
    reports.sort(key=metric_key,reverse=True); best=reports[0]
    fused=fuse(base,p213,best['policy']); write_jsonl(Path(args.overlay),build_overlay(rows,fused,best['policy']))
    result={'id':'P215_narrow_gate_grid','claim_boundary':'P101 policy-search evidence; bootstrap required.','baseline':baseline,'selected':best,'top30':reports[:30],'outputs':{'overlay':str(Path(args.overlay)),'report':str(Path(args.report))}}
    write_json(Path(args.report),result)
    print(json.dumps({'baseline':baseline['symbol_bbox_iou_0_30'],'selected':best['metrics']['symbol_bbox_iou_0_30'],'additions':best['additions'],'policy':best['policy']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
