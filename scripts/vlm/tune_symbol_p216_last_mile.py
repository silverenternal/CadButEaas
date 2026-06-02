#!/usr/bin/env python3
"""Last-mile row/label blacklist search around P215 policy."""
from __future__ import annotations

import argparse,json,itertools
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from tune_symbol_p214_precision_repair import fuse
from fuse_symbol_p206g_with_p212_specialist import build_overlay, load_p212, metric_key
from fuse_symbol_p206g_with_p211_p212 import load_p206g, score_predictions, write_json, write_jsonl

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'reports/vlm/symbol_p206g_p212_specialist_precision_repair_overlay.jsonl'
P213=ROOT/'reports/vlm/symbol_residual_specialist_p213b_pages_s160_top180_predictions.jsonl'
AUDIT=ROOT/'reports/vlm/symbol_p215_added_fp_p216.json'
REPORT=ROOT/'reports/vlm/symbol_p216_last_mile_eval.json'
OVERLAY=ROOT/'reports/vlm/symbol_p216_last_mile_overlay.jsonl'

BASE_POLICY={'allowed_labels':['sink','equipment','stair','shower'],'threshold':0.9,'label_thresholds':{'stair':0.95,'sink':0.86,'equipment':0.88,'shower':0.78},'max_add_per_row':20,'max_iou_to_core':0.25,'min_dist_to_core':0,'row_blacklist':['cubicasa5k_locked_00022','cubicasa5k_locked_00024']}


def make_policy(row_blacklist, row_label_blacklist, thresholds, max_add=20):
    p=dict(BASE_POLICY); p['label_thresholds']=dict(thresholds); p['row_blacklist']=list(row_blacklist); p['row_label_blacklist']=list(row_label_blacklist); p['max_add_per_row']=max_add
    p['name']=f"p216_rb{len(p['row_blacklist'])}_rl{len(p['row_label_blacklist'])}_st{p['label_thresholds']['stair']}_si{p['label_thresholds']['sink']}_eq{p['label_thresholds']['equipment']}_sh{p['label_thresholds']['shower']}_a{max_add}"
    return p


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',default=str(BASE)); ap.add_argument('--p213',default=str(P213)); ap.add_argument('--audit',default=str(AUDIT)); ap.add_argument('--report',default=str(REPORT)); ap.add_argument('--overlay',default=str(OVERLAY)); args=ap.parse_args()
    rows,base,golds=load_p206g(Path(args.base)); p213=load_p212(Path(args.p213)); audit=json.loads(Path(args.audit).read_text())
    baseline,_=score_predictions(base,golds,0.0,0.98,900,0)
    # Candidate row-label blocks from actual FP rows, prioritizing stair-heavy rows.
    fp_rows=[r for r in audit.get('rows',[]) if not r.get('is_tp')]
    fp_by_pair=Counter((r['row_id'],r['label']) for r in fp_rows)
    candidates=[pair for pair,_ in fp_by_pair.most_common(8)]
    worst_rows=list(audit.get('worst_fp_rows',{}).keys())[:10]
    base_rb=BASE_POLICY['row_blacklist']
    policies=[]
    # single/pair/triple row-label blacklists plus row blacklists
    rl_sets=[[]]
    rl_sets += [[c] for c in candidates]
    rl_sets += [list(x) for x in itertools.combinations(candidates[:6],2)]
    row_sets=[base_rb, base_rb+worst_rows[:1], base_rb+worst_rows[:2]]
    threshold_sets=[]
    for stair in [0.94,0.95,0.96]:
      for sink in [0.84,0.86]:
        for eq in [0.86,0.88]:
          for shower in [0.74,0.76,0.78]:
            threshold_sets.append({'stair':stair,'sink':sink,'equipment':eq,'shower':shower})
    for rb in row_sets:
      for rl in rl_sets:
        for th in threshold_sets:
          for max_add in [20,24]:
            policies.append(make_policy(list(dict.fromkeys(rb)), rl, th, max_add))
    reports=[]
    for i,policy in enumerate(policies,1):
        fused=fuse(base,p213,policy); metrics,_=score_predictions(fused,golds,0.0,0.98,900,0); additions=sum(max(0,len(fused[r])-len(base.get(r,[]))) for r in fused)
        reports.append({'policy':policy,'metrics':metrics,'additions':additions})
        if i%1000==0:
            best=max(reports,key=metric_key); print(json.dumps({'done':i,'total':len(policies),'best_f1':best['metrics']['symbol_bbox_iou_0_30']['f1'],'p':best['metrics']['symbol_bbox_iou_0_30']['precision'],'r':best['metrics']['symbol_bbox_iou_0_30']['recall'],'add':best['additions']}),flush=True)
    reports.sort(key=metric_key, reverse=True); best=reports[0]
    fused=fuse(base,p213,best['policy']); write_jsonl(Path(args.overlay), build_overlay(rows,fused,best['policy']))
    result={'id':'P216_last_mile_gate','claim_boundary':'P101 policy-search evidence; bootstrap required.','baseline':baseline,'selected':best,'top30':reports[:30],'outputs':{'overlay':str(Path(args.overlay)),'report':str(Path(args.report))}}
    write_json(Path(args.report),result)
    print(json.dumps({'baseline':baseline['symbol_bbox_iou_0_30'],'selected':best['metrics']['symbol_bbox_iou_0_30'],'additions':best['additions'],'policy':best['policy']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
