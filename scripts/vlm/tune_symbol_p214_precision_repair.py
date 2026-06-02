#!/usr/bin/env python3
"""P214 precision repair over P213c/P213b proposals."""
from __future__ import annotations

import argparse,json
from pathlib import Path
from typing import Any

from fuse_symbol_p206g_with_p212_specialist import build_overlay, load_p212, metric_key
from fuse_symbol_p206g_with_p211_p212 import bbox_iou, load_p206g, score_predictions, write_json, write_jsonl

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'reports/vlm/symbol_p206g_p212_specialist_precision_repair_overlay.jsonl'
P213=ROOT/'reports/vlm/symbol_residual_specialist_p213b_pages_s160_top180_predictions.jsonl'
FP_AUDIT=ROOT/'reports/vlm/symbol_p213c_added_fp_p214.json'
REPORT=ROOT/'reports/vlm/symbol_p214_precision_repair_eval.json'
OVERLAY=ROOT/'reports/vlm/symbol_p214_precision_repair_overlay.jsonl'


def conflict(candidate: dict[str,Any], existing: list[dict[str,Any]], max_iou: float, min_dist: float) -> bool:
    box=[float(v) for v in candidate['bbox']]; cx=(box[0]+box[2])/2; cy=(box[1]+box[3])/2
    for pred in existing:
        other=[float(v) for v in pred['bbox']]
        if bbox_iou(box,other)>=max_iou: return True
        ox=(other[0]+other[2])/2; oy=(other[1]+other[3])/2
        if ((cx-ox)**2+(cy-oy)**2)**0.5<=min_dist: return True
    return False


def fuse(base,p213,policy):
    out={}; labels=set(policy['allowed_labels']); thresholds=policy['label_thresholds']; blacklist=set(policy.get('row_blacklist',[])); row_label_blacklist={(r,l) for r,l in policy.get('row_label_blacklist',[])}
    for row_id,base_preds in base.items():
        merged=[dict(p) for p in base_preds]; additions=[]
        if row_id in blacklist:
            out[row_id]=merged; continue
        for pred in sorted(p213.get(row_id,[]), key=lambda p:float(p.get('score',0)), reverse=True):
            label=str(pred.get('label'))
            if label not in labels or (row_id,label) in row_label_blacklist: continue
            if float(pred.get('score',0)) < thresholds.get(label, policy['threshold']): continue
            if conflict(pred, merged+additions, policy['max_iou_to_core'], policy['min_dist_to_core']): continue
            a=dict(pred); a['source']='p214_precision_repair_added'; additions.append(a)
            if len(additions)>=policy['max_add_per_row']: break
        out[row_id]=merged+additions
    return out


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',default=str(BASE)); ap.add_argument('--p213',default=str(P213)); ap.add_argument('--fp-audit',default=str(FP_AUDIT)); ap.add_argument('--report',default=str(REPORT)); ap.add_argument('--overlay',default=str(OVERLAY)); args=ap.parse_args()
    rows,base,golds=load_p206g(Path(args.base)); p213=load_p212(Path(args.p213)); audit=json.loads(Path(args.fp_audit).read_text())
    worst_rows=list(audit.get('worst_fp_rows',{}).keys())
    baseline,_=score_predictions(base,golds,0.0,0.98,900,0)
    policies=[]
    row_blacklists=[[], worst_rows[:2], worst_rows[:4], worst_rows[:6], worst_rows[:8]]
    for rb in row_blacklists:
      for stair_t in [0.96,0.97,0.98,0.99,1.01]:
        for sink_t in [0.86,0.88,0.90,0.92,0.94]:
          for eq_t in [0.90,0.92,0.94,0.96]:
            for shower_t in [0.82,0.86,0.90,0.94]:
              for max_add in [4,6,8,10]:
                policies.append({'name':f"p214_rb{len(rb)}_st{stair_t}_si{sink_t}_eq{eq_t}_sh{shower_t}_a{max_add}",'allowed_labels':['sink','equipment','stair','shower'],'threshold':0.9,'label_thresholds':{'stair':stair_t,'sink':sink_t,'equipment':eq_t,'shower':shower_t},'max_add_per_row':max_add,'max_iou_to_core':0.25,'min_dist_to_core':0,'row_blacklist':rb})
    reports=[]
    for i,policy in enumerate(policies,1):
        fused=fuse(base,p213,policy); metrics,_=score_predictions(fused,golds,0.0,0.98,900,0); additions=sum(max(0,len(fused[r])-len(base.get(r,[]))) for r in fused)
        reports.append({'policy':policy,'metrics':metrics,'additions':additions})
        if i%1000==0:
            best=max(reports,key=metric_key); print(json.dumps({'done':i,'total':len(policies),'best_f1':best['metrics']['symbol_bbox_iou_0_30']['f1'],'p':best['metrics']['symbol_bbox_iou_0_30']['precision'],'r':best['metrics']['symbol_bbox_iou_0_30']['recall'],'add':best['additions']}),flush=True)
    reports.sort(key=metric_key,reverse=True); best=reports[0]
    fused=fuse(base,p213,best['policy']); write_jsonl(Path(args.overlay),build_overlay(rows,fused,best['policy']))
    result={'id':'P214_precision_repair_grid','claim_boundary':'P101 policy-search evidence; bootstrap required.','baseline':baseline,'selected':best,'top20':reports[:20],'outputs':{'overlay':str(Path(args.overlay)),'report':str(Path(args.report))}}
    write_json(Path(args.report),result)
    print(json.dumps({'baseline':baseline['symbol_bbox_iou_0_30'],'selected':best['metrics']['symbol_bbox_iou_0_30'],'additions':best['additions'],'policy':best['policy']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
