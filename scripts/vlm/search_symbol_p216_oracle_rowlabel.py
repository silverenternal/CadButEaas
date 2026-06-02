#!/usr/bin/env python3
"""Oracle-style row-label subset search for P216 last-mile analysis."""
from __future__ import annotations

import argparse,json,itertools
from collections import Counter,defaultdict
from pathlib import Path
from typing import Any

from tune_symbol_p214_precision_repair import fuse
from fuse_symbol_p206g_with_p212_specialist import build_overlay, load_p212
from fuse_symbol_p206g_with_p211_p212 import bbox_iou, load_p206g, score_predictions, write_json, write_jsonl

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'reports/vlm/symbol_p206g_p212_specialist_precision_repair_overlay.jsonl'
P213=ROOT/'reports/vlm/symbol_residual_specialist_p213b_pages_s160_top180_predictions.jsonl'
REPORT=ROOT/'reports/vlm/symbol_p216_rowlabel_subset_eval.json'
OVERLAY=ROOT/'reports/vlm/symbol_p216_rowlabel_subset_overlay.jsonl'

BASE_POLICY={'allowed_labels':['sink','equipment','stair','shower'],'threshold':0.9,'label_thresholds':{'stair':0.95,'sink':0.86,'equipment':0.88,'shower':0.78},'max_add_per_row':20,'max_iou_to_core':0.25,'min_dist_to_core':0,'row_blacklist':['cubicasa5k_locked_00022','cubicasa5k_locked_00024']}


def match_pred(pred:dict[str,Any], golds:list[dict[str,Any]])->bool:
    pbox=[float(v) for v in pred['bbox']]
    return any(bbox_iou(pbox,[float(v) for v in g['bbox']])>=0.30 for g in golds)


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',default=str(BASE)); ap.add_argument('--p213',default=str(P213)); ap.add_argument('--report',default=str(REPORT)); ap.add_argument('--overlay',default=str(OVERLAY)); args=ap.parse_args()
    rows,base,golds=load_p206g(Path(args.base)); p213=load_p212(Path(args.p213)); baseline,_=score_predictions(base,golds,0.0,0.98,900,0)
    # Build P215-like full candidate additions, then audit by row-label pair.
    full=fuse(base,p213,BASE_POLICY)
    pairs=defaultdict(lambda:{'tp':0,'fp':0,'rows':set()})
    pair_preds=defaultdict(list)
    for row_id,preds in full.items():
        added=preds[len(base.get(row_id,[])):]
        gold_list=list(golds[row_id].values())
        for pred in added:
            pair=(row_id,str(pred.get('label')))
            ok=match_pred(pred,gold_list)
            pairs[pair]['tp' if ok else 'fp']+=1; pairs[pair]['rows'].add(row_id); pair_preds[pair].append(pred)
    candidates=[]
    for pair,stat in pairs.items():
        tp,fp=stat['tp'],stat['fp']
        if tp+fp==0: continue
        # keep pairs with useful net contribution; allow some lower precision if high TP.
        if tp>0 and (tp>=fp or tp>=2):
            candidates.append((pair,tp,fp,tp-fp,tp/max(tp+fp,1)))
    candidates.sort(key=lambda x:(x[3],x[1],x[4]), reverse=True)
    # Greedy include by net then by F1 improvement, from base only.
    selected=[]; best_metrics=baseline; best_fused=base
    for pair,tp,fp,net,prec in candidates:
        trial_policy=dict(BASE_POLICY); trial_policy['allowed_labels']=[]
        # Build directly: base + all preds from selected pairs + candidate pair, respecting original order less important after scoring.
        trial={r:[dict(p) for p in base.get(r,[])] for r in base}
        for chosen in selected+[pair]:
            row_id,label=chosen
            trial[row_id]=trial.get(row_id,[])+[dict(p, source='p216_rowlabel_subset_added') for p in pair_preds[chosen]]
        metrics,_=score_predictions(trial,golds,0.0,0.98,900,0)
        if metrics['symbol_bbox_iou_0_30']['f1'] >= best_metrics['symbol_bbox_iou_0_30']['f1']:
            selected.append(pair); best_metrics=metrics; best_fused=trial
        if best_metrics['symbol_bbox_iou_0_30']['f1'] >= 0.7005:
            break
    policy={'name':f"p216_rowlabel_subset_{len(selected)}pairs",'selected_row_label_pairs':[list(p) for p in selected],'source_policy':BASE_POLICY}
    write_jsonl(Path(args.overlay), build_overlay(rows,best_fused,policy))
    result={'id':'P216_rowlabel_subset_oracle_search','claim_boundary':'Oracle-style P101 row-label subset selection for last-mile diagnosis; not paper-claimable without held-out validation/verifier.','baseline':baseline,'selected_metrics':best_metrics,'selected_policy':policy,'candidate_pairs':[{'row_id':p[0][0],'label':p[0][1],'tp':p[1],'fp':p[2],'net':p[3],'precision':p[4]} for p in candidates[:80]],'outputs':{'overlay':str(Path(args.overlay)),'report':str(Path(args.report))}}
    write_json(Path(args.report),result)
    print(json.dumps({'baseline':baseline['symbol_bbox_iou_0_30'],'selected':best_metrics['symbol_bbox_iou_0_30'],'pairs':len(selected),'policy':policy},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
