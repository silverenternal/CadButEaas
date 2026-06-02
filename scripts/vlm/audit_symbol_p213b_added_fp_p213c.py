#!/usr/bin/env python3
"""Audit added P213b proposals over P212 current best."""
from __future__ import annotations

import argparse, json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fuse_symbol_p206g_with_p211_p212 import bbox_iou, area_bucket, load_p206g, write_json

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'reports/vlm/symbol_p206g_p212_specialist_precision_repair_overlay.jsonl'
FUSED=ROOT/'reports/vlm/symbol_p212_p213b_residual_fusion_overlay.jsonl'
REPORT=ROOT/'reports/vlm/symbol_p213b_added_fp_p213c.json'
MD=ROOT/'reports/vlm/symbol_p213b_added_fp_p213c.md'


def match_pred(pred: dict[str,Any], golds: list[dict[str,Any]]) -> tuple[bool,str,str,float]:
    best_iou=0.0; best_label=''; best_bucket=''
    pbox=[float(v) for v in pred['bbox']]
    for gold in golds:
        gbox=[float(v) for v in gold['bbox']]
        iou=bbox_iou(pbox,gbox)
        if iou>best_iou:
            best_iou=iou; best_label=str(gold.get('label')); best_bucket=area_bucket(gbox)
    return best_iou>=0.30, best_label, best_bucket, best_iou


def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--base',default=str(BASE)); ap.add_argument('--fused',default=str(FUSED)); ap.add_argument('--report',default=str(REPORT)); ap.add_argument('--md',default=str(MD)); args=ap.parse_args()
    _rows,base,golds=load_p206g(Path(args.base)); _frows,fused,_fgolds=load_p206g(Path(args.fused))
    rows=[]; counts=Counter(); tp_by=Counter(); fp_by=Counter(); tp_bucket=Counter(); fp_bucket=Counter(); score_bins=Counter(); row_fp=Counter()
    for row_id, preds in fused.items():
        base_len=len(base.get(row_id,[])); added=preds[base_len:]
        gold_list=list(golds[row_id].values())
        for pred in added:
            label=str(pred.get('label')); score=float(pred.get('score') or 0.0)
            ok,glabel,gbucket,iou=match_pred(pred,gold_list)
            bucket=f'{int(score*10)/10:.1f}-{int(score*10)/10+0.1:.1f}'
            score_bins[(label,bucket,'TP' if ok else 'FP')]+=1
            counts['tp' if ok else 'fp']+=1
            if ok:
                tp_by[label]+=1; tp_bucket[gbucket]+=1
            else:
                fp_by[label]+=1; fp_bucket['unknown']+=1; row_fp[row_id]+=1
            rows.append({'row_id':row_id,'label':label,'score':score,'bbox':pred.get('bbox'),'is_tp':ok,'gold_label':glabel,'gold_bucket':gbucket,'best_iou':round(iou,4),'tile_id':pred.get('tile_id')})
    rows.sort(key=lambda r:(not r['is_tp'], r['label'], -r['score']))
    report={'id':'P213c_added_fp_audit','added_total':len(rows),'tp':counts['tp'],'fp':counts['fp'],'precision':round(counts['tp']/max(len(rows),1),6),'tp_by_label':dict(tp_by),'fp_by_label':dict(fp_by),'tp_by_bucket':dict(tp_bucket),'worst_fp_rows':dict(row_fp.most_common(20)),'score_bins':{str(k):v for k,v in score_bins.items()},'rows':rows,'claim_boundary':'Audit of P213b additions over P212 current best; gold used only for offline verifier/gate design.'}
    write_json(Path(args.report),report)
    lines=['# P213c Added FP Audit','',f"- Added total: {len(rows)}",f"- TP/FP: {counts['tp']} / {counts['fp']}",f"- Added precision: {report['precision']:.6f}",f"- TP by label: `{json.dumps(dict(tp_by), ensure_ascii=False)}`",f"- FP by label: `{json.dumps(dict(fp_by), ensure_ascii=False)}`",f"- Worst FP rows: `{json.dumps(dict(row_fp.most_common(10)), ensure_ascii=False)}`",'', '## Claim Boundary', report['claim_boundary']]
    Path(args.md).write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print(json.dumps({k:report[k] for k in ['added_total','tp','fp','precision','tp_by_label','fp_by_label','worst_fp_rows']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
