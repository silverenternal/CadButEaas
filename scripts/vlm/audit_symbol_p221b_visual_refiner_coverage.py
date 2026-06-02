#!/usr/bin/env python3
"""Audit old visual refiner streams for P221b residual coverage over P222."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fuse_symbol_p206g_with_p211_p212 import bbox_iou, load_p206g

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'reports/vlm/symbol_p222_p221a_frozen_overlay.jsonl'
CASES=ROOT/'reports/vlm/symbol_p221b_stair_equipment_residual_cases.jsonl'
OUT_JSON=ROOT/'reports/vlm/symbol_p221b_visual_refiner_coverage.json'
OUT_MD=ROOT/'reports/vlm/symbol_p221b_visual_refiner_coverage.md'
STREAMS=[
 'reports/vlm/symbol_visual_box_refiner_v46_enhanced_features_page_locked_predictions.jsonl',
 'reports/vlm/symbol_visual_box_refiner_v45_quality_policy_page_locked_predictions.jsonl',
 'reports/vlm/symbol_visual_box_refiner_v44_fulltarget_page_locked_predictions.jsonl',
 'reports/vlm/symbol_visual_box_refiner_v43_hardcases_page_locked_predictions.jsonl',
 'reports/vlm/symbol_visual_box_refiner_v42_page_locked_predictions.jsonl',
]

def pred_label(p): return str(p.get('label', p.get('symbol_type','unknown')))
def pred_score(p): return float(p.get('score', p.get('confidence',0.0)) or 0.0)
def center(b): return ((float(b[0])+float(b[2]))/2,(float(b[1])+float(b[3]))/2)
def dist(a,b):
    ax,ay=center(a); bx,by=center(b); return ((ax-bx)**2+(ay-by)**2)**0.5

def load_stream(path):
    out={}
    for line in path.read_text().splitlines():
        if not line.strip(): continue
        row=json.loads(line); rid=str(row.get('row_id') or row.get('id') or row.get('page_id'))
        preds=row.get('predicted_symbols') or row.get('symbol_candidates') or row.get('predictions') or []
        norm=[]
        for p in preds:
            q=dict(p)
            if 'symbol_type' not in q and 'label' in q: q['symbol_type']=q['label']
            if 'label' not in q and 'symbol_type' in q: q['label']=q['symbol_type']
            if 'confidence' not in q and 'score' in q: q['confidence']=q['score']
            if 'score' not in q and 'confidence' in q: q['score']=q['confidence']
            norm.append(q)
        out[rid]=norm
    return out

def main():
    cases=[json.loads(l) for l in CASES.read_text().splitlines() if l.strip()]
    results=[]
    for rel in STREAMS:
        path=ROOT/rel
        if not path.exists(): continue
        preds_by=load_stream(path)
        iou_cov=Counter(); center_cov=Counter(); cross_iou=Counter(); cross_center=Counter(); refined_cov=Counter(); total=0
        for c in cases:
            rid=c['row_id']; label=c['label']; gbox=[float(v) for v in c['bbox']]; total+=1
            best_iou=0; best_dist=None; best_refined=False
            for p in preds_by.get(rid,[]):
                if pred_label(p)!=label: continue
                pbox=[float(v) for v in p['bbox']]
                iou=bbox_iou(pbox,gbox); d=dist(pbox,gbox)
                if iou>best_iou:
                    best_iou=iou; best_refined=bool(p.get('refined_by_v38') or p.get('refined_by_v41') or p.get('refined'))
                if best_dist is None or d<best_dist: best_dist=d
            cross=f"{label}|{c['bucket']}"
            if best_iou>=0.30:
                iou_cov[label]+=1; cross_iou[cross]+=1
                if best_refined: refined_cov[label]+=1
            if best_dist is not None and best_dist<=16:
                center_cov[label]+=1; cross_center[cross]+=1
        results.append({'stream':rel,'total_cases':total,'iou_covered':sum(iou_cov.values()),'center16_covered':sum(center_cov.values()),'iou_by_label':dict(iou_cov),'center16_by_label':dict(center_cov),'iou_by_label_bucket':dict(cross_iou),'center16_by_label_bucket':dict(cross_center),'refined_iou_by_label':dict(refined_cov)})
    results.sort(key=lambda r:(r['iou_covered'],r['center16_covered']), reverse=True)
    OUT_JSON.write_text(json.dumps({'id':'P221b_visual_refiner_coverage','cases':len(cases),'results':results},ensure_ascii=False,indent=2)+'\n')
    lines=['# P221b Visual Refiner Coverage','','| Stream | IoU>=0.30 | Center<=16 | IoU by label | Center by label |','|---|---:|---:|---|---|']
    for r in results:
        lines.append(f"| `{r['stream']}` | {r['iou_covered']}/{r['total_cases']} | {r['center16_covered']}/{r['total_cases']} | `{json.dumps(r['iou_by_label'], ensure_ascii=False)}` | `{json.dumps(r['center16_by_label'], ensure_ascii=False)}` |")
    lines += ['', '## Interpretation','- If visual streams cover more stair/equipment than P213b, they can seed P221b proposals.', '- If not, prioritize new stair specialist data/training.']
    OUT_MD.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(OUT_MD),'best':results[0] if results else None},ensure_ascii=False,indent=2)[:4000])
if __name__=='__main__': main()
