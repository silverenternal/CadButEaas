#!/usr/bin/env python3
"""Audit existing proposal overlays for P221b stair/equipment residual coverage."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fuse_symbol_p206g_with_p211_p212 import bbox_iou, load_p206g

ROOT=Path(__file__).resolve().parents[2]
BASE=ROOT/'reports/vlm/symbol_p222_p221a_frozen_overlay.jsonl'
CASES=ROOT/'reports/vlm/symbol_p221b_stair_equipment_residual_cases.jsonl'
OUT_JSON=ROOT/'reports/vlm/symbol_p221b_existing_proposal_coverage.json'
OUT_MD=ROOT/'reports/vlm/symbol_p221b_existing_proposal_coverage.md'
OVERLAYS=[
 'reports/vlm/symbol_p206g_p212_specialist_precision_repair_overlay.jsonl',
 'reports/vlm/symbol_p212_p213b_residual_fusion_overlay.jsonl',
 'reports/vlm/symbol_p213c_precision_gate_overlay.jsonl',
 'reports/vlm/symbol_recall_detector_p211_20k_yolov8s_page_predictions.jsonl',
 'reports/vlm/symbol_p211_20k_yolov8s_p206g_pages_sliced_predictions.jsonl',
 'reports/vlm/symbol_p211_20k_yolov8s_p206g_pages_sliced256_img768_predictions.jsonl',
]

def pred_label(p): return str(p.get('label', p.get('symbol_type','unknown')))
def pred_score(p): return float(p.get('score', p.get('confidence',0.0)) or 0.0)
def center(b): return ((float(b[0])+float(b[2]))/2,(float(b[1])+float(b[3]))/2)
def dist(a,b):
    ax,ay=center(a); bx,by=center(b); return ((ax-bx)**2+(ay-by)**2)**0.5

def load_preds(path):
    try:
        _rows,preds,_golds=load_p206g(path)
        return preds
    except Exception:
        preds={}
        for line in path.read_text().splitlines():
            if not line.strip(): continue
            row=json.loads(line); rid=str(row.get('id') or row.get('row_id') or row.get('page_id'))
            items=row.get('symbol_candidates') or row.get('predictions') or row.get('symbols') or []
            preds[rid]=items
        return preds

def main():
    cases=[json.loads(l) for l in CASES.read_text().splitlines() if l.strip()]
    results=[]
    for rel in OVERLAYS:
        path=ROOT/rel
        if not path.exists(): continue
        preds_by_row=load_preds(path)
        covered_iou=Counter(); covered_center=Counter(); by_cross_iou=Counter(); by_cross_center=Counter(); best_scores=[]; total=0
        for c in cases:
            rid=c['row_id']; label=c['label']; gbox=[float(v) for v in c['bbox']]; total+=1
            best_iou=0.0; best_dist=None; best_score=0.0
            for p in preds_by_row.get(rid,[]):
                if pred_label(p)!=label: continue
                pbox=[float(v) for v in p['bbox']]
                iou=bbox_iou(pbox,gbox); d=dist(pbox,gbox)
                if iou>best_iou:
                    best_iou=iou; best_score=pred_score(p)
                if best_dist is None or d<best_dist: best_dist=d
            cross=f"{label}|{c['bucket']}"
            if best_iou>=0.30:
                covered_iou[label]+=1; by_cross_iou[cross]+=1; best_scores.append(best_score)
            if best_dist is not None and best_dist<=16:
                covered_center[label]+=1; by_cross_center[cross]+=1
        results.append({'overlay':rel,'total_cases':total,'iou_covered':sum(covered_iou.values()),'center16_covered':sum(covered_center.values()),'iou_by_label':dict(covered_iou),'center16_by_label':dict(covered_center),'iou_by_label_bucket':dict(by_cross_iou),'center16_by_label_bucket':dict(by_cross_center),'score_stats':{'n':len(best_scores),'avg':sum(best_scores)/len(best_scores) if best_scores else 0,'max':max(best_scores) if best_scores else 0}})
    results.sort(key=lambda r:(r['iou_covered'],r['center16_covered']), reverse=True)
    OUT_JSON.write_text(json.dumps({'id':'P221b_existing_proposal_coverage','cases':len(cases),'results':results},ensure_ascii=False,indent=2)+'\n')
    lines=['# P221b Existing Proposal Coverage','','| Overlay | IoU>=0.30 covered | Center<=16 covered | IoU by label | Center by label |','|---|---:|---:|---|---|']
    for r in results:
        lines.append(f"| `{r['overlay']}` | {r['iou_covered']}/{r['total_cases']} | {r['center16_covered']}/{r['total_cases']} | `{json.dumps(r['iou_by_label'], ensure_ascii=False)}` | `{json.dumps(r['center16_by_label'], ensure_ascii=False)}` |")
    lines += ['', '## Interpretation','- High center coverage but low IoU means a refiner/subcandidate rule may work.', '- Low center and IoU coverage means a new proposal branch or training data is needed.']
    OUT_MD.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(OUT_MD),'best':results[0] if results else None},ensure_ascii=False,indent=2)[:4000])
if __name__=='__main__': main()
