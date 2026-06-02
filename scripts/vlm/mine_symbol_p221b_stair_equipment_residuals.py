#!/usr/bin/env python3
"""Mine stair/equipment residuals after P222 frozen P221a baseline."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fuse_symbol_p206g_with_p211_p212 import area_bucket, bbox_iou, load_p206g

ROOT=Path(__file__).resolve().parents[2]
P222=ROOT/'reports/vlm/symbol_p222_p221a_frozen_overlay.jsonl'
OUT_JSONL=ROOT/'reports/vlm/symbol_p221b_stair_equipment_residual_cases.jsonl'
OUT_JSON=ROOT/'reports/vlm/symbol_p221b_stair_equipment_residual_summary.json'
OUT_MD=ROOT/'reports/vlm/symbol_p221b_stair_equipment_residual_mining.md'
TARGET_LABELS={'stair','equipment'}

def pred_label(p): return str(p.get('label', p.get('symbol_type','unknown')))
def pred_score(p): return float(p.get('score', p.get('confidence',0.0)) or 0.0)
def center(b): return ((float(b[0])+float(b[2]))/2,(float(b[1])+float(b[3]))/2)
def dist(a,b):
    ax,ay=center(a); bx,by=center(b); return ((ax-bx)**2+(ay-by)**2)**0.5

def match_label_aware(preds,golds):
    cand=[]
    for pi,p in enumerate(preds):
        pb=[float(v) for v in p['bbox']]; pl=pred_label(p)
        for gi,g in enumerate(golds):
            if pl!=str(g['label']): continue
            iou=bbox_iou(pb,[float(v) for v in g['bbox']])
            if iou>=0.30: cand.append((iou,pi,gi))
    up,ug=set(),set()
    for iou,pi,gi in sorted(cand, reverse=True):
        if pi in up or gi in ug: continue
        up.add(pi); ug.add(gi)
    return up,ug

def dist_bucket(value):
    if value is None: return 'no_prediction'
    if value<=8: return 'near_le_8'
    if value<=16: return 'near_le_16'
    if value<=32: return 'near_le_32'
    if value<=64: return 'near_le_64'
    if value<=128: return 'near_le_128'
    return 'far_gt_128'

def main():
    rows,preds_by_row,golds_by_row=load_p206g(P222)
    cases=[]; by_label=Counter(); by_bucket=Counter(); by_cross=Counter(); by_row=Counter(); nearest_label=Counter(); nearest_same_label_dist=Counter(); nearest_any_dist=Counter(); nearest_same_source=Counter()
    for row in rows:
        rid=str(row.get('id') or row.get('row_id'))
        preds=preds_by_row[rid]; golds=list(golds_by_row[rid].values())
        _up,ug=match_label_aware(preds,golds)
        for gi,g in enumerate(golds):
            if gi in ug: continue
            label=str(g['label'])
            if label not in TARGET_LABELS: continue
            gbox=[float(v) for v in g['bbox']]; bucket=area_bucket(gbox)
            nearest_any=None; nearest_same=None
            for p in preds:
                pbox=[float(v) for v in p['bbox']]
                item={'label':pred_label(p),'source':p.get('source'),'score':pred_score(p),'bbox':pbox,'iou':bbox_iou(pbox,gbox),'center_distance':dist(pbox,gbox)}
                if nearest_any is None or item['center_distance']<nearest_any['center_distance']: nearest_any=item
                if item['label']==label and (nearest_same is None or item['center_distance']<nearest_same['center_distance']): nearest_same=item
            if nearest_any: nearest_label[nearest_any['label']]+=1
            if nearest_same:
                nearest_same_label_dist[dist_bucket(nearest_same['center_distance'])]+=1
                nearest_same_source[str(nearest_same.get('source'))]+=1
            else:
                nearest_same_label_dist['no_same_label_prediction']+=1
            if nearest_any: nearest_any_dist[dist_bucket(nearest_any['center_distance'])]+=1
            by_label[label]+=1; by_bucket[bucket]+=1; by_cross[(label,bucket)]+=1; by_row[rid]+=1
            cases.append({'row_id':rid,'target_id':g.get('target_id'),'label':label,'bucket':bucket,'bbox':gbox,'nearest_any':nearest_any,'nearest_same_label':nearest_same})
    OUT_JSONL.write_text(''.join(json.dumps(c,ensure_ascii=False)+'\n' for c in cases))
    summary={'id':'P221b_stair_equipment_residual_mining','source_overlay':str(P222.relative_to(ROOT)),'case_count':len(cases),'by_label':dict(by_label),'by_bucket':dict(by_bucket),'by_label_bucket':{f'{k[0]}|{k[1]}':v for k,v in by_cross.most_common()},'worst_rows':dict(by_row.most_common(20)),'nearest_any_label':dict(nearest_label.most_common()),'nearest_any_distance':dict(nearest_any_dist.most_common()),'nearest_same_label_distance':dict(nearest_same_label_dist.most_common()),'nearest_same_label_source':dict(nearest_same_source.most_common()),'examples':cases[:80],'claim_boundary':'Offline residual mining only; gold used for analysis/evaluation, not runtime.'}
    OUT_JSON.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
    lines=['# P221b Stair/Equipment Residual Mining','','## Counts',f"- Cases: {len(cases)}",f"- By label: `{json.dumps(dict(by_label), ensure_ascii=False)}`",f"- By bucket: `{json.dumps(dict(by_bucket), ensure_ascii=False)}`",f"- Label-bucket: `{json.dumps({f'{k[0]}|{k[1]}':v for k,v in by_cross.most_common()}, ensure_ascii=False)}`",f"- Worst rows: `{json.dumps(dict(by_row.most_common(10)), ensure_ascii=False)}`",'', '## Nearest Prediction Context',f"- Nearest any label: `{json.dumps(dict(nearest_label.most_common()), ensure_ascii=False)}`",f"- Nearest any distance: `{json.dumps(dict(nearest_any_dist.most_common()), ensure_ascii=False)}`",f"- Nearest same-label distance: `{json.dumps(dict(nearest_same_label_dist.most_common()), ensure_ascii=False)}`",f"- Nearest same-label source: `{json.dumps(dict(nearest_same_source.most_common()), ensure_ascii=False)}`",'', '## Implementation Implication','- If same-label predictions are nearby, start with geometry/subcandidate/refiner rules.','- If same-label predictions are absent/far, train or reuse a proposal branch before verifier work.']
    OUT_MD.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(OUT_MD),'cases':len(cases),'by_label':dict(by_label),'nearest_same_label_distance':dict(nearest_same_label_dist.most_common())},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
