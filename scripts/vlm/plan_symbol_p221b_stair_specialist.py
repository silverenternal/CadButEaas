#!/usr/bin/env python3
"""Plan P221b stair proposal specialist from current residual evidence."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

ROOT=Path(__file__).resolve().parents[2]
CASES=ROOT/'reports/vlm/symbol_p221b_stair_equipment_residual_cases.jsonl'
COVERAGE=ROOT/'reports/vlm/symbol_p221b_existing_proposal_coverage.json'
VISUAL=ROOT/'reports/vlm/symbol_p221b_visual_refiner_coverage.json'
OUT_JSON=ROOT/'reports/vlm/symbol_p221b_stair_proposal_specialist_plan.json'
OUT_MD=ROOT/'reports/vlm/symbol_p221b_stair_proposal_specialist_plan.md'

def main():
    cases=[json.loads(l) for l in CASES.read_text().splitlines() if l.strip()]
    stair=[c for c in cases if c['label']=='stair']
    by_bucket=Counter(c['bucket'] for c in stair)
    by_row=Counter(c['row_id'] for c in stair)
    coverage=json.loads(COVERAGE.read_text())
    visual=json.loads(VISUAL.read_text())
    best_existing=coverage['results'][0]
    best_visual=visual['results'][0]
    plan={
        'id':'P221b_stair_proposal_specialist_plan',
        'baseline':'P222_frozen_P221a_sink_tiny_subcandidate',
        'stair_residual_count':len(stair),
        'stair_by_bucket':dict(by_bucket),
        'worst_rows':dict(by_row.most_common(20)),
        'existing_p213b_stair_iou_coverage':best_existing['iou_by_label'].get('stair',0),
        'visual_stair_iou_coverage':best_visual['iou_by_label'].get('stair',0),
        'diagnosis':'Stair residuals are proposal-limited: existing P213b covers only 37/121 and visual refiner covers only 13/121 by IoU>=0.30.',
        'recommended_actions':[
            'Build stair-focused residual crop dataset from P222 stair FN, emphasizing small_le_256, xlarge_gt_4096, and tiny_le_64 buckets.',
            'Include hard negatives from nearby non-stair predictions and rows with stair FPs to protect precision.',
            'Train a stair-only proposal specialist on server GPU; start with YOLO crop/tile specialist if existing p213b data builder can be adapted.',
            'Deploy as proposal branch over P222 pages, then gate with strict score/overlap verifier.',
            'Bootstrap vs P222; promote only with positive F1 and non-negative precision CI.'
        ],
        'runtime_contract':'Runtime may use raster pixels, model weights, candidate score/bbox/label, and frozen config only. No row_id/gold/SVG/parser geometry at runtime.',
        'outputs_next':['datasets/symbol_p221b_stair_specialist_yolo/build_report.json','checkpoints/symbol_p221b_stair_specialist/model.pt','reports/vlm/symbol_p221b_stair_specialist_eval.json','reports/vlm/symbol_p221b_stair_specialist_vs_p222_bootstrap.md']
    }
    OUT_JSON.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    lines=['# P221b Stair Proposal Specialist Plan','','## Diagnosis',f"- Stair residual FN after P222: {len(stair)}",f"- By bucket: `{json.dumps(dict(by_bucket), ensure_ascii=False)}`",f"- Worst rows: `{json.dumps(dict(by_row.most_common(10)), ensure_ascii=False)}`",f"- Existing P213b stair IoU coverage: {plan['existing_p213b_stair_iou_coverage']}/{len(stair)}",f"- Old visual refiner stair IoU coverage: {plan['visual_stair_iou_coverage']}/{len(stair)}",'', '## Decision','- Stair is proposal-limited, not verifier-limited.', '- Do not spend more time on tabular equipment gates unless visual/crop cache is made explicit.', '- Next metric-rescue branch should train or adapt a stair-focused proposal specialist.', '', '## Implementation Steps']
    lines += [f"- {x}" for x in plan['recommended_actions']]
    lines += ['', '## Runtime Contract', f"- {plan['runtime_contract']}"]
    OUT_MD.write_text('\n'.join(lines)+'\n')
    print(json.dumps({'report':str(OUT_MD),'stair':len(stair),'coverage':plan['existing_p213b_stair_iou_coverage'],'visual':plan['visual_stair_iou_coverage']},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
