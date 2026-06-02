#!/usr/bin/env python3
"""Sweep lightweight gates over P0-70 added sink/tiny candidates."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from apply_symbol_rtdetr_complement_policy_p065 import read_exported_golds
from sweep_symbol_added_candidate_precision_gate_p069 import max_overlap, rows_from_predictions, source_name
from sweep_symbol_rtdetr_complement_gate_p063 import read_predictions, score
from train_symbol_tile_detector_v20 import rel, write_jsonl

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_DATA=ROOT/'datasets/symbol_tile_detector_tiny_sahi_v21'
DEFAULT_YOLO_DIR=ROOT/'datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27'
DEFAULT_V28=ROOT/'reports/vlm/symbol_yolov8s_seg_rect_v28_smoke_v30_page_predictions_p062_refresh.jsonl'
DEFAULT_P070=ROOT/'reports/vlm/symbol_precision_gated_policy_p070_smoke_v30_predictions.jsonl'
DEFAULT_JSON=ROOT/'reports/vlm/symbol_added_candidate_reranker_p075_smoke_v30.json'
DEFAULT_MD=ROOT/'reports/vlm/symbol_added_candidate_reranker_p075_smoke_v30.md'
DEFAULT_PRED=ROOT/'reports/vlm/symbol_added_candidate_reranker_p075_smoke_v30_predictions.jsonl'


def gate_predictions(v28: dict[str,list[dict[str,Any]]], p070: dict[str,list[dict[str,Any]]], config: dict[str,Any]) -> dict[str,list[dict[str,Any]]]:
    out={}
    for row_id in sorted(set(v28)|set(p070)):
        base=list(v28.get(row_id,[]))
        kept=[]
        for pred in p070.get(row_id,[]):
            if source_name(pred)=='v28':
                continue
            score=float(pred.get('score',0.0))
            overlap=max_overlap(pred, base)
            if score < config['score_min'] or score >= config['score_max']:
                continue
            if overlap < config['overlap_min'] or overlap >= config['overlap_max']:
                continue
            kept.append(pred)
        kept.sort(key=lambda x: float(x.get('score',0.0)), reverse=True)
        out[row_id]=base+kept[:config['max_add_per_page']]
    return out


def summarize(metrics: dict[str,Any]) -> dict[str,Any]:
    return {k:metrics[k] for k in ['iou_0_30_recall','center_recall','candidate_inflation','precision']}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', default=str(DEFAULT_DATA)); ap.add_argument('--yolo-dir', default=str(DEFAULT_YOLO_DIR)); ap.add_argument('--split', default='smoke_v30')
    ap.add_argument('--v28-predictions', default=str(DEFAULT_V28)); ap.add_argument('--p070-predictions', default=str(DEFAULT_P070))
    ap.add_argument('--output-json', default=str(DEFAULT_JSON)); ap.add_argument('--output-md', default=str(DEFAULT_MD)); ap.add_argument('--output-predictions', default=str(DEFAULT_PRED))
    args=ap.parse_args()
    v28=read_predictions(Path(args.v28_predictions)); p070=read_predictions(Path(args.p070_predictions))
    golds=read_exported_golds(Path(args.data), Path(args.yolo_dir), args.split, set(v28)|set(p070))
    baseline=score(golds, v28, {row_id:[] for row_id in v28}, {'labels':'all','areas':'all','score_min':1.1,'max_iou_with_v28':0.0,'max_add_per_page':0})
    full=score(golds, p070, {row_id:[] for row_id in p070}, {'labels':'all','areas':'all','score_min':1.1,'max_iou_with_v28':0.0,'max_add_per_page':0})
    configs=[
        {'score_min':0.10,'score_max':1.01,'overlap_min':0.0,'overlap_max':1.01,'max_add_per_page':20},
        {'score_min':0.12,'score_max':1.01,'overlap_min':0.0,'overlap_max':1.01,'max_add_per_page':20},
        {'score_min':0.15,'score_max':1.01,'overlap_min':0.0,'overlap_max':1.01,'max_add_per_page':20},
        {'score_min':0.20,'score_max':1.01,'overlap_min':0.0,'overlap_max':1.01,'max_add_per_page':20},
        {'score_min':0.10,'score_max':0.65,'overlap_min':0.0,'overlap_max':1.01,'max_add_per_page':20},
        {'score_min':0.10,'score_max':0.50,'overlap_min':0.0,'overlap_max':1.01,'max_add_per_page':20},
        {'score_min':0.10,'score_max':1.01,'overlap_min':0.30,'overlap_max':1.01,'max_add_per_page':20},
        {'score_min':0.10,'score_max':1.01,'overlap_min':0.35,'overlap_max':1.01,'max_add_per_page':20},
        {'score_min':0.10,'score_max':1.01,'overlap_min':0.40,'overlap_max':1.01,'max_add_per_page':20},
        {'score_min':0.10,'score_max':1.01,'overlap_min':0.30,'overlap_max':0.75,'max_add_per_page':20},
        {'score_min':0.10,'score_max':1.01,'overlap_min':0.35,'overlap_max':0.75,'max_add_per_page':20},
        {'score_min':0.10,'score_max':0.65,'overlap_min':0.30,'overlap_max':0.75,'max_add_per_page':20},
        {'score_min':0.10,'score_max':0.65,'overlap_min':0.35,'overlap_max':0.75,'max_add_per_page':20},
        {'score_min':0.15,'score_max':0.65,'overlap_min':0.30,'overlap_max':0.75,'max_add_per_page':20},
        {'score_min':0.10,'score_max':0.65,'overlap_min':0.30,'overlap_max':0.65,'max_add_per_page':20},
    ]
    results=[]
    full_gain=full['iou_0_30_recall']-baseline['iou_0_30_recall']
    for cfg in configs:
        pred=gate_predictions(v28,p070,cfg)
        metrics=score(golds,pred,{row_id:[] for row_id in pred},{'labels':'all','areas':'all','score_min':1.1,'max_iou_with_v28':0.0,'max_add_per_page':0})
        metrics['config']=cfg
        metrics['delta_vs_v28']={k:round(metrics[k]-baseline[k],6) for k in ['iou_0_30_recall','center_recall','candidate_inflation','precision']}
        metrics['delta_vs_p070']={k:round(metrics[k]-full[k],6) for k in ['iou_0_30_recall','center_recall','candidate_inflation','precision']}
        metrics['retained_p070_gain_fraction']=round((metrics['iou_0_30_recall']-baseline['iou_0_30_recall'])/max(full_gain,1e-9),6)
        metrics['tiny_iou_delta']=round(metrics['per_area_iou_recall'].get('tiny_le_64',0)-baseline['per_area_iou_recall'].get('tiny_le_64',0),6)
        metrics['sink_iou_delta']=round(metrics['per_label_iou_recall'].get('sink',0)-baseline['per_label_iou_recall'].get('sink',0),6)
        results.append(metrics)
    feasible=[m for m in results if m['retained_p070_gain_fraction']>=0.65 and m['precision']>=full['precision'] and m['candidate_inflation']<full['candidate_inflation']]
    feasible.sort(key=lambda m:(m['precision'], -m['candidate_inflation'], m['iou_0_30_recall']), reverse=True)
    best=feasible[0] if feasible else None
    if best:
        write_jsonl(Path(args.output_predictions), rows_from_predictions(gate_predictions(v28,p070,best['config'])))
    report={'version':'symbol_added_candidate_reranker_p075_smoke_v30','source_integrity':'offline gold used only for sweep; runtime gate uses score/overlap with v28 from raster-derived candidates','inputs':{'v28_predictions':rel(Path(args.v28_predictions)),'p070_predictions':rel(Path(args.p070_predictions))},'baseline_v28':baseline,'full_p070':full,'best_feasible':best,'top_feasible':feasible[:20],'top_precision':sorted(results,key=lambda m:(m['precision'],m['iou_0_30_recall']),reverse=True)[:20],'sweep_count':len(results),'decision':'positive_smoke_candidate_validate_locked' if best else 'negative_no_better_reranker_gate'}
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    lines=['# P0-75 added-candidate reranker smoke sweep','','## Summary','',f"- v28 IoU / inflation / precision: `{baseline['iou_0_30_recall']:.6f}` / `{baseline['candidate_inflation']:.6f}` / `{baseline['precision']:.6f}`",f"- P0-70 IoU / inflation / precision: `{full['iou_0_30_recall']:.6f}` / `{full['candidate_inflation']:.6f}` / `{full['precision']:.6f}`",f"- decision: `{report['decision']}`"]
    if best:
        lines += [f"- best IoU / inflation / precision: `{best['iou_0_30_recall']:.6f}` / `{best['candidate_inflation']:.6f}` / `{best['precision']:.6f}`",f"- retained P0-70 gain: `{best['retained_p070_gain_fraction']:.6f}`",f"- delta vs P0-70 IoU / inflation / precision: `{best['delta_vs_p070']['iou_0_30_recall']:+.6f}` / `{best['delta_vs_p070']['candidate_inflation']:+.6f}` / `{best['delta_vs_p070']['precision']:+.6f}`",f"- config: `{json.dumps(best['config'],ensure_ascii=False)}`"]
    lines += ['','## Artifacts','',f"- `{rel(Path(args.output_json))}`",f"- `{rel(Path(args.output_md))}`",f"- `{rel(Path(args.output_predictions))}`",'']
    Path(args.output_md).write_text('\n'.join(lines))
    print(json.dumps({'decision':report['decision'],'baseline':summarize(baseline),'full_p070':summarize(full),'best':None if best is None else {**summarize(best),'retained':best['retained_p070_gain_fraction'],'delta_vs_p070':best['delta_vs_p070'],'config':best['config']}},ensure_ascii=False,indent=2)[:6000])

if __name__=='__main__': main()
