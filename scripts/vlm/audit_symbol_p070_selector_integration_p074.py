#!/usr/bin/env python3
"""Audit P0-70 added candidates for downstream selector/listwise integration."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from apply_symbol_rtdetr_complement_policy_p065 import read_exported_golds
from sweep_symbol_rtdetr_complement_gate_p063 import matched_target_ids, pred_area_bucket, read_predictions
from train_symbol_tile_detector_v20 import bbox_iou, rel, write_jsonl

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_DATA=ROOT/'datasets/symbol_tile_detector_tiny_sahi_v21'
DEFAULT_YOLO_DIR=ROOT/'datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27'
DEFAULT_V28=ROOT/'reports/vlm/symbol_yolov8s_seg_rect_v28_smoke_v30_page_predictions_p062_refresh.jsonl'
DEFAULT_P070=ROOT/'reports/vlm/symbol_precision_gated_policy_p070_smoke_v30_predictions.jsonl'
DEFAULT_JSON=ROOT/'reports/vlm/symbol_p070_selector_integration_p074_smoke_v30.json'
DEFAULT_MD=ROOT/'reports/vlm/symbol_p070_selector_integration_p074_smoke_v30.md'
DEFAULT_ROWS=ROOT/'reports/vlm/symbol_p070_selector_integration_p074_smoke_v30_added_candidates.jsonl'


def max_iou_to_golds(pred: dict[str,Any], golds: dict[str,dict[str,Any]]) -> tuple[float, str|None, str|None]:
    box=[float(v) for v in pred['bbox']]
    best=0.0; best_id=None; best_label=None
    for tid,g in golds.items():
        iou=bbox_iou(box,[float(v) for v in g['bbox']])
        if iou>best:
            best=iou; best_id=tid; best_label=str(g['label'])
    return best,best_id,best_label


def max_iou_to_preds(pred: dict[str,Any], preds: list[dict[str,Any]]) -> float:
    box=[float(v) for v in pred['bbox']]
    return max((bbox_iou(box,[float(x) for x in item['bbox']]) for item in preds), default=0.0)


def source_name(pred: dict[str,Any]) -> str:
    return str(pred.get('source_policy') or 'v28')


def score_bucket(score: float) -> str:
    if score >= 0.75: return 'score_ge_0_75'
    if score >= 0.50: return 'score_ge_0_50'
    if score >= 0.20: return 'score_ge_0_20'
    if score >= 0.10: return 'score_ge_0_10'
    return 'score_lt_0_10'


def main() -> None:
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', default=str(DEFAULT_DATA)); ap.add_argument('--yolo-dir', default=str(DEFAULT_YOLO_DIR)); ap.add_argument('--split', default='smoke_v30')
    ap.add_argument('--v28-predictions', default=str(DEFAULT_V28)); ap.add_argument('--p070-predictions', default=str(DEFAULT_P070))
    ap.add_argument('--output-json', default=str(DEFAULT_JSON)); ap.add_argument('--output-md', default=str(DEFAULT_MD)); ap.add_argument('--rows-output', default=str(DEFAULT_ROWS))
    args=ap.parse_args()
    v28=read_predictions(Path(args.v28_predictions)); p070=read_predictions(Path(args.p070_predictions))
    golds=read_exported_golds(Path(args.data), Path(args.yolo_dir), args.split, set(v28)|set(p070))
    rows=[]
    totals=Counter(); by_source=Counter(); by_label=Counter(); by_area=Counter(); by_score=Counter(); useful_by_source=Counter(); duplicate_by_source=Counter(); support_by_source=Counter(); wrong_by_source=Counter()
    for row_id,gold_map in golds.items():
        v28_items=v28.get(row_id,[]); p070_items=p070.get(row_id,[])
        v28_matched=matched_target_ids(gold_map, v28_items, 'iou')
        p070_matched=matched_target_ids(gold_map, p070_items, 'iou')
        unique_targets=p070_matched-v28_matched
        for pred in p070_items:
            src=source_name(pred)
            if src=='v28':
                continue
            label=str(pred.get('label'))
            area=pred_area_bucket(pred)
            score=float(pred.get('score',0.0))
            best_iou,best_tid,best_label=max_iou_to_golds(pred,gold_map)
            overlap_v28=max_iou_to_preds(pred,v28_items)
            if best_tid in unique_targets and best_iou>=0.30:
                bucket='unique_recovery'
                useful_by_source[src]+=1
            elif overlap_v28>=0.50:
                bucket='duplicate_of_v28'
                duplicate_by_source[src]+=1
            elif best_iou>=0.10 and best_label and best_label!=label:
                bucket='wrong_type_near_gold'
                wrong_by_source[src]+=1
            else:
                bucket='support_negative_or_background'
                support_by_source[src]+=1
            totals[bucket]+=1; totals['added']+=1; by_source[src]+=1; by_label[label]+=1; by_area[area]+=1; by_score[score_bucket(score)]+=1
            rows.append({'row_id':row_id,'source_policy':src,'label':label,'area_bucket':area,'score':score,'bbox':pred['bbox'],'bucket':bucket,'best_gold_iou':round(best_iou,6),'best_gold_label':best_label,'overlap_v28':round(overlap_v28,6),'unique_target':best_tid in unique_targets})
    precision_like=totals['unique_recovery']/max(totals['added'],1)
    report={
        'version':'symbol_p070_selector_integration_p074_smoke_v30',
        'source_integrity':'offline gold used only for selector integration audit; runtime features are raster-derived candidate fields',
        'inputs':{'v28_predictions':rel(Path(args.v28_predictions)),'p070_predictions':rel(Path(args.p070_predictions)),'data':rel(Path(args.data))},
        'totals':{k:int(v) for k,v in totals.items()},
        'added_candidate_unique_recovery_rate':round(precision_like,6),
        'by_source':dict(by_source),
        'useful_by_source':dict(useful_by_source),
        'duplicate_by_source':dict(duplicate_by_source),
        'support_negative_by_source':dict(support_by_source),
        'wrong_type_by_source':dict(wrong_by_source),
        'by_label':dict(by_label),
        'by_area':dict(by_area),
        'by_score':dict(by_score),
        'decision_hint':'If unique_recovery_rate is high and source concentrated, package source-aware selector features; if low, train/listwise reranker over added candidates.'
    }
    Path(args.output_json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_json).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    write_jsonl(Path(args.rows_output), rows)
    lines=['# P0-74 P0-70 selector integration audit','','## Summary','',f"- added candidates: `{totals['added']}`",f"- unique recovery: `{totals['unique_recovery']}` (`{precision_like:.6f}`)",f"- duplicate of v28: `{totals['duplicate_of_v28']}`",f"- support/background: `{totals['support_negative_or_background']}`",f"- wrong type near gold: `{totals['wrong_type_near_gold']}`",'', '## By Source','']
    for src,count in by_source.most_common():
        lines.append(f"- `{src}`: added `{count}`, useful `{useful_by_source[src]}`, duplicate `{duplicate_by_source[src]}`, support `{support_by_source[src]}`, wrong-type `{wrong_by_source[src]}`")
    lines += ['','## Artifacts','',f"- `{rel(Path(args.output_json))}`",f"- `{rel(Path(args.rows_output))}`",'']
    Path(args.output_md).write_text('\n'.join(lines))
    print(json.dumps({'totals':report['totals'],'unique_recovery_rate':report['added_candidate_unique_recovery_rate'],'by_source':report['by_source'],'useful_by_source':report['useful_by_source']},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
