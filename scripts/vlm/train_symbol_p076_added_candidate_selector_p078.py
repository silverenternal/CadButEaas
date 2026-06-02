#!/usr/bin/env python3
"""Lightweight selector/gate sweep for P0-76 added candidates."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
from typing import Any
SCRIPT_DIR=Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0,str(SCRIPT_DIR))
from apply_symbol_rtdetr_complement_policy_p065 import read_exported_golds
from sweep_symbol_added_candidate_reranker_p075 import gate_predictions, rows_from_predictions
from sweep_symbol_rtdetr_complement_gate_p063 import read_predictions, score, pred_area_bucket, matched_target_ids
from train_symbol_tile_detector_v20 import bbox_iou, rel, write_jsonl
ROOT=Path(__file__).resolve().parents[2]
DEFAULT_DATA=ROOT/'datasets/symbol_tile_detector_tiny_sahi_v21'
DEFAULT_YOLO_DIR=ROOT/'datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27'
DEFAULT_V28=ROOT/'reports/vlm/symbol_yolov8s_seg_rect_v28_smoke_v30_page_predictions_p062_refresh.jsonl'
DEFAULT_P076=ROOT/'reports/vlm/symbol_balanced_policy_p076_smoke_v30_predictions.jsonl'
DEFAULT_ROWS=ROOT/'reports/vlm/symbol_p076_selector_p078_smoke_v30_rows.jsonl'
DEFAULT_JSON=ROOT/'reports/vlm/symbol_p076_selector_p078_smoke_v30.json'
DEFAULT_MD=ROOT/'reports/vlm/symbol_p076_selector_p078_smoke_v30.md'
DEFAULT_PRED=ROOT/'reports/vlm/symbol_p076_selector_p078_smoke_v30_predictions.jsonl'

def source_name(p:dict[str,Any])->str: return str(p.get('source_policy') or 'v28')
def max_iou(pred:dict[str,Any], refs:list[dict[str,Any]])->float:
 box=[float(v) for v in pred['bbox']]
 return max((bbox_iou(box,[float(x) for x in r['bbox']]) for r in refs), default=0.0)
def box_features(pred):
 l,t,r,b=[float(v) for v in pred['bbox']]; w=max(1e-6,r-l); h=max(1e-6,b-t); area=w*h
 return w,h,area,w/h

def build_rows(v28,p076,golds):
 rows=[]
 for row_id,gold_map in golds.items():
  v28_items=v28.get(row_id,[]); p076_items=p076.get(row_id,[])
  v28_matched=matched_target_ids(gold_map,v28_items,'iou'); p076_matched=matched_target_ids(gold_map,p076_items,'iou'); unique=p076_matched-v28_matched
  for pred in p076_items:
   if source_name(pred)=='v28': continue
   best=0.0; best_tid=None; best_label=None
   for tid,g in gold_map.items():
    i=bbox_iou([float(v) for v in pred['bbox']],[float(v) for v in g['bbox']])
    if i>best: best=i; best_tid=tid; best_label=str(g['label'])
   w,h,area,aspect=box_features(pred); ov=max_iou(pred,v28_items)
   if best_tid in unique and best>=0.30: bucket='unique_recovery'; y=1
   elif ov>=0.50: bucket='duplicate'; y=0
   elif best>=0.10 and best_label and best_label!=str(pred.get('label')): bucket='wrong_type'; y=0
   else: bucket='support_negative'; y=0
   rows.append({'row_id':row_id,'label':str(pred.get('label')),'source_policy':source_name(pred),'score':float(pred.get('score',0.0)),'area_bucket':pred_area_bucket(pred),'bbox_area':area,'bbox_aspect':aspect,'bbox_w':w,'bbox_h':h,'overlap_v28':ov,'bucket':bucket,'target':y})
 return rows

def apply_feature_gate(v28,p076,cfg):
 out={}
 for row_id in sorted(set(v28)|set(p076)):
  base=list(v28.get(row_id,[])); kept=[]
  for pred in p076.get(row_id,[]):
   if source_name(pred)=='v28': continue
   score=float(pred.get('score',0.0)); ov=max_iou(pred,base); w,h,area,aspect=box_features(pred)
   if score < cfg['score_min'] or score >= cfg['score_max']: continue
   if ov < cfg['overlap_min'] or ov >= cfg['overlap_max']: continue
   if area < cfg['area_min'] or area >= cfg['area_max']: continue
   if aspect < cfg['aspect_min'] or aspect >= cfg['aspect_max']: continue
   kept.append(pred)
  kept.sort(key=lambda p: float(p.get('score',0.0)), reverse=True)
  out[row_id]=base+kept[:cfg['max_add_per_page']]
 return out

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--data',default=str(DEFAULT_DATA)); ap.add_argument('--yolo-dir',default=str(DEFAULT_YOLO_DIR)); ap.add_argument('--split',default='smoke_v30'); ap.add_argument('--v28-predictions',default=str(DEFAULT_V28)); ap.add_argument('--p076-predictions',default=str(DEFAULT_P076)); ap.add_argument('--rows-output',default=str(DEFAULT_ROWS)); ap.add_argument('--output-json',default=str(DEFAULT_JSON)); ap.add_argument('--output-md',default=str(DEFAULT_MD)); ap.add_argument('--output-predictions',default=str(DEFAULT_PRED)); args=ap.parse_args()
 v28=read_predictions(Path(args.v28_predictions)); p076=read_predictions(Path(args.p076_predictions)); golds=read_exported_golds(Path(args.data),Path(args.yolo_dir),args.split,set(v28)|set(p076))
 rows=build_rows(v28,p076,golds); Path(args.rows_output).parent.mkdir(parents=True,exist_ok=True); write_jsonl(Path(args.rows_output),rows)
 baseline=score(golds,v28,{r:[] for r in v28},{'labels':'all','areas':'all','score_min':1.1,'max_iou_with_v28':0.0,'max_add_per_page':0}); full=score(golds,p076,{r:[] for r in p076},{'labels':'all','areas':'all','score_min':1.1,'max_iou_with_v28':0.0,'max_add_per_page':0})
 configs=[
  {'score_min':0.10,'score_max':1.01,'overlap_min':0.4,'overlap_max':1.01,'area_min':0.0,'area_max':1e12,'aspect_min':0.0,'aspect_max':1e12,'max_add_per_page':20},
  {'score_min':0.10,'score_max':1.01,'overlap_min':0.45,'overlap_max':1.01,'area_min':0.0,'area_max':1e12,'aspect_min':0.0,'aspect_max':1e12,'max_add_per_page':20},
  {'score_min':0.12,'score_max':1.01,'overlap_min':0.4,'overlap_max':1.01,'area_min':0.0,'area_max':1e12,'aspect_min':0.0,'aspect_max':1e12,'max_add_per_page':20},
  {'score_min':0.10,'score_max':1.01,'overlap_min':0.4,'overlap_max':0.75,'area_min':0.0,'area_max':1e12,'aspect_min':0.0,'aspect_max':1e12,'max_add_per_page':20},
  {'score_min':0.10,'score_max':1.01,'overlap_min':0.4,'overlap_max':1.01,'area_min':0.0,'area_max':96,'aspect_min':0.0,'aspect_max':1e12,'max_add_per_page':20},
  {'score_min':0.10,'score_max':1.01,'overlap_min':0.4,'overlap_max':1.01,'area_min':0.0,'area_max':128,'aspect_min':0.0,'aspect_max':1e12,'max_add_per_page':20},
  {'score_min':0.10,'score_max':1.01,'overlap_min':0.35,'overlap_max':1.01,'area_min':0.0,'area_max':128,'aspect_min':0.0,'aspect_max':1e12,'max_add_per_page':20},
  {'score_min':0.15,'score_max':1.01,'overlap_min':0.35,'overlap_max':1.01,'area_min':0.0,'area_max':1e12,'aspect_min':0.0,'aspect_max':1e12,'max_add_per_page':20},
 ]
 results=[]; gain=full['iou_0_30_recall']-baseline['iou_0_30_recall']
 for cfg in configs:
  pred=apply_feature_gate(v28,p076,cfg); m=score(golds,pred,{r:[] for r in pred},{'labels':'all','areas':'all','score_min':1.1,'max_iou_with_v28':0.0,'max_add_per_page':0}); m['config']=cfg; m['delta_vs_v28']={k:round(m[k]-baseline[k],6) for k in ['iou_0_30_recall','center_recall','candidate_inflation','precision']}; m['delta_vs_p076']={k:round(m[k]-full[k],6) for k in ['iou_0_30_recall','center_recall','candidate_inflation','precision']}; m['retained_gain_fraction']=round((m['iou_0_30_recall']-baseline['iou_0_30_recall'])/max(gain,1e-9),6); results.append(m)
 feasible=[m for m in results if m['retained_gain_fraction']>=0.7 and m['precision']>=full['precision'] and m['candidate_inflation']<=full['candidate_inflation']]
 feasible.sort(key=lambda m:(m['precision'],m['iou_0_30_recall'],-m['candidate_inflation']),reverse=True); best=feasible[0] if feasible else None
 if best: write_jsonl(Path(args.output_predictions), rows_from_predictions(apply_feature_gate(v28,p076,best['config'])))
 pos=sum(r['target'] for r in rows); report={'version':'symbol_p076_selector_p078_smoke_v30','source_integrity':'offline labels for training/eval only; runtime gate uses candidate score/bbox/overlap features','dataset':{'rows':len(rows),'positive_unique_recovery':pos,'positive_rate':round(pos/max(len(rows),1),6)},'inputs':{'v28_predictions':rel(Path(args.v28_predictions)),'p076_predictions':rel(Path(args.p076_predictions))},'baseline_v28':baseline,'full_p076':full,'best_feasible':best,'top_feasible':feasible[:20],'sweep_count':len(results),'decision':'positive_smoke_candidate_validate_locked' if best else 'negative_no_selector_improvement'}
 Path(args.output_json).write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 lines=['# P0-78 P0-76 added-candidate selector smoke','',f"- rows: `{len(rows)}` positives: `{pos}` rate: `{report['dataset']['positive_rate']:.6f}`",f"- v28 IoU / inflation / precision: `{baseline['iou_0_30_recall']:.6f}` / `{baseline['candidate_inflation']:.6f}` / `{baseline['precision']:.6f}`",f"- P0-76 IoU / inflation / precision: `{full['iou_0_30_recall']:.6f}` / `{full['candidate_inflation']:.6f}` / `{full['precision']:.6f}`",f"- decision: `{report['decision']}`"]
 if best: lines += [f"- best IoU / inflation / precision: `{best['iou_0_30_recall']:.6f}` / `{best['candidate_inflation']:.6f}` / `{best['precision']:.6f}`",f"- retained gain: `{best['retained_gain_fraction']:.6f}`",f"- config: `{json.dumps(best['config'],ensure_ascii=False)}`"]
 Path(args.output_md).write_text('\n'.join(lines)+'\n')
 print(json.dumps({'decision':report['decision'],'dataset':report['dataset'],'best':None if best is None else {'iou':best['iou_0_30_recall'],'inflation':best['candidate_inflation'],'precision':best['precision'],'retained':best['retained_gain_fraction'],'config':best['config']}},ensure_ascii=False,indent=2)[:6000])
if __name__=='__main__': main()
