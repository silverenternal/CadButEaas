#!/usr/bin/env python3
"""Apply fixed P0-81 metadata selector learned on smoke."""
from __future__ import annotations
import argparse,json,sys,math
from pathlib import Path
SCRIPT_DIR=Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0,str(SCRIPT_DIR))
from apply_symbol_rtdetr_complement_policy_p065 import read_exported_golds
from train_symbol_metadata_selector_p081 import apply_model, simple_logistic_train, build_rows
from sweep_symbol_added_candidate_reranker_p075 import rows_from_predictions
from sweep_symbol_rtdetr_complement_gate_p063 import read_predictions, score
from train_symbol_tile_detector_v20 import rel, write_jsonl
ROOT=Path(__file__).resolve().parents[2]
DEFAULT_DATA=ROOT/'datasets/symbol_tile_detector_tiny_sahi_v21'
DEFAULT_YOLO_DIR=ROOT/'datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27'
TRAIN_V28=ROOT/'reports/vlm/symbol_yolov8s_seg_rect_v28_smoke_v30_page_predictions_p062_refresh.jsonl'
TRAIN_P076=ROOT/'reports/vlm/symbol_balanced_policy_p076_smoke_v30_predictions.jsonl'
FIXED_THRESHOLD=0.1
FIXED_MAX_ADD=5

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--data',default=str(DEFAULT_DATA)); ap.add_argument('--yolo-dir',default=str(DEFAULT_YOLO_DIR)); ap.add_argument('--split',required=True); ap.add_argument('--v28-predictions',required=True); ap.add_argument('--p076-predictions',required=True); ap.add_argument('--output-predictions',required=True); ap.add_argument('--output-json',required=True); ap.add_argument('--output-md',required=True); args=ap.parse_args()
 train_v28=read_predictions(TRAIN_V28); train_p076=read_predictions(TRAIN_P076); train_golds=read_exported_golds(Path(args.data),Path(args.yolo_dir),'smoke_v30',set(train_v28)|set(train_p076)); w=simple_logistic_train(build_rows(train_v28,train_p076,train_golds))
 v28=read_predictions(Path(args.v28_predictions)); p076=read_predictions(Path(args.p076_predictions)); golds=read_exported_golds(Path(args.data),Path(args.yolo_dir),args.split,set(v28)|set(p076))
 baseline=score(golds,v28,{r:[] for r in v28},{'labels':'all','areas':'all','score_min':1.1,'max_iou_with_v28':0.0,'max_add_per_page':0}); full=score(golds,p076,{r:[] for r in p076},{'labels':'all','areas':'all','score_min':1.1,'max_iou_with_v28':0.0,'max_add_per_page':0})
 pred=apply_model(v28,p076,w,FIXED_THRESHOLD,FIXED_MAX_ADD); metrics=score(golds,pred,{r:[] for r in pred},{'labels':'all','areas':'all','score_min':1.1,'max_iou_with_v28':0.0,'max_add_per_page':0})
 delta={k:round(metrics[k]-full[k],6) for k in ['iou_0_30_recall','center_recall','candidate_inflation','precision']}; retained=round((metrics['iou_0_30_recall']-baseline['iou_0_30_recall'])/max(full['iou_0_30_recall']-baseline['iou_0_30_recall'],1e-9),6)
 decision='positive_locked_metadata_selector' if retained>=0.9 and metrics['precision']>full['precision'] and metrics['candidate_inflation']<=full['candidate_inflation'] else 'negative_do_not_package'
 outp=Path(args.output_predictions); outj=Path(args.output_json); outm=Path(args.output_md); outp.parent.mkdir(parents=True,exist_ok=True); write_jsonl(outp,rows_from_predictions(pred))
 report={'version':'symbol_metadata_selector_p081_fixed','split':args.split,'source_integrity':'model trained with offline labels; runtime features metadata-only','weights':w,'threshold':FIXED_THRESHOLD,'max_add_per_page':FIXED_MAX_ADD,'baseline_v28':baseline,'full_p076':full,'metadata_policy':metrics,'delta_vs_p076':delta,'retained_p076_gain_fraction':retained,'decision':decision}
 outj.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 outm.write_text(f"""# P0-81 fixed metadata selector - {args.split}\n\n- decision: `{decision}`\n- P0-76 IoU / inflation / precision: `{full['iou_0_30_recall']:.6f}` / `{full['candidate_inflation']:.6f}` / `{full['precision']:.6f}`\n- metadata IoU / inflation / precision: `{metrics['iou_0_30_recall']:.6f}` / `{metrics['candidate_inflation']:.6f}` / `{metrics['precision']:.6f}`\n- retained P0-76 gain: `{retained:.6f}`\n- delta vs P0-76 IoU / inflation / precision: `{delta['iou_0_30_recall']:+.6f}` / `{delta['candidate_inflation']:+.6f}` / `{delta['precision']:+.6f}`\n""")
 print(json.dumps({'split':args.split,'decision':decision,'retained':retained,'delta_vs_p076':delta,'metrics':{k:metrics[k] for k in ['iou_0_30_recall','candidate_inflation','precision']}},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
