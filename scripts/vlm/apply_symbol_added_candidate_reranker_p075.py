#!/usr/bin/env python3
"""Apply fixed P0-75 added-candidate reranker gate."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
SCRIPT_DIR=Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0,str(SCRIPT_DIR))
from apply_symbol_rtdetr_complement_policy_p065 import read_exported_golds
from sweep_symbol_added_candidate_reranker_p075 import gate_predictions, rows_from_predictions
from sweep_symbol_rtdetr_complement_gate_p063 import read_predictions, score
from train_symbol_tile_detector_v20 import rel, write_jsonl
ROOT=Path(__file__).resolve().parents[2]
DEFAULT_DATA=ROOT/'datasets/symbol_tile_detector_tiny_sahi_v21'
DEFAULT_YOLO_DIR=ROOT/'datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27'
FIXED_CONFIG={'score_min':0.1,'score_max':1.01,'overlap_min':0.4,'overlap_max':1.01,'max_add_per_page':20}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--data',default=str(DEFAULT_DATA)); ap.add_argument('--yolo-dir',default=str(DEFAULT_YOLO_DIR)); ap.add_argument('--split',required=True); ap.add_argument('--v28-predictions',required=True); ap.add_argument('--p070-predictions',required=True); ap.add_argument('--output-predictions',required=True); ap.add_argument('--output-json',required=True); ap.add_argument('--output-md',required=True); args=ap.parse_args()
 v28=read_predictions(Path(args.v28_predictions)); p070=read_predictions(Path(args.p070_predictions)); golds=read_exported_golds(Path(args.data),Path(args.yolo_dir),args.split,set(v28)|set(p070))
 baseline=score(golds,v28,{r:[] for r in v28},{'labels':'all','areas':'all','score_min':1.1,'max_iou_with_v28':0.0,'max_add_per_page':0})
 full=score(golds,p070,{r:[] for r in p070},{'labels':'all','areas':'all','score_min':1.1,'max_iou_with_v28':0.0,'max_add_per_page':0})
 gated=gate_predictions(v28,p070,FIXED_CONFIG); metrics=score(golds,gated,{r:[] for r in gated},{'labels':'all','areas':'all','score_min':1.1,'max_iou_with_v28':0.0,'max_add_per_page':0})
 delta_v28={k:round(metrics[k]-baseline[k],6) for k in ['iou_0_30_recall','center_recall','candidate_inflation','precision']}; delta_p070={k:round(metrics[k]-full[k],6) for k in ['iou_0_30_recall','center_recall','candidate_inflation','precision']}
 retained=round((metrics['iou_0_30_recall']-baseline['iou_0_30_recall'])/max(full['iou_0_30_recall']-baseline['iou_0_30_recall'],1e-9),6)
 decision='positive_locked_reranker_gate' if retained>=0.65 and metrics['precision']>=full['precision'] and metrics['candidate_inflation']<full['candidate_inflation'] else 'negative_do_not_package'
 outp=Path(args.output_predictions); outj=Path(args.output_json); outm=Path(args.output_md); outp.parent.mkdir(parents=True,exist_ok=True); write_jsonl(outp,rows_from_predictions(gated))
 report={'version':'symbol_added_candidate_reranker_p075_fixed','split':args.split,'source_integrity':'runtime gate uses score and overlap with v28 from raster-derived candidates; gold validation-only','inputs':{'v28_predictions':rel(Path(args.v28_predictions)),'p070_predictions':rel(Path(args.p070_predictions))},'outputs':{'predictions':rel(outp),'json':rel(outj),'markdown':rel(outm)},'gate_config':FIXED_CONFIG,'baseline_v28':baseline,'full_p070':full,'gated_policy':metrics,'delta_vs_v28':delta_v28,'delta_vs_p070':delta_p070,'retained_p070_gain_fraction':retained,'decision':decision}
 outj.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
 outm.write_text(f"""# P0-75 fixed added-candidate reranker - {args.split}\n\n## Decision\n\n- `{decision}`\n\n## Metrics\n\n- v28 IoU / inflation / precision: `{baseline['iou_0_30_recall']:.6f}` / `{baseline['candidate_inflation']:.6f}` / `{baseline['precision']:.6f}`\n- P0-70 IoU / inflation / precision: `{full['iou_0_30_recall']:.6f}` / `{full['candidate_inflation']:.6f}` / `{full['precision']:.6f}`\n- gated IoU / inflation / precision: `{metrics['iou_0_30_recall']:.6f}` / `{metrics['candidate_inflation']:.6f}` / `{metrics['precision']:.6f}`\n- retained P0-70 gain: `{retained:.6f}`\n- delta vs P0-70 IoU / inflation / precision: `{delta_p070['iou_0_30_recall']:+.6f}` / `{delta_p070['candidate_inflation']:+.6f}` / `{delta_p070['precision']:+.6f}`\n""")
 print(json.dumps({'split':args.split,'decision':decision,'retained':retained,'delta_vs_p070':delta_p070,'gated':{k:metrics[k] for k in ['iou_0_30_recall','candidate_inflation','precision']}},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
