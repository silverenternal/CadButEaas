#!/usr/bin/env python3
"""Apply P0-76 balanced opt-in symbol policy."""
from __future__ import annotations
import argparse,json,sys
from pathlib import Path
SCRIPT_DIR=Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path: sys.path.insert(0,str(SCRIPT_DIR))
from apply_symbol_precision_gated_policy_p070 import rows_to_map
from apply_symbol_rtdetr_complement_policy_p065 import apply_policy as apply_p065_rows, read_exported_golds
from sweep_symbol_center_only_box_repair_p067 import repaired_predictions, rows_from_predictions
from sweep_symbol_added_candidate_precision_gate_p069 import gated_predictions as apply_p070_gate
from sweep_symbol_added_candidate_reranker_p075 import gate_predictions as apply_p075_gate
from sweep_symbol_rtdetr_complement_gate_p063 import read_predictions, score
from train_symbol_tile_detector_v20 import rel, write_jsonl
ROOT=Path(__file__).resolve().parents[2]
DEFAULT_CONFIG=ROOT/'configs/vlm/symbol_balanced_policy_p076.json'
DEFAULT_P070_CONFIG=ROOT/'configs/vlm/symbol_precision_gated_policy_p070.json'
DEFAULT_DATA=ROOT/'datasets/symbol_tile_detector_tiny_sahi_v21'
DEFAULT_YOLO_DIR=ROOT/'datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27'

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--config',default=str(DEFAULT_CONFIG)); ap.add_argument('--p070-config',default=str(DEFAULT_P070_CONFIG)); ap.add_argument('--data',default=str(DEFAULT_DATA)); ap.add_argument('--yolo-dir',default=str(DEFAULT_YOLO_DIR)); ap.add_argument('--split',required=True); ap.add_argument('--v28-predictions',required=True); ap.add_argument('--rtdetr-predictions',required=True); ap.add_argument('--output-predictions',required=True); ap.add_argument('--output-summary-json',required=True); ap.add_argument('--output-summary-md',required=True); args=ap.parse_args()
 cfg=json.loads(Path(args.config).read_text()); p070cfg=json.loads(Path(args.p070_config).read_text())
 v28=read_predictions(Path(args.v28_predictions)); rtdetr=read_predictions(Path(args.rtdetr_predictions))
 p065=rows_to_map(apply_p065_rows(v28,rtdetr,p070cfg['generation_steps'][0]['gate']))
 p068=repaired_predictions(p065,p070cfg['generation_steps'][1]['gate'])
 p070=apply_p070_gate(v28,p068,p070cfg['precision_gate'])
 p076=apply_p075_gate(v28,p070,cfg['balanced_reranker_gate'])
 golds=read_exported_golds(Path(args.data),Path(args.yolo_dir),args.split,set(v28)|set(rtdetr)|set(p076))
 baseline=score(golds,v28,{r:[] for r in v28},{'labels':'all','areas':'all','score_min':1.1,'max_iou_with_v28':0.0,'max_add_per_page':0})
 p070m=score(golds,p070,{r:[] for r in p070},{'labels':'all','areas':'all','score_min':1.1,'max_iou_with_v28':0.0,'max_add_per_page':0})
 metrics=score(golds,p076,{r:[] for r in p076},{'labels':'all','areas':'all','score_min':1.1,'max_iou_with_v28':0.0,'max_add_per_page':0})
 delta={k:round(metrics[k]-baseline[k],6) for k in ['iou_0_30_recall','center_recall','candidate_inflation','precision']}
 delta70={k:round(metrics[k]-p070m[k],6) for k in ['iou_0_30_recall','center_recall','candidate_inflation','precision']}
 outp=Path(args.output_predictions); outj=Path(args.output_summary_json); outm=Path(args.output_summary_md); outp.parent.mkdir(parents=True,exist_ok=True); write_jsonl(outp,rows_from_predictions(p076))
 summary={'version':'symbol_balanced_policy_p076_result','split':args.split,'source_integrity':cfg['runtime_input_boundary'],'policy_config':rel(Path(args.config)),'inputs':{'v28_predictions':rel(Path(args.v28_predictions)),'rtdetr_predictions':rel(Path(args.rtdetr_predictions))},'outputs':{'predictions':rel(outp),'summary_json':rel(outj),'summary_md':rel(outm)},'baseline_v28':baseline,'p070_recall_policy':p070m,'balanced_policy':metrics,'delta_vs_v28':delta,'delta_vs_p070':delta70,'decision':'balanced_opt_in_policy_positive' if delta['iou_0_30_recall']>0 and metrics['precision']>=baseline['precision'] else 'policy_requires_review'}
 outj.write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
 outm.write_text(f"""# P0-76 balanced symbol policy - {args.split}\n\n## Decision\n\n- `{summary['decision']}`\n\n## Metrics\n\n- v28 IoU / inflation / precision: `{baseline['iou_0_30_recall']:.6f}` / `{baseline['candidate_inflation']:.6f}` / `{baseline['precision']:.6f}`\n- P0-70 IoU / inflation / precision: `{p070m['iou_0_30_recall']:.6f}` / `{p070m['candidate_inflation']:.6f}` / `{p070m['precision']:.6f}`\n- P0-76 IoU / inflation / precision: `{metrics['iou_0_30_recall']:.6f}` / `{metrics['candidate_inflation']:.6f}` / `{metrics['precision']:.6f}`\n- delta vs v28 IoU / inflation / precision: `{delta['iou_0_30_recall']:+.6f}` / `{delta['candidate_inflation']:+.6f}` / `{delta['precision']:+.6f}`\n- delta vs P0-70 IoU / inflation / precision: `{delta70['iou_0_30_recall']:+.6f}` / `{delta70['candidate_inflation']:+.6f}` / `{delta70['precision']:+.6f}`\n""")
 print(json.dumps({'split':args.split,'decision':summary['decision'],'balanced':{k:metrics[k] for k in ['iou_0_30_recall','candidate_inflation','precision']},'delta_vs_p070':delta70},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
