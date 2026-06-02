#!/usr/bin/env python3
"""Apply fixed P0-73 shower/stair RTDETR no-center recovery gate."""

from __future__ import annotations

import argparse, json, sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from apply_symbol_rtdetr_complement_policy_p065 import read_exported_golds
from probe_symbol_shower_stair_no_center_recovery_p073 import add_candidates, rows_from_predictions
from sweep_symbol_rtdetr_complement_gate_p063 import read_predictions, score
from train_symbol_tile_detector_v20 import rel, write_jsonl

ROOT=Path(__file__).resolve().parents[2]
DEFAULT_DATA=ROOT/'datasets/symbol_tile_detector_tiny_sahi_v21'
DEFAULT_YOLO_DIR=ROOT/'datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27'
FIXED_CONFIG={"labels":"shower_stair","areas":"small_medium","score_min":0.1,"min_iou_with_base":0.0,"max_iou_with_base":0.5,"max_add_per_page":3}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--data', default=str(DEFAULT_DATA)); ap.add_argument('--yolo-dir', default=str(DEFAULT_YOLO_DIR))
    ap.add_argument('--split', required=True); ap.add_argument('--base-predictions', required=True); ap.add_argument('--rtdetr-predictions', required=True)
    ap.add_argument('--output-predictions', required=True); ap.add_argument('--output-json', required=True); ap.add_argument('--output-md', required=True)
    args=ap.parse_args()
    base=read_predictions(Path(args.base_predictions)); rtdetr=read_predictions(Path(args.rtdetr_predictions))
    golds=read_exported_golds(Path(args.data), Path(args.yolo_dir), args.split, set(base)|set(rtdetr))
    baseline=score(golds, base, {row_id: [] for row_id in base}, {"labels":"all","areas":"all","score_min":1.1,"max_iou_with_v28":0.0,"max_add_per_page":0})
    pred=add_candidates(base, rtdetr, FIXED_CONFIG)
    metrics=score(golds, pred, {row_id: [] for row_id in pred}, {"labels":"all","areas":"all","score_min":1.1,"max_iou_with_v28":0.0,"max_add_per_page":0})
    delta={k:round(metrics[k]-baseline[k],6) for k in ['iou_0_30_recall','center_recall','candidate_inflation','precision']}
    decision='positive_locked_tiny_gain' if delta['iou_0_30_recall']>0 and delta['candidate_inflation']<=0.5 and delta['precision']>=-0.003 else 'negative_do_not_package'
    outp=Path(args.output_predictions); outj=Path(args.output_json); outm=Path(args.output_md); outp.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(outp, rows_from_predictions(pred))
    report={'version':'symbol_shower_stair_no_center_recovery_p073_fixed','split':args.split,'source_integrity':'gold validation-only; runtime additions use raster-derived RTDETR predictions only','inputs':{'base_predictions':rel(Path(args.base_predictions)),'rtdetr_predictions':rel(Path(args.rtdetr_predictions))},'outputs':{'predictions':rel(outp),'json':rel(outj),'markdown':rel(outm)},'gate_config':FIXED_CONFIG,'baseline_p070':baseline,'recovered_policy':metrics,'delta_vs_p070':delta,'decision':decision}
    outj.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    outm.write_text(f"""# P0-73 fixed shower/stair no-center recovery - {args.split}\n\n## Decision\n\n- `{decision}`\n\n## Metrics\n\n- baseline IoU / inflation / precision: `{baseline['iou_0_30_recall']:.6f}` / `{baseline['candidate_inflation']:.6f}` / `{baseline['precision']:.6f}`\n- recovered IoU / inflation / precision: `{metrics['iou_0_30_recall']:.6f}` / `{metrics['candidate_inflation']:.6f}` / `{metrics['precision']:.6f}`\n- delta IoU / inflation / precision: `{delta['iou_0_30_recall']:+.6f}` / `{delta['candidate_inflation']:+.6f}` / `{delta['precision']:+.6f}`\n- shower IoU: `{baseline['per_label_iou_recall'].get('shower',0.0):.6f}` -> `{metrics['per_label_iou_recall'].get('shower',0.0):.6f}`\n- stair IoU: `{baseline['per_label_iou_recall'].get('stair',0.0):.6f}` -> `{metrics['per_label_iou_recall'].get('stair',0.0):.6f}`\n""")
    print(json.dumps({'split':args.split,'decision':decision,'delta':delta,'metrics':{k:metrics[k] for k in ['iou_0_30_recall','candidate_inflation','precision']}},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
