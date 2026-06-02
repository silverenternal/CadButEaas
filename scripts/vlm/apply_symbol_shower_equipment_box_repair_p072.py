#!/usr/bin/env python3
"""Apply fixed P0-72 shower/equipment geometric box repair."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from apply_symbol_rtdetr_complement_policy_p065 import read_exported_golds
from sweep_symbol_rtdetr_complement_gate_p063 import read_predictions, score
from sweep_symbol_shower_equipment_box_repair_p072 import repaired_predictions
from sweep_symbol_center_only_box_repair_p067 import rows_from_predictions
from train_symbol_tile_detector_v20 import rel, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21"
DEFAULT_YOLO_DIR = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27"
FIXED_CONFIG = {"labels":"shower_equipment","areas":"all","score_min":0.1,"scale_x":0.8,"scale_y":0.8,"max_add_per_page":10}

def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--data', default=str(DEFAULT_DATA))
    parser.add_argument('--yolo-dir', default=str(DEFAULT_YOLO_DIR))
    parser.add_argument('--split', required=True)
    parser.add_argument('--policy-predictions', required=True)
    parser.add_argument('--output-predictions', required=True)
    parser.add_argument('--output-json', required=True)
    parser.add_argument('--output-md', required=True)
    args=parser.parse_args()
    base=read_predictions(Path(args.policy_predictions))
    golds=read_exported_golds(Path(args.data), Path(args.yolo_dir), args.split, set(base))
    baseline=score(golds, base, {row_id: [] for row_id in base}, {"labels":"all","areas":"all","score_min":1.1,"max_iou_with_v28":0.0,"max_add_per_page":0})
    repaired_map=repaired_predictions(base, FIXED_CONFIG)
    metrics=score(golds, repaired_map, {row_id: [] for row_id in repaired_map}, {"labels":"all","areas":"all","score_min":1.1,"max_iou_with_v28":0.0,"max_add_per_page":0})
    delta={k:round(metrics[k]-baseline[k],6) for k in ['iou_0_30_recall','center_recall','candidate_inflation','precision']}
    decision='positive_locked_but_precision_cost' if delta['iou_0_30_recall']>0 and delta['candidate_inflation']<=0.75 else 'negative_do_not_apply'
    outp=Path(args.output_predictions); outj=Path(args.output_json); outm=Path(args.output_md)
    outp.parent.mkdir(parents=True, exist_ok=True)
    write_jsonl(outp, rows_from_predictions(repaired_map))
    report={
      'version':'symbol_shower_equipment_box_repair_p072_fixed',
      'split':args.split,
      'source_integrity':'offline gold is validation-only; runtime repair uses raster-derived prediction fields only',
      'inputs':{'policy_predictions':rel(Path(args.policy_predictions))},
      'outputs':{'predictions':rel(outp),'json':rel(outj),'markdown':rel(outm)},
      'repair_config':FIXED_CONFIG,
      'baseline_policy':baseline,
      'repaired_policy':metrics,
      'delta_vs_policy':delta,
      'decision':decision,
    }
    outj.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    outm.write_text(f"""# P0-72 fixed shower/equipment box repair - {args.split}\n\n## Decision\n\n- `{decision}`\n\n## Metrics\n\n- baseline IoU / inflation / precision: `{baseline['iou_0_30_recall']:.6f}` / `{baseline['candidate_inflation']:.6f}` / `{baseline['precision']:.6f}`\n- repaired IoU / inflation / precision: `{metrics['iou_0_30_recall']:.6f}` / `{metrics['candidate_inflation']:.6f}` / `{metrics['precision']:.6f}`\n- delta IoU / inflation / precision: `{delta['iou_0_30_recall']:+.6f}` / `{delta['candidate_inflation']:+.6f}` / `{delta['precision']:+.6f}`\n- shower IoU: `{baseline['per_label_iou_recall'].get('shower',0.0):.6f}` -> `{metrics['per_label_iou_recall'].get('shower',0.0):.6f}`\n- equipment IoU: `{baseline['per_label_iou_recall'].get('equipment',0.0):.6f}` -> `{metrics['per_label_iou_recall'].get('equipment',0.0):.6f}`\n""")
    print(json.dumps({'split':args.split,'decision':decision,'delta':delta,'metrics':{k:metrics[k] for k in ['iou_0_30_recall','candidate_inflation','precision']}},ensure_ascii=False,indent=2))

if __name__ == '__main__':
    main()
