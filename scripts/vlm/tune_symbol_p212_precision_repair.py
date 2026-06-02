#!/usr/bin/env python3
"""Fine tune per-label thresholds for P212 fusion."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from fuse_symbol_p206g_with_p212_specialist import P206G, P212, build_overlay, fuse, load_p212, metric_key
from fuse_symbol_p206g_with_p211_p212 import load_p206g, score_predictions, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm/symbol_p206g_p212_specialist_precision_repair_eval.json"
OVERLAY = ROOT / "reports/vlm/symbol_p206g_p212_specialist_precision_repair_overlay.jsonl"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p206g", default=str(P206G))
    parser.add_argument("--p212", default=str(P212))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--overlay", default=str(OVERLAY))
    args = parser.parse_args()
    rows, core, golds = load_p206g(Path(args.p206g))
    p212 = load_p212(Path(args.p212))
    baseline, _ = score_predictions(core, golds, 0.0, 0.98, 900, 0)
    reports=[]
    for shower_t in [0.55,0.58,0.60,0.62]:
        for sink_t in [0.60,0.65,0.70]:
            for equipment_t in [0.55,0.60,0.65]:
                for max_add in [20,30,40]:
                    for min_dist in [0,2,4]:
                        policy={
                            "name": f"p212pr_sh{shower_t}_si{sink_t}_eq{equipment_t}_a{max_add}_d{min_dist}",
                            "allowed_labels": ["sink","shower","equipment"],
                            "threshold": 0.60,
                            "label_thresholds": {"shower": shower_t, "sink": sink_t, "equipment": equipment_t},
                            "max_add_per_row": max_add,
                            "max_iou_to_core": 0.25,
                            "min_dist_to_core": min_dist,
                            "same_label_only": False,
                        }
                        fused=fuse(core,p212,policy)
                        metrics,_=score_predictions(fused,golds,0.0,0.98,900,0)
                        additions=sum(max(0,len(fused[row_id])-len(core.get(row_id,[]))) for row_id in fused)
                        reports.append({"policy":policy,"metrics":metrics,"additions":additions})
    reports.sort(key=metric_key, reverse=True)
    best=reports[0]
    fused=fuse(core,p212,best["policy"])
    write_jsonl(Path(args.overlay), build_overlay(rows,fused,best["policy"]))
    result={
        "id":"P212_precision_repair_grid",
        "claim_boundary":"P101/P206g policy-search evidence; bootstrap required after selecting repair policy.",
        "baseline":baseline,
        "selected":best,
        "top20":reports[:20],
        "outputs":{"overlay":str(Path(args.overlay)),"report":str(Path(args.report))},
    }
    write_json(Path(args.report), result)
    print(json.dumps({"baseline":baseline["symbol_bbox_iou_0_30"],"selected":best["metrics"]["symbol_bbox_iou_0_30"],"additions":best["additions"],"policy":best["policy"]},ensure_ascii=False,indent=2))

if __name__ == "__main__":
    main()
