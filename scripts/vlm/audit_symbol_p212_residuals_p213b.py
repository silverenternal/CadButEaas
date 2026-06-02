#!/usr/bin/env python3
"""Audit residual false negatives after P212 precision-repaired fusion."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fuse_symbol_p206g_with_p211_p212 import area_bucket, bbox_iou, load_p206g, write_json

ROOT = Path(__file__).resolve().parents[2]
OVERLAY = ROOT / "reports/vlm/symbol_p206g_p212_specialist_precision_repair_overlay.jsonl"
REPORT = ROOT / "reports/vlm/symbol_p212_residuals_p213b.json"
MD = ROOT / "reports/vlm/symbol_p212_residuals_p213b.md"


def matched_indices(preds: list[dict[str, Any]], golds: list[dict[str, Any]]) -> set[int]:
    matched=set(); used=set()
    for gi,gold in enumerate(golds):
        best_iou=0.0; best_pi=None
        gbox=[float(v) for v in gold["bbox"]]
        for pi,pred in enumerate(preds):
            if pi in used: continue
            iou=bbox_iou([float(v) for v in pred["bbox"]], gbox)
            if iou>best_iou:
                best_iou=iou; best_pi=pi
        if best_pi is not None and best_iou>=0.30:
            used.add(best_pi); matched.add(gi)
    return matched


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument("--overlay", default=str(OVERLAY))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--md", default=str(MD))
    args=parser.parse_args()
    rows,preds,golds=load_p206g(Path(args.overlay))
    residual=[]; by_label=Counter(); by_bucket=Counter(); by_row=Counter(); center_near=Counter()
    for row in rows:
        row_id=str(row.get("id") or row.get("row_id"))
        gold_list=list(golds[row_id].values())
        matched=matched_indices(preds.get(row_id,[]), gold_list)
        for index,gold in enumerate(gold_list):
            label=str(gold.get("label")); box=[float(v) for v in gold["bbox"]]; bucket=area_bucket(box)
            if index in matched: continue
            best_iou=0.0; best_label=None; best_dist=1e9
            cx=(box[0]+box[2])/2; cy=(box[1]+box[3])/2
            for pred in preds.get(row_id,[]):
                pbox=[float(v) for v in pred["bbox"]]
                iou=bbox_iou(pbox,box)
                if iou>best_iou:
                    best_iou=iou; best_label=pred.get("label")
                pcx=(pbox[0]+pbox[2])/2; pcy=(pbox[1]+pbox[3])/2
                best_dist=min(best_dist, ((cx-pcx)**2+(cy-pcy)**2)**0.5)
            if best_dist<=12: center_near[label]+=1
            by_label[label]+=1; by_bucket[bucket]+=1; by_row[row_id]+=1
            residual.append({"row_id":row_id,"target_id":gold.get("target_id"),"label":label,"bbox":box,"bucket":bucket,"best_iou":round(best_iou,4),"best_label":best_label,"best_center_dist":round(best_dist,2)})
    residual.sort(key=lambda r:(r["label"]!="stair", r["bucket"]!="tiny_le_64", -r["best_center_dist"]))
    report={
        "id":"P213b_residual_after_P212_precision_repair",
        "overlay":str(Path(args.overlay)),
        "total_residual_fn":len(residual),
        "by_label":dict(by_label),
        "by_bucket":dict(by_bucket),
        "center_near_by_label":dict(center_near),
        "worst_rows":dict(by_row.most_common(20)),
        "residuals":residual,
        "claim_boundary":"Residual audit after P212 precision-repaired P101 overlay; used for next training target construction only."
    }
    write_json(Path(args.report), report)
    lines=["# P213b Residual Audit", "", f"- Total residual FN: {len(residual)}", f"- By label: `{json.dumps(dict(by_label), ensure_ascii=False)}`", f"- By bucket: `{json.dumps(dict(by_bucket), ensure_ascii=False)}`", f"- Worst rows: `{json.dumps(dict(by_row.most_common(10)), ensure_ascii=False)}`", "", "## Claim Boundary", report["claim_boundary"]]
    Path(args.md).write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({k:report[k] for k in ["total_residual_fn","by_label","by_bucket","worst_rows"]}, ensure_ascii=False, indent=2))

if __name__=="__main__": main()
