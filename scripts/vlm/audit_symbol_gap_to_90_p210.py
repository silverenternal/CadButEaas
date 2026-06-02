#!/usr/bin/env python3
"""Audit gap from current P206g symbol overlay to 0.90 F1 target."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sweep_symbol_disagreement_backfill_p165 as p165
from fuse_symbol_detector_with_p182_p186 import load_jsonl, write_json

ROOT = Path(__file__).resolve().parents[2]
OVERLAY = ROOT / "reports/vlm/symbol_p206f_precision_repair_p206g_overlay.jsonl"
OUT_JSON = ROOT / "reports/vlm/symbol_gap_to_90_p210.json"
OUT_MD = ROOT / "reports/vlm/symbol_gap_to_90_p210.md"


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("row_id") or row.get("id"))


def target_label(item: dict[str, Any]) -> str:
    return str(item.get("semantic_type") or item.get("symbol_type") or item.get("raw_label") or item.get("label") or "generic_symbol")


def targets(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate((row.get("targets") or {}).get("symbol") or []):
        box = p165.bbox4(item.get("bbox"))
        if box:
            out.append({"id": str(item.get("target_id") or idx), "bbox": box, "label": target_label(item), "bucket": p165.bucket(box)})
    return out


def match_row(golds: list[dict[str, Any]], preds: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    used = set()
    tps=[]; fns=[]
    for gi,g in enumerate(golds):
        bi=None; bv=0.0
        for pi,p in enumerate(preds):
            if pi in used: continue
            v=p165.iou(g['bbox'], p['bbox'])
            if v>bv:
                bv=v; bi=pi
        if bi is not None and bv>=0.30:
            used.add(bi); tps.append({"gold":g,"pred":preds[bi],"iou":bv})
        else:
            best_center=False; best_iou=0.0; best_label=''
            for p in preds:
                best_iou=max(best_iou,p165.iou(g['bbox'],p['bbox']))
                if p165.center_covered(p['bbox'],g['bbox']):
                    best_center=True; best_label=p['label']
            fns.append({"gold":g,"best_iou":best_iou,"center_covered":best_center,"center_pred_label":best_label})
    fps=[{"pred":p} for i,p in enumerate(preds) if i not in used]
    return tps,fns,fps


def main():
    rows=load_jsonl(OVERLAY)
    by_label=Counter(); by_bucket=Counter(); fn_label=Counter(); fn_bucket=Counter(); fp_label=Counter(); fp_bucket=Counter()
    center_fn_label=Counter(); near_fn_label=Counter(); nohit_fn_label=Counter()
    row_records=[]; total_tp=total_gold=total_pred=0
    for row in rows:
        golds=targets(row); preds=p165.normalized(row.get('symbol_candidates') or [], 'p206g')
        tps,fns,fps=match_row(golds,preds)
        total_tp+=len(tps); total_gold+=len(golds); total_pred+=len(preds)
        for g in golds:
            by_label[g['label']]+=1; by_bucket[g['bucket']]+=1
        for item in fns:
            g=item['gold']; fn_label[g['label']]+=1; fn_bucket[g['bucket']]+=1
            if item['center_covered']:
                center_fn_label[g['label']]+=1
            elif item['best_iou']>=0.10:
                near_fn_label[g['label']]+=1
            else:
                nohit_fn_label[g['label']]+=1
        for item in fps:
            p=item['pred']; fp_label[p['label']]+=1; fp_bucket[p.get('bucket') or p165.bucket(p['bbox'])]+=1
        row_records.append({"row_id":row_id(row),"gold":len(golds),"pred":len(preds),"tp":len(tps),"fn":len(fns),"fp":len(fps),"f1":0 if len(preds)+len(golds)==0 else round(2*len(tps)/(len(preds)+len(golds)),6)})
    precision=total_tp/max(total_pred,1); recall=total_tp/max(total_gold,1); f1=0 if precision+recall==0 else 2*precision*recall/(precision+recall)
    needed_tp=int((0.9*(total_pred+total_gold)/2)+0.999999)
    label_table=[]
    for label,count in by_label.items():
        label_table.append({"label":label,"gold":count,"fn":fn_label[label],"recall":round((count-fn_label[label])/max(count,1),6),"center_fn":center_fn_label[label],"near_iou_fn":near_fn_label[label],"nohit_fn":nohit_fn_label[label],"fp":fp_label[label]})
    bucket_table=[]
    for bucket,count in by_bucket.items():
        bucket_table.append({"bucket":bucket,"gold":count,"fn":fn_bucket[bucket],"recall":round((count-fn_bucket[bucket])/max(count,1),6),"fp":fp_bucket[bucket]})
    report={
        "id":"P210_symbol_gap_to_90_audit",
        "metrics":{"tp":total_tp,"predicted":total_pred,"gold":total_gold,"precision":round(precision,6),"recall":round(recall,6),"f1":round(f1,6)},
        "target_0_90":{"needed_tp_at_current_pred_count":needed_tp,"extra_tp_needed":needed_tp-total_tp,"recall_gap":round(0.9-recall,6),"precision_gap":round(0.9-precision,6)},
        "fn_by_label":sorted(label_table,key=lambda x:(-x['fn'],x['label'])),
        "fn_by_bucket":sorted(bucket_table,key=lambda x:(-x['fn'],x['bucket'])),
        "worst_rows":sorted(row_records,key=lambda x:(-x['fn'],x['f1']))[:20],
        "interpretation":[
            "Current symbol F1 is far below 0.90; this is not a packaging problem.",
            "The dominant gap is recall: hundreds of missing true positives are required even before precision repair.",
            "Next work should train a stronger detector/segmenter or use gold-derived synthetic raster supervision at page/tile scale, not just gate existing candidates."
        ]
    }
    write_json(OUT_JSON, report)
    lines=["# P210 Symbol Gap-to-0.90 Audit","",f"Current F1: `{f1:.6f}`; precision `{precision:.6f}`; recall `{recall:.6f}`.",f"Extra TP needed for F1 0.90 at current prediction count: `{needed_tp-total_tp}`.","","## FN by Label","","| Label | Gold | FN | Recall | Center-only FN | Near-IoU FN | No-hit FN | FP |","|---|---:|---:|---:|---:|---:|---:|---:|"]
    for x in report['fn_by_label']:
        lines.append(f"| `{x['label']}` | {x['gold']} | {x['fn']} | {x['recall']:.6f} | {x['center_fn']} | {x['near_iou_fn']} | {x['nohit_fn']} | {x['fp']} |")
    lines += ["","## FN by Bucket","","| Bucket | Gold | FN | Recall | FP |","|---|---:|---:|---:|---:|"]
    for x in report['fn_by_bucket']:
        lines.append(f"| `{x['bucket']}` | {x['gold']} | {x['fn']} | {x['recall']:.6f} | {x['fp']} |")
    lines += ["","## Worst Rows","","| Row | Gold | Pred | TP | FN | FP | Row F1 |","|---|---:|---:|---:|---:|---:|---:|"]
    for x in report['worst_rows']:
        lines.append(f"| `{x['row_id']}` | {x['gold']} | {x['pred']} | {x['tp']} | {x['fn']} | {x['fp']} | {x['f1']:.6f} |")
    OUT_MD.write_text('\n'.join(lines)+'\n')
    print(json.dumps({"metrics":report['metrics'],"target_0_90":report['target_0_90'],"outputs":{"json":str(OUT_JSON),"md":str(OUT_MD)}},indent=2))
if __name__=='__main__': main()
