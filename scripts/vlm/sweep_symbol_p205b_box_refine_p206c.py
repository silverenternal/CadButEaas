#!/usr/bin/env python3
"""P206c compact box-scale sweep for P205b additions over P202."""
from __future__ import annotations

import argparse, copy, json
from pathlib import Path
from typing import Any

import sweep_symbol_disagreement_backfill_p165 as p165
from fuse_symbol_detector_with_p182_p186 import detector_by_row, load_jsonl, write_json, write_jsonl

WET = {"sink", "shower", "bathtub"}
TARGET = {"sink", "shower", "equipment", "stair", "appliance"}


def rid(row: dict[str, Any]) -> str:
    return str(row.get("row_id") or row.get("id"))


def targets(row: dict[str, Any]) -> list[dict[str, Any]]:
    out=[]
    for i,t in enumerate((row.get("targets") or {}).get("symbol") or []):
        b=p165.bbox4(t.get("bbox"))
        if b is not None: out.append({"id":str(t.get("target_id") or i),"bbox":b,"bucket":p165.bucket(b)})
    return out


def scale_box(box: list[float], sx: float, sy: float) -> list[float]:
    cx=(box[0]+box[2])/2; cy=(box[1]+box[3])/2; w=max(1,box[2]-box[0])*sx; h=max(1,box[3]-box[1])*sy
    return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]


def transform(pred: dict[str, Any], policy: dict[str, Any]) -> dict[str, Any]:
    sx,sy=policy["scale"]
    if pred["bucket"] in {"tiny","small"}:
        sx*=policy["small_scale"][0]; sy*=policy["small_scale"][1]
    if pred["label"] in policy.get("label_scale",{}):
        lx,ly=policy["label_scale"][pred["label"]]; sx*=lx; sy*=ly
    out=copy.deepcopy(pred); out["bbox"]=scale_box(pred["bbox"],sx,sy); out["bucket"]=p165.bucket(out["bbox"])
    raw=copy.deepcopy(out.get("raw") or {}); raw["bbox"]=out["bbox"]; raw.setdefault("metadata",{})["p206c_scale"]={"sx":sx,"sy":sy,"policy":policy["name"]}; out["raw"]=raw
    return out


def nms(preds: list[dict[str, Any]], thr: float) -> list[dict[str, Any]]:
    kept=[]
    for p in sorted(preds,key=lambda x:float(x.get("score") or 0),reverse=True):
        if all(p165.iou(p["bbox"],q["bbox"])<thr for q in kept): kept.append(p)
    return kept


def fuse(core: list[dict[str, Any]], det: list[dict[str, Any]], pol: dict[str, Any]) -> list[dict[str, Any]]:
    labels=set(pol["labels"]); adds=[]
    for d in nms(det, pol["nms"]):
        if d["label"] not in labels or float(d.get("score") or 0)<pol["min_score"]: continue
        p=transform(d,pol); bi,bd=p165.best_overlap_to_core(p,core)
        if bi>=pol["max_iou"] or bd<pol["min_dist"]: continue
        adds.append(p)
    out=sorted(core,key=lambda x:float(x.get("score") or 0),reverse=True)+sorted(adds,key=lambda x:float(x.get("score") or 0),reverse=True)[:pol["max_add"]]
    return sorted(out,key=lambda x:float(x.get("score") or 0),reverse=True)[:128]


def policies() -> list[dict[str, Any]]:
    out=[]
    label_scales=[{}, {"sink":(2.0,1.4),"shower":(1.6,1.6)}, {"sink":(1.4,2.0),"shower":(2.2,1.2)}, {"equipment":(1.8,1.8),"stair":(1.4,1.4),"appliance":(1.3,1.3)}]
    for labels in [WET,TARGET]:
      for min_score in [0.01,0.02,0.05]:
       for max_add in [0,1,2,3,5]:
        for small_scale in [(1,1),(1.5,1.5),(2.0,1.5),(1.5,2.0),(2.5,2.0)]:
         for label_scale in label_scales:
          out.append({"name":f"p206c_l{len(labels)}_s{min_score}_a{max_add}_ss{small_scale}_ls{len(label_scale)}","labels":sorted(labels),"min_score":min_score,"max_add":max_add,"small_scale":small_scale,"label_scale":label_scale,"scale":(1,1),"max_iou":0.08,"min_dist":8,"nms":0.55})
    return out


def materialize(rows, pred_map, policy):
    outs=[]
    for row in rows:
        r=copy.deepcopy(row); c=[]
        for i,p in enumerate(pred_map.get(rid(row),[])):
            item=copy.deepcopy(p.get("raw") or {}); item["bbox"]=p["bbox"]; item["symbol_type"]=p["label"]; item["confidence"]=float(p["score"]); item["id"]=f"{rid(row)}_p206c_{i:05d}"; item["target_id"]=item["id"]; item["source"]="symbol_p206c_box_refine"; c.append(item)
        r["symbol_candidates"]=c
        if isinstance(r.get("expected_json"),dict): r["expected_json"]["symbol_candidates"]=copy.deepcopy(c)
        r["symbol_policy_overlay"]={"policy_id":"p206c_box_refine","policy":policy}; outs.append(r)
    return outs


def main():
    ap=argparse.ArgumentParser(); ap.add_argument("--base-overlay",default="reports/vlm/symbol_context_verifier_p202_overlay.jsonl"); ap.add_argument("--detector-predictions",default="reports/vlm/symbol_tiled_recall_p205b_30k_p101_predictions.jsonl"); ap.add_argument("--out-json",default="configs/vlm/symbol_p205b_box_refine_p206c.json"); ap.add_argument("--out-md",default="reports/vlm/symbol_p205b_box_refine_p206c.md"); ap.add_argument("--out-overlay",default="reports/vlm/symbol_p205b_box_refine_p206c_overlay.jsonl"); a=ap.parse_args()
    rows=load_jsonl(Path(a.base_overlay)); det=detector_by_row(Path(a.detector_predictions)); gold={rid(r):targets(r) for r in rows}; core={rid(r):p165.normalized(r.get("symbol_candidates") or [],"p202") for r in rows}; base=p165.evaluate(gold,core)
    scored=[]
    for pol in policies():
        pm={k:fuse(core.get(k,[]),det.get(k,[]),pol) for k in gold}; scored.append({"policy":pol,"metrics":p165.evaluate(gold,pm)})
    scored.sort(key=lambda x:(x["metrics"]["f1"],x["metrics"]["recall"],x["metrics"]["center_recall"],-x["metrics"]["prediction_inflation"]),reverse=True); best=scored[0]
    best_map={k:fuse(core.get(k,[]),det.get(k,[]),best["policy"]) for k in gold}; write_jsonl(Path(a.out_overlay),materialize(rows,best_map,best["policy"]))
    rep={"id":"P206c_p205b_box_refine_sweep","baseline_metrics":base,"best_policy":best["policy"],"best_metrics":best["metrics"],"delta_vs_baseline":p165.delta(best["metrics"],base),"decision":"promote_candidate" if best["metrics"]["f1"]>base["f1"] else "no_promotion_keep_P202","top_candidates":scored[:30],"outputs":{"json":a.out_json,"md":a.out_md,"overlay":a.out_overlay}}
    write_json(Path(a.out_json),rep)
    lines=["# P206c P205b Box Refine Sweep","",f"Decision: **{rep['decision']}**","","| Variant | Precision | Recall | F1 | Center | Inflation |","|---|---:|---:|---:|---:|---:|",f"| `P202_baseline` | {base['precision']:.6f} | {base['recall']:.6f} | {base['f1']:.6f} | {base['center_recall']:.6f} | {base['prediction_inflation']:.6f} |",f"| `P206c_best` | {best['metrics']['precision']:.6f} | {best['metrics']['recall']:.6f} | {best['metrics']['f1']:.6f} | {best['metrics']['center_recall']:.6f} | {best['metrics']['prediction_inflation']:.6f} |","","## Best Policy","","```json",json.dumps(best["policy"],ensure_ascii=False,indent=2),"```","","## Top Policies"]
    for it in scored[:20]:
        m=it["metrics"]; lines.append(f"- `{it['policy']['name']}` F1 `{m['f1']:.6f}` P `{m['precision']:.6f}` R `{m['recall']:.6f}` center `{m['center_recall']:.6f}` inflation `{m['prediction_inflation']:.6f}`")
    Path(a.out_md).parent.mkdir(parents=True,exist_ok=True); Path(a.out_md).write_text("\n".join(lines)+"\n")
    print(json.dumps({"decision":rep["decision"],"baseline":base,"best_metrics":best["metrics"],"delta":rep["delta_vs_baseline"],"outputs":rep["outputs"]},ensure_ascii=False,indent=2))
if __name__=="__main__": main()
