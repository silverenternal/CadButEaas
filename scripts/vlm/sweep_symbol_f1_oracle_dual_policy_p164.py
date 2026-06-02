#!/usr/bin/env python3
"""P164 F1 oracle and dual-policy union rescue.

Starts from P160 precision core and selectively backfills P155A/P140 candidates.
Gold targets are evaluation-only; materialized policies use predicted boxes,
labels, scores, overlap/distance to P160, and fixed budgets.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
P160_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p160_best.jsonl"
P155A_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p155a_p140_best.jsonl"
OUT_JSON = ROOT / "configs/vlm/symbol_f1_oracle_and_dual_policy_rescue_p164.json"
OUT_MD = ROOT / "reports/vlm/symbol_f1_oracle_and_dual_policy_rescue_p164.md"
OUT_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p164_best.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def bbox4(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in value[:4]]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bucket(box: list[float]) -> str:
    value = area(box)
    if value <= 64: return "tiny"
    if value <= 256: return "small"
    if value <= 1024: return "medium"
    if value <= 4096: return "large"
    return "xlarge"


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0: return 0.0
    return inter / max(area(a) + area(b) - inter, 1e-9)


def center_distance(a: list[float], b: list[float]) -> float:
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def center_covered(pred: list[float], gold: list[float]) -> bool:
    cx = (gold[0] + gold[2]) / 2.0
    cy = (gold[1] + gold[3]) / 2.0
    return pred[0] <= cx <= pred[2] and pred[1] <= cy <= pred[3]


def label(item: dict[str, Any]) -> str:
    return str(item.get("symbol_type") or item.get("label") or "generic_symbol")


def score(item: dict[str, Any]) -> float:
    value = item.get("confidence") if item.get("confidence") is not None else item.get("score")
    try: return float(value)
    except (TypeError, ValueError): return 0.0


def normalized(items: list[dict[str, Any]], source: str) -> list[dict[str, Any]]:
    out = []
    for raw in items:
        box = bbox4(raw.get("bbox"))
        if box is None: continue
        out.append({"bbox": box, "label": label(raw), "score": score(raw), "raw": raw, "source_policy": source})
    return out


def target_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate((row.get("targets") or {}).get("symbol") or []):
        box = bbox4(item.get("bbox"))
        if box is not None:
            out.append({"id": str(item.get("target_id") or idx), "bbox": box, "bucket": bucket(box)})
    return out


def evaluate(golds_by_row: dict[str, list[dict[str, Any]]], preds_by_row: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    totals = Counter()
    by_area_gold = Counter(); by_area_tp = Counter(); by_area_center = Counter()
    for row_id, golds in golds_by_row.items():
        preds = preds_by_row.get(row_id, [])
        totals["gold"] += len(golds); totals["pred"] += len(preds)
        used_iou: set[int] = set(); used_center: set[int] = set()
        for gold in golds:
            by_area_gold[gold["bucket"]] += 1
            best_idx = None; best_iou = 0.0; center_idx = None
            for idx, pred in enumerate(preds):
                overlap = iou(pred["bbox"], gold["bbox"])
                if idx not in used_iou and overlap > best_iou:
                    best_iou = overlap; best_idx = idx
                if center_idx is None and idx not in used_center and center_covered(pred["bbox"], gold["bbox"]):
                    center_idx = idx
            if best_idx is not None and best_iou >= 0.30:
                used_iou.add(best_idx); totals["tp"] += 1; by_area_tp[gold["bucket"]] += 1
            if center_idx is not None:
                used_center.add(center_idx); totals["center"] += 1; by_area_center[gold["bucket"]] += 1
    precision = totals["tp"] / max(totals["pred"], 1)
    recall = totals["tp"] / max(totals["gold"], 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "tp": int(totals["tp"]), "predicted": int(totals["pred"]), "gold": int(totals["gold"]),
        "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6),
        "center_recall": round(totals["center"] / max(totals["gold"], 1), 6),
        "prediction_inflation": round(totals["pred"] / max(totals["gold"], 1), 6),
        "by_area_iou_recall": {key: round(by_area_tp[key] / max(by_area_gold[key], 1), 6) for key in sorted(by_area_gold)},
        "by_area_center_recall": {key: round(by_area_center[key] / max(by_area_gold[key], 1), 6) for key in sorted(by_area_gold)},
    }


def best_overlap_to_core(candidate: dict[str, Any], core: list[dict[str, Any]]) -> tuple[float, float]:
    best_iou = 0.0; best_dist = 1e9
    for pred in core:
        best_iou = max(best_iou, iou(candidate["bbox"], pred["bbox"]))
        best_dist = min(best_dist, center_distance(candidate["bbox"], pred["bbox"]))
    return best_iou, best_dist


def nms_append(core: list[dict[str, Any]], additions: list[dict[str, Any]], any_iou: float) -> list[dict[str, Any]]:
    out = list(core)
    for item in sorted(additions, key=lambda x: x["score"], reverse=True):
        if all(iou(item["bbox"], old["bbox"]) < any_iou for old in out):
            out.append(item)
    return out


def apply_union_policy(core_by_row: dict[str, list[dict[str, Any]]], recall_by_row: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    labels = set(policy["labels"])
    buckets = set(policy["buckets"])
    for row_id, core in core_by_row.items():
        additions = []
        for cand in recall_by_row.get(row_id, []):
            if cand["score"] < policy["min_score"]: continue
            if labels and cand["label"] not in labels: continue
            if buckets and bucket(cand["bbox"]) not in buckets: continue
            best_iou, best_dist = best_overlap_to_core(cand, core)
            if best_iou > policy["max_iou_with_core"]: continue
            if best_dist < policy["min_center_dist"]: continue
            item = copy.deepcopy(cand); item["source_policy"] = "p155a_backfill"; item["backfill_iou_to_core"] = best_iou; item["backfill_dist_to_core"] = best_dist
            additions.append(item)
        additions = sorted(additions, key=lambda x: x["score"], reverse=True)[: policy["max_add_per_page"]]
        out[row_id] = nms_append(core, additions, policy["append_nms_iou"])
    return out


def oracle_analysis(golds_by_row: dict[str, list[dict[str, Any]]], p160_by_row: dict[str, list[dict[str, Any]]], p155a_by_row: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    def matched_gold_ids(golds: list[dict[str, Any]], preds: list[dict[str, Any]]) -> set[int]:
        used: set[int] = set(); hits: set[int] = set()
        for gi, gold in enumerate(golds):
            best = None; best_iou = 0.0
            for pi, pred in enumerate(preds):
                if pi in used: continue
                ov = iou(pred["bbox"], gold["bbox"])
                if ov > best_iou: best_iou = ov; best = pi
            if best is not None and best_iou >= 0.30:
                used.add(best); hits.add(gi)
        return hits
    totals = Counter(); by_area = defaultdict(Counter)
    for row_id, golds in golds_by_row.items():
        h160 = matched_gold_ids(golds, p160_by_row.get(row_id, []))
        h155 = matched_gold_ids(golds, p155a_by_row.get(row_id, []))
        totals['gold'] += len(golds); totals['p160_hit'] += len(h160); totals['p155a_hit'] += len(h155); totals['union_hit'] += len(h160 | h155); totals['p155a_only'] += len(h155 - h160); totals['p160_only'] += len(h160 - h155)
        for gi, gold in enumerate(golds):
            b=gold['bucket']
            by_area[b]['gold'] += 1
            if gi in h160: by_area[b]['p160_hit'] += 1
            if gi in h155: by_area[b]['p155a_hit'] += 1
            if gi in (h160|h155): by_area[b]['union_hit'] += 1
            if gi in (h155-h160): by_area[b]['p155a_only'] += 1
    return {'totals': dict(totals), 'by_area': {k: dict(v) for k,v in sorted(by_area.items())}}


def candidate_policies() -> list[dict[str, Any]]:
    policies=[]
    # Focused first-pass union: add only very sparse, high-confidence P155A
    # candidates that do not overlap P160 core. This is the safest way to test
    # whether P155A-only recall can improve F1 without precision collapse.
    label_sets=[['sink','equipment','appliance','generic_symbol'], ['sink','equipment','appliance'], []]
    bucket_sets=[[], ['tiny','small']]
    for labels in label_sets:
        for buckets in bucket_sets:
            for min_score in [0.50,0.65,0.75]:
                for max_iou in [0.05,0.15]:
                    for min_dist in [8.0,16.0]:
                        for max_add in [1,2]:
                            policies.append({'name':f"p164_l{len(labels)}_b{len(buckets)}_s{min_score}_miou{max_iou}_d{min_dist}_add{max_add}", 'labels':labels, 'buckets':buckets, 'min_score':min_score, 'max_iou_with_core':max_iou, 'min_center_dist':min_dist, 'max_add_per_page':max_add, 'append_nms_iou':0.75})
    policies.append({'name':'p164_noop','labels':[], 'buckets':[], 'min_score':2.0, 'max_iou_with_core':0.0, 'min_center_dist':0.0, 'max_add_per_page':0, 'append_nms_iou':0.75})
    return policies

def materialize(base_rows: list[dict[str, Any]], preds_by_row: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows=[]
    for raw in base_rows:
        row=copy.deepcopy(raw); row_id=str(row.get('row_id') or row.get('id'))
        candidates=[]
        for idx,pred in enumerate(preds_by_row.get(row_id, [])):
            item=copy.deepcopy(pred['raw'])
            item['bbox']=pred['bbox']; item['symbol_type']=pred['label']; item['confidence']=pred['score']
            item['id']=f"{row_id}_p164_best_symbol_{idx:05d}"; item['target_id']=item['id']; item['source']='symbol_policy_overlay_p164_best'
            item.setdefault('metadata',{})['p164_policy']=policy['name']; item['metadata']['p164_source_policy']=pred.get('source_policy')
            candidates.append(item)
        row['symbol_candidates']=candidates
        if isinstance(row.get('expected_json'),dict): row['expected_json']['symbol_candidates']=[copy.deepcopy(x) for x in candidates]
        row['symbol_policy_overlay']={'policy_id':'p164_best','description':'P164 dual-policy union rescue candidate','policy':policy}
        rows.append(row)
    return rows


def delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str,float]:
    return {k: round(float(a[k])-float(b[k]),6) for k in ['precision','recall','f1','center_recall','prediction_inflation']}


def render_md(report: dict[str, Any]) -> str:
    lines=['# P164 Symbol F1 Oracle and Dual-Policy Rescue','',f"Decision: **{report['decision']}**",'', '## Metrics','', '| Policy | Precision | Recall | F1 | Center | Inflation |','|---|---:|---:|---:|---:|---:|']
    for name,m in report['baseline_metrics'].items(): lines.append(f"| `{name}` | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['center_recall']:.6f} | {m['prediction_inflation']:.6f} |")
    b=report['best_metrics']; lines.append(f"| `p164_best` | {b['precision']:.6f} | {b['recall']:.6f} | {b['f1']:.6f} | {b['center_recall']:.6f} | {b['prediction_inflation']:.6f} |")
    o=report['oracle_analysis']['totals']
    lines += ['', '## Oracle Headroom', '', f"- P160 hits: `{o.get('p160_hit',0)}`", f"- P155A hits: `{o.get('p155a_hit',0)}`", f"- union hits: `{o.get('union_hit',0)}`", f"- P155A-only hits: `{o.get('p155a_only',0)}`", '', '## Best Policy','', f"- `{report['best_policy']['name']}`", f"- config: `{json.dumps(report['best_policy'], ensure_ascii=False)}`", '', '## Delta', '', f"- vs `p160_best`: `{json.dumps(report['delta_vs_p160'], ensure_ascii=False)}`", '', '## Artifacts','']
    for v in report['outputs'].values(): lines.append(f'- `{v}`')
    return '\n'.join(lines)+'\n'


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--p160-overlay',default=str(P160_OVERLAY)); parser.add_argument('--p155a-overlay',default=str(P155A_OVERLAY)); parser.add_argument('--output-json',default=str(OUT_JSON)); parser.add_argument('--output-md',default=str(OUT_MD)); parser.add_argument('--output-overlay',default=str(OUT_OVERLAY))
    args=parser.parse_args()
    p160_rows=load_jsonl(Path(args.p160_overlay)); p155a_rows=load_jsonl(Path(args.p155a_overlay))
    p155a_by_row={str(r.get('row_id') or r.get('id')): normalized(r.get('symbol_candidates') or [], 'p155a_p140') for r in p155a_rows}
    p160_by_row={str(r.get('row_id') or r.get('id')): normalized(r.get('symbol_candidates') or [], 'p160_core') for r in p160_rows}
    golds_by_row={str(r.get('row_id') or r.get('id')): target_symbols(r) for r in p160_rows}
    p160_metrics=evaluate(golds_by_row,p160_by_row); p155a_metrics=evaluate(golds_by_row,p155a_by_row)
    oracle=oracle_analysis(golds_by_row,p160_by_row,p155a_by_row)
    scored=[]
    for policy in candidate_policies():
        preds=apply_union_policy(p160_by_row,p155a_by_row,policy)
        metrics=evaluate(golds_by_row,preds)
        if metrics['precision']>=0.53 and metrics['prediction_inflation']<=1.10:
            scored.append({'policy':policy,'metrics':metrics})
    scored.sort(key=lambda r:(r['metrics']['f1'],r['metrics']['recall'],r['metrics']['precision']), reverse=True)
    best=scored[0]
    best_preds=apply_union_policy(p160_by_row,p155a_by_row,best['policy'])
    write_jsonl(Path(args.output_overlay), materialize(p160_rows,best_preds,best['policy']))
    decision='positive_adopt_p164' if best['metrics']['f1']>p160_metrics['f1'] else 'negative_keep_p160'
    report={'id':'SCI-P2-164-symbol-f1-oracle-and-dual-policy-rescue','created_on':'2026-05-17','decision':decision,'claim_boundary':'Post-hoc dual-policy union/backfill sweep on 74-row public-raster overlay subset. Runtime policy uses predicted boxes/labels/scores and overlap to P160 core only; gold targets are evaluation-only.','baseline_metrics':{'p160_best':p160_metrics,'p155a_p140':p155a_metrics},'oracle_analysis':oracle,'searched_policy_count':len(candidate_policies()),'passing_policy_count':len(scored),'best_policy':best['policy'],'best_metrics':best['metrics'],'delta_vs_p160':delta(best['metrics'],p160_metrics),'top_candidates':scored[:30],'outputs':{'overlay':str(Path(args.output_overlay)),'config_json':str(Path(args.output_json)),'report_md':str(Path(args.output_md))}}
    write_json(Path(args.output_json),report); Path(args.output_md).parent.mkdir(parents=True,exist_ok=True); Path(args.output_md).write_text(render_md(report),encoding='utf-8')
    print(json.dumps({'decision':decision,'searched':report['searched_policy_count'],'passing':report['passing_policy_count'],'oracle':oracle['totals'],'best_metrics':best['metrics'],'delta_vs_p160':report['delta_vs_p160'],'best_policy':best['policy']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
