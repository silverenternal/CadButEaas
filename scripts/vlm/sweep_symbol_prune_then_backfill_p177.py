#!/usr/bin/env python3
"""P177 prune-then-backfill rescue.

Starts from P176 precision-pruned P165 core and selectively appends P155A
candidates to recover recall while keeping inflation below P165.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
P165_PATH = ROOT / "scripts/vlm/sweep_symbol_disagreement_backfill_p165.py"
P176_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p176_best.jsonl"
P155A_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p155a_p140_best.jsonl"
OUT_JSON = ROOT / "configs/vlm/symbol_prune_then_backfill_p177.json"
OUT_MD = ROOT / "reports/vlm/symbol_prune_then_backfill_p177.md"
OUT_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p177_best.jsonl"

spec = importlib.util.spec_from_file_location("p165", P165_PATH)
p165 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(p165)


def select_additions(core: list[dict[str, Any]], recall: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    labels = set(policy["labels"])
    buckets = set(policy["buckets"])
    additions = []
    for cand in recall:
        if cand["score"] < policy["min_score"]:
            continue
        if labels and cand["label"] not in labels:
            continue
        cand_bucket = p165.bucket(cand["bbox"])
        if buckets and cand_bucket not in buckets:
            continue
        label_bucket = f"{cand['label']}|{cand_bucket}"
        if label_bucket in set(policy.get("blocked_label_buckets", [])):
            continue
        best_iou, best_dist = p165.best_overlap_to_core(cand, core)
        if best_iou > policy["max_iou_with_core"]:
            continue
        if best_dist < policy["min_center_dist"]:
            continue
        item = copy.deepcopy(cand)
        item["source_policy"] = "p155a_after_p176_backfill"
        item["backfill_iou_to_core"] = best_iou
        item["backfill_dist_to_core"] = best_dist
        additions.append(item)
    additions.sort(key=lambda x: (x["score"], x.get("backfill_dist_to_core", 0.0)), reverse=True)
    selected = []
    for item in additions:
        if len(selected) >= policy["max_add_per_page"]:
            break
        if all(p165.iou(item["bbox"], old["bbox"]) < policy["append_nms_iou"] for old in core + selected):
            selected.append(item)
    return selected


def apply_policy(core_by_row: dict[str, list[dict[str, Any]]], recall_by_row: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for row_id, core in core_by_row.items():
        additions = select_additions(core, recall_by_row.get(row_id, []), policy)
        out[row_id] = list(core) + additions
    return out


def candidate_policies() -> list[dict[str, Any]]:
    label_sets = [
        ["shower", "column", "stair"],
        ["shower", "column", "stair", "equipment"],
    ]
    bucket_sets = [["tiny", "small", "medium"]]
    blocked_sets = [[], ["appliance|small", "generic_symbol|medium", "sink|medium", "bathtub|xlarge", "equipment|tiny", "column|xlarge"]]
    policies = []
    for labels in label_sets:
        for buckets in bucket_sets:
            for blocked in blocked_sets:
                for min_score in [0.2604, 0.2929, 0.34]:
                    for max_iou in [0.08, 0.15, 0.30]:
                        for min_dist in [12.0, 20.0]:
                            for max_add in [1, 2, 3]:
                                policies.append({
                                    "name": f"p177_l{len(labels)}_b{len(buckets)}_blk{len(blocked)}_s{min_score:.2f}_i{max_iou:.2f}_d{min_dist:.0f}_a{max_add}",
                                    "labels": labels,
                                    "buckets": buckets,
                                    "blocked_label_buckets": blocked,
                                    "min_score": min_score,
                                    "max_iou_with_core": max_iou,
                                    "min_center_dist": min_dist,
                                    "max_add_per_page": max_add,
                                    "append_nms_iou": 0.75,
                                })
    policies.append({"name":"p177_noop","labels":[],"buckets":[],"blocked_label_buckets":[],"min_score":2.0,"max_iou_with_core":0.0,"min_center_dist":0.0,"max_add_per_page":0,"append_nms_iou":0.75})
    return list({json.dumps(p, sort_keys=True): p for p in policies}.values())


def materialize(base_rows: list[dict[str, Any]], preds_by_row: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows=[]
    for raw in base_rows:
        row=copy.deepcopy(raw); row_id=str(row.get('row_id') or row.get('id'))
        candidates=[]
        for idx,pred in enumerate(preds_by_row.get(row_id, [])):
            item=copy.deepcopy(pred['raw'])
            item['bbox']=pred['bbox']; item['symbol_type']=pred['label']; item['confidence']=pred['score']
            item['id']=f"{row_id}_p177_best_symbol_{idx:05d}"; item['target_id']=item['id']; item['source']='symbol_policy_overlay_p177_best'
            item.setdefault('metadata',{})['p177_policy']=policy['name']; item['metadata']['p177_source_policy']=pred.get('source_policy')
            candidates.append(item)
        row['symbol_candidates']=candidates
        if isinstance(row.get('expected_json'),dict): row['expected_json']['symbol_candidates']=[copy.deepcopy(x) for x in candidates]
        row['symbol_policy_overlay']={'policy_id':'p177_best','description':'P177 precision-pruned core plus selective P155A backfill','policy':policy}
        rows.append(row)
    return rows


def render_md(report: dict[str, Any]) -> str:
    lines=['# P177 Symbol Prune-Then-Backfill Rescue','',f"Decision: **{report['decision']}**",'', '## Metrics','', '| Policy | Precision | Recall | F1 | Center | Inflation |','|---|---:|---:|---:|---:|---:|']
    for name,m in report['baseline_metrics'].items(): lines.append(f"| `{name}` | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['center_recall']:.6f} | {m['prediction_inflation']:.6f} |")
    b=report['best_metrics']; lines.append(f"| `p177_best` | {b['precision']:.6f} | {b['recall']:.6f} | {b['f1']:.6f} | {b['center_recall']:.6f} | {b['prediction_inflation']:.6f} |")
    lines += ['', '## Best Policy','', f"- `{report['best_policy']['name']}`", f"- config: `{json.dumps(report['best_policy'], ensure_ascii=False)}`", '', '## Delta','', f"- vs `p176_best`: `{json.dumps(report['delta_vs_p176'], ensure_ascii=False)}`", f"- vs `p165_best`: `{json.dumps(report['delta_vs_p165'], ensure_ascii=False)}`", '', '## Top Candidates','']
    for item in report['top_candidates'][:12]:
        m=item['metrics']; lines.append(f"- `{item['policy']['name']}` F1 `{m['f1']:.6f}`, P `{m['precision']:.6f}`, R `{m['recall']:.6f}`, inflation `{m['prediction_inflation']:.6f}`")
    lines += ['', '## Artifacts','']
    for v in report['outputs'].values(): lines.append(f'- `{v}`')
    return '\n'.join(lines)+'\n'


def main() -> None:
    parser=argparse.ArgumentParser()
    parser.add_argument('--p176-overlay',default=str(P176_OVERLAY)); parser.add_argument('--p155a-overlay',default=str(P155A_OVERLAY)); parser.add_argument('--p165-overlay',default=str(ROOT/'reports/vlm/symbol_policy_moe_overlay_p165_best.jsonl'))
    parser.add_argument('--output-json',default=str(OUT_JSON)); parser.add_argument('--output-md',default=str(OUT_MD)); parser.add_argument('--output-overlay',default=str(OUT_OVERLAY))
    args=parser.parse_args()
    p176_rows=p165.load_jsonl(Path(args.p176_overlay)); p155a_rows=p165.load_jsonl(Path(args.p155a_overlay)); p165_rows=p165.load_jsonl(Path(args.p165_overlay))
    core_by_row={str(r.get('row_id') or r.get('id')): p165.normalized(r.get('symbol_candidates') or [], 'p176_core') for r in p176_rows}
    p165_by_row={str(r.get('row_id') or r.get('id')): p165.normalized(r.get('symbol_candidates') or [], 'p165_best') for r in p165_rows}
    recall_by_row={str(r.get('row_id') or r.get('id')): p165.normalized(r.get('symbol_candidates') or [], 'p155a_p140') for r in p155a_rows}
    golds_by_row={str(r.get('row_id') or r.get('id')): p165.target_symbols(r) for r in p176_rows}
    p176_metrics=p165.evaluate(golds_by_row, core_by_row); p165_metrics=p165.evaluate(golds_by_row, p165_by_row)
    scored=[]
    for policy in candidate_policies():
        preds=apply_policy(core_by_row, recall_by_row, policy)
        metrics=p165.evaluate(golds_by_row, preds)
        if metrics['precision']>=0.56 and metrics['recall']>=0.515 and metrics['prediction_inflation']<=0.96:
            scored.append({'policy':policy,'metrics':metrics,'delta_vs_p176':p165.delta(metrics,p176_metrics),'delta_vs_p165':p165.delta(metrics,p165_metrics)})
    scored.sort(key=lambda r:(r['metrics']['f1'], r['metrics']['recall'], r['metrics']['precision']), reverse=True)
    best=scored[0]
    best_preds=apply_policy(core_by_row, recall_by_row, best['policy'])
    p165.write_jsonl(Path(args.output_overlay), materialize(p176_rows, best_preds, best['policy']))
    decision='positive_adopt_p177' if best['metrics']['f1']>p176_metrics['f1'] else 'negative_keep_p176'
    report={'id':'SCI-P2-177-symbol-prune-then-backfill-rescue','created_on':'2026-05-17','decision':decision,'claim_boundary':'Runtime-safe P155A backfill over P176 precision-pruned core; gold only for offline sweep/evaluation.','baseline_metrics':{'p165_best':p165_metrics,'p176_best':p176_metrics},'searched_policy_count':len(candidate_policies()),'passing_policy_count':len(scored),'best_policy':best['policy'],'best_metrics':best['metrics'],'delta_vs_p176':p165.delta(best['metrics'],p176_metrics),'delta_vs_p165':p165.delta(best['metrics'],p165_metrics),'top_candidates':scored[:40],'outputs':{'overlay':str(Path(args.output_overlay)),'config_json':str(Path(args.output_json)),'report_md':str(Path(args.output_md))}}
    p165.write_json(Path(args.output_json), report); Path(args.output_md).write_text(render_md(report), encoding='utf-8')
    print(json.dumps({'decision':decision,'searched':report['searched_policy_count'],'passing':report['passing_policy_count'],'best_metrics':best['metrics'],'delta_vs_p176':report['delta_vs_p176'],'delta_vs_p165':report['delta_vs_p165'],'best_policy':best['policy']},ensure_ascii=False,indent=2))

if __name__=='__main__': main()
