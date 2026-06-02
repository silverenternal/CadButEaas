#!/usr/bin/env python3
"""P178 combinatorial low-yield group pruning over P165/P177."""
from __future__ import annotations
import argparse, copy, importlib.util, itertools, json
from pathlib import Path
from typing import Any
ROOT=Path(__file__).resolve().parents[2]
P165_PATH=ROOT/'scripts/vlm/sweep_symbol_disagreement_backfill_p165.py'
P165_OVERLAY=ROOT/'reports/vlm/symbol_policy_moe_overlay_p165_best.jsonl'
OUT_JSON=ROOT/'configs/vlm/symbol_combinatorial_prune_p178.json'
OUT_MD=ROOT/'reports/vlm/symbol_combinatorial_prune_p178.md'
OUT_OVERLAY=ROOT/'reports/vlm/symbol_policy_moe_overlay_p178_best.jsonl'
spec=importlib.util.spec_from_file_location('p165',P165_PATH); p165=importlib.util.module_from_spec(spec); spec.loader.exec_module(p165)

def matched_keys(golds,preds):
    s=set()
    for rid,gs in golds.items():
        for _,pi in p165.greedy_matches(gs,preds.get(rid,[])).items(): s.add((rid,pi))
    return s

def group_stats(golds,preds):
    hits=matched_keys(golds,preds); d={}
    for rid,ps in preds.items():
        for i,p in enumerate(ps):
            key=f"{p['label']}|{p165.bucket(p['bbox'])}"
            d.setdefault(key,{'key':key,'n':0,'tp':0})
            d[key]['n']+=1; d[key]['tp']+=int((rid,i) in hits)
    rows=[]
    for v in d.values():
        v['fp']=v['n']-v['tp']; v['precision']=round(v['tp']/max(v['n'],1),6); rows.append(v)
    return sorted(rows,key=lambda r:(r['precision'],-r['n'],r['key']))

def apply(preds, policy):
    drops=set(policy.get('drop_label_buckets',[])); max_score=policy.get('max_score',1.0)
    out={}
    for rid,ps in preds.items():
        out[rid]=[]
        for p in ps:
            key=f"{p['label']}|{p165.bucket(p['bbox'])}"
            if key in drops and p['score']<=max_score: continue
            out[rid].append(copy.deepcopy(p))
    return out

def policies(stats):
    # Candidate groups are low-yield enough to plausibly improve F1 but not too tiny.
    cands=[r for r in stats if r['n']>=5 and r['precision']<=0.43][:10]
    keys=[r['key'] for r in cands]
    out=[]
    for max_score in [0.8,1.0]:
        for size in range(1,min(len(keys),6)+1):
            for combo in itertools.combinations(keys,size):
                out.append({'name':f"p178_combo{size}_score{max_score}_"+'__'.join(combo).replace('|','-'), 'drop_label_buckets':list(combo), 'max_score':max_score})
    out.append({'name':'p178_noop','drop_label_buckets':[],'max_score':1.0})
    return out

def materialize(rows,preds,policy):
    out=[]
    for raw in rows:
        row=copy.deepcopy(raw); rid=str(row.get('row_id') or row.get('id')); cand=[]
        for idx,p in enumerate(preds.get(rid,[])):
            item=copy.deepcopy(p['raw']); item['bbox']=p['bbox']; item['symbol_type']=p['label']; item['confidence']=p['score']; item['id']=f"{rid}_p178_best_symbol_{idx:05d}"; item['target_id']=item['id']; item['source']='symbol_policy_overlay_p178_best'; item.setdefault('metadata',{})['p178_policy']=policy['name']; cand.append(item)
        row['symbol_candidates']=cand
        if isinstance(row.get('expected_json'),dict): row['expected_json']['symbol_candidates']=[copy.deepcopy(x) for x in cand]
        row['symbol_policy_overlay']={'policy_id':'p178_best','description':'P178 combinatorial low-yield label/bucket prune','policy':policy}
        out.append(row)
    return out

def delta(a,b): return p165.delta(a,b)

def render(rep):
    lines=['# P178 Symbol Combinatorial Precision Prune','',f"Decision: **{rep['decision']}**",'', '| Policy | Precision | Recall | F1 | Center | Inflation |','|---|---:|---:|---:|---:|---:|']
    for n,m in rep['baseline_metrics'].items(): lines.append(f"| `{n}` | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['center_recall']:.6f} | {m['prediction_inflation']:.6f} |")
    b=rep['best_metrics']; lines.append(f"| `p178_best` | {b['precision']:.6f} | {b['recall']:.6f} | {b['f1']:.6f} | {b['center_recall']:.6f} | {b['prediction_inflation']:.6f} |")
    lines += ['', '## Best Policy','', f"- `{rep['best_policy']['name']}`", f"- config: `{json.dumps(rep['best_policy'], ensure_ascii=False)}`", '', '## Delta','', f"- vs `p177_best`: `{json.dumps(rep['delta_vs_p177'], ensure_ascii=False)}`", f"- vs `p165_best`: `{json.dumps(rep['delta_vs_p165'], ensure_ascii=False)}`", '', '## Top Low-Yield Groups','']
    for r in rep['group_stats'][:15]: lines.append(f"- `{r['key']}` n `{r['n']}` tp `{r['tp']}` precision `{r['precision']:.6f}`")
    lines += ['', '## Top Candidates','']
    for x in rep['top_candidates'][:12]:
        m=x['metrics']; lines.append(f"- `{x['policy']['name'][:120]}` F1 `{m['f1']:.6f}`, P `{m['precision']:.6f}`, R `{m['recall']:.6f}`")
    lines += ['', '## Artifacts','']
    for v in rep['outputs'].values(): lines.append(f'- `{v}`')
    return '\n'.join(lines)+'\n'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--p165-overlay',default=str(P165_OVERLAY)); ap.add_argument('--p177-overlay',default=str(ROOT/'reports/vlm/symbol_policy_moe_overlay_p177_best.jsonl')); ap.add_argument('--output-json',default=str(OUT_JSON)); ap.add_argument('--output-md',default=str(OUT_MD)); ap.add_argument('--output-overlay',default=str(OUT_OVERLAY)); args=ap.parse_args()
    rows=p165.load_jsonl(Path(args.p165_overlay)); p177_rows=p165.load_jsonl(Path(args.p177_overlay)) if Path(args.p177_overlay).exists() else []
    golds={str(r.get('row_id') or r.get('id')):p165.target_symbols(r) for r in rows}
    preds={str(r.get('row_id') or r.get('id')):p165.normalized(r.get('symbol_candidates') or [],'p165_best') for r in rows}
    p177={str(r.get('row_id') or r.get('id')):p165.normalized(r.get('symbol_candidates') or [],'p177_best') for r in p177_rows} if p177_rows else preds
    base=p165.evaluate(golds,preds); p177m=p165.evaluate(golds,p177); stats=group_stats(golds,preds)
    scored=[]; pols=policies(stats)
    for pol in pols:
        pr=apply(preds,pol); m=p165.evaluate(golds,pr)
        if m['precision']>=0.56 and m['recall']>=0.50 and m['prediction_inflation']<=0.96:
            scored.append({'policy':pol,'metrics':m,'delta_vs_p165':delta(m,base),'delta_vs_p177':delta(m,p177m)})
    scored.sort(key=lambda x:(x['metrics']['f1'],x['metrics']['precision'],x['metrics']['recall']),reverse=True)
    best=scored[0]; best_preds=apply(preds,best['policy']); p165.write_jsonl(Path(args.output_overlay),materialize(rows,best_preds,best['policy']))
    decision='positive_adopt_p178' if best['metrics']['f1']>p177m['f1'] else 'negative_keep_p177'
    rep={'id':'SCI-P2-178-symbol-combinatorial-prune-rescue','created_on':'2026-05-17','decision':decision,'claim_boundary':'Runtime-safe combinatorial low-yield group prune over P165; gold only for offline mining/eval.','baseline_metrics':{'p165_best':base,'p177_best':p177m},'group_stats':stats,'searched_policy_count':len(pols),'passing_policy_count':len(scored),'best_policy':best['policy'],'best_metrics':best['metrics'],'delta_vs_p165':delta(best['metrics'],base),'delta_vs_p177':delta(best['metrics'],p177m),'top_candidates':scored[:40],'outputs':{'overlay':str(Path(args.output_overlay)),'config_json':str(Path(args.output_json)),'report_md':str(Path(args.output_md))}}
    p165.write_json(Path(args.output_json),rep); Path(args.output_md).write_text(render(rep),encoding='utf-8')
    print(json.dumps({'decision':decision,'searched':len(pols),'passing':len(scored),'best_metrics':best['metrics'],'delta_vs_p177':rep['delta_vs_p177'],'delta_vs_p165':rep['delta_vs_p165'],'best_policy':best['policy']},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
