#!/usr/bin/env python3
from __future__ import annotations
import argparse, copy, importlib.util, json
from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]
P165_PATH=ROOT/'scripts/vlm/sweep_symbol_disagreement_backfill_p165.py'
spec=importlib.util.spec_from_file_location('p165',P165_PATH); p165=importlib.util.module_from_spec(spec); spec.loader.exec_module(p165)
P179=ROOT/'reports/vlm/symbol_policy_moe_overlay_p179_best.jsonl'; P155=ROOT/'reports/vlm/symbol_policy_moe_overlay_p155a_p140_best.jsonl'
OUTJ=ROOT/'configs/vlm/symbol_p179_targeted_backfill_p180.json'; OUTM=ROOT/'reports/vlm/symbol_p179_targeted_backfill_p180.md'; OUTO=ROOT/'reports/vlm/symbol_policy_moe_overlay_p180_best.jsonl'

def select(core, recall, pol):
    labels=set(pol['labels']); buckets=set(pol['buckets']); blocked=set(pol.get('blocked_label_buckets',[])); adds=[]
    for c in recall:
        b=p165.bucket(c['bbox']); lb=f"{c['label']}|{b}"
        if c['score']<pol['min_score'] or c['score']>pol.get('max_score',1.0): continue
        if labels and c['label'] not in labels: continue
        if buckets and b not in buckets: continue
        if lb in blocked: continue
        bi,bd=p165.best_overlap_to_core(c,core)
        if bi>pol['max_iou']: continue
        if bd<pol['min_dist']: continue
        item=copy.deepcopy(c); item['source_policy']='p155a_p180_backfill'; item['backfill_iou_to_core']=bi; item['backfill_dist_to_core']=bd; adds.append(item)
    mode=pol['sort']
    if mode=='score': adds.sort(key=lambda x:x['score'], reverse=True)
    elif mode=='far_score': adds.sort(key=lambda x:(x['backfill_dist_to_core'],x['score']), reverse=True)
    elif mode=='low_iou_score': adds.sort(key=lambda x:(-x['backfill_iou_to_core'],x['score']), reverse=True)
    selected=[]
    for a in adds:
        if len(selected)>=pol['max_add']: break
        if all(p165.iou(a['bbox'],x['bbox'])<pol['nms'] for x in core+selected): selected.append(a)
    return selected

def drop_core(core, additions, pol):
    if pol['mode']=='append': return list(core)
    keep=list(core)
    for add in additions:
        elig=list(range(len(keep)))
        if pol['mode']=='replace_low_score_under': elig=[i for i,p in enumerate(keep) if p['score']<=pol['drop_score_max']]
        elif pol['mode']=='replace_same_label': elig=[i for i,p in enumerate(keep) if p['label']==add['label'] and p['score']<=pol['drop_score_max']]
        if not elig: continue
        di=min(elig, key=lambda i: keep[i]['score']); keep.pop(di)
    return keep

def apply(core_by, rec_by, pol):
    out={}
    for rid,core in core_by.items():
        adds=select(core, rec_by.get(rid,[]), pol); keep=drop_core(core,adds,pol); out[rid]=keep+adds
    return out

def policies():
    label_sets=[['shower','stair','sink'],['shower','stair','sink','equipment']]
    bucket_sets=[['tiny','small','medium'],['tiny','small','medium','large']]
    blocked=[[],['sink|medium','generic_symbol|medium','appliance|small','sink|large','stair|xlarge','appliance|large','column|xlarge','column|large','equipment|tiny','bathtub|xlarge']]
    out=[]
    for labels in label_sets:
      for buckets in bucket_sets:
       for bl in blocked:
        for mn in [0.26,0.2929,0.34,0.45]:
         for mx_iou in [0.08,0.15,0.3]:
          for md in [12,20]:
           for max_add in [1,2,3]:
            for mode in ['append']:
             for sort in ['score','low_iou_score']:
              out.append({'name':f"p180_l{len(labels)}_b{len(buckets)}_blk{len(bl)}_s{mn}_i{mx_iou}_d{md}_a{max_add}_{mode}_{sort}", 'labels':labels,'buckets':buckets,'blocked_label_buckets':bl,'min_score':mn,'max_score':1.0,'max_iou':mx_iou,'min_dist':md,'max_add':max_add,'mode':mode,'drop_score_max':0.65,'sort':sort,'nms':0.75})
    out.append({'name':'p180_noop','labels':[],'buckets':[],'blocked_label_buckets':[],'min_score':2,'max_score':1,'max_iou':0,'min_dist':0,'max_add':0,'mode':'append','drop_score_max':0.65,'sort':'score','nms':0.75})
    return out

def materialize(rows,preds,pol):
    out=[]
    for raw in rows:
        row=copy.deepcopy(raw); rid=str(row.get('row_id') or row.get('id')); cand=[]
        for idx,p in enumerate(preds.get(rid,[])):
            item=copy.deepcopy(p['raw']); item['bbox']=p['bbox']; item['symbol_type']=p['label']; item['confidence']=p['score']; item['id']=f"{rid}_p180_best_symbol_{idx:05d}"; item['target_id']=item['id']; item['source']='symbol_policy_overlay_p180_best'; item.setdefault('metadata',{})['p180_policy']=pol['name']; cand.append(item)
        row['symbol_candidates']=cand
        if isinstance(row.get('expected_json'),dict): row['expected_json']['symbol_candidates']=[copy.deepcopy(x) for x in cand]
        row['symbol_policy_overlay']={'policy_id':'p180_best','description':'P180 targeted P155A backfill over P179 core','policy':pol}; out.append(row)
    return out

def render(rep):
    lines=['# P180 P179 Targeted Backfill','',f"Decision: **{rep['decision']}**",'', '| Policy | Precision | Recall | F1 | Center | Inflation |','|---|---:|---:|---:|---:|---:|']
    for n,m in rep['baseline_metrics'].items(): lines.append(f"| `{n}` | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['center_recall']:.6f} | {m['prediction_inflation']:.6f} |")
    b=rep['best_metrics']; lines.append(f"| `p180_best` | {b['precision']:.6f} | {b['recall']:.6f} | {b['f1']:.6f} | {b['center_recall']:.6f} | {b['prediction_inflation']:.6f} |")
    lines += ['', '## Best Policy','', f"- `{rep['best_policy']['name']}`", f"- config: `{json.dumps(rep['best_policy'], ensure_ascii=False)}`", '', '## Delta','', f"- vs `p179_best`: `{json.dumps(rep['delta_vs_p179'], ensure_ascii=False)}`", '', '## Top Candidates','']
    for x in rep['top_candidates'][:15]:
        m=x['metrics']; lines.append(f"- `{x['policy']['name']}` F1 `{m['f1']:.6f}`, P `{m['precision']:.6f}`, R `{m['recall']:.6f}`, infl `{m['prediction_inflation']:.6f}`")
    return '\n'.join(lines)+'\n'

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--p179',default=str(P179)); ap.add_argument('--p155a',default=str(P155)); ap.add_argument('--out-json',default=str(OUTJ)); ap.add_argument('--out-md',default=str(OUTM)); ap.add_argument('--out-overlay',default=str(OUTO)); args=ap.parse_args()
    rows=p165.load_jsonl(Path(args.p179)); rec_rows=p165.load_jsonl(Path(args.p155a))
    core={str(r.get('row_id') or r.get('id')):p165.normalized(r.get('symbol_candidates') or [],'p179') for r in rows}
    rec={str(r.get('row_id') or r.get('id')):p165.normalized(r.get('symbol_candidates') or [],'p155a') for r in rec_rows}
    golds={str(r.get('row_id') or r.get('id')):p165.target_symbols(r) for r in rows}
    base=p165.evaluate(golds,core); scored=[]
    for pol in policies():
        preds=apply(core,rec,pol); m=p165.evaluate(golds,preds)
        if m['precision']>=0.58 and m['recall']>=0.51 and m['prediction_inflation']<=0.93: scored.append({'policy':pol,'metrics':m,'delta_vs_p179':p165.delta(m,base)})
    scored.sort(key=lambda x:(x['metrics']['f1'],x['metrics']['recall'],x['metrics']['precision']), reverse=True)
    best=scored[0]; best_preds=apply(core,rec,best['policy']); p165.write_jsonl(Path(args.out_overlay), materialize(rows,best_preds,best['policy']))
    decision='positive_adopt_p180' if best['metrics']['f1']>base['f1'] else 'negative_keep_p179'
    rep={'id':'SCI-P2-180-symbol-p179-targeted-backfill','created_on':'2026-05-17','decision':decision,'baseline_metrics':{'p179_best':base},'searched_policy_count':len(policies()),'passing_policy_count':len(scored),'best_policy':best['policy'],'best_metrics':best['metrics'],'delta_vs_p179':p165.delta(best['metrics'],base),'top_candidates':scored[:50],'outputs':{'overlay':str(Path(args.out_overlay)),'config_json':str(Path(args.out_json)),'report_md':str(Path(args.out_md))}}
    p165.write_json(Path(args.out_json),rep); Path(args.out_md).write_text(render(rep),encoding='utf-8')
    print(json.dumps({'decision':decision,'searched':len(policies()),'passing':len(scored),'best_metrics':best['metrics'],'delta_vs_p179':rep['delta_vs_p179'],'best_policy':best['policy']},ensure_ascii=False,indent=2))
if __name__=='__main__': main()
