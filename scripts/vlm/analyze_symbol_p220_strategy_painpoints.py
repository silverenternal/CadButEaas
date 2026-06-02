#!/usr/bin/env python3
"""Analyze current symbol strategy and pain points after P217/P219."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from fuse_symbol_p206g_with_p211_p212 import area_bucket, bbox_iou, load_p206g

ROOT = Path(__file__).resolve().parents[2]
P217 = ROOT / "reports/vlm/symbol_p218_p217_frozen_overlay.jsonl"
OUT_MD = ROOT / "reports/vlm/symbol_p220_strategy_painpoints.md"
OUT_JSON = ROOT / "reports/vlm/symbol_p220_strategy_painpoints.json"


def greedy_match(preds, golds):
    pairs = []
    used_preds = set()
    used_golds = set()
    candidates = []
    for pred_index, pred in enumerate(preds):
        pred_box = [float(v) for v in pred["bbox"]]
        for gold_index, gold in enumerate(golds):
            if str(pred.get("label", "unknown")) != str(gold.get("label", "unknown")):
                continue
            iou = bbox_iou(pred_box, [float(v) for v in gold["bbox"]])
            if iou >= 0.30:
                candidates.append((iou, pred_index, gold_index))
    for iou, pred_index, gold_index in sorted(candidates, reverse=True):
        if pred_index in used_preds or gold_index in used_golds:
            continue
        used_preds.add(pred_index)
        used_golds.add(gold_index)
        pairs.append((pred_index, gold_index, iou))
    return pairs, used_preds, used_golds


def pct(n, d):
    return 0.0 if d == 0 else 100.0 * n / d


def main() -> None:
    rows, preds_by_row, golds_by_row = load_p206g(P217)
    tp_label = Counter()
    fp_label = Counter()
    fn_label = Counter()
    gold_label = Counter()
    pred_label = Counter()
    tp_bucket = Counter()
    fp_bucket = Counter()
    fn_bucket = Counter()
    fn_cross = Counter()
    fp_cross = Counter()
    row_counts = []

    for row in rows:
        row_id = str(row.get("id") or row.get("row_id"))
        preds = preds_by_row[row_id]
        golds = list(golds_by_row[row_id].values())
        pairs, used_preds, used_golds = greedy_match(preds, golds)
        for pred in preds:
            pred_label[str(pred.get("label", "unknown"))] += 1
        for gold in golds:
            gold_label[str(gold.get("label", "unknown"))] += 1
        for pred_index, gold_index, _iou in pairs:
            gold = golds[gold_index]
            label = str(gold.get("label", "unknown"))
            bucket = area_bucket([float(v) for v in gold["bbox"]])
            tp_label[label] += 1
            tp_bucket[bucket] += 1
        for pred_index, pred in enumerate(preds):
            if pred_index in used_preds:
                continue
            label = str(pred.get("label", "unknown"))
            bucket = area_bucket([float(v) for v in pred["bbox"]])
            fp_label[label] += 1
            fp_bucket[bucket] += 1
            fp_cross[(label, bucket)] += 1
        for gold_index, gold in enumerate(golds):
            if gold_index in used_golds:
                continue
            label = str(gold.get("label", "unknown"))
            bucket = area_bucket([float(v) for v in gold["bbox"]])
            fn_label[label] += 1
            fn_bucket[bucket] += 1
            fn_cross[(label, bucket)] += 1
        row_counts.append((row_id, len(golds), len(preds), len(pairs), len(golds) - len(used_golds), len(preds) - len(used_preds)))

    total_tp = sum(tp_label.values())
    total_fp = sum(fp_label.values())
    total_fn = sum(fn_label.values())
    precision = total_tp / (total_tp + total_fp)
    recall = total_tp / (total_tp + total_fn)
    f1 = 2 * precision * recall / (precision + recall)

    label_rows = []
    for label in sorted(set(gold_label) | set(pred_label) | set(tp_label) | set(fp_label) | set(fn_label)):
        support = gold_label[label]
        label_rows.append({
            "label": label,
            "gold": support,
            "pred": pred_label[label],
            "tp": tp_label[label],
            "fp": fp_label[label],
            "fn": fn_label[label],
            "recall": 0.0 if support == 0 else tp_label[label] / support,
            "precision": 0.0 if pred_label[label] == 0 else tp_label[label] / pred_label[label],
            "fn_share": pct(fn_label[label], total_fn),
        })
    label_rows.sort(key=lambda r: (-r["fn"], r["label"]))

    bucket_rows = []
    for bucket in sorted(set(tp_bucket) | set(fp_bucket) | set(fn_bucket)):
        gold_count = tp_bucket[bucket] + fn_bucket[bucket]
        pred_count = tp_bucket[bucket] + fp_bucket[bucket]
        bucket_rows.append({
            "bucket": bucket,
            "gold": gold_count,
            "pred": pred_count,
            "tp": tp_bucket[bucket],
            "fp": fp_bucket[bucket],
            "fn": fn_bucket[bucket],
            "recall": 0.0 if gold_count == 0 else tp_bucket[bucket] / gold_count,
            "precision": 0.0 if pred_count == 0 else tp_bucket[bucket] / pred_count,
            "fn_share": pct(fn_bucket[bucket], total_fn),
        })
    bucket_rows.sort(key=lambda r: (-r["fn"], r["bucket"]))

    top_fn_cross = [
        {"label": label, "bucket": bucket, "fn": count}
        for (label, bucket), count in fn_cross.most_common(15)
    ]
    top_fp_cross = [
        {"label": label, "bucket": bucket, "fp": count}
        for (label, bucket), count in fp_cross.most_common(15)
    ]
    worst_rows = [
        {"row_id": row_id, "gold": gold, "pred": pred, "tp": tp, "fn": fn, "fp": fp}
        for row_id, gold, pred, tp, fn, fp in sorted(row_counts, key=lambda x: (-x[4], -x[5], x[0]))[:20]
    ]

    result = {
        "id": "P220_strategy_painpoints_after_P217_P219",
        "source_overlay": str(P217.relative_to(ROOT)),
        "metrics": {"tp": total_tp, "fp": total_fp, "fn": total_fn, "precision": precision, "recall": recall, "f1": f1},
        "by_label": label_rows,
        "by_bucket": bucket_rows,
        "top_fn_label_bucket": top_fn_cross,
        "top_fp_label_bucket": top_fp_cross,
        "worst_rows": worst_rows,
        "diagnosis": [
            "The pipeline has moved from weak broad symbol recognition to a proposal-plus-verifier rescue architecture.",
            "Current bottleneck is recall on tiny/small raster symbols, especially sink/stair/equipment, not generic packaging.",
            "Precision is still only moderate, so adding a high-recall branch without a strong verifier can erase F1 gains.",
            "Evidence is still P101/bootstrap-bounded; independent validation is a separate paper-readiness bottleneck.",
        ],
        "recommended_mainline": [
            "Keep P217/P218 frozen as the current safe baseline.",
            "Run P220 split inventory to determine whether independent validation exists.",
            "In parallel, start P221 residual rescue focused on tiny sink/stair/equipment with strict precision gate.",
            "Do not chase >0.90 by threshold loosening; train/gate a targeted residual specialist and bootstrap against P217.",
        ],
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# P220 Strategy and Pain-Point Diagnosis",
        "",
        "## Current Mainline",
        "- Freeze P217/P218 as the runtime-safe baseline: F1={f1:.6f}, P={p:.6f}, R={r:.6f}.",
        "- Treat P219 as bounded paper packaging, not metric rescue.",
        "- Next useful work is split validation plus targeted residual rescue, not broad threshold search.",
        "",
        "## Metric Decomposition",
        f"- TP/FP/FN: {total_tp}/{total_fp}/{total_fn}",
        f"- Precision/Recall/F1: {precision:.6f}/{recall:.6f}/{f1:.6f}",
        "",
        "## Pain Points by Label",
        "| Label | Gold | Pred | TP | FP | FN | P | R | FN share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    lines[3] = lines[3].format(f1=f1, p=precision, r=recall)
    for row in label_rows:
        lines.append(f"| {row['label']} | {row['gold']} | {row['pred']} | {row['tp']} | {row['fp']} | {row['fn']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['fn_share']:.1f}% |")
    lines += [
        "",
        "## Pain Points by Size Bucket",
        "| Bucket | Gold | Pred | TP | FP | FN | P | R | FN share |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in bucket_rows:
        lines.append(f"| {row['bucket']} | {row['gold']} | {row['pred']} | {row['tp']} | {row['fp']} | {row['fn']} | {row['precision']:.3f} | {row['recall']:.3f} | {row['fn_share']:.1f}% |")
    lines += [
        "",
        "## Top FN Label-Bucket Intersections",
        "| Label | Bucket | FN |",
        "|---|---|---:|",
    ]
    for row in top_fn_cross:
        lines.append(f"| {row['label']} | {row['bucket']} | {row['fn']} |")
    lines += [
        "",
        "## Interpretation",
        "- The main architecture should remain proposal generation plus runtime-safe verification; the verifier is what made P217 deployable over the P216 oracle proof.",
        "- The largest metric ceiling is missed tiny/small sink/stair/equipment, so a residual branch should be class/size-aware instead of generic high-recall detection.",
        "- Precision is fragile: earlier P213b showed recall additions can improve recall but hurt precision CI, so every rescue branch must be verified against P217 with paired bootstrap.",
        "- Paper readiness has two independent bottlenecks: metric quality and evidence independence. P219 helps wording, but P220 validation is still needed for broad claims.",
        "",
        "## Recommended Next Step",
        "- P220: inventory available non-selected rows/splits and run frozen P217 unchanged if labels exist.",
        "- P221: start a tiny sink/stair/equipment residual specialist only after defining a precision gate and no-leakage runtime feature contract.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_MD), "json": str(OUT_JSON), "metrics": result["metrics"], "top_fn": top_fn_cross[:5]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
