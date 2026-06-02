#!/usr/bin/env python3
"""Probe runtime-safe sink tiny box calibration over frozen P217/P218 predictions."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from fuse_symbol_p206g_with_p211_p212 import area_bucket, bbox_iou, load_p206g

ROOT = Path(__file__).resolve().parents[2]
P217 = ROOT / "reports/vlm/symbol_p218_p217_frozen_overlay.jsonl"
OUT_JSON = ROOT / "reports/vlm/symbol_p221a_sink_tiny_box_calibration_probe.json"
OUT_MD = ROOT / "reports/vlm/symbol_p221a_sink_tiny_box_calibration_probe.md"


def pred_label(pred):
    return str(pred.get("label", pred.get("symbol_type", "unknown")))


def score(preds_by_row, golds_by_row):
    tp = fp = fn = 0
    tp_label = Counter(); fp_label = Counter(); fn_label = Counter()
    tp_bucket = Counter(); fp_bucket = Counter(); fn_bucket = Counter()
    for row_id, gold_map in golds_by_row.items():
        preds = preds_by_row.get(row_id, [])
        golds = list(gold_map.values())
        candidates = []
        for pi, pred in enumerate(preds):
            pbox = [float(v) for v in pred["bbox"]]
            plabel = pred_label(pred)
            for gi, gold in enumerate(golds):
                if plabel != str(gold["label"]):
                    continue
                iou = bbox_iou(pbox, [float(v) for v in gold["bbox"]])
                if iou >= 0.30:
                    candidates.append((iou, pi, gi))
        used_p, used_g = set(), set()
        for iou, pi, gi in sorted(candidates, reverse=True):
            if pi in used_p or gi in used_g:
                continue
            used_p.add(pi); used_g.add(gi)
            gold = golds[gi]
            label = str(gold["label"])
            bucket = area_bucket([float(v) for v in gold["bbox"]])
            tp += 1; tp_label[label] += 1; tp_bucket[bucket] += 1
        for pi, pred in enumerate(preds):
            if pi in used_p:
                continue
            label = pred_label(pred)
            bucket = area_bucket([float(v) for v in pred["bbox"]])
            fp += 1; fp_label[label] += 1; fp_bucket[bucket] += 1
        for gi, gold in enumerate(golds):
            if gi in used_g:
                continue
            label = str(gold["label"])
            bucket = area_bucket([float(v) for v in gold["bbox"]])
            fn += 1; fn_label[label] += 1; fn_bucket[bucket] += 1
    p = tp / max(tp + fp, 1)
    r = tp / max(tp + fn, 1)
    f1 = 2 * p * r / max(p + r, 1e-9)
    return {"tp": tp, "fp": fp, "fn": fn, "precision": p, "recall": r, "f1": f1, "tp_label": dict(tp_label), "fp_label": dict(fp_label), "fn_label": dict(fn_label), "fn_bucket": dict(fn_bucket)}


def expand_box(box, scale=None, min_size=None):
    x1, y1, x2, y2 = [float(v) for v in box]
    cx, cy = (x1+x2)/2, (y1+y2)/2
    w, h = max(1e-6, x2-x1), max(1e-6, y2-y1)
    if scale is not None:
        w *= scale; h *= scale
    if min_size is not None:
        w = max(w, min_size); h = max(h, min_size)
    return [cx-w/2, cy-h/2, cx+w/2, cy+h/2]


def transform(preds_by_row, mode):
    out = {}
    for row_id, preds in preds_by_row.items():
        new_preds = []
        for pred in preds:
            p = dict(pred)
            label = pred_label(pred)
            box = [float(v) for v in pred["bbox"]]
            bucket = area_bucket(box)
            if label == "sink" and bucket == "tiny_le_64":
                p["bbox"] = expand_box(box, scale=mode.get("scale"), min_size=mode.get("min_size"))
                meta = dict(p.get("metadata") or {})
                meta["p221a_probe"] = mode["name"]
                p["metadata"] = meta
            new_preds.append(p)
        out[row_id] = new_preds
    return out


def main():
    _rows, preds_by_row, golds_by_row = load_p206g(P217)
    base = score(preds_by_row, golds_by_row)
    modes = [{"name": "baseline"},]
    for scale in [1.25, 1.5, 1.75, 2.0, 2.5, 3.0, 4.0]:
        modes.append({"name": f"sink_tiny_scale_{scale}", "scale": scale})
    for min_size in [8, 10, 12, 14, 16, 20, 24]:
        modes.append({"name": f"sink_tiny_min_{min_size}", "min_size": min_size})
    for scale in [1.5, 2.0, 2.5, 3.0]:
        for min_size in [10, 12, 16, 20]:
            modes.append({"name": f"sink_tiny_scale_{scale}_min_{min_size}", "scale": scale, "min_size": min_size})
    results = []
    for mode in modes:
        if mode["name"] == "baseline":
            metrics = base
        else:
            metrics = score(transform(preds_by_row, mode), golds_by_row)
        results.append({"mode": mode, "metrics": metrics, "delta_f1": metrics["f1"]-base["f1"], "delta_precision": metrics["precision"]-base["precision"], "delta_recall": metrics["recall"]-base["recall"], "sink_fn": metrics["fn_label"].get("sink", 0), "tiny_fn": metrics["fn_bucket"].get("tiny_le_64", 0)})
    results.sort(key=lambda x: (x["metrics"]["f1"], x["metrics"]["precision"]), reverse=True)
    payload = {"id":"P221a_sink_tiny_box_calibration_probe", "source_overlay": str(P217.relative_to(ROOT)), "baseline": base, "top_results": results[:30], "claim_boundary":"Probe only; any rule must be frozen and bootstrapped before promotion."}
    OUT_JSON.write_text(json.dumps(payload, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = ["# P221a Sink-Tiny Box Calibration Probe", "", "## Baseline Diagnostic", f"- F1/P/R: {base['f1']:.6f}/{base['precision']:.6f}/{base['recall']:.6f}", f"- TP/FP/FN: {base['tp']}/{base['fp']}/{base['fn']}", f"- Sink FN: {base['fn_label'].get('sink',0)}", f"- Tiny FN: {base['fn_bucket'].get('tiny_le_64',0)}", "", "## Top Runtime-Safe Box Calibration Probes", "| Mode | F1 | P | R | ΔF1 | ΔP | ΔR | Sink FN | Tiny FN |", "|---|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for r in results[:15]:
        m = r["metrics"]
        lines.append(f"| {r['mode']['name']} | {m['f1']:.6f} | {m['precision']:.6f} | {m['recall']:.6f} | {r['delta_f1']:.6f} | {r['delta_precision']:.6f} | {r['delta_recall']:.6f} | {r['sink_fn']} | {r['tiny_fn']} |")
    lines += ["", "## Interpretation", "- This only changes predicted sink tiny boxes using runtime-visible geometry; no gold is used at runtime.", "- If a simple calibration improves F1 without hurting precision, P221a can start with a frozen deterministic rule before training new weights.", "- Final promotion still requires paired bootstrap against official frozen P217/P218 scorer."]
    OUT_MD.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_MD), "best": results[0]}, ensure_ascii=False, indent=2)[:3000])

if __name__ == "__main__":
    main()
