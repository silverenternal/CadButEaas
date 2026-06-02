#!/usr/bin/env python3
"""Paired bootstrap and delta audit for P212 fusion against P206g."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fuse_symbol_p206g_with_p211_p212 import area_bucket, bbox_iou, load_p206g, score_predictions

ROOT = Path(__file__).resolve().parents[2]
P206G = ROOT / "reports/vlm/symbol_p206f_precision_repair_p206g_overlay.jsonl"
P212_OVERLAY = ROOT / "reports/vlm/symbol_p206g_p212_specialist_fusion_fine_overlay.jsonl"
REPORT = ROOT / "reports/vlm/symbol_p212_vs_p206g_bootstrap_validation.md"
JSON_REPORT = ROOT / "reports/vlm/symbol_p212_vs_p206g_bootstrap_validation.json"


def load_overlay_preds(path: Path) -> tuple[list[dict[str, Any]], dict[str, list[dict[str, Any]]], dict[str, dict[str, dict[str, Any]]]]:
    rows, _core, golds = load_p206g(path)
    preds: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        row_id = str(row.get("id") or row.get("row_id"))
        row_preds = []
        for cand in row.get("symbol_candidates") or []:
            label = str(cand.get("symbol_type") or cand.get("label") or "generic_symbol")
            row_preds.append({
                "bbox": [float(v) for v in cand.get("bbox")],
                "label": label,
                "label_id": 0,
                "score": float(cand.get("confidence") or cand.get("score") or 1.0),
                "source": cand.get("source"),
            })
        preds[row_id] = row_preds
    return rows, preds, golds


def subset(mapping: dict[str, Any], row_ids: list[str]) -> dict[str, Any]:
    return {row_id: mapping[row_id] for row_id in row_ids if row_id in mapping}


def f1(metrics: dict[str, Any]) -> float:
    return float(metrics["symbol_bbox_iou_0_30"]["f1"])


def precision(metrics: dict[str, Any]) -> float:
    return float(metrics["symbol_bbox_iou_0_30"]["precision"])


def recall(metrics: dict[str, Any]) -> float:
    return float(metrics["symbol_bbox_iou_0_30"]["recall"])


def quantile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = min(len(ordered) - 1, max(0, int(round(q * (len(ordered) - 1)))))
    return ordered[idx]


def matched_gold_indices(preds: list[dict[str, Any]], golds: list[dict[str, Any]]) -> set[int]:
    matched: set[int] = set()
    used_preds: set[int] = set()
    for gold_index, gold in enumerate(golds):
        gold_box = [float(v) for v in gold["bbox"]]
        best_iou = 0.0
        best_pred = None
        for pred_index, pred in enumerate(preds):
            if pred_index in used_preds:
                continue
            iou = bbox_iou([float(v) for v in pred["bbox"]], gold_box)
            if iou > best_iou:
                best_iou = iou
                best_pred = pred_index
        if best_pred is not None and best_iou >= 0.30:
            used_preds.add(best_pred)
            matched.add(gold_index)
    return matched


def audit_added_coverage(core: dict[str, list[dict[str, Any]]], fused: dict[str, list[dict[str, Any]]], golds: dict[str, dict[str, dict[str, Any]]]) -> dict[str, Any]:
    by_label = Counter(); gained_by_label = Counter(); lost_by_label = Counter()
    by_bucket = Counter(); gained_by_bucket = Counter(); lost_by_bucket = Counter()
    added_count = 0
    added_by_label = Counter()
    for row_id, gold_map in golds.items():
        gold_list = list(gold_map.values())
        core_match = matched_gold_indices(core.get(row_id, []), gold_list)
        fused_match = matched_gold_indices(fused.get(row_id, []), gold_list)
        added_count += max(0, len(fused.get(row_id, [])) - len(core.get(row_id, [])))
        for pred in fused.get(row_id, [])[len(core.get(row_id, [])):]:
            added_by_label[str(pred.get("label"))] += 1
        for index, gold in enumerate(gold_list):
            label = str(gold.get("label"))
            bucket = area_bucket([float(v) for v in gold["bbox"]])
            by_label[label] += 1; by_bucket[bucket] += 1
            if index in fused_match and index not in core_match:
                gained_by_label[label] += 1; gained_by_bucket[bucket] += 1
            if index in core_match and index not in fused_match:
                lost_by_label[label] += 1; lost_by_bucket[bucket] += 1
    return {
        "added_predictions": added_count,
        "added_predictions_by_label": dict(added_by_label),
        "gained_tp_by_label": dict(gained_by_label),
        "lost_tp_by_label": dict(lost_by_label),
        "gained_tp_by_bucket": dict(gained_by_bucket),
        "lost_tp_by_bucket": dict(lost_by_bucket),
        "gold_by_label": dict(by_label),
        "gold_by_bucket": dict(by_bucket),
    }



def count_row(preds: list[dict[str, Any]], gold_map: dict[str, dict[str, Any]]) -> dict[str, int]:
    gold_list = list(gold_map.values())
    matched = matched_gold_indices(preds, gold_list)
    return {"matched": len(matched), "predicted": len(preds), "gold": len(gold_list)}


def aggregate_counts(counts: list[dict[str, int]]) -> dict[str, Any]:
    matched = sum(row["matched"] for row in counts)
    predicted = sum(row["predicted"] for row in counts)
    gold = sum(row["gold"] for row in counts)
    p = matched / max(predicted, 1)
    r = matched / max(gold, 1)
    return {
        "symbol_bbox_iou_0_30": {
            "matched": matched,
            "predicted": predicted,
            "gold": gold,
            "precision": round(p, 6),
            "recall": round(r, 6),
            "f1": round(0.0 if p + r == 0 else 2 * p * r / (p + r), 6),
        }
    }

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=str(P206G))
    parser.add_argument("--candidate", default=str(P212_OVERLAY))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--json-report", default=str(JSON_REPORT))
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=212)
    args = parser.parse_args()

    rows, base_preds, golds = load_overlay_preds(Path(args.baseline))
    _cand_rows, cand_preds, _cand_golds = load_overlay_preds(Path(args.candidate))
    row_ids = [str(row.get("id") or row.get("row_id")) for row in rows]
    base_metrics, _ = score_predictions(base_preds, golds, 0.0, 0.98, 900, 0)
    cand_metrics, _ = score_predictions(cand_preds, golds, 0.0, 0.98, 900, 0)
    rng = random.Random(args.seed)
    deltas = {"f1": [], "precision": [], "recall": []}
    base_row_counts = {row_id: count_row(base_preds[row_id], golds[row_id]) for row_id in row_ids}
    cand_row_counts = {row_id: count_row(cand_preds[row_id], golds[row_id]) for row_id in row_ids}
    for _ in range(args.iterations):
        sample = [rng.choice(row_ids) for _ in row_ids]
        bm = aggregate_counts([base_row_counts[row_id] for row_id in sample])
        cm = aggregate_counts([cand_row_counts[row_id] for row_id in sample])
        deltas["f1"].append(f1(cm) - f1(bm))
        deltas["precision"].append(precision(cm) - precision(bm))
        deltas["recall"].append(recall(cm) - recall(bm))
    audit = audit_added_coverage(base_preds, cand_preds, golds)
    result = {
        "id": "P212_vs_P206g_bootstrap_validation",
        "rows": len(row_ids),
        "iterations": args.iterations,
        "baseline": base_metrics,
        "candidate": cand_metrics,
        "delta_point": {
            "f1": f1(cand_metrics) - f1(base_metrics),
            "precision": precision(cand_metrics) - precision(base_metrics),
            "recall": recall(cand_metrics) - recall(base_metrics),
        },
        "bootstrap": {
            metric: {
                "ci95": [quantile(values, 0.025), quantile(values, 0.975)],
                "prob_positive": sum(1 for value in values if value > 0) / max(len(values), 1),
                "mean": sum(values) / max(len(values), 1),
            }
            for metric, values in deltas.items()
        },
        "audit": audit,
        "claim_boundary": "Paired bootstrap over the same 74 P101/P206g rows; strong internal evidence, not independent held-out validation.",
    }
    Path(args.json_report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json_report).write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# P212 vs P206g Bootstrap Validation",
        "",
        "## Metrics",
        f"- Baseline F1/P/R: {f1(base_metrics):.6f} / {precision(base_metrics):.6f} / {recall(base_metrics):.6f}",
        f"- P212 F1/P/R: {f1(cand_metrics):.6f} / {precision(cand_metrics):.6f} / {recall(cand_metrics):.6f}",
        f"- ΔF1: {result['delta_point']['f1']:.6f}",
        f"- ΔPrecision: {result['delta_point']['precision']:.6f}",
        f"- ΔRecall: {result['delta_point']['recall']:.6f}",
        "",
        "## Bootstrap 95% CI",
        f"- ΔF1 CI: [{result['bootstrap']['f1']['ci95'][0]:.6f}, {result['bootstrap']['f1']['ci95'][1]:.6f}], P>0={result['bootstrap']['f1']['prob_positive']:.3f}",
        f"- ΔPrecision CI: [{result['bootstrap']['precision']['ci95'][0]:.6f}, {result['bootstrap']['precision']['ci95'][1]:.6f}], P>0={result['bootstrap']['precision']['prob_positive']:.3f}",
        f"- ΔRecall CI: [{result['bootstrap']['recall']['ci95'][0]:.6f}, {result['bootstrap']['recall']['ci95'][1]:.6f}], P>0={result['bootstrap']['recall']['prob_positive']:.3f}",
        "",
        "## Added Coverage",
        f"- Added predictions: {audit['added_predictions']}",
        f"- Gained TP by label: `{json.dumps(audit['gained_tp_by_label'], ensure_ascii=False)}`",
        f"- Gained TP by bucket: `{json.dumps(audit['gained_tp_by_bucket'], ensure_ascii=False)}`",
        "",
        "## Claim Boundary",
        result["claim_boundary"],
    ]
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"delta": result["delta_point"], "bootstrap": result["bootstrap"], "audit": audit}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
