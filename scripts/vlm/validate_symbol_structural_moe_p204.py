#!/usr/bin/env python3
"""P204 validation for the P203 structural symbol MoE ablation.

This script recomputes row-level metrics from materialized raster-only overlay
artifacts and estimates paired bootstrap confidence intervals. Gold annotations
are used only for offline evaluation.
"""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import sweep_symbol_disagreement_backfill_p165 as p165
from run_symbol_structural_moe_p203 import STAGES

ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def target_label(item: dict[str, Any]) -> str:
    return str(item.get("semantic_type") or item.get("symbol_type") or item.get("raw_label") or item.get("label") or "generic_symbol")


def target_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate((row.get("targets") or {}).get("symbol") or []):
        box = p165.bbox4(item.get("bbox"))
        if box is not None:
            out.append({
                "id": str(item.get("target_id") or idx),
                "bbox": box,
                "bucket": p165.bucket(box),
                "label": target_label(item),
            })
    return out


def row_counts(golds: list[dict[str, Any]], preds: list[dict[str, Any]]) -> dict[str, Any]:
    used_iou: set[int] = set()
    used_center: set[int] = set()
    counts = Counter(gold=len(golds), pred=len(preds), tp=0, center=0)
    by_bucket_gold = Counter()
    by_bucket_tp = Counter()
    by_bucket_center = Counter()
    by_label_gold = Counter()
    by_label_tp = Counter()
    for gold in golds:
        by_bucket_gold[gold["bucket"]] += 1
        by_label_gold[gold["label"]] += 1
        best_idx = None
        best_iou = 0.0
        center_idx = None
        for idx, pred in enumerate(preds):
            overlap = p165.iou(pred["bbox"], gold["bbox"])
            if idx not in used_iou and overlap > best_iou:
                best_iou = overlap
                best_idx = idx
            if center_idx is None and idx not in used_center and p165.center_covered(pred["bbox"], gold["bbox"]):
                center_idx = idx
        if best_idx is not None and best_iou >= 0.30:
            used_iou.add(best_idx)
            counts["tp"] += 1
            by_bucket_tp[gold["bucket"]] += 1
            by_label_tp[gold["label"]] += 1
        if center_idx is not None:
            used_center.add(center_idx)
            counts["center"] += 1
            by_bucket_center[gold["bucket"]] += 1
    return {
        "counts": dict(counts),
        "by_bucket_gold": dict(by_bucket_gold),
        "by_bucket_tp": dict(by_bucket_tp),
        "by_bucket_center": dict(by_bucket_center),
        "by_label_gold": dict(by_label_gold),
        "by_label_tp": dict(by_label_tp),
    }


def metrics_from_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    by_bucket_gold = Counter()
    by_bucket_tp = Counter()
    by_bucket_center = Counter()
    by_label_gold = Counter()
    by_label_tp = Counter()
    for row in rows:
        totals.update(row["counts"])
        by_bucket_gold.update(row["by_bucket_gold"])
        by_bucket_tp.update(row["by_bucket_tp"])
        by_bucket_center.update(row["by_bucket_center"])
        by_label_gold.update(row["by_label_gold"])
        by_label_tp.update(row["by_label_tp"])
    precision = totals["tp"] / max(totals["pred"], 1)
    recall = totals["tp"] / max(totals["gold"], 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
    return {
        "tp": int(totals["tp"]),
        "predicted": int(totals["pred"]),
        "gold": int(totals["gold"]),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
        "center_recall": round(totals["center"] / max(totals["gold"], 1), 6),
        "prediction_inflation": round(totals["pred"] / max(totals["gold"], 1), 6),
        "by_bucket_iou_recall": {key: round(by_bucket_tp[key] / max(by_bucket_gold[key], 1), 6) for key in sorted(by_bucket_gold)},
        "by_bucket_center_recall": {key: round(by_bucket_center[key] / max(by_bucket_gold[key], 1), 6) for key in sorted(by_bucket_gold)},
        "by_label_iou_recall": {key: round(by_label_tp[key] / max(by_label_gold[key], 1), 6) for key in sorted(by_label_gold)},
    }


def stage_row_metrics(stage: dict[str, Any]) -> dict[str, Any]:
    path = ROOT / stage["overlay"]
    rows = load_jsonl(path)
    per_row = []
    for row in rows:
        row_id = str(row.get("row_id") or row.get("id"))
        golds = target_symbols(row)
        preds = p165.normalized(row.get("symbol_candidates") or [], path.stem)
        per_row.append({"row_id": row_id, **row_counts(golds, preds)})
    return {"id": stage["id"], "role": stage["role"], "overlay": stage["overlay"], "rows": per_row, "metrics": metrics_from_counts(per_row)}


def percentile(values: list[float], q: float) -> float:
    if not values:
        return 0.0
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def bootstrap(stages: list[dict[str, Any]], baseline_id: str, iterations: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    row_count = len(stages[0]["rows"])
    baseline = next(stage for stage in stages if stage["id"] == baseline_id)
    deltas: dict[str, list[float]] = {stage["id"]: [] for stage in stages}
    values: dict[str, list[float]] = {stage["id"]: [] for stage in stages}
    for _ in range(iterations):
        sample = [rng.randrange(row_count) for _ in range(row_count)]
        base_f1 = metrics_from_counts([baseline["rows"][idx] for idx in sample])["f1"]
        for stage in stages:
            f1 = metrics_from_counts([stage["rows"][idx] for idx in sample])["f1"]
            values[stage["id"]].append(f1)
            deltas[stage["id"]].append(f1 - base_f1)
    out = {}
    for stage in stages:
        sid = stage["id"]
        out[sid] = {
            "f1_ci95": [round(percentile(values[sid], 0.025), 6), round(percentile(values[sid], 0.975), 6)],
            "delta_vs_baseline_ci95": [round(percentile(deltas[sid], 0.025), 6), round(percentile(deltas[sid], 0.975), 6)],
            "delta_vs_baseline_mean": round(sum(deltas[sid]) / max(len(deltas[sid]), 1), 6),
            "prob_delta_positive": round(sum(1 for value in deltas[sid] if value > 0) / max(len(deltas[sid]), 1), 6),
        }
    return out


def top_residual_labels(best: dict[str, Any], limit: int = 12) -> list[dict[str, Any]]:
    gold = Counter()
    tp = Counter()
    for row in best["rows"]:
        gold.update(row["by_label_gold"])
        tp.update(row["by_label_tp"])
    records = []
    for label, count in gold.items():
        miss = count - tp[label]
        records.append({"label": label, "gold": int(count), "tp": int(tp[label]), "fn": int(miss), "recall": round(tp[label] / max(count, 1), 6)})
    return sorted(records, key=lambda item: (-item["fn"], item["label"]))[:limit]


def render(report: dict[str, Any]) -> str:
    baseline_id = report["baseline_id"]
    iterations = report["bootstrap_iterations"]
    row_count = report["row_count"]
    lines = [
        "# P204 Structural Symbol MoE Validation",
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
        "## Bootstrap Summary",
        "",
        f"- Baseline for paired deltas: `{baseline_id}`",
        f"- Bootstrap iterations: {iterations}",
        f"- Rows resampled: {row_count}",
        "",
        "| Stage | Precision | Recall | F1 | F1 95% CI | ΔF1 vs Baseline 95% CI | P(Δ>0) | Center | Inflation |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for stage in report["stages"]:
        stage_id = stage["id"]
        metrics = stage["metrics"]
        boot = report["bootstrap"][stage_id]
        f1_ci = boot["f1_ci95"]
        delta_ci = boot["delta_vs_baseline_ci95"]
        lines.append(
            f"| `{stage_id}` | {metrics['precision']:.6f} | {metrics['recall']:.6f} | {metrics['f1']:.6f} | "
            f"[{f1_ci[0]:.6f}, {f1_ci[1]:.6f}] | "
            f"[{delta_ci[0]:.6f}, {delta_ci[1]:.6f}] | "
            f"{boot['prob_delta_positive']:.3f} | {metrics['center_recall']:.6f} | {metrics['prediction_inflation']:.6f} |"
        )
    lines += ["", "## Best-Stage Residual Labels", "", "| Label | Gold | TP | FN | Recall |", "|---|---:|---:|---:|---:|"]
    for item in report["best_stage_residual_labels"]:
        lines.append(f"| `{item['label']}` | {item['gold']} | {item['tp']} | {item['fn']} | {item['recall']:.6f} |")
    lines += ["", "## Interpretation", ""]
    for note in report["interpretation"]:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-id", default="P197b_detector_fusion_box_refine")
    parser.add_argument("--iterations", type=int, default=2000)
    parser.add_argument("--seed", type=int, default=204)
    parser.add_argument("--out-json", default="configs/vlm/symbol_structural_moe_p204_validation.json")
    parser.add_argument("--out-md", default="reports/vlm/symbol_structural_moe_p204_validation.md")
    args = parser.parse_args()
    stages = [stage_row_metrics(stage) for stage in STAGES]
    row_count = len(stages[0]["rows"])
    if any(len(stage["rows"]) != row_count for stage in stages):
        raise ValueError("All stages must contain the same row count for paired bootstrap validation")
    boot = bootstrap(stages, args.baseline_id, args.iterations, args.seed)
    best = max(stages, key=lambda item: item["metrics"]["f1"])
    report = {
        "id": "P204_structural_symbol_moe_validation",
        "claim_boundary": "This is a paired bootstrap validation over the same 74-row P101 overlay rows, not an independent cross-dataset result. It strengthens internal evidence but still must be labeled as P101/bootstrap-bounded in the paper.",
        "baseline_id": args.baseline_id,
        "bootstrap_iterations": args.iterations,
        "seed": args.seed,
        "row_count": row_count,
        "stages": [{k: v for k, v in stage.items() if k != "rows"} for stage in stages],
        "bootstrap": boot,
        "best_stage_id": best["id"],
        "best_stage_residual_labels": top_residual_labels(best),
        "interpretation": [
            "P202 remains the best P101/bootstrap-bounded stage by F1, mostly by precision and inflation reduction.",
            "Because the lower confidence bound of the P202 delta should be checked before paper use, this report is safer than a single locked point estimate.",
            "Residual labels identify where P205 should spend GPU budget: high-FN classes rather than global detector scaling.",
        ],
        "outputs": {"json": args.out_json, "md": args.out_md},
    }
    write_json(ROOT / args.out_json, report)
    out_md = ROOT / args.out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render(report), encoding="utf-8")
    print(json.dumps({"best_stage_id": report["best_stage_id"], "best_metrics": best["metrics"], "outputs": report["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
