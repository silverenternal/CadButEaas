#!/usr/bin/env python3
"""Paired bootstrap validation for P206d against the P202 symbol baseline."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

import sweep_symbol_disagreement_backfill_p165 as p165

ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("row_id") or row.get("id"))


def row_counts(golds: list[dict[str, Any]], preds: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter(gold=len(golds), pred=len(preds), tp=0, center=0)
    used_iou: set[int] = set()
    used_center: set[int] = set()
    for gold in golds:
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
            totals["tp"] += 1
        if center_idx is not None:
            used_center.add(center_idx)
            totals["center"] += 1
    return dict(totals)


def metrics(rows: list[dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    for row in rows:
        totals.update(row["counts"])
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
    }


def percentile(values: list[float], q: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * q
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def paired_rows(base_rows: list[dict[str, Any]], candidate_rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    cand_by_id = {row_id(row): row for row in candidate_rows}
    base_out = []
    cand_out = []
    for base in base_rows:
        rid = row_id(base)
        if rid not in cand_by_id:
            raise ValueError(f"missing candidate row {rid}")
        golds = p165.target_symbols(base)
        base_preds = p165.normalized(base.get("symbol_candidates") or [], "p202")
        cand_preds = p165.normalized(cand_by_id[rid].get("symbol_candidates") or [], "p206d")
        base_out.append({"row_id": rid, "counts": row_counts(golds, base_preds)})
        cand_out.append({"row_id": rid, "counts": row_counts(golds, cand_preds)})
    return base_out, cand_out


def bootstrap(base: list[dict[str, Any]], candidate: list[dict[str, Any]], iterations: int, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    f1_delta = []
    precision_delta = []
    recall_delta = []
    center_delta = []
    for _ in range(iterations):
        sample = [rng.randrange(len(base)) for _ in range(len(base))]
        bm = metrics([base[idx] for idx in sample])
        cm = metrics([candidate[idx] for idx in sample])
        f1_delta.append(cm["f1"] - bm["f1"])
        precision_delta.append(cm["precision"] - bm["precision"])
        recall_delta.append(cm["recall"] - bm["recall"])
        center_delta.append(cm["center_recall"] - bm["center_recall"])
    def pack(values: list[float]) -> dict[str, Any]:
        return {
            "mean": round(sum(values) / max(len(values), 1), 6),
            "ci95": [round(percentile(values, 0.025), 6), round(percentile(values, 0.975), 6)],
            "prob_positive": round(sum(1 for value in values if value > 0) / max(len(values), 1), 6),
        }
    return {
        "f1_delta": pack(f1_delta),
        "precision_delta": pack(precision_delta),
        "recall_delta": pack(recall_delta),
        "center_delta": pack(center_delta),
    }


def render(report: dict[str, Any]) -> str:
    bm = report["baseline_metrics"]
    cm = report["candidate_metrics"]
    baseline_name = report.get("baseline_name", "baseline")
    candidate_name = report.get("candidate_name", "candidate")
    fd = report["bootstrap"]["f1_delta"]
    rd = report["bootstrap"]["recall_delta"]
    pd = report["bootstrap"]["precision_delta"]
    cd = report["bootstrap"]["center_delta"]
    return "\n".join([
        "# Paired Symbol Overlay Bootstrap Validation",
        "",
        "## Claim Boundary",
        "",
        report["claim_boundary"],
        "",
        "## Locked P101 Metrics",
        "",
        "| Variant | Precision | Recall | F1 | Center | Inflation |",
        "|---|---:|---:|---:|---:|---:|",
        f"| `{baseline_name}` | {bm['precision']:.6f} | {bm['recall']:.6f} | {bm['f1']:.6f} | {bm['center_recall']:.6f} | {bm['prediction_inflation']:.6f} |",
        f"| `{candidate_name}` | {cm['precision']:.6f} | {cm['recall']:.6f} | {cm['f1']:.6f} | {cm['center_recall']:.6f} | {cm['prediction_inflation']:.6f} |",
        "",
        "## Paired Bootstrap",
        "",
        f"- Iterations: `{report['bootstrap_iterations']}`",
        f"- Rows: `{report['row_count']}`",
        f"- ΔF1 mean/CI/P(>0): `{fd['mean']:.6f}` / `[{fd['ci95'][0]:.6f}, {fd['ci95'][1]:.6f}]` / `{fd['prob_positive']:.3f}`",
        f"- ΔRecall mean/CI/P(>0): `{rd['mean']:.6f}` / `[{rd['ci95'][0]:.6f}, {rd['ci95'][1]:.6f}]` / `{rd['prob_positive']:.3f}`",
        f"- ΔPrecision mean/CI/P(>0): `{pd['mean']:.6f}` / `[{pd['ci95'][0]:.6f}, {pd['ci95'][1]:.6f}]` / `{pd['prob_positive']:.3f}`",
        f"- ΔCenter mean/CI/P(>0): `{cd['mean']:.6f}` / `[{cd['ci95'][0]:.6f}, {cd['ci95'][1]:.6f}]` / `{cd['prob_positive']:.3f}`",
        "",
        "## Interpretation",
        "",
        "- Positive lower-bound ΔF1 indicates a stronger internal promotion; if the interval crosses zero, treat it as directional only.",
        "- This remains P101/bootstrap-bounded evidence unless an independent held-out split confirms it.",
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-overlay", default="reports/vlm/symbol_context_verifier_p202_overlay.jsonl")
    parser.add_argument("--candidate-overlay", default="reports/vlm/symbol_p205b_ranker_regressor_p206d_overlay.jsonl")
    parser.add_argument("--iterations", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=206)
    parser.add_argument("--out-json", default="configs/vlm/symbol_p206d_bootstrap_validation.json")
    parser.add_argument("--out-md", default="reports/vlm/symbol_p206d_bootstrap_validation.md")
    parser.add_argument("--baseline-name", default="baseline")
    parser.add_argument("--candidate-name", default="candidate")
    args = parser.parse_args()

    base_rows = load_jsonl(ROOT / args.baseline_overlay)
    cand_rows = load_jsonl(ROOT / args.candidate_overlay)
    base_per_row, cand_per_row = paired_rows(base_rows, cand_rows)
    report = {
        "id": "P206d_paired_bootstrap_validation",
        "claim_boundary": "Paired bootstrap over the same 74 P101 rows used for planning; this is internal stability evidence, not an independent held-out paper claim.",
        "baseline_overlay": args.baseline_overlay,
        "candidate_overlay": args.candidate_overlay,
        "baseline_name": args.baseline_name,
        "candidate_name": args.candidate_name,
        "bootstrap_iterations": args.iterations,
        "seed": args.seed,
        "row_count": len(base_per_row),
        "baseline_metrics": metrics(base_per_row),
        "candidate_metrics": metrics(cand_per_row),
        "bootstrap": bootstrap(base_per_row, cand_per_row, args.iterations, args.seed),
        "outputs": {"json": args.out_json, "md": args.out_md},
    }
    write_json(ROOT / args.out_json, report)
    out_md = ROOT / args.out_md
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_md.write_text(render(report), encoding="utf-8")
    print(json.dumps({"baseline": report["baseline_metrics"], "candidate": report["candidate_metrics"], "bootstrap": report["bootstrap"], "outputs": report["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
