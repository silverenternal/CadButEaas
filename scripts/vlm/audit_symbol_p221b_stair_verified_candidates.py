#!/usr/bin/env python3
"""Audit verified P221b stair candidates against locked labels for planning only."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fuse_symbol_p206g_with_p211_p212 import bbox_iou, load_p206g

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/vlm/symbol_p222_p221a_frozen_overlay.jsonl"
PRED = ROOT / "reports/vlm/symbol_p221b_stair_candidate_verified_predictions.jsonl"
OUT_JSON = ROOT / "reports/vlm/symbol_p221b_stair_verified_candidate_audit.json"
OUT_MD = ROOT / "reports/vlm/symbol_p221b_stair_verified_candidate_audit.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def bucket(value: float) -> str:
    if value >= 0.9:
        return ">=0.90"
    if value >= 0.8:
        return "0.80-0.90"
    if value >= 0.7:
        return "0.70-0.80"
    if value >= 0.6:
        return "0.60-0.70"
    if value >= 0.4:
        return "0.40-0.60"
    if value > 0:
        return "0.00-0.40"
    return "0"


def main() -> None:
    rows, base_preds, golds = load_p206g(BASE)
    ids = [str(row.get("id") or row.get("row_id")) for row in rows]
    pred_rows = {str(row.get("row_id")): row.get("predicted_symbols") or [] for row in read_jsonl(PRED)}
    gold_stairs = {rid: [g for g in golds[rid].values() if str(g.get("label")) == "stair"] for rid in ids}
    base_hit_gold: dict[str, set[int]] = defaultdict(set)
    for rid in ids:
        for gi, gold in enumerate(gold_stairs[rid]):
            gbox = [float(v) for v in gold["bbox"]]
            for pred in base_preds.get(rid, []):
                if pred.get("label") == "stair" and bbox_iou([float(v) for v in pred["bbox"]], gbox) >= 0.30:
                    base_hit_gold[rid].add(gi)
                    break

    totals = Counter()
    by_verifier = Counter()
    by_raw = Counter()
    by_fused = Counter()
    rows_with_recoverable = []
    for rid in ids:
        recoverable = 0
        for pred in pred_rows.get(rid, []):
            pbox = [float(v) for v in pred["bbox"]]
            best_iou = 0.0
            best_index = None
            for gi, gold in enumerate(gold_stairs[rid]):
                iou = bbox_iou(pbox, [float(v) for v in gold["bbox"]])
                if iou > best_iou:
                    best_iou = iou
                    best_index = gi
            is_new_tp = best_iou >= 0.30 and best_index is not None and best_index not in base_hit_gold[rid]
            totals["candidates"] += 1
            totals["new_tp_candidates" if is_new_tp else "fp_or_duplicate_candidates"] += 1
            if is_new_tp:
                recoverable += 1
            for name, key in [("verifier", "verifier_score"), ("raw", "score"), ("fused", "fused_score")]:
                group = bucket(float(pred.get(key, 0.0) or 0.0))
                target = {"verifier": by_verifier, "raw": by_raw, "fused": by_fused}[name]
                target[(group, "total")] += 1
                target[(group, "new_tp" if is_new_tp else "fp_or_dup")] += 1
        if recoverable:
            rows_with_recoverable.append({"row_id": rid, "recoverable": recoverable, "base_stair_hits": len(base_hit_gold[rid]), "gold_stairs": len(gold_stairs[rid])})

    def table(counter: Counter) -> list[dict[str, Any]]:
        out = []
        order = [">=0.90", "0.80-0.90", "0.70-0.80", "0.60-0.70", "0.40-0.60", "0.00-0.40", "0"]
        for group in order:
            total = counter[(group, "total")]
            if not total:
                continue
            tp = counter[(group, "new_tp")]
            out.append({"bucket": group, "total": total, "new_tp": tp, "fp_or_duplicate": counter[(group, "fp_or_dup")], "new_tp_rate": round(tp / total, 4)})
        return out

    report = {
        "id": "P221b_stair_verified_candidate_audit",
        "scope": "planning_only_gold_used_for_error_analysis_not_runtime",
        "baseline": str(BASE.relative_to(ROOT)),
        "predictions": str(PRED.relative_to(ROOT)),
        "totals": dict(totals),
        "rows_with_recoverable_new_tp": rows_with_recoverable[:50],
        "score_bucket_tables": {"verifier_score": table(by_verifier), "raw_score": table(by_raw), "fused_score": table(by_fused)},
        "interpretation": "The current P213b+P221b stair verification branch contains recoverable stair TPs but low precision; next step should add a learned candidate-level accept/reject gate with local raster features and context negatives before any promotion.",
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = ["# P221b Stair Verified Candidate Audit", "", "## Totals", f"- Candidates: `{totals['candidates']}`", f"- New TP candidates: `{totals['new_tp_candidates']}`", f"- FP/duplicate candidates: `{totals['fp_or_duplicate_candidates']}`", "", "## Score Buckets"]
    for name, rows in report["score_bucket_tables"].items():
        lines += ["", f"### {name}", "| Bucket | Total | New TP | FP/Dup | New TP Rate |", "|---|---:|---:|---:|---:|"]
        for row in rows:
            lines.append(f"| {row['bucket']} | {row['total']} | {row['new_tp']} | {row['fp_or_duplicate']} | {row['new_tp_rate']:.4f} |")
    lines += ["", "## Interpretation", "- Current verifier scores alone do not separate new stair TPs cleanly enough for promotion.", "- Next branch should train a candidate-level gate using P213b proposal crops, P221b detector responses, bbox/context features, and hard negatives.", ""]
    OUT_MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"totals": report["totals"], "reports": [str(OUT_MD.relative_to(ROOT)), str(OUT_JSON.relative_to(ROOT))]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
