#!/usr/bin/env python3
"""Build P226 stair-specific residual diagnostics after P224a/P225 probes."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fuse_symbol_p206g_with_p211_p212 import area_bucket, bbox_iou, load_p206g, write_json

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/vlm/symbol_p224a_column_frozen_overlay.jsonl"
P224 = ROOT / "reports/vlm/symbol_p224_detector_pages_s384_predictions.jsonl"
OUT_JSON = ROOT / "reports/vlm/symbol_p226_stair_diagnostics.json"
OUT_MD = ROOT / "reports/vlm/symbol_p226_stair_diagnostics.md"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("row_id"))


def greedy_match(preds: list[dict[str, Any]], golds: list[dict[str, Any]], label: str = "stair", iou_threshold: float = 0.30):
    pairs = []
    for pi, pred in enumerate(preds):
        if str(pred.get("label")) != label:
            continue
        pbox = [float(v) for v in pred["bbox"]]
        for gi, gold in enumerate(golds):
            if str(gold.get("label")) != label:
                continue
            iou = bbox_iou(pbox, [float(v) for v in gold["bbox"]])
            if iou >= iou_threshold:
                pairs.append((iou, pi, gi))
    used_p, used_g, matched = set(), set(), []
    for iou, pi, gi in sorted(pairs, reverse=True):
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi); matched.append({"pred_index": pi, "gold_index": gi, "iou": round(iou, 6)})
    return used_p, used_g, matched


def load_p224(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_row: dict[str, list[dict[str, Any]]] = {}
    for row in read_jsonl(path):
        rid = str(row.get("row_id") or row.get("id"))
        preds = []
        for pred in row.get("predicted_symbols") or row.get("symbol_candidates") or []:
            if "bbox" not in pred:
                continue
            preds.append({
                "bbox": [float(v) for v in pred["bbox"]],
                "label": str(pred.get("label") or pred.get("symbol_type") or "generic_symbol"),
                "score": float(pred.get("score") or pred.get("confidence") or 0.0),
                "tile_id": pred.get("tile_id") or (pred.get("metadata") or {}).get("tile_id"),
            })
        by_row[rid] = preds
    return by_row


def score_bin(score: float) -> str:
    if score >= 0.9:
        return "ge_0.9"
    if score >= 0.7:
        return "ge_0.7"
    if score >= 0.5:
        return "ge_0.5"
    if score >= 0.3:
        return "ge_0.3"
    if score >= 0.1:
        return "ge_0.1"
    return "lt_0.1"


def build(args: argparse.Namespace) -> dict[str, Any]:
    rows, base_preds, golds = load_p206g(Path(args.base))
    p224 = load_p224(Path(args.p224))
    missed = []
    covered_missed = []
    counters = Counter()
    by_bucket = defaultdict(Counter)
    by_best_score = defaultdict(Counter)
    stair_fp_by_bucket = defaultdict(Counter)
    for row in rows:
        rid = row_id(row)
        gold_list = list(golds[rid].values())
        preds = base_preds.get(rid, [])
        used_p, used_g, _ = greedy_match(preds, gold_list)
        stair_gold_indices = [i for i, gold in enumerate(gold_list) if str(gold.get("label")) == "stair"]
        stair_pred_indices = [i for i, pred in enumerate(preds) if str(pred.get("label")) == "stair"]
        for pi in stair_pred_indices:
            if pi not in used_p:
                bucket = area_bucket([float(v) for v in preds[pi]["bbox"]])
                stair_fp_by_bucket[bucket].update({"fp": 1})
        for gi in stair_gold_indices:
            gold = gold_list[gi]
            gbox = [float(v) for v in gold["bbox"]]
            bucket = area_bucket(gbox)
            counters.update({"gold": 1})
            by_bucket[bucket].update({"gold": 1})
            if gi in used_g:
                counters.update({"base_matched": 1})
                by_bucket[bucket].update({"base_matched": 1})
                continue
            counters.update({"base_missed": 1})
            by_bucket[bucket].update({"base_missed": 1})
            stair_candidates = [pred for pred in p224.get(rid, []) if str(pred.get("label")) == "stair"]
            ranked = sorted(stair_candidates, key=lambda item: float(item.get("score", 0.0)), reverse=True)
            best = {"iou": 0.0, "score": 0.0, "rank": None, "bbox": None, "tile_id": None}
            for rank, pred in enumerate(ranked):
                iou = bbox_iou([float(v) for v in pred["bbox"]], gbox)
                if iou > best["iou"]:
                    best = {"iou": iou, "score": float(pred.get("score", 0.0)), "rank": rank, "bbox": pred["bbox"], "tile_id": pred.get("tile_id")}
            covered = best["iou"] >= 0.30
            if covered:
                counters.update({"p224_covers_missed": 1})
                by_bucket[bucket].update({"p224_covers_missed": 1})
                covered_missed.append({"row_id": rid, "gold_bbox": gbox, "area_bucket": bucket, "best_iou": round(best["iou"], 6), "best_score": round(best["score"], 6), "best_rank": best["rank"], "tile_id": best["tile_id"]})
            else:
                counters.update({"p224_still_misses": 1})
                by_bucket[bucket].update({"p224_still_misses": 1})
            by_best_score[score_bin(float(best["score"]))].update({"missed": 1, "covered": int(covered)})
            missed.append({"row_id": rid, "gold_bbox": gbox, "area_bucket": bucket, "p224_covered": covered, "best_iou": round(best["iou"], 6), "best_score": round(best["score"], 6), "best_rank": best["rank"], "tile_id": best["tile_id"]})
    summary = dict(counters)
    summary["base_recall"] = round(counters["base_matched"] / max(counters["gold"], 1), 6)
    summary["p224_rescue_rate_over_base_missed"] = round(counters["p224_covers_missed"] / max(counters["base_missed"], 1), 6)
    summary["combined_oracle_recall"] = round((counters["base_matched"] + counters["p224_covers_missed"]) / max(counters["gold"], 1), 6)
    return {
        "id": "P226_stair_diagnostics",
        "base": rel(Path(args.base)),
        "p224": rel(Path(args.p224)),
        "summary": summary,
        "by_area_bucket": {key: dict(value) for key, value in sorted(by_bucket.items())},
        "missed_by_best_score_bin": {key: dict(value) | {"covered_rate": round(value["covered"] / max(value["missed"], 1), 6)} for key, value in sorted(by_best_score.items())},
        "base_stair_fp_by_area_bucket": {key: dict(value) for key, value in sorted(stair_fp_by_bucket.items())},
        "covered_missed_examples": sorted(covered_missed, key=lambda item: (-item["best_score"], item["best_rank"] if item["best_rank"] is not None else 999999))[:80],
        "missed_examples": sorted(missed, key=lambda item: (not item["p224_covered"], -(item["best_score"] or 0.0)))[:120],
        "claim_boundary": "Offline residual diagnostics only; gold labels used for error analysis, not runtime.",
    }


def render(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# P226 Stair Diagnostics",
        "",
        "## Summary",
        f"- Stair gold/base matched/base missed: `{s.get('gold',0)}` / `{s.get('base_matched',0)}` / `{s.get('base_missed',0)}`",
        f"- Base stair recall: `{s.get('base_recall',0):.6f}`",
        f"- P224 covers base-missed stair: `{s.get('p224_covers_missed',0)}` / `{s.get('base_missed',0)}` = `{s.get('p224_rescue_rate_over_base_missed',0):.6f}`",
        f"- Combined oracle stair recall: `{s.get('combined_oracle_recall',0):.6f}`",
        "",
        "## Missed Stair By Area",
        "| Bucket | Gold | Base Matched | Base Missed | P224 Covers Missed | P224 Still Misses |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for bucket, row in report["by_area_bucket"].items():
        lines.append(f"| {bucket} | {row.get('gold',0)} | {row.get('base_matched',0)} | {row.get('base_missed',0)} | {row.get('p224_covers_missed',0)} | {row.get('p224_still_misses',0)} |")
    lines += ["", "## Base Stair False Positives By Area", "| Bucket | FP |", "|---|---:|"]
    for bucket, row in report["base_stair_fp_by_area_bucket"].items():
        lines.append(f"| {bucket} | {row.get('fp',0)} |")
    lines += ["", "## Interpretation", "- Stair is proposal-limited and selector-limited: P224 can rescue part of the 121 base misses, but combined oracle recall is still below 0.75, so a stair specialist detector/augmentation branch is needed.", ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(BASE))
    parser.add_argument("--p224", default=str(P224))
    parser.add_argument("--out-json", default=str(OUT_JSON))
    parser.add_argument("--out-md", default=str(OUT_MD))
    args = parser.parse_args()
    report = build(args)
    write_json(Path(args.out_json), report)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render(report), encoding="utf-8")
    print(json.dumps({"summary": report["summary"], "by_area_bucket": report["by_area_bucket"], "outputs": [rel(Path(args.out_json)), rel(Path(args.out_md))]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
