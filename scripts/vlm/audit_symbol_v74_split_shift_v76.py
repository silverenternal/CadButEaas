#!/usr/bin/env python3
"""Audit split shift for v74 selected additions and expanded action scores."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from typing import Any

import joblib
import numpy as np

from train_symbol_expanded_action_source_policy_v74 import feature_names, vector
from train_symbol_support_suppression_v35 import load_jsonl, source_path
from train_symbol_tile_detector_v20 import write_json, write_jsonl


def bucket(row: dict[str, Any]) -> str:
    return str(row.get("bucket") or "unknown")


def reason(row: dict[str, Any]) -> str:
    return str(row.get("source_gap_reason") or "unknown")


def summarize(rows: list[dict[str, Any]], scores: np.ndarray) -> dict[str, Any]:
    out: dict[str, Any] = {"rows": len(rows)}
    counters = Counter()
    score_by_bucket: dict[str, list[float]] = defaultdict(list)
    score_by_reason: dict[str, list[float]] = defaultdict(list)
    for row, score in zip(rows, scores, strict=True):
        b = bucket(row)
        r = reason(row)
        counters[f"bucket:{b}"] += 1
        counters[f"reason:{r}"] += 1
        score_by_bucket[b].append(float(score))
        score_by_reason[r].append(float(score))
    out["counts"] = dict(counters)
    out["score_by_bucket"] = {
        key: {"n": len(vals), "mean": round(float(np.mean(vals)), 6), "p90": round(float(np.quantile(vals, 0.90)), 6), "p99": round(float(np.quantile(vals, 0.99)), 6)}
        for key, vals in score_by_bucket.items()
    }
    out["score_by_reason"] = {
        key: {"n": len(vals), "mean": round(float(np.mean(vals)), 6), "p90": round(float(np.quantile(vals, 0.90)), 6), "p99": round(float(np.quantile(vals, 0.99)), 6)}
        for key, vals in score_by_reason.items()
    }
    return out


def topk_purity(rows: list[dict[str, Any]], scores: np.ndarray, ks: list[int]) -> dict[str, Any]:
    pairs = sorted(zip(rows, scores, strict=True), key=lambda item: float(item[1]), reverse=True)
    out = {}
    for k in ks:
        subset = pairs[: min(k, len(pairs))]
        counts = Counter(bucket(row) for row, _ in subset)
        out[str(k)] = dict(counts)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--actions", default="datasets/symbol_expanded_action_source_policy_v74/actions.jsonl")
    parser.add_argument("--model", default="checkpoints/symbol_expanded_action_source_policy_v74/model.joblib")
    parser.add_argument("--output", default="reports/vlm/symbol_v74_split_shift_v76_audit.json")
    parser.add_argument("--cases-output", default="reports/vlm/symbol_v74_split_shift_v76_top_cases.jsonl")
    args = parser.parse_args()
    rows = load_jsonl(source_path(args.actions))
    bundle = joblib.load(source_path(args.model))
    model = bundle["model"]
    names = bundle.get("feature_names") or feature_names()
    by_split: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_split[str(row.get("split") or "")].append(row)
    report = {"version": "symbol_v74_split_shift_v76", "inputs": {"actions": args.actions, "model": args.model}, "splits": {}}
    cases: list[dict[str, Any]] = []
    for split, split_rows in sorted(by_split.items()):
        scores = model.predict_proba(np.asarray([vector(row, names) for row in split_rows], dtype=np.float32))[:, 1]
        report["splits"][split] = summarize(split_rows, scores)
        report["splits"][split]["topk_purity"] = topk_purity(split_rows, scores, [100, 250, 500, 1000, 2000])
        for row, score in sorted(zip(split_rows, scores, strict=True), key=lambda item: float(item[1]), reverse=True)[:200]:
            cases.append({"split": split, "score": round(float(score), 6), "page_id": row.get("page_id"), "candidate_id": row.get("candidate_id"), "bucket": bucket(row), "reason": reason(row), "label": row.get("label"), "area": row.get("area"), "features": row.get("features")})
    report["source_integrity"] = {"gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False, "offline_labels_used_for": ["split_shift_audit"]}
    write_json(source_path(args.output), report)
    write_jsonl(source_path(args.cases_output), cases)
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
