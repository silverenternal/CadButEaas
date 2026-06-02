#!/usr/bin/env python3
"""Narrow precision-safe refinement for P221c known-good policy families."""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts/vlm"))

from freeze_symbol_p222_p221a_sink_tiny import bootstrap, metrics, score_rows  # noqa: E402
from fuse_symbol_p206g_with_p211_p212 import bbox_iou, load_p206g, write_json, write_jsonl  # noqa: E402
from train_symbol_p221c_candidate_gate import build_overlay, dist, read_jsonl, row_id  # noqa: E402

BASE = ROOT / "reports/vlm/symbol_p222_p221a_frozen_overlay.jsonl"
DATASET = ROOT / "reports/vlm/symbol_p221c_candidate_gate_dataset.jsonl"
OUT = ROOT / "reports/vlm/symbol_p221c_precision_refinement_eval.json"
MD = ROOT / "reports/vlm/symbol_p221c_precision_refinement_eval.md"
OVERLAY = ROOT / "reports/vlm/symbol_p221c_precision_refinement_overlay.jsonl"


def score(row: dict[str, Any], field: str) -> float:
    if field == "min_logreg_rf_any":
        return min(float(row.get("gate_logreg_target_any_tp", 0.0)), float(row.get("gate_rf_target_any_tp", 0.0)))
    if field == "mean_logreg_rf_any":
        return (float(row.get("gate_logreg_target_any_tp", 0.0)) + float(row.get("gate_rf_target_any_tp", 0.0))) / 2
    return float(row.get(field, 0.0))


def fuse(core: dict[str, list[dict[str, Any]]], dataset: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_row: dict[str, list[dict[str, Any]]] = {}
    for row in dataset:
        if row["label"] not in policy["labels"]:
            continue
        raw = float(row["score"])
        gate = score(row, policy["score_field"])
        feat = row.get("features") or {}
        if raw < policy["min_raw_score"] or gate < policy["threshold"]:
            continue
        if feat.get("nearest_same_iou", 0.0) > policy["max_nearest_same_iou"]:
            continue
        if row["label"] == "stair" and feat.get("stair_verifier_score", 0.0) < policy.get("min_stair_verifier", 0.0):
            continue
        cand = dict(row["candidate"])
        cand.update({"score": gate, "confidence": gate, "source": "p221c_precision_refined_added", "gate_score": gate, "raw_score": raw})
        by_row.setdefault(row["row_id"], []).append(cand)
    out = {}
    for rid, base in core.items():
        merged = [dict(x) for x in base]
        adds = []
        for cand in sorted(by_row.get(rid, []), key=lambda x: (x["gate_score"], x["raw_score"]), reverse=True):
            if len(adds) >= policy["max_add_per_row"]:
                break
            label_count = sum(1 for x in adds if x["label"] == cand["label"])
            if label_count >= policy["max_add_per_label"]:
                continue
            conflict = False
            for pred in merged + adds:
                if pred.get("label") != cand.get("label"):
                    continue
                if bbox_iou(cand["bbox"], [float(v) for v in pred["bbox"]]) >= policy["max_iou_to_existing"]:
                    conflict = True; break
                if dist(cand["bbox"], [float(v) for v in pred["bbox"]]) <= policy["min_center_dist"]:
                    conflict = True; break
            if conflict:
                continue
            adds.append(cand)
        out[rid] = merged + adds
    return out


def policies() -> list[dict[str, Any]]:
    out = []
    # Extremely narrow, paper-safe candidates around previously observed useful regions.
    configs = [
        ("stair", "gate_logreg_target_any_tp", [0.2, 0.3, 0.4, 0.5], [0.90, 0.94, 0.95, 0.97], [1, 2], [0.35, 0.5, 0.8], [0.0, 0.4, 0.6]),
        ("stair", "mean_logreg_rf_any", [0.3, 0.4, 0.5, 0.6], [0.90, 0.94, 0.95, 0.97], [1, 2], [0.35, 0.5, 0.8], [0.0, 0.4, 0.6]),
        ("equipment", "score", [0.0], [0.95, 0.97, 0.975, 0.98], [1, 2, 3], [0.8, 1.1], [0.0]),
        ("equipment", "gate_logreg_target_any_tp", [0.2, 0.4, 0.6], [0.95, 0.97, 0.975], [1, 2], [0.8, 1.1], [0.0]),
    ]
    for label, sf, thresholds, raws, adds, ious, verifiers in configs:
        for th in thresholds:
            for raw in raws:
                for max_add in adds:
                    for iou in ious:
                        for min_dist in [0.0, 8.0]:
                            for max_same_iou in [0.5, 0.8, 9.0]:
                                for ver in verifiers:
                                    out.append({
                                        "name": f"p221c_ultra_{sf}_{label}_t{th:g}_raw{raw:g}_a{max_add}_iou{iou:g}_d{min_dist:g}_siou{max_same_iou:g}_sv{ver:g}",
                                        "score_field": sf, "labels": [label], "threshold": th, "min_raw_score": raw,
                                        "max_add_per_row": max_add, "max_add_per_label": max_add,
                                        "max_iou_to_existing": iou, "min_center_dist": min_dist,
                                        "max_nearest_same_iou": max_same_iou, "min_stair_verifier": ver,
                                    })
    return out


def main() -> None:
    rows, core, golds = load_p206g(BASE)
    ids = [row_id(r) for r in rows]
    dataset = read_jsonl(DATASET)
    base_per = score_rows(core, golds, ids)
    baseline = metrics(base_per)
    raw_results = []
    for p in policies():
        fused = fuse(core, dataset, p)
        added = sum(max(0, len(fused[rid]) - len(core[rid])) for rid in ids)
        if not added:
            continue
        per = score_rows(fused, golds, ids)
        m = metrics(per)
        if m["f1"] >= baseline["f1"] and m["precision"] >= baseline["precision"] - 0.001:
            raw_results.append({"policy": p, "metrics": m, "added_predictions": added})
    raw_results.sort(key=lambda x: (x["metrics"]["precision"] >= baseline["precision"], x["metrics"]["f1"], x["metrics"]["precision"], -x["added_predictions"]), reverse=True)
    booted = []
    metric_seen = set()
    for item in raw_results[:80]:
        mk = (item["metrics"]["tp"], item["metrics"]["predicted"], item["metrics"]["fp"], item["metrics"]["fn"])
        if mk in metric_seen:
            continue
        metric_seen.add(mk)
        fused = fuse(core, dataset, item["policy"])
        per = score_rows(fused, golds, ids)
        item = dict(item)
        item["bootstrap_vs_p222"] = bootstrap(base_per, per, seed=227)
        booted.append(item)
    safe = [x for x in booted if x["bootstrap_vs_p222"]["f1_delta"]["ci95"][0] > 0 and x["bootstrap_vs_p222"]["precision_delta"]["ci95"][0] >= 0]
    weak = [x for x in booted if x["bootstrap_vs_p222"]["f1_delta"]["ci95"][0] >= 0 and x["bootstrap_vs_p222"]["precision_delta"]["ci95"][0] >= 0]
    if safe:
        decision = "precision_safe_promote_candidate"; selected = sorted(safe, key=lambda x: (x["metrics"]["f1"], x["metrics"]["precision"]), reverse=True)[0]
    elif weak:
        decision = "weak_safe_ci_touches_zero"; selected = sorted(weak, key=lambda x: (x["metrics"]["f1"], x["metrics"]["precision"]), reverse=True)[0]
    else:
        decision = "no_precision_safe_policy_found"; selected = sorted(booted, key=lambda x: (x["bootstrap_vs_p222"]["precision_delta"]["ci95"][0], x["metrics"]["f1"]), reverse=True)[0]
    write_jsonl(OVERLAY, build_overlay(rows, fuse(core, dataset, selected["policy"]), selected["policy"]))
    report = {"id":"P221c_precision_refinement_eval", "baseline_metrics":baseline, "searched_policy_count":len(policies()), "candidate_count":len(raw_results), "bootstrapped_count":len(booted), "safe_count":len(safe), "weak_safe_count":len(weak), "decision":decision, "selected":selected, "top_bootstrapped":booted[:40], "outputs":{"overlay":str(OVERLAY.relative_to(ROOT)), "markdown":str(MD.relative_to(ROOT))}}
    write_json(OUT, report)
    lines=["# P221c Precision Refinement", "", f"- Decision: `{decision}`", f"- Searched policies: `{len(policies())}`", f"- Candidates: `{len(raw_results)}`", f"- Bootstrapped: `{len(booted)}`", f"- Safe / weak-safe: `{len(safe)}` / `{len(weak)}`", "", "## Selected", f"- Policy: `{selected['policy']['name']}`", f"- Added: `{selected['added_predictions']}`", f"- Metrics: F1 `{selected['metrics']['f1']:.6f}`, P `{selected['metrics']['precision']:.6f}`, R `{selected['metrics']['recall']:.6f}`", f"- ΔF1 CI: `{selected['bootstrap_vs_p222']['f1_delta']['ci95']}`", f"- ΔP CI: `{selected['bootstrap_vs_p222']['precision_delta']['ci95']}`", f"- ΔR CI: `{selected['bootstrap_vs_p222']['recall_delta']['ci95']}`", ""]
    MD.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"decision":decision,"selected":selected,"safe_count":len(safe),"weak_safe_count":len(weak),"searched":len(policies()),"candidates":len(raw_results),"booted":len(booted)}, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
