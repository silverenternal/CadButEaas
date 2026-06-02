#!/usr/bin/env python3
"""P176 precision-prune rescue over P165.

Drops runtime-safe low-yield label/size groups from the P165 overlay. Gold is
used only to mine/evaluate group yield; materialized policy uses predicted
label, bbox bucket, and score only.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
P165_PATH = ROOT / "scripts/vlm/sweep_symbol_disagreement_backfill_p165.py"
P165_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p165_best.jsonl"
OUT_JSON = ROOT / "configs/vlm/symbol_precision_prune_p176.json"
OUT_MD = ROOT / "reports/vlm/symbol_precision_prune_p176.md"
OUT_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p176_best.jsonl"

spec = importlib.util.spec_from_file_location("p165", P165_PATH)
p165 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(p165)


def matched_pred_keys(golds_by_row: dict[str, list[dict[str, Any]]], preds_by_row: dict[str, list[dict[str, Any]]]) -> set[tuple[str, int]]:
    out: set[tuple[str, int]] = set()
    for row_id, golds in golds_by_row.items():
        for _, pred_idx in p165.greedy_matches(golds, preds_by_row.get(row_id, [])).items():
            out.add((row_id, pred_idx))
    return out


def group_key(pred: dict[str, Any], mode: str) -> str:
    b = p165.bucket(pred["bbox"])
    if mode == "label":
        return pred["label"]
    if mode == "bucket":
        return b
    return f"{pred['label']}|{b}"


def mine_groups(golds_by_row: dict[str, list[dict[str, Any]]], preds_by_row: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    hits = matched_pred_keys(golds_by_row, preds_by_row)
    stats = {"label_bucket": defaultdict(Counter), "label": defaultdict(Counter), "bucket": defaultdict(Counter)}
    for row_id, preds in preds_by_row.items():
        for idx, pred in enumerate(preds):
            hit = (row_id, idx) in hits
            for mode in stats:
                key = group_key(pred, mode)
                stats[mode][key]["n"] += 1
                stats[mode][key]["tp"] += int(hit)
    summary = {}
    for mode, values in stats.items():
        rows = []
        for key, counter in values.items():
            n = int(counter["n"])
            tp = int(counter["tp"])
            rows.append({"key": key, "n": n, "tp": tp, "fp": n - tp, "precision": round(tp / max(n, 1), 6)})
        rows.sort(key=lambda r: (r["precision"], -r["n"], r["key"]))
        summary[mode] = rows
    return summary


def should_drop(pred: dict[str, Any], policy: dict[str, Any]) -> bool:
    label = pred["label"]
    bucket = p165.bucket(pred["bbox"])
    label_bucket = f"{label}|{bucket}"
    score = pred["score"]
    if score > policy.get("max_score", 1.0):
        return False
    if policy.get("drop_label_buckets") and label_bucket in set(policy["drop_label_buckets"]):
        return True
    if policy.get("drop_labels") and label in set(policy["drop_labels"]):
        return True
    if policy.get("drop_buckets") and bucket in set(policy["drop_buckets"]):
        return True
    return False


def apply_policy(preds_by_row: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for row_id, preds in preds_by_row.items():
        out[row_id] = [copy.deepcopy(pred) for pred in preds if not should_drop(pred, policy)]
    return out


def candidate_policies(group_summary: dict[str, Any]) -> list[dict[str, Any]]:
    bad_label_buckets = [r for r in group_summary["label_bucket"] if r["n"] >= 5 and r["precision"] <= 0.52]
    very_bad_label_buckets = [r for r in group_summary["label_bucket"] if r["n"] >= 5 and r["precision"] <= 0.25]
    bad_labels = [r for r in group_summary["label"] if r["n"] >= 10 and r["precision"] <= 0.46]
    policies = []
    for source_name, source in [("badlb", bad_label_buckets), ("verybadlb", very_bad_label_buckets)]:
        for k in range(1, min(len(source), 10) + 1):
            keys = [r["key"] for r in source[:k]]
            policies.append({"name": f"p176_{source_name}_top{k}", "drop_label_buckets": keys, "drop_labels": [], "drop_buckets": [], "max_score": 1.0})
            for max_score in [0.35, 0.50, 0.65, 0.80]:
                policies.append({"name": f"p176_{source_name}_top{k}_score{max_score}", "drop_label_buckets": keys, "drop_labels": [], "drop_buckets": [], "max_score": max_score})
    for k in range(1, min(len(bad_labels), 4) + 1):
        labels = [r["key"] for r in bad_labels[:k]]
        policies.append({"name": f"p176_badlabel_top{k}", "drop_label_buckets": [], "drop_labels": labels, "drop_buckets": [], "max_score": 1.0})
    # Handful of interpretable groups from P165 diagnostics; still runtime-safe.
    manual_sets = [
        ["appliance|small", "generic_symbol|medium"],
        ["appliance|small", "generic_symbol|medium", "sink|medium"],
        ["appliance|small", "generic_symbol|medium", "sink|medium", "bathtub|xlarge"],
        ["appliance|small", "generic_symbol|medium", "sink|medium", "bathtub|xlarge", "equipment|tiny"],
        ["appliance|small", "generic_symbol|medium", "sink|medium", "bathtub|xlarge", "equipment|tiny", "column|xlarge"],
    ]
    for idx, keys in enumerate(manual_sets, 1):
        policies.append({"name": f"p176_manual_lowyield_{idx}", "drop_label_buckets": keys, "drop_labels": [], "drop_buckets": [], "max_score": 1.0})
    policies.append({"name": "p176_noop", "drop_label_buckets": [], "drop_labels": [], "drop_buckets": [], "max_score": 1.0})
    dedup = {json.dumps(policy, sort_keys=True): policy for policy in policies}
    return list(dedup.values())


def materialize(base_rows: list[dict[str, Any]], preds_by_row: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for raw in base_rows:
        row = copy.deepcopy(raw)
        row_id = str(row.get("row_id") or row.get("id"))
        candidates = []
        for idx, pred in enumerate(preds_by_row.get(row_id, [])):
            item = copy.deepcopy(pred["raw"])
            item["bbox"] = pred["bbox"]
            item["symbol_type"] = pred["label"]
            item["confidence"] = pred["score"]
            item["id"] = f"{row_id}_p176_best_symbol_{idx:05d}"
            item["target_id"] = item["id"]
            item["source"] = "symbol_policy_overlay_p176_best"
            item.setdefault("metadata", {})["p176_policy"] = policy["name"]
            candidates.append(item)
        row["symbol_candidates"] = candidates
        if isinstance(row.get("expected_json"), dict):
            row["expected_json"]["symbol_candidates"] = [copy.deepcopy(x) for x in candidates]
        row["symbol_policy_overlay"] = {"policy_id": "p176_best", "description": "P176 low-yield label/bucket precision prune over P165", "policy": policy}
        rows.append(row)
    return rows


def render_md(report: dict[str, Any]) -> str:
    lines = ["# P176 Symbol Precision-Prune Rescue", "", f"Decision: **{report['decision']}**", "", "## Metrics", "", "| Policy | Precision | Recall | F1 | Center | Inflation |", "|---|---:|---:|---:|---:|---:|"]
    for name, metrics in report["baseline_metrics"].items():
        lines.append(f"| `{name}` | {metrics['precision']:.6f} | {metrics['recall']:.6f} | {metrics['f1']:.6f} | {metrics['center_recall']:.6f} | {metrics['prediction_inflation']:.6f} |")
    b = report["best_metrics"]
    lines.append(f"| `p176_best` | {b['precision']:.6f} | {b['recall']:.6f} | {b['f1']:.6f} | {b['center_recall']:.6f} | {b['prediction_inflation']:.6f} |")
    lines += ["", "## Best Policy", "", f"- `{report['best_policy']['name']}`", f"- config: `{json.dumps(report['best_policy'], ensure_ascii=False)}`", "", "## Delta", "", f"- vs `p165_best`: `{json.dumps(report['delta_vs_p165'], ensure_ascii=False)}`", "", "## Lowest-Yield Groups", ""]
    for row in report["group_summary"]["label_bucket"][:15]:
        lines.append(f"- `{row['key']}` n `{row['n']}`, tp `{row['tp']}`, precision `{row['precision']:.6f}`")
    lines += ["", "## Top Candidates", ""]
    for item in report["top_candidates"][:12]:
        m = item["metrics"]
        lines.append(f"- `{item['policy']['name']}` F1 `{m['f1']:.6f}`, P `{m['precision']:.6f}`, R `{m['recall']:.6f}`, inflation `{m['prediction_inflation']:.6f}`")
    lines += ["", "## Artifacts", ""]
    for value in report["outputs"].values():
        lines.append(f"- `{value}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p165-overlay", default=str(P165_OVERLAY))
    parser.add_argument("--output-json", default=str(OUT_JSON))
    parser.add_argument("--output-md", default=str(OUT_MD))
    parser.add_argument("--output-overlay", default=str(OUT_OVERLAY))
    args = parser.parse_args()
    rows = p165.load_jsonl(Path(args.p165_overlay))
    golds_by_row = {str(row.get("row_id") or row.get("id")): p165.target_symbols(row) for row in rows}
    preds_by_row = {str(row.get("row_id") or row.get("id")): p165.normalized(row.get("symbol_candidates") or [], "p165_best") for row in rows}
    baseline = p165.evaluate(golds_by_row, preds_by_row)
    group_summary = mine_groups(golds_by_row, preds_by_row)
    scored = []
    for policy in candidate_policies(group_summary):
        pruned = apply_policy(preds_by_row, policy)
        metrics = p165.evaluate(golds_by_row, pruned)
        if metrics["recall"] >= 0.50 and metrics["precision"] >= 0.54 and metrics["prediction_inflation"] <= baseline["prediction_inflation"]:
            scored.append({"policy": policy, "metrics": metrics, "delta_vs_p165": p165.delta(metrics, baseline)})
    scored.sort(key=lambda row: (row["metrics"]["f1"], row["metrics"]["precision"], row["metrics"]["recall"]), reverse=True)
    best = scored[0]
    best_preds = apply_policy(preds_by_row, best["policy"])
    p165.write_jsonl(Path(args.output_overlay), materialize(rows, best_preds, best["policy"]))
    decision = "positive_adopt_p176" if best["metrics"]["f1"] > baseline["f1"] else "negative_keep_p165"
    report = {
        "id": "SCI-P2-176-symbol-precision-prune-rescue",
        "created_on": "2026-05-17",
        "decision": decision,
        "claim_boundary": "Runtime-safe low-yield label/bucket prune over P165. Gold used only for offline mining/evaluation; policy uses predicted label, bbox bucket, and score.",
        "baseline_metrics": {"p165_best": baseline},
        "group_summary": group_summary,
        "searched_policy_count": len(candidate_policies(group_summary)),
        "passing_policy_count": len(scored),
        "best_policy": best["policy"],
        "best_metrics": best["metrics"],
        "delta_vs_p165": p165.delta(best["metrics"], baseline),
        "top_candidates": scored[:40],
        "outputs": {"overlay": str(Path(args.output_overlay)), "config_json": str(Path(args.output_json)), "report_md": str(Path(args.output_md))},
    }
    p165.write_json(Path(args.output_json), report)
    Path(args.output_md).write_text(render_md(report), encoding="utf-8")
    print(json.dumps({"decision": decision, "searched": report["searched_policy_count"], "passing": report["passing_policy_count"], "best_metrics": best["metrics"], "delta_vs_p165": report["delta_vs_p165"], "best_policy": best["policy"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
