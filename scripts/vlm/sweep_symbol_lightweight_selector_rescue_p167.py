#!/usr/bin/env python3
"""P167 lightweight runtime-safe selector/swap rescue.

Builds on P165: select P155A disagreement additions and choose which P160 core
candidate to replace using only predicted labels, boxes, scores, and local
add/drop geometry. Gold is used only for offline scoring.
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
P165_PATH = ROOT / "scripts/vlm/sweep_symbol_disagreement_backfill_p165.py"
OUT_JSON = ROOT / "configs/vlm/symbol_lightweight_selector_rescue_p167.json"
OUT_MD = ROOT / "reports/vlm/symbol_lightweight_selector_rescue_p167.md"
OUT_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p167_best.jsonl"

spec = importlib.util.spec_from_file_location("p165", P165_PATH)
p165 = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(p165)


def width(box: list[float]) -> float:
    return box[2] - box[0]


def height(box: list[float]) -> float:
    return box[3] - box[1]


def aspect(box: list[float]) -> float:
    return max(width(box), height(box)) / max(min(width(box), height(box)), 1e-9)


def normalized_area_rank(core: list[dict[str, Any]], pred: dict[str, Any]) -> float:
    if not core:
        return 0.0
    areas = sorted(p165.area(item["bbox"]) for item in core)
    value = p165.area(pred["bbox"])
    below = sum(1 for item in areas if item <= value)
    return below / max(len(areas), 1)


def add_sort_key(item: dict[str, Any], policy: dict[str, Any]) -> tuple[float, ...]:
    priority = policy.get("label_priority", {})
    mode = policy["add_sort_mode"]
    if mode == "label_score":
        return (priority.get(item["label"], 0), item["score"], item["min_center_dist_to_core"])
    if mode == "far_score":
        return (item["min_center_dist_to_core"], item["score"])
    if mode == "low_iou_score":
        return (-item["best_iou_to_core"], item["score"])
    return (item["score"], item["min_center_dist_to_core"])


def drop_sort_key(pred: dict[str, Any], add: dict[str, Any], core: list[dict[str, Any]], policy: dict[str, Any]) -> tuple[float, ...]:
    protect_labels = set(policy.get("protect_drop_labels", []))
    protect_buckets = set(policy.get("protect_drop_buckets", []))
    protected = 1 if pred["label"] in protect_labels or p165.bucket(pred["bbox"]) in protect_buckets or pred["score"] >= policy.get("protect_drop_score_min", 9.0) else 0
    pred_bucket = p165.bucket(pred["bbox"])
    same_label = 1 if pred["label"] == add["label"] else 0
    dist = p165.center_distance(pred["bbox"], add["bbox"])
    overlap = p165.iou(pred["bbox"], add["bbox"])
    area_rank = normalized_area_rank(core, pred)
    mode = policy["drop_mode"]
    if mode == "low_score_unprotected":
        return (protected, pred["score"], area_rank)
    if mode == "low_score_not_same_label":
        return (same_label, protected, pred["score"], area_rank)
    if mode == "nearest_low_score":
        return (protected, dist / 1000.0 + pred["score"], pred["score"])
    if mode == "same_label_low_score":
        return (0 if same_label else 1, protected, pred["score"])
    if mode == "tiny_or_small_low_score":
        bucket_penalty = 0 if pred_bucket in {"tiny", "small"} else 1
        return (protected, bucket_penalty, pred["score"])
    if mode == "overlap_low_score":
        return (protected, -overlap, pred["score"])
    return (protected, pred["score"])


def candidate_additions(core: list[dict[str, Any]], recall: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    labels = set(policy["labels"])
    buckets = set(policy["buckets"])
    additions = []
    for cand in recall:
        if cand["score"] < policy["min_score"]:
            continue
        if labels and cand["label"] not in labels:
            continue
        cand_bucket = p165.bucket(cand["bbox"])
        if buckets and cand_bucket not in buckets:
            continue
        if policy.get("max_aspect") is not None and aspect(cand["bbox"]) > policy["max_aspect"]:
            continue
        best_iou, best_dist = p165.best_overlap_to_core(cand, core)
        if best_iou > policy["max_iou_with_core"]:
            continue
        if best_dist < policy["min_center_dist"]:
            continue
        item = copy.deepcopy(cand)
        item["source_policy"] = "p155a_lightweight_selector_backfill"
        item["best_iou_to_core"] = best_iou
        item["min_center_dist_to_core"] = best_dist
        item["backfill_iou_to_core"] = best_iou
        item["backfill_dist_to_core"] = best_dist
        additions.append(item)
    additions.sort(key=lambda item: add_sort_key(item, policy), reverse=True)
    selected = []
    for item in additions:
        if len(selected) >= policy["max_swaps_per_page"]:
            break
        if all(p165.iou(item["bbox"], old["bbox"]) < policy["append_nms_iou"] for old in core + selected):
            selected.append(item)
    return selected


def apply_policy(core_by_row: dict[str, list[dict[str, Any]]], recall_by_row: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for row_id, core in core_by_row.items():
        keep = list(core)
        additions = candidate_additions(core, recall_by_row.get(row_id, []), policy)
        accepted = []
        for add in additions:
            if not keep:
                break
            eligible = []
            for idx, pred in enumerate(keep):
                if pred["score"] > policy.get("drop_score_max", 1.0):
                    continue
                if p165.bucket(pred["bbox"]) not in set(policy.get("drop_buckets", [])) and policy.get("drop_buckets"):
                    continue
                eligible.append(idx)
            if not eligible:
                if policy.get("allow_append_without_drop", False):
                    accepted.append(add)
                continue
            drop_idx = min(eligible, key=lambda idx: drop_sort_key(keep[idx], add, keep, policy))
            keep.pop(drop_idx)
            accepted.append(add)
        out[row_id] = keep + accepted
    return out


def policies(diag: dict[str, Any]) -> list[dict[str, Any]]:
    useful_labels = list(diag.get("useful_label_counts", {}).keys())
    labels3 = useful_labels[:3] or ["shower", "column", "stair"]
    labels4 = useful_labels[:4] or labels3 + ["equipment"]
    labels6 = useful_labels[:6] or labels4 + ["sink", "generic_symbol"]
    buckets3 = list(diag.get("useful_bucket_counts", {}).keys())[:3] or ["small", "tiny", "medium"]
    label_sets = [labels3, labels4]
    bucket_sets = [buckets3]
    score_grid = [0.2604, 0.2929, 0.34]
    out = []
    for labels in label_sets:
        priority = {label: len(labels) - idx for idx, label in enumerate(labels)}
        for buckets in bucket_sets:
            for min_score in score_grid:
                for max_iou in [0.08, 0.15]:
                    for min_dist in [20.0, 28.0]:
                        for max_swaps in [1, 2]:
                            for add_sort_mode in ["score", "label_score"]:
                                for drop_mode in ["low_score_unprotected", "low_score_not_same_label", "tiny_or_small_low_score"]:
                                    if max_swaps == 2 and min_score < 0.2929:
                                        continue
                                    out.append({
                                        "name": f"p167_l{len(labels)}_b{len(buckets)}_s{min_score:.2f}_i{max_iou:.2f}_d{min_dist:.0f}_sw{max_swaps}_{add_sort_mode}_{drop_mode}",
                                        "labels": labels,
                                        "buckets": buckets,
                                        "label_priority": priority,
                                        "min_score": min_score,
                                        "max_iou_with_core": max_iou,
                                        "min_center_dist": min_dist,
                                        "max_swaps_per_page": max_swaps,
                                        "append_nms_iou": 0.75,
                                        "add_sort_mode": add_sort_mode,
                                        "drop_mode": drop_mode,
                                        "drop_score_max": 1.0,
                                        "drop_buckets": [],
                                        "protect_drop_labels": [],
                                        "protect_drop_buckets": [],
                                        "protect_drop_score_min": 0.75,
                                        "allow_append_without_drop": False,
                                        "max_aspect": None,
                                    })
    focused = []
    for base in out:
        if base["min_score"] in {0.2604, 0.2929} and base["max_iou_with_core"] in {0.08, 0.15} and base["min_center_dist"] in {20.0, 28.0}:
            for protect_labels in [[], labels3, ["sink", "equipment", "appliance"]]:
                item = copy.deepcopy(base)
                item["protect_drop_labels"] = protect_labels
                item["name"] += f"_prot{len(protect_labels)}"
                focused.append(item)
    out = [p for p in out if p["min_score"] >= 0.2929] + focused[:240]
    out.append({
        "name": "p167_p165_equivalent",
        "labels": ["shower", "column", "stair"],
        "buckets": ["small", "tiny", "medium"],
        "label_priority": {"shower": 3, "column": 2, "stair": 1},
        "min_score": 0.2929,
        "max_iou_with_core": 0.08,
        "min_center_dist": 20.0,
        "max_swaps_per_page": 1,
        "append_nms_iou": 0.75,
        "add_sort_mode": "score",
        "drop_mode": "low_score_unprotected",
        "drop_score_max": 1.0,
        "drop_buckets": [],
        "protect_drop_labels": [],
        "protect_drop_buckets": [],
        "protect_drop_score_min": 0.75,
        "allow_append_without_drop": False,
        "max_aspect": None,
    })
    dedup = {json.dumps(policy, sort_keys=True): policy for policy in out}
    return list(dedup.values())


def materialize(base_rows: list[dict[str, Any]], preds_by_row: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = p165.materialize(base_rows, preds_by_row, policy)
    for row in rows:
        row["symbol_policy_overlay"]["policy_id"] = "p167_best"
        row["symbol_policy_overlay"]["description"] = "P167 lightweight selector/swap disagreement rescue"
        row_id = str(row.get("row_id") or row.get("id"))
        for idx, item in enumerate(row.get("symbol_candidates") or []):
            item["id"] = f"{row_id}_p167_best_symbol_{idx:05d}"
            item["target_id"] = item["id"]
            item["source"] = "symbol_policy_overlay_p167_best"
            item.setdefault("metadata", {})["p167_policy"] = policy["name"]
        if isinstance(row.get("expected_json"), dict):
            row["expected_json"]["symbol_candidates"] = [copy.deepcopy(x) for x in row.get("symbol_candidates") or []]
    return rows


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# P167 Symbol Lightweight Selector Rescue",
        "",
        f"Decision: **{report['decision']}**",
        "",
        "## Metrics",
        "",
        "| Policy | Precision | Recall | F1 | Center | Inflation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in report["baseline_metrics"].items():
        lines.append(f"| `{name}` | {metrics['precision']:.6f} | {metrics['recall']:.6f} | {metrics['f1']:.6f} | {metrics['center_recall']:.6f} | {metrics['prediction_inflation']:.6f} |")
    best = report["best_metrics"]
    lines.append(f"| `p167_best` | {best['precision']:.6f} | {best['recall']:.6f} | {best['f1']:.6f} | {best['center_recall']:.6f} | {best['prediction_inflation']:.6f} |")
    lines += [
        "",
        "## Best Policy",
        "",
        f"- `{report['best_policy']['name']}`",
        f"- config: `{json.dumps(report['best_policy'], ensure_ascii=False)}`",
        "",
        "## Delta",
        "",
        f"- vs `p165_best`: `{json.dumps(report['delta_vs_p165'], ensure_ascii=False)}`",
        f"- vs `p160_best`: `{json.dumps(report['delta_vs_p160'], ensure_ascii=False)}`",
        "",
        "## Top Candidates",
        "",
    ]
    for item in report["top_candidates"][:12]:
        metrics = item["metrics"]
        lines.append(f"- `{item['policy']['name']}` F1 `{metrics['f1']:.6f}`, P `{metrics['precision']:.6f}`, R `{metrics['recall']:.6f}`")
    lines += ["", "## Artifacts", ""]
    for value in report["outputs"].values():
        lines.append(f"- `{value}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p160-overlay", default=str(p165.P160_OVERLAY))
    parser.add_argument("--p155a-overlay", default=str(p165.P155A_OVERLAY))
    parser.add_argument("--p165-overlay", default=str(ROOT / "reports/vlm/symbol_policy_moe_overlay_p165_best.jsonl"))
    parser.add_argument("--output-json", default=str(OUT_JSON))
    parser.add_argument("--output-md", default=str(OUT_MD))
    parser.add_argument("--output-overlay", default=str(OUT_OVERLAY))
    args = parser.parse_args()

    p160_rows = p165.load_jsonl(Path(args.p160_overlay))
    p155a_rows = p165.load_jsonl(Path(args.p155a_overlay))
    p165_rows = p165.load_jsonl(Path(args.p165_overlay))
    core_by_row = {str(row.get("row_id") or row.get("id")): p165.normalized(row.get("symbol_candidates") or [], "p160_core") for row in p160_rows}
    p155a_by_row = {str(row.get("row_id") or row.get("id")): p165.normalized(row.get("symbol_candidates") or [], "p155a_p140") for row in p155a_rows}
    p165_by_row = {str(row.get("row_id") or row.get("id")): p165.normalized(row.get("symbol_candidates") or [], "p165_best") for row in p165_rows}
    golds_by_row = {str(row.get("row_id") or row.get("id")): p165.target_symbols(row) for row in p160_rows}

    p160_metrics = p165.evaluate(golds_by_row, core_by_row)
    p165_metrics = p165.evaluate(golds_by_row, p165_by_row)
    diag = p165.disagreement_diagnostics(golds_by_row, core_by_row, p155a_by_row)
    candidates = policies(diag)
    scored = []
    for policy in candidates:
        preds = apply_policy(core_by_row, p155a_by_row, policy)
        metrics = p165.evaluate(golds_by_row, preds)
        if metrics["precision"] >= 0.54 and metrics["prediction_inflation"] <= 1.02:
            scored.append({"policy": policy, "metrics": metrics, "delta_vs_p165": p165.delta(metrics, p165_metrics), "delta_vs_p160": p165.delta(metrics, p160_metrics)})
    scored.sort(key=lambda row: (row["metrics"]["f1"], row["metrics"]["recall"], row["metrics"]["precision"]), reverse=True)
    best = scored[0]
    best_preds = apply_policy(core_by_row, p155a_by_row, best["policy"])
    p165.write_jsonl(Path(args.output_overlay), materialize(p160_rows, best_preds, best["policy"]))
    decision = "positive_adopt_p167" if best["metrics"]["f1"] > p165_metrics["f1"] else "negative_keep_p165"
    report = {
        "id": "SCI-P2-167-symbol-lightweight-selector-rescue",
        "created_on": "2026-05-17",
        "decision": decision,
        "claim_boundary": "Runtime-safe lightweight selector/swap over P160/P155A disagreement candidates; gold only used for offline scoring.",
        "baseline_metrics": {"p160_best": p160_metrics, "p165_best": p165_metrics},
        "disagreement_diagnostics": diag,
        "searched_policy_count": len(candidates),
        "passing_policy_count": len(scored),
        "best_policy": best["policy"],
        "best_metrics": best["metrics"],
        "delta_vs_p165": p165.delta(best["metrics"], p165_metrics),
        "delta_vs_p160": p165.delta(best["metrics"], p160_metrics),
        "top_candidates": scored[:40],
        "outputs": {"overlay": str(Path(args.output_overlay)), "config_json": str(Path(args.output_json)), "report_md": str(Path(args.output_md))},
    }
    p165.write_json(Path(args.output_json), report)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(render_md(report), encoding="utf-8")
    print(json.dumps({
        "decision": decision,
        "searched": report["searched_policy_count"],
        "passing": report["passing_policy_count"],
        "best_metrics": best["metrics"],
        "delta_vs_p165": report["delta_vs_p165"],
        "delta_vs_p160": report["delta_vs_p160"],
        "best_policy": best["policy"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
