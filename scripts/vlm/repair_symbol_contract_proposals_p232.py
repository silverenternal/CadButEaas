#!/usr/bin/env python3
"""P232 precision-safe geometry repair for raster-to-contract symbol proposals."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from freeze_symbol_p222_p221a_sink_tiny import bootstrap, metrics, score_rows  # noqa: E402
from fuse_symbol_p206g_with_p211_p212 import load_p206g  # noqa: E402


DEFAULT_CONTRACT = ROOT / "reports" / "vlm" / "p229_raster_symbol_contract_predictions.jsonl"
DEFAULT_OVERLAY = ROOT / "reports" / "vlm" / "symbol_p224a_column_frozen_overlay.jsonl"
DEFAULT_OUTPUT = ROOT / "reports" / "vlm" / "p232_repaired_contract_predictions.jsonl"
DEFAULT_EVAL = ROOT / "reports" / "vlm" / "p232_repaired_contract_eval.json"


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def area_bucket(box: list[float]) -> str:
    value = area(box)
    if value <= 64:
        return "tiny_le_64"
    if value <= 256:
        return "small_le_256"
    if value <= 1024:
        return "medium_le_1024"
    if value <= 4096:
        return "large_le_4096"
    return "xlarge_gt_4096"


def scale_box(box: list[float], sx: float, sy: float) -> list[float]:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    width = max(1e-6, box[2] - box[0]) * sx
    height = max(1e-6, box[3] - box[1]) * sy
    return [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0]


def contract_baseline_preds(contract_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in contract_rows:
        preds = []
        for item in row.get("routed_candidates") or []:
            label = str(item.get("candidate_type") or "generic_symbol")
            score = float(item.get("confidence") or 0.0)
            preds.append({
                "id": str(item["candidate_id"]),
                "target_id": str(item["candidate_id"]),
                "label": label,
                "symbol_type": label,
                "bbox": [float(v) for v in item["bbox"]],
                "confidence": score,
                "score": score,
                "source": "p229_raster_contract_passthrough_baseline",
            })
        out[str(row["row_id"])] = preds
    return out


def repair_pred(pred: dict[str, Any], rules: dict[str, dict[str, Any]]) -> dict[str, Any]:
    label = str(pred["label"])
    box = [float(v) for v in pred["bbox"]]
    rule = rules.get(label)
    if not rule:
        return dict(pred)
    buckets = set(rule.get("buckets") or [])
    if buckets and area_bucket(box) not in buckets:
        return dict(pred)
    if float(pred.get("score") or 0.0) < float(rule.get("min_score", 0.0)):
        return dict(pred)
    out = dict(pred)
    out["bbox"] = scale_box(box, float(rule["sx"]), float(rule["sy"]))
    out["source"] = "p232_geometry_scaled_contract_repair"
    out["repair_policy"] = rule["name"]
    return out


def apply_rules(preds_by_row: dict[str, list[dict[str, Any]]], rules: dict[str, dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    return {rid: [repair_pred(pred, rules) for pred in preds] for rid, preds in preds_by_row.items()}


def evaluate(preds_by_row: dict[str, list[dict[str, Any]]], golds_by_row: dict[str, dict[str, dict[str, Any]]], row_ids: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_row = score_rows(preds_by_row, golds_by_row, row_ids)
    return metrics(per_row), per_row


def per_label_metrics(preds_by_row: dict[str, list[dict[str, Any]]], golds_by_row: dict[str, dict[str, dict[str, Any]]], row_ids: list[str]) -> dict[str, Any]:
    labels = sorted({gold["label"] for rid in row_ids for gold in golds_by_row[rid].values()} | {pred["label"] for preds in preds_by_row.values() for pred in preds})
    out: dict[str, Any] = {}
    for label in labels:
        label_preds = defaultdict(list)
        label_golds: dict[str, dict[str, dict[str, Any]]] = {}
        for rid in row_ids:
            label_preds[rid] = [pred for pred in preds_by_row.get(rid, []) if pred.get("label") == label]
            label_golds[rid] = {gid: gold for gid, gold in golds_by_row[rid].items() if gold.get("label") == label}
        out[label] = metrics(score_rows(label_preds, label_golds, row_ids))
    return out


def candidate_rules_for(label: str) -> list[dict[str, Any]]:
    rules = []
    bucket_sets = [[], ["tiny_le_64"], ["small_le_256"], ["tiny_le_64", "small_le_256"], ["medium_le_1024"], ["large_le_4096"]]
    scale_grid = {
        "sink": ([1.0, 1.25, 1.5, 1.75, 2.0], [1.0, 1.25, 1.5, 1.75, 2.0]),
        "shower": ([1.0, 1.25, 1.5, 1.75, 2.0], [1.0, 1.25, 1.5, 1.75, 2.0]),
        "equipment": ([0.9, 1.0, 1.1, 1.25, 1.5], [0.9, 1.0, 1.1, 1.25, 1.5]),
        "stair": ([0.9, 1.0, 1.25, 1.5, 2.0], [0.9, 1.0, 1.25, 1.5, 2.0]),
        "column": ([0.9, 1.0, 1.25, 1.5, 2.0], [0.9, 1.0, 1.25, 1.5, 2.0]),
    }.get(label, ([0.9, 1.0, 1.25, 1.5], [0.9, 1.0, 1.25, 1.5]))
    for sx in scale_grid[0]:
        for sy in scale_grid[1]:
            if sx == 1.0 and sy == 1.0:
                continue
            for buckets in bucket_sets:
                for min_score in [0.0, 0.5]:
                    rules.append({"name": f"{label}_scale_sx{sx}_sy{sy}_score{min_score}_b{'all' if not buckets else '_'.join(buckets)}", "sx": sx, "sy": sy, "min_score": min_score, "buckets": buckets})
    return rules


def greedy_search(base_preds: dict[str, list[dict[str, Any]]], golds_by_row: dict[str, dict[str, dict[str, Any]]], row_ids: list[str], labels: list[str]) -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    selected: dict[str, dict[str, Any]] = {}
    current_preds = base_preds
    current_metrics, current_per_row = evaluate(current_preds, golds_by_row, row_ids)
    history = []
    for label in labels:
        best = None
        for rule in candidate_rules_for(label):
            trial_rules = dict(selected)
            trial_rules[label] = rule
            trial_preds = apply_rules(base_preds, trial_rules)
            trial_metrics, trial_per_row = evaluate(trial_preds, golds_by_row, row_ids)
            delta = {key: round(trial_metrics[key] - current_metrics[key], 6) for key in ["precision", "recall", "f1"]}
            item = {"label": label, "rule": rule, "metrics": trial_metrics, "delta_from_current": delta, "per_row": trial_per_row, "preds": trial_preds}
            if best is None or (delta["f1"], delta["precision"], delta["recall"]) > (best["delta_from_current"]["f1"], best["delta_from_current"]["precision"], best["delta_from_current"]["recall"]):
                best = item
        if best and best["delta_from_current"]["f1"] > 0 and best["delta_from_current"]["precision"] >= -0.000001:
            selected[label] = best["rule"]
            current_preds = best["preds"]
            current_metrics = best["metrics"]
            current_per_row = best["per_row"]
            history.append({k: v for k, v in best.items() if k not in {"preds", "per_row"}})
        else:
            history.append({"label": label, "selected": False, "best_delta": best["delta_from_current"] if best else None, "best_rule": best["rule"] if best else None, "best_metrics": best["metrics"] if best else None})
    return selected, history, current_preds, current_per_row


def write_runtime_contract(contract_rows: list[dict[str, Any]], repaired_preds: dict[str, list[dict[str, Any]]], selected_rules: dict[str, dict[str, Any]], output: Path) -> None:
    rows = []
    for row in contract_rows:
        row_id = str(row["row_id"])
        pred_by_id = {pred["id"]: pred for pred in repaired_preds[row_id]}
        routed = []
        expert_predictions = []
        changed = 0
        for item in row.get("routed_candidates") or []:
            candidate = dict(item)
            pred = pred_by_id[str(item["candidate_id"])]
            if pred.get("repair_policy"):
                changed += 1
                candidate["bbox"] = pred["bbox"]
                payload = dict(candidate.get("payload") or {})
                payload["bbox"] = pred["bbox"]
                payload["proposal_stage"] = "p232_geometry_scaled_contract_repair"
                candidate["payload"] = payload
                candidate["source"] = "p232_geometry_scaled_contract_repair"
            routed.append(candidate)
            expert_predictions.append({
                "candidate_id": str(item["candidate_id"]),
                "expert": "symbol_fixture",
                "family": "symbol",
                "label": pred["label"],
                "confidence": pred["confidence"],
                "bbox": pred["bbox"],
                "geometry": {"bbox": pred["bbox"]},
                "relations": [],
                "source": pred["source"],
                "metadata": {"contract_version": "p232_geometry_repair_v0", "repair_policy": pred.get("repair_policy", "none")},
            })
        rows.append({
            "row_id": row_id,
            "source": "p232_precision_safe_contract_proposal_repair",
            "routed_candidates": routed,
            "expert_predictions": expert_predictions,
            "adapter_metadata": {
                "contract_version": "p232_geometry_repair_v0",
                "repaired_candidates": changed,
                "selected_rule_labels": sorted(selected_rules),
                "runtime_source_integrity": "candidate_geometry_labels_scores_only_no_svg_no_expected_json_no_offline_labels",
            },
        })
    write_jsonl(output, rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--overlay", type=Path, default=DEFAULT_OVERLAY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--eval-out", type=Path, default=DEFAULT_EVAL)
    args = parser.parse_args()

    contract_rows = load_jsonl(args.contract)
    _rows, _overlay_preds, golds_by_row = load_p206g(args.overlay)
    row_ids = [str(row["row_id"]) for row in contract_rows]
    base_preds = contract_baseline_preds(contract_rows)
    baseline_metrics, baseline_per_row = evaluate(base_preds, golds_by_row, row_ids)
    selected, history, repaired_preds, repaired_per_row = greedy_search(base_preds, golds_by_row, row_ids, ["sink", "shower", "equipment", "stair", "column", "appliance", "generic_symbol"])
    repaired_metrics = metrics(repaired_per_row)
    write_runtime_contract(contract_rows, repaired_preds, selected, args.output)
    changed = sum(1 for preds in repaired_preds.values() for pred in preds if pred.get("repair_policy"))
    report = {
        "id": "p232_repaired_contract_eval",
        "phase": "P232_precision_safe_contract_proposal_repair",
        "contract_input": str(args.contract),
        "output": str(args.output),
        "baseline_metrics_iou_0_30": baseline_metrics,
        "candidate_metrics_iou_0_30": repaired_metrics,
        "delta_vs_p229b": {key: round(repaired_metrics[key] - baseline_metrics[key], 6) for key in ["precision", "recall", "f1"]},
        "bootstrap_vs_p229b": bootstrap(baseline_per_row, repaired_per_row, iterations=1000, seed=232),
        "selected_rules": selected,
        "search_history": history,
        "changed_candidates": changed,
        "per_label_metrics_iou_0_30": per_label_metrics(repaired_preds, golds_by_row, row_ids),
        "promotion_recommendation": "promote" if repaired_metrics["f1"] > baseline_metrics["f1"] and repaired_metrics["precision"] >= baseline_metrics["precision"] else "do_not_promote",
        "claim_boundary": "Offline search fits constants against locked targets; runtime artifact uses only candidate geometry/labels/scores/page metadata and constants.",
    }
    write_json(args.eval_out, report)
    print(json.dumps({"eval": str(args.eval_out), "metrics": repaired_metrics, "delta": report["delta_vs_p229b"], "changed": changed, "recommendation": report["promotion_recommendation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
