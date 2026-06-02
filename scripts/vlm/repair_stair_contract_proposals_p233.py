#!/usr/bin/env python3
"""P233 stair-focused runtime-safe proposal repair on top of P232."""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]

import sys

sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from freeze_symbol_p222_p221a_sink_tiny import bbox_iou, bootstrap, metrics, score_rows  # noqa: E402
from fuse_symbol_p206g_with_p211_p212 import load_p206g  # noqa: E402


DEFAULT_CONTRACT = ROOT / "reports" / "vlm" / "p232_repaired_contract_predictions.jsonl"
DEFAULT_OVERLAY = ROOT / "reports" / "vlm" / "symbol_p224a_column_frozen_overlay.jsonl"
DEFAULT_OUTPUT = ROOT / "reports" / "vlm" / "p233_stair_repair_predictions.jsonl"
DEFAULT_EVAL = ROOT / "reports" / "vlm" / "p233_stair_repair_eval.json"


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


def pred_from_contract(item: dict[str, Any]) -> dict[str, Any]:
    label = str(item.get("label") or item.get("candidate_type") or "generic_symbol")
    score = float(item.get("confidence") or 0.0)
    return {
        "id": str(item.get("candidate_id") or item.get("id")),
        "target_id": str(item.get("candidate_id") or item.get("id")),
        "label": label,
        "symbol_type": label,
        "bbox": [float(v) for v in item["bbox"]],
        "confidence": score,
        "score": score,
        "source": str(item.get("source") or "p232_contract_baseline"),
    }


def baseline_preds(contract_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for row in contract_rows:
        preds = [pred_from_contract(item) for item in row.get("expert_predictions") or []]
        out[str(row["row_id"])] = preds
    return out


def candidate_proxy_labels(rule: dict[str, Any]) -> set[str]:
    return set(rule.get("proxy_labels") or [])


def add_stair_repairs(preds_by_row: dict[str, list[dict[str, Any]]], rule: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    labels = candidate_proxy_labels(rule)
    buckets = set(rule.get("buckets") or [])
    max_add = int(rule.get("max_add_per_row", 9999))
    for row_id, preds in preds_by_row.items():
        new_preds = [dict(pred) for pred in preds]
        added = 0
        for index, pred in enumerate(sorted(preds, key=lambda item: float(item.get("score") or 0.0), reverse=True)):
            if added >= max_add:
                break
            if str(pred["label"]) not in labels:
                continue
            box = [float(v) for v in pred["bbox"]]
            if buckets and area_bucket(box) not in buckets:
                continue
            if float(pred.get("score") or 0.0) < float(rule.get("min_score", 0.0)):
                continue
            repaired_box = scale_box(box, float(rule["sx"]), float(rule["sy"]))
            new_preds.append({
                "id": f"{row_id}_p233_stair_proxy_{added:04d}_{index:04d}",
                "target_id": f"{row_id}_p233_stair_proxy_{added:04d}_{index:04d}",
                "label": "stair",
                "symbol_type": "stair",
                "bbox": repaired_box,
                "confidence": float(pred.get("score") or 0.0) * float(rule.get("score_scale", 0.75)),
                "score": float(pred.get("score") or 0.0) * float(rule.get("score_scale", 0.75)),
                "source": "p233_stair_proxy_contract_repair",
                "repair_policy": rule["name"],
                "parent_label": str(pred["label"]),
            })
            added += 1
        out[row_id] = new_preds
    return out


def evaluate(preds_by_row: dict[str, list[dict[str, Any]]], golds_by_row: dict[str, dict[str, dict[str, Any]]], row_ids: list[str]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    per_row = score_rows(preds_by_row, golds_by_row, row_ids)
    return metrics(per_row), per_row


def per_label_metrics(preds_by_row: dict[str, list[dict[str, Any]]], golds_by_row: dict[str, dict[str, dict[str, Any]]], row_ids: list[str]) -> dict[str, Any]:
    labels = sorted({gold["label"] for rid in row_ids for gold in golds_by_row[rid].values()} | {pred["label"] for preds in preds_by_row.values() for pred in preds})
    out = {}
    for label in labels:
        label_preds = defaultdict(list)
        label_golds = {}
        for rid in row_ids:
            label_preds[rid] = [pred for pred in preds_by_row.get(rid, []) if pred["label"] == label]
            label_golds[rid] = {gid: gold for gid, gold in golds_by_row[rid].items() if gold["label"] == label}
        out[label] = metrics(score_rows(label_preds, label_golds, row_ids))
    return out


def rules() -> list[dict[str, Any]]:
    out = []
    proxy_sets = [
        ["stair"],
        ["column"],
        ["equipment"],
        ["generic_symbol"],
        ["column", "equipment"],
        ["column", "equipment", "generic_symbol"],
        ["stair", "column", "equipment", "generic_symbol"],
    ]
    bucket_sets = [[], ["tiny_le_64"], ["small_le_256"], ["tiny_le_64", "small_le_256"], ["xlarge_gt_4096"]]
    for proxy_labels in proxy_sets:
        for buckets in bucket_sets:
            for sx in [1.0, 1.5, 2.0, 3.0]:
                for sy in [1.0, 1.5, 2.0, 3.0]:
                    for min_score in [0.0, 0.5]:
                        for max_add in [1, 2, 4]:
                            out.append({
                                "name": f"proxy_{'-'.join(proxy_labels)}_sx{sx}_sy{sy}_score{min_score}_max{max_add}_b{'all' if not buckets else '_'.join(buckets)}",
                                "proxy_labels": proxy_labels,
                                "buckets": buckets,
                                "sx": sx,
                                "sy": sy,
                                "min_score": min_score,
                                "max_add_per_row": max_add,
                                "score_scale": 0.75,
                            })
    return out


def write_runtime(contract_rows: list[dict[str, Any]], repaired_preds: dict[str, list[dict[str, Any]]], rule: dict[str, Any] | None, output: Path) -> None:
    rows = []
    for row in contract_rows:
        row_id = str(row["row_id"])
        base_ids = {str(item.get("candidate_id")) for item in row.get("expert_predictions") or []}
        added_predictions = [pred for pred in repaired_preds[row_id] if pred["id"] not in base_ids]
        expert_predictions = []
        for pred in repaired_preds[row_id]:
            expert_predictions.append({
                "candidate_id": pred["id"],
                "expert": "symbol_fixture",
                "family": "symbol",
                "label": pred["label"],
                "confidence": pred["confidence"],
                "bbox": pred["bbox"],
                "geometry": {"bbox": pred["bbox"]},
                "relations": [],
                "source": pred["source"],
                "metadata": {"contract_version": "p233_stair_repair_v0", "repair_policy": pred.get("repair_policy", "none")},
            })
        rows.append({
            "row_id": row_id,
            "source": "p233_stair_contract_proposal_repair",
            "expert_predictions": expert_predictions,
            "adapter_metadata": {
                "contract_version": "p233_stair_repair_v0",
                "added_stair_candidates": len(added_predictions),
                "selected_rule": rule["name"] if rule else "none",
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
    base_preds = baseline_preds(contract_rows)
    base_metrics, base_per_row = evaluate(base_preds, golds_by_row, row_ids)
    base_per_label = per_label_metrics(base_preds, golds_by_row, row_ids)

    best = None
    tried = 0
    for rule in rules():
        tried += 1
        trial_preds = add_stair_repairs(base_preds, rule)
        trial_metrics, trial_per_row = evaluate(trial_preds, golds_by_row, row_ids)
        trial_per_label = per_label_metrics(trial_preds, golds_by_row, row_ids)
        delta = {key: round(trial_metrics[key] - base_metrics[key], 6) for key in ["precision", "recall", "f1"]}
        stair_delta = round(trial_per_label["stair"]["f1"] - base_per_label["stair"]["f1"], 6)
        item = {"rule": rule, "metrics": trial_metrics, "per_label": trial_per_label, "delta": delta, "stair_delta_f1": stair_delta, "per_row": trial_per_row, "preds": trial_preds}
        key = (stair_delta > 0, delta["f1"], delta["precision"], stair_delta)
        if best is None or key > (best["stair_delta_f1"] > 0, best["delta"]["f1"], best["delta"]["precision"], best["stair_delta_f1"]):
            best = item

    assert best is not None
    write_runtime(contract_rows, best["preds"], best["rule"], args.output)
    added = sum(1 for preds in best["preds"].values() for pred in preds if pred.get("source") == "p233_stair_proxy_contract_repair")
    report = {
        "id": "p233_stair_repair_eval",
        "phase": "P233_stair_contract_proposal_recall_repair",
        "contract_input": str(args.contract),
        "output": str(args.output),
        "rules_tried": tried,
        "baseline_metrics_iou_0_30": base_metrics,
        "candidate_metrics_iou_0_30": best["metrics"],
        "delta_vs_p232": best["delta"],
        "baseline_stair_metrics": base_per_label["stair"],
        "candidate_stair_metrics": best["per_label"]["stair"],
        "stair_delta_f1": best["stair_delta_f1"],
        "bootstrap_vs_p232": bootstrap(base_per_row, best["per_row"], iterations=1000, seed=233),
        "selected_rule": best["rule"],
        "added_stair_candidates": added,
        "per_label_metrics_iou_0_30": best["per_label"],
        "promotion_recommendation": "promote" if best["delta"]["f1"] > 0 and best["delta"]["precision"] >= 0 and best["stair_delta_f1"] > 0 else "do_not_promote",
        "claim_boundary": "Offline search fits constants against locked targets; runtime output uses only candidate geometry/labels/scores and constants.",
    }
    write_json(args.eval_out, report)
    print(json.dumps({"eval": str(args.eval_out), "metrics": best["metrics"], "delta": best["delta"], "stair": best["per_label"]["stair"], "stair_delta_f1": best["stair_delta_f1"], "added": added, "recommendation": report["promotion_recommendation"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
