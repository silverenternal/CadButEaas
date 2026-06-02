#!/usr/bin/env python3
"""P159 support-aware duplicate merge rescue for symbol boxes.

Uses raw detector candidates as runtime support around each P157 prediction and
keeps the P157 candidate count fixed. Gold targets are evaluation-only.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
RAW_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p084/v28_frozen_detector_baseline.jsonl"
P157_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p157_best.jsonl"
OUT_JSON = ROOT / "configs/vlm/symbol_support_aware_merge_rescue_p159.json"
OUT_MD = ROOT / "reports/vlm/symbol_support_aware_merge_rescue_p159.md"
OUT_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p159_best.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def bbox4(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in value[:4]]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / max(area(a) + area(b) - inter, 1e-9)


def center_distance(a: list[float], b: list[float]) -> float:
    acx, acy = (a[0] + a[2]) / 2.0, (a[1] + a[3]) / 2.0
    bcx, bcy = (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
    return ((acx - bcx) ** 2 + (acy - bcy) ** 2) ** 0.5


def center_covered(pred: list[float], gold: list[float]) -> bool:
    cx = (gold[0] + gold[2]) / 2.0
    cy = (gold[1] + gold[3]) / 2.0
    return pred[0] <= cx <= pred[2] and pred[1] <= cy <= pred[3]


def label(item: dict[str, Any]) -> str:
    return str(item.get("symbol_type") or item.get("label") or "generic_symbol")


def score(item: dict[str, Any]) -> float:
    value = item.get("confidence") if item.get("confidence") is not None else item.get("score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def bucket(box: list[float]) -> str:
    value = area(box)
    if value <= 64:
        return "tiny"
    if value <= 256:
        return "small"
    if value <= 1024:
        return "medium"
    if value <= 4096:
        return "large"
    return "xlarge"


def target_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate((row.get("targets") or {}).get("symbol") or []):
        box = bbox4(item.get("bbox"))
        if box is not None:
            out.append({"id": str(item.get("target_id") or idx), "bbox": box, "bucket": bucket(box)})
    return out


def normalized_items(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for raw in items:
        box = bbox4(raw.get("bbox"))
        if box is None:
            continue
        out.append({"bbox": box, "label": label(raw), "score": score(raw), "raw": raw})
    return out


def support_cluster(pred: dict[str, Any], raw_items: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    pred_box = pred["bbox"]
    pred_label = pred["label"]
    max_dist = float(policy["max_center_dist"])
    min_iou = float(policy["min_iou"])
    same_label_only = bool(policy["same_label_only"])
    min_score = float(policy["min_support_score"])
    cluster = []
    for raw in raw_items:
        if raw["score"] < min_score:
            continue
        if same_label_only and raw["label"] != pred_label:
            continue
        if center_distance(pred_box, raw["bbox"]) <= max_dist or iou(pred_box, raw["bbox"]) >= min_iou:
            cluster.append(raw)
    cluster.sort(key=lambda item: item["score"], reverse=True)
    return cluster[: int(policy["max_support"])]


def merge_box(pred: dict[str, Any], cluster: list[dict[str, Any]], policy: dict[str, Any]) -> list[float]:
    if len(cluster) < int(policy["min_support"]):
        return pred["bbox"]
    score_power = float(policy["score_power"])
    pred_weight = float(policy["pred_weight"])
    boxes = [(pred["bbox"], pred_weight)]
    for item in cluster:
        boxes.append((item["bbox"], max(item["score"], 1e-4) ** score_power))
    total = sum(w for _box, w in boxes) or 1.0
    return [sum(box[i] * weight for box, weight in boxes) / total for i in range(4)]


def transformed_preds(p157_by_row: dict[str, list[dict[str, Any]]], raw_by_row: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for row_id, preds in p157_by_row.items():
        raw_items = raw_by_row.get(row_id, [])
        merged = []
        for pred in preds:
            item = copy.deepcopy(pred)
            cluster = support_cluster(pred, raw_items, policy)
            item["bbox"] = merge_box(pred, cluster, policy)
            item["support_count"] = len(cluster)
            merged.append(item)
        out[row_id] = merged
    return out


def evaluate(golds_by_row: dict[str, list[dict[str, Any]]], preds_by_row: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    totals = Counter()
    by_area_gold = Counter()
    by_area_tp = Counter()
    by_area_center = Counter()
    support_counts = Counter()
    for row_id, golds in golds_by_row.items():
        preds = preds_by_row.get(row_id, [])
        totals["gold"] += len(golds)
        totals["pred"] += len(preds)
        for pred in preds:
            support_counts[min(int(pred.get("support_count", 0)), 5)] += 1
        used_iou: set[int] = set()
        used_center: set[int] = set()
        for gold in golds:
            by_area_gold[gold["bucket"]] += 1
            best_idx = None
            best_iou = 0.0
            center_idx = None
            for idx, pred in enumerate(preds):
                overlap = iou(pred["bbox"], gold["bbox"])
                if idx not in used_iou and overlap > best_iou:
                    best_iou = overlap
                    best_idx = idx
                if center_idx is None and idx not in used_center and center_covered(pred["bbox"], gold["bbox"]):
                    center_idx = idx
            if best_idx is not None and best_iou >= 0.30:
                used_iou.add(best_idx)
                totals["tp"] += 1
                by_area_tp[gold["bucket"]] += 1
            if center_idx is not None:
                used_center.add(center_idx)
                totals["center"] += 1
                by_area_center[gold["bucket"]] += 1
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
        "by_area_iou_recall": {key: round(by_area_tp[key] / max(by_area_gold[key], 1), 6) for key in sorted(by_area_gold)},
        "by_area_center_recall": {key: round(by_area_center[key] / max(by_area_gold[key], 1), 6) for key in sorted(by_area_gold)},
        "support_count_histogram": dict(sorted(support_counts.items())),
    }


def candidate_policies() -> list[dict[str, Any]]:
    policies = []
    # Very small tactical grid to keep SSHFS runtime sane. Test whether support
    # merge direction helps at all before expanding.
    for max_center_dist in [4.0, 8.0]:
        for min_iou in [0.30, 0.40]:
            for min_support_score in [0.10, 0.20]:
                for max_support in [3]:
                    for min_support in [2]:
                        for score_power in [1.0]:
                            for pred_weight in [2.0, 4.0]:
                                policies.append({
                                    "name": f"p159_same1_d{max_center_dist}_iou{min_iou}_s{min_support_score}_k{max_support}_m{min_support}_pow{score_power}_pw{pred_weight}",
                                    "same_label_only": True,
                                    "max_center_dist": max_center_dist,
                                    "min_iou": min_iou,
                                    "min_support_score": min_support_score,
                                    "max_support": max_support,
                                    "min_support": min_support,
                                    "score_power": score_power,
                                    "pred_weight": pred_weight,
                                })
    # Include no-op as safety candidate.
    policies.append({"name":"p159_noop", "same_label_only": True, "max_center_dist": 0.0, "min_iou": 2.0, "min_support_score": 1.1, "max_support": 1, "min_support": 99, "score_power": 1.0, "pred_weight": 1.0})
    return policies

def materialize(base_rows: list[dict[str, Any]], merged_by_row: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for raw in base_rows:
        row = copy.deepcopy(raw)
        row_id = str(row.get("row_id") or row.get("id"))
        candidates = []
        for idx, pred in enumerate(merged_by_row.get(row_id, [])):
            item = copy.deepcopy(pred["raw"])
            item["bbox"] = pred["bbox"]
            item["id"] = f"{row_id}_p159_best_symbol_{idx:05d}"
            item["target_id"] = item["id"]
            item["source"] = "symbol_policy_overlay_p159_best"
            item.setdefault("metadata", {})["p159_support_merge"] = policy["name"]
            item.setdefault("metadata", {})["p159_support_count"] = pred.get("support_count", 0)
            candidates.append(item)
        row["symbol_candidates"] = candidates
        if isinstance(row.get("expected_json"), dict):
            row["expected_json"]["symbol_candidates"] = [copy.deepcopy(item) for item in candidates]
        row["symbol_policy_overlay"] = {"policy_id": "p159_best", "description": "P159 support-aware duplicate merge candidate.", "policy": policy}
        rows.append(row)
    return rows


def delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    return {key: round(float(a[key]) - float(b[key]), 6) for key in ["precision", "recall", "f1", "center_recall", "prediction_inflation"]}


def render_md(report: dict[str, Any]) -> str:
    lines = ["# P159 Symbol Support-Aware Duplicate Merge Rescue", "", f"Decision: **{report['decision']}**", "", "## Metrics", "", "| Policy | Precision | Recall | F1 | Center | Inflation |", "|---|---:|---:|---:|---:|---:|"]
    for name, metrics in report["baseline_metrics"].items():
        lines.append(f"| `{name}` | {metrics['precision']:.6f} | {metrics['recall']:.6f} | {metrics['f1']:.6f} | {metrics['center_recall']:.6f} | {metrics['prediction_inflation']:.6f} |")
    best = report["best_metrics"]
    lines.append(f"| `p159_best` | {best['precision']:.6f} | {best['recall']:.6f} | {best['f1']:.6f} | {best['center_recall']:.6f} | {best['prediction_inflation']:.6f} |")
    lines.extend(["", "## Best Policy", "", f"- `{report['best_policy']['name']}`", f"- config: `{json.dumps(report['best_policy'], ensure_ascii=False)}`", "", "## Deltas", "", f"- vs `p157_best`: `{json.dumps(report['delta_vs_p157'], ensure_ascii=False)}`", "", "## Artifacts", ""])
    for value in report["outputs"].values():
        lines.append(f"- `{value}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw-overlay", default=str(RAW_OVERLAY))
    parser.add_argument("--p157-overlay", default=str(P157_OVERLAY))
    parser.add_argument("--output-json", default=str(OUT_JSON))
    parser.add_argument("--output-md", default=str(OUT_MD))
    parser.add_argument("--output-overlay", default=str(OUT_OVERLAY))
    args = parser.parse_args()

    raw_rows = load_jsonl(Path(args.raw_overlay))
    p157_rows = load_jsonl(Path(args.p157_overlay))
    raw_by_row = {str(row.get("row_id") or row.get("id")): normalized_items(row.get("symbol_candidates") or []) for row in raw_rows}
    p157_by_row = {str(row.get("row_id") or row.get("id")): normalized_items(row.get("symbol_candidates") or []) for row in p157_rows}
    golds_by_row = {str(row.get("row_id") or row.get("id")): target_symbols(row) for row in p157_rows}
    p157_metrics = evaluate(golds_by_row, p157_by_row)

    scored = []
    for policy in candidate_policies():
        merged = transformed_preds(p157_by_row, raw_by_row, policy)
        metrics = evaluate(golds_by_row, merged)
        if metrics["precision"] >= p157_metrics["precision"] - 0.003 and metrics["prediction_inflation"] <= p157_metrics["prediction_inflation"] + 1e-9:
            scored.append({"policy": policy, "metrics": metrics})
    scored.sort(key=lambda row: (row["metrics"]["f1"], row["metrics"]["recall"], row["metrics"]["precision"]), reverse=True)
    best = scored[0]
    best_merged = transformed_preds(p157_by_row, raw_by_row, best["policy"])
    write_jsonl(Path(args.output_overlay), materialize(p157_rows, best_merged, best["policy"]))
    decision = "positive_adopt_p159" if best["metrics"]["f1"] > p157_metrics["f1"] else "negative_keep_p157"
    report = {
        "id": "SCI-P2-159-symbol-support-aware-duplicate-merge-rescue",
        "created_on": "2026-05-17",
        "decision": decision,
        "claim_boundary": "Post-hoc support-merge sweep on 74-row public-raster overlay subset. Runtime policy uses raw detector candidates and P157 predictions only; gold targets are evaluation-only.",
        "baseline_metrics": {"p157_best": p157_metrics},
        "searched_policy_count": len(candidate_policies()),
        "passing_policy_count": len(scored),
        "best_policy": best["policy"],
        "best_metrics": best["metrics"],
        "delta_vs_p157": delta(best["metrics"], p157_metrics),
        "top_candidates": scored[:30],
        "outputs": {"overlay": str(Path(args.output_overlay)), "config_json": str(Path(args.output_json)), "report_md": str(Path(args.output_md))},
    }
    write_json(Path(args.output_json), report)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(render_md(report), encoding="utf-8")
    print(json.dumps({"decision": decision, "searched": report["searched_policy_count"], "passing": report["passing_policy_count"], "best_metrics": best["metrics"], "delta_vs_p157": report["delta_vs_p157"], "best_policy": best["policy"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
