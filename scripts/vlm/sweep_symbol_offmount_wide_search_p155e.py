#!/usr/bin/env python3
"""P155E bounded threshold/NMS/backfill search for symbol recovery.

This is an offline sweep: targets are used only to score candidate policies. The
materialized candidate policies themselves use only predicted boxes, labels,
scores, and fixed thresholds/NMS.
"""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
SOURCE_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p084/v28_frozen_detector_baseline.jsonl"
P155A_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p155a_p140_best.jsonl"
OUT_JSON = ROOT / "configs/vlm/symbol_offmount_wide_search_p155e.json"
OUT_MD = ROOT / "reports/vlm/symbol_offmount_wide_search_p155e.md"
OUT_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p155e_best.jsonl"

P140_THRESHOLDS = {
    "*": 0.02,
    "appliance": 0.32,
    "bathtub": 0.32,
    "column": 0.32,
    "equipment": 0.24,
    "generic_symbol": 0.45,
    "shower": 0.24,
    "sink": 0.24,
    "stair": 0.24,
}


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


def center_covered(pred: list[float], gold: list[float]) -> bool:
    cx = (gold[0] + gold[2]) / 2.0
    cy = (gold[1] + gold[3]) / 2.0
    return pred[0] <= cx <= pred[2] and pred[1] <= cy <= pred[3]


def target_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate((row.get("targets") or {}).get("symbol") or []):
        box = bbox4(item.get("bbox"))
        if box is not None:
            out.append({"id": str(item.get("target_id") or idx), "bbox": box})
    return out


def symbol_label(item: dict[str, Any]) -> str:
    return str(item.get("symbol_type") or item.get("label") or "generic_symbol")


def symbol_score(item: dict[str, Any]) -> float:
    value = item.get("confidence") if item.get("confidence") is not None else item.get("score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def nms(items: list[dict[str, Any]], same_iou: float, any_iou: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in sorted(items, key=symbol_score, reverse=True):
        box = bbox4(item.get("bbox"))
        if box is None:
            continue
        label = symbol_label(item)
        suppress = False
        for old in kept:
            old_box = bbox4(old.get("bbox"))
            if old_box is None:
                continue
            overlap = iou(box, old_box)
            if label == symbol_label(old) and overlap >= same_iou:
                suppress = True
                break
            if overlap >= any_iou:
                suppress = True
                break
        if not suppress:
            kept.append(item)
    return kept


def apply_policy(raw_items: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    thresholds = policy["thresholds"]
    selected = [item for item in raw_items if symbol_score(item) >= thresholds.get(symbol_label(item), thresholds.get("*", 0.0))]
    pre_nms_keep = int(policy.get("pre_nms_keep") or 180)
    selected = sorted(selected, key=symbol_score, reverse=True)[:pre_nms_keep]
    selected = nms(selected, policy["same_label_nms_iou"], policy["any_label_nms_iou"])
    max_label_keep = policy.get("max_label_keep")
    if max_label_keep is not None:
        by_label: Counter[str] = Counter()
        limited = []
        for item in sorted(selected, key=symbol_score, reverse=True):
            label = symbol_label(item)
            if by_label[label] < max_label_keep:
                by_label[label] += 1
                limited.append(item)
        selected = limited
    max_page_keep = policy.get("max_page_keep")
    if max_page_keep is not None:
        selected = sorted(selected, key=symbol_score, reverse=True)[: int(max_page_keep)]
    return selected


def score_rows(golds_by_row: dict[str, list[dict[str, Any]]], preds_by_row: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    totals = Counter()
    for row_id, golds in golds_by_row.items():
        preds = preds_by_row.get(row_id, [])
        totals["gold"] += len(golds)
        totals["pred"] += len(preds)
        used_iou: set[int] = set()
        used_center: set[int] = set()
        for gold in golds:
            best_idx = None
            best_iou = 0.0
            center_idx = None
            for idx, pred in enumerate(preds):
                box = bbox4(pred.get("bbox"))
                if box is None:
                    continue
                overlap = iou(box, gold["bbox"])
                if idx not in used_iou and overlap > best_iou:
                    best_iou = overlap
                    best_idx = idx
                if center_idx is None and idx not in used_center and center_covered(box, gold["bbox"]):
                    center_idx = idx
            if best_idx is not None and best_iou >= 0.30:
                used_iou.add(best_idx)
                totals["tp"] += 1
            if center_idx is not None:
                used_center.add(center_idx)
                totals["center"] += 1
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


def candidate_policies() -> list[dict[str, Any]]:
    policies = []
    # Compact two-stage-inspired grid: keep the P140 structure, search only the
    # low-precision labels identified by P155C/P155D plus a small page/NMS band.
    shower_values = [0.30, 0.32, 0.34, 0.36]
    stair_values = [0.32, 0.36, 0.40]
    column_values = [0.36, 0.40]
    generic_values = [0.45]
    page_values = [80, 90, 100]
    nms_pairs = [(0.45, 0.75), (0.50, 0.80)]
    label_keeps = [None, 60]
    for shower in shower_values:
        for stair in stair_values:
            for column in column_values:
                for generic in generic_values:
                    for max_page_keep in page_values:
                        for same_iou, any_iou in nms_pairs:
                            for max_label_keep in label_keeps:
                                thresholds = dict(P140_THRESHOLDS)
                                thresholds.update({"shower": shower, "stair": stair, "column": column, "generic_symbol": generic})
                                policies.append({
                                    "name": f"p155e_sh{shower}_st{stair}_col{column}_gen{generic}_p{max_page_keep}_same{same_iou}_any{any_iou}_l{max_label_keep}",
                                    "thresholds": thresholds,
                                    "max_page_keep": max_page_keep,
                                    "max_label_keep": max_label_keep,
                                    "same_label_nms_iou": same_iou,
                                    "any_label_nms_iou": any_iou,
                                    "pre_nms_keep": 180,
                                })
    return policies

def materialize_overlay(base_rows: list[dict[str, Any]], preds_by_row: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for raw in base_rows:
        row = copy.deepcopy(raw)
        row_id = str(row.get("row_id") or row.get("id"))
        candidates = []
        for idx, item in enumerate(preds_by_row.get(row_id, [])):
            candidate = copy.deepcopy(item)
            candidate["id"] = f"{row_id}_p155e_best_symbol_{idx:05d}"
            candidate["target_id"] = candidate["id"]
            candidate["source"] = "symbol_policy_overlay_p155e_best"
            candidate.setdefault("metadata", {})["source_policy"] = policy["name"]
            candidate.setdefault("metadata", {})["p155e_materialized"] = True
            candidates.append(candidate)
        row["symbol_candidates"] = candidates
        if isinstance(row.get("expected_json"), dict):
            row["expected_json"]["symbol_candidates"] = [copy.deepcopy(item) for item in candidates]
        row["symbol_policy_overlay"] = {
            "policy_id": "p155e_best",
            "description": "P155E bounded off-mount threshold/NMS recall-recovery candidate.",
            "policy": policy,
        }
        rows.append(row)
    return rows


def render_md(report: dict[str, Any]) -> str:
    lines = [
        "# P155E Off-Mount Wide Threshold/NMS Recall Recovery",
        "",
        f"Decision: **{report['decision']}**",
        "",
        "## Scope",
        "",
        report["claim_boundary"],
        "",
        "## Best Candidates",
        "",
        "| Rank | Policy | Precision | Recall | F1 | Center | Inflation |",
        "|---:|---|---:|---:|---:|---:|---:|",
    ]
    for idx, row in enumerate(report["top_candidates"][:10], 1):
        m = row["metrics"]
        lines.append(f"| {idx} | `{row['policy']['name']}` | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['center_recall']:.6f} | {m['prediction_inflation']:.6f} |")
    lines.extend([
        "",
        "## Recommendation",
        "",
        f"- Best candidate: `{report['best_policy']['name']}`",
        f"- Compared with P155D cleanup, F1 delta is `{report['delta_vs_p155d']['f1']:+.6f}` and recall delta is `{report['delta_vs_p155d']['recall']:+.6f}`.",
        f"- Compared with P155A/P140 rerun, F1 delta is `{report['delta_vs_p155a']['f1']:+.6f}` and precision delta is `{report['delta_vs_p155a']['precision']:+.6f}`.",
        "",
        "## Artifacts",
        "",
        f"- `{report['outputs']['overlay']}`",
        f"- `{report['outputs']['config_json']}`",
        f"- `{report['outputs']['report_md']}`",
        "",
    ])
    return "\n".join(lines)


def delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    return {key: round(float(a[key]) - float(b[key]), 6) for key in ["precision", "recall", "f1", "center_recall", "prediction_inflation"]}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-overlay", default=str(SOURCE_OVERLAY))
    parser.add_argument("--base-overlay", default=str(P155A_OVERLAY))
    parser.add_argument("--output-json", default=str(OUT_JSON))
    parser.add_argument("--output-md", default=str(OUT_MD))
    parser.add_argument("--output-overlay", default=str(OUT_OVERLAY))
    args = parser.parse_args()

    source_rows = load_jsonl(Path(args.source_overlay))
    base_rows = load_jsonl(Path(args.base_overlay))
    raw_by_row = {str(row.get("row_id") or row.get("id")): list(row.get("symbol_candidates") or []) for row in source_rows}
    golds_by_row = {str(row.get("row_id") or row.get("id")): target_symbols(row) for row in base_rows}

    p155a_preds = {str(row.get("row_id") or row.get("id")): list(row.get("symbol_candidates") or []) for row in base_rows}
    p155a_metrics = score_rows(golds_by_row, p155a_preds)
    p155d_policy = {
        "name": "p155d_label_cleanup_replay",
        "thresholds": {**P140_THRESHOLDS, "shower": 0.34, "stair": 0.40, "column": 0.40, "bathtub": 0.02},
        "max_page_keep": 90,
        "max_label_keep": None,
        "same_label_nms_iou": 0.45,
        "any_label_nms_iou": 0.75,
    }
    p155d_preds = {row_id: apply_policy(items, p155d_policy) for row_id, items in raw_by_row.items() if row_id in golds_by_row}
    p155d_metrics = score_rows(golds_by_row, p155d_preds)

    scored = []
    for policy in candidate_policies():
        preds = {row_id: apply_policy(raw_by_row[row_id], policy) for row_id in golds_by_row}
        metrics = score_rows(golds_by_row, preds)
        if metrics["precision"] >= 0.53 and metrics["prediction_inflation"] <= 1.08:
            scored.append({"policy": policy, "metrics": metrics})
    scored.sort(key=lambda row: (row["metrics"]["f1"], row["metrics"]["recall"], row["metrics"]["precision"]), reverse=True)
    best = scored[0] if scored else None
    if best is None:
        raise SystemExit("no candidate passed filters")
    best_preds = {row_id: apply_policy(raw_by_row[row_id], best["policy"]) for row_id in golds_by_row}
    overlay_rows = materialize_overlay(base_rows, best_preds, best["policy"])
    write_jsonl(Path(args.output_overlay), overlay_rows)

    decision = "positive_small_recall_recovery_over_p155d" if best["metrics"]["f1"] > p155d_metrics["f1"] else "negative_keep_p155d_cleanup"
    report = {
        "id": "SCI-P2-155E-offmount-wide-threshold-nms-recall-recovery",
        "created_on": "2026-05-17",
        "decision": decision,
        "claim_boundary": "Bounded post-hoc threshold/NMS sweep on the same 74-row public-raster overlay subset. Runtime policy uses only predicted bbox/label/score; gold targets are evaluation-only.",
        "source_integrity": "No SVG/parser geometry or gold labels are used by materialized runtime policy. Offline targets are used only for sweep scoring.",
        "inputs": {"source_overlay": str(Path(args.source_overlay)), "base_overlay": str(Path(args.base_overlay))},
        "baseline_metrics": {"p155a_p140_rerun": p155a_metrics, "p155d_label_cleanup_replay": p155d_metrics},
        "best_policy": best["policy"],
        "best_metrics": best["metrics"],
        "delta_vs_p155d": delta(best["metrics"], p155d_metrics),
        "delta_vs_p155a": delta(best["metrics"], p155a_metrics),
        "searched_policy_count": len(candidate_policies()),
        "passing_policy_count": len(scored),
        "top_candidates": scored[:30],
        "outputs": {"overlay": str(Path(args.output_overlay)), "config_json": str(Path(args.output_json)), "report_md": str(Path(args.output_md))},
    }
    write_json(Path(args.output_json), report)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(render_md(report), encoding="utf-8")
    print(json.dumps({"decision": decision, "searched": report["searched_policy_count"], "passing": len(scored), "best": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
