#!/usr/bin/env python3
"""Evaluate raster MoE target-level predictions against public raster targets.

P101 focuses on target-level evidence, not relation graph claims. It currently
scores symbol candidates from MoE overlays against raster target boxes and emits
family availability for boundary/space/text so missing model outputs remain
explicit rather than silently ignored.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_TARGETS = ROOT / "datasets/public_raster_moe_supervision_v19/locked.jsonl"
DEFAULT_REPORT = ROOT / "reports/vlm/raster_moe_target_level_eval_p101.md"
DEFAULT_SUMMARY = ROOT / "reports/vlm/raster_moe_target_level_eval_p101.json"

AREA_BUCKETS = ["tiny", "small", "medium", "large"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bbox4(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def area_bucket(box: list[float]) -> str:
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area <= 64:
        return "tiny"
    if area <= 256:
        return "small"
    if area <= 1024:
        return "medium"
    return "large"


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    if inter <= 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def center_covered(pred: list[float], gold: list[float]) -> bool:
    cx = (gold[0] + gold[2]) / 2.0
    cy = (gold[1] + gold[3]) / 2.0
    return pred[0] <= cx <= pred[2] and pred[1] <= cy <= pred[3]


def normalize_symbol_label(value: Any) -> str:
    label = str(value or "generic_symbol")
    aliases = {
        "toilet": "equipment",
        "wc": "equipment",
        "bathtub": "bathtub",
    }
    return aliases.get(label, label)


def target_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    symbols = []
    for idx, item in enumerate(((row.get("targets") or {}).get("symbol") or [])):
        box = bbox4(item.get("bbox"))
        if box is None:
            continue
        symbols.append({
            "id": str(item.get("target_id") or f"gold_{idx}"),
            "bbox": box,
            "label": normalize_symbol_label(item.get("semantic_type") or item.get("label") or item.get("raw_label")),
            "area_bucket": area_bucket(box),
        })
    return symbols


def overlay_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = row.get("symbol_candidates")
    if candidates is None:
        candidates = ((row.get("expected_json") or {}).get("symbol_candidates") or [])
    preds = []
    for idx, item in enumerate(candidates or []):
        box = bbox4(item.get("bbox"))
        if box is None:
            continue
        preds.append({
            "id": str(item.get("id") or item.get("target_id") or f"pred_{idx}"),
            "bbox": box,
            "label": normalize_symbol_label(item.get("symbol_type") or item.get("semantic_type") or item.get("label")),
            "score": float(item.get("confidence") if item.get("confidence") is not None else item.get("score") or 0.0),
        })
    return preds


def match_symbols(golds: list[dict[str, Any]], preds: list[dict[str, Any]], iou_threshold: float) -> dict[str, Any]:
    used_pred_iou: set[int] = set()
    used_pred_label: set[int] = set()
    used_pred_center: set[int] = set()
    by_label_gold = Counter(g["label"] for g in golds)
    by_label_tp = Counter()
    by_area_gold = Counter(g["area_bucket"] for g in golds)
    by_area_tp = Counter()
    by_area_center = Counter()
    tp_iou_any = 0
    tp_iou_label = 0
    center_any = 0
    for gold in golds:
        best_index = None
        best_iou = 0.0
        best_label_index = None
        best_label_iou = 0.0
        center_index = None
        for idx, pred in enumerate(preds):
            overlap = iou(pred["bbox"], gold["bbox"])
            if idx not in used_pred_iou and overlap > best_iou:
                best_iou = overlap
                best_index = idx
            if idx not in used_pred_label and pred["label"] == gold["label"] and overlap > best_label_iou:
                best_label_iou = overlap
                best_label_index = idx
            if center_index is None and idx not in used_pred_center and center_covered(pred["bbox"], gold["bbox"]):
                center_index = idx
        if best_index is not None and best_iou >= iou_threshold:
            used_pred_iou.add(best_index)
            tp_iou_any += 1
            by_area_tp[gold["area_bucket"]] += 1
        if best_label_index is not None and best_label_iou >= iou_threshold:
            used_pred_label.add(best_label_index)
            tp_iou_label += 1
            by_label_tp[gold["label"]] += 1
        if center_index is not None:
            used_pred_center.add(center_index)
            center_any += 1
            by_area_center[gold["area_bucket"]] += 1
    return {
        "tp_iou_any": tp_iou_any,
        "tp_iou_label": tp_iou_label,
        "center_any": center_any,
        "by_label_gold": by_label_gold,
        "by_label_tp": by_label_tp,
        "by_area_gold": by_area_gold,
        "by_area_tp": by_area_tp,
        "by_area_center": by_area_center,
    }


def prf(tp: int, predicted: int, gold: int) -> dict[str, Any]:
    precision = tp / max(predicted, 1)
    recall = tp / max(gold, 1)
    return {
        "tp": int(tp),
        "predicted": int(predicted),
        "gold": int(gold),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall), 6),
    }


def evaluate_policy(target_rows: dict[str, dict[str, Any]], overlay_path: Path, iou_threshold: float) -> dict[str, Any]:
    overlay_rows = load_jsonl(overlay_path)
    totals = Counter()
    by_source = defaultdict(Counter)
    by_label_gold = Counter()
    by_label_tp = Counter()
    by_area_gold = Counter()
    by_area_tp = Counter()
    by_area_center = Counter()
    missing_targets = []
    for row in overlay_rows:
        row_id = str(row.get("row_id") or row.get("id") or "")
        target = target_rows.get(row_id)
        if target is None:
            missing_targets.append(row_id)
            continue
        golds = target_symbols(target)
        preds = overlay_symbols(row)
        matched = match_symbols(golds, preds, iou_threshold)
        source = str(target.get("source_dataset") or "unknown")
        totals["records"] += 1
        totals["gold"] += len(golds)
        totals["predicted"] += len(preds)
        totals["tp_iou_any"] += matched["tp_iou_any"]
        totals["tp_iou_label"] += matched["tp_iou_label"]
        totals["center_any"] += matched["center_any"]
        by_source[source]["records"] += 1
        by_source[source]["gold"] += len(golds)
        by_source[source]["predicted"] += len(preds)
        by_source[source]["tp_iou_any"] += matched["tp_iou_any"]
        by_source[source]["tp_iou_label"] += matched["tp_iou_label"]
        by_source[source]["center_any"] += matched["center_any"]
        by_label_gold.update(matched["by_label_gold"])
        by_label_tp.update(matched["by_label_tp"])
        by_area_gold.update(matched["by_area_gold"])
        by_area_tp.update(matched["by_area_tp"])
        by_area_center.update(matched["by_area_center"])
    return {
        "input": str(overlay_path.relative_to(ROOT) if overlay_path.is_absolute() and overlay_path.is_relative_to(ROOT) else overlay_path),
        "records": int(totals["records"]),
        "missing_target_rows": len(missing_targets),
        "missing_target_row_ids_sample": missing_targets[:20],
        "symbol_iou_any": prf(totals["tp_iou_any"], totals["predicted"], totals["gold"]),
        "symbol_iou_label_exact": prf(totals["tp_iou_label"], totals["predicted"], totals["gold"]),
        "symbol_center_any": {"hit": int(totals["center_any"]), "gold": int(totals["gold"]), "recall": round(totals["center_any"] / max(totals["gold"], 1), 6)},
        "prediction_inflation": round(totals["predicted"] / max(totals["gold"], 1), 6),
        "by_source_dataset": {
            source: {
                "records": int(c["records"]),
                "symbol_iou_any": prf(c["tp_iou_any"], c["predicted"], c["gold"]),
                "symbol_center_any": {"hit": int(c["center_any"]), "gold": int(c["gold"]), "recall": round(c["center_any"] / max(c["gold"], 1), 6)},
                "prediction_inflation": round(c["predicted"] / max(c["gold"], 1), 6),
            }
            for source, c in sorted(by_source.items())
        },
        "by_label_recall": {label: round(by_label_tp[label] / max(count, 1), 6) for label, count in sorted(by_label_gold.items())},
        "by_area_iou_recall": {bucket: round(by_area_tp[bucket] / max(by_area_gold[bucket], 1), 6) for bucket in AREA_BUCKETS if by_area_gold[bucket]},
        "by_area_center_recall": {bucket: round(by_area_center[bucket] / max(by_area_gold[bucket], 1), 6) for bucket in AREA_BUCKETS if by_area_gold[bucket]},
    }


def family_availability(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts = defaultdict(Counter)
    for row in rows:
        source = str(row.get("source_dataset") or "unknown")
        targets = row.get("targets") or {}
        counts[source]["records"] += 1
        for family in ["boundary", "space", "symbol", "text"]:
            counts[source][family] += len(targets.get(family) or [])
    return {source: dict(counter) for source, counter in sorted(counts.items())}


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# P1-101 Raster MoE Target-Level Evaluation",
        "",
        "## Scope",
        f"- Target records: `{summary['target_records']}`",
        f"- IoU threshold: `{summary['iou_threshold']}`",
        "- Claim boundary: target-level symbol evaluation only; no relation graph or full scene-graph claim.",
        "",
        "## Family Availability",
        "| Source | Records | Boundary | Space | Symbol | Text |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for source, counts in summary["family_availability"].items():
        lines.append(f"| `{source}` | {counts.get('records',0)} | {counts.get('boundary',0)} | {counts.get('space',0)} | {counts.get('symbol',0)} | {counts.get('text',0)} |")
    lines.extend(["", "## Policy Metrics", "| Policy | Records | Precision | Recall | F1 | Center Recall | Inflation |", "|---|---:|---:|---:|---:|---:|---:|"])
    for policy in summary["policies"]:
        m = policy["symbol_iou_any"]
        center = policy["symbol_center_any"]["recall"]
        lines.append(f"| `{policy['policy_id']}` | {policy['records']} | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {center:.6f} | {policy['prediction_inflation']:.6f} |")
    if summary.get("comparisons"):
        lines.extend(["", "## Deltas Vs Baseline", "| Policy | ΔPrecision | ΔRecall | ΔF1 | ΔCenter | ΔInflation |", "|---|---:|---:|---:|---:|---:|"])
        for item in summary["comparisons"]:
            lines.append(f"| `{item['policy_id']}` | {item['delta_precision']:+.6f} | {item['delta_recall']:+.6f} | {item['delta_f1']:+.6f} | {item['delta_center_recall']:+.6f} | {item['delta_prediction_inflation']:+.6f} |")
    lines.extend(["", "## Caveats", "- Current model-output coverage is the 74-row locked overlay subset, not the full 992-row public raster target set.", "- Boundary/space/text targets are counted for availability but not scored unless corresponding model outputs are supplied.", "- Relation edges such as `contains` require reviewed scene-graph labels or expected-json relation supervision and are out of scope here."])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", default=str(DEFAULT_TARGETS))
    parser.add_argument("--policy", action="append", required=True, help="policy_id=overlay_jsonl")
    parser.add_argument("--baseline-policy", default="v28_frozen_detector_baseline")
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument("--summary", default=str(DEFAULT_SUMMARY))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    target_path = Path(args.targets)
    target_rows_list = load_jsonl(target_path)
    target_rows = {str(row.get("id") or row.get("row_id")): row for row in target_rows_list}
    policies = []
    for spec in args.policy:
        if "=" not in spec:
            raise SystemExit(f"invalid policy spec {spec!r}; expected policy_id=path")
        policy_id, policy_path = spec.split("=", 1)
        result = evaluate_policy(target_rows, Path(policy_path), args.iou_threshold)
        result["policy_id"] = policy_id
        policies.append(result)
    baseline = next((p for p in policies if p["policy_id"] == args.baseline_policy), None)
    comparisons = []
    if baseline is not None:
        for policy in policies:
            if policy is baseline:
                continue
            comparisons.append({
                "policy_id": policy["policy_id"],
                "baseline_policy": baseline["policy_id"],
                "delta_precision": round(policy["symbol_iou_any"]["precision"] - baseline["symbol_iou_any"]["precision"], 6),
                "delta_recall": round(policy["symbol_iou_any"]["recall"] - baseline["symbol_iou_any"]["recall"], 6),
                "delta_f1": round(policy["symbol_iou_any"]["f1"] - baseline["symbol_iou_any"]["f1"], 6),
                "delta_center_recall": round(policy["symbol_center_any"]["recall"] - baseline["symbol_center_any"]["recall"], 6),
                "delta_prediction_inflation": round(policy["prediction_inflation"] - baseline["prediction_inflation"], 6),
            })
    summary = {
        "id": "SCI-P1-101-raster-moe-target-level-evaluation",
        "target_records": str(target_path.relative_to(ROOT) if target_path.is_absolute() and target_path.is_relative_to(ROOT) else target_path),
        "target_record_count": len(target_rows_list),
        "evaluated_record_count": max((p["records"] for p in policies), default=0),
        "coverage_boundary": "model-output overlays currently cover a 74-row locked subset; full 992-row target set is counted for availability only unless predictions exist",
        "iou_threshold": args.iou_threshold,
        "claim_boundary": "target-level symbol metrics only; not relation graph quality",
        "family_availability": family_availability(target_rows_list),
        "policies": policies,
        "comparisons": comparisons,
    }
    write_json(Path(args.summary), summary)
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(render_report(summary), encoding="utf-8")
    print(json.dumps({"summary": summary["id"], "policies": [(p["policy_id"], p["symbol_iou_any"], p["prediction_inflation"]) for p in policies], "comparisons": comparisons}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
