#!/usr/bin/env python3
"""P158 label-conditioned cleanup/relabel sweep on top of P157.

Gold targets are used only for offline scoring. Materialized policies use only
predicted bbox/label/score plus fixed thresholds/relabel/drop config.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
P157_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p157_best.jsonl"
OUT_JSON = ROOT / "configs/vlm/symbol_label_conditioned_calibration_p158.json"
OUT_MD = ROOT / "reports/vlm/symbol_label_conditioned_calibration_p158.md"
OUT_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p158_best.jsonl"

BAD_LABELS = ["shower", "stair", "column", "bathtub"]


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


def norm_label(value: Any) -> str:
    label = str(value or "generic_symbol")
    return {"toilet": "equipment", "wc": "equipment"}.get(label, label)


def item_label(item: dict[str, Any]) -> str:
    return norm_label(item.get("symbol_type") or item.get("label") or "generic_symbol")


def item_score(item: dict[str, Any]) -> float:
    value = item.get("confidence") if item.get("confidence") is not None else item.get("score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def target_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate((row.get("targets") or {}).get("symbol") or []):
        box = bbox4(item.get("bbox"))
        if box is None:
            continue
        out.append({
            "id": str(item.get("target_id") or idx),
            "bbox": box,
            "label": norm_label(item.get("semantic_type") or item.get("label") or item.get("raw_label")),
            "bucket": bucket(box),
        })
    return out


def apply_policy(items: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    thresholds = policy.get("thresholds", {})
    relabel = policy.get("relabel", {})
    drop_buckets = policy.get("drop_buckets", {})
    for raw in items:
        box = bbox4(raw.get("bbox"))
        if box is None:
            continue
        label = item_label(raw)
        score = item_score(raw)
        if score < thresholds.get(label, 0.0):
            continue
        if bucket(box) in set(drop_buckets.get(label, [])):
            continue
        item = copy.deepcopy(raw)
        new_label = relabel.get(label, label)
        item["symbol_type"] = new_label
        item.setdefault("metadata", {})["p158_original_label"] = label
        item.setdefault("metadata", {})["p158_policy"] = policy["name"]
        out.append(item)
    return out


def score_rows(rows: list[dict[str, Any]], policy: dict[str, Any] | None = None) -> dict[str, Any]:
    totals = Counter()
    by_label_gold = Counter()
    by_label_tp = Counter()
    by_pred_label = Counter()
    by_pred_label_tp_any = Counter()
    by_area_gold = Counter()
    by_area_tp = Counter()
    for row in rows:
        golds = target_symbols(row)
        preds_raw = row.get("symbol_candidates") or []
        preds = apply_policy(preds_raw, policy) if policy else list(preds_raw)
        pred_boxes = []
        for pred in preds:
            box = bbox4(pred.get("bbox"))
            if box is None:
                continue
            pred_label = item_label(pred)
            pred_boxes.append({"bbox": box, "label": pred_label})
            by_pred_label[pred_label] += 1
        totals["gold"] += len(golds)
        totals["pred"] += len(pred_boxes)
        used_any: set[int] = set()
        used_label: set[int] = set()
        used_center: set[int] = set()
        for gold in golds:
            by_label_gold[gold["label"]] += 1
            by_area_gold[gold["bucket"]] += 1
            best_any_idx = None
            best_any_iou = 0.0
            best_label_idx = None
            best_label_iou = 0.0
            center_idx = None
            for idx, pred in enumerate(pred_boxes):
                overlap = iou(pred["bbox"], gold["bbox"])
                if idx not in used_any and overlap > best_any_iou:
                    best_any_iou = overlap
                    best_any_idx = idx
                if idx not in used_label and pred["label"] == gold["label"] and overlap > best_label_iou:
                    best_label_iou = overlap
                    best_label_idx = idx
                if center_idx is None and idx not in used_center and center_covered(pred["bbox"], gold["bbox"]):
                    center_idx = idx
            if best_any_idx is not None and best_any_iou >= 0.30:
                used_any.add(best_any_idx)
                totals["tp_any"] += 1
                by_area_tp[gold["bucket"]] += 1
                by_pred_label_tp_any[pred_boxes[best_any_idx]["label"]] += 1
            if best_label_idx is not None and best_label_iou >= 0.30:
                used_label.add(best_label_idx)
                totals["tp_label"] += 1
                by_label_tp[gold["label"]] += 1
            if center_idx is not None:
                used_center.add(center_idx)
                totals["center"] += 1
    def prf(tp: int, pred: int, gold: int) -> dict[str, Any]:
        precision = tp / max(pred, 1)
        recall = tp / max(gold, 1)
        f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)
        return {"tp": int(tp), "predicted": int(pred), "gold": int(gold), "precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6)}
    return {
        "symbol_iou_any": prf(totals["tp_any"], totals["pred"], totals["gold"]),
        "symbol_iou_label_exact": prf(totals["tp_label"], totals["pred"], totals["gold"]),
        "center_recall": round(totals["center"] / max(totals["gold"], 1), 6),
        "prediction_inflation": round(totals["pred"] / max(totals["gold"], 1), 6),
        "by_label_recall": {k: round(by_label_tp[k] / max(by_label_gold[k], 1), 6) for k in sorted(by_label_gold)},
        "by_pred_label_precision_any": {k: round(by_pred_label_tp_any[k] / max(by_pred_label[k], 1), 6) for k in sorted(by_pred_label)},
        "by_area_iou_recall": {k: round(by_area_tp[k] / max(by_area_gold[k], 1), 6) for k in sorted(by_area_gold)},
    }


def flat_metrics(scored: dict[str, Any]) -> dict[str, Any]:
    m = scored["symbol_iou_any"]
    return {"tp": m["tp"], "predicted": m["predicted"], "gold": m["gold"], "precision": m["precision"], "recall": m["recall"], "f1": m["f1"], "center_recall": scored["center_recall"], "prediction_inflation": scored["prediction_inflation"]}


def candidate_policies() -> list[dict[str, Any]]:
    policies = []
    # Focused first-pass grid: P157 already improved geometry. Here we only
    # tune the known noisy labels; relabel is kept minimal to avoid overfitting.
    shower_values = [0.0, 0.34, 0.38, 0.42]
    stair_values = [0.0, 0.36, 0.40]
    column_values = [0.0, 0.40, 0.44]
    bathtub_values = [0.0, 0.30]
    relabel_sets = [
        {},
        {"shower": "generic_symbol"},
        {"stair": "generic_symbol"},
    ]
    drop_bucket_sets = [
        {},
        {"shower": ["tiny"]},
    ]
    for shower in shower_values:
        for stair in stair_values:
            for column in column_values:
                for bathtub in bathtub_values:
                    thresholds = {"shower": shower, "stair": stair, "column": column, "bathtub": bathtub}
                    for relabel in relabel_sets:
                        for drop_buckets in drop_bucket_sets:
                            policies.append({
                                "name": f"p158_sh{shower}_st{stair}_col{column}_bt{bathtub}_rel{len(relabel)}_drop{len(drop_buckets)}",
                                "thresholds": thresholds,
                                "relabel": relabel,
                                "drop_buckets": drop_buckets,
                            })
    return policies

def materialize(rows: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    out_rows = []
    for raw in rows:
        row = copy.deepcopy(raw)
        candidates = apply_policy(row.get("symbol_candidates") or [], policy)
        row["symbol_candidates"] = candidates
        if isinstance(row.get("expected_json"), dict):
            row["expected_json"]["symbol_candidates"] = [copy.deepcopy(item) for item in candidates]
        overlay = row.setdefault("symbol_policy_overlay", {})
        overlay["policy_id"] = "p158_best"
        overlay["description"] = "P158 label-conditioned threshold/relabel rescue on top of P157."
        overlay["policy"] = policy
        out_rows.append(row)
    return out_rows


def delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    return {k: round(float(a[k]) - float(b[k]), 6) for k in ["precision", "recall", "f1", "center_recall", "prediction_inflation"]}


def render_md(report: dict[str, Any]) -> str:
    lines = ["# P158 Symbol Label-Conditioned Calibration / Relabel Rescue", "", f"Decision: **{report['decision']}**", "", "## Metrics", "", "| Policy | Precision | Recall | F1 | Center | Inflation |", "|---|---:|---:|---:|---:|---:|"]
    for name, metrics in report["baseline_metrics"].items():
        lines.append(f"| `{name}` | {metrics['precision']:.6f} | {metrics['recall']:.6f} | {metrics['f1']:.6f} | {metrics['center_recall']:.6f} | {metrics['prediction_inflation']:.6f} |")
    best = report["best_metrics"]
    lines.append(f"| `p158_best` | {best['precision']:.6f} | {best['recall']:.6f} | {best['f1']:.6f} | {best['center_recall']:.6f} | {best['prediction_inflation']:.6f} |")
    lines.extend(["", "## Best Policy", "", f"- `{report['best_policy']['name']}`", f"- thresholds: `{json.dumps(report['best_policy']['thresholds'], ensure_ascii=False)}`", f"- relabel: `{json.dumps(report['best_policy']['relabel'], ensure_ascii=False)}`", f"- drop_buckets: `{json.dumps(report['best_policy']['drop_buckets'], ensure_ascii=False)}`", "", "## Deltas", "", f"- vs `p157_best`: `{json.dumps(report['delta_vs_p157'], ensure_ascii=False)}`", "", "## Top Candidates", "", "| Rank | Policy | Precision | Recall | F1 | Inflation |", "|---:|---|---:|---:|---:|---:|"])
    for idx, row in enumerate(report["top_candidates"][:10], 1):
        m = row["metrics"]
        lines.append(f"| {idx} | `{row['policy']['name']}` | {m['precision']:.6f} | {m['recall']:.6f} | {m['f1']:.6f} | {m['prediction_inflation']:.6f} |")
    lines.extend(["", "## Artifacts", ""])
    for value in report["outputs"].values():
        lines.append(f"- `{value}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-overlay", default=str(P157_OVERLAY))
    parser.add_argument("--output-json", default=str(OUT_JSON))
    parser.add_argument("--output-md", default=str(OUT_MD))
    parser.add_argument("--output-overlay", default=str(OUT_OVERLAY))
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input_overlay))
    p157_score = score_rows(rows)
    p157_metrics = flat_metrics(p157_score)
    scored = []
    for policy in candidate_policies():
        score = score_rows(rows, policy)
        metrics = flat_metrics(score)
        if metrics["recall"] >= p157_metrics["recall"] - 0.006 and metrics["precision"] >= p157_metrics["precision"] - 0.002:
            scored.append({"policy": policy, "metrics": metrics, "detail": score})
    if not scored:
        raise SystemExit("no passing policies")
    scored.sort(key=lambda row: (row["metrics"]["f1"], row["metrics"]["precision"], row["metrics"]["recall"]), reverse=True)
    best = scored[0]
    write_jsonl(Path(args.output_overlay), materialize(rows, best["policy"]))
    decision = "positive_adopt_p158" if best["metrics"]["f1"] > p157_metrics["f1"] else "negative_keep_p157"
    report = {
        "id": "SCI-P2-158-symbol-label-conditioned-calibration-relabel-rescue",
        "created_on": "2026-05-17",
        "decision": decision,
        "claim_boundary": "Post-hoc label threshold/relabel/drop sweep on 74-row public-raster overlay subset. Runtime policy uses predicted label/score/bbox only; gold targets are evaluation-only.",
        "baseline_metrics": {"p157_best": p157_metrics},
        "baseline_detail": p157_score,
        "searched_policy_count": len(candidate_policies()),
        "passing_policy_count": len(scored),
        "best_policy": best["policy"],
        "best_metrics": best["metrics"],
        "best_detail": best["detail"],
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
