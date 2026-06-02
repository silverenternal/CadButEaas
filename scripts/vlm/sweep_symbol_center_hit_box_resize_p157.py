#!/usr/bin/env python3
"""P157 center-hit box resize rescue for symbol overlays.

Offline sweep uses gold targets only for scoring/diagnostics. Materialized
policies use predicted bbox/label/score and fixed config only.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
P155E_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p155e_best.jsonl"
P155A_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p155a_p140_best.jsonl"
OUT_JSON = ROOT / "configs/vlm/symbol_center_hit_box_resize_rescue_p157.json"
OUT_MD = ROOT / "reports/vlm/symbol_center_hit_box_resize_rescue_p157.md"
OUT_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p157_best.jsonl"


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


def label(item: dict[str, Any]) -> str:
    return str(item.get("symbol_type") or item.get("label") or "generic_symbol")


def score(item: dict[str, Any]) -> float:
    value = item.get("confidence") if item.get("confidence") is not None else item.get("score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def scale_box(box: list[float], sx: float, sy: float, min_size: float = 0.0) -> list[float]:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    width = max(min_size, (box[2] - box[0]) * sx)
    height = max(min_size, (box[3] - box[1]) * sy)
    return [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0]


def target_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate((row.get("targets") or {}).get("symbol") or []):
        box = bbox4(item.get("bbox"))
        if box is not None:
            out.append({"id": str(item.get("target_id") or idx), "bbox": box, "bucket": bucket(box)})
    return out


def pred_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate(row.get("symbol_candidates") or []):
        box = bbox4(item.get("bbox"))
        if box is not None:
            out.append({"idx": idx, "bbox": box, "label": label(item), "score": score(item), "raw": item})
    return out


def transform_box(pred: dict[str, Any], policy: dict[str, Any]) -> list[float]:
    pred_bucket = bucket(pred["bbox"])
    pred_label = pred["label"]
    sx, sy = policy["default_scale"]
    if pred_bucket in policy.get("bucket_scale", {}):
        sx, sy = policy["bucket_scale"][pred_bucket]
    if pred_label in policy.get("label_scale", {}):
        label_sx, label_sy = policy["label_scale"][pred_label]
        sx *= label_sx
        sy *= label_sy
    return scale_box(pred["bbox"], sx, sy, float(policy.get("min_size", 0.0)))


def evaluate(golds_by_row: dict[str, list[dict[str, Any]]], preds_by_row: dict[str, list[dict[str, Any]]], policy: dict[str, Any] | None = None) -> dict[str, Any]:
    totals = Counter()
    by_area_gold = Counter()
    by_area_tp = Counter()
    by_area_center = Counter()
    by_label_pred = Counter()
    by_label_tp = Counter()
    for row_id, golds in golds_by_row.items():
        preds = preds_by_row.get(row_id, [])
        boxes = []
        for pred in preds:
            box = transform_box(pred, policy) if policy else pred["bbox"]
            boxes.append((box, pred))
            by_label_pred[pred["label"]] += 1
        totals["gold"] += len(golds)
        totals["pred"] += len(preds)
        used_iou: set[int] = set()
        used_center: set[int] = set()
        for gold in golds:
            by_area_gold[gold["bucket"]] += 1
            best_idx = None
            best_iou = 0.0
            center_idx = None
            for idx, (box, _pred) in enumerate(boxes):
                overlap = iou(box, gold["bbox"])
                if idx not in used_iou and overlap > best_iou:
                    best_iou = overlap
                    best_idx = idx
                if center_idx is None and idx not in used_center and center_covered(box, gold["bbox"]):
                    center_idx = idx
            if best_idx is not None and best_iou >= 0.30:
                used_iou.add(best_idx)
                totals["tp"] += 1
                by_area_tp[gold["bucket"]] += 1
                by_label_tp[boxes[best_idx][1]["label"]] += 1
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
        "by_label_precision_proxy": {key: round(by_label_tp[key] / max(by_label_pred[key], 1), 6) for key in sorted(by_label_pred)},
    }


def center_gap_diagnostics(golds_by_row: dict[str, list[dict[str, Any]]], preds_by_row: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    by_area = defaultdict(Counter)
    by_label = defaultdict(Counter)
    samples = []
    for row_id, golds in golds_by_row.items():
        preds = preds_by_row.get(row_id, [])
        for gold in golds:
            best_center = None
            best_center_iou = 0.0
            for pred in preds:
                if center_covered(pred["bbox"], gold["bbox"]):
                    overlap = iou(pred["bbox"], gold["bbox"])
                    if best_center is None or overlap > best_center_iou:
                        best_center = pred
                        best_center_iou = overlap
            if best_center is not None:
                by_area[gold["bucket"]]["center_hit"] += 1
                by_label[best_center["label"]]["center_hit"] += 1
                if best_center_iou < 0.30:
                    by_area[gold["bucket"]]["center_hit_iou_fail"] += 1
                    by_label[best_center["label"]]["center_hit_iou_fail"] += 1
                    if len(samples) < 30:
                        samples.append({"row_id": row_id, "gold_bucket": gold["bucket"], "pred_label": best_center["label"], "iou": round(best_center_iou, 4), "gold_bbox": gold["bbox"], "pred_bbox": best_center["bbox"]})
    def pack(counter_map: dict[str, Counter]) -> dict[str, Any]:
        out = {}
        for key, counts in sorted(counter_map.items()):
            center_hits = int(counts["center_hit"])
            fails = int(counts["center_hit_iou_fail"])
            out[key] = {"center_hit": center_hits, "center_hit_iou_fail": fails, "fail_rate_among_center_hits": round(fails / max(center_hits, 1), 6)}
        return out
    return {"by_area": pack(by_area), "by_pred_label": pack(by_label), "samples": samples}


def candidate_policies() -> list[dict[str, Any]]:
    policies = []
    # Focused grid after the first broad attempt was too slow: P155C says the
    # major recoverable mass is tiny/small/medium center-hit IoU-fail, so keep
    # large/xlarge mostly stable and vary small-object scale/aspect.
    default_scales = [(1.0, 1.0), (1.10, 1.0)]
    tiny_scales = [(1.0, 1.0), (1.40, 1.15), (1.80, 1.25)]
    small_scales = [(1.0, 1.0), (1.20, 1.05)]
    medium_scales = [(1.0, 1.0), (1.20, 0.95)]
    large_scales = [(1.0, 1.0)]
    label_scale_sets = [
        {},
        {"generic_symbol": (1.10, 1.0)},
        {"sink": (1.05, 1.0), "equipment": (1.05, 1.0), "appliance": (1.05, 1.0)},
    ]
    min_sizes = [0.0, 8.0]
    for default in default_scales:
        for tiny in tiny_scales:
            for small in small_scales:
                for medium in medium_scales:
                    for large in large_scales:
                        for label_scales in label_scale_sets:
                            for min_size in min_sizes:
                                policies.append({
                                    "name": f"p157_d{default}_t{tiny}_s{small}_m{medium}_l{large}_ms{min_size}_ls{len(label_scales)}",
                                    "default_scale": default,
                                    "bucket_scale": {"tiny": tiny, "small": small, "medium": medium, "large": large, "xlarge": large},
                                    "label_scale": label_scales,
                                    "min_size": min_size,
                                })
    return policies

def materialize(base_rows: list[dict[str, Any]], preds_by_row: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for raw in base_rows:
        row = copy.deepcopy(raw)
        row_id = str(row.get("row_id") or row.get("id"))
        candidates = []
        for idx, pred in enumerate(preds_by_row.get(row_id, [])):
            item = copy.deepcopy(pred["raw"])
            item["bbox"] = transform_box(pred, policy)
            item["id"] = f"{row_id}_p157_best_symbol_{idx:05d}"
            item["target_id"] = item["id"]
            item["source"] = "symbol_policy_overlay_p157_best"
            item.setdefault("metadata", {})["p157_box_resize"] = policy["name"]
            candidates.append(item)
        row["symbol_candidates"] = candidates
        if isinstance(row.get("expected_json"), dict):
            row["expected_json"]["symbol_candidates"] = [copy.deepcopy(item) for item in candidates]
        row["symbol_policy_overlay"] = {"policy_id": "p157_best", "description": "P157 center-hit box resize rescue candidate.", "policy": policy}
        rows.append(row)
    return rows


def delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    return {key: round(float(a[key]) - float(b[key]), 6) for key in ["precision", "recall", "f1", "center_recall", "prediction_inflation"]}


def render_md(report: dict[str, Any]) -> str:
    lines = ["# P157 Symbol Center-Hit Box Resize Rescue", "", f"Decision: **{report['decision']}**", "", "## Metrics", "", "| Policy | Precision | Recall | F1 | Center | Inflation |", "|---|---:|---:|---:|---:|---:|"]
    for name, metrics in report["baseline_metrics"].items():
        lines.append(f"| `{name}` | {metrics['precision']:.6f} | {metrics['recall']:.6f} | {metrics['f1']:.6f} | {metrics['center_recall']:.6f} | {metrics['prediction_inflation']:.6f} |")
    best = report["best_metrics"]
    lines.append(f"| `p157_best` | {best['precision']:.6f} | {best['recall']:.6f} | {best['f1']:.6f} | {best['center_recall']:.6f} | {best['prediction_inflation']:.6f} |")
    lines.extend(["", "## Deltas", "", f"- vs `p155e_best`: `{json.dumps(report['delta_vs_p155e'], ensure_ascii=False)}`", f"- vs `p155a_p140`: `{json.dumps(report['delta_vs_p155a'], ensure_ascii=False)}`", "", "## Center-Hit Gap Diagnostics", "", "| Bucket | Center Hit | Center-Hit IoU Fail | Fail Rate |", "|---|---:|---:|---:|"])
    for bucket_name, row in report["center_gap_diagnostics"]["by_area"].items():
        lines.append(f"| `{bucket_name}` | {row['center_hit']} | {row['center_hit_iou_fail']} | {row['fail_rate_among_center_hits']:.6f} |")
    lines.extend(["", "## Best Policy", "", f"- `{report['best_policy']['name']}`", f"- default_scale: `{report['best_policy']['default_scale']}`", f"- bucket_scale: `{json.dumps(report['best_policy']['bucket_scale'], ensure_ascii=False)}`", f"- label_scale: `{json.dumps(report['best_policy']['label_scale'], ensure_ascii=False)}`", f"- min_size: `{report['best_policy']['min_size']}`", "", "## Artifacts", ""])
    for value in report["outputs"].values():
        lines.append(f"- `{value}`")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-overlay", default=str(P155A_OVERLAY))
    parser.add_argument("--input-overlay", default=str(P155E_OVERLAY))
    parser.add_argument("--output-json", default=str(OUT_JSON))
    parser.add_argument("--output-md", default=str(OUT_MD))
    parser.add_argument("--output-overlay", default=str(OUT_OVERLAY))
    args = parser.parse_args()

    base_rows = load_jsonl(Path(args.base_overlay))
    input_rows = load_jsonl(Path(args.input_overlay))
    p155a_preds = {str(row.get("row_id") or row.get("id")): pred_symbols(row) for row in base_rows}
    p155e_preds = {str(row.get("row_id") or row.get("id")): pred_symbols(row) for row in input_rows}
    golds_by_row = {str(row.get("row_id") or row.get("id")): target_symbols(row) for row in base_rows}
    p155a_metrics = evaluate(golds_by_row, p155a_preds)
    p155e_metrics = evaluate(golds_by_row, p155e_preds)
    diagnostics = center_gap_diagnostics(golds_by_row, p155e_preds)

    scored = []
    for policy in candidate_policies():
        metrics = evaluate(golds_by_row, p155e_preds, policy)
        if metrics["precision"] >= 0.52 and metrics["prediction_inflation"] <= 1.05:
            scored.append({"policy": policy, "metrics": metrics})
    scored.sort(key=lambda row: (row["metrics"]["f1"], row["metrics"]["recall"], row["metrics"]["center_recall"]), reverse=True)
    best = scored[0]
    write_jsonl(Path(args.output_overlay), materialize(base_rows, p155e_preds, best["policy"]))
    decision = "positive_adopt_p157" if best["metrics"]["f1"] > p155e_metrics["f1"] else "negative_keep_p155e"
    report = {
        "id": "SCI-P2-157-symbol-center-hit-box-resize-rescue",
        "created_on": "2026-05-17",
        "decision": decision,
        "claim_boundary": "Post-hoc box-resize sweep on 74-row public-raster overlay subset. Runtime transform uses predicted bbox/label/score and fixed config only; gold targets are evaluation-only.",
        "baseline_metrics": {"p155a_p140": p155a_metrics, "p155e_best": p155e_metrics},
        "center_gap_diagnostics": diagnostics,
        "searched_policy_count": len(candidate_policies()),
        "passing_policy_count": len(scored),
        "best_policy": best["policy"],
        "best_metrics": best["metrics"],
        "delta_vs_p155e": delta(best["metrics"], p155e_metrics),
        "delta_vs_p155a": delta(best["metrics"], p155a_metrics),
        "top_candidates": scored[:30],
        "outputs": {"overlay": str(Path(args.output_overlay)), "config_json": str(Path(args.output_json)), "report_md": str(Path(args.output_md))},
    }
    write_json(Path(args.output_json), report)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(render_md(report), encoding="utf-8")
    print(json.dumps({"decision": decision, "searched": report["searched_policy_count"], "passing": report["passing_policy_count"], "best_metrics": best["metrics"], "delta_vs_p155e": report["delta_vs_p155e"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
