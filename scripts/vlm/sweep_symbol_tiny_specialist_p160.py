#!/usr/bin/env python3
"""P160 tiny-symbol specialist box policy sweep on top of P157.

Runtime policy uses predicted bbox/label/score only. Gold targets are evaluation-only.
"""
from __future__ import annotations

import argparse
import copy
import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
P157_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p157_best.jsonl"
OUT_JSON = ROOT / "configs/vlm/symbol_tiny_specialist_rescue_p160.json"
OUT_MD = ROOT / "reports/vlm/symbol_tiny_specialist_rescue_p160.md"
OUT_OVERLAY = ROOT / "reports/vlm/symbol_policy_moe_overlay_p160_best.jsonl"


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


def center_resize(box: list[float], width: float, height: float) -> list[float]:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    return [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0]


def transform_box(box: list[float], pred_label: str, pred_score: float, policy: dict[str, Any]) -> list[float]:
    pred_bucket = bucket(box)
    if pred_score < policy["score_min"]:
        return box
    if pred_bucket not in set(policy["apply_buckets"]):
        return box
    width = box[2] - box[0]
    height = box[3] - box[1]
    mode = policy["mode"]
    if mode == "min_size":
        width = max(width, float(policy["tiny_width"]))
        height = max(height, float(policy["tiny_height"]))
    elif mode == "fixed_if_tiny":
        if pred_bucket == "tiny":
            width = float(policy["tiny_width"])
            height = float(policy["tiny_height"])
        elif pred_bucket == "small":
            width = max(width, float(policy["small_min_width"]))
            height = max(height, float(policy["small_min_height"]))
    elif mode == "label_size":
        sizes = policy.get("label_sizes", {})
        default = policy.get("default_size", [width, height])
        size = sizes.get(pred_label, default)
        if pred_bucket == "tiny":
            width, height = float(size[0]), float(size[1])
    return center_resize(box, width, height)


def target_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate((row.get("targets") or {}).get("symbol") or []):
        box = bbox4(item.get("bbox"))
        if box is not None:
            out.append({"id": str(item.get("target_id") or idx), "bbox": box, "bucket": bucket(box)})
    return out


def pred_symbols(row: dict[str, Any], policy: dict[str, Any] | None = None) -> list[dict[str, Any]]:
    out = []
    for item in row.get("symbol_candidates") or []:
        box = bbox4(item.get("bbox"))
        if box is None:
            continue
        pred_label = label(item)
        pred_score = score(item)
        if policy is not None:
            box = transform_box(box, pred_label, pred_score, policy)
        out.append({"bbox": box, "label": pred_label, "score": pred_score, "raw": item})
    return out


def evaluate(rows: list[dict[str, Any]], policy: dict[str, Any] | None = None) -> dict[str, Any]:
    totals = Counter()
    by_area_gold = Counter()
    by_area_tp = Counter()
    by_area_center = Counter()
    changed = Counter()
    for row in rows:
        golds = target_symbols(row)
        preds = pred_symbols(row, policy)
        totals["gold"] += len(golds)
        totals["pred"] += len(preds)
        if policy is not None:
            for raw in row.get("symbol_candidates") or []:
                box = bbox4(raw.get("bbox"))
                if box is not None and bucket(box) in set(policy["apply_buckets"]) and score(raw) >= policy["score_min"]:
                    changed[bucket(box)] += 1
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
        "changed_pred_buckets": dict(sorted(changed.items())),
    }


def candidate_policies() -> list[dict[str, Any]]:
    policies = []
    for mode in ["min_size", "fixed_if_tiny"]:
        for apply_buckets in [["tiny"], ["tiny", "small"]]:
            for score_min in [0.0, 0.20, 0.35, 0.50]:
                for tiny_width, tiny_height in [(6, 6), (8, 8), (10, 8), (12, 8), (12, 10), (14, 10), (16, 12)]:
                    for small_min_width, small_min_height in [(10, 10), (12, 10), (14, 12)]:
                        policies.append({
                            "name": f"p160_{mode}_b{''.join(apply_buckets)}_s{score_min}_tw{tiny_width}_th{tiny_height}_sw{small_min_width}_sh{small_min_height}",
                            "mode": mode,
                            "apply_buckets": apply_buckets,
                            "score_min": score_min,
                            "tiny_width": tiny_width,
                            "tiny_height": tiny_height,
                            "small_min_width": small_min_width,
                            "small_min_height": small_min_height,
                        })
    label_sizes = [
        {"generic_symbol": [8, 8], "sink": [10, 8], "equipment": [10, 8], "appliance": [10, 8], "shower": [8, 8], "stair": [8, 8], "column": [8, 8]},
        {"generic_symbol": [10, 8], "sink": [12, 8], "equipment": [12, 8], "appliance": [12, 8], "shower": [10, 8], "stair": [10, 8], "column": [8, 10]},
    ]
    for sizes in label_sizes:
        for score_min in [0.0, 0.20, 0.35]:
            policies.append({
                "name": f"p160_label_size_s{score_min}_{len(sizes)}",
                "mode": "label_size",
                "apply_buckets": ["tiny"],
                "score_min": score_min,
                "default_size": [8, 8],
                "label_sizes": sizes,
                "tiny_width": 8,
                "tiny_height": 8,
                "small_min_width": 10,
                "small_min_height": 10,
            })
    policies.append({"name": "p160_noop", "mode": "min_size", "apply_buckets": [], "score_min": 2.0, "tiny_width": 0, "tiny_height": 0, "small_min_width": 0, "small_min_height": 0})
    return policies


def materialize(rows: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    out_rows = []
    for raw_row in rows:
        row = copy.deepcopy(raw_row)
        candidates = []
        for idx, raw in enumerate(row.get("symbol_candidates") or []):
            box = bbox4(raw.get("bbox"))
            if box is None:
                continue
            item = copy.deepcopy(raw)
            item["bbox"] = transform_box(box, label(item), score(item), policy)
            item["id"] = f"{row.get('row_id') or row.get('id')}_p160_best_symbol_{idx:05d}"
            item["target_id"] = item["id"]
            item["source"] = "symbol_policy_overlay_p160_best"
            item.setdefault("metadata", {})["p160_tiny_specialist"] = policy["name"]
            candidates.append(item)
        row["symbol_candidates"] = candidates
        if isinstance(row.get("expected_json"), dict):
            row["expected_json"]["symbol_candidates"] = [copy.deepcopy(item) for item in candidates]
        row["symbol_policy_overlay"] = {"policy_id": "p160_best", "description": "P160 tiny-symbol specialist candidate.", "policy": policy}
        out_rows.append(row)
    return out_rows


def delta(a: dict[str, Any], b: dict[str, Any]) -> dict[str, float]:
    return {key: round(float(a[key]) - float(b[key]), 6) for key in ["precision", "recall", "f1", "center_recall", "prediction_inflation"]}


def render_md(report: dict[str, Any]) -> str:
    lines = ["# P160 Symbol Tiny Specialist Rescue", "", f"Decision: **{report['decision']}**", "", "## Metrics", "", "| Policy | Precision | Recall | F1 | Center | Inflation | Tiny IoU |", "|---|---:|---:|---:|---:|---:|---:|"]
    base = report["baseline_metrics"]["p157_best"]
    best = report["best_metrics"]
    lines.append(f"| `p157_best` | {base['precision']:.6f} | {base['recall']:.6f} | {base['f1']:.6f} | {base['center_recall']:.6f} | {base['prediction_inflation']:.6f} | {base['by_area_iou_recall'].get('tiny',0):.6f} |")
    lines.append(f"| `p160_best` | {best['precision']:.6f} | {best['recall']:.6f} | {best['f1']:.6f} | {best['center_recall']:.6f} | {best['prediction_inflation']:.6f} | {best['by_area_iou_recall'].get('tiny',0):.6f} |")
    lines.extend(["", "## Best Policy", "", f"- `{report['best_policy']['name']}`", f"- config: `{json.dumps(report['best_policy'], ensure_ascii=False)}`", "", "## Deltas", "", f"- vs `p157_best`: `{json.dumps(report['delta_vs_p157'], ensure_ascii=False)}`", f"- tiny IoU delta: `{report['tiny_iou_delta']:+.6f}`", "", "## Artifacts", ""])
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
    baseline = evaluate(rows)
    scored = []
    for policy in candidate_policies():
        metrics = evaluate(rows, policy)
        tiny_delta = metrics["by_area_iou_recall"].get("tiny", 0.0) - baseline["by_area_iou_recall"].get("tiny", 0.0)
        if metrics["f1"] >= baseline["f1"] - 0.003 or tiny_delta >= 0.02:
            scored.append({"policy": policy, "metrics": metrics, "tiny_iou_delta": round(tiny_delta, 6)})
    scored.sort(key=lambda row: (row["metrics"]["f1"], row["tiny_iou_delta"], row["metrics"]["recall"]), reverse=True)
    best = scored[0]
    write_jsonl(Path(args.output_overlay), materialize(rows, best["policy"]))
    decision = "positive_adopt_p160" if best["metrics"]["f1"] > baseline["f1"] else "negative_keep_p157"
    report = {
        "id": "SCI-P2-160-symbol-tiny-specialist-rescue",
        "created_on": "2026-05-17",
        "decision": decision,
        "claim_boundary": "Post-hoc tiny/small box-size sweep on 74-row public-raster overlay subset. Runtime policy uses predicted bbox/label/score only; gold targets are evaluation-only.",
        "baseline_metrics": {"p157_best": baseline},
        "searched_policy_count": len(candidate_policies()),
        "passing_policy_count": len(scored),
        "best_policy": best["policy"],
        "best_metrics": best["metrics"],
        "delta_vs_p157": delta(best["metrics"], baseline),
        "tiny_iou_delta": best["tiny_iou_delta"],
        "top_candidates": scored[:30],
        "outputs": {"overlay": str(Path(args.output_overlay)), "config_json": str(Path(args.output_json)), "report_md": str(Path(args.output_md))},
    }
    write_json(Path(args.output_json), report)
    Path(args.output_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output_md).write_text(render_md(report), encoding="utf-8")
    print(json.dumps({"decision": decision, "searched": report["searched_policy_count"], "passing": report["passing_policy_count"], "best_metrics": best["metrics"], "delta_vs_p157": report["delta_vs_p157"], "tiny_iou_delta": report["tiny_iou_delta"], "best_policy": best["policy"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
