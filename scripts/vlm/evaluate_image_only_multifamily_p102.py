#!/usr/bin/env python3
"""Evaluate image-only raster MoE multi-family nodes against rasterized target boxes.

The v15 image-only assets are a separate 256x256 raster supervision path. This
script scores boundary/space/symbol/text at target-box level and keeps the claim
separate from public_raster_moe_supervision_v19 relation-graph metrics.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ["boundary", "space", "symbol", "text"]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


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


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-9)


def normalize_family(value: Any) -> str:
    family = str(value or "")
    if family in FAMILIES:
        return family
    return "unknown"


def normalize_label(value: Any) -> str:
    label = str(value or "unknown")
    if label == "hard_wall":
        return "wall"
    if label in {"door", "window"}:
        return label
    if label == "room_boundary":
        return "wall"
    return label


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


def gold_by_family(row: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out = {family: [] for family in FAMILIES}
    for idx, box in enumerate(row.get("boxes") or []):
        family = normalize_family(box.get("family"))
        bbox = bbox4(box.get("bbox"))
        if family in out and bbox is not None:
            out[family].append({"id": f"gold_{idx}", "bbox": bbox, "label": normalize_label(box.get("class") or box.get("semantic_type"))})
    return out


def pred_by_family(row: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out = {family: [] for family in FAMILIES}
    for idx, node in enumerate((row.get("scene_graph") or {}).get("nodes") or []):
        family = normalize_family(node.get("family"))
        bbox = bbox4((node.get("geometry") or {}).get("bbox") or node.get("bbox"))
        if family in out and bbox is not None:
            out[family].append({"id": str(node.get("id") or f"pred_{idx}"), "bbox": bbox, "label": normalize_label(node.get("semantic_type") or node.get("label"))})
    return out


def center_covered(pred: list[float], gold: list[float]) -> bool:
    cx = (gold[0] + gold[2]) / 2.0
    cy = (gold[1] + gold[3]) / 2.0
    return pred[0] <= cx <= pred[2] and pred[1] <= cy <= pred[3]


def match(golds: list[dict[str, Any]], preds: list[dict[str, Any]], iou_threshold: float, label_exact: bool, center_mode: bool = False) -> int:
    used: set[int] = set()
    tp = 0
    for gold in golds:
        best = None
        best_iou = 0.0
        for idx, pred in enumerate(preds):
            if idx in used:
                continue
            if label_exact and pred["label"] != gold["label"]:
                continue
            if center_mode:
                if center_covered(pred["bbox"], gold["bbox"]):
                    best = idx
                    break
                continue
            overlap = iou(pred["bbox"], gold["bbox"])
            if overlap > best_iou:
                best_iou = overlap
                best = idx
        if best is not None and (center_mode or best_iou >= iou_threshold):
            used.add(best)
            tp += 1
    return tp


def evaluate(gold_rows: list[dict[str, Any]], pred_rows: list[dict[str, Any]], iou_threshold: float) -> dict[str, Any]:
    gold_map = {str(row.get("id")): row for row in gold_rows}
    totals = {family: Counter() for family in FAMILIES}
    label_totals = {family: Counter() for family in FAMILIES}
    matched_rows = 0
    missing_gold = []
    for pred_row in pred_rows:
        row_id = str(pred_row.get("id") or "")
        gold_row = gold_map.get(row_id)
        if gold_row is None:
            missing_gold.append(row_id)
            continue
        matched_rows += 1
        golds = gold_by_family(gold_row)
        preds = pred_by_family(pred_row)
        for family in FAMILIES:
            totals[family]["gold"] += len(golds[family])
            totals[family]["predicted"] += len(preds[family])
            totals[family]["tp_any"] += match(golds[family], preds[family], iou_threshold, False)
            totals[family]["tp_label"] += match(golds[family], preds[family], iou_threshold, True)
            totals[family]["tp_center"] += match(golds[family], preds[family], iou_threshold, False, center_mode=True)
            for item in golds[family]:
                label_totals[family][f"gold::{item['label']}"] += 1
            for item in preds[family]:
                label_totals[family][f"pred::{item['label']}"] += 1
    family_metrics = {}
    for family in FAMILIES:
        c = totals[family]
        family_metrics[family] = {
            "iou_any": prf(c["tp_any"], c["predicted"], c["gold"]),
            "iou_label_exact": prf(c["tp_label"], c["predicted"], c["gold"]),
            "center_any": {"hit": int(c["tp_center"]), "gold": int(c["gold"]), "recall": round(c["tp_center"] / max(c["gold"], 1), 6)},
            "prediction_inflation": round(c["predicted"] / max(c["gold"], 1), 6),
            "label_distribution": dict(sorted(label_totals[family].items())),
        }
    return {
        "matched_rows": matched_rows,
        "missing_gold_rows": len(missing_gold),
        "missing_gold_row_ids_sample": missing_gold[:20],
        "family_metrics": family_metrics,
    }


def render(summary: dict[str, Any]) -> str:
    lines = [
        "# P1-102 Image-only Multi-family Target Evaluation",
        "",
        "## Scope",
        f"- Gold/proposal boxes: `{summary['gold_boxes']}`",
        f"- Predictions: `{summary['predictions']}`",
        f"- Matched rows: `{summary['matched_rows']}`",
        f"- IoU threshold: `{summary['iou_threshold']}`",
        "- Claim boundary: separate 256x256 image-only raster supervision path; target-level boxes only, no relation-graph claim.",
        "",
        "## Family Metrics",
        "| Family | Precision | Recall | F1 | Center Recall | Label-exact F1 | Inflation | Gold | Predicted |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        m = summary["family_metrics"][family]
        any_m = m["iou_any"]
        label_m = m["iou_label_exact"]
        center = m["center_any"]["recall"]
        lines.append(f"| `{family}` | {any_m['precision']:.6f} | {any_m['recall']:.6f} | {any_m['f1']:.6f} | {center:.6f} | {label_m['f1']:.6f} | {m['prediction_inflation']:.6f} | {any_m['gold']} | {any_m['predicted']} |")
    lines.extend([
        "",
        "## Interpretation",
        "- This path proves raster-only multi-family outputs exist, but its 256x256 supervision format is separate from `public_raster_moe_supervision_v19`.",
        "- Boundary produces many coarse nodes; symbol and text are under-produced relative to gold boxes.",
        "- It is useful for P102 coverage expansion, but not directly comparable to P101 public-raster symbol overlays without a scaling/alignment adapter.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold-boxes", default="reports/vlm/image_only_multitask_proposal_v15_locked_predictions.jsonl")
    parser.add_argument("--predictions", default="reports/vlm/image_only_moe_predictions_v15.jsonl")
    parser.add_argument("--iou-threshold", type=float, default=0.30)
    parser.add_argument("--summary", default="reports/vlm/image_only_multifamily_target_eval_p102.json")
    parser.add_argument("--report", default="reports/vlm/image_only_multifamily_target_eval_p102.md")
    args = parser.parse_args()
    gold_rows = load_jsonl(Path(args.gold_boxes))
    pred_rows = load_jsonl(Path(args.predictions))
    result = evaluate(gold_rows, pred_rows, args.iou_threshold)
    summary = {
        "id": "SCI-P1-102-image-only-multifamily-target-eval",
        "gold_boxes": args.gold_boxes,
        "predictions": args.predictions,
        "iou_threshold": args.iou_threshold,
        "claim_boundary": "256x256 image-only raster supervision target-level boxes; no relation-graph claim",
        **result,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(render(summary), encoding="utf-8")
    print(json.dumps({"summary": args.summary, "matched_rows": result["matched_rows"], "family_metrics": {k: v["iou_any"] for k, v in result["family_metrics"].items()}}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
