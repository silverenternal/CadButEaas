#!/usr/bin/env python3
"""Fine score-threshold sweep over P119 selected predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from analyze_symbol_subset_failure_p117 import bbox_iou, center_covered, reconstruct_page_golds


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = ROOT / "reports/vlm/full_public_raster_symbol_eval_subset_p119_expanded_grid.json"
DEFAULT_PREDICTIONS = ROOT / "reports/vlm/full_public_raster_symbol_eval_subset_p119_expanded_grid_predictions.jsonl"
DEFAULT_JSON = ROOT / "configs/vlm/symbol_score_fine_compression_p120.json"
DEFAULT_REPORT = ROOT / "reports/vlm/symbol_score_fine_compression_p120.md"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    return {str(row.get("row_id")): list(row.get("predicted_symbols") or []) for row in load_jsonl(path)}


def area_bucket(box: list[float]) -> str:
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area <= 64:
        return "tiny_le_64"
    if area <= 256:
        return "small_le_256"
    if area <= 1024:
        return "medium_le_1024"
    if area <= 4096:
        return "large_le_4096"
    return "xlarge_gt_4096"


def score(page_golds: dict[str, dict[str, dict[str, Any]]], page_preds: dict[str, list[dict[str, Any]]], threshold: float) -> dict[str, Any]:
    totals = Counter()
    by_area = Counter()
    by_area_iou = Counter()
    typed_correct = 0
    for row_id, gold_map in page_golds.items():
        preds = [pred for pred in page_preds.get(row_id, []) if float(pred.get("score") or 0.0) >= threshold]
        used_iou: set[int] = set()
        used_center: set[int] = set()
        for gold in gold_map.values():
            gold_box = [float(v) for v in gold["bbox"]]
            bucket = area_bucket(gold_box)
            by_area[bucket] += 1
            best_iou = 0.0
            best_iou_index = None
            center_index = None
            for pred_index, pred in enumerate(preds):
                pred_box = [float(v) for v in pred["bbox"]]
                current_iou = bbox_iou(pred_box, gold_box)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_iou_index = pred_index
                if center_index is None and pred_index not in used_center and center_covered(pred_box, gold_box):
                    center_index = pred_index
            if best_iou_index is not None and best_iou >= 0.30 and best_iou_index not in used_iou:
                used_iou.add(best_iou_index)
                totals["matched_iou"] += 1
                by_area_iou[bucket] += 1
                if str(preds[best_iou_index].get("label")) == str(gold.get("label")):
                    typed_correct += 1
            if center_index is not None:
                used_center.add(center_index)
                totals["matched_center"] += 1
        totals["gold"] += len(gold_map)
        totals["predicted"] += len(preds)
    precision = totals["matched_iou"] / max(totals["predicted"], 1)
    recall = totals["matched_iou"] / max(totals["gold"], 1)
    return {
        "score_threshold": threshold,
        "matched": int(totals["matched_iou"]),
        "predicted": int(totals["predicted"]),
        "gold": int(totals["gold"]),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall), 6),
        "center_recall": round(totals["matched_center"] / max(totals["gold"], 1), 6),
        "candidate_inflation": round(totals["predicted"] / max(totals["gold"], 1), 6),
        "typed_accuracy_on_iou_matches": round(typed_correct / max(totals["matched_iou"], 1), 6),
        "area_iou_recall": {bucket: round(by_area_iou[bucket] / max(by_area[bucket], 1), 6) for bucket in sorted(by_area)},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default=str(DEFAULT_EVAL))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--thresholds", default="0.005,0.006,0.007,0.008,0.009,0.010")
    parser.add_argument("--output", default=str(DEFAULT_JSON))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    eval_path = Path(args.eval)
    eval_data = json.loads(eval_path.read_text(encoding="utf-8"))
    page_golds = reconstruct_page_golds(eval_data)
    page_preds = load_predictions(Path(args.predictions))
    thresholds = [float(item) for item in args.thresholds.split(",") if item.strip()]
    rows = [score(page_golds, page_preds, threshold) for threshold in thresholds]
    for row in rows:
        tiny = (row.get("area_iou_recall") or {}).get("tiny_le_64", 0.0)
        row["tiny_iou_recall"] = tiny
        row["passes_center"] = row["center_recall"] > 0.851394
        row["passes_tiny_iou"] = tiny > 0.393013
        row["passes_inflation"] = row["candidate_inflation"] <= 7.919152
        row["passes_all_baseline_gates"] = row["passes_center"] and row["passes_tiny_iou"] and row["passes_inflation"]
    passing = [row for row in rows if row["passes_all_baseline_gates"]]
    best = max(rows, key=lambda row: (row["passes_all_baseline_gates"], row["center_recall"], row["tiny_iou_recall"], -row["candidate_inflation"])) if rows else None
    summary = {
        "id": "SCI-P2-120-symbol-score-fine-compression-sweep",
        "claim_boundary": "Fine score compression sweep over P119 subset predictions only; not a full locked quality claim.",
        "inputs": {"eval": rel(eval_path), "predictions": rel(Path(args.predictions))},
        "baseline_gates": {"center_recall_gt": 0.851394, "tiny_iou_recall_gt": 0.393013, "candidate_inflation_lte": 7.919152},
        "rows": rows,
        "passing_rows": passing,
        "best_row": best,
        "decision": "fine_score_threshold_passes_subset_gates" if passing else "fine_score_threshold_does_not_pass_subset_gates",
        "recommendation": (
            "Validate the passing fine threshold with a proper inference/eval run and consider candidate selector adoption."
            if passing
            else "Move to support-aware candidate compression or localization/tiny-box repair; simple score threshold is insufficient."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# P2-120 Symbol Score Fine Compression Sweep",
        "",
        "## Decision",
        "",
        f"- Decision: `{summary['decision']}`",
        f"- Passing rows: `{len(passing)}`",
        f"- Recommendation: {summary['recommendation']}",
        "",
        "## Sweep",
        "",
        "| Score | Center | Tiny IoU | Inflation | F1 | Precision | Recall | Pass |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['score_threshold']:.3f} | {row['center_recall']:.6f} | {row['tiny_iou_recall']:.6f} | "
            f"{row['candidate_inflation']:.6f} | {row['f1']:.6f} | {row['precision']:.6f} | {row['recall']:.6f} | "
            f"{str(row['passes_all_baseline_gates']).lower()} |"
        )
    lines.extend(["", "## Interpretation", ""])
    if passing:
        lines.append("- A fine score threshold can satisfy the subset gates on the P119 selected prediction pool.")
        lines.append("- Because this reuses NMS=0.75 selected predictions, confirm with a proper eval run before adoption.")
    else:
        lines.append("- No fine score threshold satisfies center, tiny IoU, and inflation gates together.")
        lines.append("- The next useful step is support-aware compression/listwise selection or detector localization repair.")
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": rel(output), "report": rel(report), "decision": summary["decision"], "passing_rows": len(passing), "best_row": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
