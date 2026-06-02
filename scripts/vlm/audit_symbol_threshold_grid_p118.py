#!/usr/bin/env python3
"""Audit symbol threshold-grid tradeoffs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = ROOT / "reports/vlm/full_public_raster_symbol_eval_subset_p113_p116_subset_20260516_142010.json"
DEFAULT_JSON = ROOT / "configs/vlm/symbol_threshold_grid_tradeoff_p118.json"
DEFAULT_REPORT = ROOT / "reports/vlm/symbol_threshold_grid_tradeoff_p118.md"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def metric(row: dict[str, Any], key: str, default: float = 0.0) -> float:
    metrics = row.get("metrics") or {}
    if key == "precision":
        return float((metrics.get("symbol_bbox_iou_0_30") or {}).get("precision") or default)
    if key == "recall":
        return float((metrics.get("symbol_bbox_iou_0_30") or {}).get("recall") or default)
    if key == "f1":
        return float((metrics.get("symbol_bbox_iou_0_30") or {}).get("f1") or default)
    if key == "center":
        return float(metrics.get("symbol_bbox_center_recall") or default)
    if key == "tiny_iou":
        return float((metrics.get("area_iou_recall") or {}).get("tiny_le_64") or default)
    if key == "inflation":
        return float(metrics.get("candidate_inflation") or default)
    return default


def compact(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "score_threshold": float(row.get("score_threshold")),
        "nms_threshold": float(row.get("nms_threshold")),
        "precision": metric(row, "precision"),
        "recall": metric(row, "recall"),
        "f1": metric(row, "f1"),
        "center_recall": metric(row, "center"),
        "tiny_iou_recall": metric(row, "tiny_iou"),
        "candidate_inflation": metric(row, "inflation"),
        "passes_center": metric(row, "center") > 0.851394,
        "passes_tiny_iou": metric(row, "tiny_iou") > 0.393013,
        "passes_inflation": metric(row, "inflation") <= 7.919152,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default=str(DEFAULT_EVAL))
    parser.add_argument("--output", default=str(DEFAULT_JSON))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--id", default="SCI-P2-118-symbol-threshold-grid-tradeoff-audit")
    parser.add_argument("--title", default="P2-118 Symbol Threshold Grid Tradeoff Audit")
    args = parser.parse_args()

    eval_path = Path(args.eval)
    data = json.loads(eval_path.read_text(encoding="utf-8"))
    grid = [compact(row) for row in data.get("threshold_grid") or []]
    for row in grid:
        row["passes_all_baseline_gates"] = bool(row["passes_center"] and row["passes_tiny_iou"] and row["passes_inflation"])
    selected = data.get("selected_thresholds") or {}
    selected_row = next(
        (
            row for row in grid
            if row["score_threshold"] == float(selected.get("score_threshold"))
            and row["nms_threshold"] == float(selected.get("nms_threshold"))
        ),
        None,
    )
    passing = [row for row in grid if row["passes_all_baseline_gates"]]
    top_by = {
        "center_recall": sorted(grid, key=lambda row: (row["center_recall"], row["tiny_iou_recall"], -row["candidate_inflation"]), reverse=True)[:8],
        "tiny_iou_recall": sorted(grid, key=lambda row: (row["tiny_iou_recall"], row["center_recall"], -row["candidate_inflation"]), reverse=True)[:8],
        "f1": sorted(grid, key=lambda row: (row["f1"], row["precision"], row["recall"]), reverse=True)[:8],
        "low_inflation": sorted(grid, key=lambda row: (row["candidate_inflation"], -row["f1"]))[:8],
    }
    best_center = top_by["center_recall"][0] if top_by["center_recall"] else {}
    best_tiny = top_by["tiny_iou_recall"][0] if top_by["tiny_iou_recall"] else {}
    decision = "existing_threshold_can_pass_gates" if passing else "no_existing_threshold_passes_center_tiny_gates"
    summary = {
        "id": args.id,
        "claim_boundary": "Subset threshold-grid audit only; not a full locked model-quality claim.",
        "eval": rel(eval_path),
        "baseline_gates": {
            "center_recall_gt": 0.851394,
            "tiny_iou_recall_gt": 0.393013,
            "candidate_inflation_lte": 7.919152,
        },
        "decision": decision,
        "grid_rows": len(grid),
        "selected_row": selected_row,
        "passing_rows": passing,
        "top_by": top_by,
        "best_center_gap": round(float(best_center.get("center_recall", 0.0)) - 0.851394, 6) if best_center else None,
        "best_tiny_iou_gap": round(float(best_tiny.get("tiny_iou_recall", 0.0)) - 0.393013, 6) if best_tiny else None,
        "recommendation": (
            "This threshold grid has no operating point that passes center/tiny/inflation gates; prioritize candidate compression or localization/tiny-box repair before full locked quality claims."
            if not passing
            else "At least one existing operating point passes gates; validate it on a larger/full run before adoption."
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"# {args.title}",
        "",
        "## Decision",
        "",
        f"- Decision: `{decision}`",
        f"- Grid rows: `{len(grid)}`",
        f"- Passing rows: `{len(passing)}`",
        f"- Recommendation: {summary['recommendation']}",
        "",
        "## Selected Row",
        "",
        "```json",
        json.dumps(selected_row, ensure_ascii=False, indent=2),
        "```",
        "",
        "## Top By Center Recall",
        "",
        "| Score | NMS | Center | Tiny IoU | F1 | Inflation |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for row in top_by["center_recall"][:6]:
        lines.append(f"| {row['score_threshold']} | {row['nms_threshold']} | {row['center_recall']:.6f} | {row['tiny_iou_recall']:.6f} | {row['f1']:.6f} | {row['candidate_inflation']:.6f} |")
    lines.extend(["", "## Top By Tiny IoU Recall", "", "| Score | NMS | Tiny IoU | Center | F1 | Inflation |", "|---:|---:|---:|---:|---:|---:|"])
    for row in top_by["tiny_iou_recall"][:6]:
        lines.append(f"| {row['score_threshold']} | {row['nms_threshold']} | {row['tiny_iou_recall']:.6f} | {row['center_recall']:.6f} | {row['f1']:.6f} | {row['candidate_inflation']:.6f} |")
    lines.extend(["", "## Interpretation", ""])
    if passing:
        lines.append("- At least one grid row passes the configured subset gates; the next step is validation at larger scale.")
    else:
        lines.append("- No grid row passes center, tiny IoU, and candidate-inflation gates at the same time.")
        lines.append("- Threshold tuning alone is unlikely to resolve the subset blocker; candidate compression/selector or localization repair is needed.")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": rel(output), "report": rel(Path(args.report)), "decision": decision, "passing_rows": len(passing)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
