#!/usr/bin/env python3
"""Fast top-K page-level scoring for large P212 prediction jsonl files."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fuse_symbol_p206g_with_p211_p212 import load_p206g, score_predictions, write_json

ROOT = Path(__file__).resolve().parents[2]
P206G = ROOT / "reports/vlm/symbol_p206f_precision_repair_p206g_overlay.jsonl"
PREDICTIONS = ROOT / "reports/vlm/symbol_p211_20k_yolov8s_p206g_pages_sliced256_img768_predictions.jsonl"
REPORT = ROOT / "reports/vlm/symbol_p211_20k_yolov8s_p206g_pages_sliced256_img768_fast_eval.json"


def load_predictions(path: Path, topk: int, min_score: float) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            preds = [pred for pred in row.get("predicted_symbols") or [] if float(pred.get("score", 0.0)) >= min_score]
            preds.sort(key=lambda pred: float(pred.get("score", 0.0)), reverse=True)
            out[str(row.get("row_id"))] = preds[:topk]
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p206g", default=str(P206G))
    parser.add_argument("--predictions", default=str(PREDICTIONS))
    parser.add_argument("--output", default=str(REPORT))
    parser.add_argument("--topk", type=int, default=300)
    parser.add_argument("--min-score", type=float, default=0.001)
    parser.add_argument("--score-threshold-grid", default="0.05,0.1,0.2")
    parser.add_argument("--nms-threshold-grid", default="0.25,0.35")
    parser.add_argument("--max-per-page", type=int, default=900)
    args = parser.parse_args()
    _rows, _core, golds = load_p206g(Path(args.p206g))
    preds = load_predictions(Path(args.predictions), args.topk, args.min_score)
    grid = []
    for score in [float(x) for x in args.score_threshold_grid.split(",") if x.strip()]:
        for nms in [float(x) for x in args.nms_threshold_grid.split(",") if x.strip()]:
            metrics, _ = score_predictions(preds, golds, score, nms, args.max_per_page, 0)
            grid.append({"score_threshold": score, "nms_threshold": nms, "metrics": metrics})
    grid.sort(key=lambda row: (row["metrics"]["symbol_bbox_iou_0_30"]["f1"], row["metrics"]["symbol_bbox_iou_0_30"]["recall"]), reverse=True)
    report = {
        "id": "P212_fast_prediction_jsonl_eval",
        "predictions": str(Path(args.predictions)),
        "topk": args.topk,
        "min_score": args.min_score,
        "selected": grid[0],
        "threshold_grid": grid,
        "claim_boundary": "Fast top-K planning eval over precomputed predictions; use for direction finding, not final paper claim.",
    }
    write_json(Path(args.output), report)
    print(json.dumps({"selected": grid[0], "topk": args.topk}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
