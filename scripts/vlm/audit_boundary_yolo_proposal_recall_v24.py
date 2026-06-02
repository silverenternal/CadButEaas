#!/usr/bin/env python3
"""Audit full-page boundary proposal recall from exported YOLO tile predictions."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from apply_boundary_proposals_with_graph_node_gnn_v24 import (
    BOUNDARY_TO_GRAPH_LABEL,
    LABELS,
    YOLO_CLASS_TO_LABEL,
    bbox,
    center_covered,
    iou,
    load_jsonl,
    parse_tile_id,
    write_json,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATASET = ROOT / "datasets/boundary_expert_public_raster_v19"
DEFAULT_PREDICTIONS = ROOT / "reports/vlm/boundary_public_raster_v24_yolo_full_dev493_predictions.json"
DEFAULT_OUTPUT = ROOT / "reports/vlm/boundary_public_raster_v24_yolo_full_dev493_proposal_audit.json"


def gold_items(row: dict[str, Any]) -> list[dict[str, Any]]:
    items = []
    for target in (row.get("targets") or {}).get("boxes") or []:
        b = bbox(target.get("bbox"))
        label = BOUNDARY_TO_GRAPH_LABEL.get(str(target.get("label")), str(target.get("label")))
        if b is not None and label in LABELS:
            items.append({"bbox": b, "label": label, "target_id": str(target.get("target_id") or "")})
    return items


def load_predictions(path: Path, score_min: float, max_candidates: int) -> dict[str, list[dict[str, Any]]]:
    raw = json.loads(path.read_text(encoding="utf-8"))
    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for item in raw:
        score = float(item.get("score") or 0.0)
        if score < score_min:
            continue
        parsed = parse_tile_id(str(item.get("image_id") or Path(str(item.get("file_name") or "")).stem))
        if parsed is None:
            continue
        row_id, ox, oy = parsed
        raw_box = item.get("bbox")
        if not isinstance(raw_box, list) or len(raw_box) != 4:
            continue
        x, y, w, h = [float(value) for value in raw_box]
        label = YOLO_CLASS_TO_LABEL.get(int(item.get("category_id") or 0), "hard_wall")
        candidate_id = f"boundary_yolo_full_{len(by_row[row_id]):06d}"
        by_row[row_id].append(
            {
                "candidate_id": candidate_id,
                "bbox": [ox + x, oy + y, ox + x + w, oy + y + h],
                "prediction": label,
                "label_hint": label,
                "proposal_source": "boundary_yolo_v24_full",
                "proposal_confidence": round(score, 6),
                "score": score,
            }
        )
    for row_id, preds in list(by_row.items()):
        preds.sort(key=lambda pred: pred["score"], reverse=True)
        by_row[row_id] = preds[:max_candidates]
    return by_row


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_prediction_rows(rows: list[dict[str, Any]], pred_by_row: dict[str, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    output = []
    for row in rows:
        row_id = str(row.get("id"))
        stream = []
        for pred in pred_by_row.get(row_id, []):
            item = dict(pred)
            item["confidence"] = item.pop("score")
            stream.append(item)
        output.append(
            {
                "id": row_id,
                "image": row.get("image"),
                "image_size": row.get("image_size"),
                "source_integrity": {
                    "model_input": "exported_raster_yolo_tile_predictions_only",
                    "gold_loaded_after_inference_for_evaluation_only": False,
                    "svg_or_parser_geometry_used_at_runtime": False,
                },
                "candidate_stream": stream,
            }
        )
    return output


def evaluate(rows: list[dict[str, Any]], pred_by_row: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    proposal_hit = typed_hit = total = predicted = 0
    per_label: dict[str, Counter[str]] = defaultdict(Counter)
    nonempty_rows = 0
    empty_rows = 0
    missed = []
    for row in rows:
        row_id = str(row.get("id"))
        preds = pred_by_row.get(row_id, [])
        if preds:
            nonempty_rows += 1
        else:
            empty_rows += 1
        predicted += len(preds)
        for gold in gold_items(row):
            total += 1
            label = gold["label"]
            per_label[label]["gold"] += 1
            matches = [pred for pred in preds if center_covered(pred["bbox"], gold["bbox"]) or iou(pred["bbox"], gold["bbox"]) >= 0.30]
            if matches:
                proposal_hit += 1
                per_label[label]["proposal_matched"] += 1
            else:
                missed.append({"row_id": row_id, "target_id": gold["target_id"], "label": label, "bbox": gold["bbox"]})
            if any(pred["prediction"] == label for pred in matches):
                typed_hit += 1
                per_label[label]["typed_matched"] += 1
    return {
        "rows": len(rows),
        "nonempty_rows": nonempty_rows,
        "empty_rows": empty_rows,
        "gold": total,
        "predicted": predicted,
        "candidate_inflation": round(predicted / max(total, 1), 6),
        "proposal_recall": round(proposal_hit / max(total, 1), 6),
        "typed_hint_recall": round(typed_hit / max(total, 1), 6),
        "typed_precision_proxy": round(typed_hit / max(predicted, 1), 6),
        "per_label": {
            label: {
                "gold": counts["gold"],
                "proposal_matched": counts["proposal_matched"],
                "typed_matched": counts["typed_matched"],
                "proposal_recall": round(counts["proposal_matched"] / max(counts["gold"], 1), 6),
                "typed_hint_recall": round(counts["typed_matched"] / max(counts["gold"], 1), 6),
            }
            for label, counts in sorted(per_label.items())
        },
        "missed_summary": dict(Counter(item["label"] for item in missed)),
        "missed_examples": missed[:200],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--split", default="dev")
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--predictions-output", default="")
    parser.add_argument("--score-min", type=float, default=0.001)
    parser.add_argument("--max-candidates", type=int, default=800)
    args = parser.parse_args()

    dataset = Path(args.dataset)
    rows = load_jsonl(dataset / f"{args.split}.jsonl", args.limit or None)
    pred_by_row = load_predictions(Path(args.predictions), args.score_min, args.max_candidates)
    report = evaluate(rows, pred_by_row)
    report["source_integrity"] = {
        "model_input": "exported_raster_yolo_tile_predictions_only",
        "gold_loaded_after_inference_for_evaluation_only": True,
        "svg_or_parser_geometry_used_at_runtime": False,
    }
    write_json(Path(args.output), report)
    if args.predictions_output:
        write_jsonl(Path(args.predictions_output), build_prediction_rows(rows, pred_by_row))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
