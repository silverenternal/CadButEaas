#!/usr/bin/env python3
"""Audit whether current room proposals can support end-to-end RoomSpace."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-dir", default="checkpoints/room_proposal_scorer_v1")
    parser.add_argument("--room-report", default="reports/vlm/moe/room_space_context_predicted_upstream_dev.json")
    parser.add_argument("--output", default="reports/vlm/room_space_e2e_v1_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/room_space_e2e_v1_predictions.jsonl")
    parser.add_argument("--error-output", default="reports/vlm/room_space_e2e_v1_error_audit.json")
    parser.add_argument("--splits", default="locked_test,smoke,train")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    proposal_dir = Path(args.proposal_dir)
    split_reports = {}
    prediction_rows = []
    error_rows = []
    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        path = proposal_dir / f"{split}_predictions.jsonl"
        if not path.exists():
            continue
        report, predictions, errors = audit_split(path, split, args.iou_threshold)
        split_reports[split] = report
        prediction_rows.extend(predictions)
        error_rows.extend(errors)

    room_report = load_json(Path(args.room_report))
    inherited_room_macro_f1 = room_report.get("macro_f1")
    primary_split = "locked_test" if "locked_test" in split_reports else next(iter(split_reports), None)
    primary_recall = split_reports.get(primary_split, {}).get("room_recall_at_iou") if primary_split else None
    e2e_estimate = primary_recall * inherited_room_macro_f1 if isinstance(primary_recall, (int, float)) and isinstance(inherited_room_macro_f1, (int, float)) else None

    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "room_space_e2e_v1_proposal_bottleneck_audit",
        "proposal_dir": str(proposal_dir),
        "room_report": args.room_report,
        "iou_threshold": args.iou_threshold,
        "splits": split_reports,
        "inherited_gold_room_type_macro_f1": inherited_room_macro_f1,
        "effective_e2e_macro_f1_estimate": e2e_estimate,
        "target_first_milestone": 0.70,
        "status": "blocked",
        "finding": "Current keep/suppress scorer ranks many non-room primitive components ahead of true room regions; end-to-end RoomSpace cannot improve until proposal ranking is replaced.",
        "next_bottleneck": "Train a supervised ranking/keep head with strong negative mining and candidate-source-aware calibration.",
    }

    write_json(Path(args.output), output)
    write_jsonl(Path(args.predictions_output), prediction_rows)
    error_audit = {
        "created_at_utc": output["created_at_utc"],
        "version": "room_space_e2e_v1_error_audit",
        "errors": len(error_rows),
        "top_error_reasons": dict(Counter(row["reason"] for row in error_rows).most_common(20)),
        "cases_path": args.predictions_output,
        "sample_errors": error_rows[:200],
    }
    write_json(Path(args.error_output), error_audit)
    print(json.dumps({"output": args.output, "effective_e2e_macro_f1_estimate": e2e_estimate, "status": output["status"]}, ensure_ascii=False))


def audit_split(path: Path, split: str, threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    rows = load_jsonl(path)
    rooms = 0
    matched_rooms = 0
    pred_keep = 0
    true_positive_proposals = 0
    source_counts = Counter()
    error_reasons = Counter()
    prediction_rows = []
    error_rows = []
    for row_index, row in enumerate(rows):
        kept = [proposal for proposal in row.get("proposals") or [] if proposal.get("pred_keep")]
        pred_keep += len(kept)
        source = row.get("source_dataset") or row.get("source_bucket") or "unknown"
        source_counts[str(source)] += 1
        room_predictions = []
        for room in row.get("rooms") or []:
            rooms += 1
            best_iou, best_proposal = best_match(room, kept)
            matched = best_iou >= threshold
            matched_rooms += int(matched)
            if not matched:
                reason = classify_room_miss(best_iou, kept)
                error_reasons[reason] += 1
                error_rows.append(error_row(split, row, row_index, room, best_proposal, best_iou, reason))
            room_predictions.append(
                {
                    "id": room.get("id"),
                    "gold": room.get("room_type"),
                    "prediction": "room_proposal_matched" if matched else "room_proposal_missed",
                    "confidence": best_proposal.get("score_keep") if isinstance(best_proposal, dict) else 0.0,
                    "bbox": room.get("bbox"),
                    "iou": best_iou,
                }
            )
        for proposal in kept:
            if float(proposal.get("best_iou") or 0.0) >= threshold:
                true_positive_proposals += 1
        prediction_rows.append(
            {
                "split": split,
                "image": row.get("image"),
                "annotation": row.get("annotation"),
                "source_dataset": row.get("source_dataset"),
                "rooms": room_predictions,
            }
        )
    return (
        {
            "records": len(rows),
            "rooms": rooms,
            "pred_keep": pred_keep,
            "matched_rooms_at_iou": matched_rooms,
            "room_recall_at_iou": matched_rooms / rooms if rooms else None,
            "proposal_precision_at_iou": true_positive_proposals / pred_keep if pred_keep else None,
            "source_counts": dict(source_counts),
            "error_reasons": dict(error_reasons),
        },
        prediction_rows,
        error_rows,
    )


def classify_room_miss(best_iou: float, kept: list[dict[str, Any]]) -> str:
    if not kept:
        return "no_kept_proposals"
    if best_iou < 0.05:
        return "kept_proposals_no_overlap"
    return "kept_proposals_low_iou"


def best_match(room: dict[str, Any], proposals: list[dict[str, Any]]) -> tuple[float, dict[str, Any] | None]:
    best_iou = 0.0
    best_proposal = None
    for proposal in proposals:
        value = iou(bbox(room.get("bbox")), bbox(proposal.get("bbox")))
        if value > best_iou:
            best_iou = value
            best_proposal = proposal
    return best_iou, best_proposal


def error_row(split: str, row: dict[str, Any], row_index: int, room: dict[str, Any], proposal: dict[str, Any] | None, best_iou: float, reason: str) -> dict[str, Any]:
    return {
        "split": split,
        "sample_id": row.get("annotation") or row.get("image") or f"row_{row_index}",
        "source": row.get("source_dataset") or row.get("source_bucket"),
        "room_id": room.get("id"),
        "room_type": room.get("room_type"),
        "room_bbox": room.get("bbox"),
        "best_proposal_id": proposal.get("id") if isinstance(proposal, dict) else None,
        "best_proposal_source": proposal.get("source") if isinstance(proposal, dict) else None,
        "best_iou": best_iou,
        "reason": reason,
    }


def iou(a: list[float] | None, b: list[float] | None) -> float:
    if not a or not b:
        return 0.0
    x0 = max(a[0], b[0])
    y0 = max(a[1], b[1])
    x1 = min(a[2], b[2])
    y1 = min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = area(a) + area(b) - inter
    return inter / union if union > 0 else 0.0


def area(box: list[float] | None) -> float:
    if not box:
        return 0.0
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        x0, y0, x1, y1 = [float(item) for item in value]
    except (TypeError, ValueError):
        return None
    if x1 < x0:
        x0, x1 = x1, x0
    if y1 < y0:
        y0, y1 = y1, y0
    return [x0, y0, x1, y1]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
