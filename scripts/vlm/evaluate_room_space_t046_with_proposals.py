#!/usr/bin/env python3
"""Evaluate selected T046 RoomSpace predictions through v2 proposals.

This audit joins the selected gold-polygon RoomSpaceExpert predictions to the
kept RoomProposalExpert boxes.  It separates proposal coverage from room-type
classification so the end-to-end gap is attributed correctly.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    from train_room_space_expert import evaluate_predictions
except ImportError:
    from scripts.vlm.train_room_space_expert import evaluate_predictions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal-dir", default="checkpoints/room_proposal_ranker_v2")
    parser.add_argument("--t046-dir", default="checkpoints/cadstruct_moe_room_space_hierarchical_sklearn_v5_t046")
    parser.add_argument("--splits", default="dev,locked_test,smoke")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    parser.add_argument("--output", default="reports/vlm/room_space_t046_proposal_e2e_v1_eval.json")
    parser.add_argument("--predictions-output", default="reports/vlm/room_space_t046_proposal_e2e_v1_predictions.jsonl")
    parser.add_argument("--error-output", default="reports/vlm/room_space_t046_proposal_e2e_v1_error_audit.json")
    args = parser.parse_args()

    proposal_dir = Path(args.proposal_dir)
    t046_dir = Path(args.t046_dir)
    all_predictions: list[dict[str, Any]] = []
    all_errors: list[dict[str, Any]] = []
    split_reports: dict[str, Any] = {}

    for split in [item.strip() for item in args.splits.split(",") if item.strip()]:
        proposal_path = proposal_dir / f"{split}_predictions.jsonl"
        t046_path = t046_dir / f"{split}_predictions.jsonl"
        if not proposal_path.exists() or not t046_path.exists():
            continue
        report, predictions, errors = evaluate_split(proposal_path, t046_path, split, args.iou_threshold)
        split_reports[split] = report
        all_predictions.extend(predictions)
        all_errors.extend(errors)

    primary = split_reports.get("locked_test") or split_reports.get("dev") or {}
    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "room_space_t046_proposal_e2e_v1",
        "proposal_dir": str(proposal_dir),
        "t046_dir": str(t046_dir),
        "iou_threshold": args.iou_threshold,
        "splits": split_reports,
        "status": "first_milestone_pass" if (primary.get("macro_f1") or 0.0) >= 0.90 and (primary.get("proposal_recall_at_iou") or 0.0) >= 0.98 else "needs_review",
        "finding": "T046 type path is evaluated after proposal matching; proposal miss, type error, and matching ambiguity are reported separately.",
    }
    write_json(Path(args.output), output)
    write_jsonl(Path(args.predictions_output), all_predictions)
    error_audit = {
        "created_at_utc": output["created_at_utc"],
        "version": "room_space_t046_proposal_e2e_v1_error_audit",
        "errors": len(all_errors),
        "top_error_reasons": dict(Counter(item["reason"] for item in all_errors).most_common(30)),
        "sample_errors": all_errors[:300],
    }
    write_json(Path(args.error_output), error_audit)
    print(json.dumps({"output": args.output, "status": output["status"], "primary": primary}, ensure_ascii=False))


def evaluate_split(proposal_path: Path, t046_path: Path, split: str, threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    proposal_rows = {row_key(row): row for row in load_jsonl(proposal_path)}
    t046_rows = load_jsonl(t046_path)
    predictions = []
    errors = []
    proposal_misses = 0
    matched_rooms = 0
    total_rooms = 0
    matched_ious: list[float] = []

    for row_index, row in enumerate(t046_rows):
        key = row_key(row)
        proposal_row = proposal_rows.get(key)
        kept = [item for item in (proposal_row or {}).get("proposals", []) if item.get("pred_keep")]
        room_predictions = []
        for room in row.get("rooms") or []:
            total_rooms += 1
            best_iou, best_proposal = best_match(room, kept)
            matched = best_iou >= threshold
            if matched:
                matched_rooms += 1
                matched_ious.append(best_iou)
                item = dict(room)
                item["proposal_iou"] = best_iou
                item["proposal_id"] = best_proposal.get("id") if isinstance(best_proposal, dict) else None
                item["proposal_score"] = best_proposal.get("score_keep") if isinstance(best_proposal, dict) else None
                room_predictions.append(item)
                if item.get("gold") != item.get("prediction"):
                    errors.append(error_row(split, row, row_index, item, "type_error_after_proposal_match", best_iou, best_proposal))
            else:
                proposal_misses += 1
                missed = {
                    "id": room.get("id"),
                    "gold": room.get("gold"),
                    "prediction": "__proposal_miss__",
                    "confidence": 0.0,
                    "bbox": room.get("bbox"),
                    "iou": best_iou,
                    "proposal_iou": best_iou,
                }
                room_predictions.append(missed)
                errors.append(error_row(split, row, row_index, room, "proposal_miss", best_iou, best_proposal))
        predictions.append({"split": split, "image": row.get("image"), "annotation": row.get("annotation"), "source_dataset": row.get("source_dataset"), "rooms": room_predictions})

    metrics = evaluate_predictions(predictions)
    metrics.update(
        {
            "records": len(t046_rows),
            "proposal_recall_at_iou": matched_rooms / total_rooms if total_rooms else None,
            "proposal_misses": proposal_misses,
            "matched_rooms": matched_rooms,
            "mean_matched_iou": sum(matched_ious) / len(matched_ious) if matched_ious else None,
        }
    )
    return metrics, predictions, errors


def best_match(room: dict[str, Any], proposals: list[dict[str, Any]]) -> tuple[float, dict[str, Any] | None]:
    best_iou = 0.0
    best_proposal = None
    for proposal in proposals:
        value = iou(bbox(room.get("bbox")), bbox(proposal.get("bbox")))
        if value > best_iou:
            best_iou = value
            best_proposal = proposal
    return best_iou, best_proposal


def error_row(split: str, row: dict[str, Any], row_index: int, room: dict[str, Any], reason: str, best_iou: float, proposal: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "split": split,
        "sample_id": row.get("annotation") or row.get("image") or f"row_{row_index}",
        "source": row.get("source_dataset"),
        "room_id": room.get("id"),
        "target": room.get("gold"),
        "prediction": room.get("prediction"),
        "confidence": room.get("confidence"),
        "room_bbox": room.get("bbox"),
        "proposal_id": proposal.get("id") if isinstance(proposal, dict) else None,
        "proposal_source": proposal.get("source") if isinstance(proposal, dict) else None,
        "proposal_iou": best_iou,
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


def row_key(row: dict[str, Any]) -> str:
    return str(row.get("annotation") or row.get("annotation_path") or row.get("image") or row.get("image_path") or "")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
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
