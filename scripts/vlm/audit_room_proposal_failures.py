#!/usr/bin/env python3
"""Audit room proposal misses and false positives."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/cadstruct_room_proposals_v1")
    parser.add_argument("--splits", default="locked_test,smoke,dev")
    parser.add_argument("--output", default="reports/vlm/room_proposal_failure_audit_v1.json")
    parser.add_argument("--cases-output", default="reports/vlm/room_proposal_failure_cases_v1.jsonl")
    parser.add_argument("--iou-threshold", type=float, default=0.5)
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]
    all_cases: list[dict[str, Any]] = []
    report: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "room_proposal_failure_audit_v1",
        "dataset_dir": str(dataset_dir),
        "iou_threshold": args.iou_threshold,
        "splits": {},
    }

    for split in splits:
        path = dataset_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        split_report, cases = audit_split(path, split, args.iou_threshold)
        report["splits"][split] = split_report
        all_cases.extend(cases)

    report["summary"] = summarize_splits(report["splits"])
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    cases_output = Path(args.cases_output)
    with cases_output.open("w", encoding="utf-8") as handle:
        for case in all_cases:
            handle.write(json.dumps(case, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "cases": len(all_cases), "splits": list(report["splits"].keys())}, ensure_ascii=False))


def audit_split(path: Path, split: str, threshold: float) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    rows = load_jsonl(path)
    total_rooms = 0
    total_proposals = 0
    matched_rooms = 0
    false_positive_candidates = 0
    miss_reasons = Counter()
    false_positive_reasons = Counter()
    by_source = defaultdict(lambda: {"records": 0, "rooms": 0, "matched": 0, "proposals": 0})
    cases: list[dict[str, Any]] = []

    for row_idx, row in enumerate(rows):
        source = str(row.get("source_dataset") or row.get("source_bucket") or "unknown")
        rooms = [item for item in row.get("rooms") or [] if bbox(item.get("bbox"))]
        proposals = [item for item in row.get("proposals") or [] if bbox(item.get("bbox"))]
        total_rooms += len(rooms)
        total_proposals += len(proposals)
        by_source[source]["records"] += 1
        by_source[source]["rooms"] += len(rooms)
        by_source[source]["proposals"] += len(proposals)

        proposal_best_iou = [0.0 for _ in proposals]
        for room in rooms:
            best_iou, best_proposal = best_match(room, proposals)
            if best_iou >= threshold:
                matched_rooms += 1
                by_source[source]["matched"] += 1
            else:
                reason = classify_miss(room, proposals, best_iou)
                miss_reasons[reason] += 1
                cases.append(case_row("miss", split, row_idx, row, room, best_proposal, best_iou, reason))
            for idx, proposal in enumerate(proposals):
                proposal_best_iou[idx] = max(proposal_best_iou[idx], iou(bbox(room.get("bbox")), bbox(proposal.get("bbox"))))

        for proposal, best_iou_value in zip(proposals, proposal_best_iou):
            if best_iou_value < 0.1:
                false_positive_candidates += 1
                reason = classify_false_positive(proposal)
                false_positive_reasons[reason] += 1
                if len(cases) < 5000:
                    cases.append(case_row("false_positive", split, row_idx, row, None, proposal, best_iou_value, reason))

    recall = matched_rooms / total_rooms if total_rooms else None
    split_report = {
        "records": len(rows),
        "rooms": total_rooms,
        "proposals": total_proposals,
        "matched_rooms_at_iou": matched_rooms,
        "recall_at_iou": recall,
        "avg_proposals_per_record": total_proposals / len(rows) if rows else 0.0,
        "false_positive_candidates_iou_lt_0_1": false_positive_candidates,
        "miss_reasons": dict(miss_reasons),
        "false_positive_reasons": dict(false_positive_reasons),
        "by_source": {
            source: {
                **stats,
                "recall_at_iou": stats["matched"] / stats["rooms"] if stats["rooms"] else None,
            }
            for source, stats in by_source.items()
        },
    }
    return split_report, cases


def classify_miss(room: dict[str, Any], proposals: list[dict[str, Any]], best_iou: float) -> str:
    if not proposals:
        return "closed_region_miss_no_proposals"
    room_box = bbox(room.get("bbox"))
    room_area = area(room_box)
    if best_iou < 0.05:
        return "closed_region_miss_no_overlap"
    best_areas = sorted((area(bbox(item.get("bbox"))) for item in proposals), reverse=True)
    largest_ratio = best_areas[0] / room_area if room_area and best_areas else 0.0
    if largest_ratio > 2.5:
        return "over_merge_candidate_too_large"
    if largest_ratio < 0.35:
        return "under_segment_candidates_too_small"
    return "low_iou_boundary_quality"


def classify_false_positive(proposal: dict[str, Any]) -> str:
    source = str(proposal.get("source") or "")
    box = bbox(proposal.get("bbox"))
    proposal_area = area(box)
    if proposal_area < 1000:
        return "tiny_non_room_fp"
    if source == "component":
        return "wall_component_fp"
    if source == "semantic_region":
        return "semantic_region_non_room_fp"
    return "non_room_fp"


def best_match(room: dict[str, Any], proposals: list[dict[str, Any]]) -> tuple[float, dict[str, Any] | None]:
    room_box = bbox(room.get("bbox"))
    best_iou = 0.0
    best_proposal = None
    for proposal in proposals:
        value = iou(room_box, bbox(proposal.get("bbox")))
        if value > best_iou:
            best_iou = value
            best_proposal = proposal
    return best_iou, best_proposal


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


def case_row(kind: str, split: str, row_idx: int, row: dict[str, Any], room: dict[str, Any] | None, proposal: dict[str, Any] | None, best_iou: float, reason: str) -> dict[str, Any]:
    return {
        "kind": kind,
        "split": split,
        "sample_id": row.get("annotation") or row.get("annotation_path") or row.get("image") or f"row_{row_idx}",
        "source": row.get("source_dataset") or row.get("source_bucket"),
        "room_id": room.get("id") if isinstance(room, dict) else None,
        "room_type": room.get("room_type") if isinstance(room, dict) else None,
        "room_bbox": room.get("bbox") if isinstance(room, dict) else None,
        "proposal_id": proposal.get("id") if isinstance(proposal, dict) else None,
        "proposal_source": proposal.get("source") if isinstance(proposal, dict) else None,
        "proposal_bbox": proposal.get("bbox") if isinstance(proposal, dict) else None,
        "best_iou": best_iou,
        "reason": reason,
    }


def summarize_splits(splits: dict[str, Any]) -> dict[str, Any]:
    rooms = sum((item.get("rooms") or 0) for item in splits.values())
    matched = sum((item.get("matched_rooms_at_iou") or 0) for item in splits.values())
    miss_reasons = Counter()
    fp_reasons = Counter()
    for item in splits.values():
        miss_reasons.update(item.get("miss_reasons") or {})
        fp_reasons.update(item.get("false_positive_reasons") or {})
    return {
        "rooms": rooms,
        "matched_rooms_at_iou": matched,
        "recall_at_iou": matched / rooms if rooms else None,
        "miss_reasons": dict(miss_reasons),
        "false_positive_reasons": dict(fp_reasons),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(item, dict):
                rows.append(item)
    return rows


if __name__ == "__main__":
    main()
