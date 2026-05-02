#!/usr/bin/env python3
"""Build the normalized RoomProposalExpert dataset.

This keeps the row-level proposal/room contract used by the existing scorer,
but enriches every proposal with keep/suppress, best-IoU, quality, and
merge/split-style labels for audit and future learned scorers.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/cadstruct_room_proposals_v1")
    parser.add_argument("--output-dir", default="datasets/room_proposal_expert")
    parser.add_argument("--output-audit", default="reports/vlm/room_proposal_dataset_audit_v1.json")
    parser.add_argument("--splits", default="train,dev,locked_test,smoke")
    parser.add_argument("--keep-iou", type=float, default=0.5)
    parser.add_argument("--high-quality-iou", type=float, default=0.75)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    splits = [item.strip() for item in args.splits.split(",") if item.strip()]

    audit: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "version": "room_proposal_expert_dataset_v1",
        "input_dir": str(input_dir),
        "output_dir": str(output_dir),
        "keep_iou": args.keep_iou,
        "high_quality_iou": args.high_quality_iou,
        "splits": {},
    }

    for split in splits:
        source_path = input_dir / f"{split}.jsonl"
        if not source_path.exists():
            continue
        rows = []
        split_stats = init_stats()
        for row in load_jsonl(source_path):
            enriched = enrich_row(row, args.keep_iou, args.high_quality_iou)
            rows.append(enriched)
            update_stats(split_stats, enriched)
        write_jsonl(output_dir / f"{split}.jsonl", rows)
        audit["splits"][split] = finalize_stats(split_stats)

    manifest = {
        "version": "room_proposal_expert_v1",
        "created_at_utc": audit["created_at_utc"],
        "source": str(input_dir),
        "contract": {
            "row_fields": ["image", "annotation", "source_dataset", "source_bucket", "proposals", "rooms"],
            "proposal_training_fields": [
                "best_iou",
                "keep_label",
                "quality_label",
                "failure_label",
                "nearest_room_id",
                "nearest_room_type",
            ],
        },
        "splits": {
            split: {
                "records": stats["records"],
                "rooms": stats["rooms"],
                "proposals": stats["proposals"],
                "keep_positive": stats["keep_positive"],
                "suppress_negative": stats["suppress_negative"],
            }
            for split, stats in audit["splits"].items()
        },
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    output_audit = Path(args.output_audit)
    output_audit.parent.mkdir(parents=True, exist_ok=True)
    output_audit.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "audit": str(output_audit), "splits": list(audit["splits"].keys())}, ensure_ascii=False))


def enrich_row(row: dict[str, Any], keep_iou: float, high_quality_iou: float) -> dict[str, Any]:
    rooms = [room for room in row.get("rooms") or [] if bbox(room.get("bbox"))]
    proposals = []
    for proposal in row.get("proposals") or []:
        if not bbox(proposal.get("bbox")):
            continue
        best_iou, nearest_room = best_match(proposal, rooms)
        keep = best_iou >= keep_iou
        enriched = dict(proposal)
        metadata = dict(enriched.get("metadata") or {})
        metadata.update(
            {
                "best_iou": best_iou,
                "keep_label": int(keep),
                "quality_label": quality_label(best_iou, keep_iou, high_quality_iou),
                "failure_label": failure_label(proposal, nearest_room, best_iou, keep_iou),
                "nearest_room_id": nearest_room.get("id") if nearest_room else None,
                "nearest_room_type": nearest_room.get("room_type") if nearest_room else None,
            }
        )
        enriched["metadata"] = metadata
        enriched["best_iou"] = best_iou
        enriched["keep_label"] = int(keep)
        enriched["quality_label"] = metadata["quality_label"]
        enriched["failure_label"] = metadata["failure_label"]
        proposals.append(enriched)
    output = dict(row)
    output["proposals"] = proposals
    output["proposal_training_contract"] = {
        "keep_iou": keep_iou,
        "labels": {
            "keep_label": "1 when best_iou >= keep_iou else 0",
            "quality_label": "0 suppress, 1 partial, 2 keep, 3 high_quality",
            "failure_label": "coarse miss/fp/quality category for audit",
        },
    }
    return output


def best_match(proposal: dict[str, Any], rooms: list[dict[str, Any]]) -> tuple[float, dict[str, Any] | None]:
    proposal_box = bbox(proposal.get("bbox"))
    best_iou = 0.0
    best_room = None
    for room in rooms:
        value = iou(proposal_box, bbox(room.get("bbox")))
        if value > best_iou:
            best_iou = value
            best_room = room
    return best_iou, best_room


def quality_label(best_iou: float, keep_iou: float, high_quality_iou: float) -> int:
    if best_iou >= high_quality_iou:
        return 3
    if best_iou >= keep_iou:
        return 2
    if best_iou >= 0.1:
        return 1
    return 0


def failure_label(proposal: dict[str, Any], nearest_room: dict[str, Any] | None, best_iou: float, keep_iou: float) -> str:
    if best_iou >= keep_iou:
        return "keep"
    if best_iou < 0.05:
        return "non_room_fp"
    if nearest_room is None:
        return "no_nearest_room"
    proposal_area = area(bbox(proposal.get("bbox")))
    room_area = area(bbox(nearest_room.get("bbox")))
    ratio = proposal_area / room_area if room_area else 0.0
    if ratio > 2.5:
        return "over_merge"
    if ratio < 0.35:
        return "under_segment"
    return "low_iou_boundary_quality"


def init_stats() -> dict[str, Any]:
    return {
        "records": 0,
        "rooms": 0,
        "proposals": 0,
        "keep_positive": 0,
        "suppress_negative": 0,
        "source_counts": Counter(),
        "quality_counts": Counter(),
        "failure_counts": Counter(),
        "best_iou_histogram": Counter(),
    }


def update_stats(stats: dict[str, Any], row: dict[str, Any]) -> None:
    stats["records"] += 1
    stats["rooms"] += len(row.get("rooms") or [])
    stats["source_counts"][str(row.get("source_dataset") or row.get("source_bucket") or "unknown")] += 1
    for proposal in row.get("proposals") or []:
        stats["proposals"] += 1
        if proposal.get("keep_label"):
            stats["keep_positive"] += 1
        else:
            stats["suppress_negative"] += 1
        stats["quality_counts"][str(proposal.get("quality_label"))] += 1
        stats["failure_counts"][str(proposal.get("failure_label"))] += 1
        stats["best_iou_histogram"][hist_bucket(float(proposal.get("best_iou") or 0.0))] += 1


def finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    total = max(int(stats["proposals"]), 1)
    return {
        "records": stats["records"],
        "rooms": stats["rooms"],
        "proposals": stats["proposals"],
        "keep_positive": stats["keep_positive"],
        "suppress_negative": stats["suppress_negative"],
        "positive_rate": stats["keep_positive"] / total,
        "source_counts": dict(stats["source_counts"]),
        "quality_counts": dict(stats["quality_counts"]),
        "failure_counts": dict(stats["failure_counts"]),
        "best_iou_histogram": dict(sorted(stats["best_iou_histogram"].items())),
    }


def hist_bucket(value: float) -> str:
    lo = int(max(0.0, min(0.99, value)) * 10) / 10
    return f"{lo:.1f}-{lo + 0.1:.1f}"


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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            if isinstance(item, dict):
                rows.append(item)
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
