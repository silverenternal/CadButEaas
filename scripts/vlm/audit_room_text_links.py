#!/usr/bin/env python3
"""Audit room-label text content linked to CubiCasa room candidates."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from room_text_lexicon import ROOM_TEXT_KEYWORDS, room_text_keyword_matches
except ImportError:
    from scripts.vlm.room_text_lexicon import ROOM_TEXT_KEYWORDS, room_text_keyword_matches


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/cadstruct_cubicasa5k_moe_locked")
    parser.add_argument("--output", default="reports/vlm/room_space_text_link_audit.json")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    report = {
        "dataset_dir": str(dataset_dir),
        "keyword_policy": ROOM_TEXT_KEYWORDS,
        "splits": {},
    }
    for split in ("train", "dev", "locked_test", "smoke"):
        path = dataset_dir / f"{split}.jsonl"
        if path.exists():
            report["splits"][split] = audit_split(load_jsonl(path))

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def audit_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total_rooms = 0
    rooms_with_label_text = 0
    rooms_with_keyword_match = 0
    per_label = defaultdict(lambda: Counter())
    examples = []
    for row in rows:
        expected = row.get("expected_json") or {}
        texts = [
            text
            for text in expected.get("text_candidates") or []
            if text.get("text_type") == "room_label" and str(text.get("text") or "").strip()
        ]
        for room in expected.get("room_candidates") or []:
            bbox = normalize_bbox(room.get("bbox"))
            if bbox is None:
                continue
            label = str(room.get("room_type") or "room")
            linked = [text for text in texts if text_center_inside(bbox, text)]
            total_rooms += 1
            per_label[label]["rooms"] += 1
            if linked:
                rooms_with_label_text += 1
                per_label[label]["rooms_with_label_text"] += 1
            keyword_hit = any(room_text_keyword_matches(label, text.get("text") or "") for text in linked)
            if keyword_hit:
                rooms_with_keyword_match += 1
                per_label[label]["keyword_match"] += 1
            elif linked and len(examples) < 40:
                examples.append(
                    {
                        "gold": label,
                        "texts": [text.get("text") for text in linked[:5]],
                        "annotation": row.get("annotation_path"),
                    }
                )
    return {
        "records": len(rows),
        "rooms": total_rooms,
        "rooms_with_label_text": rooms_with_label_text,
        "label_text_coverage": rooms_with_label_text / max(total_rooms, 1),
        "rooms_with_keyword_match": rooms_with_keyword_match,
        "keyword_match_rate": rooms_with_keyword_match / max(total_rooms, 1),
        "per_label": {
            label: {
                "rooms": counts["rooms"],
                "rooms_with_label_text": counts["rooms_with_label_text"],
                "label_text_coverage": counts["rooms_with_label_text"] / max(counts["rooms"], 1),
                "keyword_match": counts["keyword_match"],
                "keyword_match_rate": counts["keyword_match"] / max(counts["rooms"], 1),
            }
            for label, counts in sorted(per_label.items())
        },
        "unmatched_examples": examples,
    }

def text_center_inside(room_bbox: list[float], text: dict[str, Any]) -> bool:
    bbox = normalize_bbox(text.get("bbox"))
    if bbox is None:
        return False
    cx = (bbox[0] + bbox[2]) / 2.0
    cy = (bbox[1] + bbox[3]) / 2.0
    return room_bbox[0] <= cx <= room_bbox[2] and room_bbox[1] <= cy <= room_bbox[3]


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
