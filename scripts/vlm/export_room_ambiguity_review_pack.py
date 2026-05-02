#!/usr/bin/env python3
"""Export review artifacts for ambiguous generic-room cases."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ambiguity-report", default="reports/vlm/room_space_v5_t046_ambiguity_adjusted.json")
    parser.add_argument("--boundary-report", default="reports/vlm/room_generic_boundary_audit_v5_t046.json")
    parser.add_argument("--output-dir", default="reports/vlm/room_ambiguity_review_pack_v1")
    args = parser.parse_args()

    ambiguity = json.loads(Path(args.ambiguity_report).read_text(encoding="utf-8"))
    boundary = json.loads(Path(args.boundary_report).read_text(encoding="utf-8"))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    candidates = sorted(
        normalize_candidates(ambiguity.get("candidates") or []),
        key=lambda item: (item["prediction"], item["annotation"], item["id"]),
    )
    examples = boundary.get("examples") or []
    example_lookup = {(item.get("annotation"), str(item.get("id"))): item for item in examples}
    rows = []
    for index, item in enumerate(candidates, start=1):
        details = example_lookup.get((item["annotation"], str(item["id"]))) or {}
        rows.append(
            {
                "review_id": f"amb_room_{index:04d}",
                "annotation": item["annotation"],
                "source_bucket": source_bucket(item["annotation"]),
                "room_id": item["id"],
                "gold": item["gold"],
                "prediction": item["prediction"],
                "confidence": item.get("confidence"),
                "texts": " | ".join(item.get("texts") or []),
                "bbox": json.dumps(details.get("bbox"), ensure_ascii=False),
                "shape_bucket": details.get("shape_bucket"),
                "shape": json.dumps(details.get("shape"), ensure_ascii=False),
                "recommended_action": "accept_typed_if_text_matches_region",
                "review_label": "",
                "review_notes": "",
            }
        )

    write_csv(output_dir / "review_queue.csv", rows)
    write_jsonl(output_dir / "review_queue.jsonl", rows)
    protocol = {
        "name": "room_ambiguity_review_pack_v1",
        "source_reports": {
            "ambiguity_report": args.ambiguity_report,
            "boundary_report": args.boundary_report,
        },
        "candidate_count": len(rows),
        "review_labels": {
            "accept_typed": "Gold room is too generic; predicted typed class is supported by room text and should be accepted.",
            "keep_room": "Gold room should remain generic despite typed text, usually because the text refers to a subregion or nearby area.",
            "unclear": "Needs visual inspection or project-specific rule.",
            "exclude": "Annotation/candidate is malformed enough to exclude from clean eval.",
        },
        "strict_metrics": pick_metrics(ambiguity.get("strict") or {}),
        "ambiguity_adjusted_metrics": pick_metrics(ambiguity.get("ambiguity_adjusted") or {}),
    }
    (output_dir / "protocol.json").write_text(json.dumps(protocol, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output_dir": str(output_dir), "candidate_count": len(rows), **protocol}, ensure_ascii=False, indent=2))


def normalize_candidates(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = []
    seen = set()
    for item in candidates:
        key = (item.get("annotation"), str(item.get("id")), item.get("prediction"))
        if key in seen:
            continue
        seen.add(key)
        normalized.append(
            {
                "annotation": str(item.get("annotation") or ""),
                "id": str(item.get("id") or ""),
                "gold": str(item.get("gold") or ""),
                "prediction": str(item.get("prediction") or ""),
                "confidence": item.get("confidence"),
                "texts": [str(text) for text in item.get("texts") or []],
            }
        )
    return normalized


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "review_id",
        "annotation",
        "source_bucket",
        "room_id",
        "gold",
        "prediction",
        "confidence",
        "texts",
        "bbox",
        "shape_bucket",
        "shape",
        "recommended_action",
        "review_label",
        "review_notes",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def pick_metrics(metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "rooms": metrics.get("rooms"),
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "room_f1": ((metrics.get("per_label") or {}).get("room") or {}).get("f1"),
    }


def source_bucket(annotation: str) -> str:
    marker = "/cubicasa5k/"
    if marker in annotation:
        return annotation.split(marker, 1)[1].split("/", 1)[0]
    return "unknown"


if __name__ == "__main__":
    main()
