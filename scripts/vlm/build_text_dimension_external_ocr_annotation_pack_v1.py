#!/usr/bin/env python3
"""Build a pending external OCR annotation pack for TextDimension."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
FLOORPLANCAD = ROOT / "datasets" / "external" / "floorplancad" / "samples.json"
REPORT = ROOT / "reports" / "vlm" / "text_dimension_external_ocr_annotation_pack_v1.json"
PENDING_JSONL = ROOT / "reports" / "vlm" / "text_dimension_external_ocr_annotation_pack_v1.pending.jsonl"


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_floorplancad() -> list[dict[str, Any]]:
    if not FLOORPLANCAD.exists():
        return []
    data = json.loads(FLOORPLANCAD.read_text(encoding="utf-8"))
    if isinstance(data, dict):
        return list(data.get("samples") or [])
    if isinstance(data, list):
        return data
    return []


def sample_id(sample: dict[str, Any], idx: int) -> str:
    oid = (((sample.get("_id") or {}).get("$oid")) if isinstance(sample.get("_id"), dict) else sample.get("_id"))
    return str(oid or f"floorplancad_{idx:04d}")


def detection_counts(sample: dict[str, Any]) -> Counter[str]:
    detections = (((sample.get("ground_truth") or {}).get("detections")) or [])
    return Counter(str(det.get("label") or "unknown") for det in detections if isinstance(det, dict))


def main() -> int:
    samples = load_floorplancad()
    existing = []
    for idx, sample in enumerate(samples):
        rel = sample.get("filepath")
        image_path = ROOT / "datasets" / "external" / "floorplancad" / str(rel)
        if rel and image_path.exists():
            existing.append((idx, sample, detection_counts(sample)))

    # Prefer visually non-empty floorplans with enough structural detections.
    ranked = sorted(existing, key=lambda item: (sum(item[2].values()), item[2].get("wall", 0)), reverse=True)
    selected = ranked[:50]
    records = []
    for order, (idx, sample, counts) in enumerate(selected):
        rel = str(Path("datasets") / "external" / "floorplancad" / str(sample.get("filepath")))
        records.append(
            {
                "pack_id": f"external_ocr_{order:03d}",
                "source_dataset": "floorplancad",
                "sample_id": sample_id(sample, idx),
                "image_path": rel,
                "selection_reason": "source-held-out raster floorplan candidate; no local human OCR transcript gold available",
                "existing_detection_label_counts": dict(counts.most_common()),
                "scan_quality": None,
                "annotation_status": "pending",
                "text_annotations": [
                    {
                        "text_id": None,
                        "bbox": None,
                        "polygon": None,
                        "verbatim_transcript": None,
                        "normalized_transcript": None,
                        "text_type": None,
                        "dimension_of": None,
                        "legibility": None,
                        "notes": None,
                    }
                ],
            }
        )

    PENDING_JSONL.parent.mkdir(parents=True, exist_ok=True)
    PENDING_JSONL.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in records) + ("\n" if records else ""), encoding="utf-8")

    report = {
        "version": "text_dimension_external_ocr_annotation_pack_v1",
        "created": "2026-05-03",
        "status": "pending_annotation",
        "source": str(FLOORPLANCAD.relative_to(ROOT)),
        "candidate_pool": {
            "total_samples": len(samples),
            "existing_images": len(existing),
            "selected_drawings": len(records),
        },
        "outputs": {
            "pending_jsonl": str(PENDING_JSONL.relative_to(ROOT)),
        },
        "required_annotation_fields": [
            "text bbox or polygon",
            "verbatim transcript",
            "normalized transcript",
            "text_type in {dimension_text, room_label, note_text, leader_line, dimension_line}",
            "dimension_of link for dimension_text where visible",
            "legibility flag",
            "scan_quality flag",
        ],
        "gold_status": {
            "has_human_gold_transcripts": False,
            "can_run_external_ocr_lock": False,
            "next_step": "Fill pending JSONL text_annotations, then run TextDimension v5 standalone + OCR exact/CER and E2E text-family smoke.",
        },
        "paper_boundary": {
            "standalone_v5": "CubiCasa SVG/OCR-enhanced expert benchmark only",
            "e2e_text_family": "scene-graph node-label contract, separate from standalone expert benchmark",
            "external_ocr": "pending annotation; do not claim broad scanned/photo OCR robustness",
        },
        "sample_preview": records[:5],
    }
    write_json(REPORT, report)
    print(f"wrote {REPORT}")
    print(f"wrote {PENDING_JSONL}")
    print(json.dumps({"selected_drawings": len(records), "status": report["status"]}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
