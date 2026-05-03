#!/usr/bin/env python3
"""Audit whether an external scanned-drawing OCR lock set is available."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
OUTPUT = REPORTS / "text_dimension_external_ocr_lock_v1.json"

CHECKED_DATASETS = [
    ROOT / "datasets" / "cadstruct_real_world_benchmark_v1" / "text_dimension" / "cubicasa5k_text_smoke_locked.jsonl",
    ROOT / "datasets" / "text_dimension_expert_v4_full_ocr_augmented" / "locked_test.jsonl",
    ROOT / "datasets" / "internal_hard_cases_round_2" / "text_dimension_candidates.jsonl",
    ROOT / "reports" / "vlm" / "ocr_backend_predictions_v1.jsonl",
    ROOT / "datasets" / "external" / "floorplancad" / "samples.json",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
    return rows


def load_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def source_summary(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "exists": False}
    rows = load_jsonl(path) if path.suffix == ".jsonl" else []
    data = load_json(path) if path.suffix == ".json" else None
    source_counts: Counter[str] = Counter()
    has_gold_text = False
    has_text_labels = False
    has_scanned_flag = False
    row_count = len(rows)
    if isinstance(data, list):
        row_count = len(data)
    elif isinstance(data, dict):
        row_count = len(data.get("samples") or data.get("images") or data)
    for row in rows:
        source_counts[str(row.get("source_dataset") or row.get("source") or "unknown")] += 1
        text_candidates = row.get("text_candidates") or []
        if isinstance(text_candidates, list) and text_candidates:
            has_text_labels = has_text_labels or any(isinstance(item, dict) and item.get("text_type") for item in text_candidates)
            has_gold_text = has_gold_text or any(isinstance(item, dict) and item.get("text") for item in text_candidates)
        has_gold_text = has_gold_text or bool(row.get("gold_text") or row.get("target_text") or row.get("ocr_text_gold"))
        tags = " ".join(str(x) for x in [row.get("source_dataset"), row.get("source"), row.get("scan_quality"), row.get("profile")])
        if any(token in tags.lower() for token in ["scan", "scanned", "photo"]):
            has_scanned_flag = True
    if isinstance(data, dict):
        text = json.dumps(data)[:20000].lower()
        has_scanned_flag = has_scanned_flag or any(token in text for token in ["scan", "scanned", "photo"])
        has_gold_text = has_gold_text or any(token in text for token in ["gold_text", "target_text", "ocr_text_gold"])
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": True,
        "row_count": row_count,
        "source_counts": dict(source_counts),
        "has_text_type_labels": has_text_labels,
        "has_gold_text_transcripts": has_gold_text,
        "has_scanned_or_photo_flag": has_scanned_flag,
        "usable_external_ocr_lock": bool(has_scanned_flag and has_gold_text),
    }


def main() -> int:
    sources = [source_summary(path) for path in CHECKED_DATASETS]
    expert_v5 = load_json(REPORTS / "text_dimension_expert_v5_eval.json") or {}
    alignment = load_json(REPORTS / "text_dimension_real_upstream_alignment_v1.json") or {}
    usable = [item for item in sources if item.get("usable_external_ocr_lock")]
    status = "passed_external_lock" if usable else "not_available_with_annotation_plan"
    report = {
        "version": "text_dimension_external_ocr_lock_v1",
        "created": "2026-05-03",
        "status": status,
        "checked_sources": sources,
        "external_lock": {
            "available": bool(usable),
            "usable_sources": usable,
            "reason_if_not_available": "Local datasets contain CubiCasa SVG/OCR-enhanced TextDimension labels and OCR backend predictions, but no source-held-out scanned/photo drawing split with human gold text transcripts and TextDimension labels.",
        },
        "current_evidence": {
            "text_dimension_v5_standalone": {
                "source": "reports/vlm/text_dimension_expert_v5_eval.json",
                "dev_macro_f1": (((expert_v5.get("splits") or {}).get("dev") or {}).get("macro_f1")),
                "dev_dimension_link_f1": ((((expert_v5.get("splits") or {}).get("dev") or {}).get("dimension_link") or {}).get("f1")),
                "locked_macro_f1": (((expert_v5.get("splits") or {}).get("locked_test") or {}).get("macro_f1")),
                "locked_dimension_link_f1": ((((expert_v5.get("splits") or {}).get("locked_test") or {}).get("dimension_link") or {}).get("f1")),
            },
            "real_upstream_e2e_text_family": {
                "source": "reports/vlm/text_dimension_real_upstream_alignment_v1.json",
                "current_text_family_macro_f1": (((alignment.get("real_upstream_e2e") or {}).get("current_text_family") or {}).get("macro_f1")),
                "note": "Standalone expert metrics and E2E text-family node metrics are separate contracts.",
            },
            "paper_main_relation": {
                "source": "reports/vlm/paper_e2e_metric_reconciliation_v1.json",
                "not_recomputed": True,
                "reason": "This audit only checks external OCR lock availability; it does not alter predictions or relation fusion.",
            },
        },
        "annotation_plan_if_not_available": {
            "minimum_drawings": 50,
            "preferred_drawings": 100,
            "sources": ["source-held-out scanned PDFs", "photo/scan-like raster floorplans", "non-CubiCasa public CAD raster where licensing permits annotation"],
            "required_gold_fields": [
                "image_id/source",
                "text bbox polygon or rectangle",
                "verbatim transcript",
                "normalized transcript",
                "text_type in {dimension_text, room_label, note_text, leader_line, dimension_line}",
                "dimension_of links for dimension_text where visible",
                "scan_quality and legibility flags",
            ],
            "acceptance_targets": {
                "standalone_text_macro_f1": ">=0.95 on external lock",
                "normalized_ocr_exact": ">=0.90 on legible non-empty text",
                "dimension_link_f1": ">=0.90 where links are annotated",
            },
        },
        "paper_guidance": {
            "can_claim_textdimension_v5_on_cubicasa_svg_ocr_augmented": True,
            "can_claim_broad_real_ocr_robustness": bool(usable),
            "must_state_external_ocr_not_available": not bool(usable),
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(json.dumps({"status": status, "usable_external_sources": len(usable)}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
