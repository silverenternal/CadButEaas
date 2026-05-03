#!/usr/bin/env python3
"""Summarize external OCR and cross-source symbol gold collection status."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
TEXT_V4 = REPORTS / "text_dimension_external_ocr_lock_v4.json"
TEXT_V5 = REPORTS / "text_dimension_external_ocr_lock_v5.json"
SYMBOL_V3 = REPORTS / "symbol_cross_source_lock_v3.json"
STATUS = REPORTS / "external_gold_collection_status_v1.json"
LIMITATIONS = REPORTS / "paper_submission_limitations_v2.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    text = load_json(TEXT_V4)
    symbol = load_json(SYMBOL_V3)
    text_v5 = {**text, "version": "text_dimension_external_ocr_lock_v5", "created": "2026-05-03", "source_v4": str(TEXT_V4.relative_to(ROOT))}
    write_json(TEXT_V5, text_v5)

    report = {
        "version": "external_gold_collection_status_v1",
        "created": "2026-05-03",
        "ocr": {
            "lock_report": str(TEXT_V5.relative_to(ROOT)),
            "status": text_v5.get("status"),
            "drawings_with_gold": (text_v5.get("human_gold") or {}).get("drawings_with_transcript_and_bbox"),
            "annotations_with_gold": (text_v5.get("human_gold") or {}).get("text_annotations_with_transcript_and_bbox"),
            "minimum_drawings_for_smoke": (text_v5.get("human_gold") or {}).get("minimum_drawings_for_smoke"),
            "preferred_drawings_for_sci2": (text_v5.get("human_gold") or {}).get("preferred_drawings_for_sci2"),
        },
        "symbol": {
            "lock_report": str(SYMBOL_V3.relative_to(ROOT)),
            "status": symbol.get("status"),
            "drawings_with_gold": ((symbol.get("human_gold") or {}).get("drawings_with_gold_9class_symbol_type")),
            "annotations_with_gold": ((symbol.get("human_gold") or {}).get("gold_symbol_annotations")),
            "minimum_drawings_for_smoke": ((symbol.get("human_gold") or {}).get("minimum_drawings_for_smoke")),
            "minimum_symbols_for_smoke": ((symbol.get("human_gold") or {}).get("minimum_symbols_for_smoke")),
        },
        "annotation_files": {
            "ocr_pending_jsonl": "reports/vlm/text_dimension_external_ocr_annotation_pack_v1.pending.jsonl",
            "ocr_checklist": "reports/vlm/text_dimension_external_ocr_annotation_checklist_v1.md",
            "symbol_pending_jsonl": "reports/vlm/symbol_cross_source_annotation_pack_v1.pending.jsonl",
            "symbol_priority_queue": "reports/vlm/symbol_long_tail_sci2_annotation_queue_v1.jsonl",
        },
        "claim_boundary": {
            "external_ocr_claim_allowed": text_v5.get("status") == "passed_external_lock",
            "cross_source_symbol_claim_allowed": symbol.get("status") == "passed_cross_source_symbol_smoke",
            "paper_limitation_required": True,
        },
        "status": "pending_no_human_gold",
    }
    if report["claim_boundary"]["external_ocr_claim_allowed"] or report["claim_boundary"]["cross_source_symbol_claim_allowed"]:
        report["status"] = "partial_external_gold_available"

    if LIMITATIONS.exists():
        text_body = LIMITATIONS.read_text(encoding="utf-8")
        marker = "External gold collection status:"
        line = (
            "\nExternal gold collection status: OCR external lock v5 and symbol cross-source lock v3 remain "
            "pending_no_human_gold as of 2026-05-03; broad OCR/wild symbol generalization is not a final claim.\n"
        )
        if marker not in text_body:
            LIMITATIONS.write_text(text_body.rstrip() + "\n" + line, encoding="utf-8")

    write_json(STATUS, report)
    print(f"wrote {STATUS}")
    print(f"wrote {TEXT_V5}")
    print(json.dumps({"status": report["status"], "ocr_gold": report["ocr"]["drawings_with_gold"], "symbol_gold": report["symbol"]["annotations_with_gold"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
