#!/usr/bin/env python3
"""Build v2 external OCR/symbol human-gold smoke status reports."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"

OCR_PACK = REPORTS / "text_dimension_external_ocr_annotation_pack_v1.pending.jsonl"
SYMBOL_PACK = REPORTS / "symbol_cross_source_annotation_pack_v1.pending.jsonl"
OCR_LOCK = REPORTS / "text_dimension_external_ocr_lock_v5.json"
SYMBOL_LOCK = REPORTS / "symbol_cross_source_lock_v3.json"
OCR_OUT = REPORTS / "external_ocr_human_gold_smoke_v2.json"
SYMBOL_OUT = REPORTS / "symbol_cross_source_human_gold_smoke_v2.json"
DECISION_OUT = REPORTS / "external_generalization_claim_decision_v2.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def count_ocr_gold(rows: list[dict[str, Any]]) -> dict[str, int]:
    drawings = 0
    annotations = 0
    typed = 0
    linked = 0
    for row in rows:
        row_has_gold = False
        for item in row.get("text_annotations") or []:
            has_bbox = bool(item.get("bbox") or item.get("polygon"))
            has_text = bool(item.get("verbatim_transcript") or item.get("normalized_transcript"))
            if has_bbox and has_text:
                annotations += 1
                row_has_gold = True
                if item.get("text_type"):
                    typed += 1
                if item.get("dimension_of"):
                    linked += 1
        if row_has_gold:
            drawings += 1
    return {
        "drawings_with_transcript_and_bbox": drawings,
        "text_annotations_with_transcript_and_bbox": annotations,
        "text_annotations_with_text_type": typed,
        "text_annotations_with_dimension_link": linked,
    }


def count_symbol_gold(rows: list[dict[str, Any]]) -> dict[str, int]:
    drawings = 0
    annotations = 0
    by_label: dict[str, int] = {}
    for row in rows:
        row_has_gold = False
        for item in row.get("symbol_annotations") or []:
            label = item.get("gold_9class_symbol_type")
            if label:
                annotations += 1
                row_has_gold = True
                by_label[str(label)] = by_label.get(str(label), 0) + 1
        if row_has_gold:
            drawings += 1
    return {
        "drawings_with_gold_9class_symbol_type": drawings,
        "gold_symbol_annotations": annotations,
        "gold_label_counts": by_label,
    }


def main() -> int:
    ocr_rows = load_jsonl(OCR_PACK)
    symbol_rows = load_jsonl(SYMBOL_PACK)
    ocr_counts = count_ocr_gold(ocr_rows)
    symbol_counts = count_symbol_gold(symbol_rows)
    ocr_lock = load_json(OCR_LOCK)
    symbol_lock = load_json(SYMBOL_LOCK)

    ocr_passed = ocr_counts["drawings_with_transcript_and_bbox"] >= 20
    symbol_passed = symbol_counts["drawings_with_gold_9class_symbol_type"] >= 20 or symbol_counts["gold_symbol_annotations"] >= 200
    ocr_report = {
        "version": "external_ocr_human_gold_smoke_v2",
        "created": "2026-05-04",
        "source_lock": str(OCR_LOCK.relative_to(ROOT)),
        "pending_annotation_pack": str(OCR_PACK.relative_to(ROOT)),
        "pending_records": len(ocr_rows),
        "minimum_drawings_for_smoke": 20,
        "preferred_drawings_for_sci2": 50,
        "human_gold": ocr_counts,
        "metrics": {
            "external_ocr_exact": None,
            "external_ocr_cer": None,
            "external_ocr_text_type_macro_f1": None,
            "external_ocr_dimension_link_f1": None,
            "reason_unavailable": None if ocr_passed else "pending pack contains no sufficient filled human transcript+bbox gold",
        },
        "previous_lock_status": ocr_lock.get("status"),
        "claim_boundary": {
            "external_ocr_claim_allowed": ocr_passed,
            "limitation_required": not ocr_passed,
            "allowed_text": "External OCR human-gold smoke may be reported only after this file contains computed metrics.",
            "blocked_text": "Do not claim broad scanned/photo OCR robustness while human gold is missing.",
        },
        "status": "passed_external_ocr_human_gold_smoke" if ocr_passed else "blocked_pending_no_human_gold",
    }

    symbol_report = {
        "version": "symbol_cross_source_human_gold_smoke_v2",
        "created": "2026-05-04",
        "source_lock": str(SYMBOL_LOCK.relative_to(ROOT)),
        "pending_annotation_pack": str(SYMBOL_PACK.relative_to(ROOT)),
        "pending_drawings": len(symbol_rows),
        "minimum_drawings_for_smoke": 20,
        "minimum_symbols_for_smoke": 200,
        "human_gold": symbol_counts,
        "metrics": {
            "accuracy": None,
            "macro_f1": None,
            "per_label": {},
            "reason_unavailable": None if symbol_passed else "pending pack contains no sufficient filled 9-class symbol human gold",
        },
        "previous_lock_status": symbol_lock.get("status"),
        "claim_boundary": {
            "cross_source_symbol_claim_allowed": symbol_passed,
            "limitation_required": not symbol_passed,
            "allowed_text": "Cross-source symbol human-gold smoke may be reported only after this file contains computed metrics.",
            "blocked_text": "Do not use locked CubiCasa symbol accuracy as cross-source symbol generalization evidence.",
        },
        "status": "passed_cross_source_symbol_human_gold_smoke" if symbol_passed else "blocked_pending_no_human_gold",
    }

    decision = {
        "version": "external_generalization_claim_decision_v2",
        "created": "2026-05-04",
        "sources": {
            "ocr_smoke": str(OCR_OUT.relative_to(ROOT)),
            "symbol_smoke": str(SYMBOL_OUT.relative_to(ROOT)),
            "ocr_lock": str(OCR_LOCK.relative_to(ROOT)),
            "symbol_lock": str(SYMBOL_LOCK.relative_to(ROOT)),
        },
        "decision": "external_generalization_claim_allowed" if ocr_passed and symbol_passed else "limitation_ready_no_external_generalization_claim",
        "human_gold_status": {
            "external_ocr_drawings_with_gold": ocr_counts["drawings_with_transcript_and_bbox"],
            "cross_source_symbol_drawings_with_gold": symbol_counts["drawings_with_gold_9class_symbol_type"],
            "cross_source_symbol_annotations_with_gold": symbol_counts["gold_symbol_annotations"],
        },
        "allowed_claims": [
            "Internal/locked metrics are reportable under their exact split and source boundary.",
            "External OCR and cross-source symbol packs are annotation-ready.",
        ],
        "blocked_claims": [] if ocr_passed and symbol_passed else [
            "Broad scanned/photo OCR robustness.",
            "Cross-source symbol generalization.",
            "In-the-wild floorplan symbol/text robustness comparable to external wild datasets.",
        ],
        "paper_limitation_text": "External OCR and cross-source symbol generalization are not claimed because the prepared external annotation packs currently contain no filled human-gold labels.",
        "done_when_check": {
            "human_gold_metrics_available": ocr_passed and symbol_passed,
            "explicit_no_human_gold_limitation_enforced": not (ocr_passed and symbol_passed),
        },
        "status": "completed_human_gold_metrics_available" if ocr_passed and symbol_passed else "completed_limitation_ready_no_human_gold",
    }
    write_json(OCR_OUT, ocr_report)
    write_json(SYMBOL_OUT, symbol_report)
    write_json(DECISION_OUT, decision)
    print(json.dumps({"ocr": ocr_report["status"], "symbol": symbol_report["status"], "decision": decision["decision"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
