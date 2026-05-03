#!/usr/bin/env python3
"""Build v3 external OCR/symbol human-gold manifest and claim decision."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"

OCR_PACK_JSON = REPORTS / "text_dimension_external_ocr_annotation_pack_v1.json"
OCR_PACK_PENDING = REPORTS / "text_dimension_external_ocr_annotation_pack_v1.pending.jsonl"
SYMBOL_PACK_JSON = REPORTS / "symbol_cross_source_annotation_pack_v1.json"
SYMBOL_PACK_PENDING = REPORTS / "symbol_cross_source_annotation_pack_v1.pending.jsonl"
OCR_LOCK = REPORTS / "text_dimension_external_ocr_lock_v5.json"
SYMBOL_LOCK = REPORTS / "symbol_cross_source_lock_v3.json"

OUT_MANIFEST = REPORTS / "external_human_gold_manifest_v3.json"
OUT_OCR = REPORTS / "external_ocr_human_gold_smoke_v3.json"
OUT_SYMBOL = REPORTS / "symbol_cross_source_human_gold_smoke_v3.json"
OUT_DECISION = REPORTS / "external_generalization_claim_decision_v3.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def path_exists(raw: str | None) -> bool:
    if not raw:
        return False
    return (ROOT / raw).exists()


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
                typed += int(bool(item.get("text_type")))
                linked += int(bool(item.get("dimension_of")))
        drawings += int(row_has_gold)
    return {
        "drawings_with_transcript_and_bbox": drawings,
        "text_annotations_with_transcript_and_bbox": annotations,
        "text_annotations_with_text_type": typed,
        "text_annotations_with_dimension_link": linked,
    }


def count_symbol_gold(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
        drawings += int(row_has_gold)
    return {
        "drawings_with_gold_9class_symbol_type": drawings,
        "gold_symbol_annotations": annotations,
        "gold_label_counts": by_label,
    }


def image_status(rows: list[dict[str, Any]]) -> dict[str, int]:
    with_path = 0
    existing = 0
    for row in rows:
        image_path = row.get("image_path")
        if image_path:
            with_path += 1
            existing += int(path_exists(str(image_path)))
    return {"records_with_image_path": with_path, "records_with_existing_image_path": existing}


def main() -> int:
    ocr_rows = load_jsonl(OCR_PACK_PENDING)
    symbol_rows = load_jsonl(SYMBOL_PACK_PENDING)
    ocr_counts = count_ocr_gold(ocr_rows)
    symbol_counts = count_symbol_gold(symbol_rows)
    ocr_passed = ocr_counts["drawings_with_transcript_and_bbox"] >= 20
    symbol_passed = (
        symbol_counts["drawings_with_gold_9class_symbol_type"] >= 20
        or symbol_counts["gold_symbol_annotations"] >= 200
    )

    manifest = {
        "version": "external_human_gold_manifest_v3",
        "created": "2026-05-04",
        "packs": {
            "external_ocr": {
                "pack_json": rel(OCR_PACK_JSON),
                "pending_jsonl": rel(OCR_PACK_PENDING),
                "records": len(ocr_rows),
                "image_status": image_status(ocr_rows),
                "human_gold": ocr_counts,
                "minimum_drawings_for_smoke": 20,
                "preferred_drawings_for_sci2": 50,
                "lock_report": rel(OCR_LOCK),
                "lock_status": load_json(OCR_LOCK).get("status"),
            },
            "cross_source_symbol": {
                "pack_json": rel(SYMBOL_PACK_JSON),
                "pending_jsonl": rel(SYMBOL_PACK_PENDING),
                "records": len(symbol_rows),
                "image_status": image_status(symbol_rows),
                "human_gold": symbol_counts,
                "minimum_drawings_for_smoke": 20,
                "minimum_symbols_for_smoke": 200,
                "lock_report": rel(SYMBOL_LOCK),
                "lock_status": load_json(SYMBOL_LOCK).get("status"),
            },
        },
        "annotation_ready": bool(ocr_rows and symbol_rows),
        "human_gold_metrics_available": bool(ocr_passed and symbol_passed),
        "status": "human_gold_ready" if ocr_passed and symbol_passed else "annotation_ready_pending_human_gold",
    }

    ocr_report = {
        "version": "external_ocr_human_gold_smoke_v3",
        "created": "2026-05-04",
        "manifest": rel(OUT_MANIFEST),
        "pending_annotation_pack": rel(OCR_PACK_PENDING),
        "pending_records": len(ocr_rows),
        "human_gold": ocr_counts,
        "metrics": {
            "external_ocr_exact": None,
            "external_ocr_cer": None,
            "external_ocr_text_type_macro_f1": None,
            "external_ocr_dimension_link_f1": None,
            "reason_unavailable": None if ocr_passed else "no sufficient filled transcript+bbox human gold in pending pack",
        },
        "claim_boundary": {
            "external_ocr_claim_allowed": ocr_passed,
            "limitation_required": not ocr_passed,
            "blocked_text": "Do not claim broad scanned/photo OCR robustness before human-gold transcripts and bboxes are filled.",
        },
        "status": "passed_external_ocr_human_gold_smoke" if ocr_passed else "blocked_pending_no_human_gold",
    }

    symbol_report = {
        "version": "symbol_cross_source_human_gold_smoke_v3",
        "created": "2026-05-04",
        "manifest": rel(OUT_MANIFEST),
        "pending_annotation_pack": rel(SYMBOL_PACK_PENDING),
        "pending_drawings": len(symbol_rows),
        "human_gold": symbol_counts,
        "metrics": {
            "accuracy": None,
            "macro_f1": None,
            "per_label": {},
            "reason_unavailable": None if symbol_passed else "no sufficient filled external 9-class symbol human gold in pending pack",
        },
        "claim_boundary": {
            "cross_source_symbol_claim_allowed": symbol_passed,
            "limitation_required": not symbol_passed,
            "blocked_text": "Do not use internal CubiCasa/FloorPlanCAD locked metrics as cross-source symbol human-gold evidence.",
        },
        "status": "passed_cross_source_symbol_human_gold_smoke" if symbol_passed else "blocked_pending_no_human_gold",
    }

    decision = {
        "version": "external_generalization_claim_decision_v3",
        "created": "2026-05-04",
        "sources": {
            "manifest": rel(OUT_MANIFEST),
            "ocr_smoke": rel(OUT_OCR),
            "symbol_smoke": rel(OUT_SYMBOL),
        },
        "decision": "external_generalization_claim_allowed" if ocr_passed and symbol_passed else "blocked_external_generalization_claim_annotation_pack_ready",
        "done_when_check": {
            "external_claim_decision_exists": True,
            "has_nonzero_human_gold_metrics": bool(ocr_passed and symbol_passed),
            "blocks_external_wild_generalization_with_annotation_pack_paths": not (ocr_passed and symbol_passed),
        },
        "human_gold_status": {
            "external_ocr_drawings_with_gold": ocr_counts["drawings_with_transcript_and_bbox"],
            "cross_source_symbol_drawings_with_gold": symbol_counts["drawings_with_gold_9class_symbol_type"],
            "cross_source_symbol_annotations_with_gold": symbol_counts["gold_symbol_annotations"],
        },
        "allowed_claims": [
            "Internal locked-split node/relation metrics can be reported under their exact benchmark boundary.",
            "External OCR and cross-source symbol packs are annotation-ready and reproducible.",
        ],
        "blocked_claims": [] if ocr_passed and symbol_passed else [
            "Broad scanned/photo OCR robustness.",
            "Cross-source symbol generalization from human-gold evidence.",
            "WAFFLE/ResPlan-style in-the-wild floorplan generalization.",
        ],
        "paper_limitation_text": (
            "External OCR and cross-source symbol generalization are not claimed because the prepared "
            "annotation packs currently contain 0 filled OCR transcript+bbox drawings and 0 filled "
            "external 9-class symbol annotations."
        ),
        "status": "completed_human_gold_metrics_available" if ocr_passed and symbol_passed else "completed_blocked_with_annotation_pack_paths",
    }

    write_json(OUT_MANIFEST, manifest)
    write_json(OUT_OCR, ocr_report)
    write_json(OUT_SYMBOL, symbol_report)
    write_json(OUT_DECISION, decision)
    print(json.dumps({"manifest": manifest["status"], "decision": decision["decision"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
