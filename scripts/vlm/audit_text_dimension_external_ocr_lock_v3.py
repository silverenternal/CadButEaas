#!/usr/bin/env python3
"""Lock external OCR annotation status for paper claims."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
PENDING = REPORTS / "text_dimension_external_ocr_annotation_pack_v1.pending.jsonl"
PACK = REPORTS / "text_dimension_external_ocr_annotation_pack_v1.json"
PRIOR_LOCK = REPORTS / "text_dimension_external_ocr_lock_v1.json"
TEXT_V5 = REPORTS / "text_dimension_expert_v5_eval.json"
RECONCILIATION = REPORTS / "paper_e2e_metric_reconciliation_v1.json"
OUTPUT = REPORTS / "text_dimension_external_ocr_lock_v3.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def has_bbox(ann: dict[str, Any]) -> bool:
    bbox = ann.get("bbox") or ann.get("bbox_normalized_xywh")
    polygon = ann.get("polygon")
    return (isinstance(bbox, list) and len(bbox) >= 4 and all(v is not None for v in bbox[:4])) or bool(polygon)


def has_transcript(ann: dict[str, Any]) -> bool:
    return bool(str(ann.get("verbatim_transcript") or ann.get("normalized_transcript") or "").strip())


def annotated_records(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int]:
    records: list[dict[str, Any]] = []
    annotation_count = 0
    for row in rows:
        anns = row.get("text_annotations") or []
        filled = [ann for ann in anns if has_transcript(ann) and has_bbox(ann)]
        if filled:
            records.append({**row, "filled_text_annotations": len(filled)})
            annotation_count += len(filled)
    return records, annotation_count


def main() -> int:
    rows = load_jsonl(PENDING)
    pack = load_json(PACK)
    prior = load_json(PRIOR_LOCK)
    text_v5 = load_json(TEXT_V5)
    reconciliation = load_json(RECONCILIATION)
    filled_records, filled_annotations = annotated_records(rows)

    has_min_lock = len(filled_records) >= 20
    status = "passed_external_lock" if has_min_lock else "pending_no_human_gold"
    report: dict[str, Any] = {
        "version": "text_dimension_external_ocr_lock_v3",
        "created": "2026-05-03",
        "status": status,
        "input_pack": str(PACK.relative_to(ROOT)),
        "pending_jsonl": str(PENDING.relative_to(ROOT)),
        "selected_records": len(rows) or pack.get("selected_records"),
        "human_gold": {
            "drawings_with_transcript_and_bbox": len(filled_records),
            "text_annotations_with_transcript_and_bbox": filled_annotations,
            "minimum_drawings_for_smoke": 20,
            "smoke_metrics_available": has_min_lock,
        },
        "metrics": {
            "external_ocr_exact": None,
            "external_ocr_cer": None,
            "external_ocr_text_type_macro_f1": None,
            "external_ocr_dimension_link_f1": None,
        },
        "current_internal_evidence": {
            "text_dimension_v5_source": str(TEXT_V5.relative_to(ROOT)) if TEXT_V5.exists() else None,
            "standalone_macro_f1": text_v5.get("macro_f1"),
            "standalone_dimension_link_f1": text_v5.get("dimension_link_f1"),
            "paper_e2e_text_family_boundary": (reconciliation.get("paper_main_metrics") or {}).get("node_per_label", {}).get("dimension_text"),
        },
        "claim_boundary": {
            "broad_real_ocr_claim_allowed": False,
            "allowed_text": "Use the external OCR pack as an annotation-ready limitation unless human gold is filled and this v3 lock is rerun.",
            "disallowed_text": "Do not claim scanned/photo OCR robustness from TextDimension standalone or CubiCasa E2E numbers.",
        },
        "prior_lock": {
            "source": str(PRIOR_LOCK.relative_to(ROOT)) if PRIOR_LOCK.exists() else None,
            "status": prior.get("status"),
        },
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(json.dumps({"status": status, "drawings_with_gold": len(filled_records), "annotations_with_gold": filled_annotations}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
