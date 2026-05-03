#!/usr/bin/env python3
"""SCI2 external OCR lock and annotation checklist for TextDimension."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
PENDING = REPORTS / "text_dimension_external_ocr_annotation_pack_v1.pending.jsonl"
PACK = REPORTS / "text_dimension_external_ocr_annotation_pack_v1.json"
TEXT_V5 = REPORTS / "text_dimension_expert_v5_eval.json"
RECONCILIATION = REPORTS / "paper_e2e_metric_reconciliation_v1.json"
OUTPUT = REPORTS / "text_dimension_external_ocr_lock_v4.json"
CHECKLIST = REPORTS / "text_dimension_external_ocr_annotation_checklist_v1.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def has_bbox(ann: dict[str, Any]) -> bool:
    bbox = ann.get("bbox") or ann.get("bbox_normalized_xywh")
    polygon = ann.get("polygon")
    return (isinstance(bbox, list) and len(bbox) >= 4 and all(v is not None for v in bbox[:4])) or bool(polygon)


def has_transcript(ann: dict[str, Any]) -> bool:
    return bool(str(ann.get("verbatim_transcript") or ann.get("normalized_transcript") or "").strip())


def has_type(ann: dict[str, Any]) -> bool:
    return bool(str(ann.get("text_type") or "").strip())


def has_dimension_link(ann: dict[str, Any]) -> bool:
    value = ann.get("dimension_of")
    return value is not None and str(value).strip() != ""


def annotated_records(rows: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], int, int, int]:
    records: list[dict[str, Any]] = []
    full_annotations = 0
    type_annotations = 0
    link_annotations = 0
    for row in rows:
        anns = row.get("text_annotations") or []
        full = [ann for ann in anns if has_transcript(ann) and has_bbox(ann)]
        if full:
            records.append({**row, "filled_text_annotations": len(full)})
            full_annotations += len(full)
            type_annotations += sum(1 for ann in full if has_type(ann))
            link_annotations += sum(1 for ann in full if has_dimension_link(ann))
    return records, full_annotations, type_annotations, link_annotations


def checklist(rows: list[dict[str, Any]]) -> str:
    sample = rows[:20]
    sample_lines = "\n".join(
        f"- `{row.get('pack_id')}` `{row.get('image_path')}`: fill every visible text bbox/polygon, verbatim transcript, normalized transcript, text_type, dimension_of, legibility."
        for row in sample
    )
    return f"""# TextDimension External OCR Annotation Checklist v1

Goal: lock an external OCR smoke set for SCI2 claims. Minimum is 20 drawings with transcript and bbox gold; preferred is 50 drawings.

Required fields per visible text item:
- `bbox` or `polygon`: tight visible text extent in image coordinates or normalized coordinates.
- `verbatim_transcript`: exact visible text, preserving unit marks and punctuation.
- `normalized_transcript`: normalized string used for matching, with obvious spacing/case cleanup only.
- `text_type`: one of `dimension_text`, `room_label`, `other_text`, `illegible_text`.
- `dimension_of`: target room/object id when the text is a dimension; use `none` for non-dimension text.
- `legibility`: one of `clear`, `partial`, `illegible`.

QA rules:
- Mark a drawing complete only after all visible text has at least transcript+bbox.
- Do not infer invisible text from surrounding geometry.
- Keep illegible text as an annotation with `text_type=illegible_text`; do not delete it.
- A second reviewer should spot-check at least 10% of drawings and all `partial`/`illegible` cases.

First 20 drawings to finish:
{sample_lines}
"""


def main() -> int:
    rows = load_jsonl(PENDING)
    pack = load_json(PACK)
    text_v5 = load_json(TEXT_V5)
    reconciliation = load_json(RECONCILIATION)
    filled_records, filled_annotations, type_annotations, link_annotations = annotated_records(rows)
    has_min_lock = len(filled_records) >= 20
    status = "passed_external_lock" if has_min_lock else "pending_no_human_gold"

    CHECKLIST.write_text(checklist(rows), encoding="utf-8")
    report = {
        "version": "text_dimension_external_ocr_lock_v4",
        "created": "2026-05-03",
        "status": status,
        "input_pack": str(PACK.relative_to(ROOT)),
        "pending_jsonl": str(PENDING.relative_to(ROOT)),
        "annotation_checklist": str(CHECKLIST.relative_to(ROOT)),
        "selected_records": len(rows) or pack.get("selected_records"),
        "human_gold": {
            "drawings_with_transcript_and_bbox": len(filled_records),
            "text_annotations_with_transcript_and_bbox": filled_annotations,
            "text_annotations_with_text_type": type_annotations,
            "text_annotations_with_dimension_link": link_annotations,
            "minimum_drawings_for_smoke": 20,
            "preferred_drawings_for_sci2": 50,
            "smoke_metrics_available": has_min_lock,
        },
        "metrics": {
            "external_ocr_exact": None,
            "external_ocr_cer": None,
            "external_ocr_text_type_macro_f1": None,
            "external_ocr_dimension_link_f1": None,
            "per_quality": {},
            "reason_unavailable": None if has_min_lock else "pending_jsonl contains no filled human transcript+bbox gold.",
        },
        "current_internal_evidence": {
            "text_dimension_v5_source": str(TEXT_V5.relative_to(ROOT)) if TEXT_V5.exists() else None,
            "standalone_macro_f1": text_v5.get("macro_f1"),
            "standalone_dimension_link_f1": text_v5.get("dimension_link_f1"),
            "paper_e2e_text_family_boundary": (reconciliation.get("paper_main_metrics") or {}).get("node_per_label", {}).get("dimension_text"),
        },
        "claim_boundary": {
            "broad_real_ocr_claim_allowed": has_min_lock,
            "allowed_text": "Report TextDimension internal/locked split metrics and describe external OCR as annotation-ready until v4 passes.",
            "disallowed_text": "Do not claim broad scanned/photo OCR robustness while status is pending_no_human_gold.",
        },
        "done_when_status": "complete_annotation_ready_lock" if not has_min_lock else "complete_metric_lock",
    }
    write_json(OUTPUT, report)
    print(f"wrote {OUTPUT}")
    print(f"wrote {CHECKLIST}")
    print(json.dumps({"status": status, "drawings_with_gold": len(filled_records), "annotations_with_gold": filled_annotations}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
