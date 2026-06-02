#!/usr/bin/env python3
"""Overlay raster objectness scores onto the safe v18 symbol stream.

This keeps the existing safe candidate universe intact. The raster objectness
expert contributes ranking/audit features only; it does not add or remove
runtime symbol candidates.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
DEFAULT_BASE = REPORT / "symbol_detector_v18_safe_routed_candidates.jsonl"
DEFAULT_SCORED = REPORT / "symbol_detector_v18_objectness_raster_routed_candidates.jsonl"
DEFAULT_OUTPUT = REPORT / "symbol_detector_v18_safe_objectness_overlay_routed_candidates.jsonl"
DEFAULT_AUDIT = REPORT / "symbol_detector_v18_safe_objectness_overlay_audit.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def candidate_key(row: dict[str, Any]) -> tuple[str, str]:
    return str(row.get("row_id")), str(row.get("candidate_id") or row.get("id"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(DEFAULT_BASE))
    parser.add_argument("--scores", default=str(DEFAULT_SCORED))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--confidence-mode", choices=["objectness", "blend", "payload-only"], default="objectness")
    args = parser.parse_args()

    base_rows = load_jsonl(Path(args.base))
    scored_rows = load_jsonl(Path(args.scores))
    score_by_key: dict[tuple[str, str], dict[str, Any]] = {candidate_key(row): row for row in scored_rows}
    counts = Counter()
    output: list[dict[str, Any]] = []

    for row in base_rows:
        item = json.loads(json.dumps(row))
        payload = dict(item.get("payload") if isinstance(item.get("payload"), dict) else {})
        score_row = score_by_key.get(candidate_key(item))
        counts["base_candidates"] += 1
        if score_row is None:
            counts["unmatched"] += 1
            payload["objectness_overlay_status"] = "unmatched_candidate_id"
            item["payload"] = payload
            output.append(item)
            continue
        counts["matched"] += 1
        score_payload = score_row.get("payload") if isinstance(score_row.get("payload"), dict) else {}
        objectness_score = float(score_payload.get("objectness_score") or score_row.get("confidence") or 0.0)
        old_confidence = float(item.get("confidence") or 0.0)
        if args.confidence_mode == "objectness":
            item["confidence"] = round(objectness_score, 6)
        elif args.confidence_mode == "blend":
            item["confidence"] = round(0.65 * objectness_score + 0.35 * old_confidence, 6)
        payload.update(
            {
                "objectness_score": round(objectness_score, 6),
                "objectness_model": score_payload.get("objectness_model") or "symbol_objectness_type_v18_raster_crop_ranker",
                "objectness_overlay_status": "matched_candidate_id",
                "objectness_overlay_confidence_mode": args.confidence_mode,
                "pre_objectness_confidence": round(old_confidence, 6),
            }
        )
        item["payload"] = payload
        trace = dict(item.get("audit_trace") if isinstance(item.get("audit_trace"), dict) else {})
        trace["symbol_objectness_overlay_v18"] = {
            "candidate_universe_changed": False,
            "score_source": str(args.scores),
            "confidence_mode": args.confidence_mode,
        }
        item["audit_trace"] = trace
        output.append(item)

    audit = {
        "task": "IMG-MOE-V18-REBUILD-002.step_symbol_objectness_overlay",
        "base": str(args.base),
        "scores": str(args.scores),
        "output": str(args.output),
        "confidence_mode": args.confidence_mode,
        "counts": dict(counts),
        "match_rate": round(counts["matched"] / max(counts["base_candidates"], 1), 6),
        "candidate_universe_changed": False,
        "new_runtime_candidates_created": False,
        "runtime_input_contract": {
            "model_input": "raster_image_only",
            "svg_or_parser_geometry_used_for_inference": False,
            "gold_used_for_inference": False,
        },
    }
    write_jsonl(Path(args.output), output)
    write_json(Path(args.audit), audit)
    print(json.dumps({"output": len(output), "matched": counts["matched"], "unmatched": counts["unmatched"], "match_rate": audit["match_rate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
