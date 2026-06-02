#!/usr/bin/env python3
"""Build family-specific hard-case manifests from visual defect audits."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


FAMILY_TO_DIR = {
    "boundary": "boundary_expert_v3_hard_cases",
    "text": "text_dimension_expert_v6_hard_negative",
    "symbol": "symbol_fixture_expert_v11_hard_negative",
    "space": "room_space_expert_v3_polygon",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="reports/vlm/visual_demo/model_defect_cases_v3.jsonl")
    parser.add_argument("--converted", default="datasets/cadstruct_real_world_benchmark_v1/room_space/cubicasa5k_reviewed_locked_test.jsonl")
    parser.add_argument("--output-dir", default="datasets/cadstruct_hard_cases_v3")
    parser.add_argument("--manifest", default="reports/vlm/hard_case_dataset_manifest_v3.json")
    args = parser.parse_args()

    converted_by_image = {str(row.get("image_path") or row.get("image")): row for row in load_jsonl(Path(args.converted))}
    cases = [normalize_case(case, converted_by_image) for case in load_jsonl(Path(args.cases))]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    by_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for case in cases:
        bucket = FAMILY_TO_DIR.get(str(case.get("family") or ""), "misc_hard_cases")
        by_bucket[bucket].append(case)

    bucket_files = {}
    for bucket, rows in sorted(by_bucket.items()):
        bucket_dir = output_dir / bucket
        bucket_dir.mkdir(parents=True, exist_ok=True)
        path = bucket_dir / "manifest.jsonl"
        write_jsonl(path, sorted(rows, key=case_key))
        bucket_files[bucket] = str(path)

    manifest = build_manifest(args, cases, by_bucket, bucket_files)
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest["summary"], ensure_ascii=False, indent=2))


def normalize_case(case: dict[str, Any], converted_by_image: dict[str, dict[str, Any]]) -> dict[str, Any]:
    image = str(case.get("image") or "")
    converted = converted_by_image.get(image) or {}
    sample_id = str(case.get("sample_id") or Path(image).parent.name)
    family = str(case.get("family") or "unknown")
    semantic = str(case.get("semantic_type") or "unknown")
    bbox = normalize_bbox(case.get("bbox"))
    gold = gold_candidate_for_case(case, converted)
    return {
        "case_id": f"{sample_id}:{case.get('node_id')}:{case.get('type')}",
        "sample_id": sample_id,
        "image": image,
        "annotation": case.get("annotation"),
        "node_id": case.get("node_id"),
        "family": family,
        "semantic_type": semantic,
        "defect_type": case.get("type"),
        "severity": case.get("severity"),
        "recommended_layer": case.get("recommended_layer"),
        "source_expert": case.get("source_expert"),
        "model_source": case.get("model_source"),
        "proposal_source": case.get("proposal_source"),
        "raw_label": case.get("raw_label"),
        "base_raw_label": case.get("base_raw_label"),
        "confidence": case.get("confidence"),
        "bbox": bbox,
        "gold_label": gold.get("label"),
        "gold_text": gold.get("text"),
        "gold_bbox": gold.get("bbox"),
        "source_geometry": gold.get("geometry"),
        "review_reason": case.get("reason"),
        "training_intent": training_intent(case),
        "split": "visual_demo_review",
        "leakage_policy": "Do not merge visual_demo_review rows into locked_test; use for hard-case training/dev only after split filtering.",
        "render_assets": case.get("render_assets") or {},
    }


def gold_candidate_for_case(case: dict[str, Any], converted: dict[str, Any]) -> dict[str, Any]:
    expected = converted.get("expected_json") if isinstance(converted.get("expected_json"), dict) else {}
    node_id = str(case.get("node_id") or "")
    family = str(case.get("family") or "")
    if family == "boundary":
        target_id = node_id.replace("boundary_", "", 1)
        for item in expected.get("semantic_candidates") or []:
            if str(item.get("target_id")) == target_id:
                return {"label": item.get("semantic_type"), "bbox": item.get("bbox"), "geometry": item.get("geometry")}
    if family == "space":
        for item in expected.get("room_candidates") or []:
            if str(item.get("id")) == node_id:
                return {"label": item.get("room_type"), "bbox": item.get("bbox"), "geometry": item.get("geometry")}
    if family == "symbol":
        for item in expected.get("symbol_candidates") or []:
            if str(item.get("id")) == node_id:
                return {"label": item.get("symbol_type"), "bbox": item.get("bbox"), "geometry": item.get("geometry")}
    if family == "text":
        for item in expected.get("text_candidates") or []:
            if str(item.get("id")) == node_id:
                return {"label": item.get("text_type"), "text": item.get("text"), "bbox": item.get("bbox"), "geometry": item.get("geometry")}
    return {}


def training_intent(case: dict[str, Any]) -> str:
    defect_type = str(case.get("type") or "")
    family = str(case.get("family") or "")
    if defect_type in {"bbox_outside_canvas", "unsupported_wall"}:
        return "geometry_negative_or_boundary_relabel"
    if defect_type in {"metadata_missing", "missing_visible_text", "label_without_room"} or family == "text":
        return "text_metadata_recall_and_non_text_rejection"
    if defect_type in {"empty_symbol", "needs_review_symbol"} or family == "symbol":
        return "symbol_visual_evidence_negative"
    if defect_type in {"extra_room", "room_without_label"} or family == "space":
        return "room_polygon_validity_negative"
    return "fusion_review"


def build_manifest(
    args: argparse.Namespace,
    cases: list[dict[str, Any]],
    by_bucket: dict[str, list[dict[str, Any]]],
    bucket_files: dict[str, str],
) -> dict[str, Any]:
    defect_counts = Counter(str(case.get("defect_type")) for case in cases)
    family_counts = Counter(str(case.get("family")) for case in cases)
    layer_counts = Counter(str(case.get("recommended_layer")) for case in cases)
    return {
        "version": "cadstruct_hard_cases_v3",
        "inputs": {"cases": args.cases, "converted": args.converted},
        "output_dir": args.output_dir,
        "summary": {
            "cases": len(cases),
            "buckets": {bucket: len(rows) for bucket, rows in sorted(by_bucket.items())},
            "defect_counts": dict(defect_counts.most_common()),
            "family_counts": dict(family_counts.most_common()),
            "recommended_layer_counts": dict(layer_counts.most_common()),
            "split_leakage_status": "visual_demo_review only; locked_test insertion is explicitly blocked by manifest policy",
        },
        "bucket_files": bucket_files,
        "sampling_policy": {
            "boundary": "oversample bbox_outside_canvas and oversized opening candidates as hard negatives",
            "text": "oversample missing_visible_text and non-readable text predictions; preserve SVG text metadata",
            "symbol": "oversample low-ink equipment/symbol false positives and outside-room symbols",
            "space": "oversample tiny/unsupported room proposals and label-room inconsistency",
        },
    }


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(item) for item in value]
    except (TypeError, ValueError):
        return None


def case_key(case: dict[str, Any]) -> tuple[str, str, str]:
    return (str(case.get("sample_id") or ""), str(case.get("defect_type") or ""), str(case.get("node_id") or ""))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


if __name__ == "__main__":
    main()
