#!/usr/bin/env python3
"""Audit raster/E2E assets and claim boundaries for CadStruct v8."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

from v8_raster_e2e_utils import ROOT, load_json, load_jsonl, markdown_table, sample_key, update_todo_remove, write_json


ASSETS = [
    "datasets/cadstruct_graph_nodes_paper_v2_source_raster",
    "datasets/cadstruct_graph_nodes_lie_topology_raster_v3",
    "datasets/symbol_fixture_detector_v2",
    "datasets/cadstruct_cubicasa5k_moe",
    "reports/vlm/paper_v2_h512_l2_source_raster_locked_test.json",
    "reports/vlm/e2e_scene_graph_v1_eval.json",
    "reports/vlm/real_model_locked_eval_v7.json",
    "reports/vlm/visual_defect_ablation_v7.json",
    "reports/vlm/real_upstream_model_predictions_model_v7.jsonl",
    "scripts/vlm/evaluate_e2e_scene_graph.py",
    "scripts/vlm/apply_node_quality_gate.py",
]


def main() -> None:
    audited = [audit_asset(path) for path in ASSETS]
    reusable = [item for item in audited if item["exists"] and item["v8_reuse"] != "reject_for_pure_raster_claim"]
    report = {
        "version": "raster_e2e_asset_audit_v8",
        "created": "2026-05-07",
        "assets": audited,
        "summary": {
            "asset_count": len(audited),
            "existing_count": sum(1 for item in audited if item["exists"]),
            "source_modes": dict(Counter(item["source_classification"] for item in audited)),
            "can_support_pure_raster_claim_without_new_detector": False,
            "reusable_for_v8": [item["path"] for item in reusable],
        },
        "claim_boundary": (
            "Existing source-raster node/classification assets use image pixels as features but use SVG/parser labels or "
            "candidate geometry for supervision/proposals. They are useful for training/evaluation labels and hybrid evidence, "
            "but they do not by themselves prove pure raster end-to-end detection. v8 therefore requires a separate image-only "
            "candidate detector stream and adoption decision."
        ),
    }
    write_json("reports/vlm/raster_e2e_asset_audit_v8.json", report)
    write_markdown(report, ROOT / "reports/vlm/raster_e2e_asset_audit_v8.md")
    update_todo_remove(["RASTER-V8-T1"])
    print(json.dumps({"output": "reports/vlm/raster_e2e_asset_audit_v8.json", "assets": len(audited)}, ensure_ascii=False, indent=2))


def audit_asset(path: str) -> dict[str, Any]:
    p = ROOT / path
    item: dict[str, Any] = {"path": path, "exists": p.exists()}
    if not p.exists():
        item.update({"source_classification": "missing", "v8_reuse": "unavailable", "evidence": {}})
        return item
    evidence: dict[str, Any] = {}
    if p.is_dir():
        files = sorted(child.name for child in p.iterdir())[:20]
        evidence["files"] = files
        for split in ["train", "dev", "locked", "smoke"]:
            rows = load_jsonl(p / f"{split}.jsonl")
            if rows:
                evidence[f"{split}_rows"] = len(rows)
                evidence[f"{split}_sample_keys"] = len({sample_key(row.get("image") or row.get("image_path")) for row in rows})
                evidence[f"{split}_keys_preview"] = sorted({sample_key(row.get("image") or row.get("image_path")) for row in rows})[:5]
        manifest = load_json(p / "manifest.json", {})
        if manifest:
            evidence["manifest_keys"] = list(manifest)[:12]
            evidence["manifest_source"] = manifest.get("source")
    elif p.suffix == ".json":
        data = load_json(p, {})
        evidence["keys"] = list(data)[:20] if isinstance(data, dict) else []
        if isinstance(data, dict):
            for key in ["dataset", "predictions", "gold", "checkpoint", "records", "nodes", "version", "claim_boundary"]:
                if key in data:
                    evidence[key] = data.get(key)
    elif p.suffix == ".jsonl":
        rows = load_jsonl(p)
        evidence["rows"] = len(rows)
        if rows:
            evidence["first_keys"] = list(rows[0])[:20]
            rt = rows[0].get("route_trace") if isinstance(rows[0].get("route_trace"), dict) else {}
            evidence["first_source_mode"] = rt.get("source_mode")
            evidence["first_candidate_geometry_source"] = rt.get("candidate_geometry_source")
    else:
        evidence["kind"] = "script_or_other"

    classification, reuse = classify(path, evidence)
    item.update({"source_classification": classification, "v8_reuse": reuse, "evidence": evidence, "notes": notes_for(path, classification)})
    return item


def classify(path: str, evidence: dict[str, Any]) -> tuple[str, str]:
    low = path.lower()
    if "real_upstream_model_predictions_model_v7" in low:
        return "svg_candidate", "baseline_only"
    if "cadstruct_cubicasa5k_moe" in low:
        return "source_raster_with_svg_labels", "offline_gold_label_source"
    if "source_raster" in low or "lie_topology_raster" in low:
        return "source_raster_with_svg_labels", "feature_pretraining_or_label_audit"
    if "symbol_fixture_detector" in low:
        return "hybrid", "symbol_crop_or_semantic_auxiliary_only"
    if "e2e_scene_graph" in low:
        return "hybrid", "baseline_metric_context"
    if "quality_gate" in low:
        return "postprocess", "postprocess_baseline_only"
    if "real_model_locked_eval_v7" in low or "visual_defect_ablation_v7" in low:
        return "svg_candidate", "baseline_metric_context"
    return "unknown", "inspect_before_use"


def notes_for(path: str, classification: str) -> str:
    if classification == "source_raster_with_svg_labels":
        return "May use PNG pixels as model features and SVG as offline labels; cannot be used as inference-time SVG proposal geometry."
    if classification == "svg_candidate":
        return "Useful as v7 baseline only; not evidence for pure raster detection."
    if classification == "hybrid":
        return "Useful for comparison or auxiliary model evidence when source mode is labeled hybrid."
    if classification == "postprocess":
        return "Can be audited separately but must not receive model-recognition credit."
    return "Requires manual inspection before claims."


def write_markdown(report: dict[str, Any], path: Path) -> None:
    rows = [["Asset", "Exists", "Class", "v8 reuse"]]
    for item in report["assets"]:
        rows.append([item["path"], item["exists"], item["source_classification"], item["v8_reuse"]])
    text = "\n".join(
        [
            "# Raster E2E Asset Audit v8",
            "",
            report["claim_boundary"],
            "",
            markdown_table(rows),
            "",
            "Conclusion: existing assets do not support a pure raster E2E claim without a new image-only detector stream.",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


if __name__ == "__main__":
    main()
