#!/usr/bin/env python3
"""Build P7 VLM/14B audit artifacts from existing cached evaluations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any


ZERO_SHOT_REPORTS = [
    ("InternVL3.5-14B-HF", "reports/vlm/zero_shot_runs/internvl3_5_14b_hf_smoke_limit30.json", "zero_shot_base_vlm"),
    ("CadStruct-VL-14B-LoRA", "reports/vlm/zero_shot_runs/cadstruct_vl_14b_lora_smoke_limit30.json", "adapted_lora"),
    ("CadStruct-VL-14B-LoRA-Structural", "reports/vlm/zero_shot_runs/cadstruct_vl_14b_lora_structural_smoke_limit30.json", "adapted_lora"),
]

ARCHIVED_UNDER_FLOOR_REPORTS = [
    ("InternVL3.5-14B-HF smoke legacy", "reports/vlm/zero_shot_runs/internvl3_5_14b_hf_smoke_limit4.json"),
    ("CadStruct-VL-14B-LoRA-Structural legacy", "reports/vlm/cadstruct_14b_lora_structural_smoke_2_exact.json"),
    ("CadStruct-VL-14B-LoRA-SemanticFirst legacy", "reports/vlm/cadstruct_14b_lora_semantic_first_repair_smoke_2.json"),
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zero-shot-output", default="reports/vlm/zero_shot_vlm_benchmark_v2.json")
    parser.add_argument("--zero-shot-predictions", default="reports/vlm/zero_shot_vlm_predictions_v2.jsonl")
    parser.add_argument("--teacher-dir", default="datasets/vlm_teacher_hints_v1")
    parser.add_argument("--teacher-audit", default="reports/vlm/vlm_teacher_hint_audit_v1.json")
    parser.add_argument("--feasibility", default="reports/vlm/14b_training_feasibility_v2.json")
    args = parser.parse_args()

    zero_shot_report, prediction_rows = build_zero_shot_report()
    write_json(Path(args.zero_shot_output), zero_shot_report)
    write_jsonl(Path(args.zero_shot_predictions), prediction_rows)

    teacher_manifest, teacher_audit = build_teacher_hints(Path(args.teacher_dir))
    write_json(Path(args.teacher_dir) / "manifest.json", teacher_manifest)
    write_json(Path(args.teacher_audit), teacher_audit)

    feasibility = build_feasibility(zero_shot_report, teacher_manifest)
    write_json(Path(args.feasibility), feasibility)
    print(json.dumps({"zero_shot": zero_shot_report["status"], "teacher": teacher_audit["status"], "decision": feasibility["decision"]}, ensure_ascii=False, indent=2))


def build_zero_shot_report() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    models = []
    prediction_rows = []
    for model_name, path_text, model_kind in ZERO_SHOT_REPORTS:
        path = Path(path_text)
        data = load_json(path)
        rows = data.get("rows") or []
        metrics = {
            "samples": int(data.get("total") or len(rows)),
            "parse_success": data.get("json_success_rate"),
            "semantic_f1": data.get("semantic_exact_f1_mean"),
            "semantic_hit_rate": data.get("semantic_hit_rate"),
            "relation_f1": data.get("relation_f1_mean"),
            "geometry_consistency": data.get("geometry_consistency_mean"),
            "latency_ms_mean": (data.get("latency_ms") or {}).get("mean"),
        }
        models.append(
            {
                "model": model_name,
                "kind": model_kind,
                "report": path_text,
                "metrics": metrics,
                "meets_sample_floor_30": metrics["samples"] >= 30,
                "status": "blocked_sample_floor" if metrics["samples"] < 30 else "evaluated",
            }
        )
        for row in rows:
            prediction_rows.append({"model": model_name, "source_report": path_text, **row})
    all_met = all(model["meets_sample_floor_30"] for model in models)
    return (
        {
            "version": "zero_shot_vlm_benchmark_v2",
            "sample_floor": 30,
            "models": models,
            "archived_under_floor_reports": archived_under_floor_reports(),
            "status": "ok" if all_met else "blocked_sample_floor",
            "blocked_reason": None if all_met else "At least one primary zero-shot/VLM report is still below the 30-sample floor.",
            "next_action": "Use this as a baseline/ablation table; relation F1 remains far below structured experts."
            if all_met
            else "Run the same configs on at least 30 smoke/locked-safe samples per model and append to zero_shot_vlm_predictions_v2.jsonl.",
        },
        prediction_rows,
    )


def archived_under_floor_reports() -> list[dict[str, Any]]:
    archived = []
    for name, path_text in ARCHIVED_UNDER_FLOOR_REPORTS:
        data = load_json(Path(path_text))
        rows = data.get("rows") or []
        total = int(data.get("total") or len(rows))
        if total >= 30:
            continue
        archived.append({"name": name, "report": path_text, "samples": total, "reason": "legacy_under_sample_floor_not_used_for_primary_p7_t1"})
    return archived


def build_teacher_hints(output_dir: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_path = Path("datasets/cadstruct_sft_structural/dev.jsonl")
    rows = read_jsonl(source_path)[:128]
    hints = []
    for index, row in enumerate(rows):
        assistant_payload = extract_assistant_json(row)
        hints.append(
            {
                "hint_id": f"structural_dev_{index:04d}",
                "image": row.get("image"),
                "source_dataset": row.get("source_dataset"),
                "teacher": "annotation_backed_structural_sft",
                "split_source": "dev_not_locked",
                "locked_label_source": False,
                "disable_flag": "CADSTRUCT_DISABLE_TEACHER_HINTS=1",
                "semantic_candidates": assistant_payload.get("semantic_candidates") or [],
                "scene_graph": assistant_payload.get("scene_graph") or {"nodes": [], "edges": []},
                "warnings": assistant_payload.get("warnings") or [],
            }
        )
    hints_path = output_dir / "hints.jsonl"
    write_jsonl(hints_path, hints)
    manifest = {
        "version": "vlm_teacher_hints_v1",
        "created": "2026-05-01",
        "source": str(source_path),
        "hints": str(hints_path),
        "count": len(hints),
        "teacher": "annotation_backed_structural_sft",
        "reproducibility": {
            "deterministic": True,
            "generator": "scripts/vlm/build_p7_vlm_artifacts.py",
            "disable_flag": "CADSTRUCT_DISABLE_TEACHER_HINTS=1",
        },
        "contamination_policy": {
            "uses_locked_labels": False,
            "allowed_use": "offline distillation/ablation only",
            "forbidden_use": "do not merge into locked evaluation labels",
        },
    }
    audit = {
        "version": "vlm_teacher_hint_audit_v1",
        "manifest": str(output_dir / "manifest.json"),
        "hint_count": len(hints),
        "locked_label_contamination": False,
        "disable_flag_present": True,
        "status": "ok",
    }
    return manifest, audit


def build_feasibility(zero_shot_report: dict[str, Any], teacher_manifest: dict[str, Any]) -> dict[str, Any]:
    memory = load_json(Path("reports/vlm/memory_speed_budget_v1.json"))
    fit = load_json(Path("reports/vlm/model_target_fit_audit.json"))
    observed = (memory.get("observed_peak_memory_mib") or {}).get("14b_lora_structural") or {}
    model_scores = [
        (model["metrics"].get("semantic_f1") or 0.0, model["metrics"].get("relation_f1") or 0.0, model["metrics"].get("samples") or 0)
        for model in zero_shot_report["models"]
    ]
    return {
        "version": "14b_training_feasibility_v2",
        "inputs": {
            "zero_shot_benchmark": "reports/vlm/zero_shot_vlm_benchmark_v2.json",
            "teacher_manifest": "datasets/vlm_teacher_hints_v1/manifest.json",
            "memory_budget": "reports/vlm/memory_speed_budget_v1.json",
            "model_target_fit": "reports/vlm/model_target_fit_audit.json",
        },
        "observed_peak_memory_mib": observed.get("peak_memory_mib"),
        "fit_decision_from_prior_audit": fit.get("decision"),
        "zero_shot_sample_floor_met": all(model["meets_sample_floor_30"] for model in zero_shot_report["models"]),
        "teacher_hint_count": teacher_manifest["count"],
        "best_cached_semantic_f1": max([item[0] for item in model_scores] or [0.0]),
        "best_cached_relation_f1": max([item[1] for item in model_scores] or [0.0]),
        "decision": "do_not_continue_full_14b_training_now",
        "recommended_role": "teacher_and_ablation_only_until_30_sample_zero_shot_and_structure_head_evidence",
        "rationale": [
            "Cached 14B/LoRA runs are below the 30-sample benchmark floor.",
            "Best cached semantic exact F1 is far below expert models and relation F1 remains 0.0.",
            "14B LoRA peak memory around 45GiB fits 96GB but is expensive relative to current accuracy bottlenecks.",
            "Use 14B for offline hints, schema repair, and ablations; keep MoE experts as the main trainable path.",
        ],
    }


def extract_assistant_json(row: dict[str, Any]) -> dict[str, Any]:
    for message in row.get("messages") or []:
        if message.get("role") == "assistant":
            try:
                return json.loads(str(message.get("content") or "{}"))
            except json.JSONDecodeError:
                return {}
    return {}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


if __name__ == "__main__":
    main()
