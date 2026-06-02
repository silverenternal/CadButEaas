#!/usr/bin/env python3
"""Audit CadStruct raster-only MoE project structure against struct/todo.

This script does not judge model quality. It makes project drift visible:
which entrypoints are canonical, which files are diagnostic/legacy, and
whether the current plan still points at low-yield relation tuning.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/vlm/cadstruct_project_structure_v18.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def classify_script(path: Path, config: dict[str, Any]) -> str:
    name = path.name
    relative = rel(path)
    lowered = relative.lower()
    classes = config.get("script_classes") or {}
    for class_name, spec in classes.items():
        explicit_paths = set(str(item) for item in spec.get("explicit_paths") or [])
        if relative in explicit_paths:
            return str(class_name)
    for class_name in (
        "legacy_svg_or_offline_label_helper",
        "diagnostic_or_review_helper",
        "shared_library_or_test",
        "model_pipeline_candidate",
    ):
        spec = classes.get(class_name) or {}
        if any(token in lowered for token in spec.get("path_contains") or []):
            return class_name
        if any(name.startswith(str(prefix)) for prefix in spec.get("filename_prefixes") or []):
            return class_name
    return "uncategorized"


def existing_paths(paths: list[str]) -> dict[str, bool]:
    return {path: (ROOT / path).exists() for path in paths}


def collect_struct_entrypoints(struct: dict[str, Any]) -> list[str]:
    entrypoints = struct.get("pipeline_entrypoints") or {}
    commands: list[str] = []
    for spec in entrypoints.values():
        command = spec.get("command")
        if isinstance(command, str):
            commands.append(command)
        for item in spec.get("commands") or []:
            if isinstance(item, str):
                commands.append(item)
    return commands


def task_status_counts(todo: dict[str, Any]) -> dict[str, int]:
    return dict(Counter(str(task.get("status") or "unknown") for task in todo.get("tasks") or []))


def low_yield_relation_tuning_active(todo: dict[str, Any]) -> list[str]:
    active: list[str] = []
    for task in todo.get("tasks") or []:
        task_id = str(task.get("id") or "unknown")
        if "GOVERNANCE" in task_id:
            continue
        status = str(task.get("status") or "")
        text = json.dumps(task, ensure_ascii=False).lower()
        if status in {"in_progress", "pending"} and "contains_symbol" in text and "cap" in text:
            active.append(task_id)
    return active


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--struct", default="struct.json")
    parser.add_argument("--todo", default="todo.json")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG.relative_to(ROOT)))
    parser.add_argument("--output", default="reports/vlm/project_structure_audit_v18.json")
    args = parser.parse_args()

    struct_path = ROOT / args.struct
    todo_path = ROOT / args.todo
    config_path = ROOT / args.config
    struct = load_json(struct_path)
    todo = load_json(todo_path)
    config = load_json(config_path)

    scripts = sorted((ROOT / "scripts/vlm").glob("*.py"))
    classified = [{"path": rel(path), "class": classify_script(path, config)} for path in scripts]
    class_counts = Counter(item["class"] for item in classified)

    required_entrypoints = config.get("required_entrypoints") or {}
    expected_new = [
        str(spec.get("path"))
        for spec in required_entrypoints.values()
        if isinstance(spec, dict) and spec.get("path")
    ]
    expected_new.append("scripts/vlm/audit_project_structure_v18.py")
    entrypoint_commands = collect_struct_entrypoints(struct)
    governance_task = next(
        (task for task in todo.get("tasks") or [] if task.get("id") == "IMG-MOE-V18-GOVERNANCE-007"),
        None,
    )

    findings: list[dict[str, Any]] = []
    if governance_task is None:
        findings.append({"severity": "high", "issue": "missing_governance_reset_task"})
    if low_yield_relation_tuning_active(todo):
        findings.append({
            "severity": "medium",
            "issue": "contains_symbol_cap_tuning_still_active",
            "tasks": low_yield_relation_tuning_active(todo),
        })
    smoke_runner_present = any("run_cadstruct_moe_smoke_v18.py" in cmd for cmd in entrypoint_commands)
    locked_runner_present = any("run_cadstruct_moe_locked_v18.py" in cmd for cmd in entrypoint_commands)
    if not smoke_runner_present:
        findings.append({"severity": "medium", "issue": "struct_missing_v18_smoke_runner_entry"})
    if not locked_runner_present:
        findings.append({"severity": "medium", "issue": "struct_missing_v18_locked_runner_entry"})
    if class_counts.get("uncategorized", 0) > 20:
        findings.append({
            "severity": "low",
            "issue": "many_uncategorized_scripts",
            "count": class_counts.get("uncategorized", 0),
        })

    audit = {
        "schema_version": "cadstruct_project_structure_audit_v18",
        "struct": rel(struct_path),
        "todo": rel(todo_path),
        "config": rel(config_path),
        "hard_contract": struct.get("hard_contract", {}),
        "gate_policy": config.get("gate_policy", {}),
        "task_status_counts": task_status_counts(todo),
        "near_term_focus": todo.get("near_term_focus"),
        "expected_governance_files": existing_paths(expected_new),
        "script_class_counts": dict(class_counts),
        "script_classification": classified,
        "canonical_or_core_assets": [
            item["path"] for item in classified if item["class"] == "canonical_or_core_asset"
        ],
        "shared_libraries_or_tests_sample": [
            item["path"] for item in classified if item["class"] == "shared_library_or_test"
        ][:80],
        "uncategorized_scripts": [
            item["path"] for item in classified if item["class"] == "uncategorized"
        ],
        "module_ownership": config.get("module_ownership", {}),
        "diagnostic_or_review_helpers_sample": [
            item["path"] for item in classified if item["class"] == "diagnostic_or_review_helper"
        ][:80],
        "legacy_svg_or_offline_label_helpers_sample": [
            item["path"] for item in classified if item["class"] == "legacy_svg_or_offline_label_helper"
        ][:80],
        "findings": findings,
        "gate_decision": "fail" if any(f["severity"] in {"high", "medium"} for f in findings) else "pass",
        "next_actions": [
            "Keep relation compression experiments diagnostic until upstream raster expert gates improve.",
            "Reduce uncategorized scripts by adding a reviewed canonical/diagnostic/legacy classification manifest.",
            "Replace manifest-only v18 runners with runners that execute the canonical smoke and locked stages.",
        ],
    }
    write_json(ROOT / args.output, audit)


if __name__ == "__main__":
    main()
