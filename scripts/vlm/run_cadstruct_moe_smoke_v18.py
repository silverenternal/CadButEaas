#!/usr/bin/env python3
"""One-command smoke manifest for the raster-only CadStruct MoE v18 path."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports/vlm/cadstruct_moe_smoke_manifest_v18.json"
PROJECT_AUDIT = ROOT / "reports/vlm/project_structure_audit_v18.json"
V17_CONTRACT_AUDIT = ROOT / "reports/vlm/moe_expert_runtime_contract_v17.json"


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def git_status_summary() -> dict[str, Any]:
    result = subprocess.run(
        ["git", "status", "--short"],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    lines = [line for line in result.stdout.splitlines() if line.strip()]
    return {"returncode": result.returncode, "changed_path_count": len(lines), "sample": lines[:40]}


def run_step(command: list[str], *, enabled: bool = True) -> dict[str, Any]:
    if not enabled:
        return {"command": command, "skipped": True, "returncode": None}
    result = subprocess.run(
        command,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    return {
        "command": command,
        "skipped": False,
        "returncode": result.returncode,
        "stdout_tail": result.stdout.splitlines()[-20:],
        "stderr_tail": result.stderr.splitlines()[-20:],
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        return {"load_error": str(exc)}


def artifact(path: str, role: str) -> dict[str, Any]:
    p = ROOT / path
    return {
        "path": path,
        "role": role,
        "exists": p.exists(),
        "size_bytes": p.stat().st_size if p.exists() else 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--skip-v17-contract", action="store_true")
    args = parser.parse_args()
    started_at = datetime.now(timezone.utc).isoformat()

    steps = [
        run_step([
            ".venv/bin/python",
            "scripts/vlm/audit_project_structure_v18.py",
            "--struct",
            "struct.json",
            "--todo",
            "todo.json",
            "--config",
            "configs/vlm/cadstruct_project_structure_v18.json",
            "--output",
            "reports/vlm/project_structure_audit_v18.json",
        ]),
        run_step([
            ".venv/bin/python",
            "scripts/vlm/image_only_moe_v17_pipeline.py",
            "audit-contract",
        ], enabled=not args.skip_v17_contract),
    ]

    project_audit = load_json(PROJECT_AUDIT)
    contract_audit = load_json(V17_CONTRACT_AUDIT)
    failed_steps = [step for step in steps if not step.get("skipped") and step.get("returncode") != 0]
    contract_gate = (contract_audit.get("source_integrity_gate") or {}) if isinstance(contract_audit, dict) else {}
    project_gate = project_audit.get("gate_decision") if isinstance(project_audit, dict) else None
    gate_decision = "pass"
    failure_reasons: list[str] = []
    if failed_steps:
        gate_decision = "fail"
        failure_reasons.append("One or more smoke runner subprocesses failed.")
    if project_gate and project_gate != "pass":
        gate_decision = "fail"
        failure_reasons.append(f"Project structure audit gate is {project_gate}.")
    if contract_gate and not bool(contract_gate.get("passed")):
        gate_decision = "fail"
        failure_reasons.append("v17 runtime contract/source-integrity gate did not pass.")
    if args.skip_v17_contract and gate_decision == "pass":
        gate_decision = "diagnostic_only"
        failure_reasons.append("v17 contract audit was skipped by flag.")

    manifest = {
        "schema_version": "cadstruct_artifact_manifest_v1",
        "run_id": "cadstruct_moe_smoke_v18",
        "pipeline_id": "cadstruct_moe_smoke_v18",
        "started_at": started_at,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "mode": "executed_smoke_contracts",
        "command_or_commands": [" ".join(step["command"]) for step in steps],
        "steps": steps,
        "cwd": str(ROOT),
        "git_status_summary": git_status_summary(),
        "inputs": [
            artifact("configs/vlm/cadstruct_project_structure_v18.json", "project governance config"),
            artifact("configs/vlm/image_only_moe_contract_v1.json", "runtime contract"),
            artifact("reports/vlm/image_only_moe_predictions_v17.jsonl", "current v17 fused prediction stream"),
            artifact("reports/vlm/relation_graph_scored_rows_cache_v18.jsonl", "fixed relation selection universe"),
        ],
        "checkpoints": [],
        "outputs": [artifact("reports/vlm/cadstruct_moe_smoke_manifest_v18.json", "this manifest")],
        "eval_reports": [],
        "audit_reports": [
            artifact("reports/vlm/project_structure_audit_v18.json", "project structure gate"),
            artifact("reports/vlm/moe_expert_runtime_contract_v17.json", "v17 runtime/source-integrity gate"),
        ],
        "review_packs": [],
        "metrics_summary": {
            "project_structure_gate": project_gate,
            "project_uncategorized_scripts": len(project_audit.get("uncategorized_scripts") or []) if isinstance(project_audit, dict) else None,
            "v17_source_integrity_passed": contract_gate.get("passed"),
            "v17_source_integrity_violations": contract_gate.get("violations"),
        },
        "gate_decision": gate_decision,
        "failure_reasons": failure_reasons,
        "source_integrity_summary": {
            "inference_input": "raster image only",
            "svg_or_parser_geometry_allowed_at_runtime": False,
            "v17_contract_gate": contract_gate,
        },
    }
    write_json(Path(args.output), manifest)


if __name__ == "__main__":
    main()
