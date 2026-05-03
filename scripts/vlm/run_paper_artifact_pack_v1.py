#!/usr/bin/env python3
"""Regenerate paper-facing CadStruct-MoE artifacts and manifest."""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
OUTPUT = REPORTS / "paper_artifact_manifest_v1.json"

COMMANDS = [
    "scripts/vlm/audit_relation_no_repair_rule_sweep_v1.py",
    "scripts/vlm/audit_text_dimension_external_ocr_lock_v3.py",
    "scripts/vlm/audit_relation_no_repair_ceiling_diagnostic_v1.py",
    "scripts/vlm/audit_symbol_cross_source_lock_v1.py",
    "scripts/vlm/reconcile_paper_e2e_metrics.py",
    "scripts/vlm/generate_paper_tables_v2.py",
    "scripts/vlm/audit_final_claim_ledger_v1.py",
]

ARTIFACTS = [
    ("reports/vlm/paper_e2e_metric_reconciliation_v1.json", "main", "scripts/vlm/reconcile_paper_e2e_metrics.py", "paper main E2E metric source"),
    ("reports/vlm/scene_graph_fusion_symbol_label_arbitrated_no_repair_v2_eval.json", "main", "scripts/vlm/audit_relation_no_repair_rule_sweep_v1.py", "no-repair v2 scene graph evaluation"),
    ("reports/vlm/relation_no_repair_rule_sweep_v1.json", "appendix", "scripts/vlm/audit_relation_no_repair_rule_sweep_v1.py", "relation rule sweep"),
    ("reports/vlm/relation_no_repair_error_taxonomy_v1.json", "appendix", "scripts/vlm/audit_relation_no_repair_rule_sweep_v1.py", "relation error taxonomy"),
    ("reports/vlm/relation_no_repair_hard_cases_v1.jsonl", "appendix", "scripts/vlm/audit_relation_no_repair_ceiling_diagnostic_v1.py", "100 relation hard cases"),
    ("reports/vlm/relation_no_repair_ceiling_diagnostic_v1.json", "appendix", "scripts/vlm/audit_relation_no_repair_ceiling_diagnostic_v1.py", "relation oracle ceiling diagnostics"),
    ("reports/vlm/text_dimension_external_ocr_lock_v3.json", "limitation", "scripts/vlm/audit_text_dimension_external_ocr_lock_v3.py", "external OCR lock boundary"),
    ("reports/vlm/symbol_cross_source_lock_v1.json", "limitation", "scripts/vlm/audit_symbol_cross_source_lock_v1.py", "cross-source symbol lock boundary"),
    ("reports/vlm/final_claim_ledger_v1.json", "gate", "scripts/vlm/audit_final_claim_ledger_v1.py", "claim ledger"),
    ("reports/vlm/final_paper_boundary_v2.json", "gate", "scripts/vlm/audit_final_claim_ledger_v1.py", "paper boundary gate"),
    ("reports/vlm/paper_tables_v2/main_results.tex", "main", "scripts/vlm/generate_paper_tables_v2.py", "main result table"),
    ("reports/vlm/paper_tables_v2/ablation_results.tex", "main", "scripts/vlm/generate_paper_tables_v2.py", "ablation table"),
    ("reports/vlm/relation_gold_id_repair_sensitivity_v1.json", "appendix_upper_bound", "scripts/vlm/audit_relation_gold_id_repair_sensitivity_v1.py", "repair-enabled relation upper bound"),
    ("reports/vlm/router_appendix_topk_v1.json", "future", "scripts/vlm/build_router_appendix_topk_v1.py", "learned/top-k router appendix"),
    ("reports/vlm/lie_se2_core_or_auxiliary_decision_v2.json", "historical", "scripts/vlm/decide_lie_se2_core_or_auxiliary_v2.py", "Lie/SE(2) auxiliary decision"),
    ("reports/vlm/sheet_layout_real_gold_boundary_v1.json", "future", "scripts/vlm/audit_sheet_layout_real_gold_boundary_v1.py", "SheetLayout non-core boundary"),
    ("reports/vlm/real_upstream_latency_resource_v1.json", "appendix", "scripts/vlm/benchmark_real_upstream_latency_resource_v1.py", "local replay latency/resource"),
]


def sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def run_command(script: str) -> dict[str, Any]:
    proc = subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, text=True, capture_output=True)
    return {
        "script": script,
        "returncode": proc.returncode,
        "stdout_tail": proc.stdout[-4000:],
        "stderr_tail": proc.stderr[-4000:],
        "status": "passed" if proc.returncode == 0 else "failed",
    }


def artifact_row(path_str: str, role: str, producer: str, description: str) -> dict[str, Any]:
    path = ROOT / path_str
    stat = path.stat() if path.exists() else None
    return {
        "path": path_str,
        "exists": path.exists(),
        "sha256": sha256(path),
        "mtime": stat.st_mtime if stat else None,
        "size_bytes": stat.st_size if stat else None,
        "paper_role": role,
        "producer_script": producer,
        "description": description,
    }


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def main() -> int:
    command_results = [run_command(script) for script in COMMANDS]
    artifacts = [artifact_row(*item) for item in ARTIFACTS]
    boundary = load_json(REPORTS / "final_paper_boundary_v2.json")
    ledger = load_json(REPORTS / "final_claim_ledger_v1.json")
    required_missing = [row["path"] for row in artifacts if row["paper_role"] in {"main", "gate", "limitation"} and not row["exists"]]
    failed_commands = [row for row in command_results if row["returncode"] != 0]
    status = "passed" if not failed_commands and not required_missing and boundary.get("status") == "passed" and ledger.get("status") == "passed" else "needs_attention"
    manifest = {
        "version": "paper_artifact_manifest_v1",
        "created": "2026-05-03",
        "status": status,
        "commands": command_results,
        "artifacts": artifacts,
        "checks": {
            "commands_passed": not failed_commands,
            "required_artifacts_present": not required_missing,
            "claim_ledger_passed": ledger.get("status") == "passed",
            "final_paper_boundary_passed": boundary.get("status") == "passed",
        },
        "required_missing": required_missing,
    }
    OUTPUT.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(json.dumps({"status": status, "failed_commands": len(failed_commands), "required_missing": required_missing}, ensure_ascii=False, indent=2))
    return 0 if status == "passed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
