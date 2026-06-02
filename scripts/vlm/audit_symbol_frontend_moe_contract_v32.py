#!/usr/bin/env python3
"""Audit the raster-only symbol frontend MoE contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONTRACT = ROOT / "configs/vlm/symbol_frontend_moe_contract_v32.json"
DEFAULT_OUTPUT = ROOT / "reports/vlm/symbol_frontend_moe_contract_v32_audit.json"

REQUIRED_TOP_LEVEL_KEYS = {
    "version",
    "runtime_contract",
    "candidate_set_schema",
    "metric_claim_policy",
    "experts",
    "current_artifact_assignment_required",
}
REQUIRED_EXPERT_KEYS = {
    "id",
    "status",
    "primary_metric",
    "must_not_optimize",
    "inputs",
    "outputs",
    "runtime_fields",
    "runtime_forbidden_fields",
    "stage_gates",
    "assets",
}
FORBIDDEN_RUNTIME_TOKENS = {
    "gold",
    "expected_json",
    "annotation_path",
    "svg",
    "parser_geometry",
    "cad vector",
    "oracle",
    "semantic_id",
}


def source_path(path: str) -> Path:
    candidate = Path(path)
    return candidate if candidate.is_absolute() else ROOT / candidate


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def contains_forbidden_runtime_token(value: str) -> str | None:
    lowered = value.lower()
    for token in sorted(FORBIDDEN_RUNTIME_TOKENS):
        if token in lowered:
            return token
    return None


def audit_contract(contract: dict[str, Any]) -> dict[str, Any]:
    errors: list[str] = []
    warnings: list[str] = []
    missing_optional_assets: list[dict[str, str]] = []
    present_required_assets: list[dict[str, str]] = []
    expert_summaries: list[dict[str, Any]] = []

    missing_top = sorted(REQUIRED_TOP_LEVEL_KEYS - set(contract))
    for key in missing_top:
        errors.append(f"missing top-level key: {key}")

    runtime_allowed = set(contract.get("runtime_contract", {}).get("allowed_inputs", []))
    runtime_forbidden = set(contract.get("runtime_contract", {}).get("forbidden_inputs", []))
    if "raster image pixels" not in runtime_allowed:
        errors.append("runtime_contract.allowed_inputs must include raster image pixels")
    for required_forbidden in ("gold labels", "expected_json", "SVG/parser geometry"):
        if required_forbidden not in runtime_forbidden:
            errors.append(f"runtime_contract.forbidden_inputs must include {required_forbidden}")

    schema = contract.get("candidate_set_schema", {})
    required_runtime_fields = set(schema.get("required_runtime_fields", []))
    forbidden_schema_fields = set(schema.get("forbidden_runtime_fields", []))
    for field in ("page_id", "candidate_id", "bbox", "score", "label", "label_id", "proposal_source"):
        if field not in required_runtime_fields:
            errors.append(f"candidate_set_schema.required_runtime_fields missing {field}")
    for field in ("gold_key", "expected_json", "annotation_path", "svg_geometry"):
        if field not in forbidden_schema_fields:
            errors.append(f"candidate_set_schema.forbidden_runtime_fields missing {field}")

    experts = contract.get("experts", [])
    if not isinstance(experts, list) or not experts:
        errors.append("experts must be a non-empty list")
        experts = []

    assigned_asset_ids: set[str] = set()
    expert_ids: set[str] = set()
    offline_asset_ids: list[str] = []
    for expert in experts:
        expert_id = str(expert.get("id", "<missing>"))
        expert_ids.add(expert_id)
        missing_expert = sorted(REQUIRED_EXPERT_KEYS - set(expert))
        for key in missing_expert:
            errors.append(f"{expert_id}: missing expert key {key}")

        runtime_fields = [str(item) for item in expert.get("runtime_fields", [])]
        expert_forbidden_fields = set(str(item) for item in expert.get("runtime_forbidden_fields", []))
        overlaps = sorted(set(runtime_fields) & (forbidden_schema_fields | expert_forbidden_fields))
        if overlaps:
            errors.append(f"{expert_id}: runtime_fields include forbidden fields {overlaps}")

        runtime_field_token_hits = [
            {"field": field, "token": token}
            for field in runtime_fields
            if (token := contains_forbidden_runtime_token(field))
        ]
        if runtime_field_token_hits:
            errors.append(f"{expert_id}: runtime_fields contain forbidden tokens {runtime_field_token_hits}")

        for input_name in [str(item) for item in expert.get("inputs", [])]:
            token = contains_forbidden_runtime_token(input_name)
            if token and not (expert_id == "page_coverage_auditor_router" and "offline eval cache" in input_name.lower()):
                errors.append(f"{expert_id}: input {input_name!r} contains forbidden runtime token {token!r}")

        for asset in expert.get("assets", []):
            asset_id = str(asset.get("id", ""))
            assigned_asset_ids.add(asset_id)
            asset_path = str(asset.get("path", ""))
            required_now = bool(asset.get("required_now", False))
            offline_only = bool(asset.get("offline_only", False))
            if offline_only:
                offline_asset_ids.append(asset_id)
            if not asset_id or not asset_path:
                errors.append(f"{expert_id}: asset must include id and path")
                continue
            resolved = source_path(asset_path)
            if resolved.exists():
                if required_now:
                    present_required_assets.append({"expert": expert_id, "asset": asset_id, "path": rel(resolved)})
            elif required_now:
                errors.append(f"{expert_id}: required asset missing: {asset_id} -> {asset_path}")
            else:
                missing_optional_assets.append({"expert": expert_id, "asset": asset_id, "path": asset_path})

        if not expert.get("primary_metric"):
            errors.append(f"{expert_id}: primary_metric is empty")
        if not expert.get("stage_gates"):
            errors.append(f"{expert_id}: stage_gates must not be empty")
        if not expert.get("must_not_optimize"):
            warnings.append(f"{expert_id}: must_not_optimize is empty")

        expert_summaries.append(
            {
                "id": expert_id,
                "status": expert.get("status"),
                "primary_metric": expert.get("primary_metric"),
                "asset_count": len(expert.get("assets", [])),
                "runtime_field_count": len(runtime_fields),
            }
        )

    expected_experts = {
        "high_recall_proposal_expert",
        "box_quality_refiner_expert",
        "duplicate_support_suppression_expert",
        "symbol_type_expert",
        "page_coverage_auditor_router",
    }
    missing_experts = sorted(expected_experts - expert_ids)
    for expert_id in missing_experts:
        errors.append(f"missing required expert: {expert_id}")

    required_assignment = set(contract.get("current_artifact_assignment_required", []))
    missing_assignment = sorted(required_assignment - assigned_asset_ids)
    for asset_id in missing_assignment:
        errors.append(f"required current artifact not assigned to any expert: {asset_id}")

    if not offline_asset_ids:
        warnings.append("no offline_only assets declared; eval caches should be marked offline_only")

    return {
        "passed": not errors,
        "contract_version": contract.get("version"),
        "summary": {
            "expert_count": len(experts),
            "assigned_asset_count": len(assigned_asset_ids),
            "required_asset_present_count": len(present_required_assets),
            "missing_optional_asset_count": len(missing_optional_assets),
            "offline_only_asset_count": len(offline_asset_ids),
            "error_count": len(errors),
            "warning_count": len(warnings),
        },
        "expert_summaries": expert_summaries,
        "present_required_assets": present_required_assets,
        "missing_optional_assets": missing_optional_assets,
        "errors": errors,
        "warnings": warnings,
        "source_integrity": {
            "runtime_allowed_inputs": sorted(runtime_allowed),
            "runtime_forbidden_inputs": sorted(runtime_forbidden),
            "offline_assets_declared": sorted(offline_asset_ids),
            "schema_forbidden_runtime_fields": sorted(forbidden_schema_fields),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    contract_path = args.contract if args.contract.is_absolute() else ROOT / args.contract
    output_path = args.output if args.output.is_absolute() else ROOT / args.output
    contract = load_json(contract_path)
    audit = audit_contract(contract)
    audit["contract_path"] = rel(contract_path)
    audit["output_path"] = rel(output_path)
    write_json(output_path, audit)
    if not audit["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
