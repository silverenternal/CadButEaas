"""Run architecture-defined CadStruct pipelines and emit artifact manifests.

This runner intentionally stays small and repo-local. struct.json owns the
pipeline command list; this script makes that contract executable and collects a
single manifest so normal workflows do not require hand-chaining commands.
"""

from __future__ import annotations

import argparse
import datetime as dt
import glob
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = REPO_ROOT / "reports" / "vlm"
RUNNER_AUDIT = REPORT_DIR / "struct_pipeline_runner_audit.json"
VALIDATOR_AUDIT = REPORT_DIR / "struct_validator_registry_audit.json"
TAIL_CHARS = 4000
JSONL_SAMPLE_LINES = 20
MAX_GIT_STATUS_ENTRIES = 200
INTEGRITY_KEYS = {
    "source_integrity",
    "source_integrity_violations",
    "source_mode",
    "svg_candidate_ids_used",
    "annotation_geometry_used_at_inference",
    "model_input",
}


def utc_now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def tail(value: str) -> str:
    if len(value) <= TAIL_CHARS:
        return value
    return value[-TAIL_CHARS:]


def rel(path: Path) -> str:
    try:
        return path.resolve().relative_to(REPO_ROOT).as_posix()
    except ValueError:
        return path.as_posix()


def path_info(path_value: str, role: str) -> dict[str, Any]:
    path = (REPO_ROOT / path_value).resolve()
    info: dict[str, Any] = {
        "path": rel(path),
        "role": role,
        "exists": path.exists(),
    }
    if not path.exists():
        return info
    stat = path.stat()
    info["size_bytes"] = stat.st_size
    info["mtime"] = dt.datetime.fromtimestamp(stat.st_mtime, dt.timezone.utc).isoformat(timespec="seconds")
    if path.is_file() and stat.st_size <= 16 * 1024 * 1024:
        h = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                h.update(chunk)
        info["sha256"] = h.hexdigest()
    return info


def classify_artifacts(entry: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    inputs: list[dict[str, Any]] = []
    checkpoints: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    eval_reports: list[dict[str, Any]] = []
    audit_reports: list[dict[str, Any]] = []
    review_packs: list[dict[str, Any]] = []

    for path in entry.get("primary_inputs", []):
        info = path_info(path, "primary_input")
        if path.startswith("checkpoints/"):
            info["role"] = "checkpoint_input"
            checkpoints.append(info)
        else:
            inputs.append(info)

    for path in entry.get("primary_outputs", []):
        info = path_info(path, "primary_output")
        lower = path.lower()
        if path.startswith("checkpoints/"):
            info["role"] = "checkpoint_output"
            checkpoints.append(info)
        elif "review_pack" in lower or "visual_demo" in lower:
            info["role"] = "review_pack"
            review_packs.append(info)
        elif "audit" in lower:
            info["role"] = "audit_report"
            audit_reports.append(info)
        elif "eval" in lower:
            info["role"] = "eval_report"
            eval_reports.append(info)
        else:
            outputs.append(info)

    return {
        "inputs": inputs,
        "checkpoints": checkpoints,
        "outputs": outputs,
        "eval_reports": eval_reports,
        "audit_reports": audit_reports,
        "review_packs": review_packs,
    }


def run_shell(command: str, *, dry_run: bool = False) -> dict[str, Any]:
    started = utc_now()
    start = time.time()
    if dry_run:
        return {
            "command": command,
            "started_at": started,
            "finished_at": utc_now(),
            "duration_seconds": 0.0,
            "returncode": 0,
            "stdout_tail": "",
            "stderr_tail": "",
            "dry_run": True,
        }
    proc = subprocess.run(
        command,
        cwd=REPO_ROOT,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    return {
        "command": command,
        "started_at": started,
        "finished_at": utc_now(),
        "duration_seconds": round(time.time() - start, 3),
        "returncode": proc.returncode,
        "stdout_tail": tail(proc.stdout or ""),
        "stderr_tail": tail(proc.stderr or ""),
        "dry_run": False,
    }


def git_status_summary() -> dict[str, Any]:
    proc = subprocess.run(
        "git status --short",
        cwd=REPO_ROOT,
        shell=True,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    entries = [line for line in (proc.stdout or "").splitlines() if line.strip()]
    return {
        "returncode": proc.returncode,
        "dirty": bool(entries),
        "entry_count": len(entries),
        "entries": entries[:MAX_GIT_STATUS_ENTRIES],
        "truncated": len(entries) > MAX_GIT_STATUS_ENTRIES,
        "stderr_tail": tail(proc.stderr or ""),
    }


def iter_json_records(path: Path, max_lines: int = JSONL_SAMPLE_LINES) -> list[Any]:
    records: list[Any] = []
    if not path.exists() or not path.is_file():
        return records
    if path.suffix == ".json":
        try:
            return [load_json(path)]
        except (OSError, json.JSONDecodeError):
            return records
    if path.suffix == ".jsonl":
        try:
            with path.open("r", encoding="utf-8") as handle:
                for idx, line in enumerate(handle):
                    if idx >= max_lines:
                        break
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        records.append(json.loads(line))
                    except json.JSONDecodeError:
                        records.append({"__json_decode_error__": idx + 1})
        except OSError:
            return records
    return records


def walk_values(value: Any) -> list[Any]:
    values = [value]
    if isinstance(value, dict):
        for child in value.values():
            values.extend(walk_values(child))
    elif isinstance(value, list):
        for child in value:
            values.extend(walk_values(child))
    return values


def collect_integrity_from_value(value: Any) -> dict[str, Any]:
    summary = {
        "source_integrity_blocks": 0,
        "violations": [],
        "models_seen": set(),
        "source_modes_seen": set(),
    }
    for item in walk_values(value):
        if not isinstance(item, dict):
            continue
        if "source_integrity" in item and isinstance(item["source_integrity"], dict):
            summary["source_integrity_blocks"] += 1
            item = item["source_integrity"]
        if "model_input" in item:
            summary["models_seen"].add(str(item.get("model_input")))
            if item.get("model_input") != "raster_image_only":
                summary["violations"].append({"field": "model_input", "value": item.get("model_input")})
        if "source_mode" in item:
            summary["source_modes_seen"].add(str(item.get("source_mode")))
        if item.get("svg_candidate_ids_used") is True:
            summary["violations"].append({"field": "svg_candidate_ids_used", "value": True})
        if item.get("annotation_geometry_used_at_inference") is True:
            summary["violations"].append({"field": "annotation_geometry_used_at_inference", "value": True})
        if isinstance(item.get("source_integrity_violations"), (int, float)) and item["source_integrity_violations"] > 0:
            summary["violations"].append(
                {"field": "source_integrity_violations", "value": item["source_integrity_violations"]}
            )
    summary["models_seen"] = sorted(summary["models_seen"])
    summary["source_modes_seen"] = sorted(summary["source_modes_seen"])
    return summary


def summarize_source_integrity(paths: list[str]) -> dict[str, Any]:
    checked: list[dict[str, Any]] = []
    total_violations: list[dict[str, Any]] = []
    blocks = 0
    for path_value in paths:
        path = REPO_ROOT / path_value
        file_summary = {
            "path": path_value,
            "exists": path.exists(),
            "sampled_records": 0,
            "source_integrity_blocks": 0,
            "violations": [],
            "models_seen": [],
            "source_modes_seen": [],
        }
        models_seen: set[str] = set()
        source_modes_seen: set[str] = set()
        for record in iter_json_records(path):
            file_summary["sampled_records"] += 1
            partial = collect_integrity_from_value(record)
            file_summary["source_integrity_blocks"] += partial["source_integrity_blocks"]
            blocks += partial["source_integrity_blocks"]
            models_seen.update(partial["models_seen"])
            source_modes_seen.update(partial["source_modes_seen"])
            for violation in partial["violations"]:
                with_path = dict(violation)
                with_path["path"] = path_value
                file_summary["violations"].append(with_path)
                total_violations.append(with_path)
        file_summary["models_seen"] = sorted(models_seen)
        file_summary["source_modes_seen"] = sorted(source_modes_seen)
        checked.append(file_summary)
    return {
        "checked_files": checked,
        "checked_file_count": len(checked),
        "source_integrity_blocks_sampled": blocks,
        "violation_count": len(total_violations),
        "violations": total_violations[:50],
        "passed": not total_violations,
    }


def compact_json_metrics(path: Path) -> dict[str, Any]:
    if not path.exists() or path.suffix != ".json":
        return {}
    try:
        data = load_json(path)
    except (OSError, json.JSONDecodeError):
        return {"json_read_error": True}
    keep: dict[str, Any] = {}
    interesting = (
        "decision",
        "gate_decision",
        "passed",
        "selected_relations",
        "candidate_reduction",
        "recall_drop_abs",
        "source_integrity_violations",
        "component_id_missing_edges",
        "total_edges",
        "positive_edges",
        "rows",
        "dataset_rows",
    )
    for key in interesting:
        if key in data and isinstance(data[key], (str, int, float, bool, type(None))):
            keep[key] = data[key]
    for key, value in data.items():
        if not isinstance(value, dict):
            continue
        nested = {
            k: v
            for k, v in value.items()
            if isinstance(v, (str, int, float, bool, type(None)))
            and any(token in k for token in ("precision", "recall", "f1", "reduction", "drop", "selected"))
        }
        if nested:
            keep[key] = nested
    return keep


def metrics_summary(eval_reports: list[dict[str, Any]], audit_reports: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for info in eval_reports + audit_reports:
        if not info.get("exists"):
            continue
        path_value = info["path"]
        metrics = compact_json_metrics(REPO_ROOT / path_value)
        if metrics:
            out[path_value] = metrics
    return out


def artifact_paths(groups: dict[str, list[dict[str, Any]]]) -> list[str]:
    paths: list[str] = []
    for items in groups.values():
        for item in items:
            paths.append(str(item["path"]))
    return paths


def validate_detector_candidate(path_value: str) -> dict[str, Any]:
    path = REPO_ROOT / path_value
    required = {"candidate_id", "row_id", "family", "bbox", "confidence", "source_integrity"}
    sampled = 0
    missing = 0
    invalid_bbox = 0
    missing_provenance = 0
    missing_audit = 0
    warnings: list[str] = []
    for record in iter_json_records(path):
        rows = []
        if isinstance(record, dict):
            if "candidate_id" in record:
                rows = [record]
            elif isinstance(record.get("candidates"), list):
                rows = record["candidates"][:JSONL_SAMPLE_LINES]
            elif isinstance((record.get("scene_graph") or {}).get("candidate_stream"), list):
                rows = (record.get("scene_graph") or {})["candidate_stream"][:JSONL_SAMPLE_LINES]
            elif isinstance(record.get("families"), dict):
                for value in record["families"].values():
                    if isinstance(value, list):
                        rows.extend(value[:JSONL_SAMPLE_LINES])
        for row in rows[:JSONL_SAMPLE_LINES]:
            sampled += 1
            if not isinstance(row, dict) or required - set(row):
                missing += 1
                continue
            bbox = row.get("bbox")
            if (
                not isinstance(bbox, list)
                or len(bbox) != 4
                or not all(isinstance(v, (int, float)) for v in bbox)
                or bbox[2] <= bbox[0]
                or bbox[3] <= bbox[1]
            ):
                invalid_bbox += 1
            if not row.get("provenance"):
                missing_provenance += 1
            if not row.get("audit_trace"):
                missing_audit += 1
    failures = []
    if sampled and missing:
        failures.append(f"{missing} sampled detector candidates miss required fields")
    if invalid_bbox:
        failures.append(f"{invalid_bbox} sampled detector candidates have invalid bbox")
    if sampled and missing_provenance:
        warnings.append(f"{missing_provenance} sampled detector candidates miss provenance")
    if sampled and missing_audit:
        warnings.append(f"{missing_audit} sampled detector candidates miss audit_trace")
    status = "not_applicable" if sampled == 0 else ("fail" if failures else "pass")
    return {
        "validator": "DetectorCandidate",
        "path": path_value,
        "status": status,
        "sampled": sampled,
        "failures": failures,
        "warnings": warnings,
    }


def validate_relation_edge_candidate(path_value: str) -> dict[str, Any]:
    path = REPO_ROOT / path_value
    required = {"relation_id", "relation", "source_candidate_id", "target_candidate_id", "component_id"}
    sampled = 0
    failures: list[str] = []
    relations = {"bounded_by", "contains_symbol", "labeled_by_text", "adjacent_to"}
    for row in iter_json_records(path):
        if not isinstance(row, dict):
            failures.append("sampled row is not an object")
            continue
        sampled += 1
        missing = required - set(row)
        if missing:
            failures.append(f"missing fields: {sorted(missing)}")
        if row.get("relation") not in relations:
            failures.append(f"unknown relation: {row.get('relation')}")
    status = "not_applicable" if sampled == 0 else ("fail" if failures else "pass")
    return {
        "validator": "RelationEdgeCandidate",
        "path": path_value,
        "status": status,
        "sampled": sampled,
        "failures": failures[:20],
    }


def validate_selected_relation_edge(path_value: str) -> dict[str, Any]:
    path = REPO_ROOT / path_value
    sampled_pages = 0
    sampled_relations = 0
    failures: list[str] = []
    warnings: list[str] = []
    for record in iter_json_records(path):
        if not isinstance(record, dict):
            failures.append("sampled page row is not an object")
            continue
        sampled_pages += 1
        relations = ((record.get("scene_graph") or {}).get("relations") or [])[:JSONL_SAMPLE_LINES]
        for relation in relations:
            if not isinstance(relation, dict):
                failures.append("sampled relation is not an object")
                continue
            sampled_relations += 1
            for key in ("relation_id", "relation", "source_candidate_id", "target_candidate_id"):
                if not relation.get(key):
                    failures.append(f"missing {key}")
            evidence = relation.get("evidence") or {}
            if not (
                relation.get("selection_trace")
                or evidence.get("listwise_policy_trace")
                or evidence.get("relation_graph_policy")
            ):
                warnings.append("selection trace/policy marker missing")
    status = "not_applicable" if sampled_relations == 0 else ("fail" if failures else "pass")
    return {
        "validator": "SelectedRelationEdge",
        "path": path_value,
        "status": status,
        "sampled_pages": sampled_pages,
        "sampled_relations": sampled_relations,
        "failures": failures[:20],
        "warnings": warnings[:20],
    }


def run_validators(struct: dict[str, Any], manifest_groups: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    registry = struct.get("validator_registry") or {}
    paths = artifact_paths(manifest_groups)
    results: list[dict[str, Any]] = []

    detector_paths = [
        p for p in paths if ("candidate" in p or "detector_adapter" in p) and p.endswith(".jsonl")
    ]
    relation_dataset_paths = [
        p for p in paths if "relation_graph_reconstruction" in p and "dataset" in p and p.endswith(".jsonl")
    ]
    selected_relation_paths = [
        p for p in paths if "relation_graph_listwise" in p and "candidate" in p and p.endswith(".jsonl")
    ]

    if "DetectorCandidate" in registry:
        results.extend(validate_detector_candidate(p) for p in detector_paths[:3])
        if not detector_paths:
            results.append({"validator": "DetectorCandidate", "status": "not_applicable", "reason": "no candidate artifact"})
    if "RelationEdgeCandidate" in registry:
        results.extend(validate_relation_edge_candidate(p) for p in relation_dataset_paths[:3])
        if not relation_dataset_paths:
            results.append({"validator": "RelationEdgeCandidate", "status": "not_applicable", "reason": "no relation dataset artifact"})
    if "SelectedRelationEdge" in registry:
        results.extend(validate_selected_relation_edge(p) for p in selected_relation_paths[:3])
        if not selected_relation_paths:
            results.append({"validator": "SelectedRelationEdge", "status": "not_applicable", "reason": "no selected relation artifact"})
    if "SourceIntegrity" in registry:
        source_summary = summarize_source_integrity(paths)
        results.append(
            {
                "validator": "SourceIntegrity",
                "status": "pass" if source_summary["passed"] else "fail",
                "checked_file_count": source_summary["checked_file_count"],
                "source_integrity_blocks_sampled": source_summary["source_integrity_blocks_sampled"],
                "violation_count": source_summary["violation_count"],
                "violations": source_summary["violations"],
            }
        )
    for name in ("FeatureLeakage", "SceneGraph"):
        if name in registry:
            results.append({"validator": name, "status": "planned", "reason": registry[name].get("status", "planned")})

    failed = [item for item in results if item.get("status") == "fail"]
    executable = [item for item in results if item.get("status") in {"pass", "fail", "not_applicable"}]
    return {
        "schema_version": "cadstruct_validator_registry_audit_v1",
        "updated": utc_now(),
        "validator_result_count": len(results),
        "executable_validator_result_count": len(executable),
        "failed_validator_count": len(failed),
        "results": results,
    }


def discover_extra_audit_paths(entry: dict[str, Any]) -> list[str]:
    paths: list[str] = []
    for command in entry.get("commands") or [entry.get("command", "")]:
        tokens = command.split()
        for idx, token in enumerate(tokens[:-1]):
            if token in {"--audit", "--audit-output", "--eval-output", "--review-pack", "--features-output"}:
                candidate = tokens[idx + 1]
                if candidate.startswith("reports/"):
                    paths.append(candidate)
    return sorted(set(paths))


def build_manifest(
    *,
    struct: dict[str, Any],
    pipeline_id: str,
    entry: dict[str, Any],
    started_at: str,
    command_results: list[dict[str, Any]],
    dry_run: bool,
) -> dict[str, Any]:
    finished_at = utc_now()
    groups = classify_artifacts(entry)
    extra_paths = discover_extra_audit_paths(entry)
    existing_paths = set(artifact_paths(groups))
    for path in extra_paths:
        if path in existing_paths:
            continue
        info = path_info(path, "discovered_command_artifact")
        lower = path.lower()
        if "review_pack" in lower:
            groups["review_packs"].append(info)
        elif "audit" in lower:
            groups["audit_reports"].append(info)
        elif "eval" in lower:
            groups["eval_reports"].append(info)
        else:
            groups["outputs"].append(info)

    missing = [
        item["path"]
        for group_name in ("inputs", "checkpoints", "outputs", "eval_reports", "audit_reports", "review_packs")
        for item in groups[group_name]
        if not item.get("exists")
    ]
    failed_commands = [item for item in command_results if item.get("returncode") != 0]
    source_summary = summarize_source_integrity(artifact_paths(groups))
    validator_audit = run_validators(struct, groups)
    write_json(VALIDATOR_AUDIT, validator_audit)

    failure_reasons: list[str] = []
    if dry_run:
        gate = "diagnostic_only"
    elif failed_commands:
        gate = "fail"
        failure_reasons.append(f"{len(failed_commands)} command(s) failed")
    elif source_summary["violation_count"] > 0:
        gate = "fail"
        failure_reasons.append(f"{source_summary['violation_count']} source integrity violation(s)")
    elif validator_audit["failed_validator_count"] > 0:
        gate = "fail"
        failure_reasons.append(f"{validator_audit['failed_validator_count']} validator failure(s)")
    elif missing:
        gate = "blocked"
        failure_reasons.append(f"{len(missing)} expected artifact(s) missing")
    else:
        gate = "pass"

    manifest = {
        "schema_version": (struct.get("artifact_manifest_contract") or {}).get(
            "schema_version", "cadstruct_artifact_manifest_v1"
        ),
        "run_id": f"{pipeline_id}_{dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')}",
        "pipeline_id": pipeline_id,
        "started_at": started_at,
        "finished_at": finished_at,
        "command_or_commands": command_results,
        "cwd": str(REPO_ROOT),
        "git_status_summary": git_status_summary(),
        "inputs": groups["inputs"],
        "checkpoints": groups["checkpoints"],
        "outputs": groups["outputs"],
        "eval_reports": groups["eval_reports"],
        "audit_reports": groups["audit_reports"],
        "review_packs": groups["review_packs"],
        "metrics_summary": metrics_summary(groups["eval_reports"], groups["audit_reports"]),
        "gate_decision": gate,
        "failure_reasons": failure_reasons,
        "source_integrity_summary": source_summary,
        "missing_artifacts": missing,
        "validator_audit": rel(VALIDATOR_AUDIT),
    }
    return manifest


def required_manifest_fields(struct: dict[str, Any]) -> list[str]:
    return list((struct.get("artifact_manifest_contract") or {}).get("required_fields") or [])


def validate_manifest_contract(struct: dict[str, Any], manifest: dict[str, Any]) -> list[str]:
    missing = [field for field in required_manifest_fields(struct) if field not in manifest]
    return missing


def run_pipeline(args: argparse.Namespace) -> int:
    if args.dry_run and args.validate_existing:
        raise SystemExit("--dry-run and --validate-existing are mutually exclusive")
    struct_path = (REPO_ROOT / args.struct).resolve()
    struct = load_json(struct_path)
    entry = (struct.get("pipeline_entrypoints") or {}).get(args.pipeline)
    if not entry:
        available = ", ".join(sorted((struct.get("pipeline_entrypoints") or {}).keys()))
        raise SystemExit(f"unknown pipeline {args.pipeline!r}; available: {available}")

    manifest_path = REPO_ROOT / entry.get("artifact_manifest", f"reports/vlm/{args.pipeline}_manifest.json")
    commands = entry.get("commands") or [entry.get("command")]
    commands = [cmd for cmd in commands if cmd and cmd != "TBD after candidate frontend, symbol type, OCR/text graph, and graph reconstruction gates are connected"]
    if not commands:
        raise SystemExit(f"pipeline {args.pipeline!r} has no executable command in struct.json")

    started_at = utc_now()
    command_results: list[dict[str, Any]] = []
    exit_code = 0
    if args.validate_existing:
        for command in commands:
            command_results.append(
                {
                    "command": command,
                    "started_at": started_at,
                    "finished_at": utc_now(),
                    "duration_seconds": 0.0,
                    "returncode": 0,
                    "stdout_tail": "",
                    "stderr_tail": "",
                    "dry_run": False,
                    "validate_existing": True,
                }
            )
    else:
        for command in commands:
            result = run_shell(command, dry_run=args.dry_run)
            command_results.append(result)
            if result["returncode"] != 0:
                exit_code = int(result["returncode"])
                if not args.continue_on_error:
                    break

    manifest = build_manifest(
        struct=struct,
        pipeline_id=args.pipeline,
        entry=entry,
        started_at=started_at,
        command_results=command_results,
        dry_run=args.dry_run,
    )
    contract_missing = validate_manifest_contract(struct, manifest)
    if contract_missing:
        manifest["gate_decision"] = "fail"
        manifest["failure_reasons"].append(f"manifest contract missing fields: {contract_missing}")
        exit_code = exit_code or 2
    write_json(manifest_path, manifest)

    runner_audit = {
        "schema_version": "cadstruct_pipeline_runner_audit_v1",
        "updated": utc_now(),
        "pipeline_id": args.pipeline,
        "struct_path": rel(struct_path),
        "manifest_path": rel(manifest_path),
        "dry_run": args.dry_run,
        "validate_existing": args.validate_existing,
        "commands_total": len(commands),
        "commands_executed": len(command_results),
        "failed_command_count": sum(1 for item in command_results if item["returncode"] != 0),
        "gate_decision": manifest["gate_decision"],
        "failure_reasons": manifest["failure_reasons"],
        "manifest_required_fields": required_manifest_fields(struct),
        "manifest_missing_required_fields": contract_missing,
    }
    write_json(RUNNER_AUDIT, runner_audit)

    print(json.dumps({"manifest": rel(manifest_path), "gate_decision": manifest["gate_decision"]}, ensure_ascii=False))
    if manifest["gate_decision"] not in {"pass", "diagnostic_only"}:
        return exit_code or 1
    return exit_code


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", required=True, help="pipeline id from struct.json:pipeline_entrypoints")
    parser.add_argument("--struct", default="struct.json", help="architecture structure JSON path")
    parser.add_argument("--dry-run", action="store_true", help="write manifest without executing commands")
    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help="validate already-produced artifacts and refresh the manifest without rerunning commands",
    )
    parser.add_argument("--continue-on-error", action="store_true", help="execute remaining commands after a failure")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    return run_pipeline(args)


if __name__ == "__main__":
    raise SystemExit(main())
