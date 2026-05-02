#!/usr/bin/env python3
"""Audit and optionally enrich the real-world benchmark manifest."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


EVAL_SCRIPT_BY_FAMILY = {
    "wall_opening": "scripts/vlm/evaluate_graph_node_classifier.py",
    "room_space": "scripts/vlm/evaluate_room_space_ambiguity_adjusted.py",
    "symbol_fixture": "scripts/vlm/train_symbol_fixture_crop_mlp.py",
    "text_dimension": "scripts/vlm/train_text_dimension_crop_mlp.py",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="datasets/cadstruct_real_world_benchmark_v1/manifest.json")
    parser.add_argument("--output", default="reports/vlm/benchmark_manifest_audit_v1.json")
    parser.add_argument("--update-manifest", action="store_true")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    base_dir = manifest_path.parent
    artifacts = manifest.setdefault("artifacts", {})
    audit: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "version": "benchmark_manifest_audit_v1",
        "files": {},
        "summary": {
            "artifact_count": 0,
            "missing_hash": 0,
            "missing_source": 0,
            "missing_eval_script": 0,
            "missing_file": 0,
        },
        "status": "ok",
    }

    for rel_path, info in sorted(artifacts.items()):
        path = (base_dir / rel_path).resolve()
        family = rel_path.split("/", 1)[0] if "/" in rel_path else "unknown"
        file_audit = {
            "path": str(path),
            "exists": path.exists(),
            "element_family": info.get("element_family") or family,
            "sources": sorted((info.get("source_counts") or {}).keys()),
            "eval_script": info.get("eval_script") or EVAL_SCRIPT_BY_FAMILY.get(family),
            "rows": info.get("rows") or info.get("records"),
            "sha256": info.get("sha256"),
            "bytes": info.get("bytes"),
            "line_count": info.get("line_count"),
            "issues": [],
        }
        if path.exists():
            file_audit["sha256_actual"] = sha256_file(path)
            file_audit["bytes_actual"] = path.stat().st_size
            file_audit["line_count_actual"] = count_lines(path)
            if args.update_manifest:
                info["sha256"] = file_audit["sha256_actual"]
                info["bytes"] = file_audit["bytes_actual"]
                info["line_count"] = file_audit["line_count_actual"]
                info["element_family"] = file_audit["element_family"]
                info["eval_script"] = file_audit["eval_script"]
        else:
            file_audit["issues"].append("missing_file")
            audit["summary"]["missing_file"] += 1
        if not file_audit.get("sha256") and not args.update_manifest:
            file_audit["issues"].append("missing_hash")
            audit["summary"]["missing_hash"] += 1
        if not file_audit["sources"]:
            file_audit["issues"].append("missing_source")
            audit["summary"]["missing_source"] += 1
        if not file_audit["eval_script"]:
            file_audit["issues"].append("missing_eval_script")
            audit["summary"]["missing_eval_script"] += 1
        audit["files"][rel_path] = file_audit

    audit["summary"]["artifact_count"] = len(audit["files"])
    if any(v for k, v in audit["summary"].items() if k != "artifact_count"):
        audit["status"] = "needs_review"

    if args.update_manifest:
        manifest.setdefault("audit", {})["last_updated_at_utc"] = audit["created_at_utc"]
        manifest["audit"]["last_audit_report"] = args.output
        manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        audit["status"] = "ok" if audit["summary"]["missing_file"] == 0 and audit["summary"]["missing_source"] == 0 and audit["summary"]["missing_eval_script"] == 0 else "needs_review"

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "status": audit["status"], **audit["summary"]}, ensure_ascii=False))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def count_lines(path: Path) -> int:
    with path.open("rb") as handle:
        return sum(1 for _ in handle)


if __name__ == "__main__":
    main()
