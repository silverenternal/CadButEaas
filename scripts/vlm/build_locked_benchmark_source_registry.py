#!/usr/bin/env python3
"""Build a source/license/coverage registry for the locked benchmark."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


SOURCE_INFO = {
    "cvc_fp": {
        "name": "CVC-FP",
        "source_url": "https://dag.cvc.uab.es/dataset/cvc-fp-database-for-structural-floor-plan-analysis/",
        "license": "CC BY-NC 4.0",
        "license_evidence": "CVC-DAG dataset page states CC BY-NC 4.0.",
        "local_license_file": None,
        "permitted_claim_scope": "research/non-commercial structural floor-plan analysis",
    },
    "floorplancad": {
        "name": "FloorPlanCAD",
        "source_url": "https://floorplancad.github.io/",
        "license": "CC BY-NC 4.0",
        "license_evidence": "Project site and local README describe Creative Commons Attribution-NonCommercial 4.0.",
        "local_license_file": "datasets/external/floorplancad/README.md",
        "permitted_claim_scope": "research/non-commercial CAD wall/opening stress source",
    },
    "cubicasa5k": {
        "name": "CubiCasa5K",
        "source_url": "https://github.com/CubiCasa/CubiCasa5k",
        "license": "Apache-2.0",
        "license_evidence": "Public CubiCasa5K repository includes LICENSE; OpenConstruction lists Apache 2.0.",
        "local_license_file": None,
        "permitted_claim_scope": "floor-plan room/symbol/text benchmark; verify downstream redistribution requirements before release",
    },
    "internal_hard_cases_round_1": {
        "name": "Internal hard cases round 1",
        "source_url": "local/manual_review_seed",
        "license": "internal/review-only",
        "license_evidence": "Local annotation seed, not a public full source.",
        "local_license_file": None,
        "permitted_claim_scope": "hard-case analysis only; not sufficient for broad real-world claims",
    },
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", default="datasets/cadstruct_real_world_benchmark_v1/manifest.json")
    parser.add_argument("--benchmark-dir", default="datasets/cadstruct_real_world_benchmark_v1")
    parser.add_argument("--output", default="reports/vlm/locked_benchmark_source_registry_v1.json")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    benchmark_dir = Path(args.benchmark_dir)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    source_coverage: dict[str, Any] = {}
    artifacts = manifest.get("artifacts") or {}
    for source, info in SOURCE_INFO.items():
        source_coverage[source] = {
            **info,
            "locked_files": [],
            "rows": 0,
            "element_counts": Counter(),
            "element_families": Counter(),
            "status": "full_locked_source" if source != "internal_hard_cases_round_1" else "annotation_seed_not_full_source",
        }

    for rel_path, artifact in artifacts.items():
        path = benchmark_dir / rel_path
        if not path.exists():
            continue
        stats_by_source = inspect_jsonl(path, str(artifact.get("element_family") or "unknown"))
        for source, stats in stats_by_source.items():
            if source not in source_coverage:
                source_coverage[source] = {
                    **SOURCE_INFO.get(source, {"name": source, "source_url": "unknown", "license": "unknown", "license_evidence": "missing", "local_license_file": None, "permitted_claim_scope": "unknown"}),
                    "locked_files": [],
                    "rows": 0,
                    "element_counts": Counter(),
                    "element_families": Counter(),
                    "status": "needs_source_review",
                }
            source_coverage[source]["locked_files"].append(rel_path)
            source_coverage[source]["rows"] += stats["rows"]
            source_coverage[source]["element_counts"].update(stats["element_counts"])
            source_coverage[source]["element_families"].update(stats["element_families"])

    internal_path = Path("datasets/internal_hard_cases_round_1/wall_opening_floorplancad_hard_cases.jsonl")
    if internal_path.exists():
        count = sum(1 for line in internal_path.read_text(encoding="utf-8").splitlines() if line.strip())
        source_coverage["internal_hard_cases_round_1"]["locked_files"].append(str(internal_path))
        source_coverage["internal_hard_cases_round_1"]["rows"] += count
        source_coverage["internal_hard_cases_round_1"]["element_counts"].update({"hard_case_records": count})
        source_coverage["internal_hard_cases_round_1"]["element_families"].update({"wall_opening": count})

    normalized_sources = {}
    issues = []
    for source, item in sorted(source_coverage.items()):
        item["locked_files"] = sorted(set(item["locked_files"]))
        item["element_counts"] = dict(item["element_counts"].most_common())
        item["element_families"] = dict(item["element_families"].most_common())
        if not item["locked_files"]:
            issues.append(f"source_without_locked_file:{source}")
        if not item.get("license") or item.get("license") == "unknown":
            issues.append(f"source_missing_license:{source}")
        if item.get("local_license_file") and not Path(str(item["local_license_file"])).exists():
            issues.append(f"local_license_file_missing:{source}:{item['local_license_file']}")
        normalized_sources[source] = item

    report = {
        "version": "locked_benchmark_source_registry_v1",
        "manifest": args.manifest,
        "benchmark_dir": args.benchmark_dir,
        "source_count": len(normalized_sources),
        "sources": normalized_sources,
        "coverage_summary": {
            "sources": sorted(normalized_sources),
            "full_locked_sources": sorted(k for k, v in normalized_sources.items() if v.get("status") == "full_locked_source"),
            "annotation_seed_sources": sorted(k for k, v in normalized_sources.items() if v.get("status") != "full_locked_source"),
        },
        "issues": issues,
        "status": "ok" if len(normalized_sources) >= 4 and not any(item.startswith("source_missing_license") for item in issues) else "needs_review",
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def inspect_jsonl(path: Path, family: str) -> dict[str, Any]:
    by_source: dict[str, dict[str, Any]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source = str(row.get("source_dataset") or "unknown")
            stats = by_source.setdefault(source, {"rows": 0, "element_counts": Counter(), "element_families": Counter()})
            stats["rows"] += 1
            counts = element_counts(row)
            stats["element_counts"].update(counts)
            for name, count in counts.items():
                stats["element_families"][family if name != "rows" else "records"] += count
    return by_source


def element_counts(row: dict[str, Any]) -> Counter[str]:
    counts: Counter[str] = Counter(rows=1)
    for key in ("nodes", "rooms", "room_candidates", "symbols", "symbol_candidates", "texts", "text_candidates"):
        value = row.get(key)
        if isinstance(value, list):
            counts[key] += len(value)
    expected = row.get("expected_json") or {}
    if isinstance(expected, dict):
        for key in ("semantic_candidates", "room_candidates", "symbol_candidates", "text_candidates"):
            value = expected.get(key)
            if isinstance(value, list):
                counts[key] += len(value)
    return counts


if __name__ == "__main__":
    main()
