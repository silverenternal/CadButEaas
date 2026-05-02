#!/usr/bin/env python3
"""Audit leakage across dataset splits from manifests.

The script is intentionally conservative: it can consume manifests with either:
- {"splits": {"train": {...}, "dev": {...}, "smoke": {...}}}
- {"train": 1000, "dev": 200, "smoke": 32}
and will try to load concrete split JSONL files when they are present.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

KNOWN_SPLITS = ["train", "dev", "locked_test", "smoke", "test"]
CVC_ROTATE_SUFFIX = re.compile(r"_(0|45|90|135|180|225|270|315)(?=\.[^.]+$)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", action="append", default=[], help="Manifest path (repeatable)")
    parser.add_argument(
        "--manifest-dir",
        default="datasets",
        help="Directory to discover manifests when --manifest is not provided.",
    )
    parser.add_argument("--max-manifests", type=int, default=200)
    parser.add_argument("--output", default="reports/vlm/all_split_leakage_audit.json")
    args = parser.parse_args()

    manifest_paths = resolve_manifest_paths(args.manifest, args.manifest_dir, args.max_manifests)
    reports = [audit_manifest(path) for path in manifest_paths]

    by_source = summarize_by_source(reports)
    overlaps = find_leakage_overlaps(by_source)
    status = "ok"
    if any(item["overlap_count"] > 0 for item in overlaps):
        status = "leakage_detected"

    output = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "manifest_count": len(reports),
        "manifests": [item["manifest_path"] for item in reports],
        "manifest_reports": reports,
        "by_source": by_source,
        "overlaps": overlaps,
    }
    output["summary"] = summarize(output)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(output["summary"], ensure_ascii=False, indent=2))


def resolve_manifest_paths(manifest_args: list[str], manifest_dir: str, max_manifests: int) -> list[Path]:
    if manifest_args:
        return [Path(item) for item in manifest_args]

    manifest_paths = []
    for path in sorted(Path(manifest_dir).rglob("manifest.json")):
        manifest_paths.append(path)
        if len(manifest_paths) >= max_manifests:
            break
    return manifest_paths


def audit_manifest(path: Path) -> dict[str, Any]:
    manifest = load_json(path)
    manifest_text = json.dumps(manifest, ensure_ascii=False, sort_keys=True)
    split_specs = collect_split_specs(manifest)

    splits_report: dict[str, dict[str, Any]] = {}
    split_files_used: list[str] = []
    for split_name, spec in split_specs.items():
        rows, status = load_split_rows(path.parent, split_name, spec)
        split_sources: defaultdict[str, int] = defaultdict(int)
        row_keys: set[str] = set()
        for row in rows:
            source = normalize_source(row.get("source_dataset") or row.get("source") or "unknown")
            group_key = make_group_key(source, row)
            split_sources[source] += 1
            row_keys.add(group_key)
        splits_report[split_name] = {
            "manifest_rows_hint": spec.get("manifest_rows_hint"),
            "rows_loaded": len(rows),
            "expected_rows": spec.get("manifest_rows"),
            "source_counts": dict(sorted(split_sources.items())),
            "unique_group_keys": len(row_keys),
            "status": status,
            "group_keys": sorted(row_keys),
        }
        split_files_used.extend(spec.get("source_files", []))

    return {
        "manifest_path": str(path),
        "manifest_dir": str(path.parent),
        "manifest_hash": hashlib.sha256(manifest_text.encode("utf-8")).hexdigest(),
        "split_names": sorted(split_specs),
        "split_reports": splits_report,
        "split_files_used": sorted(set(split_files_used)),
    }

def summarize(report: dict[str, Any]) -> dict[str, Any]:
    return {
        "manifest_count": report["manifest_count"],
        "split_pairs_with_overlap": len([x for x in report["overlaps"] if x["overlap_count"] > 0]),
        "total_overlap_records": sum(item["overlap_count"] for item in report["overlaps"]),
        "by_source_count": len(report["by_source"]),
    }


def collect_split_specs(manifest: dict[str, Any]) -> dict[str, dict[str, Any]]:
    specs: dict[str, dict[str, Any]] = {}
    split_container = manifest.get("splits")
    if isinstance(split_container, dict):
        for split_name, value in split_container.items():
            specs[split_name] = normalize_split_entry(split_name, value)
    else:
        for split_name in KNOWN_SPLITS:
            if split_name in manifest:
                specs[split_name] = normalize_split_entry(split_name, manifest.get(split_name))
    return specs


def normalize_split_entry(split_name: str, value: Any) -> dict[str, Any]:
    spec: dict[str, Any] = {"name": split_name, "manifest_rows": None, "manifest_rows_hint": None, "source_files": []}
    if isinstance(value, int):
        spec["manifest_rows"] = value
        return spec
    if isinstance(value, dict):
        if isinstance(value.get("rows"), int):
            spec["manifest_rows"] = value["rows"]
        if isinstance(value.get("path"), str):
            spec["source_files"].append(str(value["path"]))
        return spec
    return spec


def load_split_rows(dataset_dir: Path, split_name: str, spec: dict[str, Any]) -> tuple[list[dict[str, Any]], str]:
    candidates = []
    seen = set()
    for source in spec.get("source_files", []):
        path = (dataset_dir / source) if source else None
        if path is not None and path.exists():
            candidates.append(path)
    candidates.append(dataset_dir / f"{split_name}.jsonl")

    for path in candidates:
        if path.exists():
            break
    else:
        return [], "manifest_only_no_split_file"

    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            key = make_audit_row_signature(record)
            if key in seen:
                continue
            seen.add(key)
            rows.append(record)
    return rows, f"loaded_from:{path.name}"


def make_audit_row_signature(row: dict[str, Any]) -> str:
    components = [
        str(row.get("annotation_path") or ""),
        str(row.get("image_path") or ""),
        str(row.get("image") or ""),
        str(row.get("annotation") or ""),
        str(row.get("record_id") or ""),
    ]
    signature = "|".join(components)
    if signature.strip("|"):
        return signature
    return _row_signature_fallback(row)


def _row_signature_fallback(row: dict[str, Any]) -> str:
    payload = json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def summarize_by_source(manifest_reports: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    source_to_split_keys: dict[str, dict[str, set[str]]] = defaultdict(lambda: defaultdict(set))
    source_to_datasets: dict[str, set[str]] = defaultdict(set)
    for manifest in manifest_reports:
        split_reports = manifest.get("split_reports") or {}
        for split_name, split_report in split_reports.items():
            for key in split_report.get("group_keys", []):
                # group_key: source|building|page|group|stem
                source = key.split("|", 1)[0]
                source_to_datasets[source].add(manifest["manifest_path"])
                source_to_split_keys[source][split_name].add(key)
            # no-group fallback: count rows under unknown.
            if not split_report.get("group_keys"):
                source_to_split_keys["unknown_no_group"][split_name].add(f"{manifest['manifest_path']}|{split_name}")

    out: dict[str, dict[str, Any]] = {}
    for source, split_keys in sorted(source_to_split_keys.items()):
        split_rows: dict[str, dict[str, Any]] = {}
        by_split: dict[str, dict[str, Any]] = {}
        for split_name in KNOWN_SPLITS:
            values = split_keys.get(split_name, set())
            if not values:
                continue
            by_split[split_name] = {
                "rows": len(values),
                "datasets": sorted(source_to_datasets[source]),
                "group_keys": sorted(values),
            }
            split_rows[split_name] = values
        out[source] = {
            "splits": by_split,
            "total_unique_groups": sum(len(v) for v in split_keys.values()),
            "split_group_keys": {split_name: sorted(keys) for split_name, keys in split_rows.items()},
        }
    return out


def find_leakage_overlaps(by_source: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    # build split-level maps
    for source, source_report in by_source.items():
        for i, split_a in enumerate(KNOWN_SPLITS):
            keys_a = _to_set(source_report.get("split_group_keys", {}).get(split_a))
            if not keys_a:
                continue
            for split_b in KNOWN_SPLITS[i + 1 :]:
                keys_b = _to_set(source_report.get("split_group_keys", {}).get(split_b))
                if not keys_b:
                    continue
                overlap_keys = sorted(keys_a & keys_b)
                if overlap_keys:
                    outputs.append(
                        {
                            "source": source,
                            "pair": f"{split_a}:{split_b}",
                            "overlap_count": len(overlap_keys),
                            "examples": overlap_keys[:20],
                        }
                    )
    return outputs


def _to_set(value: Any) -> set[str]:
    if isinstance(value, set):
        return value
    if isinstance(value, list):
        return set(value)
    return set()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def normalize_source(source: Any) -> str:
    normalized = str(source or "unknown").strip().lower()
    return normalized or "unknown"


def strip_cvc_rotation(path_text: str) -> str:
    stem = Path(path_text).stem
    stripped_stem = CVC_ROTATE_SUFFIX.sub("", stem)
    suffix = Path(path_text).suffix
    if stripped_stem == path_text:
        return path_text
    return str(Path(path_text).with_name(f"{stripped_stem}{suffix}"))


def pick_group_base(row: dict[str, Any]) -> str:
    for key in ["image_path", "image", "annotation_path", "annotation", "image_file", "annotation_file"]:
        value = row.get(key)
        if isinstance(value, str) and value:
            return value
    return ""


def make_group_key(source: str, row: dict[str, Any]) -> str:
    source = normalize_source(source)
    building = _norm(row.get("building_id") or row.get("building") or row.get("metadata", {}).get("building_id"))
    page = _norm(row.get("page_id") or row.get("page") or row.get("metadata", {}).get("page_id"))
    group = _norm(row.get("group_id") or row.get("group") or row.get("metadata", {}).get("group_id"))
    base = pick_group_base(row)
    if base:
        base = base.split("?")[0]
        if source == "cvc_fp":
            base = strip_cvc_rotation(base)
        base = Path(base).with_suffix("").as_posix()
    else:
        base = _norm(row.get("annotation_path") or row.get("image_path"))
    return "|".join([source, building, page, group, base])


def _norm(value: Any) -> str:
    if value is None:
        return "unknown"
    value = str(value).strip().lower()
    return value if value else "unknown"


if __name__ == "__main__":
    main()
