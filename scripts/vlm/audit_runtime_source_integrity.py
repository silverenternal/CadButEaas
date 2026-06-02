#!/usr/bin/env python3
"""Audit runtime artifacts for forbidden SVG/gold/expected_json source leakage."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

DEFAULT_FORBIDDEN = [
    "expected_json",
    "model.svg",
    "svg",
    "parser_geometry",
    "parser",
    "annotation_path",
    "annotation",
    "gold",
    "raw_label",
    "semantic_type",
    "offline_id",
    "source_row_ref",
]
DEFAULT_ALLOWED_CONTEXT = [
    "source_integrity",
    "claim_boundary",
    "forbidden_runtime_fields",
    "runtime_forbidden",
    "offline_allowed",
    "audit",
]


def read_jsonish(path: Path, max_lines: int | None = None) -> list[Any]:
    suffix = path.suffix.lower()
    if suffix == ".jsonl":
        rows = []
        with path.open(encoding="utf-8") as handle:
            for index, line in enumerate(handle):
                if max_lines is not None and index >= max_lines:
                    break
                if line.strip():
                    rows.append(json.loads(line))
        return rows
    return [json.loads(path.read_text(encoding="utf-8"))]


def iter_paths(value: Any, prefix: str = "$") -> Iterable[tuple[str, Any]]:
    yield prefix, value
    if isinstance(value, dict):
        for key, child in value.items():
            yield from iter_paths(child, f"{prefix}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from iter_paths(child, f"{prefix}[{index}]")


def context_allowed(path: str, allowed_context: list[str]) -> bool:
    lowered = path.lower()
    return any(token.lower() in lowered for token in allowed_context)


def find_hits(values: list[Any], forbidden: list[str], allowed_context: list[str], scan_values: bool) -> list[dict[str, Any]]:
    hits = []
    forbidden_lower = [(term, term.lower()) for term in forbidden]
    for doc_index, value in enumerate(values):
        for path, node in iter_paths(value):
            if context_allowed(path, allowed_context):
                continue
            key_part = path.split(".")[-1].split("[")[0].lower()
            for original, term in forbidden_lower:
                if term and term in key_part:
                    hits.append({"doc_index": doc_index, "path": path, "term": original, "where": "key"})
            if scan_values and isinstance(node, str):
                lowered = node.lower()
                for original, term in forbidden_lower:
                    if term and term in lowered and not context_allowed(path, allowed_context):
                        hits.append({"doc_index": doc_index, "path": path, "term": original, "where": "value", "sample": node[:160]})
    return hits


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", help="JSON/JSONL files to audit")
    parser.add_argument("--out", default="", help="Optional JSON report path")
    parser.add_argument("--forbidden", default=",".join(DEFAULT_FORBIDDEN))
    parser.add_argument("--allowed-context", default=",".join(DEFAULT_ALLOWED_CONTEXT))
    parser.add_argument("--max-lines", type=int, default=None)
    parser.add_argument("--scan-values", action="store_true", help="Also scan string values, not only keys")
    parser.add_argument("--allow-row-id-key", action="store_true", help="Do not flag row_id keys; useful for per-row prediction JSONL where row_id is identifier not feature")
    args = parser.parse_args()

    forbidden = [item.strip() for item in args.forbidden.split(",") if item.strip()]
    if args.allow_row_id_key:
        forbidden = [item for item in forbidden if item != "row_id"]
    allowed_context = [item.strip() for item in args.allowed_context.split(",") if item.strip()]
    reports = []
    all_hits = []
    for raw in args.inputs:
        path = Path(raw)
        values = read_jsonish(path, args.max_lines)
        hits = find_hits(values, forbidden, allowed_context, args.scan_values)
        counts = Counter(hit["term"] for hit in hits)
        item = {
            "input": str(path),
            "documents_scanned": len(values),
            "pass_integrity": not hits,
            "hit_count": len(hits),
            "hit_terms": dict(counts),
            "hits": hits[:200],
        }
        reports.append(item)
        all_hits.extend(hits)
    report = {
        "id": "runtime_source_integrity_audit",
        "forbidden_terms": forbidden,
        "allowed_context_terms": allowed_context,
        "scan_values": bool(args.scan_values),
        "pass_integrity": not all_hits,
        "inputs": reports,
        "claim_boundary": "Static JSON/JSONL source-leakage scan. Passing this audit is necessary but not sufficient for runtime safety.",
    }
    text = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
    print(text)
    if all_hits:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
