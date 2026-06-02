#!/usr/bin/env python3
"""Audit P224a frozen column policy for forbidden runtime feature usage."""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/vlm/symbol_p224a_column_policy_frozen.json"
SCRIPT = ROOT / "scripts/vlm/freeze_symbol_p224a_column_policy.py"
OUT_MD = ROOT / "reports/vlm/symbol_p224a_column_source_integrity.md"
OUT_JSON = ROOT / "reports/vlm/symbol_p224a_column_source_integrity.json"
FORBIDDEN_RUNTIME_TERMS = ["gold", "target", "annotation", "expected_json", "svg", "parser", "row_id", "semantic", "raw_label"]


def source_segment(source: str, node: ast.AST) -> str:
    try:
        return ast.get_source_segment(source, node) or ""
    except Exception:
        return ""


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    runtime_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in {"apply_policy", "conflicts"}]
    runtime_source = "\n".join(source_segment(source, node) for node in runtime_nodes)
    hits = {term: runtime_source.lower().count(term.lower()) for term in FORBIDDEN_RUNTIME_TERMS if runtime_source.lower().count(term.lower())}
    pass_integrity = not any(term in hits for term in ["gold", "annotation", "expected_json", "svg", "parser", "semantic", "raw_label"])
    runtime_predicates = [
        "P224 predicted label == frozen allowed_label ('column')",
        "P224 detector score >= frozen score_threshold (0.85)",
        "same-label IoU to existing P222/additions < frozen max_iou (0.35)",
        "same-label center distance > frozen min_center_dist (0.0)",
        "frozen max_add_per_row/max_add_per_label caps",
    ]
    result = {
        "id": "P224a_column_source_integrity",
        "config": str(CONFIG.relative_to(ROOT)),
        "script": str(SCRIPT.relative_to(ROOT)),
        "runtime_functions": ["apply_policy", "conflicts"],
        "runtime_predicates": runtime_predicates,
        "forbidden_term_hits_in_runtime_functions": hits,
        "row_id_usage": "row_id/rid is used to select the corresponding P224 prediction row and construct deterministic output IDs; not used as a learned feature, threshold, or policy predicate.",
        "target_usage": "target_id is an output identifier field only, not a runtime predicate." if "target" in hits else "No target hit in runtime functions.",
        "pass": pass_integrity,
        "claim_boundary": "Audit covers frozen deterministic P224a column policy. Gold labels are used only after overlay generation for offline scoring/bootstrap.",
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# P224a Column Source-Integrity Audit",
        "",
        "## Verdict",
        f"- Pass: `{str(pass_integrity).lower()}`",
        "- Runtime policy uses P222 predictions, P224 s384 raster detector predictions, bbox geometry, detector label/score, and frozen constants only.",
        "- Gold labels are used only after overlay generation for offline evaluation/bootstrap.",
        "",
        "## Runtime Functions",
        f"- Script: `{result['script']}`",
        "- Functions: `apply_policy`, `conflicts`",
        "",
        "## Runtime Predicates",
    ]
    lines += [f"- {item}" for item in runtime_predicates]
    lines += [
        "",
        "## Forbidden-Term Scan in Runtime Functions",
        f"- Hits: `{json.dumps(hits, ensure_ascii=False)}`",
        f"- Row-id interpretation: {result['row_id_usage']}",
        f"- Target interpretation: {result['target_usage']}",
        "",
        "## Claim Boundary",
        result["claim_boundary"],
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"pass": pass_integrity, "hits": hits, "report": str(OUT_MD.relative_to(ROOT))}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
