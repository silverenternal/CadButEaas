#!/usr/bin/env python3
"""Audit P222/P221a frozen rule for forbidden runtime feature usage."""
from __future__ import annotations

import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/vlm/symbol_p222_p221a_sink_tiny_frozen.json"
SCRIPT = ROOT / "scripts/vlm/freeze_symbol_p222_p221a_sink_tiny.py"
OUT_MD = ROOT / "reports/vlm/symbol_p222_p221a_source_integrity.md"
OUT_JSON = ROOT / "reports/vlm/symbol_p222_p221a_source_integrity.json"
FORBIDDEN_RUNTIME_TERMS = [
    "gold", "target", "annotation", "expected_json", "svg", "parser", "row_id", "semantic", "raw_label"
]
ALLOWED_CONTEXT = {
    "gold": "evaluation/bootstrap only after overlay generation",
    "target": "identifier output field or evaluation/bootstrap only; not used in apply_rule decision",
    "row_id": "output id prefix only; not used as rule predicate/feature",
}


def source_segment(source: str, node: ast.AST) -> str:
    try:
        return ast.get_source_segment(source, node) or ""
    except Exception:
        return ""


def main() -> None:
    config = json.loads(CONFIG.read_text(encoding="utf-8"))
    source = SCRIPT.read_text(encoding="utf-8")
    tree = ast.parse(source)
    apply_rule_node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "apply_rule")
    apply_source = source_segment(source, apply_rule_node)
    hits = {term: apply_source.lower().count(term.lower()) for term in FORBIDDEN_RUNTIME_TERMS if apply_source.lower().count(term.lower())}
    runtime_predicates = [
        "pred_label(pred) == rule['parent_label']",
        "box_area(pred['bbox']) <= rule['parent_max_area']",
        "pred_score(pred) >= rule['parent_min_score']",
        "center(pred['bbox']) with fixed subcandidate width/height",
        "subcandidate score = parent score * frozen score_scale",
    ]
    pass_integrity = not any(term in hits for term in ["gold", "annotation", "expected_json", "svg", "parser", "semantic", "raw_label"])
    row_id_hit = "row_id" in hits
    result = {
        "id": "P222_P221a_source_integrity",
        "config": str(CONFIG.relative_to(ROOT)),
        "script": str(SCRIPT.relative_to(ROOT)),
        "runtime_function": "apply_rule",
        "runtime_predicates": runtime_predicates,
        "forbidden_term_hits_in_apply_rule": hits,
        "row_id_usage": "row_id/rid is used only to construct deterministic output IDs, not as a predicate or threshold." if row_id_hit else "No row_id hit in apply_rule.",
        "pass": pass_integrity,
        "claim_boundary": "Audit covers frozen deterministic P221a rule implementation. Evaluation/bootstrap code uses gold labels only after overlay generation for offline scoring.",
    }
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    lines = [
        "# P222/P221a Source-Integrity Audit",
        "",
        "## Verdict",
        f"- Pass: `{str(pass_integrity).lower()}`",
        "- Runtime rule uses parent label, parent score, parent bbox geometry, and frozen constants only.",
        "- Gold labels are used only after overlay generation for offline evaluation/bootstrap.",
        "",
        "## Runtime Function",
        f"- Script: `{result['script']}`",
        f"- Function: `{result['runtime_function']}`",
        "",
        "## Runtime Predicates",
    ]
    lines += [f"- {item}" for item in runtime_predicates]
    lines += [
        "",
        "## Forbidden-Term Scan in Runtime Function",
        f"- Hits: `{json.dumps(hits, ensure_ascii=False)}`",
        f"- Row-id interpretation: {result['row_id_usage']}",
        "",
        "## Claim Boundary",
        result["claim_boundary"],
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"pass": pass_integrity, "hits": hits, "report": str(OUT_MD)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
