#!/usr/bin/env python3
"""Build P220 split inventory for symbol validation independence."""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ROW_RE = re.compile(r"cubicasa5k_locked_\d{5}")
OUT_MD = ROOT / "reports/vlm/symbol_p220_split_inventory.md"
OUT_JSON = ROOT / "reports/vlm/symbol_p220_split_inventory.json"

PRIOR_SELECTION_ARTIFACTS = [
    "reports/vlm/symbol_p206f_precision_repair_p206g_overlay.jsonl",
    "reports/vlm/symbol_p206g_p212_specialist_precision_repair_overlay.jsonl",
    "reports/vlm/symbol_p213c_precision_gate_overlay.jsonl",
    "reports/vlm/symbol_p214_precision_repair_overlay.jsonl",
    "reports/vlm/symbol_p215_narrow_gate_overlay.jsonl",
    "reports/vlm/symbol_p217_verifier_fusion_overlay.jsonl",
    "reports/vlm/symbol_p218_p217_frozen_overlay.jsonl",
]
REPORT_HINTS = [
    "reports/vlm/symbol_p217_verifier_summary.md",
    "reports/vlm/symbol_p218_p217_frozen_validation.md",
    "reports/vlm/symbol_p219_paper_bounded_ablation.md",
    "reports/vlm/symbol_p220_strategy_painpoints.md",
]
DATASET_SPLIT_FILES = [
    "datasets/symbol_residual_specialist_p213b_yolo/train.txt",
    "datasets/symbol_residual_specialist_p213b_yolo/dev.txt",
    "datasets/symbol_residual_specialist_p213b_yolo/locked.txt",
    "datasets/symbol_fn_specialist_p212_yolo/train.txt",
    "datasets/symbol_fn_specialist_p212_yolo/dev.txt",
    "datasets/symbol_fn_specialist_p212_yolo/locked.txt",
]
CONFIGS = [
    "configs/vlm/raster_moe_target_level_eval_p101.json",
    "configs/vlm/symbol_locked_validation_manifest_p208.json",
    "configs/vlm/symbol_locked_validation_p208.json",
    "configs/vlm/symbol_p217_runtime_verifier_frozen.json",
]


def rows_in_text(path: Path, max_bytes: int = 20_000_000) -> set[str]:
    if not path.exists() or not path.is_file():
        return set()
    try:
        data = path.read_bytes()
    except OSError:
        return set()
    if len(data) > max_bytes:
        data = data[:max_bytes]
    text = data.decode("utf-8", errors="ignore")
    return set(ROW_RE.findall(text))


def scan_path_names(base: Path, limit: int = 500_000) -> tuple[set[str], Counter[str]]:
    rows: set[str] = set()
    by_dir: Counter[str] = Counter()
    if not base.exists():
        return rows, by_dir
    count = 0
    for path in base.rglob("*"):
        count += 1
        if count > limit:
            break
        match = ROW_RE.search(str(path))
        if match:
            row = match.group(0)
            rows.add(row)
            try:
                rel_parent = str(path.parent.relative_to(ROOT))
            except ValueError:
                rel_parent = str(path.parent)
            by_dir[rel_parent] += 1
    return rows, by_dir


def main() -> None:
    prior_rows: dict[str, list[str]] = {}
    selected_union: set[str] = set()
    for rel in PRIOR_SELECTION_ARTIFACTS:
        rows = rows_in_text(ROOT / rel)
        if rows:
            prior_rows[rel] = sorted(rows)
            selected_union |= rows
    report_rows = {rel: sorted(rows_in_text(ROOT / rel)) for rel in REPORT_HINTS if rows_in_text(ROOT / rel)}
    split_rows = {rel: sorted(rows_in_text(ROOT / rel)) for rel in DATASET_SPLIT_FILES if rows_in_text(ROOT / rel)}
    config_rows = {rel: sorted(rows_in_text(ROOT / rel)) for rel in CONFIGS if rows_in_text(ROOT / rel)}

    scanned_sources = {}
    source_rows_all: set[str] = set()
    for rel in ["datasets", "reports/vlm", "configs/vlm"]:
        rows, by_dir = scan_path_names(ROOT / rel)
        scanned_sources[rel] = {"row_count": len(rows), "rows_sample": sorted(rows)[:20], "top_dirs": dict(by_dir.most_common(20))}
        source_rows_all |= rows

    all_known_rows = set(selected_union) | source_rows_all
    for rows in split_rows.values():
        all_known_rows |= set(rows)
    for rows in config_rows.values():
        all_known_rows |= set(rows)

    candidate_heldout = sorted(all_known_rows - selected_union)
    selected_sorted = sorted(selected_union)
    risk = "high" if not candidate_heldout else "medium"
    conclusion = (
        "No row IDs outside prior P206-P219 selection were discovered in path/name inventory; independent validation is unavailable from current mounted artifacts."
        if not candidate_heldout else
        "Some row IDs outside the prior P206-P219 selected overlay were discovered, but label compatibility must be checked before claiming an independent validation split."
    )
    result = {
        "id": "P220_split_inventory",
        "prior_selection_artifacts": {k: {"row_count": len(v), "rows": v[:120]} for k, v in prior_rows.items()},
        "selected_union_count": len(selected_union),
        "selected_union_rows": selected_sorted,
        "dataset_split_files": {k: {"row_count": len(v), "rows_sample": v[:50]} for k, v in split_rows.items()},
        "config_files": {k: {"row_count": len(v), "rows_sample": v[:50]} for k, v in config_rows.items()},
        "path_name_scan": scanned_sources,
        "all_known_row_count": len(all_known_rows),
        "candidate_heldout_count": len(candidate_heldout),
        "candidate_heldout_rows_sample": candidate_heldout[:100],
        "independence_risk": risk,
        "conclusion": conclusion,
        "claim_decision": "Keep P217/P218 claims P101/bootstrap-bounded unless a label-compatible held-out split is explicitly built or provided.",
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# P220 Symbol Split Inventory",
        "",
        "## Purpose",
        "- Check whether P217/P218 can be evaluated on rows not used during P206-P219 policy selection.",
        "- This inventory only establishes row/source availability; it does not tune thresholds or promote metrics.",
        "",
        "## Current Selected Rows",
        f"- Prior P206-P219 selected union rows: {len(selected_union)}",
        f"- Selected rows sample: `{selected_sorted[:20]}`",
        "",
        "## Candidate Held-Out Rows",
        f"- Candidate row IDs outside selected union found in current artifacts: {len(candidate_heldout)}",
        f"- Sample: `{candidate_heldout[:40]}`",
        f"- Independence risk: `{risk}`",
        "",
        "## Dataset Split Files",
        "| File | Row IDs found | Note |",
        "|---|---:|---|",
    ]
    if split_rows:
        for rel, rows in split_rows.items():
            overlap = len(set(rows) & selected_union)
            outside = len(set(rows) - selected_union)
            lines.append(f"| `{rel}` | {len(rows)} | selected overlap={overlap}, outside selected={outside} |")
    else:
        lines.append("| none found | 0 | No listed split files were readable/found. |")
    lines += [
        "",
        "## Path/Name Scan Summary",
        "| Root | Row IDs in path names | Top evidence dirs |",
        "|---|---:|---|",
    ]
    for rel, info in scanned_sources.items():
        top = "; ".join(f"{k}: {v}" for k, v in list(info["top_dirs"].items())[:5])
        lines.append(f"| `{rel}` | {info['row_count']} | {top or 'none'} |")
    lines += [
        "",
        "## Decision",
        f"- {conclusion}",
        "- Claim decision: keep P217/P218 as P101/bootstrap-bounded unless a label-compatible independent split is explicitly identified.",
        "- Next metric work should continue as P221a sink-tiny residual rescue, but any promoted model still needs P222 freeze/source-integrity/bootstrap.",
    ]
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"report": str(OUT_MD), "json": str(OUT_JSON), "selected_union_count": len(selected_union), "candidate_heldout_count": len(candidate_heldout), "risk": risk}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
