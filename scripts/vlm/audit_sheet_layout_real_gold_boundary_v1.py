#!/usr/bin/env python3
"""Audit whether SheetLayout has real gold annotations for paper-main claims."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
OUTPUT = REPORTS / "sheet_layout_real_gold_boundary_v1.json"

DATASETS = [
    ROOT / "datasets" / "cadstruct_cubicasa5k_moe_locked_reviewed_v1" / "train.jsonl",
    ROOT / "datasets" / "cadstruct_cubicasa5k_moe_locked_reviewed_v1" / "dev.jsonl",
    ROOT / "datasets" / "cadstruct_real_world_benchmark_v1" / "room_space" / "cubicasa5k_reviewed_locked_test.jsonl",
    ROOT / "datasets" / "cadstruct_text_dimensions_v1" / "train.jsonl",
    ROOT / "datasets" / "cadstruct_text_dimensions_v1" / "dev.jsonl",
    ROOT / "datasets" / "cadstruct_text_dimensions_v1" / "smoke.jsonl",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def inspect_dataset(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    label_counts: Counter[str] = Counter()
    records_with_layout = 0
    for row in rows:
        expected = row.get("expected_json") or row
        regions = expected.get("layout_regions") or []
        if regions:
            records_with_layout += 1
        for region in regions:
            label_counts[str(region.get("layout_type") or region.get("label") or "unknown")] += 1
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": path.exists(),
        "rows": len(rows),
        "records_with_layout_regions": records_with_layout,
        "layout_region_count": sum(label_counts.values()),
        "label_counts": dict(label_counts),
        "has_real_layout_gold": bool(records_with_layout and any(label in label_counts for label in ["title_block", "legend", "schedule", "stamp", "viewport", "north_arrow", "scale_bar"])),
    }


def main() -> int:
    datasets = [inspect_dataset(path) for path in DATASETS]
    eval_report = load_json(REPORTS / "sheet_layout_expert_v1_eval.json")
    data_gap = load_json(REPORTS / "sheet_layout_expert_data_gap_audit.json")
    has_real_gold = any(item["has_real_layout_gold"] for item in datasets)
    status = "tiny_gold_passed" if has_real_gold else "demoted_non_core_extension"
    report = {
        "version": "sheet_layout_real_gold_boundary_v1",
        "created": "2026-05-03",
        "status": status,
        "dataset_scan": datasets,
        "existing_eval": {
            "source": "reports/vlm/sheet_layout_expert_v1_eval.json",
            "available": bool(eval_report),
            "dev_mean_ap50": (((eval_report.get("splits") or {}).get("dev") or {}).get("mean_ap50")),
            "dev_macro_f1": (((eval_report.get("splits") or {}).get("dev") or {}).get("macro_f1")),
            "dev_labels": list(((((eval_report.get("splits") or {}).get("dev") or {}).get("per_label") or {}).keys())),
            "data_audit_layout_regions": (eval_report.get("data_audit") or {}).get("dev", {}).get("layout_regions"),
            "interpretation": "Existing 1.0 SheetLayout metrics are notes-only/synthesized and do not validate title_block, legend, schedule, stamp, viewport, north_arrow, or scale_bar on real gold.",
        },
        "data_gap_report": {
            "source": "reports/vlm/sheet_layout_expert_data_gap_audit.json",
            "available": bool(data_gap),
            "status": data_gap.get("status"),
        },
        "paper_guidance": {
            "core_expert_claim": False,
            "placement": "non-core extension / future work",
            "tables": "Keep SheetLayout as N/A no gold annotations; do not report synthesized AP50 as a main result.",
            "minimum_tiny_gold_if_promoted": {
                "drawings": "20-50 source-held-out sheets",
                "labels": ["title_block", "legend", "schedule", "stamp", "viewport", "north_arrow", "scale_bar", "notes"],
                "metrics": ["AP50", "precision", "recall", "per-label support"],
            },
        },
    }
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(json.dumps({"status": status, "has_real_gold": has_real_gold}, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
