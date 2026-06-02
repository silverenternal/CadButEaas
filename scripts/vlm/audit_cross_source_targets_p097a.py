#!/usr/bin/env python3
"""Audit cross-source raster target availability for SCI baseline planning.

This is target-level only: it does not claim scene-graph relation F1.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUTS = [
    ROOT / "datasets/public_raster_moe_supervision_v19/dev.jsonl",
    ROOT / "datasets/public_raster_moe_supervision_v19/locked.jsonl",
]
DEFAULT_OUTPUT = ROOT / "reports/vlm/cross_source_target_eval_scout_p097a.json"
DEFAULT_REPORT = ROOT / "reports/vlm/cross_source_target_eval_scout_p097a.md"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def bbox4(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(v) for v in value[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def area_bucket(box: list[float] | None) -> str:
    if not box:
        return "invalid"
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area <= 32 * 32:
        return "tiny"
    if area <= 96 * 96:
        return "small"
    if area <= 192 * 192:
        return "medium"
    return "large"


def audit_path(path: Path) -> dict[str, Any]:
    rows = load_jsonl(path)
    by_source: dict[str, Counter[str]] = defaultdict(Counter)
    by_source_family: dict[str, dict[str, Counter[str]]] = defaultdict(lambda: defaultdict(Counter))
    label_counts: dict[str, Counter[str]] = defaultdict(Counter)
    area_counts: dict[str, Counter[str]] = defaultdict(Counter)
    integrity = Counter()

    for row in rows:
        source = str(row.get("source_dataset") or "unknown")
        targets = row.get("targets") if isinstance(row.get("targets"), dict) else {}
        by_source[source]["records"] += 1
        if row.get("source_integrity", {}).get("runtime_input") == "raster_image_only":
            integrity["raster_only_records"] += 1
        for family, items in targets.items():
            if not isinstance(items, list):
                continue
            by_source[source][f"{family}_targets"] += len(items)
            by_source_family[source][family]["targets"] += len(items)
            for item in items:
                label = str(item.get("semantic_type") or item.get("label") or "unknown")
                box = bbox4(item.get("bbox"))
                label_counts[f"{source}:{family}"][label] += 1
                area_counts[f"{source}:{family}"][area_bucket(box)] += 1
                if box is None:
                    by_source_family[source][family]["invalid_bbox"] += 1
                else:
                    by_source_family[source][family]["valid_bbox"] += 1
    source_summary = {source: dict(counter) for source, counter in sorted(by_source.items())}
    family_summary = {
        source: {family: dict(counter) for family, counter in sorted(families.items())}
        for source, families in sorted(by_source_family.items())
    }
    return {
        "path": str(path.relative_to(ROOT) if path.is_absolute() else path),
        "records": len(rows),
        "source_summary": source_summary,
        "family_summary": family_summary,
        "top_labels": {key: dict(counter.most_common(20)) for key, counter in sorted(label_counts.items())},
        "area_buckets": {key: dict(counter.most_common()) for key, counter in sorted(area_counts.items())},
        "integrity": dict(integrity),
    }


def render_report(summary: dict[str, Any]) -> str:
    lines = [
        "# SCI-P1-97A Cross-Source Target Evaluation Scout",
        "",
        "## Decision",
        "",
        summary["decision"],
        "",
        "## Dataset Summaries",
        "",
    ]
    for dataset in summary["datasets"]:
        lines.append(f"### `{dataset['path']}`")
        lines.append("")
        lines.append(f"- Records: `{dataset['records']}`")
        lines.append("- Source/family counts:")
        for source, counts in dataset["source_summary"].items():
            parts = [f"{key}={value}" for key, value in sorted(counts.items())]
            lines.append(f"  - `{source}`: " + ", ".join(parts))
        lines.append("")
    lines.extend([
        "## What This Supports",
        "",
        "- Cross-source target-level / node-family availability analysis.",
        "- CubiCasa5K vs FloorPlanCAD family distribution comparisons.",
        "- Planning detector/node baselines for SCI experiments.",
        "",
        "## What This Does Not Support",
        "",
        "- Scene-graph relation F1 claims.",
        "- `contains` relation generalization claims.",
        "- Full real industrial drawing robustness claims.",
        "",
        "## Next Step",
        "",
        "Use this scout to define either a target-level detector/node evaluation baseline or move to the real registry annotation pilot. For SCI2 evidence, a small manually annotated relation graph pilot is still required.",
    ])
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", default=[])
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()

    inputs = [Path(item) for item in args.input] if args.input else DEFAULT_INPUTS
    datasets = [audit_path(path if path.is_absolute() else ROOT / path) for path in inputs]
    summary = {
        "id": "SCI-P1-97A-cross-source-target-adapter-scout",
        "decision": "public_raster_moe_supervision_v19 supports cross-source target-level evaluation for CubiCasa5K and FloorPlanCAD, but not relation-graph claims without an adapter or manual relation labels.",
        "datasets": datasets,
        "claim_boundary": {
            "can_claim": ["cross-source target availability", "family/label/area distribution", "raster-only target integrity"],
            "cannot_claim": ["relation F1", "contains policy generalization", "SCI2-ready real drawing robustness"]
        },
        "next_step": "SCI-P1-97B-real-registry-annotation-pilot-plan"
    }
    write_json(Path(args.output), summary)
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(render_report(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
