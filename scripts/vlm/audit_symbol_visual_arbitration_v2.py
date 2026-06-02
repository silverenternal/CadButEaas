#!/usr/bin/env python3
"""Summarize conservative symbol/equipment arbitration for the visual pack."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/vlm/e2e_cubicasa_visual_demo_predictions.jsonl")
    parser.add_argument("--defect-audit", default="reports/vlm/visual_demo/defect_audit_v2.json")
    parser.add_argument("--output", default="reports/vlm/visual_demo/symbol_visual_arbitration_v2.json")
    parser.add_argument("--markdown", default="reports/vlm/visual_demo/symbol_visual_arbitration_v2.md")
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.predictions))
    if args.limit > 0:
        rows = rows[: args.limit]
    audit = load_json(Path(args.defect_audit))

    equipment_total = 0
    equipment_needs_review = 0
    equipment_raw = Counter()
    needs_review_raw = Counter()
    normal_raw = Counter()
    examples = []
    for row in rows:
        image = str(row.get("image") or "")
        for node in ((row.get("scene_graph") or {}).get("nodes") or []):
            if str(node.get("family")) != "symbol" or str(node.get("semantic_type")) != "equipment":
                continue
            metadata = node.get("metadata") if isinstance(node.get("metadata"), dict) else {}
            raw_label = str(metadata.get("raw_label") or "unknown")
            flags = node.get("quality_flags") if isinstance(node.get("quality_flags"), list) else []
            equipment_total += 1
            equipment_raw[raw_label] += 1
            item = {
                "image": image,
                "node_id": str(node.get("id") or ""),
                "raw_label": raw_label,
                "bbox": (node.get("geometry") or {}).get("bbox"),
                "quality_flags": flags,
                "decision": "needs_review" if "needs_review_outside_room" in flags else "recognized_equipment",
                "source_expert": node.get("source_expert"),
                "source_mode": (row.get("route_trace") or {}).get("source_mode"),
            }
            if "needs_review_outside_room" in flags:
                equipment_needs_review += 1
                needs_review_raw[raw_label] += 1
                examples.append(item)
            else:
                normal_raw[raw_label] += 1

    report = {
        "version": "symbol_visual_arbitration_v2",
        "inputs": {
            "predictions": str(args.predictions),
            "defect_audit": str(args.defect_audit),
        },
        "policy": {
            "rule": "equipment outside all room/space bboxes is marked needs_review_outside_room and rendered as warning, not normal recognized equipment.",
            "claim_boundary": "This is conservative arbitration for the CubiCasa expected_json visual chain. Real model-backed deletion of this issue still requires SymbolFixtureExpert hard-negative/arbitration evaluation.",
        },
        "summary": {
            "equipment_total": equipment_total,
            "equipment_recognized": equipment_total - equipment_needs_review,
            "equipment_needs_review": equipment_needs_review,
            "equipment_raw_label_counts": dict(equipment_raw.most_common()),
            "recognized_equipment_raw_label_counts": dict(normal_raw.most_common()),
            "needs_review_raw_label_counts": dict(needs_review_raw.most_common()),
            "defect_audit_counts": (audit.get("summary") or {}).get("defect_counts") or {},
        },
        "examples": examples[:50],
        "done_when_checks": {
            "raw_label_sources_traceable": bool(equipment_raw),
            "needs_review_not_normal_claim": equipment_needs_review == int(((audit.get("summary") or {}).get("defect_counts") or {}).get("needs_review_symbol", equipment_needs_review)),
            "normal_equipment_still_available": equipment_total > equipment_needs_review,
        },
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.markdown).write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


def render_markdown(report: dict[str, Any]) -> str:
    s = report["summary"]
    lines = [
        "# Symbol Visual Arbitration v2",
        "",
        f"- Equipment total: {s['equipment_total']}",
        f"- Recognized equipment: {s['equipment_recognized']}",
        f"- Needs review: {s['equipment_needs_review']}",
        f"- Raw label counts: {s['equipment_raw_label_counts']}",
        f"- Needs-review raw labels: {s['needs_review_raw_label_counts']}",
        "",
        "## Policy",
        "",
        report["policy"]["rule"],
        "",
        report["policy"]["claim_boundary"],
        "",
        "## Examples",
        "",
    ]
    for item in report["examples"][:20]:
        lines.append(f"- `{item['node_id']}` {item['raw_label']} {item['decision']} bbox={item['bbox']}")
    return "\n".join(lines) + "\n"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
