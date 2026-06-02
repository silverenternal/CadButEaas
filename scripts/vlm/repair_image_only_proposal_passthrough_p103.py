#!/usr/bin/env python3
"""Conservative image-only proposal passthrough repair for P103.

The v15 image-only MoE graph drops most space/text/symbol boxes. This utility
adds raster proposal boxes back as low-confidence nodes for selected families so
we can quantify whether the proposal stage contains usable target-level recall.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


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


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    bb = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(aa + bb - inter, 1e-9)


def semantic_type(box: dict[str, Any]) -> str:
    label = str(box.get("class") or box.get("semantic_type") or box.get("family") or "unknown")
    if label == "room_boundary":
        return "wall"
    return label


def node_bbox(node: dict[str, Any]) -> list[float] | None:
    return bbox4((node.get("geometry") or {}).get("bbox") or node.get("bbox"))


def should_add(box: dict[str, Any], existing: list[dict[str, Any]], duplicate_iou: float) -> bool:
    box_bbox = bbox4(box.get("bbox"))
    if box_bbox is None:
        return False
    family = str(box.get("family") or "")
    for node in existing:
        if str(node.get("family")) != family:
            continue
        existing_bbox = node_bbox(node)
        if existing_bbox is not None and iou(box_bbox, existing_bbox) >= duplicate_iou:
            return False
    return True


def proposal_node(row_id: str, box: dict[str, Any], index: int, confidence: float) -> dict[str, Any]:
    family = str(box.get("family") or "unknown")
    bbox = bbox4(box.get("bbox"))
    assert bbox is not None
    return {
        "id": f"{row_id}_p103_{family}_{index:05d}",
        "family": family,
        "semantic_type": semantic_type(box),
        "confidence": confidence,
        "geometry": {"bbox": bbox},
        "source_expert": "image_only_proposal_passthrough_p103",
        "audit_trace": {
            "origin": "image_only_multitask_proposal_v15_locked_predictions",
            "stage": "p103_passthrough_repair",
            "label_source": box.get("label_source"),
            "coordinate_space": box.get("coordinate_space"),
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposals", default="reports/vlm/image_only_multitask_proposal_v15_locked_predictions.jsonl")
    parser.add_argument("--predictions", default="reports/vlm/image_only_moe_predictions_v15.jsonl")
    parser.add_argument("--output", default="reports/vlm/image_only_moe_predictions_v15_p103_passthrough.jsonl")
    parser.add_argument("--families", default="space,symbol,text")
    parser.add_argument("--duplicate-iou", type=float, default=0.80)
    parser.add_argument("--confidence", type=float, default=0.25)
    args = parser.parse_args()

    selected = {item.strip() for item in args.families.split(",") if item.strip()}
    proposals = {str(row.get("id")): row for row in load_jsonl(Path(args.proposals))}
    rows = load_jsonl(Path(args.predictions))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    added_by_family: dict[str, int] = {family: 0 for family in selected}
    before_by_family: dict[str, int] = {}
    after_by_family: dict[str, int] = {}

    with output.open("w", encoding="utf-8") as handle:
        for row in rows:
            row_id = str(row.get("id") or "")
            nodes = list((row.get("scene_graph") or {}).get("nodes") or [])
            for node in nodes:
                fam = str(node.get("family") or "unknown")
                before_by_family[fam] = before_by_family.get(fam, 0) + 1
            proposal_row = proposals.get(row_id, {})
            added = []
            for index, box in enumerate(proposal_row.get("boxes") or []):
                family = str(box.get("family") or "")
                if family not in selected:
                    continue
                if should_add(box, nodes + added, args.duplicate_iou):
                    node = proposal_node(row_id, box, index, args.confidence)
                    added.append(node)
                    added_by_family[family] = added_by_family.get(family, 0) + 1
            nodes.extend(added)
            for node in nodes:
                fam = str(node.get("family") or "unknown")
                after_by_family[fam] = after_by_family.get(fam, 0) + 1
            row.setdefault("scene_graph", {})["nodes"] = nodes
            row.setdefault("route_trace", {})["p103_passthrough_repair"] = {
                "families": sorted(selected),
                "duplicate_iou": args.duplicate_iou,
                "confidence": args.confidence,
                "added_nodes": len(added),
                "claim_boundary": "diagnostic proposal passthrough over raster proposal boxes; not a trained model improvement",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(json.dumps({
        "output": str(output),
        "families": sorted(selected),
        "added_by_family": added_by_family,
        "before_by_family": before_by_family,
        "after_by_family": after_by_family,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
