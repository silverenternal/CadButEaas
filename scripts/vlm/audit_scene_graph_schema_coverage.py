#!/usr/bin/env python3
"""Audit whether the scene-graph ontology covers the planned MoE contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


REQUIRED_NODE_GROUPS = {
    "wall": ["hard_wall", "partition_wall", "curtain_wall"],
    "opening": ["door", "window", "opening"],
    "room": ["room", "bedroom", "living_room", "kitchen", "bathroom", "corridor"],
    "symbol": ["stair", "column", "sink", "shower", "appliance", "equipment", "generic_symbol"],
    "text": ["room_label", "dimension_text", "dimension_line", "leader_line", "note_text"],
    "layout": ["title_block", "table", "schedule", "legend", "stamp"],
}

REQUIRED_RELATIONS = {"contains", "bounds", "attached_to", "labels", "dimension_of", "adjacent_to"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ontology", default="configs/vlm/cadstruct_ontology.json")
    parser.add_argument("--output", default="reports/vlm/scene_graph_schema_coverage_v1.json")
    args = parser.parse_args()

    ontology = json.loads(Path(args.ontology).read_text(encoding="utf-8"))
    labels_by_family = {
        str(family): set(str(label) for label in cfg.get("labels") or [])
        for family, cfg in (ontology.get("families") or {}).items()
        if isinstance(cfg, dict)
    }
    all_labels = set().union(*labels_by_family.values()) if labels_by_family else set()
    relations = set(str(item) for item in ontology.get("relation_types") or [])

    node_groups: dict[str, Any] = {}
    for group, labels in REQUIRED_NODE_GROUPS.items():
        present = sorted(set(labels) & all_labels)
        missing = sorted(set(labels) - all_labels)
        node_groups[group] = {
            "required_labels": labels,
            "present": present,
            "missing": missing,
            "covered": not missing,
        }

    report = {
        "version": "scene_graph_schema_coverage_v1",
        "ontology": args.ontology,
        "schema_contract": {
            "node_groups": sorted(REQUIRED_NODE_GROUPS),
            "minimum_relation_types": sorted(REQUIRED_RELATIONS),
        },
        "node_group_coverage": node_groups,
        "relation_coverage": {
            "present_required": sorted(REQUIRED_RELATIONS & relations),
            "missing_required": sorted(REQUIRED_RELATIONS - relations),
            "all_relation_types": sorted(relations),
            "relation_type_count": len(relations),
            "covered": REQUIRED_RELATIONS.issubset(relations) and len(relations) >= 6,
        },
        "family_coverage": {
            family: sorted(labels) for family, labels in sorted(labels_by_family.items())
        },
    }
    report["status"] = "ok" if all(item["covered"] for item in node_groups.values()) and report["relation_coverage"]["covered"] else "blocked"

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
