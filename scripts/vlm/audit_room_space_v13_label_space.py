#!/usr/bin/env python3
"""Audit the room-space v13 label space and split integrity."""

from __future__ import annotations

import argparse
import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from v5_pipeline_utils import write_json


VALID_CUBICASA_LABELS = {
    "balcony",
    "bathroom",
    "bedroom",
    "closet",
    "corridor",
    "kitchen",
    "living_room",
    "office",
    "room",
    "storage",
    "toilet",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/train.jsonl")
    parser.add_argument("--dev", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/dev.jsonl")
    parser.add_argument("--locked", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/locked_test.jsonl")
    parser.add_argument("--resplan-pkl", default="datasets/external/resplan/ResPlan.pkl")
    parser.add_argument("--output", default="reports/vlm/room_space_v13_label_space_audit.json")
    args = parser.parse_args()

    split_rows = {
        "train": load_jsonl(Path(args.train)),
        "dev": load_jsonl(Path(args.dev)),
        "locked_test": load_jsonl(Path(args.locked)),
    }
    split_audit = {name: audit_split(rows) for name, rows in split_rows.items()}
    leakage = leakage_audit(split_rows)
    resplan = audit_resplan(Path(args.resplan_pkl))
    pollution = {
        "cubicasa_foreign_labels": {
            name: sorted(set(audit["room_label_counts"]) - VALID_CUBICASA_LABELS)
            for name, audit in split_audit.items()
        },
        "resplan_labels_not_allowed_in_cubicasa_head": sorted(set(resplan["room_label_counts"]) - VALID_CUBICASA_LABELS),
        "policy": "ResPlan-only labels may be used for audit/pretraining diagnostics, but not as CubiCasa locked classification labels.",
    }
    report = {
        "version": "room_space_v13_label_space_audit",
        "paths": {"train": args.train, "dev": args.dev, "locked_test": args.locked, "resplan_pkl": args.resplan_pkl},
        "valid_cubicasa_labels": sorted(VALID_CUBICASA_LABELS),
        "splits": split_audit,
        "leakage": leakage,
        "resplan": resplan,
        "label_pollution": pollution,
        "ready_for_v13_training": leakage["locked_overlap_total"] == 0
        and all(not labels for labels in pollution["cubicasa_foreign_labels"].values()),
    }
    write_json(args.output, report)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def audit_split(rows: list[dict[str, Any]]) -> dict[str, Any]:
    room_counts: Counter[str] = Counter()
    row_room_counts = []
    ids = []
    for row in rows:
        ids.append(record_id(row))
        room_candidates = (row.get("expected_json") or {}).get("room_candidates") or []
        row_room_counts.append(len(room_candidates))
        for item in room_candidates:
            if isinstance(item, dict):
                room_counts[str(item.get("room_type") or "room")] += 1
    return {
        "rows": len(rows),
        "unique_record_ids": len(set(ids)),
        "room_candidates": sum(row_room_counts),
        "max_room_candidates_per_row": max(row_room_counts) if row_room_counts else 0,
        "room_label_counts": dict(sorted(room_counts.items())),
    }


def leakage_audit(split_rows: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    ids = {name: {record_id(row) for row in rows} for name, rows in split_rows.items()}
    pairs = {}
    for left, right in [("train", "dev"), ("train", "locked_test"), ("dev", "locked_test")]:
        overlap = sorted(ids[left] & ids[right])
        pairs[f"{left}_vs_{right}"] = {"count": len(overlap), "examples": overlap[:10]}
    return {
        "pairwise": pairs,
        "locked_overlap_total": pairs["train_vs_locked_test"]["count"] + pairs["dev_vs_locked_test"]["count"],
    }


def audit_resplan(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"exists": False, "records": 0, "room_label_counts": {}}
    raw = pickle.load(path.open("rb"))
    labels = Counter()
    for record in raw:
        if not isinstance(record, dict):
            continue
        for key in ["living", "kitchen", "bedroom", "bathroom", "balcony", "garden", "parking", "pool", "inner"]:
            if record.get(key) is not None:
                labels[key] += 1
    return {"exists": True, "records": len(raw), "room_label_counts": dict(sorted(labels.items()))}


def record_id(row: dict[str, Any]) -> str:
    return str(row.get("annotation_path") or row.get("image_path") or row.get("id") or "")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
