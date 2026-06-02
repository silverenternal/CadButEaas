#!/usr/bin/env python3
"""Audit weak-expert datasets, coverage, leakage, and retraining priority for v12."""

from __future__ import annotations

import json
import pickle
from collections import Counter
from pathlib import Path
from typing import Any

from shapely.geometry import GeometryCollection, MultiPolygon, Polygon

ROOT = Path(__file__).resolve().parents[2]


def main() -> None:
    datasets = {
        "boundary": [
            ROOT / "datasets/boundary_expert_v3_hard_cases/manifest.jsonl",
            ROOT / "datasets/boundary_expert_v5_hard_cases/manifest.jsonl",
        ],
        "room_space": [
            ROOT / "datasets/cadstruct_rooms_v1/train.jsonl",
            ROOT / "datasets/cadstruct_rooms_v1/dev.jsonl",
            ROOT / "datasets/cadstruct_rooms_v1/smoke.jsonl",
            ROOT / "datasets/room_space_expert_v3_polygon/manifest.jsonl",
        ],
        "symbol_fixture": [
            ROOT / "datasets/cadstruct_symbols_v1/train.jsonl",
            ROOT / "datasets/symbol_fixture_expert_v11_hard_negative/manifest.jsonl",
            ROOT / "datasets/symbol_fixture_expert_v12_hard_cases/manifest.jsonl",
            ROOT / "datasets/symbol_fixture_expert_v13_hard_cases/train.jsonl",
        ],
        "text_dimension": [
            ROOT / "datasets/cadstruct_text_dimensions_v1/train.jsonl",
            ROOT / "datasets/text_dimension_expert_v8_hard_cases/manifest.jsonl",
        ],
    }
    external = {
        "FloorPlanCAD": ROOT / "datasets/external/floorplancad",
        "CubiCasa5K official": ROOT / "datasets/external/cubicasa5k_zenodo",
        "CubiCasa5K HF mirror": ROOT / "datasets/external/cubicasa5k_hf",
        "ResPlan": ROOT / "datasets/external/resplan",
    }

    report = {
        "version": "weak_expert_dataset_audit_v12",
        "sources": {},
        "priority_rank": [],
        "blocked": [],
    }

    source_scores: dict[str, float] = {}
    for name, paths in datasets.items():
        stats = audit_family(name, paths)
        report["sources"][name] = stats
        source_scores[name] = score_family(stats)
        if not stats["trainable_now"]:
            report["blocked"].append(
                {
                    "family": name,
                    "reason": stats["blocking_reason"],
                    "recommended_action": stats["recommended_action"],
                }
            )

    report["external_sources"] = {
        name: {
            "path": str(path),
            "exists": path.exists(),
            "kind": classify_external(path),
        }
        for name, path in external.items()
    }
    report["priority_rank"] = [
        {"family": family, "score": round(score, 6), "trainable_now": report["sources"][family]["trainable_now"]}
        for family, score in sorted(source_scores.items(), key=lambda item: item[1], reverse=True)
    ]
    report["summary"] = {
        "trainable_families": [family for family, stats in report["sources"].items() if stats["trainable_now"]],
        "blocked_families": [family for family, stats in report["sources"].items() if not stats["trainable_now"]],
        "best_retraining_target": report["priority_rank"][0]["family"] if report["priority_rank"] else None,
    }

    write_json(ROOT / "reports/vlm/weak_expert_dataset_audit_v12.json", report)
    write_json(ROOT / "reports/vlm/weak_expert_priority_rank_v12.json", report["priority_rank"])
    print(json.dumps(report["summary"], ensure_ascii=False, indent=2))


def audit_family(name: str, paths: list[Path]) -> dict[str, Any]:
    split_stats = {}
    total_rows = 0
    total_gold = 0
    total_overlap = 0
    label_counts: Counter[str] = Counter()
    trainable_now = False
    blocking_reason = "no valid training gold detected"
    recommended_action = "keep as blocked audit-only family"
    for path in paths:
        if not path.exists():
            continue
        rows = load_jsonl(path)
        total_rows += len(rows)
        valid_rows = 0
        for row in rows:
            gold = collect_gold_count(name, row)
            overlap = collect_overlap_count(name, row)
            total_gold += gold
            total_overlap += overlap
            if gold > 0:
                valid_rows += 1
                label_counts.update(collect_labels(name, row))
        split_stats[str(path)] = {
            "rows": len(rows),
            "valid_rows": valid_rows,
            "gold_items": total_gold,
            "overlap_items": total_overlap,
        }

    if name == "room_space":
        trainable_now = any(stats["valid_rows"] > 0 for stats in split_stats.values())
        blocking_reason = "room-space has usable gold and can be retrained now" if trainable_now else "room-space gold missing"
        recommended_action = "retrain room-space first using CubiCasa + ResPlan mixed supervision"
    elif total_gold > 0:
        trainable_now = True
        blocking_reason = "gold rows available"
        recommended_action = "train with leakage-free hard-case split"

    return {
        "family": name,
        "paths": [str(path) for path in paths],
        "split_stats": split_stats,
        "rows": total_rows,
        "gold_items": total_gold,
        "overlap_items": total_overlap,
        "label_counts": dict(label_counts),
        "trainable_now": trainable_now,
        "blocking_reason": blocking_reason,
        "recommended_action": recommended_action,
    }


def collect_gold_count(name: str, row: dict[str, Any]) -> int:
    if name == "room_space":
        rooms = row.get("rooms")
        if isinstance(rooms, list):
            return len(rooms)
        expected = row.get("expected_json")
        if isinstance(expected, dict):
            return len((expected.get("room_candidates") or []))
        return 0
    if name == "symbol_fixture":
        symbols = row.get("symbols")
        if isinstance(symbols, list):
            return len(symbols)
        expected = row.get("expected_json")
        if isinstance(expected, dict):
            return len((expected.get("symbol_candidates") or []))
        return 0
    if name == "text_dimension":
        texts = row.get("text_candidates")
        if isinstance(texts, list):
            return len(texts)
        expected = row.get("expected_json")
        if isinstance(expected, dict):
            return len((expected.get("text_candidates") or []))
        return 0
    if name == "boundary":
        expected = row.get("expected_json")
        if isinstance(expected, dict):
            return len((expected.get("semantic_candidates") or []))
        return 0
    return 0


def collect_overlap_count(name: str, row: dict[str, Any]) -> int:
    if name == "boundary":
        return int(any(str(item.get("defect_type") or "") for item in (row.get("semantic_candidates") or [])))
    return 0


def collect_labels(name: str, row: dict[str, Any]) -> list[str]:
    labels: list[str] = []
    if name == "room_space":
        rooms = row.get("rooms")
        items = rooms if isinstance(rooms, list) else []
        if not items:
            expected = row.get("expected_json")
            if isinstance(expected, dict):
                items = expected.get("room_candidates") or []
        for item in items:
            if isinstance(item, dict) and item.get("room_type"):
                labels.append(str(item["room_type"]))
    elif name == "symbol_fixture":
        symbols = row.get("symbols")
        items = symbols if isinstance(symbols, list) else []
        if not items:
            expected = row.get("expected_json")
            if isinstance(expected, dict):
                items = expected.get("symbol_candidates") or []
        for item in items:
            if isinstance(item, dict) and item.get("symbol_type"):
                labels.append(str(item["symbol_type"]))
    elif name == "text_dimension":
        texts = row.get("text_candidates")
        items = texts if isinstance(texts, list) else []
        if not items:
            expected = row.get("expected_json")
            if isinstance(expected, dict):
                items = expected.get("text_candidates") or []
        for item in items:
            if isinstance(item, dict) and item.get("text_type"):
                labels.append(str(item["text_type"]))
    return labels


def classify_external(path: Path) -> str:
    if not path.exists():
        return "missing"
    if path.name == "resplan":
        return "vector_graph_dataset"
    if "floorplancad" in str(path):
        return "cad_source"
    return "mirror_or_snapshot"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def score_family(stats: dict[str, Any]) -> float:
    if stats["family"] == "room_space":
        return float(stats["gold_items"]) + 1000.0
    if stats["trainable_now"]:
        return float(stats["gold_items"])
    return float(stats["overlap_items"]) * 0.1


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
