#!/usr/bin/env python3
"""Attribute unrecoverable contains_symbol gold relations to candidate-stage causes."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_topology_relations_v18 import bbox, center, center_covered, contains_point, integrity, iou, load_gold, write_json
from nms_topology_relations_v18 import load_by_id, load_jsonl, row_candidate_map

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"

DEFAULT_DATASET = REPORT / "contains_symbol_bipartite_dataset.jsonl"
DEFAULT_ADAPTER = REPORT / "detector_adapter_v18_symbol_boundary_fixed_candidates.jsonl"
DEFAULT_OUTPUT = REPORT / "contains_symbol_missing_gold_diagnostic.json"


def gold_contains_symbols(gold: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row_id, room_data in gold["rooms"].items():
        for sym in gold["symbols"].get(row_id) or []:
            sb = bbox(sym.get("bbox"))
            if sb is None:
                continue
            sx, sy = center(sb)
            for room in room_data.get("rooms") or []:
                rb = bbox(room.get("bbox"))
                if rb and contains_point(rb, sx, sy, margin=2.0):
                    rows.append(
                        {
                            "row_id": row_id,
                            "gold_room_id": str(room.get("target_id")),
                            "gold_symbol_id": str(sym.get("target_id")),
                            "gold_key": f"{row_id}|{room.get('target_id')}|{sym.get('target_id')}",
                            "room_bbox": rb,
                            "symbol_bbox": sb,
                            "symbol_type": sym.get("symbol_type") or sym.get("semantic_type") or "symbol",
                        }
                    )
                    break
    return rows


def labeler_gold_universe(gold: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    """All room-symbol bbox containment pairs accepted by the current labeler."""
    rows: list[dict[str, Any]] = []
    for row_id, room_data in gold["rooms"].items():
        for sym in gold["symbols"].get(row_id) or []:
            sb = bbox(sym.get("bbox"))
            if sb is None:
                continue
            sx, sy = center(sb)
            for room in room_data.get("rooms") or []:
                rb = bbox(room.get("bbox"))
                if rb and contains_point(rb, sx, sy, margin=2.0):
                    rows.append(
                        {
                            "row_id": row_id,
                            "gold_room_id": str(room.get("target_id")),
                            "gold_symbol_id": str(sym.get("target_id")),
                            "gold_key": f"{row_id}|{room.get('target_id')}|{sym.get('target_id')}",
                            "room_bbox": rb,
                            "symbol_bbox": sb,
                            "symbol_type": sym.get("symbol_type") or sym.get("semantic_type") or "symbol",
                        }
                    )
    return rows


def best_match(gold_box: list[float], candidates: list[dict[str, Any]], threshold: float) -> dict[str, Any]:
    best: dict[str, Any] | None = None
    for cand in candidates:
        cb = bbox(cand.get("bbox"))
        if cb is None:
            continue
        score = iou(cb, gold_box)
        center_hit = center_covered(cb, gold_box)
        if center_hit:
            score = max(score, 0.5)
        item = {
            "candidate_id": cand.get("candidate_id"),
            "family": cand.get("family"),
            "bbox": cb,
            "confidence": cand.get("confidence"),
            "iou": round(iou(cb, gold_box), 6),
            "center_covered": bool(center_hit),
            "match_score": round(score, 6),
            "passes_match_threshold": score >= threshold,
        }
        if best is None or score > float(best["match_score"]):
            best = item
    return best or {"candidate_id": None, "match_score": 0.0, "passes_match_threshold": False}


def box_center_distance(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return 999999.0
    lx, ly = center(left)
    rx, ry = center(right)
    return math.hypot(lx - rx, ly - ry)


def relation_constructible(room_candidate: dict[str, Any], symbol_candidate: dict[str, Any]) -> dict[str, Any]:
    rb = bbox(room_candidate.get("bbox"))
    sb = bbox(symbol_candidate.get("bbox"))
    if rb is None or sb is None:
        return {"constructible_by_geometry": False, "reason": "invalid_candidate_bbox"}
    sx, sy = center(sb)
    return {
        "constructible_by_geometry": contains_point(rb, sx, sy, margin=2.0),
        "symbol_center_inside_room_candidate": contains_point(rb, sx, sy, margin=2.0),
        "symbol_center_to_room_center_distance": round(box_center_distance(rb, sb), 6),
    }


def dataset_recoverable_keys(dataset_rows: list[dict[str, Any]]) -> set[str]:
    keys: set[str] = set()
    for row in dataset_rows:
        labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
        key = labels.get("gold_key")
        if key:
            keys.add(str(key))
    return keys


def rows_by_gold_key(dataset_rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in dataset_rows:
        labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
        key = labels.get("gold_key")
        if key:
            out[str(key)].append(row)
    return out


def rows_by_candidate_pair(dataset_rows: list[dict[str, Any]]) -> dict[tuple[str, str, str], list[dict[str, Any]]]:
    out: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in dataset_rows:
        out[(str(row.get("row_id")), str(row.get("source_candidate_id")), str(row.get("target_candidate_id")))].append(row)
    return out


def candidate_groups(adapter_row: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for cand in row_candidate_map(adapter_row).values():
        groups[str(cand.get("family") or "unknown")].append(cand)
    return groups


def diagnose_missing(
    gold_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    adapter_by_id: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    gold_key_set = {str(row["gold_key"]) for row in gold_rows}
    recoverable = dataset_recoverable_keys(dataset_rows) & gold_key_set
    by_gold = rows_by_gold_key(dataset_rows)
    missing_cases: list[dict[str, Any]] = []
    recovered_multi_edge = Counter()
    by_pair = rows_by_candidate_pair(dataset_rows)
    for key, rows in by_gold.items():
        recovered_multi_edge[len(rows)] += 1

    reason_counts: Counter[str] = Counter()
    symbol_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    row_counts: Counter[str] = Counter()
    for gold_item in gold_rows:
        key = gold_item["gold_key"]
        if key in recoverable:
            continue
        row_id = str(gold_item["row_id"])
        adapter = adapter_by_id.get(row_id)
        if not adapter:
            reason = "missing_adapter_row"
            case = {"gold": gold_item, "primary_reason": reason}
        else:
            groups = candidate_groups(adapter)
            room_best = best_match(gold_item["room_bbox"], groups.get("space", []), threshold=0.3)
            symbol_best = best_match(gold_item["symbol_bbox"], groups.get("symbol", []), threshold=0.25)
            room_candidate = next((cand for cand in groups.get("space", []) if cand.get("candidate_id") == room_best.get("candidate_id")), None)
            symbol_candidate = next((cand for cand in groups.get("symbol", []) if cand.get("candidate_id") == symbol_best.get("candidate_id")), None)
            room_ok = bool(room_best.get("passes_match_threshold"))
            symbol_ok = bool(symbol_best.get("passes_match_threshold"))
            geometry = relation_constructible(room_candidate or {}, symbol_candidate or {}) if room_ok and symbol_ok else {}
            if not room_ok and not symbol_ok:
                reason = "missing_room_and_symbol_candidate"
            elif not room_ok:
                reason = "missing_room_candidate"
            elif not symbol_ok:
                reason = "missing_symbol_candidate"
            elif not geometry.get("constructible_by_geometry"):
                reason = "matched_candidates_fail_contains_geometry"
            else:
                pair_rows = by_pair.get((row_id, str(room_best.get("candidate_id")), str(symbol_best.get("candidate_id"))), [])
                if not pair_rows:
                    reason = "relation_not_constructed_or_capped"
                    pair_debug = {"candidate_pair_edge_exists": False}
                else:
                    reason = "candidate_pair_exists_but_not_canonical_positive"
                    pair_debug = {
                        "candidate_pair_edge_exists": True,
                        "edge_count": len(pair_rows),
                        "edge_labels": [row.get("labels") for row in pair_rows[:5]],
                    }
            case = {
                "gold": gold_item,
                "primary_reason": reason,
                "best_room_candidate": room_best,
                "best_symbol_candidate": symbol_best,
                "relation_geometry": geometry,
                "candidate_pair_debug": pair_debug if room_ok and symbol_ok and geometry.get("constructible_by_geometry") else {},
                "candidate_counts": {family: len(groups.get(family, [])) for family in ["space", "boundary", "symbol", "text"]},
            }
        reason_counts[reason] += 1
        symbol_type_counts[str(gold_item.get("symbol_type") or "symbol")][reason] += 1
        row_counts[row_id] += 1
        if len(missing_cases) < 500:
            missing_cases.append(case)

    worst_rows = [
        {"row_id": row_id, "missing_gold": count}
        for row_id, count in row_counts.most_common(50)
    ]
    return {
        "gold_total": len(gold_rows),
        "recoverable_gold": len(recoverable),
        "missing_gold": max(0, len(gold_rows) - len(recoverable)),
        "recoverable_recall": round(len(recoverable) / max(len(gold_rows), 1), 6),
        "reason_counts": dict(reason_counts),
        "reason_rates": {key: round(value / max(len(gold_rows) - len(recoverable), 1), 6) for key, value in reason_counts.items()},
        "symbol_type_reason_counts": {symbol_type: dict(counts) for symbol_type, counts in sorted(symbol_type_counts.items())},
        "worst_rows": worst_rows,
        "recovered_multi_edge_histogram": dict(recovered_multi_edge),
        "missing_cases_sample": missing_cases,
    }


def label_space_audit(canonical_gold_rows: list[dict[str, Any]], labeler_gold_rows: list[dict[str, Any]], dataset_rows: list[dict[str, Any]]) -> dict[str, Any]:
    canonical = {row["gold_key"] for row in canonical_gold_rows}
    labeler = {row["gold_key"] for row in labeler_gold_rows}
    dataset = dataset_recoverable_keys(dataset_rows)
    return {
        "canonical_gold_keys": len(canonical),
        "labeler_gold_keys": len(labeler),
        "dataset_positive_keys": len(dataset),
        "dataset_positive_in_canonical": len(dataset & canonical),
        "dataset_positive_outside_canonical": len(dataset - canonical),
        "canonical_missing_from_dataset": len(canonical - dataset),
        "dataset_positive_in_labeler": len(dataset & labeler),
        "dataset_positive_outside_labeler": len(dataset - labeler),
        "labeler_missing_from_dataset": len(labeler - dataset),
        "canonical_recall_by_dataset": round(len(dataset & canonical) / max(len(canonical), 1), 6),
        "labeler_recall_by_dataset": round(len(dataset & labeler) / max(len(labeler), 1), 6),
        "status": "label_contract_mismatch" if dataset - canonical else "aligned_with_canonical",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    dataset_rows = load_jsonl(Path(args.dataset))
    adapter_by_id = load_by_id(Path(args.adapter))
    gold = load_gold()
    canonical_gold_rows = gold_contains_symbols(gold)
    labeler_gold_rows = labeler_gold_universe(gold)
    gold_rows = canonical_gold_rows
    if args.smoke:
        keep_rows = {str(row.get("row_id")) for row in dataset_rows[:10000]}
        gold_rows = [row for row in gold_rows if row["row_id"] in keep_rows]
    report = {
        "task": "IMG-MOE-V18-REBUILD-001.step_e_build_contains_symbol_missing_gold_diagnostic",
        "dataset": str(args.dataset),
        "adapter": str(args.adapter),
        "smoke": bool(args.smoke),
        "label_space_audit": label_space_audit(canonical_gold_rows, labeler_gold_rows, dataset_rows),
        "diagnostic": diagnose_missing(gold_rows, dataset_rows, adapter_by_id),
        "labeler_universe_diagnostic": diagnose_missing(labeler_gold_rows, dataset_rows, adapter_by_id),
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_diagnosis_only": True,
        "gold_used_for_inference": False,
    }
    write_json(Path(args.output), report)
    diag = report["diagnostic"]
    print(
        json.dumps(
            {
                "gold_total": diag["gold_total"],
                "recoverable_gold": diag["recoverable_gold"],
                "missing_gold": diag["missing_gold"],
                "recoverable_recall": diag["recoverable_recall"],
                "reason_counts": diag["reason_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
