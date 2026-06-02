#!/usr/bin/env python3
"""Diagnose candidate granularity failures behind unrecoverable contains_symbol gold.

This is an offline audit script. It loads gold only after detector/topology
outputs have been produced and never writes gold-derived fields back into an
inference stream.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_topology_relations_v18 import bbox, center, center_covered, contains_point, integrity, iou, load_gold, write_json
from diagnose_contains_symbol_missing_gold_v18 import (
    best_match,
    candidate_groups,
    dataset_recoverable_keys,
    gold_contains_symbols,
    relation_constructible,
    rows_by_candidate_pair,
)
from nms_topology_relations_v18 import load_by_id, load_jsonl

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"

DEFAULT_DATASET = REPORT / "contains_symbol_bipartite_dataset.jsonl"
DEFAULT_ADAPTER = REPORT / "detector_adapter_v18_symbol_boundary_fixed_candidates.jsonl"
DEFAULT_OUTPUT = REPORT / "contains_symbol_candidate_granularity_diagnostic.json"


def match_score(candidate_box: list[float], gold_box: list[float]) -> float:
    score = iou(candidate_box, gold_box)
    if center_covered(candidate_box, gold_box):
        score = max(score, 0.5)
    return score


def matching_gold_ids(candidate: dict[str, Any] | None, gold_items: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    cb = bbox((candidate or {}).get("bbox"))
    if cb is None:
        return []
    matches: list[dict[str, Any]] = []
    for item in gold_items:
        gb = bbox(item.get("bbox"))
        if gb is None:
            continue
        score = match_score(cb, gb)
        if score >= threshold:
            matches.append(
                {
                    "target_id": str(item.get("target_id")),
                    "score": round(score, 6),
                    "iou": round(iou(cb, gb), 6),
                    "center_covered": bool(center_covered(cb, gb)),
                }
            )
    matches.sort(key=lambda item: item["score"], reverse=True)
    return matches


def gold_centers_inside(candidate: dict[str, Any] | None, gold_items: list[dict[str, Any]]) -> list[str]:
    cb = bbox((candidate or {}).get("bbox"))
    if cb is None:
        return []
    covered: list[str] = []
    for item in gold_items:
        gb = bbox(item.get("bbox"))
        if gb is None:
            continue
        gx, gy = center(gb)
        if contains_point(cb, gx, gy, margin=2.0):
            covered.append(str(item.get("target_id")))
    return covered


def geometry_failure_bucket(room_candidate: dict[str, Any] | None, symbol_candidate: dict[str, Any] | None, gold_item: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    rb = bbox((room_candidate or {}).get("bbox"))
    sb = bbox((symbol_candidate or {}).get("bbox"))
    if rb is None or sb is None:
        return "invalid_candidate_bbox", {}
    sx, sy = center(sb)
    gx, gy = center(gold_item["symbol_bbox"])
    distances = {
        "left": round(rb[0] - sx, 6),
        "right": round(sx - rb[2], 6),
        "top": round(rb[1] - sy, 6),
        "bottom": round(sy - rb[3], 6),
    }
    outside = {side: value for side, value in distances.items() if value > 2.0}
    gold_center_inside_room_candidate = contains_point(rb, gx, gy, margin=2.0)
    if outside and gold_center_inside_room_candidate:
        bucket = "symbol_candidate_center_shifted_outside_room"
    elif outside:
        bucket = "room_candidate_bbox_misses_symbol_region"
    else:
        bucket = "geometry_contract_edge_case"
    debug = {
        "symbol_candidate_center": [round(sx, 3), round(sy, 3)],
        "gold_symbol_center": [round(gx, 3), round(gy, 3)],
        "gold_symbol_center_inside_room_candidate": bool(gold_center_inside_room_candidate),
        "outside_room_by_pixels": outside,
    }
    return bucket, debug


def pair_failure_bucket(
    pair_rows: list[dict[str, Any]],
    room_matches: list[dict[str, Any]],
    symbol_matches: list[dict[str, Any]],
    covered_symbol_ids: list[str],
    gold_item: dict[str, Any],
) -> tuple[str, dict[str, Any]]:
    gold_room = str(gold_item["gold_room_id"])
    gold_symbol = str(gold_item["gold_symbol_id"])
    room_ids = {item["target_id"] for item in room_matches}
    symbol_ids = {item["target_id"] for item in symbol_matches}
    labels = [row.get("labels") for row in pair_rows if isinstance(row.get("labels"), dict)]
    positive_keys = sorted({str(label.get("gold_key")) for label in labels if label.get("gold_key")})
    positive_rooms = sorted({str(label.get("gold_room_id")) for label in labels if label.get("gold_room_id")})
    positive_symbols = sorted({str(label.get("gold_symbol_id")) for label in labels if label.get("gold_symbol_id")})

    if gold_symbol not in symbol_ids:
        bucket = "best_symbol_candidate_matches_other_gold_symbol"
    elif gold_room not in room_ids:
        bucket = "best_room_candidate_matches_other_gold_room"
    elif len(covered_symbol_ids) > 1:
        bucket = "coarse_symbol_candidate_covers_multiple_gold_symbols"
    elif positive_keys:
        bucket = "candidate_pair_labeled_to_other_canonical_key"
    else:
        bucket = "candidate_pair_edge_exists_without_positive_label"

    debug = {
        "pair_edge_count": len(pair_rows),
        "pair_positive_keys_sample": positive_keys[:10],
        "pair_positive_rooms": positive_rooms[:10],
        "pair_positive_symbols": positive_symbols[:10],
        "room_match_ids": sorted(room_ids)[:10],
        "symbol_match_ids": sorted(symbol_ids)[:10],
        "symbol_candidate_gold_centers_inside": covered_symbol_ids[:20],
    }
    return bucket, debug


def summarize_examples(examples: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    return examples[:limit]


def diagnose(
    dataset_rows: list[dict[str, Any]],
    adapter_by_id: dict[str, dict[str, Any]],
    gold: dict[str, dict[str, Any]],
    sample_limit: int,
) -> dict[str, Any]:
    gold_rows = gold_contains_symbols(gold)
    recoverable = dataset_recoverable_keys(dataset_rows) & {str(row["gold_key"]) for row in gold_rows}
    by_pair = rows_by_candidate_pair(dataset_rows)
    granularity_counts: Counter[str] = Counter()
    primary_counts: Counter[str] = Counter()
    symbol_type_counts: dict[str, Counter[str]] = defaultdict(Counter)
    row_counts: Counter[str] = Counter()
    examples: list[dict[str, Any]] = []

    for gold_item in gold_rows:
        key = str(gold_item["gold_key"])
        if key in recoverable:
            continue
        row_id = str(gold_item["row_id"])
        row_counts[row_id] += 1
        adapter = adapter_by_id.get(row_id)
        if not adapter:
            primary = "missing_adapter_row"
            bucket = "missing_adapter_row"
            case = {"gold": gold_item, "primary_reason": primary, "granularity_bucket": bucket}
        else:
            groups = candidate_groups(adapter)
            room_best = best_match(gold_item["room_bbox"], groups.get("space", []), threshold=0.3)
            symbol_best = best_match(gold_item["symbol_bbox"], groups.get("symbol", []), threshold=0.25)
            room_candidate = next((cand for cand in groups.get("space", []) if cand.get("candidate_id") == room_best.get("candidate_id")), None)
            symbol_candidate = next((cand for cand in groups.get("symbol", []) if cand.get("candidate_id") == symbol_best.get("candidate_id")), None)
            room_ok = bool(room_best.get("passes_match_threshold"))
            symbol_ok = bool(symbol_best.get("passes_match_threshold"))
            room_gold_items = (gold["rooms"].get(row_id) or {}).get("rooms") or []
            symbol_gold_items = gold["symbols"].get(row_id) or []
            room_matches = matching_gold_ids(room_candidate, room_gold_items, threshold=0.3)
            symbol_matches = matching_gold_ids(symbol_candidate, symbol_gold_items, threshold=0.25)
            covered_symbol_ids = gold_centers_inside(symbol_candidate, symbol_gold_items)

            if not room_ok and not symbol_ok:
                primary = "missing_room_and_symbol_candidate"
                bucket = "missing_room_and_symbol_candidate"
                debug: dict[str, Any] = {}
            elif not room_ok:
                primary = "missing_room_candidate"
                bucket = "missing_room_candidate"
                debug = {"best_room_match_score": room_best.get("match_score")}
            elif not symbol_ok:
                primary = "missing_symbol_candidate"
                bucket = "missing_symbol_candidate"
                debug = {
                    "best_symbol_match_score": symbol_best.get("match_score"),
                    "symbol_candidate_gold_centers_inside": covered_symbol_ids[:20],
                }
            else:
                geometry = relation_constructible(room_candidate or {}, symbol_candidate or {})
                if not geometry.get("constructible_by_geometry"):
                    primary = "matched_candidates_fail_contains_geometry"
                    bucket, debug = geometry_failure_bucket(room_candidate, symbol_candidate, gold_item)
                else:
                    pair_rows = by_pair.get((row_id, str(room_best.get("candidate_id")), str(symbol_best.get("candidate_id"))), [])
                    if not pair_rows:
                        primary = "relation_not_constructed_or_capped"
                        bucket = "relation_not_constructed_or_capped"
                        debug = {}
                    else:
                        primary = "candidate_pair_exists_but_not_canonical_positive"
                        bucket, debug = pair_failure_bucket(pair_rows, room_matches, symbol_matches, covered_symbol_ids, gold_item)
            case = {
                "gold": gold_item,
                "primary_reason": primary,
                "granularity_bucket": bucket,
                "best_room_candidate": room_best,
                "best_symbol_candidate": symbol_best,
                "room_candidate_match_ids": [item["target_id"] for item in room_matches[:10]],
                "symbol_candidate_match_ids": [item["target_id"] for item in symbol_matches[:10]],
                "debug": debug,
                "candidate_counts": {family: len(groups.get(family, [])) for family in ["space", "boundary", "symbol", "text"]},
            }
        primary_counts[primary] += 1
        granularity_counts[bucket] += 1
        symbol_type_counts[str(gold_item.get("symbol_type") or "symbol")][bucket] += 1
        if len(examples) < sample_limit:
            examples.append(case)

    missing_total = max(0, len(gold_rows) - len(recoverable))
    return {
        "gold_total": len(gold_rows),
        "recoverable_gold": len(recoverable),
        "missing_gold": missing_total,
        "recoverable_recall": round(len(recoverable) / max(len(gold_rows), 1), 6),
        "primary_reason_counts": dict(primary_counts),
        "granularity_bucket_counts": dict(granularity_counts),
        "granularity_bucket_rates": {key: round(value / max(missing_total, 1), 6) for key, value in granularity_counts.items()},
        "symbol_type_granularity_counts": {key: dict(value) for key, value in sorted(symbol_type_counts.items())},
        "worst_rows": [{"row_id": row_id, "missing_gold": count} for row_id, count in row_counts.most_common(50)],
        "examples": summarize_examples(examples, sample_limit),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--adapter", default=str(DEFAULT_ADAPTER))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--sample-limit", type=int, default=500)
    args = parser.parse_args()

    dataset_rows = load_jsonl(Path(args.dataset))
    adapter_by_id = load_by_id(Path(args.adapter))
    gold = load_gold()
    report = {
        "task": "IMG-MOE-V18-REBUILD-001.step_f_candidate_granularity_diagnostic",
        "dataset": str(args.dataset),
        "adapter": str(args.adapter),
        "diagnostic": diagnose(dataset_rows, adapter_by_id, gold, sample_limit=args.sample_limit),
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
                "granularity_bucket_counts": diag["granularity_bucket_counts"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
