#!/usr/bin/env python3
"""Diagnose upper bounds for contains_symbol bipartite policy families."""

from __future__ import annotations

import argparse
import json
import zlib
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from build_topology_relations_v18 import bbox, center, contains_point, integrity, load_gold, write_json

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"

DEFAULT_DATASET = REPORT / "contains_symbol_bipartite_dataset.jsonl"
DEFAULT_OUTPUT = REPORT / "contains_symbol_bipartite_upper_bound.json"


def load_jsonl(path: Path, limit: int | None = None) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
                if limit is not None and len(rows) >= limit:
                    break
    return rows


def split_name(row_id: str) -> str:
    bucket = zlib.crc32(row_id.encode("utf-8")) % 10
    if bucket < 7:
        return "train"
    if bucket < 8:
        return "dev"
    return "test"


def gold_key(row: dict[str, Any]) -> str | None:
    labels = row.get("labels") if isinstance(row.get("labels"), dict) else {}
    value = labels.get("gold_key")
    return str(value) if value else None


def gold_total_by_row() -> tuple[dict[str, int], int]:
    gold = load_gold()
    by_row: dict[str, int] = Counter()
    for row_id, room_data in gold["rooms"].items():
        for sym in gold["symbols"].get(row_id) or []:
            sb = bbox(sym.get("bbox"))
            if sb is None:
                continue
            sx, sy = center(sb)
            for room in room_data.get("rooms") or []:
                rb = bbox(room.get("bbox"))
                if rb and contains_point(rb, sx, sy, margin=2.0):
                    by_row[row_id] += 1
                    break
    return dict(by_row), sum(by_row.values())


def positive_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in rows if gold_key(row)]


def summarize_selection(selected: list[dict[str, Any]], rows: list[dict[str, Any]], gold_total: int) -> dict[str, Any]:
    selected_gold = {gold_key(row) for row in selected if gold_key(row)}
    recoverable = {gold_key(row) for row in rows if gold_key(row)}
    duplicate_positive_edges = max(0, len([row for row in selected if gold_key(row)]) - len(selected_gold))
    return {
        "selected_edges": len(selected),
        "selected_positive_edges": sum(1 for row in selected if gold_key(row)),
        "selected_unique_gold": len(selected_gold),
        "recoverable_unique_gold": len(recoverable),
        "gold_total": gold_total,
        "recall_vs_recoverable": round(len(selected_gold) / max(len(recoverable), 1), 6),
        "recall_vs_all_gold": round(len(selected_gold) / max(gold_total, 1), 6),
        "candidate_reduction": round(1.0 - len(selected) / max(len(rows), 1), 6),
        "duplicate_positive_edges": duplicate_positive_edges,
    }


def single_pair_oracle(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    seen_pairs: set[str] = set()
    seen_gold: set[str] = set()
    for row in positive_rows(rows):
        pair = str(row.get("relation_pair_key"))
        key = gold_key(row)
        if pair in seen_pairs or not key or key in seen_gold:
            continue
        selected.append(row)
        seen_pairs.add(pair)
        seen_gold.add(key)
    return selected


def cap_oracle(
    rows: list[dict[str, Any]],
    room_cap: int,
    symbol_cap: int,
    component_cap: int,
    include_negatives: bool = False,
) -> list[dict[str, Any]]:
    """Greedy oracle that covers unique gold keys under room/symbol/component caps."""
    room_counts: Counter[str] = Counter()
    symbol_counts: Counter[str] = Counter()
    component_counts: Counter[str] = Counter()
    selected: list[dict[str, Any]] = []
    selected_gold: set[str] = set()
    positives = positive_rows(rows)
    by_gold: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in positives:
        key = gold_key(row)
        if key:
            by_gold[key].append(row)

    gold_order = sorted(
        by_gold,
        key=lambda key: (
            len(by_gold[key]),
            min(float(item.get("edge_features", {}).get("edge_rank_within_room") or 999999.0) for item in by_gold[key]),
        ),
    )
    for key in gold_order:
        candidates = sorted(
            by_gold[key],
            key=lambda item: (
                room_counts[str(item.get("room_instance_cluster_id"))],
                symbol_counts[str(item.get("symbol_instance_cluster_id"))],
                component_counts[str(item.get("component_id"))],
                float(item.get("edge_features", {}).get("edge_rank_within_room") or 999999.0),
                float(item.get("edge_features", {}).get("edge_rank_within_symbol") or 999999.0),
            ),
        )
        for row in candidates:
            room_id = str(row.get("room_instance_cluster_id"))
            symbol_id = str(row.get("symbol_instance_cluster_id"))
            component_id = str(row.get("component_id"))
            if room_counts[room_id] >= room_cap:
                continue
            if symbol_counts[symbol_id] >= symbol_cap:
                continue
            if component_counts[component_id] >= component_cap:
                continue
            selected.append(row)
            selected_gold.add(key)
            room_counts[room_id] += 1
            symbol_counts[symbol_id] += 1
            component_counts[component_id] += 1
            break

    if include_negatives:
        selected_ids = {str(row.get("relation_id")) for row in selected}
        for row in rows:
            if str(row.get("relation_id")) not in selected_ids and not gold_key(row):
                selected.append(row)
    return selected


def cap_sweep(rows: list[dict[str, Any]], gold_total: int) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for room_cap in [4, 8, 12, 16, 24, 32, 48, 64, 999999]:
        for symbol_cap in [1, 2, 3, 4, 6, 8, 12, 999999]:
            selected = cap_oracle(rows, room_cap=room_cap, symbol_cap=symbol_cap, component_cap=999999)
            metrics = summarize_selection(selected, rows, gold_total)
            out.append(
                {
                    "policy": {
                        "room_cap": room_cap,
                        "symbol_cap": symbol_cap,
                        "component_cap": 999999,
                        "oracle": "positive_label_greedy_no_false_positive_cost",
                    },
                    "metrics": metrics,
                }
            )
    out.sort(
        key=lambda item: (
            item["metrics"]["recall_vs_recoverable"] >= 0.98,
            item["metrics"]["candidate_reduction"],
            item["metrics"]["recall_vs_recoverable"],
            item["metrics"]["recall_vs_all_gold"],
        ),
        reverse=True,
    )
    return out


def split_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "test": [], "all": rows}
    for row in rows:
        out[split_name(str(row.get("row_id")))].append(row)
    return out


def row_gold_total(rows: list[dict[str, Any]], gold_by_row: dict[str, int]) -> int:
    row_ids = {str(row.get("row_id")) for row in rows}
    return sum(gold_by_row.get(row_id, 0) for row_id in row_ids)


def diagnose(rows: list[dict[str, Any]]) -> dict[str, Any]:
    gold_by_row, all_gold_total = gold_total_by_row()
    splits = split_rows(rows)
    split_reports: dict[str, Any] = {}
    for name, split in splits.items():
        gold_total = all_gold_total if name == "all" else row_gold_total(split, gold_by_row)
        recoverable = summarize_selection(positive_rows(split), split, gold_total)
        single_pair = summarize_selection(single_pair_oracle(split), split, gold_total)
        sweep = cap_sweep(split, gold_total)
        best_recall = max(sweep, key=lambda item: (item["metrics"]["recall_vs_recoverable"], item["metrics"]["candidate_reduction"])) if sweep else {}
        best_reduction_at_98 = next((item for item in sweep if item["metrics"]["recall_vs_recoverable"] >= 0.98), None)
        best_reduction_at_95 = next((item for item in sweep if item["metrics"]["recall_vs_recoverable"] >= 0.95), None)
        split_reports[name] = {
            "rows": len(split),
            "gold_total": gold_total,
            "no_compression_recoverable_oracle": recoverable,
            "single_pair_representative_oracle": single_pair,
            "detector_missing_or_unmatched_gold": max(0, gold_total - recoverable["recoverable_unique_gold"]),
            "detector_missing_or_unmatched_gold_rate": round(max(0, gold_total - recoverable["recoverable_unique_gold"]) / max(gold_total, 1), 6),
            "best_recall_policy": best_recall,
            "best_reduction_policy_at_recoverable_recall_0_98": best_reduction_at_98,
            "best_reduction_policy_at_recoverable_recall_0_95": best_reduction_at_95,
            "top_cap_sweep": sweep[:25],
        }
    return {
        "task": "IMG-MOE-V18-REBUILD-001.step_d_build_contains_symbol_policy_upper_bound_report",
        "rows": len(rows),
        "gold_total_all": all_gold_total,
        "split_reports": split_reports,
        "conclusions": build_conclusions(split_reports),
        "source_integrity": integrity(),
        "gold_loaded_after_inference_for_diagnosis_only": True,
        "gold_used_for_inference": False,
    }


def build_conclusions(split_reports: dict[str, Any]) -> list[str]:
    all_report = split_reports.get("all") or {}
    no_compression = (all_report.get("no_compression_recoverable_oracle") or {})
    pair = (all_report.get("single_pair_representative_oracle") or {})
    at_98 = all_report.get("best_reduction_policy_at_recoverable_recall_0_98")
    conclusions = [
        f"Existing bipartite edge rows recover {no_compression.get('recoverable_unique_gold')} / {no_compression.get('gold_total')} gold contains_symbol relations before compression.",
        f"Single pair representative compression can cover only {pair.get('selected_unique_gold')} unique gold keys, confirming the multi-positive pair bottleneck.",
    ]
    if at_98:
        metrics = at_98["metrics"]
        policy = at_98["policy"]
        conclusions.append(
            "A recall>=0.98 oracle under room/symbol caps exists only with "
            f"room_cap={policy['room_cap']} symbol_cap={policy['symbol_cap']}, "
            f"candidate_reduction={metrics['candidate_reduction']}."
        )
    else:
        conclusions.append("No swept room/symbol cap policy reaches recall>=0.98 against recoverable gold; candidate generation or component policy capacity must be improved.")
    return conclusions


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    limit = 10000 if args.smoke else args.limit
    rows = load_jsonl(Path(args.dataset), limit=limit)
    report = diagnose(rows)
    report["dataset"] = str(args.dataset)
    report["smoke"] = bool(args.smoke)
    write_json(Path(args.output), report)
    all_report = report["split_reports"]["all"]
    print(
        json.dumps(
            {
                "rows": len(rows),
                "gold_total": all_report["gold_total"],
                "recoverable": all_report["no_compression_recoverable_oracle"]["recoverable_unique_gold"],
                "recoverable_recall_vs_all_gold": all_report["no_compression_recoverable_oracle"]["recall_vs_all_gold"],
                "single_pair_unique_gold": all_report["single_pair_representative_oracle"]["selected_unique_gold"],
                "best_reduction_at_98": all_report["best_reduction_policy_at_recoverable_recall_0_98"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
