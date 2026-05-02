#!/usr/bin/env python3
"""Audit MoE router routing balance, abstain/fallback policy, and expert failure risk."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from cadstruct_moe.router import BOUNDARY_HINTS, SPACE_HINTS, SYMBOL_HINTS, TEXT_HINTS, SHEET_HINTS
    from cadstruct_moe.router import DeterministicRouter
    from cadstruct_moe.schema import label_to_family, load_ontology
except ImportError:
    from scripts.vlm.cadstruct_moe.router import (
        BOUNDARY_HINTS,
        SPACE_HINTS,
        SYMBOL_HINTS,
        TEXT_HINTS,
        SHEET_HINTS,
        DeterministicRouter,
    )
    from scripts.vlm.cadstruct_moe.schema import label_to_family, load_ontology


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="datasets/cadstruct_cubicasa5k_moe_locked/smoke.jsonl")
    parser.add_argument("--output", default="reports/vlm/moe_router_balance_audit.json")
    parser.add_argument("--low-confidence", type=float, default=0.6)
    parser.add_argument("--disable-floor", action="store_true", help="keep script compatible; no behavior change")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    report = audit_router(rows, low_confidence=args.low_confidence)
    report["input"] = args.input
    report["low_confidence_threshold"] = args.low_confidence
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def audit_router(rows: list[dict[str, Any]], low_confidence: float) -> dict[str, Any]:
    ontology = load_ontology()
    ontology_labels = set(label_to_family(ontology).keys())
    family_hints = {str(item) for item in (BOUNDARY_HINTS | TEXT_HINTS | SYMBOL_HINTS | SHEET_HINTS | SPACE_HINTS)}

    router = DeterministicRouter()
    routed_stats: Counter[tuple[str, str, str]] = Counter()
    confidence_stats: Counter[tuple[str, str, str]] = Counter()
    route_balance_source_expert: Counter[tuple[str, str]] = Counter()
    route_balance_source_family: Counter[tuple[str, str]] = Counter()
    abstain_stats: Counter[str] = Counter()
    source_stats: dict[str, Counter[str]] = defaultdict(Counter)
    source_entropy_parts: dict[str, list[float]] = defaultdict(list)
    total_rows = 0
    total_candidates = 0
    routed_candidates = 0

    disable_targets = ["wall_opening", "room_space", "symbol_fixture", "text_dimension", "sheet_layout"]
    disable_stats: dict[str, dict[str, int]] = {
        expert: {
            "rows": 0,
            "rows_emit_with_expert_disabled": 0,
            "rows_with_remaining_non_abstain_if_disabled": 0,
            "rows_with_any_input_if_disabled": 0,
            "rows_empty_if_disabled": 0,
            "candidate_count_after_disable": 0,
            "disabled_candidate_count": 0,
        }
        for expert in disable_targets
    }

    for row in rows:
        total_rows += 1
        source = str(row.get("source_dataset") or "unknown")
        routed = [item.to_dict() for item in router.route_record(row)]
        if not routed:
            source_stats[source]["rows_without_route"] += 1
            continue

        record_countable = 0
        by_expert = Counter()
        by_expert_non_abstain = Counter()
        source_stats[source]["records_with_routes"] += 1

        routed_rows = []
        for item in routed:
            candidate_id = str(item.get("candidate_id") or "")
            expert = str(item.get("expert") or "unknown")
            family = str(item.get("family") or "unknown")
            conf = _safe_float(item.get("confidence"), 1.0)
            candidate_type = str(item.get("candidate_type") or "")

            reason = classify_route(candidate_type, family, ontology_labels, family_hints)
            abstain_reason = None
            if conf < low_confidence:
                abstain_reason = "low_confidence"
            if reason == "out_of_taxonomy":
                abstain_reason = "out_of_taxonomy"

            record_countable += 1
            total_candidates += 1
            source_stats[source]["candidates"] += 1
            routed_stats[(source, expert, family)] += 1
            confidence_stats[(source, expert, confidence_bucket(conf))] += 1
            route_balance_source_expert[(source, expert)] += 1
            route_balance_source_family[(source, family)] += 1
            by_expert[expert] += 1
            if abstain_reason is None:
                by_expert_non_abstain[expert] += 1
                routed_candidates += 1
                source_stats[source]["effective_candidates"] += 1
            else:
                abstain_stats[f"{reason}:{family}:{abstain_reason}"] += 1
                source_stats[source][f"abstain_{abstain_reason}"] += 1

            routed_rows.append((expert, abstain_reason))

        entropy = entropy_from_counts({**by_expert, "abstain": sum(1 for _, abstain in routed_rows if abstain is not None)})
        source_entropy_parts[source].append(entropy)

        if record_countable:
            routed_count = len(routed_rows)
            for expert in disable_targets:
                disable_stats[expert]["rows"] += 1
                candidates_without_expert = [item for item in routed_rows if item[0] != expert and item[1] is None]
                candidates_with_expert_disabled = [item for item in routed_rows if item[0] == expert and item[1] is None]
                disable_stats[expert]["disabled_candidate_count"] += len(candidates_with_expert_disabled)
                disable_stats[expert]["candidate_count_after_disable"] += len(candidates_without_expert)
                disable_stats[expert]["rows_with_any_input_if_disabled"] += int(len(candidates_without_expert) > 0)
                if len(candidates_without_expert) > 0:
                    disable_stats[expert]["rows_emit_with_expert_disabled"] += 1
                    disable_stats[expert]["rows_with_remaining_non_abstain_if_disabled"] += 1
                else:
                    disable_stats[expert]["rows_empty_if_disabled"] += 1

    route_entropy = (
        sum(value for row_stats in source_entropy_parts.values() for value in row_stats) / max(sum(len(v) for v in source_entropy_parts.values()), 1)
    )
    confidence_distribution = {
        f"{source}|{expert}|{bucket}": count
        for (source, expert, bucket), count in sorted(confidence_stats.items())
    }
    family_distribution = {
        f"{source}|{family}": count
        for (source, family), count in sorted(route_balance_source_family.items())
    }
    expert_distribution = {
        f"{source}|{expert}": count
        for (source, expert), count in sorted(route_balance_source_expert.items())
    }
    source_entropy = {
        source: {
            "mean_entropy": round(sum(values) / len(values), 6),
            "rows": len(values),
        }
        for source, values in source_entropy_parts.items()
    }
    return {
        "records": total_rows,
        "routed_records": sum(stat for stat in [v["records_with_routes"] for v in source_stats.values()]),
        "candidates": total_candidates,
        "effective_candidates": routed_candidates,
        "effective_rate": round(routed_candidates / max(total_candidates, 1), 6),
        "route_balance": {
            "by_source_expert": expert_distribution,
            "by_source_family": family_distribution,
            "by_candidate": {f"{source}|{expert}|{family}": count for (source, expert, family), count in sorted(routed_stats.items())},
        },
        "confidence_distribution": confidence_distribution,
        "abstention": {
            "counts": dict(sorted(abstain_stats.items())),
            "rate": round(sum(abstain_stats.values()) / max(total_candidates, 1), 6),
        },
        "source_metrics": {
            source: {
                "rows": int(stats.get("records_with_routes", 0)),
                "rows_without_route": int(stats.get("rows_without_route", 0)),
                "candidates": int(stats.get("candidates", 0)),
                "effective_candidates": int(stats.get("effective_candidates", 0)),
                "effective_rate": round(int(stats.get("effective_candidates", 0)) / max(int(stats.get("candidates", 1)), 1), 6),
                "abstain_low_confidence": int(stats.get("abstain_low_confidence", 0)),
                "abstain_out_of_taxonomy": int(stats.get("abstain_out_of_taxonomy", 0)),
                "entropy": source_entropy.get(source, {}).get("mean_entropy", 0.0),
                "rows": int(stats.get("records_with_routes", 0)),
            }
            for source, stats in sorted(source_stats.items())
        },
        "route_entropy": {
            "global": round(route_entropy, 6),
            "per_source": {source: round(values, 6) for source, values in {k: sum(v) / max(len(v), 1) for k, v in source_entropy_parts.items()}.items()},
        },
        "single_expert_disable": {
            expert: {
                "rows": int(stats["rows"]),
                "rows_emit_if_disabled": int(stats["rows_emit_with_expert_disabled"]),
                "rows_with_non_abstain_if_disabled": int(stats["rows_with_remaining_non_abstain_if_disabled"]),
                "rows_any_input_if_disabled": int(stats["rows_with_any_input_if_disabled"]),
                "rows_empty_if_disabled": int(stats["rows_empty_if_disabled"]),
                "disabled_candidate_count": int(stats["disabled_candidate_count"]),
                "candidate_count_after_disable": int(stats["candidate_count_after_disable"]),
                "coverage_ratio": round(
                    int(stats["rows_emit_with_expert_disabled"]) / max(int(stats["rows"]), 1), 6
                ),
                "fallback_ratio": round(
                    int(stats["rows_with_any_input_if_disabled"]) / max(int(stats["rows"]), 1), 6
                ),
                "disabled_ratio": round(
                    int(stats["disabled_candidate_count"]) / max(int(stats["rows_emit_with_expert_disabled"]) + int(stats["disabled_candidate_count"]), 1), 6
                ),
            }
            for expert, stats in sorted(disable_stats.items())
        },
        "status": "ok",
    }


def classify_route(
    candidate_type: str,
    family: str,
    ontology_labels: set[str],
    family_hints: set[str],
) -> str:
    token = candidate_type.replace("-", "_").replace(" ", "_").strip().lower()
    if not token:
        return "unknown_type"
    if token in ontology_labels:
        return "in_taxonomy"
    if token in family_hints:
        return "heuristic_keyword"
    if family in {"boundary", "space", "symbol", "text", "sheet"}:
        return "fallback_to_family"
    return "out_of_taxonomy"


def confidence_bucket(value: float) -> str:
    if value < 0.2:
        return "[0.0,0.2)"
    if value < 0.4:
        return "[0.2,0.4)"
    if value < 0.6:
        return "[0.4,0.6)"
    if value < 0.8:
        return "[0.6,0.8)"
    return "[0.8,1.0]"


def entropy_from_counts(counts: dict[str, int] | Counter[str]) -> float:
    total = sum(counts.values())
    if total <= 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        p = count / total
        if p <= 0:
            continue
        entropy -= p * math.log2(p)
    return entropy


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def _safe_float(value: Any, default: float = 1.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return max(0.0, min(1.0, number))


if __name__ == "__main__":
    main()
