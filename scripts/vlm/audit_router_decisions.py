#!/usr/bin/env python3
"""Audit router decision traceability on dev split (S7-T1).

Runs the enhanced DeterministicRouter on all 493 dev records and verifies:
- Every routed candidate has matched_hint, routing_confidence, and abstain
- Reports abstain rate, confidence distribution, and hint match patterns

Done when: all dev candidates have route traces with matched_hint + confidence + abstain.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
DEV_SPLIT = ROOT / "datasets/cadstruct_real_world_benchmark_v1/room_space/cubicasa5k_reviewed_locked_test.jsonl"
OUTPUT = ROOT / "reports/vlm/router_decision_trace_audit.json"

sys.path.insert(0, str(ROOT / "scripts/vlm"))
from cadstruct_moe.router import DeterministicRouter


def main() -> None:
    print("=== Router Decision Trace Audit (S7-T1) ===\n")

    router = DeterministicRouter(ROOT / "configs/vlm/cadstruct_ontology.json")
    dev_records = load_jsonl(DEV_SPLIT)
    print(f"Dev split: {len(dev_records)} records")

    all_routed = []
    abstain_count = 0
    has_trace_count = 0
    missing_trace_count = 0
    hint_matches = Counter()
    family_counts = Counter()
    confidence_buckets = {"high": 0, "medium": 0, "low": 0}
    records_with_abstain = 0

    for record in dev_records:
        routed = router.route_record(record)
        all_routed.extend(routed)

        record_has_abstain = False
        for candidate in routed:
            trace = candidate.route_trace

            if trace:
                has_trace_count += 1
                if trace.get("matched_hint"):
                    hint_matches[trace["matched_hint"]] += 1
                if trace.get("abstain"):
                    abstain_count += 1
                    record_has_abstain = True

                # Confidence bucket
                conf = trace.get("routing_confidence", candidate.confidence)
                if conf >= 0.8:
                    confidence_buckets["high"] += 1
                elif conf >= 0.5:
                    confidence_buckets["medium"] += 1
                else:
                    confidence_buckets["low"] += 1
            else:
                missing_trace_count += 1

            family_counts[candidate.family] += 1

        if record_has_abstain:
            records_with_abstain += 1

    total = len(all_routed)
    print(f"Total routed candidates: {total}")
    print(f"Has route trace: {has_trace_count} ({has_trace_count/total*100:.1f}%)")
    print(f"Missing route trace: {missing_trace_count}")
    print(f"Abstain: {abstain_count} ({abstain_count/total*100:.1f}%)")
    print(f"Records with abstain: {records_with_abstain}/{len(dev_records)}")

    print(f"\nFamily distribution:")
    for family, count in family_counts.most_common():
        print(f"  {family}: {count} ({count/total*100:.1f}%)")

    print(f"\nHint match patterns:")
    for hint, count in hint_matches.most_common(15):
        print(f"  {hint}: {count}")

    print(f"\nConfidence distribution:")
    for bucket, count in confidence_buckets.items():
        print(f"  {bucket} (≥0.8/≥0.5/<0.5): {count} ({count/total*100:.1f}%)")

    # Per-family trace coverage
    per_family = {}
    for family in family_counts:
        family_candidates = [c for c in all_routed if c.family == family]
        family_with_trace = sum(1 for c in family_candidates if c.route_trace)
        family_abstain = sum(1 for c in family_candidates if c.route_trace and c.route_trace.get("abstain"))
        per_family[family] = {
            "total": len(family_candidates),
            "has_trace": family_with_trace,
            "trace_coverage": round(family_with_trace / len(family_candidates), 4),
            "abstain": family_abstain,
            "abstain_rate": round(family_abstain / len(family_candidates), 4),
        }

    # Save report
    report = {
        "version": "router_decision_trace_audit_v1",
        "dev_split": str(DEV_SPLIT),
        "dev_records": len(dev_records),
        "total_routed": total,
        "trace_coverage": {
            "has_trace": has_trace_count,
            "missing_trace": missing_trace_count,
            "coverage_rate": round(has_trace_count / total, 4) if total else 0,
        },
        "abstain_summary": {
            "abstain_count": abstain_count,
            "abstain_rate": round(abstain_count / total, 4) if total else 0,
            "records_with_abstain": records_with_abstain,
        },
        "family_distribution": dict(family_counts),
        "per_family": per_family,
        "hint_matches": dict(hint_matches.most_common(20)),
        "confidence_buckets": confidence_buckets,
        "done_when_check": {
            "all_candidates_have_trace": missing_trace_count == 0,
            "all_traces_have_matched_hint": all(
                c.route_trace.get("matched_hint") is not None or c.route_trace.get("abstain")
                for c in all_routed if c.route_trace
            ),
            "all_traces_have_confidence": all(
                "routing_confidence" in c.route_trace
                for c in all_routed if c.route_trace
            ),
            "all_traces_have_abstain": all(
                "abstain" in c.route_trace
                for c in all_routed if c.route_trace
            ),
            "effective_rate": round(1.0 - (abstain_count / total), 4) if total else 0,
        },
    }

    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport saved to {OUTPUT}")

    # Done-when check
    print("\n=== Done-when check ===")
    print(f"All candidates have trace: {'PASS' if missing_trace_count == 0 else 'FAIL'}")
    print(f"All traces have matched_hint: {'PASS' if report['done_when_check']['all_traces_have_matched_hint'] else 'FAIL'}")
    print(f"All traces have confidence: {'PASS' if report['done_when_check']['all_traces_have_confidence'] else 'FAIL'}")
    print(f"All traces have abstain: {'PASS' if report['done_when_check']['all_traces_have_abstain'] else 'FAIL'}")
    print(f"Router effective rate: {report['done_when_check']['effective_rate']} (target ≥ 0.95) "
          f"{'PASS' if report['done_when_check']['effective_rate'] >= 0.95 else 'FAIL'}")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
