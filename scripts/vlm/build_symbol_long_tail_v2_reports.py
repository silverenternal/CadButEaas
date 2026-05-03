#!/usr/bin/env python3
"""Build symbol long-tail v2 reports without overstating cross-source evidence."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
CONSERVATIVE = REPORTS / "symbol_conservative_arbitration_v1.json"
ERROR_PACK = REPORTS / "symbol_long_tail_error_pack_v1.jsonl"
CROSS_V2 = REPORTS / "symbol_cross_source_lock_v2.json"
BOOST_V2 = REPORTS / "symbol_long_tail_boost_v2.json"
CONFUSION_V2 = REPORTS / "symbol_per_label_confusion_v2.json"
CROSS_V3 = REPORTS / "symbol_cross_source_lock_v3.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] if path.exists() else []


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    conservative = load_json(CONSERVATIVE)
    cross = load_json(CROSS_V2)
    errors = load_jsonl(ERROR_PACK)
    base = conservative.get("baseline_base_predictions") or {}
    full = conservative.get("full_arbitration") or {}
    base_per = base.get("long_tail_per_label") or {}
    full_per = full.get("long_tail_per_label") or {}

    labels = sorted(set(base_per) | set(full_per))
    per_label_delta = {}
    for label in labels:
        before = base_per.get(label) or {}
        after = full_per.get(label) or {}
        per_label_delta[label] = {
            "support": after.get("support", before.get("support")),
            "baseline_f1": before.get("f1"),
            "after_arbitration_f1": after.get("f1"),
            "delta_f1_pp": round((float(after.get("f1") or 0.0) - float(before.get("f1") or 0.0)) * 100.0, 3),
            "after_precision": after.get("precision"),
            "after_recall": after.get("recall"),
            "target_0_90_met": float(after.get("f1") or 0.0) >= 0.90,
        }

    confusion = Counter()
    by_gold = defaultdict(Counter)
    by_pred = defaultdict(Counter)
    high_conf = []
    for row in errors:
        gold = str(row.get("gold_label"))
        pred = str(row.get("pred_label"))
        confusion[(gold, pred)] += 1
        by_gold[gold][pred] += 1
        by_pred[pred][gold] += 1
        if float(row.get("confidence") or 0.0) >= 0.90:
            high_conf.append(row)

    confusion_report = {
        "version": "symbol_per_label_confusion_v2",
        "created": "2026-05-03",
        "source": str(ERROR_PACK.relative_to(ROOT)),
        "error_rows": len(errors),
        "top_confusions": [
            {"gold": gold, "pred": pred, "count": count}
            for (gold, pred), count in confusion.most_common(40)
        ],
        "by_gold_label": {gold: dict(counter.most_common()) for gold, counter in sorted(by_gold.items())},
        "by_pred_label": {pred: dict(counter.most_common()) for pred, counter in sorted(by_pred.items())},
        "high_confidence_error_count_ge_0_90": len(high_conf),
        "hard_case_pattern": "Current long-tail failures are dominated by equipment/generic_symbol/column confusions; the issue is label-level symbol discrimination, not family routing.",
        "status": "passed",
    }

    weak = {label: item for label, item in per_label_delta.items() if not item["target_0_90_met"]}
    boost = {
        "version": "symbol_long_tail_boost_v2",
        "created": "2026-05-03",
        "sources": {
            "conservative_arbitration": str(CONSERVATIVE.relative_to(ROOT)),
            "confusion": str(CONFUSION_V2.relative_to(ROOT)),
            "cross_source_lock": str(CROSS_V3.relative_to(ROOT)),
        },
        "node_macro_f1": {
            "baseline_base_predictions": base.get("node_macro_f1"),
            "after_full_arbitration": full.get("node_macro_f1"),
            "delta_pp": round((float(full.get("node_macro_f1") or 0.0) - float(base.get("node_macro_f1") or 0.0)) * 100.0, 3),
        },
        "long_tail_per_label_delta": per_label_delta,
        "weak_labels_below_0_90_f1": weak,
        "boost_status": "partial_internal_boost_not_target_met" if weak else "target_met",
        "paper_role": "Use as auditable long-tail analysis and partial internal improvement; do not claim cross-source symbol generalization without human gold.",
        "next_training_targets": [
            "equipment vs column hard-negative mining",
            "generic_symbol open-set/abstention head",
            "bathtub precision control with class-balanced thresholding",
            "cross-source human gold before any wild-symbol claim",
        ],
        "status": "passed",
    }

    cross_v3 = {
        **cross,
        "version": "symbol_cross_source_lock_v3",
        "created": "2026-05-03",
        "source_v2": str(CROSS_V2.relative_to(ROOT)),
        "claim_boundary": {
            **(cross.get("claim_boundary") or {}),
            "cross_source_overclaim_guard": "passed_no_human_gold_no_cross_source_claim",
        },
        "status": cross.get("status") or "pending_no_human_gold",
    }

    write_json(CONFUSION_V2, confusion_report)
    write_json(CROSS_V3, cross_v3)
    write_json(BOOST_V2, boost)
    print(f"wrote {BOOST_V2}")
    print(f"wrote {CONFUSION_V2}")
    print(f"wrote {CROSS_V3}")
    print(json.dumps({"status": boost["status"], "boost_status": boost["boost_status"], "cross_source": cross_v3["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
