#!/usr/bin/env python3
"""Build leakage-free hard-case manifests for v5 expert optimization."""

from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path
from typing import Any

from v5_pipeline_utils import hard_case_features, load_json, load_jsonl, sample_id, write_json, write_jsonl


EXPERT_OUTPUTS = {
    "text": "datasets/text_dimension_expert_v8_hard_cases/manifest.jsonl",
    "boundary": "datasets/boundary_expert_v5_hard_cases/manifest.jsonl",
    "symbol": "datasets/symbol_fixture_expert_v12_hard_cases/manifest.jsonl",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/train.jsonl")
    parser.add_argument("--dev", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/dev.jsonl")
    parser.add_argument("--locked", default="datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/locked_test.jsonl")
    parser.add_argument("--ledger", default="reports/vlm/real_model_error_ledger_v5.json")
    parser.add_argument("--audit", default="reports/vlm/hard_case_leakage_audit_v5.json")
    args = parser.parse_args()

    train_rows = load_jsonl(args.train)
    dev_rows = load_jsonl(args.dev)
    locked_rows = load_jsonl(args.locked)
    train_dev_ids = {sample_id(row) for row in [*train_rows, *dev_rows] if sample_id(row)}
    locked_ids = {sample_id(row) for row in locked_rows if sample_id(row)}
    ledger = load_json(args.ledger, {})

    by_expert: dict[str, list[dict[str, Any]]] = {key: [] for key in EXPERT_OUTPUTS}
    rejected: list[dict[str, Any]] = []
    for case in ledger.get("cases") or []:
        sid = str(case.get("sample_id") or "")
        family = str(case.get("family") or "")
        expert = "boundary" if family == "boundary" else family
        if expert not in by_expert:
            continue
        if sid in locked_ids or sid not in train_dev_ids:
            rejected.append({"sample_id": sid, "node_id": case.get("node_id"), "family": family, "reason": "locked_or_not_train_dev"})
            continue
        by_expert[expert].append(
            {
                "source_image_id": sid,
                "node_id": case.get("node_id"),
                "family": family,
                "raw_label": case.get("raw_label"),
                "gold_label": inferred_gold_label(case),
                "pred_label": case.get("model_label"),
                "failure_reason": case.get("defect_type"),
                "primary_owner": case.get("primary_owner"),
                "features": hard_case_features(case),
            }
        )

    for expert, output in EXPERT_OUTPUTS.items():
        rows = by_expert[expert]
        if not rows:
            rows = [
                {
                    "decision": "no_valid_hard_cases",
                    "reason": "All visual-demo residual cases are locked/demo-only or not owned by this trainable expert. No synthetic training rows were fabricated.",
                }
            ]
        write_jsonl(output, rows)

    audit = {
        "version": "hard_case_leakage_audit_v5",
        "inputs": {"train": args.train, "dev": args.dev, "locked": args.locked, "ledger": args.ledger},
        "train_dev_count": len(train_dev_ids),
        "locked_count": len(locked_ids),
        "overlap_count": len(train_dev_ids & locked_ids),
        "locked_visual_demo_ids": sorted(locked_ids)[:20],
        "manifest_counts": {expert: len(rows) for expert, rows in by_expert.items()},
        "rejected_counts": dict(Counter(item["reason"] for item in rejected).most_common()),
        "rejected_cases": rejected[:200],
        "leakage_free": len(train_dev_ids & locked_ids) == 0 and all(item["sample_id"] not in locked_ids for item in rejected if item.get("reason") != "locked_or_not_train_dev"),
    }
    write_json(args.audit, audit)
    print(audit)


def inferred_gold_label(case: dict[str, Any]) -> str | None:
    defect = str(case.get("defect_type") or "")
    if defect == "empty_symbol" and str(case.get("raw_label") or "").lower() == "appliance":
        return "appliance"
    if defect == "missing_visible_text":
        return case.get("raw_label") or case.get("model_label")
    return None


if __name__ == "__main__":
    main()
