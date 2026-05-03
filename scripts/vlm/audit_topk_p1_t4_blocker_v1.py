#!/usr/bin/env python3
"""P1-T4 blocker audit for top-k gated fusion.

This report records why the current top-k/family-gating task cannot be marked
complete: the family router is not the limiting factor, and the dominant node
errors are label-level/domain-adaptation failures inside selected experts.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
OUTPUT = REPORTS / "topk_p1_t4_blocker_audit_v1.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def main() -> None:
    fusion = load_json(REPORTS / "scene_graph_fusion_real_upstream_eval.json")
    topk = load_json(REPORTS / "topk_gated_fusion_v1_eval.json")
    router = load_json(REPORTS / "moe_router_v3_fair_ablation.json")
    predictions = load_jsonl(REPORTS / "real_upstream_predictions_dev.jsonl")

    per_label = fusion["node_evaluation"]["per_label"]
    zero_or_weak = {
        label: metrics
        for label, metrics in per_label.items()
        if float(metrics.get("f1", 0.0)) < 0.10 and int(metrics.get("support", 0)) > 0
    }
    boundary_predictions = Counter(
        str(pred.get("label"))
        for pred in predictions
        if str(pred.get("family")) == "boundary"
    )
    family_error_groups: dict[str, list[str]] = defaultdict(list)
    for label in zero_or_weak:
        if label in {"door", "window", "hard_wall", "opening", "partition_wall"}:
            family_error_groups["boundary"].append(label)
        elif label in {"shower", "sink", "bathtub", "stair", "column", "equipment", "appliance", "generic_symbol"}:
            family_error_groups["symbol"].append(label)
        elif label in {"dimension_line", "dimension_text", "leader_line", "note_text", "room_label"}:
            family_error_groups["text"].append(label)
        else:
            family_error_groups["other"].append(label)

    report = {
        "version": "topk_p1_t4_blocker_audit_v1",
        "created": "2026-05-03",
        "baseline": topk["baseline"],
        "topk_done_when_check": topk["done_when_check"],
        "topk_status": topk["status"],
        "router_capacity": {
            "deterministic_wrong_expert_rate": router["models"]["deterministic_router"]["wrong_expert_rate"],
            "learned_fair_wrong_expert_rate": router["models"]["learned_fair_router_v3"]["wrong_expert_rate"],
            "top2_oracle_in_k_rate": router["models"]["top2_confidence_router"]["oracle_in_top2_rate"],
            "top3_oracle_in_k_rate": router["models"]["top3_confidence_router"]["oracle_in_top3_rate"],
        },
        "dominant_node_blockers": {
            "weak_labels_f1_lt_0_10": zero_or_weak,
            "weak_labels_by_family": dict(family_error_groups),
            "boundary_prediction_distribution": dict(boundary_predictions),
        },
        "wall_opening_domain_shift_probe": {
            "observation": "current real-upstream boundary predictions are all hard_wall after rerunning WallOpening record-by-record",
            "evidence": dict(boundary_predictions),
            "source_feature_issue": (
                "The h1024 floor-target checkpoint was trained with source_floorplancad as a dominant source feature. "
                "Real-upstream CubiCasa candidates do not match that source-feature distribution, and a quick source-one-hot probe "
                "showed outputs flip from all hard_wall to mostly door rather than calibrated door/window/hard_wall predictions."
            ),
        },
        "decision": {
            "delete_p1_t4_from_todo": False,
            "reason": "P1-T4 requires node macro F1 gain >=3pp; current family-level top-k gives 0.0pp and label-level arbitration is not implemented.",
            "required_next_step": "Train/evaluate a leakage-free label-level boundary/symbol arbitration model or a CubiCasa-compatible WallOpening checkpoint, then rerun top-k fusion.",
        },
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(json.dumps(report["decision"], ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
