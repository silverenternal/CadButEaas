#!/usr/bin/env python3
"""Summarize existing FloorPlanCAD residual candidates under the P4-T2 contract."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", default="checkpoints/wall_opening_floorplancad_residual_v1")
    parser.add_argument("--report", default="reports/vlm/wall_opening_floorplancad_residual_v1_eval.json")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    baseline = load_json(Path("reports/vlm/paper_v2_floor_target_h768doorw150_residual_fine_maxf1_blend_audit.json"))
    seed32_router = load_json(Path("reports/vlm/paper_v2_floor_target_seed32_vs_current_prediction_router_audit.json"))
    h768_router = load_json(Path("reports/vlm/paper_v2_floor_target_h768_main_vs_seed31_prediction_router_audit.json"))
    dual = load_json(Path("reports/vlm/paper_v2_floor_target_residual_dual_branch_ensemble_audit.json"))
    locked_comparison = compare_locked_predictions(
        baseline_path=Path("reports/vlm/paper_v2_h512_two_stage_router_locked_test_predictions.jsonl"),
        candidate_path=Path("reports/vlm/paper_v2_routed_h512_main_floor_aug_final_locked_test_predictions.jsonl"),
    )

    candidates = [
        candidate("fine_maxf1_blend_baseline", baseline.get("selected_dev_metrics"), baseline.get("selected_smoke_metrics"), baseline),
        candidate("seed32_vs_current_prediction_router", seed32_router.get("best_dev_metrics"), seed32_router.get("best_smoke_metrics"), seed32_router),
        candidate("h768_main_vs_seed31_prediction_router", h768_router.get("best_dev_metrics"), h768_router.get("best_smoke_metrics"), h768_router),
        candidate("dual_branch_probability_ensemble", dual.get("selected_dev_metrics"), dual.get("selected_smoke_metrics"), dual),
    ]
    candidates = [row for row in candidates if row["dev_macro_f1"] is not None or row["smoke_macro_f1"] is not None]
    selected = max(candidates, key=lambda row: (row["smoke_macro_f1"] or -1.0, row["dev_macro_f1"] or -1.0))
    base = candidates[0]

    eval_report = {
        "version": "wall_opening_floorplancad_residual_v1_eval",
        "protocol": "Consolidate existing source-residual/router candidates; do not alter the frozen WallOpeningExpert.",
        "target": {"floorplancad_macro_f1": 0.98, "cvc_fp_drop_max": 0.002},
        "baseline": base,
        "selected_candidate": selected,
        "selected_locked_by_source_candidate": locked_comparison,
        "all_candidates": candidates,
        "acceptance": {
            "floorplancad_smoke_improved": (selected["smoke_macro_f1"] or 0.0) > (base["smoke_macro_f1"] or 0.0),
            "floorplancad_smoke_reaches_0_98": (selected["smoke_macro_f1"] or 0.0) >= 0.98,
            "floorplancad_locked_improved": locked_comparison["floorplancad_delta"] > 0.0,
            "cvc_fp_no_drop_verified": locked_comparison["cvc_fp_drop"] <= 0.002,
            "cvc_fp_drop": locked_comparison["cvc_fp_drop"],
            "status": "passed" if locked_comparison["floorplancad_delta"] > 0.0 and locked_comparison["cvc_fp_drop"] <= 0.002 else "blocked_on_locked_by_source_audit",
        },
        "next_action": "Keep this as a scoped FloorPlanCAD residual/router improvement; do not claim FloorPlanCAD has reached 0.98 until a stronger residual branch is trained.",
    }
    train_summary = {
        "version": "wall_opening_floorplancad_residual_v1_train_summary",
        "status": eval_report["acceptance"]["status"],
        "checkpoint_dir": str(output_dir),
        "selected_existing_candidate": selected,
        "selected_locked_by_source_candidate": locked_comparison,
        "memory_audit": {"new_training_started": False, "oom_events": 0},
        "note": "This P4-T2 attempt reuses existing residual/router outputs and is scoped to the todo acceptance: FloorPlanCAD F1 improves while CVC-FP drop stays under 0.2%.",
    }
    (output_dir / "train_summary.json").write_text(json.dumps(train_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text(json.dumps(eval_report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(eval_report, ensure_ascii=False, indent=2))


def candidate(name: str, dev: dict[str, Any] | None, smoke: dict[str, Any] | None, raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": name,
        "dev_macro_f1": metric(dev, "macro_f1"),
        "dev_accuracy": metric(dev, "accuracy"),
        "dev_probability_r2": metric(dev, "probability_r2"),
        "smoke_macro_f1": metric(smoke, "macro_f1"),
        "smoke_accuracy": metric(smoke, "accuracy"),
        "smoke_probability_r2": metric(smoke, "probability_r2"),
        "source": source_hint(raw),
    }


def metric(payload: dict[str, Any] | None, key: str) -> float | None:
    if not isinstance(payload, dict) or payload.get(key) is None:
        return None
    return float(payload[key])


def source_hint(payload: dict[str, Any]) -> dict[str, Any]:
    return {
        key: payload.get(key)
        for key in ("dataset_dir", "base_dev", "alt_dev", "base_smoke", "alt_smoke", "selected_blend", "best_rule", "selection_protocol")
        if key in payload
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def compare_locked_predictions(baseline_path: Path, candidate_path: Path) -> dict[str, Any]:
    baseline = by_source_macro_f1(baseline_path)
    candidate = by_source_macro_f1(candidate_path)
    floor_delta = candidate.get("floorplancad", 0.0) - baseline.get("floorplancad", 0.0)
    cvc_drop = baseline.get("cvc_fp", 0.0) - candidate.get("cvc_fp", 0.0)
    return {
        "baseline": str(baseline_path),
        "candidate": str(candidate_path),
        "baseline_by_source_macro_f1": baseline,
        "candidate_by_source_macro_f1": candidate,
        "floorplancad_delta": floor_delta,
        "cvc_fp_drop": cvc_drop,
        "done_when": {
            "floorplancad_f1_improves": floor_delta > 0.0,
            "cvc_fp_f1_drop_lte_0_002": cvc_drop <= 0.002,
        },
    }


def by_source_macro_f1(path: Path) -> dict[str, float]:
    pairs_by_source: dict[str, list[tuple[str, str]]] = {}
    if not path.exists():
        return {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            source = str(row.get("source_dataset") or "unknown")
            pairs = pairs_by_source.setdefault(source, [])
            for node in row.get("nodes") or []:
                gold = node.get("label")
                pred = node.get("prediction")
                if gold is None or pred is None:
                    continue
                pairs.append((str(gold), str(pred)))
    return {source: macro_f1(pairs) for source, pairs in pairs_by_source.items()}


def macro_f1(pairs: list[tuple[str, str]]) -> float:
    labels = sorted({label for pair in pairs for label in pair})
    values = []
    for label in labels:
        tp = sum(1 for gold, pred in pairs if gold == label and pred == label)
        fp = sum(1 for gold, pred in pairs if gold != label and pred == label)
        fn = sum(1 for gold, pred in pairs if gold == label and pred != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        values.append(2 * precision * recall / max(precision + recall, 1e-12))
    return sum(values) / max(len(values), 1)


if __name__ == "__main__":
    main()
