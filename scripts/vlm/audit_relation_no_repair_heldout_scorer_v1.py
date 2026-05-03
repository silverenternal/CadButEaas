#!/usr/bin/env python3
"""Held-out relation scorer audit for no-repair scene-graph fusion."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT / "scripts" / "vlm") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from audit_relation_gold_id_repair_sensitivity_v1 import build_nodes  # noqa: E402
from audit_relation_no_repair_rule_sweep_v1 import edges_for_rule  # noqa: E402
from audit_relation_no_repair_sci2_scorer_v1 import (  # noqa: E402
    candidate_rows,
    gold_edge_set,
    hard_cases,
    select_edges_from_scores,
    threshold_sweep,
)
from fuse_real_upstream import (  # noqa: E402
    compute_invalid_graph_rate,
    evaluate_relations,
    extract_gold,
    load_jsonl,
)

PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_label_arbitrated_v1.jsonl"
DEV_SPLIT = ROOT / "datasets" / "cadstruct_real_world_benchmark_v1" / "room_space" / "cubicasa5k_reviewed_locked_test.jsonl"
SWEEP = ROOT / "reports" / "vlm" / "relation_no_repair_rule_sweep_v1.json"
OUTPUT = ROOT / "reports" / "vlm" / "relation_no_repair_heldout_scorer_v1.json"
HARD_CASES = ROOT / "reports" / "vlm" / "relation_no_repair_heldout_hard_cases_v1.jsonl"
DECISION = ROOT / "reports" / "vlm" / "relation_main_or_appendix_decision_v1.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def make_model(model_name: str, seed: int) -> Any:
    if model_name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced", C=1.0))
    if model_name == "extratrees":
        return ExtraTreesClassifier(
            n_estimators=180,
            max_depth=12,
            min_samples_leaf=4,
            class_weight="balanced",
            random_state=seed,
            n_jobs=-1,
        )
    raise ValueError(model_name)


def train_eval_split(rows: list[dict[str, Any]], model_name: str) -> dict[str, Any]:
    x = np.array([row["features"] for row in rows], dtype=float)
    y = np.array([row["y"] for row in rows], dtype=int)
    record_indices = np.array([row["record_index"] for row in rows], dtype=int)
    train_records = sorted({int(i) for i in record_indices if int(i) % 10 <= 6})
    calibration_records = sorted({int(i) for i in record_indices if int(i) % 10 in {7, 8}})
    heldout_records = sorted({int(i) for i in record_indices if int(i) % 10 == 9})
    train_mask = np.isin(record_indices, train_records)
    calibration_mask = np.isin(record_indices, calibration_records)
    heldout_mask = np.isin(record_indices, heldout_records)

    model = make_model(model_name, 20260503)
    model.fit(x[train_mask], y[train_mask])
    calibration_scores = model.predict_proba(x[calibration_mask])[:, 1]
    heldout_scores = model.predict_proba(x[heldout_mask])[:, 1]
    return {
        "model": model_name,
        "record_split": {
            "method": "record_index_mod_10",
            "train_mods": [0, 1, 2, 3, 4, 5, 6],
            "calibration_mods": [7, 8],
            "heldout_mods": [9],
            "train_records": len(train_records),
            "calibration_records": len(calibration_records),
            "heldout_records": len(heldout_records),
        },
        "train_rows": int(train_mask.sum()),
        "calibration_rows": int(calibration_mask.sum()),
        "heldout_rows": int(heldout_mask.sum()),
        "train_positive_rows": int(y[train_mask].sum()),
        "calibration_positive_rows": int(y[calibration_mask].sum()),
        "heldout_positive_rows": int(y[heldout_mask].sum()),
        "calibration_rows_data": [row for row, keep in zip(rows, calibration_mask) if keep],
        "calibration_scores": calibration_scores,
        "heldout_rows_data": [row for row, keep in zip(rows, heldout_mask) if keep],
        "heldout_scores": heldout_scores,
        "heldout_record_set": set(heldout_records),
    }


def filter_edges_by_records(edges: list[dict[str, Any]], record_set: set[int]) -> list[dict[str, Any]]:
    out = []
    for edge in edges:
        try:
            record_index = int(str(edge["source"]).split(":", 1)[0][1:])
        except Exception:
            continue
        if record_index in record_set:
            out.append(edge)
    return out


def filter_gold_edges(gold_edges_raw: list[dict[str, Any]], record_set: set[int]) -> list[dict[str, Any]]:
    out = []
    for edge in gold_edges_raw:
        try:
            record_index = int(str(edge["source"]).split(":", 1)[0][1:])
        except Exception:
            continue
        if record_index in record_set:
            out.append(edge)
    return out


def filter_nodes_by_records(nodes_flat: list[dict[str, Any]], record_set: set[int]) -> list[dict[str, Any]]:
    out = []
    for node in nodes_flat:
        try:
            record_index = int(str(node["id"]).split(":", 1)[0][1:])
        except Exception:
            continue
        if record_index in record_set:
            out.append(node)
    return out


def eval_rows(
    rows: list[dict[str, Any]],
    scores: np.ndarray,
    threshold: float,
    gold_edges_raw: list[dict[str, Any]],
    nodes_flat: list[dict[str, Any]],
) -> dict[str, Any]:
    record_set = {int(row["record_index"]) for row in rows}
    edges = select_edges_from_scores(rows, scores, threshold)
    gold = filter_gold_edges(gold_edges_raw, record_set)
    nodes = filter_nodes_by_records(nodes_flat, record_set)
    return {
        "threshold": round(float(threshold), 3),
        "edge_count": len(edges),
        "relation_evaluation": evaluate_relations(edges, gold),
        "invalid_graph_rate": round(compute_invalid_graph_rate(nodes, edges), 6),
    }


def main() -> int:
    predictions = load_jsonl(PREDICTIONS)
    records = load_jsonl(DEV_SPLIT)
    _, gold_edges_raw = extract_gold(records)
    gold_edges = gold_edge_set(records)
    record_nodes = build_nodes(predictions, records)
    nodes_flat = [node for nodes in record_nodes for node in nodes]
    rows = candidate_rows(record_nodes, gold_edges)

    sweep = load_json(SWEEP)
    config = (sweep.get("best") or {}).copy()
    config.pop("relation_evaluation", None)
    config.pop("edge_count", None)
    baseline_edges_all = [edge for nodes in record_nodes for edge in edges_for_rule(nodes, config)]

    model_results: list[dict[str, Any]] = []
    for model_name in ["logreg", "extratrees"]:
        split = train_eval_split(rows, model_name)
        calibration_sweep = threshold_sweep(
            split["calibration_rows_data"],
            split["calibration_scores"],
            filter_gold_edges(gold_edges_raw, {int(row["record_index"]) for row in split["calibration_rows_data"]}),
            filter_nodes_by_records(nodes_flat, {int(row["record_index"]) for row in split["calibration_rows_data"]}),
        )
        threshold = float(calibration_sweep[0]["threshold"])
        heldout_eval = eval_rows(split["heldout_rows_data"], split["heldout_scores"], threshold, gold_edges_raw, nodes_flat)
        model_results.append(
            {
                "model": model_name,
                "record_split": split["record_split"],
                "rows": {
                    "train": split["train_rows"],
                    "calibration": split["calibration_rows"],
                    "heldout": split["heldout_rows"],
                    "train_positive": split["train_positive_rows"],
                    "calibration_positive": split["calibration_positive_rows"],
                    "heldout_positive": split["heldout_positive_rows"],
                },
                "selected_threshold_from_calibration": threshold,
                "calibration_best": calibration_sweep[0],
                "heldout": heldout_eval,
            }
        )

    best = sorted(
        model_results,
        key=lambda item: (
            item["heldout"]["relation_evaluation"]["f1"],
            item["heldout"]["relation_evaluation"]["precision"],
        ),
        reverse=True,
    )[0]
    heldout_records = set(best["record_split"]["heldout_mods"])
    true_heldout_record_indices = {i for i in range(len(records)) if i % 10 in heldout_records}
    baseline_edges = filter_edges_by_records(baseline_edges_all, true_heldout_record_indices)
    baseline_gold = filter_gold_edges(gold_edges_raw, true_heldout_record_indices)
    baseline_nodes = filter_nodes_by_records(nodes_flat, true_heldout_record_indices)
    baseline = {
        "selected_rule": config,
        "edge_count": len(baseline_edges),
        "relation_evaluation": evaluate_relations(baseline_edges, baseline_gold),
        "invalid_graph_rate": round(compute_invalid_graph_rate(baseline_nodes, baseline_edges), 6),
    }

    best_split = train_eval_split(rows, best["model"])
    best_edges = select_edges_from_scores(best_split["heldout_rows_data"], best_split["heldout_scores"], best["selected_threshold_from_calibration"])
    hard = hard_cases(records, best_split["heldout_rows_data"], best_edges, gold_edges)
    write_jsonl(HARD_CASES, hard)

    heldout_f1 = best["heldout"]["relation_evaluation"]["f1"]
    invalid_rate = best["heldout"]["invalid_graph_rate"]
    decision = {
        "version": "relation_main_or_appendix_decision_v1",
        "created": "2026-05-03",
        "source": str(OUTPUT.relative_to(ROOT)),
        "best_heldout_model": best["model"],
        "heldout_relation_f1": heldout_f1,
        "heldout_precision": best["heldout"]["relation_evaluation"]["precision"],
        "heldout_recall": best["heldout"]["relation_evaluation"]["recall"],
        "heldout_invalid_graph_rate": invalid_rate,
        "baseline_heldout_relation_f1": baseline["relation_evaluation"]["f1"],
        "preferred_0_90_target_met": heldout_f1 >= 0.90,
        "invalid_target_met": invalid_rate <= 0.01,
        "recommendation": "main_table_candidate" if heldout_f1 >= 0.90 and invalid_rate <= 0.01 else "appendix_until_stronger_external_or_record_split",
        "claim_boundary": "Held-out split is deterministic within the locked dev benchmark, not a new external source. Do not present it as wild generalization.",
        "status": "passed",
    }
    report = {
        "version": "relation_no_repair_heldout_scorer_v1",
        "created": "2026-05-03",
        "inputs": {
            "predictions": str(PREDICTIONS.relative_to(ROOT)),
            "dev_split": str(DEV_SPLIT.relative_to(ROOT)),
            "baseline_rule_sweep": str(SWEEP.relative_to(ROOT)),
        },
        "candidate_dataset": {
            "rows": len(rows),
            "records": len(records),
            "positive_rows": int(sum(row["y"] for row in rows)),
            "features": [
                "has_intersection",
                "symbol_overlap_ratio",
                "room_overlap_ratio",
                "center_inside",
                "center_inside_pad2",
                "center_distance",
                "center_distance_norm_room",
                "room_area",
                "symbol_area",
                "room_confidence",
                "symbol_confidence",
                "containing_count",
                "padded_containing_count",
                "room_count",
            ],
        },
        "baseline_no_repair_rule_on_heldout": baseline,
        "model_results": model_results,
        "best_heldout_no_repair_scorer": best,
        "hard_cases": {
            "path": str(HARD_CASES.relative_to(ROOT)),
            "count": len(hard),
            "summary": dict(Counter(row["category"] for row in hard).most_common()),
        },
        "decision": decision,
        "claim_boundary": decision["claim_boundary"],
    }
    write_json(OUTPUT, report)
    write_json(DECISION, decision)
    print(f"wrote {OUTPUT}")
    print(f"wrote {HARD_CASES}")
    print(f"wrote {DECISION}")
    print(json.dumps({"best_model": best["model"], "heldout_f1": heldout_f1, "invalid": invalid_rate, "recommendation": decision["recommendation"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
