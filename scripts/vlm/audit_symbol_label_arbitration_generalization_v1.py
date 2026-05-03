#!/usr/bin/env python3
"""Audit SymbolFixture label arbitration generalization and feature sensitivity."""

from __future__ import annotations

import json
import random
import sys
import warnings
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier

warnings.filterwarnings(
    "ignore",
    message="`sklearn.utils.parallel.delayed` should be used",
    category=UserWarning,
)

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT / "scripts" / "vlm") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from fuse_real_upstream import (  # noqa: E402
    compute_invalid_graph_rate,
    evaluate_nodes,
    evaluate_relations,
    extract_gold,
    fuse_predictions_with_gold_id_space,
    load_jsonl,
)
from train_symbol_label_arbitration_v1 import (  # noqa: E402
    BASE_FUSION,
    BASE_PREDICTIONS,
    LABELS,
    LOCKED_SPLIT,
    MAX_TRAIN_PER_LABEL,
    MIN_TRAIN_PER_LABEL,
    TRAIN_SPLITS,
    apply_arbitration,
    extract_items,
    locked_symbol_lookup,
    metrics,
    split_images,
)

REPORT = ROOT / "reports" / "vlm" / "symbol_label_arbitration_generalization_v1.json"
MAIN_SYMBOL_REPORT = ROOT / "reports" / "vlm" / "symbol_label_arbitration_v1_eval.json"

FEATURE_SETS = {
    "geometry_only": list(range(0, 6)),
    "room_context_only": list(range(6, 10)),
    "neighbor_stats_only": list(range(10, 13)),
    "full_arbitration": list(range(0, 13)),
}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def feature_view(items: list[dict[str, Any]], cols: list[int]) -> np.ndarray:
    return np.array([[float(item["features"][idx]) for idx in cols] for item in items], dtype=np.float64)


def stratified(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rng = random.Random(20260503)
    by_label: dict[str, list[dict[str, Any]]] = {label: [] for label in LABELS}
    for item in items:
        label = str(item.get("label"))
        if label in by_label:
            by_label[label].append(item)
    selected: list[dict[str, Any]] = []
    for label in LABELS:
        rows = list(by_label[label])
        if len(rows) > max(MAX_TRAIN_PER_LABEL, MIN_TRAIN_PER_LABEL):
            rng.shuffle(rows)
            rows = rows[:MAX_TRAIN_PER_LABEL]
        selected.extend(rows)
    rng.shuffle(selected)
    return selected


def train_model(x_train: np.ndarray, y_train: list[str]) -> ExtraTreesClassifier:
    model = ExtraTreesClassifier(
        n_estimators=80,
        max_depth=18,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=20260503,
        n_jobs=1,
    )
    model.fit(x_train, y_train)
    return model


def eval_e2e(model: ExtraTreesClassifier, locked_items: list[dict[str, Any]], base_predictions: list[dict[str, Any]], locked_rows: list[dict[str, Any]]) -> dict[str, Any]:
    lookup = locked_symbol_lookup(locked_items, model)
    adjusted, application = apply_arbitration(base_predictions, locked_rows, lookup)
    gold_nodes, gold_edges = extract_gold(locked_rows)
    fused_nodes, fused_edges = fuse_predictions_with_gold_id_space(adjusted, locked_rows)
    return {
        "application": application,
        "node_evaluation": evaluate_nodes(fused_nodes, gold_nodes),
        "relation_evaluation": evaluate_relations(fused_edges, gold_edges),
        "invalid_graph_rate": round(compute_invalid_graph_rate(fused_nodes, fused_edges), 6),
        "fused": {"nodes": len(fused_nodes), "edges": len(fused_edges)},
        "gold": {"nodes": len(gold_nodes), "edges": len(gold_edges)},
    }


def per_label_delta(base: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    base_labels = (base.get("node_evaluation") or {}).get("per_label") or {}
    cur_labels = (current.get("node_evaluation") or {}).get("per_label") or {}
    watched = ["sink", "shower", "equipment", "appliance", "bathtub", "generic_symbol", "column", "stair", "table"]
    out: dict[str, Any] = {}
    for label in watched:
        b = float((base_labels.get(label) or {}).get("f1") or 0.0)
        c = float((cur_labels.get(label) or {}).get("f1") or 0.0)
        out[label] = {
            "baseline_f1": round(b, 6),
            "full_arbitration_f1": round(c, 6),
            "delta_pp": round((c - b) * 100.0, 6),
            "support": (cur_labels.get(label) or {}).get("support", 0),
        }
    return out


def cross_source_status() -> dict[str, Any]:
    candidates = [
        ROOT / "datasets" / "cadstruct_real_world_benchmark_v1" / "symbol_fixture" / "cubicasa5k_symbol_smoke_locked.jsonl",
        ROOT / "datasets" / "symbol_fixture_detector_v2" / "locked.jsonl",
        ROOT / "datasets" / "internal_hard_cases_round_2" / "symbol_fixture_candidates.jsonl",
    ]
    available = [path for path in candidates if path.exists()]
    return {
        "status": "dataset_not_available",
        "checked_files": [str(path.relative_to(ROOT)) for path in available],
        "reason": "Available symbol files are CubiCasa-derived or candidate-only and do not provide a non-CubiCasa scene-graph-compatible locked split with gold 9-class SymbolFixture labels.",
        "minimum_needed": {
            "samples": ">=50 drawings or >=1000 symbol candidates",
            "fields": ["image/source id", "symbol bbox", "9-class gold symbol_type", "optional room bbox/type for room-context features"],
            "split_policy": "source-held-out from CubiCasa train/dev/locked images",
        },
    }


def main() -> int:
    print("loading splits", flush=True)
    train_rows: list[dict[str, Any]] = []
    for path in TRAIN_SPLITS:
        train_rows.extend(load_jsonl(path))
    locked_rows = load_jsonl(LOCKED_SPLIT)

    train_images = split_images(train_rows)
    locked_images = split_images(locked_rows)
    overlap = sorted(train_images & locked_images)

    print("extracting items", flush=True)
    train_items = stratified(extract_items(train_rows))
    locked_items_full = extract_items(locked_rows)
    y_train = [str(item["label"]) for item in train_items]
    y_locked = [str(item["label"]) for item in locked_items_full]
    base_predictions = load_jsonl(BASE_PREDICTIONS)
    baseline = load_json(BASE_FUSION)

    ablations: dict[str, Any] = {}
    for name, cols in FEATURE_SETS.items():
        print(f"training {name}", flush=True)
        locked_subset = [{**item, "features": [item["features"][idx] for idx in cols]} for item in locked_items_full]
        model = train_model(feature_view(train_items, cols), y_train)
        pred = list(model.predict(feature_view(locked_items_full, cols)))
        symbol_metrics = metrics(y_locked, pred)
        e2e = eval_e2e(model, locked_subset, base_predictions, locked_rows)
        ablations[name] = {
            "feature_columns": cols,
            "locked_symbol_metrics": symbol_metrics,
            "e2e": {
                "node_macro_f1": e2e["node_evaluation"]["macro_f1"],
                "relation_precision": e2e["relation_evaluation"]["precision"],
                "relation_recall": e2e["relation_evaluation"]["recall"],
                "relation_f1": e2e["relation_evaluation"]["f1"],
                "invalid_graph_rate": e2e["invalid_graph_rate"],
            },
            "application": e2e["application"],
            "per_label_e2e_delta_vs_baseline": per_label_delta(baseline, e2e),
        }

    full = ablations["full_arbitration"]["e2e"]
    report = {
        "version": "symbol_label_arbitration_generalization_v1",
        "created": "2026-05-03",
        "main_symbol_report": str(MAIN_SYMBOL_REPORT.relative_to(ROOT)),
        "train_splits": [str(path.relative_to(ROOT)) for path in TRAIN_SPLITS],
        "locked_split": str(LOCKED_SPLIT.relative_to(ROOT)),
        "feature_policy": {
            "uses_gold_label": False,
            "uses_source_id": False,
            "uses_expert_probabilities": False,
            "features": {
                "geometry_only": "normalized bbox center/size/area/aspect",
                "room_context_only": "containing room coarse category flags",
                "neighbor_stats_only": "record symbol-count and area statistics",
                "full_arbitration": "geometry + room context + neighbor statistics",
            },
            "paper_role": "label-level post-router arbitration; not a learned family router and not a sparse-MoE routing contribution",
        },
        "leakage_check": {
            "train_images": len(train_images),
            "locked_images": len(locked_images),
            "image_overlap": len(overlap),
            "overlap_examples": overlap[:10],
            "passed": len(overlap) == 0,
        },
        "train_sampling": {
            "raw_items": len(extract_items(train_rows)),
            "sampled_items": len(train_items),
            "label_counts": dict(Counter(y_train)),
            "max_train_per_label": MAX_TRAIN_PER_LABEL,
        },
        "locked_label_counts": dict(Counter(y_locked)),
        "feature_ablation": ablations,
        "cross_source": cross_source_status(),
        "adoption_check": {
            "full_node_macro_f1_ge_085": float(full["node_macro_f1"]) >= 0.85,
            "full_relation_f1_ge_090": float(full["relation_f1"]) >= 0.90,
            "no_image_overlap": len(overlap) == 0,
        },
        "interpretation": {
            "main_result_can_keep_symbol_arbitration": bool(float(full["node_macro_f1"]) >= 0.85 and float(full["relation_f1"]) >= 0.90 and len(overlap) == 0),
            "remaining_boundary": "No non-CubiCasa symbol locked split is available locally, so cross-source symbol generalization remains a limitation rather than a claimed result.",
            "long_tail_note": "generic_symbol remains weak despite positive delta; table has zero locked support in this split.",
        },
        "status": "passed" if float(full["node_macro_f1"]) >= 0.85 and float(full["relation_f1"]) >= 0.90 and len(overlap) == 0 else "needs_attention",
    }
    write_json(REPORT, report)
    print(f"wrote {REPORT}")
    print(json.dumps(report["adoption_check"], indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
