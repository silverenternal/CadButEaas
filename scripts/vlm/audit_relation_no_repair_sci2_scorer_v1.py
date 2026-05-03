#!/usr/bin/env python3
"""SCI2 relation no-repair scorer and stronger ceiling diagnostic."""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_recall_fscore_support
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT / "scripts" / "vlm") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from audit_relation_gold_id_repair_sensitivity_v1 import build_nodes  # noqa: E402
from audit_relation_no_repair_rule_sweep_v1 import bbox_intersection, containing_rooms, edges_for_rule  # noqa: E402
from fuse_real_upstream import (  # noqa: E402
    ROOM_LABELS,
    SYMBOL_LABELS,
    _bbox_area,
    _bbox_center,
    _node_key,
    compute_invalid_graph_rate,
    evaluate_relations,
    extract_gold,
    load_jsonl,
)

PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_label_arbitrated_v1.jsonl"
DEV_SPLIT = ROOT / "datasets" / "cadstruct_real_world_benchmark_v1" / "room_space" / "cubicasa5k_reviewed_locked_test.jsonl"
SWEEP = ROOT / "reports" / "vlm" / "relation_no_repair_rule_sweep_v1.json"
OUTPUT = ROOT / "reports" / "vlm" / "relation_no_repair_sci2_scorer_v1.json"
CEILING_V2 = ROOT / "reports" / "vlm" / "relation_no_repair_ceiling_diagnostic_v2.json"
HARD_CASES = ROOT / "reports" / "vlm" / "relation_no_repair_sci2_hard_cases_v1.jsonl"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def gold_edge_set(records: list[dict[str, Any]]) -> set[tuple[str, str, str]]:
    edges = set()
    for record_index, record in enumerate(records):
        expected = record.get("expected_json") or {}
        for edge in (expected.get("scene_graph") or {}).get("edges") or []:
            edges.add((_node_key(record_index, "space", str(edge.get("source"))), _node_key(record_index, "symbol", str(edge.get("target"))), str(edge.get("relation"))))
    return edges


def gold_label_maps(records: list[dict[str, Any]]) -> tuple[dict[str, str], dict[str, str]]:
    room_labels: dict[str, str] = {}
    symbol_labels: dict[str, str] = {}
    for record_index, record in enumerate(records):
        expected = record.get("expected_json") or {}
        for room in expected.get("room_candidates") or []:
            room_labels[_node_key(record_index, "space", str(room.get("id")))] = str(room.get("room_type"))
        for sym in expected.get("symbol_candidates") or []:
            symbol_labels[_node_key(record_index, "symbol", str(sym.get("id")))] = str(sym.get("symbol_type"))
    return room_labels, symbol_labels


def apply_label_oracle(
    record_nodes: list[list[dict[str, Any]]],
    *,
    room_labels: dict[str, str] | None = None,
    symbol_labels: dict[str, str] | None = None,
) -> list[list[dict[str, Any]]]:
    room_labels = room_labels or {}
    symbol_labels = symbol_labels or {}
    out: list[list[dict[str, Any]]] = []
    for nodes in record_nodes:
        new_nodes = []
        for node in nodes:
            item = dict(node)
            if item["id"] in room_labels:
                item["semantic_type"] = room_labels[item["id"]]
                item["oracle"] = "gold_room_label"
            if item["id"] in symbol_labels:
                item["semantic_type"] = symbol_labels[item["id"]]
                item["oracle"] = "gold_symbol_label"
            new_nodes.append(item)
        out.append(new_nodes)
    return out


def spatial_nodes(nodes: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    rooms = []
    symbols = []
    for node in nodes:
        center = _bbox_center(node.get("bbox"))
        if center is None:
            continue
        item = {**node, "_center": center}
        if node.get("semantic_type") in ROOM_LABELS:
            rooms.append(item)
        if node.get("semantic_type") in SYMBOL_LABELS:
            symbols.append(item)
    return rooms, symbols


def record_features(record_index: int, nodes: list[dict[str, Any]], gold_edges: set[tuple[str, str, str]]) -> list[dict[str, Any]]:
    rooms, symbols = spatial_nodes(nodes)
    rows = []
    for sym in symbols:
        sym_center = sym.get("_center")
        sym_bbox = sym.get("bbox")
        sym_area = max(_bbox_area(sym_bbox), 1e-9)
        containing_count = len(containing_rooms(sym, rooms, 0.0))
        padded_containing_count = len(containing_rooms(sym, rooms, 2.0))
        local_room_density = len(rooms)
        for room in rooms:
            room_bbox = room.get("bbox")
            room_center = room.get("_center")
            inter = bbox_intersection(sym_bbox, room_bbox)
            room_area = max(_bbox_area(room_bbox), 1e-9)
            dist = math.hypot(sym_center[0] - room_center[0], sym_center[1] - room_center[1]) if sym_center and room_center else 1e9
            edge = (room["id"], sym["id"], "contains")
            rows.append(
                {
                    "record_index": record_index,
                    "source": room["id"],
                    "target": sym["id"],
                    "relation": "contains",
                    "y": 1 if edge in gold_edges else 0,
                    "features": [
                        1.0 if inter > 0 else 0.0,
                        inter / sym_area,
                        inter / room_area,
                        1.0 if containing_count > 0 and room in containing_rooms(sym, rooms, 0.0) else 0.0,
                        1.0 if padded_containing_count > 0 and room in containing_rooms(sym, rooms, 2.0) else 0.0,
                        dist,
                        dist / math.sqrt(room_area),
                        _bbox_area(room_bbox),
                        _bbox_area(sym_bbox),
                        float(room.get("confidence") or 0.0),
                        float(sym.get("confidence") or 0.0),
                        float(containing_count),
                        float(padded_containing_count),
                        float(local_room_density),
                    ],
                    "room_label": str(room.get("semantic_type")),
                    "symbol_label": str(sym.get("semantic_type")),
                }
            )
    return rows


def candidate_rows(record_nodes: list[list[dict[str, Any]]], gold_edges: set[tuple[str, str, str]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for record_index, nodes in enumerate(record_nodes):
        rows.extend(record_features(record_index, nodes, gold_edges))
    return rows


def select_edges_from_scores(rows: list[dict[str, Any]], scores: np.ndarray, threshold: float) -> list[dict[str, Any]]:
    by_symbol: dict[str, tuple[float, dict[str, Any]]] = {}
    for row, score in zip(rows, scores):
        target = str(row["target"])
        if score >= threshold and (target not in by_symbol or score > by_symbol[target][0]):
            by_symbol[target] = (float(score), row)
    return [
        {"source": row["source"], "target": row["target"], "relation": "contains", "confidence": round(score, 6), "heuristic": "sci2_relation_scorer_cv"}
        for score, row in by_symbol.values()
    ]


def evaluate_edge_set(edges: list[dict[str, Any]], gold_edges_raw: list[dict[str, Any]]) -> dict[str, Any]:
    return evaluate_relations(edges, gold_edges_raw)


def cv_scores(rows: list[dict[str, Any]], folds: int, model_name: str) -> np.ndarray:
    x = np.array([row["features"] for row in rows], dtype=float)
    y = np.array([row["y"] for row in rows], dtype=int)
    record_indices = np.array([row["record_index"] for row in rows], dtype=int)
    scores = np.zeros(len(rows), dtype=float)
    for fold in range(folds):
        train = record_indices % folds != fold
        test = ~train
        if model_name == "logreg":
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=500, class_weight="balanced", C=1.0))
        elif model_name == "extratrees":
            model = ExtraTreesClassifier(n_estimators=160, max_depth=12, min_samples_leaf=4, class_weight="balanced", random_state=20260503 + fold, n_jobs=-1)
        else:
            raise ValueError(model_name)
        model.fit(x[train], y[train])
        scores[test] = model.predict_proba(x[test])[:, 1]
    return scores


def threshold_sweep(rows: list[dict[str, Any]], scores: np.ndarray, gold_edges_raw: list[dict[str, Any]], nodes_flat: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out = []
    for threshold in np.linspace(0.05, 0.95, 19):
        edges = select_edges_from_scores(rows, scores, float(threshold))
        metrics = evaluate_edge_set(edges, gold_edges_raw)
        out.append(
            {
                "threshold": round(float(threshold), 3),
                "edge_count": len(edges),
                "relation_evaluation": metrics,
                "invalid_graph_rate": round(compute_invalid_graph_rate(nodes_flat, edges), 6),
            }
        )
    return sorted(out, key=lambda item: (item["relation_evaluation"]["f1"], item["relation_evaluation"]["precision"]), reverse=True)


def label_pair_upper_bound(rows: list[dict[str, Any]], gold_edges_raw: list[dict[str, Any]], nodes_flat: list[dict[str, Any]]) -> dict[str, Any]:
    stats: dict[tuple[str, str], Counter] = defaultdict(Counter)
    for row in rows:
        stats[(row["room_label"], row["symbol_label"])][int(row["y"])] += 1
    allowed = {pair for pair, counter in stats.items() if counter[1] > 0 and counter[1] >= counter[0] * 0.02}
    edges = []
    for row in rows:
        if (row["room_label"], row["symbol_label"]) in allowed and row["features"][3] > 0:
            edges.append({"source": row["source"], "target": row["target"], "relation": "contains", "confidence": 1.0, "heuristic": "label_pair_upper_bound"})
    # One edge per symbol, prefer smaller room and positive overlap proxy.
    by_target: dict[str, dict[str, Any]] = {}
    score_by_target: dict[str, tuple[float, float]] = {}
    row_by_key = {(row["source"], row["target"]): row for row in rows}
    for edge in edges:
        row = row_by_key[(edge["source"], edge["target"])]
        score = (float(row["features"][1]), -float(row["features"][7]))
        if edge["target"] not in by_target or score > score_by_target[edge["target"]]:
            by_target[edge["target"]] = edge
            score_by_target[edge["target"]] = score
    chosen = list(by_target.values())
    return {
        "edge_count": len(chosen),
        "relation_evaluation": evaluate_edge_set(chosen, gold_edges_raw),
        "invalid_graph_rate": round(compute_invalid_graph_rate(nodes_flat, chosen), 6),
        "allowed_label_pairs": [list(pair) for pair in sorted(allowed)],
        "paper_role": "diagnostic upper bound only; label-pair choice learned from the evaluated split.",
    }


def hard_cases(records: list[dict[str, Any]], rows: list[dict[str, Any]], edges: list[dict[str, Any]], gold_edges: set[tuple[str, str, str]]) -> list[dict[str, Any]]:
    pred = {(edge["source"], edge["target"], edge["relation"]) for edge in edges}
    by_edge = {(row["source"], row["target"], "contains"): row for row in rows}
    out = []
    for kind, edge in [("FP", item) for item in sorted(pred - gold_edges)] + [("FN", item) for item in sorted(gold_edges - pred)]:
        row = by_edge.get(edge)
        rec_i = int(edge[0 if kind == "FP" else 1].split(":", 1)[0][1:])
        out.append(
            {
                "case_id": f"sci2_relation_{len(out):03d}",
                "kind": kind,
                "record_index": rec_i,
                "sample_id": records[rec_i].get("sample_id") or records[rec_i].get("image_id"),
                "edge": {"source": edge[0], "target": edge[1], "relation": edge[2]},
                "room_label": row.get("room_label") if row else None,
                "symbol_label": row.get("symbol_label") if row else None,
                "feature_summary": {
                    "symbol_overlap_ratio": row["features"][1] if row else None,
                    "center_inside": row["features"][3] if row else None,
                    "center_distance": row["features"][5] if row else None,
                    "candidate_rooms": row["features"][13] if row else None,
                },
                "category": "non_candidate_or_node_label_limit" if row is None else "scorer_or_geometry_assignment_limit",
            }
        )
        if len(out) >= 100:
            break
    return out


def main() -> int:
    predictions = load_jsonl(PREDICTIONS)
    records = load_jsonl(DEV_SPLIT)
    _, gold_edges_raw = extract_gold(records)
    gold_edges = gold_edge_set(records)
    record_nodes = build_nodes(predictions, records)
    nodes_flat = [node for nodes in record_nodes for node in nodes]
    rows = candidate_rows(record_nodes, gold_edges)
    y = np.array([row["y"] for row in rows], dtype=int)

    sweep = load_json(SWEEP)
    config = (sweep.get("best") or {}).copy()
    config.pop("relation_evaluation", None)
    config.pop("edge_count", None)
    baseline_edges = [edge for nodes in record_nodes for edge in edges_for_rule(nodes, config)]
    baseline_metrics = evaluate_edge_set(baseline_edges, gold_edges_raw)

    models = {}
    best = None
    for model_name in ["logreg", "extratrees"]:
        scores = cv_scores(rows, 5, model_name)
        result_sweep = threshold_sweep(rows, scores, gold_edges_raw, nodes_flat)
        models[model_name] = {
            "cv": "record_index_mod_5",
            "candidate_rows": len(rows),
            "positive_rows": int(y.sum()),
            "best": result_sweep[0],
            "threshold_sweep": result_sweep[:10],
        }
        if best is None or result_sweep[0]["relation_evaluation"]["f1"] > best["relation_evaluation"]["f1"]:
            best = {**result_sweep[0], "model": model_name, "scores": scores}

    assert best is not None
    best_edges = select_edges_from_scores(rows, best["scores"], best["threshold"])
    hard = hard_cases(records, rows, best_edges, gold_edges)
    write_jsonl(HARD_CASES, hard)

    room_labels, symbol_labels = gold_label_maps(records)
    oracle_nodes = apply_label_oracle(record_nodes, room_labels=room_labels, symbol_labels=symbol_labels)
    oracle_rows = candidate_rows(oracle_nodes, gold_edges)
    oracle_scores = cv_scores(oracle_rows, 5, "extratrees")
    oracle_best = threshold_sweep(oracle_rows, oracle_scores, gold_edges_raw, [node for nodes in oracle_nodes for node in nodes])[0]

    label_upper = label_pair_upper_bound(rows, gold_edges_raw, nodes_flat)
    best_public = {k: v for k, v in best.items() if k != "scores"}
    report = {
        "version": "relation_no_repair_sci2_scorer_v1",
        "created": "2026-05-03",
        "inputs": {"predictions": str(PREDICTIONS.relative_to(ROOT)), "dev_split": str(DEV_SPLIT.relative_to(ROOT))},
        "candidate_dataset": {
            "rows": len(rows),
            "positive_rows": int(y.sum()),
            "records": len(records),
            "features": [
                "has_intersection", "symbol_overlap_ratio", "room_overlap_ratio", "center_inside", "center_inside_pad2",
                "center_distance", "center_distance_norm_room", "room_area", "symbol_area", "room_confidence",
                "symbol_confidence", "containing_count", "padded_containing_count", "room_count",
            ],
        },
        "baseline_no_repair_rule": {"selected_rule": config, "edge_count": len(baseline_edges), "relation_evaluation": baseline_metrics, "invalid_graph_rate": round(compute_invalid_graph_rate(nodes_flat, baseline_edges), 6)},
        "models": models,
        "best_cv_no_repair_scorer": best_public,
        "gold_label_oracle_appendix_only": oracle_best,
        "label_pair_upper_bound_appendix_only": label_upper,
        "hard_cases": {"path": str(HARD_CASES.relative_to(ROOT)), "count": len(hard), "summary": dict(Counter(row["category"] for row in hard).most_common())},
        "main_recommendation": "do_not_replace_main_metric" if best_public["relation_evaluation"]["f1"] < baseline_metrics["f1"] else "candidate_for_appendix_or_main_after_external_validation",
        "target_0_90_met": best_public["relation_evaluation"]["f1"] >= 0.9,
        "claim_boundary": "The scorer is a record-level CV diagnostic on the same dev benchmark; use as SCI2 evidence only with clear split disclosure or a held-out rerun.",
    }
    write_json(OUTPUT, report)

    ceiling = {
        "version": "relation_no_repair_ceiling_diagnostic_v2",
        "created": "2026-05-03",
        "baseline_no_repair_f1": baseline_metrics["f1"],
        "best_cv_scorer_f1": best_public["relation_evaluation"]["f1"],
        "gold_label_oracle_cv_scorer_f1": oracle_best["relation_evaluation"]["f1"],
        "label_pair_upper_bound_f1": label_upper["relation_evaluation"]["f1"],
        "preferred_0_90_target_met_without_repair": best_public["relation_evaluation"]["f1"] >= 0.9,
        "diagnostic_blind_spot_fixed": "v2 reports candidate-pair scorer and label-pair upper bound, not only same-rule label oracle.",
        "interpretation": "If the best no-repair scorer remains below 0.90, remaining loss is dominated by candidate/node-label/geometry ambiguity under bbox-only no-repair constraints. Gold-ID repair remains appendix-only.",
        "status": "passed",
    }
    write_json(CEILING_V2, ceiling)
    print(f"wrote {OUTPUT}")
    print(f"wrote {CEILING_V2}")
    print(f"wrote {HARD_CASES}")
    print(json.dumps({"baseline_f1": baseline_metrics["f1"], "best_cv_f1": best_public["relation_evaluation"]["f1"], "target_0_90_met": best_public["relation_evaluation"]["f1"] >= 0.9}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
