#!/usr/bin/env python3
"""Confidence intervals for the paper-main E2E metrics.

Node metrics are bootstrapped over records from the current paper-main
prediction stream. The current relation scorer report does not persist
per-edge scored decisions, so relation precision/recall/F1 intervals are
reported with count-based Wilson/delta approximations and an explicit
strict-bootstrap follow-up requirement.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT / "scripts" / "vlm") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from audit_relation_gold_id_repair_sensitivity_v1 import build_nodes  # noqa: E402
from fuse_real_upstream import extract_gold, load_jsonl  # noqa: E402

PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_v2_text_conservative_generic_override_v1.jsonl"
LOCKED_SPLIT = ROOT / "datasets" / "cadstruct_real_world_benchmark_v1" / "room_space" / "cubicasa5k_reviewed_locked_test.jsonl"
PAPER_MAIN = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_v2_text_conservative_generic_override_no_repair_scorer_v1_eval.json"
OLD_MAIN = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_label_arbitrated_no_repair_v2_eval.json"
OUTPUT = ROOT / "reports" / "vlm" / "paper_main_bootstrap_ci_v1.json"
MANIFEST = ROOT / "reports" / "vlm" / "paper_metric_table_manifest_v1.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def ci(values: list[float]) -> dict[str, float]:
    arr = np.array(values, dtype=float)
    return {
        "mean": round(float(np.mean(arr)), 6),
        "p2_5": round(float(np.percentile(arr, 2.5)), 6),
        "p50": round(float(np.percentile(arr, 50.0)), 6),
        "p97_5": round(float(np.percentile(arr, 97.5)), 6),
        "std": round(float(np.std(arr, ddof=1)), 6),
    }


def wilson(p: float, n: int, z: float = 1.96) -> dict[str, float | int]:
    if n <= 0:
        return {"n": n, "p": round(p, 6), "p2_5": 0.0, "p97_5": 0.0}
    denom = 1.0 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt((p * (1 - p) / n) + z * z / (4 * n * n)) / denom
    return {
        "n": n,
        "p": round(p, 6),
        "p2_5": round(max(0.0, center - half), 6),
        "p97_5": round(min(1.0, center + half), 6),
    }


def f1_from_pr(precision: float, recall: float) -> float:
    return 2 * precision * recall / (precision + recall) if precision + recall else 0.0


def relation_ci_from_counts(tp: int, fp: int, fn: int) -> dict[str, Any]:
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    p_ci = wilson(precision, tp + fp)
    r_ci = wilson(recall, tp + fn)
    f1_low = f1_from_pr(float(p_ci["p2_5"]), float(r_ci["p2_5"]))
    f1_high = f1_from_pr(float(p_ci["p97_5"]), float(r_ci["p97_5"]))
    return {
        "relation_precision_wilson_95": p_ci,
        "relation_recall_wilson_95": r_ci,
        "relation_f1_delta_method_approx_95": {
            "p": round(f1_from_pr(precision, recall), 6),
            "p2_5": round(f1_low, 6),
            "p97_5": round(f1_high, 6),
            "method": "Conservative interval induced by Wilson intervals for precision and recall; strict record bootstrap requires persisted scored relation edges.",
        },
    }


def record_confusion(fused_nodes: list[dict[str, Any]], gold_nodes: list[dict[str, Any]]) -> Counter[tuple[str, str]]:
    gold_by_id = {str(node["id"]): str(node.get("semantic_type")) for node in gold_nodes}
    fused_by_id = {str(node["id"]): str(node.get("semantic_type")) for node in fused_nodes}
    common_ids = set(gold_by_id) & set(fused_by_id)
    confusion: Counter[tuple[str, str]] = Counter()
    for nid in common_ids:
        confusion[(gold_by_id[nid], fused_by_id[nid])] += 1
    for nid in set(gold_by_id) - common_ids:
        confusion[(gold_by_id[nid], "__FN__")] += 1
    for nid in set(fused_by_id) - common_ids:
        confusion[("__FP__", fused_by_id[nid])] += 1
    return confusion


def metrics_from_confusion(confusion: Counter[tuple[str, str]]) -> dict[str, float]:
    labels = sorted({g for g, _ in confusion if g != "__FP__"} | {p for _, p in confusion if p != "__FN__"})
    f1s = []
    for label in labels:
        tp = confusion.get((label, label), 0)
        fp = sum(v for (g, p), v in confusion.items() if p == label and g != label)
        fn = sum(v for (g, p), v in confusion.items() if g == label and p != label)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    correct = sum(v for (g, p), v in confusion.items() if g == p and g not in ("__FP__", "__FN__"))
    total = sum(confusion.values())
    return {
        "macro_f1": round(sum(f1s) / len(f1s), 6) if f1s else 0.0,
        "accuracy": round(correct / total, 6) if total else 0.0,
    }


def main() -> int:
    records = load_jsonl(LOCKED_SPLIT)
    predictions = load_jsonl(PREDICTIONS)
    main_report = load_json(PAPER_MAIN)
    old_report = load_json(OLD_MAIN)
    gold_nodes, _ = extract_gold(records)
    record_nodes = build_nodes(predictions, records)
    gold_nodes_by_record: list[list[dict[str, Any]]] = [[] for _ in records]
    for node in gold_nodes:
        rid = int(str(node["id"]).split(":", 1)[0][1:])
        gold_nodes_by_record[rid].append(node)

    record_confusions = [record_confusion(nodes, gold) for nodes, gold in zip(record_nodes, gold_nodes_by_record)]
    rng = np.random.default_rng(20260504)
    n_boot = 1000
    node_macro_values: list[float] = []
    node_acc_values: list[float] = []
    for sample in rng.integers(0, len(records), size=(n_boot, len(records))):
        confusion: Counter[tuple[str, str]] = Counter()
        for old_index in sample.tolist():
            confusion.update(record_confusions[old_index])
        metrics = metrics_from_confusion(confusion)
        node_macro_values.append(float(metrics["macro_f1"]))
        node_acc_values.append(float(metrics["accuracy"]))

    pm_node = main_report.get("node_evaluation") or {}
    pm_relation = main_report.get("relation_evaluation") or {}
    old_node = old_report.get("node_evaluation") or {}
    old_relation = old_report.get("relation_evaluation") or {}
    relation_counts = {
        "tp": int(pm_relation.get("tp") or 0),
        "fp": int(pm_relation.get("fp") or 0),
        "fn": int(pm_relation.get("fn") or 0),
    }
    relation_ci = relation_ci_from_counts(relation_counts["tp"], relation_counts["fp"], relation_counts["fn"])
    report = {
        "version": "paper_main_bootstrap_ci_v1",
        "created": "2026-05-04",
        "protocol": "Node metrics use record-level bootstrap over locked drawings. Relation metrics use count-based Wilson/delta approximations because the current scorer report stores aggregate TP/FP/FN but not persisted per-record scored edges.",
        "records": len(records),
        "bootstrap_samples": n_boot,
        "random_seed": 20260504,
        "paper_main_source": str(PAPER_MAIN.relative_to(ROOT)),
        "predictions": str(PREDICTIONS.relative_to(ROOT)),
        "point_estimates": {
            "node_macro_f1": pm_node.get("macro_f1"),
            "node_accuracy": pm_node.get("accuracy"),
            "relation_f1": pm_relation.get("f1"),
            "relation_precision": pm_relation.get("precision"),
            "relation_recall": pm_relation.get("recall"),
            "invalid_graph_rate": main_report.get("invalid_graph_rate"),
        },
        "node_record_bootstrap_ci_95": {
            "node_macro_f1": ci(node_macro_values),
            "node_accuracy": ci(node_acc_values),
        },
        "relation_count_ci_95": relation_ci,
        "relation_counts": relation_counts,
        "delta_vs_old_main_pp": {
            "old_source": str(OLD_MAIN.relative_to(ROOT)),
            "node_macro_f1": round((float(pm_node.get("macro_f1") or 0.0) - float(old_node.get("macro_f1") or 0.0)) * 100.0, 3),
            "node_accuracy": round((float(pm_node.get("accuracy") or 0.0) - float(old_node.get("accuracy") or 0.0)) * 100.0, 3),
            "relation_f1": round((float(pm_relation.get("f1") or 0.0) - float(old_relation.get("f1") or 0.0)) * 100.0, 3),
            "relation_precision": round((float(pm_relation.get("precision") or 0.0) - float(old_relation.get("precision") or 0.0)) * 100.0, 3),
            "relation_recall": round((float(pm_relation.get("recall") or 0.0) - float(old_relation.get("recall") or 0.0)) * 100.0, 3),
        },
        "strict_relation_record_bootstrap_status": {
            "status": "pending_requires_persisted_scored_edges",
            "required_artifact": "reports/vlm/relation_scorer_symbol_v2_text_conservative_generic_override_scored_edges_v1.jsonl",
        },
        "status": "passed_node_bootstrap_and_relation_count_ci_generated",
    }
    manifest = {
        "version": "paper_metric_table_manifest_v1",
        "created": "2026-05-04",
        "paper_main_source": str(PAPER_MAIN.relative_to(ROOT)),
        "ci_source": str(OUTPUT.relative_to(ROOT)),
        "metrics_for_main_table": report["point_estimates"],
        "node_record_bootstrap_ci_95": report["node_record_bootstrap_ci_95"],
        "relation_count_ci_95": report["relation_count_ci_95"],
        "appendix_only_sources": [
            "reports/vlm/symbol_locked_exploratory_threshold_v1_eval.json",
            "reports/vlm/scene_graph_fusion_symbol_v2_text_conservative_generic_locked_exploratory_threshold_no_repair_scorer_v1_eval.json",
            "reports/vlm/relation_gold_id_repair_sensitivity_v1.json"
        ],
        "status": "passed_manifest_generated",
    }
    write_json(OUTPUT, report)
    write_json(MANIFEST, manifest)
    print(f"wrote {OUTPUT}")
    print(f"wrote {MANIFEST}")
    print(json.dumps({"node_macro_f1_ci": report["node_record_bootstrap_ci_95"]["node_macro_f1"], "relation_f1_ci": report["relation_count_ci_95"]["relation_f1_delta_method_approx_95"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
