#!/usr/bin/env python3
"""Evaluate relation-aware P276 symbol ensemble top-k tradeoffs."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts" / "vlm") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from fuse_relation_scorer_no_repair_v1 import main as run_relation_scorer  # noqa: E402
from fuse_real_upstream import load_jsonl  # noqa: E402
from train_symbol_class_thresholds_v1 import DEV_ONLY, TRAIN_ONLY, fast_extract_items  # noqa: E402
from train_symbol_ensemble_p276 import (  # noqa: E402
    CURRENT_MAIN,
    CURRENT_PREDICTIONS,
    apply_ensemble_labels,
    compact_symbol_metrics,
    load_json,
    per_label_delta,
    predict_labels,
    select_ensemble,
    train_candidates,
)
from train_symbol_label_arbitration_v2 import (  # noqa: E402
    LABELS,
    LOCKED_SPLIT,
    evaluate_fusion,
    metrics,
    split_images,
    stratified,
    write_json,
    write_jsonl,
)

TOP_K_CANDIDATES = [1, 2, 3, 4, 5, 8, 12, 16]
REPORT_JSON = ROOT / "reports" / "vlm" / "p276_symbol_ensemble_tradeoff.json"
REPORT_MD = ROOT / "reports" / "vlm" / "p276_symbol_ensemble_tradeoff.md"


def candidate_paths(top_k: int) -> dict[str, Path]:
    stem = f"symbol_ensemble_p276_top{top_k}"
    return {
        "predictions": ROOT / "reports" / "vlm" / f"real_upstream_predictions_dev_{stem}.jsonl",
        "fusion": ROOT / "reports" / "vlm" / f"{stem}_eval.json",
        "scorer": ROOT / "reports" / "vlm" / f"scene_graph_fusion_{stem}_no_repair_scorer_v1_eval.json",
        "decision": ROOT / "reports" / "vlm" / f"relation_scorer_{stem}_adoption_v1.json",
    }


def score_relations(predictions_path: Path, output_path: Path, decision_path: Path) -> None:
    old_argv = sys.argv[:]
    try:
        sys.argv = [
            "fuse_relation_scorer_no_repair_v1.py",
            "--predictions",
            str(predictions_path),
            "--output",
            str(output_path),
            "--decision",
            str(decision_path),
            "--baseline",
            str(CURRENT_MAIN),
        ]
        run_relation_scorer()
    finally:
        sys.argv = old_argv


def compact_delta(current: dict[str, Any], scorer: dict[str, Any]) -> dict[str, Any]:
    old_node = float((current.get("node_evaluation") or {}).get("macro_f1") or 0.0)
    new_node = float((scorer.get("node_evaluation") or {}).get("macro_f1") or 0.0)
    old_relation = float((current.get("relation_evaluation") or {}).get("f1") or 0.0)
    new_relation = float((scorer.get("relation_evaluation") or {}).get("f1") or 0.0)
    old_precision = float((current.get("relation_evaluation") or {}).get("precision") or 0.0)
    new_precision = float((scorer.get("relation_evaluation") or {}).get("precision") or 0.0)
    old_recall = float((current.get("relation_evaluation") or {}).get("recall") or 0.0)
    new_recall = float((scorer.get("relation_evaluation") or {}).get("recall") or 0.0)
    return {
        "current_node_macro_f1": round(old_node, 6),
        "new_node_macro_f1": round(new_node, 6),
        "node_macro_f1_delta_pp": round((new_node - old_node) * 100.0, 3),
        "current_relation_f1": round(old_relation, 6),
        "new_relation_f1": round(new_relation, 6),
        "relation_f1_delta_pp": round((new_relation - old_relation) * 100.0, 3),
        "current_relation_precision": round(old_precision, 6),
        "new_relation_precision": round(new_precision, 6),
        "relation_precision_delta_pp": round((new_precision - old_precision) * 100.0, 3),
        "current_relation_recall": round(old_recall, 6),
        "new_relation_recall": round(new_recall, 6),
        "relation_recall_delta_pp": round((new_recall - old_recall) * 100.0, 3),
        "invalid_graph_rate": round(float(scorer.get("invalid_graph_rate") or 0.0), 6),
    }


def recommend_candidate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    adoptable = [
        row
        for row in rows
        if row["delta_vs_current_main"]["node_macro_f1_delta_pp"] > 0.0
        and row["delta_vs_current_main"]["relation_f1_delta_pp"] >= 0.0
        and row["delta_vs_current_main"]["invalid_graph_rate"] == 0.0
    ]
    if adoptable:
        return max(
            adoptable,
            key=lambda row: (
                row["delta_vs_current_main"]["node_macro_f1_delta_pp"],
                row["locked_symbol_metrics"]["per_label"]["generic_symbol"]["f1"],
                row["delta_vs_current_main"]["relation_f1_delta_pp"],
            ),
        )
    return max(
        rows,
        key=lambda row: (
            row["delta_vs_current_main"]["node_macro_f1_delta_pp"],
            row["locked_symbol_metrics"]["per_label"]["generic_symbol"]["f1"],
            row["delta_vs_current_main"]["relation_f1_delta_pp"],
        ),
    )


def write_markdown(report: dict[str, Any]) -> None:
    best = report["recommended_candidate"]
    lines = [
        "# P276 Symbol Ensemble Tradeoff",
        "",
        "## Summary",
        f"- Recommendation: top `{best['top_k']}`.",
        f"- Adopt as mainline: `{report['adopt_as_current_best_candidate']}`.",
        f"- Node macro-F1: `{best['delta_vs_current_main']['new_node_macro_f1']:.6f}` ({best['delta_vs_current_main']['node_macro_f1_delta_pp']:+.3f} pp).",
        f"- Relation F1: `{best['delta_vs_current_main']['new_relation_f1']:.6f}` ({best['delta_vs_current_main']['relation_f1_delta_pp']:+.3f} pp).",
        f"- generic_symbol F1: `{best['locked_symbol_metrics']['per_label']['generic_symbol']['f1']:.6f}`.",
        f"- bathtub F1: `{best['locked_symbol_metrics']['per_label']['bathtub']['f1']:.6f}`.",
        "",
        "## Candidate Table",
        "| top-k | node macro-F1 | Δ node pp | relation F1 | Δ rel pp | generic F1 | bathtub F1 | adoptable |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for row in report["candidates"]:
        delta = row["delta_vs_current_main"]
        per_label = row["locked_symbol_metrics"]["per_label"]
        lines.append(
            f"| {row['top_k']} | {delta['new_node_macro_f1']:.6f} | {delta['node_macro_f1_delta_pp']:+.3f} | "
            f"{delta['new_relation_f1']:.6f} | {delta['relation_f1_delta_pp']:+.3f} | "
            f"{per_label['generic_symbol']['f1']:.6f} | {per_label['bathtub']['f1']:.6f} | {row['adoptable']} |"
        )
    lines += [
        "",
        "## Decision",
        "- Keep the current mainline unless a candidate improves node macro-F1 without relation-F1 regression.",
        "- Use non-adopted candidates as symbol-rescue evidence, not as the official scene-graph main result.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    train_rows = load_jsonl(TRAIN_ONLY)
    dev_rows = load_jsonl(DEV_ONLY)
    locked_rows = load_jsonl(LOCKED_SPLIT)
    overlap = {
        "train_dev": len(split_images(train_rows) & split_images(dev_rows)),
        "train_locked": len(split_images(train_rows) & split_images(locked_rows)),
        "dev_locked": len(split_images(dev_rows) & split_images(locked_rows)),
    }
    if any(overlap.values()):
        raise SystemExit(f"split image overlap detected: {overlap}")

    train_items = stratified(fast_extract_items(train_rows, "p276_train_items_fast_v1"))
    dev_items = fast_extract_items(dev_rows, "p276_dev_items_fast_v1")
    locked_items = fast_extract_items(locked_rows, "p276_locked_items_fast_v1")
    y_dev = [str(item["label"]) for item in dev_items]
    y_locked = [str(item["label"]) for item in locked_items]
    candidate_results, _models, dev_probs, locked_probs, classes = train_candidates(train_items, dev_items, locked_items)
    selected = select_ensemble(candidate_results, dev_probs, locked_probs, classes, y_dev, y_locked)
    current_predictions = load_jsonl(CURRENT_PREDICTIONS)
    current = load_json(CURRENT_MAIN)
    rows: list[dict[str, Any]] = []

    for top_k in TOP_K_CANDIDATES:
        members = selected["candidate_ranking"][: min(top_k, len(selected["candidate_ranking"]))]
        avg_dev = sum(dev_probs[idx] for idx in members) / len(members)
        avg_locked = sum(locked_probs[idx] for idx in members) / len(members)
        dev_metrics = metrics(y_dev, predict_labels(avg_dev, classes))
        locked_metrics = metrics(y_locked, predict_labels(avg_locked, classes))
        selected_for_apply = {"top_k": top_k, "members": members}
        adjusted, application = apply_ensemble_labels(current_predictions, locked_items, avg_locked, classes, selected_for_apply)
        paths = candidate_paths(top_k)
        write_jsonl(paths["predictions"], adjusted)
        fusion = evaluate_fusion(adjusted, locked_rows)
        fusion["version"] = f"symbol_ensemble_p276_top{top_k}_eval"
        fusion["predictions_file"] = str(paths["predictions"].relative_to(ROOT))
        write_json(paths["fusion"], fusion)
        score_relations(paths["predictions"], paths["scorer"], paths["decision"])
        scorer = load_json(paths["scorer"])
        delta = compact_delta(current, scorer)
        rows.append(
            {
                "top_k": top_k,
                "members": members,
                "paths": {key: str(path.relative_to(ROOT)) for key, path in paths.items()},
                "dev_symbol_metrics": compact_symbol_metrics(dev_metrics),
                "locked_symbol_metrics": locked_metrics,
                "application": application,
                "delta_vs_current_main": delta,
                "per_label_e2e_delta": {
                    label: per_label_delta(current, scorer).get(label)
                    for label in LABELS
                    if label in {"generic_symbol", "bathtub", "column", "equipment", "stair"}
                },
                "adoptable": (
                    delta["node_macro_f1_delta_pp"] > 0.0
                    and delta["relation_f1_delta_pp"] >= 0.0
                    and delta["invalid_graph_rate"] == 0.0
                ),
            }
        )

    recommended = recommend_candidate(rows)
    report = {
        "version": "p276_symbol_ensemble_tradeoff",
        "created": "2026-05-24",
        "protocol": "Retrain P276 44D symbol candidates once, rank by dev, evaluate top-k ensembles on locked split with no-repair relation scorer.",
        "claim_boundary": "SVG/contract normalized-candidate symbol classification; not raster detector performance.",
        "current_main": str(CURRENT_MAIN.relative_to(ROOT)),
        "current_predictions": str(CURRENT_PREDICTIONS.relative_to(ROOT)),
        "split_overlap": overlap,
        "top_k_candidates": TOP_K_CANDIDATES,
        "candidate_results_compact": [
            {
                "index": row["index"],
                "config": row["config"],
                "sampled_items": row["sampled_items"],
                "dev": compact_symbol_metrics(row["dev_symbol_metrics"]),
                "locked_audit": compact_symbol_metrics(row["locked_symbol_metrics_audit"]),
            }
            for row in candidate_results
        ],
        "dev_ranking": selected["candidate_ranking"],
        "candidates": rows,
        "recommended_candidate": recommended,
        "adopt_as_current_best_candidate": bool(recommended["adoptable"]),
        "status": "passed_adopt_candidate" if recommended["adoptable"] else "tradeoff_only_no_mainline_promotion",
    }
    write_json(REPORT_JSON, report)
    write_markdown(report)
    print(
        json.dumps(
            {
                "wrote": [str(REPORT_JSON.relative_to(ROOT)), str(REPORT_MD.relative_to(ROOT))],
                "status": report["status"],
                "recommended_top_k": recommended["top_k"],
                "adopt": report["adopt_as_current_best_candidate"],
                "delta": recommended["delta_vs_current_main"],
                "generic_symbol_f1": recommended["locked_symbol_metrics"]["per_label"]["generic_symbol"]["f1"],
                "bathtub_f1": recommended["locked_symbol_metrics"]["per_label"]["bathtub"]["f1"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
