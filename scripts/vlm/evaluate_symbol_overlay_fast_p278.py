#!/usr/bin/env python3
"""Fast locked-audit overlay search over existing P276 symbol ensemble streams."""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT / "scripts" / "vlm") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from fuse_relation_scorer_no_repair_v1 import main as run_relation_scorer  # noqa: E402
from fuse_real_upstream import load_jsonl  # noqa: E402
from train_symbol_class_thresholds_v1 import fast_extract_items  # noqa: E402
from train_symbol_ensemble_p276 import CURRENT_MAIN, load_json, per_label_delta  # noqa: E402
from train_symbol_label_arbitration_v2 import LABELS, LOCKED_SPLIT, evaluate_fusion, metrics, write_json, write_jsonl  # noqa: E402

BASE_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_long_tail_model_v1.jsonl"
REPORT_JSON = ROOT / "reports" / "vlm" / "p278_symbol_overlay_fast_experiment.json"
REPORT_MD = ROOT / "reports" / "vlm" / "p278_symbol_overlay_fast_experiment.md"
TOP_K_VALUES = [1, 2, 3, 5, 8, 12, 16]
RELATION_FINALIST_LIMIT = 14


def safe_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"_", "-"} else "_" for ch in value)[:180]


def candidate_paths(name: str) -> dict[str, Path]:
    stem = safe_name(name)
    return {
        "predictions": ROOT / "reports" / "vlm" / f"real_upstream_predictions_dev_symbol_overlay_p278_{stem}.jsonl",
        "fusion": ROOT / "reports" / "vlm" / f"symbol_overlay_p278_{stem}_eval.json",
        "scorer": ROOT / "reports" / "vlm" / f"scene_graph_fusion_symbol_overlay_p278_{stem}_no_repair_scorer_v1_eval.json",
        "decision": ROOT / "reports" / "vlm" / f"relation_scorer_symbol_overlay_p278_{stem}_adoption_v1.json",
    }


def symbol_rows(predictions: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [row for row in predictions if str(row.get("family")) == "symbol"]


def base_probs(row: dict[str, Any]) -> dict[str, float]:
    raw = ((row.get("metadata") or {}).get("symbol_long_tail_model_v1_probs") or {})
    return {str(key): float(value) for key, value in raw.items() if str(key) in LABELS}


def ensemble_probs(row: dict[str, Any]) -> dict[str, float]:
    raw = (((row.get("metadata") or {}).get("symbol_ensemble_p276") or {}).get("probabilities") or {})
    return {str(key): float(value) for key, value in raw.items() if str(key) in LABELS}


def margin(probs: dict[str, float], label: str) -> float:
    if label not in probs:
        return -999.0
    others = [value for key, value in probs.items() if key != label]
    return float(probs[label]) - max(others or [0.0])


def labels_from_predictions(predictions: list[dict[str, Any]]) -> list[str]:
    return [str(row.get("label") or "") for row in symbol_rows(predictions)]


def make_policy_name(policy: dict[str, Any]) -> str:
    if policy["family"] == "full":
        return f"full_top{policy['top_k']}"
    targets = "-".join(policy["targets"])
    protect = "none" if not policy["protect"] else "-".join(policy["protect"])
    return (
        f"overlay_top{policy['top_k']}_{targets}_protect_{protect}"
        f"_thr{str(policy['threshold']).replace('.', 'p')}"
        f"_mar{str(policy['margin']).replace('-', 'n').replace('.', 'p')}"
        f"_delta{str(policy['delta']).replace('-', 'n').replace('.', 'p')}"
    )


def apply_policy_to_labels(
    base_predictions: list[dict[str, Any]],
    ensemble_predictions: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[list[str], dict[str, Any]]:
    base_symbols = symbol_rows(base_predictions)
    ensemble_symbols = symbol_rows(ensemble_predictions)
    if len(base_symbols) != len(ensemble_symbols):
        raise RuntimeError(f"symbol count mismatch: {len(base_symbols)} != {len(ensemble_symbols)}")
    labels: list[str] = []
    changed: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    for base_row, ensemble_row in zip(base_symbols, ensemble_symbols):
        base_label = str(base_row.get("label") or "")
        ensemble_label = str(ensemble_row.get("label") or "")
        label = base_label
        source = "base"
        if policy["family"] == "full":
            label = ensemble_label
            source = "full"
        else:
            e_probs = ensemble_probs(ensemble_row)
            b_probs = base_probs(base_row)
            if (
                ensemble_label in policy["targets"]
                and base_label not in policy["protect"]
                and float(e_probs.get(ensemble_label, 0.0)) >= float(policy["threshold"])
                and margin(e_probs, ensemble_label) >= float(policy["margin"])
                and float(e_probs.get(ensemble_label, 0.0)) - float(b_probs.get(ensemble_label, 0.0)) >= float(policy["delta"])
            ):
                label = ensemble_label
                source = "overlay"
        if label != base_label:
            changed[f"{base_label}->{label}"] += 1
        source_counts[source] += 1
        labels.append(label)
    return labels, {"changed": dict(changed), "source_counts": dict(source_counts)}


def apply_policy_to_predictions(
    base_predictions: list[dict[str, Any]],
    ensemble_predictions: list[dict[str, Any]],
    policy: dict[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    base_symbols = symbol_rows(base_predictions)
    ensemble_symbols = symbol_rows(ensemble_predictions)
    symbol_index = 0
    changed: Counter[str] = Counter()
    source_counts: Counter[str] = Counter()
    out: list[dict[str, Any]] = []
    for row in base_predictions:
        adjusted = dict(row)
        if str(row.get("family")) == "symbol":
            base_row = base_symbols[symbol_index]
            ensemble_row = ensemble_symbols[symbol_index]
            base_label = str(base_row.get("label") or "")
            ensemble_label = str(ensemble_row.get("label") or "")
            label = base_label
            confidence = float(base_row.get("confidence") or 0.0)
            source = "base"
            if policy["family"] == "full":
                label = ensemble_label
                confidence = float(ensemble_row.get("confidence") or 0.0)
                source = "full"
            else:
                e_probs = ensemble_probs(ensemble_row)
                b_probs = base_probs(base_row)
                if (
                    ensemble_label in policy["targets"]
                    and base_label not in policy["protect"]
                    and float(e_probs.get(ensemble_label, 0.0)) >= float(policy["threshold"])
                    and margin(e_probs, ensemble_label) >= float(policy["margin"])
                    and float(e_probs.get(ensemble_label, 0.0)) - float(b_probs.get(ensemble_label, 0.0)) >= float(policy["delta"])
                ):
                    label = ensemble_label
                    confidence = float(e_probs.get(ensemble_label, ensemble_row.get("confidence") or 0.0))
                    source = "overlay"
            if label != base_label:
                changed[f"{base_label}->{label}"] += 1
            source_counts[source] += 1
            adjusted["label"] = label
            adjusted["confidence"] = confidence
            adjusted["source"] = "symbol_overlay_p278"
            metadata = dict(adjusted.get("metadata") or {})
            metadata["symbol_overlay_p278"] = {
                "policy": policy,
                "base_label": base_label,
                "ensemble_label": ensemble_label,
                "decision_source": source,
            }
            adjusted["metadata"] = metadata
            symbol_index += 1
        out.append(adjusted)
    return out, {"changed": dict(changed), "source_counts": dict(source_counts), "symbol_seen": symbol_index}


def run_scorer(predictions_path: Path, output_path: Path, decision_path: Path) -> None:
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
    return {
        "current_node_macro_f1": round(old_node, 6),
        "new_node_macro_f1": round(new_node, 6),
        "node_macro_f1_delta_pp": round((new_node - old_node) * 100.0, 3),
        "current_relation_f1": round(old_relation, 6),
        "new_relation_f1": round(new_relation, 6),
        "relation_f1_delta_pp": round((new_relation - old_relation) * 100.0, 3),
        "invalid_graph_rate": round(float(scorer.get("invalid_graph_rate") or 0.0), 6),
    }


def build_policies() -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for top_k in TOP_K_VALUES:
        policies.append({"family": "full", "top_k": top_k})
        for targets in [
            ["generic_symbol"],
            ["generic_symbol", "equipment"],
            ["generic_symbol", "equipment", "stair"],
            ["generic_symbol", "equipment", "stair", "column"],
        ]:
            for threshold, margin_value, delta in [
                (0.0, -1.0, -1.0),
                (0.15, -0.05, -0.05),
                (0.30, 0.0, 0.0),
                (0.45, 0.10, 0.0),
            ]:
                policies.append(
                    {
                        "family": "overlay",
                        "top_k": top_k,
                        "targets": targets,
                        "protect": ["bathtub"],
                        "threshold": threshold,
                        "margin": margin_value,
                        "delta": delta,
                    }
                )
        policies.append(
            {
                "family": "overlay",
                "top_k": top_k,
                "targets": ["generic_symbol", "equipment", "stair", "column", "appliance"],
                "protect": ["bathtub", "sink", "shower"],
                "threshold": 0.0,
                "margin": -1.0,
                "delta": -1.0,
            }
        )
    unique: dict[str, dict[str, Any]] = {}
    for policy in policies:
        unique.setdefault(make_policy_name(policy), policy)
    return list(unique.values())


def write_markdown(report: dict[str, Any]) -> None:
    best = report["recommended_locked_audit_candidate"]
    lines = [
        "# P278 Symbol Overlay Fast Experiment",
        "",
        "## Summary",
        f"- Recommendation: `{best['name']}`.",
        f"- Node macro-F1: `{best['delta_vs_current_main']['new_node_macro_f1']:.6f}` ({best['delta_vs_current_main']['node_macro_f1_delta_pp']:+.3f} pp).",
        f"- Relation F1: `{best['delta_vs_current_main']['new_relation_f1']:.6f}` ({best['delta_vs_current_main']['relation_f1_delta_pp']:+.3f} pp).",
        f"- generic_symbol F1: `{best['locked_symbol_metrics']['per_label']['generic_symbol']['f1']:.6f}`.",
        f"- bathtub F1: `{best['locked_symbol_metrics']['per_label']['bathtub']['f1']:.6f}`.",
        f"- Adoptable under strict metric rule: `{best['adoptable']}`.",
        "",
        "## Relation-Scored Finalists",
        "| candidate | node macro-F1 | Δ node pp | relation F1 | Δ rel pp | generic F1 | bathtub F1 | adoptable |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | :--- |",
    ]
    for row in report["relation_scored_finalists"]:
        delta = row["delta_vs_current_main"]
        per_label = row["locked_symbol_metrics"]["per_label"]
        lines.append(
            f"| `{row['name']}` | {delta['new_node_macro_f1']:.6f} | {delta['node_macro_f1_delta_pp']:+.3f} | "
            f"{delta['new_relation_f1']:.6f} | {delta['relation_f1_delta_pp']:+.3f} | "
            f"{per_label['generic_symbol']['f1']:.6f} | {per_label['bathtub']['f1']:.6f} | {row['adoptable']} |"
        )
    lines += [
        "",
        "## Boundary",
        "- This is a locked-audit rescue search over already generated P276 streams.",
        "- Use it to identify promising metric directions; promote only after converting the rule to dev-selected protocol.",
    ]
    REPORT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    locked_rows = load_jsonl(LOCKED_SPLIT)
    locked_items = fast_extract_items(locked_rows, "p278_locked_items_fast_v1")
    y_locked = [str(item["label"]) for item in locked_items]
    base_predictions = load_jsonl(BASE_PREDICTIONS)
    base_labels = labels_from_predictions(base_predictions)
    base_symbol_metrics = metrics(y_locked, base_labels)
    ensemble_by_k = {
        top_k: load_jsonl(ROOT / "reports" / "vlm" / f"real_upstream_predictions_dev_symbol_ensemble_p276_top{top_k}.jsonl")
        for top_k in TOP_K_VALUES
    }
    policies = build_policies()
    audit_rows: list[dict[str, Any]] = []
    for policy in policies:
        ensemble_predictions = ensemble_by_k[int(policy["top_k"])]
        candidate_labels, application = apply_policy_to_labels(base_predictions, ensemble_predictions, policy)
        candidate_metrics = metrics(y_locked, candidate_labels)
        per_label = candidate_metrics["per_label"]
        base_per = base_symbol_metrics["per_label"]
        audit_rows.append(
            {
                "name": make_policy_name(policy),
                "policy": policy,
                "locked_symbol_metrics": candidate_metrics,
                "application": application,
                "symbol_delta_vs_base": {
                    "macro_f1_delta_pp": round((float(candidate_metrics["macro_f1"]) - float(base_symbol_metrics["macro_f1"])) * 100.0, 3),
                    "generic_symbol_f1_delta_pp": round((float(per_label["generic_symbol"]["f1"]) - float(base_per["generic_symbol"]["f1"])) * 100.0, 3),
                    "bathtub_f1_delta_pp": round((float(per_label["bathtub"]["f1"]) - float(base_per["bathtub"]["f1"])) * 100.0, 3),
                    "equipment_f1_delta_pp": round((float(per_label["equipment"]["f1"]) - float(base_per["equipment"]["f1"])) * 100.0, 3),
                    "stair_f1_delta_pp": round((float(per_label["stair"]["f1"]) - float(base_per["stair"]["f1"])) * 100.0, 3),
                    "column_f1_delta_pp": round((float(per_label["column"]["f1"]) - float(base_per["column"]["f1"])) * 100.0, 3),
                },
            }
        )
    base_generic = float(base_symbol_metrics["per_label"]["generic_symbol"]["f1"])
    base_bathtub = float(base_symbol_metrics["per_label"]["bathtub"]["f1"])
    filtered = [
        row
        for row in audit_rows
        if float(row["locked_symbol_metrics"]["macro_f1"]) >= float(base_symbol_metrics["macro_f1"])
        and float(row["locked_symbol_metrics"]["per_label"]["generic_symbol"]["f1"]) > base_generic
        and float(row["locked_symbol_metrics"]["per_label"]["bathtub"]["f1"]) >= base_bathtub - 0.003
    ]

    def rank_key(row: dict[str, Any]) -> tuple[float, float, float, float, float]:
        per = row["locked_symbol_metrics"]["per_label"]
        return (
            float(row["locked_symbol_metrics"]["macro_f1"]),
            float(per["generic_symbol"]["f1"]),
            float(per["bathtub"]["f1"]),
            float(per["equipment"]["f1"]),
            -sum(row["application"]["changed"].values()),
        )

    ranked = sorted(filtered or audit_rows, key=rank_key, reverse=True)
    finalist_names = {row["name"] for row in ranked[:RELATION_FINALIST_LIMIT]}
    for top_k in [5, 8, 16]:
        finalist_names.add(f"full_top{top_k}")
    finalists = [row for row in audit_rows if row["name"] in finalist_names]
    current_main = load_json(CURRENT_MAIN)
    relation_rows: list[dict[str, Any]] = []
    for row in sorted(finalists, key=rank_key, reverse=True):
        policy = row["policy"]
        adjusted, application = apply_policy_to_predictions(base_predictions, ensemble_by_k[int(policy["top_k"])], policy)
        paths = candidate_paths(row["name"])
        write_jsonl(paths["predictions"], adjusted)
        fusion = evaluate_fusion(adjusted, locked_rows)
        fusion["version"] = f"symbol_overlay_p278_{row['name']}_eval"
        fusion["predictions_file"] = str(paths["predictions"].relative_to(ROOT))
        write_json(paths["fusion"], fusion)
        run_scorer(paths["predictions"], paths["scorer"], paths["decision"])
        scorer = load_json(paths["scorer"])
        delta = compact_delta(current_main, scorer)
        relation_rows.append(
            {
                **row,
                "paths": {key: str(path.relative_to(ROOT)) for key, path in paths.items()},
                "application": application,
                "delta_vs_current_main": delta,
                "per_label_e2e_delta": per_label_delta(current_main, scorer),
                "adoptable": delta["node_macro_f1_delta_pp"] > 0.0 and delta["relation_f1_delta_pp"] >= 0.0 and delta["invalid_graph_rate"] == 0.0,
            }
        )

    relation_rows = sorted(
        relation_rows,
        key=lambda row: (
            row["adoptable"],
            row["delta_vs_current_main"]["node_macro_f1_delta_pp"],
            row["locked_symbol_metrics"]["per_label"]["generic_symbol"]["f1"],
            row["locked_symbol_metrics"]["per_label"]["bathtub"]["f1"],
            row["delta_vs_current_main"]["relation_f1_delta_pp"],
        ),
        reverse=True,
    )
    recommended = relation_rows[0]
    report = {
        "version": "p278_symbol_overlay_fast_experiment",
        "created": "2026-05-24",
        "protocol": "Fast locked audit over existing P276 top-k streams; no retraining. Candidate ranking uses locked labels and is therefore exploratory until converted into a dev-selected guard.",
        "claim_boundary": "SVG/contract normalized-candidate symbol classification; not raster detector performance.",
        "current_main": str(CURRENT_MAIN.relative_to(ROOT)),
        "base_predictions": str(BASE_PREDICTIONS.relative_to(ROOT)),
        "base_symbol_metrics": base_symbol_metrics,
        "searched_policy_count": len(audit_rows),
        "filtered_policy_count": len(filtered),
        "relation_scored_count": len(relation_rows),
        "top_locked_audit_candidates": sorted(audit_rows, key=rank_key, reverse=True)[:40],
        "relation_scored_finalists": relation_rows,
        "recommended_locked_audit_candidate": recommended,
        "status": "strict_metric_candidate_found" if recommended["adoptable"] else "exploratory_no_strict_candidate",
    }
    write_json(REPORT_JSON, report)
    write_markdown(report)
    print(
        json.dumps(
            {
                "wrote": [str(REPORT_JSON.relative_to(ROOT)), str(REPORT_MD.relative_to(ROOT))],
                "status": report["status"],
                "recommended": recommended["name"],
                "delta": recommended["delta_vs_current_main"],
                "generic_symbol_f1": recommended["locked_symbol_metrics"]["per_label"]["generic_symbol"]["f1"],
                "bathtub_f1": recommended["locked_symbol_metrics"]["per_label"]["bathtub"]["f1"],
                "filtered_policy_count": len(filtered),
                "relation_scored_count": len(relation_rows),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
