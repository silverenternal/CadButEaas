#!/usr/bin/env python3
"""Audit weak SVG/contract symbol labels for P247c."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
FINAL_EVAL = ROOT / "reports/vlm/scene_graph_fusion_symbol_long_tail_model_no_repair_scorer_v1_eval.json"
V11_EVAL = ROOT / "reports/vlm/symbol_fixture_expert_v11_eval.json"
V13_EVAL = ROOT / "reports/vlm/symbol_fixture_expert_v13_eval.json"
OUT_JSON = ROOT / "reports/vlm/p247c_svg_weak_label_audit.json"
OUT_MD = ROOT / "reports/vlm/p247c_svg_weak_label_audit.md"

WEAK_LABELS = ["generic_symbol", "bathtub"]


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def metric(metrics: dict[str, Any], label: str) -> dict[str, Any]:
    row = metrics["per_label"][label]
    return {
        "precision": round(float(row["precision"]), 6),
        "recall": round(float(row["recall"]), 6),
        "f1": round(float(row["f1"]), 6),
        "support": int(row["support"]),
    }


def confusion(metrics: dict[str, Any], label: str) -> dict[str, int]:
    return {str(k): int(v) for k, v in (metrics.get("confusion") or {}).get(label, {}).items()}


def false_negative_total(conf: dict[str, int], label: str) -> int:
    return sum(v for k, v in conf.items() if k != label)


def main() -> None:
    final_eval = load(FINAL_EVAL)
    v11 = load(V11_EVAL)
    v13 = load(V13_EVAL)

    final_node = final_eval["node_evaluation"]
    v11_metrics = v11["locked_symbol_metrics"]
    v13_metrics = v13["locked_metrics"]
    v13_baseline = v13["baseline_locked"]
    long_tail = v13.get("long_tail_audit") or {}

    labels: dict[str, Any] = {}
    for label in WEAK_LABELS:
        v11_conf = confusion(v11_metrics, label)
        v13_conf = confusion(v13_metrics, label)
        baseline_conf = confusion(v13_baseline, label)
        labels[label] = {
            "final_contract_metric": metric(final_node, label),
            "v11_locked_symbol_metric": metric(v11_metrics, label),
            "v13_locked_symbol_metric": metric(v13_metrics, label),
            "v13_delta_vs_v11": {
                key: round(metric(v13_metrics, label)[key] - metric(v11_metrics, label)[key], 6)
                for key in ["precision", "recall", "f1"]
            },
            "train_count": int((long_tail.get("train_counts") or {}).get(label, 0)),
            "locked_count": int((long_tail.get("locked_counts") or {}).get(label, 0)),
            "pred_count_v13": int((long_tail.get("pred_counts") or {}).get(label, 0)),
            "v11_confusion_gold_to_pred": v11_conf,
            "v13_confusion_gold_to_pred": v13_conf,
            "baseline_confusion_gold_to_pred": baseline_conf,
            "v11_false_negative_total": false_negative_total(v11_conf, label),
            "v13_false_negative_total": false_negative_total(v13_conf, label),
        }

    audit = {
        "id": "p247c_svg_weak_label_audit",
        "phase": "P247c_svg_weak_label_audit",
        "sources": {
            "final_contract_eval": str(FINAL_EVAL.relative_to(ROOT)),
            "symbol_v11_eval": str(V11_EVAL.relative_to(ROOT)),
            "symbol_v13_eval": str(V13_EVAL.relative_to(ROOT)),
        },
        "claim_boundary": {
            "layer": "svg_contract_or_normalized_candidate_symbol_classification",
            "not_raster_detection": True,
            "policy": "Weak labels must remain visible in tables; any exclusion requires a declared open-set/low-support policy.",
        },
        "headline": {
            "generic_symbol": {
                "diagnosis": "Residual/open-set bucket with very low locked support and low recall.",
                "final_f1": labels["generic_symbol"]["final_contract_metric"]["f1"],
                "support": labels["generic_symbol"]["final_contract_metric"]["support"],
                "recommended_treatment": "Keep visible as open-set residual or move to appendix; do not present as solved.",
            },
            "bathtub": {
                "diagnosis": "Low-support fixture with confusion against stair/sink/equipment; v13 improves generic_symbol but hurts bathtub.",
                "final_f1": labels["bathtub"]["final_contract_metric"]["f1"],
                "support": labels["bathtub"]["final_contract_metric"]["support"],
                "recommended_treatment": "Disclose as low-support fixture limitation and consider targeted augmentation or conservative fallback.",
            },
        },
        "label_audit": labels,
        "paper_wording": [
            "We report all symbol categories, including low-support residual classes.",
            "The `generic_symbol` category is treated as an open-set residual bucket and is therefore reported separately from the main recurring fixture classes.",
            "Bathtub remains a low-support fixture limitation; its errors mainly reflect confusion with visually similar or spatially adjacent fixtures.",
            "Excluding `generic_symbol` and `bathtub`, the recurring-symbol macro-F1 is 0.939937; including them, symbol macro-F1 is 0.871436.",
        ],
        "next_actions": [
            "For paper: present both full symbol macro-F1 and recurring-symbol macro-F1 excluding explicitly declared residual/low-support classes.",
            "For engineering: add targeted bathtub augmentation or fallback arbitration against stair/sink/equipment.",
            "For generic_symbol: decide whether the label should remain a residual class, be merged, or be replaced by an open-set abstention.",
            "For ablations: include v11 vs v13 symbol expert rows because v13 improves generic_symbol from 0.558140 to 0.695652 but reduces bathtub from 0.776978 to 0.661538.",
        ],
    }

    OUT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# P247c SVG Weak-Label Audit",
        "",
        "## Scope",
        "- Layer: `svg_contract_or_normalized_candidate_symbol_classification`.",
        "- This is not a raster detection audit.",
        "- Weak labels stay visible; exclusions must be declared as residual/low-support policy.",
        "",
        "## Headline",
        "- `generic_symbol`: residual/open-set bucket with support `30`; final contract F1 `0.558140`.",
        "- `bathtub`: low-support fixture with support `72`; final contract F1 `0.773723`.",
        "- Full symbol macro-F1 is `0.871436`; recurring-symbol macro-F1 excluding `generic_symbol` and `bathtub` is `0.939937`.",
        "",
        "## Metrics And Confusions",
        "",
        "| label | source | precision | recall | F1 | support | confusion summary |",
        "|---|---|---:|---:|---:|---:|---|",
    ]
    for label in WEAK_LABELS:
        final_metric = labels[label]["final_contract_metric"]
        v11_metric = labels[label]["v11_locked_symbol_metric"]
        v13_metric = labels[label]["v13_locked_symbol_metric"]
        rows = [
            ("final_contract", final_metric, {}),
            ("symbol_v11", v11_metric, labels[label]["v11_confusion_gold_to_pred"]),
            ("symbol_v13", v13_metric, labels[label]["v13_confusion_gold_to_pred"]),
        ]
        for source, row, conf in rows:
            conf_text = ", ".join(f"{k}:{v}" for k, v in conf.items()) if conf else "not exported"
            lines.append(
                f"| {label} | {source} | {row['precision']:.6f} | {row['recall']:.6f} | "
                f"{row['f1']:.6f} | {row['support']} | {conf_text} |"
            )
    lines.extend(
        [
            "",
            "## Diagnosis",
            "- `generic_symbol` is not a normal recurring fixture. It is a low-support residual/open-set category; the main issue is recall, not precision.",
            "- `bathtub` is low-support and visually/spatially confused with stair, sink, and equipment. V13 improves generic-symbol recall but sacrifices bathtub recall.",
            "- The clean paper move is to report both all-symbol metrics and recurring-symbol metrics, while explicitly disclosing these two weak labels.",
            "",
            "## Recommended Paper Wording",
            "- We report all symbol categories, including low-support residual classes.",
            "- The `generic_symbol` category is treated as an open-set residual bucket and reported separately from recurring fixture classes.",
            "- Bathtub remains a low-support fixture limitation and is included in the weak-label audit.",
            "",
            "## Engineering Follow-Up",
            "- Add targeted bathtub augmentation or fallback arbitration against stair/sink/equipment.",
            "- Decide whether `generic_symbol` should remain residual, be merged, or become an abstention/open-set flag.",
            "- Include v11 vs v13 in ablations because they trade off `generic_symbol` and `bathtub` differently.",
        ]
    )
    OUT_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": [str(OUT_JSON.relative_to(ROOT)), str(OUT_MD.relative_to(ROOT))]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
