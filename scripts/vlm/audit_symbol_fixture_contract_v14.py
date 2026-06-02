#!/usr/bin/env python3
"""Audit symbol fixture expert contracts and raster-only readiness.

This report separates historical SVG/CubiCasa candidate classifiers from the
non-SVG raster-only symbol detector/type expert that the current MoE needs.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/vlm/symbol_fixture_contract_audit_v14.json"


def load_json(path: str) -> dict[str, Any]:
    p = ROOT / path
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def first_jsonl(path: str) -> dict[str, Any]:
    p = ROOT / path
    if not p.exists():
        return {}
    with p.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                return json.loads(line)
    return {}


def checkpoint_contract(path: str) -> dict[str, Any]:
    p = ROOT / path
    if not p.exists():
        return {"exists": False}
    data = joblib.load(p)
    contract = data.get("feature_contract") or data.get("feature_names") or data.get("features")
    labels = data.get("labels") or data.get("classes") or data.get("class_names") or []
    return {
        "exists": True,
        "keys": sorted(str(k) for k in data.keys()),
        "feature_count": len(contract) if isinstance(contract, list) else None,
        "feature_contract": contract if isinstance(contract, list) else None,
        "label_count": len(labels) if isinstance(labels, list) else None,
        "labels": labels if isinstance(labels, list) else None,
    }


def split_counts(path: str) -> dict[str, Any]:
    p = ROOT / path
    labels: Counter[str] = Counter()
    rows = 0
    feature_lengths: Counter[int] = Counter()
    metadata_present = 0
    if p.exists():
        with p.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                row = json.loads(line)
                rows += 1
                labels[str(row.get("label") or "")] += 1
                feats = row.get("features")
                if isinstance(feats, list):
                    feature_lengths[len(feats)] += 1
                if isinstance(row.get("metadata"), dict):
                    metadata_present += 1
    return {
        "rows": rows,
        "label_counts": dict(labels.most_common()),
        "feature_lengths": {str(k): v for k, v in sorted(feature_lengths.items())},
        "metadata_present_rows": metadata_present,
    }


def metric_summary(report: dict[str, Any]) -> dict[str, Any]:
    metrics = report.get("locked_metrics") or report.get("locked_symbol_metrics") or {}
    return {
        "accuracy": metrics.get("accuracy"),
        "macro_f1": metrics.get("macro_f1"),
        "per_label_f1": {
            label: values.get("f1")
            for label, values in (metrics.get("per_label") or {}).items()
            if isinstance(values, dict)
        },
    }


def main() -> None:
    v13 = load_json("reports/vlm/symbol_fixture_expert_v13_eval.json")
    v14 = load_json("reports/vlm/symbol_fixture_expert_v14_candidate_eval.json")
    long_tail = load_json("reports/vlm/symbol_long_tail_model_v1_eval.json")
    typed_v18 = load_json("reports/vlm/symbol_type_classifier_v18_eval.json")
    sample = first_jsonl("datasets/symbol_fixture_expert_v13_hard_cases/train.jsonl")

    report = {
        "version": "symbol_fixture_contract_audit_v14",
        "claim": "The historical v13 symbol score is reproducible as a CubiCasa/SVG candidate-level classifier, but it is not the raster-only symbol body/type expert required by the non-SVG MoE.",
        "contracts": {
            "v13_checkpoint": checkpoint_contract("checkpoints/symbol_fixture_expert_v13/model.joblib"),
            "v14_candidate_reproduction_checkpoint": checkpoint_contract("checkpoints/symbol_fixture_expert_v14_candidate/model.joblib"),
            "long_tail_checkpoint": checkpoint_contract("checkpoints/symbol_long_tail_model_v1/model.joblib"),
        },
        "historical_metrics": {
            "v13_report": metric_summary(v13),
            "v14_candidate_reproduction": metric_summary(v14),
            "long_tail_model_v1": metric_summary(long_tail),
            "symbol_type_classifier_v18_locked": typed_v18.get("locked") or {},
        },
        "dataset_contracts": {
            "v13_hard_case_train": split_counts("datasets/symbol_fixture_expert_v13_hard_cases/train.jsonl"),
            "v13_hard_case_dev": split_counts("datasets/symbol_fixture_expert_v13_hard_cases/dev.jsonl"),
            "v13_hard_case_locked": split_counts("datasets/symbol_fixture_expert_v13_hard_cases/locked.jsonl"),
            "first_row_feature_count": len(sample.get("features") or []) if sample else None,
            "first_row_has_metadata": isinstance(sample.get("metadata"), dict) if sample else None,
        },
        "blocking_findings": [
            {
                "id": "SYMBOL-CONTRACT-001",
                "severity": "high",
                "finding": "The v13 checkpoint uses a 9D feature contract, while the v13 hard-case dataset stores an 11D feature vector. The trainer recomputes 9D features and ignores the stored 11D row features.",
                "impact": "Metrics are reproducible, but feature-contract ownership is ambiguous unless audited explicitly.",
            },
            {
                "id": "SYMBOL-CONTRACT-002",
                "severity": "critical",
                "finding": "The v13 trainer's final feature is is_equipment_like derived from row.label. That is available during supervised dataset construction but is not a valid runtime feature for a non-SVG raster-only model.",
                "impact": "The 0.883069 locked macro F1 should be treated as historical candidate-classifier evidence, not as proof that the raster symbol expert is fixed.",
            },
            {
                "id": "SYMBOL-CONTRACT-003",
                "severity": "critical",
                "finding": "The current raster symbol type stream has very low typed precision/recall and huge candidate inflation in symbol_type_classifier_v18_eval.",
                "impact": "The immediate fix path is a public raster symbol body/type dataset and learned detector/type head, not more relation-side filtering.",
            },
        ],
        "decision": {
            "keep_v13_for": "Auditing historical SVG/CubiCasa candidate-level symbol classification.",
            "do_not_use_v13_for": "Claiming non-SVG raster-only symbol detection/type performance.",
            "next_artifact": "datasets/symbol_expert_public_raster_v19/manifest.json",
        },
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"wrote": str(OUT), "findings": len(report["blocking_findings"])}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
