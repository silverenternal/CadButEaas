#!/usr/bin/env python3
"""Fast locked-only exploratory threshold audit for symbol long-tail labels.

This script intentionally marks its output as diagnostic-only because the
locked benchmark is used to select thresholds. It is useful for finding whether
more clean dev-calibrated data/stream work is worth doing, but it must not feed
paper-main metric reconciliation.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
if str(ROOT / "scripts" / "vlm") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

from fuse_real_upstream import _predictions_by_record_family_id, load_jsonl  # noqa: E402
from train_symbol_label_arbitration_v2 import evaluate_fusion, write_json, write_jsonl  # noqa: E402


LOCKED_SPLIT = ROOT / "datasets" / "cadstruct_real_world_benchmark_v1" / "room_space" / "cubicasa5k_reviewed_locked_test.jsonl"
BASE_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_v2_text_conservative_generic_override_v1.jsonl"
BASE_FUSION = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_v2_text_conservative_generic_override_v1_eval.json"
ADJUSTED_PREDICTIONS = ROOT / "reports" / "vlm" / "real_upstream_predictions_dev_symbol_v2_text_conservative_generic_locked_exploratory_threshold_v1.jsonl"
FUSION_REPORT = ROOT / "reports" / "vlm" / "scene_graph_fusion_symbol_v2_text_conservative_generic_locked_exploratory_threshold_v1_eval.json"
REPORT = ROOT / "reports" / "vlm" / "symbol_locked_exploratory_threshold_v1_eval.json"

SEARCH_LABELS = ["bathtub", "column", "equipment", "generic_symbol", "stair", "appliance"]
THRESHOLDS = [0.02, 0.04, 0.06, 0.08, 0.10, 0.12, 0.16, 0.20, 0.24, 0.30, 0.36, 0.44, 0.52, 0.60, 0.70, 0.80]
MARGINS = [-0.35, -0.25, -0.18, -0.12, -0.06, 0.0, 0.06, 0.12, 0.18, 0.25]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def symbol_arrays(records: list[dict[str, Any]], predictions: list[dict[str, Any]]) -> tuple[list[str], list[str], list[dict[str, float]]]:
    by_record = _predictions_by_record_family_id(predictions, records)
    gold: list[str] = []
    pred: list[str] = []
    probs: list[dict[str, float]] = []
    for record_index, record in enumerate(records):
        expected = record.get("expected_json") or {}
        for symbol in expected.get("symbol_candidates") or []:
            symbol_id = str(symbol.get("id"))
            prediction = by_record.get((record_index, "symbol", symbol_id)) or {}
            gold.append(str(symbol.get("symbol_type")))
            pred.append(str(prediction.get("label") or symbol.get("symbol_type")))
            raw_probs = ((prediction.get("metadata") or {}).get("arbitration_v2_probs") or {})
            probs.append({str(k): float(v) for k, v in raw_probs.items() if not str(k).startswith("_")})
    return gold, pred, probs


def label_stats(gold: list[str], pred: list[str], labels: list[str]) -> tuple[float, dict[str, dict[str, float | int]]]:
    per_label: dict[str, dict[str, float | int]] = {}
    f1s = []
    for label in labels:
        tp = sum(1 for g, p in zip(gold, pred) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, pred) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, pred) if g == label and p != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1s.append(f1)
        per_label[label] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": sum(1 for item in gold if item == label),
        }
    return round(sum(f1s) / max(len(f1s), 1), 6), per_label


def apply_policy(base_pred: list[str], probs: list[dict[str, float]], policy: list[tuple[str, float, float]]) -> list[str]:
    out: list[str] = []
    for old_label, row_probs in zip(base_pred, probs):
        label = old_label
        if row_probs:
            for target, threshold, margin in policy:
                if target not in row_probs:
                    continue
                target_prob = float(row_probs[target])
                best_other = max(float(value) for key, value in row_probs.items() if key != target)
                if target_prob >= threshold and target_prob - best_other >= margin:
                    label = target
                    break
        out.append(label)
    return out


def apply_labels_to_predictions(predictions: list[dict[str, Any]], labels: list[str]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    out = []
    symbol_index = 0
    changed: Counter[str] = Counter()
    for prediction in predictions:
        row = dict(prediction)
        if row.get("family") == "symbol":
            new_label = labels[symbol_index]
            symbol_index += 1
            old_label = str(row.get("label"))
            if new_label != old_label:
                row["label"] = new_label
                probs = ((row.get("metadata") or {}).get("arbitration_v2_probs") or {})
                row["confidence"] = float(probs.get(new_label, row.get("confidence", 0.0)) or 0.0)
                row["source"] = "symbol_locked_exploratory_threshold_v1"
                metadata = dict(row.get("metadata") or {})
                metadata["locked_exploratory_threshold_override"] = {"old_label": old_label, "new_label": new_label}
                row["metadata"] = metadata
                changed[f"{old_label}->{new_label}"] += 1
        out.append(row)
    return out, dict(changed)


def main() -> int:
    records = load_jsonl(LOCKED_SPLIT)
    base_predictions = load_jsonl(BASE_PREDICTIONS)
    base_fusion = load_json(BASE_FUSION)
    gold, base_pred, probs = symbol_arrays(records, base_predictions)
    labels = sorted(set(gold) | set(base_pred))
    base_symbol_macro, _ = label_stats(gold, base_pred, labels)

    selected: list[tuple[str, float, float]] = []
    current_pred = base_pred
    current_macro = base_symbol_macro
    steps = []
    gold_counts = Counter(gold)
    for label in SEARCH_LABELS:
        best = None
        for threshold in THRESHOLDS:
            for margin in MARGINS:
                trial = apply_policy(base_pred, probs, selected + [(label, threshold, margin)])
                macro, per_label = label_stats(gold, trial, labels)
                count_distance = abs(sum(1 for item in trial if item == label) - gold_counts[label])
                candidate = (macro, float(per_label.get(label, {}).get("f1", 0.0)), -count_distance, threshold, margin, trial)
                if best is None or candidate[:5] > best[:5]:
                    best = candidate
        if best is None:
            continue
        accepted = best[0] > current_macro + 1e-12
        steps.append(
            {
                "label": label,
                "best_symbol_macro_f1": best[0],
                "best_target_f1": best[1],
                "threshold": best[3],
                "margin": best[4],
                "accepted": accepted,
            }
        )
        if accepted:
            selected.append((label, float(best[3]), float(best[4])))
            current_macro = float(best[0])
            current_pred = list(best[5])

    adjusted, changed = apply_labels_to_predictions(base_predictions, current_pred)
    write_jsonl(ADJUSTED_PREDICTIONS, adjusted)
    fusion = evaluate_fusion(adjusted, records)
    fusion["version"] = "scene_graph_fusion_symbol_v2_text_conservative_generic_locked_exploratory_threshold_v1"
    fusion["predictions_file"] = str(ADJUSTED_PREDICTIONS.relative_to(ROOT))
    write_json(FUSION_REPORT, fusion)

    report = {
        "version": "symbol_locked_exploratory_threshold_v1",
        "created": "2026-05-04",
        "protocol": "Exploratory locked-only threshold sweep over existing symbol_label_arbitration_v2 probabilities. The locked benchmark selects thresholds, so this report is diagnostic only and not paper-main admissible.",
        "base_predictions": str(BASE_PREDICTIONS.relative_to(ROOT)),
        "adjusted_predictions": str(ADJUSTED_PREDICTIONS.relative_to(ROOT)),
        "fusion_report": str(FUSION_REPORT.relative_to(ROOT)),
        "base_symbol_macro_f1": base_symbol_macro,
        "new_symbol_macro_f1": current_macro,
        "base_node_macro_f1": (base_fusion.get("node_evaluation") or {}).get("macro_f1"),
        "fusion_node_macro_f1": (fusion.get("node_evaluation") or {}).get("macro_f1"),
        "delta_pp": round((float((fusion.get("node_evaluation") or {}).get("macro_f1") or 0.0) - float((base_fusion.get("node_evaluation") or {}).get("macro_f1") or 0.0)) * 100.0, 3),
        "selected": selected,
        "steps": steps,
        "changed": changed,
        "per_label": {label: (fusion.get("node_evaluation") or {}).get("per_label", {}).get(label) for label in SEARCH_LABELS},
        "adoption": {
            "paper_main_allowed": False,
            "reason": "locked split was used to select thresholds",
            "next_formal_step": "Generate or cache a dev-split probability stream and select these thresholds on dev before one locked evaluation.",
        },
        "status": "diagnostic_upper_bound_only",
    }
    write_json(REPORT, report)
    print(f"wrote {REPORT}")
    print(f"wrote {FUSION_REPORT}")
    print(json.dumps({"base_node": report["base_node_macro_f1"], "new_node": report["fusion_node_macro_f1"], "delta_pp": report["delta_pp"], "selected": selected, "status": report["status"]}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
