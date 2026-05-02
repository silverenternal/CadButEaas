#!/usr/bin/env python3
"""Evaluate SymbolFixture v9 classification F1 on CubiCasa5k dev split.

Computes true classification F1 by comparing gold symbol_type vs predicted label
for each symbol candidate, using candidate_id as the matching key.
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

def main():
    predictions_path = Path("reports/vlm/symbol_v9_eval/predictions.jsonl")
    gold_path = Path("datasets/cadstruct_real_world_benchmark_v1/room_space/cubicasa5k_reviewed_locked_test.jsonl")

    # Load gold symbol labels by candidate_id
    gold_by_id = {}
    for record in load_jsonl(gold_path):
        expected = record.get("expected_json") or {}
        for sc in expected.get("symbol_candidates") or []:
            cid = str(sc.get("id"))
            label = sc.get("symbol_type", "generic_symbol")
            gold_by_id[cid] = label

    print(f"Gold symbols: {len(gold_by_id)}")
    print(f"Gold distribution: {dict(Counter(gold_by_id.values()).most_common())}")

    # Load predictions
    preds_by_id = {}
    for pred in load_jsonl(predictions_path):
        if pred.get("family") == "symbol":
            cid = str(pred.get("candidate_id"))
            label = pred.get("label")
            preds_by_id[cid] = label

    print(f"\nPredictions: {len(preds_by_id)}")
    print(f"Pred distribution: {dict(Counter(preds_by_id.values()).most_common())}")

    # Match by ID and compute confusion matrix
    matched_ids = sorted(set(gold_by_id.keys()) & set(preds_by_id.keys()))
    print(f"\nMatched by ID: {len(matched_ids)}")

    gold_labels = [gold_by_id[cid] for cid in matched_ids]
    pred_labels = [preds_by_id[cid] for cid in matched_ids]

    # Compute confusion matrix
    labels = sorted(set(gold_labels) | set(pred_labels))
    confusion = defaultdict(Counter)
    for g, p in zip(gold_labels, pred_labels):
        confusion[g][p] += 1

    # Per-class F1
    print(f"\n{'='*60}")
    print(f"Per-class F1 (classification-level):")
    print(f"{'='*60}")
    f1s = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        f1s.append(f1)
        support = sum(confusion[label].values())
        print(f"  {label:15s}: P={prec:.3f} R={rec:.3f} F1={f1:.3f} (support={support})")

    macro_f1 = sum(f1s) / max(len(f1s), 1)
    accuracy = sum(confusion[l][l] for l in labels) / max(len(matched_ids), 1)

    print(f"\n{'='*60}")
    print(f"Results:")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"{'='*60}")

    # Save results
    results = {
        "model": "symbol_fixture_v9_extra_trees",
        "dataset": "cubicasa5k_reviewed_locked_test",
        "matched_symbols": len(matched_ids),
        "accuracy": round(accuracy, 4),
        "macro_f1": round(macro_f1, 4),
        "per_class": {},
    }
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[other][label] for other in labels if other != label)
        fn = sum(confusion[label][other] for other in labels if other != label)
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2 * prec * rec / max(prec + rec, 1e-12)
        results["per_class"][label] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(f1, 4),
            "support": sum(confusion[label].values()),
        }

    output_path = Path("reports/vlm/symbol_v9_classification_f1.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(results, indent=2, ensure_ascii=False) + "\n")
    print(f"\nResults saved to {output_path}")


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
