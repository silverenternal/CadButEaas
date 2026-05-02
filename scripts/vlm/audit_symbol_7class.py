#!/usr/bin/env python3
"""Symbol Fixture 7-class audit: collapse 9->7 classes and re-evaluate with abstention.

Collapses:
  table -> generic_symbol
  sanitary_fixture -> sink

Keeps: shower, bathtub, stair, column, equipment, appliance, generic_symbol, sink, furniture

Evaluates:
  - 7-class F1, precision, recall (macro and per-class)
  - Same metrics with abstention (confidence < threshold -> "abstain", not counted as error)

Reads: reports/vlm/real_upstream_predictions_dev.jsonl
       datasets/.../cubicasa5k_reviewed_locked_test.jsonl (gold labels)
Writes: reports/vlm/symbol_fixture_7class_audit.json
"""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent

PREDICTIONS_PATH = ROOT / "reports/vlm/real_upstream_predictions_dev.jsonl"
GOLD_PATH = ROOT / "datasets/cadstruct_real_world_benchmark_v1/room_space/cubicasa5k_reviewed_locked_test.jsonl"
OUTPUT_PATH = ROOT / "reports/vlm/symbol_fixture_7class_audit.json"

# 9 -> 7 collapse mapping
COLLAPSE_MAP = {
    "table": "generic_symbol",
    "sanitary_fixture": "sink",
}

KEEP_CLASSES_7 = {"shower", "bathtub", "stair", "column", "equipment", "appliance", "generic_symbol", "sink", "furniture"}


def collapse_label(label: str) -> str:
    """Map a 9-class label to a 7-class label."""
    return COLLAPSE_MAP.get(label, label)


def load_predictions() -> dict[str, dict]:
    """Load symbol predictions, deduplicating by taking highest-confidence prediction per ID."""
    raw_preds: list[dict] = []
    with open(PREDICTIONS_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                raw_preds.append(json.loads(line))

    symbol_preds = [p for p in raw_preds if p.get("expert") == "symbol_fixture"]
    print(f"Raw symbol predictions: {len(symbol_preds)}")

    # Deduplicate: keep highest-confidence prediction per candidate_id
    best_by_id: dict[str, dict] = {}
    for p in symbol_preds:
        cid = str(p["candidate_id"])
        if cid not in best_by_id or p.get("confidence", 0) > best_by_id[cid].get("confidence", 0):
            best_by_id[cid] = p

    print(f"Deduplicated symbol predictions: {len(best_by_id)}")
    return best_by_id


def load_gold_labels() -> dict[str, str]:
    """Return {candidate_id: label} for all gold symbols."""
    gold = {}
    with open(GOLD_PATH, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            record = json.loads(line)
            expected = record.get("expected_json") or {}
            for sc in expected.get("symbol_candidates") or []:
                cid = str(sc.get("id"))
                label = sc.get("symbol_type", sc.get("label", "unknown"))
                gold[cid] = label
    return gold


def compute_metrics(
    gold_labels: dict[str, str],
    pred_labels: dict[str, str],
    classes: list[str],
    collapse_fn,
    abstain_threshold: float | None = None,
    confidences: dict[str, float] | None = None,
) -> dict:
    """Compute per-class and macro F1, precision, recall.

    If abstain_threshold is set, predictions with confidence < threshold are
    marked "abstain" and excluded from both numerator and denominator.
    """
    # Collapse gold and pred labels
    gold_collapsed = {cid: collapse_fn(lbl) for cid, lbl in gold_labels.items()}
    pred_collapsed = {cid: collapse_fn(lbl) for cid, lbl in pred_labels.items()}

    # Apply abstention
    if abstain_threshold is not None and confidences:
        pred_final = {}
        abstain_count = 0
        for cid, lbl in pred_collapsed.items():
            conf = confidences.get(cid, 1.0)
            if conf < abstain_threshold:
                pred_final[cid] = "abstain"
                abstain_count += 1
            else:
                pred_final[cid] = lbl
    else:
        pred_final = pred_collapsed
        abstain_count = 0

    # Only evaluate on non-abstain predictions
    active_preds = {cid: lbl for cid, lbl in pred_final.items() if lbl != "abstain"}

    # Per-class metrics
    per_class = {}
    for cls in classes:
        tp = sum(
            1 for cid in active_preds
            if cid in gold_collapsed and active_preds[cid] == cls and gold_collapsed[cid] == cls
        )
        fp = sum(
            1 for cid in active_preds
            if cid in gold_collapsed and (active_preds[cid] == cls and gold_collapsed[cid] != cls)
        )
        # Also count predictions for ids not in gold as FP
        fp += sum(
            1 for cid in active_preds
            if cid not in gold_collapsed and active_preds[cid] == cls
        )
        fn = sum(
            1 for cid in gold_collapsed
            if cid not in active_preds or active_preds.get(cid) != cls
            if gold_collapsed[cid] == cls
        )

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        per_class[cls] = {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": sum(1 for cid in gold_collapsed if gold_collapsed[cid] == cls),
        }

    # Macro averages
    macro_precision = sum(v["precision"] for v in per_class.values()) / len(classes)
    macro_recall = sum(v["recall"] for v in per_class.values()) / len(classes)
    macro_f1 = sum(v["f1"] for v in per_class.values()) / len(classes)

    return {
        "per_class": per_class,
        "macro_precision": round(macro_precision, 6),
        "macro_recall": round(macro_recall, 6),
        "macro_f1": round(macro_f1, 6),
        "abstain_count": abstain_count,
        "active_predictions": len(active_preds),
        "total_gold": len(gold_collapsed),
    }


def main() -> None:
    print("=== Symbol Fixture 7-Class Audit ===\n")

    # Load data
    pred_dict = load_predictions()  # {candidate_id: prediction}
    gold_labels_raw = load_gold_labels()

    print(f"Gold symbols: {len(gold_labels_raw)}")

    # Build gold label distribution (9-class)
    gold_counter = Counter(gold_labels_raw.values())
    print(f"\nGold label distribution (9-class):")
    for lbl, cnt in gold_counter.most_common():
        print(f"  {lbl}: {cnt}")

    # Build pred labels and confidences from deduplicated dict
    pred_labels_raw = {cid: p["label"] for cid, p in pred_dict.items()}
    confidences = {cid: p.get("confidence", 1.0) for cid, p in pred_dict.items()}

    pred_counter = Counter(pred_labels_raw.values())
    print(f"\nPrediction label distribution (9-class):")
    for lbl, cnt in pred_counter.most_common():
        print(f"  {lbl}: {cnt}")

    # 7-class labels
    classes_7 = sorted(KEEP_CLASSES_7)

    # Evaluate without abstention
    print("\n--- 7-class evaluation (no abstention) ---")
    results_no_abstain = compute_metrics(gold_labels_raw, pred_labels_raw, classes_7, collapse_label)
    print(f"Macro F1: {results_no_abstain['macro_f1']:.4f}")
    print(f"Macro Precision: {results_no_abstain['macro_precision']:.4f}")
    print(f"Macro Recall: {results_no_abstain['macro_recall']:.4f}")

    # Evaluate with abstention at various thresholds
    abstention_thresholds = [0.5, 0.6, 0.7, 0.8, 0.9]
    results_abstention = {}
    for threshold in abstention_thresholds:
        print(f"\n--- 7-class evaluation (abstention < {threshold}) ---")
        result = compute_metrics(
            gold_labels_raw, pred_labels_raw, classes_7, collapse_label,
            abstain_threshold=threshold, confidences=confidences
        )
        results_abstention[str(threshold)] = result
        print(f"Macro F1: {result['macro_f1']:.4f}")
        print(f"Abstained: {result['abstain_count']} / {result['total_gold']} ({result['abstain_count']/result['total_gold']*100:.1f}%)")
        print(f"Active predictions: {result['active_predictions']}")

    # Also compute 9-class baseline for comparison
    classes_9 = sorted(set(gold_counter.keys()) | set(pred_counter.keys()))
    results_9class = compute_metrics(gold_labels_raw, pred_labels_raw, classes_9, lambda x: x)
    print(f"\n--- 9-class baseline (no abstention) ---")
    print(f"Macro F1: {results_9class['macro_f1']:.4f}")

    # Per-class detail for 7-class no abstention
    print("\n--- Per-class detail (7-class, no abstention) ---")
    for cls in classes_7:
        m = results_no_abstain["per_class"][cls]
        print(f"  {cls}: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} support={m['support']}")

    # Build report
    report = {
        "description": "Symbol Fixture 7-class audit: collapse 9->7 classes (table->generic_symbol, sanitary_fixture->sink)",
        "collapse_map": COLLAPSE_MAP,
        "classes_7": classes_7,
        "gold_label_distribution_9class": dict(gold_counter.most_common()),
        "pred_label_distribution_9class": dict(pred_counter.most_common()),
        "gold_label_distribution_7class": dict(Counter(collapse_label(lbl) for lbl in gold_labels_raw.values()).most_common()),
        "pred_label_distribution_7class": dict(Counter(collapse_label(lbl) for lbl in pred_labels_raw.values()).most_common()),
        "nine_class_baseline": {
            "macro_f1": results_9class["macro_f1"],
            "macro_precision": results_9class["macro_precision"],
            "macro_recall": results_9class["macro_recall"],
        },
        "seven_class_no_abstention": {
            "macro_f1": results_no_abstain["macro_f1"],
            "macro_precision": results_no_abstain["macro_precision"],
            "macro_recall": results_no_abstain["macro_recall"],
            "per_class": results_no_abstain["per_class"],
        },
        "seven_class_with_abstention": {
            str(t): {
                "macro_f1": r["macro_f1"],
                "macro_precision": r["macro_precision"],
                "macro_recall": r["macro_recall"],
                "abstain_count": r["abstain_count"],
                "abstain_rate": round(r["abstain_count"] / r["total_gold"], 4),
                "active_predictions": r["active_predictions"],
            }
            for t, r in results_abstention.items()
        },
        "abstention_threshold_0.7_detail": results_abstention.get("0.7", {}),
    }

    # Save report
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    print(f"\nReport saved to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
