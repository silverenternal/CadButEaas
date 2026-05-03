#!/usr/bin/env python3
"""Build TextDimension v5 evaluation from the v4 ExtraTrees checkpoint.

v5 keeps the v4 model fixed and adds one calibrated, non-label-leaking rule:
low-confidence `note_text` predictions are rerouted to the highest-probability
non-note class. The cutoff is selected on the training split and then frozen
for dev/smoke/locked_test.
"""

from __future__ import annotations

import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import joblib

ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(ROOT / "scripts" / "vlm"))

import train_text_dimension_expert_v4 as v4  # noqa: E402

DATA_DIR = ROOT / "datasets" / "text_dimension_expert_v4_full_ocr_augmented"
CHECKPOINT_DIR = ROOT / "checkpoints" / "text_dimension_expert_v4_aug2"
OUTPUT_DIR = ROOT / "checkpoints" / "text_dimension_expert_v5"
REPORT_PATH = ROOT / "reports" / "vlm" / "text_dimension_expert_v5_eval.json"
LABELS = ["dimension_line", "dimension_text", "leader_line", "note_text", "room_label"]


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def metric_from_pairs(gold: list[str], pred: list[str]) -> dict[str, Any]:
    confusion: dict[str, Counter[str]] = {label: Counter() for label in LABELS}
    for g, p in zip(gold, pred):
        confusion.setdefault(g, Counter())[p] += 1
    per_label, macro_f1 = v4.classification_report(LABELS, confusion)
    correct = sum(int(g == p) for g, p in zip(gold, pred))
    return {
        "text_candidates": len(gold),
        "accuracy": correct / max(len(gold), 1),
        "macro_f1": macro_f1,
        "per_label": per_label,
        "confusion": {label: dict(counts) for label, counts in confusion.items()},
    }


def reroute_low_confidence_note(
    base_pred: list[str],
    probabilities: Any,
    class_names: list[str],
    threshold: float,
) -> list[str]:
    note_idx = class_names.index("note_text")
    result: list[str] = []
    for pred, probs in zip(base_pred, probabilities):
        if pred == "note_text" and float(probs[note_idx]) < threshold:
            best_idx = max(
                (idx for idx, label in enumerate(class_names) if label != "note_text"),
                key=lambda idx: float(probs[idx]),
            )
            result.append(class_names[best_idx])
        else:
            result.append(pred)
    return result


def flatten_split(split_name: str) -> tuple[list[dict[str, Any]], list[dict[str, float]], list[str]]:
    rows = v4.load_jsonl(DATA_DIR / f"{split_name}.jsonl")
    features, labels, _ = v4.collect_samples(rows)
    return rows, features, labels


def predict_split(bundle: dict[str, Any], split_name: str, threshold: float) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = bundle["classifier"]
    encoder = bundle["encoder"]
    class_names = list(encoder.classes_)

    rows, features, gold = flatten_split(split_name)
    vectors = [v4.feature_vector(feat) for feat in features]
    base_pred = list(encoder.inverse_transform(model.predict(vectors)))
    probabilities = model.predict_proba(vectors)
    v5_pred = reroute_low_confidence_note(base_pred, probabilities, class_names, threshold)

    pred_rows: list[dict[str, Any]] = []
    item_idx = 0
    for row in rows:
        row_items: list[dict[str, Any]] = []
        for orig_item in row.get("text_candidates") or []:
            row_items.append({
                "id": orig_item.get("id"),
                "gold": orig_item.get("text_type"),
                "prediction": v5_pred[item_idx],
                "baseline_prediction": base_pred[item_idx],
                "bbox": orig_item.get("bbox"),
            })
            item_idx += 1
        links = v4.predict_dimension_links(row_items, row.get("text_candidates") or [])
        pred_rows.append({
            "image": row.get("image"),
            "annotation": row.get("annotation"),
            "source_dataset": row.get("source_dataset"),
            "text_candidates": row_items,
            "dimension_links_gold": row.get("dimension_links") or [],
            "dimension_links_pred": links,
        })

    result = v4.evaluate_predictions(pred_rows)
    base_result = metric_from_pairs(gold, base_pred)
    result["baseline_v4"] = {
        "accuracy": base_result["accuracy"],
        "macro_f1": base_result["macro_f1"],
        "note_text_f1": base_result["per_label"]["note_text"]["f1"],
        "predicted_note_text": Counter(base_pred).get("note_text", 0),
    }
    result["calibrated_note_text"] = {
        "threshold": threshold,
        "predicted_note_text": Counter(v5_pred).get("note_text", 0),
        "rerouted_from_note_text": sum(
            int(before == "note_text" and after != "note_text")
            for before, after in zip(base_pred, v5_pred)
        ),
    }
    return pred_rows, result


def select_threshold(bundle: dict[str, Any]) -> dict[str, Any]:
    model = bundle["classifier"]
    encoder = bundle["encoder"]
    class_names = list(encoder.classes_)
    _, features, gold = flatten_split("train")
    vectors = [v4.feature_vector(feat) for feat in features]
    base_pred = list(encoder.inverse_transform(model.predict(vectors)))
    probabilities = model.predict_proba(vectors)

    candidates = [round(value / 100, 2) for value in range(50, 91)]
    best: dict[str, Any] | None = None
    for threshold in candidates:
        pred = reroute_low_confidence_note(base_pred, probabilities, class_names, threshold)
        report = metric_from_pairs(gold, pred)
        note_f1 = report["per_label"]["note_text"]["f1"]
        score = (report["macro_f1"], note_f1, -abs(threshold - 0.7))
        candidate = {
            "threshold": threshold,
            "macro_f1": report["macro_f1"],
            "note_text_f1": note_f1,
            "predicted_note_text": Counter(pred).get("note_text", 0),
            "score": score,
        }
        if best is None or candidate["score"] > best["score"]:
            best = candidate

    assert best is not None
    best.pop("score", None)
    return best


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    bundle = joblib.load(CHECKPOINT_DIR / "model_v4.joblib")
    threshold_report = select_threshold(bundle)
    threshold = float(threshold_report["threshold"])

    summary: dict[str, Any] = {
        "model_type": "text_dimension_extra_trees_v5_calibrated_note_gate",
        "source_checkpoint": str(CHECKPOINT_DIR / "model_v4.joblib"),
        "input_dir": str(DATA_DIR),
        "calibration": {
            "split": "train",
            "rule": "if prediction == note_text and P(note_text) < threshold, reroute to best non-note class",
            "selected": threshold_report,
        },
        "splits": {},
        "done_when_check": {},
    }

    for split_name in ("train", "dev", "smoke", "locked_test"):
        pred_rows, result = predict_split(bundle, split_name, threshold)
        v4.write_jsonl(OUTPUT_DIR / f"{split_name}_predictions.jsonl", pred_rows)
        summary["splits"][split_name] = result

    dev = summary["splits"]["dev"]
    note_f1 = dev["per_label"]["note_text"]["f1"]
    paper_tables_script = ROOT / "scripts" / "vlm" / "generate_paper_tables_v2.py"
    paper_tables_uses_v5 = (
        paper_tables_script.exists()
        and "text_dimension_expert_v5_eval.json" in paper_tables_script.read_text(encoding="utf-8")
    )
    summary["done_when_check"] = {
        "report_generated": True,
        "dev_macro_f1_ge_0_95": dev["macro_f1"] >= 0.95,
        "note_text_f1_ge_0_80": note_f1 >= 0.80,
        "paper_tables_v2_uses_latest_textdimension": paper_tables_uses_v5,
    }
    summary["status"] = "passed_metric_gate" if all(
        value for key, value in summary["done_when_check"].items() if key != "paper_tables_v2_uses_latest_textdimension"
    ) else "attempted_not_passed"

    write_json(OUTPUT_DIR / "train_summary.json", summary)
    write_json(REPORT_PATH, summary)
    print(f"wrote {REPORT_PATH}")
    print(f"dev macro_f1={dev['macro_f1']:.6f} note_text_f1={note_f1:.6f} threshold={threshold:.2f}")


if __name__ == "__main__":
    main()
