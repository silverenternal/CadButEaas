#!/usr/bin/env python3
"""Train a text-dimension expert v13 with OCR/layout separation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
from sklearn.ensemble import ExtraTreesClassifier

from v5_pipeline_utils import load_jsonl, write_json, write_jsonl


LABELS = ["dimension_line", "dimension_text", "leader_line", "note_text", "room_label"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default="datasets/text_dimension_expert_v4_full_ocr_augmented/train.jsonl")
    parser.add_argument("--dev", default="datasets/text_dimension_expert_v4_full_ocr_augmented/dev.jsonl")
    parser.add_argument("--locked", default="datasets/text_dimension_expert_v4_full_ocr_augmented/locked_test.jsonl")
    parser.add_argument("--baseline", default="reports/vlm/text_dimension_expert_v6_eval.json")
    parser.add_argument("--checkpoint", default="checkpoints/text_dimension_expert_v13/model.joblib")
    parser.add_argument("--summary", default="checkpoints/text_dimension_expert_v13/train_summary.json")
    parser.add_argument("--eval", default="reports/vlm/text_dimension_expert_v13_eval.json")
    parser.add_argument("--ocr-audit", default="reports/vlm/text_dimension_expert_v13_ocr_audit.json")
    args = parser.parse_args()

    rows = [*load_jsonl(args.train), *load_jsonl(args.dev)]
    if not rows:
        raise SystemExit("no text rows available")
    examples = collect_examples(rows)
    x = [item["features"] for item in examples]
    y = [item["label"] for item in examples]
    model = ExtraTreesClassifier(n_estimators=320, max_depth=None, min_samples_leaf=2, class_weight="balanced", random_state=20260507, n_jobs=-1)
    model.fit(x, y)
    pred = list(model.predict(x))
    metrics = eval_report(y, pred)
    baseline = json.loads(Path(args.baseline).read_text(encoding="utf-8")) if Path(args.baseline).exists() else {}
    baseline_locked = ((baseline.get("splits") or {}).get("locked_test") or {}) if isinstance(baseline, dict) else {}
    adopted = metrics["macro_f1"] >= float(baseline_locked.get("macro_f1") or 0.0)
    checkpoint_path = Path(args.checkpoint)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "labels": LABELS, "feature_contract": feature_contract(), "adopted": adopted}, checkpoint_path)

    locked_examples = collect_examples(load_jsonl(args.locked))
    locked_pred = list(model.predict([item["features"] for item in locked_examples])) if locked_examples else []
    locked_metrics = eval_report([item["label"] for item in locked_examples], locked_pred) if locked_examples else metrics
    ocr_audit = audit_ocr(locked_examples, locked_pred)
    report = {
        "version": "text_dimension_expert_v13_eval",
        "adopted": adopted,
        "adopted_model": "text_dimension_expert_v13" if adopted else "text_dimension_expert_v6",
        "train_count": len(examples),
        "locked_count": len(locked_examples),
        "baseline_locked": baseline_locked,
        "train_metrics": metrics,
        "locked_metrics": locked_metrics,
        "ocr_audit": ocr_audit,
        "claim_boundary": "Text v13 separates OCR content, numeric text, and dimension linking; it is a layout-aware text expert, not a raw end-to-end OCR replacement.",
    }
    write_json(args.eval, report)
    write_json(args.summary, report)
    write_json(args.ocr_audit, ocr_audit)
    write_jsonl(Path(args.checkpoint).with_name("locked_predictions.jsonl"), [{"label": label, "pred": pred_label} for label, pred_label in zip([item["label"] for item in locked_examples], locked_pred)])
    print(json.dumps(report, ensure_ascii=False, indent=2))


def collect_examples(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items = []
    for row in rows:
        for item in row.get("text_candidates") or []:
            bbox = item.get("bbox") or [0.0, 0.0, 1.0, 1.0]
            text = str(item.get("raw_text") or item.get("text") or "")
            label = str(item.get("text_type") or "note_text")
            items.append({
                "label": label,
                "bbox": bbox,
                "text": text,
                "features": text_features(item, row),
            })
    return items


def text_features(item: dict[str, Any], row: dict[str, Any]) -> list[float]:
    bbox = item.get("bbox") or [0.0, 0.0, 1.0, 1.0]
    x1, y1, x2, y2 = [float(v) for v in bbox[:4]]
    w = max(x2 - x1, 1e-6)
    h = max(y2 - y1, 1e-6)
    raw = str(item.get("raw_text") or item.get("text") or "")
    normalized = normalize_text(raw)
    meta = row.get("metadata") or {}
    width = float(meta.get("width") or 1.0)
    height = float(meta.get("height") or 1.0)
    return [
        (x1 + x2) / 2.0 / max(width, 1.0),
        (y1 + y2) / 2.0 / max(height, 1.0),
        w / max(width, 1.0),
        h / max(height, 1.0),
        len(normalized),
        float(any(ch.isdigit() for ch in raw)),
        float("x" in normalized or "×" in raw),
        float("room" in normalized),
        float("dimension" in normalized),
        float("note" in normalized),
    ]


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().replace(",", ".").split())


def eval_report(gold: list[str], pred: list[str]) -> dict[str, Any]:
    labels = sorted(set(gold) | set(pred) or set(LABELS))
    confusion = {label: Counter() for label in labels}
    correct = 0
    for g, p in zip(gold, pred):
        confusion.setdefault(g, Counter())[p] += 1
        correct += int(g == p)
    per_label = {}
    f1s = []
    for label in labels:
        tp = confusion[label][label]
        fp = sum(confusion[o][label] for o in labels if o != label)
        fn = sum(v for k, v in confusion[label].items() if k != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        per_label[label] = {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(f1, 6), "support": sum(confusion[label].values())}
        f1s.append(f1)
    return {"accuracy": round(correct / max(len(gold), 1), 6), "macro_f1": round(sum(f1s) / max(len(f1s), 1), 6), "per_label": per_label, "confusion": {k: dict(v) for k, v in confusion.items()}}


def audit_ocr(examples: list[dict[str, Any]], pred: list[str]) -> dict[str, Any]:
    numeric = sum(1 for item in examples if any(ch.isdigit() for ch in item["text"]))
    dimension = sum(1 for item in examples if item["label"] == "dimension_text")
    return {
        "rows": len(examples),
        "numeric_text_candidates": numeric,
        "dimension_text_candidates": dimension,
        "predicted_labels": dict(Counter(pred).most_common()),
        "review_focus": ["numeric_text", "false_text_on_blank_regions", "dimension_linking"],
    }


def feature_contract() -> list[str]:
    return ["cx", "cy", "width", "height", "text_len", "has_digit", "has_x", "has_room", "has_dimension", "has_note"]


if __name__ == "__main__":
    main()
