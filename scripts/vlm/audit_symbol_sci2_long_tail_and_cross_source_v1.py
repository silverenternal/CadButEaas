#!/usr/bin/env python3
"""SCI2 symbol long-tail report and cross-source lock."""

from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
PENDING = REPORTS / "symbol_cross_source_annotation_pack_v1.pending.jsonl"
PACK = REPORTS / "symbol_cross_source_annotation_pack_v1.json"
CONSERVATIVE = REPORTS / "symbol_conservative_arbitration_v1.json"
ERROR_PACK = REPORTS / "symbol_long_tail_error_pack_v1.jsonl"
LONG_TAIL_OUT = REPORTS / "symbol_long_tail_sci2_boost_v1.json"
CROSS_LOCK_OUT = REPORTS / "symbol_cross_source_lock_v2.json"
QUEUE_OUT = REPORTS / "symbol_long_tail_sci2_annotation_queue_v1.jsonl"

RISK_LABELS = ["generic_symbol", "bathtub", "equipment", "table", "sink", "shower"]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, data: Any) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def f1(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(score, 6), "tp": tp, "fp": fp, "fn": fn}


def cross_source_lock(rows: list[dict[str, Any]], pack: dict[str, Any]) -> dict[str, Any]:
    gold_rows: list[tuple[str, str]] = []
    drawings_with_gold = 0
    suggested = Counter()
    gold_labels = Counter()
    confusion = Counter()
    for row in rows:
        row_has_gold = False
        for ann in row.get("symbol_annotations") or []:
            pred = str(ann.get("suggested_9class_symbol_type") or "").strip()
            gold = str(ann.get("gold_9class_symbol_type") or "").strip()
            if pred:
                suggested[pred] += 1
            if gold:
                row_has_gold = True
                gold_rows.append((pred, gold))
                gold_labels[gold] += 1
                confusion[(gold, pred)] += 1
        if row_has_gold:
            drawings_with_gold += 1

    enough_gold = drawings_with_gold >= 20 or len(gold_rows) >= 200
    per_label: dict[str, Any] = {}
    macro_f1 = None
    accuracy = None
    if enough_gold:
        labels = sorted({label for pair in gold_rows for label in pair if label})
        correct = sum(1 for pred, gold in gold_rows if pred == gold)
        accuracy = round(correct / len(gold_rows), 6) if gold_rows else 0.0
        scores = []
        for label in labels:
            tp = sum(1 for pred, gold in gold_rows if pred == label and gold == label)
            fp = sum(1 for pred, gold in gold_rows if pred == label and gold != label)
            fn = sum(1 for pred, gold in gold_rows if pred != label and gold == label)
            per_label[label] = f1(tp, fp, fn)
            scores.append(float(per_label[label]["f1"]))
        macro_f1 = round(sum(scores) / len(scores), 6) if scores else 0.0

    return {
        "version": "symbol_cross_source_lock_v2",
        "created": "2026-05-03",
        "status": "passed_cross_source_symbol_smoke" if enough_gold else "pending_no_human_gold",
        "input_pack": str(PACK.relative_to(ROOT)),
        "pending_jsonl": str(PENDING.relative_to(ROOT)),
        "selected_drawings": len(rows) or pack.get("selected_drawings"),
        "human_gold": {
            "drawings_with_gold_9class_symbol_type": drawings_with_gold,
            "gold_symbol_annotations": len(gold_rows),
            "minimum_drawings_for_smoke": 20,
            "minimum_symbols_for_smoke": 200,
        },
        "metrics": {"accuracy": accuracy, "macro_f1": macro_f1, "per_label": per_label},
        "confusion_matrix": {f"{gold}->{pred}": count for (gold, pred), count in confusion.most_common()},
        "annotation_prior": {"suggested_label_counts": dict(suggested.most_common()), "gold_label_counts": dict(gold_labels.most_common())},
        "claim_boundary": {
            "cross_source_symbol_generalization_claim_allowed": enough_gold,
            "allowed_text": "Treat cross-source symbols as annotation-only unless this lock reports passed_cross_source_symbol_smoke.",
            "disallowed_text": "Do not use CubiCasa symbol accuracy as cross-source generalization evidence.",
        },
    }


def build_queue(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    queue: list[dict[str, Any]] = []
    priority = {label: rank for rank, label in enumerate(RISK_LABELS)}
    for row in rows:
        anns = row.get("symbol_annotations") or []
        risk_count = 0
        for ann in anns:
            pred = str(ann.get("suggested_9class_symbol_type") or ann.get("source_label") or "").strip()
            source = str(ann.get("source_label") or "").strip()
            if pred in priority or source in priority:
                risk_count += 1
        if risk_count:
            queue.append(
                {
                    "image_path": row.get("image_path"),
                    "source_dataset": row.get("source_dataset"),
                    "risk_symbol_count": risk_count,
                    "annotation_priority": "P0" if risk_count >= 3 else "P1",
                    "focus_labels": RISK_LABELS,
                    "required_field": "gold_9class_symbol_type",
                    "qa_note": "Prioritize generic/equipment/bathtub/table-like and sink-shower confusion.",
                }
            )
    queue.sort(key=lambda item: (item["annotation_priority"], -int(item["risk_symbol_count"]), str(item.get("image_path"))))
    return queue


def long_tail_report(conservative: dict[str, Any], queue: list[dict[str, Any]]) -> dict[str, Any]:
    full = conservative.get("full_arbitration") or {}
    per_label = full.get("long_tail_per_label") or {}
    supports = {label: metrics.get("support") for label, metrics in per_label.items()}
    weak = {
        label: metrics
        for label, metrics in per_label.items()
        if float(metrics.get("f1") or 0.0) < 0.9
    }
    attempted_macro = conservative.get("symbol_fixture_macro_f1") or conservative.get("macro_f1")
    return {
        "version": "symbol_long_tail_sci2_boost_v1",
        "created": "2026-05-03",
        "status": "ceiling_requires_more_training_or_gold" if weak else "long_tail_target_met",
        "source": str(CONSERVATIVE.relative_to(ROOT)) if CONSERVATIVE.exists() else None,
        "error_pack": str(ERROR_PACK.relative_to(ROOT)) if ERROR_PACK.exists() else None,
        "annotation_queue": str(QUEUE_OUT.relative_to(ROOT)),
        "current_internal_boundary": {
            "node_macro_f1_with_arbitration": full.get("node_macro_f1"),
            "symbol_fixture_macro_f1_if_available": attempted_macro,
            "long_tail_per_label": per_label,
            "weak_labels_below_0_90_f1": weak,
            "support_by_label": supports,
        },
        "sci2_boost_attempt": {
            "implemented_this_run": "audit_and_annotation_queue",
            "not_run_reason": "No filled cross-source gold and no new training budget/checkpoint in todo; current evidence is a reproducible ceiling, not a new model improvement.",
            "recommended_next_training": [
                "class-balanced loss or sampler for generic_symbol/bathtub/equipment",
                "hard-negative mining for equipment/table-like and sink/shower confusion",
                "raster crop CNN/ViT baseline evaluated with macro F1 and per-label confusion",
            ],
        },
        "queue_summary": {
            "rows": len(queue),
            "p0_rows": sum(1 for row in queue if row["annotation_priority"] == "P0"),
        },
        "claim_boundary": {
            "macro_f1_target_0_90_met": False,
            "use_accuracy_as_macro_f1": False,
            "paper_role": "long-tail ceiling and annotation plan until a stronger checkpoint or human gold is available.",
        },
    }


def main() -> int:
    rows = load_jsonl(PENDING)
    pack = load_json(PACK)
    conservative = load_json(CONSERVATIVE)
    queue = build_queue(rows)
    write_jsonl(QUEUE_OUT, queue)
    write_json(CROSS_LOCK_OUT, cross_source_lock(rows, pack))
    write_json(LONG_TAIL_OUT, long_tail_report(conservative, queue))
    print(f"wrote {LONG_TAIL_OUT}")
    print(f"wrote {CROSS_LOCK_OUT}")
    print(f"wrote {QUEUE_OUT}")
    print(json.dumps({"queue_rows": len(queue), "cross_source_status": load_json(CROSS_LOCK_OUT).get("status")}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
