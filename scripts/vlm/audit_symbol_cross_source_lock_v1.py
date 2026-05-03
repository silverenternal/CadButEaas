#!/usr/bin/env python3
"""Lock cross-source symbol annotation status for paper claims."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"
PENDING = REPORTS / "symbol_cross_source_annotation_pack_v1.pending.jsonl"
PACK = REPORTS / "symbol_cross_source_annotation_pack_v1.json"
LONG_TAIL = REPORTS / "symbol_conservative_arbitration_v1.json"
OUTPUT = REPORTS / "symbol_cross_source_lock_v1.json"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def f1(tp: int, fp: int, fn: int) -> dict[str, float | int]:
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    score = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {"precision": round(precision, 6), "recall": round(recall, 6), "f1": round(score, 6), "tp": tp, "fp": fp, "fn": fn}


def main() -> int:
    rows = load_jsonl(PENDING)
    pack = load_json(PACK)
    long_tail = load_json(LONG_TAIL)
    gold_rows: list[tuple[str, str]] = []
    drawings_with_gold = 0
    suggested = Counter()
    gold_labels = Counter()
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

    status = "passed_cross_source_symbol_smoke" if enough_gold else "pending_no_human_gold"
    report = {
        "version": "symbol_cross_source_lock_v1",
        "created": "2026-05-03",
        "status": status,
        "input_pack": str(PACK.relative_to(ROOT)),
        "pending_jsonl": str(PENDING.relative_to(ROOT)),
        "selected_drawings": len(rows) or pack.get("selected_drawings"),
        "human_gold": {
            "drawings_with_gold_9class_symbol_type": drawings_with_gold,
            "gold_symbol_annotations": len(gold_rows),
            "minimum_drawings_for_smoke": 20,
            "minimum_symbols_for_smoke": 200,
        },
        "metrics": {
            "accuracy": accuracy,
            "macro_f1": macro_f1,
            "per_label": per_label,
        },
        "annotation_prior": {
            "suggested_label_counts": dict(suggested.most_common()),
            "gold_label_counts": dict(gold_labels.most_common()),
        },
        "long_tail_boundary": {
            "source": str(LONG_TAIL.relative_to(ROOT)) if LONG_TAIL.exists() else None,
            "adoption_recommendation": long_tail.get("adoption_recommendation"),
            "risk_labels": ["generic_symbol", "bathtub", "equipment", "table"],
            "paper_role": "appendix_limitation_until_cross_source_gold_is_available",
        },
        "claim_boundary": {
            "cross_source_symbol_generalization_claim_allowed": enough_gold,
            "allowed_text": "Treat the FloorPlanCAD symbol pack as annotation-only unless this lock reports passed_cross_source_symbol_smoke.",
        },
    }
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {OUTPUT}")
    print(json.dumps({"status": status, "drawings_with_gold": drawings_with_gold, "gold_symbols": len(gold_rows)}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
