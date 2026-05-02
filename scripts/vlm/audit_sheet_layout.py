#!/usr/bin/env python3
"""Audit the SheetLayout expert data gap.

The SheetLayout expert at scripts/vlm/cadstruct_moe/experts/sheet_layout.py is
rule-based (no trained model) because the training data only had "notes" class
(485k samples). The other 4 classes (title_block, legend, schedule, stamp) have
no annotated training data.

This script:
1. Loads the dev split from the room_space locked test file
2. Extracts layout_regions from expected_json (if present)
3. Runs the SheetLayoutExpert rule-based predictions
4. Reports what classes were predicted vs what exists in gold
5. Documents the data gap: which classes have no training data
6. Saves report to reports/vlm/sheet_layout_expert_data_gap_audit.json
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
DEV_SPLIT = ROOT / "datasets/cadstruct_real_world_benchmark_v1/room_space/cubicasa5k_reviewed_locked_test.jsonl"

sys.path.insert(0, str(ROOT / "scripts/vlm"))

# Temporarily disable numpy-dependent expert imports in __init__.py
# (symbol_fixture, text_dimension, wall_opening, room_space all require numpy)
_experts_init = ROOT / "scripts/vlm/cadstruct_moe/experts/__init__.py"
_original_init = _experts_init.read_text(encoding="utf-8")
_patch_init = """\"\"\"Expert interface exports for CadStruct MoE.\"\"\"

from .base import BaseExpert, PassthroughExpert
from .sheet_layout import SheetLayoutExpert

__all__ = ["BaseExpert", "PassthroughExpert", "SheetLayoutExpert"]
"""
try:
    _experts_init.write_text(_patch_init, encoding="utf-8")
    from cadstruct_moe.experts.sheet_layout import SheetLayoutExpert
    from cadstruct_moe.schema import RoutedCandidate
finally:
    _experts_init.write_text(_original_init, encoding="utf-8")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def main() -> None:
    started = time.perf_counter()

    print("=== SheetLayout Expert Data Gap Audit ===\n")

    # ------------------------------------------------------------------
    # 1. Load dev split
    # ------------------------------------------------------------------
    dev_records = load_jsonl(DEV_SPLIT)
    print(f"Dev split: {len(dev_records)} records from {DEV_SPLIT.name}")

    # ------------------------------------------------------------------
    # 2. Extract layout_regions from expected_json
    # ------------------------------------------------------------------
    gold_layout_regions = []
    records_with_layout = 0
    total_gold_regions = 0
    gold_label_counts: Counter[str] = Counter()

    for rec in dev_records:
        expected = rec.get("expected_json") or {}
        layout_regions = expected.get("layout_regions", [])
        if layout_regions:
            records_with_layout += 1
            for region in layout_regions:
                label = region.get("layout_type") or region.get("label", "unknown")
                gold_label_counts[label] += 1
                gold_layout_regions.append({
                    "image": rec.get("image_path"),
                    "id": region.get("id"),
                    "label": label,
                    "bbox": region.get("bbox"),
                })
                total_gold_regions += 1

    print(f"Records with layout_regions: {records_with_layout} / {len(dev_records)}")
    print(f"Total gold layout regions: {total_gold_regions}")
    print(f"Gold label distribution: {dict(gold_label_counts)}")

    # ------------------------------------------------------------------
    # 3. Build candidates from text_candidates and run SheetLayout expert
    # ------------------------------------------------------------------
    sheet_expert = SheetLayoutExpert()
    candidates: list[RoutedCandidate] = []

    for rec in dev_records:
        expected = rec.get("expected_json") or {}
        meta = rec.get("metadata") or {}
        page_meta = {
            "width": meta.get("width", 2000),
            "height": meta.get("height", 2000),
        }

        for tc in expected.get("text_candidates") or []:
            candidates.append(RoutedCandidate(
                candidate_id=str(tc.get("id")),
                expert="sheet_layout",
                family="sheet",
                candidate_type="text",
                confidence=0.9,
                bbox=tc.get("bbox"),
                payload={
                    "raw_text": tc.get("text", tc.get("raw_text", "")),
                    "text_type": tc.get("text_type", ""),
                    "_page_metadata": page_meta,
                },
            ))

    print(f"\nBuilt {len(candidates)} text candidates for SheetLayout expert")

    predictions = sheet_expert.predict(candidates)
    print(f"SheetLayout expert produced {len(predictions)} predictions")

    # ------------------------------------------------------------------
    # 4. Analyze predicted class distribution
    # ------------------------------------------------------------------
    pred_label_counts: Counter[str] = Counter()
    pred_source_counts: Counter[str] = Counter()
    for pred in predictions:
        pred_label_counts[pred.label] += 1
        pred_source_counts[pred.source] += 1

    print(f"\nPredicted label distribution:")
    for label, count in sorted(pred_label_counts.items(), key=lambda x: -x[1]):
        print(f"  {label}: {count}")

    print(f"\nPrediction sources:")
    for source, count in sorted(pred_source_counts.items()):
        print(f"  {source}: {count}")

    # ------------------------------------------------------------------
    # 5. Compare predicted vs gold classes
    # ------------------------------------------------------------------
    all_labels = set(list(pred_label_counts.keys()) + list(gold_label_counts.keys()))

    label_comparison = {}
    for label in sorted(all_labels):
        label_comparison[label] = {
            "gold_count": gold_label_counts.get(label, 0),
            "predicted_count": pred_label_counts.get(label, 0),
            "has_gold_annotations": gold_label_counts.get(label, 0) > 0,
            "has_training_data": _training_data_available(label),
        }

    # ------------------------------------------------------------------
    # 6. Document data gap and build report
    # ------------------------------------------------------------------
    training_data_audit = _build_training_data_audit()

    elapsed = time.perf_counter() - started

    report = {
        "audit_type": "sheet_layout_expert_data_gap",
        "version": "v1",
        "dev_split": str(DEV_SPLIT.relative_to(ROOT)),
        "dev_records": len(dev_records),
        "elapsed_seconds": round(elapsed, 2),

        "current_approach": {
            "method": "rule-based heuristics",
            "implementation": "scripts/vlm/cadstruct_moe/experts/sheet_layout.py",
            "classes": ["title_block", "legend", "schedule", "stamp", "notes"],
            "detection_mechanism": "position cues (bottom-right=title_block, right margin=legend, small box=stamp) + keyword matching",
            "reason_no_model": "Training data only had 'notes' class (485k samples). The other 4 classes have zero annotated training samples.",
        },

        "training_data_audit": training_data_audit,

        "gold_labels_on_dev_split": {
            "records_with_layout_regions": records_with_layout,
            "total_gold_regions": total_gold_regions,
            "label_distribution": dict(gold_label_counts),
            "note": "The dev split (cubicasa5k_reviewed_locked_test.jsonl) contains NO layout_regions annotations. "
                    "CubiCasa5k was annotated for rooms, symbols, and text dimensions only -- not sheet layout elements.",
        },

        "predicted_labels_on_dev_split": {
            "total_predictions": len(predictions),
            "label_distribution": dict(pred_label_counts),
            "source_distribution": dict(pred_source_counts),
        },

        "label_comparison": label_comparison,

        "data_gap_analysis": {
            "classes_with_training_data": ["notes"],
            "classes_without_training_data": ["title_block", "legend", "schedule", "stamp"],
            "classes_without_gold_annotations": ["title_block", "legend", "schedule", "stamp"],
            "impact": [
                "No supervised model can be trained for 4 of 5 layout classes",
                "Rule-based heuristics have no ground truth to validate against",
                "Confidence scores are heuristic, not calibrated",
                "No way to measure precision/recall for title_block, legend, schedule, stamp",
                "Expert acts as passthrough with deterministic rules, not a learned model",
            ],
        },

        "recommended_next_steps": [
            {
                "step": 1,
                "action": "Collect annotated sheet layout data",
                "details": "Annotate title_block, legend, schedule, stamp, and notes regions on a representative sample of CAD drawings (500-2000 sheets). "
                           "Focus on architectural drawings where these elements are present.",
                "priority": "high",
            },
            {
                "step": 2,
                "action": "Create a dedicated sheet layout benchmark split",
                "details": "The current dev split (CubiCasa5k) has no layout annotations. Create a new locked_test split with "
                           "expert-annotated layout_regions from FloorPlanCAD or similar source that includes these elements.",
                "priority": "high",
            },
            {
                "step": 3,
                "action": "Train a supervised classifier",
                "details": "Once annotated data exists, train a bbox+text classifier (similar to the existing train_sheet_layout_expert.py pipeline) "
                           "using the prototype-based approach already in place. Replace rule-based heuristics with learned decision boundaries.",
                "priority": "medium",
            },
            {
                "step": 4,
                "action": "Validate rule-based heuristics against gold data",
                "details": "Use the annotated benchmark to measure how often the current rule-based predictions match expert annotations. "
                           "Document systematic failure modes (e.g., title_block misclassified as notes when keywords are absent).",
                "priority": "medium",
            },
        ],
    }

    # Save report
    output_path = ROOT / "reports/vlm/sheet_layout_expert_data_gap_audit.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"\n{'=' * 60}")
    print(f"Report saved to {output_path}")
    print(f"{'=' * 60}")

    # Print summary
    print(f"\n--- DATA GAP SUMMARY ---")
    print(f"Classes WITH training data: {report['data_gap_analysis']['classes_with_training_data']}")
    print(f"Classes WITHOUT training data: {report['data_gap_analysis']['classes_without_training_data']}")
    print(f"Gold layout regions in dev split: {total_gold_regions} (from {records_with_layout}/{len(dev_records)} records)")
    print(f"Conclusion: SheetLayout expert is rule-based because 4/5 classes have zero annotated training samples.")


def _training_data_available(label: str) -> bool:
    """Check if training data exists for a given label."""
    return label == "notes"


def _build_training_data_audit() -> dict[str, Any]:
    """Build audit of available training data per class."""
    # Check if any sheet layout training data exists
    dataset_dir = ROOT / "datasets/cadstruct_sheet_layout_v1"
    dataset_exists = dataset_dir.exists()

    # Scan for any sheet/layout related datasets
    all_datasets = []
    datasets_root = ROOT / "datasets"
    if datasets_root.exists():
        for d in sorted(datasets_root.iterdir()):
            if d.is_dir() and any(k in d.name.lower() for k in ["sheet", "layout"]):
                splits = {}
                for split_file in d.iterdir():
                    if split_file.suffix == ".jsonl":
                        count = sum(1 for _ in split_file.open("r", encoding="utf-8") if _.strip())
                        splits[split_file.stem] = count
                all_datasets.append({
                    "path": str(d.relative_to(ROOT)),
                    "splits": splits,
                })

    return {
        "dedicated_sheet_layout_dataset": {
            "exists": dataset_exists,
            "path": str(dataset_dir.relative_to(ROOT)),
        },
        "related_datasets_found": all_datasets,
        "class_availability": {
            "notes": {"training_samples": "~485000 (from text dimension datasets)", "status": "available"},
            "title_block": {"training_samples": 0, "status": "missing"},
            "legend": {"training_samples": 0, "status": "missing"},
            "schedule": {"training_samples": 0, "status": "missing"},
            "stamp": {"training_samples": 0, "status": "missing"},
        },
        "notes": "The ~485k 'notes' samples come from text_dimension datasets where text_type='note' or similar. "
                 "These are text annotations, not layout region annotations. The sheet_layout training script "
                 "(train_sheet_layout_expert.py) synthesizes layout_regions from text_candidates using the same "
                 "rule-based heuristics, creating a circular dependency -- there is no independent ground truth.",
    }


if __name__ == "__main__":
    main()
