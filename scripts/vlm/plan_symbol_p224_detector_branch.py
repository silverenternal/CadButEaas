#!/usr/bin/env python3
"""Plan P224 stronger detector branch from P223 error-budget evidence."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
P223 = ROOT / "reports/vlm/symbol_p223_error_budget.json"
OUT_JSON = ROOT / "reports/vlm/symbol_p224_detector_branch_plan.json"
OUT_MD = ROOT / "reports/vlm/symbol_p224_detector_branch_plan.md"

DATASET_CANDIDATES = [
    ROOT / "datasets/symbol_recall_detector_p211_yolo_20k_server/build_report.json",
    ROOT / "datasets/symbol_tiled_recall_p205b_yolo_30k/build_report.json",
    ROOT / "datasets/symbol_residual_specialist_p213b_yolo/build_report.json",
    ROOT / "datasets/symbol_p221b_stair_specialist_yolo/build_report.json",
]
KEY_SCRIPTS = [
    "scripts/vlm/build_symbol_recall_detector_p211_data.py",
    "scripts/vlm/train_symbol_recall_detector_p211.py",
    "scripts/vlm/infer_symbol_p211_on_p206g_pages_p212.py",
    "scripts/vlm/eval_symbol_yolo_sliced_page_detector_v24.py",
    "scripts/vlm/build_symbol_tiled_recall_p205b_data.py",
    "scripts/vlm/train_symbol_tile_detector_v20.py",
    "scripts/vlm/build_symbol_residual_specialist_p213b_data.py",
]


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def summarize_dataset(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {"path": str(path.relative_to(ROOT)), "exists": False}
    data = load_json(path)
    splits = data.get("splits") or {}
    counts = data.get("counts") or {}
    stats = data.get("stats") or {}
    label_counts: dict[str, int] = {}
    bucket_counts: dict[str, int] = {}
    for split in splits.values():
        for label, value in (split.get("label_counts") or {}).items():
            label_counts[label] = label_counts.get(label, 0) + int(value)
        for bucket, value in (split.get("bucket_counts") or {}).items():
            bucket_counts[bucket] = bucket_counts.get(bucket, 0) + int(value)
    if not label_counts and stats:
        for key, value in stats.items():
            if key.startswith("label_") and "bucket" not in key:
                label_counts[key.removeprefix("label_")] = int(value)
            if key.startswith("label_bucket_"):
                bucket = key.rsplit("_", 1)[-1]
                bucket_counts[bucket] = bucket_counts.get(bucket, 0) + int(value)
    return {
        "path": str(path.relative_to(ROOT)),
        "exists": True,
        "id": data.get("id"),
        "images": {name: split.get("images") for name, split in splits.items()} or counts,
        "label_counts": dict(sorted(label_counts.items(), key=lambda item: item[0])),
        "bucket_counts": dict(sorted(bucket_counts.items(), key=lambda item: item[0])),
        "strengths": infer_strengths(path, label_counts, bucket_counts),
        "gaps": infer_gaps(label_counts, bucket_counts),
    }


def infer_strengths(path: Path, labels: dict[str, int], buckets: dict[str, int]) -> list[str]:
    strengths = []
    name = path.as_posix()
    if labels.get("stair", 0) > 5000:
        strengths.append("large stair exposure")
    if labels.get("sink", 0) > 5000:
        strengths.append("sink/tiny exposure")
    if labels.get("equipment", 0) > 3000:
        strengths.append("equipment exposure")
    if buckets.get("tiny_le_64", 0) + buckets.get("tiny", 0) > 5000:
        strengths.append("tiny-object weighting")
    if "p211" in name:
        strengths.append("existing page-sliced inference path")
    if "p205b" in name:
        strengths.append("weighted high-recall sampling")
    return strengths or ["limited/unknown"]


def infer_gaps(labels: dict[str, int], buckets: dict[str, int]) -> list[str]:
    gaps = []
    if labels.get("column", 0) < 2000:
        gaps.append("column underrepresented")
    if labels.get("stair", 0) < 8000:
        gaps.append("stair may still need oversampling")
    if buckets.get("small_le_256", 0) + buckets.get("small", 0) < 4000:
        gaps.append("small-object underrepresented")
    if buckets.get("tiny_le_64", 0) + buckets.get("tiny", 0) < 5000:
        gaps.append("tiny-object underrepresented")
    return gaps or ["no obvious count gap"]


def render(report: dict[str, Any]) -> str:
    lines = [
        "# P224 Stronger Detector Branch Plan",
        "",
        "## Why Detector First",
        f"- {report['why_detector_first']}",
        "",
        "## P223 Bottlenecks",
        "| Target | Current | Union Typed Cover | Action |",
        "|---|---:|---:|---|",
    ]
    for item in report["primary_targets"]:
        target = item.get("label") or item.get("bucket")
        lines.append(f"| {target} | {item['current']:.6f} | {item['union_typed_cover']:.6f} | {item['action']} |")
    lines += ["", "## Dataset Candidates", "| Dataset | Strengths | Gaps |", "|---|---|---|"]
    for ds in report["dataset_candidates"]:
        lines.append(f"| `{ds['path']}` | {', '.join(ds.get('strengths', []))} | {', '.join(ds.get('gaps', []))} |")
    lines += ["", "## Implementation Plan"]
    for index, step in enumerate(report["implementation_steps"], 1):
        lines.append(f"{index}. {step}")
    lines += ["", "## Success Gates"]
    for gate in report["success_gates"]:
        lines.append(f"- {gate}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    p223 = load_json(P223)
    label_cover = {row["label"]: row for row in p223["oracle_ceiling"]["coverage_by_label"]}
    size_cover = {row["bucket"]: row for row in p223["oracle_ceiling"]["coverage_by_size"]}
    label_metrics = {row["label"]: row for row in p223["p222_metrics"]["by_label"]}
    size_metrics = {row["bucket"]: row for row in p223["p222_metrics"]["by_size"]}
    primary_targets = [
        {"label": "stair", "current": label_metrics["stair"]["recall"], "union_typed_cover": label_cover["stair"]["typed_cover_recall"], "action": "oversample stair and train high-res page detector; P221b crop model can seed hard examples"},
        {"label": "column", "current": label_metrics["column"]["recall"], "union_typed_cover": label_cover["column"]["typed_cover_recall"], "action": "add column-focused examples because existing proposal sources do not improve coverage"},
        {"bucket": "tiny_le_64", "current": size_metrics["tiny_le_64"]["recall"], "union_typed_cover": size_cover["tiny_le_64"]["typed_cover_recall"], "action": "train higher-resolution/tiny-weighted detector; current union barely raises coverage"},
        {"bucket": "small_le_256", "current": size_metrics["small_le_256"]["recall"], "union_typed_cover": size_cover["small_le_256"]["typed_cover_recall"], "action": "class-balanced slices and small-object sampling"},
    ]
    report = {
        "id": "P224_detector_branch_plan",
        "p223_source": str(P223.relative_to(ROOT)),
        "why_detector_first": "P223 shows candidate_add_typed recall=0.809011 and candidate_add_relabel recall=0.826555, so existing proposals cannot reach reviewer-grade F1 even with ideal gating/relabeling.",
        "primary_targets": primary_targets,
        "dataset_candidates": [summarize_dataset(path) for path in DATASET_CANDIDATES],
        "key_reusable_scripts": KEY_SCRIPTS,
        "selected_starting_point": {
            "base_dataset": "datasets/symbol_recall_detector_p211_yolo_20k_server plus targeted P205b/P213b/P221b oversampling",
            "base_training_script": "scripts/vlm/train_symbol_recall_detector_p211.py or a P224 wrapper around Ultralytics YOLO",
            "base_inference_script": "scripts/vlm/infer_symbol_p211_on_p206g_pages_p212.py adapted to P224 weights and larger slices/imgsz",
        },
        "implementation_steps": [
            "Create P224 dataset builder that combines P211 20k general recall data with P205b weighted tiny/small samples and new column/stair oversampling from P222/P223 hard rows.",
            "Export YOLO data.yaml and train/dev/locked lists with explicit label/bucket counts; do not use gold labels at runtime.",
            "Train yolov8m first on server GPU0 with imgsz 768/960, long schedule, class-balanced sampling; keep yolov8s/yolo11m as follow-up comparisons.",
            "Run page-level sliced inference with larger slice sizes and low decode conf to maximize recall; then evaluate with P223 buckets.",
            "Only after recall improves, apply NMS/gating/ensemble precision repair and bootstrap vs P222.",
        ],
        "success_gates": [
            "First milestone: P224 typed cover/recall should exceed P223 union typed recall 0.809 by a clear margin.",
            "Metric milestone: overall F1 gain at least +0.05 before considering the branch paper-relevant.",
            "Reviewer milestone: visible recall gains for stair, column, tiny_le_64, and small_le_256 buckets.",
            "If detector recall does not move, stop and switch to dataset/annotation expansion rather than threshold tuning.",
        ],
        "next_artifacts": [
            "scripts/vlm/build_symbol_p224_detector_data.py",
            "scripts/vlm/train_symbol_p224_detector.py",
            "scripts/vlm/infer_symbol_p224_detector_pages.py",
            "reports/vlm/symbol_p224_detector_page_eval.json",
        ],
        "claim_boundary": "Planning artifact only; training may use labels offline, runtime must use raster pixels/model weights/config only.",
    }
    OUT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    OUT_MD.write_text(render(report), encoding="utf-8")
    print(json.dumps({"plan": str(OUT_MD.relative_to(ROOT)), "selected_starting_point": report["selected_starting_point"], "primary_targets": primary_targets}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
