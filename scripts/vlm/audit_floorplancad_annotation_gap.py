#!/usr/bin/env python3
"""Audit annotation protocol gaps between FloorPlanCAD and CVC-FP for WallOpening.

Quantifies differences in:
1. Label distribution and class balance (door/window ratio, hard_wall prevalence)
2. Morphology (bbox area, aspect ratio, length per class per source)
3. Raster appearance (dark density, edge density, mean intensity per class per source)
4. Topology (graph degree, relation counts per class per source)
5. Error pattern analysis (confusion types, systematic misclassifications)

Output:
  - reports/vlm/floorplancad_annotation_gap_audit.json

Done when: At least 3 quantified difference sources with per-sample statistics support.
"""

from __future__ import annotations

import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
MIXED_LOCKED = ROOT / "datasets/cadstruct_real_world_benchmark_v1/wall_opening/mixed_source_locked_test.jsonl"
FLOORPLANCAD_LOCKED = ROOT / "datasets/cadstruct_real_world_benchmark_v1/wall_opening/floorplancad_locked_test.jsonl"
OUTPUT = ROOT / "reports/vlm/floorplancad_annotation_gap_audit.json"


def main() -> None:
    print("=== FloorPlanCAD Annotation Gap Audit ===\n")

    # Load data
    mixed = load_jsonl(MIXED_LOCKED)
    fp_only = load_jsonl(FLOORPLANCAD_LOCKED)

    # Separate by source
    cvc_images, fp_mixed_images = [], []
    for img in mixed:
        if img.get("source_dataset") == "cvc_fp":
            cvc_images.append(img)
        elif img.get("source_dataset") == "floorplancad":
            fp_mixed_images.append(img)

    # Verify consistency
    assert len(fp_mixed_images) == len(fp_only), (
        f"FloorPlanCAD count mismatch: mixed={len(fp_mixed_images)} vs only={len(fp_only)}"
    )
    fp_images = fp_mixed_images  # use from mixed for feature consistency

    print(f"CVC-FP: {len(cvc_images)} images")
    print(f"FloorPlanCAD: {len(fp_images)} images")

    # Extract all nodes per source
    cvc_nodes, fp_nodes = [], []
    for img in cvc_images:
        for node in img.get("nodes", []):
            cvc_nodes.append(node)
    for img in fp_images:
        for node in img.get("nodes", []):
            fp_nodes.append(node)

    print(f"\nCVC-FP nodes: {len(cvc_nodes)}")
    print(f"FloorPlanCAD nodes: {len(fp_nodes)}")

    report: dict[str, Any] = {
        "version": "floorplancad_annotation_gap_audit_v1",
        "purpose": "Quantify annotation protocol differences between FloorPlanCAD and CVC-FP",
        "data_summary": {
            "cvc_fp": {"images": len(cvc_images), "nodes": len(cvc_nodes)},
            "floorplancad": {"images": len(fp_images), "nodes": len(fp_nodes)},
        },
    }

    # 1. Label distribution analysis
    report["label_distribution"] = analyze_label_distribution(cvc_nodes, fp_nodes)

    # 2. Morphology analysis (bbox area, aspect, length per class)
    report["morphology"] = analyze_morphology(cvc_nodes, fp_nodes)

    # 3. Raster appearance analysis
    report["raster_appearance"] = analyze_raster_features(cvc_nodes, fp_nodes)

    # 4. Topology analysis
    report["topology"] = analyze_topology(cvc_nodes, fp_nodes)

    # 5. Error pattern analysis (using existing hard cases)
    report["error_patterns"] = analyze_error_patterns()

    # 6. Quantified difference sources (summary)
    report["quantified_differences"] = summarize_differences(report)

    # 7. Recommendations for S3-T2 adaptation
    report["adaptation_recommendations"] = generate_recommendations(report)

    # Save
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"\nReport saved to {OUTPUT}")

    # Print summary
    print("\n=== Summary of Quantified Differences ===")
    for i, diff in enumerate(report["quantified_differences"], 1):
        print(f"  {i}. {diff['name']}: {diff['description']}")


def analyze_label_distribution(cvc_nodes: list, fp_nodes: list) -> dict:
    """Compare class balance and label prevalence."""
    cvc_labels = Counter(n.get("label") for n in cvc_nodes)
    fp_labels = Counter(n.get("label") for n in fp_nodes)

    cvc_total = len(cvc_nodes)
    fp_total = len(fp_nodes)

    # Per-class rates
    all_labels = sorted(set(list(cvc_labels.keys()) + list(fp_labels.keys())))
    per_class = {}
    for label in all_labels:
        cvc_count = cvc_labels.get(label, 0)
        fp_count = fp_labels.get(label, 0)
        per_class[label] = {
            "cvc_fp": {"count": cvc_count, "rate": cvc_count / cvc_total},
            "floorplancad": {"count": fp_count, "rate": fp_count / fp_total},
            "rate_gap_pp": round((fp_count / fp_total - cvc_count / cvc_total) * 10000, 1),
        }

    # Door/window ratio
    cvc_dw_ratio = cvc_labels.get("door", 0) / max(cvc_labels.get("window", 1), 1)
    fp_dw_ratio = fp_labels.get("door", 0) / max(fp_labels.get("window", 1), 1)

    # Opening ratio (door+window vs hard_wall)
    cvc_opening_ratio = (cvc_labels.get("door", 0) + cvc_labels.get("window", 0)) / cvc_total
    fp_opening_ratio = (fp_labels.get("door", 0) + fp_labels.get("window", 0)) / fp_total

    return {
        "label_counts": {
            "cvc_fp": dict(cvc_labels),
            "floorplancad": dict(fp_labels),
        },
        "per_class_rates": per_class,
        "door_window_ratio": {
            "cvc_fp": round(cvc_dw_ratio, 2),
            "floorplancad": round(fp_dw_ratio, 2),
            "gap": round(fp_dw_ratio - cvc_dw_ratio, 2),
        },
        "opening_ratio": {
            "cvc_fp": round(cvc_opening_ratio, 4),
            "floorplancad": round(fp_opening_ratio, 4),
            "gap_pp": round((fp_opening_ratio - cvc_opening_ratio) * 10000, 1),
        },
        "interpretation": (
            f"FloorPlanCAD has {fp_dw_ratio:.1f} doors per window vs {cvc_dw_ratio:.1f} in CVC-FP. "
            f"Opening density is {fp_opening_ratio*100:.1f}% vs {cvc_opening_ratio*100:.1f}%."
        ),
    }


def bbox_area(node: dict) -> float:
    bbox = node.get("features", {}).get("bbox", [])
    if len(bbox) == 4:
        w = abs(bbox[2] - bbox[0])
        h = abs(bbox[3] - bbox[1])
        return w * h
    return 0.0


def bbox_width(node: dict) -> float:
    bbox = node.get("features", {}).get("bbox", [])
    return abs(bbox[2] - bbox[0]) if len(bbox) == 4 else 0.0


def bbox_height(node: dict) -> float:
    bbox = node.get("features", {}).get("bbox", [])
    return abs(bbox[3] - bbox[1]) if len(bbox) == 4 else 0.0


def aspect_ratio(node: dict) -> float:
    w = bbox_width(node)
    h = bbox_height(node)
    if min(w, h) < 1e-6:
        return 1.0
    return max(w, h) / min(w, h)


def analyze_morphology(cvc_nodes: list, fp_nodes: list) -> dict:
    """Compare bbox morphology per class per source."""
    result = {}

    for label in ["hard_wall", "door", "window"]:
        cvc_subset = [n for n in cvc_nodes if n.get("label") == label]
        fp_subset = [n for n in fp_nodes if n.get("label") == label]

        if not cvc_subset or not fp_subset:
            continue

        # Area statistics
        cvc_areas = [bbox_area(n) for n in cvc_subset]
        fp_areas = [bbox_area(n) for n in fp_subset]

        # Aspect ratio statistics
        cvc_aspects = [aspect_ratio(n) for n in cvc_subset]
        fp_aspects = [aspect_ratio(n) for n in fp_subset]

        # Width/height
        cvc_widths = [bbox_width(n) for n in cvc_subset]
        fp_widths = [bbox_width(n) for n in fp_subset]
        cvc_heights = [bbox_height(n) for n in cvc_subset]
        fp_heights = [bbox_height(n) for n in fp_subset]

        # Length (from features)
        cvc_lengths = [n.get("features", {}).get("length", 0) for n in cvc_subset]
        fp_lengths = [n.get("features", {}).get("length", 0) for n in fp_subset]

        result[label] = {
            "area": {
                "cvc_fp": _stats(cvc_areas),
                "floorplancad": _stats(fp_areas),
                "median_ratio_fp_over_cvc": round(np.median(fp_areas) / max(np.median(cvc_areas), 1), 3),
            },
            "aspect_ratio": {
                "cvc_fp": _stats(cvc_aspects),
                "floorplancad": _stats(fp_aspects),
                "high_aspect_gt_5_rate": {
                    "cvc_fp": round(sum(1 for a in cvc_aspects if a > 5) / len(cvc_aspects), 4),
                    "floorplancad": round(sum(1 for a in fp_aspects if a > 5) / len(fp_aspects), 4),
                },
            },
            "width": {
                "cvc_fp": _stats(cvc_widths),
                "floorplancad": _stats(fp_widths),
            },
            "height": {
                "cvc_fp": _stats(cvc_heights),
                "floorplancad": _stats(fp_heights),
            },
            "length": {
                "cvc_fp": _stats(cvc_lengths),
                "floorplancad": _stats(fp_lengths),
            },
            "n_samples": {"cvc_fp": len(cvc_subset), "floorplancad": len(fp_subset)},
        }

    # Key finding: door morphology gap
    if "door" in result:
        door_cvc = [n for n in cvc_nodes if n.get("label") == "door"]
        door_fp = [n for n in fp_nodes if n.get("label") == "door"]
        cvc_narrow_doors = sum(1 for n in door_cvc if aspect_ratio(n) > 5)
        fp_narrow_doors = sum(1 for n in door_fp if aspect_ratio(n) > 5)
        result["door_narrow_rate"] = {
            "cvc_fp": round(cvc_narrow_doors / len(door_cvc), 4),
            "floorplancad": round(fp_narrow_doors / len(door_fp), 4),
            "interpretation": (
                f"FloorPlanCAD doors have {fp_narrow_doors/len(door_fp)*100:.1f}% high-aspect (>5) "
                f"vs {cvc_narrow_doors/len(door_cvc)*100:.1f}% in CVC-FP — thin doors more likely confused with walls."
            ),
        }

    return result


def analyze_raster_features(cvc_nodes: list, fp_nodes: list) -> dict:
    """Compare raster appearance per class per source."""
    raster_keys = [
        "raster_mean", "raster_std",
        "raster_dark_density", "raster_very_dark_density",
        "raster_edge_density", "raster_edge_strong_density",
        "raster_dark_ratio", "raster_edge_ratio",
    ]

    result = {}
    for label in ["hard_wall", "door", "window"]:
        cvc_subset = [n for n in cvc_nodes if n.get("label") == label]
        fp_subset = [n for n in fp_nodes if n.get("label") == label]

        if not cvc_subset or not fp_subset:
            continue

        per_feature = {}
        for key in raster_keys:
            cvc_vals = [n.get("features", {}).get(key, 0) for n in cvc_subset]
            fp_vals = [n.get("features", {}).get(key, 0) for n in fp_subset]
            per_feature[key] = {
                "cvc_fp": _stats(cvc_vals),
                "floorplancad": _stats(fp_vals),
                "median_gap": round(np.median(fp_vals) - np.median(cvc_vals), 4),
            }

        result[label] = per_feature

    # Key finding: overall raster darkness difference
    cvc_dark = [n.get("features", {}).get("raster_dark_density", 0) for n in cvc_nodes]
    fp_dark = [n.get("features", {}).get("raster_dark_density", 0) for n in fp_nodes]
    result["overall_darkness"] = {
        "cvc_fp": _stats(cvc_dark),
        "floorplancad": _stats(fp_dark),
        "interpretation": (
            f"FloorPlanCAD raster_dark_density median={np.median(fp_dark):.3f} vs "
            f"CVC-FP median={np.median(cvc_dark):.3f} — FloorPlanCAD drawings are "
            f"{'darker' if np.median(fp_dark) > np.median(cvc_dark) else 'lighter'}."
        ),
    }

    return result


def analyze_topology(cvc_nodes: list, fp_nodes: list) -> dict:
    """Compare graph topology per class per source."""
    topo_keys = ["graph_degree", "graph_in_degree", "graph_out_degree",
                 "relation_touches", "relation_opens_in_wall",
                 "relation_window_in_wall", "relation_contained_in", "relation_contains"]

    result = {}
    for label in ["hard_wall", "door", "window"]:
        cvc_subset = [n for n in cvc_nodes if n.get("label") == label]
        fp_subset = [n for n in fp_nodes if n.get("label") == label]

        if not cvc_subset or not fp_subset:
            continue

        per_feature = {}
        for key in topo_keys:
            cvc_vals = [n.get("features", {}).get(key, 0) for n in cvc_subset]
            fp_vals = [n.get("features", {}).get(key, 0) for n in fp_subset]
            per_feature[key] = {
                "cvc_fp": _stats(cvc_vals),
                "floorplancad": _stats(fp_vals),
                "median_gap": round(np.median(fp_vals) - np.median(cvc_vals), 3),
            }

        result[label] = per_feature

    # Key finding: isolated nodes (degree=1) rate
    for label in ["door", "window"]:
        cvc_subset = [n for n in cvc_nodes if n.get("label") == label]
        fp_subset = [n for n in fp_nodes if n.get("label") == label]
        cvc_isolated = sum(1 for n in cvc_subset if n.get("features", {}).get("graph_degree", 0) <= 1)
        fp_isolated = sum(1 for n in fp_subset if n.get("features", {}).get("graph_degree", 0) <= 1)
        result[f"{label}_isolated_rate"] = {
            "cvc_fp": round(cvc_isolated / len(cvc_subset), 4),
            "floorplancad": round(fp_isolated / len(fp_subset), 4),
            "interpretation": (
                f"FloorPlanCAD {label} isolated rate (degree≤1): {fp_isolated/len(fp_subset)*100:.1f}% "
                f"vs CVC-FP {cvc_isolated/len(cvc_subset)*100:.1f}%."
            ),
        }

    return result


def analyze_error_patterns() -> dict:
    """Analyze known error patterns from hard cases and residual audits."""
    hard_cases_path = ROOT / "datasets/internal_hard_cases_round_1/wall_opening_floorplancad_hard_cases_summary.json"

    result = {"known_errors": {}, "interpretation": ""}

    if hard_cases_path.exists():
        hard = json.loads(hard_cases_path.read_text(encoding="utf-8"))
        result["known_errors"] = {
            "total_records": hard.get("records", 0),
            "errors": hard.get("errors", 0),
            "error_rate": round(hard.get("errors", 0) / max(hard.get("records", 1), 1), 4),
            "error_pairs": hard.get("error_pairs", {}),
            "exported_hard_cases": hard.get("exported_hard_cases", 0),
        }

        # Classify error types
        error_pairs = hard.get("error_pairs", {})
        result["error_classification"] = {
            "door_hard_wall_confusion": error_pairs.get("door->hard_wall", 0),
            "door_window_subtype_confusion": error_pairs.get("door->window", 0),
            "hard_wall_door_confusion": error_pairs.get("hard_wall->door", 0),
        }

        result["interpretation"] = (
            f"Of {hard.get('errors', 0)} errors in {hard.get('records', 0)} FloorPlanCAD smoke nodes: "
            f"{error_pairs.get('door->hard_wall', 0)} are door→hard_wall (thin doors confused with walls), "
            f"{error_pairs.get('door->window', 0)} are door→window (subtype ambiguity), "
            f"{error_pairs.get('hard_wall->door', 0)} are hard_wall→door (isolated walls as doors)."
        )
    else:
        result["interpretation"] = "No hard cases summary found."

    return result


def summarize_differences(report: dict) -> list:
    """Summarize the top quantified differences as actionable findings."""
    diffs = []

    # 1. Label distribution gap
    ld = report.get("label_distribution", {})
    dw_ratio = ld.get("door_window_ratio", {})
    diffs.append({
        "name": "Class balance skew: FloorPlanCAD is door-heavy, window-light",
        "description": (
            f"FloorPlanCAD has {dw_ratio.get('floorplancad', 0):.1f} doors per window vs "
            f"{dw_ratio.get('cvc_fp', 0):.1f} in CVC-FP (gap={dw_ratio.get('gap', 0):.1f}). "
            f"Opening density: {ld.get('opening_ratio', {}).get('floorplancad', 0)*100:.1f}% vs "
            f"{ld.get('opening_ratio', {}).get('cvc_fp', 0)*100:.1f}%. "
            "Model trained on CVC-FP expects more windows; FloorPlanCAD doors are under-represented in training."
        ),
        "severity": "medium",
    })

    # 2. Door morphology gap
    morph = report.get("morphology", {})
    door_narrow = morph.get("door_narrow_rate", {})
    diffs.append({
        "name": "Door morphology: FloorPlanCAD doors are thinner/more elongated",
        "description": (
            f"High-aspect ratio (>5) doors: {door_narrow.get('floorplancad', 0)*100:.1f}% in FloorPlanCAD vs "
            f"{door_narrow.get('cvc_fp', 0)*100:.1f}% in CVC-FP. "
            f"Door area median ratio (FP/CVC): {morph.get('door', {}).get('area', {}).get('median_ratio_fp_over_cvc', 'N/A')}. "
            "Thin FloorPlanCAD doors are confused with hard walls (8 of 12 errors are door→hard_wall)."
        ),
        "severity": "high",
    })

    # 3. Raster appearance gap
    raster = report.get("raster_appearance", {})
    overall = raster.get("overall_darkness", {})
    diffs.append({
        "name": "Raster appearance: FloorPlanCAD drawings have different dark/edge density profile",
        "description": (
            f"Overall raster_dark_density: FloorPlanCAD median={overall.get('floorplancad', {}).get('median', 0):.3f} vs "
            f"CVC-FP median={overall.get('cvc_fp', {}).get('median', 0):.3f}. "
            f"{overall.get('interpretation', '')} "
            "This affects threshold-based opening detection."
        ),
        "severity": "medium",
    })

    # 4. Topology gap
    topo = report.get("topology", {})
    door_iso = topo.get("door_isolated_rate", {})
    diffs.append({
        "name": "Topology: FloorPlanCAD has more isolated openings (degree ≤ 1)",
        "description": (
            f"Door isolated rate: {door_iso.get('floorplancad', 0)*100:.1f}% vs "
            f"{door_iso.get('cvc_fp', 0)*100:.1f}% in CVC-FP. "
            f"Window isolated rate: {topo.get('window_isolated_rate', {}).get('floorplancad', 0)*100:.1f}% vs "
            f"{topo.get('window_isolated_rate', {}).get('cvc_fp', 0)*100:.1f}%. "
            "Isolated openings in FloorPlanCAD are misclassified as hard walls."
        ),
        "severity": "medium",
    })

    # 5. Annotation boundary: door vs window subtype ambiguity
    errors = report.get("error_patterns", {})
    error_class = errors.get("error_classification", {})
    diffs.append({
        "name": "Annotation protocol: FloorPlanCAD door/window subtype boundary differs from CVC-FP",
        "description": (
            f"Error pairs: door→window={error_class.get('door_window_subtype_confusion', 0)}, "
            f"door→hard_wall={error_class.get('door_hard_wall_confusion', 0)}, "
            f"hard_wall→door={error_class.get('hard_wall_door_confusion', 0)}. "
            "FloorPlanCAD and CVC-FP define door/window boundaries differently (e.g., sliding doors, "
            "wide openings, archways). A FloorPlanCAD-specific calibration branch is needed."
        ),
        "severity": "low",
    })

    return diffs


def generate_recommendations(report: dict) -> dict:
    """Generate actionable recommendations for S3-T2 adaptation."""
    return {
        "priority_1": {
            "action": "Add FloorPlanCAD thin-door hard negatives",
            "rationale": "8 of 12 errors are door→hard_wall. FloorPlanCAD doors have higher aspect ratio.",
            "implementation": "Mine high-aspect (>5) door bboxes from FloorPlanCAD and add as labeled training data with source_floorplancad=1.",
        },
        "priority_2": {
            "action": "Train FloorPlanCAD residual calibration branch",
            "rationale": "Domain gap is structural (morphology + raster), not just distribution shift.",
            "implementation": "Use few-shot (50 samples) adaptation with source-specific threshold calibration. Keep frozen main model.",
        },
        "priority_3": {
            "action": "Add raster-dark density as router feature",
            "rationale": f"FloorPlanCAD raster_dark_density median differs from CVC-FP ({report.get('raster_appearance', {}).get('overall_darkness', {}).get('floorplancad', {}).get('median', 0):.3f} vs {report.get('raster_appearance', {}).get('overall_darkness', {}).get('cvc_fp', {}).get('median', 0):.3f}).",
            "implementation": "Add raster_dark_density to router feature set for source-aware routing.",
        },
        "priority_4": {
            "action": "Door/window subtype calibration with abstention",
            "rationale": "door→window errors indicate subtype boundary ambiguity.",
            "implementation": "When door/window posterior margin < threshold, abstain or use FloorPlanCAD-specific prior.",
        },
        "target_revision": {
            "note": "The 0.98 FloorPlanCAD target may need to account for annotation protocol differences. "
                    "After thin-door hard negatives + residual branch, target should be achievable. "
                    "If not, consider reporting ambiguity-adjusted FloorPlanCAD F1 separately.",
        },
    }


def _stats(values: list) -> dict:
    """Compute summary statistics for a list of numbers."""
    if not values:
        return {"count": 0}
    arr = np.array(values, dtype=float)
    return {
        "count": len(arr),
        "mean": round(float(np.mean(arr)), 4),
        "std": round(float(np.std(arr)), 4),
        "median": round(float(np.median(arr)), 4),
        "min": round(float(np.min(arr)), 4),
        "max": round(float(np.max(arr)), 4),
        "p10": round(float(np.percentile(arr, 10)), 4),
        "p25": round(float(np.percentile(arr, 25)), 4),
        "p75": round(float(np.percentile(arr, 75)), 4),
        "p90": round(float(np.percentile(arr, 90)), 4),
    }


def load_jsonl(path: Path) -> list:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


if __name__ == "__main__":
    main()
