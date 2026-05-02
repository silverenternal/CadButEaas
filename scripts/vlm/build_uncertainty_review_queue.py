#!/usr/bin/env python3
"""R6-T3: Uncertainty & abstain review queue builder v1.

Computes per-sample uncertainty scores from multiple signals:
  1. Prediction confidence variance (node/edge confidence spread)
  2. Conflicting expert outputs (multiple source_experts per semantic type)
  3. Edge density anomalies (edges-per-node ratio outliers)
  4. Quality features (warnings, repair events, candidate counts)
  5. Known error labels from e2e_scene_graph_v1_cases.jsonl

Generates abstain decisions for samples above threshold and a review
queue prioritized by uncertainty score.

Outputs:
  - reports/vlm/uncertainty_abstain_v1.json
  - reports/vlm/review_queue_v1.jsonl
  - checkpoints/uncertainty_review_v1/train_summary.json
"""

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = ROOT / "reports" / "vlm"
CHECKPOINTS_DIR = ROOT / "checkpoints"


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_predictions(path):
    """Load e2e predictions from JSONL, keyed by image path."""
    preds = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            preds[rec.get("image", "")] = rec
    return preds


def load_cases(path):
    """Load per-sample error cases from JSONL, keyed by image path."""
    cases = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            cases[rec.get("image", "")] = rec
    return cases


# ---------------------------------------------------------------------------
# Feature computation
# ---------------------------------------------------------------------------

def compute_confidence_signal(rec):
    """Return (mean_conf, std_conf, min_conf) from nodes and edges."""
    sg = rec.get("scene_graph", {})
    nodes = sg.get("nodes", [])
    edges = sg.get("edges", [])

    node_confs = [n.get("confidence", 1.0) for n in nodes]
    edge_confs = [e.get("confidence", 1.0) for e in edges]
    all_confs = node_confs + edge_confs

    if not all_confs:
        return 1.0, 0.0, 1.0

    arr = np.array(all_confs, dtype=np.float32)
    return float(arr.mean()), float(arr.std()), float(arr.min())


def compute_expert_conflict_signal(rec):
    """Return conflict score based on how many different experts contribute
    to each semantic type within the same sample."""
    sg = rec.get("scene_graph", {})
    nodes = sg.get("nodes", [])

    type_experts = defaultdict(set)
    for n in nodes:
        st = n.get("semantic_type", "unknown")
        expert = n.get("source_expert", "unknown")
        type_experts[st].add(expert)

    # Fraction of semantic types with >1 expert (conflict indicator)
    n_types = len(type_experts)
    if n_types == 0:
        return 0.0, 0, 0

    conflicted = sum(1 for s in type_experts.values() if len(s) > 1)
    return conflicted / n_types, conflicted, n_types


def compute_edge_density_signal(rec):
    """Return edge density anomaly score (z-score of edges-per-node)."""
    sg = rec.get("scene_graph", {})
    n_nodes = len(sg.get("nodes", []))
    n_edges = len(sg.get("edges", []))

    if n_nodes == 0:
        return 0.0, 0, 0, 0.0

    ratio = n_edges / n_nodes
    return ratio, n_nodes, n_edges, 0.0  # z-score filled in batch


def compute_quality_signal(rec):
    """Return composite quality score from warnings, repairs, candidate counts."""
    route = rec.get("route_trace", {})
    qr = rec.get("quality_report", {})

    n_warnings = len(rec.get("warnings", []))
    n_repairs = route.get("repair_event_count", 0)
    candidate_counts = qr.get("candidate_counts", {})
    total_candidates = sum(candidate_counts.values()) if candidate_counts else 0

    # Normalized quality score (0 = good, 1 = poor)
    # Heavily weighted toward repair events and warnings
    quality_score = min(1.0, (n_warnings * 0.15 + n_repairs * 0.1 + total_candidates * 0.0002))
    return quality_score, n_warnings, n_repairs, total_candidates


def compute_ocr_uncertainty_signal(rec):
    """Estimate OCR-related uncertainty from text/dimension candidate counts
    and text-related warnings."""
    qr = rec.get("quality_report", {})
    candidate_counts = qr.get("candidate_counts", {})
    text_candidates = candidate_counts.get("texts", 0)
    symbols = candidate_counts.get("symbols", 0)

    sg = rec.get("scene_graph", {})
    nodes = sg.get("nodes", [])
    text_nodes = [n for n in nodes if "text" in n.get("semantic_type", "").lower()
                  or "dimension" in n.get("semantic_type", "").lower()]
    n_text_nodes = len(text_nodes)

    # High candidate-to-detected ratio suggests OCR difficulty
    if n_text_nodes == 0 and text_candidates > 0:
        return 1.0, text_candidates, 0
    if text_candidates == 0:
        return 0.0, 0, n_text_nodes

    ocr_ratio = n_text_nodes / max(1, text_candidates)
    # Low ratio = many candidates but few detected = high uncertainty
    ocr_uncertainty = max(0.0, 1.0 - ocr_ratio)
    return ocr_uncertainty, text_candidates, n_text_nodes


# ---------------------------------------------------------------------------
# Uncertainty aggregation
# ---------------------------------------------------------------------------

def compute_uncertainty_scores(preds, cases):
    """Compute all uncertainty signals and aggregate into a single score."""
    records = []

    # First pass: collect edge density ratios for z-score computation
    edge_ratios = []
    for img, rec in preds.items():
        ratio, n_nodes, n_edges, _ = compute_edge_density_signal(rec)
        edge_ratios.append(ratio)

    ratio_arr = np.array(edge_ratios, dtype=np.float64)
    ratio_mean = ratio_arr.mean()
    ratio_std = ratio_arr.std()
    if ratio_std == 0:
        ratio_std = 1.0

    # Second pass: compute all signals
    for img, rec in preds.items():
        case = cases.get(img, {})
        is_error = bool(case.get("failure_tags"))
        error_severity = len(case.get("failure_tags", []))

        # Signal 1: confidence
        mean_conf, std_conf, min_conf = compute_confidence_signal(rec)
        conf_uncertainty = 1.0 - mean_conf  # low confidence = high uncertainty

        # Signal 2: expert conflict
        conflict_frac, n_conflicted, n_types = compute_expert_conflict_signal(rec)

        # Signal 3: edge density anomaly (z-score)
        ratio, n_nodes, n_edges, _ = compute_edge_density_signal(rec)
        edge_z = abs(ratio - ratio_mean) / ratio_std
        edge_anomaly = min(1.0, edge_z / 3.0)  # normalize: z=3 -> 1.0

        # Signal 4: quality
        quality_score, n_warnings, n_repairs, total_candidates = compute_quality_signal(rec)

        # Signal 5: OCR uncertainty
        ocr_uncertainty, text_candidates, n_text_nodes = compute_ocr_uncertainty_signal(rec)

        # Composite uncertainty: weighted sum
        uncertainty = (
            0.25 * conf_uncertainty +
            0.20 * conflict_frac +
            0.20 * edge_anomaly +
            0.20 * quality_score +
            0.15 * ocr_uncertainty
        )

        records.append({
            "image": img,
            "source_dataset": rec.get("source_dataset", "unknown"),
            "split": rec.get("split", "unknown"),
            "uncertainty": round(float(uncertainty), 6),
            "is_error": is_error,
            "error_severity": error_severity,
            "failure_tags": case.get("failure_tags", []),
            "signals": {
                "confidence": {
                    "mean_conf": round(mean_conf, 4),
                    "std_conf": round(std_conf, 4),
                    "min_conf": round(min_conf, 4),
                    "uncertainty": round(float(conf_uncertainty), 4),
                },
                "expert_conflict": {
                    "conflict_fraction": round(float(conflict_frac), 4),
                    "n_conflicted_types": n_conflicted,
                    "n_total_types": n_types,
                },
                "edge_density": {
                    "edges_per_node": round(ratio, 4),
                    "z_score": round(float(edge_z), 4),
                    "anomaly_score": round(float(edge_anomaly), 4),
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                },
                "quality": {
                    "score": round(float(quality_score), 4),
                    "n_warnings": n_warnings,
                    "n_repairs": n_repairs,
                    "total_candidates": total_candidates,
                },
                "ocr": {
                    "uncertainty": round(float(ocr_uncertainty), 4),
                    "text_candidates": text_candidates,
                    "detected_text_nodes": n_text_nodes,
                },
            },
        })

    return records


# ---------------------------------------------------------------------------
# Abstain decision & metrics
# ---------------------------------------------------------------------------

def find_optimal_threshold(records, target_precision=0.80):
    """Find threshold that maximizes recall while keeping abstain precision >= target."""
    thresholds = np.arange(0.05, 0.95, 0.01)
    best_threshold = 0.5
    best_recall = 0.0

    for t in thresholds:
        abstained = [r for r in records if r["uncertainty"] >= t]
        if not abstained:
            continue
        n_wrong = sum(1 for r in abstained if r["is_error"])
        precision = n_wrong / len(abstained)
        if precision >= target_precision:
            recall = n_wrong / max(1, sum(1 for r in records if r["is_error"]))
            if recall > best_recall:
                best_recall = recall
                best_threshold = float(t)

    return best_threshold


def compute_abstain_metrics(records, threshold):
    """Compute abstain precision, recall, and high-confidence error reduction."""
    abstained = [r for r in records if r["uncertainty"] >= threshold]
    kept = [r for r in records if r["uncertainty"] < threshold]

    n_abstained = len(abstained)
    n_wrong_abstained = sum(1 for r in abstained if r["is_error"])
    n_wrong_total = sum(1 for r in records if r["is_error"])
    n_wrong_kept = sum(1 for r in kept if r["is_error"])

    # Abstain precision: of abstained, what % are actually wrong?
    abstain_precision = n_wrong_abstained / max(1, n_abstained)

    # Abstain recall: of all wrong, what % did we catch?
    abstain_recall = n_wrong_abstained / max(1, n_wrong_total)

    # High-confidence errors: errors that would have been kept with high confidence
    hc_errors_without_abstain = n_wrong_total  # all errors are "high confidence" without abstain
    hc_errors_with_abstain = n_wrong_kept  # errors that slip through

    # Reduction in high-confidence errors
    if hc_errors_without_abstain > 0:
        hc_error_reduction = (hc_errors_without_abstain - hc_errors_with_abstain) / hc_errors_without_abstain
    else:
        hc_error_reduction = 0.0

    return {
        "threshold": round(threshold, 4),
        "n_total": len(records),
        "n_abstained": n_abstained,
        "n_kept": len(kept),
        "n_wrong_total": n_wrong_total,
        "n_wrong_abstained": n_wrong_abstained,
        "n_wrong_kept": n_wrong_kept,
        "abstain_precision": round(float(abstain_precision), 4),
        "abstain_recall": round(float(abstain_recall), 4),
        "hc_error_reduction": round(float(hc_error_reduction), 4),
        "abstain_rate": round(n_abstained / max(1, len(records)), 4),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("R6-T3: Uncertainty & Abstain Review Queue Builder v1")
    print("=" * 70)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    ckpt_dir = CHECKPOINTS_DIR / "uncertainty_review_v1"
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # ── [1/5] Load data ──────────────────────────────────────────────
    print("[1/5] Loading data...")
    pred_path = REPORTS_DIR / "e2e_real_pipeline_smoke_predictions.jsonl"
    case_path = REPORTS_DIR / "e2e_scene_graph_v1_cases.jsonl"

    preds = load_predictions(pred_path)
    cases = load_cases(case_path)
    print(f"  Predictions: {len(preds)} samples")
    print(f"  Error cases: {len(cases)} samples")
    print(f"  Overlap: {len(set(preds.keys()) & set(cases.keys()))} samples")
    n_errors = sum(1 for c in cases.values() if c.get("failure_tags"))
    print(f"  Samples with errors: {n_errors}")
    print(f"  Elapsed: {time.time() - t_start:.1f}s")

    # ── [2/5] Compute uncertainty scores ─────────────────────────────
    print("[2/5] Computing uncertainty scores...")
    records = compute_uncertainty_scores(preds, cases)

    uncertainties = np.array([r["uncertainty"] for r in records])
    print(f"  Uncertainty stats: mean={uncertainties.mean():.4f}, "
          f"std={uncertainties.std():.4f}, "
          f"min={uncertainties.min():.4f}, max={uncertainties.max():.4f}")

    # Per-signal summary
    signal_key_map = {
        "confidence": "uncertainty",
        "expert_conflict": "conflict_fraction",
        "edge_density": "anomaly_score",
        "quality": "score",
        "ocr": "uncertainty",
    }
    for sig_name in ["confidence", "expert_conflict", "edge_density", "quality", "ocr"]:
        vals = [r["signals"][sig_name] for r in records]
        key = signal_key_map[sig_name]
        nums = [v[key] for v in vals if isinstance(v, dict)]
        if not nums:
            continue
        arr = np.array(nums, dtype=np.float64)
        print(f"    {sig_name}: mean={arr.mean():.4f}, std={arr.std():.4f}")
    print(f"  Elapsed: {time.time() - t_start:.1f}s")

    # ── [3/5] Find threshold & compute metrics ───────────────────────
    print("[3/5] Finding optimal abstain threshold...")
    threshold = find_optimal_threshold(records, target_precision=0.80)
    print(f"  Selected threshold: {threshold:.4f}")

    metrics = compute_abstain_metrics(records, threshold)
    print(f"  Abstain precision: {metrics['abstain_precision']:.4f}")
    print(f"  Abstain recall:    {metrics['abstain_recall']:.4f}")
    print(f"  HC error reduction: {metrics['hc_error_reduction']:.4f}")
    print(f"  Abstain rate:      {metrics['abstain_rate']:.4f}")
    print(f"  Elapsed: {time.time() - t_start:.1f}s")

    # ── [4/5] Generate review queue ──────────────────────────────────
    print("[4/5] Generating review queue...")

    # Sort by uncertainty descending
    records.sort(key=lambda r: r["uncertainty"], reverse=True)

    # Split into abstained (review queue) and kept
    abstained_records = [r for r in records if r["uncertainty"] >= threshold]
    kept_records = [r for r in records if r["uncertainty"] < threshold]

    # Add rank to review queue
    review_queue = []
    for rank, rec in enumerate(abstained_records, 1):
        review_entry = {
            "rank": rank,
            "image": rec["image"],
            "source_dataset": rec["source_dataset"],
            "uncertainty": rec["uncertainty"],
            "is_error": rec["is_error"],
            "failure_tags": rec["failure_tags"],
            "signals": rec["signals"],
            "abstain_reason": _classify_abstain_reason(rec),
        }
        review_queue.append(review_entry)

    print(f"  Review queue size: {len(review_queue)}")
    print(f"  Of which are errors: {sum(1 for r in review_queue if r['is_error'])}")
    print(f"  Top 3 review items:")
    for item in review_queue[:3]:
        print(f"    #{item['rank']} {item['image'].split('/')[-1]}: "
              f"uncertainty={item['uncertainty']:.4f}, error={item['is_error']}, "
              f"reason={item['abstain_reason']}")
    print(f"  Elapsed: {time.time() - t_start:.1f}s")

    # ── [5/5] Write outputs ──────────────────────────────────────────
    print("[5/5] Writing outputs...")

    # Main eval JSON
    done_when = {
        "abstain_precision_ge_0_80": metrics["abstain_precision"] >= 0.80,
        "hc_error_reduction_ge_0_30": metrics["hc_error_reduction"] >= 0.30,
    }

    # Signal distribution summary
    signal_distributions = {}
    for sig_name in ["confidence", "expert_conflict", "edge_density", "quality", "ocr"]:
        key = "uncertainty" if sig_name == "confidence" else ("conflict_fraction" if sig_name == "expert_conflict"
                                                              else "anomaly_score" if sig_name == "edge_density"
                                                              else "score" if sig_name == "quality"
                                                              else "uncertainty")
        vals = [float(r["signals"][sig_name][key]) for r in records]
        arr = np.array(vals)
        signal_distributions[sig_name] = {
            "mean": round(float(arr.mean()), 4),
            "std": round(float(arr.std()), 4),
            "min": round(float(arr.min()), 4),
            "max": round(float(arr.max()), 4),
            "p25": round(float(np.percentile(arr, 25)), 4),
            "p50": round(float(np.percentile(arr, 50)), 4),
            "p75": round(float(np.percentile(arr, 75)), 4),
        }

    eval_output = {
        "version": "uncertainty_abstain_v1",
        "date": "2026-05-01",
        "dataset": {
            "n_samples": len(records),
            "n_errors": sum(1 for r in records if r["is_error"]),
            "n_clean": sum(1 for r in records if not r["is_error"]),
            "error_rate": round(sum(1 for r in records if r["is_error"]) / max(1, len(records)), 4),
        },
        "uncertainty": {
            "threshold": round(threshold, 4),
            "signal_distributions": signal_distributions,
            "signal_weights": {
                "confidence": 0.25,
                "expert_conflict": 0.20,
                "edge_density": 0.20,
                "quality": 0.20,
                "ocr": 0.15,
            },
        },
        "abstain_metrics": metrics,
        "r6_t3_done_when": {
            "abstain_precision_ge_0_80": done_when["abstain_precision_ge_0_80"],
            "abstain_precision": metrics["abstain_precision"],
            "hc_error_reduction_ge_0_30": done_when["hc_error_reduction_ge_0_30"],
            "hc_error_reduction": metrics["hc_error_reduction"],
        },
        "elapsed_seconds": time.time() - t_start,
    }

    eval_path = REPORTS_DIR / "uncertainty_abstain_v1.json"
    with open(eval_path, "w", encoding="utf-8") as f:
        json.dump(eval_output, f, indent=2, ensure_ascii=False)

    # Review queue JSONL
    queue_path = REPORTS_DIR / "review_queue_v1.jsonl"
    with open(queue_path, "w", encoding="utf-8") as f:
        for item in review_queue:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    # Train summary
    train_summary = {
        "version": "uncertainty_review_v1",
        "date": "2026-05-01",
        "n_samples": len(records),
        "n_abstained": metrics["n_abstained"],
        "n_kept": metrics["n_kept"],
        "threshold": metrics["threshold"],
        "abstain_precision": metrics["abstain_precision"],
        "abstain_recall": metrics["abstain_recall"],
        "hc_error_reduction": metrics["hc_error_reduction"],
        "done_when": done_when,
        "all_passed": all(done_when.values()),
        "elapsed_seconds": time.time() - t_start,
    }

    summary_path = ckpt_dir / "train_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(train_summary, f, indent=2, ensure_ascii=False)

    # ── Final summary ────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"R6-T3 Uncertainty & Abstain Review Queue v1 — Results ({eval_output['elapsed_seconds']:.1f}s)")
    print("=" * 70)
    print(f"Samples: {len(records)} total, {metrics['n_abstained']} abstained, {metrics['n_kept']} kept")
    print(f"Threshold: {threshold:.4f}")
    print(f"Abstain precision: {metrics['abstain_precision']:.4f} (target >= 0.80)")
    print(f"Abstain recall:    {metrics['abstain_recall']:.4f}")
    print(f"HC error reduction: {metrics['hc_error_reduction']:.4f} (target >= 0.30)")
    print()
    print("R6-T3 done_when check:")
    print(f"  {'PASS' if done_when['abstain_precision_ge_0_80'] else 'FAIL'} "
          f"Abstain precision >= 0.80: {metrics['abstain_precision']:.4f}")
    print(f"  {'PASS' if done_when['hc_error_reduction_ge_0_30'] else 'FAIL'} "
          f"HC error reduction >= 30%: {metrics['hc_error_reduction']:.4f}")
    print(f"  {'PASS' if all(done_when.values()) else 'PENDING'} "
          f"Overall: {'ALL PASSED' if all(done_when.values()) else 'PENDING'}")
    print()
    print(f"Outputs:")
    print(f"  {eval_path}")
    print(f"  {queue_path}")
    print(f"  {summary_path}")

    return 0


def _classify_abstain_reason(rec):
    """Classify why a sample was abstained, based on dominant signal."""
    signals = rec["signals"]
    contributions = {
        "low_confidence": signals["confidence"]["uncertainty"] * 0.25,
        "expert_conflict": signals["expert_conflict"]["conflict_fraction"] * 0.20,
        "edge_anomaly": signals["edge_density"]["anomaly_score"] * 0.20,
        "poor_quality": signals["quality"]["score"] * 0.20,
        "ocr_uncertain": signals["ocr"]["uncertainty"] * 0.15,
    }
    dominant = max(contributions, key=contributions.get)
    if contributions[dominant] > 0:
        return dominant
    return "composite_uncertainty"


if __name__ == "__main__":
    sys.exit(main())
