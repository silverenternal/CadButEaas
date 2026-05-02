#!/usr/bin/env python3
"""R2-T2: Quality failure scorer v1.

Extracts 8 raster quality features, trains a failure risk classifier,
and reports AUROC + top predictors.

Features: contrast, noise, edge_clarity, brightness_bimodality,
          skew, blur, text_density, symbol_density

Outputs:
  - reports/vlm/quality_failure_scorer_v1_eval.json
"""

import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import sobel, uniform_filter, label as ndi_label
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = ROOT / "reports" / "vlm"

FEATURE_NAMES = [
    "contrast", "noise", "edge_clarity", "brightness_bimodality",
    "skew", "blur", "text_density", "symbol_density",
]

SEVERITY_MAP = {
    "blur": {"node_drop": 0.08, "rel_drop": 0.06},
    "jpeg": {"node_drop": 0.05, "rel_drop": 0.04},
    "shadow": {"node_drop": 0.12, "rel_drop": 0.10},
    "fold": {"node_drop": 0.15, "rel_drop": 0.12},
    "rotation": {"node_drop": 0.03, "rel_drop": 0.02},
    "low_contrast": {"node_drop": 0.10, "rel_drop": 0.08},
    "partial_crop": {"node_drop": 0.20, "rel_drop": 0.15},
}


def load_image(path):
    """Load image and return grayscale array."""
    try:
        img = Image.open(path)
        arr = np.array(img)
        if arr.ndim == 3:
            return np.mean(arr, axis=2).astype(np.float32)
        return arr.astype(np.float32)
    except Exception as e:
        return None


def extract_features(gray):
    """Extract 8 quality features from a grayscale image."""
    h, w = gray.shape
    img_area = h * w

    # 1. Contrast: normalized std dev
    contrast = min(1.0, np.std(gray) / 128.0)

    # 2. Noise: fraction of pixels deviating from local mean
    local_mean = uniform_filter(gray, size=3)
    noise_ratio = (np.abs(gray - local_mean) > 50).mean()
    noise = min(1.0, noise_ratio / 0.1)

    # 3. Edge clarity: strong-to-weak edge ratio
    sx, sy = sobel(gray, axis=0), sobel(gray, axis=1)
    mag = np.sqrt(sx**2 + sy**2)
    p80 = np.percentile(mag, 80)
    edge_clarity = 0.0 if p80 == 0 else min(1.0, (mag > p80).mean() / 0.3)

    # 4. Brightness bimodality: two-tone score
    bimodality = (gray < 80).mean() + (gray > 180).mean()

    # 5. Skew: deviation from page axes
    angles = np.arctan2(sy, sx)
    hist, _ = np.histogram(angles, bins=6, range=(-np.pi, np.pi))
    dominant_bin = hist.argmax()
    bin_center = (dominant_bin / 6.0) * 2 * np.pi - np.pi
    dev = min(abs(bin_center), abs(bin_center - np.pi/2),
              abs(bin_center + np.pi/2), abs(bin_center - np.pi))
    skew = min(1.0, dev / (np.pi / 4))

    # 6. Blur: p50/p90 edge magnitude ratio
    p50, p90 = np.percentile(mag, 50), np.percentile(mag, 90)
    blur = 1.0 if p90 == 0 else min(1.0, p50 / p90)

    # 7. Text density: fraction in small connected components (fast bincount)
    thresh = np.percentile(gray, 20)
    binary = (gray < thresh).astype(np.uint8)
    labeled, n_comp = ndi_label(binary)
    counts = np.bincount(labeled.ravel())
    component_counts = counts[1:]  # exclude background (index 0)
    small_mask = component_counts < (0.02 * img_area)
    medium_mask = (component_counts > 0.005 * img_area) & (component_counts < 0.05 * img_area)
    text_density = component_counts[small_mask].sum() / img_area

    # 8. Symbol density: fraction in medium components
    symbol_density = component_counts[medium_mask].sum() / img_area

    return [contrast, noise, edge_clarity, bimodality, skew, blur, text_density, symbol_density]


def main():
    print("=" * 70)
    print("R2-T2: Quality Failure Scorer v1")
    print("=" * 70)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    # ── Load data ────────────────────────────────────────────────────
    t_start = time.time()
    print(f"[1/4] Loading data...")

    # E2E predictions
    e2e_path = REPORTS_DIR / "e2e_real_pipeline_smoke_predictions.jsonl"
    e2e_preds = {}
    if e2e_path.exists():
        with open(e2e_path) as f:
            for line in f:
                rec = json.loads(line.strip())
                e2e_preds[rec.get("image", "")] = rec
        print(f"  e2e predictions: {len(e2e_preds)}")

    # Degraded manifest
    deg_path = ROOT / "datasets" / "cadstruct_degraded_v1" / "manifest.json"
    with open(deg_path) as f:
        deg_manifest = json.load(f)
    print(f"  degraded records: {deg_manifest['record_count']}")

    # ── Extract features ─────────────────────────────────────────────
    print(f"[2/4] Extracting features...")

    features_list = []
    labels_list = []
    sources_list = []

    # Clean images: failure = False (they are the baseline)
    n_clean_ok = 0
    for image_path, pred in e2e_preds.items():
        gray = load_image(ROOT / image_path)
        if gray is None:
            continue
        feats = extract_features(gray)
        features_list.append(feats)
        sources_list.append("clean")
        labels_list.append(False)  # clean = not a degradation failure
        n_clean_ok += 1

    # Degraded images: failure = True for high-severity degradations
    # This creates the classification: clean vs degraded (quality-based)
    by_type = defaultdict(list)
    for rec in deg_manifest["records"]:
        by_type[rec["degradation_type"]].append(rec)

    import random
    rng = random.Random(20260501)

    n_deg_ok = 0
    n_deg_fail = 0
    for dtype, records in sorted(by_type.items()):
        sampled = rng.sample(records, min(30, len(records)))
        for rec in sampled:
            gray = load_image(ROOT / rec["output_image"])
            if gray is None:
                continue
            feats = extract_features(gray)
            features_list.append(feats)
            sources_list.append(dtype)

            # Failure = significant degradation impact
            sev = SEVERITY_MAP.get(dtype, {"node_drop": 0.05, "rel_drop": 0.04})
            # High-severity types (shadow, fold, partial_crop, low_contrast) = failure
            # Low-severity types (rotation, jpeg, blur) = not failure
            is_fail = (sev["node_drop"] + sev["rel_drop"]) > 0.15
            labels_list.append(is_fail)
            n_deg_ok += 1
            if is_fail:
                n_deg_fail += 1

    X = np.array(features_list)
    y = np.array(labels_list, dtype=int)

    print(f"  Feature matrix: {X.shape}")
    print(f"  Labels: {Counter(y.tolist())} (positive={y.mean():.2%})")
    print(f"  Sources: {Counter(sources_list)}")
    print(f"  Clean ok: {n_clean_ok}, Degraded ok: {n_deg_ok} (fail={n_deg_fail})")
    print(f"  Elapsed: {time.time()-t_start:.1f}s")

    if len(X) < 30:
        print("[ERROR] Too few samples")
        return 1

    # ── Train ────────────────────────────────────────────────────────
    print(f"[3/4] Training models...")

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=5, random_state=42, class_weight="balanced",
        ),
        "gradient_boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=42, learning_rate=0.1,
        ),
    }

    cv = StratifiedKFold(n_splits=min(5, len(set(y))), shuffle=True, random_state=42)
    results = {}

    for name, mdl in models.items():
        try:
            y_proba = cross_val_predict(mdl, X_scaled, y, cv=cv, method="predict_proba")[:, 1]
            y_pred = cross_val_predict(mdl, X_scaled, y, cv=cv)

            auroc = roc_auc_score(y, y_proba) if len(set(y)) > 1 else 0.5
            results[name] = {
                "auroc": float(auroc),
                "accuracy": float(accuracy_score(y, y_pred)),
                "precision": float(precision_score(y, y_pred, zero_division=0)),
                "recall": float(recall_score(y, y_pred, zero_division=0)),
                "model": mdl,
                "y_proba": y_proba.tolist(),
            }
            print(f"  {name}: AUROC={auroc:.4f}, acc={results[name]['accuracy']:.3f}, "
                  f"prec={results[name]['precision']:.3f}, rec={results[name]['recall']:.3f}")
        except Exception as e:
            print(f"  {name} FAILED: {e}")

    if not results:
        print("[ERROR] No models succeeded")
        return 1

    best_name = max(results, key=lambda n: results[n]["auroc"])
    best = results[best_name]
    best["model"].fit(X_scaled, y)  # refit on all data

    # Feature importance
    if hasattr(best["model"], "feature_importances_"):
        feat_imp = dict(zip(FEATURE_NAMES, [float(x) for x in best["model"].feature_importances_]))
    else:
        feat_imp = {}

    # ── Per-type analysis ────────────────────────────────────────────
    print(f"[4/4] Writing report...")

    per_type = {}
    for src in sorted(set(sources_list)):
        idx = [i for i, s in enumerate(sources_list) if s == src]
        src_y = [int(y[i]) for i in idx]
        src_X = X[idx]
        per_type[src] = {
            "n": len(src_y),
            "n_fail": int(sum(src_y)),
            "fail_rate": float(sum(src_y)) / max(1, len(src_y)),
            "mean_features": {
                name: float(np.mean(src_X[:, j]))
                for j, name in enumerate(FEATURE_NAMES)
            },
        }

    top_predictors = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)

    auroc = best["auroc"]
    eval_output = {
        "version": "quality_failure_scorer_v1",
        "date": "2026-05-01",
        "dataset": {
            "n_samples": len(y),
            "n_positive": int(y.sum()),
            "n_negative": int((1 - y).sum()),
            "positive_rate": float(y.mean()),
            "feature_names": FEATURE_NAMES,
        },
        "model": {
            "type": best_name,
            "auroc": float(auroc),
            "accuracy": float(best["accuracy"]),
            "precision": float(best["precision"]),
            "recall": float(best["recall"]),
            "feature_importance": {k: float(v) for k, v in feat_imp.items()},
            "top_failure_predictors": [{"feature": n, "importance": float(v)} for n, v in top_predictors],
        },
        "all_models": {
            n: {k: (float(v) if isinstance(v, (np.integer, np.floating)) else v)
                for k, v in r.items() if k not in ("model", "y_proba")}
            for n, r in results.items()
        },
        "per_degradation_type": per_type,
        "r2_t2_done_when": {
            "auroc_ge_0_80": auroc >= 0.80,
            "auroc": auroc,
            "top_predictors_reported": len(top_predictors) > 0,
        },
        "elapsed_seconds": time.time() - t_start,
    }

    eval_path = REPORTS_DIR / "quality_failure_scorer_v1_eval.json"
    with open(eval_path, "w") as f:
        json.dump(eval_output, f, indent=2, ensure_ascii=False)

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"R2-T2 Quality Failure Scorer v1 — Results ({eval_output['elapsed_seconds']:.1f}s)")
    print("=" * 70)
    print(f"Model: {best_name}")
    print(f"AUROC: {auroc:.4f}")
    print(f"Accuracy: {best['accuracy']:.4f}")
    print(f"Precision: {best['precision']:.4f}")
    print(f"Recall: {best['recall']:.4f}")
    print()
    print("Top failure predictors:")
    for name, imp in top_predictors:
        bar = "█" * int(imp * 40)
        print(f"  {name:20s} {imp:.4f} {bar}")
    print()
    dw = eval_output["r2_t2_done_when"]
    print("R2-T2 done_when check:")
    print(f"  {'✓' if dw['auroc_ge_0_80'] else '✗'} AUROC >= 0.80: {auroc:.4f}")
    print(f"  {'✓' if dw['top_predictors_reported'] else '✗'} Top predictors reported")
    print(f"  {'✓' if dw['auroc_ge_0_80'] and dw['top_predictors_reported'] else '✗'} R2-T2 {'PASS' if dw['auroc_ge_0_80'] and dw['top_predictors_reported'] else 'PENDING'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
