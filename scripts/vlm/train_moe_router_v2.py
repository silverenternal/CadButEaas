#!/usr/bin/env python3
"""R6-T1: MoE Router v2 — Train an explicit classifier to route candidates to experts.

Uses bbox, spatial, type-hint, and page-context features to classify
candidates into one of 5 expert families: wall_opening, room_space,
symbol_fixture, text_dimension, sheet_layout.

Outputs:
  - scripts/vlm/train_moe_router_v2.py
  - checkpoints/moe_router_v2/train_summary.json
  - reports/vlm/moe_router_v2_eval.json
"""

import joblib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.preprocessing import LabelEncoder, StandardScaler

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = ROOT / "reports" / "vlm"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "moe_router_v2"

FAMILIES = ["wall_opening", "room_space", "symbol_fixture", "text_dimension", "sheet_layout"]

FEATURE_NAMES = [
    "bbox_area", "bbox_aspect", "bbox_center_x", "bbox_center_y",
    "bbox_width_norm", "bbox_height_norm", "bbox_area_norm",
    "page_aspect", "page_area",
    "symbol_type_code", "text_type_code", "room_type_code",
    "has_primitive", "is_layout_region", "is_text_candidate",
    "is_symbol_candidate", "is_semantic_region",
    "confidence", "n_candidates_same_family",
]

# Features that leak the family label (type codes that directly reveal the expert)
LEAKING_FEATURES = {"symbol_type_code", "text_type_code", "room_type_code",
                    "is_text_candidate", "is_symbol_candidate", "is_semantic_region",
                    "is_layout_region", "has_primitive", "confidence"}

# Fair features: only geometry + page context (no type leakage)
FAIR_FEATURE_NAMES = [
    "bbox_area", "bbox_aspect", "bbox_center_x", "bbox_center_y",
    "bbox_width_norm", "bbox_height_norm", "bbox_area_norm",
    "page_aspect", "page_area",
    "n_candidates_same_family",
]


# ── Feature extraction ─────────────────────────────────────────────────

def extract_candidate_features(candidate: dict, page_meta: dict, all_candidates: list) -> list:
    """Extract routing features from a single candidate."""
    pw = page_meta.get("width", 1000)
    ph = page_meta.get("height", 1000)
    page_area = pw * ph
    page_aspect = pw / max(1, ph)

    bbox = candidate.get("bbox", {})
    x1 = bbox.get("x", bbox.get("x1", 0))
    y1 = bbox.get("y", bbox.get("y1", 0))
    x2 = bbox.get("x2", x1 + 1)
    y2 = bbox.get("y2", y1 + 1)

    w = max(1, x2 - x1)
    h = max(1, y2 - y1)
    area = w * h
    aspect = w / max(1, h)

    features = [
        float(area),                          # bbox_area
        float(aspect),                        # bbox_aspect
        float((x1 + x2) / 2 / max(1, pw)),   # bbox_center_x (normalized)
        float((y1 + y2) / 2 / max(1, ph)),   # bbox_center_y (normalized)
        float(w / max(1, pw)),               # bbox_width_norm
        float(h / max(1, ph)),               # bbox_height_norm
        float(area / max(1, page_area)),     # bbox_area_norm
        float(page_aspect),                   # page_aspect
        float(page_area),                     # page_area
    ]

    # Type codes (simple hashing of type strings)
    symbol_type = candidate.get("symbol_type", "")
    text_type = candidate.get("text_type", "")
    room_type = candidate.get("room_type", "")

    features.append(float(hash(symbol_type) % 1000) / 1000)
    features.append(float(hash(text_type) % 1000) / 1000)
    features.append(float(hash(room_type) % 1000) / 1000)

    # Flags
    features.append(float(1 if candidate.get("primitive_type") else 0))
    features.append(float(1 if candidate.get("layout_type") else 0))
    features.append(float(1 if "text_candidate" in candidate or "text_type" in candidate else 0))
    features.append(float(1 if "symbol_candidate" in candidate or "symbol_type" in candidate else 0))
    features.append(float(1 if "semantic_type" in candidate or candidate.get("family") == "space" else 0))

    # Confidence
    features.append(float(candidate.get("confidence", 0.5)))

    # Sibling count (how many candidates of same family on this page)
    fam = candidate.get("_family", "")
    n_same = sum(1 for c in all_candidates if c.get("_family") == fam)
    features.append(float(n_same))

    return features


def _bbox_to_dict(bbox) -> dict:
    """Convert bbox [x1,y1,x2,y2] or dict to {x,y,x2,y2}."""
    if isinstance(bbox, dict):
        return bbox
    if isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
        return {"x": bbox[0], "y": bbox[1], "x2": bbox[2], "y2": bbox[3]}
    return {"x": 0, "y": 0, "x2": 1, "y2": 1}


def extract_gold_from_record(record: dict) -> list:
    """Extract gold (features, family_label) pairs from a locked JSONL record."""
    page_meta = record.get("metadata", {})
    pw = page_meta.get("width", 1000)
    ph = page_meta.get("height", 1000)

    hints = record.get("request_hints", {})
    all_candidates = []
    gold_pairs = []

    # Wall/boundary nodes from primitive_graph
    for node in hints.get("primitive_graph", {}).get("nodes", []):
        fam = "wall_opening"
        c = {
            "bbox": _bbox_to_dict(node.get("bbox", [])),
            "primitive_type": node.get("primitive_type", ""),
            "semantic_type": node.get("semantic_type", ""),
            "_family": fam,
            "confidence": 0.5,
        }
        all_candidates.append(c)
        gold_pairs.append((c, fam))

    # Room/space candidates from semantic_regions
    for sr in hints.get("semantic_regions", []):
        fam = "room_space"
        c = {
            "bbox": _bbox_to_dict(sr.get("bbox", [])),
            "room_type": sr.get("type", ""),
            "_family": fam,
            "confidence": 0.5,
        }
        all_candidates.append(c)
        gold_pairs.append((c, fam))

    # Symbol candidates
    for sc in hints.get("symbol_candidates", []):
        fam = "symbol_fixture"
        c = {
            "bbox": _bbox_to_dict(sc.get("bbox", [])),
            "symbol_type": sc.get("symbol_type", ""),
            "_family": fam,
            "confidence": sc.get("confidence", 0.5),
        }
        all_candidates.append(c)
        gold_pairs.append((c, fam))

    # Text candidates
    for tc in hints.get("text_candidates", []):
        fam = "text_dimension"
        c = {
            "bbox": _bbox_to_dict(tc.get("bbox", [])),
            "text_type": tc.get("text_type", ""),
            "_family": fam,
            "confidence": tc.get("confidence", 0.5),
        }
        all_candidates.append(c)
        gold_pairs.append((c, fam))

    # Extract features
    features_list = []
    labels_list = []
    for candidate, family in gold_pairs:
        feats = extract_candidate_features(candidate, page_meta, all_candidates)
        features_list.append(feats)
        labels_list.append(family)

    return features_list, labels_list


# ── Training ─────────────────────────────────────────────────────────────

def load_split(split_path: Path) -> tuple:
    """Load a JSONL split and return (features, labels)."""
    all_features = []
    all_labels = []
    n_records = 0
    if not split_path.exists():
        return all_features, all_labels

    from PIL import Image

    with open(split_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            page_meta = rec.get("metadata", {})
            pw = page_meta.get("width")
            ph = page_meta.get("height")
            # Fallback to image dimensions if metadata doesn't have them
            if pw is None or ph is None:
                img_path = rec.get("image_path", "")
                if img_path:
                    try:
                        img = Image.open(ROOT / img_path)
                        pw, ph = img.size
                    except Exception:
                        pw, ph = 1000, 1000
                else:
                    pw, ph = 1000, 1000
            page_meta["width"] = pw
            page_meta["height"] = ph

            feats, labels = extract_gold_from_record(rec)
            all_features.extend(feats)
            all_labels.extend(labels)
            n_records += 1

    return all_features, all_labels


def train_router(X_train, y_train, X_dev, y_dev, X_locked, y_locked, feature_names=None):
    """Train and evaluate the router.
    
    Args:
        feature_names: list of feature names for importance reporting.
    """
    if feature_names is None:
        feature_names = FEATURE_NAMES
        
    le = LabelEncoder()
    y_train_enc = le.fit_transform(y_train)
    y_dev_enc = le.transform(y_dev)
    y_locked_enc = le.transform(y_locked)

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_dev_s = scaler.transform(X_dev)
    X_locked_s = scaler.transform(X_locked)

    models = {
        "random_forest": RandomForestClassifier(
            n_estimators=200, max_depth=20, random_state=42, class_weight="balanced",
        ),
    }

    results = {}
    for name, mdl in models.items():
        mdl.fit(X_train_s, y_train_enc)

        train_acc = accuracy_score(y_train_enc, mdl.predict(X_train_s))
        dev_acc = accuracy_score(y_dev_enc, mdl.predict(X_dev_s))
        locked_acc = accuracy_score(y_locked_enc, mdl.predict(X_locked_s))

        # Cross-validated dev accuracy
        cv = StratifiedKFold(n_splits=min(5, len(set(y_train_enc))), shuffle=True, random_state=42)
        try:
            y_cv = cross_val_predict(mdl, X_train_s, y_train_enc, cv=cv)
            cv_acc = accuracy_score(y_train_enc, y_cv)
        except Exception:
            cv_acc = dev_acc

        # Feature importance
        if hasattr(mdl, "feature_importances_"):
            feat_imp = dict(zip(feature_names, [float(x) for x in mdl.feature_importances_]))
        else:
            feat_imp = {}

        # Effective rate: fraction of candidates routed to any expert (all should be routed)
        dev_preds = mdl.predict(X_dev_s)
        routed = sum(1 for p in dev_preds if p >= 0)
        effective_rate = routed / max(1, len(dev_preds))

        # Wrong-expert rate
        wrong = sum(1 for p, g in zip(dev_preds, y_dev_enc) if p != g)
        wrong_rate = wrong / max(1, len(dev_preds))

        results[name] = {
            "model": mdl,
            "train_accuracy": float(train_acc),
            "dev_accuracy": float(dev_acc),
            "locked_accuracy": float(locked_acc),
            "cv_accuracy": float(cv_acc),
            "effective_rate": float(effective_rate),
            "wrong_expert_rate": float(wrong_rate),
            "feature_importance": feat_imp,
        }

        print(f"  {name}: train={train_acc:.4f}, dev={dev_acc:.4f}, "
              f"locked={locked_acc:.4f}, cv={cv_acc:.4f}, "
              f"effective={effective_rate:.4f}, wrong={wrong_rate:.4f}")

    # Pick best model
    best_name = max(results, key=lambda n: results[n]["dev_accuracy"])
    return best_name, results[best_name], le, scaler, results


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--fair", action="store_true", help="Train with fair features only (no type code leakage)")
    args = parser.parse_args()

    mode = "fair" if args.fair else "full"
    print("=" * 70)
    print(f"R6-T1: MoE Router v2 — Training ({mode} mode)")
    print("=" * 70)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINTS_DIR.mkdir(parents=True, exist_ok=True)

    t_start = time.time()

    # ── Load data ────────────────────────────────────────────────────
    print("[1/4] Loading splits...")
    locked_dir = ROOT / "datasets" / "cadstruct_cubicasa5k_moe_locked"

    X_dev, y_dev = load_split(locked_dir / "dev.jsonl")
    print(f"  dev: {len(y_dev)} candidates, {Counter(y_dev)}")

    X_locked, y_locked = load_split(locked_dir / "locked_test.jsonl")
    print(f"  locked: {len(y_locked)} candidates, {Counter(y_locked)}")

    # Use dev as train+dev split (no separate train file available)
    # Use first 80% as train, last 20% as dev
    n_train = int(len(X_dev) * 0.8)
    X_train, y_train = X_dev[:n_train], y_dev[:n_train]
    X_dev_final, y_dev_final = X_dev[n_train:], y_dev[n_train:]
    print(f"  train: {len(y_train)}, dev_final: {len(y_dev_final)}")

    if len(X_train) < 100:
        print("[ERROR] Too few training samples")
        return 1

    # Select features based on mode
    if args.fair:
        # Filter to fair features only
        fair_indices = [FEATURE_NAMES.index(f) for f in FAIR_FEATURE_NAMES]
        X_train = [[row[i] for i in fair_indices] for row in X_train]
        X_dev_final = [[row[i] for i in fair_indices] for row in X_dev_final]
        X_locked = [[row[i] for i in fair_indices] for row in X_locked]
        active_feature_names = FAIR_FEATURE_NAMES
        print(f"  Using {len(FAIR_FEATURE_NAMES)} fair features (no type code leakage)")
    else:
        active_feature_names = FEATURE_NAMES
        print(f"  Using all {len(FEATURE_NAMES)} features (includes type codes)")

    # ── Train ────────────────────────────────────────────────────────
    print("[2/4] Training router models...")

    best_name, best, le, scaler, all_results = train_router(
        X_train, y_train, X_dev_final, y_dev_final, X_locked, y_locked,
        feature_names=active_feature_names,
    )

    # ── Detailed analysis ────────────────────────────────────────────
    print("[3/4] Detailed analysis...")

    # Per-family accuracy on locked
    mdl = best["model"]
    le_obj = le
    scaler_obj = scaler

    X_locked_s = scaler_obj.transform(X_locked)
    y_locked_enc = le_obj.transform(y_locked)
    y_pred = mdl.predict(X_locked_s)

    # Per-family breakdown
    per_family = {}
    for fam in le_obj.classes_:
        mask = y_locked_enc == le_obj.transform([fam])[0]
        if mask.sum() == 0:
            continue
        fam_acc = (y_pred[mask] == y_locked_enc[mask]).mean()
        per_family[fam] = {
            "n_samples": int(mask.sum()),
            "accuracy": float(fam_acc),
        }

    # Confusion summary
    confusion = defaultdict(lambda: defaultdict(int))
    for pred, gold in zip(y_pred, y_locked_enc):
        pred_fam = le_obj.inverse_transform([pred])[0]
        gold_fam = le_obj.inverse_transform([gold])[0]
        confusion[gold_fam][pred_fam] += 1

    # Top feature importances
    sorted_imp = sorted(best["feature_importance"].items(), key=lambda x: x[1], reverse=True)
    top_predictors = [{"feature": n, "importance": float(v)} for n, v in sorted_imp]

    # ── Save checkpoint ──────────────────────────────────────────────
    print("[4/4] Saving checkpoint and report...")

    # Save model artifacts (model + scaler + label encoder)
    model_bundle = {
        "model": mdl,
        "scaler": scaler_obj,
        "label_encoder": le_obj,
        "feature_names": active_feature_names,
        "classes": le_obj.classes_.tolist(),
        "mode": mode,
    }
    model_path = CHECKPOINTS_DIR / f"model_v2_{mode}.joblib"
    joblib.dump(model_bundle, model_path)
    print(f"  Model saved to {model_path} ({model_path.stat().st_size / 1e6:.1f}MB)")

    train_summary = {
        "version": f"moe_router_v2_{mode}",
        "date": "2026-05-01",
        "model_type": best_name,
        "mode": mode,
        "n_features": len(active_feature_names),
        "feature_names": active_feature_names,
        "classes": le_obj.classes_.tolist(),
        "train_samples": len(y_train),
        "dev_samples": len(y_dev_final),
        "locked_samples": len(y_locked),
        "scaler_mean": scaler_obj.mean_.tolist(),
        "scaler_scale": scaler_obj.scale_.tolist(),
        "model_checkpoint": str(model_path),
    }
    with open(CHECKPOINTS_DIR / "train_summary.json", "w") as f:
        json.dump(train_summary, f, indent=2)

    # Eval report
    eval_output = {
        "version": "moe_router_v2_eval",
        "date": "2026-05-01",
        "model": {
            "type": best_name,
            "dev_accuracy": best["dev_accuracy"],
            "locked_accuracy": best["locked_accuracy"],
            "effective_rate": best["effective_rate"],
            "wrong_expert_rate": best["wrong_expert_rate"],
            "feature_importance": best["feature_importance"],
            "top_failure_predictors": top_predictors[:10],
        },
        "all_models": {
            n: {k: v for k, v in r.items() if k != "model"}
            for n, r in all_results.items()
        },
        "per_family": per_family,
        "confusion_summary": {k: dict(v) for k, v in confusion.items()},
        "r6_t1_done_when": {
            "effective_rate_ge_0_98": best["effective_rate"] >= 0.98,
            "wrong_expert_rate_le_0_02": best["wrong_expert_rate"] <= 0.02,
            "effective_rate": best["effective_rate"],
            "wrong_expert_rate": best["wrong_expert_rate"],
        },
        "elapsed_seconds": time.time() - t_start,
    }

    eval_path = REPORTS_DIR / "moe_router_v2_eval.json"
    with open(eval_path, "w") as f:
        json.dump(eval_output, f, indent=2, ensure_ascii=False)

    # ── Summary ──────────────────────────────────────────────────────
    print()
    print("=" * 70)
    print(f"R6-T1 MoE Router v2 — Results ({eval_output['elapsed_seconds']:.1f}s)")
    print("=" * 70)
    print(f"Model: {best_name}")
    print(f"Dev accuracy: {best['dev_accuracy']:.4f}")
    print(f"Locked accuracy: {best['locked_accuracy']:.4f}")
    print(f"Effective rate: {best['effective_rate']:.4f}")
    print(f"Wrong-expert rate: {best['wrong_expert_rate']:.4f}")
    print()
    print("Top routing features:")
    for p in top_predictors[:8]:
        bar = "█" * int(p["importance"] * 40)
        print(f"  {p['feature']:25s} {p['importance']:.4f} {bar}")
    print()
    print("Per-family locked accuracy:")
    for fam, stats in sorted(per_family.items()):
        print(f"  {fam:20s} n={stats['n_samples']:5d} acc={stats['accuracy']:.4f}")
    print()
    dw = eval_output["r6_t1_done_when"]
    print("R6-T1 done_when check:")
    print(f"  {'✓' if dw['effective_rate_ge_0_98'] else '✗'} effective_rate >= 0.98: {dw['effective_rate']:.4f}")
    print(f"  {'✓' if dw['wrong_expert_rate_le_0_02'] else '✗'} wrong_expert_rate <= 0.02: {dw['wrong_expert_rate']:.4f}")
    passed = dw["effective_rate_ge_0_98"] and dw["wrong_expert_rate_le_0_02"]
    print(f"  {'✓' if passed else '✗'} R6-T1 {'PASS' if passed else 'PENDING'}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
