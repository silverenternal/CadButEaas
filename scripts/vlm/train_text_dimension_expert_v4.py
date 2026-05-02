#!/usr/bin/env python3
"""TextDimensionExpert v4: sklearn-based classifier with full OCR features.

Upgrades from v3's prototype+scoring approach to a learned ExtraTrees
classifier with engineered features from:
  - bbox geometry (6D)
  - layout/role features (4D)
  - OCR text content patterns (10D)
  - page context features (4D)

Trains a separate model for text_type classification and a rule-based
dimension relation linker.

Input: datasets/text_dimension_expert_v4_full_ocr/{train,dev,smoke,locked_test}.jsonl
Output: checkpoints/text_dimension_expert_v4/model_v4.json
        reports/vlm/text_dimension_expert_v4_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import re
import resource
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

try:
    from sklearn.ensemble import ExtraTreesClassifier
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import LabelEncoder
except ImportError:
    print("ERROR: scikit-learn is required. pip install scikit-learn")
    import sys
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS_DIR = ROOT / "reports" / "vlm"
CHECKPOINTS_DIR = ROOT / "checkpoints" / "text_dimension_expert_v4"

FEATURE_NAMES = [
    # bbox geometry (6)
    "cx", "cy", "width", "height", "area", "log_aspect",
    # layout/role (4)
    "y_bucket_neg", "y_bucket_origin", "is_narrow_vertical", "is_horizontal_bar",
    # OCR text content (10)
    "has_raw_text", "text_len", "word_count", "is_numeric", "has_x_separator",
    "has_dimension_unit", "has_foot_inch", "is_alpha_only", "has_digit_and_alpha",
    "is_short_label",
    # page context (4)
    "page_norm_cx", "page_norm_cy", "page_norm_w", "page_norm_h",
    # text-pattern scores (5) — direct rule-based signals for each class
    "score_dimension_line", "score_dimension_text", "score_leader_line",
    "score_note_text", "score_room_label",
]


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def normalize_bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) != 4:
        return None
    try:
        return [float(x) for x in value]
    except (TypeError, ValueError):
        return None


def normalize_text(text: str) -> str:
    if not text:
        return ""
    text = text.strip().lower()
    text = re.sub(r"[\s_\-]+", " ", text)
    text = re.sub(r"[^a-z0-9\s.]", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def compute_text_pattern_scores(raw: str, normalized: str, w: float, h: float, y1: float, y2: float) -> dict[str, float]:
    """Compute rule-based scores for each text type.

    Returns {label: score} dict in [0, 1] range.
    These complement the learned features by providing strong prior knowledge.
    """
    scores = {
        "score_dimension_line": 0.0,
        "score_dimension_text": 0.0,
        "score_leader_line": 0.0,
        "score_note_text": 0.0,
        "score_room_label": 0.0,
    }

    if not raw or not normalized:
        # No text content — rely on geometry
        if 9.0 <= w <= 15.0 and 14.0 <= h <= 25.0:
            scores["score_leader_line"] = 0.6
        if y1 <= -4.5 and y2 >= -0.5 and abs(h) <= 11.0 and w <= 15.0:
            scores["score_dimension_line"] = 0.5
        return scores

    # dimension_text: ALWAYS has 'x' separator
    if 'x' in normalized.lower() or '×' in raw:
        scores["score_dimension_text"] = 0.95
        return scores

    # room_label: alpha-only short strings (Finnish room names)
    cleaned = normalized.replace(' ', '').replace('.', '')
    is_alpha_only = cleaned.isalpha() and len(cleaned) > 0

    if is_alpha_only and len(cleaned) <= 25:
        scores["score_room_label"] = 0.9

    # note_text: floor indicators with digit + alpha
    has_digit_and_alpha = bool(re.search(r'\d', raw)) and bool(re.search(r'[a-zA-Z]', raw))
    if has_digit_and_alpha and 'x' not in normalized.lower():
        scores["score_note_text"] = 0.8

    # Pure-alpha notes longer than 5 chars
    if is_alpha_only and len(cleaned) > 5:
        scores["score_note_text"] = max(scores["score_note_text"], 0.4)

    # Short alpha-only labels (2-4 chars) could be either room_label or note_text
    if is_alpha_only and 2 <= len(cleaned) <= 4:
        scores["score_room_label"] = max(scores["score_room_label"], 0.6)
        scores["score_note_text"] = max(scores["score_note_text"], 0.3)

    return scores


def item_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    value = str(item.get("id") or "")
    match = re.search(r"(\d+)$", value)
    return (int(match.group(1)) if match else 10**9, value)


def extract_features(item: dict[str, Any]) -> dict[str, float]:
    """Extract all features for a single text candidate."""
    bbox = normalize_bbox(item.get("bbox"))
    if bbox is None:
        bbox = [0.0, 0.0, 1.0, 1.0]

    x1, y1, x2, y2 = bbox
    w = max(0.0, x2 - x1)
    h = max(0.0, y2 - y1)
    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area = w * h
    aspect = math.log((w + 1.0) / max(h + 1.0, 1e-6))

    # Layout/role features
    y_bucket_neg = float(y1 <= -9.5 and abs(y2) <= 0.5)
    y_bucket_origin = float(y1 < 0.0 and y2 <= 5.0 and w <= 15.0)
    is_narrow_vertical = float(9.0 <= w <= 15.0 and 14.0 <= h <= 25.0)
    is_horizontal_bar = float(y1 <= -4.5 and y2 >= -0.5 and abs(h) <= 11.0 and w <= 15.0)

    # OCR text content features
    raw = item.get("raw_text") or ""
    normalized = normalize_text(raw)
    text_len = len(normalized)
    word_count = len(normalized.split()) if normalized else 0
    is_numeric = float(bool(re.match(r"^[\d.]+$", normalized.replace(" ", ""))))
    has_x_separator = float('x' in normalized.lower() or '×' in raw)
    has_dimension_unit = float(bool(re.search(r"\d\s*(mm|cm|m|in|ft)", raw, re.I)))
    has_foot_inch = float(bool(re.search(r"\d'[\d\"]*", raw)))
    cleaned = normalized.replace(' ', '').replace('.', '')
    is_alpha_only = float(cleaned.isalpha() and len(cleaned) > 0)
    has_digit_and_alpha = float(bool(re.search(r'\d', raw)) and bool(re.search(r'[a-zA-Z]', raw)))
    is_short_label = float(len(cleaned) <= 4 and cleaned.isalpha())

    # Page context
    meta = item.get("_metadata") or {}
    img_w = float(meta.get("width") or 1.0)
    img_h = float(meta.get("height") or 1.0)
    page_norm_cx = cx / max(img_w, 1.0) if img_w > 1.0 else 0.5
    page_norm_cy = cy / max(img_h, 1.0) if img_h > 1.0 else 0.5
    page_norm_w = w / max(img_w, 1.0) if img_w > 1.0 else 0.01
    page_norm_h = h / max(img_h, 1.0) if img_h > 1.0 else 0.01

    # Text-pattern scores
    pattern_scores = compute_text_pattern_scores(raw, normalized, w, h, y1, y2)

    return {
        "cx": cx, "cy": cy, "width": w, "height": h, "area": area, "log_aspect": aspect,
        "y_bucket_neg": y_bucket_neg, "y_bucket_origin": y_bucket_origin,
        "is_narrow_vertical": is_narrow_vertical, "is_horizontal_bar": is_horizontal_bar,
        "has_raw_text": float(bool(raw)), "text_len": text_len, "word_count": word_count,
        "is_numeric": is_numeric, "has_x_separator": has_x_separator,
        "has_dimension_unit": has_dimension_unit, "has_foot_inch": has_foot_inch,
        "is_alpha_only": is_alpha_only, "has_digit_and_alpha": has_digit_and_alpha,
        "is_short_label": is_short_label,
        "page_norm_cx": page_norm_cx, "page_norm_cy": page_norm_cy,
        "page_norm_w": page_norm_w, "page_norm_h": page_norm_h,
        "score_dimension_line": pattern_scores["score_dimension_line"],
        "score_dimension_text": pattern_scores["score_dimension_text"],
        "score_leader_line": pattern_scores["score_leader_line"],
        "score_note_text": pattern_scores["score_note_text"],
        "score_room_label": pattern_scores["score_room_label"],
    }


def collect_samples(rows: list[dict[str, Any]]) -> tuple[list[dict[str, float]], list[str], list[dict[str, Any]]]:
    """Extract feature vectors and labels from dataset rows."""
    features = []
    labels = []
    items_meta = []

    for row in rows:
        meta = row.get("metadata") or {}
        page_w = float(meta.get("width") or 1.0)
        page_h = float(meta.get("height") or 1.0)

        for item in row.get("text_candidates") or []:
            label = str(item.get("text_type") or "note_text")
            item["_metadata"] = {"width": page_w, "height": page_h}
            feats = extract_features(item)
            features.append(feats)
            labels.append(label)
            items_meta.append(item)

    return features, labels, items_meta


def feature_vector(feats: dict[str, float]) -> list[float]:
    return [feats[name] for name in FEATURE_NAMES]


# ---------------------------------------------------------------------------
# Relation prediction (dimension_text -> dimension_line linker)
# ---------------------------------------------------------------------------

def predict_dimension_links(
    pred_items: list[dict[str, Any]],
    source_items: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Predict relation targets: dimension_text -> dimension_line via nearest neighbor."""
    dimension_lines = [
        item for item in pred_items
        if item.get("prediction") == "dimension_line" and normalize_bbox(item.get("bbox"))
    ]
    links = []

    for item in pred_items:
        if item.get("prediction") != "dimension_text":
            continue
        bbox = normalize_bbox(item.get("bbox"))
        if bbox is None or not dimension_lines:
            continue

        def bbox_distance_one(a: list[float], b: list[float]) -> float:
            dx = max(a[0] - b[2], b[0] - a[2], 0.0)
            dy = max(a[1] - b[3], b[1] - a[3], 0.0)
            return (dx * dx + dy * dy) ** 0.5

        nearest = min(
            dimension_lines,
            key=lambda c: bbox_distance_one(bbox, normalize_bbox(c.get("bbox")) or bbox),
        )
        links.append({
            "source": str(item.get("id")),
            "target": str(nearest.get("id")),
            "relation": "dimension_of",
            "evidence": "nearest_v4",
        })

    return links


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def classification_report(labels: list[str], confusion: dict[str, Counter[str]]) -> tuple[dict[str, Any], float]:
    per_label = {}
    f1_values = []
    for label in labels:
        tp = confusion.get(label, Counter()).get(label, 0)
        fp = sum(confusion.get(other, Counter()).get(label, 0) for other in labels if other != label)
        fn = sum(count for pred, count in confusion.get(label, Counter()).items() if pred != label)
        precision = tp / max(tp + fp, 1)
        recall = tp / max(tp + fn, 1)
        f1 = 2 * precision * recall / max(precision + recall, 1e-12)
        f1_values.append(f1)
        per_label[label] = {"precision": precision, "recall": recall, "f1": f1, "support": sum(confusion[label].values())}
    return per_label, sum(f1_values) / max(len(f1_values), 1)


def link_report(rows: list[dict[str, Any]], gold_key: str, pred_key: str) -> dict[str, float | int]:
    def link_key(item: dict[str, Any]) -> tuple[str, str, str] | None:
        source = item.get("source")
        target = item.get("target")
        relation = item.get("relation")
        if source is None or target is None or relation is None:
            return None
        return str(source), str(target), str(relation)

    gold_total = pred_total = matched = 0
    for row in rows:
        gold = {link_key(item) for item in row.get(gold_key) or []}
        pred = {link_key(item) for item in row.get(pred_key) or []}
        gold.discard(None)
        pred.discard(None)
        gold_total += len(gold)
        pred_total += len(pred)
        matched += len(gold & pred)
    precision = matched / max(pred_total, 1)
    recall = matched / max(gold_total, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {"gold": gold_total, "predicted": pred_total, "matched": matched, "precision": precision, "recall": recall, "f1": f1}


def evaluate_predictions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = sorted({str(item["gold"]) for row in rows for item in row.get("text_candidates") or []})
    confusion = {label: Counter() for label in labels}
    total = 0
    correct = 0
    for row in rows:
        for item in row.get("text_candidates") or []:
            gold = str(item.get("gold"))
            pred = str(item.get("prediction"))
            confusion.setdefault(gold, Counter())[pred] += 1
            total += 1
            correct += int(gold == pred)

    per_label, macro_f1 = classification_report(labels, confusion)
    link_metrics = link_report(rows, "dimension_links_gold", "dimension_links_pred")

    return {
        "text_candidates": total,
        "accuracy": correct / max(total, 1),
        "macro_f1": macro_f1,
        "dimension_link": link_metrics,
        "per_label": per_label,
        "confusion": {label: dict(counts) for label, counts in confusion.items()},
    }


def memory_audit(stage: str) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"stage": stage, "max_rss_kb": int(usage.ru_maxrss)}


def split_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    item_counts = [len(row.get("text_candidates") or []) for row in rows]
    link_counts = [len(row.get("dimension_links") or []) for row in rows]
    return {
        "rows": len(rows),
        "text_candidates": sum(item_counts),
        "dimension_links": sum(link_counts),
    }


# ---------------------------------------------------------------------------
# Main training pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-dir", default="datasets/text_dimension_expert_v4_full_ocr")
    parser.add_argument("--output-dir", default="checkpoints/text_dimension_expert_v4")
    parser.add_argument("--n-estimators", type=int, default=400)
    parser.add_argument("--max-depth", type=int, default=20)
    parser.add_argument("--min-samples-leaf", type=int, default=3)
    parser.add_argument("--seed", type=int, default=20260501)
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("TextDimensionExpert v4: sklearn ExtraTrees with full OCR features")
    print("=" * 70)

    # 1. Load and extract features
    print("\n1. Loading and extracting features...")
    train_rows = load_jsonl(input_dir / "train.jsonl")
    train_features, train_labels, train_items_meta = collect_samples(train_rows)
    X_train = [feature_vector(f) for f in train_features]
    y_train_str = train_labels

    print(f"   Train: {len(X_train)} samples, {len(set(y_train_str))} classes")
    print(f"   Class distribution:")
    for label, count in sorted(Counter(y_train_str).items()):
        print(f"     {label}: {count}")

    # 2. Train ExtraTrees classifier
    print(f"\n2. Training ExtraTrees ({args.n_estimators} trees)...")
    encoder = LabelEncoder()
    y_train = encoder.fit_transform(y_train_str)

    model = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    # Feature importance audit
    importances = model.feature_importances_
    print(f"\n   Top 10 features:")
    for idx in sorted(range(len(FEATURE_NAMES)), key=lambda i: importances[i], reverse=True)[:10]:
        print(f"     {FEATURE_NAMES[idx]:<25} {importances[idx]:.4f}")

    # 3. Evaluate on all splits
    print(f"\n3. Evaluating on all splits...")
    summary: dict[str, Any] = {
        "input_dir": str(input_dir),
        "model": str(output_dir / "model_v4.json"),
        "model_type": "text_dimension_extra_trees_v4",
        "feature_names": FEATURE_NAMES,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "class_names": list(encoder.classes_),
        "feature_importances": {FEATURE_NAMES[i]: float(importances[i]) for i in range(len(FEATURE_NAMES))},
        "splits": {},
    }

    for split_name in ("train", "dev", "smoke", "locked_test"):
        split_rows = load_jsonl(input_dir / f"{split_name}.jsonl")
        if not split_rows:
            continue

        split_features, split_labels, split_items_meta = collect_samples(split_rows)
        X_split = [feature_vector(f) for f in split_features]
        y_pred_idx = model.predict(X_split)
        y_pred_str = encoder.inverse_transform(y_pred_idx)

        # Build prediction rows
        pred_rows = []
        item_idx = 0
        for row in split_rows:
            n_items = len(row.get("text_candidates") or [])
            row_items = []
            for i in range(n_items):
                orig_item = row.get("text_candidates", [])[i]
                row_items.append({
                    "id": orig_item.get("id"),
                    "gold": orig_item.get("text_type"),
                    "prediction": y_pred_str[item_idx],
                    "bbox": orig_item.get("bbox"),
                })
                item_idx += 1

            links = predict_dimension_links(row_items, row.get("text_candidates") or [])
            pred_rows.append({
                "image": row.get("image"),
                "annotation": row.get("annotation"),
                "source_dataset": row.get("source_dataset"),
                "text_candidates": row_items,
                "dimension_links_gold": row.get("dimension_links") or [],
                "dimension_links_pred": links,
            })

        eval_result = evaluate_predictions(pred_rows)
        summary["splits"][split_name] = eval_result

        print(f"   {split_name}: acc={eval_result['accuracy']:.4f}, macro_f1={eval_result['macro_f1']:.4f}, "
              f"link_f1={eval_result['dimension_link']['f1']:.4f}")

        # Write predictions
        write_jsonl(output_dir / f"{split_name}_predictions.jsonl", pred_rows)

    # 4. Save model
    model_data = {
        "model_type": "text_dimension_extra_trees_v4",
        "feature_names": FEATURE_NAMES,
        "class_names": list(encoder.classes_),
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "tree_weights": model.estimators_ if False else None,  # too large to serialize, use joblib instead
        "notes": "v4: sklearn ExtraTrees with full OCR features from EasyOCR v4 dataset.",
    }

    import joblib
    joblib.dump({
        "classifier": model,
        "encoder": encoder,
        "feature_names": FEATURE_NAMES,
    }, output_dir / "model_v4.joblib")
    model_data["joblib_model"] = "model_v4.joblib"

    write_json(output_dir / "model_v4.json", model_data)
    write_json(output_dir / "train_summary.json", summary)
    write_json(REPORTS_DIR / "text_dimension_expert_v4_eval.json", summary)

    print(f"\n{'=' * 70}")
    print(f"TextDimensionExpert v4 complete.")
    print(f"  Model: {output_dir / 'model_v4.joblib'}")
    print(f"  Summary: {output_dir / 'train_summary.json'}")
    print(f"  Report: {REPORTS_DIR / 'text_dimension_expert_v4_eval.json'}")

    # 5. Print per-class F1 for dev
    if "dev" in summary["splits"]:
        dev = summary["splits"]["dev"]
        print(f"\n  Dev per-class F1:")
        for label, metrics in sorted(dev.get("per_label", {}).items()):
            print(f"    {label:<20} F1={metrics['f1']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}, N={metrics['support']}")

    print(f"{'=' * 70}")


if __name__ == "__main__":
    main()
