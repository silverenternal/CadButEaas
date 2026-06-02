#!/usr/bin/env python3
"""Train and evaluate a lightweight runtime-feature symbol selector."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, precision_recall_fscore_support, roc_auc_score

from analyze_symbol_subset_failure_p117 import bbox_iou, center_covered, reconstruct_page_golds


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = ROOT / "reports/vlm/full_public_raster_symbol_eval_subset_p119_expanded_grid.json"
DEFAULT_TRAIN = ROOT / "datasets/symbol_selector_subset_p122/train.jsonl"
DEFAULT_DEV = ROOT / "datasets/symbol_selector_subset_p122/dev.jsonl"
DEFAULT_MODEL = ROOT / "checkpoints/symbol_selector_p124/model.joblib"
DEFAULT_JSON = ROOT / "configs/vlm/symbol_selector_p124.json"
DEFAULT_REPORT = ROOT / "reports/vlm/symbol_selector_p124.md"
DEFAULT_PREDICTIONS = ROOT / "reports/vlm/symbol_selector_p124_dev_predictions.jsonl"

CENTER_GATE = 0.851394
TINY_IOU_GATE = 0.393013
INFLATION_GATE = 7.919152


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray(
        [[float((row.get("runtime_features") or {}).get(name, 0.0)) for name in feature_names] for row in rows],
        dtype=np.float32,
    )
    y = np.asarray([1 if (row.get("offline_label") or {}).get("keep_any") else 0 for row in rows], dtype=np.int64)
    return x, y


def area_bucket(box: list[float]) -> str:
    area = max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])
    if area <= 64:
        return "tiny_le_64"
    if area <= 256:
        return "small_le_256"
    if area <= 1024:
        return "medium_le_1024"
    if area <= 4096:
        return "large_le_4096"
    return "xlarge_gt_4096"


def with_probabilities(rows: list[dict[str, Any]], feature_names: list[str], model: Any) -> list[dict[str, Any]]:
    x, _y = matrix(rows, feature_names)
    probs = model.predict_proba(x)[:, 1]
    enriched: list[dict[str, Any]] = []
    for row, prob in zip(rows, probs):
        item = dict(row)
        item["selector_score"] = float(prob)
        enriched.append(item)
    return enriched


def group_rows(rows: list[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["row_id"])].append(row)
    for page_rows in grouped.values():
        page_rows.sort(key=lambda item: float(item.get("selector_score") or 0.0), reverse=True)
    return dict(grouped)


def select_rows(page_rows: list[dict[str, Any]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    label_kept = Counter()
    for row in page_rows:
        if float(row.get("selector_score") or 0.0) < float(policy["threshold"]):
            continue
        label = str(row["predicted"].get("label") or "")
        if len(selected) >= int(policy["max_page_keep"]):
            continue
        if label_kept[label] >= int(policy["max_label_keep"]):
            continue
        selected.append(row)
        label_kept[label] += 1
    return selected


def score_policy(page_rows: dict[str, list[dict[str, Any]]], page_golds: dict[str, dict[str, dict[str, Any]]], policy: dict[str, Any]) -> dict[str, Any]:
    totals = Counter()
    by_area = Counter()
    by_area_iou = Counter()
    typed_correct = 0
    selected_offline = Counter()
    selected_labels = Counter()
    predicted_rows: list[dict[str, Any]] = []

    for row_id, gold_map in page_golds.items():
        selected = select_rows(page_rows.get(row_id, []), policy)
        used_iou: set[int] = set()
        used_center: set[int] = set()
        predicted_rows.append({
            "row_id": row_id,
            "predicted_symbols": [
                {
                    **row["predicted"],
                    "selector_score": round(float(row.get("selector_score") or 0.0), 8),
                }
                for row in selected
            ],
        })
        for row in selected:
            selected_labels[str(row["predicted"].get("label") or "")] += 1
            selected_offline["keep_any" if (row.get("offline_label") or {}).get("keep_any") else "negative"] += 1
        for gold in gold_map.values():
            gold_box = [float(v) for v in gold["bbox"]]
            bucket = area_bucket(gold_box)
            by_area[bucket] += 1
            best_iou = 0.0
            best_iou_index = None
            center_index = None
            for idx, row in enumerate(selected):
                pred_box = [float(v) for v in row["predicted"]["bbox"]]
                iou = bbox_iou(pred_box, gold_box)
                if iou > best_iou:
                    best_iou = iou
                    best_iou_index = idx
                if center_index is None and idx not in used_center and center_covered(pred_box, gold_box):
                    center_index = idx
            if best_iou_index is not None and best_iou >= 0.30 and best_iou_index not in used_iou:
                used_iou.add(best_iou_index)
                totals["matched_iou"] += 1
                by_area_iou[bucket] += 1
                if str(selected[best_iou_index]["predicted"].get("label") or "") == str(gold.get("label") or ""):
                    typed_correct += 1
            if center_index is not None:
                used_center.add(center_index)
                totals["matched_center"] += 1
        totals["gold"] += len(gold_map)
        totals["predicted"] += len(selected)

    precision = totals["matched_iou"] / max(totals["predicted"], 1)
    recall = totals["matched_iou"] / max(totals["gold"], 1)
    center = totals["matched_center"] / max(totals["gold"], 1)
    tiny_iou = by_area_iou["tiny_le_64"] / max(by_area["tiny_le_64"], 1)
    inflation = totals["predicted"] / max(totals["gold"], 1)
    return {
        "policy": policy,
        "matched": int(totals["matched_iou"]),
        "predicted": int(totals["predicted"]),
        "gold": int(totals["gold"]),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall), 6),
        "center_recall": round(center, 6),
        "tiny_iou_recall": round(tiny_iou, 6),
        "candidate_inflation": round(inflation, 6),
        "typed_accuracy_on_iou_matches": round(typed_correct / max(totals["matched_iou"], 1), 6),
        "area_iou_recall": {bucket: round(by_area_iou[bucket] / max(by_area[bucket], 1), 6) for bucket in sorted(by_area)},
        "selected_offline_label_counts": dict(selected_offline),
        "selected_label_counts": dict(selected_labels.most_common()),
        "passes_center": center > CENTER_GATE,
        "passes_tiny_iou": tiny_iou > TINY_IOU_GATE,
        "passes_inflation": inflation <= INFLATION_GATE,
        "passes_all_gates": center > CENTER_GATE and tiny_iou > TINY_IOU_GATE and inflation <= INFLATION_GATE,
        "_prediction_rows": predicted_rows,
    }


def policy_grid() -> list[dict[str, Any]]:
    policies: list[dict[str, Any]] = []
    for threshold in [0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.24, 0.32, 0.45, 0.60]:
        for max_page_keep in [80, 120, 160, 220]:
            for max_label_keep in [20, 30, 45, 60]:
                policies.append({
                    "name": f"thr{threshold}_p{max_page_keep}_l{max_label_keep}",
                    "threshold": threshold,
                    "max_page_keep": max_page_keep,
                    "max_label_keep": max_label_keep,
                })
    return policies


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default=str(DEFAULT_EVAL))
    parser.add_argument("--train", default=str(DEFAULT_TRAIN))
    parser.add_argument("--dev", default=str(DEFAULT_DEV))
    parser.add_argument("--model-output", default=str(DEFAULT_MODEL))
    parser.add_argument("--output", default=str(DEFAULT_JSON))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    parser.add_argument("--predictions-output", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--seed", type=int, default=20260516)
    args = parser.parse_args()

    train_rows = load_jsonl(Path(args.train))
    dev_rows = load_jsonl(Path(args.dev))
    feature_names = sorted((train_rows[0].get("runtime_features") or {}).keys())
    train_x, train_y = matrix(train_rows, feature_names)
    dev_x, dev_y = matrix(dev_rows, feature_names)

    pos = max(int(train_y.sum()), 1)
    neg = max(int((1 - train_y).sum()), 1)
    sample_weight = np.where(train_y == 1, neg / pos, 1.0)
    model = HistGradientBoostingClassifier(
        max_iter=180,
        learning_rate=0.05,
        max_leaf_nodes=31,
        l2_regularization=0.01,
        random_state=args.seed,
    )
    model.fit(train_x, train_y, sample_weight=sample_weight)
    dev_probs = model.predict_proba(dev_x)[:, 1]
    threshold_rows = []
    for threshold in [0.02, 0.04, 0.06, 0.08, 0.10, 0.14, 0.18, 0.24, 0.32, 0.45, 0.60]:
        pred = (dev_probs >= threshold).astype(np.int64)
        precision, recall, f1, _support = precision_recall_fscore_support(dev_y, pred, average="binary", zero_division=0)
        threshold_rows.append({
            "threshold": threshold,
            "precision": round(float(precision), 6),
            "recall": round(float(recall), 6),
            "f1": round(float(f1), 6),
            "kept_rate": round(float(pred.mean()), 6),
        })

    eval_data = json.loads(Path(args.eval).read_text(encoding="utf-8"))
    all_page_golds = reconstruct_page_golds(eval_data)
    dev_page_ids = {str(row["row_id"]) for row in dev_rows}
    dev_page_golds = {row_id: golds for row_id, golds in all_page_golds.items() if row_id in dev_page_ids}
    dev_with_probs = with_probabilities(dev_rows, feature_names, model)
    grouped_dev = group_rows(dev_with_probs)
    results = [score_policy(grouped_dev, dev_page_golds, policy) for policy in policy_grid()]
    results.sort(
        key=lambda row: (
            row["passes_all_gates"],
            row["center_recall"] > CENTER_GATE,
            row["tiny_iou_recall"] > TINY_IOU_GATE,
            row["candidate_inflation"] <= INFLATION_GATE,
            row["f1"],
            row["center_recall"],
            row["tiny_iou_recall"],
            -row["candidate_inflation"],
        ),
        reverse=True,
    )
    best = results[0]
    prediction_rows = best.pop("_prediction_rows")
    for row in results:
        row.pop("_prediction_rows", None)

    Path(args.model_output).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "model_type": "symbol_selector_p124_hist_gradient_boosting",
            "feature_names": feature_names,
            "model": model,
            "selected_policy": best["policy"],
            "training_data": rel(Path(args.train)),
            "dev_data": rel(Path(args.dev)),
        },
        args.model_output,
    )
    write_jsonl(Path(args.predictions_output), prediction_rows)

    report = {
        "id": "SCI-P2-124-symbol-lightweight-selector-train-dev",
        "claim_boundary": "P122 train/dev selector probe only; no locked/full runtime adoption claim.",
        "source_integrity": "Model inputs are P122 runtime_features only; offline labels are used for supervised training/dev evaluation only.",
        "inputs": {"eval": rel(Path(args.eval)), "train": rel(Path(args.train)), "dev": rel(Path(args.dev))},
        "artifacts": {
            "model": rel(Path(args.model_output)),
            "dev_predictions": rel(Path(args.predictions_output)),
        },
        "counts": {
            "train_rows": len(train_rows),
            "dev_rows": len(dev_rows),
            "train_pages": len({row["row_id"] for row in train_rows}),
            "dev_pages": len(dev_page_ids),
            "train_positives": int(train_y.sum()),
            "dev_positives": int(dev_y.sum()),
        },
        "candidate_level_validation": {
            "roc_auc": round(float(roc_auc_score(dev_y, dev_probs)), 6) if len(set(dev_y.tolist())) > 1 else None,
            "average_precision": round(float(average_precision_score(dev_y, dev_probs)), 6),
            "threshold_grid": threshold_rows,
        },
        "baseline_gates": {
            "center_recall_gt": CENTER_GATE,
            "tiny_iou_recall_gt": TINY_IOU_GATE,
            "candidate_inflation_lte": INFLATION_GATE,
        },
        "best_row": best,
        "top_rows": results[:20],
        "decision": "selector_passes_dev_subset_gates" if best["passes_all_gates"] else "selector_does_not_pass_dev_subset_gates",
        "recommendation": (
            "Export/apply this selected policy to the full P119 subset before any runtime adoption claim."
            if best["passes_all_gates"]
            else "Selector compressed candidates but missed at least one gate; inspect failure buckets before training a richer listwise model."
        ),
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        "# P2-124 Symbol Lightweight Selector Train/Dev",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Train/dev rows: `{len(train_rows)}` / `{len(dev_rows)}`",
        f"- Candidate AP/AUC: `{report['candidate_level_validation']['average_precision']}` / `{report['candidate_level_validation']['roc_auc']}`",
        f"- Best policy: `{best['policy']['name']}`",
        f"- Best center/tiny/inflation: `{best['center_recall']}` / `{best['tiny_iou_recall']}` / `{best['candidate_inflation']}`",
        f"- Best precision/recall/F1: `{best['precision']}` / `{best['recall']}` / `{best['f1']}`",
        f"- Model: `{rel(Path(args.model_output))}`",
        f"- Dev predictions: `{rel(Path(args.predictions_output))}`",
        "",
        "## Top Policies",
        "",
        "| Policy | Center | Tiny IoU | Inflation | F1 | Precision | Recall | Pass |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in results[:10]:
        lines.append(
            f"| `{row['policy']['name']}` | {row['center_recall']:.6f} | {row['tiny_iou_recall']:.6f} | "
            f"{row['candidate_inflation']:.6f} | {row['f1']:.6f} | {row['precision']:.6f} | {row['recall']:.6f} | "
            f"{str(row['passes_all_gates']).lower()} |"
        )
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output": rel(Path(args.output)), "report": rel(Path(args.report)), "decision": report["decision"], "best": best}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
