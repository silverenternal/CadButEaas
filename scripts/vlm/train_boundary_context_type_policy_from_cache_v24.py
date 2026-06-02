#!/usr/bin/env python3
"""Train fail-closed boundary type repair policy from v24 feature cache."""

from __future__ import annotations

import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.metrics import classification_report

from apply_boundary_proposals_with_graph_node_gnn_v24 import BOUNDARY_TO_GRAPH_LABEL, LABELS, load_jsonl, write_json
from build_boundary_context_feature_cache_v24 import OFFLINE_ONLY_COLUMNS, RUNTIME_FEATURE_COLUMNS


ROOT = Path(__file__).resolve().parents[2]
ALL_LABELS = ["background", *LABELS]
NON_FEATURE_COLUMNS = {
    "row_id",
    "candidate_id",
    "orientation",
    "collinear_chain_id",
    "yolo_hint",
    "duplicate_group_id",
}
FEATURE_NAMES = [name for name in RUNTIME_FEATURE_COLUMNS if name not in NON_FEATURE_COLUMNS]


def resolve(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def read_jsonl_gz(path: Path) -> list[dict[str, Any]]:
    rows = []
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def gold_counts(path: Path, limit: int | None = None) -> dict[str, dict[str, str]]:
    out: dict[str, dict[str, str]] = {}
    for row in load_jsonl(path, limit):
        targets = {}
        for target in (row.get("targets") or {}).get("boxes") or []:
            label = BOUNDARY_TO_GRAPH_LABEL.get(str(target.get("label")), str(target.get("label")))
            if label in LABELS:
                targets[str(target.get("target_id") or "")] = label
        out[str(row.get("id"))] = targets
    return out


def vectorize(rows: list[dict[str, Any]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(row.get(name) or 0.0) for name in FEATURE_NAMES] for row in rows], dtype=np.float32)
    y = np.asarray([str(row.get("gold_label") or "background") for row in rows])
    return x, y


def split_train_tune(rows: list[dict[str, Any]], train_pages: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered_pages = []
    seen = set()
    for row in rows:
        row_id = str(row.get("row_id"))
        if row_id not in seen:
            seen.add(row_id)
            ordered_pages.append(row_id)
    train_ids = set(ordered_pages[:train_pages])
    return [row for row in rows if row.get("row_id") in train_ids], [row for row in rows if row.get("row_id") not in train_ids]


def baseline_label(row: dict[str, Any]) -> str:
    hint = str(row.get("yolo_hint") or "")
    return hint if hint in LABELS else "hard_wall"


def apply_policy_label(row: dict[str, Any], classes: list[str], probs: np.ndarray, thresholds: dict[str, float]) -> str:
    fallback = baseline_label(row)
    if fallback != "hard_wall":
        return fallback
    by_label = {label: float(probs[idx]) for idx, label in enumerate(classes)}
    door_prob = by_label.get("door", 0.0)
    window_prob = by_label.get("window", 0.0)
    if window_prob >= thresholds["window"] and window_prob >= door_prob:
        return "window"
    if door_prob >= thresholds["door"]:
        return "door"
    return fallback


def predict_rows(rows: list[dict[str, Any]], model: Any, thresholds: dict[str, float]) -> list[str]:
    if not rows:
        return []
    x, _ = vectorize(rows)
    classes = [str(label) for label in model.classes_.tolist()]
    probs = model.predict_proba(x)
    return [apply_policy_label(row, classes, prob, thresholds) for row, prob in zip(rows, probs, strict=True)]


def evaluate_feature_rows(
    rows: list[dict[str, Any]],
    targets_by_page: dict[str, dict[str, str]],
    predicted_labels: list[str] | None = None,
) -> dict[str, Any]:
    predicted_labels = predicted_labels or [baseline_label(row) for row in rows]
    by_page_gold_candidate: dict[str, dict[str, list[tuple[str, str]]]] = defaultdict(lambda: defaultdict(list))
    predicted_count = len(rows)
    for row, pred in zip(rows, predicted_labels, strict=True):
        row_id = str(row.get("row_id"))
        gold_id = str(row.get("matched_gold_id") or "")
        gold_label = str(row.get("gold_label") or "background")
        if gold_id:
            by_page_gold_candidate[row_id][gold_id].append((gold_label, pred))

    total = proposal_hit = classified_hit = 0
    per_label: dict[str, Counter[str]] = defaultdict(Counter)
    missed = []
    wrong = []
    for row_id, targets in targets_by_page.items():
        candidates_by_gold = by_page_gold_candidate.get(row_id, {})
        for gold_id, gold_label in targets.items():
            total += 1
            per_label[gold_label]["gold"] += 1
            candidates = candidates_by_gold.get(gold_id, [])
            if candidates:
                proposal_hit += 1
                per_label[gold_label]["proposal_matched"] += 1
            else:
                missed.append({"row_id": row_id, "target_id": gold_id, "label": gold_label})
                continue
            if any(pred == gold_label for _, pred in candidates):
                classified_hit += 1
                per_label[gold_label]["classified_matched"] += 1
            else:
                wrong.append(
                    {
                        "row_id": row_id,
                        "target_id": gold_id,
                        "gold_label": gold_label,
                        "predicted_labels": sorted(Counter(pred for _, pred in candidates).items(), key=lambda x: (-x[1], x[0]))[:5],
                    }
                )
    return {
        "gold": total,
        "predicted": predicted_count,
        "candidate_inflation": round(predicted_count / max(total, 1), 6),
        "proposal_recall": round(proposal_hit / max(total, 1), 6),
        "classified_recall": round(classified_hit / max(total, 1), 6),
        "classified_precision_proxy": round(classified_hit / max(predicted_count, 1), 6),
        "per_label": {
            label: {
                "gold": counts["gold"],
                "proposal_matched": counts["proposal_matched"],
                "classified_matched": counts["classified_matched"],
                "proposal_recall": round(counts["proposal_matched"] / max(counts["gold"], 1), 6),
                "classified_recall": round(counts["classified_matched"] / max(counts["gold"], 1), 6),
            }
            for label, counts in sorted(per_label.items())
        },
        "missed_summary": dict(Counter(item["label"] for item in missed)),
        "missed_examples": missed[:100],
        "wrong_type_examples": wrong[:200],
    }


def threshold_grid() -> list[dict[str, float]]:
    return [{"door": door, "window": window} for door in np.linspace(0.10, 0.90, 9) for window in np.linspace(0.10, 0.90, 9)]


def select_thresholds(
    rows: list[dict[str, Any]],
    targets_by_page: dict[str, dict[str, str]],
    model: Any,
) -> tuple[dict[str, float], list[dict[str, Any]]]:
    baseline = evaluate_feature_rows(rows, targets_by_page)
    sweep = []
    best = {"door": 1.01, "window": 1.01}
    best_score = -1e9
    for thresholds in threshold_grid():
        preds = predict_rows(rows, model, thresholds)
        metrics = evaluate_feature_rows(rows, targets_by_page, preds)
        hard_wall = metrics["per_label"].get("hard_wall", {}).get("classified_recall", 0.0)
        door = metrics["per_label"].get("door", {}).get("classified_recall", 0.0)
        window = metrics["per_label"].get("window", {}).get("classified_recall", 0.0)
        base_hard_wall = baseline["per_label"].get("hard_wall", {}).get("classified_recall", 0.0)
        no_drop = metrics["classified_recall"] >= baseline["classified_recall"] and hard_wall >= base_hard_wall
        score = (
            (1.0 if no_drop else 0.0)
            + metrics["classified_recall"]
            + 0.5 * door
            + 0.5 * window
            + 0.05 * metrics["classified_precision_proxy"]
        )
        item = {
            "thresholds": {key: round(float(value), 4) for key, value in thresholds.items()},
            "classified_recall": metrics["classified_recall"],
            "classified_precision_proxy": metrics["classified_precision_proxy"],
            "hard_wall_recall": hard_wall,
            "door_recall": door,
            "window_recall": window,
            "no_drop_vs_baseline": no_drop,
        }
        sweep.append(item)
        if score > best_score:
            best_score = score
            best = thresholds
    return {key: round(float(value), 4) for key, value in best.items()}, sorted(sweep, key=lambda r: (r["no_drop_vs_baseline"], r["classified_recall"], r["door_recall"], r["window_recall"]), reverse=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev-cache", default="datasets/boundary_context_feature_cache_v24/dev493_features.jsonl.gz")
    parser.add_argument("--locked-cache", default="datasets/boundary_context_feature_cache_v24/locked50_features.jsonl.gz")
    parser.add_argument("--dataset", default="datasets/boundary_expert_public_raster_v19")
    parser.add_argument("--output-dir", default="checkpoints/boundary_context_type_policy_v24")
    parser.add_argument("--eval-output", default="reports/vlm/boundary_context_type_policy_v24_locked50_eval.json")
    parser.add_argument("--sweep-output", default="reports/vlm/boundary_context_type_policy_v24_threshold_sweep.json")
    parser.add_argument("--error-output", default="reports/vlm/boundary_context_type_policy_v24_error_buckets.json")
    parser.add_argument("--train-pages", type=int, default=400)
    parser.add_argument("--max-background", type=int, default=120000)
    parser.add_argument("--model-backend", choices=["hist_gradient", "extra_trees"], default="hist_gradient")
    parser.add_argument("--seed", type=int, default=20260511)
    args = parser.parse_args()

    dev_rows = read_jsonl_gz(resolve(args.dev_cache))
    locked_rows = read_jsonl_gz(resolve(args.locked_cache))
    train_rows, tune_rows = split_train_tune(dev_rows, args.train_pages)
    rng = np.random.default_rng(args.seed)
    positives = [row for row in train_rows if row.get("gold_label") != "background"]
    background = [row for row in train_rows if row.get("gold_label") == "background"]
    if len(background) > args.max_background:
        keep = rng.choice(len(background), size=args.max_background, replace=False)
        background = [background[int(idx)] for idx in keep]
    train_sample = positives + background
    rng.shuffle(train_sample)
    x_train, y_train = vectorize(train_sample)
    weights = np.asarray(
        [{"background": 0.2, "hard_wall": 1.0, "door": 3.5, "window": 3.5}.get(str(label), 1.0) for label in y_train],
        dtype=np.float32,
    )
    if args.model_backend == "extra_trees":
        model = ExtraTreesClassifier(n_estimators=500, min_samples_leaf=2, max_features="sqrt", random_state=args.seed, n_jobs=-1)
    else:
        model = HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=140,
            max_leaf_nodes=31,
            l2_regularization=0.01,
            random_state=args.seed,
        )
    model.fit(x_train, y_train, sample_weight=weights)

    dataset = resolve(args.dataset)
    dev_targets = gold_counts(dataset / "dev.jsonl", None)
    tune_page_ids = {str(row.get("row_id")) for row in tune_rows}
    tune_targets = {row_id: targets for row_id, targets in dev_targets.items() if row_id in tune_page_ids}
    locked_targets = gold_counts(dataset / "locked.jsonl", 50)
    thresholds, sweep = select_thresholds(tune_rows, tune_targets, model)
    locked_baseline = evaluate_feature_rows(locked_rows, locked_targets)
    locked_preds = predict_rows(locked_rows, model, thresholds)
    locked_eval = evaluate_feature_rows(locked_rows, locked_targets, locked_preds)

    output_dir = resolve(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "feature_names": FEATURE_NAMES,
            "labels": ALL_LABELS,
            "thresholds": thresholds,
            "runtime_feature_columns": FEATURE_NAMES,
            "offline_only_columns": OFFLINE_ONLY_COLUMNS,
            "policy_contract": "fail_closed_keep_all_candidates_retype_hard_wall_to_door_or_window_only",
        },
        model_path,
    )
    report = {
        "version": "boundary_context_type_policy_v24_locked50_eval",
        "task": "P0-02-boundary-fail-closed-type-policy",
        "model": str(model_path.relative_to(ROOT)),
        "claim_boundary": "Model is trained with offline labels from cache, but inference uses only runtime feature columns. Application is fail-closed: keep all candidates and only retype hard_wall hints to door/window above thresholds.",
        "feature_names": FEATURE_NAMES,
        "thresholds": thresholds,
        "train_counts": dict(Counter(y_train.tolist())),
        "classification_report_train_sample": classification_report(y_train, model.predict(x_train), labels=ALL_LABELS, output_dict=True, zero_division=0),
        "locked_baseline_yolo_hint": locked_baseline,
        "locked_eval": locked_eval,
        "success_gate": {
            "locked_classified_recall_min_stage_1": 0.95,
            "locked_door_recall_min_stage_1": 0.9,
            "locked_window_recall_min_stage_1": 0.9,
            "must_beat_fusion_hint_classified_recall": 0.922937,
            "baseline_classified_recall": locked_baseline["classified_recall"],
            "baseline_precision_proxy": locked_baseline["classified_precision_proxy"],
            "locked_classified_recall": locked_eval["classified_recall"],
            "locked_precision_proxy": locked_eval["classified_precision_proxy"],
            "door_recall": locked_eval["per_label"].get("door", {}).get("classified_recall", 0.0),
            "window_recall": locked_eval["per_label"].get("window", {}).get("classified_recall", 0.0),
        },
    }
    gate = report["success_gate"]
    gate["passed"] = (
        gate["locked_classified_recall"] >= gate["locked_classified_recall_min_stage_1"]
        and gate["door_recall"] >= gate["locked_door_recall_min_stage_1"]
        and gate["window_recall"] >= gate["locked_window_recall_min_stage_1"]
        and gate["locked_classified_recall"] >= gate["baseline_classified_recall"]
        and gate["locked_classified_recall"] > gate["must_beat_fusion_hint_classified_recall"]
    )
    write_json(resolve(args.eval_output), report)
    write_json(resolve(args.sweep_output), {"version": "boundary_context_type_policy_v24_threshold_sweep", "selected_thresholds": thresholds, "top_candidates": sweep[:50], "all_candidates": sweep})
    write_json(resolve(args.error_output), {"version": "boundary_context_type_policy_v24_error_buckets", "wrong_type_examples": locked_eval["wrong_type_examples"], "missed_examples": locked_eval["missed_examples"], "missed_summary": locked_eval["missed_summary"]})
    print(json.dumps({"thresholds": thresholds, "success_gate": gate}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
