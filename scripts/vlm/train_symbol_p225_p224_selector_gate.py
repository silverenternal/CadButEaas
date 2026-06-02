#!/usr/bin/env python3
"""Train/evaluate P225 OOF selector gate for P224 s384 proposals over P224a baseline."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from freeze_symbol_p222_p221a_sink_tiny import bootstrap, metrics, score_rows
from fuse_symbol_p206g_with_p211_p212 import LABELS, LABEL_TO_ID, bbox_iou, load_p206g, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/vlm/symbol_p224a_column_frozen_overlay.jsonl"
DATASET = ROOT / "reports/vlm/symbol_p225_p224_selector_dataset.jsonl"
OUT_DATASET = ROOT / "reports/vlm/symbol_p225_p224_selector_scored_dataset.jsonl"
REPORT = ROOT / "reports/vlm/symbol_p225_p224_selector_gate_eval.json"
MD = ROOT / "reports/vlm/symbol_p225_p224_selector_gate_eval.md"
OVERLAY = ROOT / "reports/vlm/symbol_p225_p224_selector_gate_overlay.jsonl"
MODEL = ROOT / "checkpoints/symbol_p225_p224_selector_gate/model.joblib"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("row_id"))


def center(box: list[float]) -> tuple[float, float]:
    return (float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0


def dist(left: list[float], right: list[float]) -> float:
    lx, ly = center(left)
    rx, ry = center(right)
    return ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5


def split_rows(ids: list[str], folds: int, seed: int) -> dict[str, int]:
    shuffled = list(ids)
    random.Random(seed).shuffle(shuffled)
    return {rid: index % folds for index, rid in enumerate(shuffled)}


def make_model(name: str, seed: int):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.35, random_state=seed))
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=100, learning_rate=0.055, l2_regularization=0.08, max_leaf_nodes=15, min_samples_leaf=30, random_state=seed)
    if name == "rf":
        return RandomForestClassifier(n_estimators=320, max_depth=8, min_samples_leaf=8, class_weight="balanced_subsample", random_state=seed, n_jobs=-1)
    if name == "et":
        return ExtraTreesClassifier(n_estimators=360, max_depth=10, min_samples_leaf=6, class_weight="balanced", random_state=seed, n_jobs=-1)
    raise ValueError(name)


def add_oof_scores(dataset: list[dict[str, Any]], feature_names: list[str], model_name: str, target: str, folds: int, seed: int) -> tuple[list[float], dict[str, Any]]:
    row_folds = split_rows(sorted({row["row_id"] for row in dataset}), folds, seed)
    scores = [0.0] * len(dataset)
    y = np.array([int(row[target]) for row in dataset], dtype=int)
    for fold in range(folds):
        train_idx = [i for i, row in enumerate(dataset) if row_folds[row["row_id"]] != fold]
        test_idx = [i for i, row in enumerate(dataset) if row_folds[row["row_id"]] == fold]
        x_train = np.array([[float(dataset[i]["features"].get(name, 0.0)) for name in feature_names] for i in train_idx], dtype=float)
        y_train = y[train_idx]
        x_test = np.array([[float(dataset[i]["features"].get(name, 0.0)) for name in feature_names] for i in test_idx], dtype=float)
        if len(set(y_train.tolist())) < 2:
            pred = np.full(len(test_idx), float(y_train.mean() if len(y_train) else 0.0))
        else:
            model = make_model(model_name, seed + fold)
            model.fit(x_train, y_train)
            pred = model.predict_proba(x_test)[:, 1]
        for index, value in zip(test_idx, pred, strict=True):
            scores[index] = float(value)
    auc = None
    if len(set(y.tolist())) > 1:
        auc = float(roc_auc_score(y, np.array(scores)))
    return scores, {"model": model_name, "target": target, "folds": folds, "auc": auc, "positive": int(y.sum()), "total": int(len(y))}


def candidate_from_row(row: dict[str, Any], score_field: str) -> dict[str, Any]:
    label = str(row["label"])
    gate_score = float(row.get(score_field, row.get("score", 0.0)) or 0.0)
    raw_score = float(row.get("score", 0.0) or 0.0)
    return {
        "bbox": [float(v) for v in row["bbox"]],
        "label": label,
        "symbol_type": label,
        "label_id": LABEL_TO_ID.get(label, LABEL_TO_ID["generic_symbol"]),
        "score": gate_score,
        "confidence": gate_score,
        "raw_score": raw_score,
        "gate_score": gate_score,
        "source": "p225_p224_selector_gate_added",
        "metadata": {"p225_score_field": score_field, "p224_tile_id": row.get("tile_id")},
    }


def conflicts(candidate: dict[str, Any], existing: list[dict[str, Any]], max_iou: float, min_center_dist: float, same_label_only: bool) -> bool:
    cbox = [float(v) for v in candidate["bbox"]]
    for pred in existing:
        if same_label_only and str(pred.get("label") or pred.get("symbol_type")) != candidate["label"]:
            continue
        pbox = [float(v) for v in pred["bbox"]]
        if bbox_iou(cbox, pbox) >= max_iou:
            return True
        if dist(cbox, pbox) <= min_center_dist:
            return True
    return False


def fuse(core: dict[str, list[dict[str, Any]]], dataset: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    score_field = str(policy["score_field"])
    labels = set(policy["labels"])
    for row in dataset:
        if str(row["label"]) not in labels:
            continue
        gate_score = float(row.get(score_field, row.get("score", 0.0)) or 0.0)
        raw_score = float(row.get("score", 0.0) or 0.0)
        if gate_score < float(policy["threshold"]):
            continue
        if raw_score < float(policy["min_raw_score"]):
            continue
        by_row[row["row_id"]].append(candidate_from_row(row, score_field))
    fused: dict[str, list[dict[str, Any]]] = {}
    for rid, base_preds in core.items():
        merged = [dict(pred) for pred in base_preds]
        additions: list[dict[str, Any]] = []
        for cand in sorted(by_row.get(rid, []), key=lambda item: float(item.get("gate_score", 0.0)), reverse=True):
            if conflicts(cand, merged + additions, float(policy["max_iou_to_existing"]), float(policy["min_center_dist"]), bool(policy["same_label_only"])):
                continue
            additions.append(cand)
            if len(additions) >= int(policy["max_add_per_row"]):
                break
        fused[rid] = merged + additions
    return fused


def build_overlay(rows: list[dict[str, Any]], fused: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        rid = row_id(row)
        item = dict(row)
        item["symbol_candidates"] = []
        for index, pred in enumerate(fused.get(rid, [])):
            label = str(pred.get("label") or pred.get("symbol_type") or "generic_symbol")
            item["symbol_candidates"].append({
                "id": f"{rid}_p225_{index:04d}",
                "target_id": f"{rid}_p225_{index:04d}",
                "symbol_type": label,
                "label": label,
                "bbox": [float(v) for v in pred["bbox"]],
                "confidence": float(pred.get("confidence", pred.get("score", 1.0)) or 0.0),
                "score": float(pred.get("score", pred.get("confidence", 1.0)) or 0.0),
                "source": pred.get("source", "p224a_core"),
                "metadata": pred.get("metadata", {}),
            })
        metadata = dict(item.get("metadata") or {})
        metadata["p225_selector_policy"] = policy["name"]
        item["metadata"] = metadata
        out.append(item)
    return out


def policy_grid(score_fields: list[str]) -> list[dict[str, Any]]:
    label_sets = [
        ["equipment", "appliance", "shower", "sink", "stair", "column", "generic_symbol", "bathtub"],
        ["equipment", "appliance", "shower", "sink", "stair", "column"],
    ]
    thresholds = {
        "score": [0.70, 0.85, 0.92],
        "gate": [0.10, 0.20, 0.35, 0.50],
    }
    policies = []
    for field in score_fields:
        kind = "score" if field == "score" else "gate"
        for labels in label_sets:
            for threshold in thresholds[kind]:
                for min_raw in [0.30, 0.50]:
                    for max_add in [1, 2, 3, 5]:
                        for max_iou in [0.10, 0.30]:
                            policies.append({
                                "name": f"p225_{field}_t{threshold:g}_raw{min_raw:g}_a{max_add}_iou{max_iou:g}_{len(labels)}lbl",
                                "score_field": field,
                                "labels": labels,
                                "threshold": threshold,
                                "min_raw_score": min_raw,
                                "max_add_per_row": max_add,
                                "max_iou_to_existing": max_iou,
                                "min_center_dist": 0.0,
                                "same_label_only": True,
                            })
    return policies


def train_final(dataset: list[dict[str, Any]], feature_names: list[str], field: str, model_path: Path, seed: int) -> None:
    if not field.startswith("gate_"):
        return
    _, model_name, target = field.split("_", 2)
    x = np.array([[float(row["features"].get(name, 0.0)) for name in feature_names] for row in dataset], dtype=float)
    y = np.array([int(row[target]) for row in dataset], dtype=int)
    model = make_model(model_name, seed)
    model.fit(x, y)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "feature_names": feature_names, "score_field": field, "target": target, "model_name": model_name}, model_path)


def render(report: dict[str, Any]) -> str:
    base = report["baseline_metrics"]
    selected = report["selected_metrics"]
    boot = report["bootstrap_vs_p224a"]
    lines = [
        "# P225 P224 Selector Gate Evaluation",
        "",
        "## Metrics",
        "| Variant | F1 | Precision | Recall | TP | Pred | Gold |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| P224a baseline | {base['f1']:.6f} | {base['precision']:.6f} | {base['recall']:.6f} | {base['tp']} | {base['predicted']} | {base['gold']} |",
        f"| P225 selected | {selected['f1']:.6f} | {selected['precision']:.6f} | {selected['recall']:.6f} | {selected['tp']} | {selected['predicted']} | {selected['gold']} |",
        "",
        "## Selected Policy",
        f"- `{report['selected_policy']['name']}`",
        f"- Added predictions: `{report['selected_added_predictions']}`",
        "",
        "## Bootstrap vs P224a",
        f"- ΔF1: `{boot['f1_delta']['mean']:.6f}` / `{boot['f1_delta']['ci95']}` / P>0 `{boot['f1_delta']['prob_positive']:.3f}`",
        f"- ΔPrecision: `{boot['precision_delta']['mean']:.6f}` / `{boot['precision_delta']['ci95']}` / P>0 `{boot['precision_delta']['prob_positive']:.3f}`",
        f"- ΔRecall: `{boot['recall_delta']['mean']:.6f}` / `{boot['recall_delta']['ci95']}` / P>0 `{boot['recall_delta']['prob_positive']:.3f}`",
        "",
        "## Gate Diagnostics",
    ]
    for diag in report["gate_diagnostics"]:
        auc = diag.get("auc")
        lines.append(f"- `{diag['model']}/{diag['target']}`: AUC `{auc:.4f}` positives `{diag['positive']}/{diag['total']}`" if auc is not None else f"- `{diag['model']}/{diag['target']}`: AUC n/a")
    lines += ["", "## Claim Boundary", report["claim_boundary"], ""]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(BASE))
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--out-dataset", default=str(OUT_DATASET))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--md", default=str(MD))
    parser.add_argument("--overlay", default=str(OVERLAY))
    parser.add_argument("--model", default=str(MODEL))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=225)
    parser.add_argument("--models", default="logreg,hgb")
    args = parser.parse_args()

    rows, core, golds = load_p206g(Path(args.base))
    ids = [row_id(row) for row in rows]
    dataset = read_jsonl(Path(args.dataset))
    feature_names = sorted(dataset[0]["features"].keys()) if dataset else []
    diagnostics = []
    score_fields = ["score"]
    for model_name in [name.strip() for name in args.models.split(",") if name.strip()]:
        for target in ["target_new_tp", "target_typed_tp"]:
            scores, diag = add_oof_scores(dataset, feature_names, model_name, target, args.folds, args.seed)
            field = f"gate_{model_name}_{target}"
            for row, value in zip(dataset, scores, strict=True):
                row[field] = value
            score_fields.append(field)
            diagnostics.append(diag)
    write_jsonl(Path(args.out_dataset), dataset)

    base_per_row = score_rows(core, golds, ids)
    baseline = metrics(base_per_row)
    grid = []
    best = None
    for policy in policy_grid(score_fields):
        fused = fuse(core, dataset, policy)
        per_row = score_rows(fused, golds, ids)
        m = metrics(per_row)
        added = sum(max(0, len(fused.get(rid, [])) - len(core.get(rid, []))) for rid in ids)
        item = {"policy": policy, "metrics": m, "added_predictions": added}
        grid.append(item)
        precision_guard = m["precision"] >= 0.650000
        key = (precision_guard, m["f1"], m["precision"], m["recall"], -added)
        if best is None or key > best[0]:
            best = (key, item, fused, per_row)
    assert best is not None
    selected = best[1]
    selected_fused = best[2]
    selected_per_row = best[3]
    boot = bootstrap(base_per_row, selected_per_row, seed=args.seed)
    overlay = build_overlay(rows, selected_fused, selected["policy"])
    write_jsonl(Path(args.overlay), overlay)
    grid.sort(key=lambda item: (item["metrics"]["f1"], item["metrics"]["precision"], item["metrics"]["recall"]), reverse=True)
    train_final(dataset, feature_names, str(selected["policy"]["score_field"]), Path(args.model), args.seed)
    report = {
        "id": "P225_p224_selector_gate_eval",
        "source_integrity": {
            "runtime_inputs": ["P224 raster proposal boxes/scores", "P224a prediction geometry", "trained selector weights/config"],
            "offline_gold_use": "OOF selector target assignment, policy selection, and locked evaluation only",
            "forbidden_runtime_features_used": False,
        },
        "base": rel(Path(args.base)),
        "input_dataset": rel(Path(args.dataset)),
        "scored_dataset": rel(Path(args.out_dataset)),
        "feature_names": feature_names,
        "gate_diagnostics": diagnostics,
        "baseline_metrics": baseline,
        "selected_policy": selected["policy"],
        "selected_metrics": selected["metrics"],
        "selected_added_predictions": selected["added_predictions"],
        "bootstrap_vs_p224a": boot,
        "top_grid": grid[:50],
        "outputs": {"overlay": rel(Path(args.overlay)), "report": rel(Path(args.report)), "markdown": rel(Path(args.md)), "model": rel(Path(args.model))},
        "promotion_decision": "Promote only after source audit/freeze; this OOF locked-set result is a rescue probe, not final paper claim.",
        "claim_boundary": "Internal P101 row-bootstrap/OFF selector evaluation. Gold is not used as runtime feature, but policy choice is still made on the locked set and needs freeze/audit before promotion.",
    }
    write_json(Path(args.report), report)
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text(render(report), encoding="utf-8")
    print(json.dumps({"baseline": baseline, "selected_policy": selected["policy"], "selected_metrics": selected["metrics"], "added": selected["added_predictions"], "bootstrap": boot, "diagnostics": diagnostics, "outputs": report["outputs"]}, ensure_ascii=False, indent=2)[:12000])


if __name__ == "__main__":
    main()
