#!/usr/bin/env python3
"""Train/evaluate P221c proposal-level gate for P213b/P221b residual candidates."""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

from freeze_symbol_p222_p221a_sink_tiny import bootstrap, metrics, score_rows
from fuse_symbol_p206g_with_p211_p212 import LABEL_TO_ID, bbox_iou, load_p206g, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/vlm/symbol_p222_p221a_frozen_overlay.jsonl"
PROPOSAL = ROOT / "reports/vlm/symbol_p212_p213b_residual_fusion_overlay.jsonl"
STAIR_VERIFIED = ROOT / "reports/vlm/symbol_p221b_stair_candidate_verified_predictions.jsonl"
REPORT = ROOT / "reports/vlm/symbol_p221c_candidate_gate_eval.json"
MD = ROOT / "reports/vlm/symbol_p221c_candidate_gate_eval.md"
OVERLAY = ROOT / "reports/vlm/symbol_p221c_candidate_gate_overlay.jsonl"
DATASET = ROOT / "reports/vlm/symbol_p221c_candidate_gate_dataset.jsonl"

LABELS = ["equipment", "stair"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("row_id"))


def cand_label(candidate: dict[str, Any]) -> str:
    return str(candidate.get("symbol_type") or candidate.get("label") or "generic_symbol")


def cand_score(candidate: dict[str, Any]) -> float:
    return float(candidate.get("confidence") or candidate.get("score") or 0.0)


def norm_candidate(candidate: dict[str, Any], default_source: str) -> dict[str, Any]:
    label = cand_label(candidate)
    return {
        "bbox": [float(v) for v in candidate["bbox"]],
        "label": label,
        "label_id": LABEL_TO_ID.get(label, LABEL_TO_ID["generic_symbol"]),
        "score": cand_score(candidate),
        "source": default_source,
        "tile_id": (candidate.get("metadata") or {}).get("tile_id"),
    }


def center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def dist(left: list[float], right: list[float]) -> float:
    lx, ly = center(left)
    rx, ry = center(right)
    return ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5


def load_stair_verified(path: Path) -> dict[tuple[str, tuple[float, ...]], dict[str, float]]:
    out: dict[tuple[str, tuple[float, ...]], dict[str, float]] = {}
    if not path.exists():
        return out
    for row in read_jsonl(path):
        rid = str(row.get("row_id"))
        for pred in row.get("predicted_symbols") or []:
            key = (rid, tuple(round(float(v), 3) for v in pred.get("bbox") or []))
            out[key] = {
                "stair_verifier_score": float(pred.get("verifier_score", 0.0) or 0.0),
                "stair_fused_score": float(pred.get("fused_score", 0.0) or 0.0),
            }
    return out


def candidate_key(rid: str, pred: dict[str, Any]) -> tuple[str, str, tuple[float, ...]]:
    return rid, pred["label"], tuple(round(float(v), 3) for v in pred["bbox"])


def build_dataset(rows: list[dict[str, Any]], core: dict[str, list[dict[str, Any]]], golds: dict[str, dict[str, dict[str, Any]]], proposal_path: Path, stair_verified_path: Path) -> list[dict[str, Any]]:
    proposal_rows = {row_id(row): row for row in read_jsonl(proposal_path)}
    stair_scores = load_stair_verified(stair_verified_path)
    dataset: list[dict[str, Any]] = []
    seen = set()
    for row in rows:
        rid = row_id(row)
        width, height = row.get("image_size") or [1, 1]
        width = max(float(width), 1.0)
        height = max(float(height), 1.0)
        core_preds = core.get(rid, [])
        proposal_candidates = proposal_rows.get(rid, {}).get("symbol_candidates") or []
        for candidate in proposal_candidates:
            label = cand_label(candidate)
            if label not in LABELS:
                continue
            pred = norm_candidate(candidate, "p221c_candidate_gate_input")
            key = candidate_key(rid, pred)
            if key in seen:
                continue
            seen.add(key)
            box = pred["bbox"]
            same_core = [q for q in core_preds if str(q.get("label")) == label]
            any_core = core_preds
            nearest_same_iou = max([bbox_iou(box, [float(v) for v in q["bbox"]]) for q in same_core] or [0.0])
            nearest_any_iou = max([bbox_iou(box, [float(v) for v in q["bbox"]]) for q in any_core] or [0.0])
            nearest_same_dist = min([dist(box, [float(v) for v in q["bbox"]]) for q in same_core] or [9999.0])
            nearest_any_dist = min([dist(box, [float(v) for v in q["bbox"]]) for q in any_core] or [9999.0])
            best_iou = 0.0
            is_new_tp = 0
            is_tp_any = 0
            for gold in golds[rid].values():
                if str(gold.get("label")) != label:
                    continue
                gbox = [float(v) for v in gold["bbox"]]
                iou = bbox_iou(box, gbox)
                best_iou = max(best_iou, iou)
                if iou >= 0.30:
                    is_tp_any = 1
                    if not any(q.get("label") == label and bbox_iou([float(v) for v in q["bbox"]], gbox) >= 0.30 for q in core_preds):
                        is_new_tp = 1
            cx, cy = center(box)
            st = stair_scores.get((rid, tuple(round(float(v), 3) for v in box)), {})
            features = {
                "score": pred["score"],
                "score_logit": math.log(max(pred["score"], 1e-6) / max(1.0 - pred["score"], 1e-6)),
                "label_is_equipment": 1.0 if label == "equipment" else 0.0,
                "label_is_stair": 1.0 if label == "stair" else 0.0,
                "area_norm": area(box) / (width * height),
                "sqrt_area_norm": math.sqrt(max(area(box), 0.0)) / math.sqrt(width * height),
                "aspect": (box[2] - box[0]) / max(box[3] - box[1], 1e-6),
                "cx_norm": cx / width,
                "cy_norm": cy / height,
                "nearest_same_iou": nearest_same_iou,
                "nearest_any_iou": nearest_any_iou,
                "nearest_same_dist_norm": min(nearest_same_dist, 9999.0) / math.sqrt(width * height),
                "nearest_any_dist_norm": min(nearest_any_dist, 9999.0) / math.sqrt(width * height),
                "core_count": float(len(core_preds)),
                "same_label_core_count": float(len(same_core)),
                "stair_verifier_score": st.get("stair_verifier_score", 0.0),
                "stair_fused_score": st.get("stair_fused_score", 0.0),
            }
            dataset.append({
                "row_id": rid,
                "label": label,
                "bbox": box,
                "score": pred["score"],
                "features": features,
                "target_new_tp": is_new_tp,
                "target_any_tp": is_tp_any,
                "best_iou_to_same_label_gold": best_iou,
                "candidate": pred,
            })
    return dataset


def split_rows(ids: list[str], folds: int, seed: int) -> dict[str, int]:
    rng = random.Random(seed)
    shuffled = list(ids)
    rng.shuffle(shuffled)
    return {rid: index % folds for index, rid in enumerate(shuffled)}


def make_model(name: str):
    if name == "logreg":
        return make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced", C=0.5))
    if name == "rf":
        return RandomForestClassifier(n_estimators=300, min_samples_leaf=3, class_weight="balanced_subsample", random_state=221, n_jobs=-1)
    if name == "hgb":
        return HistGradientBoostingClassifier(max_iter=200, learning_rate=0.04, l2_regularization=0.05, min_samples_leaf=5, random_state=221)
    raise ValueError(name)


def add_oof_scores(dataset: list[dict[str, Any]], feature_names: list[str], model_name: str, target: str, folds: int, seed: int) -> tuple[list[float], dict[str, Any]]:
    row_folds = split_rows(sorted({row["row_id"] for row in dataset}), folds, seed)
    scores = [0.0] * len(dataset)
    ys = np.array([row[target] for row in dataset], dtype=int)
    for fold in range(folds):
        train_idx = [i for i, row in enumerate(dataset) if row_folds[row["row_id"]] != fold]
        test_idx = [i for i, row in enumerate(dataset) if row_folds[row["row_id"]] == fold]
        x_train = np.array([[dataset[i]["features"][name] for name in feature_names] for i in train_idx], dtype=float)
        y_train = ys[train_idx]
        x_test = np.array([[dataset[i]["features"][name] for name in feature_names] for i in test_idx], dtype=float)
        if len(set(y_train.tolist())) < 2:
            pred = np.full(len(test_idx), float(y_train.mean() if len(y_train) else 0.0))
        else:
            model = make_model(model_name)
            model.fit(x_train, y_train)
            pred = model.predict_proba(x_test)[:, 1]
        for index, value in zip(test_idx, pred, strict=True):
            scores[index] = float(value)
    auc = None
    if len(set(ys.tolist())) > 1:
        auc = float(roc_auc_score(ys, np.array(scores)))
    return scores, {"model": model_name, "target": target, "folds": folds, "auc": auc, "positive": int(ys.sum()), "total": int(len(ys))}


def conflicts(candidate: dict[str, Any], existing: list[dict[str, Any]], max_iou: float, min_dist: float, same_label_only: bool) -> bool:
    cbox = candidate["bbox"]
    for pred in existing:
        if same_label_only and str(pred.get("label")) != candidate["label"]:
            continue
        pbox = [float(v) for v in pred["bbox"]]
        if bbox_iou(cbox, pbox) >= max_iou:
            return True
        if dist(cbox, pbox) <= min_dist:
            return True
    return False


def fuse(core: dict[str, list[dict[str, Any]]], dataset: list[dict[str, Any]], policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    by_row: dict[str, list[dict[str, Any]]] = defaultdict(list)
    score_field = policy["score_field"]
    for row in dataset:
        if row["label"] not in policy["labels"]:
            continue
        gate_score = float(row.get(score_field, 0.0))
        raw_score = float(row.get("score", 0.0))
        if gate_score < float(policy["threshold"]):
            continue
        if raw_score < float(policy.get("min_raw_score", 0.0)):
            continue
        cand = dict(row["candidate"])
        cand["score"] = gate_score
        cand["confidence"] = gate_score
        cand["source"] = "p221c_candidate_gate_added"
        cand["gate_score"] = gate_score
        cand["raw_score"] = raw_score
        by_row[row["row_id"]].append(cand)
    out: dict[str, list[dict[str, Any]]] = {}
    for rid, core_preds in core.items():
        merged = [dict(pred) for pred in core_preds]
        additions = []
        for cand in sorted(by_row.get(rid, []), key=lambda item: float(item.get("gate_score", 0.0)), reverse=True):
            if conflicts(cand, merged + additions, float(policy["max_iou_to_existing"]), float(policy["min_center_dist"]), bool(policy["same_label_only"])):
                continue
            additions.append(cand)
            if len(additions) >= int(policy["max_add_per_row"]):
                break
        out[rid] = merged + additions
    return out


def build_overlay(rows: list[dict[str, Any]], fused: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        rid = row_id(row)
        nr = dict(row)
        candidates = []
        for index, pred in enumerate(fused.get(rid, [])):
            candidates.append({
                "id": f"{rid}_p221c_symbol_{index:05d}",
                "target_id": f"{rid}_p221c_symbol_{index:05d}",
                "symbol_type": pred.get("label"),
                "bbox": pred.get("bbox"),
                "confidence": pred.get("score"),
                "source": pred.get("source"),
                "metadata": {"tile_id": pred.get("tile_id"), "fusion_policy": policy.get("name"), "raw_score": pred.get("raw_score"), "gate_score": pred.get("gate_score")},
            })
        nr["symbol_candidates"] = candidates
        nr["symbol_policy_overlay"] = {"policy_id": "p221c_candidate_gate", "policy": policy}
        out.append(nr)
    return out


def policy_grid(score_fields: list[str]) -> list[dict[str, Any]]:
    policies = []
    label_sets = [{"equipment"}, {"stair"}, {"equipment", "stair"}]
    thresholds = [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90]
    raw_thresholds = [0.0, 0.90, 0.95, 0.97]
    for score_field in score_fields:
        for labels in label_sets:
            for threshold in thresholds:
                for min_raw in raw_thresholds:
                    for max_add in [1, 2, 3, 5, 10, 20]:
                        for max_iou in [0.05, 0.20, 0.50, 1.10]:
                            policies.append({
                                "name": f"p221c_{score_field}_{'-'.join(sorted(labels))}_t{threshold:g}_raw{min_raw:g}_a{max_add}_iou{max_iou:g}",
                                "score_field": score_field,
                                "labels": sorted(labels),
                                "threshold": threshold,
                                "min_raw_score": min_raw,
                                "max_add_per_row": max_add,
                                "max_iou_to_existing": max_iou,
                                "min_center_dist": 0.0,
                                "same_label_only": True,
                            })
    return policies


def render(report: dict[str, Any]) -> str:
    bm = report["baseline_metrics"]
    sm = report["selected_metrics"]
    b = report["bootstrap_vs_p222"]
    lines = [
        "# P221c Candidate Gate Evaluation",
        "",
        "## Metrics",
        "| Variant | F1 | Precision | Recall | TP | Pred | Gold |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| P222 frozen | {bm['f1']:.6f} | {bm['precision']:.6f} | {bm['recall']:.6f} | {bm['tp']} | {bm['predicted']} | {bm['gold']} |",
        f"| P221c selected | {sm['f1']:.6f} | {sm['precision']:.6f} | {sm['recall']:.6f} | {sm['tp']} | {sm['predicted']} | {sm['gold']} |",
        "",
        "## Selected Policy",
        f"- `{report['selected_policy']['name']}`",
        f"- Added predictions: `{report['selected_added_predictions']}`",
        "",
        "## Bootstrap vs P222",
        f"- ΔF1: `{b['f1_delta']['mean']:.6f}` / `{b['f1_delta']['ci95']}` / P>0 `{b['f1_delta']['prob_positive']:.3f}`",
        f"- ΔPrecision: `{b['precision_delta']['mean']:.6f}` / `{b['precision_delta']['ci95']}` / P>0 `{b['precision_delta']['prob_positive']:.3f}`",
        f"- ΔRecall: `{b['recall_delta']['mean']:.6f}` / `{b['recall_delta']['ci95']}` / P>0 `{b['recall_delta']['prob_positive']:.3f}`",
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
    parser.add_argument("--proposal", default=str(PROPOSAL))
    parser.add_argument("--stair-verified", default=str(STAIR_VERIFIED))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--md", default=str(MD))
    parser.add_argument("--overlay", default=str(OVERLAY))
    parser.add_argument("--dataset", default=str(DATASET))
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--seed", type=int, default=221)
    args = parser.parse_args()

    rows, core, golds = load_p206g(Path(args.base))
    ids = [row_id(row) for row in rows]
    dataset = build_dataset(rows, core, golds, Path(args.proposal), Path(args.stair_verified))
    feature_names = sorted(dataset[0]["features"].keys()) if dataset else []
    diagnostics = []
    score_fields = ["score"]
    for model_name in ["logreg", "hgb", "rf"]:
        for target in ["target_new_tp", "target_any_tp"]:
            scores, diag = add_oof_scores(dataset, feature_names, model_name, target, args.folds, args.seed)
            field = f"gate_{model_name}_{target}"
            for row, value in zip(dataset, scores, strict=True):
                row[field] = value
            score_fields.append(field)
            diagnostics.append(diag)
    write_jsonl(Path(args.dataset), dataset)

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
        precision_guard = m["precision"] >= baseline["precision"] - 0.0005
        key = (precision_guard, m["f1"], m["precision"], m["recall"], -added)
        if best is None or key > best[0]:
            best = (key, item, fused, per_row)
    assert best is not None
    selected = best[1]
    boot = bootstrap(base_per_row, best[3], seed=args.seed)
    overlay = build_overlay(rows, best[2], selected["policy"])
    write_jsonl(Path(args.overlay), overlay)
    grid.sort(key=lambda item: (item["metrics"]["f1"], item["metrics"]["precision"], item["metrics"]["recall"]), reverse=True)
    report = {
        "id": "P221c_candidate_gate_eval",
        "source_integrity": {
            "runtime_inputs": ["raster-derived proposal boxes/scores", "P222 prediction geometry/scores", "trained gate weights/config"],
            "offline_gold_use": "candidate labeling and evaluation only",
            "forbidden_runtime_features_used": False,
        },
        "dataset": str(Path(args.dataset).relative_to(ROOT)),
        "dataset_counts": {f"{label}_{target}": count for (label, target), count in Counter((row["label"], row["target_new_tp"]) for row in dataset).items()},
        "feature_names": feature_names,
        "gate_diagnostics": diagnostics,
        "baseline_metrics": baseline,
        "selected_policy": selected["policy"],
        "selected_metrics": selected["metrics"],
        "selected_added_predictions": selected["added_predictions"],
        "bootstrap_vs_p222": boot,
        "top_grid": grid[:40],
        "outputs": {"overlay": str(Path(args.overlay).relative_to(ROOT)), "markdown": str(Path(args.md).relative_to(ROOT))},
        "promotion_decision": "promote_only_if_delta_f1_CI_positive_and_delta_precision_CI_non_negative",
        "claim_boundary": "Internal P101 row-bootstrap evaluation; OOF gate scores reduce direct candidate-label leakage but independent page validation is still required for broad claims.",
    }
    write_json(Path(args.report), report)
    Path(args.md).write_text(render(report), encoding="utf-8")
    print(json.dumps({"baseline": baseline, "selected_policy": selected["policy"], "selected_metrics": selected["metrics"], "added": selected["added_predictions"], "bootstrap": boot, "diagnostics": diagnostics}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
