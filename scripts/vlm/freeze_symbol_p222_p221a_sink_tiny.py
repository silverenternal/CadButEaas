#!/usr/bin/env python3
"""Freeze/reproduce P221a sink-tiny subcandidate rule from config."""
from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path
from typing import Any

from fuse_symbol_p206g_with_p211_p212 import bbox_iou, load_p206g

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs/vlm/symbol_p222_p221a_sink_tiny_frozen.json"


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("row_id"))


def pred_label(pred: dict[str, Any]) -> str:
    return str(pred.get("label", pred.get("symbol_type", "unknown")))


def pred_score(pred: dict[str, Any]) -> float:
    return float(pred.get("score", pred.get("confidence", 0.0)) or 0.0)


def box_area(box: list[float]) -> float:
    return max(0.0, float(box[2]) - float(box[0])) * max(0.0, float(box[3]) - float(box[1]))


def center(box: list[float]) -> tuple[float, float]:
    return ((float(box[0]) + float(box[2])) / 2.0, (float(box[1]) + float(box[3])) / 2.0)


def fixed_box(cx: float, cy: float, width: float, height: float) -> list[float]:
    return [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0]


def normalized_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for pred in row.get("symbol_candidates") or []:
        item = dict(pred)
        if "label" not in item and "symbol_type" in item:
            item["label"] = item["symbol_type"]
        if "symbol_type" not in item and "label" in item:
            item["symbol_type"] = item["label"]
        if "score" not in item and "confidence" in item:
            item["score"] = item["confidence"]
        if "confidence" not in item and "score" in item:
            item["confidence"] = item["score"]
        out.append(item)
    return out


def apply_rule(row: dict[str, Any], rule: dict[str, Any]) -> tuple[dict[str, Any], int]:
    rid = row_id(row)
    candidates = normalized_candidates(row)
    new_candidates = list(candidates)
    added = 0
    for index, pred in enumerate(candidates):
        if pred_label(pred) != str(rule["parent_label"]):
            continue
        bbox = [float(value) for value in pred["bbox"]]
        if box_area(bbox) > float(rule["parent_max_area"]):
            continue
        if pred_score(pred) < float(rule["parent_min_score"]):
            continue
        cx, cy = center(bbox)
        score = pred_score(pred) * float(rule["score_scale"])
        sub_id = f"{rid}_p222_p221a_sinktiny_{index:04d}"
        new_candidates.append({
            "id": sub_id,
            "target_id": sub_id,
            "symbol_type": "sink",
            "label": "sink",
            "bbox": fixed_box(cx, cy, float(rule["subcandidate_width"]), float(rule["subcandidate_height"])),
            "confidence": score,
            "score": score,
            "source": "p222_p221a_frozen_sink_tiny_subcandidate",
            "metadata": {
                "fusion_policy": rule["name"],
                "parent_id": pred.get("id") or pred.get("target_id"),
                "parent_score": pred_score(pred),
                "runtime_features": "parent_label_parent_score_parent_bbox_geometry_only"
            }
        })
        added += 1
    out = dict(row)
    out["symbol_candidates"] = new_candidates
    metadata = dict(out.get("metadata") or {})
    metadata["p222_p221a_rule"] = rule["name"]
    metadata["p222_p221a_added"] = added
    out["metadata"] = metadata
    return out, added


def score_rows(preds_by_row: dict[str, list[dict[str, Any]]], golds_by_row: dict[str, dict[str, dict[str, Any]]], ids: list[str]) -> list[dict[str, Any]]:
    per_row: list[dict[str, Any]] = []
    for rid in ids:
        preds = preds_by_row.get(rid, [])
        golds = list(golds_by_row[rid].values())
        candidates: list[tuple[float, int, int]] = []
        for pred_index, pred in enumerate(preds):
            pred_box = [float(value) for value in pred["bbox"]]
            label = pred_label(pred)
            for gold_index, gold in enumerate(golds):
                if label != str(gold["label"]):
                    continue
                iou = bbox_iou(pred_box, [float(value) for value in gold["bbox"]])
                if iou >= 0.30:
                    candidates.append((iou, pred_index, gold_index))
        used_preds: set[int] = set()
        used_golds: set[int] = set()
        for iou, pred_index, gold_index in sorted(candidates, reverse=True):
            if pred_index in used_preds or gold_index in used_golds:
                continue
            used_preds.add(pred_index)
            used_golds.add(gold_index)
        per_row.append({
            "row_id": rid,
            "counts": {
                "tp": len(used_golds),
                "pred": len(preds),
                "gold": len(golds),
                "fp": len(preds) - len(used_preds),
                "fn": len(golds) - len(used_golds),
            }
        })
    return per_row


def metrics(per_row: list[dict[str, Any]]) -> dict[str, Any]:
    counts = Counter()
    for row in per_row:
        counts.update(row["counts"])
    precision = counts["tp"] / max(counts["pred"], 1)
    recall = counts["tp"] / max(counts["gold"], 1)
    f1 = 2.0 * precision * recall / max(precision + recall, 1e-9)
    return {
        "tp": int(counts["tp"]),
        "predicted": int(counts["pred"]),
        "gold": int(counts["gold"]),
        "fp": int(counts["fp"]),
        "fn": int(counts["fn"]),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(f1, 6),
    }


def percentile(values: list[float], quantile: float) -> float:
    values = sorted(values)
    pos = (len(values) - 1) * quantile
    lo = int(pos)
    hi = min(lo + 1, len(values) - 1)
    frac = pos - lo
    return values[lo] * (1.0 - frac) + values[hi] * frac


def bootstrap(base: list[dict[str, Any]], cand: list[dict[str, Any]], iterations: int = 1000, seed: int = 222) -> dict[str, Any]:
    rng = random.Random(seed)
    deltas = {"f1": [], "precision": [], "recall": []}
    for _ in range(iterations):
        sample = [rng.randrange(len(base)) for _ in range(len(base))]
        bm = metrics([base[index] for index in sample])
        cm = metrics([cand[index] for index in sample])
        deltas["f1"].append(cm["f1"] - bm["f1"])
        deltas["precision"].append(cm["precision"] - bm["precision"])
        deltas["recall"].append(cm["recall"] - bm["recall"])
    return {
        f"{name}_delta": {
            "mean": round(sum(values) / len(values), 6),
            "ci95": [round(percentile(values, 0.025), 6), round(percentile(values, 0.975), 6)],
            "prob_positive": round(sum(value > 0 for value in values) / len(values), 6),
        }
        for name, values in deltas.items()
    }


def render(report: dict[str, Any]) -> str:
    bm = report["baseline_metrics"]
    cm = report["candidate_metrics"]
    boot = report["bootstrap"]
    fd = boot["f1_delta"]
    pd = boot["precision_delta"]
    rd = boot["recall_delta"]
    return "\n".join([
        "# P222 Frozen P221a Sink-Tiny Validation",
        "",
        "## Metrics",
        "| Variant | F1 | Precision | Recall | TP | Pred | Gold |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| P217/P218 baseline | {bm['f1']:.6f} | {bm['precision']:.6f} | {bm['recall']:.6f} | {bm['tp']} | {bm['predicted']} | {bm['gold']} |",
        f"| P222 frozen P221a | {cm['f1']:.6f} | {cm['precision']:.6f} | {cm['recall']:.6f} | {cm['tp']} | {cm['predicted']} | {cm['gold']} |",
        "",
        "## Frozen Rule",
        f"- Config: `{report['config']}`",
        f"- Rule: `{report['rule']['name']}`",
        f"- Added subcandidates: `{report['added_subcandidates']}`",
        "- Runtime features: parent symbol label, parent score, parent bbox geometry, frozen constants only.",
        "",
        "## Bootstrap vs P217/P218",
        f"- ΔF1 mean/CI/P>0: `{fd['mean']:.6f}` / `[{fd['ci95'][0]:.6f}, {fd['ci95'][1]:.6f}]` / `{fd['prob_positive']:.3f}`",
        f"- ΔPrecision mean/CI/P>0: `{pd['mean']:.6f}` / `[{pd['ci95'][0]:.6f}, {pd['ci95'][1]:.6f}]` / `{pd['prob_positive']:.3f}`",
        f"- ΔRecall mean/CI/P>0: `{rd['mean']:.6f}` / `[{rd['ci95'][0]:.6f}, {rd['ci95'][1]:.6f}]` / `{rd['prob_positive']:.3f}`",
        "",
        "## Claim Boundary",
        report["claim_boundary"],
        "",
    ])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    args = parser.parse_args()
    config_path = Path(args.config)
    config = read_json(config_path)
    base_overlay = ROOT / config["base_overlay"]
    outputs = {key: ROOT / value for key, value in config["outputs"].items()}
    rows = load_rows(base_overlay)
    frozen_rows: list[dict[str, Any]] = []
    total_added = 0
    for row in rows:
        frozen_row, added = apply_rule(row, config["rule"])
        frozen_rows.append(frozen_row)
        total_added += added
    outputs["overlay"].parent.mkdir(parents=True, exist_ok=True)
    outputs["overlay"].write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in frozen_rows) + "\n", encoding="utf-8")

    _, base_preds, golds = load_p206g(base_overlay)
    _, cand_preds, _ = load_p206g(outputs["overlay"])
    ids = [row_id(row) for row in rows]
    base_per_row = score_rows(base_preds, golds, ids)
    cand_per_row = score_rows(cand_preds, golds, ids)
    report = {
        "id": config["id"],
        "config": str(config_path.relative_to(ROOT)),
        "base_overlay": config["base_overlay"],
        "rule": config["rule"],
        "added_subcandidates": total_added,
        "baseline_metrics": metrics(base_per_row),
        "candidate_metrics": metrics(cand_per_row),
        "bootstrap": bootstrap(base_per_row, cand_per_row),
        "claim_boundary": config["claim_boundary"],
        "outputs": {key: str(path.relative_to(ROOT)) for key, path in outputs.items()},
    }
    write_json(outputs["eval"], report)
    write_json(outputs["bootstrap_json"], report)
    outputs["bootstrap_md"].write_text(render(report), encoding="utf-8")
    print(json.dumps({"candidate": report["candidate_metrics"], "bootstrap": report["bootstrap"], "added": total_added, "outputs": report["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
