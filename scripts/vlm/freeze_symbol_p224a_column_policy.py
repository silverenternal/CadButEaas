#!/usr/bin/env python3
"""Freeze P224a precision-safe column policy over P222 baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from freeze_symbol_p222_p221a_sink_tiny import bootstrap, metrics, score_rows
from fuse_symbol_p206g_with_p211_p212 import bbox_iou, load_p206g

ROOT = Path(__file__).resolve().parents[2]
CONFIG = ROOT / "configs/vlm/symbol_p224a_column_policy_frozen.json"
BASE = ROOT / "reports/vlm/symbol_p222_p221a_frozen_overlay.jsonl"
P224 = ROOT / "reports/vlm/symbol_p224_detector_pages_s384_predictions.jsonl"
OVERLAY = ROOT / "reports/vlm/symbol_p224a_column_frozen_overlay.jsonl"
EVAL = ROOT / "reports/vlm/symbol_p224a_column_frozen_eval.json"
BOOT_MD = ROOT / "reports/vlm/symbol_p224a_column_bootstrap_validation.md"
BOOT_JSON = ROOT / "reports/vlm/symbol_p224a_column_bootstrap_validation.json"

DEFAULT_CONFIG = {
    "id": "P224a_column_policy_frozen",
    "base_overlay": "reports/vlm/symbol_p222_p221a_frozen_overlay.jsonl",
    "p224_predictions": "reports/vlm/symbol_p224_detector_pages_s384_predictions.jsonl",
    "rule": {
        "name": "p224_narrow_column_t0.85_a3_iou0.35_d0",
        "allowed_label": "column",
        "score_threshold": 0.85,
        "max_add_per_row": 3,
        "max_add_per_label": 3,
        "max_iou_to_existing_same_label": 0.35,
        "min_center_dist_to_existing_same_label": 0.0,
        "same_label_only": True,
    },
    "outputs": {
        "overlay": "reports/vlm/symbol_p224a_column_frozen_overlay.jsonl",
        "eval": "reports/vlm/symbol_p224a_column_frozen_eval.json",
        "bootstrap_md": "reports/vlm/symbol_p224a_column_bootstrap_validation.md",
        "bootstrap_json": "reports/vlm/symbol_p224a_column_bootstrap_validation.json",
    },
    "claim_boundary": "Frozen runtime-safe P224a column-only policy over P222 and P224 s384 raster detector predictions; internal P101/bootstrap-bounded, not reviewer-grade final recognition.",
}


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def load_rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("row_id"))


def pred_label(pred: dict[str, Any]) -> str:
    return str(pred.get("label") or pred.get("symbol_type") or "generic_symbol")


def pred_score(pred: dict[str, Any]) -> float:
    return float(pred.get("score") or pred.get("confidence") or 0.0)


def center_dist(left: list[float], right: list[float]) -> float:
    lx = (left[0] + left[2]) / 2.0
    ly = (left[1] + left[3]) / 2.0
    rx = (right[0] + right[2]) / 2.0
    ry = (right[1] + right[3]) / 2.0
    return ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5


def normalize_base_candidates(row: dict[str, Any]) -> list[dict[str, Any]]:
    preds = []
    for cand in row.get("symbol_candidates") or []:
        if "bbox" not in cand:
            continue
        preds.append({
            "bbox": [float(v) for v in cand["bbox"]],
            "label": pred_label(cand),
            "score": pred_score(cand),
            "source": cand.get("source") or "p222",
            "tile_id": (cand.get("metadata") or {}).get("tile_id"),
        })
    return preds


def load_p224_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    by_row: dict[str, list[dict[str, Any]]] = {}
    for row in load_rows(path):
        rid = str(row.get("row_id"))
        preds = []
        for pred in row.get("predicted_symbols") or []:
            if "bbox" not in pred:
                continue
            preds.append({
                "bbox": [float(v) for v in pred["bbox"]],
                "label": str(pred.get("label") or "generic_symbol"),
                "score": float(pred.get("score", 0.0)),
                "source": "p224a_column_added",
                "tile_id": pred.get("tile_id"),
            })
        preds.sort(key=lambda item: float(item.get("score", 0.0)), reverse=True)
        by_row[rid] = preds
    return by_row


def conflicts(candidate: dict[str, Any], existing: list[dict[str, Any]], rule: dict[str, Any]) -> bool:
    cbox = [float(v) for v in candidate["bbox"]]
    for pred in existing:
        if rule.get("same_label_only", True) and pred_label(pred) != pred_label(candidate):
            continue
        pbox = [float(v) for v in pred["bbox"]]
        if bbox_iou(cbox, pbox) >= float(rule["max_iou_to_existing_same_label"]):
            return True
        if center_dist(cbox, pbox) <= float(rule["min_center_dist_to_existing_same_label"]):
            return True
    return False


def apply_policy(row: dict[str, Any], p224_by_row: dict[str, list[dict[str, Any]]], rule: dict[str, Any]) -> tuple[dict[str, Any], int]:
    rid = row_id(row)
    base_preds = normalize_base_candidates(row)
    additions = []
    per_label_count: dict[str, int] = {}
    for pred in p224_by_row.get(rid, []):
        label = pred_label(pred)
        if label != rule["allowed_label"]:
            continue
        if pred_score(pred) < float(rule["score_threshold"]):
            continue
        if per_label_count.get(label, 0) >= int(rule["max_add_per_label"]):
            continue
        if conflicts(pred, base_preds + additions, rule):
            continue
        addition = dict(pred)
        addition["source"] = "p224a_column_added"
        additions.append(addition)
        per_label_count[label] = per_label_count.get(label, 0) + 1
        if len(additions) >= int(rule["max_add_per_row"]):
            break
    new_row = dict(row)
    new_candidates = []
    for index, pred in enumerate(base_preds + additions):
        new_candidates.append({
            "id": f"{rid}_p224a_symbol_{index:05d}",
            "target_id": f"{rid}_p224a_symbol_{index:05d}",
            "symbol_type": pred_label(pred),
            "bbox": [float(v) for v in pred["bbox"]],
            "confidence": pred_score(pred),
            "source": pred.get("source"),
            "metadata": {"tile_id": pred.get("tile_id"), "fusion_policy": rule["name"]},
        })
    new_row["symbol_candidates"] = new_candidates
    new_row["symbol_policy_overlay"] = {"policy_id": "p224a_column_policy_frozen", "policy": rule}
    return new_row, len(additions)


def render(report: dict[str, Any]) -> str:
    bm = report["baseline_metrics"]
    cm = report["candidate_metrics"]
    boot = report["bootstrap"]
    return "\n".join([
        "# P224a Frozen Column Policy Validation",
        "",
        "## Metrics",
        "| Variant | F1 | Precision | Recall | TP | Pred | Gold |",
        "|---|---:|---:|---:|---:|---:|---:|",
        f"| P222 baseline | {bm['f1']:.6f} | {bm['precision']:.6f} | {bm['recall']:.6f} | {bm['tp']} | {bm['predicted']} | {bm['gold']} |",
        f"| P224a frozen | {cm['f1']:.6f} | {cm['precision']:.6f} | {cm['recall']:.6f} | {cm['tp']} | {cm['predicted']} | {cm['gold']} |",
        "",
        "## Frozen Policy",
        f"- Rule: `{report['rule']['name']}`",
        f"- Added column candidates: `{report['added_candidates']}`",
        "- Runtime features: P222 symbol predictions, P224 s384 raster detector predictions, bbox geometry, detector score/label, and frozen constants only.",
        "",
        "## Bootstrap vs P222",
        f"- ΔF1 mean/CI/P>0: `{boot['f1_delta']['mean']:.6f}` / `{boot['f1_delta']['ci95']}` / `{boot['f1_delta']['prob_positive']:.3f}`",
        f"- ΔPrecision mean/CI/P>0: `{boot['precision_delta']['mean']:.6f}` / `{boot['precision_delta']['ci95']}` / `{boot['precision_delta']['prob_positive']:.3f}`",
        f"- ΔRecall mean/CI/P>0: `{boot['recall_delta']['mean']:.6f}` / `{boot['recall_delta']['ci95']}` / `{boot['recall_delta']['prob_positive']:.3f}`",
        "",
        "## Claim Boundary",
        report["claim_boundary"],
        "",
    ])


def ensure_config(path: Path) -> dict[str, Any]:
    if not path.exists():
        write_json(path, DEFAULT_CONFIG)
    return read_json(path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(CONFIG))
    args = ap.parse_args()
    config_path = Path(args.config)
    config = ensure_config(config_path)
    base_overlay = ROOT / config["base_overlay"]
    p224_predictions = ROOT / config["p224_predictions"]
    outputs = {key: ROOT / value for key, value in config["outputs"].items()}
    rows = load_rows(base_overlay)
    p224_by_row = load_p224_predictions(p224_predictions)
    frozen_rows = []
    total_added = 0
    for row in rows:
        frozen, added = apply_policy(row, p224_by_row, config["rule"])
        frozen_rows.append(frozen)
        total_added += added
    write_jsonl(outputs["overlay"], frozen_rows)
    _, base_preds, golds = load_p206g(base_overlay)
    _, cand_preds, _ = load_p206g(outputs["overlay"])
    ids = [row_id(row) for row in rows]
    base_per = score_rows(base_preds, golds, ids)
    cand_per = score_rows(cand_preds, golds, ids)
    report = {
        "id": config["id"],
        "config": rel(config_path),
        "base_overlay": config["base_overlay"],
        "p224_predictions": config["p224_predictions"],
        "rule": config["rule"],
        "added_candidates": total_added,
        "baseline_metrics": metrics(base_per),
        "candidate_metrics": metrics(cand_per),
        "bootstrap": bootstrap(base_per, cand_per, seed=2246),
        "claim_boundary": config["claim_boundary"],
        "outputs": {key: rel(path) for key, path in outputs.items()},
    }
    write_json(outputs["eval"], report)
    write_json(outputs["bootstrap_json"], report)
    outputs["bootstrap_md"].parent.mkdir(parents=True, exist_ok=True)
    outputs["bootstrap_md"].write_text(render(report), encoding="utf-8")
    print(json.dumps({"candidate": report["candidate_metrics"], "bootstrap": report["bootstrap"], "added": total_added, "outputs": report["outputs"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
