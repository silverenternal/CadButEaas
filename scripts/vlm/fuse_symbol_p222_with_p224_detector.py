#!/usr/bin/env python3
"""Targeted fusion of P224 page-sliced detector proposals into P222 baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from freeze_symbol_p222_p221a_sink_tiny import bootstrap, metrics, score_rows
from fuse_symbol_p206g_with_p211_p212 import bbox_iou, load_p206g, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/vlm/symbol_p222_p221a_frozen_overlay.jsonl"
P224 = ROOT / "reports/vlm/symbol_p224_detector_pages_sliced_predictions.jsonl"
REPORT = ROOT / "reports/vlm/symbol_p224_p222_targeted_fusion_eval.json"
MD = ROOT / "reports/vlm/symbol_p224_p222_targeted_fusion_eval.md"
OVERLAY = ROOT / "reports/vlm/symbol_p224_p222_targeted_fusion_overlay.jsonl"


def rel_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def load_predictions(path: Path, allowed: set[str], topk: int, min_score: float) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for row in read_jsonl(path):
        rid = str(row.get("row_id"))
        if rid not in allowed:
            continue
        preds = []
        for pred in row.get("predicted_symbols") or []:
            score = float(pred.get("score", 0.0))
            if score < min_score:
                continue
            preds.append({
                "bbox": [float(v) for v in pred["bbox"]],
                "label": str(pred.get("label") or "generic_symbol"),
                "score": score,
                "source": "p224_detector_added",
                "tile_id": pred.get("tile_id"),
            })
        preds.sort(key=lambda item: float(item["score"]), reverse=True)
        out[rid] = preds[:topk]
    return out


def center_dist(left: list[float], right: list[float]) -> float:
    lx = (left[0] + left[2]) / 2.0; ly = (left[1] + left[3]) / 2.0
    rx = (right[0] + right[2]) / 2.0; ry = (right[1] + right[3]) / 2.0
    return ((lx-rx)**2 + (ly-ry)**2) ** 0.5


def conflict(cand: dict[str, Any], existing: list[dict[str, Any]], policy: dict[str, Any]) -> bool:
    box = cand["bbox"]
    for pred in existing:
        if policy.get("same_label_only", False) and pred.get("label") != cand.get("label"):
            continue
        other = [float(v) for v in pred["bbox"]]
        if bbox_iou(box, other) >= float(policy["max_iou_to_existing"]):
            return True
        if center_dist(box, other) <= float(policy["min_center_dist"]):
            return True
    return False


def fuse(core: dict[str, list[dict[str, Any]]], p224: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    thresholds = {k: float(v) for k, v in policy.get("label_thresholds", {}).items()}
    labels = set(policy["labels"])
    out = {}
    for rid, base_preds in core.items():
        merged = [dict(pred) for pred in base_preds]
        additions = []
        per_label = {}
        for pred in p224.get(rid, []):
            label = str(pred.get("label"))
            if label not in labels:
                continue
            if float(pred.get("score", 0.0)) < thresholds.get(label, float(policy["threshold"])):
                continue
            if per_label.get(label, 0) >= int(policy["max_add_per_label"]):
                continue
            if conflict(pred, merged + additions, policy):
                continue
            add = dict(pred)
            add["source"] = "p224_detector_added"
            additions.append(add)
            per_label[label] = per_label.get(label, 0) + 1
            if len(additions) >= int(policy["max_add_per_row"]):
                break
        out[rid] = merged + additions
    return out


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("row_id"))


def build_overlay(rows: list[dict[str, Any]], fused: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        rid = row_id(row)
        nr = dict(row)
        cands = []
        for idx, pred in enumerate(fused.get(rid, [])):
            cands.append({
                "id": f"{rid}_p224_symbol_{idx:05d}",
                "target_id": f"{rid}_p224_symbol_{idx:05d}",
                "symbol_type": pred.get("label"),
                "bbox": pred.get("bbox"),
                "confidence": pred.get("score"),
                "source": pred.get("source"),
                "metadata": {"tile_id": pred.get("tile_id"), "fusion_policy": policy.get("name")},
            })
        nr["symbol_candidates"] = cands
        nr["symbol_policy_overlay"] = {"policy_id": "p224_targeted_fusion", "policy": policy}
        out.append(nr)
    return out


def policies() -> list[dict[str, Any]]:
    label_sets = [["column"], ["column", "stair"], ["column", "equipment"], ["column", "stair", "equipment"], ["sink", "shower"]]
    out = []
    for labels in label_sets:
        for threshold in [0.3, 0.5, 0.7, 0.85, 0.9]:
            for max_add in [1, 2, 3]:
                for iou in [0.15, 0.35, 0.5]:
                    for dist in [0.0, 8.0]:
                        out.append({
                            "name": f"p224_narrow_{'-'.join(labels)}_t{threshold:g}_a{max_add}_iou{iou:g}_d{dist:g}",
                            "labels": labels,
                            "threshold": threshold,
                            "max_add_per_row": max_add,
                            "max_add_per_label": max_add,
                            "max_iou_to_existing": iou,
                            "min_center_dist": dist,
                            "same_label_only": True,
                        })
    return out


def render(report: dict[str, Any]) -> str:
    bm = report["baseline_metrics"]; sm = report["selected_metrics"]; b = report["bootstrap_vs_p222"]
    return "\n".join([
        "# P224 Targeted Fusion Eval", "",
        "| Variant | F1 | P | R | TP | Pred | Gold |", "|---|---:|---:|---:|---:|---:|---:|",
        f"| P222 | {bm['f1']:.6f} | {bm['precision']:.6f} | {bm['recall']:.6f} | {bm['tp']} | {bm['predicted']} | {bm['gold']} |",
        f"| P224 fused | {sm['f1']:.6f} | {sm['precision']:.6f} | {sm['recall']:.6f} | {sm['tp']} | {sm['predicted']} | {sm['gold']} |", "",
        f"- Selected policy: `{report['selected_policy']['name']}`",
        f"- Additions: `{report['selected_added_predictions']}`",
        f"- ΔF1 CI: `{b['f1_delta']['ci95']}`",
        f"- ΔPrecision CI: `{b['precision_delta']['ci95']}`",
        f"- ΔRecall CI: `{b['recall_delta']['ci95']}`", "",
    ])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--p224", default=str(P224))
    ap.add_argument("--report", default=str(REPORT))
    ap.add_argument("--md", default=str(MD))
    ap.add_argument("--overlay", default=str(OVERLAY))
    ap.add_argument("--topk", type=int, default=1000)
    ap.add_argument("--min-score", type=float, default=0.001)
    args = ap.parse_args()
    rows, core, golds = load_p206g(Path(args.base))
    ids = [row_id(row) for row in rows]
    p224 = load_predictions(Path(args.p224), set(ids), args.topk, args.min_score)
    base_per = score_rows(core, golds, ids)
    baseline = metrics(base_per)
    results = []
    best = None
    for policy in policies():
        fused = fuse(core, p224, policy)
        per = score_rows(fused, golds, ids)
        m = metrics(per)
        added = sum(max(0, len(fused[rid]) - len(core.get(rid, []))) for rid in ids)
        item = {"policy": policy, "metrics": m, "added_predictions": added}
        results.append(item)
        key = (m["f1"], m["precision"] >= baseline["precision"] - 0.01, m["recall"], -added, m["precision"])
        if best is None or key > best[0]:
            best = (key, item, fused, per)
    assert best is not None
    selected = best[1]
    boot = bootstrap(base_per, best[3], seed=2244)
    write_jsonl(Path(args.overlay), build_overlay(rows, best[2], selected["policy"]))
    results.sort(key=lambda x: (x["metrics"]["f1"], x["metrics"]["precision"], x["metrics"]["recall"]), reverse=True)
    report = {
        "id": "P224_targeted_fusion_eval",
        "baseline_metrics": baseline,
        "selected_policy": selected["policy"],
        "selected_metrics": selected["metrics"],
        "selected_added_predictions": selected["added_predictions"],
        "bootstrap_vs_p222": boot,
        "top_grid": results[:50],
        "inputs": {"base": rel_path(Path(args.base)), "p224": rel_path(Path(args.p224))},
        "outputs": {"overlay": rel_path(Path(args.overlay)), "markdown": rel_path(Path(args.md))},
        "claim_boundary": "Internal P101 policy search over P224 raster predictions; requires source audit and independent validation before claims.",
    }
    write_json(Path(args.report), report)
    Path(args.md).write_text(render(report), encoding="utf-8")
    print(json.dumps({"baseline": baseline, "selected": selected, "bootstrap": boot}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
