#!/usr/bin/env python3
"""Fuse P226 stair-specialist page proposals into P224a baseline."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from freeze_symbol_p222_p221a_sink_tiny import bootstrap, metrics, score_rows
from fuse_symbol_p206g_with_p211_p212 import bbox_iou, load_p206g, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/vlm/symbol_p224a_column_frozen_overlay.jsonl"
P226 = ROOT / "reports/vlm/symbol_p226_stair_specialist_pages_predictions.jsonl"
REPORT = ROOT / "reports/vlm/symbol_p226_stair_fusion_eval.json"
MD = ROOT / "reports/vlm/symbol_p226_stair_fusion_eval.md"
OVERLAY = ROOT / "reports/vlm/symbol_p226_stair_fusion_overlay.jsonl"


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT.resolve()))
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("row_id"))


def center_dist(left: list[float], right: list[float]) -> float:
    lx = (left[0] + left[2]) / 2.0; ly = (left[1] + left[3]) / 2.0
    rx = (right[0] + right[2]) / 2.0; ry = (right[1] + right[3]) / 2.0
    return ((lx - rx) ** 2 + (ly - ry) ** 2) ** 0.5


def load_p226(path: Path, allowed: set[str], min_score: float, topk: int) -> dict[str, list[dict[str, Any]]]:
    out = {rid: [] for rid in allowed}
    for row in read_jsonl(path):
        rid = str(row.get("row_id") or row.get("id"))
        if rid not in allowed:
            continue
        preds = []
        for pred in row.get("predicted_symbols") or []:
            if str(pred.get("label")) != "stair":
                continue
            score = float(pred.get("score", 0.0) or 0.0)
            if score < min_score:
                continue
            preds.append({
                "bbox": [float(v) for v in pred["bbox"]],
                "label": "stair",
                "score": score,
                "source": "p226_stair_specialist_added",
                "tile_id": pred.get("tile_id"),
            })
        preds.sort(key=lambda item: item["score"], reverse=True)
        out[rid] = preds[:topk]
    return out


def conflict(cand: dict[str, Any], existing: list[dict[str, Any]], max_iou: float, min_dist: float, same_label_only: bool) -> bool:
    cbox = cand["bbox"]
    for pred in existing:
        label = str(pred.get("label") or pred.get("symbol_type"))
        if same_label_only and label != "stair":
            continue
        pbox = [float(v) for v in pred["bbox"]]
        if bbox_iou(cbox, pbox) >= max_iou:
            return True
        if center_dist(cbox, pbox) <= min_dist:
            return True
    return False


def fuse(core: dict[str, list[dict[str, Any]]], p226: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for rid, base_preds in core.items():
        merged = [dict(pred) for pred in base_preds]
        additions = []
        for pred in p226.get(rid, []):
            if pred["score"] < policy["threshold"]:
                continue
            if conflict(pred, merged + additions, policy["max_iou_to_existing"], policy["min_center_dist"], policy["same_label_only"]):
                continue
            additions.append(dict(pred))
            if len(additions) >= policy["max_add_per_row"]:
                break
        out[rid] = merged + additions
    return out


def policies() -> list[dict[str, Any]]:
    out = []
    for threshold in [0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7]:
        for max_add in [1, 2, 3, 5, 8, 12]:
            for max_iou in [0.05, 0.10, 0.20, 0.35, 0.50]:
                for dist in [0.0, 4.0, 8.0, 16.0]:
                    for same_label_only in [True, False]:
                        out.append({
                            "name": f"p226_stair_t{threshold:g}_a{max_add}_iou{max_iou:g}_d{dist:g}_same{int(same_label_only)}",
                            "threshold": threshold,
                            "max_add_per_row": max_add,
                            "max_iou_to_existing": max_iou,
                            "min_center_dist": dist,
                            "same_label_only": same_label_only,
                        })
    return out


def build_overlay(rows: list[dict[str, Any]], fused: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for row in rows:
        rid = row_id(row)
        nr = dict(row)
        cands = []
        for idx, pred in enumerate(fused.get(rid, [])):
            label = str(pred.get("label") or pred.get("symbol_type") or "generic_symbol")
            cands.append({
                "id": f"{rid}_p226_symbol_{idx:05d}",
                "target_id": f"{rid}_p226_symbol_{idx:05d}",
                "symbol_type": label,
                "label": label,
                "bbox": [float(v) for v in pred["bbox"]],
                "confidence": float(pred.get("score", 1.0) or 0.0),
                "score": float(pred.get("score", 1.0) or 0.0),
                "source": pred.get("source", "p224a_core"),
                "metadata": {"tile_id": pred.get("tile_id"), "fusion_policy": policy["name"]},
            })
        nr["symbol_candidates"] = cands
        meta = dict(nr.get("metadata") or {})
        meta["p226_stair_policy"] = policy["name"]
        nr["metadata"] = meta
        out.append(nr)
    return out


def render(report: dict[str, Any]) -> str:
    bm = report["baseline_metrics"]; sm = report["selected_metrics"]; b = report["bootstrap_vs_p224a"]
    lines = [
        "# P226 Stair Fusion Eval", "",
        "| Variant | F1 | P | R | TP | Pred | Gold |", "|---|---:|---:|---:|---:|---:|---:|",
        f"| P224a | {bm['f1']:.6f} | {bm['precision']:.6f} | {bm['recall']:.6f} | {bm['tp']} | {bm['predicted']} | {bm['gold']} |",
        f"| P226 fused | {sm['f1']:.6f} | {sm['precision']:.6f} | {sm['recall']:.6f} | {sm['tp']} | {sm['predicted']} | {sm['gold']} |", "",
        f"- Selected policy: `{report['selected_policy']['name']}`",
        f"- Additions: `{report['selected_added_predictions']}`",
        f"- ΔF1 CI: `{b['f1_delta']['ci95']}`",
        f"- ΔPrecision CI: `{b['precision_delta']['ci95']}`",
        f"- ΔRecall CI: `{b['recall_delta']['ci95']}`", "",
    ]
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(BASE))
    parser.add_argument("--p226", default=str(P226))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--md", default=str(MD))
    parser.add_argument("--overlay", default=str(OVERLAY))
    parser.add_argument("--min-score", type=float, default=0.001)
    parser.add_argument("--topk", type=int, default=1200)
    args = parser.parse_args()
    rows, core, golds = load_p206g(Path(args.base))
    ids = [row_id(row) for row in rows]
    p226 = load_p226(Path(args.p226), set(ids), args.min_score, args.topk)
    base_per = score_rows(core, golds, ids)
    baseline = metrics(base_per)
    grid = []
    best = None
    for policy in policies():
        fused = fuse(core, p226, policy)
        per = score_rows(fused, golds, ids)
        m = metrics(per)
        added = sum(max(0, len(fused[rid]) - len(core.get(rid, []))) for rid in ids)
        item = {"policy": policy, "metrics": m, "added_predictions": added}
        grid.append(item)
        precision_guard = m["precision"] >= baseline["precision"] - 0.005
        key = (precision_guard, m["f1"], m["precision"], m["recall"], -added)
        if best is None or key > best[0]:
            best = (key, item, fused, per)
    assert best is not None
    selected = best[1]
    boot = bootstrap(base_per, best[3], seed=2261)
    write_jsonl(Path(args.overlay), build_overlay(rows, best[2], selected["policy"]))
    grid.sort(key=lambda item: (item["metrics"]["f1"], item["metrics"]["precision"], item["metrics"]["recall"]), reverse=True)
    report = {
        "id": "P226_stair_fusion_eval",
        "baseline_metrics": baseline,
        "selected_policy": selected["policy"],
        "selected_metrics": selected["metrics"],
        "selected_added_predictions": selected["added_predictions"],
        "bootstrap_vs_p224a": boot,
        "top_grid": grid[:80],
        "inputs": {"base": rel(Path(args.base)), "p226": rel(Path(args.p226))},
        "outputs": {"overlay": rel(Path(args.overlay)), "report": rel(Path(args.report)), "markdown": rel(Path(args.md))},
        "claim_boundary": "Internal P101 policy sweep over P226 raster stair predictions; needs freeze/audit before promotion.",
    }
    write_json(Path(args.report), report)
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text(render(report), encoding="utf-8")
    print(json.dumps({"baseline": baseline, "selected_policy": selected["policy"], "selected_metrics": selected["metrics"], "added": selected["added_predictions"], "bootstrap": boot}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
