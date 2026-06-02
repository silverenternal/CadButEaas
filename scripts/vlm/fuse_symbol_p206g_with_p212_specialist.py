#!/usr/bin/env python3
"""Fuse P206g precision overlay with P212 FN-specialist page proposals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from fuse_symbol_p206g_with_p211_p212 import bbox_iou, load_p206g, score_predictions, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
P206G = ROOT / "reports/vlm/symbol_p206f_precision_repair_p206g_overlay.jsonl"
P212 = ROOT / "reports/vlm/symbol_fn_specialist_p212_pages_s192_top150_predictions.jsonl"
REPORT = ROOT / "reports/vlm/symbol_p206g_p212_specialist_fusion_eval.json"
OVERLAY = ROOT / "reports/vlm/symbol_p206g_p212_specialist_fusion_overlay.jsonl"


def load_p212(path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            out[str(row.get("row_id"))] = row.get("predicted_symbols") or []
    return out


def conflict(candidate: dict[str, Any], existing: list[dict[str, Any]], max_iou: float, min_dist: float, same_label_only: bool) -> bool:
    box = [float(v) for v in candidate["bbox"]]
    cx = (box[0] + box[2]) / 2.0; cy = (box[1] + box[3]) / 2.0
    label = str(candidate.get("label"))
    for pred in existing:
        if same_label_only and str(pred.get("label")) != label:
            continue
        other = [float(v) for v in pred["bbox"]]
        if bbox_iou(box, other) >= max_iou:
            return True
        ox = (other[0] + other[2]) / 2.0; oy = (other[1] + other[3]) / 2.0
        if ((cx - ox) ** 2 + (cy - oy) ** 2) ** 0.5 <= min_dist:
            return True
    return False


def fuse(core: dict[str, list[dict[str, Any]]], p212: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    labels = set(policy["allowed_labels"])
    thresholds = {str(k): float(v) for k, v in policy.get("label_thresholds", {}).items()}
    default_threshold = float(policy["threshold"])
    max_add = int(policy["max_add_per_row"])
    max_iou = float(policy["max_iou_to_core"])
    min_dist = float(policy["min_dist_to_core"])
    same_label_only = bool(policy.get("same_label_only", False))
    output = {}
    for row_id, core_preds in core.items():
        merged = [dict(pred) for pred in core_preds]
        additions = []
        candidates = sorted(p212.get(row_id, []), key=lambda pred: float(pred.get("score", 0.0)), reverse=True)
        for pred in candidates:
            label = str(pred.get("label") or "generic_symbol")
            if label not in labels:
                continue
            if float(pred.get("score", 0.0)) < thresholds.get(label, default_threshold):
                continue
            if conflict(pred, merged + additions, max_iou, min_dist, same_label_only):
                continue
            addition = dict(pred)
            addition["source"] = "p212_fn_specialist_added"
            additions.append(addition)
            if len(additions) >= max_add:
                break
        output[row_id] = merged + additions
    return output


def metric_key(row: dict[str, Any]) -> tuple[float, ...]:
    m = row["metrics"]["symbol_bbox_iou_0_30"]
    return (float(m["f1"]), float(m["precision"]), float(m["recall"]), -float(row["metrics"]["candidate_inflation"]))


def build_overlay(rows: list[dict[str, Any]], fused: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        row_id = str(row.get("id") or row.get("row_id"))
        new_row = dict(row)
        new_candidates = []
        for index, pred in enumerate(fused.get(row_id, [])):
            new_candidates.append({
                "id": f"{row_id}_p212f_symbol_{index:05d}",
                "target_id": f"{row_id}_p212f_symbol_{index:05d}",
                "symbol_type": pred.get("label"),
                "bbox": pred.get("bbox"),
                "confidence": pred.get("score"),
                "source": pred.get("source", "p206g"),
                "metadata": {"tile_id": pred.get("tile_id"), "fusion_policy": policy.get("name")},
            })
        new_row["symbol_candidates"] = new_candidates
        new_row["symbol_policy_overlay"] = {"policy_id": "p212_fn_specialist_fusion", "policy": policy}
        result.append(new_row)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p206g", default=str(P206G))
    parser.add_argument("--p212", default=str(P212))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--overlay", default=str(OVERLAY))
    parser.add_argument("--max-per-page", type=int, default=900)
    args = parser.parse_args()
    rows, core, golds = load_p206g(Path(args.p206g))
    p212 = load_p212(Path(args.p212))
    baseline, _ = score_predictions(core, golds, 0.0, 0.98, args.max_per_page, 0)
    policies=[]
    label_sets=[
        ["sink","shower","equipment"],
        ["sink","shower","equipment","stair"],
        ["sink","shower"],
        ["sink","equipment"],
        ["shower","equipment"],
    ]
    for labels in label_sets:
        for threshold in [0.48,0.5,0.52,0.55,0.58,0.6,0.62]:
            for max_add in [20,30,40,60]:
                for min_dist in [0,1,2,4]:
                    policies.append({
                        "name": f"p212f_{'-'.join(labels)}_t{threshold}_a{max_add}_d{min_dist}",
                        "allowed_labels": labels,
                        "threshold": threshold,
                        "max_add_per_row": max_add,
                        "max_iou_to_core": 0.25,
                        "min_dist_to_core": min_dist,
                        "same_label_only": False,
                        "label_thresholds": {},
                    })
    reports=[]
    for index, policy in enumerate(policies, start=1):
        fused = fuse(core, p212, policy)
        metrics, _ = score_predictions(fused, golds, 0.0, 0.98, args.max_per_page, 0)
        additions = sum(max(0, len(fused[row_id]) - len(core.get(row_id, []))) for row_id in fused)
        reports.append({"policy": policy, "metrics": metrics, "additions": additions})
        if index % 100 == 0:
            best=max(reports,key=metric_key)
            print(json.dumps({"done":index,"total":len(policies),"best_f1":best["metrics"]["symbol_bbox_iou_0_30"]["f1"],"additions":best["additions"]}), flush=True)
    reports.sort(key=metric_key, reverse=True)
    best=reports[0]
    fused=fuse(core,p212,best["policy"])
    _metrics,pred_rows=score_predictions(fused,golds,0.0,0.98,args.max_per_page,0)
    write_jsonl(Path(args.overlay), build_overlay(rows, fused, best["policy"]))
    result={
        "id":"P212_fn_specialist_fusion_grid",
        "claim_boundary":"P101/P206g split policy-search evidence only; needs bootstrap/frozen validation before paper claim.",
        "inputs":{"p206g":str(Path(args.p206g)),"p212":str(Path(args.p212))},
        "baseline":baseline,
        "selected":best,
        "top10":reports[:10],
        "outputs":{"overlay":str(Path(args.overlay)),"report":str(Path(args.report))},
    }
    write_json(Path(args.report), result)
    print(json.dumps({"baseline":baseline["symbol_bbox_iou_0_30"],"selected":best["metrics"]["symbol_bbox_iou_0_30"],"additions":best["additions"],"policy":best["policy"]},ensure_ascii=False,indent=2))


if __name__ == "__main__":
    main()
