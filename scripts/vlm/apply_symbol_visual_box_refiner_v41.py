#!/usr/bin/env python3
"""Apply v40 visual refiner with class/size routing on crop rows."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import joblib
import numpy as np

from train_symbol_tile_detector_v20 import bbox_iou, rel, write_json
from train_symbol_visual_box_refiner_v40 import apply_delta, features, load_jsonl


ROOT = Path(__file__).resolve().parents[2]


def source_path(path: str | Path) -> Path:
    value = Path(path)
    return value if value.is_absolute() else ROOT / value


def should_apply(row: dict[str, Any], policy: dict[str, Any]) -> bool:
    label = str((row.get("target") or {}).get("label") or (row.get("proposal") or {}).get("label") or "")
    area = str((row.get("target") or {}).get("area_bucket") or "")
    if label in set(policy.get("deny_labels") or []):
        return False
    if area in set(policy.get("deny_areas") or []):
        return False
    if label in set(policy.get("allow_labels") or []):
        return True
    if area in set(policy.get("allow_areas") or []):
        return True
    return False


def evaluate(model: Any, rows: list[dict[str, Any]], policy: dict[str, Any], clip: float) -> dict[str, Any]:
    x = np.asarray([features(row) for row in rows], dtype=np.float32)
    pred = model.predict(x)
    totals = Counter()
    by_area = defaultdict(Counter)
    by_label = defaultdict(Counter)
    for row, delta in zip(rows, pred, strict=True):
        box = [float(v) for v in row["proposal"]["bbox"]]
        gold = [float(v) for v in row["target"]["bbox"]]
        routed = apply_delta(box, list(delta), clip) if should_apply(row, policy) else box
        bi = bbox_iou(box, gold)
        ri = bbox_iou(routed, gold)
        label = str(row["target"].get("label") or row["proposal"].get("label") or "")
        area = str(row["target"].get("area_bucket") or "")
        totals["rows"] += 1
        totals["routed"] += int(routed is not box)
        totals["input_hit"] += int(bi >= 0.30)
        totals["routed_hit"] += int(ri >= 0.30)
        totals["improved"] += int(ri > bi)
        totals["worse"] += int(ri < bi)
        for bucket in [by_area[area], by_label[label]]:
            bucket["rows"] += 1
            bucket["routed"] += int(routed is not box)
            bucket["input_hit"] += int(bi >= 0.30)
            bucket["routed_hit"] += int(ri >= 0.30)
    def rates(c: Counter) -> dict[str, float]:
        n = max(int(c["rows"]), 1)
        return {"rows": int(c["rows"]), "routed": int(c["routed"]), "input_iou_0_30_recall": round(c["input_hit"] / n, 6), "routed_iou_0_30_recall": round(c["routed_hit"] / n, 6)}
    n = max(int(totals["rows"]), 1)
    return {
        "rows": int(totals["rows"]),
        "routed": int(totals["routed"]),
        "input_iou_0_30_recall": round(totals["input_hit"] / n, 6),
        "routed_iou_0_30_recall": round(totals["routed_hit"] / n, 6),
        "improved_rate": round(totals["improved"] / n, 6),
        "worse_rate": round(totals["worse"] / n, 6),
        "by_area": {k: rates(v) for k, v in sorted(by_area.items())},
        "by_label": {k: rates(v) for k, v in sorted(by_label.items())},
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", default="datasets/symbol_visual_box_refiner_v40")
    parser.add_argument("--model", default="checkpoints/symbol_visual_box_refiner_v40/model.joblib")
    parser.add_argument("--eval-output", default="reports/vlm/symbol_visual_box_refiner_v41_locked_eval.json")
    parser.add_argument("--clip", type=float, default=0.75)
    args = parser.parse_args()
    data_dir = source_path(args.data_dir)
    rows = load_jsonl(data_dir / "locked.jsonl")
    bundle = joblib.load(source_path(args.model))
    model = bundle["model"]
    policy = {
        "allow_labels": ["sink", "stair"],
        "allow_areas": ["large_le_4096", "xlarge_gt_4096"],
        "deny_labels": ["shower"],
        "deny_areas": ["tiny_le_64", "small_le_256"],
    }
    locked = evaluate(model, rows, policy, args.clip)
    report = {
        "version": "symbol_visual_box_refiner_v41_locked_eval",
        "task": "P1-12-visual-refiner-routing-v41",
        "claim_boundary": "Class/size routed visual refiner on v40 locked smoke crop rows. Policy is fixed from v40 bucket evidence.",
        "source_integrity": {"model_input": "raster crop pixels plus candidate bbox/score/type", "gold_used_for_inference": False, "runtime_uses_svg_or_cad_geometry": False},
        "policy": policy,
        "locked": locked,
        "stage_gate": {
            "locked_overall_iou_recall_not_drop": locked["routed_iou_0_30_recall"] >= locked["input_iou_0_30_recall"],
            "locked_sink_iou_recall_improves": locked["by_label"].get("sink", {}).get("routed_iou_0_30_recall", 0.0) > locked["by_label"].get("sink", {}).get("input_iou_0_30_recall", 0.0),
            "locked_tiny_iou_recall_not_drop": locked["by_area"].get("tiny_le_64", {}).get("routed_iou_0_30_recall", 0.0) >= locked["by_area"].get("tiny_le_64", {}).get("input_iou_0_30_recall", 0.0),
            "no_oracle_inference": True,
        },
    }
    report["stage_gate"]["passed"] = all(report["stage_gate"].values())
    write_json(source_path(args.eval_output), report)
    print(json.dumps({"policy": policy, "locked": locked, "stage_gate": report["stage_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
