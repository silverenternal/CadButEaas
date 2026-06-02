#!/usr/bin/env python3
"""Mine P221a sink tiny residual cases after frozen P217/P218."""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path

from fuse_symbol_p206g_with_p211_p212 import area_bucket, bbox_iou, load_p206g

ROOT = Path(__file__).resolve().parents[2]
P217 = ROOT / "reports/vlm/symbol_p218_p217_frozen_overlay.jsonl"
OUT_JSONL = ROOT / "reports/vlm/symbol_p221a_sink_tiny_residual_cases.jsonl"
OUT_JSON = ROOT / "reports/vlm/symbol_p221a_sink_tiny_residual_summary.json"
OUT_MD = ROOT / "reports/vlm/symbol_p221a_sink_tiny_residual_mining.md"


def match_label_aware(preds, golds):
    candidates = []
    for pi, pred in enumerate(preds):
        pred_label = str(pred.get("label", pred.get("symbol_type", "unknown")))
        pred_box = [float(v) for v in pred["bbox"]]
        for gi, gold in enumerate(golds):
            if pred_label != str(gold.get("label", "unknown")):
                continue
            iou = bbox_iou(pred_box, [float(v) for v in gold["bbox"]])
            if iou >= 0.30:
                candidates.append((iou, pi, gi))
    used_p, used_g = set(), set()
    for iou, pi, gi in sorted(candidates, reverse=True):
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi)
    return used_p, used_g


def center(box):
    return ((box[0]+box[2])/2, (box[1]+box[3])/2)


def distance(a, b):
    ax, ay = center(a); bx, by = center(b)
    return ((ax-bx)**2 + (ay-by)**2) ** 0.5


def main() -> None:
    rows, preds_by_row, golds_by_row = load_p206g(P217)
    cases = []
    by_row = Counter()
    nearest_label = Counter()
    nearest_dist_bucket = Counter()
    row_image_sizes = {}
    for row in rows:
        row_id = str(row.get("id") or row.get("row_id"))
        image_size = row.get("image_size") or row.get("metadata", {}).get("image_size")
        row_image_sizes[row_id] = image_size
        preds = preds_by_row[row_id]
        golds = list(golds_by_row[row_id].values())
        _used_p, used_g = match_label_aware(preds, golds)
        for gi, gold in enumerate(golds):
            if gi in used_g:
                continue
            label = str(gold.get("label", "unknown"))
            box = [float(v) for v in gold["bbox"]]
            bucket = area_bucket(box)
            if not (label == "sink" and bucket == "tiny_le_64"):
                continue
            nearest = None
            for pred in preds:
                pred_box = [float(v) for v in pred["bbox"]]
                item = {
                    "label": str(pred.get("label", pred.get("symbol_type", "unknown"))),
                    "source": pred.get("source"),
                    "score": float(pred.get("score", pred.get("confidence", 0.0)) or 0.0),
                    "iou": bbox_iou(pred_box, box),
                    "center_distance": distance(pred_box, box),
                    "bbox": pred_box,
                }
                if nearest is None or item["center_distance"] < nearest["center_distance"]:
                    nearest = item
            if nearest is None:
                nearest = {"label": "none", "source": None, "score": 0.0, "iou": 0.0, "center_distance": None, "bbox": None}
            dist = nearest["center_distance"]
            if dist is None:
                dist_bucket = "no_prediction"
            elif dist <= 8:
                dist_bucket = "near_le_8"
            elif dist <= 16:
                dist_bucket = "near_le_16"
            elif dist <= 32:
                dist_bucket = "near_le_32"
            elif dist <= 64:
                dist_bucket = "near_le_64"
            else:
                dist_bucket = "far_gt_64"
            case = {
                "row_id": row_id,
                "target_id": gold.get("target_id"),
                "label": label,
                "bucket": bucket,
                "bbox": box,
                "image_size": image_size,
                "nearest_prediction": nearest,
                "dist_bucket": dist_bucket,
            }
            cases.append(case)
            by_row[row_id] += 1
            nearest_label[nearest["label"]] += 1
            nearest_dist_bucket[dist_bucket] += 1
    OUT_JSONL.write_text("".join(json.dumps(c, ensure_ascii=False)+"\n" for c in cases), encoding="utf-8")
    summary = {
        "id": "P221a_sink_tiny_residual_mining",
        "source_overlay": str(P217.relative_to(ROOT)),
        "case_count": len(cases),
        "rows_with_cases": len(by_row),
        "by_row_top20": dict(by_row.most_common(20)),
        "nearest_prediction_label": dict(nearest_label.most_common()),
        "nearest_distance_bucket": dict(nearest_dist_bucket.most_common()),
        "examples": cases[:50],
        "claim_boundary": "Gold used for offline residual mining only; P221a runtime branch must use raster/candidate features only.",
    }
    OUT_JSON.write_text(json.dumps(summary, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    lines = [
        "# P221a Sink-Tiny Residual Mining",
        "",
        "## Scope",
        "- Mine missed `sink` targets in `tiny_le_64` after frozen P217/P218.",
        "- This is offline mining only; it is not a runtime feature source.",
        "",
        "## Counts",
        f"- Cases: {len(cases)}",
        f"- Rows with cases: {len(by_row)}",
        f"- Worst rows: `{json.dumps(dict(by_row.most_common(10)), ensure_ascii=False)}`",
        "",
        "## Nearest Prediction Context",
        f"- Nearest prediction labels: `{json.dumps(dict(nearest_label.most_common()), ensure_ascii=False)}`",
        f"- Nearest distance buckets: `{json.dumps(dict(nearest_dist_bucket.most_common()), ensure_ascii=False)}`",
        "",
        "## P221a Implementation Implication",
        "- If many misses have a nearby non-sink prediction, prioritize verifier/relabel calibration.",
        "- If many misses are far from all predictions, prioritize a tiny-sink proposal generator/detector.",
        "- Fuse only after paired bootstrap vs frozen P217/P218 and reject negative precision CI.",
    ]
    OUT_MD.write_text("\n".join(lines)+"\n", encoding="utf-8")
    print(json.dumps({"cases": len(cases), "rows": len(by_row), "report": str(OUT_MD), "nearest_label": dict(nearest_label.most_common()), "dist": dict(nearest_dist_bucket.most_common())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
