#!/usr/bin/env python3
"""Build P225 candidate-selector dataset from P224 s384 page proposals over P224a baseline."""
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from fuse_symbol_p206g_with_p211_p212 import bbox_iou, load_p206g, write_json, write_jsonl

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/vlm/symbol_p224a_column_frozen_overlay.jsonl"
P224 = ROOT / "reports/vlm/symbol_p224_detector_pages_s384_predictions.jsonl"
OUT_JSONL = ROOT / "reports/vlm/symbol_p225_p224_selector_dataset.jsonl"
OUT_REPORT = ROOT / "reports/vlm/symbol_p225_p224_selector_dataset_report.json"
OUT_MD = ROOT / "reports/vlm/symbol_p225_p224_selector_dataset_report.md"
LABELS = ["appliance", "bathtub", "column", "equipment", "generic_symbol", "shower", "sink", "stair", "table"]


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def center(box: list[float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def dist(a: list[float], b: list[float]) -> float:
    ax, ay = center(a); bx, by = center(b)
    return ((ax-bx)**2 + (ay-by)**2) ** 0.5


def row_id(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("row_id"))


def pred_label(pred: dict[str, Any]) -> str:
    return str(pred.get("label") or pred.get("symbol_type") or "generic_symbol")


def pred_score(pred: dict[str, Any]) -> float:
    return float(pred.get("score") or pred.get("confidence") or 0.0)


def load_p224(path: Path, min_score: float, topk: int) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for row in read_jsonl(path):
        rid = str(row.get("row_id"))
        preds = []
        for pred in row.get("predicted_symbols") or []:
            score = float(pred.get("score", 0.0))
            if score < min_score or "bbox" not in pred:
                continue
            preds.append({
                "bbox": [float(v) for v in pred["bbox"]],
                "label": str(pred.get("label") or "generic_symbol"),
                "score": score,
                "tile_id": pred.get("tile_id"),
                "source": pred.get("source") or "p224_s384",
            })
        preds.sort(key=lambda item: item["score"], reverse=True)
        out[rid] = preds[:topk]
    return out


def matched_base_gold_indices(base_preds: list[dict[str, Any]], golds: list[dict[str, Any]]) -> set[int]:
    pairs = []
    for pi, pred in enumerate(base_preds):
        pbox = [float(v) for v in pred["bbox"]]
        plabel = pred_label(pred)
        for gi, gold in enumerate(golds):
            if plabel != str(gold["label"]):
                continue
            iou = bbox_iou(pbox, [float(v) for v in gold["bbox"]])
            if iou >= 0.30:
                pairs.append((iou, pi, gi))
    used_p, used_g = set(), set()
    for iou, pi, gi in sorted(pairs, reverse=True):
        if pi in used_p or gi in used_g:
            continue
        used_p.add(pi); used_g.add(gi)
    return used_g


def build(args: argparse.Namespace) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows, core, golds_by_row = load_p206g(Path(args.base))
    ids = [row_id(row) for row in rows]
    row_sizes = {row_id(row): row.get("image_size") or [1, 1] for row in rows}
    p224 = load_p224(Path(args.p224), args.min_score, args.topk)
    examples = []
    totals = Counter()
    by_label = defaultdict(Counter)
    by_score_bin = defaultdict(Counter)
    for rid in ids:
        width, height = row_sizes.get(rid, [1, 1])
        width = max(float(width), 1.0); height = max(float(height), 1.0)
        diag = math.sqrt(width * height)
        base_preds = core.get(rid, [])
        golds = list(golds_by_row[rid].values())
        base_matched = matched_base_gold_indices(base_preds, golds)
        for rank, pred in enumerate(p224.get(rid, [])):
            label = pred_label(pred)
            box = [float(v) for v in pred["bbox"]]
            score = pred_score(pred)
            best_typed_iou = 0.0; best_typed_index = None
            best_any_iou = 0.0; best_any_label = ""
            for gi, gold in enumerate(golds):
                gbox = [float(v) for v in gold["bbox"]]
                iou = bbox_iou(box, gbox)
                if iou > best_any_iou:
                    best_any_iou = iou; best_any_label = str(gold["label"])
                if str(gold["label"]) == label and iou > best_typed_iou:
                    best_typed_iou = iou; best_typed_index = gi
            same_core = [item for item in base_preds if pred_label(item) == label]
            nearest_same_iou = max([bbox_iou(box, [float(v) for v in item["bbox"]]) for item in same_core] or [0.0])
            nearest_any_iou = max([bbox_iou(box, [float(v) for v in item["bbox"]]) for item in base_preds] or [0.0])
            nearest_same_dist = min([dist(box, [float(v) for v in item["bbox"]]) for item in same_core] or [9999.0])
            nearest_any_dist = min([dist(box, [float(v) for v in item["bbox"]]) for item in base_preds] or [9999.0])
            box_area = area(box)
            cx, cy = center(box)
            target_typed_tp = int(best_typed_iou >= 0.30)
            target_new_tp = int(target_typed_tp and best_typed_index is not None and best_typed_index not in base_matched)
            score_bin = "ge_0.9" if score >= 0.9 else "ge_0.7" if score >= 0.7 else "ge_0.5" if score >= 0.5 else "ge_0.3" if score >= 0.3 else "lt_0.3"
            row = {
                "row_id": rid,
                "rank": rank,
                "label": label,
                "bbox": box,
                "score": score,
                "tile_id": pred.get("tile_id"),
                "features": {
                    "score": score,
                    "score_logit": math.log(max(score, 1e-6) / max(1.0 - score, 1e-6)),
                    "rank_norm": rank / max(args.topk, 1),
                    "area_norm": box_area / (width * height),
                    "sqrt_area_norm": math.sqrt(max(box_area, 0.0)) / max(diag, 1e-6),
                    "aspect": (box[2] - box[0]) / max(box[3] - box[1], 1e-6),
                    "cx_norm": cx / width,
                    "cy_norm": cy / height,
                    "nearest_same_iou": nearest_same_iou,
                    "nearest_any_iou": nearest_any_iou,
                    "nearest_same_dist_norm": min(nearest_same_dist, 9999.0) / max(diag, 1e-6),
                    "nearest_any_dist_norm": min(nearest_any_dist, 9999.0) / max(diag, 1e-6),
                    "same_label_core_count": float(len(same_core)),
                    "core_count": float(len(base_preds)),
                    **{f"label_is_{name}": float(label == name) for name in LABELS},
                },
                "target_typed_tp": target_typed_tp,
                "target_new_tp": target_new_tp,
                "best_typed_iou": round(best_typed_iou, 6),
                "best_any_iou": round(best_any_iou, 6),
                "best_any_label": best_any_label,
            }
            examples.append(row)
            totals.update({"candidates": 1, "typed_tp": target_typed_tp, "new_tp": target_new_tp})
            by_label[label].update({"candidates": 1, "typed_tp": target_typed_tp, "new_tp": target_new_tp})
            by_score_bin[score_bin].update({"candidates": 1, "typed_tp": target_typed_tp, "new_tp": target_new_tp})
    report = {
        "id": "P225_p224_selector_dataset",
        "base": str(Path(args.base)),
        "p224": str(Path(args.p224)),
        "config": vars(args),
        "totals": dict(totals),
        "by_label": {label: dict(counter) | {"new_tp_rate": round(counter["new_tp"] / max(counter["candidates"], 1), 6), "typed_tp_rate": round(counter["typed_tp"] / max(counter["candidates"], 1), 6)} for label, counter in sorted(by_label.items())},
        "by_score_bin": {
            key: dict(counter)
            | {
                "new_tp_rate": round(counter["new_tp"] / max(counter["candidates"], 1), 6),
                "typed_tp_rate": round(counter["typed_tp"] / max(counter["candidates"], 1), 6),
            }
            for key, counter in sorted(by_score_bin.items())
        },
        "claim_boundary": "Offline selector training dataset; gold labels used only to assign supervised targets/evaluate, not runtime features.",
    }
    return examples, report


def render(report: dict[str, Any]) -> str:
    t = report["totals"]
    lines = ["# P225 P224 Proposal Selector Dataset", "", "## Totals", f"- Candidates: `{t.get('candidates',0)}`", f"- Typed TP candidates: `{t.get('typed_tp',0)}`", f"- New TP candidates over P224a: `{t.get('new_tp',0)}`", "", "## By Label", "| Label | Candidates | Typed TP | Typed Rate | New TP | New Rate |", "|---|---:|---:|---:|---:|---:|"]
    for label, row in sorted(report["by_label"].items(), key=lambda item: -item[1].get("new_tp", 0)):
        lines.append(f"| {label} | {row.get('candidates',0)} | {row.get('typed_tp',0)} | {row.get('typed_tp_rate',0):.6f} | {row.get('new_tp',0)} | {row.get('new_tp_rate',0):.6f} |")
    lines += ["", "## Interpretation", "- P224 s384 provides many covering candidates but extremely low raw precision; this dataset is for a learned selector, not direct fusion.", ""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default=str(BASE))
    ap.add_argument("--p224", default=str(P224))
    ap.add_argument("--output", default=str(OUT_JSONL))
    ap.add_argument("--report", default=str(OUT_REPORT))
    ap.add_argument("--md", default=str(OUT_MD))
    ap.add_argument("--min-score", type=float, default=0.001)
    ap.add_argument("--topk", type=int, default=1200)
    args = ap.parse_args()
    examples, report = build(args)
    write_jsonl(Path(args.output), examples)
    report["outputs"] = {"dataset": str(Path(args.output)), "report": str(Path(args.report)), "markdown": str(Path(args.md))}
    write_json(Path(args.report), report)
    Path(args.md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.md).write_text(render(report), encoding="utf-8")
    print(json.dumps({"totals": report["totals"], "by_label": report["by_label"], "outputs": report["outputs"]}, ensure_ascii=False, indent=2)[:6000])


if __name__ == "__main__":
    main()
