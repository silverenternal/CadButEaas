#!/usr/bin/env python3
"""P199 learned-ish symbol box refiner using page-held-out grouped median box deltas.

Gold targets are used only to train/evaluate the refiner offline. Runtime features are
prediction label, bucket, score, and box geometry; no gold/vector features are used
when applying a trained policy to candidates.
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import median
from typing import Any

import sweep_symbol_disagreement_backfill_p165 as p165

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "reports/vlm/symbol_box_refiner_p197b_over_p196c_best_overlay.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def target_symbols(row: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for idx, item in enumerate((row.get("targets") or {}).get("symbol") or []):
        box = p165.bbox4(item.get("bbox"))
        if box is None:
            continue
        label = str(item.get("semantic_type") or item.get("symbol_type") or item.get("label") or "generic_symbol")
        out.append({"id": str(item.get("target_id") or idx), "bbox": box, "bucket": p165.bucket(box), "label": label})
    return out


def clamp(box: list[float], image_size: Any) -> list[float]:
    width, height = 4096.0, 4096.0
    if isinstance(image_size, list) and len(image_size) >= 2:
        width, height = float(image_size[0]), float(image_size[1])
    elif isinstance(image_size, dict):
        width = float(image_size.get("width") or image_size.get("w") or width)
        height = float(image_size.get("height") or image_size.get("h") or height)
    x1, y1, x2, y2 = box
    x1 = max(0.0, min(width - 1.0, x1))
    y1 = max(0.0, min(height - 1.0, y1))
    x2 = max(x1 + 1.0, min(width, x2))
    y2 = max(y1 + 1.0, min(height, y2))
    return [round(x1, 6), round(y1, 6), round(x2, 6), round(y2, 6)]


def key_chain(pred: dict[str, Any]) -> list[str]:
    label = str(pred.get("label") or "generic_symbol")
    bucket = str(pred.get("bucket") or p165.bucket(pred["bbox"]))
    score = float(pred.get("score") or 0.0)
    if score >= 0.75:
        score_bin = "hi"
    elif score >= 0.45:
        score_bin = "mid"
    else:
        score_bin = "lo"
    return [f"label_bucket_score:{label}|{bucket}|{score_bin}", f"label_bucket:{label}|{bucket}", f"label:{label}", f"bucket:{bucket}", "global"]


def box_delta(pred_box: list[float], gold_box: list[float]) -> list[float]:
    pcx, pcy = (pred_box[0] + pred_box[2]) / 2.0, (pred_box[1] + pred_box[3]) / 2.0
    gcx, gcy = (gold_box[0] + gold_box[2]) / 2.0, (gold_box[1] + gold_box[3]) / 2.0
    pw, ph = max(1.0, pred_box[2] - pred_box[0]), max(1.0, pred_box[3] - pred_box[1])
    gw, gh = max(1.0, gold_box[2] - gold_box[0]), max(1.0, gold_box[3] - gold_box[1])
    return [(gcx - pcx) / pw, (gcy - pcy) / ph, math.log(gw / pw), math.log(gh / ph)]


def apply_delta(box: list[float], delta: list[float], alpha: float, image_size: Any) -> list[float]:
    cx, cy = (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0
    w, h = max(1.0, box[2] - box[0]), max(1.0, box[3] - box[1])
    dx, dy, dw, dh = delta
    ncx = cx + alpha * dx * w
    ncy = cy + alpha * dy * h
    nw = max(1.0, w * math.exp(alpha * dw))
    nh = max(1.0, h * math.exp(alpha * dh))
    return clamp([ncx - nw / 2.0, ncy - nh / 2.0, ncx + nw / 2.0, ncy + nh / 2.0], image_size)


def matched_training_examples(row: dict[str, Any], max_center_dist: float) -> list[dict[str, Any]]:
    preds = p165.normalized(row.get("symbol_candidates") or [], "p199_train")
    golds = target_symbols(row)
    used_pred: set[int] = set()
    examples = []
    for gold in golds:
        best_idx = None
        best_rank = None
        for idx, pred in enumerate(preds):
            if idx in used_pred:
                continue
            same_label = 1 if str(pred["label"]) == str(gold["label"]) else 0
            center_ok = p165.center_covered(pred["bbox"], gold["bbox"])
            dist = p165.center_distance(pred["bbox"], gold["bbox"])
            overlap = p165.iou(pred["bbox"], gold["bbox"])
            if not center_ok and dist > max_center_dist and overlap < 0.10:
                continue
            rank = (same_label, center_ok, overlap, -dist)
            if best_rank is None or rank > best_rank:
                best_rank = rank
                best_idx = idx
        if best_idx is None:
            continue
        pred = preds[best_idx]
        used_pred.add(best_idx)
        examples.append({"row_id": str(row.get("row_id") or row.get("id")), "pred": pred, "gold": gold, "delta": box_delta(pred["bbox"], gold["bbox"])})
    return examples


def train_model(rows: list[dict[str, Any]], exclude_row: str | None, max_center_dist: float, min_group: int) -> dict[str, Any]:
    buckets: dict[str, list[list[float]]] = defaultdict(list)
    count_examples = 0
    for row in rows:
        row_id = str(row.get("row_id") or row.get("id"))
        if exclude_row is not None and row_id == exclude_row:
            continue
        for ex in matched_training_examples(row, max_center_dist):
            count_examples += 1
            for key in key_chain(ex["pred"]):
                buckets[key].append(ex["delta"])
    model = {"min_group": min_group, "count_examples": count_examples, "groups": {}}
    for key, values in buckets.items():
        if len(values) < (1 if key == "global" else min_group):
            continue
        model["groups"][key] = {"n": len(values), "delta": [round(median([v[i] for v in values]), 6) for i in range(4)]}
    return model


def predict_delta(pred: dict[str, Any], model: dict[str, Any]) -> tuple[list[float] | None, str | None, int]:
    groups = model.get("groups") or {}
    for key in key_chain(pred):
        if key in groups:
            item = groups[key]
            return list(item["delta"]), key, int(item.get("n") or 0)
    return None, None, 0


def apply_model(rows: list[dict[str, Any]], model: dict[str, Any], alpha: float, only_labels: set[str], only_buckets: set[str]) -> dict[str, list[dict[str, Any]]]:
    out = {}
    for row in rows:
        row_id = str(row.get("row_id") or row.get("id"))
        preds = p165.normalized(row.get("symbol_candidates") or [], "p199_apply")
        refined = []
        for pred in preds:
            item = copy.deepcopy(pred)
            if only_labels and str(item["label"]) not in only_labels:
                refined.append(item); continue
            if only_buckets and str(item["bucket"]) not in only_buckets:
                refined.append(item); continue
            delta, source_key, source_n = predict_delta(item, model)
            if delta is not None:
                item["bbox"] = apply_delta(item["bbox"], delta, alpha, row.get("image_size"))
                raw = copy.deepcopy(item.get("raw") or {})
                raw["bbox"] = item["bbox"]
                raw.setdefault("metadata", {})["p199_refiner"] = {"alpha": alpha, "source_key": source_key, "source_n": source_n}
                item["raw"] = raw
            refined.append(item)
        out[row_id] = refined
    return out


def materialize(rows: list[dict[str, Any]], preds_by_row: dict[str, list[dict[str, Any]]], policy: dict[str, Any]) -> list[dict[str, Any]]:
    out = []
    for raw_row in rows:
        row = copy.deepcopy(raw_row)
        row_id = str(row.get("row_id") or row.get("id"))
        candidates = []
        for idx, pred in enumerate(preds_by_row.get(row_id, [])):
            item = copy.deepcopy(pred.get("raw") or {})
            item["bbox"] = pred["bbox"]
            item["symbol_type"] = pred["label"]
            item["confidence"] = float(pred["score"])
            item["id"] = f"{row_id}_p199_symbol_{idx:05d}"
            item["target_id"] = item["id"]
            item["source"] = "symbol_learned_box_refiner_p199"
            item.setdefault("metadata", {})["p199_policy"] = policy
            candidates.append(item)
        row["symbol_candidates"] = candidates
        if isinstance(row.get("expected_json"), dict):
            row["expected_json"]["symbol_candidates"] = [copy.deepcopy(item) for item in candidates]
        row["symbol_policy_overlay"] = {"policy_id": "p199_learned_box_refiner", "description": "P199 grouped median learned box delta refiner", "policy": policy}
        out.append(row)
    return out


def render(report: dict[str, Any]) -> str:
    lines = [
        "# P199 Learned Box Refiner",
        "",
        "Decision: **" + str(report["decision"]) + "**",
        "",
        "## Metrics",
        "",
        "| Variant | Precision | Recall | F1 | Center | Inflation |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name, metrics in report["metrics"].items():
        lines.append(
            "| `" + str(name) + "` | "
            + f"{metrics['precision']:.6f} | {metrics['recall']:.6f} | {metrics['f1']:.6f} | "
            + f"{metrics['center_recall']:.6f} | {metrics['prediction_inflation']:.6f} |"
        )
    lines += ["", "## Best Policy", "", "```json", json.dumps(report["best_policy"], ensure_ascii=False, indent=2), "```", "", "## Notes", ""]
    for note in report["notes"]:
        lines.append(f"- {note}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-overlay", default=str(DEFAULT_BASE))
    parser.add_argument("--out-json", default="configs/vlm/symbol_learned_box_refiner_p199.json")
    parser.add_argument("--out-md", default="reports/vlm/symbol_learned_box_refiner_p199.md")
    parser.add_argument("--out-overlay", default="reports/vlm/symbol_learned_box_refiner_p199_overlay.jsonl")
    parser.add_argument("--max-center-dist", type=float, default=40.0)
    parser.add_argument("--min-group", type=int, default=8)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.base_overlay))
    golds = {str(row.get("row_id") or row.get("id")): target_symbols(row) for row in rows}
    baseline_preds = {str(row.get("row_id") or row.get("id")): p165.normalized(row.get("symbol_candidates") or [], "baseline") for row in rows}
    baseline = p165.evaluate(golds, baseline_preds)
    labels_options = [set(), {"sink", "shower"}, {"sink", "shower", "equipment", "appliance", "stair"}]
    bucket_options = [set(), {"tiny", "small"}, {"tiny", "small", "medium"}]
    alphas = [0.25, 0.5, 0.75, 1.0]

    # Leave-one-page-out policy selection evidence.
    lopo_scores = []
    row_ids = [str(row.get("row_id") or row.get("id")) for row in rows]
    for alpha in alphas:
        for labels in labels_options:
            for buckets in bucket_options:
                pred_all = {}
                for row in rows:
                    row_id = str(row.get("row_id") or row.get("id"))
                    model = train_model(rows, row_id, args.max_center_dist, args.min_group)
                    pred_all.update(apply_model([row], model, alpha, labels, buckets))
                metrics = p165.evaluate(golds, pred_all)
                lopo_scores.append({"alpha": alpha, "labels": sorted(labels), "buckets": sorted(buckets), "metrics": metrics})
    lopo_scores.sort(key=lambda x: (x["metrics"]["f1"], x["metrics"]["recall"], x["metrics"]["precision"]), reverse=True)
    best_lopo = lopo_scores[0]

    # Full-fit materialized exploratory candidate using best LOPO policy.
    full_model = train_model(rows, None, args.max_center_dist, args.min_group)
    best_policy = {"alpha": best_lopo["alpha"], "labels": best_lopo["labels"], "buckets": best_lopo["buckets"], "max_center_dist": args.max_center_dist, "min_group": args.min_group, "model_type": "grouped_median_box_delta", "selection": "leave_one_page_out_best_policy"}
    full_preds = apply_model(rows, full_model, float(best_policy["alpha"]), set(best_policy["labels"]), set(best_policy["buckets"]))
    full_metrics = p165.evaluate(golds, full_preds)
    write_jsonl(Path(args.out_overlay), materialize(rows, full_preds, best_policy))
    report = {
        "id": "P199_learned_symbol_box_refiner",
        "created_on": "2026-05-19",
        "decision": "promote_candidate" if full_metrics["f1"] > baseline["f1"] else "no_promotion_keep_baseline",
        "claim_boundary": "Offline supervised box-refiner experiment. LOPO is planning evidence; full-fit overlay still requires independent validation before paper claim.",
        "inputs": {"base_overlay": args.base_overlay},
        "training_examples_full": full_model["count_examples"],
        "group_count_full": len(full_model.get("groups") or {}),
        "best_policy": best_policy,
        "metrics": {"baseline": baseline, "lopo_best": best_lopo["metrics"], "full_fit_overlay": full_metrics},
        "delta_full_vs_baseline": p165.delta(full_metrics, baseline),
        "top_lopo_candidates": lopo_scores[:20],
        "model": full_model,
        "notes": [
            "P199 uses grouped median dx/dy/dw/dh deltas as a lightweight learned refiner without external sklearn dependency.",
            "Runtime application uses only candidate label, bucket, score, and geometry; gold is used only for offline supervised training/evaluation.",
            "If full-fit improves but LOPO does not, treat it as overfit diagnostic and do not promote for paper claim."
        ],
        "outputs": {"json": args.out_json, "md": args.out_md, "overlay": args.out_overlay},
    }
    write_json(Path(args.out_json), report)
    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text(render(report), encoding="utf-8")
    print(json.dumps({"decision": report["decision"], "metrics": report["metrics"], "delta_full_vs_baseline": report["delta_full_vs_baseline"], "best_policy": best_policy}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
