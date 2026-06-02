#!/usr/bin/env python3
"""Attribute P116 symbol subset failures without importing inference packages."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_EVAL = ROOT / "reports/vlm/full_public_raster_symbol_eval_subset_p113_p116_subset_20260516_142010.json"
DEFAULT_PREDICTIONS = ROOT / "reports/vlm/full_public_raster_symbol_eval_subset_p113_p116_subset_20260516_142010_predictions.jsonl"
DEFAULT_JSON = ROOT / "configs/vlm/symbol_subset_failure_attribution_p117.json"
DEFAULT_REPORT = ROOT / "reports/vlm/symbol_subset_failure_attribution_p117.md"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def bbox_area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def area_bucket(box: list[float]) -> str:
    area = bbox_area(box)
    if area <= 64:
        return "tiny_le_64"
    if area <= 256:
        return "small_le_256"
    if area <= 1024:
        return "medium_le_1024"
    if area <= 4096:
        return "large_le_4096"
    return "xlarge_gt_4096"


def bbox_iou(left: list[float], right: list[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    return inter / max(bbox_area(left) + bbox_area(right) - inter, 1e-9)


def center_covered(pred: list[float], gold: list[float], margin: float = 2.0) -> bool:
    cx = (gold[0] + gold[2]) / 2.0
    cy = (gold[1] + gold[3]) / 2.0
    return pred[0] - margin <= cx <= pred[2] + margin and pred[1] - margin <= cy <= pred[3] + margin


def nwd_similarity(left: list[float], right: list[float], normalizer: float = 32.0) -> float:
    lcx = (left[0] + left[2]) / 2.0
    lcy = (left[1] + left[3]) / 2.0
    rcx = (right[0] + right[2]) / 2.0
    rcy = (right[1] + right[3]) / 2.0
    lw = max(0.0, left[2] - left[0])
    lh = max(0.0, left[3] - left[1])
    rw = max(0.0, right[2] - right[0])
    rh = max(0.0, right[3] - right[1])
    distance = ((lcx - rcx) ** 2 + (lcy - rcy) ** 2 + ((lw - rw) ** 2 + (lh - rh) ** 2) / 4.0) ** 0.5
    return math.exp(-distance / max(normalizer, 1e-6))


def target_area_buckets(row: dict[str, Any]) -> set[str]:
    buckets: set[str] = set()
    for target in ((row.get("targets") or {}).get("boxes") or []):
        box = target.get("page_bbox") or target.get("bbox") or []
        if isinstance(box, list) and len(box) == 4:
            buckets.add(area_bucket([float(v) for v in box]))
    return buckets


def sample_tiles_area_aware(rows: list[dict[str, Any]], limit: int, seed: int, positive_ratio: float, small_positive_ratio: float) -> list[dict[str, Any]]:
    if not limit or len(rows) <= limit:
        return list(rows)
    rng = random.Random(seed)
    positives = [row for row in rows if int((row.get("target_counts") or {}).get("symbols") or 0) > 0]
    empties = [row for row in rows if int((row.get("target_counts") or {}).get("symbols") or 0) == 0]
    small = [row for row in positives if target_area_buckets(row) & {"tiny_le_64", "small_le_256"}]
    small_ids = {id(row) for row in small}
    other = [row for row in positives if id(row) not in small_ids]
    for group in (small, other, empties):
        rng.shuffle(group)
    pos_n = min(len(positives), int(limit * positive_ratio))
    small_n = min(len(small), int(pos_n * small_positive_ratio))
    selected = small[:small_n] + other[: max(0, pos_n - small_n)]
    if len(selected) < pos_n:
        selected.extend(small[small_n : small_n + (pos_n - len(selected))])
    selected.extend(empties[: max(0, limit - len(selected))])
    if len(selected) < limit:
        used = {id(row) for row in selected}
        leftovers = [row for row in rows if id(row) not in used]
        rng.shuffle(leftovers)
        selected.extend(leftovers[: limit - len(selected)])
    rng.shuffle(selected)
    return selected[:limit]


def image_path_for_yolo_tile(yolo_dir: Path, split: str, row: dict[str, Any]) -> Path:
    if split == "dev":
        candidate_splits = ["val"]
    elif split.startswith("smoke"):
        candidate_splits = [split, "smoke", "smoke_v30", "locked"]
    else:
        candidate_splits = [split]
    for yolo_split in candidate_splits:
        path = yolo_dir / "images" / yolo_split / f"{row['id']}.jpg"
        if path.exists():
            return path
    return yolo_dir / "images" / candidate_splits[0] / f"{row['id']}.jpg"


def reconstruct_page_golds(eval_data: dict[str, Any]) -> dict[str, dict[str, dict[str, Any]]]:
    cfg = eval_data["config"]
    split = str(cfg.get("split") or "locked")
    data_dir = ROOT / str(cfg["data"])
    yolo_dir = ROOT / str(cfg["yolo_dir"])
    rows = load_jsonl(data_dir / f"{split}.jsonl")
    exported = [row for row in rows if image_path_for_yolo_tile(yolo_dir, split, row).exists()]
    sampled = sample_tiles_area_aware(
        exported,
        int(cfg.get("limit_tiles") or 0),
        int(cfg.get("seed") or 20260510) + (2 if split == "locked" else 3 if split.startswith("smoke") else 1),
        float(cfg.get("positive_ratio") or 0.85),
        float(cfg.get("small_positive_ratio") or 0.75),
    )
    page_golds: dict[str, dict[str, dict[str, Any]]] = defaultdict(dict)
    for row in sampled:
        row_id = str(row.get("row_id"))
        for gold in ((row.get("targets") or {}).get("boxes") or []):
            target_id = str(gold.get("target_id") or f"{row_id}_{len(page_golds[row_id])}")
            page_golds[row_id][target_id] = {
                "target_id": target_id,
                "bbox": [float(v) for v in gold.get("page_bbox") or gold.get("bbox")],
                "label": str(gold.get("label") or "generic_symbol"),
            }
    return dict(page_golds)


def load_predictions(path: Path) -> dict[str, list[dict[str, Any]]]:
    rows = load_jsonl(path)
    return {str(row.get("row_id")): list(row.get("predicted_symbols") or []) for row in rows}


def cause_for(best_iou: float, center_hit: bool, best_nwd: float, type_correct: bool) -> str:
    if best_iou >= 0.30:
        return "iou_match_type_correct" if type_correct else "iou_match_type_confusion"
    if center_hit:
        return "center_hit_iou_miss_box_quality"
    if best_nwd >= 0.70:
        return "nwd_near_iou_miss_tiny_box_quality"
    if best_iou >= 0.10:
        return "partial_overlap_below_iou"
    if best_iou >= 0.01:
        return "weak_overlap_or_duplicate_miss"
    return "no_near_prediction"


def analyze(eval_path: Path, predictions_path: Path) -> dict[str, Any]:
    eval_data = load_json(eval_path)
    page_golds = reconstruct_page_golds(eval_data)
    page_preds = load_predictions(predictions_path)

    causes = Counter()
    by_label: dict[str, Counter[str]] = defaultdict(Counter)
    by_area: dict[str, Counter[str]] = defaultdict(Counter)
    label_support = Counter()
    area_support = Counter()
    false_positive_by_label = Counter()
    rows = []
    examples = []
    totals = Counter()

    for row_id, gold_map in page_golds.items():
        preds = page_preds.get(row_id, [])
        used_iou: set[int] = set()
        row_cause = Counter()
        for gold in gold_map.values():
            gold_box = [float(v) for v in gold["bbox"]]
            label = str(gold["label"])
            bucket = area_bucket(gold_box)
            label_support[label] += 1
            area_support[bucket] += 1
            best_iou = 0.0
            best_iou_index = None
            best_nwd = 0.0
            center_index = None
            for pred_index, pred in enumerate(preds):
                pred_box = [float(v) for v in pred["bbox"]]
                current_iou = bbox_iou(pred_box, gold_box)
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_iou_index = pred_index
                best_nwd = max(best_nwd, nwd_similarity(pred_box, gold_box))
                if center_index is None and pred_index not in used_iou and center_covered(pred_box, gold_box):
                    center_index = pred_index
            matched = best_iou_index is not None and best_iou >= 0.30 and best_iou_index not in used_iou
            if matched:
                used_iou.add(int(best_iou_index))
            pred_label = str(preds[best_iou_index]["label"]) if best_iou_index is not None else ""
            type_correct = matched and pred_label == label
            cause = cause_for(best_iou if best_iou_index is not None else 0.0, center_index is not None, best_nwd, type_correct)
            causes[cause] += 1
            by_label[label][cause] += 1
            by_area[bucket][cause] += 1
            row_cause[cause] += 1
            totals["gold"] += 1
            if matched:
                totals["matched_iou"] += 1
            if center_index is not None:
                totals["center_hit"] += 1
            if best_nwd >= 0.70:
                totals["nwd_070"] += 1
            if cause != "iou_match_type_correct" and len(examples) < 80:
                examples.append({
                    "row_id": row_id,
                    "target_id": gold["target_id"],
                    "label": label,
                    "area_bucket": bucket,
                    "cause": cause,
                    "best_iou": round(best_iou, 6),
                    "best_nwd": round(best_nwd, 6),
                    "best_pred_label": pred_label,
                    "gold_bbox": gold_box,
                    "best_pred_bbox": preds[best_iou_index]["bbox"] if best_iou_index is not None else None,
                })
        fp = max(0, len(preds) - len(used_iou))
        totals["predicted"] += len(preds)
        totals["false_positive_or_unmatched"] += fp
        for pred_index, pred in enumerate(preds):
            if pred_index not in used_iou:
                false_positive_by_label[str(pred.get("label") or "unknown")] += 1
        if row_cause:
            rows.append({"row_id": row_id, "gold": len(gold_map), "predicted": len(preds), "unmatched_predictions": fp, "causes": dict(row_cause)})

    def normalize(counter: Counter[str], support: int) -> dict[str, Any]:
        return {
            cause: {"count": int(count), "rate": round(count / max(support, 1), 6)}
            for cause, count in counter.most_common()
        }

    label_report = {
        label: {"support": int(label_support[label]), "causes": normalize(counter, label_support[label])}
        for label, counter in sorted(by_label.items())
    }
    area_report = {
        bucket: {"support": int(area_support[bucket]), "causes": normalize(counter, area_support[bucket])}
        for bucket, counter in sorted(by_area.items())
    }
    worst_labels = sorted(
        (
            {
                "label": label,
                "support": int(label_support[label]),
                "center_miss_rate": round(1.0 - (counter["iou_match_type_correct"] + counter["iou_match_type_confusion"] + counter["center_hit_iou_miss_box_quality"]) / max(label_support[label], 1), 6),
                "iou_success_rate": round((counter["iou_match_type_correct"] + counter["iou_match_type_confusion"]) / max(label_support[label], 1), 6),
                "dominant_cause": counter.most_common(1)[0][0] if counter else "",
            }
            for label, counter in by_label.items()
        ),
        key=lambda item: (item["iou_success_rate"], -item["support"]),
    )

    return {
        "id": "SCI-P2-117-symbol-subset-failure-attribution",
        "claim_boundary": "Failure attribution for the P116 2000-tile subset only; not a full locked model-quality claim.",
        "inputs": {"eval": rel(eval_path), "predictions": rel(predictions_path)},
        "eval_gate": eval_data.get("gate"),
        "selected_thresholds": eval_data.get("selected_thresholds"),
        "totals": {key: int(value) for key, value in totals.items()},
        "cause_summary": normalize(causes, totals["gold"]),
        "by_label": label_report,
        "by_area": area_report,
        "false_positive_by_label": dict(false_positive_by_label.most_common()),
        "worst_labels_by_iou_success": worst_labels[:12],
        "worst_rows_by_failure_count": sorted(
            rows,
            key=lambda row: (row["gold"] - row["causes"].get("iou_match_type_correct", 0) - row["causes"].get("iou_match_type_confusion", 0), row["unmatched_predictions"]),
            reverse=True,
        )[:30],
        "examples": examples,
        "interpretation": [
            "Type accuracy is high once IoU matching succeeds; localization and box-quality misses dominate the failed gates.",
            "center_hit_iou_miss_box_quality indicates proposals roughly locate the symbol but boxes do not overlap enough for IoU@0.30.",
            "nwd_near_iou_miss_tiny_box_quality is especially relevant for tiny boxes where NWD sees proximity but IoU remains harsh.",
            "no_near_prediction and weak_overlap_or_duplicate_miss are coverage/localization misses that a full run will not fix by itself.",
        ],
    }


def write_report(path: Path, data: dict[str, Any]) -> None:
    totals = data["totals"]
    lines = [
        "# P2-117 Symbol Subset Failure Attribution",
        "",
        "## Decision",
        "",
        "- P116 subset failed because localization/box quality is still the main bottleneck.",
        "- Full locked evaluation should remain evidence expansion only until these subset failure modes are addressed.",
        "- Claim boundary: P116 subset attribution only.",
        "",
        "## Totals",
        "",
        "| Metric | Value |",
        "|---|---:|",
    ]
    for key in ["gold", "predicted", "matched_iou", "center_hit", "nwd_070", "false_positive_or_unmatched"]:
        lines.append(f"| `{key}` | {totals.get(key, 0)} |")
    lines.extend(["", "## Cause Summary", "", "| Cause | Count | Rate |", "|---|---:|---:|"])
    for cause, item in data["cause_summary"].items():
        lines.append(f"| `{cause}` | {item['count']} | {item['rate']:.6f} |")
    lines.extend(["", "## Area Buckets", "", "| Area | Support | Dominant Cause | Dominant Rate |", "|---|---:|---|---:|"])
    for area, item in data["by_area"].items():
        dominant, value = next(iter(item["causes"].items()))
        lines.append(f"| `{area}` | {item['support']} | `{dominant}` | {value['rate']:.6f} |")
    lines.extend(["", "## Worst Labels", "", "| Label | Support | IoU Success | Center Miss Rate | Dominant Cause |", "|---|---:|---:|---:|---|"])
    for item in data["worst_labels_by_iou_success"][:10]:
        lines.append(f"| `{item['label']}` | {item['support']} | {item['iou_success_rate']:.6f} | {item['center_miss_rate']:.6f} | `{item['dominant_cause']}` |")
    lines.extend(["", "## Interpretation"])
    for item in data["interpretation"]:
        lines.append(f"- {item}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval", default=str(DEFAULT_EVAL))
    parser.add_argument("--predictions", default=str(DEFAULT_PREDICTIONS))
    parser.add_argument("--output", default=str(DEFAULT_JSON))
    parser.add_argument("--report", default=str(DEFAULT_REPORT))
    args = parser.parse_args()
    data = analyze(Path(args.eval), Path(args.predictions))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    write_report(Path(args.report), data)
    print(json.dumps({"output": rel(output), "report": rel(Path(args.report)), "top_causes": data["cause_summary"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
