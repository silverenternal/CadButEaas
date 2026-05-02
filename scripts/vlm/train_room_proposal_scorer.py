#!/usr/bin/env python3
"""Train a lightweight room proposal keep/suppress scorer and quality head."""

from __future__ import annotations

import argparse
import json
import math
import resource
from collections import defaultdict, Counter
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/cadstruct_room_proposals_v1")
    parser.add_argument("--output-dir", default="checkpoints/cadstruct_room_proposal_scorer_v1")
    parser.add_argument("--report-path", default="reports/vlm/room_proposal_scorer_locked_test.json")
    parser.add_argument("--iou-positive", type=float, default=0.5)
    parser.add_argument("--train-split", default="train")
    parser.add_argument("--eval-split", default="locked_test")
    parser.add_argument("--smoke-split", default="smoke")
    parser.add_argument("--iou-nms", type=float, default=0.95)
    parser.add_argument("--nms-max-per-image", type=int, default=80)
    parser.add_argument("--iou-match-thresholds", default="0.5,0.75,0.9")
    args = parser.parse_args()

    thresholds = [float(item) for item in args.iou_match_thresholds.split(",") if item]
    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = Path(args.report_path)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    split_paths = {split: dataset_dir / f"{split}.jsonl" for split in (args.train_split, args.eval_split, args.smoke_split)}
    for split, path in split_paths.items():
        if not path.exists():
            raise SystemExit(f"split file missing: {path}")

    rows = {split: load_jsonl(path) for split, path in split_paths.items()}
    train_rows = rows[args.train_split]

    train_examples = build_examples(train_rows, args.iou_positive, thresholds=thresholds, add_quality=True)
    if not train_examples:
        raise SystemExit("no training examples generated")

    keep_model = train_centroid_model([example for example in train_examples if example["label"] == 1])
    suppress_model = train_centroid_model([example for example in train_examples if example["label"] == 0])
    quality_model = train_centroid_model(
        [example for example in train_examples if example["label"] == 1 and example["quality_label"] is not None]
    )

    model = {
        "model_type": "room_proposal_centroid_scorer_v1",
        "feature_names": FEATURE_NAMES,
        "keep_model": keep_model,
        "suppress_model": suppress_model,
        "quality_model": quality_model,
        "iou_positive_threshold": args.iou_positive,
        "iou_match_thresholds": thresholds,
        "iou_nms": args.iou_nms,
        "nms_max_per_image": args.nms_max_per_image,
        "notes": "Deterministic prototype scorer for room proposal keep/suppress and coarse quality head.",
    }

    model_path = output_dir / "model.json"
    model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    train_summary = {
        "dataset_dir": str(dataset_dir),
        "model_path": str(model_path),
        "model_type": model["model_type"],
        "memory_audit": memory_audit("after_training"),
        "data": {
            "train_examples": len(train_examples),
            "keep_pos": sum(1 for item in train_examples if item["label"] == 1),
            "suppress_neg": sum(1 for item in train_examples if item["label"] == 0),
            "quality_samples": sum(1 for item in train_examples if item["quality_label"] is not None),
            "quality_bins": Counter(item["quality_label"] for item in train_examples if item["quality_label"] is not None),
        },
        "splits": {},
    }

    for split in (args.train_split, args.eval_split, args.smoke_split):
        split_rows = rows[split]
        scored = score_and_nms_rows(split_rows, model)
        summary = evaluate_split(split, scored, args.iou_positive, args.iou_nms, nms_max_per_image=args.nms_max_per_image)
        write_jsonl(output_dir / f"{split}_predictions.jsonl", scored)
        train_summary["splits"][split] = summary

    train_summary["memory_audit"] = memory_audit("after_evaluation")
    (output_dir / "train_summary.json").write_text(json.dumps(train_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report = build_report(train_summary, args)
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_examples(
    rows: list[dict[str, Any]],
    iou_positive: float,
    thresholds: list[float],
    add_quality: bool = True,
) -> list[dict[str, Any]]:
    examples: list[dict[str, Any]] = []
    for row in rows:
        width = float(row.get("width") or 1.0)
        height = float(row.get("height") or 1.0)
        page_area = max(width * height, 1.0)
        proposals = row.get("proposals") or []
        rooms = row.get("rooms") or []
        for proposal in proposals:
            feature = proposal_features(proposal, width, height, page_area)
            if feature is None:
                continue
            best_iou, quality_label = proposal_best_iou_and_quality(proposal, rooms, thresholds, iou_positive)
            label = 1 if best_iou >= iou_positive else 0
            if not add_quality:
                quality_label = None
            examples.append(
                {
                    "features": feature,
                    "label": int(label),
                    "quality_label": int(quality_label) if quality_label is not None else None,
                    "best_iou": best_iou,
                    "proposal_id": str(proposal.get("id") or ""),
                    "source": str(proposal.get("metadata", {}).get("source_dataset") or proposal.get("source") or ""),
                }
            )
    if not examples:
        raise SystemExit("No valid proposal examples from dataset")
    return examples


def proposal_best_iou_and_quality(
    proposal: dict[str, Any],
    rooms: list[dict[str, Any]],
    thresholds: list[float],
    iou_positive: float,
) -> tuple[float, int | None]:
    proposal_box = proposal.get("bbox") or []
    if not valid_box(proposal_box):
        return 0.0, None
    best = 0.0
    for room in rooms:
        room_box = room.get("bbox") or []
        if not valid_box(room_box):
            continue
        overlap = iou(proposal_box, room_box)
        if overlap > best:
            best = overlap
    quality = None
    for idx, threshold in enumerate(sorted(thresholds), start=1):
        if best >= threshold:
            quality = idx
    if quality is None and best >= iou_positive:
        quality = 0
    return best, quality


def train_centroid_model(samples: list[dict[str, Any]]) -> dict[str, Any]:
    means: dict[str, list[float]] = defaultdict(list)
    for sample in samples:
        for name, value in zip(FEATURE_NAMES, sample["features"]):
            means[name].append(float(value))

    if not samples:
        return {
            "count": 0,
            "mean": [0.0] * len(FEATURE_NAMES),
            "std": [1.0] * len(FEATURE_NAMES),
            "quality_bins": {},
        }

    mean = [sum(values) / len(values) for values in means.values()]
    std = [
        max(math.sqrt(sum((value - mean[idx]) ** 2 for value in means[name]) / max(len(values), 1)), 1e-6)
        for idx, (name, values) in enumerate(means.items())
    ]

    model: dict[str, Any] = {"count": len(samples), "mean": mean, "std": std}
    bins = defaultdict(list)
    for sample in samples:
        quality = sample.get("quality_label")
        if quality is None:
            continue
        bins[str(int(quality))].append(sample["features"])
    model["quality_bins"] = {
        key: [sum(row[idx] for row in rows) / max(len(rows), 1) for idx in range(len(FEATURE_NAMES))]
        for key, rows in {
            quality: rows for quality, rows in bins.items()
        }.items()
        if rows
    }
    model["quality_bins_count"] = {key: len(rows) for key, rows in bins.items() if rows}
    return model


def score_keep_probability(features: list[float], model_keep: dict[str, Any], model_suppress: dict[str, Any]) -> float:
    keep_distance = distance_to_centroid(features, model_keep["mean"], model_keep["std"])
    suppress_distance = distance_to_centroid(features, model_suppress["mean"], model_suppress["std"])
    if not math.isfinite(keep_distance) or not math.isfinite(suppress_distance):
        return 0.0
    diff = keep_distance - suppress_distance
    diff = max(min(diff, 60.0), -60.0)
    return 1.0 / (1.0 + math.exp(diff))


def predict_quality_bin(features: list[float], quality_model: dict[str, Any]) -> int:
    bins = quality_model.get("quality_bins") or {}
    if not bins:
        return 0
    best_label = None
    best_distance = float("inf")
    for key, center in bins.items():
        dist = euclidean(features, center)
        if dist < best_distance:
            best_distance = dist
            best_label = key
    if best_label is None:
        return 0
    return int(best_label)


def score_and_nms_rows(rows: list[dict[str, Any]], model: dict[str, Any]) -> list[dict[str, Any]]:
    outputs: list[dict[str, Any]] = []
    for row in rows:
        width = float(row.get("width") or 1.0)
        height = float(row.get("height") or 1.0)
        page_area = max(width * height, 1.0)
        proposals = row.get("proposals") or []
        scored = []
        for proposal in proposals:
            feature = proposal_features(proposal, width, height, page_area)
            if feature is None:
                continue
            keep_prob = score_keep_probability(feature, model["keep_model"], model["suppress_model"])
            quality = predict_quality_bin(feature, model["quality_model"])
            candidate = dict(proposal)
            candidate["score_keep"] = keep_prob
            candidate["score"] = keep_prob
            candidate["pred_keep"] = int(keep_prob >= 0.5)
            candidate["pred_quality_bin"] = quality
            scored.append(candidate)

        kept = nms_sorted(scored, iou_threshold=model["iou_nms"], max_count=model["nms_max_per_image"])
        outputs.append(
            {
                "image": row.get("image"),
                "annotation": row.get("annotation"),
                "source_dataset": row.get("source_dataset"),
                "proposals": kept,
                "rooms": row.get("rooms", []),
                "proposal_scores": {str(item["id"]): item["score_keep"] for item in kept},
            }
        )
    return outputs


def evaluate_split(split: str, rows: list[dict[str, Any]], iou_positive: float, iou_nms: float, nms_max_per_image: int) -> dict[str, Any]:
    total_gt = total_pred_keep = total_tp_keep = total_preditions = 0
    iou_list: list[float] = []
    quality_confusion = Counter()
    all_labels: list[int] = []
    all_scores: list[float] = []
    ap = 0.0
    for row in rows:
        proposals = row.get("proposals") or []
        rooms = row.get("rooms") or []
        total_preditions += len(proposals)
        labels = []
        scores = []
        used = set()
        for proposal in proposals:
            feature = proposal_features(proposal, 1.0, 1.0, 1.0)
            iou_score = 0.0
            for room in rooms:
                candidate = iou(proposal.get("bbox") or [], room.get("bbox") or [])
                if candidate > iou_score:
                    iou_score = candidate
            gt_keep = 1 if iou_score >= iou_positive else 0
            pred_keep = int(float(proposal.get("score_keep") or 0.0) >= 0.5)
            labels.append(gt_keep)
            scores.append(float(proposal.get("score_keep") or 0.0))
            total_gt += gt_keep
            total_pred_keep += pred_keep
            total_tp_keep += int(gt_keep == 1 and pred_keep == 1)
            if proposal.get("pred_quality_bin") is not None and gt_keep:
                quality_confusion[f"quality_pred_{proposal['pred_quality_bin']}_vs_gt_{quality_bucket(iou_score)}"] += 1
            iou_list.append(iou_score)
            if gt_keep == 1 and pred_keep == 1:
                pass
        all_scores.extend(scores)
        all_labels.extend(labels)
    precision = total_tp_keep / max(total_pred_keep, 1)
    recall = total_tp_keep / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    ap50 = compute_average_precision(all_scores, all_labels)
    return {
        "split": split,
        "proposals_scored": total_preditions,
        "gt_keep": total_gt,
        "pred_keep": total_pred_keep,
        "tp_keep": total_tp_keep,
        "precision_keep": precision,
        "recall_keep": recall,
        "f1_keep": f1,
        "ap50": ap50,
        "nms_iou": iou_nms,
        "nms_max_per_image": nms_max_per_image,
        "mean_best_iou": sum(iou_list) / max(len(iou_list), 1),
        "quality_confusion": {k: v for k, v in sorted(quality_confusion.items())},
        "finding": {
            "recall_pass": recall >= 0.98,
            "gt_support": {"rows": len(rows), "gt_rooms": sum(len(row.get("rooms") or []) for row in rows)},
        },
    }


def compute_average_precision(scores: list[float], labels: list[int]) -> float:
    if not scores:
        return 0.0
    # labels are interpreted by keep_gt after scoring threshold-free and sorted by score
    paired = sorted(zip(scores, labels), key=lambda item: item[0], reverse=True)
    total_positives = sum(1 for _, label in paired if label >= 1)
    if total_positives == 0:
        return 0.0

    tp = fp = 0.0
    precisions: list[float] = []
    recalls: list[float] = []
    for score, label in paired:
        if label >= 1:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / total_positives)

    # 11-point interpolated AP as compact baseline
    ap = 0.0
    for level in [i / 10 for i in range(11)]:
        p = max((p for p, r in zip(precisions, recalls) if r >= level), default=0.0)
        ap += p / 11.0
    return ap


def nms_sorted(proposals: list[dict[str, Any]], iou_threshold: float, max_count: int) -> list[dict[str, Any]]:
    proposals = sorted(proposals, key=lambda item: float(item.get("score_keep") or 0.0), reverse=True)
    selected: list[dict[str, Any]] = []
    for proposal in proposals:
        keep = True
        for kept in selected:
            if iou(proposal.get("bbox") or [], kept.get("bbox") or []) >= iou_threshold:
                keep = False
                break
        if keep:
            selected.append(proposal)
        if len(selected) >= max_count:
            break
    return selected


def proposal_features(proposal: dict[str, Any], width: float, height: float, page_area: float) -> list[float] | None:
    bbox = proposal.get("bbox")
    if not valid_box(bbox):
        return None
    x1, y1, x2, y2 = [float(v) for v in bbox]
    proposal_area = max(0.0, (x2 - x1) * (y2 - y1))
    if proposal_area <= 0:
        return None
    cx = (x1 + x2) / 2.0 / max(width, 1.0)
    cy = (y1 + y2) / 2.0 / max(height, 1.0)
    w = (x2 - x1) / max(width, 1.0)
    h = (y2 - y1) / max(height, 1.0)
    area = proposal_area / max(page_area, 1.0)
    aspect = math.log((w + 1e-6) / (h + 1e-6))
    conf = float(proposal.get("confidence") or 0.0)
    semantic_type = str((proposal.get("metadata") or {}).get("semantic_type") or "room").lower()
    semantic_room = 1.0 if semantic_type == "room" else 0.0
    semantic_window = 1.0 if semantic_type == "window" else 0.0
    return [cx, cy, w, h, area, aspect, conf, semantic_room, semantic_window]


def quality_bucket(best_iou: float) -> int:
    if best_iou >= 0.9:
        return 2
    if best_iou >= 0.75:
        return 1
    return 0


def distance_to_centroid(features: list[float], mean: list[float], std: list[float]) -> float:
    dist2 = 0.0
    for index, value in enumerate(features):
        denom = max(std[index], 1e-6)
        delta = (value - mean[index]) / denom
        dist2 += delta * delta
    return math.sqrt(dist2)


def euclidean(left: list[float], right: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(left, right)))


def centroid(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    return [sum(value[index] for value in vectors) / len(vectors) for index in range(len(vectors[0]))]


def iou(left: list[float], right: list[float]) -> float:
    if not valid_box(left) or not valid_box(right):
        return 0.0
    left_x1, left_y1, left_x2, left_y2 = [float(item) for item in left]
    right_x1, right_y1, right_x2, right_y2 = [float(item) for item in right]
    inter_x1 = max(left_x1, right_x1)
    inter_y1 = max(left_y1, right_y1)
    inter_x2 = min(left_x2, right_x2)
    inter_y2 = min(left_y2, right_y2)
    inter_area = max(0.0, inter_x2 - inter_x1) * max(0.0, inter_y2 - inter_y1)
    if inter_area <= 0:
        return 0.0
    left_area = max(0.0, (left_x2 - left_x1) * (left_y2 - left_y1))
    right_area = max(0.0, (right_x2 - right_x1) * (right_y2 - right_y1))
    union = left_area + right_area - inter_area
    if union <= 0:
        return 0.0
    return inter_area / union


def valid_box(bbox: Any) -> bool:
    if not isinstance(bbox, list) or len(bbox) != 4:
        return False
    try:
        x1, y1, x2, y2 = [float(item) for item in bbox]
    except (TypeError, ValueError):
        return False
    return x2 > x1 and y2 > y1


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def build_report(train_summary: dict[str, Any], args: argparse.Namespace) -> dict[str, Any]:
    return {
        "task_id": "P1.2",
        "model_type": "room_proposal_centroid_scorer_v1",
        "dataset_dir": str(Path(args.dataset_dir)),
        "train_split": args.train_split,
        "eval_split": args.eval_split,
        "smoke_split": args.smoke_split,
        "report_path": args.report_path,
        "iou_positive": args.iou_positive,
        "iou_nms": args.iou_nms,
        "nms_max_per_image": args.nms_max_per_image,
        "memory_audit": memory_audit("report"),
        "splits": {name: train_summary["splits"][name] for name in ("train", "locked_test", "smoke") if name in train_summary["splits"]},
        "finding": "Centroid scorer used for room proposal keep/suppress baseline; keep this as audit baseline until learned ranking/model scoring is introduced.",
    }


def memory_audit(stage: str) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return {"stage": stage, "max_rss_kb": int(usage), "note": "ru_maxrss is KiB on Linux."}


FEATURE_NAMES = ["cx", "cy", "w", "h", "area", "aspect", "conf", "semantic_room", "semantic_window"]


if __name__ == "__main__":
    main()
