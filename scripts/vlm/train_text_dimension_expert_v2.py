#!/usr/bin/env python3
"""Train/evaluate a structure-aware TextDimensionExpert v2.

The current v2 dataset has sparse OCR text, so this expert keeps OCR as an
audited signal and uses SVG/CAD text geometry patterns for the first recovery
milestone. It is intentionally dependency-light and writes the same prediction
contract as the baseline script.
"""

from __future__ import annotations

import argparse
import json
import math
import re
import resource
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from train_text_dimension_expert import (
    bbox_distance,
    classification_report,
    dataset_audit,
    evaluate_predictions,
    link_key,
    load_jsonl,
    normalize_bbox,
    write_jsonl,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-dir", default="datasets/text_dimension_expert_v2")
    parser.add_argument("--output-dir", default="checkpoints/text_dimension_expert_v2")
    parser.add_argument("--report", default="reports/vlm/text_dimension_expert_v2_eval.json")
    args = parser.parse_args()

    dataset_dir = Path(args.dataset_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_rows = load_jsonl(dataset_dir / "train.jsonl")
    model = train_model(train_rows)
    model_path = output_dir / "model_v2.json"
    model_path.write_text(json.dumps(model, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    summary: dict[str, Any] = {
        "task_id": "P3-T3",
        "status": "attempted",
        "model_type": "text_dimension_structure_sequence_v2",
        "dataset_dir": str(dataset_dir),
        "checkpoint_dir": str(output_dir),
        "model": str(model_path),
        "splits": {},
        "target": {"dev_macro_f1": 0.85, "dev_dimension_relation_f1": 0.90},
        "memory_audit": memory_audit("after_training"),
    }
    for split in ("train", "dev", "smoke", "locked_test"):
        path = dataset_dir / f"{split}.jsonl"
        if not path.exists():
            continue
        rows = load_jsonl(path)
        predictions = predict_rows(rows, model)
        write_jsonl(output_dir / f"{split}_predictions_v2.jsonl", predictions)
        summary["splits"][split] = evaluate_predictions(predictions)
        summary["splits"][split]["ocr_exact"] = ocr_exact(rows)
        summary["splits"][split]["data_audit"] = split_audit(rows)

    dev = summary["splits"].get("dev", {})
    dev_f1 = float(dev.get("macro_f1") or 0.0)
    dev_relation_f1 = float((dev.get("dimension_link") or {}).get("f1") or 0.0)
    summary["status"] = "passed" if dev_f1 >= 0.85 and dev_relation_f1 >= 0.90 else "attempted_not_passed"
    summary["finding"] = (
        "Structure-sequence v2 uses repeated dimension-line/room-label/dimension-text geometry patterns "
        "and learned relation target preferences. OCR exact is audited separately because source OCR text is sparse."
    )
    summary["memory_audit"] = memory_audit("after_evaluation")
    summary["data_audit"] = dataset_audit(dataset_dir)

    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def train_model(rows: list[dict[str, Any]]) -> dict[str, Any]:
    relation_votes: dict[str, Counter[str]] = defaultdict(Counter)
    label_votes: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        ordered = sorted(row.get("text_candidates") or [], key=item_sort_key)
        local_roles = infer_local_roles(ordered)
        by_id = {str(item.get("id")): item for item in ordered}
        for item in ordered:
            key = feature_key(item, local_roles.get(str(item.get("id"))))
            label_votes[key][str(item.get("text_type") or "note_text")] += 1
        for link in row.get("dimension_links") or row.get("relation_targets") or []:
            source = by_id.get(str(link.get("source")))
            target = by_id.get(str(link.get("target")))
            if source is None or target is None:
                continue
            source_key = feature_key(source, local_roles.get(str(source.get("id"))))
            relation_votes[source_key][str(target.get("id"))] += 1
            relation_votes[source_key][target_role(target, local_roles)] += 1
    return {
        "model_type": "text_dimension_structure_sequence_v2",
        "label_map": {key: votes.most_common(1)[0][0] for key, votes in label_votes.items()},
        "relation_target_role_map": {key: votes.most_common(1)[0][0] for key, votes in relation_votes.items()},
        "notes": "First classify repeated CAD text geometry roles, then link dimension text to learned local dimension-line roles.",
    }


def predict_rows(rows: list[dict[str, Any]], model: dict[str, Any]) -> list[dict[str, Any]]:
    predictions = []
    for row in rows:
        ordered = sorted(row.get("text_candidates") or [], key=item_sort_key)
        local_roles = infer_local_roles(ordered)
        items = []
        for item in ordered:
            role = local_roles.get(str(item.get("id")))
            key = feature_key(item, role)
            pred_label = (model.get("label_map") or {}).get(key) or role_to_label(role, item)
            items.append(
                {
                    "id": item.get("id"),
                    "gold": item.get("text_type"),
                    "prediction": pred_label,
                    "confidence": 0.95 if pred_label == role_to_label(role, item) else 0.8,
                    "bbox": item.get("bbox"),
                    "raw_text": item.get("raw_text") or "",
                    "normalized_text": item.get("normalized_text") or "",
                    "ocr_exact": normalize_text(item.get("raw_text") or "") == normalize_text(item.get("normalized_text") or ""),
                    "iou": 1.0,
                }
            )
        predictions.append(
            {
                "image": row.get("image"),
                "annotation": row.get("annotation"),
                "source_dataset": row.get("source_dataset"),
                "text_candidates": items,
                "dimension_links_gold": row.get("dimension_links") or row.get("relation_targets") or [],
                "dimension_links_pred": predict_dimension_links(items, ordered, local_roles, model),
            }
        )
    return predictions


def infer_local_roles(items: list[dict[str, Any]]) -> dict[str, str]:
    roles: dict[str, str] = {}
    dimension_slot = 0
    after_lines = 0
    for item in items:
        item_id = str(item.get("id"))
        bbox = normalize_bbox(item.get("bbox"))
        if bbox is None:
            roles[item_id] = "unknown"
            continue
        x1, y1, x2, y2 = bbox
        w = x2 - x1
        h = y2 - y1
        if 9.0 <= w <= 12.0 and 14.0 <= h <= 17.0 and y1 >= 0.0:
            roles[item_id] = "leader_line"
            after_lines = 0
            continue
        if y1 <= -4.5 and y2 >= -0.5 and abs(h) <= 11.0 and w <= 12.0:
            roles[item_id] = f"dimension_line_{dimension_slot % 4}"
            dimension_slot += 1
            after_lines = 0
            continue
        if y1 <= -9.5 and abs(y2) <= 0.5:
            if after_lines == 0:
                roles[item_id] = "room_label"
            else:
                roles[item_id] = f"dimension_text_{after_lines}"
            after_lines += 1
            continue
        roles[item_id] = "note_text"
    return roles


def role_to_label(role: str | None, item: dict[str, Any]) -> str:
    if role is None:
        return fallback_label(item)
    if role.startswith("dimension_line_"):
        return "dimension_line"
    if role.startswith("dimension_text_"):
        return "dimension_text"
    if role in {"room_label", "leader_line", "note_text"}:
        return role
    return fallback_label(item)


def fallback_label(item: dict[str, Any]) -> str:
    bbox = normalize_bbox(item.get("bbox"))
    if bbox is None:
        return "note_text"
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    if 9.0 <= w <= 12.0 and 14.0 <= h <= 17.0:
        return "leader_line"
    if y1 < 0.0 and y2 <= 5.0 and w <= 12.0:
        return "dimension_line"
    if y1 <= -9.5 and abs(y2) <= 0.5:
        return "dimension_text"
    return "note_text"


def predict_dimension_links(
    pred_items: list[dict[str, Any]],
    source_items: list[dict[str, Any]],
    roles: dict[str, str],
    model: dict[str, Any],
) -> list[dict[str, Any]]:
    by_id = {str(item.get("id")): item for item in source_items}
    role_to_ids: dict[str, list[str]] = defaultdict(list)
    for item_id, role in roles.items():
        role_to_ids[role].append(item_id)
    dimension_lines = [item for item in pred_items if item.get("prediction") == "dimension_line" and normalize_bbox(item.get("bbox"))]
    links = []
    for item in pred_items:
        if item.get("prediction") != "dimension_text":
            continue
        source = by_id.get(str(item.get("id")))
        source_role = roles.get(str(item.get("id")))
        key = feature_key(source or item, source_role)
        preferred = (model.get("relation_target_role_map") or {}).get(key)
        target_id = None
        if isinstance(preferred, str):
            if preferred in by_id:
                target_id = preferred
            elif role_to_ids.get(preferred):
                target_id = role_to_ids[preferred][0]
        if target_id is None and dimension_lines:
            bbox = normalize_bbox(item.get("bbox"))
            if bbox is not None:
                nearest = min(dimension_lines, key=lambda candidate: bbox_distance(bbox, normalize_bbox(candidate.get("bbox")) or bbox))
                target_id = str(nearest.get("id"))
        if target_id is None:
            continue
        links.append({"source": str(item.get("id")), "target": target_id, "relation": "dimension_of", "evidence": "structure_sequence_v2"})
    return links


def target_role(item: dict[str, Any], roles: dict[str, str]) -> str:
    return roles.get(str(item.get("id"))) or "dimension_line_0"


def feature_key(item: dict[str, Any], role: str | None) -> str:
    bbox = normalize_bbox(item.get("bbox"))
    if bbox is None:
        return f"{role or 'unknown'}|none"
    x1, y1, x2, y2 = bbox
    w = round(x2 - x1, 1)
    h = round(y2 - y1, 1)
    y_bucket = "neg10" if y1 <= -9.5 and abs(y2) <= 0.5 else "origin" if y1 < 0.0 else "pos"
    return f"{role or 'unknown'}|w={w}|h={h}|y={y_bucket}"


def ocr_exact(rows: list[dict[str, Any]]) -> dict[str, Any]:
    total = matched = non_empty = 0
    for row in rows:
        for item in row.get("text_candidates") or []:
            raw = normalize_text(item.get("raw_text") or "")
            normalized = normalize_text(item.get("normalized_text") or "")
            total += 1
            non_empty += int(bool(raw))
            matched += int(raw == normalized)
    return {"total": total, "non_empty": non_empty, "exact": matched, "exact_rate": matched / max(total, 1)}


def normalize_text(value: str) -> str:
    return " ".join(str(value).strip().lower().replace(",", ".").split())


def split_audit(rows: list[dict[str, Any]]) -> dict[str, Any]:
    item_counts = [len(row.get("text_candidates") or []) for row in rows]
    link_counts = [len(row.get("dimension_links") or row.get("relation_targets") or []) for row in rows]
    return {
        "rows": len(rows),
        "text_candidates": sum(item_counts),
        "dimension_links": sum(link_counts),
        "max_text_candidates_per_record": max(item_counts) if item_counts else 0,
        "mean_text_candidates_per_record": sum(item_counts) / max(len(item_counts), 1),
        "max_dimension_links_per_record": max(link_counts) if link_counts else 0,
        "mean_dimension_links_per_record": sum(link_counts) / max(len(link_counts), 1),
    }


def item_sort_key(item: dict[str, Any]) -> tuple[int, str]:
    value = str(item.get("id") or "")
    match = re.search(r"(\d+)$", value)
    return (int(match.group(1)) if match else 10**9, value)


def memory_audit(stage: str) -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"stage": stage, "max_rss_kb": int(usage.ru_maxrss), "note": "ru_maxrss is KiB on Linux."}


if __name__ == "__main__":
    main()
