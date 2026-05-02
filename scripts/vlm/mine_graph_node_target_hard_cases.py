#!/usr/bin/env python3
"""Mine target-domain graph-node hard cases for adaptation and annotation."""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--target-source", default="")
    parser.add_argument("--fragile-labels", default="door,window")
    parser.add_argument("--low-confidence-threshold", type=float, default=0.9)
    parser.add_argument("--top-k", type=int, default=200)
    args = parser.parse_args()

    fragile_labels = {item.strip() for item in args.fragile_labels.split(",") if item.strip()}
    dataset = load_dataset(Path(args.dataset))
    predictions = load_predictions(Path(args.predictions))
    rows = join_rows(dataset, predictions)
    if args.target_source:
        rows = [row for row in rows if str(row.get("source_dataset") or "") == args.target_source]

    hard_cases = [row for row in rows if is_hard_case(row, fragile_labels, args.low_confidence_threshold)]
    hard_cases = sorted(hard_cases, key=hard_case_sort_key)[: args.top_k]
    summary = build_summary(rows, hard_cases, args)

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in hard_cases:
            handle.write(json.dumps(compact_case(row), ensure_ascii=False) + "\n")

    summary_path = Path(args.summary_output)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def load_dataset(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = {}
    for sample in load_jsonl(path):
        image = str(sample.get("image"))
        source = str(sample.get("source_dataset") or "unknown")
        node_count = len(sample.get("nodes") or [])
        edge_count = len(sample.get("edges") or [])
        for node in sample.get("nodes") or []:
            features = node.get("features") or {}
            rows[(image, int(node["id"]))] = {
                "image": image,
                "source_dataset": source,
                "id": int(node["id"]),
                "label": str(node.get("label")),
                "bbox": features.get("bbox"),
                "orientation": features.get("orientation"),
                "primitive_type": features.get("primitive_type"),
                "node_count": node_count,
                "edge_count": edge_count,
                "features": features,
            }
    return rows


def load_predictions(path: Path) -> dict[tuple[str, int], dict[str, Any]]:
    rows = {}
    for sample in load_jsonl(path):
        image = str(sample.get("image"))
        for node in sample.get("nodes") or []:
            rows[(image, int(node["id"]))] = {
                "prediction": str(node.get("prediction")),
                "confidence": float(node.get("confidence", 0.0) or 0.0),
            }
    return rows


def join_rows(dataset: dict[tuple[str, int], dict[str, Any]], predictions: dict[tuple[str, int], dict[str, Any]]) -> list[dict[str, Any]]:
    output = []
    for key, row in dataset.items():
        prediction = predictions.get(key)
        if prediction is None:
            continue
        output.append({**row, **prediction, "correct": row["label"] == prediction["prediction"]})
    return output


def is_hard_case(row: dict[str, Any], fragile_labels: set[str], low_confidence_threshold: float) -> bool:
    if not row["correct"]:
        return True
    if row["confidence"] < low_confidence_threshold:
        return True
    return row["label"] in fragile_labels and row["confidence"] < 0.98


def hard_case_sort_key(row: dict[str, Any]) -> tuple[int, int, float, str, int]:
    label = row["label"]
    prediction = row["prediction"]
    boundary_error = int(bool({label, prediction} & {"hard_wall"} and {label, prediction} & {"door", "window"}))
    wrong = int(not row["correct"])
    fragile = int(label in {"door", "window"} or prediction in {"door", "window"})
    return (-wrong, -boundary_error, -fragile, row["confidence"], row["image"], row["id"])


def build_summary(rows: list[dict[str, Any]], hard_cases: list[dict[str, Any]], args: argparse.Namespace) -> dict[str, Any]:
    errors = [row for row in rows if not row["correct"]]
    low_conf = [row for row in rows if row["confidence"] < args.low_confidence_threshold]
    by_pair = Counter(f"{row['label']}->{row['prediction']}" for row in errors)
    by_image = defaultdict(lambda: {"records": 0, "errors": 0, "hard_cases": 0})
    hard_keys = {(row["image"], row["id"]) for row in hard_cases}
    for row in rows:
        bucket = by_image[row["image"]]
        bucket["records"] += 1
        bucket["errors"] += int(not row["correct"])
        bucket["hard_cases"] += int((row["image"], row["id"]) in hard_keys)
    return {
        "dataset": args.dataset,
        "predictions": args.predictions,
        "target_source": args.target_source,
        "fragile_labels": sorted({item.strip() for item in args.fragile_labels.split(",") if item.strip()}),
        "low_confidence_threshold": args.low_confidence_threshold,
        "records": len(rows),
        "errors": len(errors),
        "low_confidence_records": len(low_conf),
        "exported_hard_cases": len(hard_cases),
        "error_pairs": dict(by_pair.most_common()),
        "worst_images": sorted(
            [
                {"image": image, **counts, "error_rate": round(counts["errors"] / counts["records"], 6)}
                for image, counts in by_image.items()
                if counts["errors"] or counts["hard_cases"]
            ],
            key=lambda item: (item["errors"], item["hard_cases"], item["error_rate"]),
            reverse=True,
        )[:20],
    }


def compact_case(row: dict[str, Any]) -> dict[str, Any]:
    features = row.get("features") or {}
    return {
        "image": row["image"],
        "source_dataset": row["source_dataset"],
        "node_id": row["id"],
        "label": row["label"],
        "prediction": row["prediction"],
        "confidence": round(row["confidence"], 6),
        "correct": row["correct"],
        "bbox": row.get("bbox"),
        "primitive_type": row.get("primitive_type"),
        "orientation": row.get("orientation"),
        "node_count": row.get("node_count"),
        "edge_count": row.get("edge_count"),
        "raster_dark_density": round(float(features.get("raster_dark_density", 0.0) or 0.0), 6),
        "raster_edge_density": round(float(features.get("raster_edge_density", 0.0) or 0.0), 6),
        "relation_contains": float(features.get("relation_contains", 0.0) or 0.0),
        "relation_contained_in": float(features.get("relation_contained_in", 0.0) or 0.0),
        "graph_degree": float(features.get("graph_degree", 0.0) or 0.0),
    }


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


if __name__ == "__main__":
    main()
