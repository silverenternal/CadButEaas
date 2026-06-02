#!/usr/bin/env python3
"""Pretrain a room-space expert on CubiCasa + ResPlan mixed supervision."""

from __future__ import annotations

import argparse
import json
import pickle
import random
import resource
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
from shapely.geometry import GeometryCollection, MultiPolygon, Polygon
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder

try:
    from train_room_space_context_sklearn import (
        ENHANCED_FEATURE_NAMES,
        evaluate_predictions,
        load_jsonl,
        predict_rows,
        row_context,
        room_feature,
        write_jsonl,
        enhanced_room_feature,
    )
except ImportError:
    from scripts.vlm.train_room_space_context_sklearn import (
        ENHANCED_FEATURE_NAMES,
        evaluate_predictions,
        load_jsonl,
        predict_rows,
        row_context,
        room_feature,
        write_jsonl,
        enhanced_room_feature,
    )

ROOM_KEYS = ["living", "kitchen", "bedroom", "bathroom", "balcony", "garden", "parking", "pool", "inner"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cubi-dir", default="datasets/cadstruct_rooms_v1")
    parser.add_argument("--resplan-pkl", default="datasets/external/resplan/ResPlan.pkl")
    parser.add_argument("--output-dir", default="checkpoints/room_space_expert_v12")
    parser.add_argument("--mixed-output-dir", default="datasets/room_space_expert_v12_mixed")
    parser.add_argument("--n-estimators", type=int, default=640)
    parser.add_argument("--min-samples-leaf", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260507)
    args = parser.parse_args()

    random.seed(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    mixed_dir = Path(args.mixed_output_dir)
    mixed_dir.mkdir(parents=True, exist_ok=True)

    cubi_dir = Path(args.cubi_dir)
    train_rows = load_jsonl(cubi_dir / "train.jsonl")
    dev_rows = load_jsonl(cubi_dir / "dev.jsonl")
    smoke_rows = load_jsonl(cubi_dir / "smoke.jsonl")
    resplan_rows = build_resplan_rows(Path(args.resplan_pkl))
    resplan_train, resplan_dev, resplan_smoke = split_resplan_rows(resplan_rows, args.seed)
    mixed_train = [*train_rows, *resplan_train]
    mixed_dev = [*dev_rows, *resplan_dev]
    mixed_smoke = [*smoke_rows, *resplan_smoke]
    write_jsonl(mixed_dir / "train.jsonl", mixed_train)
    write_jsonl(mixed_dir / "dev.jsonl", mixed_dev)
    write_jsonl(mixed_dir / "smoke.jsonl", mixed_smoke)
    write_json(
        mixed_dir / "manifest.json",
        {
            "source_cubi_dir": str(cubi_dir),
            "resplan_pkl": str(args.resplan_pkl),
            "train": len(mixed_train),
            "dev": len(mixed_dev),
            "smoke": len(mixed_smoke),
            "resplan_rows": len(resplan_rows),
        },
    )

    train_items = collect_items(mixed_train)
    labels = sorted({item["label"] for item in train_items})
    encoder = LabelEncoder()
    y = encoder.fit_transform([item["label"] for item in train_items])
    x = [item["feature"] for item in train_items]

    model = ExtraTreesClassifier(
        n_estimators=args.n_estimators,
        min_samples_leaf=args.min_samples_leaf,
        class_weight="balanced",
        random_state=args.seed,
        n_jobs=-1,
    )
    model.fit(x, y)

    model_path = output_dir / "model.joblib"
    joblib.dump({"model": model, "label_encoder": encoder, "feature_names": ENHANCED_FEATURE_NAMES}, model_path)

    summary: dict[str, Any] = {
        "version": "room_space_expert_v12_train_summary",
        "model": str(model_path),
        "output_dir": str(output_dir),
        "mixed_dir": str(mixed_dir),
        "train_item_counts": dict(Counter(item["label"] for item in train_items)),
        "resplan_rows": len(resplan_rows),
        "splits": {},
    }
    for split_name, rows in [("dev", dev_rows), ("smoke", smoke_rows)]:
        predictions = predict_mixed_rows(rows, model, encoder)
        write_jsonl(output_dir / f"{split_name}_predictions.jsonl", predictions)
        metrics = evaluate_predictions(predictions)
        metrics["source_audit"] = {"cubi_rows": len(rows), "resplan_rows": len(resplan_dev) if split_name == "dev" else len(resplan_smoke)}
        summary["splits"][split_name] = metrics

    locked_rows = load_jsonl(Path("datasets/cadstruct_cubicasa5k_moe_locked_reviewed_v1/locked_test.jsonl"))
    locked_predictions = predict_mixed_rows(locked_rows, model, encoder)
    write_jsonl(output_dir / "locked_test_predictions.jsonl", locked_predictions)
    locked_metrics = evaluate_predictions(locked_predictions)
    baseline = load_json(Path("reports/vlm/room_space_expert_v3_eval.json"))
    baseline_locked = ((baseline.get("splits") or {}).get("locked_test") or {})
    summary["splits"]["locked_test"] = locked_metrics
    summary["baseline_locked_test"] = {
        "macro_f1": baseline_locked.get("macro_f1"),
        "accuracy": baseline_locked.get("accuracy"),
        "source": "reports/vlm/room_space_expert_v3_eval.json",
    }
    adopted = float(locked_metrics.get("macro_f1") or 0.0) >= float(baseline_locked.get("macro_f1") or 0.0)
    summary["adoption"] = {
        "adopted": adopted,
        "reason": "locked macro_f1 improved over v3 baseline" if adopted else "retain v3 baseline; v12 is diagnostic only",
    }
    summary["memory_audit"] = memory_audit()
    (output_dir / "train_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def build_resplan_rows(path: Path) -> list[dict[str, Any]]:
    try:
        raw = pickle.load(path.open("rb"))
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"failed to load ResPlan: {exc}")
    rows: list[dict[str, Any]] = []
    for index, record in enumerate(raw):
        width = 256.0
        height = 256.0
        rooms = []
        symbols = []
        boundaries = []
        for key in ROOM_KEYS:
            geom = record.get(key)
            if geom is None:
                continue
            for part in iter_parts(geom):
                bbox = geom_bbox(part)
                if bbox is None:
                    continue
                rooms.append(
                    {
                        "id": f"{record.get('id', index)}_{key}_{len(rooms)}",
                        "room_type": key,
                        "bbox": bbox,
                        "shape_features": geom_shape_features(part, width, height),
                    }
                )
        for key in ["wall", "door", "window"]:
            geom = record.get(key)
            if geom is None:
                continue
            for part in iter_parts(geom):
                bbox = geom_bbox(part)
                if bbox is None:
                    continue
                symbols.append(
                    {
                        "id": f"{record.get('id', index)}_{key}_{len(symbols)}",
                        "symbol_type": key if key != "wall" else "hard_wall",
                        "bbox": bbox,
                        "rotation": 0.0,
                    }
                )
                boundaries.append(
                    {
                        "id": f"{record.get('id', index)}_{key}_{len(boundaries)}",
                        "semantic_type": key if key != "wall" else "hard_wall",
                        "bbox": bbox,
                    }
                )
        rows.append(
            {
                "image": f"datasets/external/resplan/{record.get('id', index)}.png",
                "annotation": f"datasets/external/resplan/ResPlan.pkl:{record.get('id', index)}",
                "source_dataset": "resplan",
                "rooms": rooms,
                "symbols": symbols,
                "texts": [],
                "boundaries": boundaries,
                "metadata": {"width": width, "height": height, "room_count": len(rooms)},
            }
        )
    return rows


def iter_parts(geom: Any):
    if hasattr(geom, "geoms"):
        for part in geom.geoms:
            yield from iter_parts(part)
    elif isinstance(geom, (Polygon,)):
        yield geom


def geom_bbox(geom: Polygon | MultiPolygon | GeometryCollection | Any) -> list[float] | None:
    if geom is None:
        return None
    try:
        min_x, min_y, max_x, max_y = geom.bounds
    except Exception:
        return None
    if min_x == max_x or min_y == max_y:
        return None
    return [float(min_x), float(min_y), float(max_x), float(max_y)]


def geom_shape_features(geom: Any, width: float, height: float) -> dict[str, float]:
    bbox = geom_bbox(geom)
    if bbox is None:
        return {}
    area = float(getattr(geom, "area", 0.0))
    perimeter = float(getattr(geom, "length", 0.0))
    bbox_area = max(bbox[2] - bbox[0], 1e-6) * max(bbox[3] - bbox[1], 1e-6)
    point_count = 0.0
    exterior = getattr(geom, "exterior", None)
    if exterior is not None and hasattr(exterior, "coords"):
        point_count = float(len(list(exterior.coords)))
    return {
        "point_count": point_count,
        "polygon_area": area,
        "polygon_perimeter": perimeter,
        "bbox_fill_ratio": area / bbox_area,
        "compactness": (4.0 * 3.141592653589793 * area) / max(perimeter * perimeter, 1e-6),
    }


def split_resplan_rows(rows: list[dict[str, Any]], seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    rng = random.Random(seed)
    ordered = list(rows)
    rng.shuffle(ordered)
    smoke = ordered[: max(1, len(ordered) // 200)]
    dev = ordered[len(smoke) : len(smoke) + max(1, len(ordered) // 10)]
    train = ordered[len(smoke) + len(dev) :]
    return train, dev, smoke


def collect_items(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for row in rows:
        context = room_context(row)
        for room in context["rooms"]:
            feature = enhanced_room_feature(room, context)
            if feature is None:
                continue
            items.append({"id": room["id"], "label": room["room_type"], "feature": feature})
    return items


def predict_mixed_rows(rows: list[dict[str, Any]], model: ExtraTreesClassifier, encoder: LabelEncoder) -> list[dict[str, Any]]:
    predictions = []
    for row in rows:
        context = room_context(row)
        room_predictions = []
        features = []
        rooms = []
        for room in context["rooms"]:
            feature = enhanced_room_feature(room, context)
            if feature is None:
                continue
            rooms.append(room)
            features.append(feature)
        if features:
            pred_indices = model.predict(features)
            for room, pred_index in zip(rooms, pred_indices):
                room_predictions.append(
                    {
                        "id": room["id"],
                        "gold": room["room_type"],
                        "prediction": str(encoder.inverse_transform([int(pred_index)])[0]),
                        "confidence": 1.0,
                        "bbox": room["bbox"],
                        "iou": 1.0,
                    }
                )
        predictions.append(
            {
                "image": row.get("image"),
                "annotation": row.get("annotation"),
                "source_dataset": row.get("source_dataset"),
                "rooms": room_predictions,
            }
        )
    return predictions


def room_context(row: dict[str, Any]) -> dict[str, Any]:
    if isinstance(row.get("rooms"), list) and row.get("rooms"):
        rooms = []
        for index, item in enumerate(row.get("rooms") or []):
            if not isinstance(item, dict):
                continue
            bbox = item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            rooms.append(
                {
                    "id": str(item.get("id") or f"room_{index}"),
                    "room_type": str(item.get("room_type") or "room"),
                    "bbox": [float(v) for v in bbox[:4]],
                    "shape_features": item.get("shape_features") if isinstance(item.get("shape_features"), dict) else {},
                }
            )
        return {
            "width": float((row.get("metadata") or {}).get("width") or 1.0),
            "height": float((row.get("metadata") or {}).get("height") or 1.0),
            "rooms": rooms,
            "symbols": row.get("symbols") if isinstance(row.get("symbols"), list) else [],
            "texts": row.get("texts") if isinstance(row.get("texts"), list) else [],
            "boundaries": row.get("boundaries") if isinstance(row.get("boundaries"), list) else [],
            "adjacency": adjacency_from_row(row, rooms),
        }
    return row_context(row)


def adjacency_from_row(row: dict[str, Any], rooms: list[dict[str, Any]]) -> dict[str, int]:
    adjacency = {room["id"]: 0 for room in rooms}
    for edge in row.get("adjacency_edges") or []:
        if not isinstance(edge, dict):
            continue
        source = str(edge.get("source") or "")
        target = str(edge.get("target") or "")
        if source in adjacency:
            adjacency[source] += 1
        if target in adjacency:
            adjacency[target] += 1
    return adjacency


def load_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def memory_audit() -> dict[str, Any]:
    usage = resource.getrusage(resource.RUSAGE_SELF)
    return {"max_rss_kb": int(usage.ru_maxrss)}


if __name__ == "__main__":
    main()
