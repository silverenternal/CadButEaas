#!/usr/bin/env python3
"""P0-48: build focus detector train view for stair/tiny symbol proposal pivot."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from build_symbol_seg_hardcase_train_view_v29 import NAMES, image_to_label, parse_simple_yaml, polygon_bbox_area_bucket, resolve_data_path, write_yaml
from train_symbol_tile_detector_v20 import write_json

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BASE = ROOT / "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_seg_rect_v27/data.yaml"
DEFAULT_OUT = ROOT / "datasets/symbol_focus_detector_v48"
FOCUS_LABELS = {"stair", "sink", "shower", "equipment"}
FOCUS_AREAS = {"tiny_le_64", "small_le_256"}


def label_profile(image: Path, image_size: int) -> tuple[Counter[str], Counter[str], Counter[str]]:
    labels = Counter()
    areas = Counter()
    pairs = Counter()
    label_path = image_to_label(image)
    if not label_path.exists():
        return labels, areas, pairs
    for raw in label_path.read_text(encoding="utf-8").splitlines():
        parts = raw.split()
        if len(parts) < 9:
            continue
        cls_id = int(float(parts[0]))
        label = NAMES.get(cls_id, str(cls_id))
        area = polygon_bbox_area_bucket(parts, image_size)
        labels[label] += 1
        areas[area] += 1
        pairs[f"{label}/{area}"] += 1
    return labels, areas, pairs


def focus_score(labels: Counter[str], areas: Counter[str], pairs: Counter[str]) -> tuple[int, dict[str, int]]:
    score = 0
    hits = Counter()
    for label in FOCUS_LABELS:
        if labels[label]:
            weight = 4 if label == "stair" else 2
            score += weight * labels[label]
            hits[f"label:{label}"] += labels[label]
    for area in FOCUS_AREAS:
        if areas[area]:
            score += 2 * areas[area]
            hits[f"area:{area}"] += areas[area]
    for label in FOCUS_LABELS:
        for area in FOCUS_AREAS:
            key = f"{label}/{area}"
            if pairs[key]:
                score += 6 * pairs[key]
                hits[f"pair:{key}"] += pairs[key]
    return score, dict(hits)


def leakage_audit(train_rows: list[str], val_path: Path, test_path: Path) -> dict[str, object]:
    train_set = {str(Path(x).resolve()) for x in train_rows}
    val_set = {str((val_path / p.name).resolve()) for p in val_path.glob("*.jpg")} if val_path.exists() else set()
    test_set = {str((test_path / p.name).resolve()) for p in test_path.glob("*.jpg")} if test_path.exists() else set()
    return {
        "train_rows": len(train_rows),
        "unique_train_images": len(train_set),
        "val_overlap_count": len(train_set & val_set),
        "test_overlap_count": len(train_set & test_set),
        "passes": len(train_set & val_set) == 0 and len(train_set & test_set) == 0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-data", default=str(DEFAULT_BASE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUT))
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--target-multiplier", type=float, default=2.0)
    parser.add_argument("--max-repeat", type=int, default=6)
    args = parser.parse_args()

    base_yaml = Path(args.base_data).resolve()
    out = Path(args.output_dir).resolve()
    out.mkdir(parents=True, exist_ok=True)
    base = parse_simple_yaml(base_yaml)
    train_txt = resolve_data_path(base_yaml, base["train"])
    val_dir = resolve_data_path(base_yaml, base["val"])
    test_dir = resolve_data_path(base_yaml, base["test"])
    train_images = [Path(line.strip()) for line in train_txt.read_text(encoding="utf-8").splitlines() if line.strip()]

    scored = []
    counts = Counter()
    hit_hist = Counter()
    for idx, image in enumerate(train_images):
        labels, areas, pairs = label_profile(image, args.image_size)
        score, hits = focus_score(labels, areas, pairs)
        scored.append((score, idx, image, hits, dict(labels), dict(areas), dict(pairs)))
        counts["base_train_images"] += 1
        for label, count in labels.items():
            counts[f"base_label:{label}"] += count
        for area, count in areas.items():
            counts[f"base_area:{area}"] += count
        for pair, count in pairs.items():
            if pair.split("/")[0] in FOCUS_LABELS or pair.split("/")[1] in FOCUS_AREAS:
                counts[f"base_pair:{pair}"] += count

    target_rows = int(round(len(train_images) * args.target_multiplier))
    extra_budget = max(0, target_rows - len(train_images))
    rows = [str(image) for image in train_images]
    extras_by_idx = Counter()
    selected_examples = []
    for score, idx, image, hits, labels, areas, pairs in sorted(scored, key=lambda item: item[0], reverse=True):
        if extra_budget <= 0 or score <= 0:
            break
        extra = min(args.max_repeat - 1, extra_budget, max(1, score // 8))
        rows.extend([str(image)] * extra)
        extras_by_idx[idx] += extra
        extra_budget -= extra
        hit_hist.update(hits)
        if len(selected_examples) < 30:
            selected_examples.append({"image": str(image), "score": score, "extra": extra, "hits": hits, "labels": labels, "areas": areas})

    repeat_hist = Counter(1 + extras_by_idx[idx] for idx in range(len(train_images)))
    out_train = out / "train_focus_v48.txt"
    out_train.write_text("\n".join(rows) + ("\n" if rows else ""), encoding="utf-8")
    write_yaml(out, out_train, base_yaml)
    audit = leakage_audit(rows, val_dir, test_dir)
    manifest = {
        "version": "symbol_focus_detector_train_view_p048",
        "dataset": str(out.relative_to(ROOT) if out.is_relative_to(ROOT) else out),
        "base_data": str(base_yaml.relative_to(ROOT) if base_yaml.is_relative_to(ROOT) else base_yaml),
        "train_txt": str(out_train.relative_to(ROOT) if out_train.is_relative_to(ROOT) else out_train),
        "data_yaml": str((out / "data.yaml").relative_to(ROOT) if (out / "data.yaml").is_relative_to(ROOT) else out / "data.yaml"),
        "focus_labels": sorted(FOCUS_LABELS),
        "focus_areas": sorted(FOCUS_AREAS),
        "counts": dict(counts),
        "view": {
            "base_train_images": len(train_images),
            "train_rows": len(rows),
            "target_multiplier": args.target_multiplier,
            "max_repeat": args.max_repeat,
            "repeat_hist": dict(sorted(repeat_hist.items())),
            "hit_hist": dict(hit_hist.most_common()),
            "selected_examples": selected_examples,
        },
        "leakage_audit": audit,
        "stage_gate": {
            "data_integrity_no_val_test_overlap": audit["passes"],
            "ready_for_training": audit["passes"] and len(rows) > len(train_images),
            "requires_smoke_eval_before_dev": True,
        },
        "source_integrity": {
            "claim_boundary": "Train rows are base train-split images only; val/test paths are referenced only by data.yaml for evaluation, not duplicated into train_focus_v48.txt.",
            "gold_used_for_inference": False,
            "runtime_uses_svg_or_cad_geometry": False,
            "offline_labels_used_for": ["train view weighting", "leakage audit"],
        },
    }
    write_json(out / "manifest.json", manifest)
    write_json(ROOT / "reports/vlm/symbol_focus_detector_train_view_p048_audit.json", manifest)
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
