#!/usr/bin/env python3
"""Convert existing CubiCasa host_links into offline assignment supervision.

This script intentionally does not create runtime detector candidates or new
relation edges. It materializes training labels from the already converted
CubiCasa host_links and, when possible, aligns them to the image-only v16 split
ids used by the v18 scored-row cache.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SYMBOL_ROOT = ROOT / "datasets/cadstruct_symbols_v1"
DEFAULT_IMAGE_ONLY_ROOT = ROOT / "datasets/image_only_structured_targets_v16"
DEFAULT_OUTPUT_ROOT = ROOT / "datasets/external_supervision/cubicasa_contains_symbol_assignment_v18"
DEFAULT_AUDIT = ROOT / "reports/vlm/cubicasa_host_link_assignment_supervision_v18_audit.json"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True) + "\n")


def source_key_from_path(value: str) -> str:
    path = Path(value)
    if path.name == "model.svg" and path.parent.name:
        return path.parent.name
    if path.name.startswith("F1_") and path.parent.name:
        return path.parent.name
    return ""


def bbox(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    try:
        x1, y1, x2, y2 = [float(item) for item in value[:4]]
    except (TypeError, ValueError):
        return None
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def center(box: list[float]) -> tuple[float, float]:
    return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)


def area(box: list[float]) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def iou(left: list[float] | None, right: list[float] | None) -> float:
    if left is None or right is None:
        return 0.0
    ix1 = max(left[0], right[0])
    iy1 = max(left[1], right[1])
    ix2 = min(left[2], right[2])
    iy2 = min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    union = area(left) + area(right) - inter
    return inter / max(union, 1e-9)


def inside_ratio(inner: list[float] | None, outer: list[float] | None) -> float:
    if inner is None or outer is None:
        return 0.0
    ix1 = max(inner[0], outer[0])
    iy1 = max(inner[1], outer[1])
    ix2 = min(inner[2], outer[2])
    iy2 = min(inner[3], outer[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return inter / max(area(inner), 1e-9)


def scaled_box(box: list[float] | None, src_size: tuple[float, float], dst_size: tuple[float, float]) -> list[float] | None:
    if box is None:
        return None
    sw, sh = src_size
    dw, dh = dst_size
    if sw <= 0 or sh <= 0 or dw <= 0 or dh <= 0:
        return None
    return [box[0] * dw / sw, box[1] * dh / sh, box[2] * dw / sw, box[3] * dh / sh]


def load_image_only_index(root: Path) -> dict[str, dict[str, Any]]:
    index: dict[str, dict[str, Any]] = {}
    for split in ["train", "dev", "locked", "smoke"]:
        for row in load_jsonl(root / f"{split}.jsonl"):
            source_key = str(row.get("source_key") or "")
            if source_key and source_key not in index:
                index[source_key] = row
    return index


def page_source_size(page: dict[str, Any], image_only_row: dict[str, Any] | None) -> tuple[float, float] | None:
    metadata = page.get("metadata") if isinstance(page.get("metadata"), dict) else {}
    width = float(metadata.get("width") or 0.0)
    height = float(metadata.get("height") or 0.0)
    if width > 0 and height > 0:
        return width, height
    if image_only_row is not None:
        original_size = image_only_row.get("original_image_size")
        if isinstance(original_size, list) and len(original_size) >= 2:
            width = float(original_size[0] or 0.0)
            height = float(original_size[1] or 0.0)
            if width > 0 and height > 0:
                return width, height
    max_x = 0.0
    max_y = 0.0
    for item in list(page.get("rooms") or []) + list(page.get("symbols") or []):
        item_box = bbox(item.get("bbox")) if isinstance(item, dict) else None
        if item_box is None:
            continue
        max_x = max(max_x, item_box[2])
        max_y = max(max_y, item_box[3])
    if max_x > 0 and max_y > 0:
        return max_x, max_y
    return None


def nearest_items(
    items: list[dict[str, Any]],
    target_box: list[float],
    source_size: tuple[float, float],
    image_only_row: dict[str, Any] | None,
    *,
    limit: int,
) -> list[dict[str, Any]]:
    if limit <= 0:
        return []
    target_center = center(target_box)
    scored = []
    for item in items:
        item_box = bbox(item.get("bbox"))
        if item_box is None:
            continue
        if image_only_row is not None:
            dst = tuple(float(x) for x in (image_only_row.get("image_size") or [512, 512])[:2])
            item_box_for_score = scaled_box(item_box, source_size, dst) or item_box
            target_box_for_score = scaled_box(target_box, source_size, dst) or target_box
        else:
            item_box_for_score = item_box
            target_box_for_score = target_box
        icx, icy = center(item_box_for_score)
        tcx, tcy = center(target_box_for_score)
        diag = math.sqrt(max(area(item_box_for_score), 1.0))
        distance = math.sqrt((icx - tcx) ** 2 + (icy - tcy) ** 2) / max(diag, 1.0)
        scored.append((inside_ratio(target_box_for_score, item_box_for_score), -distance, item))
    scored.sort(reverse=True, key=lambda value: (value[0], value[1]))
    return [item for _inside, _distance, item in scored[:limit]]


def best_aligned_id(
    raw_item: dict[str, Any] | None,
    aligned_items: list[dict[str, Any]],
    source_size: tuple[float, float],
    image_only_row: dict[str, Any] | None,
) -> tuple[str | None, float]:
    if raw_item is None or image_only_row is None:
        return None, 0.0
    raw_box = bbox(raw_item.get("bbox"))
    if raw_box is None:
        return None, 0.0
    dst = tuple(float(x) for x in (image_only_row.get("image_size") or [512, 512])[:2])
    raw_scaled = scaled_box(raw_box, source_size, dst)
    best_id: str | None = None
    best_score = 0.0
    for item in aligned_items:
        item_box = bbox(item.get("bbox"))
        score = iou(raw_scaled, item_box)
        if score > best_score:
            best_score = score
            best_id = str(item.get("id") or "")
    return best_id, best_score


def split_for_row(symbol_split: str, source_key: str, image_only: dict[str, dict[str, Any]]) -> str:
    row = image_only.get(source_key)
    if row is not None:
        split = str(row.get("split") or "")
        if split == "locked":
            return "locked"
        if split in {"train", "dev"}:
            return split
    return "dev" if symbol_split == "dev" else "train"


def build_rows(
    symbol_root: Path,
    image_only_root: Path,
    negatives_per_positive: int,
) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    image_only = load_image_only_index(image_only_root)
    by_split: dict[str, list[dict[str, Any]]] = {"train": [], "dev": [], "locked": []}
    counts = Counter()
    align_counts = Counter()
    leakage_sources: dict[str, set[str]] = defaultdict(set)

    for symbol_split in ["train", "dev", "smoke"]:
        for page in load_jsonl(symbol_root / f"{symbol_split}.jsonl"):
            annotation = str(page.get("annotation") or "")
            source_key = source_key_from_path(annotation) or source_key_from_path(str(page.get("image") or ""))
            if not source_key:
                counts["missing_source_key_pages"] += 1
                continue
            out_split = split_for_row(symbol_split, source_key, image_only)
            if out_split not in by_split:
                continue
            leakage_sources[source_key].add(out_split)
            img_row = image_only.get(source_key)
            structured = img_row.get("structured") if isinstance(img_row, dict) else {}
            aligned_rooms = structured.get("rooms") if isinstance(structured, dict) and isinstance(structured.get("rooms"), list) else []
            aligned_symbols = structured.get("symbols") if isinstance(structured, dict) and isinstance(structured.get("symbols"), list) else []
            source_size = page_source_size(page, img_row)
            if source_size is None:
                counts["invalid_page_size"] += 1
                continue
            rooms = {str(item.get("id")): item for item in page.get("rooms") or []}
            symbols = {str(item.get("id")): item for item in page.get("symbols") or []}
            positive_pairs = {
                (str(link.get("source")), str(link.get("target")))
                for link in page.get("host_links") or []
                if link.get("relation") in {None, "contains"}
            }
            counts["pages"] += 1
            counts[f"pages_{out_split}"] += 1
            counts["positive_links_seen"] += len(positive_pairs)

            positive_by_symbol: dict[str, set[str]] = defaultdict(set)
            link_payload: dict[tuple[str, str], dict[str, Any]] = {}
            for link in page.get("host_links") or []:
                if link.get("relation") not in {None, "contains"}:
                    continue
                room_id = str(link.get("source"))
                symbol_id = str(link.get("target"))
                if room_id not in rooms or symbol_id not in symbols:
                    counts["host_link_missing_endpoint"] += 1
                    continue
                positive_by_symbol[symbol_id].add(room_id)
                link_payload[(room_id, symbol_id)] = link

            emitted_pairs: set[tuple[str, str]] = set()
            for symbol_id, positive_rooms in positive_by_symbol.items():
                symbol = symbols.get(symbol_id)
                symbol_box = bbox(symbol.get("bbox")) if isinstance(symbol, dict) else None
                if symbol is None or symbol_box is None:
                    counts["positive_missing_symbol_bbox"] += len(positive_rooms)
                    continue
                hard_negative_rooms = [
                    item for item in nearest_items(list(rooms.values()), symbol_box, source_size, img_row, limit=len(positive_rooms) * max(negatives_per_positive + 1, 1))
                    if str(item.get("id")) not in positive_rooms
                ][: len(positive_rooms) * negatives_per_positive]
                for room_id in sorted(positive_rooms):
                    room = rooms.get(room_id)
                    if room is None:
                        continue
                    emitted_pairs.add((room_id, symbol_id))
                    link = link_payload.get((room_id, symbol_id), {})
                    row = make_row(
                        page=page,
                        split=out_split,
                        source_key=source_key,
                        image_only_row=img_row,
                        source_size=source_size,
                        room=room,
                        symbol=symbol,
                        aligned_rooms=aligned_rooms,
                        aligned_symbols=aligned_symbols,
                        label=1,
                        negative_rank=None,
                        link=link,
                    )
                    by_split[out_split].append(row)
                    counts["rows_positive"] += 1
                    align_counts.update(row.get("alignment", {}).get("counters", {}))
                for rank, room in enumerate(hard_negative_rooms, start=1):
                    room_id = str(room.get("id"))
                    if (room_id, symbol_id) in emitted_pairs:
                        continue
                    emitted_pairs.add((room_id, symbol_id))
                    row = make_row(
                        page=page,
                        split=out_split,
                        source_key=source_key,
                        image_only_row=img_row,
                        source_size=source_size,
                        room=room,
                        symbol=symbol,
                        aligned_rooms=aligned_rooms,
                        aligned_symbols=aligned_symbols,
                        label=0,
                        negative_rank=rank,
                        link={},
                    )
                    by_split[out_split].append(row)
                    counts["rows_negative"] += 1
                    align_counts.update(row.get("alignment", {}).get("counters", {}))

    leakage = {source: sorted(splits) for source, splits in leakage_sources.items() if len(splits) > 1}
    for split, rows in by_split.items():
        counts[f"rows_{split}"] = len(rows)
        counts[f"positives_{split}"] = sum(1 for row in rows if row["label"] == 1)
        counts[f"negatives_{split}"] = sum(1 for row in rows if row["label"] == 0)
    audit = {
        "task": "IMG-MOE-V18-REBUILD-006.step_cubicasa_host_link_assignment_supervision",
        "symbol_root": str(symbol_root),
        "image_only_root": str(image_only_root),
        "counts": dict(counts),
        "alignment_counts": dict(align_counts),
        "negative_sampling": {
            "method": "nearest_room_hard_negatives_for_offline_training_only",
            "negatives_per_positive": negatives_per_positive,
            "runtime_candidates_created": False,
        },
        "split_leakage_sources": leakage,
        "split_leakage_count": len(leakage),
        "source_integrity": {
            "source_mode": "offline_cubicasa_host_link_supervision",
            "model_input": "labels_only_not_runtime_input",
            "svg_candidate_ids_used": False,
            "annotation_geometry_used_at_inference": False,
            "gold_used_for_inference": False,
        },
    }
    return by_split, audit


def make_row(
    *,
    page: dict[str, Any],
    split: str,
    source_key: str,
    image_only_row: dict[str, Any] | None,
    source_size: tuple[float, float],
    room: dict[str, Any],
    symbol: dict[str, Any],
    aligned_rooms: list[dict[str, Any]],
    aligned_symbols: list[dict[str, Any]],
    label: int,
    negative_rank: int | None,
    link: dict[str, Any],
) -> dict[str, Any]:
    room_box = bbox(room.get("bbox"))
    symbol_box = bbox(symbol.get("bbox"))
    dst_size = tuple(float(x) for x in ((image_only_row or {}).get("image_size") or [512, 512])[:2])
    room_scaled = scaled_box(room_box, source_size, dst_size)
    symbol_scaled = scaled_box(symbol_box, source_size, dst_size)
    aligned_room_id, room_iou = best_aligned_id(room, aligned_rooms, source_size, image_only_row)
    aligned_symbol_id, symbol_iou = best_aligned_id(symbol, aligned_symbols, source_size, image_only_row)
    counters = {
        "image_only_aligned_rows": int(image_only_row is not None),
        "room_alignment_iou_ge_050": int(room_iou >= 0.5),
        "symbol_alignment_iou_ge_050": int(symbol_iou >= 0.5),
    }
    relation_key = "|".join([source_key, str(room.get("id")), str(symbol.get("id"))])
    return {
        "schema_version": "cubicasa_contains_symbol_assignment_supervision_v18",
        "split": split,
        "source_dataset": "cubicasa5k",
        "source_key": source_key,
        "image_only_row_id": (image_only_row or {}).get("id"),
        "image": (image_only_row or {}).get("image") or page.get("image"),
        "original_image": (image_only_row or {}).get("original_image") or page.get("image"),
        "annotation": page.get("annotation"),
        "relation_key": relation_key,
        "label": int(label),
        "label_name": "contains_symbol_positive" if label else "hard_negative_room_symbol_pair",
        "room": {
            "id": room.get("id"),
            "room_type": room.get("room_type") or room.get("semantic_type"),
            "bbox": room_box,
            "bbox_512": room_scaled,
            "aligned_image_only_id": aligned_room_id,
            "alignment_iou": round(room_iou, 6),
        },
        "symbol": {
            "id": symbol.get("id"),
            "symbol_type": symbol.get("symbol_type") or symbol.get("semantic_type"),
            "bbox": symbol_box,
            "bbox_512": symbol_scaled,
            "aligned_image_only_id": aligned_symbol_id,
            "alignment_iou": round(symbol_iou, 6),
        },
        "assignment_features_offline": {
            "symbol_inside_room_ratio": round(inside_ratio(symbol_scaled, room_scaled), 6),
            "room_symbol_iou": round(iou(room_scaled, symbol_scaled), 6),
            "negative_rank": negative_rank,
        },
        "host_link": {
            "room_type": link.get("room_type"),
            "symbol_type": link.get("symbol_type"),
            "relation": link.get("relation") or ("contains" if label else "not_contains"),
        },
        "alignment": {
            "counters": counters,
            "contract": "alignment is for offline supervision/audit only; no runtime candidate or relation edge is created",
        },
        "source_integrity": {
            "source_mode": "offline_cubicasa_host_link_supervision",
            "model_input": "labels_only_not_runtime_input",
            "svg_candidate_ids_used": False,
            "annotation_geometry_used_at_inference": False,
            "gold_used_for_inference": False,
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--symbol-root", default=str(DEFAULT_SYMBOL_ROOT))
    parser.add_argument("--image-only-root", default=str(DEFAULT_IMAGE_ONLY_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--audit-output", default=str(DEFAULT_AUDIT))
    parser.add_argument("--negatives-per-positive", type=int, default=4)
    args = parser.parse_args()

    output_root = Path(args.output_root)
    by_split, audit = build_rows(Path(args.symbol_root), Path(args.image_only_root), args.negatives_per_positive)
    for split in ["train", "dev", "locked"]:
        write_jsonl(output_root / f"{split}.jsonl", by_split.get(split, []))
    manifest = {
        "schema_version": "cubicasa_contains_symbol_assignment_supervision_v18",
        "dataset": str(output_root),
        "splits": {split: len(by_split.get(split, [])) for split in ["train", "dev", "locked"]},
        "positive_rows": {split: sum(1 for row in by_split.get(split, []) if row["label"] == 1) for split in ["train", "dev", "locked"]},
        "negative_rows": {split: sum(1 for row in by_split.get(split, []) if row["label"] == 0) for split in ["train", "dev", "locked"]},
        "inference_contract": "offline labels only; raster-only runtime contract unchanged; no new runtime candidates or relation edges",
        "audit": str(Path(args.audit_output)),
    }
    write_json(output_root / "manifest.json", manifest)
    audit["output_root"] = str(output_root)
    audit["manifest"] = manifest
    write_json(Path(args.audit_output), audit)
    print(json.dumps({"output_root": str(output_root), "splits": manifest["splits"], "audit": args.audit_output}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
