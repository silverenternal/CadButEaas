#!/usr/bin/env python3
"""Build non-oracle image-only nodes from predicted raster masks.

This P104 policy reads `pred_masks` / `pred_probs` PNGs from the v15 image-only
multitask output and extracts connected-component boxes. It does not read gold
boxes, target ids, or SVG geometry at runtime.
"""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from collections import deque
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
MASK_TO_FAMILY_LABEL = {
    "wall": ("boundary", "wall"),
    "opening": ("boundary", "opening"),
    "window": ("boundary", "window"),
    "room": ("space", "room"),
    "symbol": ("symbol", "symbol"),
    "text": ("text", "text"),
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def read_png_gray(path: Path) -> tuple[int, int, list[bytearray]]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"not a PNG: {path}")
    pos = 8
    width = height = bit_depth = color_type = None
    chunks: list[bytes] = []
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        ctype = data[pos + 4 : pos + 8]
        chunk = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if ctype == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", chunk)
        elif ctype == b"IDAT":
            chunks.append(chunk)
        elif ctype == b"IEND":
            break
    if width is None or height is None or bit_depth != 8 or color_type not in {0, 2, 6}:
        raise ValueError(f"unsupported PNG format for {path}: bit_depth={bit_depth}, color_type={color_type}")
    channels = {0: 1, 2: 3, 6: 4}[int(color_type)]
    raw = zlib.decompress(b"".join(chunks))
    stride = int(width) * channels
    rows: list[bytearray] = []
    prev = bytearray(stride)
    idx = 0
    for _ in range(int(height)):
        filter_type = raw[idx]
        idx += 1
        scan = bytearray(raw[idx : idx + stride])
        idx += stride
        recon = bytearray(stride)
        for i, val in enumerate(scan):
            left = recon[i - channels] if i >= channels else 0
            up = prev[i]
            up_left = prev[i - channels] if i >= channels else 0
            if filter_type == 0:
                out = val
            elif filter_type == 1:
                out = (val + left) & 255
            elif filter_type == 2:
                out = (val + up) & 255
            elif filter_type == 3:
                out = (val + ((left + up) // 2)) & 255
            elif filter_type == 4:
                p = left + up - up_left
                pa, pb, pc = abs(p - left), abs(p - up), abs(p - up_left)
                pred = left if pa <= pb and pa <= pc else up if pb <= pc else up_left
                out = (val + pred) & 255
            else:
                raise ValueError(f"unsupported PNG filter {filter_type} in {path}")
            recon[i] = out
        if channels == 1:
            rows.append(recon)
        else:
            rows.append(bytearray(recon[i] for i in range(0, stride, channels)))
        prev = recon
    return int(width), int(height), rows


def components(rows: list[bytearray], threshold: int, min_area: int, max_components: int) -> list[dict[str, Any]]:
    h = len(rows)
    w = len(rows[0]) if h else 0
    seen = [bytearray(w) for _ in range(h)]
    comps: list[dict[str, Any]] = []
    for y in range(h):
        row = rows[y]
        for x in range(w):
            if seen[y][x] or row[x] < threshold:
                continue
            q = deque([(x, y)])
            seen[y][x] = 1
            min_x = max_x = x
            min_y = max_y = y
            area = 0
            score_sum = 0
            while q:
                cx, cy = q.popleft()
                area += 1
                score_sum += rows[cy][cx]
                min_x, max_x = min(min_x, cx), max(max_x, cx)
                min_y, max_y = min(min_y, cy), max(max_y, cy)
                for nx, ny in ((cx + 1, cy), (cx - 1, cy), (cx, cy + 1), (cx, cy - 1)):
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and rows[ny][nx] >= threshold:
                        seen[ny][nx] = 1
                        q.append((nx, ny))
            if area >= min_area:
                comps.append({
                    "bbox": [float(min_x), float(min_y), float(max_x + 1), float(max_y + 1)],
                    "area": int(area),
                    "mean_mask_value": round(score_sum / max(area, 1) / 255.0, 6),
                })
    comps.sort(key=lambda item: (item["area"], item["mean_mask_value"]), reverse=True)
    return comps[:max_components]


def load_mask(path: str | None) -> tuple[int, int, list[bytearray]] | None:
    if not path:
        return None
    p = Path(path)
    if not p.is_absolute():
        p = ROOT / p
    if not p.exists():
        return None
    return read_png_gray(p)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="reports/vlm/image_only_multitask_proposal_v15_locked_predictions.jsonl")
    parser.add_argument("--base", default="reports/vlm/image_only_moe_predictions_v15.jsonl")
    parser.add_argument("--output", default="reports/vlm/image_only_moe_predictions_v15_p104_mask_policy.jsonl")
    parser.add_argument("--use-prob-masks", action="store_true", help="Use pred_probs PNGs instead of binarized pred_masks.")
    parser.add_argument("--families", default="space,symbol,text")
    parser.add_argument("--threshold", type=int, default=128)
    parser.add_argument("--min-area", default="space=20,symbol=2,text=2,boundary=4")
    parser.add_argument("--max-components", default="space=80,symbol=120,text=120,boundary=200")
    args = parser.parse_args()

    selected_families = {item.strip() for item in args.families.split(",") if item.strip()}
    min_area = {k: int(v) for k, v in (item.split("=") for item in args.min_area.split(",") if item.strip())}
    max_components = {k: int(v) for k, v in (item.split("=") for item in args.max_components.split(",") if item.strip())}
    source_rows = {str(row.get("id")): row for row in load_jsonl(Path(args.source))}
    base_rows = load_jsonl(Path(args.base))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    added_by_family: dict[str, int] = {family: 0 for family in selected_families}

    with output.open("w", encoding="utf-8") as handle:
        for row in base_rows:
            row_id = str(row.get("id") or "")
            source = source_rows.get(row_id, {})
            masks = source.get("pred_probs") if args.use_prob_masks else source.get("pred_masks")
            masks = masks or {}
            nodes = [node for node in ((row.get("scene_graph") or {}).get("nodes") or []) if str(node.get("family")) not in selected_families]
            added_total = 0
            for mask_name, (family, label) in MASK_TO_FAMILY_LABEL.items():
                if family not in selected_families:
                    continue
                loaded = load_mask(masks.get(mask_name))
                if loaded is None:
                    continue
                _, _, rows = loaded
                comps = components(rows, args.threshold, min_area.get(family, 1), max_components.get(family, 100))
                for index, comp in enumerate(comps):
                    node = {
                        "id": f"{row_id}_p104_{mask_name}_{index:05d}",
                        "family": family,
                        "semantic_type": label,
                        "confidence": comp["mean_mask_value"],
                        "geometry": {"bbox": comp["bbox"]},
                        "source_expert": "image_only_mask_component_policy_p104",
                        "metadata": {
                            "mask_name": mask_name,
                            "component_area": comp["area"],
                            "threshold": args.threshold,
                            "runtime_features": ["predicted_raster_mask_pixels"],
                        },
                    }
                    nodes.append(node)
                    added_by_family[family] = added_by_family.get(family, 0) + 1
                    added_total += 1
            row.setdefault("scene_graph", {})["nodes"] = nodes
            row.setdefault("route_trace", {})["p104_mask_component_policy"] = {
                "families": sorted(selected_families),
                "threshold": args.threshold,
                "min_area": min_area,
                "max_components": max_components,
                "added_nodes": added_total,
                "runtime_gold_used": False,
                "use_prob_masks": bool(args.use_prob_masks),
                "claim_boundary": "non-oracle connected components from predicted raster masks",
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(output), "added_by_family": added_by_family}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
