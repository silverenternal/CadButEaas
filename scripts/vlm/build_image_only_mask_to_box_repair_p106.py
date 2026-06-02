#!/usr/bin/env python3
"""P106 non-oracle mask-to-box repair for image-only room/symbol nodes.

This script uses predicted raster masks/probability maps only. For rooms it can
subtract predicted wall/opening/window/room-boundary masks before connected
components; for symbols it uses high-threshold component extraction with area
filters. Gold masks/boxes are not read at runtime.
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


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.open("r", encoding="utf-8") if line.strip()]


def read_png_gray(path: Path) -> tuple[int, int, list[bytearray]]:
    data = path.read_bytes()
    if not data.startswith(b"\x89PNG\r\n\x1a\n"):
        raise ValueError(f"not PNG: {path}")
    pos = 8
    width = height = bit_depth = color_type = None
    chunks: list[bytes] = []
    while pos < len(data):
        length = struct.unpack(">I", data[pos:pos+4])[0]
        ctype = data[pos+4:pos+8]
        chunk = data[pos+8:pos+8+length]
        pos += 12 + length
        if ctype == b"IHDR":
            width, height, bit_depth, color_type, _, _, _ = struct.unpack(">IIBBBBB", chunk)
        elif ctype == b"IDAT":
            chunks.append(chunk)
        elif ctype == b"IEND":
            break
    if width is None or height is None or bit_depth != 8 or color_type not in {0, 2, 6}:
        raise ValueError(f"unsupported PNG {path}: bit_depth={bit_depth}, color_type={color_type}")
    channels = {0: 1, 2: 3, 6: 4}[int(color_type)]
    raw = zlib.decompress(b"".join(chunks))
    stride = int(width) * channels
    rows: list[bytearray] = []
    prev = bytearray(stride)
    idx = 0
    for _ in range(int(height)):
        filt = raw[idx]
        idx += 1
        scan = bytearray(raw[idx:idx+stride])
        idx += stride
        recon = bytearray(stride)
        for i, val in enumerate(scan):
            left = recon[i-channels] if i >= channels else 0
            up = prev[i]
            up_left = prev[i-channels] if i >= channels else 0
            if filt == 0:
                out = val
            elif filt == 1:
                out = (val + left) & 255
            elif filt == 2:
                out = (val + up) & 255
            elif filt == 3:
                out = (val + ((left + up) // 2)) & 255
            elif filt == 4:
                p = left + up - up_left
                pa, pb, pc = abs(p-left), abs(p-up), abs(p-up_left)
                pred = left if pa <= pb and pa <= pc else up if pb <= pc else up_left
                out = (val + pred) & 255
            else:
                raise ValueError(f"unsupported PNG filter {filt}: {path}")
            recon[i] = out
        rows.append(recon if channels == 1 else bytearray(recon[i] for i in range(0, stride, channels)))
        prev = recon
    return int(width), int(height), rows


def load_mask(path_value: str | None) -> list[bytearray] | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return None
    return read_png_gray(path)[2]


def empty_like(rows: list[bytearray]) -> list[bytearray]:
    return [bytearray(len(row)) for row in rows]


def combine_room(room: list[bytearray], blockers: list[list[bytearray]], room_th: int, block_th: int) -> list[bytearray]:
    h = len(room)
    out = empty_like(room)
    for y in range(h):
        w = len(room[y])
        for x in range(w):
            if room[y][x] < room_th:
                continue
            blocked = False
            for block in blockers:
                if y < len(block) and x < len(block[y]) and block[y][x] >= block_th:
                    blocked = True
                    break
            if not blocked:
                out[y][x] = 255
    return out


def threshold_mask(rows: list[bytearray], threshold: int) -> list[bytearray]:
    return [bytearray(255 if value >= threshold else 0 for value in row) for row in rows]


def component_boxes(rows: list[bytearray], min_area: int, max_area: int, min_side: int, max_side: int, max_components: int) -> list[dict[str, Any]]:
    h = len(rows)
    w = len(rows[0]) if h else 0
    seen = [bytearray(w) for _ in range(h)]
    boxes: list[dict[str, Any]] = []
    for y in range(h):
        for x in range(w):
            if seen[y][x] or rows[y][x] == 0:
                continue
            q = deque([(x, y)])
            seen[y][x] = 1
            min_x = max_x = x
            min_y = max_y = y
            area = 0
            while q:
                cx, cy = q.popleft()
                area += 1
                min_x, max_x = min(min_x, cx), max(max_x, cx)
                min_y, max_y = min(min_y, cy), max(max_y, cy)
                for nx, ny in ((cx+1, cy), (cx-1, cy), (cx, cy+1), (cx, cy-1)):
                    if 0 <= nx < w and 0 <= ny < h and not seen[ny][nx] and rows[ny][nx] > 0:
                        seen[ny][nx] = 1
                        q.append((nx, ny))
            bw = max_x - min_x + 1
            bh = max_y - min_y + 1
            if area < min_area or area > max_area or bw < min_side or bh < min_side or bw > max_side or bh > max_side:
                continue
            boxes.append({"bbox": [float(min_x), float(min_y), float(max_x+1), float(max_y+1)], "area": area, "width": bw, "height": bh})
    boxes.sort(key=lambda item: item["area"], reverse=True)
    return boxes[:max_components]


def grid_split_large(boxes: list[dict[str, Any]], max_side: int, min_side: int) -> list[dict[str, Any]]:
    split: list[dict[str, Any]] = []
    for box in boxes:
        x1, y1, x2, y2 = box["bbox"]
        width = int(x2 - x1)
        height = int(y2 - y1)
        nx = max(1, round(width / max_side))
        ny = max(1, round(height / max_side))
        if nx == 1 and ny == 1:
            split.append(box)
            continue
        cell_w = width / nx
        cell_h = height / ny
        for iy in range(ny):
            for ix in range(nx):
                cx1 = x1 + ix * cell_w
                cy1 = y1 + iy * cell_h
                cx2 = x1 + (ix + 1) * cell_w
                cy2 = y1 + (iy + 1) * cell_h
                if cx2 - cx1 >= min_side and cy2 - cy1 >= min_side:
                    split.append({"bbox": [cx1, cy1, cx2, cy2], "area": int((cx2-cx1)*(cy2-cy1)), "width": cx2-cx1, "height": cy2-cy1, "split_from": box["bbox"]})
    return split


def node(row_id: str, family: str, semantic_type: str, box: dict[str, Any], index: int, source: str) -> dict[str, Any]:
    return {
        "id": f"{row_id}_p106_{family}_{index:05d}",
        "family": family,
        "semantic_type": semantic_type,
        "confidence": 0.5,
        "geometry": {"bbox": box["bbox"]},
        "source_expert": source,
        "metadata": {"component_area": box.get("area"), "runtime_features": ["predicted_raster_masks"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="reports/vlm/image_only_multitask_proposal_v15_locked_predictions.jsonl")
    parser.add_argument("--base", default="reports/vlm/image_only_moe_predictions_v15.jsonl")
    parser.add_argument("--output", default="reports/vlm/image_only_moe_predictions_v15_p106_mask_to_box.jsonl")
    parser.add_argument("--room-threshold", type=int, default=160)
    parser.add_argument("--block-threshold", type=int, default=96)
    parser.add_argument("--symbol-threshold", type=int, default=192)
    parser.add_argument("--room-max-side", type=int, default=96)
    parser.add_argument("--room-min-area", type=int, default=40)
    parser.add_argument("--symbol-min-area", type=int, default=2)
    parser.add_argument("--symbol-max-area", type=int, default=500)
    parser.add_argument("--max-room-components", type=int, default=200)
    parser.add_argument("--max-symbol-components", type=int, default=300)
    args = parser.parse_args()

    source_rows = {str(row.get("id")): row for row in load_jsonl(Path(args.source))}
    base_rows = load_jsonl(Path(args.base))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    added = {"space": 0, "symbol": 0}
    with out.open("w", encoding="utf-8") as handle:
        for row in base_rows:
            row_id = str(row.get("id") or "")
            source = source_rows.get(row_id, {})
            probs = source.get("pred_probs") or {}
            nodes = [n for n in ((row.get("scene_graph") or {}).get("nodes") or []) if str(n.get("family")) not in {"space", "symbol"}]

            room = load_mask(probs.get("room"))
            if room is not None:
                blockers = [mask for name in ["wall", "opening", "window", "room_boundary"] for mask in [load_mask(probs.get(name))] if mask is not None]
                room_binary = combine_room(room, blockers, args.room_threshold, args.block_threshold)
                room_boxes = component_boxes(room_binary, args.room_min_area, 256*256, 4, 256, args.max_room_components)
                room_boxes = grid_split_large(room_boxes, args.room_max_side, 8)[:args.max_room_components]
                for index, box in enumerate(room_boxes):
                    nodes.append(node(row_id, "space", "room", box, index, "image_only_room_mask_to_box_p106"))
                added["space"] += len(room_boxes)

            symbol = load_mask(probs.get("symbol"))
            if symbol is not None:
                symbol_binary = threshold_mask(symbol, args.symbol_threshold)
                symbol_boxes = component_boxes(symbol_binary, args.symbol_min_area, args.symbol_max_area, 1, 64, args.max_symbol_components)
                for index, box in enumerate(symbol_boxes):
                    nodes.append(node(row_id, "symbol", "symbol", box, index, "image_only_symbol_mask_to_box_p106"))
                added["symbol"] += len(symbol_boxes)

            row.setdefault("scene_graph", {})["nodes"] = nodes
            row.setdefault("route_trace", {})["p106_mask_to_box_repair"] = vars(args) | {"runtime_gold_used": False, "added_counts_so_far": dict(added)}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(out), "added": added}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
