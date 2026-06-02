#!/usr/bin/env python3
"""Build image-only symbol boxes from predicted symbol probability peaks.

P107 is a non-oracle localization repair: it reads predicted symbol probability
maps only, extracts local maxima, emits fixed-size boxes, and applies simple NMS.
Gold boxes/masks are not runtime inputs.
"""

from __future__ import annotations

import argparse
import json
import struct
import zlib
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


def load_mask(path_value: str | None) -> tuple[int, int, list[bytearray]] | None:
    if not path_value:
        return None
    path = Path(path_value)
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return None
    return read_png_gray(path)


def iou(a: list[float], b: list[float]) -> float:
    ix1, iy1 = max(a[0], b[0]), max(a[1], b[1])
    ix2, iy2 = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    aa = max(0.0, a[2]-a[0]) * max(0.0, a[3]-a[1])
    bb = max(0.0, b[2]-b[0]) * max(0.0, b[3]-b[1])
    return inter / max(aa + bb - inter, 1e-9)


def nms(items: list[dict[str, Any]], threshold: float) -> list[dict[str, Any]]:
    kept: list[dict[str, Any]] = []
    for item in sorted(items, key=lambda x: x["score"], reverse=True):
        if all(iou(item["bbox"], kept_item["bbox"]) <= threshold for kept_item in kept):
            kept.append(item)
    return kept


def peak_boxes(rows: list[bytearray], threshold: int, radius: int, box_size: int, stride: int, max_per_image: int, nms_threshold: float) -> list[dict[str, Any]]:
    h = len(rows)
    w = len(rows[0]) if h else 0
    half = box_size / 2.0
    peaks: list[dict[str, Any]] = []
    for y in range(radius, max(radius, h - radius), max(1, stride)):
        for x in range(radius, max(radius, w - radius), max(1, stride)):
            value = rows[y][x]
            if value < threshold:
                continue
            is_peak = True
            for yy in range(y - radius, y + radius + 1):
                row = rows[yy]
                for xx in range(x - radius, x + radius + 1):
                    if row[xx] > value:
                        is_peak = False
                        break
                if not is_peak:
                    break
            if not is_peak:
                continue
            peaks.append({
                "bbox": [max(0.0, x - half), max(0.0, y - half), min(float(w), x + half), min(float(h), y + half)],
                "score": value / 255.0,
                "peak": [x, y],
            })
    return nms(peaks, nms_threshold)[:max_per_image]


def node(row_id: str, item: dict[str, Any], index: int) -> dict[str, Any]:
    return {
        "id": f"{row_id}_p107_symbol_peak_{index:05d}",
        "family": "symbol",
        "semantic_type": "symbol",
        "confidence": round(float(item["score"]), 6),
        "geometry": {"bbox": item["bbox"]},
        "source_expert": "image_only_symbol_peak_policy_p107",
        "metadata": {"peak": item["peak"], "runtime_features": ["predicted_symbol_probability_map"]},
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", default="reports/vlm/image_only_multitask_proposal_v15_locked_predictions.jsonl")
    parser.add_argument("--base", default="reports/vlm/image_only_moe_predictions_v15_p106_s80_sym192_a500.jsonl")
    parser.add_argument("--output", default="reports/vlm/image_only_moe_predictions_v15_p107_symbol_peak.jsonl")
    parser.add_argument("--threshold", type=int, default=192)
    parser.add_argument("--radius", type=int, default=5)
    parser.add_argument("--box-size", type=int, default=12)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--max-per-image", type=int, default=120)
    parser.add_argument("--nms-threshold", type=float, default=0.2)
    args = parser.parse_args()

    sources = {str(row.get("id")): row for row in load_jsonl(Path(args.source))}
    base_rows = load_jsonl(Path(args.base))
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    total = 0
    with out.open("w", encoding="utf-8") as handle:
        for row in base_rows:
            row_id = str(row.get("id") or "")
            source = sources.get(row_id, {})
            prob_path = (source.get("pred_probs") or {}).get("symbol")
            loaded = load_mask(prob_path)
            nodes = [n for n in ((row.get("scene_graph") or {}).get("nodes") or []) if str(n.get("family")) != "symbol"]
            if loaded is not None:
                _, _, rows = loaded
                boxes = peak_boxes(rows, args.threshold, args.radius, args.box_size, args.stride, args.max_per_image, args.nms_threshold)
                for idx, item in enumerate(boxes):
                    nodes.append(node(row_id, item, idx))
                total += len(boxes)
            row.setdefault("scene_graph", {})["nodes"] = nodes
            row.setdefault("route_trace", {})["p107_symbol_peak_policy"] = vars(args) | {"runtime_gold_used": False, "added_symbols_so_far": total}
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps({"output": str(out), "symbols": total, "threshold": args.threshold, "radius": args.radius, "box_size": args.box_size}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
