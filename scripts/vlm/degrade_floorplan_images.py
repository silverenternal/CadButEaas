#!/usr/bin/env python3
"""Generate reproducible degraded floor-plan PNGs with stdlib-only PNG I/O."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import struct
import zlib
from collections import Counter
from pathlib import Path
from typing import Any


DEGRADATIONS = ["blur", "jpeg", "shadow", "fold", "rotation", "low_contrast", "partial_crop"]
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--inputs", nargs="*", default=["datasets/cadstruct_real_world_benchmark_v3/dev.jsonl", "datasets/cadstruct_real_world_benchmark_v3/smoke.jsonl"])
    parser.add_argument("--output-dir", default="datasets/cadstruct_degraded_v1")
    parser.add_argument("--manifest", default="datasets/cadstruct_degraded_v1/manifest.json")
    parser.add_argument("--audit", default="reports/vlm/degraded_generation_audit_v1.json")
    parser.add_argument("--per-type", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260501)
    parser.add_argument("--max-side", type=int, default=256)
    args = parser.parse_args()

    rng = random.Random(args.seed)
    sources = collect_sources([Path(item) for item in args.inputs])
    if len(sources) < args.per_type:
        raise RuntimeError(f"need at least {args.per_type} readable PNG sources, found {len(sources)}")
    sources = sources[: args.per_type]

    output_dir = Path(args.output_dir)
    image_dir = output_dir / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    failures: list[dict[str, Any]] = []

    for degradation in DEGRADATIONS:
        for index, source in enumerate(sources):
            seed = args.seed + index * 101 + DEGRADATIONS.index(degradation) * 100003
            local_rng = random.Random(seed)
            try:
                original_width, original_height, channels, pixels = read_png_rgb(source["image"])
                width, height, pixels, resize_params = resize_to_max_side(
                    original_width, original_height, channels, pixels, args.max_side
                )
                degraded, params = apply_degradation(degradation, width, height, channels, pixels, local_rng)
                out_path = image_dir / degradation / f"{index:04d}_{Path(source['image']).parent.name}_{Path(source['image']).stem}.png"
                out_path.parent.mkdir(parents=True, exist_ok=True)
                write_png_rgb(out_path, width, height, channels, degraded)
                records.append(
                    {
                        "id": f"degraded_v1_{degradation}_{index:04d}",
                        "degradation_type": degradation,
                        "seed": seed,
                        "source_image": source["image"],
                        "source_dataset": source.get("source_dataset") or "unknown",
                        "source_annotation": source.get("annotation"),
                        "output_image": str(out_path),
                        "width": width,
                        "height": height,
                        "original_width": original_width,
                        "original_height": original_height,
                        "channels": channels,
                        "params": {**params, **resize_params},
                        "source_sha256": sha256_file(Path(source["image"])),
                        "output_sha256": sha256_file(out_path),
                        "traceable_to_original": True,
                    }
                )
            except Exception as exc:  # pragma: no cover - audit path
                failures.append(
                    {
                        "degradation_type": degradation,
                        "source_image": source["image"],
                        "error": type(exc).__name__,
                        "message": str(exc),
                    }
                )

    manifest = {
        "version": "cadstruct_degraded_v1",
        "created": "2026-05-01",
        "seed": args.seed,
        "source_inputs": args.inputs,
        "degradation_types": DEGRADATIONS,
        "records": records,
        "failures": failures,
        "record_count": len(records),
    }
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    by_type = Counter(item["degradation_type"] for item in records)
    audit = {
        "version": "degraded_generation_audit_v1",
        "manifest": str(manifest_path),
        "record_count": len(records),
        "failure_count": len(failures),
        "by_degradation_type": dict(sorted(by_type.items())),
        "done_when_checks": {
            "each_type_at_least_100": all(by_type.get(item, 0) >= 100 for item in DEGRADATIONS),
            "all_records_trace_to_original": all(item.get("source_image") and item.get("params") for item in records),
            "records_have_seed": all("seed" in item for item in records),
            "no_generation_failures": not failures,
        },
    }
    audit_path = Path(args.audit)
    audit_path.parent.mkdir(parents=True, exist_ok=True)
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


def collect_sources(paths: list[Path]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    out: list[dict[str, Any]] = []
    for path in paths:
        if not path.exists():
            continue
        if path.suffix == ".jsonl":
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                row = json.loads(line)
                image = row.get("image_path") or row.get("image")
                if not image or image in seen or not Path(image).exists() or Path(image).suffix.lower() != ".png":
                    continue
                seen.add(image)
                out.append(
                    {
                        "image": image,
                        "annotation": row.get("annotation_path") or row.get("annotation"),
                        "source_dataset": row.get("source_dataset"),
                    }
                )
        elif path.suffix == ".json":
            data = json.loads(path.read_text(encoding="utf-8"))
            for row in data.get("records") or []:
                image = row.get("image")
                if not image or image in seen or not Path(image).exists() or Path(image).suffix.lower() != ".png":
                    continue
                seen.add(image)
                out.append({"image": image, "annotation": row.get("annotation"), "source_dataset": row.get("source_dataset")})
    out.sort(key=lambda item: (Path(item["image"]).stat().st_size, item["image"]))
    return out


def apply_degradation(
    kind: str, width: int, height: int, channels: int, pixels: bytearray, rng: random.Random
) -> tuple[bytearray, dict[str, Any]]:
    if kind == "blur":
        radius = rng.choice([1, 2])
        return box_blur(width, height, channels, pixels, radius), {"radius": radius}
    if kind == "jpeg":
        block = rng.choice([4, 8])
        levels = rng.choice([16, 24, 32])
        return block_quantize(width, height, channels, pixels, block, levels), {"block": block, "levels": levels}
    if kind == "shadow":
        strength = rng.uniform(0.25, 0.45)
        return add_shadow(width, height, channels, pixels, strength), {"strength": round(strength, 4)}
    if kind == "fold":
        x = rng.randint(max(1, width // 4), max(1, width * 3 // 4))
        strength = rng.uniform(0.25, 0.55)
        return add_fold(width, height, channels, pixels, x, strength), {"x": x, "strength": round(strength, 4)}
    if kind == "rotation":
        turns = rng.choice([1, 2, 3])
        return rotate_90(width, height, channels, pixels, turns), {"turns_90deg": turns, "note": "canvas preserved with nearest-neighbor remap"}
    if kind == "low_contrast":
        factor = rng.uniform(0.45, 0.7)
        return low_contrast(channels, pixels, factor), {"factor": round(factor, 4)}
    if kind == "partial_crop":
        margin = rng.uniform(0.04, 0.12)
        return partial_crop(width, height, channels, pixels, margin), {"masked_margin_fraction": round(margin, 4)}
    raise ValueError(kind)


def resize_to_max_side(
    width: int, height: int, channels: int, pixels: bytearray, max_side: int
) -> tuple[int, int, bytearray, dict[str, Any]]:
    longest = max(width, height)
    if max_side <= 0 or longest <= max_side:
        return width, height, pixels, {"resize": "none"}
    scale = max_side / longest
    new_width = max(1, int(round(width * scale)))
    new_height = max(1, int(round(height * scale)))
    out = bytearray(new_width * new_height * channels)
    for y in range(new_height):
        src_y = min(height - 1, int(y / scale))
        for x in range(new_width):
            src_x = min(width - 1, int(x / scale))
            src = (src_y * width + src_x) * channels
            dst = (y * new_width + x) * channels
            out[dst : dst + channels] = pixels[src : src + channels]
    return (
        new_width,
        new_height,
        out,
        {
            "resize": "nearest_for_degraded_benchmark_preview",
            "resize_scale": round(scale, 6),
            "max_side": max_side,
        },
    )


def box_blur(width: int, height: int, channels: int, pixels: bytearray, radius: int) -> bytearray:
    del radius
    out = bytearray(len(pixels))
    for y in range(height):
        for x in range(width):
            totals = [0] * channels
            count = 1
            idx = (y * width + x) * channels
            for c in range(channels):
                totals[c] = pixels[idx + c]
            for xx, yy in ((x - 1, y), (x + 1, y), (x, y - 1), (x, y + 1)):
                if 0 <= xx < width and 0 <= yy < height:
                    nidx = (yy * width + xx) * channels
                    for c in range(channels):
                        totals[c] += pixels[nidx + c]
                    count += 1
            idx = (y * width + x) * channels
            for c in range(channels):
                out[idx + c] = totals[c] // count
    return out


def block_quantize(width: int, height: int, channels: int, pixels: bytearray, block: int, levels: int) -> bytearray:
    out = bytearray(pixels)
    step = max(1, 256 // levels)
    for y0 in range(0, height, block):
        for x0 in range(0, width, block):
            totals = [0] * channels
            count = 0
            for y in range(y0, min(height, y0 + block)):
                for x in range(x0, min(width, x0 + block)):
                    idx = (y * width + x) * channels
                    for c in range(min(3, channels)):
                        totals[c] += pixels[idx + c]
                    if channels == 4:
                        totals[3] += pixels[idx + 3]
                    count += 1
            avg = [min(255, (totals[c] // max(count, 1) // step) * step) for c in range(channels)]
            for y in range(y0, min(height, y0 + block)):
                for x in range(x0, min(width, x0 + block)):
                    idx = (y * width + x) * channels
                    for c in range(channels):
                        out[idx + c] = avg[c]
    return out


def add_shadow(width: int, height: int, channels: int, pixels: bytearray, strength: float) -> bytearray:
    out = bytearray(pixels)
    for y in range(height):
        for x in range(width):
            shade = 1.0 - strength * ((x / max(width - 1, 1)) * 0.65 + (y / max(height - 1, 1)) * 0.35)
            idx = (y * width + x) * channels
            for c in range(min(3, channels)):
                out[idx + c] = clamp(out[idx + c] * shade)
    return out


def add_fold(width: int, height: int, channels: int, pixels: bytearray, fold_x: int, strength: float) -> bytearray:
    out = bytearray(pixels)
    band = max(2, width // 80)
    for y in range(height):
        for x in range(max(0, fold_x - band * 4), min(width, fold_x + band * 4)):
            distance = abs(x - fold_x) / max(band * 4, 1)
            shade = 1.0 - strength * max(0.0, 1.0 - distance)
            idx = (y * width + x) * channels
            for c in range(min(3, channels)):
                out[idx + c] = clamp(out[idx + c] * shade)
    return out


def rotate_90(width: int, height: int, channels: int, pixels: bytearray, turns: int) -> bytearray:
    out = bytearray([255] * len(pixels))
    for y in range(height):
        for x in range(width):
            nx, ny = x, y
            if turns == 1:
                nx = int(y * width / max(height, 1))
                ny = height - 1 - int(x * height / max(width, 1))
            elif turns == 2:
                nx = width - 1 - x
                ny = height - 1 - y
            elif turns == 3:
                nx = width - 1 - int(y * width / max(height, 1))
                ny = int(x * height / max(width, 1))
            if 0 <= nx < width and 0 <= ny < height:
                src = (y * width + x) * channels
                dst = (ny * width + nx) * channels
                out[dst : dst + channels] = pixels[src : src + channels]
    return out


def low_contrast(channels: int, pixels: bytearray, factor: float) -> bytearray:
    out = bytearray(pixels)
    for idx in range(0, len(out), channels):
        for c in range(min(3, channels)):
            out[idx + c] = clamp(128 + (out[idx + c] - 128) * factor)
    return out


def partial_crop(width: int, height: int, channels: int, pixels: bytearray, margin: float) -> bytearray:
    out = bytearray(pixels)
    mx = int(width * margin)
    my = int(height * margin)
    for y in range(height):
        for x in range(width):
            if x < mx or y < my or x >= width - mx or y >= height - my:
                idx = (y * width + x) * channels
                for c in range(min(3, channels)):
                    out[idx + c] = 255
                if channels == 4:
                    out[idx + 3] = 255
    return out


def read_png_rgb(path_text: str) -> tuple[int, int, int, bytearray]:
    path = Path(path_text)
    data = path.read_bytes()
    if not data.startswith(PNG_SIGNATURE):
        raise ValueError("not_png")
    pos = len(PNG_SIGNATURE)
    width = height = color_type = bit_depth = None
    idat = bytearray()
    while pos < len(data):
        length = struct.unpack(">I", data[pos : pos + 4])[0]
        chunk_type = data[pos + 4 : pos + 8]
        chunk_data = data[pos + 8 : pos + 8 + length]
        pos += 12 + length
        if chunk_type == b"IHDR":
            width, height, bit_depth, color_type, _, _, interlace = struct.unpack(">IIBBBBB", chunk_data)
            if bit_depth != 8 or color_type not in (2, 6) or interlace != 0:
                raise ValueError(f"unsupported_png:bit_depth={bit_depth}:color_type={color_type}:interlace={interlace}")
        elif chunk_type == b"IDAT":
            idat.extend(chunk_data)
        elif chunk_type == b"IEND":
            break
    if width is None or height is None or color_type is None:
        raise ValueError("missing_ihdr")
    channels = 3 if color_type == 2 else 4
    raw = zlib.decompress(bytes(idat))
    stride = width * channels
    pixels = bytearray(width * height * channels)
    prev = bytearray(stride)
    offset = 0
    for y in range(height):
        filter_type = raw[offset]
        offset += 1
        scan = bytearray(raw[offset : offset + stride])
        offset += stride
        recon = unfilter_scanline(filter_type, scan, prev, channels)
        pixels[y * stride : (y + 1) * stride] = recon
        prev = recon
    return width, height, channels, pixels


def unfilter_scanline(filter_type: int, scan: bytearray, prev: bytearray, bpp: int) -> bytearray:
    out = bytearray(scan)
    for i in range(len(out)):
        left = out[i - bpp] if i >= bpp else 0
        up = prev[i] if prev else 0
        up_left = prev[i - bpp] if prev and i >= bpp else 0
        if filter_type == 0:
            continue
        if filter_type == 1:
            out[i] = (out[i] + left) & 0xFF
        elif filter_type == 2:
            out[i] = (out[i] + up) & 0xFF
        elif filter_type == 3:
            out[i] = (out[i] + ((left + up) // 2)) & 0xFF
        elif filter_type == 4:
            out[i] = (out[i] + paeth(left, up, up_left)) & 0xFF
        else:
            raise ValueError(f"unsupported_filter:{filter_type}")
    return out


def write_png_rgb(path: Path, width: int, height: int, channels: int, pixels: bytearray) -> None:
    color_type = 2 if channels == 3 else 6
    raw = bytearray()
    stride = width * channels
    for y in range(height):
        raw.append(0)
        raw.extend(pixels[y * stride : (y + 1) * stride])
    chunks = [
        png_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, color_type, 0, 0, 0)),
        png_chunk(b"IDAT", zlib.compress(bytes(raw), level=6)),
        png_chunk(b"IEND", b""),
    ]
    path.write_bytes(PNG_SIGNATURE + b"".join(chunks))


def png_chunk(chunk_type: bytes, payload: bytes) -> bytes:
    crc = zlib.crc32(chunk_type + payload) & 0xFFFFFFFF
    return struct.pack(">I", len(payload)) + chunk_type + payload + struct.pack(">I", crc)


def paeth(a: int, b: int, c: int) -> int:
    p = a + b - c
    pa = abs(p - a)
    pb = abs(p - b)
    pc = abs(p - c)
    if pa <= pb and pa <= pc:
        return a
    if pb <= pc:
        return b
    return c


def clamp(value: float) -> int:
    return max(0, min(255, int(round(value))))


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


if __name__ == "__main__":
    main()
