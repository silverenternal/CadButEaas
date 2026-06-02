#!/usr/bin/env python3
"""Audit image-only predicted masks against target masks at pixel level.

Pure-Python PNG reader is used to avoid PIL/numpy import stalls on sshfs. This
is an offline evaluation only; target masks are never runtime inputs.
"""

from __future__ import annotations

import argparse
import json
import struct
import zlib
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
FAMILIES = ["wall", "opening", "window", "room", "room_boundary", "symbol", "text"]


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


def resize_nearest(rows: list[bytearray], out_w: int, out_h: int) -> list[bytearray]:
    in_h = len(rows)
    in_w = len(rows[0]) if in_h else 0
    if in_w == out_w and in_h == out_h:
        return rows
    out = []
    for y in range(out_h):
        sy = min(in_h - 1, int(y * in_h / max(out_h, 1)))
        row = bytearray(out_w)
        for x in range(out_w):
            sx = min(in_w - 1, int(x * in_w / max(out_w, 1)))
            row[x] = rows[sy][sx]
        out.append(row)
    return out


def score_masks(pred_rows: list[bytearray], target_rows: list[bytearray], threshold: int) -> dict[str, int]:
    h = len(target_rows)
    w = len(target_rows[0]) if h else 0
    if not pred_rows or not target_rows:
        return {"tp": 0, "pred_pos": 0, "gold_pos": 0, "union": 0}
    pred = resize_nearest(pred_rows, w, h)
    tp = pred_pos = gold_pos = union = 0
    for y in range(h):
        prow = pred[y]
        grow = target_rows[y]
        for x in range(w):
            p = prow[x] >= threshold
            g = grow[x] > 0
            if p:
                pred_pos += 1
            if g:
                gold_pos += 1
            if p and g:
                tp += 1
            if p or g:
                union += 1
    return {"tp": tp, "pred_pos": pred_pos, "gold_pos": gold_pos, "union": union}


def metrics(counts: dict[str, int]) -> dict[str, Any]:
    tp = counts["tp"]
    pred = counts["pred_pos"]
    gold = counts["gold_pos"]
    union = counts["union"]
    precision = tp / max(pred, 1)
    recall = tp / max(gold, 1)
    return {
        "tp_pixels": int(tp),
        "pred_pixels": int(pred),
        "gold_pixels": int(gold),
        "precision": round(precision, 6),
        "recall": round(recall, 6),
        "f1": round(0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall), 6),
        "iou": round(tp / max(union, 1), 6),
        "pred_to_gold_ratio": round(pred / max(gold, 1), 6),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", default="reports/vlm/image_only_multitask_proposal_v15_locked_predictions.jsonl")
    parser.add_argument("--thresholds", default="32,64,96,128,160,192")
    parser.add_argument("--families", default=",".join(FAMILIES))
    parser.add_argument("--summary", default="reports/vlm/image_only_mask_quality_audit_p105.json")
    parser.add_argument("--report", default="reports/vlm/image_only_mask_quality_audit_p105.md")
    args = parser.parse_args()

    rows = load_jsonl(Path(args.predictions))
    thresholds = [int(item) for item in args.thresholds.split(",") if item.strip()]
    families = [item.strip() for item in args.families.split(",") if item.strip()]
    totals = {family: {threshold: {"tp": 0, "pred_pos": 0, "gold_pos": 0, "union": 0} for threshold in thresholds} for family in families}
    availability = {family: {"pred_masks": 0, "pred_probs": 0, "target_masks": 0} for family in families}

    for row in rows:
        pred_masks = row.get("pred_masks") or {}
        pred_probs = row.get("pred_probs") or {}
        target_masks = row.get("target_masks") or {}
        for family in families:
            pred_path = pred_probs.get(family) or pred_masks.get(family)
            target_path = target_masks.get(family)
            pred_loaded = load_mask(pred_path)
            target_loaded = load_mask(target_path)
            if pred_masks.get(family):
                availability[family]["pred_masks"] += 1
            if pred_probs.get(family):
                availability[family]["pred_probs"] += 1
            if target_path:
                availability[family]["target_masks"] += 1
            if pred_loaded is None or target_loaded is None:
                continue
            _, _, pred_rows = pred_loaded
            _, _, target_rows = target_loaded
            for threshold in thresholds:
                counts = score_masks(pred_rows, target_rows, threshold)
                total = totals[family][threshold]
                for key, value in counts.items():
                    total[key] += value

    family_reports = {}
    for family in families:
        threshold_metrics = {str(threshold): metrics(totals[family][threshold]) for threshold in thresholds}
        best_f1_threshold = max(thresholds, key=lambda th: threshold_metrics[str(th)]["f1"])
        best_iou_threshold = max(thresholds, key=lambda th: threshold_metrics[str(th)]["iou"])
        family_reports[family] = {
            "availability": availability[family],
            "threshold_metrics": threshold_metrics,
            "best_f1": {"threshold": best_f1_threshold, **threshold_metrics[str(best_f1_threshold)]},
            "best_iou": {"threshold": best_iou_threshold, **threshold_metrics[str(best_iou_threshold)]},
        }

    summary = {
        "id": "SCI-P1-105-image-only-mask-quality-audit",
        "predictions": args.predictions,
        "records": len(rows),
        "thresholds": thresholds,
        "claim_boundary": "offline pixel-level mask audit; target masks are evaluation-only and not runtime inputs",
        "families": family_reports,
    }
    Path(args.summary).parent.mkdir(parents=True, exist_ok=True)
    Path(args.summary).write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# P1-105 Image-only Mask Quality Audit", "", "## Scope", f"- Predictions: `{args.predictions}`", f"- Records: `{len(rows)}`", "- Claim boundary: offline pixel-level mask audit only; target masks are evaluation-only.", "", "## Best F1 by Family", "| Family | Threshold | Precision | Recall | F1 | IoU | Pred/Gold |", "|---|---:|---:|---:|---:|---:|---:|"]
    for family in families:
        best = family_reports[family]["best_f1"]
        lines.append(f"| `{family}` | {best['threshold']} | {best['precision']:.6f} | {best['recall']:.6f} | {best['f1']:.6f} | {best['iou']:.6f} | {best['pred_to_gold_ratio']:.6f} |")
    lines.extend(["", "## Interpretation", "- High pixel recall with poor component boxes would indicate extraction/threshold issues.", "- Near-zero pixel recall indicates mask generator/training failure for that family.", "- Use these results to decide whether P106 should tune extraction or retrain the image-only proposal model."])
    Path(args.report).parent.mkdir(parents=True, exist_ok=True)
    Path(args.report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({family: family_reports[family]["best_f1"] for family in families}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
