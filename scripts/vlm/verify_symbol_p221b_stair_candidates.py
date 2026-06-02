#!/usr/bin/env python3
"""Use P221b stair crop specialist to re-score existing P213b stair proposals."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from PIL import Image
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "reports/vlm/symbol_p222_p221a_frozen_overlay.jsonl"
PROPOSAL = ROOT / "reports/vlm/symbol_p212_p213b_residual_fusion_overlay.jsonl"
WEIGHTS = ROOT / "checkpoints/symbol_p221b_stair_specialist_yolo/model.pt"
OUTPUT = ROOT / "reports/vlm/symbol_p221b_stair_candidate_verified_predictions.jsonl"
REPORT = ROOT / "reports/vlm/symbol_p221b_stair_candidate_verifier_report.json"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""), encoding="utf-8")


def rid(row: dict[str, Any]) -> str:
    return str(row.get("id") or row.get("row_id"))


def image_path(row: dict[str, Any]) -> Path:
    raw = row.get("image_path") or row.get("image")
    path = Path(str(raw))
    return path if path.is_absolute() else ROOT / path


def label(candidate: dict[str, Any]) -> str:
    return str(candidate.get("symbol_type") or candidate.get("label") or "generic_symbol")


def score(candidate: dict[str, Any]) -> float:
    return float(candidate.get("confidence") or candidate.get("score") or 0.0)


def crop_box(box: list[float], width: int, height: int, crop_size: int) -> tuple[int, int, int, int]:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    half = crop_size / 2.0
    left = max(0, min(int(round(cx - half)), max(width - crop_size, 0)))
    top = max(0, min(int(round(cy - half)), max(height - crop_size, 0)))
    return left, top, min(left + crop_size, width), min(top + crop_size, height)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", default=str(BASE))
    parser.add_argument("--proposal", default=str(PROPOSAL))
    parser.add_argument("--weights", default=str(WEIGHTS))
    parser.add_argument("--output", default=str(OUTPUT))
    parser.add_argument("--report", default=str(REPORT))
    parser.add_argument("--crop-size", type=int, default=192)
    parser.add_argument("--imgsz", type=int, default=256)
    parser.add_argument("--decode-conf", type=float, default=0.001)
    parser.add_argument("--decode-iou", type=float, default=0.7)
    parser.add_argument("--predict-batch", type=int, default=64)
    parser.add_argument("--device", default="0")
    args = parser.parse_args()

    base_rows = {rid(row): row for row in read_jsonl(Path(args.base))}
    proposal_rows = {rid(row): row for row in read_jsonl(Path(args.proposal))}
    model = YOLO(args.weights)
    output_by_row: dict[str, list[dict[str, Any]]] = {key: [] for key in base_rows}
    audit = {"id": "P221b_stair_candidate_verifier", "rows": len(base_rows), "proposal": str(Path(args.proposal).relative_to(ROOT)), "total_candidates": 0, "scored_candidates": 0}
    batch_images: list[Image.Image] = []
    batch_meta: list[tuple[str, dict[str, Any], int, int]] = []

    def flush() -> None:
        nonlocal batch_images, batch_meta
        if not batch_images:
            return
        results = model.predict(source=batch_images, imgsz=args.imgsz, conf=args.decode_conf, iou=args.decode_iou, max_det=50, device=args.device, batch=len(batch_images), stream=False, verbose=False)
        for (row_id, cand, left, top), result in zip(batch_meta, results, strict=True):
            cbox = [float(v) for v in cand["bbox"]]
            ccx = (cbox[0] + cbox[2]) / 2.0
            ccy = (cbox[1] + cbox[3]) / 2.0
            best = 0.0
            best_box = None
            if result.boxes is not None:
                boxes = result.boxes.xyxy.detach().cpu().tolist()
                confs = result.boxes.conf.detach().cpu().tolist()
                for box, conf in zip(boxes, confs, strict=True):
                    page_box = [float(box[0] + left), float(box[1] + top), float(box[2] + left), float(box[3] + top)]
                    if page_box[0] <= ccx <= page_box[2] and page_box[1] <= ccy <= page_box[3] and float(conf) > best:
                        best = float(conf)
                        best_box = page_box
            out = {
                "bbox": cbox,
                "label": "stair",
                "label_id": 8,
                "score": score(cand),
                "verifier_score": best,
                "fused_score": (score(cand) * max(best, 1e-6)) ** 0.5,
                "tile_id": (cand.get("metadata") or {}).get("tile_id"),
                "source": "p221b_p213b_stair_verified",
            }
            if best_box is not None:
                out["verifier_bbox"] = best_box
            output_by_row[row_id].append(out)
            audit["scored_candidates"] += 1
        batch_images = []
        batch_meta = []

    for row_id, base_row in base_rows.items():
        prop_row = proposal_rows.get(row_id)
        if not prop_row:
            continue
        candidates = [cand for cand in prop_row.get("symbol_candidates") or [] if label(cand) == "stair"]
        audit["total_candidates"] += len(candidates)
        if not candidates:
            continue
        with Image.open(image_path(base_row)) as image:
            image = image.convert("RGB")
            width, height = image.size
            for cand in candidates:
                box = [float(v) for v in cand["bbox"]]
                left, top, right, bottom = crop_box(box, width, height, args.crop_size)
                batch_images.append(image.crop((left, top, right, bottom)))
                batch_meta.append((row_id, cand, left, top))
                if len(batch_images) >= args.predict_batch:
                    flush()
        flush()
    rows = [{"row_id": key, "predicted_symbols": output_by_row[key]} for key in sorted(output_by_row)]
    write_jsonl(Path(args.output), rows)
    audit["output"] = str(Path(args.output).relative_to(ROOT))
    write_json(Path(args.report), audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
