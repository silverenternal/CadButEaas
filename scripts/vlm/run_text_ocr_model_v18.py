#!/usr/bin/env python3
"""Run raster-only OCR over v18 text detector candidates.

Gold transcripts are loaded only after model inference, for locked evaluation.
The OCR input is the raster crop from each predicted candidate bbox.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
GOLD_LOCKED = ROOT / "datasets/image_only_text_ocr_v18/locked.jsonl"


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def normalize_text(value: str | None) -> str:
    text = (value or "").strip().lower()
    text = re.sub(r"[^a-z0-9.]+", "", text)
    return text


def bbox_iou(left: list[float] | None, right: list[float] | None) -> float:
    if not left or not right:
        return 0.0
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    la = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    ra = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(la + ra - inter, 1e-9)


def center_covered(pred: list[int], gold: list[int], margin: int = 1) -> bool:
    cx = (gold[0] + gold[2]) / 2.0
    cy = (gold[1] + gold[3]) / 2.0
    return pred[0] - margin <= cx <= pred[2] + margin and pred[1] - margin <= cy <= pred[3] + margin


def crop_candidate(image: Image.Image, bbox: list[int], pad: int, scale: int) -> np.ndarray:
    width, height = image.size
    x1, y1, x2, y2 = [int(v) for v in bbox]
    x1 = max(0, x1 - pad)
    y1 = max(0, y1 - pad)
    x2 = min(width, x2 + pad)
    y2 = min(height, y2 + pad)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), dtype=np.uint8)
    crop = image.crop((x1, y1, x2, y2)).convert("L")
    crop = ImageOps.autocontrast(crop)
    crop = crop.resize((max(8, crop.width * scale), max(8, crop.height * scale)), Image.Resampling.BICUBIC)
    rgb = Image.merge("RGB", (crop, crop, crop))
    return np.asarray(rgb)


class EasyOcrBackend:
    def __init__(self) -> None:
        import easyocr

        self.reader = easyocr.Reader(["en"], gpu=False, verbose=False)

    def read(self, crop: np.ndarray) -> tuple[str, float | None, str]:
        result = self.reader.readtext(
            crop,
            detail=1,
            paragraph=False,
            text_threshold=0.05,
            low_text=0.05,
            link_threshold=0.05,
            mag_ratio=2.0,
        )
        pieces: list[tuple[str, float]] = []
        for item in result:
            if len(item) >= 3:
                text = str(item[1] or "").strip()
                if text:
                    pieces.append((text, float(item[2] or 0.0)))
        if not pieces:
            return "", None, "no_text"
        pieces.sort(key=lambda item: item[1], reverse=True)
        if len(pieces) == 1:
            return pieces[0][0], pieces[0][1], "ok"
        text = " ".join(piece[0] for piece in pieces)
        confidence = sum(piece[1] for piece in pieces) / len(pieces)
        return text, confidence, "ok"


def init_backend(name: str) -> tuple[Any | None, str | None]:
    if name != "easyocr":
        return None, f"unsupported_backend:{name}"
    if importlib.util.find_spec("easyocr") is None:
        return None, "easyocr_not_installed"
    try:
        return EasyOcrBackend(), None
    except Exception as exc:  # pragma: no cover - depends on local model cache/download.
        return None, f"{type(exc).__name__}: {exc}"


def load_gold(path: Path) -> dict[str, dict[str, Any]]:
    return {row["id"]: row for row in load_jsonl(path)}


def select_candidates(row: dict[str, Any], limit: int | None) -> list[dict[str, Any]]:
    candidates = list(row.get("predicted_text") or [])
    if limit is None or limit <= 0:
        return candidates
    return candidates[:limit]


def evaluate_ocr(pred_rows: list[dict[str, Any]], gold_rows: dict[str, dict[str, Any]]) -> dict[str, Any]:
    totals = Counter()
    misses: list[dict[str, Any]] = []
    examples: list[dict[str, Any]] = []
    semantic_total = Counter()
    semantic_norm = Counter()

    for pred_row in pred_rows:
        gold_texts = ((gold_rows.get(pred_row["id"]) or {}).get("targets") or {}).get("texts") or []
        predictions = pred_row.get("predicted_text") or []
        used: set[int] = set()
        row_hits = Counter()
        for gold_index, gold in enumerate(gold_texts):
            gb = [int(v) for v in gold["bbox"]]
            semantic = str(gold.get("semantic_type") or "unknown")
            semantic_total[semantic] += 1
            totals["gold"] += 1
            best_index: int | None = None
            best_key = (-1, 0.0, -1.0)
            for pred_index, pred in enumerate(predictions):
                if pred_index in used:
                    continue
                bbox = [int(v) for v in pred.get("bbox") or []]
                if len(bbox) != 4:
                    continue
                covers = center_covered(bbox, gb)
                iou = bbox_iou(bbox, gb)
                has_text = 1 if normalize_text((pred.get("payload") or {}).get("ocr_text")) else 0
                key = (1 if covers or iou >= 0.30 else 0, iou, has_text)
                if key > best_key:
                    best_key = key
                    best_index = pred_index
            if best_index is None or best_key[0] == 0:
                misses.append({"row_id": pred_row["id"], "gold_index": gold_index, "bbox": gb, "reason": "not_localized"})
                continue
            used.add(best_index)
            matched = predictions[best_index]
            payload = matched.get("payload") or {}
            ocr_text = str(payload.get("ocr_text") or "")
            norm_pred = normalize_text(ocr_text)
            norm_gold = normalize_text(gold.get("normalized_text") or gold.get("raw_text"))
            totals["localized"] += 1
            totals["ocr_nonempty"] += 1 if norm_pred else 0
            exact_ok = ocr_text.strip() == str(gold.get("raw_text") or "").strip()
            norm_ok = bool(norm_gold) and norm_pred == norm_gold
            totals["exact_match"] += 1 if exact_ok else 0
            totals["normalized_match"] += 1 if norm_ok else 0
            row_hits["localized"] += 1
            row_hits["normalized_match"] += 1 if norm_ok else 0
            semantic_norm[semantic] += 1 if norm_ok else 0
            if not norm_ok:
                misses.append({
                    "row_id": pred_row["id"],
                    "gold_index": gold_index,
                    "bbox": gb,
                    "gold": gold.get("raw_text"),
                    "predicted": ocr_text,
                    "predicted_bbox": matched.get("bbox"),
                    "ocr_status": payload.get("ocr_status"),
                    "reason": "ocr_mismatch_or_empty",
                })
        if len(examples) < 12:
            examples.append({
                "id": pred_row["id"],
                "gold": len(gold_texts),
                "ocr_candidates": len(predictions),
                "localized_gold": int(row_hits["localized"]),
                "normalized_match": int(row_hits["normalized_match"]),
                "top_ocr": [
                    {
                        "bbox": item.get("bbox"),
                        "text": (item.get("payload") or {}).get("ocr_text"),
                        "confidence": (item.get("payload") or {}).get("ocr_confidence"),
                        "status": (item.get("payload") or {}).get("ocr_status"),
                    }
                    for item in predictions[:5]
                ],
            })

    gold = max(totals["gold"], 1)
    localized = max(totals["localized"], 1)
    return {
        "gold_texts": int(totals["gold"]),
        "localized_gold_with_selected_candidates": int(totals["localized"]),
        "ocr_nonempty_localized": int(totals["ocr_nonempty"]),
        "candidate_limited_localization_recall": round(totals["localized"] / gold, 6),
        "exact_accuracy_all_gold": round(totals["exact_match"] / gold, 6),
        "normalized_accuracy_all_gold": round(totals["normalized_match"] / gold, 6),
        "exact_accuracy_localized_only": round(totals["exact_match"] / localized, 6),
        "normalized_accuracy_localized_only": round(totals["normalized_match"] / localized, 6),
        "semantic_normalized_accuracy_all_gold": {
            key: round(semantic_norm[key] / max(semantic_total[key], 1), 6)
            for key in sorted(semantic_total)
        },
        "miss_examples": misses[:40],
        "page_examples": examples,
    }


def run(args: argparse.Namespace) -> tuple[dict[str, Any], list[dict[str, Any]], list[dict[str, Any]]]:
    detector_rows = load_jsonl(Path(args.input))
    if args.limit_pages:
        detector_rows = detector_rows[: args.limit_pages]
    gold_rows = load_gold(Path(args.gold))
    backend, backend_error = init_backend(args.backend)

    prediction_rows: list[dict[str, Any]] = []
    routed_rows: list[dict[str, Any]] = []
    totals = Counter()
    status_counts = Counter()

    for row in detector_rows:
        image_path = ROOT / row["image"] if not Path(row["image"]).is_absolute() else Path(row["image"])
        image = Image.open(image_path).convert("RGB")
        selected = select_candidates(row, args.limit_candidates_per_page)
        ocr_candidates: list[dict[str, Any]] = []
        for cand in selected:
            item = json.loads(json.dumps(cand))
            payload = dict(item.get("payload") or {})
            if backend is None:
                text, confidence, status = "", None, "backend_unavailable"
            else:
                try:
                    crop = crop_candidate(image, [int(v) for v in item["bbox"]], args.crop_pad, args.crop_scale)
                    text, confidence, status = backend.read(crop)
                except Exception as exc:  # pragma: no cover - OCR backend/runtime dependent.
                    text, confidence, status = "", None, f"ocr_error:{type(exc).__name__}"
            payload["ocr_text"] = text
            payload["raw_text"] = text
            payload["normalized_text"] = normalize_text(text)
            payload["ocr_confidence"] = None if confidence is None else round(float(confidence), 6)
            payload["ocr_status"] = status
            payload["ocr_backend"] = args.backend
            payload["source"] = "raster_text_detector_v18_easyocr_crop"
            item["payload"] = payload
            item["semantic_type"] = "ocr_text" if payload["normalized_text"] else item.get("semantic_type", "unknown_text")
            ocr_candidates.append(item)
            totals["candidates_ocr_attempted"] += 1
            status_counts[status] += 1
            routed_rows.append({
                "candidate_id": item["id"],
                "row_id": row["id"],
                "family": "text",
                "route": "text_dimension",
                "bbox": item["bbox"],
                "confidence": item.get("confidence", 0.0),
                "payload": item["payload"],
                "source_integrity": {
                    "model_input": "raster_image_only",
                    "gold_used_for_inference": False,
                    "ocr_transcript_from_gold": False,
                },
            })
        prediction_rows.append({
            "id": row["id"],
            "image": row["image"],
            "predicted_text": ocr_candidates,
            "detector_prediction_count_before_ocr_cap": row.get("prediction_count_before_export_cap"),
            "ocr_candidate_limit_per_page": args.limit_candidates_per_page,
            "source_integrity": {
                "model_input": "raster_image_only",
                "gold_used_for_inference": False,
                "ocr_transcript_from_gold": False,
            },
        })

    metrics = evaluate_ocr(prediction_rows, gold_rows)
    exact = metrics["exact_accuracy_all_gold"]
    normalized = metrics["normalized_accuracy_all_gold"]
    report = {
        "task": "IMG-MOE-V18-P0-006",
        "run_mode": "raster_only_crop_ocr",
        "backend": args.backend,
        "backend_error": backend_error,
        "source_integrity": {
            "model_input": "raster_image_only",
            "gold_loaded_after_inference_for_evaluation_only": True,
            "gold_used_for_inference": False,
            "ocr_transcript_from_gold": False,
        },
        "inputs": {
            "detector_predictions": str(args.input),
            "gold_eval_file": str(args.gold),
            "rows": len(detector_rows),
            "limit_candidates_per_page": args.limit_candidates_per_page,
            "crop_pad": args.crop_pad,
            "crop_scale": args.crop_scale,
        },
        "status_counts": dict(sorted(status_counts.items())),
        "metrics": metrics,
        "ocr": {
            "backends_available": {name: importlib.util.find_spec(name) is not None for name in ["easyocr", "pytesseract", "paddleocr"]},
            "exact_accuracy": exact,
            "normalized_accuracy": normalized,
            "status": "reported" if backend is not None else "backend_unavailable",
        },
        "success_criteria": {
            "ocr_accuracy_reported": exact is not None and normalized is not None,
            "roomspace_text_evidence_ready": len(routed_rows) > 0,
        },
        "artifacts": {
            "predictions": str(args.output_predictions),
            "routed_candidates": str(args.output_routed),
            "eval": str(args.output_eval),
        },
    }
    return report, prediction_rows, routed_rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(REPORT / "text_detector_v18_locked_predictions.jsonl"))
    parser.add_argument("--gold", default=str(GOLD_LOCKED))
    parser.add_argument("--backend", default="easyocr")
    parser.add_argument("--limit-candidates-per-page", type=int, default=50)
    parser.add_argument("--limit-pages", type=int, default=None)
    parser.add_argument("--crop-pad", type=int, default=4)
    parser.add_argument("--crop-scale", type=int, default=8)
    parser.add_argument("--output-eval", default=str(REPORT / "text_ocr_v18_eval.json"))
    parser.add_argument("--output-predictions", default=str(REPORT / "text_ocr_v18_predictions.jsonl"))
    parser.add_argument("--output-routed", default=str(REPORT / "text_ocr_v18_routed_candidates.jsonl"))
    args = parser.parse_args()

    args.input = Path(args.input)
    args.gold = Path(args.gold)
    args.output_eval = Path(args.output_eval)
    args.output_predictions = Path(args.output_predictions)
    args.output_routed = Path(args.output_routed)

    report, predictions, routed = run(args)
    write_json(args.output_eval, report)
    write_jsonl(args.output_predictions, predictions)
    write_jsonl(args.output_routed, routed)

    print("task IMG-MOE-V18-P0-006 OCR")
    print("rows", report["inputs"]["rows"])
    print("attempted", sum(report["status_counts"].values()))
    print("status", report["ocr"]["status"])
    print("exact_accuracy_all_gold", report["ocr"]["exact_accuracy"])
    print("normalized_accuracy_all_gold", report["ocr"]["normalized_accuracy"])
    print("candidate_limited_localization_recall", report["metrics"]["candidate_limited_localization_recall"])


if __name__ == "__main__":
    main()
