#!/usr/bin/env python3
"""Build OCR-teacher pseudo labels for the v19 raster text expert.

The teacher consumes raster images only. Offline gold text boxes are loaded
after teacher inference only to audit coverage on train/dev. Locked rows are
forbidden by default and should not be used for pseudo-label training.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "datasets/text_expert_raster_v19"
OUT = ROOT / "datasets/text_ocr_teacher_pseudolabels_v19"
REPORT = ROOT / "reports/vlm"
ALLOWED_SPLITS = ("train", "dev")


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
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


def abs_path(value: str | Path) -> Path:
    path = Path(value)
    return path if path.is_absolute() else ROOT / path


def normalize_text(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("ö", "o").replace("ä", "a").replace("å", "a")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def bbox_from_poly(poly: Any) -> list[float] | None:
    if not isinstance(poly, list) or not poly:
        return None
    try:
        xs = [float(point[0]) for point in poly]
        ys = [float(point[1]) for point in poly]
    except Exception:
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def bbox_iou(left: list[float], right: list[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0:
        return 0.0
    la = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    ra = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(la + ra - inter, 1e-9)


def center_covered(pred: list[float], gold: list[float], margin: int = 2) -> bool:
    cx = (gold[0] + gold[2]) / 2.0
    cy = (gold[1] + gold[3]) / 2.0
    return pred[0] - margin <= cx <= pred[2] + margin and pred[1] - margin <= cy <= pred[3] + margin


def raster_gold_targets(row: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        item for item in row.get("text_targets") or []
        if item.get("can_train_localizer") and item.get("bbox") and len(item["bbox"]) == 4
    ]


class EasyOcrTeacher:
    def __init__(self, gpu: bool) -> None:
        if importlib.util.find_spec("easyocr") is None:
            raise RuntimeError("easyocr is not installed")
        import easyocr

        self.reader = easyocr.Reader(["en"], gpu=gpu, verbose=False)

    def read(self, image_path: Path, width: int) -> list[dict[str, Any]]:
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            if width and image.width > width:
                scale = width / max(image.width, 1)
                rgb = rgb.resize((width, max(1, int(round(image.height * scale)))), Image.Resampling.BICUBIC)
            else:
                scale = 1.0
        results = self.reader.readtext(
            str(image_path),
            detail=1,
            paragraph=False,
            text_threshold=0.05,
            low_text=0.05,
            link_threshold=0.05,
            mag_ratio=2.0,
        )
        labels: list[dict[str, Any]] = []
        for idx, item in enumerate(results):
            if len(item) < 3:
                continue
            poly, text, conf = item[0], str(item[1] or "").strip(), float(item[2] or 0.0)
            bbox = bbox_from_poly(poly)
            if bbox is None or bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
                continue
            if scale != 1.0:
                bbox = [value / scale for value in bbox]
                poly = [[float(p[0]) / scale, float(p[1]) / scale] for p in poly]
            labels.append(
                {
                    "teacher_id": f"easyocr_{idx}",
                    "bbox": [round(float(v), 3) for v in bbox],
                    "polygon": [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in poly],
                    "raw_text": text,
                    "normalized_text": normalize_text(text),
                    "confidence": round(conf, 6),
                    "teacher": "easyocr_full_image",
                    "can_train_localizer": True,
                    "can_train_ocr": bool(normalize_text(text)),
                }
            )
        return labels


def audit_row(row: dict[str, Any], pseudo: list[dict[str, Any]]) -> dict[str, Any]:
    golds = raster_gold_targets(row)
    used: set[int] = set()
    totals = Counter()
    semantic_total = Counter()
    semantic_hit = Counter()
    for gold in golds:
        gb = [float(v) for v in gold["bbox"]]
        semantic = str(gold.get("semantic_type") or "unknown")
        semantic_total[semantic] += 1
        totals["gold"] += 1
        best_iou, best_index, center_index = 0.0, None, None
        for pred_index, pred in enumerate(pseudo):
            if pred_index in used:
                continue
            pb = [float(v) for v in pred["bbox"]]
            iou = bbox_iou(pb, gb)
            if iou > best_iou:
                best_iou, best_index = iou, pred_index
            if center_index is None and center_covered(pb, gb):
                center_index = pred_index
        if best_index is not None and best_iou >= 0.30:
            used.add(best_index)
            totals["matched_iou_0_30"] += 1
            totals["matched_center"] += 1
            semantic_hit[semantic] += 1
        elif center_index is not None:
            used.add(center_index)
            totals["matched_center"] += 1
            semantic_hit[semantic] += 1
        else:
            totals["missed"] += 1
            if (gb[2] - gb[0]) * (gb[3] - gb[1]) <= 25:
                totals["missed_tiny_text"] += 1
    totals["pseudo"] = len(pseudo)
    totals["pseudo_with_text"] = sum(1 for item in pseudo if item.get("normalized_text"))
    return {
        "counts": dict(totals),
        "semantic_total": dict(semantic_total),
        "semantic_center_hit": dict(semantic_hit),
    }


def merge_counts(left: Counter[str], row_audit: dict[str, Any]) -> None:
    left.update(row_audit.get("counts") or {})


def build(args: argparse.Namespace) -> dict[str, Any]:
    if args.split not in ALLOWED_SPLITS:
        raise SystemExit(f"split {args.split!r} is forbidden; allowed splits: {ALLOWED_SPLITS}")
    rows = load_jsonl(SOURCE / f"{args.split}.jsonl")
    if args.source == "v18_only":
        rows = [row for row in rows if row.get("source_dataset") == "image_only_text_ocr_v18"]
    if args.max_pages:
        rows = rows[: args.max_pages]
    teacher = EasyOcrTeacher(gpu=args.gpu)
    out_rows: list[dict[str, Any]] = []
    totals = Counter()
    semantic_total = Counter()
    semantic_hit = Counter()
    errors: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        image_path = abs_path(row["image"])
        pseudo: list[dict[str, Any]] = []
        error = None
        try:
            pseudo = teacher.read(image_path, args.image_width)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            errors.append({"row_id": row.get("source_row_id") or row.get("id"), "error": error})
        row_audit = audit_row(row, pseudo)
        merge_counts(totals, row_audit)
        semantic_total.update(row_audit.get("semantic_total") or {})
        semantic_hit.update(row_audit.get("semantic_center_hit") or {})
        out_rows.append(
            {
                "id": row.get("id"),
                "source_row_id": row.get("source_row_id"),
                "source_dataset": row.get("source_dataset"),
                "split": args.split,
                "image": row.get("image"),
                "image_size": row.get("image_size"),
                "pseudo_text_targets": pseudo,
                "teacher_error": error,
                "audit_against_offline_gold": row_audit,
                "source_integrity": {
                    "model_input": "raster_image_only",
                    "teacher_input": "raster_image_only",
                    "gold_used_for_teacher_inference": False,
                    "gold_used_after_inference_for_audit": True,
                    "locked_split_used": False,
                    "runtime_uses_svg_or_cad_geometry": False,
                },
            }
        )
        if args.progress_every and (index + 1) % args.progress_every == 0:
            print(json.dumps({"split": args.split, "processed": index + 1, "pseudo": int(totals["pseudo"]), "gold": int(totals["gold"])}, ensure_ascii=False))
    write_jsonl(OUT / f"{args.split}.jsonl", out_rows)
    gold = max(int(totals["gold"]), 1)
    pseudo = max(int(totals["pseudo"]), 1)
    audit = {
        "schema_version": "text_ocr_teacher_pseudolabels_v19",
        "split": args.split,
        "source": args.source,
        "rows": len(out_rows),
        "source_dataset": str(SOURCE.relative_to(ROOT)),
        "output": str((OUT / f"{args.split}.jsonl").relative_to(ROOT)),
        "source_integrity": {
            "model_input": "raster_image_only",
            "teacher_input": "raster_image_only",
            "gold_used_for_teacher_inference": False,
            "gold_used_after_inference_for_audit": True,
            "locked_split_used": False,
            "runtime_uses_svg_or_cad_geometry": False,
        },
        "counts": dict(totals),
        "metrics_against_offline_gold": {
            "center_recall": round(totals["matched_center"] / gold, 6),
            "iou_0_30_recall": round(totals["matched_iou_0_30"] / gold, 6),
            "pseudo_to_gold_ratio": round(totals["pseudo"] / gold, 6),
            "pseudo_text_rate": round(totals["pseudo_with_text"] / pseudo, 6),
            "semantic_center_recall": {
                key: round(semantic_hit[key] / max(semantic_total[key], 1), 6)
                for key in sorted(semantic_total)
            },
        },
        "errors": errors[:50],
    }
    write_json(REPORT / f"text_ocr_teacher_pseudolabels_v19_{args.split}_audit.json", audit)
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", required=True, choices=ALLOWED_SPLITS)
    parser.add_argument("--source", choices=["v18_only", "all"], default="v18_only")
    parser.add_argument("--max-pages", type=int, default=0)
    parser.add_argument("--image-width", type=int, default=0)
    parser.add_argument("--gpu", action="store_true")
    parser.add_argument("--progress-every", type=int, default=25)
    args = parser.parse_args()
    audit = build(args)
    print(json.dumps({"split": audit["split"], "rows": audit["rows"], "metrics": audit["metrics_against_offline_gold"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
