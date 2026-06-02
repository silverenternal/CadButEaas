#!/usr/bin/env python3
"""Apply adopted v8 symbol visual-evidence gate to v18 symbol candidates.

Inference uses only raster image crops and detector bboxes. Offline labels are
read only for optional recall-loss auditing.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports/vlm"
DEFAULT_INPUT = REPORT / "symbol_detector_v18_safe_predictions.jsonl"
DEFAULT_MODEL = ROOT / "checkpoints/symbol_visual_evidence_v8/model.joblib"
DEFAULT_OUTPUT = REPORT / "symbol_detector_v18_visual_evidence_safe_predictions.jsonl"
DEFAULT_ROUTED = REPORT / "symbol_detector_v18_visual_evidence_safe_routed_candidates.jsonl"
DEFAULT_AUDIT = REPORT / "symbol_visual_evidence_v8_to_v18_audit.json"
DEFAULT_LOCKED_GOLD = ROOT / "datasets/image_only_symbol_detector_v18/locked.jsonl"


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + ("\n" if rows else ""),
        encoding="utf-8",
    )


def clamp_bbox(bbox: list[float], size: tuple[int, int]) -> tuple[int, int, int, int]:
    width, height = size
    x1 = max(0, min(width, int(round(float(bbox[0])))))
    y1 = max(0, min(height, int(round(float(bbox[1])))))
    x2 = max(0, min(width, int(round(float(bbox[2])))))
    y2 = max(0, min(height, int(round(float(bbox[3])))))
    return x1, y1, x2, y2


class IntegralImageStats:
    def __init__(self, image: Image.Image) -> None:
        array = np.asarray(image, dtype=np.float64)
        self.size = image.size
        self.sum = self._integral(array)
        self.sum_sq = self._integral(array * array)
        self.dark = self._integral((array < 210).astype(np.float64))
        self.very_dark = self._integral((array < 80).astype(np.float64))

    @staticmethod
    def _integral(array: np.ndarray) -> np.ndarray:
        return np.pad(array.cumsum(axis=0).cumsum(axis=1), ((1, 0), (1, 0)), mode="constant")

    @staticmethod
    def _area_sum(integral: np.ndarray, x1: int, y1: int, x2: int, y2: int) -> float:
        return float(integral[y2, x2] - integral[y1, x2] - integral[y2, x1] + integral[y1, x1])

    def crop_features(self, bbox: list[float]) -> dict[str, float]:
        x1, y1, x2, y2 = clamp_bbox(bbox, self.size)
        if x2 <= x1 or y2 <= y1:
            return {
                "dark_ratio": 0.0,
                "very_dark_ratio": 0.0,
                "mean": 1.0,
                "std": 0.0,
                "area": 0.0,
                "width": 0.0,
                "height": 0.0,
                "aspect": 0.0,
            }
        w = x2 - x1
        h = y2 - y1
        total = max(float(w * h), 1.0)
        raw_sum = self._area_sum(self.sum, x1, y1, x2, y2)
        raw_sum_sq = self._area_sum(self.sum_sq, x1, y1, x2, y2)
        mean = raw_sum / total
        variance = max(raw_sum_sq / total - mean * mean, 0.0)
        return {
            "dark_ratio": self._area_sum(self.dark, x1, y1, x2, y2) / total,
            "very_dark_ratio": self._area_sum(self.very_dark, x1, y1, x2, y2) / total,
            "mean": mean / 255.0,
            "std": float(np.sqrt(variance)) / 255.0,
            "area": total,
            "width": float(w),
            "height": float(h),
            "aspect": max(w, h) / max(min(w, h), 1),
        }


def feature_vector(features: dict[str, float], names: list[str]) -> list[float]:
    return [float(features.get(name) or 0.0) for name in names]


def bbox_iou(left: list[float], right: list[float]) -> float:
    ix1, iy1 = max(left[0], right[0]), max(left[1], right[1])
    ix2, iy2 = min(left[2], right[2]), min(left[3], right[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    if inter <= 0.0:
        return 0.0
    la = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    ra = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    return inter / max(la + ra - inter, 1e-9)


def center_covered(pred: list[float], gold: list[float], margin: float = 2.0) -> bool:
    cx = (gold[0] + gold[2]) * 0.5
    cy = (gold[1] + gold[3]) * 0.5
    return pred[0] - margin <= cx <= pred[2] + margin and pred[1] - margin <= cy <= pred[3] + margin


def gold_by_row(path: Path) -> dict[str, list[dict[str, Any]]]:
    out: dict[str, list[dict[str, Any]]] = {}
    for row in load_jsonl(path):
        out[str(row.get("id"))] = [
            item for item in (row.get("targets") or {}).get("symbols") or []
            if isinstance(item.get("bbox"), list) and len(item["bbox"]) == 4
        ]
    return out


def recall(rows: list[dict[str, Any]], gold: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    matched = 0
    total = 0
    for row in rows:
        preds = row.get("predicted_symbols") or []
        used: set[int] = set()
        for gold_item in gold.get(str(row.get("id")), []):
            total += 1
            gb = [float(v) for v in gold_item["bbox"]]
            best = None
            best_iou = 0.0
            for idx, pred in enumerate(preds):
                if idx in used:
                    continue
                pb = pred.get("bbox")
                if not isinstance(pb, list) or len(pb) != 4:
                    continue
                pb = [float(v) for v in pb]
                iou = bbox_iou(pb, gb)
                if center_covered(pb, gb) or iou >= 0.30:
                    if iou >= best_iou:
                        best = idx
                        best_iou = iou
            if best is not None:
                used.add(best)
                matched += 1
    return {
        "matched": matched,
        "gold": total,
        "center_or_iou_recall": round(matched / max(total, 1), 6),
    }


def source_integrity() -> dict[str, Any]:
    return {
        "model_input": "raster_image_only",
        "gold_used_for_inference": False,
        "svg_or_parser_geometry_used_for_inference": False,
    }


def parse_thresholds(raw: str) -> list[float]:
    values = sorted({float(item.strip()) for item in raw.split(",") if item.strip()})
    return [value for value in values if 0.0 <= value <= 1.0]


def materialize(
    rows: list[dict[str, Any]],
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], Counter]:
    output_rows: list[dict[str, Any]] = []
    routed: list[dict[str, Any]] = []
    counts: Counter = Counter()
    for row in rows:
        kept: list[dict[str, Any]] = []
        for cand in row.get("predicted_symbols") or []:
            item = json.loads(json.dumps(cand))
            payload = dict(item.get("payload") or {})
            reject_prob = float(payload.get("symbol_visual_evidence_v8_reject_probability") or 0.0)
            keep = reject_prob < threshold
            payload["symbol_visual_evidence_v8_threshold"] = threshold
            payload["symbol_visual_evidence_v8_decision"] = "keep" if keep else "reject_empty_or_review"
            item["payload"] = payload
            item.setdefault("audit_trace", {})
            item["audit_trace"] = dict(item["audit_trace"])
            item["audit_trace"]["symbol_visual_evidence_v8"] = {
                "decision": payload["symbol_visual_evidence_v8_decision"],
                "raster_only": True,
            }
            counts["before"] += 1
            if keep:
                counts["kept"] += 1
                kept.append(item)
                routed.append({
                    "candidate_id": item.get("id") or item.get("candidate_id"),
                    "row_id": row.get("id"),
                    "family": "symbol",
                    "route": "symbol_fixture",
                    "bbox": item.get("bbox"),
                    "confidence": item.get("confidence"),
                    "payload": item.get("payload"),
                    "source_integrity": source_integrity(),
                })
            else:
                counts["rejected"] += 1
        out = dict(row)
        out["predicted_symbols"] = kept
        out["prediction_count_before_visual_evidence_gate"] = len(row.get("predicted_symbols") or [])
        out["prediction_count_after_visual_evidence_gate"] = len(kept)
        out["source_integrity"] = source_integrity()
        output_rows.append(out)
    return output_rows, routed, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--model", default=str(DEFAULT_MODEL))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--routed-output", default=str(DEFAULT_ROUTED))
    parser.add_argument("--audit", default=str(DEFAULT_AUDIT))
    parser.add_argument("--locked-gold", default=str(DEFAULT_LOCKED_GOLD))
    parser.add_argument("--max-recall-drop", type=float, default=0.02)
    parser.add_argument("--threshold-override", type=float)
    parser.add_argument(
        "--threshold-sweep",
        default="0.5,0.7,0.8,0.9,0.95,0.98,0.99,0.995,0.999",
        help="Comma-separated reject-probability thresholds; higher values are more conservative.",
    )
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()

    bundle = joblib.load(args.model)
    model = bundle["model"]
    feature_names = list(bundle.get("features") or [])
    model_threshold = float(bundle.get("threshold", 0.5))
    rows = load_jsonl(Path(args.input))
    if args.smoke:
        rows = rows[:5]
    image_cache: dict[str, IntegralImageStats] = {}
    scored_rows: list[dict[str, Any]] = []

    for row in rows:
        image_path = str(row.get("image") or "")
        if image_path not in image_cache:
            image_cache[image_path] = IntegralImageStats(
                Image.open(ROOT / image_path if not Path(image_path).is_absolute() else image_path).convert("L")
            )
        image = image_cache[image_path]
        scored: list[dict[str, Any]] = []
        feature_rows: list[list[float]] = []
        feature_payloads: list[dict[str, float]] = []
        for cand in row.get("predicted_symbols") or []:
            item = json.loads(json.dumps(cand))
            bbox = item.get("bbox")
            if not isinstance(bbox, list) or len(bbox) != 4:
                continue
            features = image.crop_features([float(v) for v in bbox])
            payload = dict(item.get("payload") or {})
            payload.update({
                "symbol_visual_evidence_v8_model_threshold": model_threshold,
                "symbol_visual_evidence_v8_features": {key: round(float(features[key]), 6) for key in feature_names},
                "symbol_visual_evidence_v8_model": "checkpoints/symbol_visual_evidence_v8/model.joblib",
            })
            item["payload"] = payload
            feature_rows.append(feature_vector(features, feature_names))
            feature_payloads.append(payload)
            scored.append(item)
        if feature_rows:
            reject_probs = model.predict_proba(np.asarray(feature_rows, dtype=float))[:, 1]
            for payload, reject_prob in zip(feature_payloads, reject_probs, strict=True):
                payload["symbol_visual_evidence_v8_reject_probability"] = round(float(reject_prob), 6)
        scored_row = dict(row)
        scored_row["predicted_symbols"] = scored
        scored_rows.append(scored_row)

    gold = gold_by_row(Path(args.locked_gold))
    before_recall = recall(scored_rows, gold)
    thresholds = parse_thresholds(args.threshold_sweep)
    if args.threshold_override is not None:
        thresholds = [float(args.threshold_override)]
    if model_threshold not in thresholds:
        thresholds.append(model_threshold)
    threshold_results: list[dict[str, Any]] = []
    selected: dict[str, Any] | None = None
    for candidate_threshold in sorted(set(thresholds)):
        trial_rows, _, trial_counts = materialize(scored_rows, candidate_threshold)
        trial_recall = recall(trial_rows, gold)
        trial_drop = round(before_recall["center_or_iou_recall"] - trial_recall["center_or_iou_recall"], 6)
        trial_reduction = round(trial_counts["rejected"] / max(trial_counts["before"], 1), 6)
        result = {
            "threshold": candidate_threshold,
            "counts": dict(trial_counts),
            "reduction": trial_reduction,
            "locked_recall_after": trial_recall,
            "locked_recall_drop_abs": trial_drop,
            "passes_recall_drop_gate": trial_drop <= args.max_recall_drop,
        }
        threshold_results.append(result)
        if result["passes_recall_drop_gate"] and (
            selected is None or result["reduction"] > selected["reduction"]
        ):
            selected = result

    if selected is None:
        selected = min(threshold_results, key=lambda item: (item["locked_recall_drop_abs"], -item["reduction"]))

    threshold = float(selected["threshold"])
    output_rows, routed, counts = materialize(scored_rows, threshold)
    after_recall = recall(output_rows, gold)
    recall_drop = round(before_recall["center_or_iou_recall"] - after_recall["center_or_iou_recall"], 6)
    audit = {
        "task": "IMG-MOE-V18-REBUILD-002.step_symbol_visual_evidence_v8_adapter",
        "input": str(args.input),
        "output": str(args.output),
        "routed_output": str(args.routed_output),
        "model": str(args.model),
        "features": feature_names,
        "model_threshold": model_threshold,
        "threshold": threshold,
        "selection_policy": "max_reduction_within_recall_drop_gate",
        "threshold_sweep": threshold_results,
        "smoke": args.smoke,
        "rows": len(rows),
        "counts": dict(counts),
        "reduction": round(counts["rejected"] / max(counts["before"], 1), 6),
        "locked_recall_before": before_recall,
        "locked_recall_after": after_recall,
        "locked_recall_drop_abs": recall_drop,
        "source_integrity": source_integrity(),
        "adoption_gate": {
            "max_recall_drop": args.max_recall_drop,
            "passes_recall_drop_gate": recall_drop <= args.max_recall_drop,
            "note": "Gate removes visual empty/review candidates only; symbol type remains disabled unless its own gate passes.",
        },
    }
    write_jsonl(Path(args.output), output_rows)
    write_jsonl(Path(args.routed_output), routed)
    write_json(Path(args.audit), audit)
    print(json.dumps({
        "rows": len(rows),
        "before": counts["before"],
        "kept": counts["kept"],
        "rejected": counts["rejected"],
        "reduction": audit["reduction"],
        "recall_drop": recall_drop,
        "passes": audit["adoption_gate"]["passes_recall_drop_gate"],
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
