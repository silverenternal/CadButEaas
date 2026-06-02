#!/usr/bin/env python3
"""Materialize P155D symbol box-calibration / label-cleanup rescue variants.

The transformations are runtime-safe: they use only predicted bbox, label, and
score from the P155A/P140 overlay. Offline targets are used only by the separate
P101 evaluator after materialization.
"""

from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT = ROOT / "reports/vlm/symbol_policy_moe_overlay_p155a_p140_best.jsonl"
DEFAULT_ANISO_OUTPUT = ROOT / "reports/vlm/symbol_policy_moe_overlay_p155d_aniso120_095.jsonl"
DEFAULT_LABEL_OUTPUT = ROOT / "reports/vlm/symbol_policy_moe_overlay_p155d_label_cleanup.jsonl"

LABEL_CLEANUP_THRESHOLDS = {
    "shower": 0.34,
    "stair": 0.40,
    "column": 0.40,
    "bathtub": 0.02,
}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(json.dumps(row, ensure_ascii=False) for row in rows) + "\n", encoding="utf-8")


def bbox4(value: Any) -> list[float] | None:
    if not isinstance(value, list) or len(value) < 4:
        return None
    x1, y1, x2, y2 = [float(v) for v in value[:4]]
    if x2 <= x1 or y2 <= y1:
        return None
    return [x1, y1, x2, y2]


def scale_box(box: list[float], sx: float, sy: float) -> list[float]:
    cx = (box[0] + box[2]) / 2.0
    cy = (box[1] + box[3]) / 2.0
    width = (box[2] - box[0]) * sx
    height = (box[3] - box[1]) * sy
    return [cx - width / 2.0, cy - height / 2.0, cx + width / 2.0, cy + height / 2.0]


def item_label(item: dict[str, Any]) -> str:
    return str(item.get("symbol_type") or item.get("label") or "generic_symbol")


def item_score(item: dict[str, Any]) -> float:
    value = item.get("confidence") if item.get("confidence") is not None else item.get("score")
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def sync_expected_symbols(row: dict[str, Any]) -> None:
    expected = row.get("expected_json")
    if isinstance(expected, dict):
        expected["symbol_candidates"] = [copy.deepcopy(item) for item in row.get("symbol_candidates") or []]


def materialize_aniso(rows: list[dict[str, Any]], sx: float, sy: float) -> list[dict[str, Any]]:
    output = []
    for raw in rows:
        row = copy.deepcopy(raw)
        for item in row.get("symbol_candidates") or []:
            box = bbox4(item.get("bbox"))
            if box is None:
                continue
            item["bbox"] = scale_box(box, sx, sy)
            item.setdefault("metadata", {})["p155d_box_calibration"] = f"aniso_scale_x{sx:.2f}_y{sy:.2f}"
        sync_expected_symbols(row)
        overlay = row.setdefault("symbol_policy_overlay", {})
        overlay["policy_id"] = "p155d_aniso120_095"
        overlay["description"] = "P155D anisotropic runtime-safe box calibration on P155A/P140 core."
        overlay["p155d_box_scale"] = {"sx": sx, "sy": sy}
        output.append(row)
    return output


def materialize_label_cleanup(rows: list[dict[str, Any]], thresholds: dict[str, float]) -> tuple[list[dict[str, Any]], dict[str, int]]:
    output = []
    counts = {"kept": 0, "removed": 0}
    for raw in rows:
        row = copy.deepcopy(raw)
        kept_items = []
        for item in row.get("symbol_candidates") or []:
            if item_score(item) >= thresholds.get(item_label(item), 0.0):
                item.setdefault("metadata", {})["p155d_label_cleanup"] = "bad_label_score_floor"
                kept_items.append(item)
                counts["kept"] += 1
            else:
                counts["removed"] += 1
        row["symbol_candidates"] = kept_items
        sync_expected_symbols(row)
        overlay = row.setdefault("symbol_policy_overlay", {})
        overlay["policy_id"] = "p155d_label_cleanup"
        overlay["description"] = "P155D low-precision label cleanup on P155A/P140 core."
        overlay["p155d_thresholds"] = thresholds
        output.append(row)
    return output, counts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default=str(DEFAULT_INPUT))
    parser.add_argument("--aniso-output", default=str(DEFAULT_ANISO_OUTPUT))
    parser.add_argument("--label-output", default=str(DEFAULT_LABEL_OUTPUT))
    parser.add_argument("--sx", type=float, default=1.20)
    parser.add_argument("--sy", type=float, default=0.95)
    args = parser.parse_args()

    rows = load_jsonl(Path(args.input))
    aniso_rows = materialize_aniso(rows, args.sx, args.sy)
    label_rows, label_counts = materialize_label_cleanup(rows, LABEL_CLEANUP_THRESHOLDS)
    write_jsonl(Path(args.aniso_output), aniso_rows)
    write_jsonl(Path(args.label_output), label_rows)
    print(json.dumps({
        "input": args.input,
        "aniso_output": args.aniso_output,
        "label_output": args.label_output,
        "label_cleanup_thresholds": LABEL_CLEANUP_THRESHOLDS,
        "label_cleanup_counts": label_counts,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
