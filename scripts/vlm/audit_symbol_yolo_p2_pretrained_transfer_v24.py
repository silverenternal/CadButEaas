#!/usr/bin/env python3
"""Audit and export a pretrained-transfer YOLOv8-P2 symbol detector init."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from ultralytics import YOLO
from ultralytics import __version__ as ultralytics_version
from ultralytics.nn.tasks import DetectionModel

from train_symbol_tile_detector_v20 import rel, write_json


ROOT = Path(__file__).resolve().parents[2]
P2_CFG = ROOT / ".venv/lib/python3.14/site-packages/ultralytics/cfg/models/v8/yolov8-p2.yaml"
SOURCE_WEIGHTS = ROOT / "runs/detect/runs/vlm/symbol_yolov8n_pretrained_v22_dedup_hi640_probe/weights/best.pt"
OUT_WEIGHTS = ROOT / "checkpoints/symbol_yolo_p2_pretrained_transfer_v24/init.pt"
REPORT = ROOT / "reports/vlm/symbol_yolo_p2_pretrained_transfer_audit_v24.json"


def tensor_numel(value: torch.Tensor) -> int:
    return int(value.numel())


def module_index(name: str) -> str:
    parts = name.split(".")
    if len(parts) >= 2 and parts[0] == "model":
        return parts[1]
    return "other"


def summarize_modules(keys: list[str]) -> dict[str, int]:
    counts = Counter(module_index(key) for key in keys)
    return dict(sorted(counts.items(), key=lambda item: (item[0] != "other", int(item[0]) if item[0].isdigit() else 9999)))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--p2-cfg", default=str(P2_CFG))
    parser.add_argument("--source-weights", default=str(SOURCE_WEIGHTS))
    parser.add_argument("--output-weights", default=str(OUT_WEIGHTS))
    parser.add_argument("--report-output", default=str(REPORT))
    parser.add_argument("--nc", type=int, default=9)
    args = parser.parse_args()

    p2_cfg = Path(args.p2_cfg)
    source_weights = Path(args.source_weights)
    output_weights = Path(args.output_weights)
    report_output = Path(args.report_output)

    if not p2_cfg.exists():
        raise FileNotFoundError(p2_cfg)
    if not source_weights.exists():
        raise FileNotFoundError(source_weights)

    source = YOLO(str(source_weights))
    source_state = source.model.state_dict()
    target_model = DetectionModel(str(p2_cfg), nc=args.nc, verbose=False)
    target_model.args = dict(source.model.args)
    target_model.args.update({"model": str(p2_cfg), "task": "detect"})
    target_model.task = "detect"
    target_model.names = {idx: name for idx, name in enumerate(source.model.names.values())}
    target_model.stride = target_model.model[-1].stride
    target_model.eval()
    target_model.float()
    target_state = target_model.state_dict()

    compatible: dict[str, torch.Tensor] = {}
    skipped_shape: list[dict[str, Any]] = []
    missing_in_source: list[str] = []
    for key, value in target_state.items():
        source_value = source_state.get(key)
        if source_value is None:
            missing_in_source.append(key)
            continue
        if tuple(source_value.shape) == tuple(value.shape):
            compatible[key] = source_value.detach().clone()
        else:
            skipped_shape.append(
                {
                    "key": key,
                    "target_shape": list(value.shape),
                    "source_shape": list(source_value.shape),
                    "target_module": module_index(key),
                }
            )

    updated_state = dict(target_state)
    updated_state.update(compatible)
    target_model.load_state_dict(updated_state, strict=True)
    output_weights.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "date": "2026-05-11T00:00:00",
            "version": ultralytics_version,
            "license": "AGPL-3.0 License (https://ultralytics.com/license)",
            "docs": "https://docs.ultralytics.com",
            "epoch": -1,
            "best_fitness": None,
            "model": target_model.half(),
            "ema": None,
            "updates": None,
            "optimizer": None,
            "scaler": None,
            "train_args": {
                "task": "detect",
                "mode": "train",
                "model": str(p2_cfg),
                "data": "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_v22/data.yaml",
                "epochs": 0,
                "pretrained": str(source_weights),
                "nc": args.nc,
            },
            "train_metrics": None,
            "train_results": None,
        },
        output_weights,
    )

    target_params = sum(tensor_numel(value) for value in target_state.values())
    compatible_params = sum(tensor_numel(value) for value in compatible.values())
    skipped_shape_params = sum(tensor_numel(target_state[item["key"]]) for item in skipped_shape)
    missing_params = sum(tensor_numel(target_state[key]) for key in missing_in_source)
    compatible_keys = sorted(compatible)
    skipped_shape_keys = [item["key"] for item in skipped_shape]
    report = {
        "version": "symbol_yolo_p2_pretrained_transfer_audit_v24",
        "claim_boundary": "Offline initialization audit only. Runtime model input remains raster pixels only.",
        "source_integrity": {
            "source_weights": rel(source_weights),
            "target_cfg": rel(p2_cfg),
            "output_weights": rel(output_weights),
            "raw_label_or_semantic_type_used_as_runtime_feature": False,
            "pretrained_transfer": "Shape-compatible tensors are copied from the current adopted v22 YOLO symbol baseline into the P2 architecture.",
            "from_scratch_p2_repeated": False,
        },
        "model_shapes": {
            "source_state_keys": len(source_state),
            "target_state_keys": len(target_state),
            "compatible_state_keys": len(compatible),
            "missing_in_source_keys": len(missing_in_source),
            "shape_mismatch_keys": len(skipped_shape),
            "target_parameter_tensors": len(target_state),
            "target_parameters": target_params,
            "compatible_parameters": compatible_params,
            "compatible_parameter_ratio": round(compatible_params / max(target_params, 1), 6),
            "missing_parameters": missing_params,
            "shape_mismatch_parameters": skipped_shape_params,
        },
        "module_summary": {
            "compatible_by_module": summarize_modules(compatible_keys),
            "missing_by_module": summarize_modules(missing_in_source),
            "shape_mismatch_by_module": summarize_modules(skipped_shape_keys),
        },
        "detect_head": {
            "target_detect_scales": 4,
            "expected_target_heads": ["P2/4", "P3/8", "P4/16", "P5/32"],
            "source_detect_scales": 3,
            "expected_source_heads": ["P3/8", "P4/16", "P5/32"],
            "new_or_partially_new_head_policy": "P2-specific head tensors and shape-mismatched head tensors remain initialized from target cfg; shared backbone/FPN tensors are transferred where shapes match.",
        },
        "examples": {
            "missing_in_source": missing_in_source[:40],
            "shape_mismatch": skipped_shape[:40],
            "compatible": compatible_keys[:40],
        },
        "training_recommendation": {
            "safe_to_probe": compatible_params > 0 and len(missing_in_source) > 0,
            "train_from": rel(output_weights),
            "data": "datasets/symbol_tile_detector_tiny_sahi_v21_yolo_v22/data.yaml",
            "epochs": 3,
            "imgsz": 640,
            "batch": 24,
            "evaluation": [
                "scripts/vlm/eval_symbol_yolo_tile_detector_v22.py",
                "scripts/vlm/audit_symbol_yolo_page_errors_v22.py",
            ],
        },
    }
    write_json(report_output, report)
    print(
        json.dumps(
            {
                "report": rel(report_output),
                "output_weights": rel(output_weights),
                "compatible_parameter_ratio": report["model_shapes"]["compatible_parameter_ratio"],
                "compatible_state_keys": len(compatible),
                "missing_in_source_keys": len(missing_in_source),
                "shape_mismatch_keys": len(skipped_shape),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
