#!/usr/bin/env python3
"""Build a reuse-first registry for raster-only MoE expert assets.

The registry is deliberately conservative: strong candidate-level experts are
marked reusable only inside their valid inference boundary, while detector
probes, oracle crop heads, SVG-assisted reports, and smoke-only evidence are
kept out of production promotion.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports/vlm/reusable_expert_asset_registry_v24.json"


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"_load_error": "json_decode_error"}
    return data if isinstance(data, dict) else {"_load_error": "not_json_object"}


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def pick(data: dict[str, Any], path: str, default: Any = None) -> Any:
    cur: Any = data
    for key in path.split("."):
        if isinstance(cur, dict):
            cur = cur.get(key)
        elif isinstance(cur, list) and key.isdigit():
            index = int(key)
            if index >= len(cur):
                return default
            cur = cur[index]
        else:
            return default
    return cur if cur is not None else default


def path_status(paths: list[str]) -> dict[str, bool]:
    return {item: (ROOT / item).exists() for item in paths}


def metric_block(report_path: str, mapping: dict[str, str]) -> dict[str, Any]:
    data = load_json(ROOT / report_path)
    return {name: pick(data, source) for name, source in mapping.items()}


def make_entry(
    *,
    asset_id: str,
    family: str,
    role: str,
    production_status: str,
    checkpoint_paths: list[str],
    report_paths: list[str],
    metric_scope: str,
    runtime_input_contract: str,
    output_contract: str,
    offline_label_use: str,
    metrics: dict[str, Any],
    reuse_policy: str,
    caveats: list[str],
    next_action: str,
) -> dict[str, Any]:
    all_paths = checkpoint_paths + report_paths
    return {
        "asset_id": asset_id,
        "family": family,
        "role": role,
        "production_status": production_status,
        "checkpoint_paths": checkpoint_paths,
        "report_paths": report_paths,
        "path_exists": path_status(all_paths),
        "metric_scope": metric_scope,
        "runtime_input_contract": runtime_input_contract,
        "output_contract": output_contract,
        "offline_label_use": offline_label_use,
        "metrics": metrics,
        "reuse_policy": reuse_policy,
        "caveats": caveats,
        "next_action": next_action,
    }


def build_registry() -> dict[str, Any]:
    strong_assets = [
        make_entry(
            asset_id="boundary_node_crop_gnn_h384_c32_ms3_l2_e24",
            family="boundary_node_classifier",
            role="classify raster-derived graph-node/crop proposals as hard_wall/door/window/opening-like boundary semantics",
            production_status="reuse_first_candidate_classifier",
            checkpoint_paths=["checkpoints/cadstruct_graph_node_crop_gnn_h384_c32_ms3_l2_e24/model_best.pt"],
            report_paths=["reports/vlm/cadstruct_direct_split_expert_audit.json"],
            metric_scope="direct split candidate-level expert audit, not full-page raster proposal detection",
            runtime_input_contract="Raster pixels cropped around candidate graph nodes plus raster-derived candidate geometry/features. No SVG/parser geometry at runtime.",
            output_contract="Per-candidate boundary label probabilities and confidence suitable for MoE fusion after proposal generation.",
            offline_label_use="SVG/CubiCasa labels may supervise/evaluate candidate labels offline only.",
            metrics={
                "dev_accuracy": 0.991147,
                "dev_macro_f1": 0.986746,
                "hard_wall_f1": 0.993981,
                "door_f1": 0.987724,
                "window_f1": 0.978533,
            },
            reuse_policy="Do not retrain first. Build a raster proposal frontend and attach this classifier before changing classifier architecture.",
            caveats=[
                "Cannot localize full-page boundaries by itself.",
                "Any production metric must separate proposal recall from classifier accuracy.",
            ],
            next_action="Use in P0-BOUNDARY-PROPOSAL-002 after generating graph-node/segment proposals.",
        ),
        make_entry(
            asset_id="room_space_expert_v13",
            family="room_space_classifier",
            role="classify room/space candidates",
            production_status="reuse_first_candidate_classifier",
            checkpoint_paths=["checkpoints/room_space_expert_v13/model.joblib"],
            report_paths=["reports/vlm/room_space_expert_v13_eval.json"],
            metric_scope="locked candidate-level room classification with reviewed CubiCasa labels",
            runtime_input_contract="Raster-derived room polygon/candidate geometry and crop/layout features. No gold room ids or SVG polygons at runtime.",
            output_contract="Per-room semantic label probabilities.",
            offline_label_use="Structured labels may be used for training/evaluation only.",
            metrics=metric_block(
                "reports/vlm/room_space_expert_v13_eval.json",
                {
                    "locked_accuracy": "baseline_locked.accuracy",
                    "locked_macro_f1": "baseline_locked.macro_f1",
                    "locked_mean_iou": "baseline_locked.mean_iou",
                },
            ),
            reuse_policy="Do not retrain until raster room proposals are reliable; weak point is proposal generation.",
            caveats=["Mean IoU of 1.0 reflects candidate/gold geometry scope, not image-only room polygon recovery."],
            next_action="Attach after P0-ROOM-PROPOSAL-005 produces raster room candidates.",
        ),
        make_entry(
            asset_id="text_dimension_expert_v13",
            family="text_dimension_classifier",
            role="classify already localized text/dimension/leader candidates",
            production_status="reuse_first_candidate_classifier",
            checkpoint_paths=["checkpoints/text_dimension_expert_v13/model.joblib"],
            report_paths=["reports/vlm/text_dimension_expert_v13_eval.json"],
            metric_scope="locked candidate-level text/dimension classification",
            runtime_input_contract="Raster text/line crops and OCR/layout features from a raster-only localizer.",
            output_contract="Per-candidate text/dimension semantic label and dimension-link confidence.",
            offline_label_use="Text labels may be used for training/evaluation only.",
            metrics=metric_block(
                "reports/vlm/text_dimension_expert_v13_eval.json",
                {
                    "locked_accuracy": "baseline_locked.accuracy",
                    "locked_macro_f1": "baseline_locked.macro_f1",
                    "dimension_link_f1": "baseline_locked.dimension_link.f1",
                    "dimension_line_f1": "baseline_locked.per_label.dimension_line.f1",
                    "dimension_text_f1": "baseline_locked.per_label.dimension_text.f1",
                    "note_text_f1": "baseline_locked.per_label.note_text.f1",
                },
            ),
            reuse_policy="Do not retrain for localization failures; fix raster text localizer first.",
            caveats=["note_text is below target and should be audited after localization is fixed."],
            next_action="Use in P0-TEXT-LOCALIZATION-004 once text boxes/lines are found from raster.",
        ),
        make_entry(
            asset_id="symbol_visual_evidence_v8",
            family="symbol_visual_evidence_gate",
            role="reject empty/review symbol proposal crops while preserving visual positives",
            production_status="reuse_first_visual_gate",
            checkpoint_paths=["checkpoints/symbol_visual_evidence_v8/model.joblib"],
            report_paths=["reports/vlm/symbol_visual_evidence_v8_eval.json"],
            metric_scope="locked crop-level visual evidence gate",
            runtime_input_contract="Raster crop pixels and simple crop geometry from detector/proposal output.",
            output_contract="keep vs empty_or_review probability for symbol proposal crops.",
            offline_label_use="Offline labels may train/evaluate keep/reject; not runtime features.",
            metrics=metric_block(
                "reports/vlm/symbol_visual_evidence_v8_eval.json",
                {
                    "locked_accuracy": "locked_eval.classification_report.accuracy",
                    "locked_macro_f1": "locked_eval.classification_report.macro avg.f1-score",
                    "reject_precision": "locked_eval.reject_precision",
                    "reject_recall": "locked_eval.reject_recall",
                },
            ),
            reuse_policy="Reuse to audit and filter symbol proposal streams; do not treat it as detector or type classifier.",
            caveats=["May improve precision but must be evaluated fail-open/fail-closed so center recall is not damaged."],
            next_action="Apply to current YOLO body proposal predictions in P0-SYMBOL-PROPOSAL-003.",
        ),
    ]

    diagnostic_assets = [
        make_entry(
            asset_id="symbol_yolov8n_pretrained_v22_dedup_hi640_probe",
            family="symbol_body_detector",
            role="raster symbol body proposal baseline",
            production_status="weak_proposal_baseline",
            checkpoint_paths=["runs/detect/runs/vlm/symbol_yolov8n_pretrained_v22_dedup_hi640_probe/weights/best.pt"],
            report_paths=["reports/vlm/symbol_yolov8n_pretrained_v22_dedup_hi640_probe_page_eval.json"],
            metric_scope="locked page-level detector proposal evaluation",
            runtime_input_contract="Raster tile pixels only.",
            output_contract="Symbol body boxes and confidence scores.",
            offline_label_use="Gold boxes for evaluation and hard-case mining only.",
            metrics=metric_block(
                "reports/vlm/symbol_yolov8n_pretrained_v22_dedup_hi640_probe_page_eval.json",
                {
                    "center_recall": "threshold_grid.0.metrics.symbol_bbox_center_recall",
                    "iou_0_30_precision": "threshold_grid.0.metrics.symbol_bbox_iou_0_30.precision",
                    "iou_0_30_recall": "threshold_grid.0.metrics.symbol_bbox_iou_0_30.recall",
                    "candidate_inflation": "threshold_grid.0.metrics.candidate_inflation",
                },
            ),
            reuse_policy="Keep as proposal stream only; not sufficient as production symbol expert.",
            caveats=["Recall and precision are far below 0.98.", "Useful baseline to beat and source of detector-proposal crops."],
            next_action="Fuse with symbol_visual_evidence_v8 and build proposal-distribution type calibration.",
        ),
        make_entry(
            asset_id="symbol_crop_context_convnext_v20_finetune",
            family="symbol_type_head",
            role="type classifier for symbol crops",
            production_status="diagnostic_oracle_crop_head",
            checkpoint_paths=["checkpoints/symbol_crop_context_pretrained_v20_convnext_tiny_finetune/model.pt"],
            report_paths=["reports/vlm/symbol_yolo_convnext_two_stage_v24_eval.json"],
            metric_scope="two-stage audit on noisy YOLO proposals; oracle crop evidence is not production evidence",
            runtime_input_contract="Raster crops from detector proposals only if used; no gold crop boxes.",
            output_contract="Symbol type probabilities.",
            offline_label_use="Gold types for supervised calibration/evaluation only.",
            metrics=metric_block(
                "reports/vlm/symbol_yolo_convnext_two_stage_v24_eval.json",
                {
                    "typed_predictions": "classify_audit.typed_predictions",
                    "real_yolo_body_center_recall": "body_only_metrics_from_input_predictions.symbol_bbox_center_recall",
                    "real_yolo_iou_recall": "body_only_metrics_from_input_predictions.symbol_bbox_iou_0_30.recall",
                },
            ),
            reuse_policy="Do not promote directly; retrain/calibrate on detector-proposal crop distribution if needed.",
            caveats=["Oracle crop quality does not transfer to detector boxes.", "Type accuracy is capped by upstream proposal quality."],
            next_action="Use only for diagnostic comparison in P0-SYMBOL-PROPOSAL-003.",
        ),
        make_entry(
            asset_id="boundary_public_raster_v24_yolo_probe",
            family="boundary_yolo_probe",
            role="raster tile boundary box proposal probe",
            production_status="weak_proposal_baseline",
            checkpoint_paths=["runs/detect/runs/detect/runs/vlm/boundary_public_raster_v24_yolo_probe/weights/best.pt"],
            report_paths=["reports/vlm/boundary_public_raster_v24_yolo_probe_eval.json"],
            metric_scope="short tile-level locked YOLO probe, not page-level after-cap recall",
            runtime_input_contract="Raster tile pixels only.",
            output_contract="Wall/opening/window boxes and confidence scores.",
            offline_label_use="Gold boxes for training/evaluation only.",
            metrics=metric_block(
                "reports/vlm/boundary_public_raster_v24_yolo_probe_eval.json",
                {
                    "tile_precision": "tile_locked_metrics.metrics/precision(B)",
                    "tile_recall": "tile_locked_metrics.metrics/recall(B)",
                    "tile_map50": "tile_locked_metrics.metrics/mAP50(B)",
                    "tile_map50_95": "tile_locked_metrics.metrics/mAP50-95(B)",
                },
            ),
            reuse_policy="Use as one proposal stream only; final route should be segment/graph proposals plus crop-GNN classifier.",
            caveats=["Tile detection numbers are below target and not equivalent to graph topology recovery."],
            next_action="Bootstrap P0-BOUNDARY-PROPOSAL-002 but do not optimize this alone.",
        ),
        make_entry(
            asset_id="text_expert_v19_localizer",
            family="raster_text_localizer",
            role="text bounding-box localization baseline",
            production_status="weak_localizer_baseline",
            checkpoint_paths=["checkpoints/text_expert_v19/model_best.pt"],
            report_paths=["reports/vlm/text_expert_v19_eval.json"],
            metric_scope="raster text localization dev/locked audit",
            runtime_input_contract="Raster page image only.",
            output_contract="Text boxes and scores.",
            offline_label_use="Gold text boxes for training/evaluation only.",
            metrics=metric_block(
                "reports/vlm/text_expert_v19_eval.json",
                {
                    "dev_iou_recall": "dev.text_bbox_iou_0_30.recall",
                    "dev_center_recall": "dev.text_bbox_center_recall",
                    "dev_candidate_inflation": "dev.candidate_inflation",
                },
            ),
            reuse_policy="Do not attach strong text_dimension_v13 behind this until localization is replaced or fixed.",
            caveats=["Current localization is the bottleneck, not the text/dimension classifier."],
            next_action="P0-TEXT-LOCALIZATION-004 must replace or substantially improve this stage.",
        ),
    ]

    all_entries = strong_assets + diagnostic_assets
    missing_required = []
    for entry in all_entries:
        for path, exists in entry["path_exists"].items():
            if not exists and path in entry["checkpoint_paths"] + entry["report_paths"]:
                missing_required.append({"asset_id": entry["asset_id"], "path": path})

    no_oracle_promoted = all(
        "oracle" not in item["production_status"] or not item["production_status"].startswith("reuse_first")
        for item in all_entries
    )
    strong_have_report_or_checkpoint = all(
        any((ROOT / path).exists() for path in item["checkpoint_paths"] + item["report_paths"])
        for item in strong_assets
    )

    return {
        "schema_version": "reusable_expert_asset_registry_v24",
        "created": "2026-05-11",
        "purpose": "Prevent repeated training and metric confusion by fixing reusable expert boundaries for the raster-only MoE.",
        "hard_contract": {
            "runtime_input": "Raster image pixels and raster-derived proposals only.",
            "forbidden_runtime_features": ["svg_geometry", "cad_geometry", "expected_json", "gold_labels", "annotation_path", "vector_ids"],
            "offline_label_use": ["dataset_conversion", "supervised_training", "dev_selection", "locked_evaluation", "hard_case_mining", "audit"],
        },
        "summary": {
            "entries": len(all_entries),
            "reuse_first_assets": [item["asset_id"] for item in strong_assets],
            "diagnostic_or_weak_assets": [item["asset_id"] for item in diagnostic_assets],
            "missing_required_paths": missing_required,
            "main_bottleneck": "Raster proposal/localization frontend, not already-strong candidate classifiers.",
        },
        "assets": all_entries,
        "promotion_rules": [
            "A candidate classifier can be reused only after a raster-only proposal source exists.",
            "Oracle crop, SVG/parser-assisted, smoke-only, and dev-only reports cannot be marked production.",
            "Every page-level claim must report proposal recall, classifier accuracy, post-cap recall, precision, and candidate inflation separately.",
            "Do not retrain a strong classifier until the upstream proposal/localizer has been audited and shown not to be the bottleneck.",
        ],
        "success_gate": {
            "registry_exists": True,
            "all_strong_assets_have_checkpoint_or_report": strong_have_report_or_checkpoint,
            "no_oracle_asset_marked_production": no_oracle_promoted,
            "passed": strong_have_report_or_checkpoint and no_oracle_promoted and not missing_required,
        },
        "next_task": {
            "id": "P0-BOUNDARY-PROPOSAL-002",
            "reason": "The registry confirms the boundary crop-GNN is strong but lacks a raster-only proposal frontend.",
        },
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    registry = build_registry()
    write_json(Path(args.output), registry)
    print(json.dumps({"output": rel(Path(args.output)), "success_gate": registry["success_gate"]}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
