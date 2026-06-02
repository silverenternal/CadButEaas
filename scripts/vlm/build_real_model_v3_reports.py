#!/usr/bin/env python3
"""Build v3 retraining/evaluation reports from completed expert runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent.parent


def load_json(path: str) -> dict[str, Any]:
    value = ROOT / path
    if not value.exists():
        return {}
    return json.loads(value.read_text(encoding="utf-8"))


def write_json(path: str, data: Any) -> None:
    value = ROOT / path
    value.parent.mkdir(parents=True, exist_ok=True)
    value.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def count_lines(path: str) -> int:
    value = ROOT / path
    if not value.exists():
        return 0
    with value.open("r", encoding="utf-8") as handle:
        return sum(1 for line in handle if line.strip())


def split_metrics(summary: dict[str, Any], split: str) -> dict[str, Any]:
    return dict((summary.get("splits") or {}).get(split) or {})


def compact_scene_eval(report: dict[str, Any]) -> dict[str, Any]:
    node = report.get("node_evaluation") or {}
    relation = report.get("relation_evaluation") or {}
    return {
        "node_macro_f1": node.get("macro_f1"),
        "node_accuracy": node.get("accuracy"),
        "per_family_f1": {
            label: metrics.get("f1")
            for label, metrics in sorted((node.get("per_label") or {}).items())
        },
        "relation_precision": relation.get("precision"),
        "relation_recall": relation.get("recall"),
        "relation_f1": relation.get("f1"),
        "invalid_graph_rate": report.get("invalid_graph_rate"),
    }


def main() -> None:
    text = load_json("checkpoints/text_dimension_expert_v6/train_summary.json")
    room = load_json("checkpoints/room_space_expert_v3/train_summary.json")
    symbol_geom = load_json("checkpoints/symbol_fixture_expert_v11/train_summary.json")
    symbol_long_tail = load_json("reports/vlm/symbol_long_tail_model_v1_eval.json")
    boundary = load_json("reports/vlm/boundary_label_arbitration_v1_eval.json")
    raw_scene = load_json("reports/vlm/scene_graph_fusion_real_upstream_eval.json")
    boundary_scene = load_json("reports/vlm/scene_graph_fusion_topk_label_arbitrated_v1_eval.json")
    retrained_scene = load_json("reports/vlm/scene_graph_fusion_symbol_long_tail_model_no_repair_scorer_v1_eval.json")
    raw_visual = load_json("reports/vlm/visual_demo/model_defect_summary_v3.json")
    post_visual = load_json("reports/vlm/visual_demo/model_defect_summary_postprocessed_v3.json")
    gate = load_json("reports/vlm/node_quality_gate_sweep_v3.json")

    text_report = {
        "version": "text_dimension_expert_v6_eval",
        "checkpoint_dir": "checkpoints/text_dimension_expert_v6",
        "model_type": text.get("model_type"),
        "dataset": text.get("input_dir"),
        "hard_case_manifest": "datasets/text_dimension_expert_v6_hard_negative/manifest.jsonl",
        "hard_case_count": count_lines("datasets/text_dimension_expert_v6_hard_negative/manifest.jsonl"),
        "splits": {
            "dev": split_metrics(text, "dev"),
            "locked_test": split_metrics(text, "locked_test"),
            "smoke": split_metrics(text, "smoke"),
        },
        "visual_demo_after_quality_gate": {
            "missing_visible_text": (post_visual.get("defect_counts") or {}).get("missing_visible_text"),
            "text_family_cases": (post_visual.get("family_counts") or {}).get("text"),
        },
        "done_when": {
            "locked_dimension_text_reported": "dimension_text" in ((split_metrics(text, "locked_test").get("per_label") or {})),
            "locked_room_label_reported": "room_label" in ((split_metrics(text, "locked_test").get("per_label") or {})),
            "hard_negative_manifest_available": count_lines("datasets/text_dimension_expert_v6_hard_negative/manifest.jsonl") > 0,
        },
    }
    write_json("reports/vlm/text_dimension_expert_v6_eval.json", text_report)

    boundary_report = {
        "version": "boundary_expert_v3_eval",
        "checkpoint": "checkpoints/boundary_label_arbitration_v1/model.joblib",
        "model_type": "boundary_label_arbitration_v1_extra_trees",
        "hard_case_manifest": "datasets/boundary_expert_v3_hard_cases/manifest.jsonl",
        "hard_case_count": count_lines("datasets/boundary_expert_v3_hard_cases/manifest.jsonl"),
        "locked_boundary_metrics": boundary.get("locked_boundary_metrics"),
        "e2e_delta": boundary.get("e2e_delta"),
        "done_when_candidate": boundary.get("done_when_candidate"),
        "status": boundary.get("status"),
    }
    write_json("reports/vlm/boundary_expert_v3_eval.json", boundary_report)
    write_json(
        "reports/vlm/boundary_postprocess_ablation_v3.json",
        {
            "version": "boundary_postprocess_ablation_v3",
            "raw_visual_defects": (raw_visual.get("defect_counts") or {}),
            "postprocessed_visual_defects": (post_visual.get("defect_counts") or {}),
            "quality_gate_boundary_events": (gate.get("family_counts") or {}).get("boundary"),
            "locked_scene_raw": compact_scene_eval(raw_scene),
            "locked_scene_boundary_arbitrated": compact_scene_eval(boundary_scene),
        },
    )

    symbol_report = {
        "version": "symbol_fixture_expert_v11_eval",
        "checkpoint_dir": "checkpoints/symbol_fixture_expert_v11",
        "geometry_only_train_summary": "checkpoints/symbol_fixture_expert_v11/train_summary.json",
        "adopted_checkpoint": "checkpoints/symbol_long_tail_model_v1/model.joblib",
        "adopted_predictions": "reports/vlm/real_upstream_predictions_v3.jsonl",
        "hard_case_manifest": "datasets/symbol_fixture_expert_v11_hard_negative/manifest.jsonl",
        "hard_case_count": count_lines("datasets/symbol_fixture_expert_v11_hard_negative/manifest.jsonl"),
        "geometry_only_acceptance": symbol_geom.get("acceptance"),
        "geometry_only_note": "The bbox/context-only v11 run did not meet the target; the adopted v11 path uses the stronger long-tail model plus visual evidence quality gate.",
        "locked_symbol_metrics": symbol_long_tail.get("locked_symbol_metrics"),
        "visual_gate_effect": {
            "raw_empty_symbol": (raw_visual.get("defect_counts") or {}).get("empty_symbol"),
            "postprocessed_empty_symbol": (post_visual.get("defect_counts") or {}).get("empty_symbol"),
            "symbol_gate_events": (gate.get("family_counts") or {}).get("symbol"),
        },
        "status": symbol_long_tail.get("status"),
    }
    write_json("reports/vlm/symbol_fixture_expert_v11_eval.json", symbol_report)
    write_json(
        "reports/vlm/symbol_evidence_gate_v3.json",
        {
            "version": "symbol_evidence_gate_v3",
            "quality_gate_report": "reports/vlm/node_quality_gate_sweep_v3.json",
            "raw_empty_symbol": (raw_visual.get("defect_counts") or {}).get("empty_symbol"),
            "postprocessed_empty_symbol": (post_visual.get("defect_counts") or {}).get("empty_symbol"),
            "symbol_gate_event_count": (gate.get("family_counts") or {}).get("symbol"),
            "reason_counts": gate.get("reason_counts"),
        },
    )

    room_report = {
        "version": "room_space_expert_v3_eval",
        "checkpoint_dir": "checkpoints/room_space_expert_v3",
        "model_type": room.get("model_type"),
        "hard_case_manifest": "datasets/room_space_expert_v3_polygon/manifest.jsonl",
        "hard_case_count": count_lines("datasets/room_space_expert_v3_polygon/manifest.jsonl"),
        "splits": {
            "dev": split_metrics(room, "dev"),
            "locked_test": split_metrics(room, "locked_test"),
            "smoke": split_metrics(room, "smoke"),
        },
        "visual_demo_after_quality_gate": {
            "extra_room": (post_visual.get("defect_counts") or {}).get("extra_room", 0),
            "room_without_label": (post_visual.get("defect_counts") or {}).get("room_without_label"),
            "label_without_room": (post_visual.get("defect_counts") or {}).get("label_without_room"),
        },
    }
    write_json("reports/vlm/room_space_expert_v3_eval.json", room_report)
    write_json(
        "reports/vlm/room_validity_gate_v3.json",
        {
            "version": "room_validity_gate_v3",
            "quality_gate_report": "reports/vlm/node_quality_gate_sweep_v3.json",
            "raw_extra_room": (raw_visual.get("defect_counts") or {}).get("extra_room"),
            "postprocessed_extra_room": (post_visual.get("defect_counts") or {}).get("extra_room", 0),
            "raw_room_without_label": (raw_visual.get("defect_counts") or {}).get("room_without_label"),
            "postprocessed_room_without_label": (post_visual.get("defect_counts") or {}).get("room_without_label"),
            "raw_label_without_room": (raw_visual.get("defect_counts") or {}).get("label_without_room"),
            "postprocessed_label_without_room": (post_visual.get("defect_counts") or {}).get("label_without_room"),
            "room_gate_events": (gate.get("family_counts") or {}).get("space"),
        },
    )

    retrain_summary = {
        "version": "expert_retrain_summary_v3",
        "environment": {
            "runner": "uv run python",
            "torch_sklearn_pil_available_in_project_venv": True,
        },
        "unified_prediction_stream": "reports/vlm/real_upstream_predictions_v3.jsonl",
        "experts": {
            "text_dimension_v6": {
                "checkpoint": "checkpoints/text_dimension_expert_v6/model_v4.joblib",
                "report": "reports/vlm/text_dimension_expert_v6_eval.json",
                "locked_macro_f1": split_metrics(text, "locked_test").get("macro_f1"),
            },
            "boundary_v3": {
                "checkpoint": "checkpoints/boundary_label_arbitration_v1/model.joblib",
                "report": "reports/vlm/boundary_expert_v3_eval.json",
                "locked_macro_f1": (boundary.get("locked_boundary_metrics") or {}).get("macro_f1"),
            },
            "symbol_fixture_v11": {
                "checkpoint": "checkpoints/symbol_long_tail_model_v1/model.joblib",
                "geometry_only_checkpoint": "checkpoints/symbol_fixture_expert_v11/model_v8.joblib",
                "report": "reports/vlm/symbol_fixture_expert_v11_eval.json",
                "locked_macro_f1": (symbol_long_tail.get("locked_symbol_metrics") or {}).get("macro_f1"),
                "note": "Geometry-only v11 failed; adopted v11 combines long-tail model output with evidence gate.",
            },
            "room_space_v3": {
                "checkpoint": "checkpoints/room_space_expert_v3/model.joblib",
                "report": "reports/vlm/room_space_expert_v3_eval.json",
                "locked_macro_f1": split_metrics(room, "locked_test").get("macro_f1"),
            },
        },
        "locked_scene": {
            "raw_model": compact_scene_eval(raw_scene),
            "postprocess_or_boundary_arbitrated": compact_scene_eval(boundary_scene),
            "retrained_model_stream": compact_scene_eval(retrained_scene),
        },
        "visual_demo_5": {
            "raw_model": {
                "case_count": raw_visual.get("case_count"),
                "defect_counts": raw_visual.get("defect_counts"),
            },
            "postprocessed": {
                "case_count": post_visual.get("case_count"),
                "defect_counts": post_visual.get("defect_counts"),
            },
        },
    }
    write_json("reports/vlm/expert_retrain_summary_v3.json", retrain_summary)

    locked_eval = {
        "version": "real_model_locked_eval_v3",
        "created": "2026-05-07",
        "scope": "CubiCasa reviewed locked split plus the 5-sample visual demo proxy.",
        "claim_boundary": "Saved expert models classify parser/SVG candidate geometry; this is not pure raster end-to-end detection.",
        "columns": {
            "raw_model": {
                "source": "reports/vlm/real_upstream_predictions_dev.jsonl",
                "scene_graph_report": "reports/vlm/scene_graph_fusion_real_upstream_eval.json",
            },
            "postprocess": {
                "source": "reports/vlm/real_upstream_predictions_dev_boundary_arbitrated_v1.jsonl",
                "scene_graph_report": "reports/vlm/scene_graph_fusion_topk_label_arbitrated_v1_eval.json",
            },
            "retrained_model": {
                "source": "reports/vlm/real_upstream_predictions_v3.jsonl",
                "scene_graph_report": "reports/vlm/scene_graph_fusion_symbol_long_tail_model_no_repair_scorer_v1_eval.json",
            },
        },
        "locked_split_metrics": {
            "raw_model": compact_scene_eval(raw_scene),
            "postprocess": compact_scene_eval(boundary_scene),
            "retrained_model": compact_scene_eval(retrained_scene),
        },
        "expert_locked_metrics": {
            "text_dimension_v6": split_metrics(text, "locked_test"),
            "boundary_v3": boundary.get("locked_boundary_metrics"),
            "symbol_fixture_v11": symbol_long_tail.get("locked_symbol_metrics"),
            "room_space_v3": split_metrics(room, "locked_test"),
        },
        "visual_demo_5": {
            "raw_model": {
                "case_count": raw_visual.get("case_count"),
                "defect_counts": raw_visual.get("defect_counts"),
            },
            "postprocess": {
                "case_count": post_visual.get("case_count"),
                "defect_counts": post_visual.get("defect_counts"),
            },
            "retrained_model": {
                "source": "reports/vlm/real_upstream_predictions_v3.jsonl",
                "note": "Locked stream is available; visual re-render can be regenerated from this stream if presentation needs current checkpoint overlays.",
            },
        },
        "oracle_policy": "expected_json/oracle outputs are renderer smoke tests only and must not be mixed with real model metrics.",
    }
    write_json("reports/vlm/real_model_locked_eval_v3.json", locked_eval)


if __name__ == "__main__":
    main()
