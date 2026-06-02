#!/usr/bin/env python3
"""Build RoomSpace v4 visual/linking evaluation report."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parent.parent.parent


def main() -> None:
    room_v3 = load_json(ROOT / "reports/vlm/room_space_expert_v3_eval.json")
    residual = load_json(ROOT / "reports/vlm/room_space_visual_residual_audit_v3.json")
    link = load_json(ROOT / "reports/vlm/room_label_link_resolver_v3.json")
    gate = load_json(ROOT / "reports/vlm/room_validity_gate_v4.json")
    ablation = load_json(ROOT / "reports/vlm/room_validity_gate_ablation_v4.json")
    summary = load_json(ROOT / "reports/vlm/visual_demo/model_defect_summary_roomspace_v4.json")
    old_summary = load_json(ROOT / "reports/vlm/visual_demo/model_defect_summary_postprocessed_v3.json")
    report = {
        "version": "room_space_v4_visual_eval",
        "created": "2026-05-07",
        "claim_boundary": "RoomSpace v4 is a visual/postprocess improvement over saved expert-model labels and parser/SVG candidate geometry; it is not pure raster room segmentation.",
        "locked_room_type_classification": {
            "source": "reports/vlm/room_space_expert_v3_eval.json",
            "checkpoint_dir": room_v3.get("checkpoint_dir"),
            "locked_test": ((room_v3.get("splits") or {}).get("locked_test") or {}),
        },
        "visual_room_label_linking": {
            "source": "reports/vlm/room_label_link_resolver_v3.json",
            "summary": link.get("summary"),
            "events_file": "reports/vlm/room_label_link_resolver_v3.json",
        },
        "visual_room_validity_gate": {
            "source": "reports/vlm/room_validity_gate_v4.json",
            "summary": gate.get("summary"),
            "policy": gate.get("policy"),
        },
        "polygon_rendering": {
            "source": "reports/vlm/room_space_visual_residual_audit_v3.json",
            "polygon_status_counts": ((residual.get("summary") or {}).get("polygon_status_counts") or {}),
            "room_polygon_limitation": "Current CubiCasa converted room candidates in the visual stream are bbox-only; renderer already supports polygon when provided.",
        },
        "visual_defect_delta": {
            "before_postprocessed_v3": {
                "source": "reports/vlm/visual_demo/model_defect_summary_postprocessed_v3.json",
                "defect_counts": old_summary.get("defect_counts"),
            },
            "after_roomspace_v4": {
                "source": "reports/vlm/visual_demo/model_defect_summary_roomspace_v4.json",
                "defect_counts": summary.get("defect_counts"),
            },
            "done_when": {
                "extra_room_lte_3": int((summary.get("defect_counts") or {}).get("extra_room", 0)) <= 3,
                "room_without_label_lte_2": int((summary.get("defect_counts") or {}).get("room_without_label", 0)) <= 2,
                "label_without_room_lte_2": int((summary.get("defect_counts") or {}).get("label_without_room", 0)) <= 2,
                "roomspace_visual_residuals_zero": int((summary.get("defect_counts") or {}).get("extra_room", 0)) == 0
                and int((summary.get("defect_counts") or {}).get("room_without_label", 0)) == 0
                and int((summary.get("defect_counts") or {}).get("label_without_room", 0)) == 0,
            },
        },
        "visual_demo_ablation_scope": {
            "source": "reports/vlm/room_validity_gate_ablation_v4.json",
            "scope": ablation.get("scope"),
            "metrics": ((ablation.get("variants") or {}).get("roomlink_plus_review_gate") or {}),
            "note": "This ablation is scoped to the five visual-demo CubiCasa drawings only.",
        },
        "outputs": {
            "predictions": "reports/vlm/real_upstream_model_postprocessed_predictions_roomlink_v3.jsonl",
            "review_pack": "reports/vlm/visual_demo_roomspace_v4/review_pack_v2/index.html",
            "defect_cases": "reports/vlm/visual_demo/model_defect_cases_roomspace_v4.jsonl",
        },
    }
    write_json(ROOT / "reports/vlm/room_space_v4_visual_eval.json", report)
    print(json.dumps(report["visual_defect_delta"]["done_when"], ensure_ascii=False, indent=2))


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
