#!/usr/bin/env python3
"""Generate paper-submission claim, limitation, and ablation notes."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent.parent
REPORTS = ROOT / "reports" / "vlm"


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def metric(data: dict[str, Any], key: str) -> Any:
    return (data.get("paper_main_metrics") or {}).get(key)


def write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    recon = load_json(REPORTS / "paper_e2e_metric_reconciliation_v1.json")
    ceiling = load_json(REPORTS / "relation_no_repair_ceiling_diagnostic_v1.json")
    ocr = load_json(REPORTS / "text_dimension_external_ocr_lock_v3.json")
    symbol = load_json(REPORTS / "symbol_cross_source_lock_v1.json")
    ledger = load_json(REPORTS / "final_claim_ledger_v1.json")
    manifest = load_json(REPORTS / "paper_artifact_manifest_v1.json")
    pm = recon.get("paper_main_metrics") or {}
    rel = pm.get("relation_f1")
    node = pm.get("node_macro_f1")
    invalid = pm.get("invalid_graph_rate")

    claims = f"""# Paper Submission Claims v1

## Main Claim

CadStruct-MoE should be framed as an auditable structured MoE pipeline for floorplan scene-graph parsing, using deterministic family routing, expert decomposition, label-level arbitration, and constraint-aware no-repair fusion.

## Main Numbers

- Setting: `{recon.get("paper_main_setting")}`
- Source: `{recon.get("paper_main_source_file")}`
- Records: {pm.get("records")}
- Node Macro F1: {node}
- Node Accuracy: {pm.get("node_accuracy")}
- Relation F1 (no repair): {rel}
- Relation Precision / Recall: {pm.get("relation_precision")} / {pm.get("relation_recall")}
- Invalid Graph Rate: {invalid}

## Appendix-Only Numbers

- Repair-enabled relation score remains an upper-bound diagnostic because it uses gold source/target/relation labels.
- Learned/top-k router experiments belong in appendix or future work; deterministic family routing is the main mechanism.
- Lie/SE(2) and SheetLayout are auxiliary, historical, or non-core extensions unless new matched baselines are added.

## Do Not Claim

- Broad scanned/photo OCR robustness while `{ocr.get("status")}`.
- Cross-source symbol generalization while `{symbol.get("status")}`.
- General CAD understanding beyond the locked floorplan scene-graph setting.

Claim ledger status: `{ledger.get("status")}`. Artifact manifest status: `{manifest.get("status")}`.
"""

    limitations = f"""# Paper Submission Limitations v1

## Relation Boundary

The main no-repair relation F1 is {rel}, which clears the submission minimum but does not reach the preferred 0.90 target. The relation ceiling diagnostic is appendix-only and attributes remaining errors to bbox-only containment ambiguity, upstream room/symbol label errors, and hard multi-room cases.

Hard-case export: `reports/vlm/relation_no_repair_hard_cases_v1.jsonl`.

## External OCR

External OCR lock status is `{ocr.get("status")}` with {((ocr.get("human_gold") or {}).get("drawings_with_transcript_and_bbox"))} drawings containing transcript+bbox gold. The paper may describe the annotation pack and internal TextDimension evidence, but not broad OCR robustness.

## Cross-Source Symbols

Cross-source symbol lock status is `{symbol.get("status")}` with {((symbol.get("human_gold") or {}).get("gold_symbol_annotations"))} gold symbol annotations. Generic symbol, bathtub, equipment, and table-like long-tail cases should be discussed as limitations.

## Non-Core Extensions

SheetLayout is non-core/future work. Learned routing and Lie/SE(2) modules should remain appendix, auxiliary, or historical unless the paper is reframed around those mechanisms and new matched baselines are run.
"""

    ablation = f"""# Paper Submission Ablation Story v1

## Narrative

The ablation story should separate architecture from evidence. The main result is not a learned router story; it is a structured decomposition story where deterministic family routing isolates the search space and makes each expert auditable.

## Components

- Deterministic family router: main routing mechanism; avoids the learned-router wrong-expert rate observed in fair routing ablations.
- Expert decomposition: separates boundary, room/space, symbol/fixture, and text/dimension predictions.
- Boundary and symbol label arbitration: raises real-upstream node quality; current main node macro F1 is {node}.
- Constraint-aware no-repair fusion: produces valid scene graphs without gold ID-space relation repair; current relation F1 is {rel}, invalid graph rate is {invalid}.
- Repair-enabled relation score: appendix upper bound only.

## Recommended Table Roles

- Main table: no-repair v2 E2E, expert standalone metrics, invalid graph rate.
- Ablation table: deterministic router, arbitration steps, no-repair fusion variants.
- Appendix: top-k/learned router, relation ceiling diagnostics, hard cases, OCR and cross-source annotation locks, Lie/SE(2), SheetLayout.
"""

    write(REPORTS / "paper_submission_claims_v1.md", claims)
    write(REPORTS / "paper_submission_limitations_v1.md", limitations)
    write(REPORTS / "paper_submission_ablation_story_v1.md", ablation)
    print("wrote paper submission writing pack")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
