# CadStruct-MoE P308 Submission Package

Generated: 2026-06-02T11:32:56

This directory packages the current generic P308 manuscript and the material needed to defend the CadStruct-MoE submission boundary.

## Compile

```bash
cd paper_submission/cadstruct_moe_p308
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

The expected PDF is `main.pdf`.

## Main Source

- `main.tex`: copied from `reports/vlm/p308_generic_submission_manuscript.tex`
- `claim_boundary.md`: reviewer-safe claim boundary and common-method positioning
- `evidence_manifest.json`: packaged source/material inventory

## Figures

- `paper_submission/cadstruct_moe_p308/figures/paper_candidate_figure_v13_real_e2e.png` from `reports/vlm/visual_demo_model_v13_real_e2e/paper_candidate_figure_v13_real_e2e.png`
- `paper_submission/cadstruct_moe_p308/figures/paper_candidate_figure_v13_real_e2e.svg` from `reports/vlm/visual_demo_model_v13_real_e2e/paper_candidate_figure_v13_real_e2e.svg`

## Materials

- `paper_submission/cadstruct_moe_p308/materials/p308_submission_package.json` from `reports/vlm/p308_submission_package.json`
- `paper_submission/cadstruct_moe_p308/materials/p308_submission_static_check.json` from `reports/vlm/p308_submission_static_check.json`
- `paper_submission/cadstruct_moe_p308/materials/p308_submission_static_check.md` from `reports/vlm/p308_submission_static_check.md`
- `paper_submission/cadstruct_moe_p308/materials/p308_submission_readiness.json` from `reports/vlm/p308_submission_readiness.json`
- `paper_submission/cadstruct_moe_p308/materials/p308_submission_readiness.md` from `reports/vlm/p308_submission_readiness.md`
- `paper_submission/cadstruct_moe_p308/materials/p307_manuscript_refresh_package.json` from `reports/vlm/p307_manuscript_refresh_package.json`
- `paper_submission/cadstruct_moe_p308/materials/p307_manuscript_refresh_package.md` from `reports/vlm/p307_manuscript_refresh_package.md`
- `paper_submission/cadstruct_moe_p308/materials/p307_manuscript_refresh_tables.tex` from `reports/vlm/p307_manuscript_refresh_tables.tex`
- `paper_submission/cadstruct_moe_p308/materials/p306_paper_materials_package.json` from `reports/vlm/p306_paper_materials_package.json`
- `paper_submission/cadstruct_moe_p308/materials/p306_paper_materials_package.md` from `reports/vlm/p306_paper_materials_package.md`
- `paper_submission/cadstruct_moe_p308/materials/p305c_external_generalization_claim_decision.json` from `reports/vlm/p305c_external_generalization_claim_decision.json`
- `paper_submission/cadstruct_moe_p308/materials/p305c_external_generalization_claim_decision.md` from `reports/vlm/p305c_external_generalization_claim_decision.md`
- `paper_submission/cadstruct_moe_p308/materials/p304_claim_consistency_check.json` from `reports/vlm/p304_claim_consistency_check.json`
- `paper_submission/cadstruct_moe_p308/materials/p304_claim_consistency_check.md` from `reports/vlm/p304_claim_consistency_check.md`
- `paper_submission/cadstruct_moe_p308/materials/p304_p301_reviewed_metric_package.json` from `reports/vlm/p304_p301_reviewed_metric_package.json`
- `paper_submission/cadstruct_moe_p308/materials/p304_p301_reviewed_metric_package.md` from `reports/vlm/p304_p301_reviewed_metric_package.md`
- `paper_submission/cadstruct_moe_p308/materials/p303_model_code_review.json` from `reports/vlm/p303_model_code_review.json`
- `paper_submission/cadstruct_moe_p308/materials/p303_model_code_review.md` from `reports/vlm/p303_model_code_review.md`
- `paper_submission/cadstruct_moe_p308/materials/p301_relation_confidence_preserved_conservative_rescue.json` from `reports/vlm/p301_relation_confidence_preserved_conservative_rescue.json`
- `paper_submission/cadstruct_moe_p308/materials/p301_relation_confidence_preserved_conservative_rescue.md` from `reports/vlm/p301_relation_confidence_preserved_conservative_rescue.md`
- `paper_submission/cadstruct_moe_p308/materials/sci2_final_submission_evidence_pack_v2.json` from `reports/vlm/sci2_final_submission_evidence_pack_v2.json`
- `paper_submission/cadstruct_moe_p308/materials/sci2_overclaim_scan_v2.json` from `reports/vlm/sci2_overclaim_scan_v2.json`
- `paper_submission/cadstruct_moe_p308/materials/final_claim_ledger_v2.json` from `reports/vlm/final_claim_ledger_v2.json`
- `paper_submission/cadstruct_moe_p308/materials/cadstruct-paper-core-contributions-v2.md` from `docs/cadstruct/paper/cadstruct-paper-core-contributions-v2.md`
- `paper_submission/cadstruct_moe_p308/materials/real-world-capability-boundary-v3.md` from `docs/cadstruct/paper/real-world-capability-boundary-v3.md`
- `paper_submission/cadstruct_moe_p308/materials/model-asset-inventory.md` from `docs/cadstruct/current/model-asset-inventory.md`
- `paper_submission/cadstruct_moe_p308/materials/struct.json` from `struct.json`
- `paper_submission/cadstruct_moe_p308/materials/struct_audit.json` from `struct_audit.json`
- `paper_submission/cadstruct_moe_p308/materials/todo.json` from `todo.json`

## Submission Boundary

The package does not frame artificial/manual/external data collection as the core contribution. The core submission path is the reviewed SVG/contract CadStruct-MoE evidence line, backed by common-method comparisons, ablations, claim ledgers, and source-integrity boundaries.
