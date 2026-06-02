# CadStruct Visual Chain Model Decision v2

Date: 2026-05-07

## Current Decision

The CubiCasa visual review pack now uses `source_mode=real_upstream_saved_model_predictions` for node labels, followed by `NodeQualityGate v3` postprocessing. Geometry proposals still come from parser/SVG candidates.

This means the pack is a real saved-model label visualization over candidate geometry. It is not an `expected_json` oracle-smoke result, and it is not pure raster end-to-end detection from the original floorplan image.

## What Changed

- Raw model predictions: `reports/vlm/e2e_cubicasa_visual_demo_model_predictions.jsonl`
- Postprocessed predictions: `reports/vlm/real_upstream_model_postprocessed_predictions_v3.jsonl`
- Visual defect audit: `reports/vlm/visual_demo/model_defect_summary_postprocessed_v3.json`
- Candidate geometry audit: `reports/vlm/candidate_geometry_audit_v3.json`
- Claim boundary: `reports/vlm/real_model_visual_claim_boundary_v3.md`

The previous `expected_json`/oracle path is only valid as an output-contract or renderer smoke test. It must not be mixed into real model performance claims.

## Remaining Model Work

TextDimension still needs metadata/retraining work because many SVG dimension-like candidates have no readable text, while a small number of true text candidates are filtered as geometry-invalid. Boundary, SymbolFixture, and RoomSpace also need hard-case retraining rather than relying only on quality gating.
