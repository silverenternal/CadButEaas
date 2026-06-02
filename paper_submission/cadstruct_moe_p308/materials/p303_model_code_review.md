# P303 Model Code Review

Status: `passed_with_packaging_restrictions`.

## Decision

P304 may proceed, but only as reviewed SVG/contract experiment-line packaging with explicit internal locked fine-threshold relation-audit language. Do not present P301 relation F1 as external validation or runtime raster performance.

## Findings

### P303-F1 - high

**Fine relation thresholds are selected on the locked benchmark and cannot be reported as external validation.**

Impact: P301 relation F1=0.929782 is useful internal evidence, but paper packaging must label it as an internal locked fine-threshold audit. It is not a held-out external metric.

Required action: P304 may proceed only if all tables and prose mark P292/P294/P295/P296/P297/P300/P301/P302 relation values as internal locked threshold audits.

Evidence:
- scripts/vlm/evaluate_symbol_relation_confidence_preserved_rescue_p301.py:122-145 builds locked fine_eval, sweeps thresholds, selects sweep[0], and reports selected_threshold.
- scripts/vlm/audit_relation_no_repair_sci2_scorer_v1.py:207-220 sorts threshold_sweep by relation F1/precision on the same evaluated rows.
- reports/vlm/scene_graph_fusion_symbol_relation_confidence_preserved_rescue_p301_fine_relation_no_repair_scorer_v1_eval.json reports selected_threshold=0.985 and claim_boundary=Internal locked fine-threshold audit; not external validation.

### P303-F2 - medium

**Evaluation/fusion helpers use expected_json/gold-compatible IDs and must stay offline-only.**

Impact: This is acceptable for SVG/contract offline audit, but it cannot be described as a runtime raster inference path.

Required action: P304 must keep SVG/contract normalized-candidate claims separate from runtime raster claims and must not promote these helpers as runtime-safe inference artifacts.

Evidence:
- scripts/vlm/fuse_real_upstream.py:56-62 documents reconstructing from dev split to preserve gold-compatible scene graph node IDs.
- scripts/vlm/audit_relation_no_repair_sci2_scorer_v1.py:58-64 builds gold_edge_set from expected_json for labels/evaluation.
- todo.json hard_contract forbids expected_json, gold labels, annotation paths, offline object IDs, and row_id as runtime features.

### P303-F3 - medium

**Prediction relabeling relies on symbol stream order matching locked_items order.**

Impact: Existing reports are probably consistent because counts match and previous pipeline preserves order, but a future shuffled prediction file could silently relabel the wrong candidate while passing the count check.

Required action: Before changing prediction sources, add candidate_id/record_index alignment assertions or a keyed relabeler. P304 can proceed using existing frozen artifacts with this fragility documented.

Evidence:
- scripts/vlm/evaluate_symbol_relation_confidence_preserved_rescue_p301.py:148-186 increments symbol_index over base_predictions and compares only total symbol count to locked_items length.
- scripts/vlm/evaluate_symbol_pairwise_micro_rescue_p302.py:275-302 has the same order-based mapping pattern.
- Metadata writes record_index/candidate_id after assignment, but there is no per-row assert that prediction candidate_id equals locked_items[symbol_index].

### P303-F4 - low

**Node residual relabeler selection keeps train/dev/locked separation for P301.**

Impact: No blocker-level leakage was found in P301 node-label model/threshold selection during this pass.

Required action: Keep P301 as the current reviewed SVG/contract experiment-line candidate, subject to F1/F2/F3 packaging restrictions.

Evidence:
- scripts/vlm/evaluate_symbol_relation_confidence_preserved_rescue_p301.py:216-260 trains on train, scores dev candidates, and selects candidate from dev metrics.
- scripts/vlm/evaluate_symbol_relation_confidence_preserved_rescue_p301.py:261-292 applies the selected policy to locked once and writes adjusted predictions/fine audit.
- scripts/vlm/evaluate_symbol_bathtub_binary_rescue_p289.py:136-146 and evaluate_symbol_conservative_multilabel_overlay_p285.py:276-285 include split image overlap checks.

### P303-F5 - low

**P301 relation confidence preservation is implemented as claimed.**

Impact: The implementation supports the P301 confidence-preservation narrative.

Required action: P304 should describe this as a conservative residual relabeler on top of P297, not as a new broad architecture.

Evidence:
- scripts/vlm/evaluate_symbol_relation_confidence_preserved_rescue_p301.py:163-167 changes label but copies previous_confidence back into row['confidence'].
- scripts/vlm/evaluate_symbol_relation_confidence_preserved_rescue_p301.py:169-180 records relation_confidence_preserved_from_p297 metadata.
- reports/vlm/p301_relation_confidence_preserved_conservative_rescue.json states relation_confidence_policy=preserve P297 prediction confidence when label changes.

## P304 Gate

P304 may proceed only under these restrictions:
- P301 is labeled SVG/contract normalized-candidate experiment-line, not raster detector performance.
- Relation F1 values from P292/P294/P295/P296/P297/P300/P301/P302 are labeled internal locked fine-threshold audits.
- Packaging does not claim external validation unless a new external-source evaluation is run.
- Prediction source changes require candidate_id/record_index alignment assertions before promotion.
