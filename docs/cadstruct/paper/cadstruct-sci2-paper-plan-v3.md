# CadStruct SCI2 Paper Plan v4

> **Version**: v4.0 | **Date**: 2026-05-01 | **Supersedes**: v3

This plan maps current claims to datasets, metrics, baselines, ablations, and status. All 9 phases (R0-R8) are complete; evidence is collected and paper-ready.

## C1. Structured MoE pipeline emits schema-valid scene graphs.

- **Status**: ✅ **SUPPORTED** (smoke benchmark)
- Dataset: `datasets/cadstruct_real_world_benchmark_v3/smoke.jsonl` (64 records)
- Metrics: schema_valid_rate=1.0, node_f1=1.0, relation_f1=0.918, invalid_graph_rate=0.0
- Baseline: expected-json deterministic upstream; VLM zero-shot (semantic F1=0.27)
- Ablation: no-constraint-fusion → invalid rate 14.8% (vs 0.0% with fusion)
- Evidence: `reports/vlm/e2e_real_pipeline_smoke_audit.json`, `reports/vlm/scene_graph_fusion_v2_eval.json`

## C2. WallOpening recognition reaches production-grade accuracy.

- **Status**: ✅ **SUPPORTED** (locked benchmark)
- Dataset: benchmark v3 locked split (CubiCasa5K + CVC-FP + FloorPlanCAD)
- Metrics: accuracy=0.993, macro F1=0.989, R²=0.980
- By source: CVC-FP F1=0.989 ✅, FloorPlanCAD F1=0.969 ⚠️ (target 0.98)
- Evidence: `reports/vlm/wall_opening_floorplancad_residual_v1_eval.json`

## C3. Room type classification is near-perfect with proposal assistance.

- **Status**: ✅ **SUPPORTED** (locked benchmark)
- Metrics: macro F1=0.982 (strict), 0.989 (review-adjusted), proposal recall@IoU0.5=1.0
- 11 room classes; toilet/corridor/bedroom reach F1=1.0
- Evidence: `reports/vlm/room_space_v5_t046_review_adjusted_auto_accept.json`

## C4. Symbol classification works for 7 core classes via MLP with hand-crafted features.

- **Status**: ⚠️ **PARTIALLY SUPPORTED** (7-class达标, 9-class未达标)
- Metrics: 7-class macro F1=0.921 ✅, 9-class macro F1=0.717 ❌
- Root cause: `generic_symbol` (0 train) and `table` (1 train) are invalid classes
- CNN approach failed (F1=0.12) due to tiny crop sizes (3-11px)
- Evidence: `reports/vlm/symbol_fixture_crop_encoder_v65_eval.json`

## C5. TextDimension with OCR text pattern matching.

- **Status**: ⚠️ **PARTIALLY SUPPORTED** (OCR exact达标, macro F1未达标)
- Metrics: macro F1=0.858 (target 0.95), relation F1=0.868 (target 0.95), OCR exact=1.0 ✅
- Gap: 2,687 dimension_text items (23%) have empty raw_text, causing recall=0.773
- OCR pattern matching: 'x' separator → dimension_text (100% precision), alpha-only → room_label (96% accuracy)
- Evidence: `reports/vlm/text_dimension_expert_v3_eval.json`

## C6. Constraint-guaranteed scene graph fusion.

- **Status**: ✅ **SUPPORTED** (ALL done-when checks passed)
- Metrics: node F1=1.0, relation F1=0.918, invalid_graph_rate=0.0
- 6 relation types, 6 gated repair rules, 52 symbol_room_containment repairs
- Without fusion: invalid rate jumps to 14.8%
- Evidence: `reports/vlm/scene_graph_fusion_v2_eval.json`

## C7. MoE router achieves perfect routing accuracy.

- **Status**: ✅ **SUPPORTED**
- Metrics: effective_rate=1.0, wrong_expert_rate=0.0, locked accuracy=1.0
- 134,043 candidates across 4 families (wall_opening/text_dimension/symbol_fixture/room_space)
- Top features: room_type_code (0.133), symbol_type_code (0.127), confidence (0.116)
- Evidence: `reports/vlm/moe_router_v2_eval.json`

## C8. Degradation-aware robustness pipeline.

- **Status**: ✅ **SUPPORTED**
- 7 degradation types, 700 records generated traceably
- Quality scorer AUROC=0.869 (target 0.80), router accuracy=0.845 (target 0.80)
- Node F1 drop under degradation: 1.75pp (target ≤5pp)
- Evidence: `reports/vlm/degraded_robustness_v1_eval.json`, `reports/vlm/quality_failure_scorer_v1_eval.json`

## C9. Source generalization gaps are explicitly reported.

- **Status**: ✅ **SUPPORTED**
- LOSO matrix across 4 sources, few-shot curves (3 experts × 4 strategies × 5 shot levels)
- Domain generalization: adversarial training reduces FloorPlanCAD gap from 13.7pp to 7.3pp
- All 21 leakage audit checks pass
- Evidence: `reports/vlm/loso_eval_matrix_v3.json`, `reports/vlm/few_shot_adaptation_curve_v1.json`, `reports/vlm/domain_generalization_ablation_v1.json`

## C10. VLM is a baseline, not the main recognizer.

- **Status**: ✅ **SUPPORTED** (negative control in ablation)
- VLM-as-main: node F1 drops 25.2pp, relation F1 drops 8.0pp, invalid rate +19.9pp, 42x slower
- Zero-shot: InternVL3.5-14B semantic F1=0.27, relation F1=0.19
- Evidence: `reports/vlm/innovation_ablation_v2.json`

## C11. Comprehensive innovation ablation with 7 controls.

- **Status**: ✅ **SUPPORTED** (all 50 checks pass)
- 7 ablation controls: no-moe, no-geometry, no-constraint-fusion, no-quality-router, no-hardcase-loop, no-router-trace, vlm-as-main
- Key finding: no-geometry kills relations (-11.3pp), vlm-as-main is worst (-25.2pp)
- Evidence: `reports/vlm/innovation_ablation_v2.json`

## C12. Training audit and reproducibility.

- **Status**: ✅ **SUPPORTED**
- 5-class coverage: WallOpening, Room, Symbol, Text, Router
- Each run: git hash, env hash, dataset hash, peak memory, OOM/skip audit
- CI smoke passes all regression thresholds
- Evidence: `reports/vlm/training_contract_coverage_v2.json`, `reports/vlm/ci_regression_thresholds_v1.json`

## C13. Benchmark v3: 4-source, zero-leakage floorplan benchmark.

- **Status**: ✅ **SUPPORTED** (automated work complete; human review pending for internal-real-v3)
- 1,574 records, 4 sources (CubiCasa5K/CVC-FP/FloorPlanCAD/internal-real-v3)
- Zero leakage (image hash + annotation hash + path overlap + teacher contamination)
- 181 unique internal-real candidates, 100-record double-review pack ready
- Evidence: `datasets/cadstruct_real_world_benchmark_v3/manifest.json`, `reports/vlm/benchmark_v3_leakage_audit.json`

## C14. Real-world 0.98 all-element claim.

- **Status**: ❌ **NOT YET SUPPORTED**
- Blocked by: (1) human review for internal-real-v3 locked split, (2) TextDimension F1 gap, (3) Symbol 9-class gap, (4) FloorPlanCAD domain gap
- Evidence: `reports/vlm/capability_boundary_v3.json`

---

## Summary: Claims Readiness

| Claim | Status | Key Metric | Target |
|-------|--------|------------|--------|
| C1. Schema-valid scene graphs | ✅ | invalid=0.0, node=1.0, rel=0.918 | ≥0.90/≥0.85/≤0.03 |
| C2. WallOpening production-grade | ✅ | acc=0.993, F1=0.989 | ≥0.99/≥0.98 |
| C3. Room classification | ✅ | F1=0.982 | ≥0.98 |
| C4. Symbol 7-class | ✅ | F1=0.921 | ≥0.90 |
| C4b. Symbol 9-class | ❌ | F1=0.717 | ≥0.90 |
| C5. TextDimension | ⚠️ | F1=0.858, OCR=1.0 | ≥0.95/≥0.90 |
| C6. Constraint fusion | ✅ | node=1.0, rel=0.918, invalid=0.0 | ≥0.90/≥0.85/≤0.03 |
| C7. MoE router | ✅ | effective=1.0, wrong=0.0 | ≥0.98/≤0.02 |
| C8. Degraded robustness | ✅ | AUROC=0.869, drop=1.75pp | ≥0.80/≤5pp |
| C9. Source generalization | ✅ | LOSO + few-shot + DG ablation | — |
| C10. VLM baseline | ✅ | VLM-as-main: -25.2pp | — |
| C11. Innovation ablation | ✅ | 7 controls, 50/50 checks | — |
| C12. Training audit | ✅ | 5-class contract, CI pass | — |
| C13. Benchmark v3 | ✅ | 1574 records, 4 sources, 0 leak | — |
| C14. All-element 0.98 | ❌ | Blocked by 4 items | — |

**Paper-ready claims**: C1, C2, C3, C4 (7-class), C6, C7, C8, C9, C10, C11, C12, C13
**Partially supported**: C5 (TextDimension OCR gap)
**Not supported**: C4b (Symbol 9-class), C14 (all-element 0.98)
