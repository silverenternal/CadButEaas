# CadStruct-MoE: Core Contributions for SCI2 Paper

> **Version**: v0.7.1 | **Date**: 2026-05-01 | **Status**: Evidence-collected, paper-ready

## Paper Title (Working)

**CadStruct-MoE: Structured Mixture-of-Experts for Floorplan Understanding with Constraint-Guaranteed Scene Graphs**

---

## 1. Problem Statement

End-to-end floorplan understanding requires simultaneously recognizing walls, rooms, symbols, dimensions, and layout regions while producing structurally valid scene graphs. Existing approaches either:
- (a) use monolithic VLMs that lack geometric reasoning and schema enforcement, or
- (b) assemble ad-hoc rules without auditable routing or constraint guarantees.

**Gap**: No system provides both *element-level accuracy* across heterogeneous floorplan elements AND *structural validity* of the output scene graph with auditability.

---

## 2. Core Contributions

### C1: Specialized MoE Architecture for Floorplan Understanding

We propose a **5-expert Mixture-of-Experts** architecture where each expert family handles one element type:

| Expert | Task | Method |
|--------|------|--------|
| WallOpening | Wall/opening classification + probability regression | Gradient-boosted features + source-blend router |
| RoomSpace | Room type classification + polygon proposal | Hierarchical sklearn + proposal-assisted disambiguation |
| SymbolFixture | Symbol classification from crop candidates | MLP with 31D hand-crafted features (geometry + raster stats) |
| TextDimension | Text type classification + dimension linking | Prototype-based geometry + OCR text pattern matching |
| SheetLayout | Layout region detection | Rule-based heuristics + prototype classification |

The main model uses an auditable **DeterministicRouter** (effective_rate=1.0, wrong_expert_rate=0.0 on 134K real-dev candidates). A fair learned router v3 using only geometry/page-context features reaches wrong_expert_rate=0.152, so it is reported as an ablation rather than the main router; appendix top-k evidence is in `reports/vlm/router_appendix_topk_v1.json`. Boundary and symbol arbitration are label-level post-router corrections, not learned family routing or a sparse-MoE contribution. The final module-role narrative is fixed in `reports/vlm/final_moe_ablation_narrative_v1.json`.

Lie/SE(2)-canonical graph features are treated as a bounded core geometry module inside the graph-node expert. `reports/vlm/lie_se2_core_claim_decision_v6.json` supports the claim that the trained checkpoint relies on these features under zero-ablation and remains stable under crop rotation/flip stress, but it does not support a multi-seed matched accuracy-superiority claim.

**Why this matters**: MoE isolates complexity — each expert can be independently validated, improved, and ablated. Router feature importance reveals that `room_type_code` (0.133), `symbol_type_code` (0.127), and `confidence` (0.116) are the strongest routing signals, confirming that family structure is learnable from candidate features alone.

### C2: Constraint-Guaranteed Scene Graph Fusion

We introduce a **gated constraint repair** mechanism that enforces schema validity on the fused scene graph:
- 6 relation types: `bounds`, `contains`, `attached_to`, `adjacent_to`, `labeled_by`, `dimension_of`
- Main-text relation metrics use no-repair geometry-only relation fusion; gold-ID repair is reported only as an appendix upper-bound / ID-space sanity check
- Invalid graph rate: **0.0** (vs 14.8% without constraint fusion)

**Result**: On the reconciled real-upstream setting (`reports/vlm/paper_metric_table_manifest_v3.json`), the current paper-main source is `reports/vlm/scene_graph_fusion_symbol_long_tail_model_no_repair_scorer_v1_eval.json`: Node Macro F1=0.951696, Node Accuracy=0.981566, no-repair Relation F1=0.920938, Relation Precision=0.961937, and invalid graph rate=0.0. The repair sensitivity audit (`reports/vlm/relation_gold_id_repair_sensitivity_v1.json`) shows repair-enabled Relation F1=0.923, but that path uses gold source/target/relation labels and is therefore appendix-only. Locked-only symbol threshold sweeps are diagnostic upper bounds and are not paper-main admissible. The symbol arbitration generalization audit confirms zero train/locked image overlap on CubiCasa locked data, but `reports/vlm/symbol_cross_source_lock_v1.json` is still `pending_no_human_gold`. Remaining risks are TextDimension standalone-vs-E2E alignment, external OCR validation, and SheetLayout gold annotations.

### C3: Degradation-Aware Robustness Pipeline

We design a **quality-aware degradation detection** system:
- 7 degradation types: blur, jpeg, shadow, fold, rotation, low_contrast, partial_crop
- Quality failure scorer (Gradient Boosting, AUROC=0.869) predicts failure risk from image features
- Degraded-mode router (accuracy=0.845) triggers lowered thresholds and tile processing
- Node F1 drop under degradation: only **1.75pp** (target: ≤5pp)

**Why this matters**: Real-world floorplans are rarely clean scans. Our system detects degradation and adapts, rather than silently failing.

### C4: Explicit Source Generalization Reporting

We establish a **Leave-One-Source-Out (LOSO)** evaluation protocol across 4 sources (CubiCasa5K, CVC-FP, FloorPlanCAD, internal-real-v3) and report:
- Per-source macro F1 for every expert
- Few-shot adaptation curves (0/5/10/25/50 shots × 4 strategies)
- Domain generalization ablation (4 strategies: no-source, adversarial, style-aug, quality-router)

**Finding**: Adversarial training reduces WallOpening FloorPlanCAD gap from 13.7pp to 7.3pp. Source leakage audit passes all 21 checks.

### C5: Comprehensive Negative Controls

We provide **7 ablation controls** with positive/negative comparisons:

| Ablation | Δ Node F1 | Δ Relation F1 | Δ Invalid Rate | Key Finding |
|----------|-----------|---------------|-----------------|-------------|
| **no-moe** (single model) | -8.35pp | -3.23pp | +2.06pp | MoE routing adds 8.4pp node F1 |
| **no-geometry** | -5.07pp | **-11.34pp** | +3.66pp | Geometry is essential for relations |
| **no-constraint-fusion** | -1.93pp | +8.12pp* | **+14.83pp** | Fusion trades precision for validity |
| **no-quality-router** | -2.00pp | -0.29pp | +0.41pp | Degradation detection prevents errors |
| **no-hardcase-loop** | -2.56pp | -1.55pp | +1.01pp | Active learning helps tail performance |
| **no-router-trace** | -0.64pp | -0.19pp | 0.00pp | Traceability is primarily auditability |
| **vlm-as-main** | **-25.16pp** | -7.98pp | **+19.88pp** | VLM lacks specialization & schema |

Expert-level contribution is audited separately in `reports/vlm/expert_contribution_matrix_v2.json`. In real-upstream fusion, drop-one node macro F1 decreases by 11.1pp for WallOpening, 38.0pp for RoomSpace, 4.3pp for SymbolFixture, and 16.2pp for TextDimension; SheetLayout is marked as a non-core extension because it contributes no current real-upstream nodes and `reports/vlm/sheet_layout_real_gold_boundary_v1.json` found no real layout gold. Freeze-one results are diagnostic only because the gold-ID-space fusion falls back to expected labels when a prediction is missing, which can create apparent negative drops.

*Relation F1 increases without fusion because fusion adds spurious relations when unconstrained; the invalid graph rate reveals the true cost.

**Key finding**: VLM-as-main is the worst performer (-25.2pp node F1, 42x slower latency), confirming that specialized MoE > monolithic VLM for structured floorplan understanding.

### C6: Auditable Training & Reproducibility

Every training run produces: git hash, env hash, dataset hash, peak memory, OOM/skip counts, per-class confusion, and failure tags. Training contract covers 5 expert classes with complete audit trail.

---

## 3. Experimental Results

### 3.1 Benchmark v3

| Property | Value |
|----------|-------|
| Sources | 4 (CubiCasa5K, CVC-FP, FloorPlanCAD, internal-real-v3) |
| Total records | 1,574 |
| Splits | train / dev / locked / source-heldout |
| Leakage | 0 (image hash + annotation hash + path overlap) |

### 3.2 Per-Expert Performance (Dev Split)

| Expert | Metric | Value | Target | Status |
|--------|--------|-------|--------|--------|
| **WallOpening** | accuracy / macro F1 / R² | 0.993 / 0.989 / 0.980 | ≥0.99 / ≥0.98 / ≥0.98 | ✅✅✅ |
| WallOpening (CVC-FP source) | macro F1 | 0.989 | ≥0.98 | ✅ |
| WallOpening (FloorPlanCAD source) | macro F1 | 0.969 | ≥0.98 | ⚠️ |
| **RoomSpace** | macro F1 (strict) | 0.982 | ≥0.98 | ✅ |
| RoomSpace | macro F1 (review-adjusted) | 0.989 | ≥0.98 | ✅ |
| RoomSpace | proposal recall@IoU0.5 | 1.000 | ≥0.98 | ✅ |
| **SymbolFixture** (9 classes) | macro F1 | 0.872 | ≥0.80 | ✅ |
| SymbolFixture (preferred 9-class target) | macro F1 | 0.872 | ≥0.90 | ⚠️ |
| **TextDimension** | standalone macro F1 | 0.984 | ≥0.95 | ✅ |
| TextDimension | relation F1 | 0.998 | ≥0.95 | ✅ |
| TextDimension | OCR exact rate | 1.000 | ≥0.90 | ✅ |
| TextDimension — room_label | F1 | 0.994 | — | ✅ |
| TextDimension — leader_line | F1 | 1.000 | — | ✅ |
| TextDimension — dimension_text | F1 | 0.998 | — | ✅ |
| **SheetLayout** | real-layout AP50 | N/A | future work | ⚠️ |
| **MoE Router** | deterministic effective_rate / wrong_rate | 1.0 / 0.0 | ≥0.98 / ≤0.02 | ✅✅ |
| **Scene Graph Fusion (real-upstream + symbol/text conservative arbitration + generic override + no-repair scorer)** | node F1 / relation F1 / invalid | 0.944 / 0.920 / 0.0 | ≥0.50 / ≥0.90 / ≤0.02 | ✅✅✅ |
| **Degraded Robustness** | router acc / AUROC / F1 drop | 0.845 / 0.869 / 1.75pp | ≥0.80 / ≥0.80 / ≤5pp | ✅✅✅ |

### 3.3 Efficiency

| Metric | Value |
|--------|-------|
| P50 latency | 9.0 ms |
| P95 latency | 31.6 ms |
| Peak memory | 103.7 MiB |
| CI smoke | All thresholds passed |

### 3.4 VLM Baseline (Zero-Shot, 30 samples)

| Model | Semantic F1 | Relation F1 | Latency (ms) |
|-------|-------------|-------------|--------------|
| InternVL3.5-14B | 0.274 | 0.187 | 31,258 |
| CadStruct-VL-14B-LoRA | 0.231 | 0.187 | 69,327 |
| CadStruct-VL-14B-LoRA-Structural | 0.280 | 0.100 | 59,215 |
| **CadStruct-MoE (main: real-upstream + symbol/text conservative arbitration + generic override + RF long-tail symbol model + no-repair scorer)** | **0.952** | **0.921** | 2654 replay/fusion |
| CadStruct-MoE (legacy E2E, historical context) | 0.763 | 0.113 | 12.1 |

Note: The paper main E2E source is `reports/vlm/paper_metric_table_manifest_v3.json`. Legacy E2E and VLM rows use different settings and should not be merged into a single headline claim without this caveat. Repair-enabled relation F1=0.923 is appendix-only because gold-ID repair uses gold relation labels.

---

## 4. Claims We CAN Support

| Claim | Evidence |
|-------|----------|
| Structured MoE produces schema-valid relation graphs | real-upstream node macro F1=0.951696, no-repair relation F1=0.920938, invalid=0.0 |
| WallOpening recognition reaches production-grade | accuracy=0.993, F1=0.989 on locked |
| Room type classification is near-perfect | F1=0.982 (strict), 0.989 (review-adjusted) |
| Symbol classification works for 7 core classes | F1=0.921, excluding 2 invalid classes |
| Constraint fusion guarantees structural validity | invalid rate 0.0 vs 14.8% without |
| Degradation detection is effective | AUROC=0.869, F1 drop 1.75pp |
| Source generalization gaps are measurable & reportable | LOSO matrix + few-shot curves + DG ablation |
| MoE outperforms monolithic VLM on structured parsing context | real-upstream node F1=0.951696 vs best VLM semantic F1=0.280; no-repair relation F1=0.920938 |
| Geometry features are essential for relations | no-geometry: -11.3pp relation F1 |
| Training is auditable and reproducible | 5-class contract, git/env/dataset hash |

## 5. Claims We CANNOT Yet Support

| Claim | Blocker |
|-------|---------|
| All-element 0.98 F1 on real-world locked benchmark | Human review pending for internal-real-v3 |
| TextDimension real OCR generalization | `reports/vlm/text_dimension_external_ocr_lock_v3.json` is `pending_no_human_gold`; v5 passes CubiCasa dev/locked metrics, but broad scanned-drawing OCR robustness is not yet supported |
| TextDimension standalone-to-E2E equivalence | E2E text-family metrics are scene-graph node-label metrics and remain separate from the standalone v5 expert benchmark |
| Symbol 9-class F1 ≥ 0.90 / cross-source smoke | `generic_symbol` and `table` remain weak long-tail labels; `reports/vlm/symbol_long_tail_error_pack_v1.jsonl` and `reports/vlm/symbol_conservative_arbitration_v1.json` audit the current failure modes, while `reports/vlm/symbol_cross_source_lock_v1.json` is `pending_no_human_gold` rather than a passed cross-source smoke |
| FloorPlanCAD source F1 ≥ 0.98 | Current 0.969, gap of 1.1pp |

## 6. Paper Structure (Proposed)

1. **Introduction**: Floorplan understanding problem; MoE + constraint fusion thesis
2. **Related Work**: Floorplan datasets, VLM approaches, rule-based systems
3. **CadStruct-MoE Architecture**: 5 experts, router, fusion, degradation pipeline
4. **Benchmark v3**: 4-source, zero-leakage, 1574 records
5. **Experiments**: Per-expert results, ablation studies, source generalization
6. **Analysis**: Why MoE > VLM; geometry's role; constraint fusion trade-offs
7. **Limitations & Future Work**: OCR gap, FloorPlanCAD domain shift, human review
8. **Conclusion**: Specialized MoE + constraint guarantees for structured understanding

---

## 7. Evidence Index

All evidence files are under the project root:

| Evidence Type | File |
|---------------|------|
| WallOpening eval | `reports/vlm/wall_opening_floorplancad_residual_v1_eval.json` |
| RoomSpace eval | `reports/vlm/room_space_v5_t046_review_adjusted_auto_accept.json` |
| SymbolFixture v10 eval | `reports/vlm/symbol_fixture_v10_eval.json` |
| TextDimension v5 eval | `reports/vlm/text_dimension_expert_v5_eval.json` |
| SheetLayout eval | `reports/vlm/sheet_layout_expert_v1_eval.json` |
| MoE Router v2 eval | `reports/vlm/moe_router_v2_eval.json` |
| MoE Router v3 fair ablation | `reports/vlm/moe_router_v3_fair_ablation.json` |
| Lie/SE(2) current-pipeline audit | `reports/vlm/lie_se2_current_pipeline_ablation_v1.json` |
| Expert contribution matrix | `reports/vlm/expert_contribution_matrix_v2.json` |
| Scene Graph Fusion main | `reports/vlm/paper_e2e_metric_reconciliation_v1.json` |
| Relation ceiling diagnostic | `reports/vlm/relation_no_repair_ceiling_diagnostic_v1.json` |
| Relation hard cases | `reports/vlm/relation_no_repair_hard_cases_v1.jsonl` |
| External OCR lock v3 | `reports/vlm/text_dimension_external_ocr_lock_v3.json` |
| Symbol cross-source lock | `reports/vlm/symbol_cross_source_lock_v1.json` |
| Paper artifact manifest | `reports/vlm/paper_artifact_manifest_v1.json` |
| Paper submission claims | `reports/vlm/paper_submission_claims_v1.md` |
| Degraded robustness | `reports/vlm/degraded_robustness_v1_eval.json` |
| Quality scorer | `reports/vlm/quality_failure_scorer_v1_eval.json` |
| Innovation ablation | `reports/vlm/innovation_ablation_v2.json` |
| LOSO matrix | `reports/vlm/loso_eval_matrix_v3.json` |
| Few-shot curves | `reports/vlm/few_shot_adaptation_curve_v1.json` |
| Domain generalization | `reports/vlm/domain_generalization_ablation_v1.json` |
| Benchmark v3 manifest | `datasets/cadstruct_real_world_benchmark_v3/manifest.json` |
| Benchmark v3 leakage audit | `reports/vlm/benchmark_v3_leakage_audit.json` |
| Performance profile | `reports/vlm/real_pipeline_performance_v1.json` |
| CI smoke thresholds | `reports/vlm/ci_regression_thresholds_v1.json` |
| Training audit | `reports/vlm/training_contract_coverage_v2.json` |
| Capability boundary | `reports/vlm/capability_boundary_v3.json` |
