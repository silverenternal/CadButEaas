# Real-World Capability Boundary v5 (Updated 2026-05-04)

## Supported ✅

- **E2E Scene Graph Relation/Validity**: real-upstream + symbol/text conservative arbitration + generic override + RF long-tail symbol model + no-repair relation scorer reaches node macro F1=0.951696, node accuracy=0.981566, relation F1=0.920938, relation precision=0.961937, invalid=0.0 from `reports/vlm/paper_metric_table_manifest_v3.json`. Boundary/symbol/text arbitration is label-level post-router arbitration, not a learned family router.
- **WallOpening**: Locked accuracy=0.993, macro F1=0.989, R²=0.980 — production-grade
- **WallOpening (CVC-FP source)**: macro F1=0.989 ✅
- **RoomSpace**: Locked macro F1=0.982 (strict), 0.989 (review-adjusted), proposal recall@IoU0.5=1.0
- **SymbolFixture (9 classes)**: v10/v9 ExtraTrees with 13D geometry-context features, dev macro F1=0.872. This passes the main-table minimum boundary but not the preferred 0.90 target.
- **TextDimension — OCR**: Exact rate=1.0 on non-empty samples ✅
- **TextDimension — room_label**: F1=0.995 ✅
- **TextDimension — leader_line**: F1=1.000 ✅
- **MoE Router**: DeterministicRouter is the main router (effective_rate=1.0, wrong_expert_rate=0.0 on 134K candidates). Fair learned router v3 is an ablation/future-work component because wrong_expert_rate=0.152 on real dev candidates.
- **Constraint Fusion v2**: No-repair relation scoring with schema validity, invalid rate=0.0 (vs 14.8% without)
- **Lie/SE(2)**: Bounded core geometry module. `reports/vlm/lie_se2_core_claim_decision_v6.json` supports zero-ablation reliance and crop rotation/flip stress stability, but blocks any claim of sole/dominant performance source or proven multi-seed matched accuracy lead.
- **Degraded Robustness**: Router accuracy=0.845, quality scorer AUROC=0.869, F1 drop=1.75pp
- **Source Generalization**: LOSO matrix (4 sources), few-shot curves (3×4×5), DG ablation (4 strategies, all 21 leakage checks pass)
- **Innovation Ablation**: 7 controls, all 50 checks pass; VLM-as-main worst (-25.2pp node F1)
- **Benchmark v3**: 1,574 records, 4 sources, zero leakage, dataset_hash=34e20b4b
- **Performance**: P50=9.0ms, P95=31.6ms, peak memory=103.7 MiB, CI smoke all pass
- **Training Audit**: 5-class contract (WallOpening/Room/Symbol/Text/Router), git/env/dataset hash
- **Uncertainty/Abstain**: Abstain precision=1.0, high-confidence error reduction=1.0

## Partially Supported ⚠️

- **E2E Scene Graph Nodes**: real-upstream current paper-main node macro F1=0.951696 clears the preferred publication boundary, but standalone expert metrics and E2E node-label metrics remain separate contracts.
- **E2E Scene Graph Relations**: no-repair relation scorer F1=0.920938 clears the preferred 0.90 target, with strict record-bootstrap F1 95% CI [0.913169, 0.928493]. Repair-enabled F1=0.923 is appendix-only because gold-ID repair uses gold relation labels; locked-only threshold sweeps are diagnostic upper bounds and are not main-table admissible.
- **WallOpening (FloorPlanCAD source)**: macro F1=0.969 (target 0.98, gap 1.1pp)
- **SymbolFixture (9 classes, preferred target)**: macro F1=0.872; `generic_symbol` and `table` remain weak long-tail labels, so do not claim ≥0.90 yet.
- **Symbol arbitration cross-source generalization**: `reports/vlm/symbol_label_arbitration_generalization_v1.json` passes leakage and feature-ablation checks on CubiCasa locked data, but `reports/vlm/symbol_cross_source_lock_v1.json` is `pending_no_human_gold`; no non-CubiCasa symbol locked split with compatible 9-class gold labels is locally available.
- **TextDimension (external real OCR)**: `reports/vlm/text_dimension_external_ocr_lock_v3.json` is `pending_no_human_gold`; the 50 FloorPlanCAD raster candidates still need human transcript/bbox annotation. v5 passes CubiCasa dev/locked targets, but no source-held-out scanned/photo drawing split with human gold transcripts is locally available.
- **TextDimension E2E alignment**: runtime uses the v5-calibrated note gate, but scene-graph E2E text-family metrics are not identical to the standalone v5 benchmark.
- **SheetLayout**: Non-core extension / future work. `reports/vlm/sheet_layout_real_gold_boundary_v1.json` found no real layout gold; current rule heuristic is retained only as a placeholder and is excluded from main results.

## Not Supported As Final Claim ❌

- **All-source all-element 0.98 F1**: Blocked by human review + Symbol 9-class long tail + FloorPlanCAD gaps + external real OCR validation
- **Reviewed internal-real-v3 locked benchmark**: 100 records need human double-annotation (review pack ready)
- **TextDimension broad real OCR robustness**: Not supported. Needs external scanned-drawing OCR lock test beyond CubiCasa SVG/OCR-enhanced splits.
- **Symbol 9-class macro F1 ≥ 0.90**: Needs more long-tail samples and/or real raster crop pixels for CNN/ViT.
- **FloorPlanCAD WallOpening F1 ≥ 0.98**: Current 0.969, needs stronger residual branch or domain adaptation

## Blocking Items

| Blocker | Action Required | Impact |
|---------|----------------|--------|
| Human review | 100 internal-real-v3 records at `reports/vlm/internal_real_v3_review_pack_v2/` | Unlocks locked benchmark |
| External OCR lock test | Add 50-100 scanned/photo drawing OCR labels beyond CubiCasa SVG/OCR-enhanced splits | Required before any broad real OCR robustness claim |
| Symbol cross-source lock test | Fill `gold_9class_symbol_type` in the FloorPlanCAD symbol pack | Required before any cross-source symbol generalization claim |
| Symbol CNN/ViT | Train on real crop pixels (not raster stats) | Reaches 0.90 on 9 classes |
| FloorPlanCAD domain | Stronger residual branch or adversarial fine-tuning | Closes 1.1pp gap |

## Key Metric Improvements (Before → After)

| Metric | Before | After | Target |
|--------|--------|-------|--------|
| Scene Graph Node Macro F1 | 0.182 broken ID scope | **0.944 real-upstream + symbol/text conservative arbitration + generic override** | ≥0.50 / ≥0.70 ✅ |
| Scene Graph Relation F1 | 0.151 legacy | **0.920 no-repair real-upstream scorer** | ≥0.85 ✅ / ≥0.90 ✅ |
| Scene Graph Invalid Rate | 0.0 | **0.000 real-upstream** | ≤0.02 ✅ |
| WallOpening Macro F1 | 0.9885 | **0.989** | ≥0.98 ✅ |
| RoomSpace Macro F1 | 0.9821 | **0.989** (review-adjusted) | ≥0.98 ✅ |
| Symbol 9-class F1 | 0.717 old | **0.872** | ≥0.80 ✅ / ≥0.90 ⚠️ |
| TextDimension Standalone Macro F1 | 0.61 | **0.984** | ≥0.95 ✅ |
| TextDimension Relation F1 | 0.87 | **0.998** | ≥0.95 ✅ |
| TextDimension OCR Exact | 0.0 | **1.000** | ≥0.90 ✅ |
| MoE Router Effective Rate | — | **1.000** | ≥0.98 ✅ |
| Degraded F1 Drop | — | **1.75pp** | ≤5pp ✅ |

## Completed Outputs Summary

| Phase | Scripts | Reports | Checkpoints | Status |
|-------|---------|---------|-------------|--------|
| R0. Benchmark v3 | 2 | 3 | 0 | ✅ automated done; human review pending |
| R1. E2E Pipeline | 3 | 6 | 0 | ✅ done |
| R2. Degraded Robustness | 3 | 4 | 1 | ✅ done |
| R3. SymbolFixture | 4 | 5 | 2 | ✅ done (v10/v9: 9-class F1=0.872, preferred 0.90 still open) |
| R4. OCR/Text/Sheet | 5 | 6 | 1 | ✅ done (TextDimension v5 macro F1=0.984; external real OCR robustness still pending) |
| R5. Source Generalization | 3 | 3 | 0 | ✅ done |
| R6. Router/Fusion | 3 | 5 | 1 | ✅ done (fusion v2: node=1.0, rel=0.918) |
| R7. Engineering | 3 | 4 | 0 | ✅ done |
| R8. Paper/Ablation | 3 | 6 | 0 | ✅ done (7 controls, 50/50 checks) |
| R9. Submission lock pack | 5 | 8 | 0 | ✅ done (`paper_artifact_manifest_v1.json` passed; OCR/symbol external locks pending human gold) |

**Total**: ~29 scripts, ~42 reports, ~6 checkpoints across 9 phases. All automated work complete.
