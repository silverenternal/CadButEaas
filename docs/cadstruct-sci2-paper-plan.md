# CadStruct-MoE SCI 2 Paper Plan

## Positioning

CadStruct-MoE should be positioned as a structure-aware mixture-of-experts system for floor-plan recognition, not as a generic VLM fine-tuning paper. The core claim is that real drawings contain heterogeneous element families with different evidence, metrics, and annotation density, so decomposed experts plus auditable graph fusion are a better fit than one monolithic head.

Current evidence supports a scoped paper if the claims are conservative. Wall/opening recognition is strong on the selected mixed-source locked protocol, but pure cross-source robustness and end-to-end scene graph recognition are not solved yet.

## Main Contributions

1. Structure-aware MoE decomposition for floor-plan elements: WallOpening, RoomSpace, SymbolFixture, TextDimension, SheetLayout, and SceneGraphFusion are trained/evaluated as separate modules with shared schemas.
2. Geometry and topology aware recognition: SE(2)-canonical features, primitive graph conditioning, raster crop evidence, and relation-aware audits target the structured nature of drawings.
3. Auditable domain-generalization protocol: source-mixed, leave-one-source-out, few-shot target adaptation, target-only upper bound, and hard-case before/after are reported separately.
4. Constraint-aware scene graph export: expert outputs are fused into typed nodes and relations with invalid graph rate and warning traces.
5. Engineering reproducibility layer: training runs, peak memory, configs, hashes, and OOM/nonfinite audit fields are indexed by `scripts/vlm/training_audit.py`.

## Current Evidence Boundary

WallOpening is the strongest result. The selected mixed-source locked test reaches accuracy `0.992637`, macro F1 `0.988548`, and probability R2 `0.980085`. This can support a structural recognition claim.

Cross-source zero-shot robustness is not paper-ready. The existing leave-one-source-out table drops sharply, and few-shot/target-domain adaptation remains the credible path. Claims must say "few-shot target adaptation improves domain transfer" rather than "domain-invariant recognition is solved."

RoomSpace reaches macro F1 above `0.98` only under gold room polygons. End-to-end room proposal, mask/polygon quality, and type classification need a separate table before claiming real-world room recognition.

SymbolFixture and TextDimension remain below paper-grade targets. They should be presented as MoE extension modules unless new locked-domain results lift their per-class metrics.

SceneGraphFusion is an engineering scaffold with smoke evidence. It needs node F1, relation F1, and invalid graph rate before becoming a central quantitative claim.

## Required Experiments

| Table | Purpose | Required Metrics | Current Asset |
| --- | --- | --- | --- |
| T1 | Mixed-source structural recognition | acc, macro F1, R2, per-source F1 | `reports/vlm/paper_v2_two_stage_router_summary.json` |
| T2 | Leave-one-source-out robustness | acc, macro F1, R2, confusion | `reports/vlm/generalization_benchmark_v1.json` |
| T3 | Few-shot target adaptation | zero-shot vs few-shot vs target-only | `reports/vlm/paper_v2_generalization_followup_summary.json` |
| T4 | Room proposal to room type | recall@IoU, AP50, mIoU, type F1 | pending |
| T5 | Symbol and text experts | per-class F1, dimension relation F1 | current baselines below target |
| T6 | Scene graph fusion | node F1, relation F1, invalid graph rate | pending |
| T7 | Ablations | no MoE, no graph, no crop, no text, no constraints | `reports/vlm/ablation_matrix_v1.json` |
| T8 | Efficiency | peak memory, latency, compression ablations | `reports/vlm/memory_speed_budget_v1.json` |

## Claim Discipline

Use strict metrics as the main result. Ambiguity-adjusted, diagnostic final-epoch, oracle grouping, and target-only upper-bound numbers must stay in separate rows.

Do not claim 98% generalization until leave-one-source-out or held-out internal real drawings reach that level. The current result supports a strong in-protocol structural expert, not universal drawing understanding.

Do not frame the 14B VLM as the core model. It is a teacher, OCR/layout helper, or fallback. The novel model component is the structured expert system and its graph-aware features/fusion.

## Submission Gate

Before targeting SCI 2, the paper package should contain:

- at least three locked source families in the benchmark registry;
- a real or internal hard-case table with human-reviewed labels;
- end-to-end room proposal/type metrics;
- symbol/text per-class metrics with long-tail disclosure;
- scene graph relation metrics;
- complete ablation matrix with positive and negative controls;
- reproducible training index and memory/speed budget.

## Next Work Items

1. Human-review `datasets/internal_hard_cases_round_1` and only then use it as training data.
2. Add end-to-end RoomProposal metrics on the reviewed CubiCasa locked split.
3. Run the ablation matrix in `reports/vlm/ablation_matrix_v1.json`.
4. Extend the benchmark with a full internal real-drawing locked split.
5. Generate paper tables directly from audited JSON reports, not hand-copied numbers.

## V2 Claim-To-Evidence Matrix

This section is the current paper-facing contract. Claims not listed here should not be used in the abstract, conclusion, README, or rebuttal material.

| Claim | Dataset/Evidence | Metric | Baseline | Required Ablation | Claim Status |
| --- | --- | --- | --- | --- | --- |
| Structure-aware MoE is a better fit than a monolithic raster/VLM head for heterogeneous floor-plan elements. | `reports/vlm/paper_v2_two_stage_router_summary.json`, `reports/vlm/source_heldout_eval_batch_v1.json` | accuracy, macro F1, R2, per-source F1 | single shared wall/opening classifier; zero-shot VLM report | `no_moe_shared_head`, `router_oracle_vs_learned` | Supported for WallOpening only; not yet all elements. |
| Geometry-aware recognition improves structured element discrimination. | WallOpening locked runs plus room proposal/gold room evaluations | macro F1, R2, confusion by class, proposal recall | raster-only crop classifier | `no_geometry_features`, `no_se2_canonicalization` | Partially supported; needs full ablation run for final paper table. |
| Auditable routing and constraint fusion reduce silent graph failures. | `reports/vlm/scene_graph_f1_eval_v1.json`, `reports/vlm/scene_graph_error_attribution_v1.json` | node F1, relation F1, invalid graph rate, residual attribution counts | unconstrained JSON merge | `no_constraint_fusion`, `no_router_trace` | Supported on smoke/expected-json scene graph, not yet real end-to-end scenes. |
| Hard-case active learning closes long-tail gaps without contaminating locked tests. | `datasets/internal_hard_cases_round_2/manifest.json`, `reports/vlm/hard_case_mining_round_2.jsonl` | before/after F1, long-tail per-class F1, locked-test no-drop | pre-hardcase checkpoint | `no_hardcase_loop`, `unreviewed_hardcase_excluded` | Pipeline supported; metric uplift still pending for Symbol/Text. |
| Memory-aware expert training makes the system reproducible on bounded GPUs. | `reports/vlm/memory_budget_check_v2.json`, `reports/vlm/expert_compression_ablation_v1.json`, `reports/vlm/crop_inference_tile_audit_v1.json` | peak memory estimate, OOM-risk class, latency delta, F1 delta | unbounded batch/all-pairs inference | `no_memory_guard`, `compression_enabled_vs_disabled` | Supported as engineering contribution; compression is not adopted until measured. |

## SCI2 Submission Stance

The current package is close to a credible SCI2 systems paper only if the title and claims are scoped to structure-aware MoE floor-plan recognition with auditable fusion. It is not ready for a broad claim such as "general real-world drawing understanding" or "all element classes above 98%".

Paper-ready quantitative claims today:

- WallOpening in the selected locked mixed-source protocol exceeds the 98% macro-F1 target and has accuracy above 99%.
- Scene-graph fusion can reach high relation F1 on the expected-json smoke protocol with invalid graph rate equal to zero.
- The benchmark registry, source-heldout runner, hard-case mining, and memory audit are reproducible scaffolding for a publishable experimental protocol.

Claims that remain blocked:

- SymbolFixture and TextDimension are below paper-grade strict macro F1.
- Source-heldout generalization is incomplete for elements that only have one locked source.
- Zero-shot VLM comparison is still below the required sample floor.
- Real-world end-to-end drawing recognition still lacks broad coverage of complex MEP symbols, non-standard legends, poor scans, and multilingual annotations.

The recommended submission framing is therefore: "A structure-aware, auditable MoE framework for floor-plan recognition with strong wall/opening results and extensible expert routing", not "a finished universal CAD recognition model".
