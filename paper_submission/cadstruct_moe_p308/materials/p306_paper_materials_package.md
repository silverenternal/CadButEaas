# P306 Paper Materials Package

Created: 2026-05-27

Status: paper materials and end-to-end outputs packaged.

## Claim Boundary

The paper should present CadStruct-MoE as a reviewed SVG/contract normalized-candidate graph-recognition result.

Use P301 as the main quantitative line:

| Metric | Value | Boundary |
| --- | ---: | --- |
| Node macro-F1 | 0.962907 | reviewed SVG/contract normalized-candidate |
| Node accuracy | 0.982498 | reviewed SVG/contract normalized-candidate |
| Relation precision | 0.994900 | internal locked fine-threshold audit |
| Relation recall | 0.872665 | internal locked fine-threshold audit |
| Relation F1 | 0.929782 | internal locked fine-threshold audit |
| Invalid graph rate | 0.0 | reviewed SVG/contract graph audit |

Primary source: `reports/vlm/p304_p301_reviewed_metric_package.json`.

Do not describe this as external validation or runtime raster detector performance.

## External Validation Status

P305 blocks external/source-heldout generalization language.

Allowed wording:

- P301 is a reviewed SVG/contract experiment-line candidate under normalized-candidate evaluation.
- P301 relation F1 is an internal locked fine-threshold audit, not external validation.
- Relation scorer has supporting internal record-heldout robustness within the locked benchmark.
- WallOpening source-transfer evidence is weak and does not support a zero-shot source-transfer claim.

Forbidden wording:

- Do not call P301 externally validated.
- Do not claim source-heldout generalization for RoomSpace, SymbolFixture, or TextDimension.
- Do not use `relation_no_repair_heldout_scorer_v1` as external source validation.
- Do not claim runtime raster generalization from SVG/contract metrics.

Source-heldout batch result: `reports/vlm/p305b_source_heldout_eval_batch.json`.

| Expert | Status | Evidence |
| --- | --- | --- |
| WallOpening | evaluated but weak | cvc_fp to floorplancad macro-F1 0.485651; floorplancad to cvc_fp macro-F1 0.159075 |
| RoomSpace | blocked | only CubiCasa locked source; needs non-CubiCasa locked room labels |
| SymbolFixture | blocked | only CubiCasa locked source; needs FloorPlanCAD/internal symbol fixture labels |
| TextDimension | blocked | only CubiCasa locked source; needs FloorPlanCAD/internal OCR/text labels |

## End-To-End Outputs

### Main E2E Visualization

Use `reports/vlm/visual_demo_model_v13_real_e2e` for qualitative paper figures.

Key files:

- `reports/vlm/visual_demo_model_v13_real_e2e/paper_candidate_figure_v13_real_e2e.png`
- `reports/vlm/visual_demo_model_v13_real_e2e/paper_candidate_figure_v13_real_e2e.svg`
- `reports/vlm/visual_demo_model_v13_real_e2e/review_pack_v3/index.html`
- `reports/vlm/visual_demo_model_v13_real_e2e/sample_manifest_v1.json`

Samples in the manifest:

| Sample | Nodes | Edges | Rendered nodes | Recommended asset |
| --- | ---: | ---: | ---: | --- |
| cubicasa5k_00_11563 | 160 | 24 | 159 | `reports/vlm/visual_demo_model_v13_real_e2e/cubicasa5k_00_11563/side_by_side.png` |
| cubicasa5k_01_13277 | 204 | 30 | 196 | `reports/vlm/visual_demo_model_v13_real_e2e/cubicasa5k_01_13277/side_by_side.png` |
| cubicasa5k_02_13624 | 196 | 25 | 196 | `reports/vlm/visual_demo_model_v13_real_e2e/cubicasa5k_02_13624/side_by_side.png` |
| cubicasa5k_03_5039 | 243 | 34 | 242 | `reports/vlm/visual_demo_model_v13_real_e2e/cubicasa5k_03_5039/side_by_side.png` |
| cubicasa5k_04_9351 | 280 | 8 | 280 | `reports/vlm/visual_demo_model_v13_real_e2e/cubicasa5k_04_9351/side_by_side.png` |

Use this as qualitative E2E output: input, overlay, and side-by-side graph rendering. Do not describe it as source-heldout or pure raster.

### SVG-Derived E2E Smoke Path

Use `reports/vlm/cubicasa_svg_case/model_fused_scene_graph_locked_smoke_eval.json` as an E2E export/smoke metric, not as the headline model result.

| Metric | Value |
| --- | ---: |
| Cases | 64 |
| Node F1 | 0.723279 |
| Relation F1 | 1.0 |
| Boundary node F1 | 0.774974 |
| Space node F1 | 0.886297 |
| Symbol node F1 | 0.456018 |
| Text node F1 | 0.994737 |

Interpretation from the report: actual registered expert checkpoints were run on SVG-derived candidates; SVG still supplies candidate boxes and topology, while labels are predicted by loaded expert models.

Do not use `reports/vlm/cubicasa_svg_case/scene_graph_f1_locked_smoke_eval.json` as model recognition evidence. Its perfect node/relation F1 is only a contract/export sanity check.

### Image-Only E2E Negative Evidence

Use `reports/vlm/image_only_moe_e2e_v14_ablation_dashboard.json` as a limitation/failure-analysis artifact.

Important values:

- Source-integrity gate passed on 16 checked rows.
- Proposal mean F1 is 0.0.
- Final scene graph has 17 nodes and 0 edges.
- The artifact is explicitly not adopted.

This proves the image-only path can satisfy source-integrity constraints, but current proposal quality is not adoption-ready.

### Runtime Raster Secondary Bridge

Use `reports/vlm/p263_secondary_raster_adapter_package.json` only as bounded secondary runtime-raster evidence.

Best frozen secondary raster adapter:

| Metric | Value |
| --- | ---: |
| Overall F1 | 0.729861 |
| Equipment F1 | 0.729524 |

Claim boundary: secondary runtime raster adapter evidence only. The main paper claim remains SVG/contract CadStruct-MoE.

## Visualization Code Notes

The relevant visual code has three roles:

| Script | Role | Paper use |
| --- | --- | --- |
| `scripts/vlm/render_scene_graph_visual_demo.py` | SVG/HTML-first scene-graph demo renderer; emits overlays, side-by-side figures, manifests, coverage audits, and review packs | primary qualitative figure source |
| `scripts/vlm/render_visual_demo_image_only_moe_v17.py` | thin entrypoint into `image_only_moe_v17_pipeline.main("render")` | image-only visual appendix if needed |
| `scripts/vlm/build_visual_hard_case_pack_v18.py` | deterministic hard-case renderer for raster-only MoE debugging | error analysis or appendix only |

The main renderer is already designed for paper/review use: it writes `overlay_only`, `overlay_on_image`, `side_by_side`, per-sample summaries, a sample manifest, a coverage audit, and a review pack. It also includes warning/uncertain cases so the figure set is not just best-case recognition.

## Manuscript-Ready Paragraph

CadStruct-MoE is strongest as a reviewed SVG/contract normalized-candidate graph-recognition system. The packaged P301 line reaches node macro-F1 0.962907 and node accuracy 0.982498, with relation F1 0.929782 under an internal locked fine-threshold audit and invalid graph rate 0.0. Qualitative end-to-end outputs are available through the v13 real E2E visual demo and SVG-derived locked-smoke scene-graph exports. External/source-heldout generalization should not be claimed: P305 found only weak WallOpening transfer evidence and blocked RoomSpace, SymbolFixture, and TextDimension pending new external labels.

## Next Paper Work

1. Use P301/P304 for the main result table.
2. Use P305c for claim-boundary and limitations wording.
3. Use `visual_demo_model_v13_real_e2e` for qualitative E2E figures.
4. Use SVG-derived E2E smoke output only as an export-chain demonstration.
5. Keep image-only v14/v18 and P262/P263 in limitations or secondary runtime bridge sections.
