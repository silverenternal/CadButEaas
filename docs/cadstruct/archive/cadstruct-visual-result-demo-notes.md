# CadStruct-MoE Visual Result Demo Notes

This note records the claim boundary for the latest CubiCasa5K visual result demo.

- Config: `configs/vlm/cubicasa_visual_demo_model_v13_real_e2e.json`
- Dataset: this demo uses CubiCasa5K/CubiCasa floorplan samples only.
- Pipeline: visuals are generated from `scene_graph.nodes` and `scene_graph.edges` after model_v13 real RoomSpace rerun, relation gating, and v14 SVG/parser recovery/refiner stages.
- Source mode: RoomSpace v13 is rerun from its checkpoint. Boundary v14, room proposal recovery v14, and symbol visual gate v14 use CubiCasa SVG/parser candidate/raw-label recovery over saved visual candidates; they are not pure raster detectors. Text metrics use a text-aware gold adapter built from CubiCasa SVG text_candidates.
- Interpretation: colored overlays show recognized scene-graph elements. Skipped/missing bbox counts and source-mode notes are limitations, not hidden successes.
- Evaluation boundary: latest strict text-aware visual report: `reports/vlm/model_v13_real_visual_e2e_text_aware_boundary_room_symbol_v14_eval.json`; original reviewed-gold report without text adapter remains separate.

Generated samples: 5

## Model Source And Boundary Audit

- Finding: The latest boundary gain comes from suppressing saved-model boundary semantic pollution by reverting to reviewed-compatible SVG/parser raw labels, not from a new raster wall detector.
- Evidence: Boundary F1 improved from 0.359307 to 0.965368 on the 5-sample reviewed visual set after v14 raw-label selection.
- Parser fix: Room and symbol recall were improved by recovering missing CubiCasa SVG/parser candidates and then applying specialist/classification policy where available.
- Renderer fix: The renderer uses annotation-aligned input backgrounds, source canvas checks, suspicious bbox skip rules, side-by-side views, overlay views, and coverage audits.
- Residual interpretation: Remaining misses are primarily text candidate recall and contains relation misses; current outputs are best interpreted as CubiCasa SVG/parser candidate recognition and refinement, not external raster generalization.
