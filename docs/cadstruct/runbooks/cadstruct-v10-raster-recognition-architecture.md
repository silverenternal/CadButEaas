# CadStruct v10 Raster Recognition Architecture

v10 is a separate raster recovery branch. It does not replace the existing v7/v8 MoE expert pipeline.

## Boundary

- v7/v8: SVG/parser candidates plus typed experts, refiners, fusion/router, and visual evidence.
- v9: rejected pure raster attempt.
- v10: 512px CubiCasa raster supervision, residual U-Net segmentation, MuraNet-style heat/detection audit, graph/polygon/text postprocess, and source integrity gates.

SVG is used only for offline labels and locked gold. Adopted v10 inference must use `source_mode=model_v10_raster` and `svg_candidate_ids_used=false`.

## Locked Decision

- adopted: `False`
- report: `reports/vlm/model_v10_raster_locked_eval.json`
- claim gate: `reports/vlm/model_v10_paper_claim_gate.json`

## Research Basis

CubiCasa5K, MuraNet, Raster-to-Graph, PolyRoom, Raster2Seq, recent OCR/vectorization floor-plan work, and FloorSAM motivate the branch. The current implementation keeps graph, sequence, and foundation-model outputs exploratory unless locked gates pass.
