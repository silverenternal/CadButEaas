# CadStruct v11 Frontier Architecture

v11 is a frontier recovery branch around the existing CadStruct-MoE system. It does not overwrite or retrain the good v7/v8 expert models.

## Boundary

- v7/v8: protected MoE baseline with parser/SVG candidates, typed experts, refiners, fusion/router, and visual evidence.
- v10: rejected raster branch with locked evidence.
- v11: target repair, official baseline audit, edge graph proposals, polygon sequence smoke, topology refiner smoke, small-object/text protocol, foundation-model audit, source-gated assembly, and advisor visual evidence.

SVG/parser geometry is used only as offline gold. Adopted v11 inference must declare `svg_candidate_ids_used=false`.

## Research Basis

- CubiCasa5K official repository: https://github.com/cubicasa/cubicasa5k
- MuraNet: https://arxiv.org/abs/2309.00348
- PolyRoom: https://arxiv.org/abs/2407.10439
- Raster2Seq: https://arxiv.org/abs/2602.09016
- CAGE: https://arxiv.org/abs/2509.15459
- FloorSAM: https://arxiv.org/abs/2509.15750
- ResPlan: https://arxiv.org/abs/2508.14006

## Locked Decision

Adopted: `True`. See `reports/vlm/model_v11_adoption_decisions.json`.
