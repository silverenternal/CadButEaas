# CadStruct MoE Dataset Roadmap

Date: 2026-04-30

## Current Scope

The current strongest CadStruct path is a primitive graph model for `hard_wall`, `door`, and `window`.
This is not yet a full drawing-understanding model. It is best described as a wall/opening expert.

For real architectural drawings, the missing element families are:

- room and space regions, including room type labels and adjacency;
- fixtures, furniture, stairs, columns, balconies, and other architectural symbols;
- text, dimensions, legends, and notes;
- title blocks, schedules, tables, and sheet-level metadata;
- noisy scan/PDF style variation across offices and sources.

This makes a modular MoE-style architecture appropriate. The goal is to add new element families without
making the existing wall/opening expert more complex or less auditable.

Detailed architecture, training, fusion, metrics, and ablation planning is tracked in
`docs/cadstruct-moe-architecture-plan.md`.

## Dataset Findings

| Dataset | Use | Strength | Limitation | Priority |
|---|---|---|---|---|
| FloorPlanCAD | existing wall/opening CAD target domain | already local; useful for line/opening morphology | narrow current label surface | keep as core |
| CVC-FP | existing wall/opening robustness domain | already local; useful for cross-source checks | annotation format needs careful conversion | keep as core |
| CubiCasa5K | rooms, walls, openings, object/icon classes | 5,000 samples; polygon/SVG annotations; 80+ floorplan object categories | official package is large; local download is still partial | P0 |
| DeepFloorplan / R2V / R3D annotations | room-boundary and room-type segmentation | explicitly models hierarchy between room-boundary and room-type elements | old code stack; masks rather than rich graph labels | P1 |
| RPLAN | room layout/topology pretraining | large-scale residential plan layout corpus, useful for room graph priors | not primarily symbol/text detection | P1 |
| ResPlan | vector graph and room connectivity | 17,000 structurally rich vector-graph residential plans; includes walls, openings, balconies, rooms | new 2025 dataset; availability/license must be verified before depending on it | P1 research |
| SESYD / GREC symbol spotting | symbol expert pretraining/evaluation | public synthetic symbol spotting benchmark with degradation/noise variants | synthetic and small; limited symbol notation diversity | P2 |
| DocLayNet / PubLayNet | title block/table/text-layout transfer | large document-layout datasets with COCO-style boxes | not floorplan-specific; useful only for document-layout pretraining | P2 |

External references:

- CubiCasa5K official repository describes 5,000 floorplan samples with dense polygon annotations over 80+ categories: https://github.com/cubicasa/cubicasa5k
- DeepFloorplan provides wall, door/window, and room-type mask annotations for multi-task floorplan recognition: https://github.com/zlzeng/DeepFloorplan
- ResPlan is a 2025 vector-graph dataset with 17,000 residential floorplans, room connectivity, and architectural elements: https://arxiv.org/abs/2508.14006
- SESYD is used as a public synthetic architectural symbol spotting benchmark with noise/degradation variants in CVPRW 2020 symbol-spotting work: https://openaccess.thecvf.com/content_CVPRW_2020/papers/w34/Rezvanifar_Symbol_Spotting_on_Digital_Architectural_Floor_Plans_Using_a_Deep_CVPRW_2020_paper.pdf
- DocLayNet has 80,863 manually annotated pages with 11 document-layout classes: https://research.ibm.com/publications/doclaynet-a-large-human-annotated-dataset-for-document-layout-segmentation
- PubLayNet has over 360,000 document images for scientific article layout analysis: https://research.ibm.com/publications/publaynet-largest-dataset-ever-for-document-layout-analysis

## Proposed Expert Split

### WallOpeningExpert

Owns the current `hard_wall`, `door`, and `window` primitive graph task.

Implementation policy:

- freeze the selected paper-v2 path as a stable baseline;
- keep the FloorPlanCAD target-domain residual path separate;
- expose calibrated probabilities, confidence, margin, and graph-neighborhood features to later fusion.

This expert should not absorb room, text, or symbol labels.

### RoomSpaceExpert

Owns room polygons, room type labels, and room adjacency.

Primary data:

- CubiCasa5K SVG polygons;
- DeepFloorplan room masks;
- RPLAN or ResPlan for room topology priors if license and access are usable.

Outputs:

- room polygon or mask;
- room type;
- adjacency graph;
- boundary support links to wall/opening primitives.

Metrics:

- room IoU;
- room-type macro F1;
- adjacency precision/recall;
- boundary consistency against WallOpeningExpert.

### SymbolFixtureExpert

Owns fixtures, furniture, stairs, columns, appliances, sanitary symbols, and domain-specific repeated symbols.

Primary data:

- CubiCasa5K object/icon categories;
- SESYD/GREC for symbol-spotting stress tests;
- internal pseudo-label or manual annotation for real project symbols.

Outputs:

- symbol class;
- bounding box or polygon;
- orientation;
- host relation, e.g. attached wall, inside room, connected opening.

Metrics:

- mAP/AP50;
- per-symbol macro F1;
- orientation accuracy;
- host-link accuracy.

### TextDimensionExpert

Owns OCR text, dimension strings, dimension lines, leaders, callouts, and legends.

Primary data:

- internal annotated floorplan dimensions are required;
- DocLayNet/PubLayNet can only pretrain page/table/title-block layout sensitivity;
- OCR engines can generate weak labels but should not be treated as ground truth.

Outputs:

- text boxes and recognized text;
- dimension line geometry;
- text-to-line linkage;
- normalized numeric dimension value when possible.

Metrics:

- OCR exact/normalized accuracy;
- dimension-line detection F1;
- text-line linkage F1;
- numeric dimension tolerance accuracy.

### SheetLayoutExpert

Owns title block, tables, schedules, legends, stamps, and sheet-level metadata.

Primary data:

- DocLayNet/PubLayNet for generic layout pretraining;
- internal CAD/PDF sheet annotations for target labels.

Outputs:

- title block/table/legend boxes;
- metadata key-value candidates;
- sheet crop routing regions.

Metrics:

- layout mAP;
- key-value extraction F1;
- region crop recall.

## Router And Fusion Design

Start with an auditable router, not a full sparse-transformer MoE.

Router inputs:

- primitive type and geometry;
- raster/crop statistics;
- OCR/layout candidates;
- local graph degree and relation type;
- source/domain metadata only when explicitly audited.

Routing policy:

- line/opening primitives route to WallOpeningExpert;
- closed or near-closed regions route to RoomSpaceExpert;
- compact repeated glyph-like crops route to SymbolFixtureExpert;
- text-like connected components and leader/dimension-line structures route to TextDimensionExpert;
- large sheet-margin/table regions route to SheetLayoutExpert.

Fusion constraints:

- doors and windows must attach to or interrupt walls;
- rooms should be bounded by walls/openings;
- room labels should lie inside or near their room polygon;
- dimensions should link to dimension lines or extension lines;
- fixtures/symbols should attach to a host room, wall, or opening when the class implies one.

The learned router should come after enough labels exist. Before that, deterministic candidate routing gives better auditability and less risk of training-set leakage.

## Engineering Plan

### Phase 0: Lock Existing Expert Boundary

- Treat the current wall/opening model as `WallOpeningExpert`.
- Export a stable inference schema with labels, probabilities, confidence, crop coordinates, and graph-neighborhood context.
- Keep current source-specific and residual branches behind explicit router rules.

### Phase 1: Dataset Registry And Ontology

Create a dataset registry that maps external labels to an internal ontology:

- `boundary`: wall, door, window, opening;
- `space`: room polygon, room type, adjacency;
- `symbol`: fixture, furniture, equipment, stair, column, appliance;
- `text`: room label, dimension, annotation, callout;
- `sheet`: title block, table, legend, schedule.

Expected outputs:

- `configs/vlm/dataset_registry.json`;
- `configs/vlm/cadstruct_ontology.json`;
- converter stubs for CubiCasa5K SVG, DeepFloorplan masks, and symbol/layout datasets.

### Phase 2: Complete P0 Data Ingestion

- Resume and unpack the official CubiCasa5K package.
- Convert SVG polygons into unified room/object/opening records.
- Build split manifests with leakage controls.
- Produce `cadstruct_rooms_v1`, `cadstruct_symbols_v1`, and `cadstruct_integrated_v1` manifests.

### Phase 3: MoE Scaffolding

Add a small framework layer:

- `scripts/vlm/cadstruct_moe/router.py`;
- `scripts/vlm/cadstruct_moe/experts/wall_opening.py`;
- `scripts/vlm/cadstruct_moe/experts/room_space.py`;
- `scripts/vlm/cadstruct_moe/experts/symbol_fixture.py`;
- `scripts/vlm/cadstruct_moe/experts/text_dimension.py`;
- `scripts/vlm/cadstruct_moe/fusion.py`;
- `scripts/vlm/cadstruct_moe/schema.py`.

The first version can call existing checkpoints and deterministic converters. It does not need end-to-end training yet.

### Phase 4: Train New Experts Independently

Training order:

1. RoomSpaceExpert on CubiCasa5K/DeepFloorplan.
2. SymbolFixtureExpert on CubiCasa5K icons plus SESYD stress tests.
3. TextDimensionExpert after internal dimension labels exist.
4. SheetLayoutExpert only when PDF/sheet-level data is collected.

Each expert needs an independent dev/locked split and a no-leakage report.

### Phase 5: Integrated Scene Graph Evaluation

Report both expert metrics and integrated drawing metrics:

- wall/opening macro F1, accuracy, R2;
- room IoU and room-type F1;
- symbol mAP and per-class F1;
- text/dimension extraction accuracy;
- relation graph F1;
- cross-dataset generalization by source.

The paper claim should move from "wall/opening primitive recognition" to "structured floorplan scene graph recognition" only after integrated metrics are stable.

## Immediate Next Actions

1. Resume CubiCasa5K official download and unpack it.
2. Implement the ontology and dataset registry.
3. Write a CubiCasa5K SVG inspection/conversion audit that reports class counts before training.
4. Build the MoE schema and deterministic router skeleton.
5. Add RoomSpaceExpert as the first new expert while keeping WallOpeningExpert unchanged.
6. Decide whether ResPlan is practically usable by checking its data availability and license.
7. Collect or annotate a small internal dimension/text set, because public floorplan data is weak for dimension understanding.

## Implementation Status

Completed on 2026-04-30:

- added `configs/vlm/cadstruct_ontology.json`;
- added `configs/vlm/dataset_registry.json`;
- added lightweight MoE schema/router/fusion scaffolding under `scripts/vlm/cadstruct_moe/`;
- added expert wrappers for wall/opening, room/space, symbol/fixture, text/dimension, and sheet layout;
- added `scripts/vlm/audit_cubicasa5k_svg.py` for pre-conversion SVG class audits;
- resumed the CubiCasa5K official download from 96 MiB to 392 MiB and confirmed the Zenodo link supports continuation.

Validation:

- JSON syntax validation passed for the ontology and dataset registry;
- Python compile validation passed for the MoE scaffold and CubiCasa audit script;
- direct no-pytest smoke validation passed for deterministic routing and fusion constraints;
- `pytest` is not installed in the current Python environment, so the lightweight tests were run by direct function invocation.

## Research Position

The SCI-oriented innovation should not be "we used MoE" by itself. A defensible contribution is:

- a structured floorplan MoE that routes by geometric/object family;
- expert-local training that avoids catastrophic forgetting of wall/opening recognition;
- graph-consistency fusion enforcing architectural constraints;
- cross-dataset, element-family-aware evaluation;
- auditable routing and ablation against monolithic multi-task baselines.

This is better aligned with the problem structure than a single enlarged classifier.
