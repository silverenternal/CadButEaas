# CadStruct MoE Architecture Plan

Date: 2026-04-30

## Objective

Build a modular MoE architecture for raster drawing recognition.

The immediate target is not to replace the current wall/opening model. The current strongest model already works
well for `hard_wall`, `door`, and `window` on the paper-v2 locked test. The MoE architecture is for scaling from
wall/opening primitive labeling to structured drawing scene understanding while keeping each element family
auditable and independently trainable.

The target output is a structured scene graph:

- boundary primitives: walls, doors, windows, openings;
- space regions: rooms, room types, room adjacency;
- symbols and fixtures: stairs, columns, furniture, sanitary fixtures, equipment;
- text and dimensions: OCR text, dimension values, dimension lines, leaders, callouts;
- sheet-level layout: title blocks, tables, legends, schedules, stamps;
- cross-family relations: `bounds`, `attached_to`, `inside`, `labels`, `dimension_of`, `adjacent_to`.

## Design Principle

Use MoE as structural decomposition, not as a generic large sparse transformer.

For this project, a good MoE is:

- explicit about which expert owns which element family;
- able to freeze or replace one expert without retraining the others;
- auditable at every routing and fusion decision;
- evaluated both per-expert and as an integrated scene graph;
- robust to missing experts and partial annotations.

A bad MoE would be a monolithic model with a learned router that is hard to inspect and trained on incomplete labels.
That would make the existing model more complex without solving the real scaling problem.

## Current Baseline Boundary

Current strong component:

- task: primitive node classification;
- labels: `hard_wall`, `door`, `window`;
- core method: local raster crop encoder + SE(2)/geometry/topology features + primitive graph message passing;
- paper-v2 locked test: accuracy `0.992637`, macro F1 `0.988548`, probability R2 `0.980085`;
- target-domain FloorPlanCAD diagnostic best: accuracy `0.982709`, macro F1 `0.978327`, R2 `0.927768`.

Interpretation:

- the current model is a strong WallOpeningExpert;
- it is not yet a complete drawing-recognition model;
- it should be wrapped as an expert and protected from catastrophic forgetting.

## Architecture Overview

```text
raster drawing / PDF page
        |
        v
shared preprocessing
  - raster normalization
  - primitive extraction
  - connected components
  - OCR candidates
  - candidate regions
  - primitive graph
        |
        v
candidate generator
  - line/opening primitive candidates
  - room/closed-region candidates
  - symbol/icon candidates
  - text/dimension candidates
  - sheet-layout candidates
        |
        v
auditable router
        |
        +--> WallOpeningExpert
        +--> RoomSpaceExpert
        +--> SymbolFixtureExpert
        +--> TextDimensionExpert
        +--> SheetLayoutExpert
        |
        v
constraint-aware fusion
        |
        v
structured scene graph + warnings + per-decision audit trace
```

## Shared Representation

The shared representation should be lightweight and stable.

Inputs:

- image metadata: width, height, DPI when available, page crop origin;
- primitive graph: nodes, bbox, polyline geometry, edge relations;
- raster crops: multi-scale local crops around primitives/regions;
- connected components: text-like and symbol-like blobs;
- OCR candidates: raw text, bbox, confidence, source;
- layout regions: title block/table/legend candidates;
- source metadata: dataset/source id, only used when explicitly audited.

Shared feature blocks:

- normalized bbox and centroid;
- length, angle, aspect ratio, area ratio;
- SE(2)-canonicalized geometry;
- graph degree and typed relation counts;
- local raster statistics;
- optional learned crop embedding;
- optional document-layout crop embedding.

Rule:

Shared features can be reused by all experts, but expert training losses and checkpoint selection remain separate.

## Expert Definitions

### WallOpeningExpert

Purpose:

- classify structural boundary primitives as wall/door/window/opening;
- preserve existing high F1 on the current structural task;
- provide boundary support for rooms, symbols, and dimensions.

Inputs:

- primitive graph nodes;
- edge relations;
- multi-scale raster crops;
- existing paper-v2 source/raster features;
- optional target-domain residual branch.

Outputs:

- `hard_wall`, `partition_wall`, `door`, `window`, `opening`;
- calibrated probabilities;
- confidence, margin, entropy;
- relation candidates such as `interrupted_by` and `attached_to`.

Training policy:

- freeze the current selected model as baseline;
- train new variants only under a separate checkpoint namespace;
- route target-domain residuals explicitly and audit them by source;
- do not add room/text/symbol labels to this expert.

Metrics:

- accuracy;
- macro F1;
- per-class F1;
- probability R2;
- wall/opening relation consistency;
- cross-source split by FloorPlanCAD/CVC-FP/CubiCasa5K.

### RoomSpaceExpert

Purpose:

- detect room/space regions;
- classify room types;
- infer room adjacency and boundary support.

Inputs:

- closed or near-closed region candidates;
- wall/opening predictions;
- raster crops at region scale;
- room-label text candidates;
- topology from boundary primitives.

Outputs:

- room polygon or mask;
- room type;
- room adjacency edges;
- `bounded_by` and `contains` relations.

Training data:

- CubiCasa5K SVG polygons;
- DeepFloorplan masks;
- RPLAN/ResPlan if access and license are usable;
- internal target-domain room annotations later.

Recommended model:

- region proposal from deterministic geometry first;
- small mask/polygon head for room segmentation;
- graph head for room adjacency;
- text-assisted room type refinement after TextDimensionExpert is available.

Metrics:

- room IoU;
- room boundary F1;
- room-type macro F1;
- adjacency precision/recall/F1;
- boundary support consistency.

### SymbolFixtureExpert

Purpose:

- detect and classify repeated architectural symbols and fixtures.

Inputs:

- connected components and compact glyph-like crops;
- region context from RoomSpaceExpert;
- host boundary candidates from WallOpeningExpert;
- orientation-normalized crops.

Outputs:

- symbol class;
- bbox or polygon;
- orientation;
- host relation: inside room, attached to wall, connected to opening.

Training data:

- CubiCasa5K object/icon classes;
- SESYD/GREC synthetic symbol spotting for robustness checks;
- internal symbol labels for project-specific notation.

Recommended model:

- detector-style head for symbols;
- orientation classifier;
- host-relation classifier;
- hard-negative mining from text blobs and dense hatch/texture areas.

Metrics:

- mAP/AP50;
- per-class macro F1;
- orientation accuracy;
- host-link accuracy;
- false-positive rate on text/title-block regions.

### TextDimensionExpert

Purpose:

- extract text, dimensions, dimension lines, leaders, callouts, and notes.

Inputs:

- OCR candidates;
- text-like connected components;
- line candidates around text;
- local geometry for extension/dimension lines;
- sheet-layout regions.

Outputs:

- text bbox and normalized text;
- dimension text class;
- dimension line and extension line geometry;
- text-to-line links;
- normalized numeric value and unit when possible.

Training data:

- internal annotated dimension/text set is required for credible claims;
- OCR weak labels can bootstrap candidates;
- DocLayNet/PubLayNet can only support layout pretraining, not final dimension claims.

Recommended model:

- use OCR as proposal/teacher, not ground truth;
- add a relation head for `dimension_text -> dimension_line`;
- normalize numeric text with rule-based postprocessing;
- separate room labels from dimensions and notes.

Metrics:

- OCR exact accuracy;
- normalized numeric accuracy;
- dimension-line detection F1;
- text-line linkage F1;
- dimension tolerance accuracy;
- false merge/split rate.

### SheetLayoutExpert

Purpose:

- identify title blocks, tables, schedules, legends, stamps, and metadata regions.

Inputs:

- page-level raster;
- large layout connected components;
- OCR blocks;
- table/grid line candidates.

Outputs:

- layout region boxes;
- table/title/legend classes;
- key-value candidate pairs;
- routing masks to suppress false symbol/text matches in title blocks.

Training data:

- DocLayNet/PubLayNet for generic layout transfer;
- internal CAD/PDF sheet-level annotations for final target evidence.

Metrics:

- layout mAP;
- table/title-block F1;
- key-value extraction F1;
- downstream false-positive suppression gain.

## Router Design

### Stage 1: Deterministic Router

This is the first production and paper-audit router.

Rules:

- primitive graph line/opening nodes route to WallOpeningExpert;
- closed regions and large polygons route to RoomSpaceExpert;
- compact non-text glyph crops route to SymbolFixtureExpert;
- OCR/text-like components and dimension-line candidates route to TextDimensionExpert;
- large margin/table/title regions route to SheetLayoutExpert.

Outputs:

- routed candidate id;
- target expert;
- feature family;
- routing reason;
- confidence and fallback path.

Why this first:

- it handles incomplete labels;
- it is reproducible;
- it is easier to debug than a learned router;
- it avoids training-set leakage through implicit source/domain shortcuts.

### Stage 2: Learned Router

Add only after each expert has enough labels and stable locked splits.

Candidate model:

- shallow MLP or gradient-boosted classifier over shared candidate features;
- top-k routing with explicit load/capacity limits;
- reject option for unknown or ambiguous candidates;
- optional source/domain feature ablation.

Training labels:

- expert suitability labels derived from routed candidate success/failure;
- hard negatives from false positives;
- route labels selected on train/dev only, never locked test.

Metrics:

- route accuracy;
- expert load balance;
- downstream integrated F1/mAP/IoU;
- false-route recovery rate;
- ablation against deterministic routing.

### Stage 3: Sparse MoE Inside Experts

Use only where needed:

- SymbolFixtureExpert may benefit from symbol-family sub-experts;
- TextDimensionExpert may benefit from text/dimension/callout sub-experts;
- RoomSpaceExpert may benefit from mask/polygon/topology sub-experts.

Do not put a sparse 14B-style router on the critical path until the smaller expert system shows a bottleneck.

## Fusion Design

Fusion turns independent expert outputs into one coherent scene graph.

Hard constraints:

- doors/windows/openings should attach to or interrupt walls;
- rooms should be bounded by wall/opening predictions;
- room labels should lie inside or near the corresponding room polygon;
- symbols should be inside rooms or attached to a plausible host;
- dimensions should link text to dimension/extension/leader lines;
- title-block/table regions should suppress ordinary symbol/text detections unless explicitly allowed.

Soft constraints:

- room type should agree with nearby text when confidence is high;
- fixtures should be plausible for the room type;
- repeated symbols should share class and scale distributions;
- dimensions should be parallel/perpendicular to nearby extension lines when applicable.

Fusion outputs:

- accepted scene graph;
- rejected candidates with reason;
- relation edges;
- warning list;
- per-candidate audit trace.

Fusion methods by maturity:

1. rule constraints and confidence calibration;
2. pairwise relation classifiers;
3. graph optimization with constraint penalties;
4. learned scene-graph refinement after enough integrated labels exist.

## Training Strategy

### Step 1: Freeze And Wrap Current WallOpeningExpert

Deliverables:

- stable expert inference wrapper;
- exported probabilities and audit metadata;
- no behavior change to current locked-test path.

### Step 2: Build Data Converters

Priority:

1. CubiCasa5K SVG to room/symbol/boundary records;
2. DeepFloorplan masks to room-space records;
3. CVC-FP room annotations if conversion is reliable;
4. internal text/dimension annotation format.

Required converter audits:

- source file count;
- class counts;
- bbox/polygon/mask validity;
- empty/invalid annotation rate;
- split leakage report.

### Step 3: Train RoomSpaceExpert

Reason:

- rooms are the most structurally tied to the existing wall/opening model;
- room metrics can be evaluated without solving all symbol/text tasks;
- CubiCasa5K directly supports this.

First target:

- room region IoU above `0.85`;
- room-type macro F1 above `0.85`;
- adjacency F1 above `0.80`;
- no degradation to WallOpeningExpert metrics.

### Step 4: Train SymbolFixtureExpert

Reason:

- symbols add real drawing utility;
- they are separable enough for an independent expert;
- false positives can be contained by sheet/text suppression and room context.

First target:

- AP50 above `0.85` on common CubiCasa categories;
- macro F1 above `0.80` for high-frequency symbols;
- explicit long-tail reporting instead of hiding rare-class failures.

### Step 5: Train TextDimensionExpert

Reason:

- this is high practical value but weakly supported by public floorplan datasets;
- it should wait for internal labels or a clear weak-supervision pipeline.

First target:

- dimension text normalized accuracy above `0.90` on internal dev;
- text-to-dimension-line F1 above `0.85`;
- exact OCR reported separately from normalized numeric accuracy.

### Step 6: Integrated Fine-Tuning

Only after independent experts work.

Options:

- freeze experts, train fusion/relation heads;
- fine-tune only router and calibration;
- joint train with loss masks for partially annotated samples;
- avoid full end-to-end training until label coverage is broad.

## Losses And Objectives

Use separate losses by output type:

- boundary labels: cross entropy / focal only if class imbalance demands it;
- rooms: mask/polygon loss + room-type cross entropy + adjacency BCE;
- symbols: detector classification + box regression + orientation loss;
- text/dimensions: OCR sequence loss if trained, plus relation/linking BCE;
- fusion: relation classification + constraint violation penalty.

Important:

- partial labels must be masked, not treated as negatives;
- source/domain balancing should be audited, because prior source-balanced attempts hurt the wall/opening model;
- optimize per-expert first, integrated score second.

## Evaluation Plan

### Per-Expert Metrics

WallOpeningExpert:

- accuracy;
- macro F1;
- per-class F1;
- probability R2;
- cross-source macro F1.

RoomSpaceExpert:

- room IoU;
- room boundary F1;
- room-type macro F1;
- room adjacency F1.

SymbolFixtureExpert:

- mAP/AP50/AP75;
- per-class F1;
- orientation accuracy;
- host-link F1.

TextDimensionExpert:

- OCR exact accuracy;
- normalized dimension accuracy;
- dimension-line F1;
- text-line linkage F1.

SheetLayoutExpert:

- layout mAP;
- title/table/legend F1;
- key-value extraction F1.

### Integrated Metrics

- scene graph node F1;
- scene graph edge F1;
- relation consistency score;
- valid drawing constraint rate;
- cross-dataset generalization;
- expert ablation impact.

### Split Policy

Use three levels:

- dev: architecture selection;
- locked test: final reported numbers;
- cross-source locked test: generalization evidence.

Leakage controls:

- group rotation/augmentation variants;
- keep source-stratified splits;
- never select router rules on locked test;
- report seed mean/std for final models.

## Ablation Plan

Required ablations:

- monolithic multitask model vs MoE expert decomposition;
- deterministic router vs learned router;
- with/without WallOpeningExpert boundary support for rooms;
- with/without graph-consistency fusion;
- with/without SE(2) geometry features;
- with/without learned crop evidence;
- with/without source/domain metadata;
- frozen experts vs joint fine-tuning;
- complete vs partial-label masking.

Efficiency ablations:

- dense expert heads vs Tensor-Ring compressed heads;
- tiled crop evaluation vs full-batch crop evaluation;
- top-1 vs top-k routing;
- expert cache enabled/disabled.

Failure audits:

- false room merges/splits;
- symbol/text confusion;
- dimension text linked to wrong line;
- door/window not attached to wall;
- room type disagreement with text;
- title-block false positives.

## OOM And Efficiency Plan

Known risk:

- full-page high-resolution crops and long VLM context can dominate memory;
- integrated multi-expert training can multiply crop tensors;
- symbol/text detectors can create many candidates per page.

Controls:

- keep expert training separate;
- cap image side and crop count per sample;
- tile crop evaluation;
- cache shared features and crop embeddings;
- use CPU-backed datasets and move only each batch to CUDA;
- use gradient accumulation rather than oversized batches;
- add candidate NMS before expert inference;
- use Tensor-Ring or low-rank heads only after quality baseline is stable;
- log peak memory per expert and per split.

Memory audit fields:

- max candidates/page;
- max crop tensors/page;
- max sequence length for VLM-assisted runs;
- CUDA peak allocated/reserved;
- skipped sample count and reason.

## Engineering Deliverables

Already started:

- `configs/vlm/cadstruct_ontology.json`;
- `configs/vlm/dataset_registry.json`;
- `scripts/vlm/cadstruct_moe/schema.py`;
- `scripts/vlm/cadstruct_moe/router.py`;
- `scripts/vlm/cadstruct_moe/fusion.py`;
- expert wrapper skeletons;
- `scripts/vlm/audit_cubicasa5k_svg.py`;
- `scripts/vlm/convert_cubicasa5k_svg.py`;
- `scripts/vlm/prepare_room_space_dataset.py`;
- `scripts/vlm/train_room_space_expert.py`;
- `scripts/vlm/evaluate_room_space_expert.py`;
- `scripts/vlm/export_moe_scene_graph.py`;
- `scripts/vlm/audit_moe_scene_graph.py`.

Next files:

- `reports/vlm/moe_architecture_validation_summary.json`.

Directory targets:

- `datasets/cadstruct_rooms_v1`;
- `datasets/cadstruct_symbols_v1`;
- `datasets/cadstruct_integrated_v1`;
- `checkpoints/cadstruct_moe_wall_opening`;
- `checkpoints/cadstruct_moe_room_space`;
- `checkpoints/cadstruct_moe_symbol_fixture`;
- `reports/vlm/moe/`.

## Paper Contribution Framing

The strongest paper story is:

```text
CadStruct-MoE is a structure-aware floorplan scene-graph parser that decomposes raster drawing recognition into
auditable geometric experts. It preserves high-confidence wall/opening primitive recognition, adds room/symbol/text
specialists through routed candidate families, and enforces architectural graph constraints during fusion.
```

Core claims to prove:

- expert decomposition beats monolithic multitask training under partial labels;
- graph-constraint fusion reduces structurally invalid predictions;
- preserving the wall/opening expert avoids catastrophic forgetting;
- cross-family routing improves extensibility and auditability;
- integrated scene graph metrics improve over independent expert outputs.

Claims not yet supported:

- complete CAD understanding;
- broad zero-shot generalization across all drawing domains;
- dimension extraction without internal target labels;
- superiority over all commercial OCR/CAD systems.

## Milestones

### M0: Planning And Scaffold

Status: mostly done.

Exit criteria:

- ontology and registry exist;
- MoE router/schema/fusion skeleton compiles;
- roadmap and architecture plan exist.

### M1: CubiCasa5K Room/Symbol Ingestion

Exit criteria:

- official CubiCasa5K package downloaded and unpacked;
- SVG audit reports class counts;
- room/symbol converters produce manifests;
- leakage-aware train/dev/locked splits exist.

Current implementation status:

- official CubiCasa5K is downloaded, ZIP-verified, and unpacked under
  `datasets/external/cubicasa5k_zenodo/unpacked`;
- the real SVG audit covers 5,000 SVG files with zero parse errors;
- the audit confirms useful structural classes: `Wall`, `Door`, `Window`, `Space`, room types, fixtures,
  dimensions, and leader/direction marks;
- real CubiCasa5K conversion is available in `datasets/cadstruct_cubicasa5k_moe`;
- the converter supports inherited SVG group semantics, CamelCase labels, text/dimension candidates, and sparse
  wall attachment relations for OOM control;
- room-space extraction code is available in `scripts/vlm/prepare_room_space_dataset.py`;
- dependency-free RoomSpaceExpert baseline training/evaluation is available in
  `scripts/vlm/train_room_space_expert.py` and `scripts/vlm/evaluate_room_space_expert.py`;
- the converter intentionally treats Pillow as optional so annotation-only SVG audits can run before image dependencies are installed;
- a smoke test with a synthetic SVG validates wall, door, room, and symbol extraction;
- an end-to-end temporary JSONL smoke test validates room-space baseline training and evaluation;
- synthetic integrated pipeline validation now covers SVG conversion, room data preparation, room baseline training,
  fused scene-graph export, and scene-graph audit.

Current real-data counts:

- CubiCasa5K MoE records: train `4,443`, dev `493`, smoke `64`;
- family counts: boundary `601,567`, space `76,789`, symbol `211,238`, text `545,043`;
- room-space rows: train `4,436`, dev `493`, smoke `64`;
- room instances: train `68,286`, dev `7,491`, smoke `927`;
- room adjacency weak labels: train `31,202`, dev `3,392`, smoke `389`;
- symbol fixture records: train `188,283`, dev `20,542`, smoke `2,413`;
- symbol host links: train `56,600`, dev `6,331`, smoke `743`;
- text/dimension candidates: train `485,740`, dev `52,631`, smoke `6,672`;
- dimension weak links: train `109,438`, dev `11,840`, smoke `1,510`;
- fused smoke scene graph: `17,385` nodes, `35,982` edges, `5` records with residual warnings.

Current OOM audit result:

- dense pairwise boundary relations produced a 34 MiB fused smoke file and 190,549 smoke edges;
- sparse opening-to-wall relations keep fused smoke at 12 MiB and 35,982 edges after adding symbol/text candidates;
- converted train JSONL is 491 MiB, room-space train JSONL is 144 MiB, symbol train JSONL is 35 MiB,
  and text/dimension train JSONL is 62 MiB.

Current quality status:

- the bbox-prototype RoomSpaceExpert is a pipeline check only;
- dev accuracy is `0.347083`, dev macro F1 is `0.222628`, and mean IoU is `1.0` because it reuses ground-truth boxes;
- the first learned RoomSpaceExpert crop-MLP improves dev accuracy to `0.459485` and dev macro F1 to `0.297775`,
  but this remains far below paper quality because it still uses gold room boxes and no mask/topology/text fusion;
- the first structure-aware RoomSpaceExpert context-MLP improves dev accuracy to `0.577390` and dev macro F1 to
  `0.449065` by adding contained-symbol, boundary-touch, room-label-count, and adjacency features;
- the streamed RoomSpace context implementation preserves those metrics while reducing peak RSS from about `4.0`
  GiB to about `1.6` GiB and writing an auditable training feature cache;
- replacing gold symbol/text labels with current learned SymbolFixture/TextDimension predictions gives dev accuracy
  `0.576990` and dev macro F1 `0.448567`, only `-0.000498` macro F1 below the gold-context result;
- a RandomForest capacity baseline over the same context features improves RoomSpace dev accuracy to `0.639381`,
  dev macro F1 to `0.525586`, smoke accuracy to `0.628910`, and smoke macro F1 to `0.515591`;
- enhanced structural features further improve the best dev model to accuracy `0.690975` and macro F1 `0.581734`;
- enhanced HistGBDT gives the strongest smoke macro F1 so far at `0.608901`, but with lower dev macro F1 `0.566881`;
- SVG polygon shape features improve the best dev model further to accuracy `0.699640` and macro F1 `0.589379`;
- the bbox-prototype SymbolFixtureExpert is also weak: dev accuracy is `0.396748`, dev macro F1 is `0.286522`;
- the first learned SymbolFixtureExpert crop-MLP improves dev accuracy to `0.814721` and dev macro F1 to
  `0.614068`, but leaves `bathtub` and `generic_symbol` at zero F1;
- the bbox-prototype TextDimensionExpert is more useful but incomplete: dev accuracy is `0.843970`,
  dev macro F1 is `0.607816`, and weak `dimension_of` F1 is `0.581716`;
- the first learned TextDimensionExpert crop-MLP improves dev accuracy to `0.940738`, dev macro F1 to `0.729927`,
  and weak `dimension_of` F1 to `0.861986` with only `50.0` MiB CUDA reserved memory;
- next accuracy work must replace RoomSpace prototype with a learned crop/mask/topology expert, add long-tail handling
  to SymbolFixtureExpert, and add OCR/content features to TextDimensionExpert.

RoomSpace context result:

- local crop statistics are not enough for room typing;
- contained fixtures and boundary/topology context produce the largest RoomSpace gain so far;
- predicted SymbolFixture/TextDimension outputs do not materially hurt RoomSpace context under gold boxes;
- classifier capacity was part of the problem, but the RandomForest result is still far from paper-grade;
- enhanced room topology/context features help, which validates the structure-aware direction;
- room polygon shape is useful but not enough by itself;
- this supports the paper's structure-aware MoE claim, but the current context MLP still uses gold room boxes and gold
  boundary semantics;
- next step is to add CubiCasa-aligned WallOpening predictions and then replace gold room boxes with detector/mask
  proposals.

Environment status:

- system `python` does not currently expose `torch`, `Pillow`, or `numpy`;
- `.venv-vlm/bin/python` does expose `torch`, `Pillow`, and `numpy`;
- learned expert training should use `.venv-vlm/bin/python` unless the system environment is rebuilt.

### M2: RoomSpaceExpert V1

Exit criteria:

- room region and room-type training runs;
- room IoU/type F1/adjacency F1 reported;
- fusion can link rooms to wall/opening boundaries;
- no regression to WallOpeningExpert outputs.

### M3: SymbolFixtureExpert V1

Exit criteria:

- common symbol categories trained and evaluated;
- text/title-block false-positive audit exists;
- host-link relation head reported.

### M4: Integrated MoE Scene Graph V1

Exit criteria:

- deterministic router runs end to end;
- all available experts export normalized predictions;
- fusion produces scene graph and warnings;
- integrated node/edge metrics reported on locked split.

### M5: Learned Router And Paper Ablations

Exit criteria:

- learned router beats deterministic router or is reported as negative;
- monolithic baseline is trained;
- seed mean/std table exists;
- paper-ready ablation table is complete.

## Immediate Execution Order

1. Replace the bbox-prototype RoomSpaceExpert with a learned crop/mask/topology RoomSpaceExpert.
2. Add room adjacency labels and metrics from CubiCasa5K topology.
3. Add symbol fixture extraction datasets from the converted CubiCasa labels.
4. Add TextDimensionExpert weak-supervision records from `dimension_line` and `leader_line` candidates.
5. Run integrated wall/opening + room evaluation with locked cross-dataset splits.
6. Add candidate-count and peak-memory logging to every expert training entry point.
7. Train a monolithic multitask baseline for the MoE ablation table.

This order gives the fastest path from current wall/opening recognition to a real MoE drawing parser.
