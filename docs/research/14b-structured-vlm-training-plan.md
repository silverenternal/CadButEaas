# 14B Structured VLM Training Plan

Date: 2026-04-29

## Recommendation

Use `OpenGVLab/InternVL3_5-14B-HF` as the trainable 14B-class base model, with `Qwen/Qwen3-VL-32B-Instruct` as the teacher and paper-grade comparison baseline.

This is the best fit because the task is not generic image captioning. Raster CAD and floor-plan drawings are structured visual artifacts: long straight segments, repeated symbols, layer-like regions, dimension text, topology, containment, and metric constraints. A plain VLM fine-tune will improve JSON following, but it will still miss geometric consistency unless we expose structured primitives during training and decoding.

## Current Frontier

### 1. General open VLMs

- `Qwen3-VL`: strong latest open family for general multimodal reasoning, long context, grounding, and document/video inputs. The family includes dense 2B/4B/8B/32B and MoE 30B-A3B/235B-A22B variants, but no native 14B dense model. Use 32B as teacher/baseline, not as the 14B target.
- `InternVL3.5`: strongest practical 14B-class candidate. The 14B HF checkpoint is about 15B parameters, Apache-2.0, initialized from Qwen3-family LLMs plus InternViT, and trained with CPT, SFT, and cascade RL. It explicitly reports OCR, chart, document, visual grounding, GUI, SVG, and multilingual evaluations.

### 2. Document-structure specialists

- `PaddleOCR-VL-0.9B`: small but very relevant for document parsing. It uses NaViT-style dynamic resolution and is optimized for element recognition across multilingual documents. It should be used as a specialist OCR/layout auxiliary, not as the main reasoning model.
- `DeepSeek-OCR`: relevant because it frames document understanding as optical context compression. It is useful for long sheets, dense text, tables, and symbol-heavy regions where token budget becomes the bottleneck.
- Coarse-to-fine document parsing is becoming a clear pattern: detect/route important regions first, then run detailed recognition on selected crops. This is more appropriate for large CAD sheets than sending the whole page to a single VLM at fixed resolution.

### 3. CAD/floor-plan structure methods

- CAD generation work such as DeepCAD/Text2CAD/TransCAD uses sequence or hierarchical transformers over structured CAD operations rather than plain pixels.
- Floor-plan parsing work repeatedly benefits from graph representations: rooms/walls/openings as nodes and spatial adjacency/containment as edges.
- Recent floor-plan + LLM/VLM papers report that graph-based spatial representations outperform direct visual reasoning for downstream tasks. This supports a geometry-augmented VLM design.

## Proposed Model

Name for internal use: `CadStruct-VL-14B`.

Base:

- `OpenGVLab/InternVL3_5-14B-HF`.

Model type:

- Primary model type: autoregressive multimodal language model with a geometry/graph adapter.
- It is a discriminative-to-structured-generation model: image/primitive/graph inputs -> JSON and scene graph outputs.
- It is not primarily a diffusion or flow-matching model in Stage 1, because the first research target is faithful extraction and structural parsing rather than open-ended plan generation.

Added inputs:

- Original raster image or high-resolution crops.
- OCR candidates: text, bbox, confidence.
- Vector primitive candidates from our pipeline: line, arc, polyline, contour, fitted rectangle, symbol candidate.
- Graph tokens: node type, normalized bbox/centroid, angle/length, adjacency, intersection, containment, parallel/perpendicular hints.
- Request hints: expected schema, domain, units, target classes.

Outputs:

- Strict JSON matching `RasterVlmOutput`.
- Optional normalized scene graph for training/eval:
  - nodes: wall, partition_wall, door, window, room, dimension_text, centerline, symbol.
  - edges: adjacent_to, opens_into, bounds, intersects, parallel_to, dimension_of, contained_in.

Training objective:

- SFT for schema-following and candidate extraction.
- Auxiliary losses for node classification, relation classification, and bbox/line endpoint regression where labels exist.
- Preference/RL stage with geometry rewards:
  - valid JSON.
  - no impossible topology.
  - dimension values consistent with OCR and scale.
  - doors/windows attached to walls.
  - centerlines remain aligned with elongated geometry.

## Training Stages

### Stage 0: Baseline Evaluation

Evaluate unmodified models on the same data:

- `InternVL3.5-14B-HF`.
- `Qwen3-VL-8B-Instruct`.
- `Qwen3-VL-32B-Instruct`.
- `PaddleOCR-VL-0.9B` for OCR/layout only.
- `DeepSeek-OCR` for dense document/OCR conversion only.

Datasets:

- Synthetic `datasets/raster_vlm`.
- FloorPlanCAD.
- CVC-FP.
- CubiCasa5K after official download completes.
- A small internal real-drawing annotation set.

Metrics:

- JSON success.
- dimension hit.
- symbol F1.
- semantic class F1.
- relation F1.
- geometry consistency score.
- latency and VRAM.

### Stage 1: LoRA SFT

Train LoRA/QLoRA first:

- Freeze most of the base VLM.
- Train language adapters and multimodal projector.
- Add structured primitive text serialization as prompt context.
- Target: stable schema and obvious semantic lift over base.

Expected hardware:

- GPU0 RTX PRO 6000 96GB: primary training.
- GPU1 RTX 5090 32GB: eval/data generation.

### Stage 2: Structure Adapter

Add a small graph encoder:

- Encode primitive graph with a lightweight transformer/GNN.
- Project graph embedding into VLM token space.
- Keep the 14B base mostly frozen for the first run.

This is the likely publishable differentiator: not just "we fine-tuned a VLM", but "we inject CAD topology and geometric primitives into a 14B VLM and train with geometry-consistency objectives."

### Stage 3: Teacher-Generated Data

Use `Qwen3-VL-32B-Instruct` as a teacher:

- Generate structured JSON for difficult samples.
- Use our vectorizer/OCR to verify and filter labels.
- Keep only samples passing schema and geometry checks.

Avoid relying on teacher text alone. For a paper, the filtering and consistency checks matter more than raw pseudo-label volume.

### Stage 4: Preference / RL

Create pairwise preferences:

- valid vs invalid JSON.
- attached vs floating door/window.
- correct vs wrong dimension association.
- complete vs missing wall graph.

Use lightweight DPO/ORPO first. Only move to heavier RL if SFT + preference tuning plateaus.

## Data Plan

Immediate usable data:

- FloorPlanCAD: main external floor-plan/CAD corpus.
- CVC-FP: wall/opening/room robustness and segmentation-style labels.
- Synthetic data: exact dimensions and semantic classes.

Needs completion:

- CubiCasa5K official Zenodo package.
- 200-500 internal real raster drawings with manual correction, focused on dimensions, doors/windows, centerlines, and engineering symbols.

Label format:

```json
{
  "image": "path/to/page.png",
  "primitives": [{"type": "line", "bbox": [0, 0, 1, 1], "angle": 0.0}],
  "ocr": [{"text": "80", "bbox": [0, 0, 1, 1]}],
  "target": {
    "dimensions": [],
    "symbols": [],
    "semantic_regions": [],
    "scene_graph": {"nodes": [], "edges": []}
  }
}
```

## Paper Angle

The strongest paper framing is:

> Geometry-augmented multimodal training for raster CAD understanding.

Core claims to test:

1. A 14B VLM can be competitive if given vector primitives and topology, even against larger generic VLMs.
2. Structure-aware inputs improve door/window/wall/centerline semantics more than image-only fine-tuning.
3. Geometry-consistency filtering improves pseudo-label quality for domain adaptation.
4. Coarse-to-fine crop routing reduces token cost without losing small symbols.

## Flow Matching Position

Flow matching is promising, but it should not be the first-stage backbone for this project.

Why not as the main model:

- Our immediate task is conditional extraction from an observed raster drawing. We need deterministic, auditable JSON, symbol labels, dimensions, and relations.
- Flow matching is strongest when learning a continuous generative transport from noise to data. It is a better fit for generating or repairing layouts than for precise OCR-linked extraction.
- CAD/floor-plan outputs contain mixed discrete-continuous structure: symbol classes, topology, graph edges, line endpoints, dimension text, and units. A pure flow model would need extra machinery for discrete graph validity.

Where it is suitable:

- A second-stage layout prior that refines noisy wall/room/opening graphs.
- Generating synthetic floor-plan variants conditioned on room graphs or text prompts.
- Continuous endpoint/bbox refinement after the VLM predicts classes and relations.
- A publishable extension: `VLM parser + flow-matching structural refiner`.

Recommended stance:

- Stage 1 paper path: `InternVL3.5-14B + graph adapter + geometry-consistency SFT/DPO`.
- Stage 2 extension: conditional flow matching over normalized scene graphs or vector primitives for repair/generation.
- Do not replace the VLM with flow matching for OCR, instruction following, or schema generation.

## Near-Term Execution

1. Download `OpenGVLab/InternVL3_5-14B-HF`. Done: local path `models/vlm/internvl3_5_14b_hf`, about 29 GiB.
2. Add `configs/vlm/internvl3_5_14b_train.json`. Started as `configs/vlm/cadstruct_14b_lora.json`.
3. Extend the dataset generator to include primitive graph JSON. Done: `request_hints.primitive_graph` and `expected_json.scene_graph`.
4. Build an evaluator for relation F1 and geometry consistency. Done: `relation_f1_mean` and `geometry_consistency_mean`.
5. Run baseline eval on InternVL3.5-14B and Qwen3-VL-32B. Started: InternVL3.5-14B 1-sample smoke passed JSON/dimension, missed semantics.
6. Start LoRA SFT on 200 synthetic + FloorPlanCAD/CVC-FP converted samples. Next.

Current execution status:

- `scripts/vlm/download_model.py` downloads model weights into git-ignored `models/`.
- `configs/vlm/internvl3_5_14b_eval.json` points to the local InternVL3.5-14B weights.
- Rust `RasterVlmInput` now carries `primitive_graph`; `RasterVlmOutput` can carry optional `scene_graph`.
- `scripts/vlm/convert_external_dataset.py` converts FloorPlanCAD and CVC-FP into `datasets/cadstruct`.
- First converted CadStruct set: 2,132 train, 236 dev, 32 smoke samples; 1,200 FloorPlanCAD + 1,200 CVC-FP.
- `scripts/vlm/prepare_sft_dataset.py` converts CadStruct records into SFT-ready multimodal messages at `datasets/cadstruct_sft`.
- Mock graph smoke report: `reports/vlm/mock_graph_smoke_8.json`.
- Mock CadStruct smoke report: `reports/vlm/mock_cadstruct_smoke_32.json`.
- InternVL3.5-14B smoke report: `reports/vlm/internvl3_5_14b_smoke_1.json`.
- InternVL3.5-14B first smoke result: JSON success 1.0, dimension hit 1.0, semantic hit 0.0, geometry consistency 1.0, latency about 21.9s.
- CadStruct-VL 14B LoRA smoke training completed for 2 steps and saved adapter to `checkpoints/cadstruct_vl_14b_lora`.
- `configs/vlm/cadstruct_14b_lora_eval.json` loads the base model plus the PEFT adapter through the same HTTP sidecar.
- CadStruct-VL 14B LoRA 1-sample eval passed JSON after loading the PEFT adapter through the sidecar; report path `reports/vlm/cadstruct_14b_lora_smoke_1.json`.
- SFT prompt compaction now caps polyline samples, primitive graph nodes/edges, and target candidates so assistant labels survive truncation.
- `scripts/vlm/profile_sft_budget.py` profiles token count, supervised-token count, and vision tiles before training.
- Latest compact SFT budget profile over 128 records at `max_length=6144`, `max_image_side=512`: input tokens mean 3753.56, p95 6144, max 6144; supervised tokens mean 1236.57; vision tiles mean 3.95, p95 13, max 13; zero-supervised 0; at-max-length 7.
- Current OOM guardrails in `configs/vlm/cadstruct_14b_lora.json`: `per_device_train_batch_size=1`, `gradient_accumulation_steps=16`, `load_in_4bit=true`, `max_length=6144`, `max_image_side=512`, `max_vision_tiles=8`, `skip_at_max_length=true`, and `skip_oom_samples=true`.
- Training now skips samples that exceed the token/tile budget, have no supervised labels, produce nonfinite loss, or hit CUDA OOM.
- CadStruct-VL 14B LoRA 16-effective-step compact SFT run completed without OOM, NaN, or skipped samples; mean loss 0.494663; peak allocated CUDA memory was 44,167 MiB.
- CadStruct-VL 14B LoRA 2-sample eval still emits valid JSON but no semantic candidates, so the next step is prompt/target shaping for explicit non-empty structural extraction before scaling training.
- SFT encoding, sample budget accounting, JSONL loading, and device transfer are now isolated in `scripts/vlm/sft_utils.py`; `train_lora.py` and `profile_sft_budget.py` share the same budget rules.
- The trainer now checks token/tile budget on CPU before moving tensors to GPU, so known over-budget samples do not allocate CUDA memory.
- Short refactor smoke run passed: 1 effective step, no checkpoint save, no OOM, peak allocated CUDA memory 19,796 MiB, audit log `reports/vlm/train_refactor_audit.jsonl`.
- Fixed `normalize_semantic` in `scripts/vlm/server.py`; malformed model JSON can now recover complete semantic candidate objects from raw output and records `partial_json_recovered`.
- SFT and inference prompts were reshaped to semantic-first output order: `semantic_candidates`, `scene_graph`, `symbol_candidates`, `dimension_candidates`, `warnings`.
- Latest semantic-first 16-effective-step run completed with mean loss 0.327823, one budget skip for a 13-tile sample, no OOM/NaN, and peak allocated CUDA memory 45,057 MiB.
- Latest 2-sample LoRA eval: JSON success 1.0, dimension hit 1.0, semantic hit 0.5, relation F1 0.0, geometry consistency 0.58, report `reports/vlm/cadstruct_14b_lora_semantic_first_repair_smoke_2.json`.
- Remaining gap: the model still emits an empty structure for one CVC-FP smoke sample, and recovered semantics have weak geometry consistency, so the next model-side step is more semantic-first SFT steps plus stronger target filtering/negative examples.
- Output parsing/repair/normalization is now decoupled into `scripts/vlm/output_contract.py`; the HTTP sidecar imports that contract instead of owning parsing logic.
- Added `scripts/vlm/test_output_contract.py` for fast parser regression checks covering fenced JSON, scene-graph semantic repair, and partial semantic recovery.
- `scripts/vlm/evaluate_backend.py` now reports audit fields: `empty_semantic_rate`, `semantic_count_mean`, `partial_recovery_count`, and `warning_counts`.
- Added offline report auditing at `scripts/vlm/audit_eval_report.py`; audit for the latest 2-sample report is `reports/vlm/cadstruct_14b_lora_semantic_first_repair_smoke_2.audit.json`.
- Evaluation metrics are now decoupled into `scripts/vlm/eval_metrics.py`, with regression checks in `scripts/vlm/test_eval_metrics.py`.
- Added CadStruct target-distribution audit at `scripts/vlm/audit_cadstruct_dataset.py`; latest report `reports/vlm/cadstruct_dataset_audit.json`.
- Dataset audit findings: train/dev/smoke have zero empty semantic rows and zero invalid semantic target references; raw graph size is large, with train scene graph edges max 2,073, so SFT graph/target caps remain necessary for OOM control.
- Added model-target fit auditing at `scripts/vlm/audit_model_target_fit.py`; latest report `reports/vlm/model_target_fit_audit.json`.
- Fit audit decision: `InternVL3.5-14B + LoRA` is a reasonable schema-following VLM backbone, but it is not by itself matched to full dense graph extraction. It should be paired with scoped targets now and a graph adapter/node-edge classifier next.
- Main mismatch findings: dense node classification fit is low, relation graph generation fit is low, taxonomy alignment is low because targets include furniture/equipment classes outside the structural prompt, and dimension extraction fit is low because external converted data has no dimension labels.
- Added structural-core SFT generation: `datasets/cadstruct_sft_structural` keeps hard_wall/door/window targets, drops furniture/equipment classes, and drops rows that become empty after filtering.
- Structural-core SFT set: train 2,052 rows, dev 228 rows, smoke 32 rows; no empty semantic rows after filtering; train semantic mean 10.13 candidates and edge mean 13.76.
- Structural-core budget profile `reports/vlm/sft_budget_128_structural_core.json`: input token mean 3,749.42, supervised token mean 1,071.11, p95 still 6,144, and budget skips remain 7 samples due to 13 vision tiles.
- Structural-core LoRA 16-effective-step run completed in `checkpoints/cadstruct_vl_14b_lora_structural`: mean loss 0.458882, skipped_budget 2, no OOM/NaN, peak allocated CUDA memory 45,224 MiB.
- Structural-core 2-sample eval with strict target-id metric: semantic hit 1.0, but semantic exact F1 mean only 0.0187, relation F1 0.0, geometry consistency 0.0, and partial recovery on both samples. This confirms the model can emit non-empty structural labels but is not yet solving per-node classification.
- Added structural node-classification dataset generation at `scripts/vlm/prepare_graph_node_dataset.py`; latest `datasets/cadstruct_graph_nodes` has train 2,052 rows / 84,345 nodes, dev 228 rows / 9,601 nodes, smoke 32 rows / 1,245 nodes with labels hard_wall/door/window.
- Updated conclusion: Stage 1 should use the VLM for image/context conditioning and schema, but structural correctness needs a node classifier or graph adapter trained on `datasets/cadstruct_graph_nodes`.
- Implemented first CadStruct-owned structural model at `scripts/vlm/train_graph_node_classifier.py`.
- Node classifier checkpoint: `checkpoints/cadstruct_graph_node_classifier/model_best.pt`.
- Node classifier dev result after 20 epochs: accuracy 0.65212, macro F1 0.552994; per-label F1 hard_wall 0.77671, door 0.552314, window 0.329958.
- Node classifier smoke result: accuracy 0.650602, macro F1 0.55446; per-label F1 hard_wall 0.772069, door 0.580407, window 0.310905; report `reports/vlm/graph_node_classifier_smoke.json`.
- This confirms we now have a project-owned structure module. It is still a baseline MLP, but it directly addresses the observed failure mode: per-node semantic assignment.

## OOM Risk Notes

The current stable setting fits comfortably on GPU0, but the budget profile shows the next OOM risks clearly:

- CVC-FP rotated samples can still produce 13 vision tiles at `max_image_side=512`; the current training config skips anything above 8 tiles.
- Seven of the first 128 profiled SFT records land exactly at `max_length=6144`; those are treated as unsafe because assistant labels may be truncated or attention cost may spike.
- Increasing `max_image_side`, `max_length`, LoRA rank, or trainable modules should be preceded by a fresh `profile_sft_budget.py` run and a short `--max-steps` smoke run.
- GPU1 has about 32 GiB VRAM and should stay on eval/data-generation duties for this 14B QLoRA path unless the model is sharded deliberately.

## Sources

- Qwen3-VL Technical Report: https://arxiv.org/abs/2511.21631
- Qwen3-VL-32B-Instruct model card: https://huggingface.co/Qwen/Qwen3-VL-32B-Instruct
- InternVL3.5 Technical Report: https://arxiv.org/abs/2508.18265
- InternVL3.5-14B-HF model card: https://huggingface.co/OpenGVLab/InternVL3_5-14B-HF
- PaddleOCR-VL paper: https://arxiv.org/abs/2510.14528
- PaddleOCR-VL Transformers docs: https://huggingface.co/docs/transformers/en/model_doc/paddleocr_vl
- DeepSeek-OCR paper: https://arxiv.org/abs/2510.18234
- Flow Matching for Generative Modeling: https://openreview.net/forum?id=PqvMRDCJT9t
- Conditional Flow Matching overview: https://iclr-blogposts.github.io/2025/blog/conditional-flow-matching/
- DeepCAD paper: https://arxiv.org/abs/2105.09492
- TransCAD paper: https://arxiv.org/abs/2407.12702
- Floorplan2Guide: https://arxiv.org/abs/2512.12177
