# Raster VLM Sidecar

## CadStruct asset ownership

This directory contains both canonical CubiCasa-derived MoE assets and later raster-only experiments. Use these two files before choosing an entry point:

- `docs/cadstruct/legacy-cubicasa-moe.md`: human-readable ownership and module boundaries for the historical CubiCasa/MoE expert stack.
- `configs/vlm/cadstruct_legacy_moe_registry.json`: machine-readable registry for canonical datasets, checkpoints, reports, and experimental raster rebuild assets.

Do not treat every script in this directory as part of the current production path. Many files are ablations, diagnostics, or failed raster-only experiments kept for auditability.

Create an isolated environment with `uv`:

```bash
uv venv .venv-vlm
source .venv-vlm/bin/activate
uv pip install -r scripts/vlm/requirements.txt
```

Start the mock backend:

```bash
python scripts/vlm/server.py --config configs/vlm/default.json
```

Start the Qwen3-VL smoke backend:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vlm/server.py --config configs/vlm/qwen3_vl_8b_smoke.json
```

Start the paper-grade primary backend:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vlm/server.py --config configs/vlm/qwen3_vl_32b_paper.json
```

Start the 14B CadStruct baseline backend:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vlm/server.py --config configs/vlm/internvl3_5_14b_eval.json
```

Start the trained CadStruct LoRA backend:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vlm/server.py --config configs/vlm/cadstruct_14b_lora_eval.json
```

Smoke test:

```bash
python scripts/vlm/client_smoke_test.py
```

Evaluate a JSONL split:

```bash
python scripts/vlm/evaluate_backend.py --dataset datasets/raster_vlm/smoke.jsonl --output reports/vlm/qwen3_vl_8b_smoke.json
```

Audit an existing evaluation report without starting a model:

```bash
python scripts/vlm/audit_eval_report.py \
  reports/vlm/cadstruct_14b_lora_semantic_first_repair_smoke_2.json \
  --output reports/vlm/cadstruct_14b_lora_semantic_first_repair_smoke_2.audit.json
```

Audit CadStruct target distributions:

```bash
python scripts/vlm/audit_cadstruct_dataset.py \
  --output reports/vlm/cadstruct_dataset_audit.json
```

Audit model-target fit:

```bash
python scripts/vlm/audit_model_target_fit.py \
  --output reports/vlm/model_target_fit_audit.json
```

Generate synthetic smoke data:

```bash
python scripts/vlm/generate_dataset.py
```

Convert downloaded external datasets into the CadStruct JSONL schema:

```bash
python scripts/vlm/convert_external_dataset.py
```

Prepare multimodal SFT records:

```bash
python scripts/vlm/prepare_sft_dataset.py
```

Prepare a model-target aligned structural-core SFT set:

```bash
python scripts/vlm/prepare_sft_dataset.py \
  --output-dir datasets/cadstruct_sft_structural \
  --target-scope structural_core \
  --drop-empty-targets
```

Prepare a structural node-classification dataset for the graph adapter path:

```bash
python scripts/vlm/prepare_graph_node_dataset.py \
  --output-dir datasets/cadstruct_graph_nodes

python scripts/vlm/prepare_graph_node_dataset.py \
  --output-dir datasets/cadstruct_graph_nodes_topology \
  --include-topology-features

python scripts/vlm/prepare_graph_node_dataset.py \
  --output-dir datasets/cadstruct_graph_nodes_lie_topology \
  --include-topology-features \
  --include-lie-features
```

Train and evaluate the CadStruct-owned node classifier:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vlm/train_graph_node_classifier.py \
  --epochs 20 \
  --batch-size 4096 \
  --output-dir checkpoints/cadstruct_graph_node_classifier

CUDA_VISIBLE_DEVICES=0 python scripts/vlm/evaluate_graph_node_classifier.py \
  --dataset datasets/cadstruct_graph_nodes/smoke.jsonl \
  --output reports/vlm/graph_node_classifier_smoke.json \
  --predictions-output reports/vlm/graph_node_classifier_smoke_predictions.jsonl

CUDA_VISIBLE_DEVICES=0 python scripts/vlm/train_graph_node_classifier.py \
  --dataset-dir datasets/cadstruct_graph_nodes_topology \
  --output-dir checkpoints/cadstruct_graph_node_classifier_topology \
  --epochs 20 \
  --batch-size 4096 \
  --seed 20260429

CUDA_VISIBLE_DEVICES=0 python scripts/vlm/evaluate_graph_node_classifier.py \
  --checkpoint checkpoints/cadstruct_graph_node_classifier_topology/model_best.pt \
  --dataset datasets/cadstruct_graph_nodes_topology/smoke.jsonl \
  --output reports/vlm/graph_node_classifier_topology_smoke.json \
  --predictions-output reports/vlm/graph_node_classifier_topology_smoke_predictions.jsonl

CUDA_VISIBLE_DEVICES=0 python scripts/vlm/train_graph_node_classifier.py \
  --dataset-dir datasets/cadstruct_graph_nodes_lie_topology \
  --output-dir checkpoints/cadstruct_graph_node_classifier_lie_gated \
  --model-type gated \
  --experts 3 \
  --epochs 20 \
  --batch-size 4096 \
  --seed 20260429

CUDA_VISIBLE_DEVICES=0 python scripts/vlm/evaluate_graph_node_classifier.py \
  --checkpoint checkpoints/cadstruct_graph_node_classifier_lie_gated/model_best.pt \
  --dataset datasets/cadstruct_graph_nodes_lie_topology/smoke.jsonl \
  --output reports/vlm/graph_node_classifier_lie_gated_smoke.json \
  --predictions-output reports/vlm/graph_node_classifier_lie_gated_smoke_predictions.jsonl

CUDA_VISIBLE_DEVICES=0 python scripts/vlm/train_graph_node_classifier.py \
  --dataset-dir datasets/cadstruct_graph_nodes_lie_topology \
  --output-dir checkpoints/cadstruct_graph_node_classifier_lie_tr_gated \
  --model-type tr_gated \
  --experts 3 \
  --tr-rank 4 \
  --epochs 20 \
  --batch-size 4096 \
  --eval-tile-size 4096 \
  --seed 20260429

CUDA_VISIBLE_DEVICES=0 python scripts/vlm/train_graph_node_classifier.py \
  --dataset-dir datasets/cadstruct_graph_nodes_lie_topology \
  --output-dir checkpoints/cadstruct_graph_node_classifier_lie_tr_gated_balanced \
  --model-type tr_gated \
  --experts 3 \
  --tr-rank 4 \
  --routing-balance-weight 0.05 \
  --epochs 20 \
  --batch-size 4096 \
  --eval-tile-size 4096 \
  --seed 20260429

CUDA_VISIBLE_DEVICES=0 python scripts/vlm/train_graph_node_classifier.py \
  --dataset-dir datasets/cadstruct_graph_nodes_lie_topology \
  --output-dir checkpoints/cadstruct_graph_node_classifier_lie_tr_gated_rank8 \
  --model-type tr_gated \
  --experts 3 \
  --tr-rank 8 \
  --epochs 20 \
  --batch-size 4096 \
  --eval-tile-size 4096 \
  --seed 20260429

CUDA_VISIBLE_DEVICES=0 python scripts/vlm/evaluate_graph_node_classifier.py \
  --checkpoint checkpoints/cadstruct_graph_node_classifier_lie_tr_gated/model_best.pt \
  --dataset datasets/cadstruct_graph_nodes_lie_topology/smoke.jsonl \
  --output reports/vlm/graph_node_classifier_lie_tr_gated_smoke.json \
  --predictions-output reports/vlm/graph_node_classifier_lie_tr_gated_smoke_predictions.jsonl \
  --eval-tile-size 4096
```

Export graph-node classifier predictions as auditable RasterVlmOutput-style candidates:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vlm/export_graph_node_predictions.py \
  --checkpoint checkpoints/cadstruct_graph_node_classifier_topology/model_best.pt \
  --dataset datasets/cadstruct/smoke.jsonl \
  --output reports/vlm/graph_node_classifier_topology_smoke_candidates.jsonl \
  --max-candidates 64
```

The shared graph-node model code lives in `scripts/vlm/graph_node_model.py`; training, evaluation, dataset preparation, and candidate export all reuse that module instead of importing each other. Training keeps feature tensors on CPU and moves only CUDA tiles/batches to GPU; evaluation and routing summaries also accept `--eval-tile-size`.

Use `reports/vlm/graph_node_classifier_research_comparison.json` for the current geometry/topology/SE(2)/gated ablation table. Use `reports/vlm/graph_node_classifier_compression_audit.json` for Tensor-Ring compression and CUDA tiling memory tradeoffs.
Use `reports/vlm/graph_node_classifier_routing_balance_audit.json` for the current routing-collapse versus F1 tradeoff; the unbalanced Tensor-Ring checkpoint is still the best compressed checkpoint.
Use `reports/vlm/graph_node_classifier_tr_rank_audit.json` for the rank-4/rank-8 Tensor-Ring tradeoff; rank 8 is the current best accuracy checkpoint.
The graph-node evaluators now report probability R² in addition to accuracy and F1.

Prepare the current raster-feature graph-node dataset and evaluate the current calibrated quality path:

```bash
python scripts/vlm/prepare_graph_node_dataset.py \
  --input-dir datasets/cadstruct \
  --output-dir datasets/cadstruct_graph_nodes_lie_topology_raster_v3 \
  --include-topology-features \
  --include-lie-features \
  --include-raster-features

CUDA_VISIBLE_DEVICES=0 python scripts/vlm/evaluate_graph_node_ensemble.py \
  --checkpoints checkpoints/cadstruct_graph_node_classifier_lie_raster_gated_h256_e40/model_best.pt \
  --weights 1.0 \
  --class-bias 1.5,1.15,0.7 \
  --dataset datasets/cadstruct_graph_nodes_lie_topology_raster_v3/smoke.jsonl \
  --output reports/vlm/graph_node_classifier_lie_raster_gated_h256_e40_calibrated_smoke.json \
  --predictions-output reports/vlm/graph_node_classifier_lie_raster_gated_h256_e40_calibrated_smoke_predictions.jsonl
```

Audit ensemble weights and class bias on dev before accepting a quality path:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vlm/audit_graph_node_ensemble_search.py \
  --checkpoints checkpoints/cadstruct_graph_node_classifier_lie_raster_gated_h256_e40/model_best.pt \
  --initial-weights 1.0 \
  --dev-dataset datasets/cadstruct_graph_nodes_lie_topology_raster_v3/dev.jsonl \
  --smoke-dataset datasets/cadstruct_graph_nodes_lie_topology_raster_v3/smoke.jsonl \
  --output reports/vlm/graph_node_lie_raster_h256_calibration_audit.json
```

The current calibrated raster quality path reaches dev macro F1 `0.961939` with probability R² `0.938179`, and smoke macro F1 `0.963124` with probability R² `0.937717`. A small post-hoc wall/opening boundary refiner is available for audit:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vlm/train_graph_node_boundary_refiner.py \
  --base-checkpoint checkpoints/cadstruct_graph_node_classifier_lie_raster_gated_h256_e40/model_best.pt \
  --dataset-dir datasets/cadstruct_graph_nodes_lie_topology_raster_v3 \
  --output-dir checkpoints/cadstruct_graph_node_boundary_refiner_lie_raster_h128_e40 \
  --class-bias 1.5,1.15,0.7 \
  --hidden-dim 128 \
  --epochs 40 \
  --batch-size 4096 \
  --learning-rate 7e-4 \
  --weight-decay 1e-4 \
  --eval-tile-size 4096
```

The boundary refiner reaches dev macro F1 `0.962557` and smoke macro F1 `0.963635`, but probability R² drops. Treat it as a diagnostic: decision-layer boundary routing is not enough for the 98% target.

Train the learned local crop encoder path:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vlm/train_graph_node_crop_classifier.py \
  --dataset-dir datasets/cadstruct_graph_nodes_lie_topology_raster_v3 \
  --output-dir checkpoints/cadstruct_graph_node_crop_classifier_h256_c32_e20 \
  --crop-size 32 \
  --hidden-dim 256 \
  --epochs 20 \
  --batch-size 2048 \
  --learning-rate 7e-4 \
  --weight-decay 1e-4 \
  --eval-tile-size 4096
```

The uncalibrated crop model is too opening-heavy, but dev-selected class bias `hard_wall=3.0,door=1.15,window=0.7` makes it the current best quality path: dev macro F1 `0.963178` with probability R² `0.942271`, and smoke macro F1 `0.968028` with probability R² `0.945715`. Use:

- `reports/vlm/graph_node_crop_classifier_h256_c32_e20_calibrated_dev.json`
- `reports/vlm/graph_node_crop_classifier_h256_c32_e20_calibrated_smoke.json`
- `reports/vlm/graph_node_crop_classifier_h256_c32_e20_calibration_audit.json`

Train the current multi-scale crop quality path:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vlm/train_graph_node_crop_classifier.py \
  --dataset-dir datasets/cadstruct_graph_nodes_lie_topology_raster_v3 \
  --output-dir checkpoints/cadstruct_graph_node_crop_classifier_h256_c32_ms3_e24 \
  --crop-size 32 \
  --crop-pad-scales 0.15,0.35,0.8 \
  --hidden-dim 256 \
  --epochs 24 \
  --batch-size 2048 \
  --learning-rate 7e-4 \
  --weight-decay 1e-4 \
  --eval-tile-size 4096

CUDA_VISIBLE_DEVICES=0 python scripts/vlm/audit_graph_node_crop_calibration.py \
  --checkpoint checkpoints/cadstruct_graph_node_crop_classifier_h256_c32_ms3_e24/model_best.pt \
  --dataset-dir datasets/cadstruct_graph_nodes_lie_topology_raster_v3 \
  --class-bias-grid '1.25,1.35,1.45,1.5,1.55,1.65,1.75;0.85,0.9,0.95,1.0,1.05,1.1,1.15;0.7,0.75,0.8,0.85,0.9,0.95,1.0' \
  --output reports/vlm/graph_node_crop_classifier_h256_c32_ms3_e24_fine_calibration_audit.json \
  --dev-report reports/vlm/graph_node_crop_classifier_h256_c32_ms3_e24_fine_calibrated_dev.json \
  --smoke-report reports/vlm/graph_node_crop_classifier_h256_c32_ms3_e24_fine_calibrated_smoke.json \
  --dev-predictions-output reports/vlm/graph_node_crop_classifier_h256_c32_ms3_e24_fine_calibrated_dev_predictions.jsonl \
  --smoke-predictions-output reports/vlm/graph_node_crop_classifier_h256_c32_ms3_e24_fine_calibrated_smoke_predictions.jsonl
```

The multi-scale crop model reaches dev macro F1 `0.975450` with probability R² `0.960990`, and smoke macro F1 `0.976790` with probability R² `0.961526`. A single-scale plus multi-scale probability ensemble was audited at `reports/vlm/graph_node_crop_ensemble_single_ms3_calibration_audit.json`, but dev search selected `single=0.0, ms3=1.0`, so it adds no gain.

Train the current graph-message-passing quality path on top of the same c32-ms3 crops:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vlm/train_graph_node_crop_gnn_classifier.py \
  --dataset-dir datasets/cadstruct_graph_nodes_lie_topology_raster_v3 \
  --output-dir checkpoints/cadstruct_graph_node_crop_gnn_h256_c32_ms3_l2_e24 \
  --crop-size 32 \
  --crop-pad-scales 0.15,0.35,0.8 \
  --hidden-dim 256 \
  --message-layers 2 \
  --epochs 24 \
  --batch-samples 64 \
  --learning-rate 7e-4 \
  --weight-decay 1e-4

CUDA_VISIBLE_DEVICES=0 python scripts/vlm/audit_graph_node_crop_calibration.py \
  --checkpoint checkpoints/cadstruct_graph_node_crop_gnn_h256_c32_ms3_l2_e24/model_best.pt \
  --dataset-dir datasets/cadstruct_graph_nodes_lie_topology_raster_v3 \
  --output reports/vlm/graph_node_crop_gnn_h256_c32_ms3_l2_e24_fine_calibration_audit.json \
  --dev-report reports/vlm/graph_node_crop_gnn_h256_c32_ms3_l2_e24_fine_calibrated_dev.json \
  --smoke-report reports/vlm/graph_node_crop_gnn_h256_c32_ms3_l2_e24_fine_calibrated_smoke.json \
  --dev-predictions-output reports/vlm/graph_node_crop_gnn_h256_c32_ms3_l2_e24_fine_calibrated_dev_predictions.jsonl \
  --smoke-predictions-output reports/vlm/graph_node_crop_gnn_h256_c32_ms3_l2_e24_fine_calibrated_smoke_predictions.jsonl \
  --class-bias-grid '1.25,1.3,1.35,1.4,1.5,1.65,1.8;0.45,0.55,0.65,0.7,0.75,0.8;0.4,0.5,0.55,0.6,0.65,0.7' \
  --batch-samples 64
```

The first h256 message-passing model reaches dev macro F1 `0.984812` with probability R² `0.975768`, and smoke macro F1 `0.983646` with probability R² `0.970320`. To clear the 99% accuracy target, train the wider h384 l2 variant:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vlm/train_graph_node_crop_gnn_classifier.py \
  --dataset-dir datasets/cadstruct_graph_nodes_lie_topology_raster_v3 \
  --output-dir checkpoints/cadstruct_graph_node_crop_gnn_h384_c32_ms3_l2_e24 \
  --crop-size 32 \
  --crop-pad-scales 0.15,0.35,0.8 \
  --hidden-dim 384 \
  --message-layers 2 \
  --epochs 24 \
  --batch-samples 64 \
  --learning-rate 7e-4 \
  --weight-decay 1e-4

CUDA_VISIBLE_DEVICES=0 python scripts/vlm/audit_graph_node_crop_calibration.py \
  --checkpoint checkpoints/cadstruct_graph_node_crop_gnn_h384_c32_ms3_l2_e24/model_best.pt \
  --dataset-dir datasets/cadstruct_graph_nodes_lie_topology_raster_v3 \
  --output reports/vlm/graph_node_crop_gnn_h384_c32_ms3_l2_e24_fine_calibration_audit.json \
  --dev-report reports/vlm/graph_node_crop_gnn_h384_c32_ms3_l2_e24_fine_calibrated_dev.json \
  --smoke-report reports/vlm/graph_node_crop_gnn_h384_c32_ms3_l2_e24_fine_calibrated_smoke.json \
  --dev-predictions-output reports/vlm/graph_node_crop_gnn_h384_c32_ms3_l2_e24_fine_calibrated_dev_predictions.jsonl \
  --smoke-predictions-output reports/vlm/graph_node_crop_gnn_h384_c32_ms3_l2_e24_fine_calibrated_smoke_predictions.jsonl \
  --class-bias-grid '1.0,1.2,1.4,1.6,1.8,2.0,2.25,2.5,2.75,3.0;0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7;0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.7' \
  --batch-samples 64
```

The h384 l2 model is now the selected quality path: dev accuracy `0.991147`, dev macro F1 `0.986746`, probability R² `0.978723`; smoke accuracy `0.991165`, smoke macro F1 `0.986652`, probability R² `0.978659`. Peak training memory was `2252.545` MiB. The residual errors are still wall/opening boundary cases; the next accuracy work should target source-balanced hard examples and boundary ambiguity, not only global hyperparameters.

Two crop-only follow-up ablations are recorded in `reports/vlm/graph_node_classifier_performance_audit.json`: c48 ms3 crop resolution reaches dev macro F1 `0.972076`, and c32 ms3 focal-gamma 1.0 reaches dev macro F1 `0.974457`. Neither replaces the c32-ms3 crop backbone; the selected full path is now c32-ms3 plus graph message passing.

For the older three-model ensemble:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/vlm/export_graph_node_ensemble_predictions.py \
  --checkpoints checkpoints/cadstruct_graph_node_classifier_lie_gated/model_best.pt,checkpoints/cadstruct_graph_node_classifier_lie_tr_gated/model_best.pt,checkpoints/cadstruct_graph_node_classifier_lie_tr_gated_rank8/model_best.pt \
  --weights 0.2,0.5,0.3 \
  --dataset datasets/cadstruct/smoke.jsonl \
  --output reports/vlm/graph_node_classifier_ensemble_weighted_smoke_candidates.jsonl
```

Use `reports/vlm/graph_node_classifier_performance_audit.json` for the current best single-model, ensemble, calibration, F1, and R² comparison.

Prepare and audit oracle object-group diagnostics:

```bash
python scripts/vlm/prepare_graph_object_dataset.py \
  --input-dir datasets/cadstruct \
  --output-dir datasets/cadstruct_graph_objects_oracle

CUDA_VISIBLE_DEVICES=0 python scripts/vlm/train_graph_node_classifier.py \
  --dataset-dir datasets/cadstruct_graph_objects_oracle \
  --record-key groups \
  --output-dir checkpoints/cadstruct_graph_object_oracle_classifier_v3_patch \
  --model-type gated \
  --experts 4 \
  --hidden-dim 256 \
  --epochs 40 \
  --batch-size 4096 \
  --eval-tile-size 4096 \
  --seed 20260429

CUDA_VISIBLE_DEVICES=0 python scripts/vlm/evaluate_graph_node_classifier.py \
  --checkpoint checkpoints/cadstruct_graph_object_oracle_classifier_v3_patch/model_best.pt \
  --dataset datasets/cadstruct_graph_objects_oracle/smoke.jsonl \
  --record-key groups \
  --output reports/vlm/graph_object_oracle_classifier_v3_patch_smoke.json \
  --predictions-output reports/vlm/graph_object_oracle_classifier_v3_patch_smoke_predictions.jsonl \
  --eval-tile-size 4096
```

Use `reports/vlm/graph_object_oracle_audit.json` for the object-group diagnostic. It is intentionally not an inference path: groups are same-label connected components built from ground-truth labels. The current finding is that scalar object/member aggregates are not enough, while adding local raster patch statistics lifts oracle-group smoke macro F1 to `0.913949`. The next model should replace these scalar patch statistics with a learned crop encoder and add graph message passing over primitive/object proposals.

Prepare and evaluate deployable topology/singleton object proposals:

```bash
python scripts/vlm/prepare_graph_object_dataset.py \
  --input-dir datasets/cadstruct \
  --output-dir datasets/cadstruct_graph_objects_topology_singleton_proposals \
  --grouping topology \
  --proposal-relations touches \
  --include-singleton-proposals

CUDA_VISIBLE_DEVICES=0 python scripts/vlm/train_graph_node_classifier.py \
  --dataset-dir datasets/cadstruct_graph_objects_topology_singleton_proposals \
  --record-key groups \
  --output-dir checkpoints/cadstruct_graph_object_topology_singleton_classifier \
  --model-type gated \
  --experts 4 \
  --hidden-dim 256 \
  --epochs 25 \
  --batch-size 4096 \
  --eval-tile-size 4096 \
  --seed 20260429

CUDA_VISIBLE_DEVICES=0 python scripts/vlm/evaluate_graph_node_classifier.py \
  --checkpoint checkpoints/cadstruct_graph_object_topology_singleton_classifier/model_best.pt \
  --dataset datasets/cadstruct_graph_objects_topology_singleton_proposals/smoke.jsonl \
  --record-key groups \
  --output reports/vlm/graph_object_topology_singleton_classifier_smoke.json \
  --predictions-output reports/vlm/graph_object_topology_singleton_classifier_smoke_predictions.jsonl \
  --eval-tile-size 4096
```

Use `reports/vlm/graph_object_proposal_audit.json` for the deployable proposal diagnostic. The current topology+singleton proposal classifier reaches smoke macro F1 `0.838425`; the gap to oracle group+patch `0.913949` points to proposal selection/NMS or a learned keep/suppress head as the next bottleneck.

Audit primitive-expanded proposal selection:

```bash
python scripts/vlm/audit_graph_object_proposal_selection.py \
  --dataset datasets/cadstruct_graph_objects_topology_singleton_proposals/smoke.jsonl \
  --predictions reports/vlm/graph_object_topology_singleton_classifier_smoke_predictions.jsonl \
  --output reports/vlm/graph_object_proposal_selection_audit.json \
  --selected-output reports/vlm/graph_object_topology_singleton_selected_predictions.jsonl
```

Use `reports/vlm/graph_object_proposal_selection_audit.json` to separate proposal classification from final primitive-expanded output. Raw confidence argmax collapses to macro F1 `0.385076` because large wall components swallow openings; member-count penalized selection recovers to `0.839073`, which shows that hand-tuned selection is not enough and a learned keep/suppress scorer is the right next step.

Prepare and train the first keep/suppress proposal-selection head:

```bash
python scripts/vlm/prepare_graph_object_selection_dataset.py \
  --input-dir datasets/cadstruct_graph_objects_topology_singleton_proposals \
  --output-dir datasets/cadstruct_graph_object_selection

CUDA_VISIBLE_DEVICES=0 python scripts/vlm/train_graph_node_classifier.py \
  --dataset-dir datasets/cadstruct_graph_object_selection \
  --record-key groups \
  --labels suppress,keep \
  --output-dir checkpoints/cadstruct_graph_object_selection_classifier \
  --model-type gated \
  --experts 3 \
  --hidden-dim 128 \
  --epochs 20 \
  --batch-size 4096 \
  --eval-tile-size 4096 \
  --seed 20260429

CUDA_VISIBLE_DEVICES=0 python scripts/vlm/evaluate_graph_node_classifier.py \
  --checkpoint checkpoints/cadstruct_graph_object_selection_classifier/model_best.pt \
  --dataset datasets/cadstruct_graph_object_selection/smoke.jsonl \
  --record-key groups \
  --output reports/vlm/graph_object_selection_classifier_smoke.json \
  --predictions-output reports/vlm/graph_object_selection_classifier_smoke_predictions.jsonl \
  --eval-tile-size 4096
```

The first binary selection head reaches smoke macro F1 `0.943441` on preferred-proposal labels. This is rule distillation with sparse suppress labels, so the next implementation should combine semantic logits and keep probability in one proposal scorer and evaluate final primitive-expanded semantic F1.

Audit joint semantic + keep/suppress proposal scoring:

```bash
python scripts/vlm/audit_graph_object_joint_proposal_scorer.py \
  --dataset datasets/cadstruct_graph_objects_topology_singleton_proposals/smoke.jsonl \
  --semantic-predictions reports/vlm/graph_object_topology_singleton_classifier_smoke_predictions.jsonl \
  --selection-predictions reports/vlm/graph_object_selection_classifier_smoke_predictions.jsonl \
  --output reports/vlm/graph_object_joint_proposal_scorer_audit.json \
  --predictions-output reports/vlm/graph_object_joint_proposal_scorer_predictions.jsonl
```

Current joint scorer result is a negative audit: semantic-only, keep-gated, and semantic*keep all stay at primitive-expanded macro F1 `0.839073`. The keep labels are too rule-like/sparse, so the next labels should come from final semantic errors and include hard negatives for false opening proposals.

Prepare and test error-driven keep/suppress labels:

```bash
python scripts/vlm/prepare_graph_object_error_selection_dataset.py \
  --proposal-dir datasets/cadstruct_graph_objects_topology_singleton_proposals \
  --prediction-dir reports/vlm \
  --prediction-prefix graph_object_topology_singleton_classifier \
  --output-dir datasets/cadstruct_graph_object_error_selection

CUDA_VISIBLE_DEVICES=0 python scripts/vlm/train_graph_node_classifier.py \
  --dataset-dir datasets/cadstruct_graph_object_error_selection \
  --record-key groups \
  --labels suppress,keep \
  --output-dir checkpoints/cadstruct_graph_object_error_selection_classifier \
  --model-type gated \
  --experts 3 \
  --hidden-dim 128 \
  --epochs 20 \
  --batch-size 4096 \
  --eval-tile-size 4096 \
  --seed 20260429

CUDA_VISIBLE_DEVICES=0 python scripts/vlm/evaluate_graph_node_classifier.py \
  --checkpoint checkpoints/cadstruct_graph_object_error_selection_classifier/model_best.pt \
  --dataset datasets/cadstruct_graph_object_error_selection/smoke.jsonl \
  --record-key groups \
  --output reports/vlm/graph_object_error_selection_classifier_smoke.json \
  --predictions-output reports/vlm/graph_object_error_selection_classifier_smoke_predictions.jsonl \
  --eval-tile-size 4096

python scripts/vlm/audit_graph_object_joint_proposal_scorer.py \
  --dataset datasets/cadstruct_graph_objects_topology_singleton_proposals/smoke.jsonl \
  --semantic-predictions reports/vlm/graph_object_topology_singleton_classifier_smoke_predictions.jsonl \
  --selection-predictions reports/vlm/graph_object_error_selection_classifier_smoke_predictions.jsonl \
  --output reports/vlm/graph_object_error_joint_proposal_scorer_audit.json \
  --predictions-output reports/vlm/graph_object_error_joint_proposal_scorer_predictions.jsonl
```

The error-driven labels add hard negatives, including false openings on wall proposals, but independent keep-gating over-suppresses true openings. Current result: hard keep-gate drops final macro F1 to `0.532801`, while soft keep scoring returns to `0.839073`. The next step should be a per-primitive conflict resolver or a joint semantic+selection scorer, not another independent post-hoc head.

Prepare and audit per-primitive conflict resolver rows:

```bash
python scripts/vlm/prepare_graph_object_conflict_dataset.py \
  --proposal-dir datasets/cadstruct_graph_objects_topology_singleton_proposals \
  --prediction-dir reports/vlm \
  --prediction-prefix graph_object_topology_singleton_classifier \
  --output-dir datasets/cadstruct_graph_object_conflicts

CUDA_VISIBLE_DEVICES=0 python scripts/vlm/train_graph_node_classifier.py \
  --dataset-dir datasets/cadstruct_graph_object_conflicts \
  --record-key groups \
  --labels suppress,keep \
  --output-dir checkpoints/cadstruct_graph_object_conflict_resolver \
  --model-type gated \
  --experts 3 \
  --hidden-dim 128 \
  --epochs 20 \
  --batch-size 8192 \
  --eval-tile-size 8192 \
  --seed 20260430

CUDA_VISIBLE_DEVICES=0 python scripts/vlm/evaluate_graph_node_classifier.py \
  --checkpoint checkpoints/cadstruct_graph_object_conflict_resolver/model_best.pt \
  --dataset datasets/cadstruct_graph_object_conflicts/smoke.jsonl \
  --record-key groups \
  --output reports/vlm/graph_object_conflict_resolver_smoke.json \
  --predictions-output reports/vlm/graph_object_conflict_resolver_smoke_predictions.jsonl \
  --eval-tile-size 8192

python scripts/vlm/audit_graph_object_conflict_resolver.py \
  --dataset datasets/cadstruct_graph_object_conflicts/smoke.jsonl \
  --predictions reports/vlm/graph_object_conflict_resolver_smoke_predictions.jsonl \
  --output reports/vlm/graph_object_conflict_resolver_audit.json \
  --selected-output reports/vlm/graph_object_conflict_resolver_selected_predictions.jsonl
```

Use `reports/vlm/graph_object_conflict_resolver_research_audit.json` for the current finding. Independent conflict classification reaches only final primitive-expanded macro F1 `0.792451`; a simple class-bias/member-penalty search tops out at `0.842869`. The next resolver should use pairwise candidate-delta features or graph message passing.

Audit zero-shot/base-model evidence and source-level structure-model generalization:

```bash
python scripts/vlm/audit_zero_shot_performance.py \
  --output reports/vlm/zero_shot_performance_audit.json
```

Run adapter-free base-model smoke/dev benchmarks with isolated server lifecycle:

```bash
python scripts/vlm/run_zero_shot_benchmark.py \
  --configs configs/vlm/internvl3_5_14b_eval.json \
  --dataset datasets/cadstruct/smoke.jsonl \
  --limit 4 \
  --output-dir reports/vlm/zero_shot_runs \
  --skip-existing
```

Use `reports/vlm/zero_shot_performance_audit.json` to keep zero-shot claims separate from trained structure-model claims. Current base-VLM reports are smoke-only (`total < 30`), so they are compatibility checks, not paper-grade zero-shot evidence. The local adapter-free InternVL3.5-14B smoke run at `reports/vlm/zero_shot_runs/internvl3_5_14b_hf_smoke_limit4.json` reaches only `semantic_exact_f1_mean=0.1275`, `relation_f1_mean=0.0`, `geometry_consistency_mean=0.54`, and has `3/4` partial JSON recoveries. For paper claims, run `evaluate_backend.py` or `run_zero_shot_benchmark.py` on `datasets/cadstruct/dev.jsonl` for each base model without LoRA/adapters and report semantic exact F1, geometry consistency, relation F1, empty semantic rate, and latency.

Profile SFT token and vision-tile budgets before training:

```bash
python scripts/vlm/profile_sft_budget.py \
  --limit 128 \
  --max-length 6144 \
  --max-image-side 512 \
  --max-vision-tiles 8 \
  --skip-at-max-length \
  --output reports/vlm/sft_budget_128.json
```

Download the 14B base weights outside git:

```bash
python scripts/vlm/download_model.py --model OpenGVLab/InternVL3_5-14B-HF --local-dir models/vlm/internvl3_5_14b_hf
```

Validate the LoRA smoke configuration:

```bash
python scripts/vlm/train_lora.py --dry-run
python scripts/vlm/train_lora.py --config configs/vlm/cadstruct_14b_lora.json --dry-run
```

Run a short auditable LoRA smoke step without saving a checkpoint:

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/vlm/train_lora.py \
  --config configs/vlm/cadstruct_14b_lora.json \
  --max-steps 1 \
  --limit 4 \
  --no-save \
  --audit-output reports/vlm/train_refactor_audit.jsonl
```

Training and budget profiling share `scripts/vlm/sft_utils.py`, so token counts, supervised-token counts, vision-tile counts, and budget skip reasons stay consistent across dry-runs, profiling, and training.

The SFT prompt and inference prompt use a semantic-first JSON contract:

```json
{"semantic_candidates":[],"scene_graph":{"nodes":[],"edges":[]},"symbol_candidates":[],"dimension_candidates":[],"warnings":[]}
```

The sidecar also attempts partial recovery when the model emits malformed JSON. If complete semantic candidate objects can be recovered from raw text, the response includes `partial_json_recovered` in `warnings`.

The output parser is isolated in `scripts/vlm/output_contract.py`; run its fast regression checks with:

```bash
python scripts/vlm/test_output_contract.py
```

Evaluation metrics are isolated in `scripts/vlm/eval_metrics.py`; run their fast regression checks with:

```bash
python scripts/vlm/test_eval_metrics.py
```

Model matrix:

| Role | Config | Notes |
|------|--------|-------|
| CI/mock | `configs/vlm/default.json` | deterministic, no model weights |
| Primary smoke | `configs/vlm/qwen3_vl_8b_smoke.json` | first real local VLM target |
| Primary paper | `configs/vlm/qwen3_vl_32b_paper.json` | preferred main result model on the 96GB GPU |
| Trainable 14B | `configs/vlm/internvl3_5_14b_eval.json` | CadStruct-VL 14B base/eval target |
| Trainable 14B LoRA | `configs/vlm/cadstruct_14b_lora.json` | first own-model training config |
| Baseline | `configs/vlm/internvl3_5_baseline.json` | strong open VLM comparison |
| Baseline | `configs/vlm/glm4_6v_baseline.json` | reasoning-oriented comparison |
| Efficiency baseline | `configs/vlm/kimi_vl_efficiency_baseline.json` | low active-parameter MoE comparison |

Keep model weights outside the Rust build; this sidecar exposes only the stable `/analyze_raster` JSON contract. If the sidecar fails or times out, Rust falls back to heuristic extraction.
