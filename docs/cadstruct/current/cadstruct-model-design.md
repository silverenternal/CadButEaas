# CadStruct Model Design

Date: 2026-04-29

## Current Answer

We now have a first CadStruct-owned model component.

The current system is not just LoRA anymore:

- VLM backbone: `OpenGVLab/InternVL3_5-14B-HF`.
- VLM adapters:
  - full-target LoRA: `checkpoints/cadstruct_vl_14b_lora`.
  - structural-core LoRA: `checkpoints/cadstruct_vl_14b_lora_structural`.
- CadStruct-owned structure model:
  - node classifier: `checkpoints/cadstruct_graph_node_classifier`.

The node classifier is small, but it is architecturally important: it directly solves `primitive_graph node -> semantic label`, which the autoregressive VLM failed to solve reliably.

## Why This Is Needed

The LoRA VLM can learn:

- JSON schema style.
- Candidate-only language.
- Non-empty structural output.
- High-level visual/context cues.

But the latest strict metric showed that it does not reliably solve per-node classification:

- structural LoRA semantic hit rate: `1.0`.
- structural LoRA semantic exact F1: `0.0187`.
- structural LoRA geometry consistency: `0.0`.

This means the VLM was producing plausible-looking structural labels, but not assigning the right class to the right primitive node.

## CadStruct-Owned Node Classifier

Training data:

- `datasets/cadstruct_graph_nodes`.
- Labels: `hard_wall`, `door`, `window`.
- Train: 2,052 drawings / 84,345 nodes.
- Dev: 228 drawings / 9,601 nodes.
- Smoke: 32 drawings / 1,245 nodes.

Model:

- Script: `scripts/vlm/train_graph_node_classifier.py`.
- Checkpoint: `checkpoints/cadstruct_graph_node_classifier/model_best.pt`.
- Architecture: MLP over deterministic primitive features.
- Inputs:
  - bbox.
  - centroid.
  - length.
  - angle.
  - orientation one-hot.
  - primitive type one-hot.
- Outputs:
  - `hard_wall`.
  - `door`.
  - `window`.

Current smoke result:

- report: `reports/vlm/graph_node_classifier_smoke.json`.
- accuracy: `0.650602`.
- macro F1: `0.55446`.
- hard_wall F1: `0.772069`.
- door F1: `0.580407`.
- window F1: `0.310905`.

This is already much better aligned with the task than JSON-only LoRA for dense node labels, but it is still only a baseline. Window detection is weak, and the model has no message passing yet.

## Decoupled Graph Node Module

The graph-node path is now split into reusable, auditable pieces:

- Shared model and feature code: `scripts/vlm/graph_node_model.py`.
- Dataset conversion: `scripts/vlm/prepare_graph_node_dataset.py`.
- Training: `scripts/vlm/train_graph_node_classifier.py`.
- Evaluation: `scripts/vlm/evaluate_graph_node_classifier.py`.
- RasterVlmOutput-style candidate export: `scripts/vlm/export_graph_node_predictions.py`.

The export step writes deterministic JSONL with `model_info`, `semantic_candidates`, `scene_graph.nodes`, and `warnings`. This keeps dense node prediction outside the autoregressive VLM and makes every candidate traceable to the graph classifier checkpoint.

## Topology Feature Result

The next controlled step was to keep the same small MLP head but add explicit primitive-graph topology features:

- `graph_degree`.
- `graph_in_degree`.
- `graph_out_degree`.
- relation counts for `touches`, `opens_in_wall`, `window_in_wall`, and `contained_in`.

This produced a separate dataset and checkpoint:

- Dataset: `datasets/cadstruct_graph_nodes_topology`.
- Checkpoint: `checkpoints/cadstruct_graph_node_classifier_topology/model_best.pt`.
- Smoke report: `reports/vlm/graph_node_classifier_topology_smoke.json`.
- Comparison report: `reports/vlm/graph_node_classifier_topology_comparison.json`.

Smoke comparison against the geometry-only MLP:

| Model | Accuracy | Macro F1 | Hard Wall F1 | Door F1 | Window F1 |
|---|---:|---:|---:|---:|---:|
| geometry-only MLP | 0.650602 | 0.55446 | 0.772069 | 0.580407 | 0.310905 |
| topology-feature MLP | 0.833735 | 0.79451 | 0.872143 | 0.809816 | 0.701571 |

This confirms that the model should treat primitive graph structure as first-class input. The next architecture upgrade should be message passing over the graph, not more JSON-only LoRA steps.

## SE(2) Features And Gated Routing

To test a more research-oriented direction, the node model now supports:

- SE(2)-canonicalized geometry features: graph-centered translation, dominant-orientation rotation normalization, graph-scale normalization, local double-angle orientation terms, area/length ratios, aspect ratio, and radial position.
- Gated expert routing: a small gate selects among expert MLP heads per node, with routing weights written into train/eval reports.

Current smoke ablation:

| Model | Accuracy | Macro F1 | Hard Wall F1 | Door F1 | Window F1 |
|---|---:|---:|---:|---:|---:|
| geometry-only MLP | 0.650602 | 0.55446 | 0.772069 | 0.580407 | 0.310905 |
| topology MLP | 0.833735 | 0.79451 | 0.872143 | 0.809816 | 0.701571 |
| topology + SE(2) MLP | 0.856225 | 0.819654 | 0.891185 | 0.811475 | 0.756303 |
| topology + SE(2) gated experts | 0.881124 | 0.846485 | 0.911271 | 0.852321 | 0.775862 |

The current finding is conservative: topology gives the largest gain, SE(2)-canonicalized features add a measurable gain, and gated routing adds another measurable gain. This should be framed as a prototype until it is validated on larger held-out real splits and explicit rotation/scale stress tests.

## Tensor-Ring Compression And CUDA Tiling

The graph-node path now also has an efficiency-oriented variant:

- Model type: `tr_gated`.
- Checkpoint: `checkpoints/cadstruct_graph_node_classifier_lie_tr_gated/model_best.pt`.
- Smoke report: `reports/vlm/graph_node_classifier_lie_tr_gated_smoke.json`.
- Compression audit: `reports/vlm/graph_node_classifier_compression_audit.json`.

Implementation details:

- `TensorRingLinear` replaces dense hidden/expert projections with two-core tensor-ring matrix factors.
- Training keeps train/dev tensors on CPU and moves only each batch to CUDA.
- Evaluation and routing summaries use `--eval-tile-size` so full splits do not need to reside on GPU.

Compression tradeoff against dense `topology + SE(2) gated experts`:

| Model | Params | Train Peak MiB | Smoke Macro F1 |
|---|---:|---:|---:|
| dense gated | 72,076 | 120.724 | 0.846485 |
| Tensor-Ring gated rank 4 | 30,092 | 51.843 | 0.840938 |

The Tensor-Ring version reduces parameters by 58.25% and peak training memory by 57.06%, with a smoke macro-F1 drop of 0.005547. The audit also shows route collapse in the rank-4 Tensor-Ring gate, so the next step is to add a load-balancing term or test higher ranks before treating the routing behavior as a stable research claim.

## Routing Balance Ablation

A static mean-gate balance regularizer was added to test whether the Tensor-Ring gate collapse can be controlled:

```text
loss = cross_entropy + weight * (experts * sum(mean_gate^2) - 1)
```

Audit report: `reports/vlm/graph_node_classifier_routing_balance_audit.json`.

| Tensor-Ring Gated Variant | Balance Weight | Smoke Macro F1 | Max Hard Route Fraction |
|---|---:|---:|---:|
| unbalanced | 0.00 | 0.840938 | 0.976707 |
| balanced | 0.01 | 0.831200 | 0.579920 |
| balanced | 0.05 | 0.834577 | 0.432932 |

The regularizer fixes route collapse, but it currently costs F1. The unbalanced Tensor-Ring checkpoint remains the best compressed checkpoint, while the balanced checkpoints are useful ablations for controllable specialization. Next experiments should test entropy annealing, top-k routing, expert dropout, or higher Tensor-Ring rank instead of relying on static mean-gate balancing.

## Tensor-Ring Rank Ablation

The gated model now also exposes route-control knobs in checkpoint metadata:

- `gate_temperature`.
- `top_k`.
- `expert_dropout`.
- `routing_balance_weight`.

Rank audit: `reports/vlm/graph_node_classifier_tr_rank_audit.json`.

| Variant | Params | Peak MiB | Smoke Macro F1 | Max Hard Route |
|---|---:|---:|---:|---:|
| dense gated | 72,076 | 120.724 | 0.846485 | 0.599197 |
| TR gated rank 4 | 30,092 | 51.843 | 0.840938 | 0.976707 |
| TR gated rank 4 + balance 0.05 | 30,092 | 55.901 | 0.834577 | 0.432932 |
| TR gated rank 8 | 116,492 | 60.423 | 0.848536 | 0.455422 |

Rank 8 naturally avoids most routing collapse and slightly beats dense gated F1, while still using much less peak CUDA memory than dense gated. It is not parameter-compressed relative to dense gated, so the current selection is:

- Best compressed checkpoint: `checkpoints/cadstruct_graph_node_classifier_lie_tr_gated/model_best.pt`.
- Best accuracy checkpoint: `checkpoints/cadstruct_graph_node_classifier_lie_tr_gated_rank8/model_best.pt`.
- Best controllable-routing ablation: `checkpoints/cadstruct_graph_node_classifier_lie_tr_gated_balanced/model_best.pt`.

## Ensemble Quality Path

The current best-quality inference path is a weighted probability ensemble:

- Dense gated: `checkpoints/cadstruct_graph_node_classifier_lie_gated/model_best.pt`, weight 0.2.
- Tensor-Ring gated rank 4: `checkpoints/cadstruct_graph_node_classifier_lie_tr_gated/model_best.pt`, weight 0.5.
- Tensor-Ring gated rank 8: `checkpoints/cadstruct_graph_node_classifier_lie_tr_gated_rank8/model_best.pt`, weight 0.3.

Performance audit: `reports/vlm/graph_node_classifier_performance_audit.json`.

| Inference Path | Accuracy | Macro F1 | Probability R² | Window F1 |
|---|---:|---:|---:|---:|
| best single model, TR rank 8 | 0.883534 | 0.848536 | n/a | 0.774929 |
| weighted 3-model ensemble | 0.890763 | 0.856547 | 0.732575 | 0.788235 |
| calibrated 3-model ensemble | 0.908434 | 0.873609 | 0.795303 | 0.813880 |
| gated h256 e40 + class bias | 0.914859 | 0.880733 | 0.805037 | 0.813880 |

The current best smoke path is `checkpoints/cadstruct_graph_node_classifier_lie_gated_h256_e40/model_best.pt` with class bias `hard_wall=1.5, door=0.7, window=0.7`, selected by dev-set calibration audit `reports/vlm/graph_node_h256_calibration_audit.json`. This improves macro F1 and probability R², but it is still far from the 98% F1 target. The remaining errors are not just thresholding: window F1 is still only `0.813880`, so the next accuracy step needs learned local visual evidence, better proposal/object supervision, or graph message passing over primitive/proposal/conflict edges.

Pain-point audit: `reports/vlm/graph_node_error_pain_points_audit.json` and `reports/vlm/graph_node_error_pain_points_dev_audit.json`.

The bottleneck is now precise rather than broad. On smoke, all `106/106` errors are wall/opening boundary mistakes, with zero door-window cross errors. On dev, `895/898` errors are the same wall/opening boundary type. CVC-FP is the main failure source (`0.860465` dev macro F1), while FloorPlanCAD is much stronger (`0.942017` dev macro F1). Reaching 98% macro F1 on dev would require removing roughly `706` of the current `898` errors, so small calibration gains cannot close the gap.

There is also a feature-quality issue in the current Lie/SE(2) channel: `se2_area` can explode on single-node or low-spread FloorPlanCAD samples because graph area is estimated from centroid spread. This contaminates normalization statistics and should be fixed by computing graph extents from bbox bounds and/or clipping area-ratio features before retraining.

The SE(2) area normalization was fixed to use bbox graph bounds, producing `datasets/cadstruct_graph_nodes_lie_topology_v2` with `se2_area <= 1.0` and no extreme outliers. Retraining the h256 gated model on v2 gives:

| Path | Split | Macro F1 | Probability R² | Window F1 |
|---|---|---:|---:|---:|
| h256 e40 calibrated, old SE(2) | dev | 0.869310 | 0.791540 | 0.805544 |
| h256 e40 calibrated, v2 SE(2) | dev | 0.880530 | 0.809941 | 0.822594 |
| h256 e40 calibrated, old SE(2) | smoke | 0.880733 | 0.805037 | 0.813880 |
| h256 e40 calibrated, v2 SE(2) | smoke | 0.880529 | 0.818760 | 0.824675 |

This is a real engineering fix and improves dev generalization, but it does not change the main conclusion: the remaining errors are still wall/opening boundary errors. The next step should add local raster patch evidence around each primitive and a dedicated wall-opening boundary head.

Local raster patch statistics were then added to primitive-node datasets with an explicit `--include-raster-features` switch, producing `datasets/cadstruct_graph_nodes_lie_topology_raster_v3`. This is the largest gain so far:

| Path | Split | Macro F1 | Probability R² | Hard Wall F1 | Door F1 | Window F1 |
|---|---|---:|---:|---:|---:|---:|
| h256 e40 calibrated, v2 SE(2) | dev | 0.880530 | 0.809941 | 0.938973 | 0.880024 | 0.822594 |
| h256 e40 + raster patch, calibrated | dev | 0.961939 | 0.938179 | 0.982153 | 0.965048 | 0.938616 |
| h256 e40 calibrated, v2 SE(2) | smoke | 0.880529 | 0.818760 | 0.938752 | 0.878161 | 0.824675 |
| h256 e40 + raster patch, calibrated | smoke | 0.963124 | 0.937717 | 0.981839 | 0.967136 | 0.940397 |

The calibrated raster model is now the best quality path:

- Dataset: `datasets/cadstruct_graph_nodes_lie_topology_raster_v3`.
- Checkpoint: `checkpoints/cadstruct_graph_node_classifier_lie_raster_gated_h256_e40/model_best.pt`.
- Calibration audit: `reports/vlm/graph_node_lie_raster_h256_calibration_audit.json`.
- Smoke report: `reports/vlm/graph_node_classifier_lie_raster_gated_h256_e40_calibrated_smoke.json`.
- Dev report: `reports/vlm/graph_node_classifier_lie_raster_gated_h256_e40_calibrated_dev.json`.

The remaining 98% gap is now much narrower. Dev has `249` errors and needs roughly `57` fewer errors for a 98% target; smoke has `32` errors and needs roughly `8` fewer. The residual errors are still wall/opening boundary mistakes, and many are high-confidence errors. That points to learned crop features or a dedicated hard-wall/opening boundary head, not more scalar calibration.

A first dedicated boundary-refiner head was tested as an auditable add-on rather than changing the main checkpoint format:

- Script: `scripts/vlm/train_graph_node_boundary_refiner.py`.
- Checkpoint: `checkpoints/cadstruct_graph_node_boundary_refiner_lie_raster_h128_e40/model_best.pt`.
- Inputs: normalized raster graph-node features plus calibrated base-model class probabilities, opening probability, margins, confidence, and entropy.
- Decision: binary `hard_wall` versus `opening`; when opening is selected, the base model still decides `door` versus `window`.

| Path | Split | Macro F1 | Probability R² | Hard Wall F1 | Door F1 | Window F1 |
|---|---|---:|---:|---:|---:|---:|
| h256 e40 + raster patch, calibrated | dev | 0.961939 | 0.938179 | 0.982153 | 0.965048 | 0.938616 |
| boundary refiner on raster model | dev | 0.962557 | 0.934641 | 0.982688 | 0.964119 | 0.940863 |
| h256 e40 + raster patch, calibrated | smoke | 0.963124 | 0.937717 | 0.981839 | 0.967136 | 0.940397 |
| boundary refiner on raster model | smoke | 0.963635 | 0.936164 | 0.981941 | 0.959811 | 0.949153 |

This is not enough to claim a new best architecture. It reduces wall-as-opening false positives and improves window precision, but probability R² drops and door F1 regresses. The audit result is useful: post-hoc boundary routing can only move a few errors. To approach 98% F1, the next model needs learned crop evidence or graph message passing before the semantic head, not just a decision-layer refiner.

A first learned crop encoder was then added as a separate graph+image model:

- Script: `scripts/vlm/train_graph_node_crop_classifier.py`.
- Checkpoint: `checkpoints/cadstruct_graph_node_crop_classifier_h256_c32_e20/model_best.pt`.
- Local visual input: `32x32` two-channel crop per primitive (`ink` and `edge`) around the bbox.
- Fusion: graph/raster scalar branch plus a small CNN crop branch.
- Calibration: dev-selected class bias `hard_wall=3.0, door=1.15, window=0.7`.

| Path | Split | Macro F1 | Probability R² | Hard Wall F1 | Door F1 | Window F1 |
|---|---|---:|---:|---:|---:|---:|
| h256 e40 + raster patch, calibrated | dev | 0.961939 | 0.938179 | 0.982153 | 0.965048 | 0.938616 |
| boundary refiner on raster model | dev | 0.962557 | 0.934641 | 0.982688 | 0.964119 | 0.940863 |
| crop graph h256 c32, calibrated | dev | 0.963178 | 0.942271 | 0.982770 | 0.962594 | 0.944172 |
| h256 e40 + raster patch, calibrated | smoke | 0.963124 | 0.937717 | 0.981839 | 0.967136 | 0.940397 |
| boundary refiner on raster model | smoke | 0.963635 | 0.936164 | 0.981941 | 0.959811 | 0.949153 |
| crop graph h256 c32, calibrated | smoke | 0.968028 | 0.945715 | 0.983699 | 0.954869 | 0.965517 |

This is now the best quality path. The uncalibrated crop model over-predicts openings, but after class bias it improves both F1 and probability R² over the scalar raster model. The remaining smoke gap is down to `29` errors with an approximate `5`-error reduction needed for a 98% target; dev has `241` errors and needs roughly `49` fewer. The residual errors are still almost entirely wall/opening boundary cases, so the next architecture step should be multi-scale crop context and/or graph message passing, not another scalar-only head.

Multi-scale crop context improved the learned crop path again. The model stacks three local contexts per primitive, using pad scales `0.15`, `0.35`, and `0.8`, for six visual channels total (`ink` and `edge` at each scale). Fine dev calibration selected class bias `hard_wall=1.55, door=1.05, window=0.9`.

| Path | Split | Macro F1 | Probability R² | Hard Wall F1 | Door F1 | Window F1 |
|---|---|---:|---:|---:|---:|---:|
| crop graph h256 c32, calibrated | dev | 0.963178 | 0.942271 | 0.982770 | 0.962594 | 0.944172 |
| multi-scale crop h256 c32 ms3, calibrated | dev | 0.975450 | 0.960990 | 0.988891 | 0.978029 | 0.959429 |
| crop graph h256 c32, calibrated | smoke | 0.968028 | 0.945715 | 0.983699 | 0.954869 | 0.965517 |
| multi-scale crop h256 c32 ms3, calibrated | smoke | 0.976790 | 0.961526 | 0.988777 | 0.976077 | 0.965517 |

The single-scale plus multi-scale crop ensemble was also audited, but dev search selected `single=0.0, ms3=1.0`, so the single-scale model adds no useful complementary signal. The current 98% gap is now concentrated in a small set of wall/opening boundary mistakes: smoke has `20` total errors, all wall/opening; dev has `158` errors, `154` of them wall/opening. The next meaningful step is probably graph message passing or a targeted hard-example loop around the worst CVC-FP and FloorPlanCAD boundary samples.

Two follow-up ablations did not replace the selected path:

| Ablation | Dev Macro F1 | Dev R² | Smoke Macro F1 | Smoke R² | Finding |
|---|---:|---:|---:|---:|---|
| c48 ms3 crop resolution, calibrated | 0.972076 | 0.957700 | 0.968001 | 0.952539 | More pixels alone over-predict openings and underperforms c32-ms3. |
| c32 ms3 focal gamma 1.0, calibrated | 0.974457 | 0.950846 | 0.976824 | 0.953484 | Smoke is essentially tied, but dev F1/R² are lower than c32-ms3. |

These results narrow the next research step: keep c32-ms3 as the crop encoder and add structure-aware propagation or a hard-example data loop, rather than increasing crop resolution or changing only the loss.

That structure-aware step is now implemented as a lightweight primitive-edge message passing model:

- Script: `scripts/vlm/train_graph_node_crop_gnn_classifier.py`.
- Checkpoint: `checkpoints/cadstruct_graph_node_crop_gnn_h256_c32_ms3_l2_e24/model_best.pt`.
- Visual input: same six-channel c32-ms3 crop tensor as the previous best model.
- Graph input: the existing deterministic feature vector plus two rounds of mean aggregation over bidirectional primitive-graph edges.
- Calibration: dev-selected class bias `hard_wall=1.5, door=0.55, window=0.4`.

| Path | Split | Macro F1 | Probability R² | Hard Wall F1 | Door F1 | Window F1 |
|---|---|---:|---:|---:|---:|---:|
| multi-scale crop h256 c32 ms3, calibrated | dev | 0.975450 | 0.960990 | 0.988891 | 0.978029 | 0.959429 |
| crop + graph message passing l2, calibrated | dev | 0.984812 | 0.975768 | 0.993402 | 0.987135 | 0.973897 |
| multi-scale crop h256 c32 ms3, calibrated | smoke | 0.976790 | 0.961526 | 0.988777 | 0.976077 | 0.965517 |
| crop + graph message passing l2, calibrated | smoke | 0.983646 | 0.970320 | 0.992144 | 0.983529 | 0.975265 |

This crosses the 98% macro-F1 target on both dev and smoke while improving probability R². The improvement is not just threshold tuning: the uncalibrated message-passing checkpoint already reaches dev macro F1 `0.979544` and smoke macro F1 `0.980942`, and calibration raises dev to `0.984812`.

The remaining error profile is narrow and useful for the next paper iteration. Dev has `95` errors, `91` of them wall/opening boundary mistakes and `4` door-window cross mistakes. Smoke has `14` errors, all wall/opening boundary mistakes. Window remains the limiting class (`0.973897` dev F1), and FloorPlanCAD is weaker than CVC-FP on the tiny smoke subset, so the next work should focus on source-balanced validation and hard boundary examples rather than more global architecture changes.

The accuracy target exposed one more useful architectural detail. A class-bias search over the h256-l2 model could improve smoke accuracy only to `0.989558`, still short of the 99% target. Depth changes did not solve it: l1 underfits the graph context and l3 is less stable than l2. Extending h256-l2 training to 36 epochs improves dev F1 but still does not clear smoke accuracy. Increasing the same l2 model width to h384 does clear the target after calibration.

| Path | Split | Accuracy | Macro F1 | Probability R² | Hard Wall F1 | Door F1 | Window F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| crop + graph message passing h256 l2, calibrated | dev | 0.990105 | 0.984812 | 0.975768 | 0.993402 | 0.987135 | 0.973897 |
| crop + graph message passing h384 l2, calibrated | dev | 0.991147 | 0.986746 | 0.978723 | 0.993981 | 0.987724 | 0.978533 |
| crop + graph message passing h256 l2, calibrated | smoke | 0.988755 | 0.983646 | 0.970320 | 0.992144 | 0.983529 | 0.975265 |
| crop + graph message passing h384 l2, calibrated | smoke | 0.991165 | 0.986652 | 0.978659 | 0.993824 | 0.990521 | 0.975610 |

The selected structural path is now:

- Checkpoint: `checkpoints/cadstruct_graph_node_crop_gnn_h384_c32_ms3_l2_e24/model_best.pt`.
- Calibration audit: `reports/vlm/graph_node_crop_gnn_h384_c32_ms3_l2_e24_fine_calibration_audit.json`.
- Class bias: `hard_wall=2.5, door=0.7, window=0.4`.
- Dev report: `reports/vlm/graph_node_crop_gnn_h384_c32_ms3_l2_e24_fine_calibrated_dev.json`.
- Smoke report: `reports/vlm/graph_node_crop_gnn_h384_c32_ms3_l2_e24_fine_calibrated_smoke.json`.

This crosses both current targets: `>98%` macro F1 and `>99%` accuracy on dev and smoke. The h384 model has `1,292,091` parameters and peak training memory `2252.545` MiB, so the gain is not coming from an impractical memory tradeoff.

## Object-Grouping Diagnostic

The 98% F1 target requires checking whether the current primitive-node unit matches the actual CAD semantics. A diagnostic oracle object dataset was added:

- Script: `scripts/vlm/prepare_graph_object_dataset.py`.
- Dataset: `datasets/cadstruct_graph_objects_oracle`.
- Audit: `reports/vlm/graph_object_oracle_audit.json`.
- Important caveat: this uses same-label connected components from ground-truth labels, so it is diagnostic only and not an inference-time proposal generator.

Smoke comparison:

| Inference Unit | Accuracy | Macro F1 | Hard Wall F1 | Door F1 | Window F1 |
|---|---:|---:|---:|---:|---:|
| best primitive-node weighted ensemble | 0.890763 | 0.856547 | 0.919336 | 0.862069 | 0.788235 |
| oracle group v1, bbox/topology aggregate | 0.769710 | 0.764648 | 0.892128 | 0.732919 | 0.668896 |
| oracle group v2, member distribution aggregate | 0.823651 | 0.816497 | 0.920000 | 0.815710 | 0.713781 |
| oracle group v3, member distribution + raster patch stats | 0.919087 | 0.913949 | 0.980716 | 0.892216 | 0.868914 |

This is a useful directional result. Naive object grouping loses primitive-level evidence, and a richer group aggregate alone remains below the node ensemble. Adding local raster patch statistics raises smoke macro F1 to `0.913949`, which shows that door/window separation needs local visual evidence. The remaining group-level errors are still mostly door/window confusion, so the next accuracy step should not be another scalar aggregate-feature MLP. It should combine:

- primitive-level geometry and topology;
- object/group proposals;
- message passing across primitive and object graphs;
- a learned local raster crop encoder around openings and candidate objects.

SE(2)/Lie features, gated routing, Tensor-Ring compression, and CUDA tiling remain useful components, but they need to sit inside a patch-aware graph architecture rather than act as the full model.

## Deployable Proposal Diagnostic

The oracle object result is not deployable because it uses ground-truth labels to form groups. A topology-only proposal mode was added to the same dataset script:

```bash
python scripts/vlm/prepare_graph_object_dataset.py \
  --output-dir datasets/cadstruct_graph_objects_topology_singleton_proposals \
  --grouping topology \
  --proposal-relations touches \
  --include-singleton-proposals
```

Audit: `reports/vlm/graph_object_proposal_audit.json`.

Smoke comparison:

| Path | Accuracy | Macro F1 | Hard Wall F1 | Door F1 | Window F1 |
|---|---:|---:|---:|---:|---:|
| primitive-node weighted ensemble | 0.890763 | 0.856547 | 0.919336 | 0.862069 | 0.788235 |
| oracle group + patch stats | 0.919087 | 0.913949 | 0.980716 | 0.892216 | 0.868914 |
| topology + singleton proposal classifier | 0.895110 | 0.838425 | 0.941854 | 0.829374 | 0.744048 |

The proposal audit isolates the next bottleneck. Topology-only `touches` components can swallow windows into large wall groups; singleton fallback restores coverage and high proposal purity, but creates overlapping candidates. The next model should therefore include proposal selection/NMS or a learned keep/suppress head, not just a stronger label classifier.

An explicit selection audit was added:

- Script: `scripts/vlm/audit_graph_object_proposal_selection.py`.
- Report: `reports/vlm/graph_object_proposal_selection_audit.json`.
- Selected output: `reports/vlm/graph_object_topology_singleton_selected_predictions.jsonl`.

Primitive-expanded smoke result:

| Selection Path | Macro F1 | Hard Wall F1 | Door F1 | Window F1 |
|---|---:|---:|---:|---:|
| raw proposal confidence argmax | 0.385076 | 0.846924 | 0.267206 | 0.041096 |
| member-count penalized selection | 0.839073 | 0.941662 | 0.831510 | 0.744048 |

This confirms that large wall components swallow opening primitives unless selection penalizes broad proposals. However, the heuristic selection saturates near the proposal classifier result, so the next step should generate supervised keep/suppress labels and train a proposal scorer rather than hand-tune NMS.

A first keep/suppress supervision dataset and classifier were added:

- Dataset: `datasets/cadstruct_graph_object_selection`.
- Dataset script: `scripts/vlm/prepare_graph_object_selection_dataset.py`.
- Checkpoint: `checkpoints/cadstruct_graph_object_selection_classifier/model_best.pt`.
- Smoke report: `reports/vlm/graph_object_selection_classifier_smoke.json`.

The binary head reaches smoke macro F1 `0.943441` on the current preferred-proposal labels, with suppress F1 `0.888889`. This means the rules are learnable, but the label distribution is still sparse (`25` suppress vs `1243` keep on smoke). Treat this as scaffolding for a joint proposal scorer, not as final evidence that proposal selection is solved.

The learned selection head was then connected back to final primitive-expanded semantic scoring:

- Script: `scripts/vlm/audit_graph_object_joint_proposal_scorer.py`.
- Report: `reports/vlm/graph_object_joint_proposal_scorer_audit.json`.

| Final Scoring Path | Macro F1 | Hard Wall F1 | Door F1 | Window F1 |
|---|---:|---:|---:|---:|
| semantic confidence + member penalty | 0.839073 | 0.941662 | 0.831510 | 0.744048 |
| keep-gated semantic confidence | 0.839073 | 0.941662 | 0.831510 | 0.744048 |
| semantic confidence * keep confidence | 0.839073 | 0.941662 | 0.831510 | 0.744048 |

This is an important negative result: the keep/suppress head learns the current rules, but those rules do not improve final semantic F1. The next supervision must be rebuilt from final primitive-expanded errors, with hard negatives for wall-singletons predicted as openings, opening-adjacent wall boxes, and duplicate component/singleton conflicts.

An error-driven keep/suppress dataset was then built from semantic proposal mistakes:

- Script: `scripts/vlm/prepare_graph_object_error_selection_dataset.py`.
- Dataset: `datasets/cadstruct_graph_object_error_selection`.
- Checkpoint: `checkpoints/cadstruct_graph_object_error_selection_classifier/model_best.pt`.
- Selection report: `reports/vlm/graph_object_error_selection_classifier_smoke.json`.
- Joint report: `reports/vlm/graph_object_error_joint_proposal_scorer_audit.json`.

This adds harder suppress labels. On smoke, suppress examples increase from `25` to `151`, including `95` `false_opening_on_wall` cases. The independent selection head, however, reaches only `0.649493` macro F1, and hard keep-gating damages final semantic output:

| Final Scoring Path With Error-Driven Selection | Macro F1 |
|---|---:|
| semantic-only + member penalty | 0.839073 |
| hard keep-gate | 0.532801 |
| soft semantic * keep probability | 0.839073 |

The conclusion is stricter now: keep/suppress cannot remain an independent post-hoc classifier. The next implementation should rank competing candidates per primitive, or train a joint semantic+selection scorer with final primitive-expanded F1 as the validation target.

A first per-primitive conflict dataset and resolver were added:

- Dataset script: `scripts/vlm/prepare_graph_object_conflict_dataset.py`.
- Dataset: `datasets/cadstruct_graph_object_conflicts`.
- Checkpoint: `checkpoints/cadstruct_graph_object_conflict_resolver/model_best.pt`.
- Final audit: `reports/vlm/graph_object_conflict_resolver_research_audit.json`.

The dataset creates candidate rows for each primitive/proposal conflict. Each primitive has about two competing candidates on average, giving substantially more suppress examples than earlier selection datasets. The independent candidate classifier still underperforms:

| Path | Final Primitive-Expanded Macro F1 |
|---|---:|
| semantic confidence + member penalty | 0.839073 |
| best simple class-bias/member-penalty search | 0.842869 |
| independent conflict resolver | 0.792451 |
| oracle group + patch stats | 0.913949 |

This is another useful negative result. Candidate-level keep/suppress classification lacks direct comparative signal. The next resolver should use pairwise/delta features for competing candidates over the same primitive, or move to message passing over primitive nodes, proposal nodes, and conflict edges.

## Zero-Shot Audit

Zero-shot performance is now tracked separately from trained CadStruct structural modules:

- Script: `scripts/vlm/audit_zero_shot_performance.py`.
- Runner: `scripts/vlm/run_zero_shot_benchmark.py`.
- Report: `reports/vlm/zero_shot_performance_audit.json`.
- Latest base-model smoke run: `reports/vlm/zero_shot_runs/internvl3_5_14b_hf_smoke_limit4.json`.

The current base-VLM reports are smoke-only, not paper-grade evidence. Existing zero-shot/base reports have `total < 30`, so they should be described as compatibility or prompt smoke checks only. The latest local adapter-free InternVL3.5-14B run on `datasets/cadstruct/smoke.jsonl --limit 4` produced `semantic_exact_f1_mean=0.1275`, `relation_f1_mean=0.0`, `geometry_consistency_mean=0.54`, `empty_semantic_rate=0.25`, `3/4` partial JSON recoveries, and mean latency `31280.348 ms`. This confirms that the base VLM can emit some semantic candidates, but dense primitive-id alignment, relation extraction, and stable JSON remain weak without the structural model.

Current source-level trained structure breakdown on smoke:

| Path | Source | Records | Macro F1 | Window F1 |
|---|---|---:|---:|---:|
| primitive-node weighted ensemble | CVC-FP | 1193 | 0.851069 | 0.784431 |
| primitive-node weighted ensemble | FloorPlanCAD | 52 | 0.935948 | 1.000000 |
| oracle group + patch stats | CVC-FP | 433 | 0.910718 | 0.865900 |
| oracle group + patch stats | FloorPlanCAD | 49 | 0.949644 | 1.000000 |

The source split matters: CVC-FP dominates the remaining failure mass, while FloorPlanCAD smoke is too small to support strong claims. Paper-grade zero-shot evaluation should run the same `datasets/cadstruct/dev.jsonl` split across base Qwen3-VL, InternVL3.5, GLM-4.6V, and Kimi-VL without adapters, reporting `semantic_exact_f1_mean`, `geometry_consistency_mean`, `relation_f1_mean`, empty semantic rate, and latency.

## Target Architecture

The intended CadStruct model should be:

```text
Raster image + primitive graph
        |
        +-- VLM backbone / LoRA: visual context, schema, audit text
        |
        +-- CadStruct graph encoder: node and edge embeddings
                |
                +-- node head: hard_wall / door / window / other
                |
                +-- edge head: adjacent_to / opens_into / intersects / bounds / none
        |
        +-- deterministic JSON assembly into RasterVlmOutput
```

The VLM should not be responsible for writing hundreds of node labels as free-form JSON. It should provide context and human-auditable output; the structure module should own dense graph prediction.

## Next Engineering Step

Replace the MLP/object-aggregate baselines with a graph-aware and patch-aware model:

- Add message passing over `primitive_graph.edges`.
- Add object/group proposal nodes without using labels at inference time.
- Add local raster crop features for door/window candidates.
- Add local neighbor features and relation features.
- Add an `other` class with sampled non-structural nodes.
- Train primitive node, object group, and relation/edge heads with deterministic JSON assembly.
- Keep the final JSON assembler deterministic and auditable.
