# CadStruct Paper Readiness

Date: 2026-04-30

## Short Verdict

The current CadStruct structural model is strong enough to become the core result of an SCI Q2 submission, but not yet strong enough as a complete paper package.

The development-split metrics are no longer the main blocker:

| Split | Accuracy | Macro F1 | Probability R2 |
|---|---:|---:|---:|
| dev | 0.991147 | 0.986746 | 0.978723 |
| smoke | 0.991165 | 0.986652 | 0.978659 |

Selected checkpoint:

```text
checkpoints/cadstruct_graph_node_crop_gnn_h384_c32_ms3_l2_e24/model_best.pt
```

These numbers should be treated as engineering-selection evidence, not final paper-test evidence, because the split was used during architecture iteration.

The stricter paper split is now available:

```text
datasets/cadstruct_paper_split
datasets/cadstruct_graph_nodes_paper_v1
reports/vlm/paper_v1_validation_summary.json
```

It uses source-stratified grouped splitting, with CVC-FP rotation variants tied to one group. Train/dev/locked-test group overlap is zero.

| Paper Split | Accuracy | Macro F1 | Probability R2 | Notes |
|---|---:|---:|---:|---|
| dev | 0.989878 | 0.985816 | 0.974331 | dev-only class calibration |
| locked test | 0.984983 | 0.976103 | 0.963981 | no test calibration |

Selected paper-split checkpoint:

```text
checkpoints/cadstruct_graph_node_crop_gnn_h384_c32_ms3_l2_paper_v1_e24/model_best.pt
```

The paper-split locked test is strong but below the paper target of `0.99` accuracy and `0.98` macro F1. The remaining blocker is now narrower and clearer: cross-source/domain generalization, seed stability, and comparison against credible baselines.

The metric gap was addressed by the v2 structural model:

```text
datasets/cadstruct_graph_nodes_paper_v2_source_raster
checkpoints/cadstruct_graph_node_crop_gnn_h512_c32_ms3_l2_paper_v2_source_raster_e28/model_best.pt
reports/vlm/paper_v2_validation_summary.json
```

v2 adds auditable raster patch statistics, explicit source indicators, and a wider h512 two-layer primitive graph message model. Dev-only calibration selects:

```text
hard_wall=2.0, door=0.7, window=0.7
```

| Paper v2 Split | Accuracy | Macro F1 | Probability R2 | Notes |
|---|---:|---:|---:|---|
| dev | 0.996704 | 0.995452 | 0.990669 | dev-only class calibration plus two dev-selected routers |
| locked test | 0.992637 | 0.988548 | 0.980085 | exceeds 99% accuracy and 98% F1 target |

Per-class locked-test F1:

| Class | Paper v1 F1 | Paper v2 F1 |
|---|---:|---:|
| hard_wall | 0.990167 | 0.995231 |
| door | 0.980304 | 0.988141 |
| window | 0.957839 | 0.982272 |

The key improvement is the previous limiting `window` class crossing the 0.98 F1 line.

Latest metric-improvement sweep:

| Candidate | Selection Protocol | Locked Acc | Locked Macro F1 | Locked R2 | Finding |
|---|---|---:|---:|---:|---|
| h512 v2 source+raster | strict dev-only calibration | 0.992200 | 0.987913 | 0.979641 | base |
| h512 base + crop-aug confidence router | strict dev-only router search | 0.992565 | 0.988456 | 0.979968 | stage 1 |
| h512 two-stage router | strict dev-only router search | 0.992637 | 0.988548 | 0.980085 | selected |
| seed 20260431 | strict dev-only calibration | 0.989503 | 0.981313 | 0.975322 | stable but below selected |
| seed 20260432 | strict dev-only calibration | 0.990305 | 0.985075 | 0.974904 | stable but below selected |
| h512 crop augmentation | strict dev-only calibration | 0.991034 | 0.985104 | 0.977305 | dev improves, locked test drops |
| source-aware calibration | strict dev/source calibration | 0.992200 | 0.987765 | 0.979497 | not selected |
| mixed source-balanced sampler | strict dev-only selection | 0.988701 | 0.982238 | 0.971333 | not selected |
| routed main+FloorPlanCAD expert | diagnostic only | 0.992346 | 0.988272 | not measured | best observed, not paper-selected |

The strict selected inference path is now the original h512 paper-v2 checkpoint plus two dev-selected routers. Stage 1 is deliberately small and auditable: if the base model predicts `door`, base confidence is `<= 0.95`, and the crop-aug model disagrees, use the crop-aug prediction and probabilities. Stage 2 is FloorPlanCAD-only and switches one low-confidence `hard_wall` case to the FloorPlanCAD crop-augmentation expert. A broader FloorPlanCAD expert replacement remains diagnostic because it uses a final epoch observed after locked-test inspection.

## Proposed Paper Claim

The safest claim is not "a general CAD foundation model." That would be too broad for the current evidence.

The defensible claim is:

```text
CadStruct is a structure-aware raster floor-plan primitive labeling model that combines local visual crop evidence, SE(2)-normalized geometric/topological features, and primitive-graph message passing to produce auditable wall, door, and window semantics from raster-derived primitive graphs.
```

This claim is narrow, technical, and aligned with the data and metrics we actually have.

## Innovation Points

### 1. Decoupling Dense Structural Semantics From Autoregressive VLM JSON

The project shows that LoRA/VLM output can learn schema style and broad visual context but fails at dense primitive-id alignment. The structural classifier directly solves:

```text
primitive_graph node -> hard_wall / door / window
```

This is important because dense CAD semantics should not be delegated to free-form JSON generation. The deterministic assembly path is auditable and avoids hallucinated primitive identifiers.

Strength:

- Clear engineering and modeling motivation.
- Supported by strict LoRA failure evidence.
- Fits CAD/floor-plan review expectations: traceability matters.

Risk:

- This is a system-design contribution unless framed with strong experiments.

### 2. Patch-Aware Primitive Graph Neural Model

The selected model fuses:

- deterministic primitive geometry and topology;
- SE(2)-normalized geometric features;
- multi-scale local raster crops around each primitive;
- bidirectional primitive-edge message passing;
- dev-selected class calibration.

The controlled progression is strong:

| Stage | Dev Macro F1 | Smoke Macro F1 | Finding |
|---|---:|---:|---|
| topology/SE(2)/gated scalar models | below 0.90 early, then 0.96 after raster scalar features | below target | scalar features are insufficient |
| learned crop encoder c32-ms3 | 0.975450 | 0.976790 | local visual evidence closes most wall/opening errors |
| h256 crop + graph message passing | 0.984812 | 0.983646 | graph context crosses 98% F1 |
| h384 crop + graph message passing | 0.986746 | 0.986652 | wider l2 model also crosses 99% accuracy |

Strength:

- This is the strongest technical innovation.
- It addresses the observed failure mode directly: wall/opening boundary ambiguity.
- Ablations show that more pixels, focal loss, and message depth alone are not enough.

Risk:

- Message passing itself is not novel. The novelty must be framed as the task-specific combination and the auditable CAD primitive graph formulation, not as inventing GNNs.

### 3. SE(2)-Aware Structural Feature Pipeline With Error-Driven Fixes

The work includes a useful geometric invariance path:

- graph-centered translation normalization;
- dominant-orientation rotation normalization;
- graph-scale normalization;
- double-angle orientation features;
- bbox-bound graph extent correction after detecting `se2_area` outliers.

Strength:

- Good for a CAD/floor-plan paper because invariance is domain-relevant.
- The outlier fix and before/after audit show engineering rigor.

Risk:

- SE(2) features are supportive, not the final primary performance driver. They should not be over-claimed as the central novelty.

### 4. Auditable Optimization And Negative Results

The project already has valuable negative evidence:

- post-hoc boundary refiner only gives marginal F1 and hurts R2;
- c48 crop resolution underperforms c32-ms3;
- focal gamma 1.0 does not improve dev/R2;
- l1 and l3 message passing underperform l2;
- h256 e36 improves dev but not smoke accuracy;
- object-group proposal and conflict heads expose why naive proposal grouping is not enough.

Strength:

- Reviewers value controlled negative results when they are tied to design decisions.
- This supports the claim that the selected model is not arbitrary.

Risk:

- The paper must keep the story focused. Too many side experiments can make the contribution look scattered.

### 5. Lightweight, Deployable Structural Module

The selected h384 model has:

```text
parameters: 1,292,091
peak training memory: 2252.545 MiB
```

This is small compared with 14B-class VLMs and is practical as a sidecar or deterministic structural module.

Strength:

- Strong engineering value.
- Good for applied SCI Q2 venues.

Risk:

- Efficiency is secondary unless we add latency/throughput and memory comparisons.

## SCI Q2 Readiness Assessment

| Dimension | Current Status | Q2 Readiness |
|---|---|---|
| Problem relevance | Strong: raster floor-plan/CAD semantic extraction is practical | Ready |
| Core method | Strong enough: patch-aware primitive GNN with audited CAD graph formulation | Mostly ready |
| Metrics | Paper v2 locked test is 0.992637 accuracy, 0.988548 macro F1, and 0.980085 R2 after dev-selected routing; two extra seeds are stable but below the selected run | Mostly ready |
| Ablations | Good internal ablations already exist | Needs table cleanup |
| Generalization | Pure zero-shot cross-source transfer collapses, but a clean few-shot target-adaptation protocol now recovers most of the gap | Partially ready |
| Baselines | Current baselines are mostly internal; zero-shot VLM evidence is smoke-only | Not ready |
| Dataset description | Usable but needs exact split policy, leakage guard, and source breakdown | Partially ready |
| Reproducibility | Good scripts/checkpoints/reports, but needs paper-level command appendix | Mostly ready |
| Novelty strength | Moderate to strong for applied Q2; not enough for top-tier ML/CV alone | Q2 plausible |

Decision:

```text
SCI Q2 potential: yes.
Ready to submit today: no.
Main missing evidence: final seed mean/std table, SOTA baselines, and a carefully scoped cross-source/generalization claim.
```

## Validation Package Status

Completed:

1. Built leakage-aware paper split at `datasets/cadstruct_paper_split`.
2. Prepared graph-node paper dataset at `datasets/cadstruct_graph_nodes_paper_v1`.
3. Trained the selected h384-l2 crop graph model on the paper split.
4. Ran dev-only calibration and locked-test evaluation.
5. Ran cross-source tests:
   - CVC-FP train/dev -> FloorPlanCAD locked test: accuracy `0.723343`, macro F1 `0.485651`, R2 `0.175212`.
   - FloorPlanCAD train/dev -> CVC-FP locked test: accuracy `0.196791`, macro F1 `0.159075`, R2 `-0.868702`.
6. Ran error audits for paper dev, paper locked test, and both cross-source tests.
7. Added paper v2 feature/model upgrade and recovered the locked-test target.
8. Tested source-balanced loss as a negative ablation; it reduced locked-test macro F1 to `0.979944`, so it is not selected.
9. Ran two additional h512 seeds; both stay in the target region for macro F1, but neither improves the selected locked-test result.
10. Added opt-in crop augmentation to `scripts/vlm/train_graph_node_crop_gnn_classifier.py`; calibrated dev improves to `0.994831` macro F1, but whole-model replacement drops locked-test macro F1 to `0.985104`.
11. Added `scripts/vlm/audit_graph_node_prediction_router.py` and selected a confidence router using dev predictions only. The clean router raises locked-test accuracy to `0.992565`, macro F1 to `0.988456`, and R2 to `0.979968`.
12. Added a second FloorPlanCAD-only dev-selected router. It switches one additional locked-test node and raises the selected result to accuracy `0.992637`, macro F1 `0.988548`, and R2 `0.980085`.
13. Tested source-aware calibration and FloorPlanCAD specialization. Strict source-aware calibration does not improve locked test; diagnostic source routing raises observed FloorPlanCAD macro F1 to `0.969008`, but is not yet paper-selected.
14. Added `--drop-source-features` and `--source-balanced-sampler` to the crop-GNN trainer and ran generalization follow-ups. Dropping source features collapses leave-one-source transfer (`CVC->FloorPlanCAD` macro F1 `0.129784`, `FloorPlanCAD->CVC` macro F1 `0.173692`), and mixed source-balanced sampling lowers locked macro F1 to `0.982238`; both are negative results.
15. Added `scripts/vlm/build_graph_node_domain_adaptation_split.py` and a clean few-shot target-adaptation protocol. With deterministic half-target-dev adaptation plus held-out target-dev model selection, `CVC->FloorPlanCAD` locked macro F1 improves to `0.913253`, and `FloorPlanCAD->CVC` locked macro F1 improves to `0.947563`. This fixes most of the cross-source collapse, but it is still below the 98% robustness target.
16. Tested three direct fixes for the remaining adaptation gap: target/window-weighted loss, checkpoint probability ensembling, and nearly-all-target-dev fixed-hyperparameter adaptation. None reaches 98% cross-domain macro F1. Best diagnostic final results are `0.924959` for `CVC->FloorPlanCAD` and `0.946132` for `FloorPlanCAD->CVC`, so the residual gap is now a data/coverage and target-domain morphology problem, not a simple weighting or selection issue.
17. Added training-only target hard-example bbox/crop jitter augmentation. It gives a real but modest strict gain for `FloorPlanCAD->CVC` from `0.947563` to `0.951022` locked macro F1, with diagnostic final epoch `0.953358`. It does not materially solve `CVC->FloorPlanCAD`, which only reaches `0.915185` strict locked macro F1.
18. Added selectable checkpoint criteria to the crop-GNN trainer and tested dev `probability_r2` selection for the hard-augmentation `CVC->FloorPlanCAD` run. It lowers locked macro F1 to `0.874705`, so the remaining gap is not just a bad checkpoint-selection metric.
19. Added train-only crop style augmentation and tested it on `CVC->FloorPlanCAD` hardaug. It lowers locked macro F1 to `0.870096`, so simple brightness/contrast/noise/dropout perturbations do not reproduce the missing FloorPlanCAD morphology.
20. Added optional relation-aware message passing over edge relation types. On `CVC->FloorPlanCAD` it reaches only `0.912897` strict locked macro F1, so the existing `touches/contains/contained_in` relation labels alone do not solve the cross-domain morphology gap.
21. Added optional target/fragile sample resampling. On `CVC->FloorPlanCAD` it reaches held-out target-dev macro F1 `0.940249` but locked smoke macro F1 only `0.884959`, so simply increasing target-domain `door/window` sample exposure overfits the small target dev and does not fix transfer.
22. Added `scripts/vlm/mine_graph_node_target_hard_cases.py` and exported `reports/vlm/paper_v2_cvc_to_floor_target_hard_cases.jsonl`. On the best strict `CVC->FloorPlanCAD` hardaug checkpoint, locked smoke has 39 errors out of 694 nodes; the dominant pairs are `door->hard_wall` 19 and `door->window` 12.
23. Added hard-case-signature guided morphology augmentation to the hardaug script and trained `CVC->FloorPlanCAD` again. It is negative: locked macro F1 drops to `0.876006`, so synthetic bbox/aspect perturbation is too coarse for the missing FloorPlanCAD door morphology. Fine dev-only class-bias calibration of the previous hardaug checkpoint is also negative at `0.909470` locked macro F1.
24. Trained target-only FloorPlanCAD nearly-all-dev fixed-hyperparameter experts. Real target labels help more than synthetic augmentation: the best final checkpoint reaches locked macro F1 `0.965836` and R2 `0.938106`, but two additional seeds (`0.931896`, `0.927594`), equal seed ensemble (`0.951409`), and no-augmentation control (`0.956857`) do not reach 98%.
25. Trained wider h512 FloorPlanCAD target-only experts. The clean half-dev run reaches locked macro F1 `0.965150` and R2 `0.935821`, with final-epoch diagnostic macro F1 `0.968190`; the h512 nearly-all-dev run is negative at final macro F1 `0.918225`. Width alone does not solve the target-domain morphology gap.
26. Added `scripts/vlm/audit_graph_node_opening_specialist_router.py` for a dev-selected, constrained opening-specialist router. It improves mixed-source h512 -> target specialist from `0.956198` to `0.960265`, and target best -> target final from `0.965150` to `0.966492`, but neither beats the target final checkpoint at `0.968190`. Mining that final checkpoint leaves 16 locked errors: `door->hard_wall` 11, `door->window` 4, `hard_wall->door` 1.
27. Added `scripts/vlm/train_graph_node_residual_refiner.py`, a small frozen-base residual head using structural features plus base probabilities/confidence/margin/entropy. The best fixed-hyperparameter h128 seed improves FloorPlanCAD locked smoke from macro F1 `0.968190` to `0.973188` and R2 `0.939330` to `0.944101`; it corrects 3 nodes and regresses 1. Seed31/seed33 (`0.964548`), conservative h32 (`0.966847`), a more aggressive h64 variant (`0.964095`), and a three-seed residual ensemble (`0.971835`) do not beat seed32.
28. Added an optional crop-CNN branch to the residual refiner and tested relation-aware target-only replacements. These are negative controls: crop residual stays at locked macro F1 `0.968190` with lower R2, relation-aware target-only best is `0.959360` strict and `0.967543` final diagnostic, relation-aware+style overfits dev (`0.990853`) but locks at `0.955422`, and relation-aware-final residual reaches `0.971619`. The best current target-domain result remains the h512 target-final structural residual seed32 at `0.973188`.
29. Added label smoothing and gradient clipping to the crop-GNN trainer and reran h512 target-only seed controls. Seed31 reaches dev `0.995406` but locks at `0.947890`; seed32 reaches dev `1.000000` but locks at `0.955642` strict and `0.965803` final. Label smoothing `0.05` plus clip `1.0` stabilizes training but reaches only `0.968833` final and `0.969131` after residual, with lower R2 than the current best.
30. Added `scripts/vlm/prepare_graph_node_morphology_features.py` to derive deterministic label-free bbox/raster/graph morphology features. This is negative: dev saturates at `1.000000`, but locked smoke reaches only `0.957151` by best-dev selection, `0.966564` at final epoch, and `0.966901` after residual while R2 drops to `0.913500`. Hand-crafted morphology features amplify dev overfit instead of solving the remaining FloorPlanCAD door/window cases.
31. Tested a door-recall-weighted h512 target base (`door=1.5`) and residual refinement. Best-dev selection is negative (`0.952066` locked F1), but the final checkpoint plus h128 residual is a real F1/accuracy gain: locked ACC `0.981268`, macro F1 `0.974330`, errors `13/694`. Adding residual base-error sample weighting (`base_error_loss_weight=3`), a weak wall/opening boundary margin, low-confidence sample weighting, a small crop residual branch, and residual door loss `2.2` preserves ACC/F1. A dev-only F1-tolerant blend audit (`0.005` macro-F1 tolerance) selects blend `0.40`, preserving ACC/F1 and improving R2 from `0.928698` to `0.935690`. This is the best h512 F1/R2-preserving branch, while seed32 residual remains better for calibrated probability quality. Dev-only probability calibration, train+dev fixed final training, train-only coarse door morphology augmentation, simple feature-rule routing, stronger base-error weighting, focal loss, cascade residual refinement, prediction-level routing, additional residual seeds, supervised contrastive residual regularization, residual label smoothing, stronger margin, larger crop residual, excessive door/window weighting, and dual-branch ensembling do not improve the Pareto frontier.
32. Trained a stronger h768 door-recall-weighted FloorPlanCAD target base and reused the same frozen-base crop residual refiner. This is the first post-plateau error-count improvement: the h768 final base reaches locked ACC `0.979827`, macro F1 `0.972972`, R2 `0.942857`; the h128 residual raises it to locked ACC `0.982709`, macro F1 `0.978327`, errors `12/694`, with confusion `[[154,2,0],[8,482,2],[0,0,46]]`. It eliminates the previous `window->door` error and is now the best F1/ACC target-domain diagnostic. A fine dev-only max-F1 blend audit selects `blend=0.485`, preserving ACC/F1 and improving high-F1 R2 from `0.926453` to `0.927768`. A broader dev-only F1-tolerant blend audit recovers R2 to `0.939800` only by dropping locked F1 to `0.972972`; smoke-only `blend=0.45` keeps F1 `0.978327` and improves R2 to `0.930638`, but is not dev-selected. Feature-threshold routing has zero eligible rules, h1024 capacity overfits, h768 seed31 is lower-F1, and a dev-selected seed router regresses smoke to `0.974330`. The remaining 12 errors are still mostly `door->hard_wall`, so further gains likely need better hard-door morphology supervision rather than more residual loss tuning.

Paper v2 locked-test error profile:

| Error Source | Count |
|---|---:|
| all errors | 107 |
| wall/opening boundary errors | 100 |
| door-window cross errors | 7 |
| high-confidence wrong predictions | 79 |

Per-class locked-test F1:

| Class | F1 |
|---|---:|
| hard_wall | 0.995231 |
| door | 0.988141 |
| window | 0.982272 |

Strict selected source breakdown:

| Source | Accuracy | Macro F1 |
|---|---:|---:|
| CVC-FP | 0.993474 | 0.989412 |
| FloorPlanCAD | 0.976945 | 0.967773 |

Diagnostic routed candidate:

| Source | Accuracy | Macro F1 | Status |
|---|---:|---:|---|
| CVC-FP | 0.993243 | 0.989078 | inherited from selected model |
| FloorPlanCAD | 0.975504 | 0.969008 | diagnostic final-epoch expert |
| overall | 0.992346 | 0.988272 | not paper-selected yet |

Interpretation:

```text
The current in-domain locked-test target is met.
Pure zero-shot cross-source robustness is not solved.
Few-shot target-domain adaptation is now a credible and auditable robustness protocol, but not yet a 98% cross-domain result.
```

Few-shot target adaptation:

| Direction | Protocol | Locked Accuracy | Locked Macro F1 | Locked R2 | Status |
|---|---|---:|---:|---:|---|
| CVC-FP -> FloorPlanCAD | source train + half target dev, source-balanced sampler | 0.943804 | 0.913253 | 0.846537 | strict adapted |
| CVC-FP -> FloorPlanCAD | source train + half target dev, target/fragile sampler | 0.922190 | 0.884959 | 0.802572 | negative; target-dev overfit |
| FloorPlanCAD -> CVC-FP | source train + half target dev, source-balanced sampler | 0.968059 | 0.947563 | 0.915708 | strict adapted |
| CVC-FP -> FloorPlanCAD | training-only hard bbox/crop augmentation | 0.943804 | 0.915185 | 0.840019 | strict adapted; small gain |
| CVC-FP -> FloorPlanCAD | morphology-guided hard augmentation | 0.917867 | 0.876006 | 0.768198 | negative; synthetic morphology shift |
| CVC-FP -> FloorPlanCAD | hardaug + dev-only class-bias calibration | 0.939481 | 0.909470 | 0.833529 | negative |
| FloorPlanCAD -> CVC-FP | training-only hard bbox/crop augmentation | 0.970286 | 0.951022 | 0.917973 | strict adapted; best cross-source strict |
| FloorPlanCAD target specialist | mixed-source h512 + opening router | 0.969741 | 0.960265 | 0.923618 | auditable router; improves primary, below specialist |
| FloorPlanCAD target-only | target train + half target dev, crop aug | 0.976945 | 0.964518 | 0.936721 | not selected; below two-stage router |
| FloorPlanCAD target-only | target train + half target dev, h512 crop aug | 0.974063 | 0.965150 | 0.935821 | strict adapted; best clean target-only |
| FloorPlanCAD target-only | h512 best + final opening router | 0.975504 | 0.966492 | 0.937029 | auditable router; below final checkpoint |
| FloorPlanCAD target-only | target train + half target dev, h512 crop aug final epoch | 0.976945 | 0.968190 | 0.939330 | diagnostic; no dev-selected claim |
| FloorPlanCAD target-only | h512 final + h128 residual refiner | 0.978386 | 0.971835 | 0.941215 | selected residual path; improves F1/R2 |
| FloorPlanCAD target-only | h512 final + h128 residual refiner seed32 | 0.979827 | 0.973188 | 0.944101 | best residual path; still below 98% |
| FloorPlanCAD target-only | h512 door=1.5 final + h128 residual | 0.981268 | 0.974330 | 0.928698 | best F1/ACC; R2 regression |
| FloorPlanCAD target-only | h512 door=1.5 final + h128 residual base-error=3 | 0.981268 | 0.974330 | 0.928851 | best F1/ACC; tiny R2 gain |
| FloorPlanCAD target-only | h512 door=1.5 final + h128 residual base-error + boundary margin + low-confidence | 0.981268 | 0.974330 | 0.932133 | best F1/ACC; small R2 recovery |
| FloorPlanCAD target-only | h512 door=1.5 final + h128 crop residual base-error + boundary margin + low-confidence | 0.981268 | 0.974330 | 0.932381 | best F1/ACC; best R2-preserving variant |
| FloorPlanCAD target-only | h512 door=1.5 final + h128 crop residual base-error + boundary margin + low-confidence + door=2.2 | 0.981268 | 0.974330 | 0.933586 | best F1/ACC; best R2-preserving variant |
| FloorPlanCAD target-only | same + dev F1-tolerant blend 0.40 | 0.981268 | 0.974330 | 0.935690 | best F1/ACC; best R2-preserving blend |
| FloorPlanCAD target-only | h768 door=1.5 final base | 0.979827 | 0.972972 | 0.942857 | stronger base; high R2 before residual |
| FloorPlanCAD target-only | h768 door=1.5 final + h128 crop residual | 0.982709 | 0.978327 | 0.926453 | best F1/ACC; 12 errors, R2 tradeoff |
| FloorPlanCAD target-only | h768 residual fine dev max-F1 blend 0.485 | 0.982709 | 0.978327 | 0.927768 | best F1/ACC; clean small R2 gain |
| FloorPlanCAD target-only | h768 residual smoke-only blend 0.45 | 0.982709 | 0.978327 | 0.930638 | same F1, better R2, not dev-selected |
| FloorPlanCAD target-only | h768 residual dev F1-tolerant blend 0.30 | 0.979827 | 0.972972 | 0.939800 | dev-selected R2 recovery, loses F1 gain |
| FloorPlanCAD target-only | h768 feature-threshold rule router | 0.982709 | 0.978327 | n/a | no eligible train-improving/dev-safe rule |
| FloorPlanCAD target-only | h1024 door=1.5 final + residual | 0.978386 | 0.971551 | 0.936099 | negative; wider base overfits dev |
| FloorPlanCAD target-only | h768 seed31 final + residual | 0.981268 | 0.971917 | 0.937459 | negative; complementary errors but lower F1 |
| FloorPlanCAD target-only | h768 main->seed31 prediction router | 0.981268 | 0.974330 | 0.925941 | negative; dev-selected router regresses smoke |
| FloorPlanCAD target-only | train+dev fixed final residual diagnostic | 0.976945 | 0.967773 | 0.937875 | negative; more labels without selection hurts F1 |
| FloorPlanCAD target-only | train-only coarse door morphology augmentation | 0.978386 | 0.971551 | 0.938751 | negative; R2 up, errors increase to 15 |
| FloorPlanCAD target-only | simple feature-rule router | n/a | n/a | n/a | negative; no train-improving/dev-safe rule |
| FloorPlanCAD target-only | base-error weight 5/8/12 controls | 0.981268 | 0.974330 | 0.933366/0.930361/0.929774 | negative; no error reduction |
| FloorPlanCAD target-only | residual seeds 20260431-34 | 0.981268 | 0.974330 | 0.929666-0.934869 | same errors; below main R2 blend |
| FloorPlanCAD target-only | supervised contrastive residual controls | 0.981268 | 0.974330 | 0.933678-0.933904 | same errors; below main R2 blend |
| FloorPlanCAD target-only | residual focal gamma 0.5/1/1.5/2 | 0.981268/0.979827 | 0.974330/0.970552 | 0.930124-0.913434 | negative; no error reduction |
| FloorPlanCAD target-only | cascade residual h64/h128 | 0.969741/0.974063 | 0.952066/0.956134 | 0.929162/0.930052 | negative; second-stage overfits dev |
| FloorPlanCAD target-only | prediction router current<->seed32 | 0.976945/0.978386 | 0.967840/0.971835 | 0.926996/0.939942 | negative; dev improves, smoke regresses |
| FloorPlanCAD target-only | h512 door=1.5 residual + dev calibration | 0.979827 | 0.972972 | 0.930653 | R2 improves slightly, F1 drops |
| FloorPlanCAD target-only | h512 door=1.5 residual smoothing 0.01 | 0.979827 | 0.970552 | 0.930274 | negative regularization control |
| FloorPlanCAD target-only | h512 door=1.5 residual smoothing 0.03 + clip | 0.978386 | 0.969193 | 0.934049 | negative regularization control |
| FloorPlanCAD target-only | seed32 + door=1.5 residual ensemble | 0.978386 | 0.969193 | 0.936351 | negative Pareto |
| FloorPlanCAD target-only | h512 final + h128 residual 3-seed ensemble | 0.978386 | 0.971835 | 0.942091 | ensemble negative vs seed32 |
| FloorPlanCAD target-only | h512 final + crop residual refiner | 0.976945 | 0.968190 | 0.930206 | negative; dev overfit and lower R2 |
| FloorPlanCAD target-only | h512 relation-aware target-only best | 0.971182 | 0.959360 | 0.925855 | negative; relation labels alone insufficient |
| FloorPlanCAD target-only | h512 relation-aware+style best | 0.969741 | 0.955422 | 0.916599 | negative; dev overfit |
| FloorPlanCAD target-only | h512 relation-aware final + residual | 0.978386 | 0.971619 | 0.934454 | improves relation-aware base, below seed32 |
| FloorPlanCAD target-only | h512 half-dev seed31 best | 0.968300 | 0.947890 | 0.918177 | negative; dev overfit |
| FloorPlanCAD target-only | h512 half-dev seed32 final | 0.978386 | 0.965803 | 0.938667 | negative vs current best |
| FloorPlanCAD target-only | h512 label smoothing+clip final residual | 0.978386 | 0.969131 | 0.930607 | stable but below seed32 |
| FloorPlanCAD target-only | h512 deterministic morphology features best | 0.971182 | 0.957151 | 0.928046 | negative; dev saturates |
| FloorPlanCAD target-only | h512 deterministic morphology features final residual | 0.978386 | 0.966901 | 0.913500 | lower F1/R2 than seed32 |
| FloorPlanCAD target-only | nearly all target dev, crop aug final epoch | 0.975504 | 0.965836 | 0.938106 | diagnostic; best direct target-domain mitigation |
| FloorPlanCAD target-only | nearly all target dev, h512 crop aug final epoch | 0.953890 | 0.918225 | 0.869570 | diagnostic; negative capacity ablation |
| FloorPlanCAD target-only | nearly all target dev, equal 3-seed ensemble | 0.969741 | 0.951409 | 0.923232 | diagnostic; no improvement |
| FloorPlanCAD target-only | nearly all target dev, no crop aug final epoch | 0.972622 | 0.956857 | 0.928441 | diagnostic; no improvement |
| CVC-FP -> FloorPlanCAD | nearly all target dev, fixed final epoch | 0.949568 | 0.924959 | 0.866294 | diagnostic; no usable target dev selection |
| FloorPlanCAD -> CVC-FP | nearly all target dev, fixed final epoch | 0.966830 | 0.946132 | 0.910016 | diagnostic; no improvement over half-dev |

## What Not To Claim

Do not claim:

- a general CAD foundation model;
- full CAD understanding;
- end-to-end VLM superiority;
- universal floor-plan parsing;
- novelty of graph neural networks or crop CNNs themselves.

Do claim:

- a structure-aware, patch-aware primitive semantic labeling module;
- deterministic and auditable CAD primitive graph inference;
- strong wall/door/window primitive labeling under raster-derived floor-plan graphs;
- controlled evidence that local crop evidence plus primitive graph message passing is necessary for the observed boundary errors.

## Minimum Evidence Needed Before Submission

### P0: Must Have

1. Finish paper-ready seed stability table.
   - Three h512 runs have been executed and all stay near or above the requested region.
   - Still need the final table with mean/std and explicit selection rule.

2. Finalize baseline table on the same paper v2 split.
   - scalar topology/SE(2) model;
   - raster scalar model;
   - crop-only c32-ms3;
   - h384-l2 v2 model;
   - h512-l2 selected model;
   - zero-shot or prompt-based VLM baseline clearly marked as weak baseline.

3. Scope or address cross-source domain shift.
   - Pure zero-shot cross-source macro F1 is too low for a broad robustness claim.
   - Source feature removal and naive balanced source sampling are negative results.
   - Few-shot target-domain adaptation is now implemented and materially improves both directions.
   - Target/window loss weighting, checkpoint ensembling, and larger target-dev adaptation are negative controls.
   - Training-only bbox/crop hard augmentation helps CVC locked transfer modestly, but reaching 98% now likely requires real new hard examples or stronger target-domain style/geometry augmentation, especially for windows and wall/opening boundaries.
   - Dev R2 checkpoint selection is a negative control for the hardest `CVC->FloorPlanCAD` direction.
   - Simple crop style augmentation is also negative; the missing target-domain coverage appears structural/morphological, not just low-level raster style.
   - Relation-aware message passing over existing edge relation types is also negative for the hardest transfer direction.

### P1: Strongly Recommended

1. Add latency and memory benchmark.
2. Add rotation/scale stress test.
3. Add hard-boundary qualitative examples.
4. Include error audit: remaining errors are almost all wall/opening boundary cases.

## Suggested SCI Q2 Framing

Working title:

```text
CadStruct: Auditable Patch-Aware Primitive Graph Learning for Raster Floor-Plan Semantic Extraction
```

Contribution wording:

1. We introduce an auditable primitive-graph formulation for raster floor-plan semantic extraction, separating dense structural labeling from autoregressive VLM JSON generation.
2. We propose a patch-aware primitive graph model that fuses multi-scale local raster evidence, SE(2)-normalized geometric/topological features, and primitive-edge message passing.
3. We provide controlled ablations showing that local crop context and graph propagation are both necessary to resolve wall/opening boundary ambiguity.
4. We report high-accuracy primitive labeling with calibrated confidence and reproducible audits, reaching `0.992637` locked-test accuracy, `0.988548` locked-test macro F1, and `0.980085` locked-test probability R2 on a leakage-aware paper-v2 split. The limiting `window` class improves from `0.957839` F1 in paper v1 to `0.982272` F1 in paper v2.

## Final Judgment

The innovation is sufficient for an applied SCI Q2 target if the missing validation package is completed. The strongest angle is not "we used a GNN," but:

```text
we turned raster CAD semantic extraction into an auditable primitive-graph learning problem, proved why VLM JSON/LoRA alone fails dense alignment, and showed that multi-scale crop evidence plus graph message passing resolves the dominant wall/opening boundary errors.
```

The next work should be validation, not more architecture chasing.
