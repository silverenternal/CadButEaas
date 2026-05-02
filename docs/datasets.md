# Dataset Inventory

External datasets are stored under `datasets/external/` and ignored by git.

| Dataset | Local path | Status | Current size | Notes |
|---------|------------|--------|--------------|-------|
| FloorPlanCAD | `datasets/external/floorplancad` | Complete | 465 MiB | Hugging Face snapshot with 5,308 PNG files and metadata. Use as the main CAD/floor-plan semantic corpus. |
| CVC-FP Figshare package | `datasets/external/cvc_fp_figshare` | Complete, unpacked | 1.1 GiB | Includes `1-WID512_ROTATE`, `test_data`, and `5_folds_result`; unpacked set has 5,408 PNG files and 74,592 shapefile-sidecar annotation files. Good for wall/opening/room robustness evaluation. |
| CubiCasa5K HF mirror | `datasets/external/cubicasa5k_hf` | Complete but not primary | 40 KiB | This mirror only contains a small instruction JSON, not the full image corpus. |
| CubiCasa5K official Zenodo | `datasets/external/cubicasa5k_zenodo/` | Complete, unpacked, SVG-audited | ZIP 5.1 GiB | Official package verified with `unzip -t`; unpacked corpus has 5,000 SVG files and 12,342 PNG files. Converted MoE records live under `datasets/cadstruct_cubicasa5k_moe/`. |

The official CubiCasa5K download command was:

```bash
curl -L -C - \
  -o datasets/external/cubicasa5k_zenodo/cubicasa5k.zip \
  'https://zenodo.org/records/2613548/files/cubicasa5k.zip?download=1'
```

Unpack command:

```bash
mkdir -p datasets/external/cubicasa5k_zenodo/unpacked
unzip -q datasets/external/cubicasa5k_zenodo/cubicasa5k.zip \
  -d datasets/external/cubicasa5k_zenodo/unpacked
```

Audit and convert SVG annotations:

```bash
python scripts/vlm/audit_cubicasa5k_svg.py \
  --input-dir datasets/external/cubicasa5k_zenodo/unpacked \
  --output reports/vlm/cubicasa5k_svg_audit.json

python scripts/vlm/convert_cubicasa5k_svg.py \
  --input-dir datasets/external/cubicasa5k_zenodo/unpacked \
  --output-dir datasets/cadstruct_cubicasa5k_moe

python scripts/vlm/prepare_room_space_dataset.py \
  --input-dir datasets/cadstruct_cubicasa5k_moe \
  --output-dir datasets/cadstruct_rooms_v1

python scripts/vlm/prepare_symbol_fixture_dataset.py \
  --input-dir datasets/cadstruct_cubicasa5k_moe \
  --output-dir datasets/cadstruct_symbols_v1

python scripts/vlm/prepare_text_dimension_dataset.py \
  --input-dir datasets/cadstruct_cubicasa5k_moe \
  --output-dir datasets/cadstruct_text_dimensions_v1

python scripts/vlm/train_room_space_expert.py \
  --dataset-dir datasets/cadstruct_rooms_v1 \
  --output-dir checkpoints/cadstruct_moe_room_space_baseline

python scripts/vlm/train_symbol_fixture_expert.py \
  --dataset-dir datasets/cadstruct_symbols_v1 \
  --output-dir checkpoints/cadstruct_moe_symbol_fixture_baseline

python scripts/vlm/train_text_dimension_expert.py \
  --dataset-dir datasets/cadstruct_text_dimensions_v1 \
  --output-dir checkpoints/cadstruct_moe_text_dimension_baseline

python scripts/vlm/evaluate_room_space_expert.py \
  --dataset datasets/cadstruct_rooms_v1/smoke.jsonl \
  --model checkpoints/cadstruct_moe_room_space_baseline/model.json \
  --output reports/vlm/moe/room_space_baseline_smoke.json \
  --predictions-output reports/vlm/moe/room_space_baseline_smoke_predictions.jsonl

python scripts/vlm/export_moe_scene_graph.py \
  --input datasets/cadstruct_cubicasa5k_moe/smoke.jsonl \
  --output reports/vlm/moe/fused_scene_graph_smoke.jsonl

python scripts/vlm/audit_moe_scene_graph.py \
  --input reports/vlm/moe/fused_scene_graph_smoke.jsonl \
  --output reports/vlm/moe/fused_scene_graph_smoke_audit.json
```

Current CubiCasa5K MoE conversion:

| Split | Rows | File size |
|-------|------|-----------|
| train | 4,443 | 491 MiB |
| dev | 493 | 54 MiB |
| smoke | 64 | 6.7 MiB |

Converted CubiCasa5K family counts:

| Family | Instances |
|--------|-----------|
| boundary | 601,567 |
| space | 76,789 |
| symbol | 211,238 |
| text | 545,043 |

High-frequency labels include `hard_wall` 285,825, `door` 157,677, `window` 92,457,
`dimension_line` 245,576, `dimension_text` 122,788, `room_label` 61,394, `leader_line` 113,120,
`room` 20,358, `toilet` 15,390, `bedroom` 7,993, `bathroom` 7,288, `stair` 28,751,
`shower` 46,841, `sink` 49,656, `equipment` 49,462, and `appliance` 23,484.

The converter uses sparse architectural relations for OOM control: each door/window/opening keeps at most four
nearby wall attachments instead of all pairwise boundary intersections.

Current room-space extraction from CubiCasa5K:

| Split | Rows | Rooms | Adjacency edges |
|-------|------|-------|-----------------|
| train | 4,436 | 68,286 | 31,202 |
| dev | 493 | 7,491 | 3,392 |
| smoke | 64 | 927 | 389 |

The first dependency-free bbox-prototype RoomSpaceExpert is only a pipeline sanity baseline, not a paper model:
dev accuracy is `0.347083` and dev macro F1 is `0.222628`.

The first learned RoomSpaceExpert crop-MLP improves the complete dev split to accuracy `0.459485`
and macro F1 `0.297775` with `46.0` MiB CUDA reserved memory. This is still far from paper-grade:
mean IoU is `1.0` only because gold room boxes are reused, and room types such as `kitchen`, `closet`,
`office`, and `storage` remain weak without mask/topology/text fusion.

The first structure-aware RoomSpaceExpert context-MLP uses contained symbols, boundary-touch counts, room-label
counts, and adjacency degree. It improves complete dev accuracy to `0.577390` and macro F1 to `0.449065`.
This is the strongest evidence so far that room recognition needs structured context, not just local crops.
The streamed implementation writes an auditable feature cache at
`checkpoints/cadstruct_moe_room_space_context_mlp_streamed/train_features.jsonl` and lowers peak RSS from
about `4.0` GiB to about `1.6` GiB without changing the dev/smoke metrics.

Predicted-upstream RoomSpace evaluation is available in
`reports/vlm/room_space_predicted_upstream_comparison.json`. Replacing gold symbol/text labels with the current
learned SymbolFixture/TextDimension predictions gives dev accuracy `0.576990` and macro F1 `0.448567`, only
`-0.000498` macro F1 below the gold-context result. Boundary semantics and room boxes are still gold in this
ablation, so the next bottleneck is room proposal/mask/topology modeling rather than symbol/text classification.

The first model-capacity audit is available in `reports/vlm/room_space_model_capacity_audit.json`. A sklearn
RandomForest over the same context features improves RoomSpace dev accuracy to `0.639381` and macro F1 to
`0.525586`, with smoke accuracy `0.628910` and macro F1 `0.515591`. This is a useful stronger baseline but still
far below paper-grade, because the task is still bbox-level room-type classification without room masks, OCR room
name content, or graph message passing over rooms.

The first enhanced RoomSpace structural-feature audit is available in
`reports/vlm/room_space_enhanced_feature_audit.json`. Enhanced features add page margins, area rank, room-neighbor
geometry, symbol overlap/nearby evidence, text overlap counts, and boundary intersection/contact features. The best
dev model is enhanced RandomForest with dev accuracy `0.690975`, dev macro F1 `0.581734`, smoke accuracy
`0.690399`, and smoke macro F1 `0.577700`. Enhanced HistGBDT gives the best smoke macro F1, `0.608901`, but lower
dev macro F1, `0.566881`. These gains confirm the representation was underpowered, but the result is still not
paper-grade.

The converter now preserves CubiCasa room polygon shape summaries in `room_candidates[*].shape_features`.
The shape-feature audit is `reports/vlm/room_space_shape_feature_audit.json`. Shape RandomForest reaches dev
accuracy `0.699640`, dev macro F1 `0.589379`, smoke accuracy `0.704423`, and smoke macro F1 `0.593150`.
The shape features are useful but only modestly improve the best dev model, so remaining errors are mostly
semantic/OCR/topology issues rather than bbox geometry alone.

Current symbol fixture extraction:

| Split | Rows | Symbols | Host links |
|-------|------|---------|------------|
| train | 4,426 | 188,283 | 56,600 |
| dev | 491 | 20,542 | 6,331 |
| smoke | 63 | 2,413 | 743 |

The first dependency-free SymbolFixtureExpert bbox prototype is also a pipeline sanity baseline:
dev accuracy is `0.396748`, dev macro F1 is `0.286522`, and deterministic host-link F1 is `1.0`
because host links are recomputed from gold boxes.

The first learned SymbolFixtureExpert crop-MLP improves the complete dev split to accuracy `0.814721`
and macro F1 `0.614068` with `50.0` MiB CUDA reserved memory. It is still not paper-grade:
`bathtub` and `generic_symbol` remain at zero F1, and host-link F1 is deterministic from gold boxes.

Current text/dimension extraction:

| Split | Rows | Text candidates | Dimension links |
|-------|------|-----------------|-----------------|
| train | 4,441 | 485,740 | 109,438 |
| dev | 493 | 52,631 | 11,840 |
| smoke | 64 | 6,672 | 1,510 |

The first dependency-free TextDimensionExpert bbox prototype has dev accuracy `0.843970`,
dev macro F1 `0.607816`, and weak `dimension_of` link F1 `0.581716`. It still performs poorly on
`room_label`, so text content/OCR and crop evidence are required before paper use.

The first learned TextDimensionExpert crop-MLP uses `.venv-vlm/bin/python` with torch/Pillow and improves the
complete dev split to accuracy `0.940738`, macro F1 `0.729927`, and `dimension_of` link F1 `0.861986`.
Its peak CUDA reserved memory was only `50.0` MiB, but process RSS reached about `2.0` GiB because PIL crop
feature extraction is CPU-side.

Converted training/evaluation data lives under `datasets/cadstruct/` and is generated from external datasets by:

```bash
python scripts/vlm/convert_external_dataset.py
```

MoE expansion planning now lives in `docs/cadstruct-moe-dataset-roadmap.md` and
`docs/cadstruct-moe-architecture-plan.md`.
The first implementation layer adds `configs/vlm/cadstruct_ontology.json`,
`configs/vlm/dataset_registry.json`, and the lightweight `scripts/vlm/cadstruct_moe/`
schema/router/fusion scaffold.

Current CadStruct conversion:

| Split | Rows |
|-------|------|
| train | 2,132 |
| dev | 236 |
| smoke | 32 |

Source balance: 1,200 FloorPlanCAD samples and 1,200 CVC-FP samples. The converter keeps labels only in `expected_json`; `request_hints.primitive_graph` contains geometry only.

SFT-ready records live under `datasets/cadstruct_sft/` and are generated by:

```bash
python scripts/vlm/prepare_sft_dataset.py
```

The SFT converter intentionally caps prompt and target size. This keeps the assistant answer inside the training sequence and avoids extreme graph samples dominating memory.

Current compact SFT budget profile:

| Profile | Value |
|---------|-------|
| Report | `reports/vlm/sft_budget_128_compact.json` |
| Samples | 128 |
| `max_length` | 6,144 |
| `max_image_side` | 512 |
| Input tokens | min 576, mean 3,753.56, p95 6,144, max 6,144 |
| Supervised tokens | min 68, mean 1,236.57, p95 1,926, max 1,966 |
| Vision tiles | min 1, mean 3.95, p95 13, max 13 |
| At max length | 7 |
| Zero supervised labels | 0 |

OOM-sensitive samples are mostly CVC-FP rotations with 13 vision tiles or sequences exactly at `max_length`. The current training config skips samples above 8 tiles and skips samples that hit the max sequence length.
