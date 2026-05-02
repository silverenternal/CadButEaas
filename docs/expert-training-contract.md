# Expert Training Contract

CadStruct expert training configs use `configs/vlm/expert_training_schema.json`.

Supported shared keys include `dataset_dir`, `output_dir`, `epochs`, `batch_size`,
`hidden_dim`, `dropout`, `learning_rate`, item caps, class weighting, seed, and
memory/OOM policy fields.

Current schema-enabled entry points:

- `scripts/vlm/train_symbol_fixture_crop_mlp.py --config path/to/config.json`
- `scripts/vlm/train_text_dimension_crop_mlp.py --config path/to/config.json`

The config path is part of the auditable run contract and should be copied into
the run artifact bundle when launching paper-grade runs. Locked evaluation
labels must not be referenced by training configs.
