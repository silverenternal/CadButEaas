# CadStruct v18 Quality Runbook

This runbook is for the raster-only MoE recovery branch. Model-credit inference input is raster image only; offline labels are used only for dataset construction, training, locked evaluation, and diagnostics.

## Preflight

```bash
scripts/remote_moe_v18.sh --group preflight
```

The preflight validates `todo.json`, required report presence, and key JSON reports. It writes `reports/vlm/v18_reproducibility_check.json` and appends logs to `logs/remote_moe_run.log`.

## Topology Relations

Smoke:

```bash
scripts/remote_moe_v18.sh --group topology-smoke
```

Locked:

```bash
scripts/remote_moe_v18.sh --group topology-locked
```

Expected artifacts:

- `reports/vlm/topology_relations_v18_candidates.jsonl`
- `reports/vlm/topology_relations_v18_rerank_features.jsonl`
- `reports/vlm/topology_relations_v18_eval.json`
- `reports/vlm/topology_relations_v18_cap_sweep.json`
- `reports/vlm/topology_relations_v18_warning_audit.json`
- `reports/vlm/topology_relations_v18_source_integrity.json`

Current locked status: source integrity passes, but relation precision is still low. The relation-aware cap sweep improves small-cap recall, but does not reach the symbol and boundary adoption floors.

## Refiner

```bash
scripts/remote_moe_v18.sh --group refiner-locked
```

Expected artifacts:

- `datasets/image_only_scene_graph_refiner_v18/locked.jsonl`
- `datasets/image_only_scene_graph_refiner_v18/manifest.json`
- `checkpoints/scene_graph_refiner_v18/refiner_policy.json`
- `reports/vlm/scene_graph_refiner_v18_eval.json`
- `reports/vlm/scene_graph_refiner_v18_decisions.jsonl`
- `reports/vlm/scene_graph_refiner_v18_final_predictions.jsonl`
- `reports/vlm/scene_graph_refiner_v18_source_integrity.json`

Adoption floors:

- Candidate count after refiner decreases by at least 70% from 92,800.
- Boundary and space recall drop from adapter is at most 0.05.
- Symbol typed precision and F1 are at least 0.05.

Current locked status: not adoptable. The selected relation-support policy reduces candidates by 76.5517%, but boundary/space recall drop is too high and symbol typed precision/F1 remain below floor.

## Visual Hard Cases

Smoke:

```bash
scripts/remote_moe_v18.sh --group visual-smoke
```

Locked:

```bash
scripts/remote_moe_v18.sh --group visual-locked
```

Expected artifacts:

- `reports/vlm/visual_hard_cases_v18/index.html`
- `reports/vlm/visual_hard_cases_v18/manifest.json`
- `reports/vlm/visual_hard_cases_v18/assets/`

Current locked pack renders 40 pages with five deterministic buckets: OCR miss, symbol confusion, boundary duplicate flood, room topology gap, and false-positive flood.

## One-Command Smoke

```bash
scripts/remote_moe_v18.sh --smoke
```

This runs preflight, topology smoke, and visual smoke. Use it before longer locked runs.

## Current Blockers

- OCR is not usable for semantic room typing yet: all-gold normalized accuracy is 0.004666 and localized-only normalized accuracy is 0.032258.
- Symbol typing is still below adoption floor: original typed F1 is 0.005376; after refiner F1 is 0.018587.
- Relation candidates expose topology evidence, but precision is still near 1% to 2%.
- Refiner cannot both remove the false-positive flood and preserve boundary/space recall at the current detector quality.
