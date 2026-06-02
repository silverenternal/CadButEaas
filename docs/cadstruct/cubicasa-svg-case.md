# CubiCasa/SVG 场景完整案例

最后更新: 2026-05-10

## 案例定位

这个案例用于说明“之前已经做好的 CubiCasa/SVG 派生专家和 MoE 链路”是怎么从输入走到输出的。它是 SVG/结构化监督场景下的 canonical/oracle-style 案例，不是非 SVG 原始 raster-only 推理案例。

边界要先说清楚：

- 输入里允许使用 CubiCasa 的 `model.svg` 和转换后的 `expected_json`，因为这里验证的是历史专家契约、路由、融合和 scene graph schema。
- 这里的 1.0 scene graph F1 说明 SVG 结构化输入到 MoE 融合输出这条链路是闭合的。
- 它不能直接证明“任意 PNG/JPG 图纸端到端识别已经 1.0”，后者还需要 raster candidate frontend、OCR、symbol body/type 和 relation compression。

## 本次复跑产物

| 产物 | 路径 |
|---|---|
| 输入 split | `datasets/cadstruct_cubicasa5k_moe_locked/smoke.jsonl` |
| 融合 scene graph | `reports/vlm/cubicasa_svg_case/fused_scene_graph_locked_smoke.jsonl` |
| 融合审计 | `reports/vlm/cubicasa_svg_case/fused_scene_graph_locked_smoke_audit.json` |
| scene graph F1 | `reports/vlm/cubicasa_svg_case/scene_graph_f1_locked_smoke_eval.json` |
| mismatch cases | `reports/vlm/cubicasa_svg_case/scene_graph_f1_locked_smoke_cases.jsonl` |
| 真实专家模型融合输出 | `reports/vlm/cubicasa_svg_case/model_fused_scene_graph_locked_smoke.jsonl` |
| 真实专家模型指标 | `reports/vlm/cubicasa_svg_case/model_fused_scene_graph_locked_smoke_eval.json` |
| 真实专家 runtime audit | `reports/vlm/cubicasa_svg_case/model_expert_runtime_audit.json` |
| smoke manifest rerun | `reports/vlm/cubicasa_svg_case/cadstruct_moe_smoke_manifest_v18_rerun.json` |
| locked gate rerun | `reports/vlm/cubicasa_svg_case/cadstruct_moe_locked_manifest_v18_rerun.json` |

## 输入

原始 CubiCasa 资产已经下载在：

```text
datasets/external/cubicasa5k_zenodo/unpacked/cubicasa5k/
```

单条样本包含：

```text
high_quality_architectural/<case_id>/F1_scaled.png
high_quality_architectural/<case_id>/model.svg
```

转换后的 MoE 记录位于：

```text
datasets/cadstruct_cubicasa5k_moe_locked/smoke.jsonl
```

每条记录的关键字段：

| 字段 | 含义 |
|---|---|
| `image_path` | CubiCasa raster floorplan 图片，例如 `F1_scaled.png` |
| `annotation_path` | CubiCasa SVG 标注，例如 `model.svg` |
| `request_hints.primitive_graph.nodes` | 从 SVG 解析出的 boundary primitive/candidate 节点 |
| `request_hints.primitive_graph.edges` | primitive 间的拓扑关系 |
| `expected_json.semantic_candidates` | boundary/wall/opening/window 专家契约输入 |
| `expected_json.room_candidates` | room/space 专家契约输入 |
| `expected_json.symbol_candidates` | symbol fixture 专家契约输入 |
| `expected_json.text_candidates` | text/dimension 专家契约输入 |

## 数据流

```text
CubiCasa image + model.svg
  -> convert_cubicasa5k_svg.py
  -> datasets/cadstruct_cubicasa5k_moe_locked/*.jsonl
  -> export_moe_scene_graph.py --source expected_json
  -> predictions_from_record()
  -> ExpertPrediction list
  -> fuse_predictions()
  -> scene_graph.nodes + scene_graph.edges
  -> audit_moe_scene_graph.py
  -> evaluate_scene_graph_f1.py
```

### Step 1: CubiCasa/SVG 转 MoE 记录

历史转换入口：

```bash
.venv/bin/python scripts/vlm/convert_cubicasa5k_svg.py
.venv/bin/python scripts/vlm/split_cubicasa_moe_locked.py
```

现有产物：

```text
datasets/cadstruct_cubicasa5k_moe/manifest.json
datasets/cadstruct_cubicasa5k_moe_locked/manifest.json
```

`datasets/cadstruct_cubicasa5k_moe_locked/manifest.json` 是按 record 分组的 train/dev/locked/smoke 切分，避免同一图纸泄漏到多个 split。

### Step 2: 转成专家输出契约

本案例使用：

```bash
.venv/bin/python scripts/vlm/export_moe_scene_graph.py \
  --input datasets/cadstruct_cubicasa5k_moe_locked/smoke.jsonl \
  --output reports/vlm/cubicasa_svg_case/fused_scene_graph_locked_smoke.jsonl \
  --source expected_json
```

`export_moe_scene_graph.py` 内部做了四类专家契约映射：

| 输入候选 | 归属专家族 | 输出 family | 输出示例 |
|---|---|---|---|
| `semantic_candidates` | `wall_opening` | `boundary` | `hard_wall`, `door`, `window`, `opening` |
| `room_candidates` | `room_space` | `space` | `bedroom`, `bathroom`, `kitchen`, `room` |
| `symbol_candidates` | `symbol_fixture` | `symbol` | `sink`, `shower`, `stair`, `equipment` |
| `text_candidates` | `text_dimension` | `text` | `room_label`, `dimension_text`, `note_text` |

这些会被统一成 `ExpertPrediction`：

```text
candidate_id, expert, family, label, confidence, bbox, geometry, relations, source, metadata
```

### Step 3: MoE 融合

融合入口：

```text
scripts/vlm/cadstruct_moe/fusion.py::fuse_predictions
```

融合做三件事：

1. 把各专家输出合并为 `scene_graph.nodes`。
2. 把专家附带的 `relations` 合并为 `scene_graph.edges`。
3. 跑基础建筑约束检查，输出 `warnings`，例如 opening 没有关联 wall、room 没有 boundary 关系。

本次 locked smoke 融合输出：

```json
{
  "records": 64,
  "nodes": 10198,
  "edges": 21855,
  "warning_counts": {
    "opening_without_wall_relation:boundary_32": 1,
    "opening_without_wall_relation:boundary_36": 1,
    "opening_without_wall_relation:boundary_42": 1,
    "room_without_boundary_relation:svg_3": 1,
    "room_without_boundary_relation:svg_7": 1
  }
}
```

### Step 4: 审计输出

命令：

```bash
.venv/bin/python scripts/vlm/audit_moe_scene_graph.py \
  --input reports/vlm/cubicasa_svg_case/fused_scene_graph_locked_smoke.jsonl \
  --output reports/vlm/cubicasa_svg_case/fused_scene_graph_locked_smoke_audit.json
```

审计结果：

| 指标 | 值 |
|---|---:|
| records | 64 |
| total_nodes | 10198 |
| total_edges | 21855 |
| nodes_per_record | 159.34375 |
| edges_per_record | 341.484375 |
| records_with_warnings | 4 |
| boundary nodes | 6657 |
| space nodes | 686 |
| symbol nodes | 2285 |
| text nodes | 570 |

主要关系计数：

| relation | count |
|---|---:|
| `bounds` | 11557 |
| `attached_to` | 8114 |
| `contains` | 2184 |

主要标签计数：

| label | count |
|---|---:|
| `hard_wall` | 3208 |
| `door` | 1795 |
| `window` | 932 |
| `sink` | 582 |
| `room_label` | 570 |
| `opening` | 561 |

### Step 5: scene graph F1 评估

命令：

```bash
.venv/bin/python scripts/vlm/evaluate_scene_graph_f1.py \
  --input reports/vlm/cubicasa_svg_case/fused_scene_graph_locked_smoke.jsonl \
  --source-records datasets/cadstruct_cubicasa5k_moe_locked/smoke.jsonl \
  --output reports/vlm/cubicasa_svg_case/scene_graph_f1_locked_smoke_eval.json \
  --cases-output reports/vlm/cubicasa_svg_case/scene_graph_f1_locked_smoke_cases.jsonl
```

结果：

| 指标 | precision | recall | F1 |
|---|---:|---:|---:|
| nodes | 1.0 | 1.0 | 1.0 |
| relations | 1.0 | 1.0 | 1.0 |

其他结果：

| 指标 | 值 |
|---|---:|
| records | 64 |
| node tp/pred/gold | 10198 / 10198 / 10198 |
| relation tp/pred/gold | 21855 / 21855 / 21855 |
| invalid_graph_rate | 0.0 |
| mismatch case_count | 0 |

这个结果说明：在 CubiCasa/SVG 结构化输入下，专家契约映射和 MoE 融合输出可以无损复现 expected scene graph。

## 真实专家模型复跑

上面的 `export_moe_scene_graph.py --source expected_json` 是 oracle-style 契约闭环，不是模型预测。为了看真实模型效果，补充了专用 runner：

```bash
.venv/bin/python scripts/vlm/run_cubicasa_svg_model_case.py
```

这个 runner 做的是：

```text
CubiCasa/SVG-derived candidates
  -> build RoutedCandidate per record
  -> load registered expert checkpoints
  -> expert.predict()
  -> fuse_predictions()
  -> compare predicted labels against expected_json labels
```

本次所有专家都成功加载 checkpoint，没有 fallback：

| family | source | fallback |
|---|---|---:|
| boundary | `wall_opening_crop_gnn` | 0 |
| space | `room_space_sklearn_context` | 0 |
| symbol | `symbol_fixture_v9_extra_trees` | 0 |
| text | `text_dimension_v5_calibrated_note_gate` | 0 |

真实模型结果：

| 指标 | precision | recall | F1 |
|---|---:|---:|---:|
| all nodes | 0.465091 | 0.465091 | 0.465091 |
| boundary nodes | 0.472285 | 0.472285 | 0.472285 |
| space nodes | 0.886297 | 0.886297 | 0.886297 |
| symbol nodes | 0.185558 | 0.185558 | 0.185558 |
| text nodes | 0.994737 | 0.994737 | 0.994737 |
| relations | 1.0 | 1.0 | 1.0 |

输出文件：

```text
reports/vlm/cubicasa_svg_case/model_fused_scene_graph_locked_smoke.jsonl
reports/vlm/cubicasa_svg_case/model_fused_scene_graph_locked_smoke_eval.json
reports/vlm/cubicasa_svg_case/model_expert_runtime_audit.json
reports/vlm/cubicasa_svg_case/model_fused_scene_graph_locked_smoke_cases.jsonl
```

解读：

- `1.0` 是 oracle-style SVG 契约闭环，不是模型识别效果。
- 真实模型在这个 smoke 上最强的是 text，已经到 `0.994737`。
- space 中等，`0.886297`。
- boundary 和 symbol 在当前 registered expert wrapper/checkpoint 组合下很差，分别是 `0.472285` 和 `0.185558`。
- relation F1 是 `1.0`，因为本次 SVG 仍提供候选拓扑关系；它不代表模型自己从图像中学会了关系。

这里也暴露出一个重要工程事实：registry 当前加载的 boundary wrapper 用的是 `cadstruct_graph_node_crop_gnn_h1024...`，不是前文记录的 h384 canonical 质量 checkpoint；symbol wrapper 加载的是 v9 ExtraTrees，不是 v13 contribution matrix 里的 symbol v13 路线。因此“模型实跑案例”比“历史报告里的最佳专家指标”低很多，后续要把 registry 的 canonical checkpoint 和案例 runner 对齐。

## 专家指标补充

这些是同一历史 CubiCasa/MoE 资产体系下的可复用专家指标：

| 专家 | 报告 | 指标 |
|---|---|---|
| boundary v13 | `reports/vlm/boundary_expert_v13_eval.json` | locked accuracy `1.0`, macro F1 `1.0` |
| graph-node crop/GNN h384 | `reports/vlm/graph_node_crop_gnn_h384_c32_ms3_l2_e24_calibrated_dev.json` | dev accuracy `0.991147`, macro F1 `0.986746` |
| graph-node crop/GNN h384 | `reports/vlm/graph_node_crop_gnn_h384_c32_ms3_l2_e24_calibrated_smoke.json` | smoke accuracy `0.991165`, macro F1 `0.986652` |
| room v13 | `reports/vlm/expert_contribution_matrix_v13.json` | locked macro F1 `0.9821291711858883` |
| text v13 | `reports/vlm/expert_contribution_matrix_v13.json` | locked macro F1 `0.973128` |
| symbol v13 | `reports/vlm/expert_contribution_matrix_v13.json` | locked macro F1 `0.883069` |
| deterministic structured router | `reports/vlm/domain_structured_moe_route_audit_v1.json` | route accuracy `1.0`, wrong expert rate `0.0` |

解读：

- boundary、graph-node、room 是历史强专家资产。
- text 接近 0.98，但仍受 OCR/text candidate 质量影响。
- symbol 是历史体系里相对弱的专家，非 SVG/raster-only 下必须另做 symbol body/type detector。

## 本次额外 runner 结果

命令：

```bash
.venv/bin/python scripts/vlm/run_cadstruct_moe_smoke_v18.py \
  --output reports/vlm/cubicasa_svg_case/cadstruct_moe_smoke_manifest_v18_rerun.json

.venv/bin/python scripts/vlm/run_cadstruct_moe_locked_v18.py \
  --output reports/vlm/cubicasa_svg_case/cadstruct_moe_locked_manifest_v18_rerun.json
```

结果：

- smoke manifest gate: `fail`，原因是 `project_structure_audit` 当前 gate 失败；`v17_source_integrity_passed=true`，`violations=0`。
- locked gate aggregator: `fail`，因为它聚合的是当前 v18 raster-only/image-only 资产，62 个 locked 指标低于 0.98，且尚无完整集成 locked v18 raster-to-graph runner。

这两个 runner 的失败不推翻本案例的 SVG/CubiCasa scene graph 结果。它们说明的是当前工程治理和 raster-only 集成链路还未闭合。

## 怎么复跑

最小复跑：

```bash
mkdir -p reports/vlm/cubicasa_svg_case

.venv/bin/python scripts/vlm/export_moe_scene_graph.py \
  --input datasets/cadstruct_cubicasa5k_moe_locked/smoke.jsonl \
  --output reports/vlm/cubicasa_svg_case/fused_scene_graph_locked_smoke.jsonl \
  --source expected_json

.venv/bin/python scripts/vlm/audit_moe_scene_graph.py \
  --input reports/vlm/cubicasa_svg_case/fused_scene_graph_locked_smoke.jsonl \
  --output reports/vlm/cubicasa_svg_case/fused_scene_graph_locked_smoke_audit.json

.venv/bin/python scripts/vlm/evaluate_scene_graph_f1.py \
  --input reports/vlm/cubicasa_svg_case/fused_scene_graph_locked_smoke.jsonl \
  --source-records datasets/cadstruct_cubicasa5k_moe_locked/smoke.jsonl \
  --output reports/vlm/cubicasa_svg_case/scene_graph_f1_locked_smoke_eval.json \
  --cases-output reports/vlm/cubicasa_svg_case/scene_graph_f1_locked_smoke_cases.jsonl
```

验收标准：

- `scene_graph_f1_locked_smoke_eval.json.node_f1.f1 == 1.0`
- `scene_graph_f1_locked_smoke_eval.json.relation_f1.f1 == 1.0`
- `scene_graph_f1_locked_smoke_eval.json.invalid_graph_rate == 0.0`
- `scene_graph_f1_locked_smoke_cases.jsonl` 为空

## 对后续非 SVG/raster-only 的意义

这个案例给后续工作提供了清晰目标：非 SVG 输入不是要重写整个 MoE，而是要把 raster frontend 的输出对齐到这套专家契约。

后续 raster-only 应该复用的接口：

```text
raster candidate frontend
  -> semantic_candidates / room_candidates / symbol_candidates / text_candidates
  -> ExpertPrediction contract
  -> deterministic structured router
  -> fuse_predictions
  -> scene_graph
  -> same audit/eval
```

也就是说，当前最大工程任务不是继续在下游融合层“补丁式调参”，而是让 raster-only 上游稳定地产生和 CubiCasa/SVG 案例一致的专家契约输入。
