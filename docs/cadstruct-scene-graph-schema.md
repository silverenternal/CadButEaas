# CadStruct Scene Graph 合同（v1）

本文档定义 MoE 融合链路的统一 Scene Graph 数据合同。所有专家输出在入图阶段前后应满足该合同。

## 设计目标

- 结构化可审计：每个实体/关系都要可追溯（`source_expert`、`audit_trace`）。
- 约束友好：允许在规则层或学习层统一约束（几何包含、邻接、关系合理性）。
- 与现有报告兼容：保留 `id`、`semantic_type` 等主标签语义；补齐 `source_expert/geometry/confidence`。

## 统一字段

### SceneGraph 根对象

- `version`：字符串，当前为 `cadstruct-moe-scene-graph-v1`
- `nodes`：列表，每个元素为 Node
- `edges`：列表，每个元素为 Edge
- `metadata`：对象

### Node 契约

必需字段：

1. `id`（string）
2. `semantic_type`（string）
3. `family`（string）
4. `source_expert`（string）
5. `confidence`（float，0~1）
6. `geometry`（object）
7. `audit_trace`（object）

可选字段：

- `metadata`（object）

几何子字段要求：
- `geometry.bbox` 为 `[x1, y1, x2, y2]` 四元组，且 `x2 > x1`, `y2 > y1`。
- 允许额外几何字段（如 `area`, `mask`, `raster_stats`），用于专家自身溯源。

### Edge 契约

必需字段：

1. `source`（string，指向 Node.id）
2. `target`（string，指向 Node.id）
3. `relation`（string，必须属于白名单关系）
4. `source_expert`（string）
5. `confidence`（float，0~1）
6. `geometry`（object）
7. `audit_trace`（object）

可选字段：

- `metadata`（object）

## 标签与关系白名单

标签来自 `configs/vlm/cadstruct_ontology.json`：
- boundary: hard_wall / partition_wall / door / window / opening / curtain_wall
- space: room / bedroom / living_room / kitchen / bathroom / toilet / corridor / balcony / closet / office / storage / unknown_room
- symbol: stair / column / sink / bathtub / toilet_fixture / shower / sofa / bed / table / chair / appliance / equipment / generic_symbol
- text: room_label / dimension_text / dimension_line / extension_line / leader_line / callout / legend_text / note_text
- sheet: title_block / table / schedule / legend / stamp / key_value_field

关系白名单：
- touches / contains / contained_in / bounds / interrupted_by / attached_to / inside / labels / dimension_of / adjacent_to

## 验证规则（`scene_graph_schema.py`）

- `nodes` / `edges` 必须是列表；
- 每个 Node/Edge 必须包含上述必填字段；
- Node id 不允许重复；
- Edge source/target 必须在 nodes 中存在；
- 非法 bbox、未知标签、未知 relation、缺失 audit trace 均判定为校验失败；
- 审计输出（`reports/vlm/*audit*.json`）应保存 `validation_errors`。

## 输出与消费

`scripts/vlm/scene_graph_schema.py` 提供：
- `convert_predictions_to_scene_graph()`：专家预测 -> SceneGraph
- `validate_scene_graph()`：返回 `(is_valid, errors)`
- `assert_scene_graph_contract()`：失败即抛异常

建议：
- 在每次 `fuse_predictions()` 前后都执行一次 contract 校验；
- 在每个 split 的融合报告中附带校验失败样本数和错误码分布；
- 任何 `smoke-only` 结果不得替代主 claim。
