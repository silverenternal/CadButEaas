# CadStruct raster-only MoE ideal architecture v18

本文描述 CadStruct 面向非矢量图纸的理想模型结构，以及数据从输入到输出的完整流向。这里的“非矢量图纸”指 PNG、JPG、扫描图、截图、PDF 栅格页等只有像素可用的输入；模型推理阶段不得读取 SVG、DXF、PDF 矢量对象、解析器几何、`expected_json` 或人工标注几何。

## 目标

CadStruct raster-only MoE 的目标不是把图纸简单 OCR 成文本，也不是输出一堆检测框，而是把单页或多页栅格图纸识别为可审计的结构化 scene graph：

```text
raster drawing
-> normalized page image
-> high-recall visual proposals
-> family-specific experts
-> relation-aware topology policy
-> calibrated MoE fusion
-> schema-valid scene graph
-> visual review pack and machine-readable JSON
```

理想输出需要同时满足四类要求：

- 几何完整：墙、门窗、房间、符号、文字、标注、图框等实体有稳定位置、范围和置信度。
- 语义正确：实体不仅有 objectness，还要有类型，例如 `door`、`window`、`bed`、`room_label`、`dimension_text`。
- 拓扑可信：实体之间有关系，例如 `bounded_by`、`contains_symbol`、`labeled_by_text`、`adjacent_to`、`dimension_of`。
- 可审计：每个节点和边都能追溯到输入像素、候选来源、专家输出、路由决策、置信度校准和后处理依据。

## 硬输入合同

推理阶段只允许使用：

- 原始栅格图像像素。
- 从原始像素计算出的图像金字塔、灰度图、二值图、边缘图、连通域、局部密度、纹理、patch embedding。
- 模型自身从像素预测出的候选框、mask、关键点、文本框、关系候选和置信度。
- 与输入无关的固定配置，例如 ontology、schema、阈值、checkpoint、类别白名单。

推理阶段禁止使用：

- SVG、DXF、IFC、CAD parser 输出。
- PDF 内部矢量对象。
- `expected_json`、人工标注 bbox、人工标注关系。
- 任何从标注 ID、文件名或数据集转换脚本泄漏出的真实对象 ID。

离线标签只允许用于：

- 构建训练集。
- 训练检测器、分类器、OCR、关系模型、融合模型。
- dev policy selection、locked evaluation、错误归因和上界分析。
- 生成报告，不得作为线上推理输入。

## 领域对象

理想 scene graph 至少覆盖以下对象族：

| Family | 典型节点 | 主要任务 |
| --- | --- | --- |
| `boundary` | wall、partition、door、window、opening | 识别建筑边界和开口，支持房间闭合与邻接判断 |
| `space` | room、corridor、balcony、kitchen、bathroom | 识别房间或空间区域、轮廓、中心、面积和类型 |
| `symbol` | bed、sofa、sink、toilet、stair、table、appliance | 识别图纸符号实例和符号类型 |
| `text` | room label、dimension text、note、legend text | 定位文字、识别内容、归一化文本 |
| `dimension` | dimension line、extension line、leader、arrow | 识别尺寸链和被标注对象 |
| `sheet` | title block、legend、schedule、stamp | 识别图框、图例、表格和元信息区域 |

核心关系包括：

- `bounded_by`: space -> boundary
- `contains_symbol`: space -> symbol
- `labeled_by_text`: space -> text
- `adjacent_to`: space -> space 或 boundary/space 邻接
- `attached_to`: symbol/opening -> boundary
- `dimension_of`: dimension/text -> measured object
- `inside` / `contains`: 通用包含关系

## 总体模块

理想系统由七层组成。

### 1. Page normalizer

输入是任意来源的栅格页，输出统一坐标系下的标准页面对象。

职责：

- 解码 PNG、JPG、PDF raster page、扫描图。
- 统一 DPI、方向、页面尺寸和颜色空间。
- 估计图纸有效区域，剔除大面积空白边。
- 生成多尺度图像金字塔。
- 输出基础 raster features：灰度、二值、边缘、距离变换、连通域、局部黑像素密度。

输出结构：

```json
{
  "page_id": "string",
  "image": {
    "width": 4096,
    "height": 3072,
    "channels": 3,
    "dpi_estimate": 300
  },
  "coordinate_frame": "page_pixel_v1",
  "raster_features": {
    "pyramid_levels": ["1x", "0.5x", "0.25x"],
    "binary_map": "artifact_ref",
    "edge_map": "artifact_ref",
    "connected_components": "artifact_ref"
  },
  "audit": {
    "source_mode": "raster_only",
    "vector_input_used": false
  }
}
```

### 2. High-recall proposal layer

这一层追求 recall，不急于做强压缩。它给各个专家提供候选，而不是直接宣布最终结果。

子模块：

- Boundary proposal detector：输出墙线、门窗、开口、长线段、厚墙候选。
- Space proposal detector：输出房间区域、空间中心点、粗 mask、候选 polygon。
- Symbol objectness detector：输出可能是图纸符号的 bbox/mask，不直接承诺类型。
- Text proposal detector：输出文字行、文字块、旋转框。
- Dimension proposal detector：输出尺寸线、箭头、延长线、标注串候选。
- Sheet proposal detector：输出图框、标题栏、图例、表格区域。

关键原则：

- 候选层可以有较多重复，但必须保留足够召回。
- 候选必须是像素推理结果，不允许带入标注 ID。
- 每个候选保存来源、尺度、patch 证据和 raster stats，便于后续专家复核。

统一候选结构：

```json
{
  "candidate_id": "page_001:symbol_prop:000123",
  "family": "symbol",
  "bbox": [100.0, 200.0, 132.0, 236.0],
  "mask_ref": null,
  "objectness": 0.91,
  "proposal_source": "symbol_objectness_detector_v18",
  "features": {
    "local_dark_density": 0.38,
    "edge_density": 0.24,
    "scale": 1.0,
    "patch_embedding_ref": "artifact_ref"
  },
  "audit_trace": {
    "input": "raster_pixels",
    "vector_input_used": false
  }
}
```

### 3. Family expert layer

MoE 的专家不是一个通用大模型随意输出 JSON，而是按图纸对象族拆开的专门模型。每个专家只负责自己的局部问题，并输出可校准的候选解释。

#### Boundary expert

职责：

- 对 wall、partition、door、window、opening 分类。
- 估计线段、厚墙区域、开口端点和方向。
- 合并碎片化墙段，同时保留 junction 和 opening 证据。

理想输出：

- `boundary` node candidates。
- 几何字段包括 bbox、polyline、thickness、orientation、junction refs。
- 对门窗输出 hinge/opening direction 或 window span。

#### Space expert

职责：

- 从墙体和像素区域推断房间候选。
- 输出 room mask、polygon、center point、area。
- 识别空间类型，但空间类型应与 text expert 和 symbol expert 联合校准。

理想输出：

- `space` node candidates。
- 每个 space 有 polygon/mask、center、boundary support、enclosure confidence。

#### Symbol expert

职责：

- 将 symbol objectness proposal 分类为具体符号类型。
- 支持 `unknown_symbol` / abstain，避免低置信 typed label 污染拓扑。
- 使用多尺度上下文识别贴墙、贴文字、半遮挡符号。

理想输出：

- `symbol` node candidates。
- `objectness_confidence` 和 `type_confidence` 分离。
- 低置信类型只保留 generic symbol，不进入 typed downstream claim。

#### Text and OCR expert

职责：

- 定位文字行和文字块。
- 识别文字内容。
- 按项目规则归一化文本，例如大小写、空格、单位、尺寸格式。
- 区分 room label、dimension text、note、legend text。

理想输出：

- `text` node candidates。
- `raw_text`、`normalized_text`、`language/script`、`orientation`、`recognition_confidence`。

#### Dimension expert

职责：

- 识别尺寸线、箭头、延长线、leader line。
- 将尺寸文本和被测对象关联。
- 输出测量关系和可解析数值。

理想输出：

- `dimension` nodes。
- `dimension_of` edges。
- 数值字段包括 `value`、`unit`、`normalized_value`、`parse_confidence`。

#### Sheet expert

职责：

- 识别标题栏、图例、表格、比例尺、页号、专业信息。
- 防止标题栏和图例里的文字/符号污染主图区域识别。

理想输出：

- `sheet` nodes。
- sheet 区域内对象带 `sheet_region_id`，供后续路由降权或隔离。

### 4. Router and gating layer

Router 决定哪些候选进入哪些专家、哪些专家结果可以进入关系建模和最终融合。它不是硬编码 NMS 的替代品，而是可审计的决策层。

输入：

- 页面级 raster features。
- proposal candidates。
- family expert logits。
- 局部上下文，例如候选周围的 room、boundary、text、symbol 密度。

输出：

- 每个候选的 route decision。
- 专家启用/禁用理由。
- abstain 或 fallback 标记。

理想策略：

- 对 objectness 高但 type 不稳的 symbol，保留节点候选但不输出 typed claim。
- 对 OCR 不稳的 text，保留 bbox 但不输出 room label claim。
- 对边界和空间候选优先保 recall，交给 relation-aware topology 压缩。
- 对 sheet 区域候选降低主图拓扑参与权重。

route decision 示例：

```json
{
  "candidate_id": "page_001:symbol_prop:000123",
  "routes": ["symbol_expert", "contains_symbol_relation_policy"],
  "blocked_routes": ["typed_symbol_output"],
  "reason": {
    "symbol_objectness": 0.91,
    "type_confidence": 0.54,
    "typed_output_floor": 0.98
  }
}
```

### 5. Relation-aware topology layer

这一层是 v18 当前最大的架构重点。它负责把节点候选连接成结构化关系，并在保 recall 的前提下压缩重复候选。

不能依赖简单硬 NMS 或固定 cap，因为它们会把真实关系一起删掉。理想做法是 relation-aware compression：

- 先保留高召回关系候选。
- 对每种关系训练专门 reranker/policy。
- 对 `contains_symbol` 使用 joint room-symbol assignment，而不是只拆 symbol 或只按 source cap。
- 对 `bounded_by` 使用 room-boundary 几何一致性和 enclosure support。
- 对 `adjacent_to` 使用共享边界、距离、门洞、空间 mask 接触证据。
- 对 `labeled_by_text` 同时要求 text localization、OCR confidence 和 room-text 空间关系。

关系候选结构：

```json
{
  "edge_candidate_id": "page_001:contains_symbol:000456",
  "relation": "contains_symbol",
  "source_candidate_id": "page_001:space_prop:000010",
  "target_candidate_id": "page_001:symbol_prop:000123",
  "raw_score": 0.74,
  "features": {
    "center_inside_source": true,
    "bbox_overlap_ratio": 0.12,
    "relative_offset_x": 0.34,
    "relative_offset_y": -0.18,
    "source_cluster_spread": 21.5,
    "target_cluster_spread": 8.2,
    "local_density": 0.31
  },
  "instance_keys": {
    "source_cluster_id": "space_cluster_07",
    "target_cluster_id": "symbol_cluster_31",
    "room_instance_cluster_id": "space_cluster_07:inst_0",
    "symbol_instance_cluster_id": "symbol_cluster_31:inst_1"
  }
}
```

#### Joint room-symbol assignment

`contains_symbol` 的理想实现必须同时判断 room side 和 symbol side：

- 一个 room cluster 可能包含多个真实 room anchor。
- 一个 symbol cluster 可能包含多个相邻真实 symbol。
- 多个 room anchor 和多个 symbol anchor 之间可能存在真实多边关系。
- 压缩时不能把一个 cluster pair 强行压成一条边。

理想策略：

```text
space candidates + symbol candidates + relation candidates
-> build local bipartite room-symbol graph
-> infer room_instance_cluster_id and symbol_instance_cluster_id
-> score candidate edges inside each joint pair
-> keep one or several representatives only when topology supports it
-> emit auditable compressed contains_symbol edges
```

这比单纯 symbol splitter 更关键，因为当前主要损失来自 cluster pair 内代表边选错和 source cap 丢失，而不是纯目标侧重复。

### 6. Fusion and constraint layer

Fusion 把各专家节点和关系候选合成为最终 scene graph。它应该是 fail-closed 的：没有足够证据时宁可 abstain 或保留低级候选，不应编造高置信结构。

职责：

- 去重：合并同一实体的多个候选。
- 冲突解决：例如一个候选不能同时是 door 和 bed。
- 关系一致性：例如 `contains_symbol` 的 source 应是 space，target 应是 symbol。
- schema 校验：输出必须满足 scene graph contract。
- 校准：所有 confidence 应能在 locked evaluation 上解释实际正确率。
- 审计：保留每个保留/删除/合并动作的理由。

理想输出节点：

```json
{
  "id": "page_001:symbol:bed:0007",
  "semantic_type": "bed",
  "family": "symbol",
  "confidence": 0.985,
  "geometry": {
    "bbox": [100.0, 200.0, 132.0, 236.0],
    "mask_ref": "artifact_ref"
  },
  "source_expert": "symbol_type_model_v18",
  "audit_trace": {
    "proposal_id": "page_001:symbol_prop:000123",
    "objectness_confidence": 0.992,
    "type_confidence": 0.985,
    "router_decision": "typed_symbol_enabled",
    "vector_input_used": false
  }
}
```

理想输出关系：

```json
{
  "source": "page_001:space:room:0002",
  "target": "page_001:symbol:bed:0007",
  "relation": "contains_symbol",
  "confidence": 0.982,
  "source_expert": "relation_topology_policy_v18",
  "geometry": {
    "source_bbox": [40.0, 120.0, 260.0, 360.0],
    "target_bbox": [100.0, 200.0, 132.0, 236.0]
  },
  "audit_trace": {
    "edge_candidate_id": "page_001:contains_symbol:000456",
    "source_cluster_id": "space_cluster_07",
    "target_cluster_id": "symbol_cluster_31",
    "room_instance_cluster_id": "space_cluster_07:inst_0",
    "symbol_instance_cluster_id": "symbol_cluster_31:inst_1",
    "compression_policy": "joint_room_symbol_assignment_v18",
    "vector_input_used": false
  }
}
```

### 7. Review, metrics, and feedback layer

最终系统必须持续生成机器可读报告和人工可看的 review pack。

输出：

- `scene_graph.jsonl`: 每页最终结构化结果。
- `candidates.jsonl`: 高召回候选。
- `relations.jsonl`: 关系候选和压缩后关系。
- `source_integrity.json`: 证明没有使用矢量输入或标注泄漏。
- `eval.json`: locked precision、recall、F1、per-class、per-relation 指标。
- `audit.json`: 错误归因、丢失原因、阈值选择、abstain 统计。
- `visual_review_pack/`: 原图叠加节点、关系、错误类别、置信度。

反馈闭环：

```text
locked eval failures
-> error attribution
-> hard-case mining
-> dataset builder
-> expert training
-> calibration
-> relation policy retraining
-> locked eval
```

## 端到端数据流

### Step 1: Input ingestion

输入：

```text
file path / bytes / page image
```

处理：

- 解码图像。
- 转成 page pixel coordinate frame。
- 记录 source integrity。

输出：

```text
PageImage
```

### Step 2: Raster feature extraction

输入：

```text
PageImage
```

处理：

- 生成多尺度图像。
- 提取二值图、边缘图、连通域、局部密度、patch embedding。

输出：

```text
PageRasterFeatures
```

### Step 3: Proposal generation

输入：

```text
PageImage + PageRasterFeatures
```

处理：

- 各 family proposal detector 并行运行。
- 产出高召回候选。
- 保留重复候选和低置信候选，等待后续专家与关系层判断。

输出：

```text
ProposalSet {
  boundary_candidates,
  space_candidates,
  symbol_candidates,
  text_candidates,
  dimension_candidates,
  sheet_candidates
}
```

### Step 4: Expert inference

输入：

```text
ProposalSet + image crops + context windows
```

处理：

- boundary expert 分类并补几何。
- space expert 输出 room mask/polygon/type。
- symbol expert 输出 objectness/type/abstain。
- text expert 输出 localization/OCR/normalized text。
- dimension expert 输出尺寸结构。
- sheet expert 输出标题栏、图例、表格区域。

输出：

```text
ExpertCandidateSet
```

### Step 5: Routing and calibration

输入：

```text
ExpertCandidateSet + proposal evidence + page context
```

处理：

- 对每个候选做 route decision。
- 校准 objectness、type、OCR、relation readiness。
- 低置信 typed label 或 OCR 进入 abstain，不污染最终语义。

输出：

```text
RoutedCandidateSet
```

### Step 6: Relation candidate construction

输入：

```text
RoutedCandidateSet
```

处理：

- 构建候选关系图。
- 用空间索引减少全连接爆炸。
- 为每条关系提取几何、视觉、上下文和 cluster features。

输出：

```text
RelationCandidateSet
```

### Step 7: Relation-aware compression

输入：

```text
RelationCandidateSet
```

处理：

- 按关系类型调用专门 topology policy。
- `contains_symbol` 使用 joint room-symbol assignment。
- `bounded_by` 保持房间边界闭合和墙体支持。
- `adjacent_to` 保持共享边界或门洞证据。
- `labeled_by_text` 依赖高质量 OCR 和 room-text alignment。
- 在保证 recall 的前提下降低重复边和候选洪水。

输出：

```text
CompressedRelationSet
```

### Step 8: Scene graph fusion

输入：

```text
RoutedCandidateSet + CompressedRelationSet
```

处理：

- 节点去重和类型仲裁。
- 关系去重和 schema 检查。
- 约束校验和 fail-closed 输出。
- 生成 audit trace。

输出：

```text
SceneGraph
```

### Step 9: Export and review

输入：

```text
SceneGraph + candidates + relations + audit
```

处理：

- 写 JSONL。
- 渲染 overlay。
- 输出指标和错误归因。

输出：

```text
machine-readable scene graph
visual review pack
locked evaluation reports
```

## 训练数据流

训练阶段可以读取离线标签，但要和推理路径隔离。

```text
raster image + offline labels
-> dataset builders
-> crop/mask/relation samples
-> expert training
-> calibration and threshold selection
-> locked evaluation
-> checkpoints
```

每个训练样本必须记录：

- label 来源。
- 对应 raster crop 或 page region。
- 是否用于 train/dev/locked。
- 是否参与 threshold selection。
- 是否可能泄漏标注 ID。

训练产物必须通过 source integrity gate：

- checkpoint 可以携带 learned weights。
- checkpoint 不得携带 locked page 的标注对象 ID。
- inference config 不得引用 gold geometry 文件。

## 当前 v18 和理想态的关键差距

当前已有基础：

- raster-only source integrity 可以校验。
- high-recall detector adapter 已能产生 space、boundary、symbol 等候选。
- relation reranker 可以显著压缩候选数量。
- visual hard case pack 可以辅助诊断。

距离理想态的主要缺口：

- `contains_symbol` 需要 joint room-symbol assignment，不能继续依赖硬 NMS、固定 cap 或 symbol-only splitter。
- symbol 当前主要是 objectness，缺少可靠 typed symbol model。
- OCR/text localization 和 recognition 都不足以支撑 `labeled_by_text`。
- final refiner 只有在上游 recall-preserving compression 解决后才有意义。
- 0.98 目标要求每个 family 和 relation 都有独立模型、校准、abstain 和 locked evaluation，不是单个后处理脚本能补出来的。

## 理想质量门槛

生产采用前，核心 locked 指标应达到：

| Area | Minimum |
| --- | --- |
| space detection recall | 0.98 |
| boundary detection recall | 0.98 |
| symbol detection recall | 0.98 |
| symbol type precision/recall/F1 | 0.98 |
| text localization precision/recall | 0.98 |
| OCR normalized accuracy | 0.98 |
| `bounded_by` precision/recall | 0.98 |
| `contains_symbol` precision/recall | 0.98 |
| `adjacent_to` precision/recall | 0.98 |
| `labeled_by_text` precision/recall | 0.98 |
| final scene graph precision/recall | 0.98 |
| source integrity violations | 0 |

中间阶段可以使用较低的工程 gate 来验证方向，但任何未达到 0.98 的模块都不能被描述为生产可用。

## 推荐文件边界

理想工程拆分如下：

```text
configs/vlm/
  image_only_moe_contract_v1.json
  cadstruct_ontology.json
  detector_output_schema_v18.json

scripts/vlm/
  normalize_raster_page_v18.py
  train_boundary_detector_v18.py
  train_room_space_detector_v18.py
  train_symbol_objectness_v18.py
  train_symbol_type_model_v18.py
  train_text_detector_v18.py
  train_ocr_recognizer_v18.py
  build_topology_relations_v18.py
  train_relation_reranker_v18.py
  train_contains_symbol_joint_policy_v18.py
  fuse_scene_graph.py
  scene_graph_schema.py

reports/vlm/
  detector_adapter_v18_*.jsonl
  topology_relations_v18_*.jsonl
  *_eval.json
  *_audit.json
  visual_hard_cases_v18/

checkpoints/
  boundary_detector_v18/
  room_space_detector_v18/
  symbol_objectness_v18/
  symbol_type_model_v18/
  text_detector_v18/
  ocr_recognizer_v18/
  relation_reranker_v18/
  contains_symbol_joint_policy_v18/
```

## One-page summary

理想 CadStruct raster-only MoE 是一个从像素到 scene graph 的分层系统：

```text
1. Normalize raster page
2. Generate high-recall family proposals
3. Run family-specific experts
4. Route, calibrate, and abstain unsafe claims
5. Build relation candidates
6. Apply relation-aware topology compression
7. Fuse schema-valid scene graph
8. Export JSON, visual review, audits, and locked metrics
```

它的核心原则是：推理只看像素；专家分工明确；类型、OCR、关系都要独立校准；压缩必须 relation-aware；所有输出必须能追溯到像素证据和模型决策。
