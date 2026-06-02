# CadStruct MoE 真实场景覆盖与 gap 分析（版本 2026-05-01）

## 1. 当前可宣称能力（按专家模块）

### WallOpening（核心结构主干）
- **已闭环**：在论文 split 的严格 locked-test 上，boundary 家族指标可达 98%+（acc 0.9926, macro-F1 0.9885，R2 0.9801）。
- **可复现资产**：`checkpoints/cadstruct_graph_node_crop_gnn_h512_c32_ms3_l2_paper_v2_source_raster_e28/model_best.pt`，报告为 `reports/vlm/paper_v2_validation_summary.json`。
- **可直接用于当前主指标之一**：边界识别主任务（严格）。

### RoomSpace（端到端尚未闭环）
- **当前表现**：在 gold room polygon 条件下，分类准确率较高（acc 0.9868，macro-F1 0.9821）。
- **问题**：不是端到端房间识别（缺 proposal recall / mask 入口质量评估）。
- **后果**：论文主陈述不能把当前结果写成“端到端 room 识别”。

### SymbolFixture
- **当前表现**：crop+crop-mlp 相比 bbox prototype 有明显提升，但 dev macro-F1 仍偏低（约 0.61）。
- **问题**：long-tail 类别（如 `bathtub`、`generic_symbol`）不足，真实场景风格多样性覆盖不足。

### TextDimension
- **当前表现**：比 bbox 原型有提升，dimension link-F1 有进步。
- **问题**：`note_text` 与 OCR 语义抽取仍不足；未形成可用于 SCI 级文本理解主线的完整链路。

### Scene Graph（MoE 融合层）
- **当前状态**：已有 smoke 融合与审计（`reports/vlm/moe/fused_scene_graph_smoke_audit.json`），但尚未形成生产级 end-to-end locked 一体验证。
- **主要问题**：边界/房间约束外推及关系 F1 在真实场景仍缺乏多源验证。

## 2. 真实场景覆盖差距（已确认）

1. **Room pipeline 断点**
   - 当前 room-space 主要是 gold 条件分类，没有 `room proposal` recall、mask/polygon 质量、NMS/false-positive 控制的严谨闭环。

2. **符号类别泛化断层**
   - 真实图纸中 fixture/equipment/furniture 风格差异大，现有分类/定位对长尾类别支持不足。

3. **文本语义链断层**
- 目前文本链路偏向结构性分类，缺少 OCR+数值值抽取+错误传播控制。

4. **跨域泛化仍未可用于主 claim**
   - Wall/opening 的 source-aware 迁移性能在 source train/source test 不同方向上崩坏，说明缺少更强的真实域泛化协议。

5. **scene graph 评估边界混淆风险**
   - 目前缺少统一的统一可审计 contract 与跨 source 的 locked 级别融合指标，不能把所有 warning 当作错误或当作提升证据。

## 3. 建议的指标口径（避免混淆）

### 已经可以报告为“论文可引用”：
- WallOpening 主指标（严格 locked-test 的 acc/macro-F1/R2）。
- RoomSpace 的 **gold-candidate** room type 分类（严格与 ambiguity-adjusted）。
- Symbol/Text 的 dev/smoke 对比（作为“原型线”而非最终论文线）。
- 零样本模型：仅 smoke-only baseline。

### 不能与论文主线混用：
- 所有总数低于 30 的结果；
- 未区分 gold-candidate 与 end-to-end pipeline 的 room 指标；
- 未提供 source-specific locked 报告的跨源泛化 claim；
- scene graph warnings 统计替代完整 relation-F1。

## 4. 下一步优先级（针对“真实场景闭环”）

1. **补齐 Room proposal 与 proposal recall**
   - 在真实图纸 split 上报告 recall@0.3/0.5/0.75 与 false-positive rate。

2. **符号域增强**
   - 用更接近真实场景的 sub-manifolds 进行长尾再训练与主观类补齐（generic/special fixture 家族）。

3. **文本/OCR 结构化链路**
   - 将 room label 与 dimension line 作为可分任务训练，并输出关系 F1 与 parse 失败审计。

4. **scene graph contract 稳定化**
- 统一 contract（node/edge source_expert/confidence/geometry/audit_trace），并把 warnings 分为 recoverable/fatal。

5. **可发表 claim 的三源验证**
   - 至少三源 locked 一致口径对比：CubiCasa5K / FloorPlanCAD / CVC-FP，并明确 failure mode。

## 5. 文档与证据索引

- 能力矩阵：`reports/vlm/real_world_capability_matrix.json`
- 训练/评估清单：`todo.json` 中 P0.1/P0.2 任务条目与执行状态
- 融合审计：`reports/vlm/moe/fused_scene_graph_smoke_audit.json`
- Split 泄漏审计：`reports/vlm/all_split_leakage_audit.json`（本次补充）
