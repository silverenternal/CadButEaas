# CadStruct 文档入口

最后更新: 2026-05-11

本目录只放 CadStruct 图纸识别模型相关文档。根目录 `ARCHITECTURE.md` 描述 CAD EaaS/Rust 服务架构；这里描述 Python/VLM/MoE 图纸识别实验线。

## 当前主线

| 主线 | 状态 | 权威说明 | 机器索引/执行入口 |
|---|---|---|---|
| CubiCasa/SVG 派生强专家和结构化 MoE | canonical historical baseline | [legacy-cubicasa-moe.md](legacy-cubicasa-moe.md) | [configs/vlm/cadstruct_legacy_moe_registry.json](../../configs/vlm/cadstruct_legacy_moe_registry.json) |
| CubiCasa/SVG 输入到 scene graph 完整案例 | reproducible canonical case | [cubicasa-svg-case.md](cubicasa-svg-case.md) | [reports/vlm/cubicasa_svg_case/case_manifest.json](../../reports/vlm/cubicasa_svg_case/case_manifest.json) |
| 模型资产、训练清单和指标台账 | current inventory | [current/model-asset-inventory.md](current/model-asset-inventory.md) | [configs/vlm/cadstruct_legacy_moe_registry.json](../../configs/vlm/cadstruct_legacy_moe_registry.json) |
| 非 SVG/raster-only MoE 重建 | active experimental rebuild | [current/cadstruct-raster-moe-ideal-architecture-v18.md](current/cadstruct-raster-moe-ideal-architecture-v18.md) | [struct.json](../../struct.json), [todo.json](../../todo.json) |
| Scene graph 数据合同 | current contract | [current/cadstruct-scene-graph-schema.md](current/cadstruct-scene-graph-schema.md) | `scripts/vlm/scene_graph_schema.py` |
| 论文与 claim 边界 | paper evidence | [paper/cadstruct-paper-core-contributions-v2.md](paper/cadstruct-paper-core-contributions-v2.md) | [paper/real-world-capability-boundary-v3.md](paper/real-world-capability-boundary-v3.md) |

## 子目录

| 子目录 | 用途 |
|---|---|
| `current/` | 当前仍参与理解架构、合同、质量控制的文档 |
| `paper/` | 论文主张、贡献、能力边界、MoE 定位 |
| `runbooks/` | v7-v11 训练和架构复跑说明 |
| `archive/` | v14-v17、advisor、roadmap、阶段性结论、失败实验记录 |

## 判断规则

- 历史强专家和 raster-only 失败实验必须分开判断。强专家证明专家路线有效；当前低 precision 主要暴露 raster proposal、symbol/text localization 和 relation compression 问题。
- 当前训练过哪些模型、哪些指标强、哪些指标弱，以 [current/model-asset-inventory.md](current/model-asset-inventory.md) 为准。
- `struct.json` 是当前架构和数据流机器可读说明。
- `todo.json` 是当前执行计划和真实状态。
- `docs/cadstruct/archive/` 内文档只作追溯，不能直接当作当前主线结论。
- 新增专家或指标时，必须同步脚本、报告、registry 或 `todo.json`，不要只写散文档。
