# CubiCasa/MoE 历史强资产说明

最后更新: 2026-05-10

## 结论

项目中确实已经形成过一条基于 CubiCasa 数据集的多专家 MoE 路线。这条路线的核心成果不是“一个端到端 raster-only 模型已经完成”，而是：

- CubiCasa 数据已经被转换成可训练的结构化专家数据。
- boundary、room/space、text、graph-node 等专家已有多个高指标 checkpoint/report。
- MoE 路由和融合已有 smoke/locked manifest 和审计报告。
- 当前低 precision 的主要来源是后续非 SVG/raster-only 前端、候选生成和关系压缩链路，不是这些历史专家整体失效。

## 数据资产

| 资产 | 角色 | 状态 |
|---|---|---|
| `datasets/cadstruct_cubicasa5k_moe` | CubiCasa 原始 SVG/语义结构转换后的 MoE 数据视图 | canonical |
| `datasets/cadstruct_cubicasa5k_moe_locked` | 按 record 分组的 train/dev/locked/smoke 切分 | canonical |
| `datasets/cadstruct_graph_nodes_lie_topology_raster_v3` | graph-node 专家训练视图，含 topology、Lie/SE(2)、raster crop 特征 | canonical expert dataset |
| `datasets/public_raster_moe_supervision_v19` | 最近 raster-only 重建用的公共监督视图 | experimental rebuild input |
| `datasets/boundary_expert_public_raster_v19` | 最近 raster-only boundary 重建视图 | experimental rebuild input |

## 专家模块

| 专家族 | 典型职责 | 当前可复用资产 | 已知边界 |
|---|---|---|---|
| boundary / wall-opening-window | hard_wall、door/opening、window 等结构节点分类和修正 | `reports/vlm/boundary_expert_v13_eval.json`, `reports/vlm/boundary_geometry_refiner_v7_eval.json` | 强项来自结构化/候选级输入；不能直接等价为原始 raster 全图检测能力 |
| graph-node crop/GNN | primitive/crop graph node -> hard_wall/door/window | `checkpoints/cadstruct_graph_node_crop_gnn_h384_c32_ms3_l2_e24/model_best.pt`, `reports/vlm/graph_node_crop_gnn_h384_c32_ms3_l2_e24_calibrated_dev.json` | 依赖已有 node/crop/proposal，不能替代候选前端 |
| room/space | 房间/空间候选分类 | `reports/vlm/room_space_expert_v13_eval.json`, `checkpoints/room_space_expert_v13/model.joblib` | 候选级 CubiCasa room classification，不是端到端房间 polygon detector |
| symbol | fixture/symbol 候选分类、长尾符号处理 | `reports/vlm/symbol_fixture_expert_v13_eval.json` | 历史指标弱于 boundary/room/text；非 SVG 下仍需要 symbol body/type detector |
| text/dimension | OCR 内容、尺寸文本、文本关系 | `reports/vlm/text_dimension_expert_v13_eval.json`, `reports/vlm/text_dimension_expert_v2_eval.json` | 不是原始 OCR 引擎替代；依赖 text candidate 或 OCR backend |
| router/fusion | typed candidate stream -> 正确专家 -> scene graph | `reports/vlm/domain_structured_moe_route_audit_v1.json`, `reports/vlm/cadstruct_moe_smoke_manifest_v18.json` | deterministic structured router 是主路线；learned router 只是消融 |

## 关键指标证据

| 报告 | 指标摘录 | 解释 |
|---|---:|---|
| `reports/vlm/cadstruct_direct_split_expert_audit.json` | boundary `1.0`, graph-node `0.986746`, room `0.982129`, symbol `0.883069`, text `0.973128` | 专家本体 direct-split 权威审计；不要与集成 smoke 指标混用 |
| `reports/vlm/boundary_expert_v13_eval.json` | locked accuracy/macro F1 = `1.0` | boundary 几何修正专家是历史强资产 |
| `reports/vlm/graph_node_crop_gnn_h384_c32_ms3_l2_e24_calibrated_dev.json` | dev accuracy `0.991147`, macro F1 `0.986746` | 当前 graph-node/crop/GNN 质量路径 |
| `reports/vlm/graph_node_crop_gnn_h384_c32_ms3_l2_e24_calibrated_smoke.json` | smoke accuracy `0.991165`, macro F1 `0.986652` | smoke 与 dev 一致，说明不是单个 split 偶然值 |
| `reports/vlm/expert_contribution_matrix_v13.json` | boundary `1.0`, room `0.982129`, text `0.973128`, symbol `0.883069` | 专家贡献矩阵；symbol 是相对弱项 |
| `reports/vlm/domain_structured_moe_route_audit_v1.json` | deterministic structured router accuracy `1.0` | MoE 路由在 typed candidate 流上是可审计强项 |
| `reports/vlm/cadstruct_moe_locked_manifest_v18.json` | locked gate `fail`; image/relation 指标低 | 失败点在后续 image-only/raster-only 集成，不代表历史专家本身失败 |

## direct-split 指标审计

权威审计入口：

```bash
.venv/bin/python scripts/vlm/audit_direct_split_expert_metrics.py
```

输出：

```text
reports/vlm/cadstruct_direct_split_expert_audit.json
```

该审计只汇总已经存在的专家训练/评估报告，不新造 split、不把 smoke 集成结果当专家本体指标。当前 direct-split 结论：

| 专家族 | direct-split macro F1 | 状态 |
|---|---:|---|
| boundary | `1.0` | strong |
| graph-node crop/GNN | `0.986746` | strong |
| room/space | `0.9821291711858883` | strong |
| text/dimension | `0.973128` | usable, below 0.98 |
| symbol/fixture | `0.883069` | weak; 需要重建/增强 |

`reports/vlm/cubicasa_svg_case/model_fused_scene_graph_locked_smoke_eval.json` 是当前 wrapper/fusion 的集成 smoke 回归，不是专家本体指标。它可以暴露 wrapper 接错、label space 错位和融合链路问题，但不能替代 direct-split 专家评估。

## 标准数据流

```text
CubiCasa SVG/annotation
  -> convert_cubicasa5k_svg.py / split_cubicasa_moe_locked.py
  -> expert dataset views
  -> expert training scripts
  -> checkpoint + eval report + audit report
  -> domain structured router
  -> fusion/export scene graph
  -> manifest + locked audit
```

对于当前非 SVG/raster-only 目标，正确复用方式是：

```text
raster image
  -> raster candidate frontend / pseudo-vector / learned tile detector
  -> convert candidates into the same expert-facing contracts
  -> reuse canonical experts where input contract matches
  -> route/fuse/audit exactly as historical MoE
```

如果某个 raster-only 实验直接在全图上输出一堆候选，且没有转成专家契约，就不能和历史强专家的指标直接比较。

## 开发入口

| 任务 | 推荐入口 |
|---|---|
| 查看历史强资产索引 | `configs/vlm/cadstruct_legacy_moe_registry.json` |
| 运行 smoke manifest 控制点 | `.venv/bin/python scripts/vlm/run_cadstruct_moe_smoke_v18.py` |
| 运行 locked manifest 控制点 | `.venv/bin/python scripts/vlm/run_cadstruct_moe_locked_v18.py` |
| 查看 graph-node 训练/评估命令 | `scripts/vlm/README.md` |
| 查看非 SVG/raster-only 当前架构 | `struct.json` |
| 查看非 SVG/raster-only 当前待办 | `todo.json` |

## 清理原则

现阶段不移动已有 `datasets/`、`reports/`、`checkpoints/` 下的历史产物，因为很多脚本硬编码或引用这些路径。整理方式采用“索引和责任边界优先”：

- canonical: 可作为当前基线或复用资产。
- experimental: 有价值但不是主线的实验。
- diagnostic: 只用于定位问题或证明某条路不行。
- archived: 历史报告，保留追溯价值，不作为当前决策依据。
