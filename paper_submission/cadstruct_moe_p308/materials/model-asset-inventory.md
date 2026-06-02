# CadStruct Model Asset Inventory

最后更新: 2026-05-11

本文是图纸识别模型任务的当前资产台账。它回答四个问题：

- 我们训练过哪些模型。
- 这些模型属于整体工作架构的哪一层。
- 哪些指标还能看，哪些指标需要提升。
- 哪些资产可以复用，哪些只是失败证据或诊断资产。

## 项目目标边界

当前目标是 **非 SVG / raster-only 图纸识别 MoE**：推理输入只能是 PNG/JPG/扫描图/PDF 栅格页等像素图，不允许读取 SVG、DXF、PDF 矢量对象、`expected_json`、人工标注几何或数据集转换时的真实对象 ID。

历史 CubiCasa/SVG 派生专家仍然有价值，但它们是 **candidate/contract-level 专家**，不能直接等同于“全图 raster detector 已经完成”。正确关系是：

```text
raster image
  -> page normalizer
  -> high-recall raster proposal frontend
  -> candidate contract adapter
  -> reusable family experts
  -> deterministic MoE router / fusion
  -> relation/topology policy
  -> schema-valid scene graph + audit reports
```

当前低 precision/recall 主要来自 raster proposal、symbol/text localization、relation compression，而不是所有历史专家本体失效。

## 整体 Work 架构

| 层级 | 职责 | 当前状态 | 关键资产 |
|---|---|---|---|
| 数据转换 | CubiCasa/FloorPlanCAD/internal 数据转成专家训练视图 | CubiCasa/MoE 已成型；raster-only 仍在重建 | `datasets/cadstruct_cubicasa5k_moe`, `datasets/cadstruct_cubicasa5k_moe_locked`, `datasets/public_raster_moe_supervision_v19` |
| 高召回候选 | 从 raster 全图生成 boundary/symbol/text/room candidates | boundary proposal 已接近可用；symbol/text/room 仍弱 | `datasets/boundary_public_raster_v24_yolo_full`, YOLO v24/v22 reports |
| 专家模型 | boundary、room、symbol、text、graph-node 等局部专家 | 部分强，symbol/text-localizer 弱 | checkpoints 和 reports 见下表 |
| MoE 路由 | typed candidate stream -> family expert | deterministic router 强；learned router 弱 | `reports/vlm/domain_structured_moe_route_audit_v1.json` |
| 融合/关系 | 节点压缩、关系预测、topology policy | 受上游候选质量限制，暂不应主攻 | `struct.json`, `todo.json` |
| 审计 | 每层输出 source_integrity、指标、错误桶 | 已形成大量脚本和报告，是重要项目资产 | `scripts/vlm/audit_*.py`, `reports/vlm/*.json` |

## 已训练模型资产

### 强资产，可优先复用

| 模型族 | 资产 | 指标 | 结论 |
|---|---|---:|---|
| boundary candidate-level expert | `reports/vlm/boundary_expert_v13_eval.json` | locked accuracy/macro F1 = `1.0` | 强专家，但依赖候选输入，不是 raster 全图 detector |
| graph-node crop/GNN | `checkpoints/cadstruct_graph_node_crop_gnn_h384_c32_ms3_l2_e24/model_best.pt` | dev accuracy `0.991147`, macro F1 `0.986746`; smoke macro F1 `0.986652` | boundary/door/window 节点分类强资产 |
| room/space classifier | `checkpoints/room_space_expert_v13/model.joblib` | locked accuracy `0.986811`, macro F1 `0.982129` | 房间候选分类强资产，room proposal 仍需加强 |
| text/dimension classifier | `text_dimension_expert_v13`, `reports/vlm/text_dimension_expert_v13_eval.json` | locked accuracy `0.998501`, macro F1 `0.967754`; dimension link F1 `0.998342` | 分类强，note_text/room_label 长尾低于 0.98；依赖 text localization |
| symbol visual evidence gate | `checkpoints/symbol_visual_evidence_v8/model.joblib` | locked accuracy `0.992808`, macro F1 `0.991157`, reject recall `1.0` | keep/reject crop gate 强资产，不是 symbol detector |
| deterministic MoE router | `reports/vlm/domain_structured_moe_route_audit_v1.json` | router accuracy `1.0`, wrong_expert_rate `0.0` | typed candidate routing 强资产；learned router 只是消融 |

### 可用但未达 0.98 的资产

| 模型族 | 资产 | 指标 | 问题 |
|---|---|---:|---|
| symbol fixture expert v13 | `checkpoints/symbol_fixture_expert_v13/model.joblib`, `reports/vlm/symbol_fixture_expert_v13_eval.json` | direct-split macro F1 `0.883069` | 历史专家中最弱，长尾符号和类型泛化不足 |
| text/dimension v13 | `reports/vlm/text_dimension_expert_v13_eval.json` | macro F1 `0.967754`, note_text F1 `0.84375` | 总体可用，但未到 0.98；note_text 是弱项 |
| boundary full-YOLO proposal frontend | `runs/detect/runs/detect/runs/vlm/boundary_public_raster_v24_yolo_probe/weights/best.pt` + full tile dataset | full-dev proposal recall `0.980579`; locked50 `0.980474` | proposal recall 可看；typed precision proxy 很低，door/window type recall 不足 |
| symbol YOLO body detector | `runs/detect/runs/vlm/symbol_yolov8n_pretrained_v22_dedup_hi640_probe/weights/best.pt` | center recall `0.911595`, IoU@0.30 recall `0.719572`, precision `0.096685` | body localization 远未到 0.98，候选多、IoU 弱 |

### 失败证据 / 诊断资产

| 模型族 | 资产 | 结果 | 结论 |
|---|---|---:|---|
| boundary door/window tabular specialist | `checkpoints/boundary_door_window_specialist_v24/model.joblib` | locked50 classified recall `0.913043` | 未超过 fusion/hint baseline，不推广 |
| boundary crop/context CNN | `checkpoints/boundary_door_window_crop_context_v24/model.pt` | locked50 classified recall `0.912523` | crop-only 路线弱 |
| boundary fail-closed ResNet18 | `checkpoints/boundary_door_window_crop_context_failclosed_v24/model.pt` | default classified `0.907316`; sweep best `0.915126` | door/window 有弱正向，但掉 hard_wall/overall，不能 promoted |
| boundary context type policy smoke | `checkpoints/boundary_context_type_policy_v24_smoke/model.joblib` | locked10 `0.914718 -> 0.918845`, window `0.916667 -> 0.958333`, door `0.843750` | 有弱正向，但 full locked50 训练链路被 feature-cache 缺失阻塞 |
| symbol ConvNeXt proposal type head | `checkpoints/symbol_crop_context_pretrained_v20_convnext_tiny_finetune/model.pt` | real proposal typed accuracy `0.743429` | oracle crop type head不能直接迁移到 noisy detector proposal |
| raster text localizer v19 | `checkpoints/text_expert_v19/model_best.pt` | dev text IoU recall `0.0`, center recall `0.123539` | text localization 基本不可用，必须重建 |

## 当前优秀指标

这些指标仍然有参考价值，但必须按边界解释。

| 指标 | 数值 | 边界 |
|---|---:|---|
| boundary expert v13 locked macro F1 | `1.0` | 候选/结构化输入专家本体 |
| graph-node crop/GNN dev macro F1 | `0.986746` | node/crop/proposal 输入，不是全图 detector |
| room_space v13 locked macro F1 | `0.982129` | room candidate-level classifier |
| symbol_visual_evidence_v8 locked macro F1 | `0.991157` | crop keep/reject gate |
| text_dimension_v13 locked accuracy | `0.998501` | text/dimension candidate classifier |
| deterministic structured router wrong_expert_rate | `0.0` | typed candidate stream routing |
| boundary full-YOLO full-dev proposal recall | `0.980579` | raster boundary proposal recall，类型仍不足 |
| boundary full-YOLO locked50 proposal recall | `0.980474` | same-frontend locked subset |

## 当前待提升指标

| 模块 | 当前指标 | 目标 | 需要大改的点 |
|---|---:|---:|---|
| boundary type repair | full-dev typed_hint `0.954223`; door `0.867544`; window `0.927282` | `>=0.98` | 不再盲训 crop-only；先做 boundary context feature cache，再训练 fail-closed type policy |
| boundary precision/compression | full-dev typed_precision_proxy `0.081281`; locked50 `0.091357` | 接近实际可用 precision | 候选压缩必须 relation/topology aware，不能硬 NMS/cap 伤 recall |
| symbol body detector | center recall `0.911595`; IoU recall `0.719572`; precision `0.096685` | recall/precision/F1 `>=0.98` | 需要真正小目标/符号 detector 或 segmentation/heatmap，不是只接 oracle type head |
| symbol type on detector proposals | typed accuracy `0.743429` | `>=0.98` | type head 必须在 detector-proposal 分布上训练/校准 |
| text localization | IoU recall `0.0`; center recall `0.123539` | `>=0.95` localization recall | 需要 OCR/text detector 重建；text_dimension 专家本体不是瓶颈 |
| room proposal | 无稳定 raster room proposal 指标 | `>=0.95` proposal recall | 需要从 boundary mask/segment/topology 生成 room polygons，再接 room_space_v13 |
| relation/topology final precision | 曾接近 `0.003` | `>=0.98` | 上游节点不稳时优化 relation 无意义；待 node/proposal 达标后训练 listwise policy |

## 资产使用决策

| 决策 | 说明 |
|---|---|
| 不重复训练强专家 | boundary、graph-node、room_space、symbol_visual_evidence、text_dimension 先复用 |
| 先修 raster frontend | 当前端到端差主要不是专家本体，而是从像素到候选合同的转换失败 |
| 不推广弱正向模型 | boundary crop/context、tabular specialist、context smoke 均作为诊断，不进入生产主线 |
| 保留审计脚本 | `scripts/vlm/audit_*.py`、candidate stream、error buckets 是重要项目资产 |
| 下一步 P0 | 为 boundary context policy 构建离线 feature cache，再训练 fail-closed policy |

## 关键入口

| 用途 | 路径 |
|---|---|
| 当前架构机器说明 | `struct.json` |
| 当前执行计划 | `todo.json` |
| 历史强专家 registry | `configs/vlm/cadstruct_legacy_moe_registry.json` |
| 历史强资产说明 | `docs/cadstruct/legacy-cubicasa-moe.md` |
| direct-split 专家审计 | `reports/vlm/cadstruct_direct_split_expert_audit.json` |
| raster-only 理想架构 | `docs/cadstruct/current/cadstruct-raster-moe-ideal-architecture-v18.md` |
| boundary v24 full-dev audit | `reports/vlm/boundary_public_raster_v24_yolo_full_dev493_proposal_audit.json` |
| boundary v24 locked50 audit | `reports/vlm/boundary_public_raster_v24_yolo_full_locked50_proposal_audit.json` |
