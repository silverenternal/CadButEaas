# 项目文档索引

最后更新: 2026-05-11

本目录现在按用途分层。根目录只保留这个总索引和少量跨模块资料；专项文档进入子目录；阶段性 CadStruct 实验报告进入 `docs/cadstruct/archive/`。

## 先读什么

| 目标 | 推荐入口 |
|---|---|
| 了解项目整体能力和启动方式 | [README.md](../README.md) |
| 了解服务架构和 Rust/CAD EaaS 主链路 | [ARCHITECTURE.md](../ARCHITECTURE.md) |
| 对接 API | [API.md](../API.md), [product/后端 API 概览.md](product/后端%20API%20概览.md) |
| 准备交付或评审 | [product/功能介绍.md](product/功能介绍.md), [product/交付目标对照表.md](product/交付目标对照表.md) |
| 开发 Web UI | [web-ui/README.md](web-ui/README.md) |
| CadStruct 图纸识别/MoE | [cadstruct/README.md](cadstruct/README.md) |
| 清点 CadStruct 模型资产和指标 | [cadstruct/current/model-asset-inventory.md](cadstruct/current/model-asset-inventory.md) |
| 外部数据集资产 | [datasets.md](datasets.md) |
| VLM/大模型训练调研 | [research/README.md](research/README.md) |

## 目录划分

| 目录 | 内容 | 权威性 |
|---|---|---|
| `product/` | 产品功能、后端 API 概览、DXF 解析渲染建议、交付对照 | 当前可读入口 |
| `web-ui/` | Web UI 迁移、组件、API 集成文档 | 当前前端专项 |
| `cadstruct/` | 图纸识别/MoE 文档入口、当前结构、论文资料、历史实验归档 | CadStruct 权威入口 |
| `research/` | raster VLM、14B 结构化训练等调研/训练计划 | 研究计划 |
| `archive/` | 非 CadStruct 的历史阶段报告 | 只作追溯 |

## CadStruct 当前权威入口

CadStruct 已拆为三层：

| 子目录 | 内容 |
|---|---|
| [cadstruct/README.md](cadstruct/README.md) | CadStruct 总入口，说明 canonical/active/archive 边界 |
| `cadstruct/current/model-asset-inventory.md` | 当前模型资产、训练清单、优秀指标和待提升指标 |
| `cadstruct/current/` | 当前仍有参考价值的模型结构、scene graph 合同、raster-only 理想架构、质量 runbook |
| `cadstruct/paper/` | 论文贡献、claim boundary、real-world capability boundary、domain-structured MoE 定位 |
| `cadstruct/runbooks/` | v7-v11 历史训练/架构复跑说明 |
| `cadstruct/archive/` | v14-v17、advisor、roadmap、失败或阶段性实验说明 |

当前工程执行仍以根目录 [struct.json](../struct.json) 和 [todo.json](../todo.json) 为准；文档只负责解释资产边界和历史背景。

## 维护规则

1. 新文档先判断读者：产品/API/Web/CadStruct/研究/历史，不再直接堆到 `docs/` 顶层。
2. CadStruct 相关文档必须从 `docs/cadstruct/README.md` 能找到。
3. 失败实验、阶段报告、旧 roadmap 不删除，移动到 `archive/` 并在入口说明“只作追溯”。
4. 同一事实只保留一个当前权威入口，其他文档链接过去。
5. 修改路径后，同步更新 `README.md`、`docs/INDEX.md`、`docs/cadstruct/README.md` 中的入口链接。
