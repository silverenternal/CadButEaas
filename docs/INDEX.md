# 项目文档索引

**最后更新**: 2026 年 4 月 29 日  
**项目版本**: v0.1.0  
**前端版本**: v0.2.0

本索引用于快速定位项目文档。根目录文档保留项目级入口，`docs/` 放专项说明和评审材料，`docs/archive/` 保留历史阶段报告。

## 先读什么

| 目标 | 推荐文档 |
|------|----------|
| 快速了解项目能力和运行方式 | [README.md](../README.md) |
| 理解服务拆分、处理流水线和数据流 | [ARCHITECTURE.md](../ARCHITECTURE.md) |
| 对接后端 HTTP/WebSocket API | [API.md](../API.md)、[后端 API 概览.md](后端%20API%20概览.md) |
| 启动或开发 Web 前端 | [cad-web/README.md](../cad-web/README.md) |
| 跑测试、贡献代码、提交 PR | [CONTRIBUTING.md](../CONTRIBUTING.md) |
| 做性能回归或基准测试 | [BENCHMARK_GUIDE.md](../BENCHMARK_GUIDE.md) |
| 准备验收或甲方评审 | [功能介绍.md](功能介绍.md)、[交付目标对照表.md](交付目标对照表.md) |

## 项目级文档

| 文档 | 内容 | 主要读者 |
|------|------|----------|
| [README.md](../README.md) | 项目介绍、能力边界、快速开始、架构概览、模块说明 | 所有读者 |
| [ARCHITECTURE.md](../ARCHITECTURE.md) | EaaS 设计、服务职责、调用链、核心数据结构 | 开发者、架构评审 |
| [API.md](../API.md) | 后端 API、WebSocket、错误处理、配置接口 | 集成开发者 |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | 开发流程、测试命令、提交规范、代码风格 | 贡献者 |
| [CHANGELOG.md](../CHANGELOG.md) | 版本变更记录 | 维护者、用户 |
| [todo.md](../todo.md) | 技术路线图和未完成事项 | 维护者 |
| [TEST_FILES.md](../TEST_FILES.md) | 测试样例文件说明 | 测试人员 |
| [BENCHMARK_GUIDE.md](../BENCHMARK_GUIDE.md) | Rust/前端性能基准、KPI、回归检测 | 性能负责人 |

## 运行与配置

| 文档或文件 | 内容 |
|------------|------|
| [cad_config.example.toml](../cad_config.example.toml) | 完整配置示例 |
| [cad_config.profiles.toml](../cad_config.profiles.toml) | architectural、raster 等预设配置 |
| [scripts/manage.sh](../scripts/manage.sh) | 启停前后端服务，默认后端 3000、前端 5173 |
| [scripts/start.sh](../scripts/start.sh) | 项目启动脚本 |
| [monitoring/README.md](../monitoring/README.md) | Prometheus/Grafana 监控栈启动和面板说明 |

常用命令：

```bash
cargo run --package cad-cli -- serve --port 3000

cd cad-web
pnpm install
pnpm dev

./scripts/manage.sh start
./scripts/manage.sh status
./scripts/manage.sh stop
```

## 后端与算法文档

| 文档 | 内容 |
|------|------|
| [后端 API 概览.md](后端%20API%20概览.md) | 后端接口速查，适合对接前快速浏览 |
| [功能介绍.md](功能介绍.md) | 面向技术评审的功能说明 |
| [DXF_PARSE_RENDER_RECOMMENDATIONS.md](DXF_PARSE_RENDER_RECOMMENDATIONS.md) | DXF 解析与渲染落地建议 |
| [multimodal-vlm-plan.md](multimodal-vlm-plan.md) | 光栅图纸多模态 VLM 后端、schema、fallback 和 QLoRA smoke 路线 |
| [14b-structured-vlm-training-plan.md](14b-structured-vlm-training-plan.md) | 14B 级结构增强 VLM 训练方案、前沿模型调研和论文实验路线 |
| [crates/cad-viewer/README.md](../crates/cad-viewer/README.md) | egui 查看器说明 |
| [benches/README.md](../benches/README.md) | 基准测试数据说明 |

核心 Rust workspace 模块：

| Crate | 职责 |
|-------|------|
| `common-types` | 公共几何、请求响应、错误、场景类型 |
| `service-kit` | 服务 trait、健康检查、指标采集、服务治理工具 |
| `scene-builder` | RawEntity/Polyline 到 SceneState 片段的转换、边提取、标注摘要 |
| `parser` | DXF/DWG/PDF/SVG/STL 解析 |
| `vectorize` | 光栅图像和扫描图纸矢量化；`RasterSemanticExtractor` 薄封装 OCR/VLM 尺寸解析、符号检测、图元拟合和语义候选 |
| `topo` | 拓扑构建、Halfedge、空间索引 |
| `validator` | 几何合法性校验 |
| `export` | JSON/Bincode/SVG 导出 |
| `orchestrator` | API 网关和处理流程编排；API DTO、上传解析、边响应适配、交互/WebSocket/导出 handler 与流水线辅助已拆入 `api/`、`pipeline/` 子模块 |
| `interact` | 选边、圈选、缺口检测等交互能力 |
| `acoustic` | 声学分析、材料统计、RT60 计算 |
| `config` | 配置加载和预设管理 |
| `cad-cli` | 命令行和 HTTP 服务入口 |
| `cad-viewer` | Rust egui 查看器 |
| `raster-loader` | PNG/JPG/BMP/TIFF/WebP 加载 |
| `accelerator-*` | CPU/wgpu 加速器抽象和实现 |
| `vector-graph` | 图结构特征和 GNN 训练实验模块 |

## 多模态 VLM

| 路径 | 内容 |
|------|------|
| [multimodal-vlm-plan.md](multimodal-vlm-plan.md) | Rust schema、候选融合规则、fallback 和评估计划 |
| [14b-structured-vlm-training-plan.md](14b-structured-vlm-training-plan.md) | 14B 级结构增强 VLM 训练方案 |
| [cadstruct-model-design.md](cadstruct-model-design.md) | CadStruct 自有结构模型设计、LoRA 边界和节点分类器结果 |
| [cadstruct-paper-readiness.md](cadstruct-paper-readiness.md) | CadStruct 论文创新点、SCI Q2 就绪度、缺失验证和投稿叙事 |
| [cadstruct-moe-dataset-roadmap.md](cadstruct-moe-dataset-roadmap.md) | MoE 扩展所需图纸数据集、专家划分和数据优先级 |
| [cadstruct-moe-architecture-plan.md](cadstruct-moe-architecture-plan.md) | 面向图纸识别的 CadStruct-MoE 架构、训练、融合、指标和消融规划 |
| [cadstruct-moe-98-metric-recovery-plan.md](cadstruct-moe-98-metric-recovery-plan.md) | CadStruct-MoE 指标拉升到 98%+ 的系统性攻关计划、阶段目标和风险边界 |
| [datasets.md](datasets.md) | 外部图纸数据集下载位置、规模、用途和续传命令 |
| [scripts/vlm/README.md](../scripts/vlm/README.md) | Python sidecar、mock backend、数据生成和 LoRA smoke 命令 |
| [configs/vlm/](../configs/vlm/) | 默认 HTTP sidecar 与 QLoRA smoke 配置 |

## Web UI 文档

| 文档 | 内容 |
|------|------|
| [cad-web/README.md](../cad-web/README.md) | 前端启动、技术栈、目录结构、测试命令 |
| [web-ui-index.md](web-ui-index.md) | Web UI 文档入口 |
| [web-ui-api-integration.md](web-ui-api-integration.md) | 前端与后端 API 集成规范 |
| [web-ui-component-spec.md](web-ui-component-spec.md) | Web UI 组件规范 |
| [web-ui-migration-plan.md](web-ui-migration-plan.md) | Web UI 迁移计划 |
| [cad-web/IMPLEMENTATION.md](../cad-web/IMPLEMENTATION.md) | 前端实现说明 |
| [cad-web/IMPLEMENTATION_SUMMARY.md](../cad-web/IMPLEMENTATION_SUMMARY.md) | 前端实现摘要 |
| [cad-web/UPLOAD_TEST_GUIDE.md](../cad-web/UPLOAD_TEST_GUIDE.md) | 上传测试指南 |

前端常用命令：

```bash
cd cad-web
pnpm dev
pnpm build
pnpm lint
pnpm test
pnpm test:e2e
```

## 演示、样例与测试资产

| 路径 | 内容 |
|------|------|
| [demo/](../demo/) | 演示页面、E2E demo、演示视频脚本 |
| [examples/](../examples/) | WebSocket demo、DXF debug 示例 |
| [dxfs/](../dxfs/) | DXF 测试文件和问题样例 |
| [testpdf/](../testpdf/) | PDF 测试文件 |
| [fuzz/](../fuzz/) | fuzz target 和 fuzz 配置 |
| [checkpoints/](../checkpoints/) | GNN/训练检查点和训练指标 |

## 验收与交付

| 文档 | 内容 |
|------|------|
| [交付目标.md](../交付目标.md) | 原始交付目标文档，作为验收基准 |
| [交付目标对照表.md](交付目标对照表.md) | 交付目标与系统实现的映射 |
| [功能介绍.md](功能介绍.md) | 面向甲方技术评审的功能介绍 |
| [demo/演示视频脚本.md](../demo/演示视频脚本.md) | 演示录制和讲解脚本 |

## 历史归档

`docs/archive/` 中的文档是阶段性报告、旧专项说明或已整合内容。归档文档用于追溯，不作为当前实现的唯一依据。

| 文档 | 说明 |
|------|------|
| [FUZZ_TESTING.md](archive/FUZZ_TESTING.md) | 模糊测试框架指南 |
| [GAP_ANALYSIS_REPORT.md](archive/GAP_ANALYSIS_REPORT.md) | 商业化 CAD 差距分析报告 |
| [limitations-and-p2-plan.md](archive/limitations-and-p2-plan.md) | 系统限制与 P2 计划 |
| [MIRI_GUIDE.md](archive/MIRI_GUIDE.md) | Miri 内存检测指南 |
| [OPENCV_USAGE.md](archive/OPENCV_USAGE.md) | OpenCV 使用说明，核心内容已整合到 README |
| [P0-7_ROBUST_GEOMETRY_REPORT.md](archive/P0-7_ROBUST_GEOMETRY_REPORT.md) | 稳健几何内核阶段报告 |
| [P0_GPU_RENDER_REPORT.md](archive/P0_GPU_RENDER_REPORT.md) | GPU 渲染阶段报告 |
| [P1-3_SPATIAL_INDEX_REPORT.md](archive/P1-3_SPATIAL_INDEX_REPORT.md) | 空间索引阶段报告 |
| [P1_COMPLETION_REPORT.md](archive/P1_COMPLETION_REPORT.md) | P1 完成报告 |
| [PDF 矢量化说明.md](archive/PDF%20矢量化说明.md) | PDF 矢量化历史说明 |
| [qa-preparation.md](archive/qa-preparation.md) | 验收 Q&A 预案 |
| [test_report.md](archive/test_report.md) | 测试报告 |

## 文档维护规则

1. 项目级入口放在根目录，专项说明放在 `docs/`，阶段报告放在 `docs/archive/`。
2. 同一事实只保留一个权威来源，其他文档用链接引用。
3. 修改接口、配置、命令或目录结构时，同步更新相关文档和本索引。
4. 历史报告不直接删除；过时后移动到归档并在索引中标注。
5. 新增面向用户或对接方的能力时，至少更新 `README.md`、`API.md` 或对应专项文档之一。
