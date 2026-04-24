# 项目文档索引

**最后更新**: 2026 年 4 月 15 日
**版本**: v0.1.0

---

## 📚 核心文档

这些是项目的主要文档，保持最新和准确：

| 文档 | 说明 | 目标读者 |
|------|------|----------|
| [README.md](../README.md) | 项目介绍、快速开始、功能概览 | 所有用户 |
| [ARCHITECTURE.md](../ARCHITECTURE.md) | 架构设计、服务详解、数据类型 | 开发者、架构师 |
| [CONTRIBUTING.md](../CONTRIBUTING.md) | 贡献指南、代码风格、提交流程 | 贡献者 |
| [todo.md](../todo.md) | 技术路线图、任务规划 | 项目维护者 |
| [CHANGELOG.md](../CHANGELOG.md) | 变更日志、版本历史 | 所有用户 |
| [API.md](../API.md) | 后端 API 完整文档 | 集成开发人员 |
| [TEST_FILES.md](../TEST_FILES.md) | 测试文件说明 | 测试人员 |
| [BENCHMARK_GUIDE.md](../BENCHMARK_GUIDE.md) | 性能基准测试指南 | 性能优化负责人 |

---

## 📋 专项文档

### 配置与示例

| 文档 | 说明 |
|------|------|
| [cad_config.example.toml](../cad_config.example.toml) | 配置文件示例 |
| [cad_config.profiles.toml](../cad_config.profiles.toml) | 预设配置模板 |

### 演示与验收

| 文档 | 说明 |
|------|------|
| [交付目标.md](../交付目标.md) | 甲方交付目标文档（不修改） |
| [交付目标对照表.md](交付目标对照表.md) | 系统实现与交付目标的对照说明 |

### 面向甲方的功能介绍

| 文档 | 说明 | 目标读者 |
|------|------|----------|
| [功能介绍.md](功能介绍.md) | 后端功能详细介绍 | 甲方技术评审 |
| [后端 API 概览.md](后端 API 概览.md) | API 接口快速参考 | 集成开发人员 |
| [交付目标对照表.md](交付目标对照表.md) | 与交付目标的对应关系 | 甲方技术评审 |

### 技术规范

| 文档 | 说明 |
|------|------|
| [DXF_PARSE_RENDER_RECOMMENDATIONS.md](DXF_PARSE_RENDER_RECOMMENDATIONS.md) | DXF 解析与渲染落地建议 |

### Web UI 规范

| 文档 | 说明 |
|------|------|
| [web-ui-api-integration.md](web-ui-api-integration.md) | Web UI API 集成指南 |
| [web-ui-component-spec.md](web-ui-component-spec.md) | Web UI 组件规范 |
| [web-ui-index.md](web-ui-index.md) | Web UI 索引文档 |
| [web-ui-migration-plan.md](web-ui-migration-plan.md) | Web UI 迁移计划 |

---

## 🗄️ 历史文档归档

以下文档已归档到 `docs/archive/` 目录：

| 文档 | 说明 |
|------|------|
| FUZZ_TESTING.md | 模糊测试框架指南 |
| GAP_ANALYSIS_REPORT.md | 商业化 CAD 差距分析报告 |
| limitations-and-p2-plan.md | 系统限制说明与 P2 计划 |
| MIRI_GUIDE.md | Miri 内存检测指南 |
| OPENCV_USAGE.md | OpenCV 使用说明（已整合到 README） |
| P0-7_ROBUST_GEOMETRY_REPORT.md | P0-7 稳健几何内核报告 |
| P0_GPU_RENDER_REPORT.md | P0 阶段 GPU 渲染报告 |
| P1-3_SPATIAL_INDEX_REPORT.md | P1-3 空间索引报告 |
| P1_COMPLETION_REPORT.md | P1 优先级任务完成报告 |
| PDF 矢量化说明.md | PDF 矢量化说明（已整合到 README） |
| qa-preparation.md | 验收 Q&A 预案 |
| test_report.md | 性能回归检测报告 |

**归档原因**: 阶段性报告已完成历史使命，或内容已整合到核心文档中。保留归档供历史参考。

---

## 📁 子 Crate 文档

每个子 crate 可能有自己的 README 文档：

| Crate | 文档 | 说明 |
|-------|------|------|
| `cad-viewer` | [crates/cad-viewer/README.md](../crates/cad-viewer/README.md) | egui 前端说明 |

---

## 🔍 文档用途速查

### 我想了解项目
→ 阅读 [README.md](../README.md)

### 我想了解架构
→ 阅读 [ARCHITECTURE.md](../ARCHITECTURE.md)

### 我想贡献代码
→ 阅读 [CONTRIBUTING.md](../CONTRIBUTING.md)

### 我想了解路线图
→ 阅读 [todo.md](../todo.md)

### 我想查看变更历史
→ 阅读 [CHANGELOG.md](../CHANGELOG.md)

### 我想了解测试文件
→ 阅读 [TEST_FILES.md](../TEST_FILES.md)

### 我想了解性能基准
→ 阅读 [BENCHMARK_GUIDE.md](../BENCHMARK_GUIDE.md)

### 我想查看历史文档
→ 查看 [docs/archive/](archive/) 目录

---

## 📝 文档维护规范

### 更新频率

| 文档 | 更新频率 | 负责人 |
|------|----------|--------|
| README.md | 每个版本 | 维护团队 |
| ARCHITECTURE.md | 重大架构变更时 | 架构师 |
| CONTRIBUTING.md | 贡献流程变更时 | 维护团队 |
| todo.md | 每月或里程碑时 | 项目经理 |
| CHANGELOG.md | 每个版本发布时 | 发布经理 |
| TEST_FILES.md | 添加测试文件时 | 测试负责人 |

### 文档命名规范

- **核心文档**: 大写英文命名（如 `README.md`, `ARCHITECTURE.md`）
- **专项文档**: 小写英文或中文命名（如 `test_report.md`）
- **历史文档**: 中文命名，归档到 `docs/archive/`

### 文档整合原则

1. **避免冗余**: 相同内容只在一个地方详细说明
2. **保持更新**: 过时的文档会误导读者，应及时更新或归档
3. **清晰索引**: 通过文档索引帮助读者快速定位
4. **历史可追溯**: 重要历史文档归档保留，不直接删除

---

## 📊 文档统计

| 类别 | 数量 |
|------|------|
| 核心文档 | 8 |
| 专项文档 | 11 |
| 历史归档文档 | 12 |
| 子 crate 文档 | 1 |
| **总计** | **32** |

---

**维护者**: CAD 项目团队
**联系方式**: https://github.com/your-org/cad/issues
