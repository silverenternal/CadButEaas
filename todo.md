# CAD 几何智能处理系统 - 技术路线图

**版本**: v0.1.0 (2026 年 2 月 28 日)
**设计哲学**: 一切皆服务 (EaaS)

---

## 📊 当前状态

### 🔥 焚诀循环（2026-04-29 — PNG/JPG 光栅图纸几何语义提取）

| 优化项 | 状态 | 说明 |
|--------|------|------|
| raster-loader 光栅加载层 | ✅ | 支持 PNG/JPG/BMP/TIFF/WebP 等格式检测、加载、DPI 元数据、大小限制与预处理模块 |
| ProcessingPipeline 光栅入口 | ✅ | 新增 `process_raster_file` / `process_raster_bytes`，光栅图片直接进入 Vectorize → Topo → Validator → Export |
| HTTP API 光栅集成 | ✅ | `/process` 自动识别光栅图片，新增专用 `/process/raster` 上传端点 |
| `photo_sketch` 预设 | ✅ | API、配置层、CLI 文案均支持照片/手绘草图场景 |
| CLI 光栅处理 | ✅ | `cad process input.png --profile photo_sketch` 通过 pipeline 自动路由光栅处理 |
| 测试覆盖 | ✅ | 新增 raster-loader 有效/无效/大文件/一致性测试、orchestrator 光栅端到端测试、API 上传测试 |

### ✅ 已完成 (v0.1.0)

| 模块 | 状态 | 测试 | 说明 |
|------|------|------|------|
| DXF 解析 | ✅ | 57 测试 | 支持所有实体类型，NURBS 离散化 |
| PDF 解析 | ✅ | 8 测试 | 矢量/光栅 PDF 支持 |
| 光栅矢量化 | ✅ | 46 测试 | 自动矢量化（线条清晰图纸） |
| 拓扑构建 | ✅ | 28 测试 | R*-tree，O(n log n)，Halfedge 已集成 |
| 交互 API | ✅ | 9 测试 | 后端完成，前端完成 |
| 几何验证 | ✅ | 21 测试 | 恢复建议生成 |
| 场景导出 | ✅ | 2 测试 | JSON/Binary |
| E2E 测试 | ✅ | 6 测试 | 完整流程覆盖 |

**总计**: 347+ 测试（全部通过，0 失败）| Clippy: 0 警告（lib），~20 test/bench 警告已修复

### 📈 模块化优化（2026-04-14 焚诀循环）

| 优化项 | 状态 | 说明 |
|--------|------|------|
| EzdxfParser 实现 DxfParserTrait | ✅ | 纳入 trait 抽象体系，支持缓存包装 |
| ParserService 使用 DxfParserEnum | ✅ | 消除硬编码依赖，支持工厂创建 |
| ezdxf-bridge feature flag | ✅ | `--no-default-features` 编译无 Python 依赖 |
| 统一 ezdxf 错误处理 | ✅ | String → CadError，消除类型转换 |
| 焚诀工作流 API 优化 | ✅ | 每轮 agent 调用 4-6→2 次（~60%节省） |
| 修复 VectorizeFailed 语义误用 | ✅ | tokio join 错误 → InternalError(Panic)，ezdxf 错误 → DxfParseError |
| 新增 DxfParseReason::ParseError | ✅ | 通用解析失败原因枚举，支持 ezdxf 错误上报 |
| 补全 DxfParserEnum::with_layer_filter | ✅ | 所有解析器类型（Sync/Async/Cached/Ezdxf）均支持图层过滤 |
| SyncDxfParser::with_layer_filter | ✅ | builder pattern 链式 API |

### 🔧 解耦优化（2026-04-14 第二轮）

| 优化项 | 状态 | 说明 |
|--------|------|------|
| 声学类型迁移至 acoustic crate | ⛔ 停止开发 | 声学功能已标记为停止开发，不再推进 |
| Frequency 保留在 common-types | ✅ | 仅保留此类型（Material 需要），其余全部迁出 |
| 修复 lasso_selection 测试掩盖 bug | ✅ | LineString::contains() → Polygon::contains()，真正检测点是否在多边形内部 |
| 清理未使用依赖（4 个） | ✅ | anyhow, euclid, insta, proptest — 声明但零使用 |
| 清理 20+ clippy 警告 | ✅ | length>0→!is_empty(), assert_eq!(x,true)→assert!(x), map_or 简化等 |
| 移除 assert!(true) | ✅ | parser/tests/test_nurbs_sampling.rs 无意义断言 |

### 🔥 焚诀循环（2026-04-14 第五轮 — P2 并发优化）

| 优化项 | 状态 | 说明 |
|--------|------|------|
| 矢量化预处理并行化 | ✅ | `median_filter`, `gaussian_blur`, `clahe`, `non_local_means`, `threshold_binary/inv` — 全部 `par_chunks_mut` 并行，小图像串行 fallback |
| ValidatorService 并行验证 | ✅ | `check_self_intersection` O(n²)、`check_micro_features`、`check_hole_containment`、`check_convexity` — 4 个独立重检查通过 `rayon::join` 并行执行 |

### 🔥 焚诀循环（2026-04-14 第六轮 — P2 并发优化 + CCITTFaxDecode）

| 优化项 | 状态 | 说明 |
|--------|------|------|
| CCITTFaxDecode 支持 | ✅ | 使用 `fax` crate (v0.2) 解码 CCITT Group 3/4 传真编码图像 |
| Pipeline `fill_scene_edges` 并行化 | ✅ | 串行 for-loop → `par_iter().flat_map()` 并行提取边 + 后处理分配 ID |

### 🔥 焚诀循环（2026-04-14 第七轮 — P2 并发优化收官）

| 优化项 | 状态 | 说明 |
|--------|------|------|
| `auto_infer_boundaries` 并行化 | ✅ | 串行 O(E×N) 逐段匹配 → `par_iter` 并行推断边界语义，外轮廓 + 孔洞分段并行处理 |
| 并行化覆盖率审计 | ✅ | parser (DXF 大文件 `par_iter`、PDF 多页 `par_iter`)、topo (overlap/intersection `par_iter`、snap/DP 并行)、validator (`rayon::join` 4 路)、vectorize (5 个预处理 `par_chunks_mut`)、orchestrator (`fill_scene_edges` + `auto_infer_boundaries`) |
| 剩余 O(n²) 标注 | 记录 | `snap_endpoints_global` (vectorize) 为 O(n²) 但作用于小集合，更适合空间索引优化而非并行化 |

### 🔥 焚诀循环（2026-04-14 第八轮 — 功能完备性增强：HATCH 集成）

| 优化项 | 状态 | 说明 |
|--------|------|------|
| HATCH 填充图案集成主流水线 | ✅ | `ParserService.parse_file()` 自动调用 `HatchParser` 补充提取 HATCH 实体，建筑图纸填充图案不再丢失 |
| `with_hatch_ignore_solid` 配置 | ✅ | 支持忽略 Solid Fill 类型 HATCH |
| 新测试 | ✅ | `test_parser_service_with_hatch_config` — 验证 HATCH 配置传递 |

### 🔥 焚诀循环（2026-04-14 第九轮 — DXF 实体覆盖度增强）

| 优化项 | 状态 | 说明 |
|--------|------|------|
| POINT 实体支持 | ✅ | `EntityType::ModelPoint` → `RawEntity::Point { position, metadata, semantic }`，测绘标记点不再丢失 |
| IMAGE 实体支持 | ✅ | `EntityType::Image` → `RawEntity::Image { image_def, position, size, metadata, semantic }`，栅格图片引用 |
| ATTRIB 实体支持 | ✅ | `EntityType::Attribute` → `RawEntity::Attribute { tag, value, position, height, rotation, metadata, semantic }`，块属性值（门号/房间名等） |
| ATTDEF 实体支持 | ✅ | `EntityType::AttributeDefinition` → `RawEntity::AttributeDefinition { tag, default_value, prompt, position, height, rotation, metadata, semantic }`，属性定义 |
| 联动更新 | ✅ | `common-types/geometry.rs`（4 新枚举变体 + 5 辅助方法），`unit_converter`、`cache`、`recovery`、`orchestrator/api.rs`、`tests` 全部更新 |
| 测试 | ✅ | 350+ 测试全部通过，clippy 0 警告 |

### 🔥 焚诀循环（2026-04-14 第十轮 — HATCH 拓扑集成 + 管线收敛审计）

| 优化项 | 状态 | 说明 |
|--------|------|------|
| HATCH 边界集成 topology pipeline | ✅ | `extract_polylines_from_entities` 新增 HATCH 分支，提取 Polyline/Arc 边界路径为多段线，参与拓扑构建 |
| 文字标注提取注释完善 | ✅ | `extract_text_annotations` 添加注释说明 MTEXT 已映射为 `RawEntity::Text` |

### 🔥 焚诀循环（2026-04-15 第十一轮 — 配置传播链修复 + parser_factory 编译修复）

| 优化项 | 状态 | 说明 |
|--------|------|------|
| 修复 DxfParserEnum::with_ignore_* 编译错误 | ✅ | `CachedSync` 变体中 `p.inner().config` → `p.inner().inner().config`（双层解引用） |
| `ParserService::with_dxf_filter` | ✅ | 新增方法接受 `ignore_text/ignore_dimensions/ignore_hatch` 三个 bool 参数 |
| `ProcessingPipeline::new_with_config` 配置传播 | ✅ | 新增 `create_parser_service` 辅助函数，从 `CadConfig.parser.dxf` 读取 ignore_* 设置并应用到 ParserService |
| 新测试 | ✅ | `test_parser_service_with_dxf_filter` |

### 🔥 焚诀循环（2026-04-15 第十二轮 — DXF 实体覆盖度增强：LEADER + RAY/XLINE + 管线集成修复）

| 优化项 | 状态 | 说明 |
|--------|------|------|
| `RawEntity::Leader` 新增 | ✅ | 引线标注实体：`points: Polyline, annotation_text: Option<String>` |
| `RawEntity::Ray` 新增 | ✅ | 射线/构造线实体：`start: Point2, direction: Point2`（统一表示 RAY 和 XLINE） |
| dxf crate API 适配修复 | ✅ | `Leader.vertices` (字段非方法), `Ray.unit_direction_vector`, `XLine.unit_direction_vector` |
| `transform_entity` 支持 | ✅ | Leader 变换所有顶点，Ray 变换起点和方向 |
| `compute_entities_hash` 支持 | ✅ | Leader 哈希点数+标注文字，Ray 哈希起点+方向 |
| `convert_entity` (单位转换) 支持 | ✅ | Leader 坐标转 mm，Ray 起点+方向转 mm |
| `is_valid_entity` (实体验证) 支持 | ✅ | Leader 验证点有效+≥2 点，Ray 验证起点+方向有效 |
| `entities_to_edges` (边提取) 支持 | ✅ | Leader 分解为线段序列，Ray 用 start→start+dir*10000 长线段 |
| `extract_polylines_from_entities` 支持 | ✅ | Leader points 直接提取为 Polyline，Ray 暂用 `_ => None` |
| `fill_scene_edges` 支持 | ✅ | Leader/Ray 暂用 `_ => Vec::new()` |
| `ProcessResult` 新增 `text_annotations` | ✅ | 4 个 ProcessResult 构造函数全部更新，`extract_polylines` 返回 tuple |
| `StageContext` 新增 `text_annotations` | ✅ | configurable.rs 增加字段 + Default 支持 |
| `test_real_dxf_files.rs` 更新 | ✅ | 4 处 match 全部添加 Leader/Ray 图层提取 |
| 测试 | ✅ | 347 passed, 0 failed, 1 ignored — clippy 0 警告 |

### 🔥 焚诀循环（2026-04-15 第十三轮 — DXF 实体覆盖度增强：MLINE 多线实体支持）

| 优化项 | 状态 | 说明 |
|--------|------|------|
| `RawEntity::MLine` 新增 | ✅ | 多线实体（建筑图纸中常用于表示墙体）：`center_line: Polyline, closed, style_name, scale_factor` |
| 真实 DXF 文件审计 | ✅ | 审计 10 张真实 DXF 文件：MLINE (25 实例), SOLID (17 实例), WIPEOUT (3 实例) 为未支持实体 |
| dxf_parser.rs MLine 解析 | ✅ | 提取中心线顶点序列，flags 判断是否闭合 |
| `transform_entity` 支持 | ✅ | MLine 中心线所有顶点变换 |
| `compute_entities_hash` 支持 | ✅ | MLine 哈希顶点数+样式名+闭合标志 |
| `convert_entity` (单位转换) 支持 | ✅ | MLine 中心线坐标转 mm |
| `is_valid_entity` (实体验证) 支持 | ✅ | MLine 验证点有效+≥2 点 |
| `entities_to_edges` (边提取) 支持 | ✅ | MLine 中心线分解为线段序列 |
| 5 个 exhaustive match 方法更新 | ✅ | `semantic`, `set_semantic`, `layer`, `color`, `metadata`, `entity_type_name` |
| `test_real_dxf_files.rs` 更新 | ✅ | 4 处 match 全部添加 MLine 图层提取 |
| 测试 | ✅ | 347 passed, 0 failed, 1 ignored — clippy 0 警告 |

### 🔥 焚诀循环（2026-04-14 第三轮）

#### 声学功能停止开发
- [x] **声学功能标记停止开发** — acoustic crate 标注 `#![deprecated]`，orchestrator 添加 `#[allow(deprecated)]`
- [ ] 后续从 Cargo.toml 移除 acoustic 依赖（待确认）

#### 逻辑链路修复
- [x] **拓扑结果 WebSocket 推送** — `InteractionState` 新增 `topology_ready` 标志，后台完成后设置，WS handler 检测并推送
- [x] **auto_trace 追踪逻辑实现** — 从空壳改为实际追踪算法（`quantize_point` 网格桶查找相邻边，闭环/分叉检测），4 个新测试通过
- [x] **配置传递链修复** — `OrchestratorService::with_config()` 接受 `CadConfig`，`serve()` 加载 profile 传递到 pipeline，伪 Clone 修复
- [x] **缺口检测 UI 更新** — 检测结果写入 loading state 的 `gap_markers`，`update()` 循环同步到 `self.gap_markers`

### 📈 完成度

| 交付目标 | 完成度 | P11 锐评说明 |
|----------|--------|--------------|
| 图纸解析 | 95% | ✅ |
| 几何清洗 | 90% | ✅ |
| 拓扑构建 | 95% | Halfedge 已集成，修正之前的评估 |
| 交互确认 | 90% | WebSocket 前后端均已实现 |
| 验证导出 | 95% | ✅ |
| **整体** | **93%** | P11 锐评：85 分项目，P2 阶段可补至 95 分 |

### 📊 架构指标

| 指标 | 值 | 说明 |
|------|-----|------|
| Crate 数量 | 16 | 独立可编译模块 |
| Rust 源文件 | 157 个 | 约 69.8k 行代码 |
| 测试总数 | 347+ | 全部通过，0 失败 |
| Clippy lib 警告 | 0 | 0 警告 |
| 未使用依赖 | 0 | 已清理 anyhow, euclid, insta, proptest |
| 声学类型解耦 | ⛔ 停止开发 | 声学功能已停止开发 |

---

## 📋 路线图

### P0（已完成）✅

核心功能实现：
- DXF 完整解析（嵌套块、NURBS、曲率自适应采样）
- 智能图层识别（AIA 标准 + 中文变体）
- 拓扑构建（端点吸附、交点切分）
- 交互 API 后端（8 个 trait 方法）
- 错误恢复建议系统
- E2E 测试套件
- 配置预设模板

### P1（已完成）✅

工程化增强：
- PDF 矢量化功能
- OpenCV 加速集成（4.5x 提升）
- 质量评估系统
- 性能基准测试
- 栈溢出修复（迭代算法）
- 大图像优化

### P2（规划中）📋

**时间**: 验收后 4-8 周

#### 核心任务（P11 锐评后优先级调整）
- [x] ~~**WebSocket 实时交互**~~ (已完成) - 前后端均已完成
- [x] ~~**Halfedge 结构集成**~~ (已完成) - `TopoAlgorithm::Halfedge` 已是默认算法
- [x] ~~**声学类型解耦**~~ (已完成) - 从 common-types 迁至 acoustic crate
- [x] ~~**依赖清理**~~ (已完成) - 清理 anyhow, euclid, insta, proptest

- [x] **PDF 矢量化增强** (4-6 周) - **P1 优先级** — 子任务 1-5 全部完成
  - [x] 复杂扫描图纸支持（pipeline bug 修复、DPI 自适应缩放、质量评估增强）
  - [x] 虚线/中心线识别（跨线段共线分组 + dash-gap 模式分析）
  - [x] 文字标注分离（连通分量标记 + 启发式文字筛选 + DXF Text 实体提取）
  - [x] 质量评估增强（偏斜检测 + 分辨率检查）
  - [x] CCITTFaxDecode 支持（Group 3/4 传真解码，使用 fax crate）
  - **理由**: 扩大适用范围，当前仅适用于线条清晰的图纸

- [ ] **并发处理优化** (2 周) - **P1 优先级**
  - rayon `par_iter()` 已在多个 crate 中使用（topo、parser、vectorize 等）
  - 大文件并行解析优化
  - 多线程几何处理扩展
  - **理由**: 提升大文件性能（当前 541,216 实体 PDF 约 1.5s，已部分并行化）

- [ ] **OpenTelemetry 集成** (2 周) - **P2 优先级**
  - 分布式链路追踪
  - 微服务调试支持
  - 监控指标收集
  - **理由**: 微服务拆分后调试困难

#### 次要任务
- [ ] 语义标注 UI 校正 (2 周) - **P2 优先级**
  - 允许用户手动指定边界语义
  - 弥补图层命名不规范的问题
  
- [ ] 配置热加载 (1 周) - **P2 优先级**
  - 无需重启服务即可更新配置
  - 提升用户体验

- [ ] 真实图纸测试集（50+ 张）

### P3（进行中）🔮

**时间**: 验收后 2-3 个月

- [x] Web UI 基础架构（React + Vite + Tailwind + shadcn/ui）
- [x] Canvas 渲染（React Konva）
- [x] API 客户端封装
- [x] WebSocket 实时通信（前后端均已完成）
- [x] 状态管理（Zustand）
- [ ] Web UI 性能优化（LOD、虚拟滚动）
- [ ] 微服务拆分（HTTP/gRPC）
- [ ] CI/CD 配置
- [ ] WASM 前端嵌入
- [ ] 数据库集成

#### Web UI 迁移进度

| 模块 | 状态 | 说明 |
|------|------|------|
| 项目初始化 | ✅ 完成 | Vite + React + TypeScript + Tailwind |
| 基础 UI 组件 | ✅ 完成 | 16 个 shadcn/ui 组件 |
| API 集成 | ✅ 完成 | HTTP + WebSocket 客户端 |
| 状态管理 | ✅ 完成 | 4 个 Zustand stores |
| Canvas 渲染 | ✅ 完成 | React Konva 实现 |
| 工具栏/面板 | ✅ 完成 | 主工具栏、Canvas 工具栏、图层/属性面板 |
| 自定义 Hooks | ✅ 完成 | useWebSocket, useFileUpload, useAutoTrace |
| 测试框架 | ✅ 完成 | Vitest + Playwright |
| 部署配置 | ✅ 完成 | Docker + Nginx |
| 性能优化 | ⏳ 待开始 | LOD、虚拟滚动、批处理渲染 |
| 动画过渡 | ⏳ 待开始 | Framer Motion 集成 |

**详情**: 见 `cad-web/IMPLEMENTATION.md`

---

## 🎯 当前优先级

### 验收准备（立即）
1. ~~修复 cad-viewer 测试 abort~~ → 已完成
2. ~~修复 Clippy 警告~~ → 已完成（69 → 0，2026-04-14）
3. 演示环境准备
4. 验收演示脚本排练
5. Q&A 准备

### P2 阶段（验收后）
1. PDF 矢量化增强
2. 并发处理优化
3. OpenTelemetry 集成

---

## 📊 技术债务

| 问题 | 优先级 | 计划 | P11 锐评状态 |
|------|--------|------|-------------|
| 复杂扫描图纸支持 | P1 | P2 阶段 (4-6 周) | ⚠️ 隐患 2 |
| 语义标注无 UI 校正 | P2 | P2 阶段 (2 周) | ⚠️ 隐患 3 |
| 并发处理不完善 | P1 | P2 阶段 (2 周) | ⚠️ 不足 1（已有部分并行化，待完善） |
| 监控和链路追踪未集成 | P2 | P2 阶段 (2 周) | ⚠️ 不足 2 |
| wgpu 加速器边缘检测 | ✅ 完成 | 计算着色器 Sobel 边缘检测已实现 | - |
| wgpu 轮廓提取 | ✅ 完成 | GPU 连通分量标记 + CPU 轮廓跟踪，完整加速 pipeline | 焚诀第十五轮 |
| 扫描图纸预处理增强 | ✅ 完成 | 默认启用 CLAHE 对比度增强，支持多种去噪方法 | 焚诀第十五轮 |
| 文字分离算法增强 | ✅ 完成 | 行启发式聚类、自适应膨胀擦除、过滤水平线/垂直线更准确 | 焚诀第十五轮 |
| 缺口填充/虚线连接增强 | ✅ 完成 | 默认启用 gap_filling 和 line_type_detection 更好处理间断线 | 焚诀第十五轮 |
| wgpu 圆弧拟合/端点吸附 | ✅ 完成 | GPU 加速版本已实现 | 焚诀第十六轮 |
| clippy test/bench 警告 | P3 | 20+ 剩余（非 lib 代码） | 可容忍 |

**已解决（原 P11 锐评指出）**:
- ~~WebSocket 前端缺失~~ → 已实现（`cad-web/src/services/websocket-client.ts` + `useWebSocket` hook）
- ~~WebSocket 后端缺失~~ → 已实现（`crates/orchestrator/src/api.rs` 完整 handler）
- ~~Halfedge 未集成主流程~~ → 已集成（`TopoAlgorithm::Halfedge` 为默认）
- ~~Clippy 警告~~ → 已修复（69 → 0 警告）
- ~~cad-viewer test abort~~ → 已修复（`RegionCode` 从 enum 改为 struct 支持位标志）

**已解决（第二轮优化）**:
- ~~声学类型耦合~~ → 已解耦（`common_types::acoustic` → `acoustic::acoustic_types`）
- ~~未使用依赖~~ → 已清理（anyhow, euclid, insta, proptest）
- ~~lasso_selection 测试 bug~~ → 已修复（LineString → Polygon contains）
- ~~20+ clippy 警告~~ → 已修复（length>0→!is_empty(), assert_eq! literal bool 等）

---

## 📝 文档状态

| 文档 | 状态 | 说明 |
|------|------|------|
| README.md | ✅ | 主文档，最新 |
| CONTRIBUTING.md | ✅ | 贡献指南 |
| todo.md | ✅ | 本文档 |
| 交付目标.md | ✅ | 甲方文件（不修改） |
| docs/archive/ | ✅ | 历史文档归档 |

---

**最后更新**: 2026 年 4 月 20 日（焚诀第十五轮：扫描图纸识别能力增强 + wgpu 轮廓提取 GPU 实现完成）
**下次评审**: P2 阶段中期（验收后 4 周）
