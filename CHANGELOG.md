# 变更日志 (CHANGELOG)

所有重要的项目变更都将记录在此文件中。

格式基于 [Keep a Changelog](https://keepachangelog.com/zh-CN/1.0.0/)，项目遵循 [语义化版本](https://semver.org/lang/zh-CN/)。

---

## [未发布]

### 新增

#### 多格式解析增强
- **DWG 解析**
  - AutoCAD 默认格式支持（R13-R2018 版本）
  - 通过 libredwg 外部转换器集成
  - 实体映射到 RawEntity 标准格式

- **SVG 导入/导出**
  - 导入：解析 SVG `<line>/<path>/<circle>/<text>/<polygon>` 等元素 → RawEntity
  - 导出：RawEntity → SVG XML 格式
  - 自动计算 viewBox（基于实体边界框 + 10% 边距）
  - 支持图层可见性过滤
  - 支持 RawEntity 类型：Line, Polyline, Circle, Arc, Text, Path, Triangle (XY 投影)

- **STL 解析**
  - 支持二进制和 ASCII 双格式
  - 自动检测格式类型
  - 三角面片 → `RawEntity::Triangle`
  - 法线向量记录
  - 新增 `RawEntity::Triangle` 变体（3D 几何支持）

#### PDF 增强
- **PDF 文字提取**
  - 支持 Tj/TJ 操作符（显示文字）
  - BT/ET 文字对象解析
  - Tm 变换矩阵合成
  - Td/T*/Tf 文字定位和字体设置
  - 变换矩阵应用于 Path 和 Text 实体位置

#### 模块拆分
- **raster-loader** 独立 crate
  - 统一光栅图片加载接口
  - 支持 PNG/JPG/BMP/TIFF/WebP 格式
  - 自动格式检测和元数据提取
  - 解耦 vectorize 和图片加载依赖

### 改进

- 测试套件从 310+ 扩展到 585+ 测试

#### wgpu 加速器功能完备化
- **圆弧拟合 GPU 实现** (`accelerator-wgpu/src/arc_fit.rs`)
  - Kåsa 算法 GPU 并行版本，workgroup 归约计算累加和
  - GPU 计算统计量，CPU 求解线性方程组得到圆心半径
- **端点吸附 GPU 实现** (`accelerator-wgpu/src/snap.rs`)
  - 工作组内存缓存 + 并行最近邻搜索
  - 在容差范围内吸附到最近端点
- 完整实现 Accelerator trait 所有四个操作：边缘检测 ✅ + 轮廓提取 ✅ + 圆弧拟合 ✅ + 端点吸附 ✅
- wgpu 加速器现在功能完备，所有操作都有 GPU 加速版本
- export crate 新增 SVG 导出器（SvgWriter）
- parser crate 新增 SVG 导入器（SvgParser）、STL 解析器（StlParser）、DWG 解析器（DwgParser）
- common-types 新增 `RawEntity::Triangle` 变体
- parser 缓存系统支持 Triangle 实体哈希计算
- parser 恢复系统支持 Triangle 实体验证

---

## [0.1.0] - 2026-02-28

**稳定版本** - 核心功能完整，220+ 测试全部通过

### 新增

#### 核心功能
- **DXF 解析**
  - 支持 LINE, LWPOLYLINE, ARC, CIRCLE, SPLINE, ELLIPSE 实体
  - 嵌套块递归展开
  - NURBS 曲率自适应采样（弦高误差 < 0.1mm）
  - 智能图层识别（AIA 标准 + 中文变体）
  - 单位解析与自动标定
  - 颜色/线宽过滤
  - 3D 曲线自动投影到 2D

- **PDF 解析**
  - 矢量 PDF 直接提取路径/线段
  - 光栅 PDF 自动矢量化（适用于线条清晰的图纸）
  - 压缩流支持（FlateDecode/DCTDecode/LZW/RunLength）
  - 质量评估与错误报告

- **拓扑构建**
  - R*-tree 空间索引加速（O(n log n) 复杂度）
  - 端点吸附与零长度过滤
  - 交点切分（Bentley-Ottmann 扫描线）
  - 闭合环提取（DFS + 夹角最小启发式）
  - 孔洞检测与包含关系验证

- **交互 API**
  - 选边追踪（Edge Picking + Auto Trace）
  - 圈选区域（Lasso/Polygon Selection）
  - 缺口检测与分层补全
  - 边界语义标注

- **几何验证**
  - 闭合性检查
  - 自相交检测
  - 孔洞包含关系验证
  - 微小特征检测（短边/尖角）
  - 错误恢复建议生成（RecoverySuggestion）

- **场景导出**
  - JSON 格式（人类可读）
  - Binary 格式（bincode 高性能）
  - Schema v1.2 兼容仿真模块

#### 工程化
- **测试套件**
  - 220+ 测试全部通过
  - 单元测试（133 个）
  - 边界情况测试（14 个）
  - NURBS 验证测试（6 个）
  - 真实文件测试（20 个：9 个 DXF + 4 个 PDF）
  - 基准测试（19 个）
  - 集成测试（7 个）
  - E2E 测试（6 个）
  - 用户故事测试（6 个）

- **性能基准**
  - 100 线段：13.4ms
  - 1000 线段：131.9ms
  - 复杂度验证：O(n log n)

- **配置系统**
  - 5 个场景化配置预设
  - 命令行 `--profile` 参数支持
  - 配置验证功能

- **OpenCV 加速**（可选 feature）
  - 边缘检测 5.3x 提升
  - 轮廓提取 4.7x 提升
  - 总计 4.5x 性能提升

#### 前端原型
- egui 完整实现
- 点选/标注/导出功能
- 后端 API 集成
- 加载动画与错误提示

### 修复

- 栈溢出问题（所有递归算法改为迭代实现）
- 零长度线段未过滤问题
- 闭合多段线首尾重复点问题
- 测试逻辑不符合 DXF 规范问题
- API 路径不匹配问题
- 嵌套 runtime 测试失败问题

### 改进

- 错误恢复建议系统（提供可操作的修复建议）
- 递归深度限制（MAX_DEPTH=20）
- 大图像优化（堆分配 + 尺寸限制）
- DPI 自适应参数调整
- 圆弧拟合算法（Kåsa 算法）
- 图像质量预评估

### 文档

- README.md 完整更新
- ARCHITECTURE.md 架构文档
- CONTRIBUTING.md 贡献指南
- todo.md 技术路线图
- CHANGELOG.md 变更日志
- 历史文档归档到 `docs/archive/`

---

## [0.1.0-beta] - 2026-02-15

**Beta 版本** - 核心功能基本完成

### 新增
- DXF 基础解析功能
- 拓扑构建核心算法
- 几何验证基础功能
- 场景导出 JSON 格式

### 已知问题
- 嵌套块不支持
- NURBS 采样点数不足
- 栈溢出风险
- 测试覆盖不完整

---

## [0.0.1] - 2026-01-18

**初始版本** - 项目启动

### 新增
- 项目骨架
- Workspace 配置
- 基础类型定义
- 交付目标文档

---

## 版本说明

### 版本号规则

- **主版本号 (Major)**: 不兼容的 API 修改
- **次版本号 (Minor)**: 向下兼容的功能性新增
- **修订号 (Patch)**: 向下兼容的问题修正

### 当前状态

- **最新版本**: v0.1.0 (稳定版本)
- **测试状态**: 585+ 测试（584 通过，1 已知失败）
- **代码质量**: Clippy 0 警告
- **整体完成度**: 92%

### P2 阶段计划

验收后 4-8 周内完成：
- WebSocket 实时交互
- Halfedge 主流程集成
- PDF 矢量化增强
- 配置热加载
- 微服务拆分（HTTP/gRPC）
- 语义标注 UI 校正入口

---

## 快速链接

- [README.md](README.md) - 项目介绍
- [ARCHITECTURE.md](ARCHITECTURE.md) - 架构文档
- [CONTRIBUTING.md](CONTRIBUTING.md) - 贡献指南
- [todo.md](todo.md) - 技术路线图
- [交付目标.md](交付目标.md) - 甲方文档

---

**最后更新**: 2026 年 4 月 15 日
