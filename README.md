# CAD 几何智能处理系统

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust Version](https://img.shields.io/badge/rust-1.75+-blue.svg)](https://github.com/rust-lang/rust)

基于「一切皆服务」(Everything-as-a-Service, EaaS) 设计哲学的工业级 CAD 几何智能处理系统，融合**计算机视觉、计算几何、图形拓扑建模、人机协同交互与工程仿真接口**。

## 📊 项目状态

> ### ✅ **核心功能已完成**（v0.1.0 稳定版本）
>
> **当前系统支持**：
> - ✅ DXF 文件（AutoCAD 矢量格式，AC1015 及以上版本）
>   - ✅ 支持实体：LINE, LWPOLYLINE, ARC, CIRCLE, SPLINE, ELLIPSE, HATCH
>   - ✅ BLOCK/INSERT（块定义与引用，嵌套块支持）
>   - ✅ 智能图层识别（AIA 标准 + 中文变体）
>   - ✅ NURBS 精确离散化（弦高误差 < 0.1mm）
>   - ✅ HATCH 填充边界提取
>   - ✅ 单位解析与自动标定
>   - ✅ 颜色/线宽过滤
> - ✅ 矢量 PDF 文件（可直接提取路径/线段）
> - ✅ 光栅 PDF 文件（扫描版/截图，自动矢量化 - 适用于线条清晰的图纸）
> - ✅ 光栅图片文件（PNG/JPG/BMP/TIFF/WebP，自动矢量化）
> - ✅ DWG 文件（AutoCAD 默认格式，R13-R2018 版本）
> - ✅ SVG 文件（Web 端 CAD 交换格式，导入/导出双向支持）
> - ✅ STL 文件（3D 制造/打印工作流，二进制/ASCII 双格式）
>
> **光栅 PDF/图片矢量化特性**：
> - 支持扫描版 PDF 和 PNG/JPG/BMP/TIFF/WebP 自动矢量化（图像预处理 + 边缘检测 + 线结构提取）
> - 包含质量评估和错误报告
> - 推荐图像尺寸 < 2000x2000 像素
> - 最大支持 250 万像素（约 1581x1581）
> - 测试通过率 100%
>
> **性能参考**：
> - 500x500 像素：~50ms
> - 1000x1000 像素：~200ms
> - 2000x2000 像素：~800ms
> - OpenCV 加速（2000x3000 像素）：~220ms（4.5x 提升）
>
> **适用场景**：
> - ✅ 线条清晰的建筑平面图（推荐）
> - ✅ 扫描质量良好的图纸
> - ✅ 高对比度的工程图
>
> **限制说明**：
> - ⚠️ 对于存在严重阴影/折痕/褪色的复杂图纸，建议先转换为 DXF 格式（行业通用做法）
> - ⚠️ 虚线/中心线/剖面线混合场景的识别能力计划于 P2 阶段增强（验收后 4-6 周）
> - ⚠️ 语义标注依赖图层命名规范（非标准命名时需手动校正，P2 阶段增加 UI 校正入口）
> - ✅ 复杂拓扑（嵌套孔洞/非流形几何）处理：Halfedge 结构已集成到主流程，`TopoAlgorithm::Halfedge` 为默认算法
>
> **如何判断 PDF 类型**：
> - 矢量 PDF：文件小（< 1MB），放大后边缘清晰，包含 LINE/PATH 等矢量图元
> - 光栅 PDF：文件大（> 5MB），放大后有锯齿，内部为位图图像

| 服务 | 状态 | 测试覆盖 | 说明 |
|------|------|----------|------|
| `common-types` | ✅ | 19 单元测试 | 公共类型定义、错误处理、恢复建议 |
| `parser` | ✅ | 74 测试 | DXF/DWG/SVG/STL/PDF 解析 + 缓存/恢复 |
| `vectorize` | ✅ | 46 测试 | 矢量化算法 + 光栅 PDF 测试 |
| `topo` | ✅ | 28 测试 | 拓扑构建 + Halfedge + 基准测试 |
| `interact` | ✅ | 10 单元测试 | 交互 API + 脏矩形追踪 |
| `validator` | ✅ | 21 单元测试 | 几何验证 + 恢复建议 |
| `export` | ✅ | 8 测试 | JSON/Binary/SVG 导出 |
| `orchestrator` | ✅ | 22 测试 | API 网关 + E2E + WebSocket |
| `config` | ✅ | 4 单元测试 | 配置管理 + 5 场景预设 |
| `acoustic` | ✅ | 48 测试 | 声学分析（选区材料统计、混响时间计算、多区域对比） |
| `accelerator-api` | ✅ | 14 测试 | 加速器抽象接口 |
| `accelerator-cpu` | ✅ | 10 测试 | CPU 加速器实现 |
| `accelerator-registry` | ✅ | 7 测试 | 加速器注册中心 |
| `accelerator-wgpu` | ⚠️ | Stub (TODO) | wgpu 加速器（CPU fallback 已工作） |
| `raster-loader` | ✅ | 27 测试 | 光栅图片加载（PNG/JPG/BMP/TIFF/WebP） |

**总计**: ✅ 585+ 测试（584 通过，1 已知失败） | Clippy: 0 错误（4 个良性复杂度警告）

## 🚀 快速开始

### 环境要求

- Rust 1.75+ (stable)
- Windows 10/11、Linux 或 macOS
- (可选) OpenCV 4.x（用于高级矢量化加速）

#### OpenCV 可选加速

启用 OpenCV 后可获得 **4.5x 性能提升**：

| 操作 | 纯 Rust | OpenCV | 提升 |
|------|--------|--------|------|
| 边缘检测 | ~450ms | ~85ms | 5.3x |
| 轮廓提取 | ~280ms | ~60ms | 4.7x |
| **总计** | ~1000ms | ~220ms | **4.5x** |

**启用方式**:
```bash
# 从 cad-cli 构建（推荐）
cargo build --release --features cad-cli/opencv

# 或从 workspace 根目录构建
cargo build --release --features vectorize/opencv
```

**系统要求**:
- Windows: 安装 OpenCV 4.x 并设置 `OpenCV_DIR`
- Linux: `sudo apt-get install libopencv-dev`
- macOS: `brew install opencv`

### 构建与运行

```bash
# 构建整个项目
cargo build --workspace

# 运行测试
cargo test --workspace

# 构建 Release 版本（性能优化）
cargo build --release
```

### 使用方式

#### 方式一：命令行工具

```bash
# 处理 DXF 文件
cargo run --package cad-cli -- process input.dxf --output scene.json

# 处理 PDF 文件（矢量或光栅自动识别）
cargo run --package cad-cli -- process input.pdf --output scene.json

# 处理光栅图片文件，可显式选择光栅策略和 DPI/尺度
cargo run --package cad-cli -- process input.png \
  --profile raster_semantic \
  --raster-strategy auto \
  --dpi-override 300,300 \
  --output scene.json

# 使用预设配置（architectural/mechanical/scanned/photo_sketch/raster_clean/raster_scan/raster_photo/raster_sketch/raster_semantic/quick）
cargo run --package cad-cli -- process input.dxf --profile architectural

# 自定义参数
cargo run --package cad-cli -- process input.dxf \
  --snap-tolerance 0.5 \
  --min-line-length 1.0 \
  --closure-tolerance 0.3

# 查看预设配置
cargo run --package cad-cli -- list-profiles
cargo run --package cad-cli -- show-profile architectural

# 构建独立二进制文件
cargo build --release --package cad-cli
# 生成的可执行文件：target/release/cad
```

#### 方式二：HTTP 服务

```bash
# 启动 HTTP 服务（默认端口 3000）
cargo run --package cad-cli -- serve --port 3000

# 使用预设配置启动
cargo run --package cad-cli -- serve --port 3000 --profile architectural
```

服务启动后，可通过 API 调用：

```bash
# 健康检查
curl http://localhost:3000/health

# 处理 DXF 文件
curl -X POST http://localhost:3000/process -F "file=@file.dxf"

# 处理 PDF 文件
curl -X POST http://localhost:3000/process -F "file=@file.pdf"

# 处理光栅图片文件
curl -X POST http://localhost:3000/process -F "file=@drawing.png"

# 专用光栅图片端点，返回 raster_report、semantic_candidates、尺度信息
curl -X POST http://localhost:3000/process/raster \
  -F "file=@drawing.png" \
  -F "strategy=auto" \
  -F "dpi_override=300,300" \
  -F "max_retries=3" \
  -F "debug_artifacts=false"

# 声学分析（选区材料统计）
curl -X POST http://localhost:3000/acoustic/analyze \
  -H "Content-Type: application/json" \
  -d '{"type":"SELECTION_MATERIAL_STATS","boundary":{"type":"RECT","min":[0,0],"max":[10,10]}}'

# 房间混响时间计算
curl -X POST http://localhost:3000/acoustic/analyze \
  -H "Content-Type: application/json" \
  -d '{"type":"ROOM_REVERBERATION","room_id":0,"formula":"SABINE","room_height":3.0}'
```

#### 方式三：GUI 查看器（交互式界面）

```bash
# 启动 GUI 查看器（egui 界面）
cargo run --package cad-viewer

# 启用 GPU 加速（需要独立显卡）
cargo run --package cad-viewer --features gpu
```

**GUI 功能**：
- 📐 Canvas 渲染（线段绘制/缩放/平移）
- 🖱️ 鼠标点选边（射线检测 + 容差）
- ✨ 实时高亮追踪路径
- 🏷️ 语义标注 ComboBox
- 📤 文件上传/导出
- 🔍 缺口可视化
- ⬜ 圈选工具（Lasso/Polygon Selection）
- 🎨 macOS 风格主题（浅色/深色模式）
- 🚀 GPU 加速渲染（毛玻璃效果/实例化/MSAA）
- 🔌 WebSocket 实时交互（选边/缺口检测/ping）

**界面布局**：
```
┌────────────────────────────────────────────────────┐
│  工具栏 (Toolbar)                                   │
├──────────┬──────────────────────────────┬──────────┤
│  图层    │                              │  属性     │
│  面板    │      Canvas 画布              │  面板     │
│          │   - 缩放/平移                │          │
│          │   - 点选/圈选                │          │
│          │   - 语义标注                 │          │
├──────────┴──────────────────────────────┴──────────┤
│  底部状态栏 (坐标/线段数/性能指标)                    │
└────────────────────────────────────────────────────┘
```

#### 方式四：库调用

在 Rust 项目中直接使用：

```rust
use orchestrator::OrchestratorService;

#[tokio::main]
async fn main() {
    let service = OrchestratorService::default();
    let result = service.process_file("input.dxf").await.unwrap();
    println!("处理完成：{:?}", result);
}
```

### 输出格式

支持两种导出格式（在 `cad_config.toml` 中配置）：

- **JSON**: 人类可读，带美化输出
- **Bincode**: 高性能二进制格式

### 配置文件

创建 `cad_config.toml` 自定义处理参数：

```toml
[topology]
snap_tolerance_mm = 0.5
min_line_length_mm = 1.0
merge_angle_tolerance_deg = 5.0
max_gap_bridge_length_mm = 2.0

[validator]
closure_tolerance_mm = 0.3
min_area_m2 = 0.5
min_edge_length_mm = 100.0
min_angle_deg = 15.0

[export]
format = "json"
json_indent = 2
auto_validate = true
```

或使用预设配置（无需手动编写配置文件）。

## 🏗️ 架构概述

```
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator Service                      │
│                   (API 网关 / 流程编排)                        │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Core Processing Pipeline                   │
├──────────────┬──────────────┬──────────────┬────────────────┤
│  ParserSvc   │ VectorizeSvc │  TopoSvc     │ ValidatorSvc   │
│  图纸解析     │ 图像矢量化    │ 拓扑建模      │ 几何验证        │
│  DXF/PDF     │ OpenCV/Rust  │ R*-tree      │ 单位标定        │
└──────────────┴──────────────┴──────────────┴────────────────┘
                              ↓
                    ┌─────────────────┐
                    │   ExportSvc     │
                    │   场景导出       │
                    │  JSON/Binary    │
                    └─────────────────┘
                              ↓
                    ┌─────────────────┐
                    │  AcousticSvc    │
                    │   声学分析       │
                    │  RT60/材料统计   │
                    └─────────────────┘
```

### EaaS 架构实现状态

| 服务 | 进程内调用 | HTTP API | WebSocket | gRPC | 熔断 | 监控 | 链路追踪 |
|------|-----------|---------|-----------|------|------|------|----------|
| ParserSvc | ✅ | ✅ | 🔲 | 🔲 | ⚠️ | ✅ | ⚠️ |
| VectorizeSvc | ✅ | ✅ | 🔲 | 🔲 | ⚠️ | ✅ | ⚠️ |
| TopoSvc | ✅ | ✅ | 🔲 | 🔲 | ⚠️ | ✅ | ⚠️ |
| ValidatorSvc | ✅ | ✅ | 🔲 | 🔲 | ⚠️ | ✅ | ⚠️ |
| ExportSvc | ✅ | ✅ | 🔲 | 🔲 | ⚠️ | ✅ | ⚠️ |
| InteractSvc | ✅ | ✅ | ✅ | 🔲 | ⚠️ | ✅ | ⚠️ |
| AcousticSvc | ✅ | ✅ | 🔲 | 🔲 | ⚠️ | ✅ | ⚠️ |

**说明**: ✅ 已完成 | ⚠️ 已实现未集成 | 🔲 P2 计划

### 架构演进记录

| 阶段 | 日期 | 关键变更 |
|------|------|---------|
| 多格式解析增强 | 2026-04-15 | DWG/SVG/STL 解析、PDF 文字提取、RawEntity::Triangle |
| 首轮焚诀优化 | 2026-04-14 | EzdxfParser 抽象化、错误语义修复、焚诀 API 优化 |
| 解耦优化 | 2026-04-14 | 声学类型解耦（common-types → acoustic）、清理 4 个未使用依赖、修复 lasso_selection bug |

### 服务调用链

```
process_with_services()
    ↓
1. ParserService::process()    → ParseResult
2. TopoService::process()      → SceneState
3. ValidatorService::process() → ValidationReport
4. ExportService::process()    → ExportResult
    ↓
ProcessResult { scene, validation, output_bytes }
```

**当前部署**: 单体部署（进程内服务调用）
**P2 计划**: HTTP/gRPC 微服务部署

## 📦 项目结构

```
CAD/
├── Cargo.toml
├── README.md
├── CONTRIBUTING.md
├── docs/
│   ├── INDEX.md                    # 文档索引
│   ├── 功能介绍.md                  # 面向甲方的功能介绍
│   ├── 后端 API 概览.md               # 后端 API 功能概览
│   ├── 交付目标对照表.md             # 与交付目标的对应关系
│   ├── web-ui-*.md                 # Web UI 相关文档
│   └── archive/                    # 历史文档归档
├── dxfs/                           # DXF 测试文件 (9 个)
├── testpdf/                        # PDF 测试文件 (4 个)
└── crates/
    ├── common-types/               # 公共类型定义
    ├── parser/                     # 图纸解析服务 (DXF/DWG/PDF/SVG/STL)
    ├── vectorize/                  # 图像矢量化服务
    ├── topo/                       # 拓扑建模服务
    ├── validator/                  # 几何验证服务
    ├── export/                     # 场景导出服务 (JSON/Binary/SVG)
    ├── interact/                   # 交互协同服务
    ├── orchestrator/               # 流程编排服务
    ├── acoustic/                   # 声学分析服务
    ├── config/                     # 配置管理
    ├── cad-cli/                    # 命令行工具
    ├── cad-viewer/                 # GUI 查看器 (egui)
    ├── accelerator-api/            # 加速器抽象接口
    ├── accelerator-cpu/            # CPU 加速器实现
    ├── accelerator-registry/       # 加速器注册中心
    ├── accelerator-wgpu/           # wgpu 加速器 (stub)
    └── raster-loader/              # 光栅图片加载器
```

## 🔧 核心服务

### 1. common-types - 公共类型库

**测试**: 26 单元测试

提供共享类型：
- 几何类型：`Point2`, `Point3`, `Polyline`, `ClosedLoop`
- 场景状态：`SceneState`, `BoundarySegment`
- 统一错误：`CadError` + `RecoverySuggestion`
- 语义推断：图层/颜色映射

### 2. parser - 图纸解析服务

**测试**: 74 测试

**支持格式**:
- **DXF**: LINE, LWPOLYLINE, ARC, CIRCLE, SPLINE, ELLIPSE, HATCH
  - 智能图层识别（AIA 标准 + 中文变体）
  - NURBS 精确离散化（弦高误差 < 0.1mm）
  - 嵌套块递归展开
  - 曲率自适应采样
  - HATCH 填充边界提取
  - 单位解析与标定
  - 颜色/线宽过滤
- **DWG**: AutoCAD 默认格式（R13-R2018 版本）
  - 外部转换器集成（libredwg）
  - 实体映射到 RawEntity 标准格式
- **PDF**:
  - 矢量 PDF：直接提取路径 + 文字标注（Tj/TJ 操作符）
  - 光栅 PDF：自动矢量化
  - 变换矩阵合成（BT/ET/Tm/Td）
- **SVG**: Web 端 CAD 交换格式
  - 导入：解析 `<line>/<path>/<circle>/<text>` 等元素 → RawEntity
  - 导出：RawEntity → SVG XML（自动计算 viewBox，支持图层过滤）
- **STL**: 3D 制造/打印工作流
  - 支持二进制和 ASCII 双格式
  - 三角面片 → `RawEntity::Triangle`
  - 自动检测格式类型

### 3. vectorize - 图像矢量化服务

**测试**: 46 测试

**功能**:
- 边缘检测（Sobel / OpenCV Canny）
- 二值化（Otsu 自适应）
- 骨架化（Zhang-Suen）
- 轮廓提取（迭代 DFS）
- Douglas-Peucker 简化
- 端点吸附（R*-tree 加速）
- 圆弧拟合（Kåsa 算法）
- 质量评估（自动评分）

### 4. topo - 拓扑建模服务

**测试**: 28 测试

**核心算法**:
1. 端点吸附：R*-tree 空间索引，O(n log n)
2. 交点切分：Bentley-Ottmann 扫描线
3. 平面图构建：节点 - 边 - 邻接表
4. 闭合环提取：DFS + 夹角最小启发式
5. 孔洞判定：负面积 + 射线法

**性能**:
- 100 线段：13.4ms
- 1000 线段：131.9ms
- 复杂度：O(n log n) ✅

### 5. interact - 交互协同服务

**测试**: 9 单元测试

**功能**:
- 模式 A：选边追踪（Edge Picking + Auto Trace）
- 模式 B：圈选区域（Lasso/Polygon Selection）
- 缺口检测与分层补全
- 边界语义标注

### 6. validator - 几何验证服务

**测试**: 21 单元测试

**验证项**:
- 闭合性检查（环首尾误差）
- 自相交检测
- 孔洞包含关系
- 微小特征检测
- 单位标定

**错误代码**:
- `E001`: 环未闭合
- `E002`: 自相交
- `E003`: 孔洞在外边界外
- `W001`: 短边
- `W002`: 尖角

### 7. export - 场景导出服务

**测试**: 8 测试

**格式**:
- JSON：人类可读
- Binary：bincode 高性能
- SVG：矢量图形导出（Line→`<line>`, Circle→`<circle>`, Path→`<path>`, Triangle→`<polygon>` XY 投影）

**Schema v1.2**:
```json
{
  "schema_version": "1.2",
  "units": "m",
  "geometry": {
    "outer": [[0,0],[10,0],[10,8],[0,8]],
    "holes": [[[2,2],[4,2],[4,3],[2,3]]]
  },
  "boundaries": [...],
  "sources": [...]
}
```

### 8. orchestrator - 流程编排服务

**测试**: 22 测试

**API 端点**:
- `GET /health` - 健康检查
- `POST /process` - 处理文件

### 9. acoustic - 声学分析服务（新增）

**功能**:
- **选区材料统计**：计算选定区域内的材料分布和等效吸声面积
- **房间混响时间计算**：支持 SABINE/EYRING 公式，计算 T60/EDT
- **多区域对比分析**：对比不同区域的声学特性

**API 端点**:
- `POST /acoustic/analyze` - 执行声学分析

**支持的声学指标**:
- 混响时间 T60（125Hz-4kHz 倍频程）
- 早期衰变时间 EDT
- 平均吸声系数
- 等效吸声面积

### 10. config - 配置管理服务

**测试**: 4 单元测试

**预设配置**:
- `architectural`: 建筑图纸预设
- `mechanical`: 机械图纸预设
- `scanned`: 扫描图纸预设
- `photo_sketch`: 照片/手绘光栅预设
- `raster_clean`, `raster_scan`, `raster_photo`, `raster_sketch`, `raster_semantic`: 光栅专用预设
- `quick`: 快速原型预设

**API 端点**:
- `GET /config/profiles` - 列出预设配置
- `GET /config/profile/:name` - 获取配置详情

### 11. raster-loader - 光栅图片加载服务

**测试**: 3 单元测试

**功能**:
- 支持多种光栅图片格式：PNG, JPG, BMP, TIFF, WebP
- 自动检测文件格式
- 提取图片元数据（尺寸、PNG pHYs/JPEG JFIF 或 EXIF/TIFF resolution DPI）
- 输出 `image::DynamicImage` 直接对接 VectorizeService

**使用场景**:
- 独立图片文件矢量化
- 扫描图纸直接处理
- 截图/照片导入

## 📈 性能基准

### Parser 性能
- 1000 实体 DXF: <100ms
- 541,216 PDF 实体：1.5s

### Topo 性能
| 线段数 | 时间 | 每线段 |
|--------|------|--------|
| 100 | 13.4ms | 134μs |
| 500 | 67.6ms | 135μs |
| 1000 | 131.9ms | 132μs |

**复杂度**: O(n log n) ✅

### Vectorize 性能
| 像素 | 纯 Rust | OpenCV |
|------|--------|--------|
| 500×500 | ~50ms | - |
| 1000×1000 | ~200ms | - |
| 2000×2000 | ~800ms | - |
| 2000×3000 | ~1000ms | ~220ms |

### 端到端性能
| 场景 | 线段数 | 时间 |
|------|--------|------|
| 小型会议室 | 100 | 14.55ms |
| 中型报告厅 | 300 | 13.20ms |
| 大型礼堂 | 1000 | 9.80ms |

## 🧪 测试

```bash
# 运行所有测试
cargo test --workspace

# 运行特定 crate 测试
cargo test -p parser
cargo test -p topo

# 运行基准测试
cargo test --test benchmarks -- --nocapture

# 运行 E2E 测试
cargo test --test e2e_tests

# 运行用户故事测试
cargo test --test user_story_tests
```

**测试文件**:
- DXF: `dxfs/` (9 个真实建筑图纸)
- PDF: `testpdf/` (4 个矢量 PDF)

**测试覆盖**:
- 单元测试：133 个
- 边界测试：14 个
- NURBS 测试：6 个
- 真实文件测试：20 个
- 基准测试：19 个
- 集成测试：7 个
- E2E 测试：6 个
- 用户故事测试：6 个

**总计**: 220+ 测试全部通过

## 🛠️ 开发指南

### 添加新依赖

在根 `Cargo.toml` 的 `[workspace.dependencies]` 中添加：

```toml
[workspace.dependencies]
your-crate = "version"
```

### 代码风格

```bash
# 格式化
cargo fmt --workspace

# Clippy 检查
cargo clippy --workspace --lib
```

## 📋 路线图

### P0（已完成）✅
- DXF 解析完整功能
- PDF 解析集成
- 拓扑构建核心算法
- 交互 API 后端
- 错误恢复建议系统
- E2E 测试套件

### P1（已完成）✅
- PDF 矢量化功能
- OpenCV 加速集成
- 质量评估系统
- 性能基准测试
- 配置预设模板（`cad_config.profiles.toml` 已定义）
- **WebSocket 实时交互**（后端 `/ws` 端点 + 前端 cad-viewer 集成）
- HTTP API 完整实现（`/health`, `/process`）
- CI/CD 配置（`.github/workflows/ci.yml`）
- 性能基线（`benches/baseline_v0.1.0.txt`）
- **声学分析服务**（选区材料统计、混响时间计算）
- **多格式解析**（DWG/DXF/PDF/SVG/STL）
- **PDF 文字提取**（Tj/TJ 操作符 + 变换矩阵合成）
- **SVG 导入/导出**（RawEntity ↔ SVG XML 双向转换）
- **STL 解析**（二进制/ASCII → `RawEntity::Triangle`）

### P2（规划中）📋
- ✅ Halfedge 结构集成到主流程（已完成，默认启用，支持嵌套孔洞）
- rayon 并行化优化（依赖已引入，待扩展到 parser/vectorize 全流程）
- PDF 矢量化增强（虚线/中心线/剖面线识别）
- 配置热加载
- 微服务拆分（HTTP/gRPC）
- UI 语义标注校正入口

### P3（未来）🔮
- WASM 前端嵌入
- 数据库集成
- OpenTelemetry 链路追踪

## 📄 License

MIT License

## 📞 联系方式

- 项目地址：https://github.com/your-org/cad
- 问题反馈：https://github.com/your-org/cad/issues

---

**最后更新**: 2026 年 4 月 15 日
**版本**: v0.1.0 (稳定版本)
**测试状态**: ✅ 585+ 测试（584 通过，1 已知失败）| Clippy: 0 警告
