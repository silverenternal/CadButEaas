# CAD 几何智能处理系统 - 架构文档

**版本**: v0.1.0
**最后更新**: 2026 年 4 月 15 日

---

## 一、架构概述

### 1.1 设计哲学

本系统基于「**一切皆服务**」(Everything-as-a-Service, EaaS) 设计哲学：

- **服务化**: 每个功能模块都是独立的服务，有明确的输入输出契约
- **可组合**: 服务间通过统一接口组合，形成处理流水线
- **可测试**: 每个服务可独立测试，支持 Mock 和 Stub
- **可演进**: 服务内部实现可独立演进，不影响其他服务

**架构说明**: 当前为 Rust 单体架构（monolithic），所有服务运行在同一进程中。EaaS 是设计哲学而非微服务架构。

### 1.2 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Orchestrator Service                      │
│                   (API 网关 / 流程编排)                        │
│                                                              │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐         │
│  │  HTTP API   │  │ WebSocket   │  │   CLI       │         │
│  │  /process   │  │   /ws       │  │  cad-cli    │         │
│  └─────────────┘  └─────────────┘  └─────────────┘         │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                   Core Processing Pipeline                   │
│                                                              │
│  ┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐ │
│  │ Parser   │ → │  Topo    │ → │Validator │ → │ Export   │ │
│  │  Service │   │ Service  │   │ Service  │   │ Service  │ │
│  │          │   │          │   │          │   │          │ │
│  │ DXF/PDF  │   │ R*-tree  │   │ 闭合性   │   │ JSON     │ │
│  │ 实体解析  │   │ 交点切分  │   │ 自相交   │   │ Binary   │ │
│  │ 图层识别  │   │ 闭合环    │   │ 孔洞检查  │   │          │ │
│  └──────────┘   └──────────┘   └──────────┘   └──────────┘ │
└─────────────────────────────────────────────────────────────┘
                              ↓
                    ┌─────────────────┐
                    │  Interaction    │
                    │   Service       │
                    │                 │
                    │ 选边追踪        │
                    │ 圈选区域        │
                    │ 缺口检测        │
                    └─────────────────┘
```

### 1.3 服务调用链

```
process_with_services()
    ↓
1. ParserService::process(request)    → Response<ParseResult>
2. TopoService::process(request)      → Response<SceneState>
3. ValidatorService::process(request) → Response<ValidationReport>
4. ExportService::process(request)    → Response<ExportResult>
    ↓
ProcessResult {
    scene: SceneState,
    validation: ValidationReport,
    output_bytes: Vec<u8>
}
```

**当前部署**: 单体部署（进程内服务调用）
**P2 计划**: HTTP/gRPC 微服务部署

---

## 二、服务详细设计

### 2.1 Parser Service

**职责**: 解析 DXF/DWG/PDF/SVG/STL 文件，输出标准化几何原语

**输入**:
```rust
pub struct ParserRequest {
    pub file_path: PathBuf,
    pub config: ParserConfig,
}
```

**输出**:
```rust
pub struct ParseResult {
    pub entities: Vec<RawEntity>,
    pub layers: Vec<LayerInfo>,
    pub units: LengthUnit,
    pub metadata: FileMetadata,
}
```

**核心功能**:
- DXF 实体解析（LINE, LWPOLYLINE, ARC, CIRCLE, SPLINE, ELLIPSE, HATCH）
- DWG 文件解析（R13-R2018，通过 libredwg 转换器）
- SVG 导入/导出（RawEntity ↔ SVG XML）
- STL 解析（二进制/ASCII → `RawEntity::Triangle`）
- PDF 矢量/光栅判定 + 文字提取
- 嵌套块递归展开
- NURBS 曲率自适应采样
- HATCH 填充边界提取
- 智能图层识别
- 单位解析与标定
- 文件缓存（Content-Hash 去重）

**依赖**: `dxf` crate, `lopdf` crate, `curvo` crate, `usvg` crate, `stl_io` crate

---

### 2.2 Vectorize Service

**职责**: 光栅图像矢量化

**输入**:
```rust
pub struct VectorizeRequest {
    pub image: DynamicImage,
    pub config: VectorizeConfig,
}
```

**输出**:
```rust
pub struct VectorizeResult {
    pub polylines: Vec<Polyline>,
    pub quality_score: f64,
    pub report: QualityReport,
}
```

**处理流程**:
```
图像预处理 → 边缘检测 → 骨架化 → 轮廓提取 → 矢量化 → 质量评估
```

**核心算法**:
- 自适应阈值二值化（Otsu）
- Sobel/Canny 边缘检测
- Zhang-Suen 骨架化
- DFS 轮廓提取（迭代实现）
- Douglas-Peucker 简化（迭代实现）
- Kåsa 圆弧拟合
- R*-tree 端点吸附

**可选加速**: OpenCV feature（4.5x 性能提升）

---

### 2.3 Topo Service

**职责**: 拓扑构建与几何清洗

**输入**:
```rust
pub struct TopoRequest {
    pub polylines: Vec<Polyline>,
    pub config: TopoConfig,
}
```

**输出**:
```rust
pub struct TopologyResult {
    pub points: Vec<Point2>,
    pub edges: Vec<(usize, usize)>,
    pub outer: Option<ClosedLoop>,
    pub holes: Vec<ClosedLoop>,
}
```

**核心算法**:
1. **端点吸附**: R*-tree 空间索引，O(n log n)
2. **交点切分**: Bentley-Ottmann 扫描线
3. **平面图构建**: 节点 - 边 - 邻接表
4. **闭合环提取**: DFS + 夹角最小启发式
5. **孔洞判定**: 负面积 + 射线法

**性能**:
- 100 线段：13.4ms
- 1000 线段：131.9ms
- 复杂度：O(n log n)

**P2 阶段增强计划**（P11 锐评后优先级调整）:
- [x] **Halfedge 结构实现** (P0 优先级，已完成)
  - 半边数据结构用于复杂拓扑处理
  - 支持嵌套孔洞（孔中孔）场景
  - 支持非流形几何处理
  - 面枚举算法优化
  - 孔洞遍历优化
  - **状态**: ✅ 已完成集成，`TopoAlgorithm::Halfedge` 为默认算法
- [ ] **并发处理优化** (P1 优先级)
  - rayon 并行化实际使用
  - 大文件并行解析
  - 多线程几何处理
  - **状态**: rayon 依赖已引入，待扩展到 parser/vectorize 全流程

---

### 2.4.1 Halfedge 结构（已集成到主流程）

**状态**: ✅ 已完成集成（1076 行代码在 `crates/topo/src/halfedge.rs`）
**当前默认**: `TopoAlgorithm::Halfedge` 是默认算法，支持嵌套孔洞和非流形几何

**设计目标**:
- 处理复杂拓扑关系（多重孔洞、嵌套边界、非流形几何）
- 替代当前 DFS + 夹角最小启发式方案
- 提供标准计算几何数据结构

**数据结构**:
```rust
/// 半边结构
pub struct Halfedge {
    pub origin: usize,           // 起点索引
    pub target: usize,           // 终点索引
    pub twin: usize,             // 对称半边索引
    pub face: usize,             // 所属面索引
    pub next: usize,             // 下一条半边索引
    pub prev: usize,             // 上一条半边索引
    pub edge_data: EdgeMetadata, // 边元数据
}

/// 面（用于表示外边界和孔洞）
pub struct Face {
    pub halfedge: usize,  // 组成该面的任意一条半边
    pub is_outer: bool,   // 是否为外边界
    pub area: f64,        // 面积（正：外边界，负：孔洞）
}

/// Halfedge 网格
pub struct HalfedgeMesh {
    pub vertices: Vec<Point2>,
    pub halfedges: Vec<Halfedge>,
    pub faces: Vec<Face>,
}
```

**核心算法**:
1. **构建**: 从线段集合构建 Halfedge 网格
2. **面枚举**: 遍历所有半边，提取闭合面
3. **孔洞识别**: 通过面积符号判断外边界/孔洞
4. **嵌套关系**: 通过射线法判断孔洞包含关系

**与当前方案对比**:

| 特性 | 当前方案 (DFS) | Halfedge 方案 |
|------|---------------|--------------|
| 简单闭合环 | ✅ 支持 | ✅ 支持 |
| 嵌套孔洞（孔中孔） | ⚠️ 可能出错 | ✅ 完全支持 |
| 非流形几何 | ❌ 不支持 | ✅ 支持 |
| 面遍历 | O(n²) | O(n) |
| 拓扑查询 | O(n) | O(1) |

**实施计划**:
1. 第 1 周：Halfedge 数据结构实现 + 单元测试
2. 第 2 周：主流程集成 + E2E 测试验证

---

### 2.4 Validator Service

**职责**: 几何质量验证

**输入**:
```rust
pub struct ValidatorRequest {
    pub scene: &SceneState,
    pub config: ValidatorConfig,
}
```

**输出**:
```rust
pub struct ValidationReport {
    pub passed: bool,
    pub issues: Vec<ValidationIssue>,
}

pub struct ValidationIssue {
    pub code: String,      // E001, E002, W001, ...
    pub message: String,
    pub severity: Severity,
    pub recovery_suggestion: Option<RecoverySuggestion>,
}
```

**验证项**:
- `E001`: 环未闭合
- `E002`: 自相交
- `E003`: 孔洞在外边界外
- `W001`: 短边
- `W002`: 尖角

**恢复建议**:
```rust
pub struct RecoverySuggestion {
    pub action: String,
    pub config_change: Option<(String, serde_json::Value)>,
    pub priority: u8,  // 1-10
}
```

---

### 2.5 Export Service

**职责**: 场景导出

**输入**:
```rust
pub struct ExportRequest {
    pub scene: SceneState,
    pub format: ExportFormat,
}
```

**输出**:
```rust
pub struct ExportResult {
    pub bytes: Vec<u8>,
    pub format: String,
}
```

**支持格式**:
- JSON: 人类可读，带美化输出
- Binary: bincode 高性能二进制
- SVG: 矢量图形导出（Line→`<line>`, Polyline→`<polyline>/<polygon>`, Circle→`<circle>`, Arc→`<path>`, Text→`<text>`, Triangle→`<polygon>` XY 投影）

**Schema v1.2**:
```json
{
  "schema_version": "1.2",
  "units": "m",
  "coordinate_system": "right_handed_y_up",
  "geometry": {
    "outer": [[0,0],[10,0],[10,8],[0,8]],
    "holes": [[[2,2],[4,2],[4,3],[2,3]]]
  },
  "boundaries": [...],
  "sources": [...]
}
```

---

### 2.6 Interact Service

**职责**: 交互协同

**输入**: WebSocket/HTTP 消息流

**输出**: `InteractionState`

**功能**:
- **模式 A**: 选边追踪（Edge Picking + Auto Trace）
- **模式 B**: 圈选区域（Lasso/Polygon Selection）
- **缺口检测**: 开放端点配对
- **分层补全**: Snap → Bridge → Semantic → Manual
- **边界语义标注**: HardWall/AbsorptiveWall/Opening/Window/Door

**API**:
```rust
pub trait InteractService {
    async fn select_edge(&self, edge_id: usize) -> Result<TraceResult>;
    async fn auto_trace(&self, edge_id: usize) -> Result<TraceResult>;
    async fn lasso(&self, polygon: &[Point2]) -> Result<Vec<ClosedLoop>>;
    async fn detect_gaps(&self, tolerance: f64) -> Result<Vec<GapInfo>>;
    async fn apply_snap_bridge(&self, gap_id: usize) -> Result<()>;
    async fn set_boundary_semantic(&self, segment_id: usize, semantic: BoundarySemantic) -> Result<()>;
}
```

---

### 2.7 Orchestrator Service

**职责**: API 网关与流程编排

**API 端点**:
- `GET /health` - 健康检查
- `POST /process` - 处理文件
- `GET /ws` - WebSocket 连接

**处理流水线**:
```rust
pub async fn process_with_services(
    &self,
    file_path: &Path,
) -> Result<ProcessResult> {
    // 1. ParserService
    let parse_result = self.parser.process(request).await?;
    
    // 2. TopoService
    let topo_result = self.topo.process(request).await?;
    
    // 3. ValidatorService
    let validation = self.validator.process(request).await?;
    
    // 4. ExportService
    let export_result = self.export.process(request).await?;
    
    Ok(ProcessResult {
        scene: topo_result.scene,
        validation,
        output_bytes: export_result.bytes,
    })
}
```

---

## 三、数据类型

### 3.1 核心类型

```rust
// 几何原语
pub type Point2 = [f64; 2];
pub type Point3 = [f64; 3];
pub type Polyline = Vec<Point2>;

// 标准化实体
pub enum RawEntity {
    Line { start: Point2, end: Point2, metadata: EntityMetadata, semantic: Option<SemanticLabel> },
    Polyline { points: Vec<Point2>, closed: bool, metadata: EntityMetadata, semantic: Option<SemanticLabel> },
    Circle { center: Point2, radius: f64, metadata: EntityMetadata, semantic: Option<SemanticLabel> },
    Arc { center: Point2, radius: f64, start_angle: f64, end_angle: f64, metadata: EntityMetadata, semantic: Option<SemanticLabel> },
    Text { position: Point2, content: String, height: f64, metadata: EntityMetadata, semantic: Option<SemanticLabel> },
    Path { commands: Vec<PathCommand>, metadata: EntityMetadata, semantic: Option<SemanticLabel> },
    Triangle { vertices: [Point3; 3], normal: Point3, metadata: EntityMetadata, semantic: Option<SemanticLabel> },
    // BlockReference, Dimension, Hatch, Image, etc.
}

// 场景状态
pub struct SceneState {
    pub outer: Option<ClosedLoop>,
    pub holes: Vec<ClosedLoop>,
    pub boundaries: Vec<BoundarySegment>,
    pub sources: Vec<SoundSource>,
    pub units: LengthUnit,
}

// 闭合环
pub struct ClosedLoop {
    pub points: Vec<Point2>,
    pub signed_area: f64,
}
```

### 3.2 错误类型

```rust
pub enum CadError {
    DxfParseError {
        message: String,
        source: Option<Box<dyn Error>>,
        file: Option<PathBuf>,
    },
    TopologyConstructionError {
        stage: String,
        message: String,
    },
    ValidationFailed {
        issues: Vec<ValidationIssue>,
    },
    VectorizeFailed {
        message: String,
    },
    // ...
}
```

---

## 四、EaaS 架构实现

### 4.1 服务契约

每个服务实现统一的 trait：

```rust
pub trait Service: Send + Sync {
    type Request;
    type Response;
    type Error;
    
    async fn process(&self, request: Self::Request) 
        -> Result<Self::Response, Self::Error>;
}
```

### 4.2 服务通信

**当前**（单体部署）:
```rust
// 进程内直接调用
let result = service.process(request).await?;
```

**P2 计划**（微服务部署）:
```rust
// HTTP 远程调用
let response = client
    .post("http://parser-service/process")
    .json(&request)
    .send()
    .await?;

// gRPC 远程调用
let response = client.process(request).await?;
```

### 4.3 链路追踪

```rust
pub struct RequestMetadata {
    pub trace_id: String,      // 全局追踪 ID
    pub span_id: String,       // 当前 span ID
    pub parent_span_id: Option<String>,  // 父 span ID
    pub timestamp: u64,
}
```

支持 OpenTelemetry 集成（P2 阶段完整实现）。

---

## 五、性能优化

### 5.1 空间索引

使用 R*-tree 加速空间查询：

```rust
use rstar::RTree;

let mut rtree = RTree::bulk_load(endpoints);
let nearby = rtree.locate_within_distance(point, tolerance);
```

**复杂度**: O(n²) → O(n log n)

### 5.2 迭代算法

所有递归算法改为迭代实现，避免栈溢出：

- `douglas_peucker()` - 迭代实现
- `extract_contours()` - 迭代 DFS
- `subdivide_curve()` - 带最大深度限制

### 5.3 零拷贝优化

使用 `Arc<T>` 共享大对象：

```rust
pub struct ProcessResult {
    pub scene: Arc<SceneState>,
    pub validation: Arc<ValidationReport>,
}
```

**性能**: 1000 次 Arc::clone() = 12-13μs（深拷贝的 1/2000）

### 5.4 并发处理优化（P2 阶段计划）

**P11 锐评后 P1 优先级任务**

**现状**:
- rayon 依赖已引入但未实际使用
- 大文件（541,216 实体 PDF）解析约 1.5s
- 串行处理限制性能提升

**优化方案**:

```rust
use rayon::prelude::*;

// 1. 并行解析大文件
pub fn parse_entities_parallel(entities: &[Entity]) -> Vec<RawEntity> {
    entities.par_iter()
        .map(|e| parse_entity(e))
        .collect()
}

// 2. 并行几何处理
pub fn process_geometries_parallel(polylines: Vec<Polyline>) -> Vec<Polyline> {
    polylines.into_par_iter()
        .map(|poly| {
            let simplified = douglas_peucker(poly, 0.1);
            snap_endpoints(simplified, 0.01)
        })
        .collect()
}

// 3. 并行端点吸附
pub fn snap_endpoints_parallel(points: &[Point2], tolerance: f64) -> Vec<Point2> {
    points.par_iter()
        .map(|p| snap_to_nearest(p, tolerance))
        .collect()
}
```

**预期性能提升**:

| 操作 | 当前（串行） | 目标（并行） | 提升 |
|------|-------------|-------------|------|
| 大文件解析 | 1.5s | ~0.5s | 3x |
| 几何处理 | O(n) | O(n/cores) | 4-8x |
| 端点吸附 | O(n log n) | O((n/cores) log n) | 3-5x |

**实施计划**:
1. rayon 并行化集成 + 基准测试
2. 性能验证 + 回归测试

---

## 六、测试策略

### 6.1 测试层次

```
单元测试 → 集成测试 → E2E 测试 → 用户故事测试
```

### 6.2 测试覆盖

| 测试类型 | 数量 | 说明 |
|----------|------|------|
| 单元测试 | 133 | 各 crate 内部逻辑 |
| 边界测试 | 14 | 极端情况处理 |
| NURBS 测试 | 6 | 曲线离散化验证 |
| 真实文件测试 | 20 | 9 个 DXF + 4 个 PDF |
| 基准测试 | 19 | 性能验证 |
| 集成测试 | 7 | 服务间集成 |
| E2E 测试 | 6 | 完整流程 |
| 用户故事测试 | 6 | 实际工作流 |

**总计**: 585+ 测试（584 通过，1 已知失败）

### 6.3 性能回归

CI/CD 集成性能回归检测：
- 警告阈值：10%
- Blocking 阈值：20%

---

## 七、部署架构

### 7.1 当前（单体）

```
┌─────────────────────┐
│   cad-binary        │
│  ┌───────────────┐  │
│  │ ParserSvc     │  │
│  │ TopoSvc       │  │
│  │ ValidatorSvc  │  │
│  │ ExportSvc     │  │
│  └───────────────┘  │
└─────────────────────┘
```

**接口支持**:
- ✅ HTTP API: `/health`, `/process`
- ✅ WebSocket: `/ws` (实时交互：选边/缺口检测/ping)
- ✅ CLI: `cad-cli` 命令

### 7.2 P2 计划（微服务）

```
┌─────────────┐     ┌─────────────┐
│  API Gateway│────→│ ParserSvc   │
└─────────────┘     └─────────────┘
       ↓                   ↓
┌─────────────┐     ┌─────────────┐
│  TopoSvc    │←────│ VectorizeSvc│
└─────────────┘     └─────────────┘
       ↓
┌─────────────┐
│ValidatorSvc │
└─────────────┘
```

---

## 八、技术栈

| 类别 | Crate | 用途 |
|------|-------|------|
| **Web** | axum 0.7 | HTTP API |
| **异步** | tokio | 异步运行时 |
| **几何** | geo 0.28 | 基础几何算法 |
| **空间索引** | rstar 0.12 | R*-tree |
| **线性代数** | nalgebra 0.34 | 矩阵/向量 |
| **NURBS** | curvo 0.1 | NURBS 曲线 |
| **图像** | image 0.25 | 图像处理 |
| **OpenCV** | opencv 0.92 | 可选加速 |
| **CAD** | dxf 0.6 | DXF 解析 |
| **CAD** | acadrust | DWG 解析 |
| **PDF** | lopdf 0.34 | PDF 解析 |
| **SVG** | usvg 0.44 | SVG 导入 |
| **STL** | stl_io 0.9 | STL 解析 |
| **序列化** | serde + json | JSON |
| **二进制** | bincode | 高性能二进制 |

---

**最后更新**: 2026 年 4 月 15 日
**版本**: v0.1.0
