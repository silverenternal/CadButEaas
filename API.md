# CAD 几何智能处理系统 - 后端 API 文档

**版本**: v0.1.0  
**最后更新**: 2026 年 3 月 21 日  
**作者**: CAD Team

---

## 目录

1. [概述](#1-概述)
2. [快速开始](#2-快速开始)
3. [HTTP API](#3-http-api)
4. [WebSocket API](#4-websocket-api)
5. [交互 API](#5-交互-api)
6. [声学分析 API](#6-声学分析-api)
7. [数据类型](#7-数据类型)
8. [错误处理](#8-错误处理)
9. [服务架构](#9-服务架构)
10. [配置管理](#10-配置管理)

---

## 1. 概述

### 1.1 系统简介

CAD 几何智能处理系统是基于「一切皆服务」(EaaS) 设计哲学的工业级 CAD 几何智能处理系统，支持：

- **DXF 文件解析**（AutoCAD 矢量格式，AC1015 及以上版本）
- **PDF 文件处理**（矢量 PDF 直接提取 / 光栅 PDF 自动矢量化）
- **拓扑构建**（R*-tree 空间索引，Bentley-Ottmann 交点检测）
- **几何验证**（闭合性、自相交、孔洞关系检查）
- **声学分析**（选区材料统计、混响时间计算）
- **实时交互**（WebSocket 选边追踪、圈选区域）

### 1.2 支持的输入格式

| 格式 | 类型 | 支持实体 | 说明 |
|------|------|----------|------|
| **DXF** | 矢量 | LINE, LWPOLYLINE, ARC, CIRCLE, SPLINE, ELLIPSE, BLOCK/INSERT, HATCH, TEXT, DIMENSION | AutoCAD R14 (AC1015) 及以上 |
| **PDF** | 矢量 | LINE, PATH, RECT, CURVE | 直接提取矢量图元 |
| **PDF** | 光栅 | 图像 | 自动矢量化（边缘检测 + 骨架化 + 轮廓提取） |

### 1.3 输出格式

| 格式 | 扩展名 | 说明 | 适用场景 |
|------|--------|------|----------|
| **JSON** | `.json` | 人类可读，带美化输出 | 调试、数据交换 |
| **Bincode** | `.bin` | 高性能二进制格式 | 生产环境、快速加载 |

### 1.4 输出 Schema v1.2

```json
{
  "schema_version": "1.2",
  "units": "m",
  "coordinate_system": "right_handed_y_up",
  "geometry": {
    "outer": [[0,0],[10,0],[10,8],[0,8]],
    "holes": [[[2,2],[4,2],[4,3],[2,3]]]
  },
  "boundaries": [
    {
      "segment": [0, 1],
      "semantic": "hard_wall",
      "material": "concrete"
    }
  ],
  "sources": [
    {
      "id": "speaker-1",
      "position": [5.0, 4.0, 2.5],
      "source_type": "omnidirectional",
      "gain_db": 0.0,
      "delay_ms": 0.0
    }
  ],
  "seat_zones": [],
  "render_config": {
    "recommended_lod": "detailed",
    "seat_render_threshold": 500,
    "auto_lod": true
  }
}
```

---

## 2. 快速开始

### 2.1 启动 HTTP 服务

```bash
# 默认端口 3000
cargo run --package cad-cli -- serve --port 3000

# 使用预设配置启动
cargo run --package cad-cli -- serve --port 3000 --profile architectural
```

### 2.2 健康检查

```bash
curl http://localhost:3000/health
```

**响应示例**:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "api_version": "v1"
}
```

### 2.3 处理文件

```bash
# 处理 DXF 文件
curl -X POST http://localhost:3000/process \
  -F "file=@/path/to/file.dxf"

# 处理 PDF 文件
curl -X POST http://localhost:3000/process \
  -F "file=@/path/to/file.pdf"
```

---

## 3. HTTP API

### 3.1 基础 API

#### 3.1.1 健康检查

**端点**: `GET /health`

**描述**: 检查服务健康状态

**响应**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `status` | string | `healthy` / `unhealthy` / `degraded` |
| `version` | string | 服务版本 |
| `api_version` | string | API 版本 |

**示例**:
```bash
curl http://localhost:3000/health
```

---

#### 3.1.2 处理文件

**端点**: `POST /process`

**描述**: 上传并处理 DXF/PDF 文件，返回渐进式渲染结果

**请求**:
- **Content-Type**: `multipart/form-data`
- **表单字段**:
  - `file` (required): 文件二进制数据

**响应**:

| 字段 | 类型 | 说明 |
|------|------|------|
| `job_id` | string | 任务 ID |
| `status` | string | `completed` / `partial` / `failed` |
| `message` | string | 状态消息 |
| `result` | object | 处理结果详情 |
| `edges` | array | 边数据（用于快速渲染） |

**ProcessResult 结构**:
```json
{
  "scene_summary": {
    "outer_boundaries": 1,
    "holes": 2,
    "total_points": 156
  },
  "validation_summary": {
    "error_count": 0,
    "warning_count": 2,
    "passed": true
  },
  "output_size": 4096
}
```

**示例**:
```bash
curl -X POST http://localhost:3000/process \
  -F "file=@/path/to/floor_plan.dxf" \
  -H "Authorization: Bearer <token>"
```

**渐进式渲染流程**:

1. **阶段 1（快速，~1 秒）**: 解析 DXF → 提取原始边 → 立即返回
2. **阶段 2（后台）**: 构建拓扑 → 完成后通过 WebSocket 推送更新

---

### 3.2 配置 API

#### 3.2.1 列出预设配置

**端点**: `GET /config/profiles`

**描述**: 列出所有可用的预设配置

**响应**:
```json
{
  "profiles": [
    {
      "name": "architectural",
      "description": "建筑图纸预设"
    },
    {
      "name": "mechanical",
      "description": "机械图纸预设"
    },
    {
      "name": "scanned",
      "description": "扫描图纸预设"
    },
    {
      "name": "quick",
      "description": "快速原型预设"
    }
  ]
}
```

---

#### 3.2.2 获取预设配置详情

**端点**: `GET /config/profile/:name`

**描述**: 获取指定预设配置的详细参数

**路径参数**:
- `name`: 预设配置名称

**响应**:
```json
{
  "name": "architectural",
  "topology": {
    "snap_tolerance_mm": 0.5,
    "min_line_length_mm": 1.0,
    "merge_angle_tolerance_deg": 5.0,
    "max_gap_bridge_length_mm": 2.0
  },
  "validator": {
    "closure_tolerance_mm": 0.3,
    "min_area_m2": 0.5,
    "min_edge_length_mm": 100.0,
    "min_angle_deg": 15.0
  },
  "export": {
    "format": "json",
    "json_indent": 2,
    "auto_validate": true
  }
}
```

---

## 4. WebSocket API

### 4.1 连接 WebSocket

**端点**: `GET /ws`

**描述**: 建立 WebSocket 连接，用于实时交互和进度推送

**连接示例** (JavaScript):
```javascript
const ws = new WebSocket('ws://localhost:3000/ws');

ws.onopen = () => {
  console.log('WebSocket 连接已建立');
};

ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  console.log('收到消息:', message);
};

ws.onerror = (error) => {
  console.error('WebSocket 错误:', error);
};
```

---

### 4.2 消息格式

#### 4.2.1 客户端 → 服务器消息

| 消息类型 | 描述 | 负载结构 |
|----------|------|----------|
| `select_edge` | 选择边 | `{ "edge_id": number }` |
| `auto_trace` | 自动追踪 | `{ "edge_id": number }` |
| `lasso` | 圈选区域 | `{ "polygon": number[][] }` |
| `detect_gaps` | 缺口检测 | `{ "tolerance": number }` |
| `snap_bridge` | 缺口桥接 | `{ "gap_id": number }` |
| `set_boundary_semantic` | 设置边界语义 | `{ "segment_id": number, "semantic": string }` |
| `ping` | 心跳检测 | `{}` |

**发送示例**:
```javascript
ws.send(JSON.stringify({
  type: 'select_edge',
  payload: { edge_id: 42 }
}));
```

---

#### 4.2.2 服务器 → 客户端消息

| 消息类型 | 描述 | 负载结构 |
|----------|------|----------|
| `edge_selected` | 边选择结果 | `{ "edge_id": number, "trace_result": object }` |
| `auto_trace_result` | 自动追踪结果 | `{ "loop_points": number[][], "path": number[] }` |
| `lasso_result` | 圈选结果 | `{ "selected_edges": number[], "loops": number[][][] }` |
| `gap_detection` | 缺口检测结果 | `{ "gaps": GapInfo[] }` |
| `topology_update` | 拓扑更新推送 | `{ "edges": Edge[], "loops": Loop[] }` |
| `parse_progress` | 解析进度 | `{ "stage": string, "progress": number }` |
| `pong` | 心跳响应 | `{ "latency_ms": number }` |

**接收示例**:
```javascript
ws.onmessage = (event) => {
  const message = JSON.parse(event.data);
  
  switch (message.type) {
    case 'topology_update':
      console.log('拓扑更新:', message.payload.edges);
      break;
    case 'parse_progress':
      console.log(`解析进度：${message.payload.stage} - ${message.payload.progress}%`);
      break;
  }
};
```

---

### 4.3 交互 API 端点

#### 4.3.1 选边自动追踪

**端点**: `POST /interact/auto_trace`

**描述**: 从选中的边自动追踪闭合环

**请求体**:
```json
{
  "edge_id": 42
}
```

**响应**:
```json
{
  "success": true,
  "loop_points": [[0,0],[10,0],[10,8],[0,8]],
  "message": "追踪到闭合环"
}
```

---

#### 4.3.2 圈选区域

**端点**: `POST /interact/lasso`

**描述**: 从圈选多边形中提取闭合环

**请求体**:
```json
{
  "polygon": [[0,0],[20,0],[20,15],[0,15]]
}
```

**响应**:
```json
{
  "selected_edges": [1, 5, 12, 18],
  "loops": [[[2,2],[8,2],[8,6],[2,6]]],
  "connected_components": 1
}
```

---

#### 4.3.3 缺口检测

**端点**: `POST /interact/detect_gaps`

**描述**: 检测边界上的缺口

**请求体**:
```json
{
  "tolerance": 2.0
}
```

**响应**:
```json
{
  "gaps": [
    {
      "id": 0,
      "start": [10.0, 0.0],
      "end": [10.5, 0.0],
      "length": 0.5,
      "gap_type": "collinear"
    }
  ],
  "total_count": 1
}
```

---

#### 4.3.4 缺口桥接

**端点**: `POST /interact/snap_bridge`

**描述**: 应用端点吸附补全缺口

**请求体**:
```json
{
  "gap_id": 0
}
```

**响应**:
```json
{
  "success": true,
  "message": "缺口已桥接"
}
```

---

#### 4.3.5 设置边界语义

**端点**: `POST /interact/set_boundary_semantic`

**描述**: 为边界段设置语义标注

**请求体**:
```json
{
  "segment_id": 5,
  "semantic": "hard_wall"
}
```

**语义类型**:
- `hard_wall`: 硬墙
- `absorptive_wall`: 吸声墙
- `opening`: 开口/门洞
- `window`: 窗户
- `door`: 门

**响应**:
```json
{
  "success": true,
  "message": "边界语义已设置"
}
```

---

#### 4.3.6 获取交互状态

**端点**: `GET /interact/state`

**描述**: 获取当前交互状态

**响应**:
```json
{
  "total_edges": 156,
  "selected_edges": [1, 5, 12],
  "detected_gaps": [
    {
      "id": 0,
      "start": [10.0, 0.0],
      "end": [10.5, 0.0],
      "length": 0.5,
      "gap_type": "collinear"
    }
  ]
}
```

---

## 5. 交互 API

### 5.1 交互模式

系统支持两种交互模式：

#### 模式 A：选边追踪 (Edge Picking + Auto Trace)

用户点选一段墙线后，系统沿拓扑自动追踪形成闭合环：

1. 在每个节点选择"最可能延续"的下一条边
2. 优先小转角、同层/同属性
3. 遇到分叉时提示用户选择
4. 成功闭合则输出一个环

**API 调用序列**:
```
1. POST /interact/auto_trace { edge_id: 1 }
   → 返回追踪路径
2. WebSocket: select_edge { edge_id: 1 }
   ← 收到 edge_selected 事件
3. POST /interact/set_boundary_semantic { segment_id: 1, semantic: "hard_wall" }
   ← 边界语义已标注
```

---

#### 模式 B：圈选区域 (Lasso/Polygon Selection)

用户圈定区域后，系统在选区内提取闭合环：

1. 连通组件分析，提取闭合候选
2. 按面积与闭合度排序
3. 最大可用区域通常为主区域
4. 其它闭合环作为洞/障碍物候选

**API 调用序列**:
```
1. POST /interact/lasso { polygon: [[0,0],[20,0],[20,15],[0,15]] }
   → 返回选中的边和闭合环
2. POST /interact/detect_gaps { tolerance: 2.0 }
   → 返回检测到的缺口
3. POST /interact/snap_bridge { gap_id: 0 }
   → 桥接缺口
```

---

### 5.2 缺口检测与修复

#### 缺口类型

| 类型 | 描述 | 处理策略 |
|------|------|----------|
| `collinear` | 共线缺口（两端方向一致） | 自动桥接 |
| `orthogonal` | 正交缺口（两端方向垂直） | 手动确认 |
| `angled` | 斜角缺口 | 手动确认 |
| `small` | 小缺口（长度<2mm） | 自动吸附 |

#### 分层补全策略

1. **端点吸附 (Snap)**: 端点距离<ε时合并
2. **短缺口桥接 (Auto Bridge)**: 缺口短且方向一致时自动补线
3. **门洞语义补全**: 用户指定缺口为"封闭边界"或"开口边界"
4. **受控绘制**: 提供吸附、正交约束工具手动补全

---

## 6. 声学分析 API

### 6.1 声学分析请求

**端点**: `POST /acoustic/analyze`

**描述**: 执行声学分析（选区材料统计、混响时间计算）

**请求体** (选区材料统计):
```json
{
  "type": "SELECTION_MATERIAL_STATS",
  "boundary": {
    "type": "RECT",
    "min": [0.0, 0.0],
    "max": [10.0, 10.0]
  },
  "mode": "SMART"
}
```

**请求体** (房间级混响时间):
```json
{
  "type": "ROOM_REVERBERATION",
  "room_id": 0,
  "formula": "SABINE",
  "room_height": 3.0
}
```

**请求体** (多区域对比分析):
```json
{
  "type": "COMPARATIVE_ANALYSIS",
  "selections": [
    {
      "name": "区域 A",
      "boundary": {
        "type": "RECT",
        "min": [0.0, 0.0],
        "max": [10.0, 10.0]
      }
    },
    {
      "name": "区域 B",
      "boundary": {
        "type": "POLYGON",
        "points": [[0,0],[20,0],[20,15],[0,15]]
      }
    }
  ],
  "metrics": ["AREA", "AVERAGE_ABSORPTION", "EQUIVALENT_ABSORPTION_AREA"]
}
```

---

### 6.2 声学分析响应

**选区材料统计响应**:
```json
{
  "result": {
    "type": "SELECTION_MATERIAL_STATS",
    "surface_ids": [1, 5, 12, 18],
    "total_area": 45.6,
    "material_distribution": [
      {
        "material_name": "concrete",
        "area": 30.2,
        "percentage": 66.2
      },
      {
        "material_name": "glass",
        "area": 15.4,
        "percentage": 33.8
      }
    ],
    "equivalent_absorption_area": {
      "HZ500": 2.34,
      "HZ1K": 1.87
    },
    "average_absorption_coefficient": {
      "HZ500": 0.051,
      "HZ1K": 0.041
    }
  },
  "computation_time": 0.045,
  "metrics": {
    "surface_count": 4,
    "computation_time_ms": 45.0
  }
}
```

**房间混响时间响应**:
```json
{
  "result": {
    "type": "ROOM_REVERBERATION",
    "volume": 136.8,
    "total_surface_area": 152.4,
    "formula": "SABINE",
    "t60": {
      "HZ125": 1.82,
      "HZ250": 1.65,
      "HZ500": 1.51,
      "HZ1K": 1.42,
      "HZ2K": 1.38,
      "HZ4K": 1.35
    },
    "edt": {
      "HZ125": 1.65,
      "HZ250": 1.50,
      "HZ500": 1.38,
      "HZ1K": 1.30,
      "HZ2K": 1.26,
      "HZ4K": 1.23
    }
  },
  "computation_time": 0.032,
  "metrics": {
    "surface_count": 12,
    "computation_time_ms": 32.0
  }
}
```

---

### 6.3 频率定义

声学分析使用的倍频程频率：

| 枚举值 | 频率 | 说明 |
|--------|------|------|
| `HZ125` | 125 Hz | 低频 |
| `HZ250` | 250 Hz | 中低频 |
| `HZ500` | 500 Hz | 中频（参考频率） |
| `HZ1K` | 1000 Hz | 中高频 |
| `HZ2K` | 2000 Hz | 高频 |
| `HZ4K` | 4000 Hz | 超高频 |

---

### 6.4 混响时间公式

| 公式 | 适用条件 | 计算公式 |
|------|----------|----------|
| `SABINE` | α < 0.2（低吸声房间） | T60 = 0.161 × V / A |
| `EYRING` | α > 0.2（高吸声房间） | T60 = 0.161 × V / (-S × ln(1-α)) |
| `AUTO` | 自动选择 | 根据平均吸声系数自动选择 |

---

## 7. 数据类型

### 7.1 几何类型

#### Point2 (2D 点)
```typescript
type Point2 = [number, number]  // [x, y]，单位：mm
```

#### Point3 (3D 点)
```typescript
type Point3 = [number, number, number]  // [x, y, z]，单位：mm
```

#### Polyline (多段线)
```typescript
type Polyline = Point2[]
```

#### ClosedLoop (闭合环)
```typescript
interface ClosedLoop {
  points: Point2[]       // 点序列（首尾相连）
  signed_area: number    // 有符号面积（>0 为外轮廓，<0 为孔洞）
}
```

#### Edge (边)
```typescript
interface Edge {
  id: number
  start: Point2
  end: Point2
  layer?: string         // 所属图层
  is_wall: boolean       // 是否为墙体
  visible?: boolean      // 是否可见
  line_style?: LineStyle // 线型
  line_width?: LineWidth // 线宽
}
```

---

### 7.2 语义类型

#### BoundarySemantic (边界语义)
```typescript
type BoundarySemantic =
  | "hard_wall"          // 硬墙
  | "absorptive_wall"    // 吸声墙
  | "opening"            // 开口/门洞
  | "window"             // 窗户
  | "door"               // 门
  | "custom"             // 自定义
```

#### LineStyle (线型)
```typescript
type LineStyle =
  | "solid"              // 实线
  | "dashed"             // 虚线
  | "dotted"             // 点线
  | "dash_dot"           // 点划线
  | "dash_dot_dot"       // 双点划线
  | "long_dash"          // 长划线
  | "long_dash_dot"      // 长划 - 点 - 长划
  | "custom"             // 自定义
```

#### LineWidth (线宽，24 级)
```typescript
type LineWidth =
  | "w0"   // 0.00mm
  | "w1"   // 0.05mm
  | "w2"   // 0.09mm
  | "w3"   // 0.13mm
  | "w4"   // 0.15mm
  | "w5"   // 0.18mm
  | "w6"   // 0.20mm
  | "w7"   // 0.25mm
  | "w8"   // 0.30mm
  | "w9"   // 0.35mm
  | "w10"  // 0.40mm
  | "w11"  // 0.50mm
  | "w12"  // 0.53mm
  | "w13"  // 0.60mm
  | "w14"  // 0.70mm
  | "w15"  // 0.80mm
  | "w16"  // 0.90mm
  | "w17"  // 1.00mm
  | "w18"  // 1.06mm
  | "w19"  // 1.20mm
  | "w20"  // 1.40mm
  | "w21"  // 1.58mm
  | "w22"  // 2.00mm
  | "w23"  // 2.11mm
  | "by_layer"  // ByLayer（跟随图层）
```

---

### 7.3 场景类型

#### SceneState (场景状态)
```typescript
interface SceneState {
  outer: ClosedLoop | null           // 外轮廓
  holes: ClosedLoop[]                // 孔洞列表
  boundaries: BoundarySegment[]      // 边界语义标注
  sources: SoundSource[]             // 声源列表
  edges: RawEdge[]                   // 原始边数据（用于前端显示）
  units: LengthUnit                  // 单位
  coordinate_system: CoordinateSystem // 坐标系
  seat_zones: SeatZone[]             // 座椅区域
  render_config?: RenderConfig       // 渲染配置
}
```

#### LengthUnit (长度单位)
```typescript
type LengthUnit =
  | "mm"          // 毫米
  | "cm"          // 厘米
  | "m"           // 米
  | "inch"        // 英寸
  | "foot"        // 英尺
  | "yard"        // 码
  | "mile"        // 英里
  | "micron"      // 微米
  | "kilometer"   // 千米
  | "point"       // 点 (1/72 英寸)
  | "pica"        // 派卡 (12 点)
  | "unspecified" // 未指定
```

---

### 7.4 验证类型

#### ValidationIssue (验证问题)
```typescript
interface ValidationIssue {
  code: string           // 错误代码 (E001, W001, ...)
  message: string        // 问题描述
  severity: Severity     // 严重性
  location?: Location    // 位置信息
  suggestion?: string    // 修复建议
}
```

#### Severity (严重性)
```typescript
type Severity =
  | "INFO"       // 信息
  | "WARNING"    // 警告
  | "ERROR"      // 错误
  | "CRITICAL"   // 严重错误
```

#### 错误代码表

| 代码 | 严重性 | 描述 | 修复建议 |
|------|--------|------|----------|
| `E000` | Error | 缺少外轮廓 | 确保图纸包含闭合外轮廓 |
| `E001` | Error | 环未闭合 | 调整端点位置或增大闭合容差 |
| `E002` | Error | 自相交多边形 | 在交点处切分线段 |
| `E003` | Error | 孔洞在外边界外 | 重新定位孔洞 |
| `W001` | Warning | 短边 | 增大最小边长阈值 |
| `W002` | Warning | 尖角 | 添加圆弧过渡 |
| `W003` | Warning | 未指定单位 | 标定单位或指定参考尺寸 |
| `I001` | Info | 未标注边界语义 | 为边界段添加语义标注 |

---

## 8. 错误处理

### 8.1 错误响应格式

```json
{
  "request_id": "req-123456789",
  "status": "FAILURE",
  "error": {
    "code": "DXF_PARSE_ERROR",
    "message": "DXF 解析失败：无效的实体句柄",
    "details": {
      "type": "DXF_PARSE",
      "file": "/path/to/file.dxf",
      "line": 1234,
      "raw_error": "Invalid entity handle"
    },
    "retryable": false,
    "suggestion": "检查 DXF 文件是否损坏，或尝试使用 AUDIT 命令修复"
  },
  "latency_ms": 45
}
```

---

### 8.2 错误代码分类

#### 解析错误 (PARSER)

| 错误码 | 描述 | 修复建议 |
|--------|------|----------|
| `DXF_PARSE_ERROR` | DXF 解析失败 | 使用 AUDIT 命令修复文件 |
| `PDF_PARSE_ERROR` | PDF 解析失败 | 检查 PDF 是否加密或损坏 |
| `UNSUPPORTED_FORMAT` | 不支持的文件格式 | 转换为 DXF 或 PDF 格式 |
| `RASTER_TOO_LARGE` | 光栅图像过大 | 降低图像分辨率或裁剪 |

#### 拓扑错误 (TOPOLOGY)

| 错误码 | 描述 | 修复建议 |
|--------|------|----------|
| `TOPOLOGY_CONSTRUCTION_FAILED` | 拓扑构建失败 | 检查线段是否过于密集 |
| `LOOP_EXTRACTION_FAILED` | 闭合环提取失败 | 增大端点吸附容差 |
| `INTERSECTION_DETECTION_FAILED` | 交点检测失败 | 跳过交点检测或简化几何 |

#### 验证错误 (VALIDATION)

| 错误码 | 描述 | 修复建议 |
|--------|------|----------|
| `VALIDATION_FAILED` | 验证失败 | 查看 issues 数组详情 |
| `CALIBRATION_FAILED` | 标定失败 | 确保标定点不重合 |
| `UNIT_NOT_SPECIFIED` | 未指定单位 | 使用两点标定功能 |

#### 内部错误 (INTERNAL)

| 错误码 | 描述 | 修复建议 |
|--------|------|----------|
| `SERVICE_UNAVAILABLE` | 服务不可用 | 检查服务状态 |
| `TIMEOUT` | 请求超时 | 增大超时时间或简化输入 |
| `PANIC` | 内部错误 | 查看日志详情 |

---

### 8.3 恢复建议系统

系统为每个错误提供恢复建议：

```json
{
  "code": "E001",
  "message": "环未闭合，端点间距 0.8mm",
  "suggestion": {
    "action": "检测到环未闭合。建议：1) 调整端点位置使其闭合 2) 增大闭合容差 3) 使用 InteractSvc 桥接缺口",
    "config_change": {
      "key": "validator.closure_tolerance_mm",
      "value": 0.45
    },
    "priority": 10
  }
}
```

**优先级说明**:
- 10: 最高优先级，必须处理
- 7-9: 高优先级，建议处理
- 4-6: 中优先级，可选处理
- 1-3: 低优先级，仅供参考

---

## 9. 服务架构

### 9.1 服务调用链

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
```

---

### 9.2 服务接口定义

每个服务实现统一的 `Service` trait：

```rust
#[async_trait]
pub trait Service: Send + Sync {
    type Payload;
    type Response;
    type Error;

    async fn process(&self, request: Request<Self::Payload>) 
        -> Result<Response<Self::Response>, Self::Error>;
    
    fn health_check(&self) -> ServiceHealth;
    fn version(&self) -> ServiceVersion;
    fn service_name(&self) -> &'static str;
    fn metrics(&self) -> &ServiceMetrics;
}
```

---

### 9.3 服务清单

| 服务 | Crate | 职责 | 性能指标 |
|------|-------|------|----------|
| **ParserService** | `parser` | DXF/PDF 解析 | 1000 实体 <100ms |
| **VectorizeService** | `vectorize` | 光栅矢量化 | 2000x2000 像素 ~800ms (OpenCV ~220ms) |
| **TopoService** | `topo` | 拓扑构建 | 1000 线段 ~132ms |
| **ValidatorService** | `validator` | 几何验证 | 100 环 ~15ms |
| **ExportService** | `export` | 场景导出 | JSON ~5ms, Binary ~2ms |
| **InteractionService** | `interact` | 交互协同 | 实时响应 <50ms |
| **AcousticService** | `acoustic` | 声学分析 | 选区统计 ~45ms |

---

### 9.4 部署架构

#### 当前部署（单体）

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

**特点**:
- 进程内服务调用，性能最优
- 单一二进制文件，部署简单
- 适合单机部署和开发环境

---

#### P2 计划（微服务）

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

**特点**:
- HTTP/gRPC 远程调用
- 支持独立部署和弹性伸缩
- 集成服务发现、熔断、链路追踪

---

## 10. 配置管理

### 10.1 配置文件格式

创建 `cad_config.toml` 自定义处理参数：

```toml
[topology]
snap_tolerance_mm = 0.5           # 端点吸附容差
min_line_length_mm = 1.0          # 最小线段长度
merge_angle_tolerance_deg = 5.0   # 合并角度容差
max_gap_bridge_length_mm = 2.0    # 最大缺口桥接长度
use_halfedge = true               # 使用 Halfedge 结构
skip_intersection_check = false   # 跳过交点检测
enable_parallel = true            # 启用并行处理
parallel_threshold = 1000         # 并行处理阈值

[validator]
closure_tolerance_mm = 0.3        # 闭合容差
min_area_m2 = 0.5                 # 最小面积
min_edge_length_mm = 100.0        # 最小边长
min_angle_deg = 15.0              # 最小角度

[export]
format = "json"                   # 导出格式：json / bincode
json_indent = 2                   # JSON 缩进（0 为紧凑输出）
auto_validate = true              # 自动验证

[parser.pdf]
threshold = 128                   # 二值化阈值 (0-255)
vectorize_tolerance_px = 2.0      # 矢量化容差（像素）
min_line_length_px = 10           # 最小线段长度（像素）
```

---

### 10.2 预设配置

系统提供 4 个预设配置：

#### architectural (建筑图纸)

适用于 AutoCAD 导出的建筑平面图：

```toml
[topology]
snap_tolerance_mm = 0.5
min_line_length_mm = 1.0
merge_angle_tolerance_deg = 5.0

[validator]
closure_tolerance_mm = 0.3
min_area_m2 = 0.5
min_edge_length_mm = 100.0
```

---

#### mechanical (机械图纸)

适用于高精度机械图纸：

```toml
[topology]
snap_tolerance_mm = 0.2
min_line_length_mm = 0.5
merge_angle_tolerance_deg = 2.0

[validator]
closure_tolerance_mm = 0.1
min_area_m2 = 0.1
min_edge_length_mm = 50.0
```

---

#### scanned (扫描图纸)

适用于线条清晰的扫描版图纸：

```toml
[parser.pdf]
threshold = 128
vectorize_tolerance_px = 2.0
min_line_length_px = 15

[topology]
snap_tolerance_mm = 1.0
min_line_length_mm = 2.0
```

---

#### quick (快速原型)

低精度要求，快速处理：

```toml
[topology]
snap_tolerance_mm = 2.0
min_line_length_mm = 5.0
merge_angle_tolerance_deg = 10.0

[validator]
closure_tolerance_mm = 1.0
min_area_m2 = 1.0
```

---

### 10.3 配置验证

```bash
# 验证配置文件
cargo run --package cad-cli -- validate-config cad_config.toml

# 查看预设配置
cargo run --package cad-cli -- list-profiles

# 查看预设配置详情
cargo run --package cad-cli -- show-profile architectural
```

---

## 附录

### A. 性能基准

#### Parser 性能
| 场景 | 实体数 | 时间 |
|------|--------|------|
| 小型会议室 | 100 | <10ms |
| 中型报告厅 | 300 | <30ms |
| 大型礼堂 | 1000 | <100ms |
| 超大 PDF | 541,216 | ~1.5s |

#### Topo 性能
| 线段数 | 时间 | 每线段 |
|--------|------|--------|
| 100 | 13.4ms | 134μs |
| 500 | 67.6ms | 135μs |
| 1000 | 131.9ms | 132μs |

**复杂度**: O(n log n) ✅

#### Vectorize 性能
| 像素 | 纯 Rust | OpenCV | 提升 |
|------|--------|--------|------|
| 500×500 | ~50ms | - | - |
| 1000×1000 | ~200ms | - | - |
| 2000×2000 | ~800ms | - | - |
| 2000×3000 | ~1000ms | ~220ms | **4.5x** |

---

### B. 测试覆盖

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

**总计**: 220+ 测试全部通过 (100% 通过率)  
**Clippy**: 0 警告

---

### C. 支持的文件格式

#### DXF 支持实体

| 实体类型 | 支持程度 | 说明 |
|----------|----------|------|
| LINE | ✅ 完全支持 | 直线段 |
| LWPOLYLINE | ✅ 完全支持 | 多段线（含闭合） |
| ARC | ✅ 完全支持 | 圆弧（离散化为线段） |
| CIRCLE | ✅ 完全支持 | 圆（离散化为 32 段） |
| SPLINE | ✅ 完全支持 | NURBS 精确离散化（弦高误差<0.1mm） |
| ELLIPSE | ✅ 完全支持 | 椭圆/椭圆弧 |
| BLOCK/INSERT | ✅ 完全支持 | 块定义与引用（嵌套块支持） |
| HATCH | ✅ 完全支持 | 填充图案（边界提取） |
| TEXT | ✅ 完全支持 | 文字（渲染为边界框） |
| DIMENSION | ✅ 完全支持 | 尺寸标注（渲染为尺寸线） |
| XREF | ⚠️ 部分支持 | 外部参照（待完整实现） |

#### PDF 支持

| PDF 类型 | 支持程度 | 说明 |
|----------|----------|------|
| 矢量 PDF | ✅ 完全支持 | 直接提取 LINE/PATH/RECT/CURVE |
| 光栅 PDF | ✅ 完全支持 | 自动矢量化（边缘检测 + 骨架化） |

---

### D. 常见问题 (FAQ)

#### Q1: 如何处理 DWG 文件？

DWG 是 Autodesk 专有格式，需要先转换为 DXF：

```bash
# 使用 AutoCAD
DWGTODXF input.dwg output.dxf

# 使用 LibreCAD
librecad --input input.dwg --export-dxf output.dxf
```

---

#### Q2: PDF 矢量化效果不佳怎么办？

1. **检查图像质量**: 确保扫描清晰、对比度高
2. **调整阈值**: `--threshold 100` (0-255)
3. **启用 OpenCV 加速**: `cargo build --features opencv`
4. **转换为 DXF**: 使用专业软件（如 Inkscape）手动矢量化

---

#### Q3: 如何处理超大文件？

1. **启用并行处理**:
   ```toml
   [topology]
   enable_parallel = true
   parallel_threshold = 1000
   ```

2. **跳过交点检测**（适用于已清理的 DXF）:
   ```toml
   [topology]
   skip_intersection_check = true
   ```

3. **使用渐进式渲染**: API 会先返回原始边，后台构建拓扑

---

#### Q4: 如何集成到现有工作流？

1. **CLI 集成**:
   ```bash
   cad process input.dxf --output scene.json
   ```

2. **HTTP API 集成**:
   ```bash
   curl -X POST http://localhost:3000/process -F "file=@input.dxf"
   ```

3. **Rust 库调用**:
   ```rust
   use orchestrator::OrchestratorService;
   
   let service = OrchestratorService::default();
   let result = service.process_file("input.dxf").await?;
   ```

---

### E. 版本历史

| 版本 | 日期 | 主要更新 |
|------|------|----------|
| v0.1.0 | 2026-03-11 | 初始稳定版本，220+ 测试通过 |
| v0.1.1 | 2026-03-21 | 完善 API 文档，添加 WebSocket 支持 |

---

**文档维护**: CAD Team  
**反馈**: https://github.com/your-org/cad/issues
