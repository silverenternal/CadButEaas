# 后端 API 功能概览

**版本**: v0.1.0  
**最后更新**: 2026 年 3 月 22 日  
**面向**: 甲方技术评审、集成开发人员

---

## 一、API 概述

### 1.1 接口类型

系统提供三种接口类型：

| 接口类型 | 协议 | 用途 | 状态 |
|----------|------|------|------|
| **HTTP REST** | HTTP/1.1 | 文件处理、配置管理、声学分析 | ✅ 已完成 |
| **WebSocket** | WebSocket | 实时交互、进度推送 | ✅ 已完成 |
| **CLI** | 命令行 | 批处理、脚本集成 | ✅ 已完成 |

### 1.2 服务地址

| 环境 | 地址 | 说明 |
|------|------|------|
| **开发环境** | `http://localhost:3000` | 本地启动 |
| **生产环境** | 待定 | 部署后配置 |

### 1.3 启动服务

```bash
# 启动 HTTP 服务（默认端口 3000）
cargo run --package cad-cli -- serve --port 3000

# 使用预设配置启动
cargo run --package cad-cli -- serve --port 3000 --profile architectural
```

---

## 二、核心 API 端点

### 2.1 基础 API

#### 2.1.1 健康检查

**端点**: `GET /health`

**描述**: 检查服务健康状态

**响应示例**:
```json
{
  "status": "healthy",
  "version": "0.1.0",
  "api_version": "v1"
}
```

**状态码**:
| 状态 | 说明 |
|------|------|
| `healthy` | 服务正常 |
| `unhealthy` | 服务异常 |
| `degraded` | 服务降级 |

---

#### 2.1.2 处理文件

**端点**: `POST /process`

**描述**: 上传并处理 DXF/PDF 文件，返回处理结果

**请求**:
- **Content-Type**: `multipart/form-data`
- **表单字段**: `file` (required) - 文件二进制数据

**响应示例**:
```json
{
  "job_id": "uuid-123456",
  "status": "completed",
  "message": "处理完成",
  "result": {
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
}
```

**curl 示例**:
```bash
curl -X POST http://localhost:3000/process \
  -F "file=@/path/to/floor_plan.dxf"
```

---

### 2.2 配置 API

#### 2.2.1 列出预设配置

**端点**: `GET /config/profiles`

**描述**: 列出所有可用的预设配置

**响应示例**:
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

#### 2.2.2 获取预设配置详情

**端点**: `GET /config/profile/:name`

**描述**: 获取指定预设配置的详细参数

**路径参数**:
- `name`: 预设配置名称

**响应示例**:
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

### 2.3 交互 API

#### 2.3.1 选边自动追踪

**端点**: `POST /interact/auto_trace`

**描述**: 从选中的边自动追踪闭合环

**请求体**:
```json
{
  "edge_id": 42
}
```

**响应示例**:
```json
{
  "success": true,
  "loop_points": [[0,0],[10,0],[10,8],[0,8]],
  "message": "追踪到闭合环"
}
```

---

#### 2.3.2 圈选区域

**端点**: `POST /interact/lasso`

**描述**: 从圈选多边形中提取闭合环

**请求体**:
```json
{
  "polygon": [[0,0],[20,0],[20,15],[0,15]]
}
```

**响应示例**:
```json
{
  "selected_edges": [1, 5, 12, 18],
  "loops": [[[2,2],[8,2],[8,6],[2,6]]],
  "connected_components": 1
}
```

---

#### 2.3.3 缺口检测

**端点**: `POST /interact/detect_gaps`

**描述**: 检测边界上的缺口

**请求体**:
```json
{
  "tolerance": 2.0
}
```

**响应示例**:
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

#### 2.3.4 缺口桥接

**端点**: `POST /interact/snap_bridge`

**描述**: 应用端点吸附补全缺口

**请求体**:
```json
{
  "gap_id": 0
}
```

**响应示例**:
```json
{
  "success": true,
  "message": "缺口已桥接"
}
```

---

#### 2.3.5 设置边界语义

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
| 语义值 | 说明 |
|--------|------|
| `hard_wall` | 硬墙 |
| `absorptive_wall` | 吸声墙 |
| `opening` | 开口/门洞 |
| `window` | 窗户 |
| `door` | 门 |
| `custom` | 自定义 |

**响应示例**:
```json
{
  "success": true,
  "message": "边界语义已设置"
}
```

---

#### 2.3.6 获取交互状态

**端点**: `GET /interact/state`

**描述**: 获取当前交互状态

**响应示例**:
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

### 2.4 声学分析 API

#### 2.4.1 选区材料统计

**端点**: `POST /acoustic/analyze`

**描述**: 计算选定区域内的材料分布和等效吸声面积

**请求体**:
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

**响应示例**:
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

---

#### 2.4.2 房间混响时间计算

**端点**: `POST /acoustic/analyze`

**描述**: 计算房间的混响时间 T60 和早期衰变时间 EDT

**请求体**:
```json
{
  "type": "ROOM_REVERBERATION",
  "room_id": 0,
  "formula": "SABINE",
  "room_height": 3.0
}
```

**响应示例**:
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

#### 2.4.3 多区域对比分析

**端点**: `POST /acoustic/analyze`

**描述**: 对比不同区域的声学特性

**请求体**:
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

**响应示例**:
```json
{
  "result": {
    "type": "COMPARATIVE_ANALYSIS",
    "zones": [
      {
        "name": "区域 A",
        "area": 100.0,
        "average_absorption": {
          "HZ500": 0.05,
          "HZ1K": 0.04
        }
      },
      {
        "name": "区域 B",
        "area": 300.0,
        "average_absorption": {
          "HZ500": 0.08,
          "HZ1K": 0.06
        }
      }
    ]
  },
  "computation_time": 0.078
}
```

---

## 三、WebSocket API

### 3.1 连接 WebSocket

**端点**: `GET /ws`

**描述**: 建立 WebSocket 连接，用于实时交互和进度推送

**JavaScript 示例**:
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

### 3.2 消息格式

#### 3.2.1 客户端 → 服务器

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

#### 3.2.2 服务器 → 客户端

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

## 四、命令行接口 (CLI)

### 4.1 处理文件

```bash
# 处理 DXF 文件
cad process input.dxf --output scene.json

# 使用预设配置
cad process input.dxf --profile architectural --output scene.json

# 处理 PDF 文件
cad process input.pdf --profile scanned --output scene.json

# 自定义参数
cad process input.dxf \
  --snap-tolerance 0.5 \
  --min-line-length 1.0 \
  --closure-tolerance 0.3
```

### 4.2 配置管理

```bash
# 列出预设配置
cad list-profiles

# 显示预设配置详情
cad show-profile architectural
```

### 4.3 启动服务

```bash
# 启动 HTTP 服务
cad serve --port 3000

# 使用预设配置启动
cad serve --port 3000 --profile architectural
```

---

## 五、错误处理

### 5.1 错误响应格式

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

### 5.2 错误代码分类

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

#### 错误代码与恢复建议

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

## 六、输出 Schema

### 6.1 Schema v1.2

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

### 6.2 字段说明

| 字段 | 类型 | 说明 |
|------|------|------|
| `schema_version` | string | Schema 版本号 |
| `units` | string | 单位 (mm/cm/m/inch 等) |
| `coordinate_system` | string | 坐标系 |
| `geometry.outer` | array | 外边界点序列 |
| `geometry.holes` | array | 洞点序列列表 |
| `boundaries` | array | 边界段材料/语义映射 |
| `sources` | array | 声源列表 |
| `seat_zones` | array | 座椅区域 |
| `render_config` | object | 渲染配置 |

---

## 七、快速集成示例

### 7.1 Python 集成

```python
import requests

# 处理 DXF 文件
with open('floor_plan.dxf', 'rb') as f:
    files = {'file': f}
    response = requests.post('http://localhost:3000/process', files=files)
    result = response.json()
    print(result)

# 声学分析
payload = {
    "type": "ROOM_REVERBERATION",
    "room_id": 0,
    "formula": "SABINE",
    "room_height": 3.0
}
response = requests.post('http://localhost:3000/acoustic/analyze', json=payload)
result = response.json()
print(result)
```

### 7.2 Node.js 集成

```javascript
const WebSocket = require('ws');

// WebSocket 连接
const ws = new WebSocket('ws://localhost:3000/ws');

ws.on('open', () => {
  // 发送选边消息
  ws.send(JSON.stringify({
    type: 'select_edge',
    payload: { edge_id: 42 }
  }));
});

ws.on('message', (data) => {
  const message = JSON.parse(data);
  console.log('收到:', message);
});
```

---

## 八、API 完整文档

完整的 API 文档请参考：[API.md](../API.md)

---

**最后更新**: 2026 年 3 月 22 日  
**版本**: v0.1.0
