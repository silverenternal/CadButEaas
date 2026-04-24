# CAD 几何智能处理系统 - Claude Code 配置

## 项目概览

Rust 单体架构 CAD 几何智能处理系统，遵循「一切皆服务」(EaaS) 设计哲学。

- **版本**: v0.1.0
- **测试**: 585+ 测试（584 通过，1 预存在失败 `test_polyline_zero_length_edge_filtering`），Clippy 0 错误（4 个良性复杂度警告）
- **工作区**: 17 个 crates

## 开发命令

```bash
# 构建
cargo build --workspace

# 测试（580+ 测试）
cargo test --workspace

# Clippy 检查（必须 0 警告）
cargo clippy --workspace --lib

# 格式化
cargo fmt --workspace

# 基准测试
cargo test --test benchmarks -- --nocapture

# Release 构建（LTO + opt-level 3）
cargo build --release

# OpenCV 加速（可选 feature）
cargo build --release --features cad-cli/opencv
```

## 代码风格规则

### Rust

- **错误处理**: 使用 `thiserror` 定义错误类型，不使用 `unwrap()` / `expect()`（测试除外）
- **迭代优先**: 所有递归算法必须改为迭代实现（避免栈溢出）
- **零拷贝**: 大对象使用 `Arc<T>` 共享，避免深拷贝
- **命名**: Rust 标准命名约定（snake_case 函数/变量，PascalCase 类型）
- **注释**: 仅在逻辑不明显时添加注释，pub 项必须有文档注释

### 禁止事项

- 不要在代码中添加 emoji
- 不要添加 TODO/FIXME 注释——直接实现或不做
- 不要添加超出任务范围的错误处理或防御性代码
- 不要创建未使用的抽象层（helper/utility 仅在有实际复用时创建）
- 不要使用 `--no-verify` 跳过 pre-commit hooks
- 不要 force push 到 main/master

### TypeScript (cad-web/)

- 使用 biome 进行格式化和 linting
- 组件使用 PascalCase，hooks 使用 camelCase
- 优先使用函数组件，不使用 class 组件

## 架构关键路径

### 处理流水线

```
Parser → Topo → Validator → Export
         ↓
   Acoustic / Interact
```

### 核心模块位置

| 功能 | 路径 |
|------|------|
| DXF/PDF 解析 | `crates/parser/src/` |
| 拓扑建模 | `crates/topo/src/` |
| 几何验证 | `crates/validator/src/` |
| 流程编排 | `crates/orchestrator/src/` |
| 声学分析 | `crates/acoustic/src/` |
| HTTP API | `crates/orchestrator/src/api.rs` |
| 光栅图片加载 | `crates/raster-loader/src/` |
| 公共类型 | `crates/common-types/src/` |
| 前端 | `cad-web/src/` |

### P2 规划中（重要上下文）

- **Halfedge 结构** (`crates/topo/src/halfedge.rs`): 已完成，`TopoAlgorithm::Halfedge` 为默认算法
- **rayon 并行化**: `crates/topo/src/parallel.rs` 已有部分并行函数，待扩展到 parser/vectorize
- **wgpu 加速器**: `crates/accelerator-wgpu/` 为 stub（TODO），CPU fallback 已工作
- **微服务拆分**: HTTP/gRPC 部署

## 测试规范

- 所有新公共函数必须有对应的单元测试
- 几何算法优先使用 proptest 进行属性测试
- 修改现有代码时，确保不破坏已有测试
- 新测试文件名格式：`test_<feature>.rs`

## 性能要求

- Topo: 1000 线段 < 150ms (当前 131.9ms)
- Parser: 1000 实体 DXF < 100ms
- Vectorize: 2000x2000 像素 < 1s
- 不允许性能回归超过 10%（CI 警告阈值）
