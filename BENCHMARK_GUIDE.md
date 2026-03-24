# 性能基准测试指南

**版本**: v1.0
**创建日期**: 2026 年 3 月 22 日

---

## 概述

本指南介绍如何运行和解读项目的性能基准测试，帮助开发者发现和解决性能瓶颈。

---

## Rust 基准测试

### 运行基准测试

```bash
# 运行所有基准测试
cargo bench

# 运行特定 crate 的基准测试
cargo bench --package topo
cargo bench --package parser

# 运行特定测试
cargo bench --bench topology_bench
cargo bench --bench parallel_bench
```

### 基准测试位置

| Crate | 路径 | 说明 |
|-------|------|------|
| topo | `crates/topo/benches/` | 拓扑构建性能 |
| parser | `crates/parser/benches/` | DXF 解析性能 |
| orchestrator | `crates/orchestrator/benches/` | 服务编排性能 |

### 关键性能指标 (KPI)

#### TopoService

| 测试 | 目标 | 说明 |
|------|------|------|
| `build_topology_100_segments` | < 20ms | 100 线段拓扑构建 |
| `build_topology_1000_segments` | < 150ms | 1000 线段拓扑构建 |
| `snap_endpoints_parallel` | 3-5x 加速 | 并行端点吸附性能 |
| `halfedge_construction` | < 50ms | Halfedge 构建性能 |

#### ParserService

| 测试 | 目标 | 说明 |
|------|------|------|
| `parse_small_dxf` | < 50ms | 小文件解析 (<100KB) |
| `parse_medium_dxf` | < 200ms | 中等文件解析 (100-500KB) |
| `parse_large_dxf` | < 1s | 大文件解析 (>500KB) |
| `nurbs_discretize` | < 10ms | NURBS 离散化性能 |

### 解读结果

```
build_topology/100_segments
                        time:   [13.390 ms 13.426 ms 13.465 ms]
                        change: [-2.1% -1.5% -0.9%] (p = 0.00 < 0.05)
                        Performance: ✅ 符合目标 (< 20ms)
```

- **time**: 执行时间（越低越好）
- **change**: 与上次测试的变化（负值 = 性能提升）
- **p-value**: 统计显著性（< 0.05 = 可信）

### 性能回归检测

```bash
# 使用 cargo-criterion 进行精确比较
cargo install cargo-criterion
cargo criterion

# 查看历史对比
cargo criterion --baseline main
```

---

## 前端基准测试

### 运行前端性能测试

```bash
cd cad-web

# 单元测试（包含性能断言）
pnpm test

# 性能专项测试（如果有）
pnpm test:perf

# E2E 测试（包含加载时间测量）
pnpm test:e2e
```

### 关键性能指标 (KPI)

| 指标 | 目标 | 说明 |
|------|------|------|
| FPS | >= 60 | 渲染帧率 |
| 首屏加载 | < 2s | 首次渲染时间 |
| 边渲染耗时 | < 16ms | 单帧渲染时间（1000 边） |
| HATCH 渲染耗时 | < 50ms | 单帧渲染时间（100 HATCH） |
| 内存使用 | < 200MB | JavaScript 堆内存 |

### 使用性能监控组件

开发模式下自动显示性能监控面板：

```
📊 性能监控
FPS: 60          (>= 60 绿色，30-60 黄色，< 30 红色)
帧时间：16.7 ms
渲染耗时：8.2 ms
边数量：1250
HATCH 数量：45
内存：156 MB
缩放：100%
```

### Chrome DevTools 性能分析

1. 打开 Chrome DevTools (F12)
2. 切换到 Performance 标签
3. 点击录制按钮
4. 执行操作（缩放、平移、上传文件）
5. 停止录制并分析

**关键指标**:
- Green frames: 60 FPS
- Yellow frames: 30-60 FPS
- Red frames: < 30 FPS

---

## CI 基准测试

### 自动触发

每次 PR 提交自动运行基准测试：

```yaml
# .github/workflows/benchmark.yml
name: Performance Benchmark
on: [push, pull_request]
```

### 性能回归检查

CI 会自动检测性能回归：

- **警告**: 性能下降 5-10%
- **错误**: 性能下降 > 10%

### 查看报告

1. GitHub Actions 页面查看运行结果
2. 下载 `benchmark-results`  artifact
3. PR 自动评论包含性能对比

---

## 性能优化建议

### Rust 优化

1. **启用并行化**
   ```rust
   TopoConfig {
       enable_parallel: true,
       parallel_threshold: 1000,
   }
   ```

2. **减少内存分配**
   ```rust
   // ❌ 避免
   let mut result = Vec::new();
   for item in items {
       result.push(process(item));
   }

   // ✅ 推荐
   let mut result = Vec::with_capacity(items.len());
   for item in items {
       result.push(process(item));
   }
   ```

3. **使用迭代器而非循环**
   ```rust
   // ❌ 避免
   let mut sum = 0.0;
   for point in points {
       sum += point[0];
   }

   // ✅ 推荐
   let sum: f64 = points.iter().map(|p| p[0]).sum();
   ```

### 前端优化

1. **LOD 分层渲染**
   ```typescript
   // zoom < 0.3: 仅渲染墙体
   // zoom 0.3-0.7: 渲染墙体 + 门窗
   // zoom > 0.7: 渲染全部
   ```

2. **批量渲染**
   ```tsx
   // ❌ 避免
   edges.map(edge => <Edge key={edge.id} {...edge} />)

   // ✅ 推荐
   <BatchedEdges edges={edges} />
   ```

3. **缓存优化**
   ```typescript
   // 使用 useMemo 缓存计算结果
   const filteredEdges = useMemo(() => {
     return edges.filter(e => e.visible)
   }, [edges])
   ```

---

## 故障排查

### 性能突然下降

1. 运行基准测试对比历史数据
2. 使用 `git bisect` 定位问题提交
3. 检查是否有新的内存分配或循环

### 前端 FPS 过低

1. 打开性能监控组件查看边数量
2. 检查是否有过多 HATCH 渲染
3. 使用 Chrome DevTools 分析长任务

### 内存泄漏

1. Rust: 使用 `cargo flamegraph` 分析
2. 前端：使用 Chrome Memory 标签拍摄快照对比

---

## 参考资源

- [criterion.rs 文档](https://bheisler.github.io/criterion.rs/book/)
- [Chrome DevTools Performance](https://developer.chrome.com/docs/devtools/performance/)
- [React Performance Optimization](https://react.dev/learn/render-and-commit)

---

**维护者**: P11 Engineering Team
**最后更新**: 2026 年 3 月 22 日
