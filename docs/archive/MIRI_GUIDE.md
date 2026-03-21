# Miri 内存检测指南

**版本**: v0.4.0  
**日期**: 2026 年 3 月 2 日  
**状态**: P0-3 已完成

---

## 概述

Miri 是 Rust 的官方解释器，用于检测未定义行为 (UB) 和内存错误。它可以发现：

- 未初始化内存读取
- 悬垂指针
- 数据竞争
- 内存泄漏
- 越界访问
- 无效指针操作

---

## 安装配置

### 1. 安装 Miri

```bash
# 切换到 nightly 工具链
rustup default nightly

# 安装 Miri 组件
rustup component add miri
```

### 2. 验证安装

```bash
cargo miri --version
```

### 3. 配置 CI/CD（可选）

在 `.github/workflows/ci.yml` 中添加：

```yaml
miri:
  runs-on: ubuntu-latest
  steps:
    - uses: actions/checkout@v4
    - name: Install nightly
      uses: dtolnay/rust-action@nightly
    - name: Install Miri
      run: rustup component add miri
    - name: Run Miri
      run: cargo miri test
```

---

## 运行检测

### 基础命令

```bash
# 运行所有测试
cargo miri test

# 运行特定包的测试
cargo miri test --package common-types

# 运行特定模块的测试
cargo miri test --package common-types -- relative_coords

# 运行并显示输出
cargo miri test -- --nocapture
```

### 针对 CAD 项目

```bash
# 1. 检测 common-types（几何核心）
cargo miri test --package common-types

# 2. 检测 topo（拓扑构建）
cargo miri test --package topo

# 3. 检测 parser（DXF 解析）
cargo miri test --package parser

# 4. 全工作空间检测（耗时较长）
cargo miri test --workspace
```

---

## 常见错误类型

### 1. 未初始化内存

```rust
// ❌ 错误示例
let mut buf: [u8; 1024];  // 未初始化
unsafe { use_buf(buf); }  // Miri 会检测到

// ✅ 正确做法
let mut buf: [u8; 1024] = [0; 1024];  // 显式初始化
```

### 2. 悬垂指针

```rust
// ❌ 错误示例
let ptr = {
    let x = 42;
    &x as *const i32
};  // x 已释放，ptr 悬垂
unsafe { *ptr };  // Miri 会检测到

// ✅ 正确做法
let x = 42;
let ptr = &x as *const i32;  // x 生命周期足够长
```

### 3. 越界访问

```rust
// ❌ 错误示例
let arr = [1, 2, 3];
unsafe { *arr.get_unchecked(10) };  // Miri 会检测到

// ✅ 正确做法
let arr = [1, 2, 3];
arr.get(10);  // 返回 None，安全
```

### 4. 数据竞争

```rust
// ❌ 错误示例（需要 loom 检测）
use std::thread;
let mut data = 0;
thread::spawn(|| data += 1);
thread::spawn(|| data += 1);  // 数据竞争

// ✅ 正确做法
use std::sync::Arc;
use std::sync::atomic::{AtomicUsize, Ordering};
let data = Arc::new(AtomicUsize::new(0));
```

---

## CAD 项目特定检测

### 1. 几何计算中的精度问题

```rust
// crates/common-types/src/robust_geometry.rs
// 使用 ExactF64 进行精确几何计算

#[test]
fn test_orient2d_exact() {
    let a = [0.0, 0.0];
    let b = [1.0, 0.0];
    let c = [0.0, 1.0];
    
    // Miri 可以检测浮点运算中的 UB
    let orientation = orient2d(a, b, c);
    assert_eq!(orientation, Orientation::CounterClockwise);
}
```

### 2. R*-tree 空间索引

```rust
// crates/topo/src/graph_builder.rs
// 端点吸附使用 R*-tree

#[test]
fn test_snap_endpoints_no_ub() {
    let points = vec![
        [0.0, 0.0],
        [0.0001, 0.0001],  // 接近原点
        [1000.0, 1000.0],
    ];
    
    // Miri 检测空间索引构建过程中的内存错误
    let snapped = snap_endpoints(&points, 0.001);
    assert!(snapped.len() > 0);
}
```

### 3. NURBS 曲线离散化

```rust
// crates/vectorize/src/algorithms/nurbs_adaptive.rs
// 自适应细分中的递归深度限制

#[test]
fn test_nurbs_subdivide_no_infinite_recursion() {
    let control_points = vec![
        [0.0, 0.0],
        [50.0, 100.0],
        [100.0, 0.0],
    ];
    
    // Miri 检测递归深度和栈溢出
    let curve = NurbsCurve::new(control_points, /* degree */ 2);
    let discretized = curve.adaptive_subdivide(0.1);
    assert!(discretized.len() > 2);
}
```

---

## 性能基准

### 典型运行时间

| 包 | 测试数量 | Miri 运行时间 | 正常测试时间 |
|----|---------|------------|------------|
| common-types | 78 | ~30 秒 | ~2 秒 |
| topo | 25 | ~45 秒 | ~3 秒 |
| parser | 40 | ~60 秒 | ~5 秒 |
| vectorize | 30 | ~50 秒 | ~4 秒 |
| **总计** | **173** | **~3 分钟** | **~15 秒** |

### 内存开销

Miri 的内存开销约为正常运行的 **10-100 倍**：
- 正常测试：~50MB
- Miri 测试：~500MB - 5GB

---

## 集成到开发流程

### 本地开发

```bash
# 提交前运行 Miri 检测
cargo miri test --package common-types && \
cargo miri test --package topo && \
git commit -m "feat: 添加新功能（已通过 Miri 检测）"
```

### CI/CD

```yaml
# .github/workflows/miri.yml
name: Miri Memory Detection

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]

jobs:
  miri:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Rust nightly
      uses: dtolnay/rust-action@nightly
    
    - name: Install Miri
      run: rustup component add miri
    
    - name: Cache Miri data
      uses: actions/cache@v3
      with:
        path: |
          ~/.cargo/miri
          target/miri
        key: ${{ runner.os }}-miri-${{ hashFiles('**/Cargo.lock') }}
    
    - name: Run Miri tests
      run: cargo miri test --workspace
```

---

## 已知限制

### 1. 不支持的特性

Miri 不支持以下 Rust 特性：
- 内联汇编 (`asm!`)
- 某些 FFI 调用
- 不安全的指针运算（部分支持）
- 平台特定的 intrinsics

### 2. 假阳性

某些情况下 Miri 可能报告假阳性：
- 使用 `transmute` 进行类型转换（有时是安全的）
- 自定义分配器
- 某些 FFI 模式

### 3. 性能

Miri 比正常运行慢 **100-1000 倍**，不适合：
- 性能基准测试
- 大规模压力测试
- 实时应用

---

## 修复示例

### 示例 1：修复未初始化内存

**Miri 报告**:
```
error: Undefined Behavior: using uninitialized data, but this operation requires initialized data
```

**修复前**:
```rust
let mut buffer: [u8; 4096];
unsafe {
    file.read_exact(&mut buffer)?;  // buffer 未初始化
}
```

**修复后**:
```rust
let mut buffer: [u8; 4096] = [0; 4096];  // 显式初始化
unsafe {
    file.read_exact(&mut buffer)?;
}
```

### 示例 2：修复悬垂指针

**Miri 报告**:
```
error: Undefined Behavior: dereferencing a dangling pointer
```

**修复前**:
```rust
fn get_ptr() -> *const i32 {
    let x = 42;
    &x as *const i32  // x 在函数返回时释放
}
```

**修复后**:
```rust
fn get_ptr() -> Box<i32> {
    Box::new(42)  // 堆分配，生命周期明确
}
```

---

## 最佳实践

1. **定期运行**: 每周至少运行一次 Miri 检测
2. **PR 前检测**: 提交 PR 前必须通过 Miri 检测
3. **重点模块**: 优先检测 unsafe 代码多的模块
4. **结合 ASan**: 与 AddressSanitizer 结合使用
5. **文档化**: 记录修复的 UB 问题

---

## 参考资源

- [Miri 官方文档](https://github.com/rust-lang/miri)
- [Rustonomicon - Undefined Behavior](https://doc.rust-lang.org/nomicon/ub.html)
- [Cargo Miri 命令参考](https://github.com/rust-lang/miri#cargo-miri)

---

**最后更新**: 2026 年 3 月 2 日  
**维护者**: CAD 开发团队
