# 模糊测试 (Fuzz Testing) 指南

**版本**: v0.4.0  
**日期**: 2026 年 3 月 2 日  
**状态**: P0-4 已完成

---

## 概述

模糊测试是一种自动化软件测试技术，通过向程序提供随机、畸形或意外的输入来发现漏洞和崩溃。

### 模糊测试能发现什么

- **崩溃 (Crashes)**: 导致程序终止的输入
- **内存错误**: 缓冲区溢出、越界访问
- **未定义行为**: Rust 的 UB 问题
- **无限循环**: 导致程序挂起的输入
- **断言失败**: 触发 `assert!` 或 `panic!` 的输入

### CAD 项目模糊测试目标

| 目标 | 测试内容 | 预期发现 |
|------|---------|---------|
| `parse_dxf` | DXF 文件解析 | 解析崩溃、内存泄漏 |
| `build_topology` | 拓扑构建算法 | 死循环、栈溢出 |
| `nurbs_discretize` | NURBS 离散化 | 数值不稳定、精度问题 |

---

## 快速开始

### 1. 安装 cargo-fuzz

```bash
cargo install cargo-fuzz
```

### 2. 验证安装

```bash
cd fuzz
cargo fuzz --version
```

### 3. 运行第一个模糊测试

```bash
# 运行 DXF 解析模糊测试
cargo fuzz run parse_dxf

# 按 Ctrl+C 停止
```

---

## 详细使用

### 运行模糊测试

```bash
# 基础运行
cargo fuzz run parse_dxf

# 使用多个 CPU 核心并行运行（推荐）
cargo fuzz run parse_dxf -j 8

# 运行指定时间（秒）
cargo fuzz run parse_dxf -- -max_total_time=3600

# 运行直到找到崩溃
cargo fuzz run parse_dxf -- -max_total_time=1800 -detect_leaks=0

# 使用特定的语料库种子
cargo fuzz run parse_dxf fuzz/corpus/parse_dxf
```

### 检查崩溃

当模糊测试找到崩溃时，会保存到一个文件：

```bash
# 列出所有崩溃
ls fuzz/artifacts/parse_dxf/

# 重现崩溃
cargo fuzz run parse_dxf fuzz/artifacts/parse_dxf/crash-xxxxx

# 调试崩溃
cat fuzz/artifacts/parse_dxf/crash-xxxxx | hexdump -C
```

### 最小化测试用例

```bash
# 最小化崩溃文件
cargo fuzz tmin parse_dxf fuzz/artifacts/parse_dxf/crash-xxxxx

# 最小化后会生成 crash-xxxxx.min 文件
```

---

## 语料库管理

### 添加种子语料

创建有意义的测试用例作为种子：

```bash
# 创建种子目录
mkdir -p fuzz/corpus/parse_dxf

# 添加真实的 DXF 文件作为种子
cp ../test_files/*.dxf fuzz/corpus/parse_dxf/

# 添加最小测试用例
echo "0\nSECTION\n2\nENTITIES\n0\nENDSEC\n0\nEOF" > fuzz/corpus/parse_dxf/minimal.dxf
```

### 语料库优化

```bash
# 合并多个语料库
cargo fuzz cmin parse_dxf fuzz/corpus/parse_dxf

# 这会移除冗余的测试用例，保留最小集合
```

---

## CAD 项目特定目标

### 1. parse_dxf - DXF 解析器模糊测试

**测试内容**:
- DXF 文件头解析
- 实体段解析
- 块定义解析
- 图层表解析

**运行命令**:
```bash
cargo fuzz run parse_dxf -j 8 -- -max_total_time=7200
```

**预期发现**:
- 畸形 DXF 代码导致的 panic
- 内存泄漏
- 无限循环

### 2. build_topology - 拓扑构建模糊测试

**测试内容**:
- 端点吸附算法
- 交点检测
- 闭合环提取
- 孔洞判定

**运行命令**:
```bash
cargo fuzz run build_topology -j 8 -- -max_total_time=7200
```

**预期发现**:
- R*-tree 构建失败
- 交点检测死循环
- 栈溢出（递归过深）

### 3. nurbs_discretize - NURBS 离散化模糊测试

**测试内容**:
- 节点插入算法
- 曲线细分
- 连续性分析

**运行命令**:
```bash
cargo fuzz run nurbs_discretize -j 8 -- -max_total_time=7200
```

**预期发现**:
- 数值不稳定
- 除零错误
- 无限细分

---

## 集成到 CI/CD

### GitHub Actions 配置

```yaml
# .github/workflows/fuzz.yml
name: Fuzz Testing

on:
  push:
    branches: [ main, develop ]
  pull_request:
    branches: [ main ]
  schedule:
    # 每天运行一次
    - cron: '0 2 * * *'

jobs:
  fuzz:
    runs-on: ubuntu-latest
    strategy:
      matrix:
        target: [parse_dxf, build_topology, nurbs_discretize]
    
    steps:
    - uses: actions/checkout@v4
    
    - name: Install Rust
      uses: dtolnay/rust-action@stable
    
    - name: Install cargo-fuzz
      run: cargo install cargo-fuzz
    
    - name: Cache fuzz artifacts
      uses: actions/cache@v3
      with:
        path: |
          fuzz/target
          fuzz/corpus
        key: ${{ runner.os }}-fuzz-${{ hashFiles('fuzz/Cargo.lock') }}
    
    - name: Run fuzz test
      run: |
        cd fuzz
        cargo fuzz run ${{ matrix.target }} -- -max_total_time=600
    
    - name: Check for crashes
      run: |
        if [ -d "fuzz/artifacts/${{ matrix.target }}" ]; then
          echo "::error::Found crashes in ${{ matrix.target }}"
          ls fuzz/artifacts/${{ matrix.target }}/
          exit 1
        fi
```

---

## 修复模糊测试发现的问题

### 步骤 1：重现问题

```bash
# 使用崩溃文件重现
cargo fuzz run parse_dxf fuzz/artifacts/parse_dxf/crash-12345
```

### 步骤 2：调试

```bash
# 使用 gdb/lldb 调试
gdb --args target/fuzz/debug/build/parse_dxf crash-12345

# 或使用 cargo-fuzz 内置调试
cargo fuzz run parse_dxf fuzz/artifacts/parse_dxf/crash-12345 --dev
```

### 步骤 3：修复并验证

```rust
// 修复前
pub fn parse_dxf(content: &str) -> Result<ParseResult> {
    // 可能 panic 的代码
    let entity = entities[index];  // 越界访问
    ...
}

// 修复后
pub fn parse_dxf(content: &str) -> Result<ParseResult> {
    if index >= entities.len() {
        return Err(CadError::ParseError("Entity index out of bounds"));
    }
    let entity = entities[index];
    ...
}
```

### 步骤 4：回归测试

```bash
# 将崩溃文件添加到回归测试
cp fuzz/artifacts/parse_dxf/crash-12345 fuzz/regression_tests/

# 运行回归测试
cargo fuzz run parse_dxf fuzz/regression_tests/
```

---

## 最佳实践

### 1. 持续运行

- **开发阶段**: 每次提交前运行 5-10 分钟
- **CI 阶段**: 每次 PR 运行 10-30 分钟
- **夜间任务**: 每天运行 2-4 小时

### 2. 语料库质量

- 使用真实文件作为种子
- 定期优化语料库（`cargo fuzz cmin`）
- 分享有趣的测试用例

### 3. 多目标并行

```bash
# 同时运行多个模糊测试目标
cargo fuzz run parse_dxf -j 4 &
cargo fuzz run build_topology -j 4 &
cargo fuzz run nurbs_discretize -j 4 &
wait
```

### 4. 监控进度

```bash
# 查看覆盖率
cargo fuzz coverage parse_dxf

# 生成覆盖率报告
cargo fuzz coverage parse_dxf --output-dir=coverage-report
```

---

## 性能优化

### 1. 启用释放模式

```bash
# 使用释放模式构建（更快）
cargo fuzz run parse_dxf --release

# 或使用 lto（更慢但覆盖率更高）
cargo fuzz run parse_dxf --release -- -lto
```

### 2. 调整模糊器参数

```bash
# 增加单次运行的最大长度
cargo fuzz run parse_dxf -- -max_len=4096

# 禁用泄漏检测（更快）
cargo fuzz run parse_dxf -- -detect_leaks=0

# 使用更快的随机数生成器
cargo fuzz run parse_dxf -- -use_traces=0
```

---

## 已知限制

### 1. 不支持的特性

- 异步代码（需要特殊处理）
- FFI 调用（可能需要 mock）
- 文件系统操作（需要 mock）

### 2. 假阳性

某些崩溃可能是预期的错误处理：

```rust
// 这是预期的错误处理，不是 bug
if invalid_input {
    return Err(ParseError::InvalidFormat);
}
```

### 3. 资源限制

- 内存限制：默认 2GB
- 时间限制：默认无限制
- 可以手动调整

---

## 参考资源

- [cargo-fuzz 官方文档](https://github.com/rust-fuzz/cargo-fuzz)
- [libFuzzer 文档](https://llvm.org/docs/LibFuzzer.html)
- [Rust Fuzz 项目](https://github.com/rust-fuzz)

---

**最后更新**: 2026 年 3 月 2 日  
**维护者**: CAD 开发团队
