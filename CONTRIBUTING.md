# 贡献指南

欢迎为 CAD 几何智能处理系统做出贡献！

## 🚀 快速开始

### 环境要求

- Rust 1.75+ (stable)
- Windows 10/11 或 Linux/macOS
- 至少 4GB RAM（推荐 8GB）

### 安装依赖

```bash
# 克隆仓库
git clone https://github.com/your-org/cad.git
cd cad

# 安装 Rust 工具链
rustup install stable
rustup default stable
```

## 📋 开发流程

### 1. 代码检查

在提交前，请确保通过所有检查：

```bash
# 代码格式
cargo fmt --all

# Clippy 检查（0 警告）
cargo clippy --workspace --all-targets

# 构建
cargo build --workspace --all-targets

# 测试（220+ 测试全部通过）
cargo test --workspace --all-targets
```

### 2. 创建分支

```bash
# 功能开发
git checkout -b feature/your-feature-name

# Bug 修复
git checkout -b fix/issue-123

# 性能优化
git checkout -b perf/optimize-graph-builder
```

### 3. 提交规范

遵循 [Conventional Commits](https://www.conventionalcommits.org/) 规范：

```
feat: 添加新的 DXF 实体类型支持
fix: 修复端点吸附精度问题
perf: 优化 R*-tree 查询性能
docs: 更新 API 文档
test: 添加集成测试
refactor: 重构错误处理模块
```

### 4. 创建 Pull Request

1. Fork 仓库
2. 推送到你的分支
3. 创建 Pull Request 到 `main` 分支
4. 等待 CI 检查通过
5. 代码审查通过后合并

## 🧪 测试指南

### 单元测试

```bash
# 运行所有单元测试
cargo test --workspace

# 运行特定模块测试
cargo test -p topo

# 运行特定测试
cargo test test_graph_builder_basic
```

### 集成测试

```bash
# 运行所有集成测试
cargo test --test integration_tests

# 运行 E2E 测试
cargo test --test e2e_tests

# 运行用户故事测试
cargo test --test user_story_tests
```

### 性能基准测试

```bash
# 运行所有基准测试
cargo test --test benchmarks -- --nocapture

# 运行性能回归测试
cargo test --test perf_regression
```

## 🔒 安全指南

### 依赖审计

```bash
# 运行安全审计
cargo audit

# 检查过时依赖
cargo outdated
```

### 报告安全漏洞

发现安全漏洞请通过 GitHub Security Advisory 报告，不要公开披露。

## 📝 文档指南

### API 文档

```bash
# 生成本地文档
cargo doc --workspace --open
```

### 代码注释

- 公共 API 必须有文档注释
- 复杂算法需要说明实现思路
- 性能关键代码需要标注时间复杂度

## 🎯 代码风格

### Rust 风格

遵循 [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/)

### 命名约定

```rust
// 类型使用 PascalCase
pub struct GraphBuilder;
pub enum CadError;

// 函数使用 snake_case
pub fn build_scene(&self) -> Result<SceneState>;

// 常量使用 SCREAMING_SNAKE_CASE
pub const API_VERSION: &str = "v1";
```

### 错误处理

```rust
// 使用 Result 而非 panic
pub fn parse_file(path: &Path) -> Result<ParseResult, CadError>

// 提供有意义的错误信息
Err(CadError::DxfParseError {
    message: "无法解析 ENTITIES 段".to_string(),
    source: Some(Box::new(e)),
    file: Some(path.to_path_buf()),
})
```

## 🏗️ 架构原则

### EaaS 设计

每个服务遵循「一切皆服务」原则：
- 明确的输入输出契约
- 独立的测试套件
- 清晰的错误类型
- 文档注释完整

### 服务调用链

```
ParserService → TopoService → ValidatorService → ExportService
```

当前为单体部署（进程内调用），P2 阶段支持 HTTP/gRPC 微服务部署。

## 🤝 社区准则

- 保持友好和包容
- 对事不对人
- 接受建设性批评
- 帮助新贡献者

## 📧 联系方式

- GitHub Issues: 报告 Bug 和提出新功能
- GitHub Discussions: 一般讨论和问题解答

---

感谢你的贡献！🎉
