# 测试文件目录说明

**最后更新**: 2026 年 2 月 28 日
**版本**: v0.1.0

---

## 概述

本项目包含两个测试文件目录，用于验证 DXF 和 PDF 解析功能：

| 目录 | 文件数量 | 用途 |
|------|----------|------|
| `dxfs/` | 9 个 DXF 文件 | DXF 解析测试、拓扑构建测试 |
| `testpdf/` | 4 个 PDF 文件 | PDF 解析测试 |

---

## dxfs/ - DXF 测试文件

### 文件清单

| 文件名 | 实体数 | 用途 | 测试类型 |
|--------|--------|------|----------|
| `报告厅 1.dxf` | 273 实体 | 解析测试、拓扑测试、基准测试 | 真实场景 |
| `报告厅 2.dxf` | 148 实体 | 解析测试、拓扑测试、基准测试 | 真实场景 |
| `报告厅 3.dxf` | - | 解析测试、语义识别测试 | 真实场景 |
| `报告厅 4.dxf` | - | 解析测试、语义识别测试 | 真实场景 |
| `报告厅 5.dxf` | - | 解析测试、语义识别测试 | 真实场景 |
| `会议室 1.dxf` | - | 解析测试、语义识别测试 | 真实场景 |
| `会议室 2.dxf` | - | 解析测试、语义识别测试 | 真实场景 |
| `问题文件 - 端点错位 0.3mm.dxf` | - | 边界情况测试、错误恢复测试 | 问题文件 |
| `问题文件 - 自相交多边形.dxf` | - | 验证服务测试、错误恢复测试 | 问题文件 |

### 测试覆盖

- **解析测试**: 9 个文件全部通过
- **拓扑构建测试**: 报告厅 1-2 用于基准测试
- **语义识别测试**: 会议室 1-2 用于图层识别验证
- **边界情况测试**: 问题文件用于容错能力验证
- **错误恢复测试**: 自相交文件用于恢复建议生成验证

### 性能参考

| 文件 | 实体数 | 解析时间 | 拓扑构建时间 |
|------|--------|----------|--------------|
| 报告厅 1.dxf | 273 | < 50ms | 2.45ms |
| 报告厅 2.dxf | 148 | < 30ms | 1.82ms |

---

## testpdf/ - PDF 测试文件

### 文件清单

| 文件名 | 类型 | 实体数 | 用途 |
|--------|------|--------|------|
| `20x40-house-with-4-bedrooms.pdf` | 矢量 PDF | - | PDF 解析测试 |
| `36x32-house-with-4-bedroom.pdf` | 矢量 PDF | - | PDF 解析测试 |
| `45x40-house-with-3-Bedooms.pdf` | 矢量 PDF | - | PDF 解析测试 |
| `64x60-house-plan-with4-bedrooms.pdf` | 矢量 PDF | 541,216 实体 | PDF 解析测试、性能测试 |

### 测试覆盖

- **矢量 PDF 解析**: 4 个文件全部通过
- **压缩流支持**: FlateDecode/DCTDecode/LZW/RunLength
- **性能测试**: 541,216 实体解析约 1.5s

### 光栅 PDF 矢量化测试

使用内存中的光栅图像进行测试（12 个测试用例）：

| 测试类型 | 数量 | 说明 |
|----------|------|------|
| 基本图形 | 4 | 矩形、水平线、垂直线 |
| 圆弧拟合 | 1 | Kåsa 算法验证 |
| 缺口检测 | 2 | 缺口填充测试 |
| 质量评估 | 3 | 自动评分测试 |
| DPI 自适应 | 2 | 参数动态调整 |

**测试通过率**: 100%（12/12）

### 性能参考

| 像素 | 纯 Rust | OpenCV 加速 |
|------|--------|-------------|
| 500×500 | ~50ms | - |
| 1000×1000 | ~200ms | - |
| 2000×2000 | ~800ms | - |
| 2000×3000 | ~1000ms | ~220ms (4.5x) |

---

## 测试命令

### 运行 DXF 测试

```bash
# 运行所有 DXF 解析测试
cargo test --package parser test_parse_all_real_dxf_files -- --nocapture

# 运行拓扑构建测试
cargo test --package topo benchmark_real -- --nocapture

# 运行边界情况测试
cargo test --package parser test_edge_cases -- --nocapture
```

### 运行 PDF 测试

```bash
# 运行所有 PDF 解析测试
cargo test --package parser test_parse_pdf_files -- --nocapture

# 运行光栅 PDF 矢量化测试
cargo test --package vectorize test_raster_pdf -- --nocapture
```

### 运行 E2E 测试

```bash
# 端到端测试（DXF → JSON）
cargo test --test e2e_tests test_e2e_dxf_to_json -- --nocapture

# 端到端测试（PDF → JSON）
cargo test --test e2e_tests test_e2e_pdf_to_json -- --nocapture
```

---

## 测试文件来源

### DXF 文件
- 报告厅系列：真实建筑图纸（AutoCAD 导出）
- 会议室系列：真实建筑图纸（AutoCAD 导出）
- 问题文件：人工构造的边界情况测试文件

### PDF 文件
- 4 个矢量 PDF：开源房屋平面图（AutoCAD 导出）
- 光栅图像：内存生成（用于矢量化测试）

---

## 添加测试文件

如需添加新的测试文件，请遵循以下规范：

### DXF 文件命名
- 正常文件：`[建筑类型][编号].dxf`
- 问题文件：`问题文件 - [问题描述].dxf`

### PDF 文件命名
- 使用描述性英文名称
- 包含尺寸信息（如 `20x40-house-...`）

### 测试文件要求
1. 文件大小合理（< 10MB）
2. 有明确的测试目的
3. 在测试代码中正确引用
4. 更新本文档

---

## 相关文件

- [README.md](../README.md) - 项目介绍
- [ARCHITECTURE.md](../ARCHITECTURE.md) - 架构文档
- [CONTRIBUTING.md](../CONTRIBUTING.md) - 贡献指南

---

**维护者**: CAD 项目团队
**联系方式**: https://github.com/your-org/cad/issues
