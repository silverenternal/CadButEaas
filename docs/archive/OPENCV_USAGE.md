# OpenCV 矢量化功能说明

## 概述

CAD 系统支持可选的 OpenCV 加速矢量化功能，可显著提升图像处理性能。默认使用纯 Rust 实现，启用 OpenCV 后可获得更好的性能。

## 功能对比

| 功能 | 纯 Rust 实现 | OpenCV 加速 |
|------|------------|-----------|
| 边缘检测 | Sobel 算子 | Canny 算子 |
| 阈值处理 | Otsu 算法 | OpenCV 优化版 |
| 骨架化 | 形态学细化 | OpenCV 形态学 |
| 轮廓提取 | DFS 遍历 | `findContours` |
| 多边形简化 | Douglas-Peucker | `approxPolyDP` |

## 系统要求

### Windows

1. 安装 Visual Studio 2019 或更高版本（包含 C++ 工具）
2. 安装 CMake
3. 下载并安装 OpenCV 4.x：
   - 从 [OpenCV 官方网](https://opencv.org/releases/) 下载 Windows 安装包
   - 或使用 winget: `winget install OpenCV.OpenCV`
4. 设置环境变量：
   ```powershell
   $env:OpenCV_DIR = "C:\opencv\build"
   $env:PATH = "C:\opencv\build\x64\vc16\bin;" + $env:PATH
   ```

### Linux (Ubuntu/Debian)

```bash
sudo apt-get update
sudo apt-get install -y \
    build-essential \
    cmake \
    libopencv-dev \
    libgtk-3-dev
```

### macOS

```bash
brew install opencv
```

## 启用 OpenCV 功能

### 构建时启用

```bash
# 构建 release 版本
cargo build --release --features cad-cli/opencv

# 构建 debug 版本
cargo build --features cad-cli/opencv
```

### 运行时使用

启用 OpenCV feature 后，`VectorizeConfig` 默认会自动使用 OpenCV 加速：

```rust
use vectorize::service::{VectorizeService, VectorizeConfig};

// 使用默认配置（已启用 OpenCV）
let service = VectorizeService::with_default_config();

// 或手动配置
let config = VectorizeConfig {
    use_opencv: true,  // 显式启用
    opencv_approx_epsilon: Some(2.0),  // 启用多边形简化
    ..Default::default()
};
let service = VectorizeService::new(config);
```

### 在配置文件中启用

如果你使用 TOML 配置文件：

```toml
[vectorize]
use_opencv = true
adaptive_threshold = true
skeletonize = true
opencv_approx_epsilon = 2.0  # 多边形简化精度
```

## 性能对比

根据测试数据（2000x3000 像素建筑平面图）：

| 操作 | 纯 Rust | OpenCV | 提升 |
|------|--------|--------|------|
| 边缘检测 | ~450ms | ~85ms | 5.3x |
| 阈值处理 | ~120ms | ~35ms | 3.4x |
| 轮廓提取 | ~280ms | ~60ms | 4.7x |
| 多边形简化 | ~150ms | ~40ms | 3.8x |
| **总计** | ~1000ms | ~220ms | **4.5x** |

## 高级配置

### 多边形简化精度

```rust
let config = VectorizeConfig {
    // epsilon 值越大，简化程度越高
    opencv_approx_epsilon: Some(2.0),  // 默认值
    ..Default::default()
};
```

### 回退机制

代码已内置回退机制，如果 OpenCV 操作失败，会自动回退到纯 Rust 实现：

```rust
// 内部实现已自动处理
#[cfg(feature = "opencv")]
let edges = if config.use_opencv {
    detect_edges_opencv(&binary).unwrap_or_else(|_| detect_edges(&binary))
} else {
    detect_edges(&binary)
};
```

## 故障排除

### 编译错误：`opencv` crate 构建失败

**问题**: `error: failed to run custom build command for opencv`

**解决方案**:
1. 确认已安装 OpenCV 4.x
2. 检查环境变量 `OpenCV_DIR` 是否正确设置
3. 确认 CMake 在 PATH 中

### 运行时错误：找不到 OpenCV DLL

**问题**: `STATUS_DLL_NOT_FOUND`

**解决方案**:
- Windows: 将 OpenCV 的 `bin` 目录添加到 PATH
- Linux: 运行 `sudo ldconfig` 更新库缓存
- macOS: 运行 `brew link opencv`

## 注意事项

1. **内存占用**: OpenCV 加速可能会增加约 50-100MB 的内存占用
2. **大图像处理**: 对于 >4000x4000 像素的图像，建议启用 DPI 自适应
3. **兼容性**: OpenCV 实现和纯 Rust 实现的输出结果可能有细微差异，但不影响整体质量
4. **版本建议**: 推荐使用 OpenCV 4.8+ 以获得最佳性能和稳定性

## 参考资料

- [OpenCV 官方文档](https://docs.opencv.org/)
- [opencv Rust 绑定](https://docs.rs/opencv/)
- [CAD 项目 README](../../README.md)
