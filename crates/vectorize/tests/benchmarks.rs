//! 矢量化性能基准测试
//!
//! 验证矢量化服务在不同图像尺寸下的性能表现
//!
//! # 测试覆盖
//!
//! - 100x100 → 2000x2000 像素性能曲线
//! - 不同图像尺寸的处理时间
//! - 质量评分与性能权衡

use common_types::PdfRasterImage;
use std::time::Instant;
use vectorize::{VectorizeConfig, VectorizeService};

/// 生成测试图像（带噪声的网格线，模拟建筑图纸）
fn generate_test_raster(width: u32, height: u32, noise_ratio: f64) -> PdfRasterImage {
    let mut pixels = Vec::with_capacity((width * height) as usize);

    // 简单伪随机数生成器
    let seed = 42u64;
    let mut rng = seed;

    for y in 0..height {
        for x in 0..width {
            // 绘制网格线（模拟墙体）
            let grid_spacing = 20u32;
            let is_grid = (y % grid_spacing == 0) || (x % grid_spacing == 0);

            if is_grid {
                pixels.push(255u8);
            } else {
                // 添加随机噪声
                rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
                if (rng & 0xFFFFFF) as f64 / 16777216.0 < noise_ratio {
                    pixels.push(255u8);
                } else {
                    pixels.push(0u8);
                }
            }
        }
    }

    PdfRasterImage::new(
        "benchmark".to_string(),
        width,
        height,
        pixels,
        Some((72.0, 72.0)),
        [1.0, 0.0, 0.0, 1.0, 0.0, 0.0],
    )
}

#[test]
#[ignore] // 需要较长时间运行，默认忽略
fn benchmark_vectorize_100x100() {
    let raster = generate_test_raster(100, 100, 0.01);
    let service = VectorizeService::with_default();

    let start = Instant::now();
    let polylines = service.vectorize_from_pdf(&raster, None).unwrap();
    let duration = start.elapsed();

    println!(
        "100x100 矢量化耗时：{:?}，输出 {} 条多段线",
        duration,
        polylines.len()
    );
    assert!(duration.as_millis() < 100, "100x100 图像应 < 100ms");
}

#[test]
#[ignore] // 需要较长时间运行，默认忽略
fn benchmark_vectorize_500x500() {
    let raster = generate_test_raster(500, 500, 0.01);
    let service = VectorizeService::with_default();

    let start = Instant::now();
    let polylines = service.vectorize_from_pdf(&raster, None).unwrap();
    let duration = start.elapsed();

    println!(
        "500x500 矢量化耗时：{:?}，输出 {} 条多段线",
        duration,
        polylines.len()
    );
    assert!(duration.as_millis() < 300, "500x500 图像应 < 300ms");
}

#[test]
#[ignore] // 需要较长时间运行，默认忽略
fn benchmark_vectorize_1000x1000() {
    let raster = generate_test_raster(1000, 1000, 0.01);
    let service = VectorizeService::with_default();

    let start = Instant::now();
    let polylines = service.vectorize_from_pdf(&raster, None).unwrap();
    let duration = start.elapsed();

    println!(
        "1000x1000 矢量化耗时：{:?}，输出 {} 条多段线",
        duration,
        polylines.len()
    );
    assert!(duration.as_millis() < 1000, "1000x1000 图像应 < 1s");
}

#[test]
#[ignore] // 需要较长时间运行，默认忽略
fn benchmark_vectorize_2000x2000() {
    let raster = generate_test_raster(2000, 2000, 0.01);
    let service = VectorizeService::with_default();

    let start = Instant::now();
    let polylines = service.vectorize_from_pdf(&raster, None).unwrap();
    let duration = start.elapsed();

    println!(
        "2000x2000 矢量化耗时：{:?}，输出 {} 条多段线",
        duration,
        polylines.len()
    );
    assert!(duration.as_millis() < 5000, "2000x2000 图像应 < 5s");
}

/// 性能曲线测试：100x100 → 2000x2000
#[test]
#[ignore] // 需要较长时间运行，默认忽略
fn benchmark_vectorize_performance_curve() {
    println!("\n=== 矢量化性能曲线测试 ===\n");
    println!(
        "{:<15} {:<15} {:<15} {:<15}",
        "尺寸", "耗时 (ms)", "输出线段", "像素/秒"
    );
    println!("{:-<65}", "");

    let sizes = [100, 200, 500, 1000, 2000];
    let service = VectorizeService::with_default();

    for &size in &sizes {
        let raster = generate_test_raster(size, size, 0.01);
        let pixel_count = (size * size) as f64;

        let start = Instant::now();
        let polylines = service.vectorize_from_pdf(&raster, None).unwrap();
        let duration = start.elapsed();
        let pixels_per_sec = pixel_count / duration.as_secs_f64();

        println!(
            "{:<15} {:<15.2} {:<15} {:<15.0}",
            format!("{}x{}", size, size),
            duration.as_secs_f64() * 1000.0,
            polylines.len(),
            pixels_per_sec
        );
    }
}

/// 质量评估性能测试
#[test]
#[ignore] // 需要较长时间运行，默认忽略
fn benchmark_quality_assessment() {
    println!("\n=== 质量评估性能测试 ===\n");

    let sizes = [500, 1000, 2000];
    let service = VectorizeService::with_default();

    for &size in &sizes {
        let raster = generate_test_raster(size, size, 0.01);

        let start = Instant::now();
        let polylines = service.vectorize_from_pdf(&raster, None).unwrap();
        let duration = start.elapsed();

        println!(
            "{}x{} 矢量化 + 质量评估：{:?}，输出 {} 条多段线",
            size,
            size,
            duration,
            polylines.len()
        );
    }
}

/// 不同预处理配置的性能对比
#[test]
#[ignore] // 需要较长时间运行，默认忽略
fn benchmark_preprocessing_configs() {
    println!("\n=== 预处理配置性能对比 ===\n");

    let size = 1000;
    let raster = generate_test_raster(size, size, 0.05); // 5% 噪声

    println!("{:<25} {:<15}", "配置", "耗时 (ms)");
    println!("{:-<45}", "");

    // 无预处理
    let config_no_preprocess = VectorizeConfig {
        preprocessing: vectorize::config::PreprocessingConfig {
            denoise: false,
            ..Default::default()
        },
        ..Default::default()
    };
    let service = VectorizeService::new(
        Box::new(accelerator_cpu::CpuAccelerator::new()),
        config_no_preprocess,
    );
    let start = Instant::now();
    let _ = service.vectorize_from_pdf(&raster, None).unwrap();
    println!(
        "{:<25} {:<15.2}",
        "无预处理",
        start.elapsed().as_secs_f64() * 1000.0
    );

    // 中值滤波
    let config_median = VectorizeConfig {
        preprocessing: vectorize::config::PreprocessingConfig {
            denoise: true,
            denoise_method: "median".to_string(),
            denoise_strength: 3.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let service = VectorizeService::new(
        Box::new(accelerator_cpu::CpuAccelerator::new()),
        config_median,
    );
    let start = Instant::now();
    let _ = service.vectorize_from_pdf(&raster, None).unwrap();
    println!(
        "{:<25} {:<15.2}",
        "中值滤波 (3x3)",
        start.elapsed().as_secs_f64() * 1000.0
    );

    // 高斯模糊
    let config_gaussian = VectorizeConfig {
        preprocessing: vectorize::config::PreprocessingConfig {
            denoise: true,
            denoise_method: "gaussian".to_string(),
            denoise_strength: 1.0,
            ..Default::default()
        },
        ..Default::default()
    };
    let service = VectorizeService::new(
        Box::new(accelerator_cpu::CpuAccelerator::new()),
        config_gaussian,
    );
    let start = Instant::now();
    let _ = service.vectorize_from_pdf(&raster, None).unwrap();
    println!(
        "{:<25} {:<15.2}",
        "高斯模糊 (σ=1.0)",
        start.elapsed().as_secs_f64() * 1000.0
    );

    // CLAHE
    let config_clahe = VectorizeConfig {
        preprocessing: vectorize::config::PreprocessingConfig {
            denoise: false,
            enhance_contrast: true,
            clahe_clip_limit: 3.0,
            clahe_tile_size: 8,
            ..Default::default()
        },
        ..Default::default()
    };
    let service = VectorizeService::new(
        Box::new(accelerator_cpu::CpuAccelerator::new()),
        config_clahe,
    );
    let start = Instant::now();
    let _ = service.vectorize_from_pdf(&raster, None).unwrap();
    println!(
        "{:<25} {:<15.2}",
        "CLAHE (8x8)",
        start.elapsed().as_secs_f64() * 1000.0
    );
}
