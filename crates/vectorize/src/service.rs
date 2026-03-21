//! 矢量化服务实现（使用 Accelerator trait）

use std::sync::Arc;
use std::time::Instant;

use accelerator_api::Accelerator;
use common_types::{
    Polyline, CadError,
    InternalErrorReason, PdfRasterImage, ServiceMetrics,
};
use image::DynamicImage;
use log::debug;

use crate::config::{VectorizeConfig, PreprocessingConfig};
use crate::quality::evaluate_image_quality;
use crate::algorithms::{
    threshold, detect_edges, skeletonize, extract_contours,
    preprocessing, threshold as threshold_algo,
    douglas_peucker,
};

#[cfg(feature = "opencv")]
use crate::algorithms::{
    threshold_opencv, detect_edges_opencv, skeletonize_opencv, find_contours_opencv,
    simplify_contours_opencv,
};

/// 矢量化服务（使用 Accelerator trait）
pub struct VectorizeService {
    accelerator: Box<dyn Accelerator>,
    config: VectorizeConfig,
    metrics: Arc<ServiceMetrics>,
}

impl VectorizeService {
    /// 创建新的矢量化服务
    pub fn new(accelerator: Box<dyn Accelerator>, config: VectorizeConfig) -> Self {
        Self {
            accelerator,
            config,
            metrics: Arc::new(ServiceMetrics::new("VectorizeService")),
        }
    }

    /// 使用默认配置和 CPU 加速器创建
    pub fn with_default() -> Self {
        use accelerator_cpu::CpuAccelerator;
        Self::new(
            Box::new(CpuAccelerator::new()),
            VectorizeConfig::default(),
        )
    }

    /// 使用指定加速器创建
    pub fn with_accelerator(accelerator: Box<dyn Accelerator>, config: VectorizeConfig) -> Self {
        Self::new(accelerator, config)
    }

    /// 获取加速器名称
    pub fn accelerator_name(&self) -> &str {
        self.accelerator.name()
    }

    /// 获取服务配置
    pub fn config(&self) -> &VectorizeConfig {
        &self.config
    }

    /// 获取服务指标
    pub fn metrics(&self) -> &ServiceMetrics {
        &self.metrics
    }

    /// 从 DynamicImage 矢量化（使用内部加速器）
    pub fn vectorize_image(&self, img: &DynamicImage) -> Result<Vec<Polyline>, CadError> {
        self.vectorize_image_with_config(img, &self.config)
    }

    /// 从 DynamicImage 矢量化（使用指定配置）
    pub fn vectorize_image_with_config(
        &self,
        img: &DynamicImage,
        config: &VectorizeConfig,
    ) -> Result<Vec<Polyline>, CadError> {
        let start = Instant::now();
        let gray = img.to_luma8();
        let (width, height) = gray.dimensions();

        // 图像尺寸检查
        const MAX_PIXELS: usize = 2_500_000;
        if (width as usize * height as usize) > MAX_PIXELS {
            return Err(CadError::InternalError {
                reason: InternalErrorReason::Panic {
                    message: format!(
                        "图像尺寸过大：{}x{} = {} 像素，最大支持 {} 像素",
                        width, height, width as usize * height as usize, MAX_PIXELS
                    ),
                },
                location: None,
            });
        }

        // 图像质量预检查
        if config.quality_assessment {
            let quality_score = evaluate_image_quality(&gray);
            const MIN_QUALITY_SCORE: f64 = 60.0;
            if quality_score < MIN_QUALITY_SCORE {
                return Err(CadError::VectorizeFailed {
                    message: format!(
                        "图像质量过低：{:.1}/100，最低要求 {:.1} 分",
                        quality_score, MIN_QUALITY_SCORE
                    ),
                });
            }
            debug!("图像质量评分：{:.1}/100", quality_score);
        }

        // 1. 图像预处理
        let preprocessed = self.preprocess(&gray, &config.preprocessing)?;

        // 2. 二值化
        #[cfg(feature = "opencv")]
        let binary = if config.use_opencv && config.adaptive_threshold {
            threshold_opencv(&preprocessed, true).map_err(|e| CadError::InternalError {
                reason: InternalErrorReason::Panic { message: format!("OpenCV 阈值处理失败：{}", e) },
                location: None,
            })?
        } else if config.adaptive_threshold {
            threshold_algo::binary_with_otsu(&preprocessed)
        } else {
            threshold(&preprocessed, config.threshold)
        };

        #[cfg(not(feature = "opencv"))]
        let binary = if config.adaptive_threshold {
            threshold_algo::binary_with_otsu(&preprocessed)
        } else {
            threshold(&preprocessed, config.threshold)
        };

        // 3. 边缘检测
        #[cfg(feature = "opencv")]
        let edges = if config.use_opencv {
            detect_edges_opencv(&binary).unwrap_or_else(|_| detect_edges(&binary))
        } else {
            detect_edges(&binary)
        };

        #[cfg(not(feature = "opencv"))]
        let edges = detect_edges(&binary);

        // 4. 骨架化
        let skeleton = if config.skeletonize {
            #[cfg(feature = "opencv")]
            {
                if config.use_opencv {
                    skeletonize_opencv(&edges).unwrap_or_else(|_| skeletonize(&edges))
                } else {
                    skeletonize(&edges)
                }
            }
            #[cfg(not(feature = "opencv"))]
            {
                skeletonize(&edges)
            }
        } else {
            edges
        };

        // 5. 轮廓提取
        let min_len = config.min_line_length_px as usize;
        #[cfg(feature = "opencv")]
        let contours = if config.use_opencv {
            find_contours_opencv(&skeleton, min_len)
                .unwrap_or_else(|_| extract_contours(&skeleton, min_len))
        } else {
            extract_contours(&skeleton, min_len)
        };

        #[cfg(not(feature = "opencv"))]
        let contours = extract_contours(&skeleton, min_len);

        // 5.5. OpenCV 多边形简化（approxPolyDP）
        #[cfg(feature = "opencv")]
        let contours = if config.use_opencv && config.opencv_approx_epsilon.is_some() {
            let epsilon = config.opencv_approx_epsilon.unwrap_or(2.0);
            match simplify_contours_opencv(&contours, epsilon) {
                Ok(simplified) => {
                    debug!("OpenCV approxPolyDP 简化：{} -> {} 个轮廓", contours.len(), simplified.len());
                    simplified
                }
                Err(e) => {
                    warn!("OpenCV 多边形简化失败：{}, 使用原始轮廓", e);
                    contours
                }
            }
        } else {
            contours
        };

        // 6. 简化和吸附
        let simplified = self.simplify_polylines(&contours, config);

        let duration = start.elapsed();
        debug!(
            "矢量化完成：{} 条轮廓 -> {} 条多段线，耗时 {:?}",
            contours.len(),
            simplified.len(),
            duration
        );

        Ok(simplified)
    }

    /// 从 PdfRasterImage 矢量化
    pub fn vectorize_from_pdf(
        &self,
        raster: &PdfRasterImage,
        config: Option<&VectorizeConfig>,
    ) -> Result<Vec<Polyline>, CadError> {
        let config = config.unwrap_or(&self.config);
        let img = raster.to_image();
        self.vectorize_image_with_config(&img, config)
    }

    /// 图像预处理
    fn preprocess(
        &self,
        image: &image::GrayImage,
        config: &PreprocessingConfig,
    ) -> Result<image::GrayImage, CadError> {
        let mut result = image.clone();

        if config.denoise && config.denoise_method == "median" {
            let strength = (config.denoise_strength as u32).min(3);
            result = preprocessing::median_filter(&result, strength);
        }

        Ok(result)
    }

    /// 简化多段线并吸附端点
    fn simplify_polylines(
        &self,
        polylines: &[Polyline],
        config: &VectorizeConfig,
    ) -> Vec<Polyline> {
        let simplified: Vec<Polyline> = polylines
            .iter()
            .filter_map(|pl| {
                if pl.len() < 2 {
                    return None;
                }
                let simplified = douglas_peucker(pl, config.snap_tolerance_px);
                if simplified.len() >= 2 {
                    Some(simplified)
                } else {
                    None
                }
            })
            .collect();

        self.snap_endpoints_global(&simplified, config.snap_tolerance_px)
    }

    /// 全局端点吸附
    fn snap_endpoints_global(&self, polylines: &[Polyline], tolerance: f64) -> Vec<Polyline> {
        if polylines.is_empty() {
            return polylines.to_vec();
        }

        let tol_sq = tolerance * tolerance;
        let mut result = polylines.to_vec();

        // 简单迭代吸附
        for _ in 0..3 {
            let mut changed = false;
            for i in 0..result.len() {
                if result[i].len() < 2 {
                    continue;
                }
                let first = result[i][0];
                let last = result[i][result[i].len() - 1];
                let len = result[i].len();

                // 检查与其他线段端点的距离
                for j in 0..result.len() {
                    if i == j || result[j].len() < 2 {
                        continue;
                    }
                    let other_first = result[j][0];
                    let other_last = result[j][result[j].len() - 1];

                    // 检查 first 与 other_first
                    let dist_sq = (first[0] - other_first[0]).powi(2) + (first[1] - other_first[1]).powi(2);
                    if dist_sq < tol_sq && dist_sq > 0.0 {
                        result[i][0] = other_first;
                        changed = true;
                    }

                    // 检查 first 与 other_last
                    let dist_sq = (first[0] - other_last[0]).powi(2) + (first[1] - other_last[1]).powi(2);
                    if dist_sq < tol_sq && dist_sq > 0.0 {
                        result[i][0] = other_last;
                        changed = true;
                    }

                    // 检查 last 与 other_first
                    let dist_sq = (last[0] - other_first[0]).powi(2) + (last[1] - other_first[1]).powi(2);
                    if dist_sq < tol_sq && dist_sq > 0.0 {
                        result[i][len - 1] = other_first;
                        changed = true;
                    }

                    // 检查 last 与 other_last
                    let dist_sq = (last[0] - other_last[0]).powi(2) + (last[1] - other_last[1]).powi(2);
                    if dist_sq < tol_sq && dist_sq > 0.0 {
                        result[i][len - 1] = other_last;
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        result
    }
}

impl Default for VectorizeService {
    fn default() -> Self {
        Self::with_default()
    }
}

/// 矢量化请求
#[derive(Debug, Clone)]
pub struct VectorizeRequest {
    pub image_bytes: Vec<u8>,
}

/// 矢量化响应数据
#[derive(Debug, Clone)]
pub struct VectorizeResponseData {
    pub polylines: Vec<Polyline>,
    pub quality_score: f64,
    pub latency_ms: f32,
}
