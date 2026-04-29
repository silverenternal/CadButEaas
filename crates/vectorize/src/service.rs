//! 矢量化服务实现（使用 Accelerator trait）

use std::sync::Arc;
use std::time::Instant;

use accelerator_api::Accelerator;
use common_types::{CadError, PdfRasterImage, Polyline, ServiceMetrics};
use image::{DynamicImage, GenericImageView, GrayImage};
use log::{debug, warn};

use crate::algorithms::{
    adaptive_params, architectural_rules, detect_edges, detect_line_types_from_polylines,
    douglas_peucker, extract_contours, fill_gaps, hough_assisted_gap_filling, paper_detection,
    perspective_correction, preprocessing, skeletonize, threshold, threshold as threshold_algo,
    FitData, OcrBackend, SkeletonConfig,
};
use crate::config::{PreprocessingConfig, RasterStrategy, VectorizeConfig};
use crate::line_type::LineType;
use crate::pipeline::{
    strategy_for_kind, PrimitiveCandidate, RasterVectorizationOutput, SemanticCandidate,
    SymbolCandidate, TextCandidate, VectorizationAttemptReport, VectorizePipelineBuilder,
};
use crate::quality::evaluate_image_quality;
use accelerator_api::{AcceleratorOp, EdgeMap, Image as AcceleratorImage};

#[cfg(feature = "opencv")]
use crate::algorithms::{
    detect_edges_opencv, find_contours_opencv, simplify_contours_opencv, skeletonize_opencv,
    threshold_opencv,
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
        Self::new(Box::new(CpuAccelerator::new()), VectorizeConfig::default())
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

    /// 从 DynamicImage 矢量化并返回结构化报告。
    pub fn vectorize_image_detailed(
        &self,
        img: &DynamicImage,
        config: &VectorizeConfig,
        debug_artifacts: bool,
    ) -> Result<RasterVectorizationOutput, CadError> {
        let gray = img.to_luma8();
        let quality_score = evaluate_image_quality(&gray);
        let (detected_kind, _) = crate::pipeline::detect_raster_kind(&gray, quality_score);

        let mut configs = Vec::new();
        if config.raster_strategy == RasterStrategy::Auto {
            let detected_strategy = strategy_for_kind(detected_kind);
            let mut detected_config = VectorizeConfig::preset(detected_strategy);
            detected_config.max_pixels = config.max_pixels;
            detected_config.max_retries = config.max_retries;
            configs.push(detected_config);
        } else {
            configs.push(config.clone());
        }

        for fallback in [
            RasterStrategy::CleanLineArt,
            RasterStrategy::ScannedPlan,
            RasterStrategy::LowContrast,
            RasterStrategy::PhotoPerspective,
            RasterStrategy::HandSketch,
        ] {
            if configs
                .iter()
                .all(|existing| existing.raster_strategy != fallback)
            {
                let mut fallback_config = VectorizeConfig::preset(fallback);
                fallback_config.max_pixels = config.max_pixels;
                fallback_config.max_retries = config.max_retries;
                configs.push(fallback_config);
            }
        }

        let max_attempts = config.max_retries.max(1).min(configs.len());
        configs.truncate(max_attempts);

        let mut attempts = Vec::new();
        let mut best: Option<(RasterVectorizationOutput, f64)> = None;

        for (idx, attempt_config) in configs.iter().enumerate() {
            let pipeline = VectorizePipelineBuilder::from_config(attempt_config).build();
            match pipeline.process_detailed(img, debug_artifacts && idx == 0) {
                Ok(mut output) => {
                    let score = attempt_score(
                        output.report.quality_score,
                        output.report.final_polyline_count,
                        output.report.contour_count,
                    );
                    attempts.push(VectorizationAttemptReport {
                        attempt_index: idx + 1,
                        strategy: attempt_config.raster_strategy,
                        threshold: attempt_config.threshold,
                        snap_tolerance_px: attempt_config.snap_tolerance_px,
                        min_line_length_px: attempt_config.min_line_length_px,
                        quality_score: output.report.quality_score,
                        polyline_count: output.report.final_polyline_count,
                        score,
                        error_code: None,
                        message: None,
                    });
                    output.report.selected_strategy = attempt_config.raster_strategy;

                    if best
                        .as_ref()
                        .is_none_or(|(_, best_score)| score > *best_score)
                    {
                        best = Some((output, score));
                    }
                }
                Err(err) => {
                    attempts.push(VectorizationAttemptReport {
                        attempt_index: idx + 1,
                        strategy: attempt_config.raster_strategy,
                        threshold: attempt_config.threshold,
                        snap_tolerance_px: attempt_config.snap_tolerance_px,
                        min_line_length_px: attempt_config.min_line_length_px,
                        quality_score,
                        polyline_count: 0,
                        score: 0.0,
                        error_code: Some("pipeline_failed".to_string()),
                        message: Some(err.to_string()),
                    });
                }
            }
        }

        let (mut output, _) = best.ok_or_else(|| CadError::VectorizeFailed {
            message: "所有光栅矢量化策略均失败".to_string(),
        })?;
        output.report.attempts = attempts;
        output.report.primitive_candidates = primitive_candidates(&output.polylines);
        output.report.text_candidates = text_candidates(&gray);
        output.report.symbol_candidates = symbol_candidates(&gray);
        output.report.semantic_candidates = semantic_candidates(&output.polylines);
        Ok(output)
    }

    /// 从 DynamicImage 矢量化（使用指定配置）
    pub fn vectorize_image_with_config(
        &self,
        img: &DynamicImage,
        config: &VectorizeConfig,
    ) -> Result<Vec<Polyline>, CadError> {
        let start = Instant::now();

        // 根据配置调整图像尺寸
        let max_pixels = config.max_pixels;
        let mut gray = if (img.width() as usize * img.height() as usize) > max_pixels {
            let (w, h) = img.dimensions();
            let total = (w as usize) * (h as usize);
            let scale = (max_pixels as f64 / total as f64).sqrt();
            let new_w = (w as f64 * scale).round() as u32;
            let new_h = (h as f64 * scale).round() as u32;
            debug!(
                "图像过大 ({}x{}={} 像素)，缩放到 {}x{}={} 像素（比例 {:.2}%）",
                w,
                h,
                total,
                new_w,
                new_h,
                new_w as usize * new_h as usize,
                scale * 100.0
            );
            img.resize(new_w, new_h, image::imageops::FilterType::Lanczos3)
                .to_luma8()
        } else {
            img.to_luma8()
        };

        // [新增] 自动纸张检测与裁剪
        if config.auto_crop_paper {
            let cropped = paper_detection::detect_and_crop(&gray);
            if cropped.width() < gray.width() || cropped.height() < gray.height() {
                debug!(
                    "自动纸张裁剪: {}x{} -> {}x{}",
                    gray.width(),
                    gray.height(),
                    cropped.width(),
                    cropped.height()
                );
                gray = cropped;
            }
        }

        // [新增] 透视校正
        if config.perspective_correction {
            if let Some(region) = paper_detection::detect_paper(&gray) {
                if region.confidence > 0.5 {
                    let corrected = perspective_correction::correct_perspective(&gray, &region);
                    debug!(
                        "透视校正完成: {}x{} -> {}x{}",
                        gray.width(),
                        gray.height(),
                        corrected.width(),
                        corrected.height()
                    );
                    gray = corrected;
                }
            }
        }

        let (width, height) = gray.dimensions();

        // 图像质量评估 + [新增] 自适应参数调整
        let adapted_config = if config.quality_assessment && config.adaptive_params {
            let quality_score = evaluate_image_quality(&gray);
            debug!("图像质量评分：{:.1}/100", quality_score);
            // 自适应参数调整
            adaptive_params::adapt_parameters(quality_score, width, height, config)
        } else {
            config.clone()
        };

        // 质量预检查（使用调整后的配置判断是否继续）
        if adapted_config.quality_assessment {
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
        }

        // 1. 图像预处理（使用自适应调整后的参数）
        let preprocessed = self.preprocess(&gray, &adapted_config.preprocessing)?;

        // 2. 二值化
        #[cfg(feature = "opencv")]
        let binary = if adapted_config.use_opencv && adapted_config.adaptive_threshold {
            threshold_opencv(&preprocessed, true).map_err(|e| CadError::InternalError {
                reason: InternalErrorReason::Panic {
                    message: format!("OpenCV 阈值处理失败：{}", e),
                },
                location: None,
            })?
        } else if adapted_config.adaptive_threshold {
            threshold_algo::binary_with_otsu(&preprocessed)
        } else {
            threshold(&preprocessed, adapted_config.threshold)
        };

        #[cfg(not(feature = "opencv"))]
        let binary = if adapted_config.adaptive_threshold {
            threshold_algo::binary_with_otsu(&preprocessed)
        } else {
            threshold(&preprocessed, adapted_config.threshold)
        };

        // 2.5. 文字 blob 过滤（如果启用）
        let binary = if adapted_config.text_separation {
            // 降低最小面积以检测更小字符，使用聚类合并邻近字符
            let blobs = crate::algorithms::detect_text_blobs(&binary, 5, 5000);
            if !blobs.is_empty() {
                debug!("文字标注分离: 检测到 {} 个文字块", blobs.len());
            }
            crate::algorithms::erase_text_blobs(&binary, &blobs)
        } else {
            binary
        };

        // 3. 边缘检测
        // 优先级: GPU 加速器 (如果可用并启用) → OpenCV → CPU 纯 Rust
        let edges = if adapted_config.use_accelerator_edge_detect
            && self.accelerator.supports_op(AcceleratorOp::EdgeDetect)
        {
            // 使用加速器（GPU）边缘检测
            let accel_image = AcceleratorImage {
                width: binary.width(),
                height: binary.height(),
                data: binary.as_ref().to_vec(),
            };
            let edge_config = adapted_config.to_edge_detect_config();

            match pollster::block_on(self.accelerator.edge_detect(&accel_image, &edge_config)) {
                Ok(edge_map) => {
                    debug!("wgpu 边缘检测完成：{}x{}", edge_map.width, edge_map.height);
                    edge_map_to_gray_image(&edge_map)
                }
                Err(e) => {
                    warn!("GPU 边缘检测失败，回退到 CPU: {}", e);
                    #[cfg(feature = "opencv")]
                    {
                        if adapted_config.use_opencv {
                            detect_edges_opencv(&binary).unwrap_or_else(|_| detect_edges(&binary))
                        } else {
                            detect_edges(&binary)
                        }
                    }
                    #[cfg(not(feature = "opencv"))]
                    {
                        detect_edges(&binary)
                    }
                }
            }
        } else if cfg!(feature = "opencv") && adapted_config.use_opencv {
            #[cfg(feature = "opencv")]
            {
                detect_edges_opencv(&binary).unwrap_or_else(|_| detect_edges(&binary))
            }
            #[cfg(not(feature = "opencv"))]
            {
                detect_edges(&binary)
            }
        } else {
            detect_edges(&binary)
        };

        // 4. 骨架化
        let skeleton = if adapted_config.skeletonize {
            #[cfg(feature = "opencv")]
            {
                if adapted_config.use_opencv {
                    skeletonize_opencv(&edges).unwrap_or_else(|_| skeletonize(&edges))
                } else {
                    let config = SkeletonConfig::default();
                    skeletonize(&edges, &config)
                }
            }
            #[cfg(not(feature = "opencv"))]
            {
                let config = SkeletonConfig::default();
                skeletonize(&edges, &config)
            }
        } else {
            edges
        };

        // 5. 轮廓提取
        let min_len = adapted_config.min_line_length_px as usize;
        #[cfg(feature = "opencv")]
        let contours = if adapted_config.use_opencv {
            find_contours_opencv(&skeleton, min_len)
                .unwrap_or_else(|_| extract_contours(&skeleton, min_len))
        } else {
            extract_contours(&skeleton, min_len)
        };

        #[cfg(not(feature = "opencv"))]
        let contours = extract_contours(&skeleton, min_len);

        // 5.5. OpenCV 多边形简化（approxPolyDP）
        #[cfg(feature = "opencv")]
        let contours =
            if adapted_config.use_opencv && adapted_config.opencv_approx_epsilon.is_some() {
                let epsilon = adapted_config.opencv_approx_epsilon.unwrap_or(2.0);
                match simplify_contours_opencv(&contours, epsilon) {
                    Ok(simplified) => {
                        debug!(
                            "OpenCV approxPolyDP 简化：{} -> {} 个轮廓",
                            contours.len(),
                            simplified.len()
                        );
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
        let mut simplified = self.simplify_polylines(&contours, &adapted_config);

        // 7. 线型检测（如果启用）
        if adapted_config.line_type_detection {
            let line_types = detect_line_types_from_polylines(
                &simplified,
                adapted_config.max_angle_dev_deg,
                adapted_config.snap_tolerance_px * 3.0,
            );
            let dashed = line_types
                .iter()
                .filter(|lt| lt.line_type == LineType::Dashed)
                .count();
            let center = line_types
                .iter()
                .filter(|lt| lt.line_type == LineType::Center)
                .count();
            let group_count = line_types
                .iter()
                .map(|lt| lt.group_id)
                .max()
                .map(|g| g + 1)
                .unwrap_or(0);
            debug!(
                "线型检测: {} 条虚线, {} 条中心线, {} 个共线组",
                dashed, center, group_count
            );
        }

        // 8. 缺口填充（原有基础填充）
        if adapted_config.gap_filling {
            let before = simplified.len();
            // 使用 snap_tolerance * 3 作为最大缺口，max_angle_dev_deg 作为角度容差
            simplified = fill_gaps(
                &simplified,
                adapted_config.snap_tolerance_px * 3.0,
                adapted_config.max_angle_dev_deg,
            );
            debug!("基础缺口填充: {} -> {} 条多段线", before, simplified.len());
        }

        // [新增] 9. 霍夫变换辅助缺口填充（处理更大间隔）
        if adapted_config.hough_gap_filling {
            let before = simplified.len();
            simplified = hough_assisted_gap_filling(
                &binary,
                &simplified,
                adapted_config.snap_tolerance_px * 6.0,
                adapted_config.max_angle_dev_deg,
                adapted_config.hough_threshold,
            );
            debug!(
                "霍夫辅助缺口填充: {} -> {} 条多段线",
                before,
                simplified.len()
            );
        }

        // [新增] 10. 建筑规则几何校正（正交性/平行性校正）
        if adapted_config.architectural_correction {
            let before = simplified.len();
            architectural_rules::correct_all(
                &mut simplified,
                5.0,                                    // 5 度容差
                0.15,                                   // 15% 间距容差
                adapted_config.snap_tolerance_px * 4.0, // 最大闭合间隙
            );
            debug!(
                "建筑规则校正: {} -> {} 条多段线（正交化+平行均匀化+闭合）",
                before,
                simplified.len()
            );
        }

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

        // 对比度增强（在去噪之前，因为增强后去噪更有效）
        if config.enhance_contrast {
            result = preprocessing::clahe(&result, config.clahe_clip_limit, config.clahe_tile_size);
        }

        // 去噪
        if config.denoise {
            match config.denoise_method.as_str() {
                "median" => {
                    let strength = (config.denoise_strength as u32).clamp(1, 5);
                    result = preprocessing::median_filter(&result, strength);
                }
                "gaussian" => {
                    let sigma = config.denoise_strength.max(0.5);
                    result = preprocessing::gaussian_blur(&result, sigma);
                }
                "nlmeans" => {
                    let h = config.denoise_strength.max(1.0);
                    result = preprocessing::non_local_means(&result, h);
                }
                _ => {
                    // 默认回退到中值滤波
                    result = preprocessing::median_filter(&result, 3);
                }
            }
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
                    let dist_sq =
                        (first[0] - other_first[0]).powi(2) + (first[1] - other_first[1]).powi(2);
                    if dist_sq < tol_sq && dist_sq > 0.0 {
                        result[i][0] = other_first;
                        changed = true;
                    }

                    // 检查 first 与 other_last
                    let dist_sq =
                        (first[0] - other_last[0]).powi(2) + (first[1] - other_last[1]).powi(2);
                    if dist_sq < tol_sq && dist_sq > 0.0 {
                        result[i][0] = other_last;
                        changed = true;
                    }

                    // 检查 last 与 other_first
                    let dist_sq =
                        (last[0] - other_first[0]).powi(2) + (last[1] - other_first[1]).powi(2);
                    if dist_sq < tol_sq && dist_sq > 0.0 {
                        result[i][len - 1] = other_first;
                        changed = true;
                    }

                    // 检查 last 与 other_last
                    let dist_sq =
                        (last[0] - other_last[0]).powi(2) + (last[1] - other_last[1]).powi(2);
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

/// 将加速器返回的 EdgeMap 转换为 image crate 的 GrayImage
fn edge_map_to_gray_image(edge_map: &EdgeMap) -> GrayImage {
    use image::{GrayImage, Luma};

    let mut result = GrayImage::new(edge_map.width, edge_map.height);

    // EdgeMap.data 中: 0 = 边缘（黑色），255 = 非边缘（白色）
    // 这与我们现有的表示方式一致，直接复制即可
    for (y, row) in result.rows_mut().enumerate() {
        for (x, pixel) in row.enumerate() {
            let idx = y * (edge_map.width as usize) + x;
            *pixel = Luma([edge_map.data[idx]]);
        }
    }

    result
}

fn attempt_score(quality_score: f64, polyline_count: usize, contour_count: usize) -> f64 {
    let geometry_score = if polyline_count == 0 {
        0.0
    } else {
        (polyline_count as f64).log10().min(2.0) * 15.0
    };
    let contour_penalty = if contour_count > 10_000 { 15.0 } else { 0.0 };
    (quality_score + geometry_score - contour_penalty).clamp(0.0, 100.0)
}

fn primitive_candidates(polylines: &[Polyline]) -> Vec<PrimitiveCandidate> {
    polylines
        .iter()
        .enumerate()
        .take(256)
        .filter_map(|(_, polyline)| {
            crate::algorithms::fit_best_primitive(polyline, 2.0).map(|fit| {
                let (primitive_type, start, end) = match &fit.data {
                    FitData::Line(line) => ("line", line.start, line.end),
                    FitData::Arc(arc) => {
                        let start = [
                            arc.center[0] + arc.radius * arc.start_angle.cos(),
                            arc.center[1] + arc.radius * arc.start_angle.sin(),
                        ];
                        let end = [
                            arc.center[0] + arc.radius * arc.end_angle.cos(),
                            arc.center[1] + arc.radius * arc.end_angle.sin(),
                        ];
                        ("arc", start, end)
                    }
                    FitData::Bezier(bezier) => ("bezier", bezier.p0, bezier.p3),
                };

                PrimitiveCandidate {
                    primitive_type: primitive_type.to_string(),
                    start,
                    end,
                    rms_error: fit.rms_error,
                    confidence: (1.0 / (1.0 + fit.rms_error)).clamp(0.0, 1.0),
                }
            })
        })
        .collect()
}

fn text_candidates(gray: &GrayImage) -> Vec<TextCandidate> {
    let ocr = crate::algorithms::HeuristicOcrBackend::new();
    ocr.recognize(gray)
        .into_iter()
        .map(|text| {
            let accepted = text.confidence >= 0.5;
            TextCandidate {
                content: text.text,
                confidence: text.confidence,
                bbox: [
                    text.bbox.x_min as f64,
                    text.bbox.y_min as f64,
                    text.bbox.x_max as f64,
                    text.bbox.y_max as f64,
                ],
                rotation: text.orientation,
                accepted,
            }
        })
        .collect()
}

fn symbol_candidates(gray: &GrayImage) -> Vec<SymbolCandidate> {
    let classifier = crate::algorithms::SymbolClassifier::new();
    classifier
        .classify(gray)
        .into_iter()
        .map(|symbol| SymbolCandidate {
            symbol_type: symbol.symbol_type.to_string(),
            confidence: symbol.confidence,
            bbox: [
                symbol.x as f64,
                symbol.y as f64,
                (symbol.x + symbol.width) as f64,
                (symbol.y + symbol.height) as f64,
            ],
            rotation: symbol.rotation,
        })
        .collect()
}

fn semantic_candidates(polylines: &[Polyline]) -> Vec<SemanticCandidate> {
    polylines
        .iter()
        .enumerate()
        .filter_map(|(idx, polyline)| {
            if polyline.len() < 2 {
                return None;
            }
            let length = polyline
                .windows(2)
                .map(|segment| {
                    let dx = segment[1][0] - segment[0][0];
                    let dy = segment[1][1] - segment[0][1];
                    (dx * dx + dy * dy).sqrt()
                })
                .sum::<f64>();
            let semantic_type = if length > 80.0 {
                "hard_wall"
            } else if length > 25.0 {
                "opening"
            } else {
                "detail_line"
            };
            Some(SemanticCandidate {
                target_id: idx,
                semantic_type: semantic_type.to_string(),
                confidence: if length > 80.0 { 0.72 } else { 0.55 },
                source: "rule".to_string(),
            })
        })
        .collect()
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
