//! 矢量化 pipeline - 模块化 trait 设计
//!
//! 使用 Stage trait 模式将矢量化流程分解为可替换、可组合的阶段：
//! - 图像输入与上下文包装
//! - 预处理（去噪、对比度增强、阴影去除等）
//! - 二值化（Otsu 自适应阈值或固定阈值）
//! - 骨架化（Zhang-Suen 细化算法）
//! - 轮廓追踪（提取 1 像素宽的线段）
//! - 几何拟合（Douglas-Peucker 简化 + 端点吸附）

use std::time::Instant;

use base64::{engine::general_purpose, Engine as _};
use common_types::{CadError, Polyline, ServiceMetrics};
use image::{DynamicImage, GenericImageView, GrayImage};
use serde::{Deserialize, Serialize};
use thiserror::Error;

use crate::algorithms::{
    adaptive_params, douglas_peucker, extract_contours, fill_gaps, perspective_correction,
    preprocessing, skeletonize, threshold as threshold_algo, SkeletonAlgorithm, SkeletonConfig,
    SubpixelPoint,
};
use crate::config::{RasterStrategy, VectorizeConfig};
use crate::quality::evaluate_image_quality;

/// Pipeline 阶段错误
#[derive(Debug, Error)]
pub enum StageError {
    #[error("图像处理失败: {0}")]
    ImageProcessing(String),
    #[error("配置错误: {0}")]
    Config(String),
    #[error("算法执行失败: {0}")]
    Algorithm(String),
    #[error(transparent)]
    Cad(#[from] CadError),
}

pub type StageResult<T> = Result<T, StageError>;

/// 阶段执行统计
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct StageStats {
    pub stage_name: String,
    pub duration_ms: u64,
    pub input_size: Option<(u32, u32)>,
    pub output_size: Option<(u32, u32)>,
    pub extra: serde_json::Value,
}

/// 自动识别的光栅图纸类型。
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, Default)]
#[serde(rename_all = "snake_case")]
pub enum RasterKind {
    #[default]
    LineArt,
    Scan,
    Photo,
    Sketch,
    LowContrast,
}

/// 单个调试中间图，默认不生成。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DebugArtifact {
    pub name: String,
    pub mime_type: String,
    pub width: u32,
    pub height: u32,
    pub data_base64: String,
}

/// 一次质量反馈尝试的摘要。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorizationAttemptReport {
    pub attempt_index: usize,
    pub strategy: RasterStrategy,
    pub threshold: u8,
    pub snap_tolerance_px: f64,
    pub min_line_length_px: f64,
    pub quality_score: f64,
    pub polyline_count: usize,
    pub score: f64,
    pub error_code: Option<String>,
    pub message: Option<String>,
}

/// 几何基元候选。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PrimitiveCandidate {
    pub primitive_type: String,
    pub start: common_types::Point2,
    pub end: common_types::Point2,
    pub rms_error: f64,
    pub confidence: f64,
}

/// OCR 文本候选。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TextCandidate {
    pub content: String,
    pub confidence: f64,
    pub bbox: [f64; 4],
    pub rotation: f64,
    pub accepted: bool,
}

/// CAD 符号候选。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SymbolCandidate {
    pub symbol_type: String,
    pub confidence: f64,
    pub bbox: [f64; 4],
    pub rotation: f64,
}

/// 光栅语义候选。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct SemanticCandidate {
    pub target_id: usize,
    pub semantic_type: String,
    pub confidence: f64,
    pub source: String,
}

/// 光栅矢量化结构化报告。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RasterVectorizationReport {
    pub schema_version: String,
    pub input_width: u32,
    pub input_height: u32,
    pub working_width: u32,
    pub working_height: u32,
    pub scale_factor: f64,
    pub quality_score: f64,
    pub threshold: u8,
    pub contour_count: usize,
    pub final_polyline_count: usize,
    pub gap_fill_count: usize,
    pub detected_raster_kind: RasterKind,
    pub detected_raster_kind_confidence: f64,
    pub selected_strategy: RasterStrategy,
    pub stage_stats: Vec<StageStats>,
    pub attempts: Vec<VectorizationAttemptReport>,
    pub failure_code: Option<String>,
    pub failure_reason: Option<String>,
    pub recommendations: Vec<String>,
    pub debug_artifacts: Vec<DebugArtifact>,
    pub primitive_candidates: Vec<PrimitiveCandidate>,
    pub text_candidates: Vec<TextCandidate>,
    pub symbol_candidates: Vec<SymbolCandidate>,
    pub semantic_candidates: Vec<SemanticCandidate>,
}

impl RasterVectorizationReport {
    pub fn failed(
        input_width: u32,
        input_height: u32,
        code: impl Into<String>,
        reason: impl Into<String>,
    ) -> Self {
        let code = code.into();
        let reason = reason.into();
        Self {
            schema_version: "raster-report-1.0".to_string(),
            input_width,
            input_height,
            working_width: input_width,
            working_height: input_height,
            scale_factor: 1.0,
            quality_score: 0.0,
            threshold: 128,
            contour_count: 0,
            final_polyline_count: 0,
            gap_fill_count: 0,
            detected_raster_kind: RasterKind::LineArt,
            detected_raster_kind_confidence: 0.0,
            selected_strategy: RasterStrategy::Auto,
            stage_stats: Vec::new(),
            attempts: Vec::new(),
            failure_code: Some(code),
            failure_reason: Some(reason),
            recommendations: vec![
                "尝试显式设置 raster_strategy".to_string(),
                "提供 dpi_override 或 scale_calibration 以恢复可靠尺度".to_string(),
                "开启 debug_artifacts 检查阈值和骨架化结果".to_string(),
            ],
            debug_artifacts: Vec::new(),
            primitive_candidates: Vec::new(),
            text_candidates: Vec::new(),
            symbol_candidates: Vec::new(),
            semantic_candidates: Vec::new(),
        }
    }
}

/// 结构化矢量化输出。
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RasterVectorizationOutput {
    pub polylines: Vec<Polyline>,
    pub report: RasterVectorizationReport,
}

/// 可插拔的 Pipeline Stage trait
///
/// 每个阶段接收上一阶段的输出，产生本阶段的输出。
/// 通过泛型 Input/Output 实现类型安全的流水线组合。
pub trait Stage<Input, Output>: Send + Sync {
    /// 阶段名称（用于日志和指标）
    fn name(&self) -> &'static str;

    /// 执行阶段处理
    fn process(&self, input: Input) -> StageResult<Output>;

    /// 执行阶段处理并返回统计信息
    fn process_with_stats(&self, input: Input) -> StageResult<(Output, StageStats)>
    where
        Self: Sized,
    {
        let start = Instant::now();
        let result = self.process(input)?;
        let duration = start.elapsed();

        let stats = StageStats {
            stage_name: self.name().to_string(),
            duration_ms: duration.as_millis() as u64,
            input_size: None,
            output_size: None,
            extra: serde_json::Value::Null,
        };

        Ok((result, stats))
    }
}

/// 携带上下文的图像数据
///
/// 在 Pipeline 各阶段间传递，保留原始元数据和中间结果引用
#[derive(Clone)]
pub struct ImageWithContext {
    /// 当前灰度图像
    pub image: GrayImage,
    /// 原始图像尺寸
    pub original_dimensions: (u32, u32),
    /// 应用的缩放比例（如果有缩放）
    pub scale_factor: f64,
    /// 质量评分（如果已计算）
    pub quality_score: Option<f64>,
    /// 原始图像引用（可选，用于需要回溯的阶段）
    pub original_image: Option<GrayImage>,
}

impl ImageWithContext {
    /// 从 DynamicImage 创建
    pub fn from_dynamic(img: &DynamicImage, max_pixels: usize) -> Self {
        let (orig_w, orig_h) = img.dimensions();
        let total = (orig_w as usize) * (orig_h as usize);

        if total > max_pixels {
            let scale = (max_pixels as f64 / total as f64).sqrt();
            let new_w = (orig_w as f64 * scale).round() as u32;
            let new_h = (orig_h as f64 * scale).round() as u32;
            let scaled = img
                .resize(new_w, new_h, image::imageops::FilterType::Lanczos3)
                .to_luma8();

            Self {
                image: scaled.clone(),
                original_dimensions: (orig_w, orig_h),
                scale_factor: scale,
                quality_score: None,
                original_image: Some(scaled),
            }
        } else {
            let gray = img.to_luma8();
            Self {
                image: gray.clone(),
                original_dimensions: (orig_w, orig_h),
                scale_factor: 1.0,
                quality_score: None,
                original_image: Some(gray),
            }
        }
    }

    /// 获取当前尺寸
    pub fn dimensions(&self) -> (u32, u32) {
        self.image.dimensions()
    }
}

/// 骨架化结果
#[derive(Clone)]
pub struct SkeletonResult {
    /// 骨架图像
    pub skeleton: GrayImage,
    /// 亚像素精度骨架点
    pub subpixel_points: Vec<SubpixelPoint>,
    /// 原始二值图像
    pub binary: GrayImage,
    /// 图像上下文
    pub context: ImageWithContext,
}

/// 轮廓提取结果
#[derive(Clone)]
pub struct TraceResult {
    /// 提取的原始轮廓
    pub contours: Vec<Polyline>,
    /// 骨架图像
    pub skeleton: GrayImage,
    /// 二值图像
    pub binary: GrayImage,
    /// 图像上下文
    pub context: ImageWithContext,
}

/// 拟合结果
#[derive(Clone)]
pub struct FitResult {
    /// 最终的多段线集合
    pub polylines: Vec<Polyline>,
    /// 图像上下文
    pub context: ImageWithContext,
}

// ============================================================================
// Stage 1: 预处理阶段
// ============================================================================

/// 预处理阶段配置
#[derive(Debug, Clone)]
pub struct PreprocessStageConfig {
    /// 启用自动纸张裁剪
    pub auto_crop_paper: bool,
    /// 启用透视校正
    pub perspective_correction: bool,
    /// 启用质量评估
    pub quality_assessment: bool,
    /// 启用自适应参数调整
    pub adaptive_params: bool,
    /// 原始矢量化配置（用于自适应参数）
    pub base_config: VectorizeConfig,
}

/// 预处理阶段：纸张检测、透视校正、质量评估、去噪、对比度增强
pub struct PreprocessStage {
    config: PreprocessStageConfig,
}

impl PreprocessStage {
    pub fn new(config: PreprocessStageConfig) -> Self {
        Self { config }
    }
}

impl Stage<ImageWithContext, (ImageWithContext, VectorizeConfig)> for PreprocessStage {
    fn name(&self) -> &'static str {
        "Preprocess"
    }

    fn process(
        &self,
        mut input: ImageWithContext,
    ) -> StageResult<(ImageWithContext, VectorizeConfig)> {
        let mut gray = input.image.clone();

        // 自动纸张检测与裁剪
        if self.config.auto_crop_paper {
            let cropped = crate::algorithms::paper_detection::detect_and_crop(&gray);
            if cropped.width() < gray.width() || cropped.height() < gray.height() {
                gray = cropped;
            }
        }

        // 透视校正
        if self.config.perspective_correction {
            if let Some(region) = crate::algorithms::paper_detection::detect_paper(&gray) {
                if region.confidence > 0.5 {
                    gray = perspective_correction::correct_perspective(&gray, &region);
                }
            }
        }

        // 质量评估
        let quality_score = if self.config.quality_assessment {
            let score = evaluate_image_quality(&gray);
            input.quality_score = Some(score);
            Some(score)
        } else {
            None
        };

        // 自适应参数调整
        let adapted_config =
            if let (true, Some(score)) = (self.config.adaptive_params, quality_score) {
                let (w, h) = gray.dimensions();
                adaptive_params::adapt_parameters(score, w, h, &self.config.base_config)
            } else {
                self.config.base_config.clone()
            };

        // 去噪 + 对比度增强
        let config = &adapted_config.preprocessing;
        let mut result = gray.clone();

        if config.enhance_contrast {
            result = preprocessing::clahe(&result, config.clahe_clip_limit, config.clahe_tile_size);
        }

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
                    result = preprocessing::median_filter(&result, 3);
                }
            }
        }

        input.image = result;

        Ok((input, adapted_config))
    }
}

// ============================================================================
// Stage 2: 二值化阶段
// ============================================================================

/// 二值化阶段配置
#[derive(Debug, Clone)]
pub struct ThresholdStageConfig {
    /// 使用自适应 Otsu 阈值
    pub use_otsu: bool,
    /// 固定阈值（不使用 Otsu 时）
    pub fixed_threshold: u8,
    /// 启用文字分离
    pub text_separation: bool,
}

/// 二值化阶段：Otsu 自适应阈值或固定阈值
pub struct ThresholdStage {
    config: ThresholdStageConfig,
}

impl ThresholdStage {
    pub fn new(config: ThresholdStageConfig) -> Self {
        Self { config }
    }
}

impl Stage<ImageWithContext, ImageWithContext> for ThresholdStage {
    fn name(&self) -> &'static str {
        "Threshold"
    }

    fn process(&self, mut input: ImageWithContext) -> StageResult<ImageWithContext> {
        let binary = if self.config.use_otsu {
            threshold_algo::binary_with_otsu(&input.image)
        } else {
            threshold_algo::threshold_binary(&input.image, self.config.fixed_threshold)
        };

        // 文字 blob 过滤
        let binary = if self.config.text_separation {
            let blobs = crate::algorithms::detect_text_blobs(&binary, 5, 5000);
            crate::algorithms::erase_text_blobs(&binary, &blobs)
        } else {
            binary
        };

        input.image = binary;
        Ok(input)
    }
}

// ============================================================================
// Stage 3: 骨架化阶段（Zhang-Suen 细化算法）
// ============================================================================

/// 骨架化阶段配置
#[derive(Debug, Clone)]
pub struct SkeletonizeStageConfig {
    /// 启用边缘检测前置
    pub edge_detect: bool,
    /// 骨架化算法
    pub algorithm: SkeletonAlgorithm,
    /// 启用去毛刺后处理
    pub enable_de_spur: bool,
    /// 毛刺最大长度（像素）
    pub max_spur_length: usize,
    /// 启用亚像素精度
    pub enable_subpixel: bool,
}

impl Default for SkeletonizeStageConfig {
    fn default() -> Self {
        Self {
            edge_detect: true,
            algorithm: SkeletonAlgorithm::ZhangSuen,
            enable_de_spur: true,
            max_spur_length: 3,
            enable_subpixel: true,
        }
    }
}

/// 骨架化阶段：Zhang-Suen/Guo-Hall 细化算法 + 亚像素精度
pub struct SkeletonizeStage {
    config: SkeletonizeStageConfig,
}

impl SkeletonizeStage {
    pub fn new(config: SkeletonizeStageConfig) -> Self {
        Self { config }
    }
}

impl Stage<ImageWithContext, SkeletonResult> for SkeletonizeStage {
    fn name(&self) -> &'static str {
        "Skeletonize"
    }

    fn process(&self, input: ImageWithContext) -> StageResult<SkeletonResult> {
        let edges = if self.config.edge_detect {
            crate::algorithms::detect_edges(&input.image)
        } else {
            input.image.clone()
        };

        let skeleton_config = SkeletonConfig {
            algorithm: self.config.algorithm,
            enable_de_spur: self.config.enable_de_spur,
            max_spur_length: self.config.max_spur_length,
            enable_subpixel: self.config.enable_subpixel,
        };

        let (skeleton, subpixel_points) = if self.config.enable_subpixel {
            crate::algorithms::skeletonize_with_subpixel(&edges, &skeleton_config)
        } else {
            let skeleton = skeletonize(&edges, &skeleton_config);
            (skeleton, Vec::new())
        };

        Ok(SkeletonResult {
            skeleton,
            subpixel_points,
            binary: input.image.clone(),
            context: input,
        })
    }
}

// ============================================================================
// Stage 4: 轮廓追踪阶段
// ============================================================================

/// 轮廓追踪配置
#[derive(Debug, Clone)]
pub struct TraceStageConfig {
    /// 最小线段长度（像素）
    pub min_line_length_px: f64,
}

/// 轮廓追踪阶段：从骨架图像提取矢量轮廓
pub struct TraceStage {
    config: TraceStageConfig,
}

impl TraceStage {
    pub fn new(config: TraceStageConfig) -> Self {
        Self { config }
    }
}

impl Stage<SkeletonResult, TraceResult> for TraceStage {
    fn name(&self) -> &'static str {
        "Trace"
    }

    fn process(&self, input: SkeletonResult) -> StageResult<TraceResult> {
        let min_len = self.config.min_line_length_px as usize;
        let contours = extract_contours(&input.skeleton, min_len);

        Ok(TraceResult {
            contours,
            skeleton: input.skeleton,
            binary: input.binary,
            context: input.context,
        })
    }
}

// ============================================================================
// Stage 5: 几何拟合阶段
// ============================================================================

/// 拟合阶段配置
#[derive(Debug, Clone)]
pub struct FitStageConfig {
    /// Douglas-Peucker 简化容差
    pub simplify_tolerance: f64,
    /// 端点吸附容差
    pub snap_tolerance: f64,
    /// 启用缺口填充
    pub gap_filling: bool,
    /// 最大角度偏差（度）
    pub max_angle_dev_deg: f64,
}

/// 拟合阶段：Douglas-Peucker 简化 + 端点吸附 + 缺口填充
pub struct FitStage {
    config: FitStageConfig,
}

impl FitStage {
    pub fn new(config: FitStageConfig) -> Self {
        Self { config }
    }
}

impl Stage<TraceResult, FitResult> for FitStage {
    fn name(&self) -> &'static str {
        "Fit"
    }

    fn process(&self, input: TraceResult) -> StageResult<FitResult> {
        // 1. Douglas-Peucker 简化
        let simplified: Vec<Polyline> = input
            .contours
            .iter()
            .filter_map(|pl| {
                if pl.len() < 2 {
                    return None;
                }
                let simplified = douglas_peucker(pl, self.config.simplify_tolerance);
                if simplified.len() >= 2 {
                    Some(simplified)
                } else {
                    None
                }
            })
            .collect();

        // 2. 端点吸附
        let snapped = snap_endpoints_global(&simplified, self.config.snap_tolerance);

        // 3. 缺口填充
        let final_polylines = if self.config.gap_filling {
            fill_gaps(
                &snapped,
                self.config.snap_tolerance * 3.0,
                self.config.max_angle_dev_deg,
            )
        } else {
            snapped
        };

        Ok(FitResult {
            polylines: final_polylines,
            context: input.context,
        })
    }
}

// ============================================================================
// Stage 6: 精炼优化阶段
// ============================================================================

/// 精炼优化阶段配置
#[derive(Debug, Clone)]
pub struct RefineStageConfig {
    /// 启用重合线段合并
    pub merge_overlapping: bool,
    /// 重合检测容差
    pub merge_tolerance: f64,
    /// 启用共线线段合并
    pub merge_colinear: bool,
    /// 共线角度容差（度）
    pub colinear_angle_tolerance: f64,
    /// 启用太短线段过滤
    pub filter_too_short: bool,
    /// 最小线段长度
    pub min_segment_length: f64,
}

impl Default for RefineStageConfig {
    fn default() -> Self {
        Self {
            merge_overlapping: true,
            merge_tolerance: 1.5,
            merge_colinear: true,
            colinear_angle_tolerance: 5.0,
            filter_too_short: true,
            min_segment_length: 3.0,
        }
    }
}

/// 精炼优化阶段：拓扑清理、线段合并、过滤
pub struct RefineStage {
    config: RefineStageConfig,
}

impl RefineStage {
    pub fn new(config: RefineStageConfig) -> Self {
        Self { config }
    }
}

impl Stage<FitResult, FitResult> for RefineStage {
    fn name(&self) -> &'static str {
        "Refine"
    }

    fn process(&self, input: FitResult) -> StageResult<FitResult> {
        let mut polylines = input.polylines;

        // 1. 过滤太短线段
        if self.config.filter_too_short {
            polylines.retain(|pl| {
                pl.len() >= 2 && polyline_length(pl) >= self.config.min_segment_length
            });
        }

        // 2. 合并共线线段
        if self.config.merge_colinear {
            polylines = merge_colinear_segments(&polylines, self.config.colinear_angle_tolerance);
        }

        // 3. 合并重合端点
        if self.config.merge_overlapping {
            polylines = snap_endpoints_global(&polylines, self.config.merge_tolerance);
        }

        Ok(FitResult {
            polylines,
            context: input.context,
        })
    }
}

// ============================================================================
// 矢量化策略枚举
// ============================================================================

/// 矢量化策略类型
///
/// 支持多种矢量化策略，可根据输入质量和硬件环境选择：
/// - Traditional: 纯传统 CV 方法，全平台兼容（默认）
/// - DeepLearningAssisted: 深度学习辅助（需要启用相应 feature）
/// - Hybrid: 混合策略，传统 CV 为主，DL 处理复杂区域
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum VectorizationStrategy {
    /// 传统计算机视觉方法（默认，纯 CPU，全兼容）
    #[default]
    Traditional,
    /// 深度学习辅助（需要相应 feature，推荐 GPU 环境）
    DeepLearningAssisted,
    /// 混合策略（传统 CV 为主，DL 用于复杂区域）
    Hybrid,
}

// ============================================================================
// Pipeline Builder
// ============================================================================

/// 矢量化 Pipeline Builder
///
/// 用于构建自定义的矢量化流水线，支持替换各阶段的实现
pub struct VectorizePipelineBuilder {
    max_pixels: usize,
    strategy: VectorizationStrategy,
    preprocess_config: Option<PreprocessStageConfig>,
    threshold_config: Option<ThresholdStageConfig>,
    skeletonize_config: Option<SkeletonizeStageConfig>,
    trace_config: Option<TraceStageConfig>,
    fit_config: Option<FitStageConfig>,
    refine_config: Option<RefineStageConfig>,
}

impl Default for VectorizePipelineBuilder {
    fn default() -> Self {
        Self::new()
    }
}

impl VectorizePipelineBuilder {
    /// 创建新的 Builder
    pub fn new() -> Self {
        Self {
            max_pixels: 30_000_000,
            strategy: VectorizationStrategy::default(),
            preprocess_config: None,
            threshold_config: None,
            skeletonize_config: None,
            trace_config: None,
            fit_config: None,
            refine_config: None,
        }
    }

    /// 设置矢量化策略
    pub fn strategy(mut self, strategy: VectorizationStrategy) -> Self {
        self.strategy = strategy;
        self
    }

    /// 设置最大像素限制
    pub fn max_pixels(mut self, max: usize) -> Self {
        self.max_pixels = max;
        self
    }

    /// 设置预处理配置
    pub fn preprocess_config(mut self, config: PreprocessStageConfig) -> Self {
        self.preprocess_config = Some(config);
        self
    }

    /// 设置二值化配置
    pub fn threshold_config(mut self, config: ThresholdStageConfig) -> Self {
        self.threshold_config = Some(config);
        self
    }

    /// 设置骨架化配置
    pub fn skeletonize_config(mut self, config: SkeletonizeStageConfig) -> Self {
        self.skeletonize_config = Some(config);
        self
    }

    /// 设置轮廓追踪配置
    pub fn trace_config(mut self, config: TraceStageConfig) -> Self {
        self.trace_config = Some(config);
        self
    }

    /// 设置拟合配置
    pub fn fit_config(mut self, config: FitStageConfig) -> Self {
        self.fit_config = Some(config);
        self
    }

    /// 设置精炼优化配置
    pub fn refine_config(mut self, config: RefineStageConfig) -> Self {
        self.refine_config = Some(config);
        self
    }

    /// 使用默认 VectorizeConfig 快速构建
    pub fn from_config(config: &VectorizeConfig) -> Self {
        Self::new()
            .max_pixels(config.max_pixels)
            .preprocess_config(PreprocessStageConfig {
                auto_crop_paper: config.auto_crop_paper,
                perspective_correction: config.perspective_correction,
                quality_assessment: config.quality_assessment,
                adaptive_params: config.adaptive_params,
                base_config: config.clone(),
            })
            .threshold_config(ThresholdStageConfig {
                use_otsu: config.adaptive_threshold,
                fixed_threshold: config.threshold,
                text_separation: config.text_separation,
            })
            .skeletonize_config(SkeletonizeStageConfig {
                edge_detect: config.skeletonize,
                ..Default::default()
            })
            .trace_config(TraceStageConfig {
                min_line_length_px: config.min_line_length_px,
            })
            .fit_config(FitStageConfig {
                simplify_tolerance: config.snap_tolerance_px,
                snap_tolerance: config.snap_tolerance_px,
                gap_filling: config.gap_filling,
                max_angle_dev_deg: config.max_angle_dev_deg,
            })
            .refine_config(RefineStageConfig::default())
    }

    /// 构建完整的 Pipeline
    pub fn build(self) -> VectorizePipeline {
        let base_config = self
            .preprocess_config
            .as_ref()
            .map(|c| c.base_config.clone())
            .unwrap_or_default();

        VectorizePipeline {
            max_pixels: self.max_pixels,
            strategy: self.strategy,
            preprocess: PreprocessStage::new(self.preprocess_config.unwrap_or_else(|| {
                PreprocessStageConfig {
                    auto_crop_paper: true,
                    perspective_correction: true,
                    quality_assessment: true,
                    adaptive_params: true,
                    base_config: VectorizeConfig::default(),
                }
            })),
            threshold: ThresholdStage::new(self.threshold_config.unwrap_or(ThresholdStageConfig {
                use_otsu: true,
                fixed_threshold: 128,
                text_separation: true,
            })),
            skeletonize: SkeletonizeStage::new(self.skeletonize_config.unwrap_or(
                SkeletonizeStageConfig {
                    edge_detect: true,
                    ..Default::default()
                },
            )),
            trace: TraceStage::new(self.trace_config.unwrap_or(TraceStageConfig {
                min_line_length_px: 10.0,
            })),
            fit: FitStage::new(self.fit_config.unwrap_or(FitStageConfig {
                simplify_tolerance: 2.0,
                snap_tolerance: 2.0,
                gap_filling: true,
                max_angle_dev_deg: 5.0,
            })),
            refine: RefineStage::new(self.refine_config.unwrap_or_default()),
            metrics: ServiceMetrics::new("VectorizePipeline"),
            base_config,
        }
    }
}

/// 完整的矢量化 Pipeline
pub struct VectorizePipeline {
    max_pixels: usize,
    strategy: VectorizationStrategy,
    preprocess: PreprocessStage,
    threshold: ThresholdStage,
    skeletonize: SkeletonizeStage,
    trace: TraceStage,
    fit: FitStage,
    refine: RefineStage,
    metrics: ServiceMetrics,
    base_config: VectorizeConfig,
}

impl Default for VectorizePipeline {
    fn default() -> Self {
        VectorizePipelineBuilder::from_config(&VectorizeConfig::default()).build()
    }
}

impl VectorizePipeline {
    /// 使用默认配置创建 Pipeline
    pub fn new() -> Self {
        Self::default()
    }

    /// 获取指标
    pub fn metrics(&self) -> &ServiceMetrics {
        &self.metrics
    }

    /// 获取基础配置
    pub fn config(&self) -> &VectorizeConfig {
        &self.base_config
    }

    /// 获取当前使用的矢量化策略
    pub fn strategy(&self) -> VectorizationStrategy {
        self.strategy
    }

    /// 执行完整的矢量化流程
    pub fn process(&self, img: &DynamicImage) -> StageResult<Vec<Polyline>> {
        let start = Instant::now();

        // 阶段 0: 图像包装
        let context = ImageWithContext::from_dynamic(img, self.max_pixels);

        // 阶段 1: 预处理
        let (context, _adapted_config) = self.preprocess.process(context)?;

        // 阶段 2: 二值化
        let thresholded = self.threshold.process(context)?;

        // 阶段 3: 骨架化
        let skeleton = self.skeletonize.process(thresholded)?;

        // 阶段 4: 轮廓追踪
        let traced = self.trace.process(skeleton)?;

        // 阶段 5: 拟合
        let fitted = self.fit.process(traced)?;

        // 阶段 6: 精炼优化
        let result = self.refine.process(fitted)?;

        let _duration = start.elapsed();
        Ok(result.polylines)
    }

    /// 执行完整流程并返回各阶段统计
    pub fn process_with_stats(
        &self,
        img: &DynamicImage,
    ) -> StageResult<(Vec<Polyline>, Vec<StageStats>)> {
        let mut stats = Vec::with_capacity(6);

        // 阶段 0: 图像包装
        let context = ImageWithContext::from_dynamic(img, self.max_pixels);

        // 阶段 1: 预处理
        let (context, _adapted_config) = {
            let (r, s) = self.preprocess.process_with_stats(context)?;
            stats.push(s);
            (r.0, r.1)
        };

        // 阶段 2: 二值化
        let (thresholded, stat) = self.threshold.process_with_stats(context)?;
        stats.push(stat);

        // 阶段 3: 骨架化
        let (skeleton, stat) = self.skeletonize.process_with_stats(thresholded)?;
        stats.push(stat);

        // 阶段 4: 轮廓追踪
        let (traced, stat) = self.trace.process_with_stats(skeleton)?;
        stats.push(stat);

        // 阶段 5: 拟合
        let (fitted, stat) = self.fit.process_with_stats(traced)?;
        stats.push(stat);

        // 阶段 6: 精炼优化
        let (result, stat) = self.refine.process_with_stats(fitted)?;
        stats.push(stat);

        Ok((result.polylines, stats))
    }

    /// 执行完整流程并返回结构化报告和可选调试中间图。
    pub fn process_detailed(
        &self,
        img: &DynamicImage,
        debug_artifacts: bool,
    ) -> StageResult<RasterVectorizationOutput> {
        let mut stats = Vec::with_capacity(6);
        let mut artifacts = Vec::new();

        let context = ImageWithContext::from_dynamic(img, self.max_pixels);
        let input_width = context.original_dimensions.0;
        let input_height = context.original_dimensions.1;
        let scale_factor = context.scale_factor;

        if debug_artifacts {
            artifacts.push(encode_gray_debug("gray", &context.image)?);
        }

        let (context, adapted_config) = {
            let (r, s) = self.preprocess.process_with_stats(context)?;
            stats.push(with_sizes(
                s,
                Some((input_width, input_height)),
                Some(r.0.dimensions()),
            ));
            (r.0, r.1)
        };

        let quality_score = context
            .quality_score
            .unwrap_or_else(|| evaluate_image_quality(&context.image));
        let (detected_kind, kind_confidence) = detect_raster_kind(&context.image, quality_score);

        if debug_artifacts {
            artifacts.push(encode_gray_debug("preprocessed", &context.image)?);
        }

        let (thresholded, stat) = self.threshold.process_with_stats(context)?;
        let stat = with_sizes(
            stat,
            Some((thresholded.image.width(), thresholded.image.height())),
            Some((thresholded.image.width(), thresholded.image.height())),
        );
        stats.push(stat);
        if debug_artifacts {
            artifacts.push(encode_gray_debug("binary", &thresholded.image)?);
        }

        let (skeleton, stat) = self.skeletonize.process_with_stats(thresholded)?;
        stats.push(with_sizes(
            stat,
            Some((skeleton.binary.width(), skeleton.binary.height())),
            Some((skeleton.skeleton.width(), skeleton.skeleton.height())),
        ));
        if debug_artifacts {
            artifacts.push(encode_gray_debug("skeleton", &skeleton.skeleton)?);
        }

        let (traced, stat) = self.trace.process_with_stats(skeleton)?;
        let contour_count = traced.contours.len();
        stats.push(with_extra(
            stat,
            serde_json::json!({
                "contours": contour_count,
            }),
        ));

        let (fitted, stat) = self.fit.process_with_stats(traced)?;
        let fitted_count = fitted.polylines.len();
        stats.push(with_extra(
            stat,
            serde_json::json!({
                "polylines": fitted_count,
            }),
        ));

        let (result, stat) = self.refine.process_with_stats(fitted)?;
        stats.push(with_extra(
            stat,
            serde_json::json!({
                "polylines": result.polylines.len(),
            }),
        ));

        let working_width = result.context.image.width();
        let working_height = result.context.image.height();
        let final_polyline_count = result.polylines.len();
        let gap_fill_count = final_polyline_count.saturating_sub(fitted_count);

        let report = RasterVectorizationReport {
            schema_version: "raster-report-1.0".to_string(),
            input_width,
            input_height,
            working_width,
            working_height,
            scale_factor,
            quality_score,
            threshold: adapted_config.threshold,
            contour_count,
            final_polyline_count,
            gap_fill_count,
            detected_raster_kind: detected_kind,
            detected_raster_kind_confidence: kind_confidence,
            selected_strategy: adapted_config.raster_strategy,
            stage_stats: stats,
            attempts: Vec::new(),
            failure_code: None,
            failure_reason: None,
            recommendations: quality_recommendations(quality_score, final_polyline_count),
            debug_artifacts: artifacts,
            primitive_candidates: Vec::new(),
            text_candidates: Vec::new(),
            symbol_candidates: Vec::new(),
            semantic_candidates: Vec::new(),
        };

        Ok(RasterVectorizationOutput {
            polylines: result.polylines,
            report,
        })
    }
}

// ============================================================================
// 辅助函数
// ============================================================================

/// 计算多段线长度
fn polyline_length(pl: &Polyline) -> f64 {
    if pl.len() < 2 {
        return 0.0;
    }
    let mut length = 0.0;
    for i in 0..pl.len() - 1 {
        let dx = pl[i + 1][0] - pl[i][0];
        let dy = pl[i + 1][1] - pl[i][1];
        length += (dx * dx + dy * dy).sqrt();
    }
    length
}

/// 合并共线线段
fn merge_colinear_segments(polylines: &[Polyline], angle_tolerance_deg: f64) -> Vec<Polyline> {
    if polylines.len() < 2 {
        return polylines.to_vec();
    }

    let mut result = polylines.to_vec();
    let tol_rad = angle_tolerance_deg.to_radians();

    for _ in 0..2 {
        let mut merged = false;
        let mut i = 0;

        while i < result.len() {
            if result[i].len() != 2 {
                i += 1;
                continue;
            }

            let dir1 = [
                result[i][1][0] - result[i][0][0],
                result[i][1][1] - result[i][0][1],
            ];
            let len1 = (dir1[0].powi(2) + dir1[1].powi(2)).sqrt();
            if len1 < 0.1 {
                i += 1;
                continue;
            }
            let norm_dir1 = [dir1[0] / len1, dir1[1] / len1];

            let mut j = i + 1;
            while j < result.len() {
                if result[j].len() != 2 {
                    j += 1;
                    continue;
                }

                let dir2 = [
                    result[j][1][0] - result[j][0][0],
                    result[j][1][1] - result[j][0][1],
                ];
                let len2 = (dir2[0].powi(2) + dir2[1].powi(2)).sqrt();
                if len2 < 0.1 {
                    j += 1;
                    continue;
                }
                let norm_dir2 = [dir2[0] / len2, dir2[1] / len2];

                // 检查角度相似性（同向或反向）
                let dot = norm_dir1[0] * norm_dir2[0] + norm_dir1[1] * norm_dir2[1];
                if dot.abs() < (1.0 - tol_rad).cos() {
                    j += 1;
                    continue;
                }

                // 检查端点是否重合
                let p1_end = result[i][result[i].len() - 1];
                let p2_start = result[j][0];
                let dist =
                    ((p1_end[0] - p2_start[0]).powi(2) + (p1_end[1] - p2_start[1]).powi(2)).sqrt();

                if dist < 2.0 {
                    // 合并两条线段
                    let mut new_pl = result[i].clone();
                    new_pl.extend_from_slice(&result[j][1..]);
                    result[i] = new_pl;
                    result.remove(j);
                    merged = true;
                } else {
                    j += 1;
                }
            }

            i += 1;
        }

        if !merged {
            break;
        }
    }

    result
}

/// 全局端点吸附
fn snap_endpoints_global(polylines: &[Polyline], tolerance: f64) -> Vec<Polyline> {
    if polylines.is_empty() {
        return polylines.to_vec();
    }

    let tol_sq = tolerance * tolerance;
    let mut result = polylines.to_vec();

    for _ in 0..3 {
        let mut changed = false;
        for i in 0..result.len() {
            if result[i].len() < 2 {
                continue;
            }
            let first = result[i][0];
            let last = result[i][result[i].len() - 1];
            let len = result[i].len();

            for j in 0..result.len() {
                if i == j || result[j].len() < 2 {
                    continue;
                }
                let other_first = result[j][0];
                let other_last = result[j][result[j].len() - 1];

                // first <-> other_first
                let dist_sq =
                    (first[0] - other_first[0]).powi(2) + (first[1] - other_first[1]).powi(2);
                if dist_sq < tol_sq && dist_sq > 0.0 {
                    result[i][0] = other_first;
                    changed = true;
                }

                // first <-> other_last
                let dist_sq =
                    (first[0] - other_last[0]).powi(2) + (first[1] - other_last[1]).powi(2);
                if dist_sq < tol_sq && dist_sq > 0.0 {
                    result[i][0] = other_last;
                    changed = true;
                }

                // last <-> other_first
                let dist_sq =
                    (last[0] - other_first[0]).powi(2) + (last[1] - other_first[1]).powi(2);
                if dist_sq < tol_sq && dist_sq > 0.0 {
                    result[i][len - 1] = other_first;
                    changed = true;
                }

                // last <-> other_last
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

fn with_sizes(
    mut stats: StageStats,
    input_size: Option<(u32, u32)>,
    output_size: Option<(u32, u32)>,
) -> StageStats {
    stats.input_size = input_size;
    stats.output_size = output_size;
    stats
}

fn with_extra(mut stats: StageStats, extra: serde_json::Value) -> StageStats {
    stats.extra = extra;
    stats
}

fn encode_gray_debug(name: &str, image: &GrayImage) -> StageResult<DebugArtifact> {
    let mut cursor = std::io::Cursor::new(Vec::new());
    DynamicImage::ImageLuma8(image.clone())
        .write_to(&mut cursor, image::ImageFormat::Png)
        .map_err(|e| StageError::ImageProcessing(format!("调试图导出失败: {}", e)))?;

    Ok(DebugArtifact {
        name: name.to_string(),
        mime_type: "image/png".to_string(),
        width: image.width(),
        height: image.height(),
        data_base64: general_purpose::STANDARD.encode(cursor.into_inner()),
    })
}

pub fn detect_raster_kind(image: &GrayImage, quality_score: f64) -> (RasterKind, f64) {
    let (width, height) = image.dimensions();
    let total = (width as f64 * height as f64).max(1.0);
    let pixels = image.as_raw();

    let mut dark = 0usize;
    let mut mid = 0usize;
    let mut bright = 0usize;
    let mut sum = 0.0;
    for &px in pixels {
        sum += px as f64;
        if px < 64 {
            dark += 1;
        } else if px > 192 {
            bright += 1;
        } else {
            mid += 1;
        }
    }

    let mean = sum / total;
    let variance = pixels
        .iter()
        .map(|&px| {
            let d = px as f64 - mean;
            d * d
        })
        .sum::<f64>()
        / total;
    let contrast = variance.sqrt();
    let dark_ratio = dark as f64 / total;
    let bright_ratio = bright as f64 / total;
    let mid_ratio = mid as f64 / total;
    let bimodal_ratio = dark_ratio + bright_ratio;

    let edge_map = crate::algorithms::detect_edges(image);
    let edge_pixels = edge_map.as_raw().iter().filter(|&&px| px > 0).count() as f64;
    let edge_density = edge_pixels / total;

    if contrast < 18.0 || mid_ratio > 0.65 {
        (RasterKind::LowContrast, 0.75)
    } else if edge_density > 0.22 && quality_score < 65.0 {
        (RasterKind::Sketch, 0.70)
    } else if bimodal_ratio > 0.88 && edge_density < 0.18 {
        (RasterKind::LineArt, 0.85)
    } else if quality_score > 70.0 && dark_ratio < 0.25 {
        (RasterKind::Scan, 0.70)
    } else {
        (RasterKind::Photo, 0.65)
    }
}

pub fn strategy_for_kind(kind: RasterKind) -> RasterStrategy {
    match kind {
        RasterKind::LineArt => RasterStrategy::CleanLineArt,
        RasterKind::Scan => RasterStrategy::ScannedPlan,
        RasterKind::Photo => RasterStrategy::PhotoPerspective,
        RasterKind::Sketch => RasterStrategy::HandSketch,
        RasterKind::LowContrast => RasterStrategy::LowContrast,
    }
}

fn quality_recommendations(quality_score: f64, final_polyline_count: usize) -> Vec<String> {
    let mut recommendations = Vec::new();
    if quality_score < 60.0 {
        recommendations.push("图像质量较低，建议提高扫描分辨率或增强对比度".to_string());
    }
    if final_polyline_count == 0 {
        recommendations
            .push("未提取到线段，建议尝试 low_contrast 或 scanned_plan 策略".to_string());
    }
    if recommendations.is_empty() {
        recommendations.push("结果质量正常，可按需要开启语义或 OCR 后处理".to_string());
    }
    recommendations
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{GrayImage, Luma};

    #[test]
    fn test_pipeline_builder_creates_valid_pipeline() {
        let pipeline = VectorizePipelineBuilder::new()
            .max_pixels(1_000_000)
            .build();

        assert_eq!(pipeline.max_pixels, 1_000_000);
    }

    #[test]
    fn test_image_context_creation() {
        let img = GrayImage::from_pixel(100, 100, Luma([255u8]));
        let dynamic = DynamicImage::ImageLuma8(img);

        let context = ImageWithContext::from_dynamic(&dynamic, 1_000_000);
        assert_eq!(context.dimensions(), (100, 100));
        assert_eq!(context.scale_factor, 1.0);
    }

    #[test]
    fn test_stage_names() {
        let preprocess = PreprocessStage::new(PreprocessStageConfig {
            auto_crop_paper: true,
            perspective_correction: true,
            quality_assessment: true,
            adaptive_params: true,
            base_config: VectorizeConfig::default(),
        });
        assert_eq!(preprocess.name(), "Preprocess");

        let threshold = ThresholdStage::new(ThresholdStageConfig {
            use_otsu: true,
            fixed_threshold: 128,
            text_separation: true,
        });
        assert_eq!(threshold.name(), "Threshold");

        let skeletonize = SkeletonizeStage::new(SkeletonizeStageConfig {
            edge_detect: true,
            ..Default::default()
        });
        assert_eq!(skeletonize.name(), "Skeletonize");

        let trace = TraceStage::new(TraceStageConfig {
            min_line_length_px: 10.0,
        });
        assert_eq!(trace.name(), "Trace");

        let fit = FitStage::new(FitStageConfig {
            simplify_tolerance: 2.0,
            snap_tolerance: 2.0,
            gap_filling: true,
            max_angle_dev_deg: 5.0,
        });
        assert_eq!(fit.name(), "Fit");

        let refine = RefineStage::new(RefineStageConfig::default());
        assert_eq!(refine.name(), "Refine");
    }

    #[test]
    fn test_vectorization_strategy_default() {
        assert_eq!(
            VectorizationStrategy::default(),
            VectorizationStrategy::Traditional
        );
    }

    #[test]
    fn test_pipeline_with_strategy() {
        let pipeline = VectorizePipelineBuilder::new()
            .strategy(VectorizationStrategy::Traditional)
            .build();

        assert_eq!(pipeline.strategy(), VectorizationStrategy::Traditional);
    }

    #[test]
    fn test_refine_stage_config_default() {
        let config = RefineStageConfig::default();
        assert!(config.merge_overlapping);
        assert!(config.merge_colinear);
        assert!(config.filter_too_short);
    }
}
