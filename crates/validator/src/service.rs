//! 验证服务主模块
use std::sync::Arc;
use std::time::Instant;

use crate::checks::{
    check_closure, check_convexity, check_hole_containment, check_micro_features,
    check_self_intersection, Severity, ValidationIssue, ValidationLocation, ValidationReport,
    ValidationSummary,
};
use common_types::request::Request;
use common_types::response::Response;
use common_types::{
    CadError, GeometryConstructionReason, LengthUnit, Point2, RecoverySuggestion, SceneState,
    Service, ServiceHealth, ServiceMetrics, ServiceVersion,
};

/// 单位标定结果
#[derive(Debug, Clone)]
pub struct CalibrationRequirement {
    /// 是否需要标定
    pub required: bool,
    /// 提示信息
    pub message: String,
}

/// 单位标定配置
#[derive(Debug, Clone)]
pub struct CalibrationConfig {
    /// 目标单位
    pub target_unit: LengthUnit,
    /// 最小标定距离
    pub min_calibration_distance: f64,
}

impl Default for CalibrationConfig {
    fn default() -> Self {
        Self {
            target_unit: LengthUnit::Mm,
            min_calibration_distance: 100.0,
        }
    }
}

/// 验证配置
#[derive(Debug, Clone)]
pub struct ValidatorConfig {
    /// 闭合容差 (mm)
    pub closure_tolerance: f64,
    /// 最小边长 (mm)
    pub min_edge_length: f64,
    /// 最小角度 (度)
    pub min_angle_degrees: f64,
}

impl Default for ValidatorConfig {
    fn default() -> Self {
        Self {
            closure_tolerance: 0.5,
            min_edge_length: 10.0,
            min_angle_degrees: 15.0,
        }
    }
}

/// 验证服务
#[derive(Clone)]
pub struct ValidatorService {
    config: ValidatorConfig,
    #[allow(dead_code)]
    calibration_config: CalibrationConfig,
    /// 标定比例因子 (1.0 = 无缩放)
    scale_factor: f64,
    metrics: Arc<ServiceMetrics>,
}

impl ValidatorService {
    pub fn new(config: ValidatorConfig) -> Self {
        Self {
            config,
            calibration_config: CalibrationConfig::default(),
            scale_factor: 1.0,
            metrics: Arc::new(ServiceMetrics::new("ValidatorService")),
        }
    }

    /// P11 锐评落实：添加 with_config 方法，支持动态配置
    pub fn with_config(config: &ValidatorConfig) -> Self {
        Self::new(config.clone())
    }

    pub fn with_default_config() -> Self {
        Self::new(ValidatorConfig::default())
    }

    pub fn with_calibration_config(calibration_config: CalibrationConfig) -> Self {
        Self {
            config: ValidatorConfig::default(),
            calibration_config,
            scale_factor: 1.0,
            metrics: Arc::new(ServiceMetrics::new("ValidatorService")),
        }
    }

    /// 获取服务指标
    pub fn metrics(&self) -> &ServiceMetrics {
        &self.metrics
    }

    /// 检查是否需要单位标定
    pub fn check_unit_calibration(&self, scene: &SceneState) -> CalibrationRequirement {
        if matches!(scene.units, LengthUnit::Unspecified) {
            CalibrationRequirement {
                required: true,
                message: "图纸未指定单位，请标定两点真实距离".to_string(),
            }
        } else {
            CalibrationRequirement {
                required: false,
                message: String::new(),
            }
        }
    }

    /// 应用单位标定
    pub fn apply_calibration(
        &mut self,
        point_a: Point2,
        point_b: Point2,
        real_distance: f64,
    ) -> Result<f64, CadError> {
        let dx = point_b[0] - point_a[0];
        let dy = point_b[1] - point_a[1];
        let current_distance = (dx * dx + dy * dy).sqrt();

        if current_distance < 1e-10 {
            return Err(CadError::ValidationError {
                issue_code: "CALIBRATION_INVALID".to_string(),
                reason: "标定点重合，无法计算比例".to_string(),
            });
        }

        if real_distance <= 0.0 {
            return Err(CadError::ValidationError {
                issue_code: "CALIBRATION_INVALID".to_string(),
                reason: "真实距离必须为正".to_string(),
            });
        }

        let scale = real_distance / current_distance;
        self.scale_factor = scale;

        Ok(scale)
    }

    /// 获取当前缩放比例
    pub fn get_scale_factor(&self) -> f64 {
        self.scale_factor
    }

    /// 应用标定到场景
    pub fn apply_calibration_to_scene(&self, scene: &SceneState) -> SceneState {
        if (self.scale_factor - 1.0).abs() < 1e-10 {
            return scene.clone();
        }

        let mut new_scene = scene.clone();

        if let Some(outer) = &mut new_scene.outer {
            for pt in &mut outer.points {
                pt[0] *= self.scale_factor;
                pt[1] *= self.scale_factor;
            }
            outer.signed_area *= self.scale_factor * self.scale_factor;
        }

        for hole in &mut new_scene.holes {
            for pt in &mut hole.points {
                pt[0] *= self.scale_factor;
                pt[1] *= self.scale_factor;
            }
            hole.signed_area *= self.scale_factor * self.scale_factor;
        }

        for source in &mut new_scene.sources {
            source.position[0] *= self.scale_factor;
            source.position[1] *= self.scale_factor;
            source.position[2] *= self.scale_factor;
        }

        new_scene
    }

    /// 验证场景状态（并行化版本）
    pub fn validate(&self, scene: &SceneState) -> Result<ValidationReport, CadError> {
        let mut issues = Vec::new();
        let mut summary = ValidationSummary::new();
        let mut recovery_suggestions = Vec::new();

        // 轻量检查：先执行 closure
        if let Some(outer) = &scene.outer {
            if let Some(issue) = check_closure(outer, self.config.closure_tolerance) {
                self.categorize_issue(&issue, &mut summary);
                if let Some(suggestion) = self.generate_recovery_suggestion(&issue) {
                    recovery_suggestions.push(suggestion);
                }
                issues.push(issue);
            }
        } else {
            issues.push(ValidationIssue {
                code: "E000".to_string(),
                message: "缺少外轮廓".to_string(),
                severity: Severity::Error,
                location: None,
                suggestion: Some("确保图纸包含一个闭合的外轮廓".to_string()),
            });
            summary.error_count += 1;
        }

        // 并行执行 4 个独立重检查（self_intersection, micro, hole, convexity）
        let outer_ref = scene.outer.clone();
        let holes_ref = scene.holes.clone();
        let min_edge_length = self.config.min_edge_length;
        let min_angle = self.config.min_angle_degrees;

        let (si_issues, (mf_issues, (hc_issues, cx_issues))) = rayon::join(
            || {
                outer_ref
                    .as_ref()
                    .and_then(check_self_intersection)
                    .map(|i| vec![i])
                    .unwrap_or_default()
            },
            || {
                rayon::join(
                    || {
                        outer_ref
                            .as_ref()
                            .map(|outer| check_micro_features(outer, min_edge_length, min_angle))
                            .unwrap_or_default()
                    },
                    || {
                        rayon::join(
                            || {
                                outer_ref
                                    .as_ref()
                                    .map(|outer| check_hole_containment(outer, &holes_ref))
                                    .unwrap_or_default()
                            },
                            || outer_ref.as_ref().map(check_convexity).unwrap_or_default(),
                        )
                    },
                )
            },
        );

        // 合并并行检查结果
        let all_parallel_issues = [si_issues, mf_issues, hc_issues, cx_issues];
        for issue in all_parallel_issues.into_iter().flatten() {
            self.categorize_issue(&issue, &mut summary);
            if let Some(suggestion) = self.generate_recovery_suggestion(&issue) {
                recovery_suggestions.push(suggestion);
            }
            issues.push(issue);
        }

        // 孔洞闭合和自相交检查（依赖 scene.holes 枚举，串行）
        for (i, hole) in scene.holes.iter().enumerate() {
            if let Some(issue) = check_closure(hole, self.config.closure_tolerance) {
                let mut issue = issue.clone();
                issue.location = Some(ValidationLocation {
                    loop_index: Some(i),
                    ..issue.location.unwrap_or_default()
                });
                self.categorize_issue(&issue, &mut summary);
                if let Some(suggestion) = self.generate_recovery_suggestion(&issue) {
                    recovery_suggestions.push(suggestion);
                }
                issues.push(issue);
            }

            if let Some(issue) = check_self_intersection(hole) {
                let mut issue = issue.clone();
                issue.location = Some(ValidationLocation {
                    loop_index: Some(i),
                    ..issue.location.unwrap_or_default()
                });
                self.categorize_issue(&issue, &mut summary);
                if let Some(suggestion) = self.generate_recovery_suggestion(&issue) {
                    recovery_suggestions.push(suggestion);
                }
                issues.push(issue);
            }
        }

        self.check_boundary_completeness(scene, &mut issues, &mut summary);

        if matches!(scene.units, common_types::LengthUnit::Unspecified) {
            issues.push(ValidationIssue {
                code: "W003".to_string(),
                message: "未指定单位".to_string(),
                severity: Severity::Warning,
                location: None,
                suggestion: Some("请标定单位或指定参考尺寸".to_string()),
            });
            summary.warning_count += 1;
        }

        recovery_suggestions.sort_by_key(|s| std::cmp::Reverse(s.priority));

        Ok(ValidationReport {
            passed: summary.error_count == 0,
            issues,
            summary,
            recovery_suggestions,
        })
    }

    fn generate_recovery_suggestion(&self, issue: &ValidationIssue) -> Option<RecoverySuggestion> {
        match issue.code.as_str() {
            "E001" | "E002" => {
                Some(RecoverySuggestion::new(
                    "检测到环未闭合。建议：1) 调整端点位置使其闭合 2) 增大闭合容差 3) 使用 InteractSvc 桥接缺口"
                )
                .with_config_change("validator.closure_tolerance_mm", serde_json::json!(self.config.closure_tolerance * 1.5))
                .with_priority(10))
            }
            "E003" => {
                Some(RecoverySuggestion::new(
                    "检测到自相交多边形（蝴蝶结形状）。建议：1) 在交点处切分线段 2) 重新数字化几何，3) 简化复杂形状"
                )
                .with_priority(9))
            }
            "W001" => {
                Some(RecoverySuggestion::new(
                    "检测到过短线段（可能是噪点）。建议：1) 增大最小边长阈值，2) 运行去噪算法移除短边"
                )
                .with_config_change("validator.min_edge_length_mm", serde_json::json!(2.0))
                .with_priority(5))
            }
            "W002" => {
                Some(RecoverySuggestion::new(
                    "检测到尖锐角度。建议：1) 增大最小角度阈值，2) 在转角处添加圆弧过渡"
                )
                .with_config_change("validator.min_angle_degrees", serde_json::json!(20.0))
                .with_priority(5))
            }
            "W003" => {
                Some(RecoverySuggestion::new(
                    "图纸未指定单位。建议：1) 使用两点标定功能设置真实距离 2) 指定目标单位（mm/m）"
                )
                .with_priority(8))
            }
            _ => None,
        }
    }

    fn categorize_issue(&self, issue: &ValidationIssue, summary: &mut ValidationSummary) {
        match issue.severity {
            Severity::Error => summary.error_count += 1,
            Severity::Warning => summary.warning_count += 1,
            Severity::Info => summary.info_count += 1,
        }
    }

    fn check_boundary_completeness(
        &self,
        scene: &SceneState,
        issues: &mut Vec<ValidationIssue>,
        summary: &mut ValidationSummary,
    ) {
        if scene.outer.is_some() && scene.boundaries.is_empty() {
            issues.push(ValidationIssue {
                code: "I001".to_string(),
                message: "未标注边界语义".to_string(),
                severity: Severity::Info,
                location: None,
                suggestion: Some("为每个边界段添加语义标注（墙体/开口/玻璃等）".to_string()),
            });
            summary.info_count += 1;
        }
    }
}

impl Default for ValidatorService {
    fn default() -> Self {
        Self::with_default_config()
    }
}

// ============================================================================
// Service Trait 实现
// ============================================================================

#[async_trait::async_trait]
impl Service for ValidatorService {
    type Payload = ValidateRequest;
    type Data = ValidationReport;
    type Error = CadError;

    async fn process(
        &self,
        request: Request<Self::Payload>,
    ) -> std::result::Result<Response<Self::Data>, Self::Error> {
        let start = Instant::now();

        // 真正的处理入口：解析场景数据并执行验证
        let scene: SceneState = serde_json::from_str(&request.payload.scene_json).map_err(|e| {
            CadError::GeometryConstructionError {
                reason: GeometryConstructionReason::InvalidPoint {
                    x: 0.0,
                    y: 0.0,
                    reason: format!("解析场景数据失败：{}", e),
                },
                operation: "deserialize_scene".to_string(),
                details: None,
            }
        })?;

        let result = self.validate(&scene);
        let latency = start.elapsed().as_secs_f64() * 1000.0;

        // 记录指标
        self.metrics.record_request(result.is_ok(), latency);

        let data = result?;
        Ok(Response::success(request.id, data, latency as u64))
    }

    fn health_check(&self) -> ServiceHealth {
        ServiceHealth::healthy(self.version().semver.clone())
    }

    fn version(&self) -> ServiceVersion {
        ServiceVersion::new(env!("CARGO_PKG_VERSION"))
    }

    fn service_name(&self) -> &'static str {
        "ValidatorService"
    }

    fn metrics(&self) -> &ServiceMetrics {
        &self.metrics
    }
}

/// 验证服务请求
#[derive(Debug, Clone)]
pub struct ValidateRequest {
    pub scene_json: String,
}

impl ValidateRequest {
    pub fn new(scene_json: impl Into<String>) -> Self {
        Self {
            scene_json: scene_json.into(),
        }
    }
}
