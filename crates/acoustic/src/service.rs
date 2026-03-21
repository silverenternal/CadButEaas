//! AcousticService 实现
//!
//! 实现 Service trait，提供声学分析服务
//!
//! # 设计说明
//!
//! AcousticService 使用无状态设计，所有内部组件（SelectionCalculator、ReverberationCalculator、ComparativeAnalyzer）
//! 均为无状态结构。process_sync 方法使用 &self，支持多线程并发调用。
//!
//! ## 内部可变性
//!
//! - ServiceMetrics 内部使用 Arc 实现指标收集
//! - ComparativeAnalyzer 使用 RwLock 包装，支持 &self 下的可变调用（用于 R*-tree 缓存）

use std::time::Instant;
use std::sync::RwLock;
use async_trait::async_trait;
use tracing::{info, instrument};

use common_types::service::{Service, ServiceHealth, ServiceMetrics, ServiceVersion};
use common_types::request::Request;
use common_types::response::Response;
use common_types::acoustic::{
    AcousticInput, AcousticOutput, AcousticRequest, AcousticResult,
    AcousticMetrics, AcousticError, ReverberationFormula,
};
use common_types::scene::SceneState;

use crate::selection::SelectionCalculator;
use crate::reverberation::ReverberationCalculator;
use crate::comparative::ComparativeAnalyzer;

/// AcousticService 配置
#[derive(Debug, Clone)]
pub struct AcousticServiceConfig {
    /// 默认房间高度 (m)
    pub default_room_height: f64,
    /// 默认混响公式
    pub default_formula: ReverberationFormula,
}

impl Default for AcousticServiceConfig {
    fn default() -> Self {
        Self {
            default_room_height: 3.0,
            default_formula: ReverberationFormula::Sabine,
        }
    }
}

/// 声学分析服务
///
/// 提供以下功能：
/// - 选区材料统计
/// - 选区等效吸声面积计算
/// - 房间级混响时间计算
/// - 多区域对比分析
///
/// # 设计说明
///
/// 本服务采用无状态设计：
/// - SelectionCalculator、ReverberationCalculator、ComparativeAnalyzer 均为无状态组件
/// - process_sync 使用 &self，支持多线程并发调用
/// - ServiceMetrics 内部使用 Arc，保证指标正确累积
/// - ComparativeAnalyzer 使用 RwLock 包装，支持 &self 下的可变调用（用于 R*-tree 缓存优化）
pub struct AcousticService {
    config: AcousticServiceConfig,
    metrics: ServiceMetrics,
    calculator: SelectionCalculator,
    reverberation_calc: ReverberationCalculator,
    comparative: RwLock<ComparativeAnalyzer>,
}

impl AcousticService {
    /// 创建新的 AcousticService
    pub fn new(config: AcousticServiceConfig) -> Self {
        Self {
            config,
            metrics: ServiceMetrics::new("AcousticService"),
            calculator: SelectionCalculator::new(),
            reverberation_calc: ReverberationCalculator::new(),
            comparative: RwLock::new(ComparativeAnalyzer::new()),
        }
    }

    /// 获取配置
    pub fn config(&self) -> &AcousticServiceConfig {
        &self.config
    }

    /// 处理声学分析请求
    ///
    /// # 注意
    ///
    /// 本方法使用 &self，支持并发调用。指标通过 ServiceMetrics 内部的 Arc 进行累积。
    #[instrument(skip(self, input), fields(scene_edges = input.scene.edges.len()))]
    pub fn process_sync(&self, input: AcousticInput) -> Result<AcousticOutput, AcousticError> {
        let start = Instant::now();

        let result = match input.request {
            AcousticRequest::SelectionMaterialStats { boundary, mode } => {
                info!("执行选区材料统计");
                let stats = self.calculator.calculate(&input.scene, boundary, mode)?;
                AcousticResult::SelectionMaterialStats(stats)
            }
            AcousticRequest::RoomReverberation { room_id, formula, room_height } => {
                info!("执行房间混响时间计算，room_id={:?}", room_id);
                let formula = formula.unwrap_or(self.config.default_formula);
                let height = room_height.unwrap_or(self.config.default_room_height);
                let rev = self.reverberation_calc.calculate(&input.scene, room_id, formula, height)?;
                AcousticResult::RoomReverberation(rev)
            }
            AcousticRequest::ComparativeAnalysis { selections, metrics } => {
                info!("执行多区域对比分析，regions={}", selections.len());
                let comparison = self.comparative
                    .write()
                    .map_err(|e| AcousticError::CalculationFailed {
                        message: format!("锁中毒：{}", e),
                        suggestion: Some(common_types::acoustic::AcousticRecoverySuggestion::new(
                            "重试请求，锁中毒通常是暂时性的"
                        )),
                    })?
                    .analyze(&input.scene, selections, metrics)?;
                AcousticResult::ComparativeAnalysis(comparison)
            }
        };

        let computation_time = start.elapsed();
        let computation_time_ms = computation_time.as_secs_f64() * 1000.0;

        // 记录指标（ServiceMetrics 内部使用 Arc，保证正确累积）
        self.metrics.record_request(true, computation_time_ms);

        Ok(AcousticOutput {
            result,
            computation_time,
            metrics: AcousticMetrics {
                surface_count: input.scene.edges.len(),
                computation_time_ms,
            },
        })
    }

    /// 计算选区材料统计（便捷方法）
    pub fn calculate_selection_material_stats(
        &self,
        scene: &SceneState,
        boundary: common_types::acoustic::SelectionBoundary,
        mode: common_types::acoustic::SelectionMode,
    ) -> Result<common_types::acoustic::SelectionMaterialStatsResult, AcousticError> {
        self.calculator.calculate(scene, boundary, mode)
    }

    /// 计算房间混响时间（便捷方法）
    pub fn calculate_room_reverberation(
        &self,
        scene: &SceneState,
        room_id: common_types::scene::SurfaceId,
        formula: ReverberationFormula,
        room_height: f64,
    ) -> Result<common_types::acoustic::ReverberationResult, AcousticError> {
        self.reverberation_calc.calculate(scene, room_id, formula, room_height)
    }

    /// 多区域对比分析（便捷方法）
    pub fn calculate_comparative_analysis(
        &self,
        scene: &SceneState,
        selections: Vec<common_types::acoustic::NamedSelection>,
        metrics: Vec<common_types::acoustic::ComparisonMetric>,
    ) -> Result<common_types::acoustic::ComparativeAnalysisResult, AcousticError> {
        self.comparative
            .write()
            .map_err(|e| AcousticError::CalculationFailed {
                message: format!("锁中毒：{}", e),
                suggestion: Some(common_types::acoustic::AcousticRecoverySuggestion::new(
                    "重试请求，锁中毒通常是暂时性的"
                )),
            })?
            .analyze(scene, selections, metrics)
    }
}

#[async_trait]
impl Service for AcousticService {
    type Payload = AcousticInput;
    type Data = AcousticOutput;
    type Error = AcousticError;

    async fn process(&self, request: Request<Self::Payload>) -> Result<Response<Self::Data>, Self::Error> {
        let start = Instant::now();

        // 直接使用 &self 调用 process_sync，无需创建临时实例
        // ✅ 指标正确累积到 self.metrics
        // ✅ 支持并发调用
        let result = self.process_sync(request.payload);
        let latency = start.elapsed().as_secs_f64() * 1000.0;

        // 记录指标（ServiceMetrics 内部使用 Arc，保证正确累积）
        self.metrics.record_request(result.is_ok(), latency);

        let data = result?;
        Ok(Response::success(
            request.id,
            data,
            latency as u64,
        ))
    }

    fn health_check(&self) -> ServiceHealth {
        ServiceHealth::healthy(env!("CARGO_PKG_VERSION"))
    }

    fn version(&self) -> ServiceVersion {
        ServiceVersion::new(env!("CARGO_PKG_VERSION"))
    }

    fn service_name(&self) -> &'static str {
        "AcousticService"
    }

    fn metrics(&self) -> &ServiceMetrics {
        &self.metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common_types::service::HealthStatus;
    use common_types::scene::RawEdge;

    #[allow(dead_code)]
    fn create_test_scene() -> SceneState {
        let mut scene = SceneState::default();

        // 添加测试边
        for i in 0..10 {
            scene.edges.push(RawEdge {
                id: i,
                start: [i as f64 * 1000.0, 0.0],
                end: [(i + 1) as f64 * 1000.0, 0.0],
                layer: Some("WALL".to_string()),
                color_index: None,
            });
        }

        scene
    }

    #[test]
    fn test_acoustic_service_creation() {
        let config = AcousticServiceConfig::default();
        let service = AcousticService::new(config);
        assert_eq!(service.config().default_room_height, 3.0);
        assert_eq!(service.config().default_formula, ReverberationFormula::Sabine);
    }

    #[test]
    fn test_acoustic_service_config() {
        let config = AcousticServiceConfig {
            default_room_height: 4.0,
            default_formula: ReverberationFormula::Eyring,
        };
        let service = AcousticService::new(config);
        assert_eq!(service.config().default_room_height, 4.0);
        assert_eq!(service.config().default_formula, ReverberationFormula::Eyring);
    }

    #[test]
    fn test_acoustic_service_health() {
        let service = AcousticService::new(AcousticServiceConfig::default());
        let health = service.health_check();
        assert_eq!(health.status, HealthStatus::Healthy);
    }

    #[test]
    fn test_acoustic_service_version() {
        let service = AcousticService::new(AcousticServiceConfig::default());
        let version = service.version();
        assert!(!version.semver.is_empty());
    }

    #[test]
    fn test_acoustic_service_name() {
        let service = AcousticService::new(AcousticServiceConfig::default());
        assert_eq!(service.service_name(), "AcousticService");
    }
}
