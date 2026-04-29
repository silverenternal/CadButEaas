//! 导出服务主模块

use std::sync::Arc;
use std::time::Instant;

use crate::formats::{ExportFormat, SceneJson};
use common_types::request::Request;
use common_types::response::Response;
use common_types::{
    CadError, InternalErrorReason, IoErrorReason, SceneState, Service, ServiceHealth,
    ServiceMetrics, ServiceVersion,
};
use std::fmt::Debug;

/// 导出配置
#[derive(Debug, Clone)]
pub struct ExportConfig {
    /// 导出格式
    pub format: ExportFormat,
    /// JSON 是否美化输出
    pub pretty_json: bool,
    /// 单位转换（导出时转换为目标单位）
    pub target_units: Option<String>,
}

impl Default for ExportConfig {
    fn default() -> Self {
        Self {
            format: ExportFormat::Json,
            pretty_json: true,
            target_units: None,
        }
    }
}

/// 导出请求
#[derive(Debug, Clone)]
pub struct ExportRequest {
    /// 场景状态
    pub scene: SceneState,
    /// 输出路径（可选）
    pub output_path: Option<String>,
}

impl ExportRequest {
    pub fn new(scene: SceneState) -> Self {
        Self {
            scene,
            output_path: None,
        }
    }

    pub fn with_output_path(mut self, path: impl Into<String>) -> Self {
        self.output_path = Some(path.into());
        self
    }
}

/// 导出服务
#[derive(Clone)]
pub struct ExportService {
    config: ExportConfig,
    /// 标定比例（从 ValidatorService 传入）
    scale_factor: f64,
    metrics: Arc<ServiceMetrics>,
}

impl ExportService {
    pub fn new(config: ExportConfig) -> Self {
        Self {
            config,
            scale_factor: 1.0,
            metrics: Arc::new(ServiceMetrics::new("ExportService")),
        }
    }

    /// P11 锐评落实：添加 with_config 方法，支持动态配置
    pub fn with_config(config: &ExportConfig) -> Self {
        Self::new(config.clone())
    }

    pub fn with_default_config() -> Self {
        Self::new(ExportConfig::default())
    }

    /// 设置标定比例（从 ValidatorService 传入）
    pub fn with_scale_factor(mut self, scale_factor: f64) -> Self {
        self.scale_factor = scale_factor;
        self
    }

    /// 设置标定比例
    pub fn set_scale_factor(&mut self, scale_factor: f64) {
        self.scale_factor = scale_factor;
    }

    /// 获取标定比例
    pub fn get_scale_factor(&self) -> f64 {
        self.scale_factor
    }

    /// 获取服务指标
    pub fn metrics(&self) -> &ServiceMetrics {
        &self.metrics
    }

    /// 应用标定到场景
    fn apply_scale_factor(&self, scene: &SceneState) -> SceneState {
        if (self.scale_factor - 1.0).abs() < 1e-10 {
            return scene.clone();
        }

        let mut new_scene = scene.clone();

        // 缩放外轮廓
        if let Some(outer) = &mut new_scene.outer {
            for pt in &mut outer.points {
                pt[0] *= self.scale_factor;
                pt[1] *= self.scale_factor;
            }
            outer.signed_area *= self.scale_factor * self.scale_factor;
        }

        // 缩放孔洞
        for hole in &mut new_scene.holes {
            for pt in &mut hole.points {
                pt[0] *= self.scale_factor;
                pt[1] *= self.scale_factor;
            }
            hole.signed_area *= self.scale_factor * self.scale_factor;
        }

        // 缩放声源位置
        for source in &mut new_scene.sources {
            source.position[0] *= self.scale_factor;
            source.position[1] *= self.scale_factor;
            source.position[2] *= self.scale_factor;
        }

        new_scene
    }

    /// 导出场景为字节
    pub fn export(&self, scene: &SceneState) -> Result<ExportResult, CadError> {
        // 应用标定比例
        let scaled_scene = self.apply_scale_factor(scene);
        let scene_json = SceneJson::from_scene_state(&scaled_scene);

        let bytes = match self.config.format {
            ExportFormat::Json => {
                scene_json
                    .to_json_bytes(self.config.pretty_json)
                    .map_err(|e| CadError::InternalError {
                        reason: InternalErrorReason::Panic {
                            message: format!("JSON 序列化失败：{}", e),
                        },
                        location: None,
                    })?
            }
            ExportFormat::Binary => {
                scene_json
                    .to_binary_bytes()
                    .map_err(|e| CadError::InternalError {
                        reason: InternalErrorReason::Panic {
                            message: format!("二进制序列化失败：{}", e),
                        },
                        location: None,
                    })?
            }
        };

        let extension = match self.config.format {
            ExportFormat::Json => "json",
            ExportFormat::Binary => "bin",
        };

        Ok(ExportResult {
            bytes,
            format: self.config.format,
            extension: extension.to_string(),
        })
    }

    /// 导出为 JSON 字符串
    pub fn export_to_json_string(&self, scene: &SceneState) -> Result<String, CadError> {
        let result = self.export(scene)?;
        String::from_utf8(result.bytes).map_err(|e| CadError::InternalError {
            reason: InternalErrorReason::Panic {
                message: format!("UTF-8 转换失败：{}", e),
            },
            location: None,
        })
    }

    /// 导出到文件
    pub fn export_to_file(
        &self,
        scene: &SceneState,
        path: impl AsRef<std::path::Path>,
    ) -> Result<(), CadError> {
        let result = self.export(scene)?;
        let path_buf = path.as_ref().to_path_buf();
        std::fs::write(path.as_ref(), &result.bytes)
            .map_err(|e| CadError::io_path(&path_buf, IoErrorReason::WriteFailed, e))?;
        Ok(())
    }
}

/// 导出结果
#[derive(Debug)]
pub struct ExportResult {
    /// 字节数据
    pub bytes: Vec<u8>,
    /// 导出格式
    pub format: ExportFormat,
    /// 文件扩展名
    pub extension: String,
}

// P11 锐评 v3.0 修复：局部导入 async_trait，避免文件顶部全局导入
#[async_trait::async_trait]
impl Service for ExportService {
    type Payload = ExportRequest;
    type Data = ExportResult;
    type Error = CadError;

    async fn process(
        &self,
        request: Request<Self::Payload>,
    ) -> Result<Response<Self::Data>, Self::Error> {
        let start = Instant::now();

        // 如果需要导出到文件
        let result = if let Some(path) = request.payload.output_path {
            let path_buf = std::path::PathBuf::from(path);
            self.export_to_file(&request.payload.scene, &path_buf)?;
            self.export(&request.payload.scene)
        } else {
            self.export(&request.payload.scene)
        };

        let latency = start.elapsed().as_secs_f64() * 1000.0;

        // 记录指标
        self.metrics.record_request(result.is_ok(), latency);

        let data = result?;
        Ok(Response::success(request.id, data, latency as u64))
    }

    fn health_check(&self) -> ServiceHealth {
        ServiceHealth::healthy(env!("CARGO_PKG_VERSION"))
    }

    fn version(&self) -> ServiceVersion {
        ServiceVersion::new(env!("CARGO_PKG_VERSION"))
    }

    fn service_name(&self) -> &'static str {
        "ExportService"
    }

    fn metrics(&self) -> &ServiceMetrics {
        &self.metrics
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common_types::{ClosedLoop, CoordinateSystem, LengthUnit, SceneState};

    #[test]
    fn test_export_service_json() {
        let config = ExportConfig {
            format: ExportFormat::Json,
            pretty_json: false,
            ..Default::default()
        };
        let service = ExportService::new(config);

        let scene = SceneState {
            outer: Some(ClosedLoop {
                points: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                signed_area: 100.0,
            }),
            holes: vec![],
            boundaries: vec![],
            sources: vec![],
            edges: vec![],
            raster_metadata: None,
            units: LengthUnit::Mm,
            coordinate_system: CoordinateSystem::RightHandedYUp,
            seat_zones: vec![],
            render_config: None,
        };

        let result = service.export(&scene).unwrap();
        assert_eq!(result.extension, "json");

        let json_str = String::from_utf8_lossy(&result.bytes);
        assert!(json_str.contains("schema_version"));
        assert!(json_str.contains("outer"));
    }

    #[test]
    fn test_export_service_binary() {
        let config = ExportConfig {
            format: ExportFormat::Binary,
            ..Default::default()
        };
        let service = ExportService::new(config);

        let scene = SceneState {
            outer: Some(ClosedLoop {
                points: vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
                signed_area: 100.0,
            }),
            holes: vec![],
            boundaries: vec![],
            sources: vec![],
            edges: vec![],
            raster_metadata: None,
            units: LengthUnit::Mm,
            coordinate_system: CoordinateSystem::RightHandedYUp,
            seat_zones: vec![],
            render_config: None,
        };

        let result = service.export(&scene).unwrap();
        assert_eq!(result.extension, "bin");
        assert!(!result.bytes.is_empty());
    }
}
