//! 编排服务主模块

use crate::api::{create_router_with_cors, ApiState};
use crate::pipeline::ProcessingPipeline;
#[allow(deprecated)]
use acoustic::{
    AcousticError, AcousticInput, AcousticOutput, AcousticService, AcousticServiceConfig,
};
use async_trait::async_trait;
use common_types::error::InternalErrorReason;
use common_types::{CadError, Request, Response, Service, ServiceHealth, ServiceVersion};
use config::CadConfig;
use interact::InteractionService;
use std::fmt::Debug;
use std::net::SocketAddr;
use std::sync::Arc;
use std::time::Instant;
use tokio::net::TcpListener;
use tokio::sync::Mutex;
use tracing_subscriber::{layer::SubscriberExt, util::SubscriberInitExt};

/// 编排服务配置
#[derive(Debug, Clone)]
pub struct OrchestratorConfig {
    /// HTTP 监听地址
    pub listen_addr: String,
    /// 是否启用 API 服务
    pub enable_api: bool,
}

impl Default for OrchestratorConfig {
    fn default() -> Self {
        Self {
            listen_addr: "0.0.0.0:3000".to_string(),
            enable_api: true,
        }
    }
}

/// 编排服务请求
#[derive(Debug, Clone)]
pub struct OrchestratorRequest {
    /// 文件路径
    pub path: std::path::PathBuf,
    /// 是否启动 HTTP 服务
    pub run_http: bool,
}

impl OrchestratorRequest {
    pub fn new(path: impl AsRef<std::path::Path>) -> Self {
        Self {
            path: path.as_ref().to_path_buf(),
            run_http: false,
        }
    }

    pub fn with_http(mut self) -> Self {
        self.run_http = true;
        self
    }
}

/// 编排服务
pub struct OrchestratorService {
    config: OrchestratorConfig,
    cad_config: Option<CadConfig>,
    pipeline: ProcessingPipeline,
    metrics: Arc<common_types::ServiceMetrics>,
    /// 声学服务（无状态设计，使用 Arc 共享引用，无需锁）
    /// (deprecated - 声学功能已停止开发)
    #[allow(deprecated)]
    acoustic_service: Arc<AcousticService>,
}

impl OrchestratorService {
    pub fn new(config: OrchestratorConfig) -> Self {
        Self::with_config(config, None)
    }

    /// 创建编排服务（带可选的 CadConfig）
    ///
    /// 当提供 CadConfig 时，处理流水线将使用自定义配置而非默认配置。
    pub fn with_config(config: OrchestratorConfig, cad_config: Option<CadConfig>) -> Self {
        // 使用 try_init() 避免多次初始化 panic
        let _ = tracing_subscriber::registry()
            .with(
                tracing_subscriber::fmt::layer()
                    .with_target(true)
                    .with_thread_ids(true),
            )
            .try_init();

        let pipeline = match &cad_config {
            Some(cfg) => ProcessingPipeline::new_with_config(cfg),
            None => ProcessingPipeline::new(),
        };

        Self {
            config,
            cad_config,
            pipeline,
            metrics: Arc::new(common_types::ServiceMetrics::new("OrchestratorService")),
            #[allow(deprecated)]
            acoustic_service: Arc::new(AcousticService::new(AcousticServiceConfig::default())),
        }
    }

    /// 获取处理流水线
    pub fn pipeline(&self) -> &ProcessingPipeline {
        &self.pipeline
    }

    /// 计算声学分析
    ///
    /// (deprecated - 声学功能已停止开发，此方法将在未来版本中移除)
    #[deprecated(since = "0.1.0", note = "声学功能已停止开发")]
    #[allow(deprecated)]
    pub async fn calculate_acoustic(
        &self,
        input: AcousticInput,
    ) -> Result<AcousticOutput, CadError> {
        // 直接调用 process_sync 方法（使用 &self，无需锁）
        // 注意：使用底层方法而非 Service trait 的 process()
        let result = tokio::task::spawn_blocking({
            let acoustic_service = Arc::clone(&self.acoustic_service);
            move || acoustic_service.process_sync(input)
        })
        .await
        .unwrap_or_else(|e| {
            Err(AcousticError::CalculationFailed {
                message: format!("任务执行失败：{}", e),
                suggestion: None,
            })
        });

        result.map_err(|e| {
            CadError::internal(InternalErrorReason::ServiceUnavailable {
                service: format!("AcousticService: {}", e),
            })
        })
    }

    /// 运行 HTTP 服务
    pub async fn run(&self) -> Result<(), Box<dyn std::error::Error>> {
        if !self.config.enable_api {
            tracing::info!("API 服务已禁用");
            return Ok(());
        }

        let addr: SocketAddr = self.config.listen_addr.parse()?;
        tracing::info!("启动 HTTP 服务，监听：{}", addr);

        // 初始化交互服务（空状态，等待用户上传文件后动态创建）
        // 实际使用时从解析结果中加载边数据
        let interact_service = InteractionService::new(vec![]);

        // 使用配置的 cad_config 创建流水线（如果有）
        let pipeline = match &self.cad_config {
            Some(cfg) => ProcessingPipeline::new_with_config(cfg),
            None => ProcessingPipeline::new(),
        };

        let app = create_router_with_cors().with_state(ApiState {
            pipeline,
            interact: Arc::new(Mutex::new(interact_service)),
        });

        let listener = TcpListener::bind(addr).await?;
        axum::serve(listener, app).await?;

        Ok(())
    }

    /// 处理文件（不启动 HTTP 服务）
    pub async fn process_file(
        &self,
        path: impl AsRef<std::path::Path>,
    ) -> Result<common_types::SceneState, common_types::CadError> {
        let result = self.pipeline.process_file(path).await?;
        Ok(result.scene)
    }
}

// ProcessingPipeline Clone 已在 pipeline.rs 中实现

#[async_trait]
impl Service for OrchestratorService {
    type Payload = OrchestratorRequest;
    type Data = common_types::SceneState;
    type Error = CadError;

    async fn process(
        &self,
        request: Request<Self::Payload>,
    ) -> Result<Response<Self::Data>, Self::Error> {
        let start = Instant::now();

        if request.payload.run_http {
            // 启动 HTTP 服务（后台运行）
            let config = self.config.clone();
            tokio::spawn(async move {
                let service = OrchestratorService::new(config);
                let _ = service.run().await;
            });
        }

        // 处理文件（调用底层方法，而非 process()）
        let result = self.process_file(&request.payload.path).await;
        let latency = start.elapsed().as_millis() as u64;

        match result {
            Ok(data) => Ok(Response::success(request.id, data, latency)),
            Err(e) => Err(e),
        }
    }

    fn health_check(&self) -> ServiceHealth {
        // 检查 Pipeline 健康状态
        let pipeline_health = self.pipeline.health_check();

        // 构建依赖健康状态
        let mut health = ServiceHealth::healthy(env!("CARGO_PKG_VERSION"))
            .with_uptime(0)
            .with_dependency(common_types::DependencyHealth {
                name: "ProcessingPipeline".to_string(),
                status: pipeline_health.status,
                message: None,
            });

        // 添加 Pipeline 的子服务依赖
        for dep in &pipeline_health.dependencies {
            health = health.with_dependency(dep.clone());
        }

        health
    }

    fn version(&self) -> ServiceVersion {
        ServiceVersion::new(env!("CARGO_PKG_VERSION"))
    }

    fn service_name(&self) -> &'static str {
        "OrchestratorService"
    }

    fn metrics(&self) -> &common_types::ServiceMetrics {
        &self.metrics
    }
}

impl Default for OrchestratorService {
    fn default() -> Self {
        Self::new(OrchestratorConfig::default())
    }
}
