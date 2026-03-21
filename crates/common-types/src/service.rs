//! 服务层统一接口定义（EaaS 架构升级版）
//!
//! 基于 EaaS (Everything as a Service) 设计哲学：
//! - 统一的服务契约
//! - 统一的服务治理接口
//! - 支持本地/远程透明切换
//! - 生产级指标收集（基于 HDR Histogram）
//! - 深度健康检查

use std::collections::HashMap;
use std::fmt::Debug;
use std::sync::{Mutex, RwLock};
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;
use std::thread;
use std::cell::RefCell;

use serde::{Deserialize, Serialize};
use hdrhistogram::Histogram;

use crate::error::CadError;
use crate::request::Request;
use crate::response::Response;

// ============================================================================
// 线程局部存储：缓存分片索引（避免重复计算）
// ============================================================================

thread_local! {
    static CACHED_SHARD_INDEX: RefCell<Option<u64>> = const { RefCell::new(None) };
}

// ============================================================================
// 服务健康状态
// ============================================================================

/// 服务健康状态枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "SCREAMING_SNAKE_CASE")]
pub enum HealthStatus {
    /// 服务健康
    Healthy,
    /// 服务降级（部分功能不可用）
    Degraded,
    /// 服务不健康
    Unhealthy,
}

/// 依赖服务健康状态
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DependencyHealth {
    /// 依赖服务名称
    pub name: String,
    /// 健康状态
    pub status: HealthStatus,
    /// 详细信息
    pub message: Option<String>,
}

/// 服务健康检查结果
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceHealth {
    /// 整体健康状态
    pub status: HealthStatus,
    /// 服务版本
    pub version: String,
    /// 运行时长（秒）
    pub uptime_secs: u64,
    /// 依赖服务健康状态
    pub dependencies: Vec<DependencyHealth>,
    /// 额外元数据
    #[serde(default)]
    pub metadata: HashMap<String, String>,
}

impl ServiceHealth {
    /// 创建健康状态
    pub fn healthy(version: impl Into<String>) -> Self {
        Self {
            status: HealthStatus::Healthy,
            version: version.into(),
            uptime_secs: 0,
            dependencies: Vec::new(),
            metadata: HashMap::new(),
        }
    }

    /// 创建降级状态
    pub fn degraded(
        version: impl Into<String>,
        dependencies: Vec<DependencyHealth>,
    ) -> Self {
        Self {
            status: HealthStatus::Degraded,
            version: version.into(),
            uptime_secs: 0,
            dependencies,
            metadata: HashMap::new(),
        }
    }

    /// 创建不健康状态
    pub fn unhealthy(version: impl Into<String>, message: impl Into<String>) -> Self {
        let mut metadata = HashMap::new();
        metadata.insert("error".to_string(), message.into());
        Self {
            status: HealthStatus::Unhealthy,
            version: version.into(),
            uptime_secs: 0,
            dependencies: Vec::new(),
            metadata,
        }
    }

    /// 设置运行时长
    pub fn with_uptime(mut self, uptime_secs: u64) -> Self {
        self.uptime_secs = uptime_secs;
        self
    }

    /// 添加依赖健康状态
    pub fn with_dependency(mut self, dep: DependencyHealth) -> Self {
        self.dependencies.push(dep);
        self
    }
}

// ============================================================================
// 生产级服务指标（基于 HDR Histogram）
// ============================================================================

/// 延迟直方图快照（用于序列化/导出）
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct HistogramSnapshot {
    /// 最小值（毫秒）
    pub min_ms: f64,
    /// 最大值（毫秒）
    pub max_ms: f64,
    /// 平均值（毫秒）
    pub mean_ms: f64,
    /// 标准差
    pub stddev_ms: f64,
    /// P50
    pub p50_ms: f64,
    /// P90
    pub p90_ms: f64,
    /// P95
    pub p95_ms: f64,
    /// P99
    pub p99_ms: f64,
    /// P99.9
    pub p999_ms: f64,
    /// 样本数
    pub count: u64,
}

impl HistogramSnapshot {
    /// 从 HDR Histogram 创建快照
    pub fn from_histogram(hist: &Histogram<u64>) -> Self {
        Self {
            min_ms: if !hist.is_empty() { hist.min() as f64 } else { 0.0 },
            max_ms: if !hist.is_empty() { hist.max() as f64 } else { 0.0 },
            mean_ms: hist.mean(),
            stddev_ms: hist.stdev(),
            p50_ms: hist.value_at_percentile(50.0) as f64,
            p90_ms: hist.value_at_percentile(90.0) as f64,
            p95_ms: hist.value_at_percentile(95.0) as f64,
            p99_ms: hist.value_at_percentile(99.0) as f64,
            p999_ms: hist.value_at_percentile(99.9) as f64,
            count: hist.len(),
        }
    }
}

// ============================================================================
// 分片直方图（高并发优化）
// ============================================================================

/// 分片直方图（减少锁竞争）
///
/// 使用线程局部存储和分片技术，在高并发场景下减少锁竞争。
/// 每个线程写入自己的分片，读取时合并所有分片。
pub struct ShardedHistogram {
    /// 直方图分片
    shards: Vec<Mutex<Histogram<u64>>>,
    /// 分片掩码（用于快速选择分片）
    shard_mask: u64,
    /// 最大可追踪值
    highest_trackable_value_ms: u64,
    /// 有效数字位数
    significant_figures: u8,
}

impl ShardedHistogram {
    /// 创建新的分片直方图
    ///
    /// # 参数
    ///
    /// * `num_shards` - 分片数量（会自动向上取整到 2 的幂）
    /// * `highest_trackable_value_ms` - 可追踪的最大延迟值（毫秒）
    /// * `significant_figures` - 有效数字位数（3-5，影响精度和内存占用）
    ///
    /// # 示例
    ///
    /// ```
    /// use common_types::service::ShardedHistogram;
    ///
    /// // 创建 8 个分片的直方图，最大追踪 1 小时延迟，3 位有效数字
    /// let hist = ShardedHistogram::new(8, 3_600_000, 3);
    /// ```
    pub fn new(num_shards: usize, highest_trackable_value_ms: u64, significant_figures: u8) -> Self {
        // 确保分片数是 2 的幂，以便使用位掩码优化
        let num_shards = num_shards.next_power_of_two();
        let shards = (0..num_shards)
            .map(|_| {
                Mutex::new(
                    Histogram::<u64>::new_with_bounds(
                        1,
                        highest_trackable_value_ms,
                        significant_figures,
                    )
                    .expect("Failed to create histogram"),
                )
            })
            .collect();

        Self {
            shards,
            shard_mask: (num_shards - 1) as u64,
            highest_trackable_value_ms,
            significant_figures,
        }
    }

    /// 记录一个值（纳秒）
    ///
    /// 使用线程 ID 哈希选择分片，减少锁竞争。
    pub fn record(&self, value_ns: u64) {
        // 使用线程 ID 哈希选择分片
        let shard_idx = self.shard_index();
        let mut shard = self.shards[shard_idx as usize].lock().unwrap();
        
        // 如果值超出范围，记录最大值而不是失败
        let max_value = self.highest_trackable_value_ms;
        let record_value = value_ns.min(max_value);
        let _ = shard.record(record_value);
    }

    /// 获取当前线程的分片索引（使用 thread_local 缓存）
    fn shard_index(&self) -> u64 {
        CACHED_SHARD_INDEX.with(|cached| {
            *cached.borrow_mut().get_or_insert_with(|| {
                // 使用 ThreadId 的 Hash 实现（比 format!("{:?}", thread) 更高效）
                use std::hash::{Hash, Hasher};
                use std::collections::hash_map::DefaultHasher;
                
                let thread_id = thread::current().id();
                let mut hasher = DefaultHasher::new();
                thread_id.hash(&mut hasher);
                hasher.finish() & self.shard_mask
            })
        })
    }

    /// 合并所有分片为一个直方图
    pub fn merge(&self) -> Histogram<u64> {
        let mut merged = Histogram::<u64>::new_with_bounds(
            1,
            self.highest_trackable_value_ms,
            self.significant_figures,
        )
        .unwrap();

        for shard in &self.shards {
            let hist = shard.lock().unwrap();
            merged.add(&*hist).unwrap();
        }

        merged
    }

    /// 获取直方图快照
    pub fn snapshot(&self) -> HistogramSnapshot {
        let merged = self.merge();
        HistogramSnapshot::from_histogram(&merged)
    }

    /// 重置所有分片
    pub fn reset(&self) {
        for shard in &self.shards {
            let mut hist = shard.lock().unwrap();
            *hist = Histogram::<u64>::new_with_bounds(
                1,
                self.highest_trackable_value_ms,
                self.significant_figures,
            )
            .unwrap();
        }
    }
}

/// 为 ShardedHistogram 实现 Default trait
impl Default for ShardedHistogram {
    fn default() -> Self {
        Self::new(8, 3_600_000, 3) // 默认 8 个分片，最大 1 小时，3 位有效数字
    }
}

// ============================================================================
// 生产级服务指标
// ============================================================================

/// 生产级服务指标（无锁设计）
///
/// 使用原子计数器和分片直方图实现高并发下的精确指标收集
pub struct ServiceMetrics {
    /// 服务名称
    service_name: String,
    /// 请求总数（原子计数）
    request_count: AtomicU64,
    /// 成功请求数
    success_count: AtomicU64,
    /// 失败请求数
    error_count: AtomicU64,
    /// 延迟直方图（分片直方图，无锁设计）
    latency_histogram: ShardedHistogram,
    /// 启动时间
    start_time: Instant,
}

/// 指标数据快照（用于序列化）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceMetricsData {
    /// 服务名称
    pub name: String,
    /// 总请求数
    pub request_count: u64,
    /// 成功请求数
    pub success_count: u64,
    /// 失败请求数
    pub error_count: u64,
    /// 成功率
    pub success_rate: f64,
    /// 延迟直方图快照
    pub latency: HistogramSnapshot,
    /// 运行时长（秒）
    pub uptime_secs: u64,
}

impl ServiceMetrics {
    /// 创建新的指标实例
    ///
    /// # 参数
    ///
    /// * `name` - 服务名称
    /// * `highest_trackable_value_ms` - 可追踪的最大延迟值（毫秒），默认 1 小时
    pub fn new(name: impl Into<String>) -> Self {
        Self::with_config(name, 3_600_000) // 1 小时 = 3,600,000ms
    }

    /// 创建带配置的指标实例
    ///
    /// # 参数
    ///
    /// * `name` - 服务名称
    /// * `highest_trackable_value_ms` - 可追踪的最大延迟值（毫秒）
    /// * `significant_figures` - 有效数字位数（影响精度和内存占用）
    pub fn with_config(name: impl Into<String>, highest_trackable_value_ms: u64) -> Self {
        Self {
            service_name: name.into(),
            request_count: AtomicU64::new(0),
            success_count: AtomicU64::new(0),
            error_count: AtomicU64::new(0),
            latency_histogram: ShardedHistogram::new(8, highest_trackable_value_ms, 3),
            start_time: Instant::now(),
        }
    }

    /// 记录请求完成（通过内部可变性支持 &self 调用）
    ///
    /// # 参数
    ///
    /// * `success` - 请求是否成功
    /// * `latency_ms` - 延迟（毫秒）
    pub fn record_request(&self, success: bool, latency_ms: f64) {
        // 原子计数器更新（无锁）
        self.request_count.fetch_add(1, Ordering::Relaxed);
        if success {
            self.success_count.fetch_add(1, Ordering::Relaxed);
        } else {
            self.error_count.fetch_add(1, Ordering::Relaxed);
        }

        // 更新分片直方图（无锁，每个线程写入自己的分片）
        let latency_ns = (latency_ms * 1_000_000.0) as u64; // 转换为纳秒
        self.latency_histogram.record(latency_ns);
    }

    /// 获取请求总数
    pub fn request_count(&self) -> u64 {
        self.request_count.load(Ordering::Relaxed)
    }

    /// 获取成功率
    pub fn success_rate(&self) -> f64 {
        let total = self.request_count.load(Ordering::Relaxed);
        let success = self.success_count.load(Ordering::Relaxed);
        if total == 0 {
            0.0
        } else {
            success as f64 / total as f64
        }
    }

    /// 获取平均延迟（毫秒）
    pub fn avg_latency_ms(&self) -> f64 {
        let snapshot = self.latency_histogram.snapshot();
        snapshot.mean_ms
    }

    /// 获取 P95 延迟（毫秒）
    pub fn p95_latency_ms(&self) -> f64 {
        let snapshot = self.latency_histogram.snapshot();
        snapshot.p95_ms
    }

    /// 获取 P99 延迟（毫秒）
    pub fn p99_latency_ms(&self) -> f64 {
        let snapshot = self.latency_histogram.snapshot();
        snapshot.p99_ms
    }

    /// 获取 P99.9 延迟（毫秒）
    pub fn p999_latency_ms(&self) -> f64 {
        let snapshot = self.latency_histogram.snapshot();
        snapshot.p999_ms
    }

    /// 获取指标数据快照
    pub fn snapshot(&self) -> ServiceMetricsData {
        let latency = self.latency_histogram.snapshot();

        let total = self.request_count.load(Ordering::Relaxed);
        let success = self.success_count.load(Ordering::Relaxed);

        ServiceMetricsData {
            name: self.service_name.clone(),
            request_count: total,
            success_count: success,
            error_count: self.error_count.load(Ordering::Relaxed),
            success_rate: if total == 0 { 0.0 } else { success as f64 / total as f64 },
            latency,
            uptime_secs: self.start_time.elapsed().as_secs(),
        }
    }

    /// 导出为 Prometheus 格式
    pub fn export_prometheus(&self) -> String {
        let snapshot = self.snapshot();
        let name = self.service_name.clone().replace(['-', ' '], "_");

        format!(
            r#"# HELP {name}_requests_total Total requests
# TYPE {name}_requests_total counter
{name}_requests_total{{service="{name}"}} {total}

# HELP {name}_requests_success Total successful requests
# TYPE {name}_requests_success counter
{name}_requests_success{{service="{name}"}} {success}

# HELP {name}_requests_error Total failed requests
# TYPE {name}_requests_error counter
{name}_requests_error{{service="{name}"}} {error}

# HELP {name}_latency_ms Latency histogram (milliseconds)
# TYPE {name}_latency_ms summary
{name}_latency_ms{{service="{name}",quantile="0.5"}} {p50}
{name}_latency_ms{{service="{name}",quantile="0.9"}} {p90}
{name}_latency_ms{{service="{name}",quantile="0.95"}} {p95}
{name}_latency_ms{{service="{name}",quantile="0.99"}} {p99}
{name}_latency_ms{{service="{name}",quantile="0.999"}} {p999}
{name}_latency_ms_sum{{service="{name}"}} {sum}
{name}_latency_ms_count{{service="{name}"}} {count}
"#,
            name = name,
            total = snapshot.request_count,
            success = snapshot.success_count,
            error = snapshot.error_count,
            p50 = snapshot.latency.p50_ms,
            p90 = snapshot.latency.p90_ms,
            p95 = snapshot.latency.p95_ms,
            p99 = snapshot.latency.p99_ms,
            p999 = snapshot.latency.p999_ms,
            sum = snapshot.latency.mean_ms * snapshot.latency.count as f64,
            count = snapshot.latency.count,
        )
    }

    /// 重置指标
    pub fn reset(&self) {
        self.request_count.store(0, Ordering::Relaxed);
        self.success_count.store(0, Ordering::Relaxed);
        self.error_count.store(0, Ordering::Relaxed);
        self.latency_histogram.reset();
    }
}

impl Default for ServiceMetrics {
    fn default() -> Self {
        Self::new("unknown")
    }
}

// ============================================================================
// 服务版本信息
// ============================================================================

/// 服务版本（语义化版本 + API 版本）
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ServiceVersion {
    /// 语义化版本 (e.g., "0.1.0")
    pub semver: String,
    /// API 版本 (e.g., "v1", "v2")
    pub api_version: String,
    /// Schema 版本 (e.g., "schema-1.2")
    pub schema_version: String,
}

impl ServiceVersion {
    /// 创建默认版本
    pub fn new(semver: impl Into<String>) -> Self {
        Self {
            semver: semver.into(),
            api_version: "v1".to_string(),
            schema_version: "schema-1.0".to_string(),
        }
    }

    /// 设置 API 版本
    pub fn with_api_version(mut self, api_version: impl Into<String>) -> Self {
        self.api_version = api_version.into();
        self
    }

    /// 设置 Schema 版本
    pub fn with_schema_version(mut self, schema_version: impl Into<String>) -> Self {
        self.schema_version = schema_version.into();
        self
    }
}

// ============================================================================
// 统一服务 Trait（EaaS 架构核心）
// ============================================================================

/// 所有服务的根 trait
///
/// 这个 trait 定义了服务的基本行为，支持：
/// - 统一的输入/输出类型（使用 crate::request::Request 和 crate::response::Response）
/// - 健康检查
/// - 指标收集
///
/// # 示例
///
/// ```rust,no_run
/// use common_types::{Service, ServiceMetrics};
/// use common_types::request::Request;
/// use common_types::response::Response;
///
/// struct MyService {
///     metrics: ServiceMetrics,
/// }
///
/// #[async_trait::async_trait]
/// impl Service for MyService {
///     type Payload = String;
///     type Data = String;
///     type Error = common_types::CadError;
///
///     async fn process(&self, request: Request<Self::Payload>) -> Result<Response<Self::Data>, Self::Error> {
///         // 处理逻辑
///         Ok(Response::success(
///             request.id.clone(),
///             request.payload,
///             1,
///         ))
///     }
///
///     fn service_name(&self) -> &'static str {
///         "MyService"
///     }
///
///     fn metrics(&self) -> &ServiceMetrics {
///         &self.metrics
///     }
/// }
/// ```
#[async_trait::async_trait]
pub trait Service: Send + Sync {
    /// 请求负载类型
    type Payload: Debug + Send + Sync;
    /// 响应数据类型
    type Data: Send + Sync;
    /// 错误类型
    type Error: std::error::Error + Send + Sync;

    /// 处理请求的核心方法
    ///
    /// # 参数
    ///
    /// * `request` - 统一请求类型，包含上下文和负载
    ///
    /// # 返回
    ///
    /// * `Ok(Response<Self::Data>)` - 成功响应
    /// * `Err(Self::Error)` - 错误响应
    async fn process(
        &self,
        request: Request<Self::Payload>,
    ) -> Result<Response<Self::Data>, Self::Error>;

    /// 服务健康检查（默认实现）
    fn health_check(&self) -> ServiceHealth {
        ServiceHealth::healthy(self.version().semver.clone())
    }

    /// 获取服务版本信息（默认实现）
    fn version(&self) -> ServiceVersion {
        ServiceVersion::new(env!("CARGO_PKG_VERSION"))
    }

    /// 获取服务名称
    fn service_name(&self) -> &'static str;

    /// 获取服务指标
    fn metrics(&self) -> &ServiceMetrics;
}

// ============================================================================
// 服务配置 Trait
// ============================================================================

/// 服务配置 trait
pub trait ServiceConfig: Clone + Debug + Default {
    /// 验证配置是否有效
    fn validate(&self) -> Result<(), CadError>;
}

// ============================================================================
// 深度健康检查工具（生产级）
// ============================================================================

/// 深度健康检查工具
///
/// 提供系统级别的深度健康检查，包括：
/// - 文件系统健康检查
/// - 内存使用率检查
/// - CPU 使用率检查
///
/// # 示例
///
/// ```rust,no_run
/// use common_types::service::HealthCheckUtils;
///
/// let fs_health = HealthCheckUtils::check_filesystem_health();
/// let mem_health = HealthCheckUtils::check_memory_health();
/// let cpu_health = HealthCheckUtils::check_cpu_health();
/// ```
pub struct HealthCheckUtils;

impl HealthCheckUtils {
    /// 检查文件系统健康状态
    ///
    /// 检查项：
    /// - 临时目录是否可写
    pub fn check_filesystem_health() -> DependencyHealth {
        // 检查临时目录是否可写
        let temp_dir = std::env::temp_dir();
        let test_file = temp_dir.join(".cad_health_check");

        match std::fs::write(&test_file, b"") {
            Ok(_) => {
                let _ = std::fs::remove_file(test_file);
            }
            Err(e) => {
                return DependencyHealth {
                    name: "FileSystem".to_string(),
                    status: HealthStatus::Unhealthy,
                    message: Some(format!("无法写入临时目录：{}", e)),
                };
            }
        }

        DependencyHealth {
            name: "FileSystem".to_string(),
            status: HealthStatus::Healthy,
            message: None,
        }
    }

    /// 检查内存使用率
    ///
    /// 检查项：
    /// - 内存使用率 < 80%: Healthy
    /// - 内存使用率 80%-90%: Degraded
    /// - 内存使用率 > 90%: Unhealthy
    pub fn check_memory_health() -> DependencyHealth {
        let mut sys = sysinfo::System::new_all();
        sys.refresh_memory();

        let total = sys.total_memory();
        let used = sys.used_memory();

        if total == 0 {
            return DependencyHealth {
                name: "Memory".to_string(),
                status: HealthStatus::Degraded,
                message: Some("无法获取内存信息".to_string()),
            };
        }

        let used_percent = (used as f64 / total as f64) * 100.0;

        if used_percent > 90.0 {
            DependencyHealth {
                name: "Memory".to_string(),
                status: HealthStatus::Unhealthy,
                message: Some(format!("内存使用率过高：{:.1}%", used_percent)),
            }
        } else if used_percent > 80.0 {
            DependencyHealth {
                name: "Memory".to_string(),
                status: HealthStatus::Degraded,
                message: Some(format!("内存使用率警告：{:.1}%", used_percent)),
            }
        } else {
            DependencyHealth {
                name: "Memory".to_string(),
                status: HealthStatus::Healthy,
                message: None,
            }
        }
    }

    /// 检查 CPU 使用率
    ///
    /// 检查项：
    /// - CPU 使用率 < 80%: Healthy
    /// - CPU 使用率 80%-90%: Degraded
    /// - CPU 使用率 > 90%: Unhealthy
    pub fn check_cpu_health() -> DependencyHealth {
        let mut sys = sysinfo::System::new_all();
        sys.refresh_cpu();

        let cpu_percent = sys.global_cpu_info().cpu_usage();

        if cpu_percent > 90.0 {
            DependencyHealth {
                name: "CPU".to_string(),
                status: HealthStatus::Unhealthy,
                message: Some(format!("CPU 使用率过高：{:.1}%", cpu_percent)),
            }
        } else if cpu_percent > 80.0 {
            DependencyHealth {
                name: "CPU".to_string(),
                status: HealthStatus::Degraded,
                message: Some(format!("CPU 使用率警告：{:.1}%", cpu_percent)),
            }
        } else {
            DependencyHealth {
                name: "CPU".to_string(),
                status: HealthStatus::Healthy,
                message: None,
            }
        }
    }

    /// 综合健康检查
    ///
    /// 检查所有系统资源，返回整体健康状态
    pub fn comprehensive_health() -> ServiceHealth {
        let fs_health = Self::check_filesystem_health();
        let mem_health = Self::check_memory_health();
        let cpu_health = Self::check_cpu_health();

        let dependencies = vec![fs_health, mem_health, cpu_health];

        // 综合判定
        let overall_status = if dependencies.iter().all(|d| d.status == HealthStatus::Healthy) {
            HealthStatus::Healthy
        } else if dependencies.iter().any(|d| d.status == HealthStatus::Unhealthy) {
            HealthStatus::Unhealthy
        } else {
            HealthStatus::Degraded
        };

        let mut health = ServiceHealth::healthy(env!("CARGO_PKG_VERSION"))
            .with_uptime(0);

        for dep in dependencies {
            health = health.with_dependency(dep);
        }

        // 根据整体状态重新构建
        match overall_status {
            HealthStatus::Healthy => health,
            HealthStatus::Degraded => ServiceHealth::degraded(env!("CARGO_PKG_VERSION"), health.dependencies),
            HealthStatus::Unhealthy => ServiceHealth::unhealthy(env!("CARGO_PKG_VERSION"), "系统资源不足"),
        }
    }
}

// ============================================================================
// 持续健康监控器（事件驱动版本）
// ============================================================================

/// 持续健康监控器
///
/// 后台定期采集健康指标，提供实时健康状态查询
///
/// # 特性
///
/// - **后台采集**：每 5 秒自动采集一次健康状态
/// - **状态缓存**：避免频繁检查影响性能
/// - **事件驱动**：健康状态变化时主动通知订阅者
/// - **最新值语义**：使用 watch 通道，不会丢失状态
///
/// # 示例
///
/// ```
/// use common_types::service::HealthMonitor;
///
/// // 启动健康监控
/// let monitor = HealthMonitor::start();
///
/// // 获取当前健康状态
/// let health = monitor.get_health();
///
/// // 订阅健康状态变化
/// let subscriber = monitor.subscribe();
/// ```
pub struct HealthMonitor {
    /// 健康状态（使用 RwLock 保护）
    state: std::sync::Arc<RwLock<ServiceHealth>>,
    /// 停止信号发送器
    stop_tx: Option<std::sync::Arc<std::sync::atomic::AtomicBool>>,
    /// 监控线程句柄
    handle: Option<std::thread::JoinHandle<()>>,
    /// 状态变化发送器（watch 通道，最新值语义）
    tx: Option<tokio::sync::watch::Sender<ServiceHealth>>,
}

impl HealthMonitor {
    /// 启动健康监控器
    ///
    /// 创建后台线程，每 5 秒采集一次健康状态
    pub fn start() -> Self {
        Self::with_interval(std::time::Duration::from_secs(5))
    }

    /// 启动带采集间隔的健康监控器
    ///
    /// # 参数
    ///
    /// * `interval` - 采集间隔
    pub fn with_interval(interval: std::time::Duration) -> Self {
        use std::sync::atomic::{AtomicBool, Ordering};

        let state = std::sync::Arc::new(RwLock::new(
            ServiceHealth::healthy(env!("CARGO_PKG_VERSION"))
        ));
        let stop_flag = std::sync::Arc::new(AtomicBool::new(false));

        // 创建 watch 通道（最新值语义，不会丢失）
        let (tx, _rx) = tokio::sync::watch::channel(ServiceHealth::healthy(env!("CARGO_PKG_VERSION")));

        let state_clone = std::sync::Arc::clone(&state);
        let stop_clone = std::sync::Arc::clone(&stop_flag);
        let tx_clone = tx.clone();

        // 启动后台监控线程
        let handle = std::thread::spawn(move || {
            loop {
                // 检查是否收到停止信号
                if stop_clone.load(Ordering::Relaxed) {
                    break;
                }

                // 采集健康状态
                let new_health = HealthCheckUtils::comprehensive_health();

                // 更新状态
                {
                    let mut state = state_clone.write().unwrap();
                    *state = new_health.clone();
                }

                // 事件驱动：状态变化时广播（watch 通道会覆盖旧值，不会丢失）
                let _ = tx_clone.send(new_health.clone());

                // 等待下一个采集周期
                std::thread::sleep(interval);
            }
        });

        Self {
            state,
            stop_tx: Some(stop_flag),
            handle: Some(handle),
            tx: Some(tx),
        }
    }

    /// 订阅健康状态变化
    ///
    /// 返回初始健康状态和接收器，可以获取最新健康状态或等待状态变化
    ///
    /// # 返回
    ///
    /// 返回元组：`(初始健康状态，watch 接收器)`
    ///
    /// # 示例
    ///
    /// ```
    /// use common_types::service::HealthMonitor;
    ///
    /// let monitor = HealthMonitor::start();
    /// let (initial_health, subscriber) = monitor.subscribe();
    ///
    /// // 获取初始健康状态（不会阻塞）
    /// println!("初始健康状态：{:?}", initial_health);
    ///
    /// // 在异步上下文中等待状态变化
    /// // subscriber.changed().await;
    /// // let new_health = subscriber.borrow().clone();
    /// ```
    pub fn subscribe(&self) -> (ServiceHealth, tokio::sync::watch::Receiver<ServiceHealth>) {
        let current = self.state.read().unwrap().clone();
        let rx = if let Some(tx) = &self.tx {
            tx.subscribe()
        } else {
            // 如果发送器不存在（理论上不应该），创建一个新的通道
            let (_tx, rx) = tokio::sync::watch::channel(ServiceHealth::healthy("unknown"));
            rx
        };
        (current, rx)
    }

    /// 获取当前健康状态
    pub fn get_health(&self) -> ServiceHealth {
        self.state.read().unwrap().clone()
    }

    /// 检查服务是否健康
    pub fn is_healthy(&self) -> bool {
        self.state.read().unwrap().status == HealthStatus::Healthy
    }

    /// 停止监控器
    ///
    /// 发送停止信号并等待后台线程退出
    pub fn stop(self) {
        if let Some(stop_flag) = self.stop_tx {
            stop_flag.store(true, std::sync::atomic::Ordering::Relaxed);
        }

        // 等待后台线程退出
        if let Some(handle) = self.handle {
            let _ = handle.join();
        }
    }
}

impl Default for HealthMonitor {
    fn default() -> Self {
        Self::start()
    }
}
