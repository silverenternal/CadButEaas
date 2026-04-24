//! DXF 解析结果缓存系统
//!
//! ## 设计目标
//!
//! 1. **二次打开加速**：相同文件的二次解析速度提升 10x+
//! 2. **内存效率**：使用 LRU 策略自动淘汰旧缓存
//! 3. **线程安全**：支持多线程并发访问
//! 4. **自动失效**：文件修改后自动失效缓存
//!
//! ## 缓存策略
//!
//! | 策略 | 描述 | 适用场景 |
//! |------|------|---------|
//! | LRU | 最近最少使用淘汰 | 通用场景 |
//! | TTL | 超时自动失效 | 频繁修改的文件 |
//! | Size-based | 基于文件大小淘汰 | 内存受限场景 |
//!
//! ## 使用示例
//!
//! ```rust
//! use parser::cache::{DxfCache, CacheConfig};
//! use parser::parser_trait::{DxfParserTrait, SyncDxfParser};
//! use std::sync::Arc;
//!
//! // 创建缓存配置
//! let config = CacheConfig {
//!     max_entries: 100,
//!     ttl: Some(std::time::Duration::from_secs(3600)),
//!     max_memory_mb: Some(500.0),
//! };
//!
//! // 创建缓存解析器
//! let inner_parser = SyncDxfParser::new();
//! let cached_parser = DxfCache::with_config(inner_parser, config);
//!
//! // 第一次解析（从文件读取）
//! let (entities1, report1) = cached_parser.parse_file_with_report("file.dxf")?;
//!
//! // 第二次解析（从缓存读取，速度提升 10x+）
//! let (entities2, report2) = cached_parser.parse_file_with_report("file.dxf")?;
//! ```

use crate::parser_trait::DxfParserTrait;
use crate::{DxfConfig, DxfParseReport};
use common_types::{CadError, RawEntity};
use dashmap::DashMap;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::io::Read;
use std::path::{Path, PathBuf};
use std::sync::{Arc, RwLock};
use std::time::{Duration, Instant};

/// 缓存条目
#[derive(Debug, Clone)]
struct CacheEntry {
    /// 解析后的实体
    entities: Arc<Vec<RawEntity>>,
    /// 解析报告
    report: Arc<DxfParseReport>,
    /// 创建时间
    created_at: Instant,
    /// 最后访问时间
    last_accessed: Instant,
    /// 文件修改时间（用于失效检测）
    file_modified: Option<std::time::SystemTime>,
    /// 缓存条目大小（估算字节数）
    size_bytes: usize,
    /// 文件内容哈希（用于精确检测文件变化）
    file_hash: u64,
    /// 实体哈希（用于增量解析对比）
    entities_hash: u64,
}

impl CacheEntry {
    fn new(
        entities: Vec<RawEntity>,
        report: DxfParseReport,
        file_modified: Option<std::time::SystemTime>,
        file_hash: u64,
    ) -> Self {
        let now = Instant::now();
        // 估算大小：每个实体约 200 字节
        let size_bytes = entities.len() * 200 + report.layer_distribution.len() * 50;

        // 计算实体哈希（用于增量对比）
        let entities_hash = compute_entities_hash(&entities);

        Self {
            entities: Arc::new(entities),
            report: Arc::new(report),
            created_at: now,
            last_accessed: now,
            file_modified,
            size_bytes,
            file_hash,
            entities_hash,
        }
    }

    fn touch(&mut self) {
        self.last_accessed = Instant::now();
    }

    fn is_stale(&self, path: &Path, current_hash: u64) -> bool {
        // 优先使用文件哈希检测变化（更精确）
        if self.file_hash != current_hash && current_hash != 0 {
            return true;
        }

        // 备用：检查文件修改时间
        if let Some(cached_mtime) = self.file_modified {
            if let Ok(metadata) = std::fs::metadata(path) {
                if let Ok(current_mtime) = metadata.modified() {
                    return cached_mtime != current_mtime;
                }
            }
        }
        false
    }

    /// 检查实体是否发生变化（用于增量解析）
    #[allow(dead_code)] // 预留用于未来增量解析功能
    pub fn entities_changed(&self, new_hash: u64) -> bool {
        self.entities_hash != new_hash
    }
}

/// 计算文件内容的快速哈希
fn compute_file_hash(path: &Path) -> Result<u64, CadError> {
    let mut file = std::fs::File::open(path).map_err(|e| {
        CadError::dxf_parse_with_source(path, common_types::DxfParseReason::FileNotFound, e)
    })?;

    let mut hasher = DefaultHasher::new();
    let mut buffer = [0u8; 8192];

    loop {
        let bytes_read = file.read(&mut buffer).map_err(|e| {
            CadError::dxf_parse_with_source(
                path,
                common_types::DxfParseReason::EncodingError("读取文件失败".to_string()),
                e,
            )
        })?;

        if bytes_read == 0 {
            break;
        }

        hasher.write(&buffer[..bytes_read]);
    }

    Ok(hasher.finish())
}

/// 计算实体列表的哈希（用于增量对比）
fn compute_entities_hash(entities: &[RawEntity]) -> u64 {
    let mut hasher = DefaultHasher::new();

    // 哈希实体数量
    entities.len().hash(&mut hasher);

    // 哈希前 100 个实体的关键信息（性能考虑）
    for entity in entities.iter().take(100) {
        // 只哈希实体的基本特征，而非完整数据
        // 注意：f64 不实现 Hash，需要使用 to_bits() 转换为 u64
        match entity {
            RawEntity::Line { start, end, .. } => {
                "Line".hash(&mut hasher);
                start[0].to_bits().hash(&mut hasher);
                start[1].to_bits().hash(&mut hasher);
                end[0].to_bits().hash(&mut hasher);
                end[1].to_bits().hash(&mut hasher);
            }
            RawEntity::Polyline { points, closed, .. } => {
                "Polyline".hash(&mut hasher);
                points.len().hash(&mut hasher);
                closed.hash(&mut hasher);
            }
            RawEntity::Arc {
                center,
                radius,
                start_angle,
                end_angle,
                ..
            } => {
                "Arc".hash(&mut hasher);
                center[0].to_bits().hash(&mut hasher);
                center[1].to_bits().hash(&mut hasher);
                radius.to_bits().hash(&mut hasher);
                start_angle.to_bits().hash(&mut hasher);
                end_angle.to_bits().hash(&mut hasher);
            }
            RawEntity::Circle { center, radius, .. } => {
                "Circle".hash(&mut hasher);
                center[0].to_bits().hash(&mut hasher);
                center[1].to_bits().hash(&mut hasher);
                radius.to_bits().hash(&mut hasher);
            }
            RawEntity::Text {
                content, position, ..
            } => {
                "Text".hash(&mut hasher);
                content.hash(&mut hasher);
                position[0].to_bits().hash(&mut hasher);
                position[1].to_bits().hash(&mut hasher);
            }
            RawEntity::Path { commands, .. } => {
                "Path".hash(&mut hasher);
                commands.len().hash(&mut hasher);
            }
            RawEntity::BlockReference {
                block_name,
                insertion_point,
                ..
            } => {
                "BlockReference".hash(&mut hasher);
                block_name.hash(&mut hasher);
                insertion_point[0].to_bits().hash(&mut hasher);
                insertion_point[1].to_bits().hash(&mut hasher);
            }
            RawEntity::Dimension {
                definition_points,
                measurement,
                ..
            } => {
                "Dimension".hash(&mut hasher);
                definition_points.len().hash(&mut hasher);
                measurement.to_bits().hash(&mut hasher);
            }
            RawEntity::Hatch {
                boundary_paths,
                pattern,
                ..
            } => {
                "Hatch".hash(&mut hasher);
                boundary_paths.len().hash(&mut hasher);
                // 哈希填充图案类型
                match pattern {
                    common_types::HatchPattern::Predefined { name } => {
                        "Predefined".hash(&mut hasher);
                        name.hash(&mut hasher);
                    }
                    common_types::HatchPattern::Custom { pattern_def } => {
                        "Custom".hash(&mut hasher);
                        pattern_def.name.hash(&mut hasher);
                    }
                    common_types::HatchPattern::Solid { .. } => {
                        "Solid".hash(&mut hasher);
                    }
                }
            }
            RawEntity::XRef {
                file_path,
                insertion_point,
                ..
            } => {
                // P1-1: XREF 外部参照支持 - 待完整实现
                // 仅哈希基本特征，不进行完整几何处理
                "XRef".hash(&mut hasher);
                file_path.hash(&mut hasher);
                insertion_point[0].to_bits().hash(&mut hasher);
                insertion_point[1].to_bits().hash(&mut hasher);
            }
            RawEntity::Point { position, .. } => {
                "Point".hash(&mut hasher);
                position[0].to_bits().hash(&mut hasher);
                position[1].to_bits().hash(&mut hasher);
            }
            RawEntity::Image {
                image_def,
                position,
                ..
            } => {
                "Image".hash(&mut hasher);
                image_def.hash(&mut hasher);
                position[0].to_bits().hash(&mut hasher);
                position[1].to_bits().hash(&mut hasher);
            }
            RawEntity::Attribute {
                tag,
                value,
                position,
                ..
            } => {
                "Attribute".hash(&mut hasher);
                tag.hash(&mut hasher);
                value.hash(&mut hasher);
                position[0].to_bits().hash(&mut hasher);
                position[1].to_bits().hash(&mut hasher);
            }
            RawEntity::AttributeDefinition {
                tag,
                default_value,
                position,
                ..
            } => {
                "AttributeDefinition".hash(&mut hasher);
                tag.hash(&mut hasher);
                default_value.hash(&mut hasher);
                position[0].to_bits().hash(&mut hasher);
                position[1].to_bits().hash(&mut hasher);
            }
            RawEntity::Leader {
                points,
                annotation_text,
                ..
            } => {
                "Leader".hash(&mut hasher);
                points.len().hash(&mut hasher);
                if let Some(text) = annotation_text {
                    text.hash(&mut hasher);
                }
            }
            RawEntity::Ray {
                start, direction, ..
            } => {
                "Ray".hash(&mut hasher);
                start[0].to_bits().hash(&mut hasher);
                start[1].to_bits().hash(&mut hasher);
                direction[0].to_bits().hash(&mut hasher);
                direction[1].to_bits().hash(&mut hasher);
            }
            RawEntity::MLine {
                center_line,
                style_name,
                closed,
                ..
            } => {
                "MLine".hash(&mut hasher);
                center_line.len().hash(&mut hasher);
                closed.hash(&mut hasher);
                style_name.hash(&mut hasher);
            }
            RawEntity::Triangle {
                vertices, normal, ..
            } => {
                "Triangle".hash(&mut hasher);
                for v in vertices {
                    v[0].to_bits().hash(&mut hasher);
                    v[1].to_bits().hash(&mut hasher);
                    v[2].to_bits().hash(&mut hasher);
                }
                normal[0].to_bits().hash(&mut hasher);
                normal[1].to_bits().hash(&mut hasher);
                normal[2].to_bits().hash(&mut hasher);
            }
        }
    }

    hasher.finish()
}

/// 缓存配置
#[derive(Debug, Clone)]
pub struct CacheConfig {
    /// 最大缓存条目数
    pub max_entries: usize,
    /// 缓存存活时间（TTL），None 表示永不过期
    pub ttl: Option<Duration>,
    /// 最大内存占用（MB），None 表示无限制
    pub max_memory_mb: Option<f64>,
    /// 是否检查文件修改（启用后文件修改会自动失效缓存）
    pub check_file_modified: bool,
}

impl Default for CacheConfig {
    fn default() -> Self {
        Self {
            max_entries: 50,
            ttl: Some(Duration::from_secs(3600)), // 1 小时
            max_memory_mb: Some(200.0),
            check_file_modified: true,
        }
    }
}

impl CacheConfig {
    /// 创建无限制缓存配置
    pub fn unlimited() -> Self {
        Self {
            max_entries: usize::MAX,
            ttl: None,
            max_memory_mb: None,
            check_file_modified: true,
        }
    }

    /// 创建激进缓存配置（更多条目，更长 TTL）
    pub fn aggressive() -> Self {
        Self {
            max_entries: 200,
            ttl: Some(Duration::from_secs(86400)), // 24 小时
            max_memory_mb: Some(500.0),
            check_file_modified: true,
        }
    }

    /// 创建保守缓存配置（较少条目，更短 TTL）
    pub fn conservative() -> Self {
        Self {
            max_entries: 20,
            ttl: Some(Duration::from_secs(300)), // 5 分钟
            max_memory_mb: Some(50.0),
            check_file_modified: true,
        }
    }
}

/// 缓存统计信息
#[derive(Debug, Clone, Default)]
pub struct CacheStats {
    /// 命中次数
    pub hits: usize,
    /// 未命中次数
    pub misses: usize,
    /// 淘汰次数
    pub evictions: usize,
    /// 当前缓存条目数
    pub entries: usize,
    /// 当前内存占用（MB）
    pub memory_mb: f64,
}

impl CacheStats {
    /// 获取命中率
    pub fn hit_rate(&self) -> f64 {
        let total = self.hits + self.misses;
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

/// 缓存条目信息（公开用于 API 返回）
#[derive(Debug, Clone)]
pub struct CacheEntryInfo {
    /// 实体数量
    pub entities_count: usize,
    /// 创建至今经过的时间（秒）
    pub created_at: u64,
    /// 最后访问至今经过的时间（秒）
    pub last_accessed: u64,
    /// 估算大小（字节）
    pub size_bytes: usize,
    /// 文件哈希
    pub file_hash: u64,
    /// 实体哈希
    pub entities_hash: u64,
}

/// DXF 解析结果缓存
///
/// 包装任意 DxfParserTrait 实现，添加缓存功能
pub struct DxfCache<P: DxfParserTrait> {
    /// 内部解析器
    inner: P,
    /// 缓存存储
    cache: Arc<DashMap<PathBuf, CacheEntry>>,
    /// 缓存配置
    config: CacheConfig,
    /// 缓存统计
    stats: Arc<RwLock<CacheStats>>,
}

impl<P: DxfParserTrait + Clone> Clone for DxfCache<P> {
    fn clone(&self) -> Self {
        Self {
            inner: self.inner.clone(),
            cache: Arc::clone(&self.cache),
            config: self.config.clone(),
            stats: Arc::clone(&self.stats),
        }
    }
}

impl<P: DxfParserTrait> DxfCache<P> {
    /// 创建新的缓存解析器
    pub fn new(inner: P) -> Self {
        Self::with_config(inner, CacheConfig::default())
    }

    /// 使用配置创建缓存解析器
    pub fn with_config(inner: P, config: CacheConfig) -> Self {
        Self {
            inner,
            cache: Arc::new(DashMap::new()),
            config,
            stats: Arc::new(RwLock::new(CacheStats::default())),
        }
    }

    /// 获取内部解析器的不可变引用
    pub fn inner(&self) -> &P {
        &self.inner
    }

    /// 获取内部解析器的可变引用
    pub fn inner_mut(&mut self) -> &mut P {
        &mut self.inner
    }

    /// 清空缓存
    pub fn clear(&self) {
        self.cache.clear();
        self.update_stats(|stats| {
            stats.entries = 0;
            stats.memory_mb = 0.0;
        });
    }

    /// 从缓存中移除指定路径
    pub fn remove(&self, path: impl AsRef<Path>) {
        let path = path.as_ref().to_path_buf();
        self.cache.remove(&path);
        self.recalculate_stats();
    }

    /// 检查路径是否在缓存中
    pub fn contains(&self, path: impl AsRef<Path>) -> bool {
        let path = path.as_ref().to_path_buf();
        self.cache.contains_key(&path)
    }

    /// 获取缓存统计信息
    pub fn stats(&self) -> CacheStats {
        self.stats.read().unwrap().clone()
    }

    // ========================================================================
    // P0-2 增量解析 API
    // ========================================================================

    /// 增量解析文件 - 仅当文件内容变化时才重新解析
    ///
    /// # 返回说明
    /// - `Ok((entities, report, true))` - 文件已变化，已重新解析
    /// - `Ok((entities, report, false))` - 文件未变化，从缓存返回
    pub fn parse_file_incremental(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<(Vec<RawEntity>, DxfParseReport, bool), CadError> {
        let path = path.as_ref();
        let (entities, report) = self.get_or_parse(path)?;

        // 检查是否从缓存返回
        let from_cache = self
            .cache
            .get(&path.to_path_buf())
            .map(|entry| !entry.is_stale(path, entry.value().file_hash))
            .unwrap_or(false);

        Ok((
            Arc::unwrap_or_clone(entities),
            Arc::unwrap_or_clone(report),
            !from_cache,
        ))
    }

    /// 后台预解析 - 在后台线程中解析文件并更新缓存
    ///
    /// 注意：此方法需要解析器实现 Send + Sync + Clone
    ///
    /// # 使用示例
    /// ```rust
    /// let cache = DxfCache::new(parser);
    /// cache.prefetch_async("file.dxf");
    /// // ... 稍后调用 parse_file 时会立即返回（已缓存）
    /// ```
    pub fn prefetch_async(&self, path: impl AsRef<Path> + Send + 'static)
    where
        P: Clone + Send + Sync + 'static,
    {
        let path = path.as_ref().to_path_buf();
        let cache_clone = Arc::clone(&self.cache);
        let inner = self.inner.clone();
        let stats = Arc::clone(&self.stats);

        tokio::spawn(async move {
            // 在后台线程中执行解析
            let path_clone = path.clone();
            let result =
                tokio::task::spawn_blocking(move || inner.parse_file_with_report(&path_clone))
                    .await;

            match result {
                Ok(Ok((entities, report))) => {
                    // 获取文件信息
                    let file_modified = std::fs::metadata(&path)
                        .ok()
                        .and_then(|m| m.modified().ok());
                    let file_hash = compute_file_hash(&path).unwrap_or(0);

                    // 存入缓存
                    let entry = CacheEntry::new(entities, report, file_modified, file_hash);
                    cache_clone.insert(path, entry);

                    // 更新统计
                    let mut stats_guard = stats.write().unwrap();
                    stats_guard.hits += 1;
                }
                Ok(Err(e)) => {
                    tracing::warn!("后台预解析失败：{}", e);
                }
                Err(e) => {
                    tracing::error!("后台任务执行失败：{}", e);
                }
            }
        });
    }

    /// 获取缓存条目的详细信息
    pub fn get_cache_info(&self, path: impl AsRef<Path>) -> Option<CacheEntryInfo> {
        let path = path.as_ref().to_path_buf();
        self.cache.get(&path).map(|entry| CacheEntryInfo {
            entities_count: entry.entities.len(),
            created_at: entry.created_at.elapsed().as_secs(),
            last_accessed: entry.last_accessed.elapsed().as_secs(),
            size_bytes: entry.size_bytes,
            file_hash: entry.file_hash,
            entities_hash: entry.entities_hash,
        })
    }

    /// 强制刷新缓存 - 无论文件是否变化都重新解析
    pub fn refresh_cache(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<(Vec<RawEntity>, DxfParseReport), CadError> {
        let path = path.as_ref();

        // 移除旧缓存
        self.remove(path);

        // 重新解析
        let (entities, report) = self.get_or_parse(path)?;
        Ok((Arc::unwrap_or_clone(entities), Arc::unwrap_or_clone(report)))
    }

    // ========================================================================
    // 内部辅助方法
    // ========================================================================

    /// 更新统计信息
    fn update_stats<F>(&self, f: F)
    where
        F: FnOnce(&mut CacheStats),
    {
        let mut stats = self.stats.write().unwrap();
        f(&mut stats);
    }

    /// 重新计算统计信息
    fn recalculate_stats(&self) {
        let entries = self.cache.len();
        let memory_bytes: usize = self.cache.iter().map(|e| e.value().size_bytes).sum();
        let memory_mb = memory_bytes as f64 / (1024.0 * 1024.0);

        self.update_stats(|stats| {
            stats.entries = entries;
            stats.memory_mb = memory_mb;
        });
    }

    /// 检查是否需要淘汰缓存
    fn needs_eviction(&self) -> bool {
        if self.cache.len() > self.config.max_entries {
            return true;
        }

        if let Some(max_mb) = self.config.max_memory_mb {
            let current_mb: f64 = self
                .cache
                .iter()
                .map(|e| e.value().size_bytes)
                .sum::<usize>() as f64
                / (1024.0 * 1024.0);
            if current_mb > max_mb {
                return true;
            }
        }

        false
    }

    /// 执行缓存淘汰（LRU 策略）
    fn evict(&self) {
        if self.cache.is_empty() {
            return;
        }

        // 收集所有条目并排序
        let mut entries: Vec<_> = self.cache.iter().collect();
        entries.sort_by_key(|e| e.value().last_accessed);

        // 淘汰最旧的条目
        let to_remove = (entries.len() - self.config.max_entries).max(1);
        for entry in entries.iter().take(to_remove) {
            self.cache.remove(entry.key());
        }

        self.update_stats(|stats| {
            stats.evictions += to_remove;
        });
        self.recalculate_stats();
    }

    /// 检查 TTL 是否过期
    fn is_expired(&self, entry: &CacheEntry) -> bool {
        if let Some(ttl) = self.config.ttl {
            entry.created_at.elapsed() > ttl
        } else {
            false
        }
    }

    /// 从缓存获取或解析文件
    fn get_or_parse(
        &self,
        path: &Path,
    ) -> Result<(Arc<Vec<RawEntity>>, Arc<DxfParseReport>), CadError> {
        let path_buf = path.to_path_buf();

        // 计算当前文件哈希
        let current_file_hash = compute_file_hash(path).unwrap_or(0);

        // 检查缓存
        if let Some(mut entry) = self.cache.get_mut(&path_buf) {
            // 检查是否过期或失效（使用文件哈希检测）
            if self.is_expired(&entry)
                || (self.config.check_file_modified && entry.is_stale(path, current_file_hash))
            {
                drop(entry);
                self.cache.remove(&path_buf);
            } else {
                // 缓存命中
                entry.touch();
                self.update_stats(|stats| stats.hits += 1);
                return Ok((Arc::clone(&entry.entities), Arc::clone(&entry.report)));
            }
        }

        // 缓存未命中，执行解析
        self.update_stats(|stats| stats.misses += 1);

        let (entities, report) = self.inner.parse_file_with_report(path)?;

        // 获取文件修改时间
        let file_modified = self
            .config
            .check_file_modified
            .then(|| std::fs::metadata(path).ok()?.modified().ok())
            .flatten();

        // 存入缓存（包含文件哈希）
        let entry = CacheEntry::new(entities, report, file_modified, current_file_hash);
        self.cache.insert(path_buf, entry);

        // 检查是否需要淘汰
        if self.needs_eviction() {
            self.evict();
        }

        self.recalculate_stats();

        // 返回刚存入的数据
        let entry = self.cache.get(&path.to_path_buf()).unwrap();
        Ok((Arc::clone(&entry.entities), Arc::clone(&entry.report)))
    }
}

#[async_trait::async_trait]
impl<P: DxfParserTrait + 'static> DxfParserTrait for DxfCache<P> {
    fn parse_file_with_report(
        &self,
        path: impl AsRef<Path>,
    ) -> Result<(Vec<RawEntity>, DxfParseReport), CadError> {
        let (entities, report) = self.get_or_parse(path.as_ref())?;
        Ok((Arc::unwrap_or_clone(entities), Arc::unwrap_or_clone(report)))
    }

    async fn parse_file_async(
        &self,
        path: impl AsRef<Path> + Send,
    ) -> Result<Vec<RawEntity>, CadError> {
        let (entities, _report) = self.get_or_parse(path.as_ref())?;
        Ok(Arc::unwrap_or_clone(entities))
    }

    async fn parse_file_with_report_async(
        &self,
        path: impl AsRef<Path> + Send,
    ) -> Result<(Vec<RawEntity>, DxfParseReport), CadError> {
        let (entities, report) = self.get_or_parse(path.as_ref())?;
        Ok((Arc::unwrap_or_clone(entities), Arc::unwrap_or_clone(report)))
    }

    fn parse_bytes_with_report(
        &self,
        bytes: &[u8],
    ) -> Result<(Vec<RawEntity>, DxfParseReport), CadError> {
        // 字节解析不缓存（没有路径作为 key）
        self.inner.parse_bytes_with_report(bytes)
    }

    fn config(&self) -> &DxfConfig {
        self.inner.config()
    }

    fn name(&self) -> &'static str {
        "CachedDxfParser"
    }
}

// ============================================================================
// 辅助 Trait：支持克隆的解析器
// ============================================================================

/// 可克隆的解析器 Trait
///
/// 用于需要克隆解析器的场景（如缓存、组合等）
pub trait CloneableParser: DxfParserTrait + Clone {
    fn clone_for_thread(&self) -> Self;
}

impl<T> CloneableParser for T
where
    T: DxfParserTrait + Clone,
{
    fn clone_for_thread(&self) -> Self {
        self.clone()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser_trait::SyncDxfParser;

    #[test]
    fn test_cache_config() {
        let config = CacheConfig::default();
        assert_eq!(config.max_entries, 50);
        assert!(config.ttl.is_some());
        assert!(config.max_memory_mb.is_some());

        let unlimited = CacheConfig::unlimited();
        assert_eq!(unlimited.max_entries, usize::MAX);

        let aggressive = CacheConfig::aggressive();
        assert_eq!(aggressive.max_entries, 200);

        let conservative = CacheConfig::conservative();
        assert_eq!(conservative.max_entries, 20);
    }

    #[test]
    fn test_cache_stats() {
        let mut stats = CacheStats::default();
        stats.hits = 80;
        stats.misses = 20;
        assert!((stats.hit_rate() - 0.8).abs() < 0.01);

        stats.hits = 0;
        stats.misses = 0;
        assert_eq!(stats.hit_rate(), 0.0);
    }

    #[test]
    fn test_cache_creation() {
        let inner = SyncDxfParser::new();
        let cache = DxfCache::new(inner);
        assert_eq!(cache.name(), "CachedDxfParser");
        assert_eq!(cache.stats().entries, 0);
    }

    #[test]
    fn test_cache_with_config() {
        let inner = SyncDxfParser::new();
        let config = CacheConfig::aggressive();
        let cache = DxfCache::with_config(inner, config);
        assert_eq!(cache.stats().entries, 0);
    }
}
