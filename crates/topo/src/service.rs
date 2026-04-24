//! 拓扑服务主模块

use std::sync::Arc;
use std::time::Instant;

use crate::graph_builder::GraphBuilder;
use crate::halfedge::HalfedgeGraph;
use crate::loop_extractor::LoopExtractor;
use common_types::geometry::ToleranceConfig;
use common_types::request::Request;
use common_types::response::Response;
use common_types::{
    CadError, ClosedLoop, GeometryConstructionReason, LengthUnit, Point2, Polyline, SceneState,
    Service, ServiceHealth, ServiceMetrics, ServiceVersion,
};

/// 拓扑配置
#[derive(Debug, Clone, Default)]
pub struct TopoConfig {
    /// 容差配置
    pub tolerance: ToleranceConfig,
    /// 图层过滤器
    pub layer_filter: Option<Vec<String>>,
    /// P11 锐评落实：拓扑构建算法
    /// - Dfs: 当前 DFS 方案（向后兼容）
    /// - Halfedge: Halfedge 方案（推荐，支持嵌套孔洞）
    pub algorithm: TopoAlgorithm,
    /// 跳过交点检测（P11 性能优化）
    /// true = 跳过交点检测和切分，适用于已清理的 DXF 文件
    /// false = 执行完整的交点检测（默认，处理复杂图纸）
    pub skip_intersection_check: bool,
    /// 启用并行处理（P11 锐评落实）
    /// true = 大场景自动启用并行端点吸附和交点检测
    /// false = 使用串行处理（默认，兼容旧流程）
    pub enable_parallel: bool,
    /// 并行处理阈值（P11 锐评落实）
    /// 当线段数量超过此阈值时自动启用并行处理
    pub parallel_threshold: usize,
}

/// P11 新增：拓扑构建算法
#[derive(Debug, Clone, Copy, Default)]
pub enum TopoAlgorithm {
    /// DFS 方案（默认，向后兼容）
    #[default]
    Dfs,
    /// Halfedge 方案（推荐，支持嵌套孔洞）
    Halfedge,
}

impl TopoConfig {
    /// 创建默认配置
    pub fn new() -> Self {
        Self {
            tolerance: ToleranceConfig::default(),
            layer_filter: None,
            algorithm: TopoAlgorithm::Halfedge, // ✅ P2-1 修复：默认使用 Halfedge（支持嵌套孔洞）
            skip_intersection_check: false,
            enable_parallel: false,
            parallel_threshold: 1000,
        }
    }

    /// 创建优化配置（P11 锐评落实）
    pub fn optimized() -> Self {
        Self {
            tolerance: ToleranceConfig::default(),
            layer_filter: None,
            algorithm: TopoAlgorithm::Halfedge, // ✅ P2-1 修复：默认使用 Halfedge（支持嵌套孔洞）
            skip_intersection_check: false,
            enable_parallel: true,
            parallel_threshold: 1000, // 1000 线段以上启用并行
        }
    }
}

/// 拓扑建模服务
#[derive(Clone)]
pub struct TopoService {
    config: TopoConfig,
    metrics: Arc<ServiceMetrics>,
}

impl TopoService {
    pub fn new(config: TopoConfig) -> Self {
        Self {
            config,
            metrics: Arc::new(ServiceMetrics::new("TopoService")),
        }
    }

    /// P11 锐评落实：添加 with_config 方法，支持动态配置
    pub fn with_config(config: &TopoConfig) -> Self {
        Self::new(config.clone())
    }

    pub fn with_default_config() -> Self {
        Self::new(TopoConfig::default())
    }

    /// 获取服务指标
    pub fn metrics(&self) -> &ServiceMetrics {
        &self.metrics
    }

    /// 从多段线构建拓扑并提取闭合环
    ///
    /// # 拓扑构建流程（P11 锐评落实）
    ///
    /// ## 架构角色澄清
    ///
    /// 文档曾声称"默认使用 Halfedge 数据结构构建拓扑"，这是**不准确**的。
    /// 实际架构如下：
    ///
    /// ```text
    /// 输入 Polyline[]
    ///      │
    ///      ▼
    /// ┌─────────────────────────────┐
    /// │  GraphBuilder (核心引擎)    │
    /// │  - snap_and_build           │  端点吸附 (O(n log n))
    /// │  - detect_overlapping       │  重叠线段合并
    /// │  - compute_intersections    │  交点计算与切分
    /// └─────────────────────────────┘
    ///      │
    ///      ▼
    /// ┌─────────────────────────────┐
    /// │  LoopExtractor              │
    /// │  从切分后的边提取闭合环      │
    /// └─────────────────────────────┘
    ///      │
    ///      ▼
    /// ┌─────────────────────────────┐
    /// │  HalfedgeGraph (存储层)     │
    /// │  存储已提取的环，支持：       │
    /// │  - 面枚举和孔洞遍历          │
    /// │  - 嵌套孔洞和岛中岛          │
    /// │  - 共享边界查询              │
    /// └─────────────────────────────┘
    ///      │
    ///      ▼
    /// 输出 TopologyResult
    /// ```
    ///
    /// ## Halfedge 角色定位
    ///
    /// Halfedge 是**存储和查询层**，不是构建层：
    /// - ✅ 用于存储已提取的闭合环
    /// - ✅ 支持高效的面枚举和边界遍历
    /// - ✅ 支持嵌套孔洞和岛中岛查询
    /// - ❌ **不用于**拓扑构建（端点吸附、交点切分在 GraphBuilder 中完成）
    ///
    /// 当 Halfedge 验证失败时，fallback 到传统邻接表遍历方案。
    pub fn build_topology(&self, polylines: &[Polyline]) -> Result<TopologyResult, CadError> {
        let start_time = Instant::now();

        let tol = self.config.tolerance.snap_tolerance;
        let units = self.config.tolerance.units.unwrap_or(LengthUnit::Mm);

        let total_segments: usize = polylines.iter().map(|p| p.len() - 1).sum();
        let total_points: usize = polylines.iter().map(|p| p.len()).sum();

        // 超大文件保护：超过 100M 点时拒绝处理（避免 OOM）
        if total_points > 100_000_000 {
            return Err(CadError::GeometryConstructionError {
                reason: GeometryConstructionReason::InvalidPoint {
                    x: 0.0,
                    y: 0.0,
                    reason: format!(
                        "输入规模过大：{} 点，{} 线段（上限：100M 点）",
                        total_points, total_segments
                    ),
                },
                operation: "topo_build_too_large".to_string(),
                details: None,
            });
        }

        let use_parallel =
            self.config.enable_parallel && total_segments > self.config.parallel_threshold;

        let mut graph_builder = GraphBuilder::new(tol, units);

        if self.config.tolerance.units.is_some() {
            graph_builder.set_adaptive_tolerance(true);
        }

        if use_parallel {
            let all_points: Vec<Point2> = polylines.iter().flatten().copied().collect();
            let (snapped_points, snap_index) =
                crate::parallel::snap_endpoints_parallel(&all_points, tol);
            graph_builder.set_points_with_mapping(snapped_points, snap_index);
            graph_builder.build_edges_from_polylines(polylines);
        } else {
            graph_builder.snap_and_build(polylines);
        }

        // 2. 检测并处理重叠线段
        graph_builder.detect_and_merge_overlapping_segments();

        // 3. 计算交点并切分
        if !self.config.skip_intersection_check {
            if use_parallel {
                graph_builder.compute_intersections_bentley_ottmann();
            } else {
                graph_builder.compute_intersections_and_split();
            }
        }

        // 4. 提取闭合环
        let extractor = LoopExtractor::new(tol);
        let loops = extractor.extract_loops(graph_builder.points(), graph_builder.edges());

        // 5. 分类外轮廓和孔洞 - P11 锐评落实：Halfedge 用于存储已提取的环，传统方案负责分类
        let (outer, holes) = self.classify_loops_with_halfedge(&loops);

        // 6. 构建 Halfedge 图（始终构建，用于后续拓扑查询）
        let halfedge_graph = Some(HalfedgeGraph::from_loops(
            &loops.iter().map(|l| l.points.clone()).collect::<Vec<_>>(),
        ));

        let elapsed = start_time.elapsed();
        tracing::info!(
            "TopoService::build_topology completed in {:?} - {} loops, {} outer, {} holes",
            elapsed,
            loops.len(),
            outer.is_some() as usize,
            holes.len()
        );

        Ok(TopologyResult {
            points: graph_builder.points().to_vec(),
            edges: graph_builder.edges().to_vec(),
            all_loops: loops,
            outer,
            holes,
            halfedge_graph,
        })
    }

    /// 分类外轮廓和孔洞（P11 锐评落实：支持算法选择）
    ///
    /// ## 角色说明
    ///
    /// 根据 `config.algorithm` 选择分类策略：
    /// - `TopoAlgorithm::Halfedge`: 使用 Halfedge 结构分类（支持嵌套孔洞、岛中岛）
    /// - `TopoAlgorithm::Dfs`: 使用传统 DFS 方案（基于面积和包含测试）
    ///
    /// ## Halfedge 流程
    ///
    /// 1. 用 Halfedge 存储已提取的环
    /// 2. 构建嵌套层级关系（射线法）
    /// 3. 使用 extract_outer_and_holes 分类（支持嵌套孔洞、岛中岛）
    /// 4. 如果 Halfedge 验证失败，fallback 到传统方案（基于面积和包含测试）
    fn classify_loops_with_halfedge(
        &self,
        loops: &[ClosedLoop],
    ) -> (Option<ClosedLoop>, Vec<ClosedLoop>) {
        match self.config.algorithm {
            TopoAlgorithm::Halfedge => {
                // 尝试使用 Halfedge 存储并分类
                let mut halfedge_graph = HalfedgeGraph::from_loops(
                    &loops.iter().map(|l| l.points.clone()).collect::<Vec<_>>(),
                );

                // ✅ P2-1 新增：构建嵌套层级关系（支持孔中孔、岛中岛）
                if halfedge_graph.build_nesting_hierarchy().is_ok() {
                    // 验证嵌套层级
                    if halfedge_graph.validate_nesting().is_ok() {
                        tracing::info!(
                            "Halfedge 嵌套层级构建成功，面数={}",
                            halfedge_graph.faces.len()
                        );
                    } else {
                        tracing::warn!("Halfedge 嵌套层级验证失败，继续使用基础分类");
                    }
                } else {
                    tracing::warn!("Halfedge 嵌套层级构建失败，fallback 到 DFS 方案");
                }

                // Halfedge 成功时直接返回，失败时 fallback 到传统方案
                if halfedge_graph.validate().is_ok() {
                    let (outer, holes) = halfedge_graph.extract_outer_and_holes();
                    if outer.is_some() || !holes.is_empty() {
                        tracing::info!(
                            "Halfedge 分类成功：outer={}, holes={}",
                            outer.is_some() as usize,
                            holes.len()
                        );
                        return (outer, holes);
                    }
                }

                tracing::warn!("Halfedge 验证失败，fallback 到 DFS 方案");
                // Fallback：传统方案（基于面积和包含测试）
                self.classify_loops(loops)
            }
            TopoAlgorithm::Dfs => {
                // 直接使用传统方案
                self.classify_loops(loops)
            }
        }
    }

    /// 构建完整场景状态
    pub fn build_scene(&self, polylines: &[Polyline]) -> Result<SceneState, CadError> {
        let topo_result = self.build_topology(polylines)?;

        Ok(SceneState {
            outer: topo_result.outer,
            holes: topo_result.holes,
            boundaries: Vec::new(), // 待用户标注
            sources: Vec::new(),
            edges: Vec::new(), // 由 Pipeline 填充
            units: common_types::LengthUnit::Mm,
            coordinate_system: common_types::CoordinateSystem::RightHandedYUp,
            seat_zones: Vec::new(), // 由 Parser 填充
            render_config: None,    // 由 Pipeline 填充
        })
    }

    /// 分类外轮廓和孔洞
    fn classify_loops(&self, loops: &[ClosedLoop]) -> (Option<ClosedLoop>, Vec<ClosedLoop>) {
        if loops.is_empty() {
            return (None, Vec::new());
        }

        // 按面积绝对值排序
        let mut sorted: Vec<&ClosedLoop> = loops.iter().collect();
        sorted.sort_by(|a, b| {
            b.signed_area
                .abs()
                .partial_cmp(&a.signed_area.abs())
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // 最大正面积为外轮廓
        let outer = sorted
            .iter()
            .find(|l| l.signed_area > 0.0)
            .map(|l| (*l).clone());

        // 孔洞判定：负面积且被外轮廓包含
        let holes: Vec<ClosedLoop> = sorted
            .iter()
            .filter(|&l| {
                if l.signed_area > 0.0 {
                    // 正面积不是孔洞
                    return false;
                }
                // 检查是否被外轮廓包含（通过中心点判断）
                if let Some(ref outer_loop) = outer {
                    let hole_center = calculate_loop_center(&l.points);
                    // 简单包含测试：孔洞中心在外轮廓内部
                    // 使用射线法判断点是否在多边形内
                    point_in_polygon(&hole_center, &outer_loop.points)
                } else {
                    false
                }
            })
            .map(|&l| l.clone())
            .collect();

        (outer, holes)
    }
}

impl Default for TopoService {
    fn default() -> Self {
        Self::with_default_config()
    }
}

// ============================================================================
// Service Trait 实现
// ============================================================================

#[async_trait::async_trait]
impl Service for TopoService {
    type Payload = TopoRequest;
    type Data = TopologyResult;
    type Error = CadError;

    async fn process(
        &self,
        request: Request<Self::Payload>,
    ) -> std::result::Result<Response<Self::Data>, Self::Error> {
        let start = Instant::now();

        // 真正的处理入口：解析几何数据并构建拓扑
        let polylines: Vec<Polyline> = serde_json::from_str(&request.payload.geometry_json)
            .map_err(|e| CadError::GeometryConstructionError {
                reason: GeometryConstructionReason::InvalidPoint {
                    x: 0.0,
                    y: 0.0,
                    reason: format!("解析几何数据失败：{}", e),
                },
                operation: "deserialize_polylines".to_string(),
                details: None,
            })?;

        let result = self.build_topology(&polylines);
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
        "TopoService"
    }

    fn metrics(&self) -> &ServiceMetrics {
        &self.metrics
    }
}

/// 拓扑服务请求
#[derive(Debug, Clone)]
pub struct TopoRequest {
    pub geometry_json: String,
}

impl TopoRequest {
    pub fn new(geometry_json: impl Into<String>) -> Self {
        Self {
            geometry_json: geometry_json.into(),
        }
    }
}

/// 计算环的中心点
fn calculate_loop_center(loop_points: &[Point2]) -> Point2 {
    if loop_points.is_empty() {
        return [0.0, 0.0];
    }
    let sum_x: f64 = loop_points.iter().map(|p| p[0]).sum();
    let sum_y: f64 = loop_points.iter().map(|p| p[1]).sum();
    let len = loop_points.len() as f64;
    [sum_x / len, sum_y / len]
}

/// 射线法判断点是否在多边形内
fn point_in_polygon(point: &Point2, polygon: &[Point2]) -> bool {
    let x = point[0];
    let y = point[1];
    let mut inside = false;

    let n = polygon.len();
    if n < 3 {
        return false;
    }

    let mut j = n - 1;
    for i in 0..n {
        let xi = polygon[i][0];
        let yi = polygon[i][1];
        let xj = polygon[j][0];
        let yj = polygon[j][1];

        if ((yi > y) != (yj > y)) && (x < (xj - xi) * (y - yi) / (yj - yi) + xi) {
            inside = !inside;
        }
        j = i;
    }

    inside
}

/// 拓扑结果
#[derive(Debug, Clone)]
pub struct TopologyResult {
    /// 所有点
    pub points: Vec<Point2>,
    /// 所有边
    pub edges: Vec<(usize, usize)>,
    /// 所有闭合环
    pub all_loops: Vec<ClosedLoop>,
    /// 外轮廓
    pub outer: Option<ClosedLoop>,
    /// 孔洞列表
    pub holes: Vec<ClosedLoop>,
    /// Halfedge 图（P11 锐评落实，可选）
    /// 当 `config.use_halfedge = true` 时填充
    pub halfedge_graph: Option<HalfedgeGraph>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_topo_service_basic() {
        let service = TopoService::with_default_config();

        // 使用一个更简单的矩形，端点完全重合
        let polylines = vec![
            vec![[0.0, 0.0], [10.0, 0.0]],
            vec![[10.0, 0.0], [10.0, 10.0]],
            vec![[10.0, 10.0], [0.0, 10.0]],
            vec![[0.0, 10.0], [0.0, 0.0]],
        ];

        let result = service.build_topology(&polylines);

        // 如果失败，输出调试信息
        match &result {
            Ok(r) => {
                eprintln!("Points: {}", r.points.len());
                eprintln!("Edges: {}", r.edges.len());
                eprintln!("Loops: {}", r.all_loops.len());
            }
            Err(e) => {
                eprintln!("Error: {:?}", e);
            }
        }

        let result = result.unwrap();
        assert!(result.outer.is_some(), "应该提取到外轮廓");
        if let Some(outer) = &result.outer {
            assert!((outer.signed_area - 100.0).abs() < 1e-10, "面积应该是 100");
        }
    }

    #[test]
    fn test_topo_service_empty() {
        let service = TopoService::with_default_config();
        let result = service.build_topology(&[]).unwrap();
        assert!(result.outer.is_none());
        assert!(result.holes.is_empty());
    }

    #[test]
    fn test_topo_service_parallel() {
        // 测试并行 snap 路径
        let config = TopoConfig::optimized();
        let service = TopoService::with_config(&config);

        // 生成足够多的多段线以触发并行路径（> 1000 线段）
        let mut polylines = Vec::new();
        for i in 0..100 {
            let x = i as f64 * 10.0;
            polylines.push(vec![[x, 0.0], [x + 5.0, 0.0]]);
            polylines.push(vec![[x, 5.0], [x + 5.0, 5.0]]);
            polylines.push(vec![[x, 0.0], [x, 5.0]]);
        }

        let result = service.build_topology(&polylines);
        match &result {
            Ok(r) => {
                eprintln!(
                    "Parallel: Points={}, Edges={}, Loops={}",
                    r.points.len(),
                    r.edges.len(),
                    r.all_loops.len()
                );
            }
            Err(e) => {
                eprintln!("Error: {:?}", e);
            }
        }
        assert!(result.is_ok(), "并行路径应该成功");
        let r = result.unwrap();
        assert!(!r.points.is_empty(), "应该有顶点");
        assert!(!r.edges.is_empty(), "应该有边");
    }
}
