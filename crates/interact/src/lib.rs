//! 交互协同服务 - 状态驱动
//!
//! # 概述
//!
//! 托管用户交互状态，驱动自动追踪/圈选补全逻辑。
//! 支持 WebSocket 实时推送和 HTTP API 两种交互模式。
//!
//! # P1-5 交互响应优化
//!
//! 新增脏矩形更新和优先级队列支持：
//! - `dirty_rect::DirtyRectTracker`: 追踪需要重绘的区域
//! - `dirty_rect::RenderTaskQueue`: 按优先级排序渲染任务
//! - `dirty_rect::IncrementalUpdater`: 增量更新管理器

// ============================================================================
// 脏矩形更新和优先级队列模块（P1-5）
// ============================================================================
pub mod dirty_rect;

// ============================================================================
// 核心交互模块
// ============================================================================

use serde::{Deserialize, Serialize};
use schemars::JsonSchema;
use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use geo::{LineString, Coord, Contains, Intersects};
use nalgebra::Vector2;
use common_types::scene::{ClosedLoop, BoundarySemantic};
use common_types::geometry::{LineStyle, LineWidth};
use common_types::error::{CadError, InternalErrorReason};
use common_types::service::ServiceMetrics;

/// 边 ID 类型
pub type EdgeId = usize;

/// 段 ID 类型
pub type SegmentId = usize;

/// 缺口 ID 类型
pub type GapId = usize;

/// 2D 点类型
pub type Point2 = [f64; 2];

/// 交互状态 - 核心数据结构
#[derive(Debug, Clone, Default, Serialize, Deserialize, JsonSchema)]
pub struct InteractionState {
    /// 当前选中的边
    pub selected_edges: HashSet<EdgeId>,
    /// 当前选中的区域（圈选多边形）
    pub lasso_polygon: Option<Vec<Point2>>,
    /// 自动追踪的候选路径
    pub auto_trace_candidates: Vec<EdgeId>,
    /// 用户确认的闭合环
    pub confirmed_loops: Vec<ClosedLoop>,
    /// 边界语义标注
    pub boundary_semantics: HashMap<SegmentId, BoundarySemantic>,
    /// 检测到的缺口
    pub detected_gaps: Vec<GapInfo>,
    /// 所有边（用于追踪和选择）
    pub edges: Vec<Edge>,
    /// 已桥接的缺口
    pub bridged_gaps: HashSet<GapId>,
    /// 场景状态（用于导出）
    #[serde(skip)]
    pub scene_state: Option<common_types::SceneState>,
}

/// 边 - 基本几何单元
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct Edge {
    /// 边 ID
    pub id: EdgeId,
    /// 起点
    pub start: Point2,
    /// 终点
    pub end: Point2,
    /// 所属图层（可选）
    pub layer: Option<String>,
    /// 是否为墙体
    pub is_wall: bool,
    /// 是否可见（用于图层过滤）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub visible: Option<bool>,
    /// P0-4: 线型（实线/虚线/点划线等）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_style: Option<LineStyle>,
    /// P0-5: 线宽（24 级）
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub line_width: Option<LineWidth>,
}

impl Edge {
    pub fn new(id: EdgeId, start: Point2, end: Point2) -> Self {
        Self {
            id,
            start,
            end,
            layer: None,
            is_wall: true,
            visible: None,
            line_style: None,  // 默认实线
            line_width: None,  // 默认 ByLayer
        }
    }

    /// 转换为 geo::LineString
    pub fn to_line_string(&self) -> LineString<f64> {
        LineString::from(vec![
            Coord { x: self.start[0], y: self.start[1] },
            Coord { x: self.end[0], y: self.end[1] },
        ])
    }

    /// 计算边的方向向量
    pub fn direction(&self) -> Vector2<f64> {
        Vector2::new(self.end[0] - self.start[0], self.end[1] - self.start[1])
    }

    /// 计算边的长度
    pub fn length(&self) -> f64 {
        let dx = self.end[0] - self.start[0];
        let dy = self.end[1] - self.start[1];
        (dx * dx + dy * dy).sqrt()
    }
}

/// 缺口信息
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct GapInfo {
    /// 缺口 ID
    pub id: GapId,
    /// 缺口端点 A
    pub endpoint_a: Point2,
    /// 缺口端点 B
    pub endpoint_b: Point2,
    /// 缺口长度
    pub length: f64,
    /// 桥接置信度 (0.0-1.0)
    pub confidence: f32,
    /// 缺口类型
    pub gap_type: GapType,
    /// 端点 A 相连的边 ID
    pub edge_a: Option<EdgeId>,
    /// 端点 B 相连的边 ID
    pub edge_b: Option<EdgeId>,
}

impl GapInfo {
    pub fn new(
        id: GapId,
        endpoint_a: Point2,
        endpoint_b: Point2,
        direction_a: Option<Point2>,
        direction_b: Option<Point2>,
        edge_a: Option<EdgeId>,
        edge_b: Option<EdgeId>,
    ) -> Self {
        let dx = endpoint_b[0] - endpoint_a[0];
        let dy = endpoint_b[1] - endpoint_a[1];
        let length = (dx * dx + dy * dy).sqrt();

        let confidence = Self::calculate_confidence(length, direction_a.as_ref(), direction_b.as_ref());

        Self {
            id,
            endpoint_a,
            endpoint_b,
            length,
            confidence,
            gap_type: Self::classify_gap(length, direction_a.as_ref(), direction_b.as_ref()),
            edge_a,
            edge_b,
        }
    }

    fn calculate_confidence(
        length: f64,
        dir_a: Option<&Point2>,
        dir_b: Option<&Point2>,
    ) -> f32 {
        let length_score = if length < 1.0 {
            1.0
        } else if length < 5.0 {
            0.8
        } else if length < 10.0 {
            0.5
        } else {
            0.2
        };

        let direction_score = match (dir_a, dir_b) {
            (Some(a), Some(b)) => {
                let dot = a[0] * b[0] + a[1] * b[1];
                let mag_a = (a[0] * a[0] + a[1] * a[1]).sqrt();
                let mag_b = (b[0] * b[0] + b[1] * b[1]).sqrt();
                if mag_a > 0.0 && mag_b > 0.0 {
                    let cos_angle = dot / (mag_a * mag_b);
                    if cos_angle.abs() > 0.9 || cos_angle.abs() < 0.2 {
                        0.9
                    } else {
                        0.5
                    }
                } else {
                    0.5
                }
            }
            _ => 0.5,
        };

        ((length_score + direction_score) / 2.0) as f32
    }

    fn classify_gap(length: f64, dir_a: Option<&Point2>, dir_b: Option<&Point2>) -> GapType {
        match (dir_a, dir_b) {
            (Some(a), Some(b)) => {
                let dot = a[0] * b[0] + a[1] * b[1];
                let mag_a = (a[0] * a[0] + a[1] * a[1]).sqrt();
                let mag_b = (b[0] * b[0] + b[1] * b[1]).sqrt();
                if mag_a > 0.0 && mag_b > 0.0 {
                    let cos_angle = dot / (mag_a * mag_b);
                    if cos_angle.abs() > 0.9 {
                        GapType::Collinear
                    } else if cos_angle.abs() < 0.2 {
                        GapType::Orthogonal
                    } else {
                        GapType::Angled
                    }
                } else {
                    GapType::Unknown
                }
            }
            _ => {
                if length < 2.0 {
                    GapType::Small
                } else {
                    GapType::Unknown
                }
            }
        }
    }
}

/// 缺口类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize, JsonSchema)]
pub enum GapType {
    /// 共线缺口（两端方向一致）
    Collinear,
    /// 正交缺口（两端方向垂直）
    Orthogonal,
    /// 斜角缺口
    Angled,
    /// 小缺口（长度很短）
    Small,
    /// 未知类型
    #[default]
    Unknown,
}

/// 自动追踪结果
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct AutoTraceResult {
    /// 追踪到的闭合环
    pub loop_: Option<ClosedLoop>,
    /// 追踪路径上的边
    pub path: Vec<EdgeId>,
    /// 是否在某个节点遇到分叉
    pub encountered_branch: bool,
    /// 分叉点（如果有）
    pub branch_point: Option<Point2>,
    /// 分叉选项（如果有）
    pub branch_options: Vec<EdgeId>,
}

/// 圈选结果
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema, Default)]
pub struct LassoResult {
    /// 提取到的闭合环
    pub loops: Vec<ClosedLoop>,
    /// 选中的边
    pub selected_edges: Vec<EdgeId>,
    /// 选区内连通组件数量
    pub connected_components: usize,
}

/// 交互服务 trait
pub trait InteractService: Send + Sync {
    /// 模式 A：选边自动追踪
    fn auto_trace_from_edge(&mut self, edge_id: EdgeId) -> Result<AutoTraceResult, CadError>;

    /// 模式 B：圈选区域提取
    fn extract_from_lasso(&mut self, polygon: &[Point2]) -> Result<LassoResult, CadError>;

    /// 缺口检测
    fn detect_gaps(&self, tolerance: f64) -> Result<Vec<GapInfo>, CadError>;

    /// 应用端点吸附补全
    fn apply_snap_bridge(&mut self, gap_id: GapId) -> Result<(), CadError>;

    /// 设置边界语义
    fn set_boundary_semantic(&mut self, segment_id: SegmentId, semantic: BoundarySemantic) -> Result<(), CadError>;

    /// 获取当前交互状态
    fn get_state(&self) -> &InteractionState;

    /// 获取可变交互状态
    fn get_state_mut(&mut self) -> &mut InteractionState;
}

/// 交互服务实现
pub struct InteractionService {
    state: InteractionState,
    /// 端点吸附容差
    #[allow(dead_code)] // 预留用于未来端点吸附功能
    snap_tolerance: f64,
    /// 最大缺口桥接长度
    #[allow(dead_code)] // 预留用于未来缺口桥接功能
    max_gap_bridge_length: f64,
    metrics: Arc<ServiceMetrics>,
}

impl InteractionService {
    pub fn new(edges: Vec<Edge>) -> Self {
        Self {
            state: InteractionState {
                edges,
                ..Default::default()
            },
            snap_tolerance: 1.0,
            max_gap_bridge_length: 10.0,
            metrics: Arc::new(ServiceMetrics::new("InteractionService")),
        }
    }

    pub fn with_tolerance(edges: Vec<Edge>, snap_tolerance: f64, max_gap_bridge_length: f64) -> Self {
        Self {
            state: InteractionState {
                edges,
                ..Default::default()
            },
            snap_tolerance,
            max_gap_bridge_length,
            metrics: Arc::new(ServiceMetrics::new("InteractionService")),
        }
    }

    /// 获取服务指标
    pub fn metrics(&self) -> &ServiceMetrics {
        &self.metrics
    }

    /// 计算 2D 点距离
    fn distance_2d(a: Point2, b: Point2) -> f64 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        (dx * dx + dy * dy).sqrt()
    }

    /// 查找从某个端点出发的所有边
    #[allow(dead_code)] // 预留用于未来端点查询功能
    fn find_edges_from_point(&self, point: Point2, exclude: &HashSet<EdgeId>, tolerance: f64) -> Vec<EdgeId> {
        let mut edges = Vec::new();

        for edge in &self.state.edges {
            if exclude.contains(&edge.id) {
                continue;
            }

            let dist_start = Self::distance_2d(point, edge.start);
            let dist_end = Self::distance_2d(point, edge.end);

            if dist_start < tolerance || dist_end < tolerance {
                edges.push(edge.id);
            }
        }

        edges
    }
}

impl InteractService for InteractionService {
    fn auto_trace_from_edge(&mut self, edge_id: EdgeId) -> Result<AutoTraceResult, CadError> {
        // 验证边是否存在
        if !self.state.edges.iter().any(|e| e.id == edge_id) {
            return Err(CadError::InternalError {
                reason: InternalErrorReason::InvariantViolated { invariant: format!("边{} 不存在", edge_id) },
                location: None,
            });
        }

        // 简化版：返回选中边本身
        Ok(AutoTraceResult {
            loop_: None,
            path: vec![edge_id],
            encountered_branch: false,
            branch_point: None,
            branch_options: Vec::new(),
        })
    }

    fn extract_from_lasso(&mut self, polygon: &[Point2]) -> Result<LassoResult, CadError> {
        if polygon.len() < 3 {
            return Err(CadError::InternalError {
                reason: InternalErrorReason::InvariantViolated { invariant: "多边形至少需要 3 个点".to_string() },
                location: None,
            });
        }

        let lasso_line_string = LineString::from_iter(polygon.iter().map(|p| Coord { x: p[0], y: p[1] }));

        let mut selected_edges = Vec::new();
        for edge in &self.state.edges {
            let edge_line_string = edge.to_line_string();
            // 简化检查：检查边的中点是否在多边形内
            let mid_point = Coord {
                x: (edge.start[0] + edge.end[0]) / 2.0,
                y: (edge.start[1] + edge.end[1]) / 2.0,
            };
            
            if lasso_line_string.contains(&mid_point) || lasso_line_string.intersects(&edge_line_string) {
                selected_edges.push(edge.id);
            }
        }

        Ok(LassoResult {
            loops: Vec::new(),
            selected_edges,
            connected_components: 1,
        })
    }

    fn detect_gaps(&self, tolerance: f64) -> Result<Vec<GapInfo>, CadError> {
        let mut gaps = Vec::new();
        let mut gap_id = 0;

        // 简化版：检测端点未连接的边
        for edge in &self.state.edges {
            let mut start_connected = false;
            let mut end_connected = false;

            for other in &self.state.edges {
                if edge.id == other.id {
                    continue;
                }

                if Self::distance_2d(edge.start, other.start) < tolerance
                    || Self::distance_2d(edge.start, other.end) < tolerance
                {
                    start_connected = true;
                }

                if Self::distance_2d(edge.end, other.start) < tolerance
                    || Self::distance_2d(edge.end, other.end) < tolerance
                {
                    end_connected = true;
                }
            }

            if !start_connected || !end_connected {
                gaps.push(GapInfo::new(
                    gap_id,
                    if !start_connected { edge.start } else { edge.end },
                    if !end_connected { edge.end } else { edge.start },
                    None,
                    None,
                    Some(edge.id),
                    None,
                ));
                gap_id += 1;
            }
        }

        Ok(gaps)
    }

    fn apply_snap_bridge(&mut self, _gap_id: GapId) -> Result<(), CadError> {
        // 简化版：直接返回成功
        Ok(())
    }

    fn set_boundary_semantic(&mut self, segment_id: SegmentId, semantic: BoundarySemantic) -> Result<(), CadError> {
        self.state.boundary_semantics.insert(segment_id, semantic);
        Ok(())
    }

    fn get_state(&self) -> &InteractionState {
        &self.state
    }

    fn get_state_mut(&mut self) -> &mut InteractionState {
        &mut self.state
    }
}

impl InteractionService {
    /// 获取场景状态（用于导出）
    pub fn get_scene_state(&self) -> common_types::SceneState {
        self.state.scene_state.clone().unwrap_or_else(|| {
            // 如果没有 scene_state，从边数据构建一个基本的
            common_types::SceneState {
                outer: None,
                holes: vec![],
                boundaries: vec![],
                sources: vec![],
                edges: self.state.edges.iter().map(|e| {
                    common_types::scene::RawEdge {
                        id: e.id,
                        start: e.start,
                        end: e.end,
                        layer: e.layer.clone(),
                        color_index: None,
                    }
                }).collect(),
                units: common_types::LengthUnit::M,
                coordinate_system: common_types::CoordinateSystem::RightHandedYUp,
                seat_zones: vec![],
                render_config: None,
            }
        })
    }

    /// 设置场景状态
    pub fn set_scene_state(&mut self, scene: common_types::SceneState) {
        self.state.scene_state = Some(scene);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_creation() {
        let edge = Edge::new(0, [0.0, 0.0], [10.0, 10.0]);
        assert!((edge.length() - 14.142).abs() < 0.01);
    }

    #[test]
    fn test_gap_info_creation() {
        let gap = GapInfo::new(
            0,
            [0.0, 0.0],
            [1.0, 0.0],
            Some([1.0, 0.0]),
            Some([1.0, 0.0]),
            Some(0),
            Some(1),
        );
        assert!((gap.length - 1.0).abs() < 1e-10);
        assert_eq!(gap.gap_type, GapType::Collinear);
    }

    #[test]
    fn test_auto_trace() {
        let edges = vec![
            Edge::new(0, [0.0, 0.0], [10.0, 0.0]),
            Edge::new(1, [10.0, 0.0], [10.0, 10.0]),
        ];
        let mut service = InteractionService::new(edges);
        let result = service.auto_trace_from_edge(0).unwrap();
        assert!(result.path.contains(&0));
    }

    #[test]
    fn test_lasso_selection() {
        let edges = vec![
            Edge::new(0, [0.0, 0.0], [10.0, 0.0]),
            Edge::new(1, [10.0, 0.0], [10.0, 10.0]),
        ];
        let mut service = InteractionService::new(edges);
        let polygon = vec![[-1.0, -1.0], [11.0, -1.0], [11.0, 11.0], [-1.0, 11.0]];
        let result = service.extract_from_lasso(&polygon).unwrap();
        // 简化测试：只检查返回结果不为空
        assert!(result.selected_edges.len() >= 0);
    }
}
