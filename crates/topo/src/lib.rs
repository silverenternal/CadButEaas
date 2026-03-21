//! 拓扑建模服务
//!
//! # 概述
//!
//! 将原始线段（Polyline）构建为平面图（Planar Graph），提取闭合环（外轮廓/孔洞）。
//! 支持端点吸附、线段合并、交点切分、去噪等几何清洗功能。
//!
//! # 输入输出
//!
//! ## 输入
//! - `Vec<Polyline>`: 原始多段线集合
//! - `TopoConfig`: 拓扑配置（容差、最小线长等）
//!
//! ## 输出
//! - `TopologyResult`: 包含外边界（outer）、孔洞（holes）、所有环（all_loops）
//!
//! # 核心算法
//!
//! ## 1. 端点吸附（Snap）
//! 使用 R*-tree 空间索引加速，复杂度 O(n log n)。
//! 合并距离小于 `snap_tolerance` 的端点。
//!
//! ## 2. 交点切分
//! 使用 Bentley-Ottmann 扫描线算法计算线段交点，
//! 在交点处将线段切分为两段。
//!
//! ## 3. 线段合并
//! 合并共线且相邻的短线段，支持角度容差和间隙容差配置。
//!
//! ## 4. 闭合环提取
//! 从平面图遍历提取闭合环，采用"小转角优先"策略。
//! 按有符号面积分类：正面积 = 外轮廓，负面积 = 孔洞。
//!
//! # 容差参数
//!
//! | 参数 | 默认值 | 说明 |
//! |------|--------|------|
//! | `snap_tolerance` | 0.5mm | 端点吸附距离 |
//! | `min_line_length` | 1.0mm | 最小线段长度（去噪） |
//! | `merge_angle_tolerance` | 5° | 共线合并角度容差 |
//! | `merge_gap_tolerance` | 0.5mm | 共线合并间隙容差 |
//!
//! # 使用示例
//!
//! ```rust,no_run
//! use topo::{TopoService, service::TopoConfig};
//! use common_types::geometry::Polyline;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let service = TopoService::new(TopoConfig::default());
//!
//! let polylines: Vec<Polyline> = vec![
//!     vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]],
//! ];
//!
//! let result = service.build_topology(&polylines)?;
//!
//! if let Some(outer) = result.outer {
//!     println!("外边界：{:?} 个点", outer.points.len());
//!     println!("孔洞数量：{}", result.holes.len());
//! }
//! # Ok(())
//! # }
//! ```
//!
//! # 常见错误
//!
//! | 错误代码 | 说明 | 解决方法 |
//! |----------|------|----------|
//! | `E001` | 环未闭合 | 检查输入线段连续性，调整 snap_tolerance |
//! | `E002` | 自相交 | 输入数据存在交叉，需先执行交点切分 |
//! | `W001` | 短边警告 | 存在长度 < min_line_length 的边，已自动移除 |
//! | `W002` | 尖角警告 | 存在角度 < 15° 的尖角，可能导致数值不稳定 |
//!
//! # Halfedge 结构（P2 新增）
//!
//! Halfedge 结构用于支持复杂平面图遍历：
//! - 嵌套孔洞、岛中岛
//! - 共享边界查询
//! - 面枚举和边界操作
//!
//! ```rust,no_run
//! use topo::halfedge::HalfedgeGraph;
//! use common_types::geometry::Point2;
//!
//! let outer = vec![
//!     [0.0, 0.0], [20.0, 0.0], [20.0, 20.0], [0.0, 20.0],
//! ];
//! let hole = vec![
//!     [8.0, 8.0], [12.0, 8.0], [12.0, 12.0], [8.0, 12.0],
//! ];
//!
//! let graph = HalfedgeGraph::from_loops(&[outer, hole]);
//! assert_eq!(graph.faces().count(), 2); // 1 个外轮廓 + 1 个孔洞
//! ```

pub mod service;
pub mod graph_builder;
pub mod loop_extractor;
pub mod halfedge;
pub mod spatial_index;  // P1-3 新增：分层空间索引渲染
pub mod bentley_ottmann;  // P1-3 新增：Bentley-Ottmann 扫描线算法
pub mod parallel;  // P1-4 新增：并行化处理
pub mod union_find;  // P11-3 新增：并查集数据结构

pub use service::TopoService;
pub use graph_builder::GraphBuilder;
pub use loop_extractor::LoopExtractor;
pub use halfedge::{HalfedgeGraph, Halfedge, HalfedgeId, Vertex, VertexId, Face, FaceId};
pub use spatial_index::{SpatialIndex, SpatialIndexConfig, SpatialIndexStats, RenderEntity, ViewportCuller};
pub use bentley_ottmann::{BentleyOttmann, Segment, Intersection, brute_force_intersections};
pub use parallel::{snap_endpoints_parallel, process_geometries_parallel, find_intersections_parallel};
pub use union_find::UnionFind;
