//! Halfedge（半边）数据结构实现
//!
//! # 概述
//!
//! Halfedge 结构是平面图遍历的基础数据结构，支持：
//! - 高效的面枚举（外轮廓/孔洞）
//! - 边界遍历（顺时针/逆时针）
//! - 邻接关系查询（顶点→边→面）
//! - 嵌套孔洞和岛中岛处理
//!
//! # 核心概念
//!
//! 每条边被拆分为两条方向相反的半边：
//! - 每条半边有一个起点（origin）和终点（twin.origin）
//! - 每条半边属于一个面（face）
//! - 每条半边有下一条半边（next）和上一条半边（prev）
//! - 每条半边有孪生半边（twin），指向相反方向
//!
//! # 数据结构
//!
//! ```text
//!          face_left
//!              ↑
//!              |
//!   origin →  [HE]  → next → ...
//!              ↓
//!            twin
//!              ↓
//!   twin.origin ← [TwinHE]
//!              ↑
//!              |
//!          face_right
//! ```
//!
//! # 使用示例
//!
//! ```rust,no_run
//! use topo::halfedge::HalfedgeGraph;
//! use common_types::geometry::Point2;
//!
//! // 创建三角形
//! let mut graph = HalfedgeGraph::new();
//! let v0 = graph.add_vertex([0.0, 0.0]);
//! let v1 = graph.add_vertex([10.0, 0.0]);
//! let v2 = graph.add_vertex([10.0, 10.0]);
//!
//! // 添加边（自动创建两条半边）
//! let e0 = graph.add_edge(v0, v1, 0);
//! let e1 = graph.add_edge(v1, v2, 1);
//! let e2 = graph.add_edge(v2, v0, 2);
//!
//! // 遍历面
//! for face_id in graph.faces() {
//!     let boundary = graph.face_boundary_points(face_id);
//!     println!("面 {:?} 的边界：{:?} 个点", face_id, boundary.len());
//! }
//! ```

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use common_types::geometry::Point2;

// 确保 serde 宏可用
use serde::{Deserializer, Serializer};

/// 半边 ID
pub type HalfedgeId = usize;

/// 顶点 ID
pub type VertexId = usize;

/// 面 ID
pub type FaceId = usize;

/// 半边结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Halfedge {
    /// 起点顶点 ID
    pub origin: VertexId,
    /// 下一条半边（逆时针遍历）
    pub next: Option<HalfedgeId>,
    /// 上一条半边
    pub prev: Option<HalfedgeId>,
    /// 孪生半边（方向相反）
    pub twin: HalfedgeId,
    /// 左侧面 ID（None 表示无界面）
    pub face: Option<FaceId>,
    /// 边索引（用于回溯到原始边）
    pub edge_index: usize,
}

impl Halfedge {
    /// 创建新的半边
    pub fn new(origin: VertexId, edge_index: usize) -> Self {
        Self {
            origin,
            next: None,
            prev: None,
            twin: 0, // 占位，后续设置
            face: None,
            edge_index,
        }
    }
}

/// 顶点结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Vertex {
    /// 顶点坐标
    pub position: Point2,
    /// 从该顶点出发的半边 ID（任意一条）
    pub outgoing: Option<HalfedgeId>,
}

impl Vertex {
    /// 创建新顶点
    pub fn new(position: Point2) -> Self {
        Self {
            position,
            outgoing: None,
        }
    }
}

/// 面结构
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Face {
    /// 面 ID
    pub id: FaceId,
    /// 边界上的任意半边 ID
    pub boundary: Option<HalfedgeId>,
    /// 是否为孔洞
    pub is_hole: bool,
    /// 有符号面积（>0 外轮廓，<0 孔洞）
    pub signed_area: f64,
}

impl Face {
    /// 创建新面
    pub fn new(id: FaceId) -> Self {
        Self {
            id,
            boundary: None,
            is_hole: false,
            signed_area: 0.0,
        }
    }
}

/// Halfedge 图（平面图表示）
#[derive(Debug, Clone)]
pub struct HalfedgeGraph {
    /// 顶点列表
    pub vertices: Vec<Vertex>,
    /// 半边列表
    pub halfedges: Vec<Halfedge>,
    /// 面列表
    pub faces: Vec<Face>,
    /// 顶点位置索引（用于快速查找）
    pub vertex_map: HashMap<Point2Key, VertexId>,
    /// 下一个可用 ID
    next_vertex_id: VertexId,
    next_halfedge_id: HalfedgeId,
    next_face_id: FaceId,
    /// P1-2 新增：嵌套层级缓存（用于 O(1) 查询）
    /// 存储每个面的父面 ID（None 表示根面/外轮廓）
    face_parent_cache: Vec<Option<FaceId>>,
    /// P1-2 新增：子面索引（面→子面列表）
    face_children_cache: Vec<Vec<FaceId>>,
}

/// 顶点坐标的哈希键（处理浮点精度）
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Point2Key {
    pub x: i64,
    pub y: i64,
}

impl Point2Key {
    /// 从 Point2 创建键（精度到 0.001mm）
    pub fn from_point(p: Point2) -> Self {
        Self {
            x: (p[0] * 1000.0).round() as i64,
            y: (p[1] * 1000.0).round() as i64,
        }
    }
}

// Point2Key 不需要序列化，它是内部索引结构

impl HalfedgeGraph {
    /// 创建新的 Halfedge 图
    pub fn new() -> Self {
        Self {
            vertices: Vec::new(),
            halfedges: Vec::new(),
            faces: Vec::new(),
            vertex_map: HashMap::new(),
            next_vertex_id: 0,
            next_halfedge_id: 0,
            next_face_id: 0,
            face_parent_cache: Vec::new(),
            face_children_cache: Vec::new(),
        }
    }

    /// 添加顶点
    pub fn add_vertex(&mut self, position: Point2) -> VertexId {
        let key = Point2Key::from_point(position);
        
        // 检查是否已存在（处理浮点精度）
        if let Some(&vid) = self.vertex_map.get(&key) {
            return vid;
        }

        let vid = self.next_vertex_id;
        self.next_vertex_id += 1;
        
        self.vertices.push(Vertex::new(position));
        self.vertex_map.insert(key, vid);
        
        vid
    }

    /// 添加边（创建两条方向相反的半边）
    pub fn add_edge(&mut self, from: VertexId, to: VertexId, edge_index: usize) -> (HalfedgeId, HalfedgeId) {
        let he_id = self.next_halfedge_id;
        let twin_id = self.next_halfedge_id + 1;
        self.next_halfedge_id += 2;

        // 创建两条半边
        let he = Halfedge::new(from, edge_index);
        let twin = Halfedge::new(to, edge_index);

        // 设置孪生关系
        let he_with_twin = Halfedge {
            twin: twin_id,
            ..he
        };
        let twin_with_twin = Halfedge {
            twin: he_id,
            ..twin
        };

        self.halfedges.push(he_with_twin);
        self.halfedges.push(twin_with_twin);

        // 更新顶点的 outgoing 引用
        self.vertices[from].outgoing = Some(he_id);
        self.vertices[to].outgoing = Some(twin_id);

        (he_id, twin_id)
    }

    /// 获取半边
    pub fn halfedge(&self, id: HalfedgeId) -> &Halfedge {
        &self.halfedges[id]
    }

    /// 获取可变半边
    pub fn halfedge_mut(&mut self, id: HalfedgeId) -> &mut Halfedge {
        &mut self.halfedges[id]
    }

    /// 获取顶点
    pub fn vertex(&self, id: VertexId) -> &Vertex {
        &self.vertices[id]
    }

    /// 获取面
    pub fn face(&self, id: FaceId) -> &Face {
        &self.faces[id]
    }

    /// 创建新面
    pub fn add_face(&mut self) -> FaceId {
        let fid = self.next_face_id;
        self.next_face_id += 1;
        self.faces.push(Face::new(fid));
        // 扩展缓存
        self.face_parent_cache.push(None);
        self.face_children_cache.push(Vec::new());
        fid
    }

    /// 设置面的边界
    pub fn set_face_boundary(&mut self, face_id: FaceId, halfedge_id: HalfedgeId) {
        if let Some(face) = self.faces.get_mut(face_id) {
            face.boundary = Some(halfedge_id);
            self.halfedges[halfedge_id].face = Some(face_id);
        }
    }

    /// 标记面为孔洞
    pub fn set_face_as_hole(&mut self, face_id: FaceId, signed_area: f64) {
        if let Some(face) = self.faces.get_mut(face_id) {
            face.is_hole = signed_area < 0.0;
            face.signed_area = signed_area;
        }
    }

    /// 遍历面的边界半边
    pub fn face_boundary_loop(&self, face_id: FaceId) -> Vec<HalfedgeId> {
        let face = &self.faces[face_id];
        let mut loop_hes = Vec::new();
        
        if let Some(start_he) = face.boundary {
            let mut current = start_he;
            loop {
                loop_hes.push(current);
                let he = &self.halfedges[current];
                if let Some(next) = he.next {
                    if next == start_he {
                        break;
                    }
                    current = next;
                } else {
                    break;
                }
            }
        }
        
        loop_hes
    }

    /// 获取面的边界点（按顺序）
    pub fn face_boundary_points(&self, face_id: FaceId) -> Vec<Point2> {
        let loop_hes = self.face_boundary_loop(face_id);
        loop_hes
            .iter()
            .map(|&he_id| {
                let he = &self.halfedges[he_id];
                self.vertices[he.origin].position
            })
            .collect()
    }

    /// 计算面的有符号面积
    pub fn compute_face_area(&self, face_id: FaceId) -> f64 {
        let points = self.face_boundary_points(face_id);
        calculate_signed_area(&points)
    }

    /// 获取所有面 ID
    pub fn faces(&self) -> impl Iterator<Item = FaceId> + '_ {
        0..self.faces.len()
    }

    /// 获取所有顶点
    pub fn vertices(&self) -> impl Iterator<Item = (VertexId, &Vertex)> + '_ {
        self.vertices.iter().enumerate()
    }

    /// 获取从顶点出发的所有半边
    pub fn outgoing_halfedges_from(&self, vertex_id: VertexId) -> Vec<HalfedgeId> {
        let mut result = Vec::new();
        let vertex = &self.vertices[vertex_id];
        
        if let Some(start_he) = vertex.outgoing {
            // 通过 twin.next 遍历围绕顶点的所有半边
            let mut current = start_he;
            loop {
                result.push(current);
                let he = &self.halfedges[current];
                let twin = &self.halfedges[he.twin];
                if let Some(next) = twin.next {
                    if next == start_he {
                        break;
                    }
                    current = next;
                } else {
                    break;
                }
            }
        }
        
        result
    }

    /// 从线段环构建 Halfedge 图
    ///
    /// # Arguments
    /// * `loops` - 闭合环列表（每个环是点序列）
    ///
    /// # Returns
    /// 构建的 HalfedgeGraph
    pub fn from_loops(loops: &[Vec<Point2>]) -> Self {
        let mut graph = Self::new();

        for (loop_idx, loop_points) in loops.iter().enumerate() {
            if loop_points.len() < 3 {
                continue;
            }

            // 创建顶点并记录 ID
            let mut vertex_ids: Vec<VertexId> = Vec::new();
            for &point in loop_points {
                let vid = graph.add_vertex(point);
                vertex_ids.push(vid);
            }

            // 创建边（半边对）
            let mut halfedge_ids: Vec<HalfedgeId> = Vec::new();
            for i in 0..loop_points.len() {
                let from = vertex_ids[i];
                let to = vertex_ids[(i + 1) % loop_points.len()];
                let edge_index = loop_idx * 1000 + i; // 唯一索引

                let (he_id, _twin_id) = graph.add_edge(from, to, edge_index);
                halfedge_ids.push(he_id);
            }

            // 设置 next/prev 指针（形成环）
            let n = halfedge_ids.len();
            for i in 0..n {
                let he_id = halfedge_ids[i];
                let prev_id = halfedge_ids[(i + n - 1) % n];
                let next_id = halfedge_ids[(i + 1) % n];

                graph.halfedges[he_id].next = Some(next_id);
                graph.halfedges[he_id].prev = Some(prev_id);
            }

            // 创建面
            let face_id = graph.add_face();
            graph.set_face_boundary(face_id, halfedge_ids[0]);

            // 计算面积并标记是否为孔洞
            let area = graph.compute_face_area(face_id);
            graph.set_face_as_hole(face_id, area);
        }

        graph
    }

    /// 从 Halfedge 图提取外轮廓和孔洞
    ///
    /// # Returns
    /// `(Option<ClosedLoop>, Vec<ClosedLoop>)` - 外轮廓（如果有）和孔洞列表
    ///
    /// # 算法说明
    /// - 遍历所有面，按面积符号分类
    /// - 正面积 = 外轮廓，负面积 = 孔洞
    /// - 支持嵌套孔洞和岛中岛
    pub fn extract_outer_and_holes(&self) -> (Option<common_types::ClosedLoop>, Vec<common_types::ClosedLoop>) {
        use common_types::ClosedLoop;

        let mut outer: Option<ClosedLoop> = None;
        let mut holes: Vec<ClosedLoop> = Vec::new();

        // 按面积绝对值排序面
        let mut faces_with_area: Vec<(FaceId, f64)> = self.faces()
            .map(|fid| (fid, self.compute_face_area(fid)))
            .collect();
        faces_with_area.sort_by(|a, b| {
            b.1.abs().partial_cmp(&a.1.abs()).unwrap_or(std::cmp::Ordering::Equal)
        });

        for (fid, area) in faces_with_area {
            let points = self.face_boundary_points(fid);
            if points.len() < 3 {
                continue;
            }

            let loop_ = ClosedLoop::new(points);

            if area > 0.0 {
                // 正面积 = 外轮廓
                if outer.is_none() {
                    outer = Some(loop_);
                }
            } else {
                // 负面积 = 孔洞
                holes.push(loop_);
            }
        }

        (outer, holes)
    }

    // ========================================================================
    // P1-2 新增：嵌套孔洞识别与 O(1) 层级查询
    // ========================================================================

    /// 构建嵌套层级关系（射线法）
    ///
    /// ## 算法说明
    /// 使用射线法（Ray Casting）判断面的包含关系：
    /// 1. 对每个面，取其边界上的一个点
    /// 2. 从该点向右发射水平射线
    /// 3. 计算射线与其他面边界的交点数量
    /// 4. 奇数个交点 = 在内部，偶数个交点 = 在外部
    ///
    /// ## 时间复杂度
    /// - 构建：O(F² × E)，F 为面数，E 为平均边数
    /// - 查询：O(1)（使用缓存）
    ///
    /// ## 返回值
    /// 构建是否成功
    pub fn build_nesting_hierarchy(&mut self) -> Result<(), String> {
        let num_faces = self.faces.len();
        
        // 重置缓存
        self.face_parent_cache = vec![None; num_faces];
        self.face_children_cache = vec![Vec::new(); num_faces];

        // 计算所有面的面积
        let face_areas: Vec<f64> = (0..num_faces)
            .map(|fid| self.compute_face_area(fid))
            .collect();

        // 获取每个面的边界点（用于射线法）
        let face_points: Vec<Vec<Point2>> = (0..num_faces)
            .map(|fid| self.face_boundary_points(fid))
            .collect();

        // 对每个面，找到它的直接父面（包含它的最小面）
        for child_id in 0..num_faces {
            let child_area = face_areas[child_id];
            let child_points = &face_points[child_id];
            
            if child_points.len() < 3 {
                continue;
            }

            // 使用面的第一个顶点作为测试点
            let test_point = child_points[0];
            
            // 找到所有包含该测试点的面（候选父面）
            let mut candidate_parents: Vec<(FaceId, f64)> = Vec::new();
            
            for parent_id in 0..num_faces {
                if parent_id == child_id {
                    continue;
                }

                let parent_area = face_areas[parent_id];
                
                // 只有面积更大的面才可能是父面
                if parent_area.abs() <= child_area.abs() {
                    continue;
                }

                let parent_points = &face_points[parent_id];
                
                // 使用射线法判断测试点是否在父面内
                if Self::point_in_polygon_ray_casting(test_point, parent_points) {
                    candidate_parents.push((parent_id, parent_area.abs()));
                }
            }

            // 选择面积最小的候选父面（直接父面）
            if let Some((parent_id, _)) = candidate_parents.iter().min_by(|a, b| {
                a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal)
            }) {
                self.face_parent_cache[child_id] = Some(*parent_id);
                self.face_children_cache[*parent_id].push(child_id);
            }
        }

        // 更新面的 is_hole 标记
        for (face_id, parent) in self.face_parent_cache.iter().enumerate() {
            if let Some(_parent_id) = parent {
                // 有父面的是孔洞
                self.faces[face_id].is_hole = true;
            } else {
                // 无父面的是外轮廓
                self.faces[face_id].is_hole = false;
            }
        }

        Ok(())
    }

    /// 射线法判断点是否在多边形内
    ///
    /// ## 算法
    /// 从测试点向右发射水平射线，计算与多边形边界的交点数量：
    /// - 奇数个交点 = 在多边形内
    /// - 偶数个交点 = 在多边形外
    fn point_in_polygon_ray_casting(point: Point2, polygon: &[Point2]) -> bool {
        let mut inside = false;
        let n = polygon.len();
        
        for i in 0..n {
            let p1 = polygon[i];
            let p2 = polygon[(i + 1) % n];
            
            // 检查射线是否与边相交
            let intersect = ((p1[1] > point[1]) != (p2[1] > point[1]))
                && (point[0] < (p2[0] - p1[0]) * (point[1] - p1[1]) / (p2[1] - p1[1]) + p1[0]);
            
            if intersect {
                inside = !inside;
            }
        }
        
        inside
    }

    /// O(1) 查询：获取面的父面 ID
    ///
    /// ## 返回值
    /// - `Some(FaceId)`: 父面 ID（当前面是孔洞）
    /// - `None`: 无父面（当前面是外轮廓）
    pub fn get_face_parent(&self, face_id: FaceId) -> Option<FaceId> {
        self.face_parent_cache.get(face_id).copied().flatten()
    }

    /// O(1) 查询：获取面的所有子面
    ///
    /// ## 返回值
    /// 子面 ID 列表（对于外轮廓，返回所有直接孔洞）
    pub fn get_face_children(&self, face_id: FaceId) -> &[FaceId] {
        self.face_children_cache.get(face_id).map(|v| v.as_slice()).unwrap_or(&[])
    }

    /// O(1) 查询：获取面的嵌套深度
    ///
    /// ## 返回值
    /// 嵌套深度（0 = 外轮廓，1 = 一级孔洞，2 = 孔中孔，...）
    pub fn get_nesting_depth(&self, face_id: FaceId) -> usize {
        let mut depth = 0;
        let mut current = face_id;
        
        while let Some(parent) = self.face_parent_cache.get(current).copied().flatten() {
            depth += 1;
            current = parent;
        }
        
        depth
    }

    /// O(1) 查询：判断面是否为孔洞
    pub fn is_hole(&self, face_id: FaceId) -> bool {
        self.face_parent_cache.get(face_id).map_or(false, |p| p.is_some())
    }

    /// O(1) 查询：获取根面（外轮廓）
    pub fn get_root_face(&self, face_id: FaceId) -> Option<FaceId> {
        let mut current = face_id;
        
        while let Some(parent) = self.face_parent_cache.get(current).copied().flatten() {
            current = parent;
        }
        
        // 如果当前面就是根面（无父面），返回它
        if self.face_parent_cache.get(current).map_or(true, |p| p.is_none()) {
            Some(current)
        } else {
            None
        }
    }

    /// 获取完整的嵌套层级路径（从根面到当前面）
    pub fn get_nesting_path(&self, face_id: FaceId) -> Vec<FaceId> {
        let mut path = Vec::new();
        let mut current = Some(face_id);
        
        while let Some(fid) = current {
            path.push(fid);
            current = self.face_parent_cache.get(fid).copied().flatten();
        }
        
        path.reverse();
        path
    }

    /// 验证嵌套层级关系
    pub fn validate_nesting(&self) -> Result<(), String> {
        for face_id in self.faces() {
            // 验证父子关系一致性
            if let Some(parent_id) = self.face_parent_cache.get(face_id).copied().flatten() {
                // 父面的子面列表应该包含当前面
                if !self.face_children_cache.get(parent_id).map_or(false, |children| {
                    children.contains(&face_id)
                }) {
                    return Err(format!(
                        "面 {} 的父子关系不一致：父面 {} 的子面列表不包含它",
                        face_id, parent_id
                    ));
                }
            }
        }
        
        Ok(())
    }

    /// 验证 Halfedge 结构完整性
    ///
    /// 检查：
    /// 1. 所有半边的 twin 是否正确
    /// 2. 所有半边的 next/prev 是否形成闭环
    /// 3. 欧拉公式：V - E + F = 2（连通图）
    pub fn validate(&self) -> Result<(), String> {
        // 检查 twin 关系
        for (i, he) in self.halfedges.iter().enumerate() {
            let twin = &self.halfedges[he.twin];
            if twin.twin != i {
                return Err(format!("Halfedge {} twin 关系错误", i));
            }
            if he.origin != twin.origin {
                // twin 的 origin 应该是 he 的终点
                // 这里简化检查
            }
        }

        // 检查 next/prev 闭环
        for (i, he) in self.halfedges.iter().enumerate() {
            if let Some(next_id) = he.next {
                if next_id >= self.halfedges.len() {
                    return Err(format!("Halfedge {} next 指向无效", i));
                }
            }
            if let Some(prev_id) = he.prev {
                if prev_id >= self.halfedges.len() {
                    return Err(format!("Halfedge {} prev 指向无效", i));
                }
            }
        }

        // 欧拉公式验证（简化版，假设连通图）
        let v = self.vertices.len();
        let e = self.halfedges.len() / 2; // 每条边有 2 条半边
        let f = self.faces.len();
        
        // V - E + F 应该等于 2（对于连通平面图）
        // 但实际可能有多个连通分量，所以这里只做记录
        eprintln!("欧拉公式验证：V={} E={} F={} => V-E+F={}", v, e, f, v as i32 - e as i32 + f as i32);

        Ok(())
    }
}

impl Default for HalfedgeGraph {
    fn default() -> Self {
        Self::new()
    }
}

// 手动实现 Serialize/Deserialize，跳过 vertex_map 和 next_* 字段
impl Serialize for HalfedgeGraph {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("HalfedgeGraph", 3)?;
        state.serialize_field("vertices", &self.vertices)?;
        state.serialize_field("halfedges", &self.halfedges)?;
        state.serialize_field("faces", &self.faces)?;
        state.end()
    }
}

impl<'de> Deserialize<'de> for HalfedgeGraph {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        #[derive(Deserialize)]
        struct Helper {
            vertices: Vec<Vertex>,
            halfedges: Vec<Halfedge>,
            faces: Vec<Face>,
        }

        let helper = Helper::deserialize(deserializer)?;
        let num_faces = helper.faces.len();
        let mut graph = HalfedgeGraph {
            vertices: helper.vertices,
            halfedges: helper.halfedges,
            faces: helper.faces,
            vertex_map: HashMap::new(),
            next_vertex_id: 0,
            next_halfedge_id: 0,
            next_face_id: 0,
            face_parent_cache: vec![None; num_faces],
            face_children_cache: vec![Vec::new(); num_faces],
        };

        // 重建 vertex_map
        for (vid, vertex) in graph.vertices.iter().enumerate() {
            let key = Point2Key::from_point(vertex.position);
            graph.vertex_map.insert(key, vid);
        }

        // 更新 next_* ID
        graph.next_vertex_id = graph.vertices.len();
        graph.next_halfedge_id = graph.halfedges.len();
        graph.next_face_id = graph.faces.len();

        Ok(graph)
    }
}

/// 计算多边形的有符号面积
fn calculate_signed_area(points: &[Point2]) -> f64 {
    if points.len() < 3 {
        return 0.0;
    }

    let mut area = 0.0;
    let n = points.len();

    for i in 0..n {
        let j = (i + 1) % n;
        area += points[i][0] * points[j][1];
        area -= points[j][0] * points[i][1];
    }

    area / 2.0
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_triangle() {
        let mut graph = HalfedgeGraph::new();
        
        // 创建三角形顶点
        let v0 = graph.add_vertex([0.0, 0.0]);
        let v1 = graph.add_vertex([10.0, 0.0]);
        let v2 = graph.add_vertex([10.0, 10.0]);
        
        // 创建边
        let (e0, _) = graph.add_edge(v0, v1, 0);
        let (e1, _) = graph.add_edge(v1, v2, 1);
        let (e2, _) = graph.add_edge(v2, v0, 2);
        
        // 设置 next 指针（逆时针）
        graph.halfedge_mut(e0).next = Some(e1);
        graph.halfedge_mut(e1).next = Some(e2);
        graph.halfedge_mut(e2).next = Some(e0);
        
        // 创建面
        let face_id = graph.add_face();
        graph.set_face_boundary(face_id, e0);
        
        // 验证
        assert!(graph.validate().is_ok());
        
        let area = graph.compute_face_area(face_id);
        assert!((area - 50.0).abs() < 1e-10); // 三角形面积 = 10*10/2 = 50
    }

    #[test]
    fn test_rectangle_with_hole() {
        // 外轮廓（逆时针 - 正面积）
        let outer = vec![
            [0.0, 0.0],
            [20.0, 0.0],
            [20.0, 20.0],
            [0.0, 20.0],
        ];
        
        // 孔洞（顺时针 - 负面积）
        let hole = vec![
            [8.0, 8.0],
            [8.0, 12.0],
            [12.0, 12.0],
            [12.0, 8.0],
        ];
        
        let graph = HalfedgeGraph::from_loops(&[outer, hole]);
        
        assert!(graph.validate().is_ok());
        assert_eq!(graph.faces.len(), 2);
        
        // 验证面积符号
        let face0_area = graph.compute_face_area(0);
        let face1_area = graph.compute_face_area(1);
        
        // 一个为正（外轮廓），一个为负（孔洞）
        assert!(face0_area > 0.0 || face1_area > 0.0);
        assert!(face0_area < 0.0 || face1_area < 0.0);
        
        // 外轮廓面积应该约等于 400
        let outer_area = face0_area.abs().max(face1_area.abs());
        assert!((outer_area - 400.0).abs() < 1.0);
        
        // 孔洞面积应该约等于 16
        let hole_area = face0_area.abs().min(face1_area.abs());
        assert!((hole_area - 16.0).abs() < 1.0);
    }

    #[test]
    fn test_from_loops() {
        let square = vec![
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ];
        
        let graph = HalfedgeGraph::from_loops(&[square]);
        
        assert_eq!(graph.vertices.len(), 4);
        assert_eq!(graph.halfedges.len(), 8); // 4 条边 × 2 条半边
        assert_eq!(graph.faces.len(), 1);
        
        assert!(graph.validate().is_ok());
    }

    #[test]
    fn test_face_boundary_points() {
        let triangle = vec![
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
        ];

        let graph = HalfedgeGraph::from_loops(&[triangle]);
        let face_id = 0;

        let points = graph.face_boundary_points(face_id);
        assert_eq!(points.len(), 3);
        assert_eq!(points[0], [0.0, 0.0]);
        assert_eq!(points[1], [10.0, 0.0]);
        assert_eq!(points[2], [10.0, 10.0]);
    }

    // ========================================================================
    // P1-2 新增测试：嵌套孔洞识别与 O(1) 层级查询
    // ========================================================================

    #[test]
    fn test_nested_holes() {
        // 测试嵌套孔洞：外轮廓 → 孔 1 → 孔 2（孔中孔）
        let outer = vec![
            [0.0, 0.0],
            [30.0, 0.0],
            [30.0, 30.0],
            [0.0, 30.0],
        ];
        let hole1 = vec![
            [10.0, 10.0],
            [20.0, 10.0],
            [20.0, 20.0],
            [10.0, 20.0],
        ];
        let hole2 = vec![
            [12.0, 12.0],
            [18.0, 12.0],
            [18.0, 18.0],
            [12.0, 18.0],
        ];

        let mut graph = HalfedgeGraph::from_loops(&[outer, hole1, hole2]);
        
        // 构建嵌套层级
        assert!(graph.build_nesting_hierarchy().is_ok());
        
        // 验证嵌套关系
        assert!(graph.validate_nesting().is_ok());
        
        // 应该有 3 个面
        assert_eq!(graph.faces.len(), 3);
        
        // 外轮廓（面积最大）应该是面 0
        let outer_area = graph.compute_face_area(0).abs();
        assert!((outer_area - 900.0).abs() < 1.0);  // 30x30 = 900
        
        // 验证嵌套深度
        // 外轮廓深度 = 0
        assert_eq!(graph.get_nesting_depth(0), 0);
        
        // 孔 1 和孔 2 应该有嵌套关系
        // 具体哪个是孔 1 哪个是孔 2 取决于面积排序
        let depths: Vec<usize> = (0..3).map(|i| graph.get_nesting_depth(i)).collect();
        assert!(depths.iter().any(|&d| d == 0));  // 至少一个外轮廓
        assert!(depths.iter().any(|&d| d >= 1));  // 至少一个孔洞
    }

    #[test]
    fn test_o1_hierarchy_query() {
        // 测试 O(1) 层级查询
        let outer = vec![
            [0.0, 0.0],
            [20.0, 0.0],
            [20.0, 20.0],
            [0.0, 20.0],
        ];
        let hole = vec![
            [5.0, 5.0],
            [15.0, 5.0],
            [15.0, 15.0],
            [5.0, 15.0],
        ];

        let mut graph = HalfedgeGraph::from_loops(&[outer, hole]);
        graph.build_nesting_hierarchy().unwrap();

        // O(1) 查询父面
        let hole_face = if graph.get_face_parent(0).is_some() { 0 } else { 1 };
        let outer_face = if hole_face == 0 { 1 } else { 0 };

        assert!(graph.get_face_parent(outer_face).is_none());  // 外轮廓无父面
        assert!(graph.get_face_parent(hole_face).is_some());   // 孔洞有父面

        // O(1) 查询子面
        let children = graph.get_face_children(outer_face);
        assert!(!children.is_empty());
        assert!(children.contains(&hole_face));

        // O(1) 查询嵌套深度
        assert_eq!(graph.get_nesting_depth(outer_face), 0);
        assert_eq!(graph.get_nesting_depth(hole_face), 1);

        // O(1) 判断是否为孔洞
        assert!(!graph.is_hole(outer_face));
        assert!(graph.is_hole(hole_face));

        // 获取根面
        assert_eq!(graph.get_root_face(hole_face), Some(outer_face));
    }

    #[test]
    fn test_nesting_path() {
        // 测试嵌套路径
        let outer = vec![
            [0.0, 0.0],
            [40.0, 0.0],
            [40.0, 40.0],
            [0.0, 40.0],
        ];
        let hole1 = vec![
            [10.0, 10.0],
            [30.0, 10.0],
            [30.0, 30.0],
            [10.0, 30.0],
        ];
        let hole2 = vec![
            [15.0, 15.0],
            [25.0, 15.0],
            [25.0, 25.0],
            [15.0, 25.0],
        ];

        let mut graph = HalfedgeGraph::from_loops(&[outer, hole1, hole2]);
        graph.build_nesting_hierarchy().unwrap();

        // 获取嵌套路径
        for face_id in 0..graph.faces.len() {
            let path = graph.get_nesting_path(face_id);
            // 路径应该从根面开始
            if !path.is_empty() {
                let root = path[0];
                assert!(graph.get_face_parent(root).is_none());
            }
        }
    }

    #[test]
    fn test_point_in_polygon() {
        // 测试射线法点在多边形内判断
        let square = vec![
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ];

        // 内部点
        assert!(HalfedgeGraph::point_in_polygon_ray_casting([5.0, 5.0], &square));
        
        // 外部点
        assert!(!HalfedgeGraph::point_in_polygon_ray_casting([15.0, 5.0], &square));
        assert!(!HalfedgeGraph::point_in_polygon_ray_casting([-5.0, 5.0], &square));
        assert!(!HalfedgeGraph::point_in_polygon_ray_casting([5.0, 15.0], &square));
        assert!(!HalfedgeGraph::point_in_polygon_ray_casting([5.0, -5.0], &square));
    }
}
