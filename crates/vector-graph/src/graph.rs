//! CAD 图结构定义
//!
//! 将矢量化的多段线转换为图结构：
//! - 节点：端点、交点
//! - 边：线段连接关系 + 几何属性

use std::collections::HashMap;

use common_types::Polyline;
use petgraph::graph::{Graph, NodeIndex};
use petgraph::Directed;

/// 节点类型 - 基于连接度的连接分类
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum NodeType {
    /// 端点 (度数=1)
    Endpoint,
    /// L型连接 (度数=2)
    LShape,
    /// T型连接 (度数=3)
    TShape,
    /// X型连接 (度数=4)
    XShape,
    /// 多路交叉 (度数>4)
    MultiJunction,
    /// 中间点（沿线上的点）
    Midpoint,
}

impl NodeType {
    /// 根据度数推断节点类型
    pub fn from_degree(degree: usize) -> Self {
        match degree {
            0 | 1 => NodeType::Endpoint,
            2 => NodeType::LShape,
            3 => NodeType::TShape,
            4 => NodeType::XShape,
            _ => NodeType::MultiJunction,
        }
    }

    /// 转换为 one-hot 编码向量 (5 维)
    pub fn to_one_hot(&self) -> [f32; 5] {
        match self {
            NodeType::Endpoint => [1.0, 0.0, 0.0, 0.0, 0.0],
            NodeType::LShape => [0.0, 1.0, 0.0, 0.0, 0.0],
            NodeType::TShape => [0.0, 0.0, 1.0, 0.0, 0.0],
            NodeType::XShape => [0.0, 0.0, 0.0, 1.0, 0.0],
            NodeType::MultiJunction => [0.0, 0.0, 0.0, 0.0, 1.0],
            NodeType::Midpoint => [0.5, 0.5, 0.0, 0.0, 0.0], // 混合表示
        }
    }
}

/// 边类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum EdgeType {
    /// 直线段连接
    Line,
    /// 圆弧连接
    Arc,
    /// 相邻（共享端点）
    Adjacent,
    /// 相交（交叉）
    Intersecting,
}

/// CAD 图节点
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CadNode {
    /// 坐标位置
    pub x: f64,
    /// 坐标位置
    pub y: f64,
    /// 节点类型
    pub node_type: NodeType,
    /// 连接的边数量（度数）
    pub degree: usize,
    /// 所属多段线 ID
    pub polyline_ids: Vec<usize>,
    /// 局部曲率（转弯程度）
    pub curvature: f64,
    /// 估算的线宽（像素）
    pub line_width: f64,
}

/// CAD 图边
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CadEdge {
    /// 边类型
    pub edge_type: EdgeType,
    /// 长度（像素）
    pub length: f64,
    /// 角度（弧度）
    pub angle: f64,
    /// 所属多段线 ID
    pub polyline_id: Option<usize>,
    /// 与其他边的平行度得分 [0,1]
    pub parallelism_score: f64,
    /// 与其他边的共线性得分 [0,1]
    pub collinearity_score: f64,
    /// 与相邻边的平均距离
    pub avg_neighbor_distance: f64,
}

/// CAD 图结构
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CadGraph {
    /// 内部图表示
    #[serde(skip)]
    pub graph: Graph<CadNode, CadEdge, Directed>,
    /// 坐标到节点索引的映射（用于去重）
    #[serde(skip)]
    pub coord_to_node: HashMap<(i64, i64), NodeIndex>,
    /// 容差阈值（用于点重合判断，像素）
    pub tolerance: f64,
}

impl CadGraph {
    /// 创建空的 CAD 图
    pub fn new(tolerance: f64) -> Self {
        Self {
            graph: Graph::new(),
            coord_to_node: HashMap::new(),
            tolerance,
        }
    }

    /// 获取或创建坐标对应的节点
    pub fn get_or_create_node(&mut self, x: f64, y: f64, node_type: NodeType) -> NodeIndex {
        // 量化坐标以进行哈希比较
        let key = (
            (x / self.tolerance).round() as i64,
            (y / self.tolerance).round() as i64,
        );

        if let Some(&idx) = self.coord_to_node.get(&key) {
            // 更新现有节点度数 - 先计算再借用可变引用
            let degree = self.graph.neighbors(idx).count();
            if let Some(node) = self.graph.node_weight_mut(idx) {
                node.degree = degree;
            }
            return idx;
        }

        let node = CadNode {
            x,
            y,
            node_type,
            degree: 0,
            polyline_ids: Vec::new(),
            curvature: 0.0,
            line_width: 1.0,
        };

        let idx = self.graph.add_node(node);
        self.coord_to_node.insert(key, idx);
        idx
    }

    /// 添加一条边
    pub fn add_edge(&mut self, from: NodeIndex, to: NodeIndex, edge: CadEdge) {
        self.graph.add_edge(from, to, edge);

        // 更新节点度数 - 先计算度数再获取可变引用
        let from_degree = self.graph.neighbors(from).count();
        let to_degree = self.graph.neighbors(to).count();

        if let Some(node) = self.graph.node_weight_mut(from) {
            node.degree = from_degree;
        }
        if let Some(node) = self.graph.node_weight_mut(to) {
            node.degree = to_degree;
        }
    }

    /// 从多段线集合构建 CAD 图
    pub fn from_polylines(polylines: &[Polyline], tolerance: f64) -> Self {
        let mut graph = Self::new(tolerance);

        for (poly_id, polyline) in polylines.iter().enumerate() {
            if polyline.len() < 2 {
                continue;
            }

            // 添加所有点作为节点
            let mut node_indices = Vec::new();
            for (i, point) in polyline.iter().enumerate() {
                let node_type = if i == 0 || i == polyline.len() - 1 {
                    NodeType::Endpoint
                } else {
                    NodeType::Midpoint
                };

                let idx = graph.get_or_create_node(point[0], point[1], node_type);

                // 记录所属多段线
                if let Some(node) = graph.graph.node_weight_mut(idx) {
                    if !node.polyline_ids.contains(&poly_id) {
                        node.polyline_ids.push(poly_id);
                    }
                }

                node_indices.push(idx);
            }

            // 在相邻点之间添加边
            for window in node_indices.windows(2) {
                let from = window[0];
                let to = window[1];

                let from_node = &graph.graph[from];
                let to_node = &graph.graph[to];

                let dx = to_node.x - from_node.x;
                let dy = to_node.y - from_node.y;
                let length = (dx * dx + dy * dy).sqrt();
                let angle = dy.atan2(dx);

                let edge = CadEdge {
                    edge_type: EdgeType::Line,
                    length,
                    angle,
                    polyline_id: Some(poly_id),
                    parallelism_score: 0.0,
                    collinearity_score: 0.0,
                    avg_neighbor_distance: 0.0,
                };

                graph.add_edge(from, to, edge);
            }
        }

        // 检测相邻关系（共享端点的线段）
        graph.detect_adjacent_connections();

        // 根据连接度数更新节点类型
        graph.update_node_types_by_degree();

        graph
    }

    /// 检测相邻连接关系（共享端点的线段之间添加边）
    fn detect_adjacent_connections(&mut self) {
        let node_indices: Vec<NodeIndex> = self.graph.node_indices().collect();

        for &node_idx in &node_indices {
            let neighbors: Vec<NodeIndex> = self.graph.neighbors(node_idx).collect();

            // 如果一个节点连接了多条边，在这些边的另一端之间建立相邻关系
            if neighbors.len() >= 2 {
                for i in 0..neighbors.len() {
                    for j in (i + 1)..neighbors.len() {
                        let a = neighbors[i];
                        let b = neighbors[j];

                        // 检查是否已有边
                        if !self.graph.contains_edge(a, b) && !self.graph.contains_edge(b, a) {
                            let node_a = &self.graph[a];
                            let node_b = &self.graph[b];

                            let dx = node_b.x - node_a.x;
                            let dy = node_b.y - node_a.y;
                            let length = (dx * dx + dy * dy).sqrt();
                            let angle = dy.atan2(dx);

                            let edge = CadEdge {
                                edge_type: EdgeType::Adjacent,
                                length,
                                angle,
                                polyline_id: None,
                                parallelism_score: 0.0,
                                collinearity_score: 0.0,
                                avg_neighbor_distance: 0.0,
                            };

                            self.add_edge(a, b, edge);
                        }
                    }
                }
            }
        }
    }

    /// 获取节点数量
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// 获取边数量
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// 获取所有节点
    pub fn nodes(&self) -> Vec<&CadNode> {
        self.graph.node_weights().collect()
    }

    /// 获取所有边
    pub fn edges(&self) -> Vec<&CadEdge> {
        self.graph.edge_weights().collect()
    }

    /// 根据度数更新所有节点的类型
    pub fn update_node_types_by_degree(&mut self) {
        let node_indices: Vec<NodeIndex> = self.graph.node_indices().collect();
        for idx in node_indices {
            let degree = self.graph.neighbors(idx).count();
            if let Some(node) = self.graph.node_weight_mut(idx) {
                node.degree = degree;
                // 保留 Midpoint 类型，其他根据度数更新
                if node.node_type != NodeType::Midpoint {
                    node.node_type = NodeType::from_degree(degree);
                }
            }
        }
    }

    /// 计算图的统计信息
    pub fn statistics(&self) -> GraphStatistics {
        let degrees: Vec<usize> = self.graph.node_weights().map(|n| n.degree).collect();

        let avg_degree = if degrees.is_empty() {
            0.0
        } else {
            degrees.iter().sum::<usize>() as f64 / degrees.len() as f64
        };

        let max_degree = degrees.iter().max().copied().unwrap_or(0);

        let endpoint_count = self
            .graph
            .node_weights()
            .filter(|n| n.node_type == NodeType::Endpoint)
            .count();

        let lshape_count = self
            .graph
            .node_weights()
            .filter(|n| n.node_type == NodeType::LShape)
            .count();

        let tshape_count = self
            .graph
            .node_weights()
            .filter(|n| n.node_type == NodeType::TShape)
            .count();

        let xshape_count = self
            .graph
            .node_weights()
            .filter(|n| n.node_type == NodeType::XShape)
            .count();

        let multi_junction_count = self
            .graph
            .node_weights()
            .filter(|n| n.node_type == NodeType::MultiJunction)
            .count();

        let line_count = self
            .graph
            .edge_weights()
            .filter(|e| e.edge_type == EdgeType::Line)
            .count();

        let adjacent_count = self
            .graph
            .edge_weights()
            .filter(|e| e.edge_type == EdgeType::Adjacent)
            .count();

        GraphStatistics {
            node_count: self.node_count(),
            edge_count: self.edge_count(),
            avg_degree,
            max_degree,
            endpoint_count,
            lshape_count,
            tshape_count,
            xshape_count,
            multi_junction_count,
            line_count,
            adjacent_count,
        }
    }
}

/// 图统计信息
#[derive(Debug, Clone, serde::Serialize)]
pub struct GraphStatistics {
    pub node_count: usize,
    pub edge_count: usize,
    pub avg_degree: f64,
    pub max_degree: usize,
    pub endpoint_count: usize,
    pub lshape_count: usize,
    pub tshape_count: usize,
    pub xshape_count: usize,
    pub multi_junction_count: usize,
    pub line_count: usize,
    pub adjacent_count: usize,
}

/// GraphML 导出支持
impl CadGraph {
    /// 将图导出为 GraphML 格式字符串
    pub fn to_graphml(&self) -> String {
        let mut output = String::new();

        // XML 头部
        output.push_str(
            r#"<?xml version="1.0" encoding="UTF-8"?>
"#,
        );
        output.push_str(
            r#"<graphml xmlns="http://graphml.graphdrawing.org/xmlns"
"#,
        );
        output.push_str(
            r#"    xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
"#,
        );
        output.push_str(
            r#"    xsi:schemaLocation="http://graphml.graphdrawing.org/xmlns
"#,
        );
        output.push_str(
            r#"    http://graphml.graphdrawing.org/xmlns/1.0/graphml.xsd">
"#,
        );

        // 定义节点属性
        output.push_str(
            r#"  <key id="d0" for="node" attr.name="x" attr.type="double"/>
"#,
        );
        output.push_str(
            r#"  <key id="d1" for="node" attr.name="y" attr.type="double"/>
"#,
        );
        output.push_str(
            r#"  <key id="d2" for="node" attr.name="node_type" attr.type="string"/>
"#,
        );
        output.push_str(
            r#"  <key id="d3" for="node" attr.name="degree" attr.type="int"/>
"#,
        );
        output.push_str(
            r#"  <key id="d4" for="node" attr.name="curvature" attr.type="double"/>
"#,
        );
        output.push_str(
            r#"  <key id="d5" for="node" attr.name="line_width" attr.type="double"/>
"#,
        );

        // 定义边属性
        output.push_str(
            r#"  <key id="d6" for="edge" attr.name="edge_type" attr.type="string"/>
"#,
        );
        output.push_str(
            r#"  <key id="d7" for="edge" attr.name="length" attr.type="double"/>
"#,
        );
        output.push_str(
            r#"  <key id="d8" for="edge" attr.name="angle" attr.type="double"/>
"#,
        );
        output.push_str(
            r#"  <key id="d9" for="edge" attr.name="polyline_id" attr.type="int"/>
"#,
        );
        output.push_str(
            r#"  <key id="d10" for="edge" attr.name="parallelism_score" attr.type="double"/>
"#,
        );
        output.push_str(
            r#"  <key id="d11" for="edge" attr.name="collinearity_score" attr.type="double"/>
"#,
        );
        output.push_str(
            r#"  <key id="d12" for="edge" attr.name="avg_neighbor_distance" attr.type="double"/>
"#,
        );

        // 图开始
        output.push_str(
            r#"  <graph id="cad_graph" edgedefault="directed">
"#,
        );

        // 导出节点
        for (idx, node) in self.graph.node_indices().zip(self.graph.node_weights()) {
            output.push_str(&format!(
                r#"    <node id="n{}">
"#,
                idx.index()
            ));
            output.push_str(&format!(
                r#"      <data key="d0">{:.6}</data>
"#,
                node.x
            ));
            output.push_str(&format!(
                r#"      <data key="d1">{:.6}</data>
"#,
                node.y
            ));
            output.push_str(&format!(
                r#"      <data key="d2">{:?}</data>
"#,
                node.node_type
            ));
            output.push_str(&format!(
                r#"      <data key="d3">{}</data>
"#,
                node.degree
            ));
            output.push_str(&format!(
                r#"      <data key="d4">{:.6}</data>
"#,
                node.curvature
            ));
            output.push_str(&format!(
                r#"      <data key="d5">{:.6}</data>
"#,
                node.line_width
            ));
            output.push_str(
                r#"    </node>
"#,
            );
        }

        // 导出边
        for edge_idx in self.graph.edge_indices() {
            let (source, target) = self.graph.edge_endpoints(edge_idx).unwrap();
            let edge = &self.graph[edge_idx];
            output.push_str(&format!(
                r#"    <edge id="e{}" source="n{}" target="n{}">
"#,
                edge_idx.index(),
                source.index(),
                target.index()
            ));
            output.push_str(&format!(
                r#"      <data key="d6">{:?}</data>
"#,
                edge.edge_type
            ));
            output.push_str(&format!(
                r#"      <data key="d7">{:.6}</data>
"#,
                edge.length
            ));
            output.push_str(&format!(
                r#"      <data key="d8">{:.6}</data>
"#,
                edge.angle
            ));
            if let Some(poly_id) = edge.polyline_id {
                output.push_str(&format!(
                    r#"      <data key="d9">{}</data>
"#,
                    poly_id
                ));
            }
            output.push_str(&format!(
                r#"      <data key="d10">{:.6}</data>
"#,
                edge.parallelism_score
            ));
            output.push_str(&format!(
                r#"      <data key="d11">{:.6}</data>
"#,
                edge.collinearity_score
            ));
            output.push_str(&format!(
                r#"      <data key="d12">{:.6}</data>
"#,
                edge.avg_neighbor_distance
            ));
            output.push_str(
                r#"    </edge>
"#,
            );
        }

        output.push_str(
            r#"  </graph>
"#,
        );
        output.push_str(
            r#"</graphml>
"#,
        );

        output
    }

    /// 导出到 GraphML 文件
    pub fn export_graphml(&self, path: &std::path::Path) -> std::io::Result<()> {
        std::fs::write(path, self.to_graphml())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_create_empty_graph() {
        let graph = CadGraph::new(1e-3);
        assert_eq!(graph.node_count(), 0);
        assert_eq!(graph.edge_count(), 0);
    }

    #[test]
    fn test_graph_from_simple_polyline() {
        let polyline = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];

        let graph = CadGraph::from_polylines(&[polyline], 1e-3);

        // 3 个节点，2 条边
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_node_deduplication() {
        // 两条共享端点的线段
        let p1 = vec![[0.0, 0.0], [1.0, 0.0]];
        let p2 = vec![[1.0, 0.0], [2.0, 0.0]];

        let graph = CadGraph::from_polylines(&[p1, p2], 1e-3);

        // 共享一个端点，所以应该是 3 个节点
        assert_eq!(graph.node_count(), 3);
        // 2 条原始边（A->B, B->C）
        // B 有入边和出边，但在有向图中 neighbors() 只返回出边，所以 B 只有 1 个邻居 C
        // 没有添加额外相邻边
        assert_eq!(graph.edge_count(), 2);
    }

    #[test]
    fn test_graph_statistics() {
        let polyline = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
        let graph = CadGraph::from_polylines(&[polyline], 1e-3);
        let stats = graph.statistics();

        assert_eq!(stats.node_count, 3);
        assert_eq!(stats.edge_count, 2);
        assert_eq!(stats.endpoint_count, 2);
    }

    #[test]
    fn test_graphml_export() {
        let polyline = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
        let graph = CadGraph::from_polylines(&[polyline], 1e-3);

        let graphml = graph.to_graphml();

        // 验证基本结构
        assert!(graphml.contains("<graphml"));
        assert!(graphml.contains("<graph"));
        assert!(graphml.contains("<node"));
        assert!(graphml.contains("<edge"));

        // 验证包含所有节点和边
        assert_eq!(graphml.matches("<node").count(), 3);
        assert_eq!(graphml.matches("<edge").count(), 2);

        // 验证包含节点属性
        assert!(graphml.contains("x"));
        assert!(graphml.contains("y"));
        assert!(graphml.contains("node_type"));
        assert!(graphml.contains("degree"));

        // 验证包含边属性
        assert!(graphml.contains("edge_type"));
        assert!(graphml.contains("length"));
        assert!(graphml.contains("angle"));
    }
}
