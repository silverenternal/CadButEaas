//! 图特征提取器
//!
//! 从 CAD 图中提取节点和边的特征向量，用于 GNN 输入

use crate::graph::{CadEdge, CadGraph, CadNode, EdgeType};
use nalgebra::DMatrix;

/// 节点特征提取器
pub trait NodeFeatureExtractor {
    /// 提取单个节点的特征
    fn extract_node(&self, node: &CadNode, graph: &CadGraph) -> Vec<f32>;

    /// 特征维度
    fn feature_dim(&self) -> usize;
}

/// 边特征提取器
pub trait EdgeFeatureExtractor {
    /// 提取单个边的特征
    fn extract_edge(&self, edge: &CadEdge, graph: &CadGraph) -> Vec<f32>;

    /// 特征维度
    fn feature_dim(&self) -> usize;
}

/// 几何特征提取器
#[derive(Debug, Clone)]
pub struct GeometryExtractor {
    /// 是否归一化坐标
    pub normalize_coords: bool,
}

impl Default for GeometryExtractor {
    fn default() -> Self {
        Self {
            normalize_coords: true,
        }
    }
}

impl GeometryExtractor {
    /// 创建新的几何特征提取器
    pub fn new(normalize_coords: bool) -> Self {
        Self { normalize_coords }
    }

    /// 计算坐标归一化参数
    fn compute_bounds(&self, graph: &CadGraph) -> (f64, f64, f64, f64) {
        let nodes = graph.nodes();
        if nodes.is_empty() {
            return (0.0, 0.0, 1.0, 1.0);
        }

        let mut min_x = f64::MAX;
        let mut max_x = f64::MIN;
        let mut min_y = f64::MAX;
        let mut max_y = f64::MIN;

        for node in nodes {
            min_x = min_x.min(node.x);
            max_x = max_x.max(node.x);
            min_y = min_y.min(node.y);
            max_y = max_y.max(node.y);
        }

        (min_x, min_y, max_x, max_y)
    }
}

impl NodeFeatureExtractor for GeometryExtractor {
    fn extract_node(&self, node: &CadNode, graph: &CadGraph) -> Vec<f32> {
        let mut features = Vec::new();

        // 坐标（归一化或原始）
        let (x, y) = if self.normalize_coords {
            let (min_x, min_y, max_x, max_y) = self.compute_bounds(graph);
            let range_x = (max_x - min_x).max(1e-6);
            let range_y = (max_y - min_y).max(1e-6);
            (
                ((node.x - min_x) / range_x) as f32,
                ((node.y - min_y) / range_y) as f32,
            )
        } else {
            (node.x as f32, node.y as f32)
        };
        features.push(x);
        features.push(y);

        // 节点类型的 one-hot 编码 (5 维)
        features.extend_from_slice(&node.node_type.to_one_hot());

        // 度数（归一化）
        features.push((node.degree as f32) / 10.0); // 假设最大度数为 10

        // 所属多段线数量
        features.push((node.polyline_ids.len() as f32) / 5.0); // 假设最多属于 5 条多段线

        // 局部曲率（归一化）
        features.push((node.curvature as f32).min(1.0));

        // 线宽（归一化）
        features.push((node.line_width as f32) / 10.0); // 假设最大线宽为 10

        features
    }

    fn feature_dim(&self) -> usize {
        2 + 5 + 1 + 1 + 1 + 1 // 坐标 + 类型 + 度数 + 多段线数量 + 曲率 + 线宽
    }
}

impl EdgeFeatureExtractor for GeometryExtractor {
    fn extract_edge(&self, edge: &CadEdge, _graph: &CadGraph) -> Vec<f32> {
        let mut features = Vec::new();

        // 边类型的 one-hot 编码
        match edge.edge_type {
            EdgeType::Line => {
                features.push(1.0);
                features.push(0.0);
                features.push(0.0);
                features.push(0.0);
            }
            EdgeType::Arc => {
                features.push(0.0);
                features.push(1.0);
                features.push(0.0);
                features.push(0.0);
            }
            EdgeType::Adjacent => {
                features.push(0.0);
                features.push(0.0);
                features.push(1.0);
                features.push(0.0);
            }
            EdgeType::Intersecting => {
                features.push(0.0);
                features.push(0.0);
                features.push(0.0);
                features.push(1.0);
            }
        }

        // 长度（归一化，假设最大 1000 像素）
        features.push((edge.length as f32) / 1000.0);

        // 角度特征（edge.angle 是 f64，计算后转 f32）
        let angle_f32 = edge.angle as f32;
        features.push((angle_f32.cos() + 1.0) / 2.0); // 归一化到 [0, 1]
        features.push((angle_f32.sin() + 1.0) / 2.0);

        // 是否属于某条多段线
        features.push(if edge.polyline_id.is_some() { 1.0 } else { 0.0 });

        // 平行度得分 [0, 1]
        features.push(edge.parallelism_score as f32);

        // 共线性得分 [0, 1]
        features.push(edge.collinearity_score as f32);

        // 平均邻边距离（归一化）
        features.push((edge.avg_neighbor_distance as f32) / 100.0);

        features
    }

    fn feature_dim(&self) -> usize {
        4 + 1 + 2 + 1 + 3 // 类型 + 长度 + 角度 + 所属关系 + 平行度 + 共线性 + 距离
    }
}

/// 通用特征提取器组合
#[derive(Debug, Clone, Default)]
pub struct FeatureExtractor {
    geometry: GeometryExtractor,
}

impl FeatureExtractor {
    /// 创建新的特征提取器
    pub fn new(geometry: GeometryExtractor) -> Self {
        Self { geometry }
    }

    /// 提取所有节点特征为矩阵
    pub fn extract_node_features(&self, graph: &CadGraph) -> DMatrix<f32> {
        let dim = NodeFeatureExtractor::feature_dim(&self.geometry);
        let n_nodes = graph.node_count();

        let mut features = DMatrix::zeros(n_nodes, dim);

        for (i, node) in graph.nodes().iter().enumerate() {
            let node_features = self.geometry.extract_node(node, graph);
            for (j, &val) in node_features.iter().enumerate() {
                features[(i, j)] = val;
            }
        }

        features
    }

    /// 提取所有边特征为矩阵
    pub fn extract_edge_features(&self, graph: &CadGraph) -> DMatrix<f32> {
        let dim = EdgeFeatureExtractor::feature_dim(&self.geometry);
        let n_edges = graph.edge_count();

        let mut features = DMatrix::zeros(n_edges, dim);

        for (i, edge) in graph.edges().iter().enumerate() {
            let edge_features = self.geometry.extract_edge(edge, graph);
            for (j, &val) in edge_features.iter().enumerate() {
                features[(i, j)] = val;
            }
        }

        features
    }

    /// 构建邻接矩阵（COO 格式，用于 GNN）
    pub fn adjacency_matrix_coo(&self, graph: &CadGraph) -> (Vec<usize>, Vec<usize>) {
        let mut src = Vec::new();
        let mut dst = Vec::new();

        for edge_idx in graph.graph.edge_indices() {
            let (source, target) = graph.graph.edge_endpoints(edge_idx).unwrap();
            src.push(source.index());
            dst.push(target.index());
        }

        (src, dst)
    }

    /// 节点特征维度
    pub fn node_feature_dim(&self) -> usize {
        NodeFeatureExtractor::feature_dim(&self.geometry)
    }

    /// 边特征维度
    pub fn edge_feature_dim(&self) -> usize {
        EdgeFeatureExtractor::feature_dim(&self.geometry)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::CadGraph;

    #[test]
    fn test_node_feature_extraction() {
        let polyline = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
        let graph = CadGraph::from_polylines(&[polyline], 1e-3);

        let extractor = GeometryExtractor::default();
        let features = extractor.extract_node(graph.nodes()[0], &graph);

        assert_eq!(
            features.len(),
            NodeFeatureExtractor::feature_dim(&extractor)
        );
        assert_eq!(NodeFeatureExtractor::feature_dim(&extractor), 11);
    }

    #[test]
    fn test_edge_feature_extraction() {
        let polyline = vec![[0.0, 0.0], [1.0, 0.0]];
        let graph = CadGraph::from_polylines(&[polyline], 1e-3);

        let extractor = GeometryExtractor::default();
        let features = extractor.extract_edge(graph.edges()[0], &graph);

        assert_eq!(
            features.len(),
            EdgeFeatureExtractor::feature_dim(&extractor)
        );
        assert_eq!(EdgeFeatureExtractor::feature_dim(&extractor), 11);
    }

    #[test]
    fn test_feature_extractor_node_matrix() {
        let polyline = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let graph = CadGraph::from_polylines(&[polyline], 1e-3);

        let extractor = FeatureExtractor::default();
        let node_features = extractor.extract_node_features(&graph);

        assert_eq!(node_features.nrows(), 4); // 4 个节点
        assert_eq!(node_features.ncols(), 11); // 11 维特征
    }

    #[test]
    fn test_adjacency_matrix_coo() {
        let polyline = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
        let graph = CadGraph::from_polylines(&[polyline], 1e-3);

        let extractor = FeatureExtractor::default();
        let (src, dst) = extractor.adjacency_matrix_coo(&graph);

        assert_eq!(src.len(), 2); // 2 条边
        assert_eq!(dst.len(), 2);
    }
}
