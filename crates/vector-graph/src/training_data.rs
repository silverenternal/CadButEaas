//! GNN 训练数据生成
//!
//! 将 vectorize 的合成训练数据转换为图结构用于 GNN 训练

#[cfg(feature = "pytorch")]
use std::path::Path;
#[cfg(feature = "pytorch")]
use vectorize::test_data::{
    generate_training_dataset, generate_training_sample, AugmentationConfig, DrawingType,
    TrainingSample,
};

#[cfg(feature = "pytorch")]
use crate::{CadEdge, CadGraph, EdgeType, NodeType};

#[cfg(feature = "pytorch")]
use crate::{DomainType, GeometryType, SemanticType};

/// 带标签的图训练样本
#[derive(Debug, Clone)]
#[cfg(feature = "pytorch")]
pub struct GraphTrainingSample {
    /// 图结构
    pub graph: CadGraph,
    /// 几何类型标签（每条边）
    pub geometry_labels: Vec<GeometryType>,
    /// 领域类型标签（图级别）
    pub domain_label: DomainType,
    /// 语义类型标签（每条边）
    pub semantic_labels: Vec<SemanticType>,
    /// 质量等级
    pub quality_level: String,
}

/// 将 DrawingType 转换为 DomainType
#[cfg(feature = "pytorch")]
fn drawing_to_domain(dtype: DrawingType) -> DomainType {
    match dtype {
        DrawingType::Architectural => DomainType::Architectural,
        DrawingType::Mechanical => DomainType::Mechanical,
        DrawingType::Circuit => DomainType::Circuit,
        DrawingType::HandDrawn => DomainType::General,
    }
}

/// 将 TrainingSample 转换为 CadGraph 用于 GNN 训练
#[cfg(feature = "pytorch")]
pub fn training_sample_to_graph(sample: &TrainingSample) -> GraphTrainingSample {
    let mut graph = CadGraph::new(1.0); // 1 像素容差

    // 从 line_coords 构建节点和边
    for &((x1, y1), (x2, y2)) in &sample.line_coords {
        // 添加或获取节点
        let idx1 = graph.get_or_create_node(x1 as f64, y1 as f64, NodeType::Endpoint);
        let idx2 = graph.get_or_create_node(x2 as f64, y2 as f64, NodeType::Endpoint);

        // 添加边
        let edge = CadEdge {
            edge_type: EdgeType::Line,
            length: (((x2 as f64 - x1 as f64).powi(2) + (y2 as f64 - y1 as f64).powi(2)) as f64)
                .sqrt(),
            angle: (y2 as f64 - y1 as f64).atan2(x2 as f64 - x1 as f64),
            polyline_id: None,
            parallelism_score: 0.0,
            collinearity_score: 0.0,
            avg_neighbor_distance: 0.0,
        };
        graph.add_edge(idx1, idx2, edge);
    }

    // 根据节点度更新节点类型
    graph.update_node_types_by_degree();

    // 根据图纸类型推断标签
    let domain_label = drawing_to_domain(sample.drawing_type);
    let default_geometry = GeometryType::Line;
    let default_semantic = match sample.drawing_type {
        DrawingType::Architectural => SemanticType::Wall,
        DrawingType::Mechanical => SemanticType::Outline,
        _ => SemanticType::Other,
    };

    let num_edges = graph.edges().len();
    let geometry_labels = vec![default_geometry; num_edges];
    let semantic_labels = vec![default_semantic; num_edges];

    GraphTrainingSample {
        graph,
        geometry_labels,
        domain_label,
        semantic_labels,
        quality_level: sample.quality_level.clone(),
    }
}

/// PyTorch 数据集导出器
#[cfg(feature = "pytorch")]
pub struct DatasetExporter;

#[cfg(feature = "pytorch")]
impl DatasetExporter {
    /// 导出单个样本到目录（GraphML + 元数据）
    pub fn export_sample(
        sample: &GraphTrainingSample,
        output_dir: &Path,
        index: usize,
    ) -> std::io::Result<()> {
        std::fs::create_dir_all(output_dir)?;

        // 导出 GraphML
        let graphml_path = output_dir.join(format!("sample_{:06}.graphml", index));
        sample.graph.export_graphml(&graphml_path)?;

        // 导出标签元数据（JSON）
        let metadata = serde_json::json!({
            "index": index,
            "domain_label": format!("{:?}", sample.domain_label),
            "geometry_labels": sample.geometry_labels.iter().map(|g| format!("{:?}", g)).collect::<Vec<_>>(),
            "semantic_labels": sample.semantic_labels.iter().map(|s| format!("{:?}", s)).collect::<Vec<_>>(),
            "quality_level": sample.quality_level,
            "num_nodes": sample.graph.nodes().len(),
            "num_edges": sample.graph.edges().len(),
        });

        let metadata_path = output_dir.join(format!("sample_{:06}.json", index));
        std::fs::write(&metadata_path, serde_json::to_string_pretty(&metadata)?)?;

        Ok(())
    }

    /// 导出整个数据集
    pub fn export_dataset(
        samples: &[GraphTrainingSample],
        output_dir: &Path,
    ) -> std::io::Result<()> {
        std::fs::create_dir_all(output_dir)?;

        // 导出每个样本
        for (i, sample) in samples.iter().enumerate() {
            Self::export_sample(sample, output_dir, i)?;
        }

        // 导出数据集索引和统计
        let dataset_stats = serde_json::json!({
            "total_samples": samples.len(),
            "domain_distribution": {
                "Architectural": samples.iter().filter(|s| matches!(s.domain_label, DomainType::Architectural)).count(),
                "Mechanical": samples.iter().filter(|s| matches!(s.domain_label, DomainType::Mechanical)).count(),
                "Circuit": samples.iter().filter(|s| matches!(s.domain_label, DomainType::Circuit)).count(),
                "General": samples.iter().filter(|s| matches!(s.domain_label, DomainType::General)).count(),
            },
            "avg_nodes_per_graph": samples.iter().map(|s| s.graph.nodes().len()).sum::<usize>() as f64 / samples.len().max(1) as f64,
            "avg_edges_per_graph": samples.iter().map(|s| s.graph.edges().len()).sum::<usize>() as f64 / samples.len().max(1) as f64,
        });

        let stats_path = output_dir.join("dataset_stats.json");
        std::fs::write(&stats_path, serde_json::to_string_pretty(&dataset_stats)?)?;

        // 导出 PyTorch Geometric 加载脚本
        let loader_script = r#"import os
import json
import torch
from torch_geometric.data import Data, Dataset
import xml.etree.ElementTree as ET

class CadGraphDataset(Dataset):
    """CAD 图纸 GNN 训练数据集"""

    def __init__(self, root_dir, transform=None):
        super().__init__(root_dir, transform)
        self.root_dir = root_dir
        self.samples = sorted([f for f in os.listdir(root_dir) if f.endswith('.graphml')])

    def len(self):
        return len(self.samples)

    def get(self, idx):
        graphml_file = self.samples[idx]
        json_file = graphml_file.replace('.graphml', '.json')

        # 加载图结构 (简化的 GraphML 解析)
        tree = ET.parse(os.path.join(self.root_dir, graphml_file))
        root = tree.getroot()

        # 加载标签
        with open(os.path.join(self.root_dir, json_file)) as f:
            meta = json.load(f)

        # 返回 PyG Data 对象
        return Data(
            num_nodes=meta['num_nodes'],
            domain_label=meta['domain_label'],
            quality=meta['quality_level'],
        )
"#;
        let loader_path = output_dir.join("dataset_loader.py");
        std::fs::write(&loader_path, loader_script)?;

        Ok(())
    }
}

/// 简化的端到端：生成数据集 → 转换为图 → 导出
#[cfg(feature = "pytorch")]
pub fn generate_and_export_graph_dataset(
    num_samples: usize,
    width: u32,
    height: u32,
    output_dir: &Path,
) -> std::io::Result<()> {
    // 生成图像训练样本
    let image_samples = generate_training_dataset(num_samples, width, height);

    // 转换为图训练样本
    let graph_samples: Vec<GraphTrainingSample> = image_samples
        .iter()
        .map(|s| training_sample_to_graph(s))
        .collect();

    // 导出
    DatasetExporter::export_dataset(&graph_samples, output_dir)?;

    println!(
        "成功导出 {} 个训练样本到 {}",
        graph_samples.len(),
        output_dir.display()
    );

    Ok(())
}

/// 创建用于小样本学习的基准数据集
/// 每种领域类型生成平衡的数据集
#[cfg(feature = "pytorch")]
pub fn generate_few_shot_benchmark(
    shots_per_class: usize,
    width: u32,
    height: u32,
    output_dir: &Path,
) -> std::io::Result<()> {
    let drawing_types = [
        DrawingType::Architectural,
        DrawingType::Mechanical,
        DrawingType::Circuit,
        DrawingType::HandDrawn,
    ];

    let aug_config = AugmentationConfig::default();

    let mut all_samples = Vec::new();

    for &dtype in &drawing_types {
        for _ in 0..shots_per_class {
            let sample = generate_training_sample(dtype, width, height, &aug_config);
            all_samples.push(training_sample_to_graph(&sample));
        }
    }

    DatasetExporter::export_dataset(&all_samples, output_dir)?;

    println!(
        "小样本基准数据集已生成: {} 类 × {} = {} 样本",
        drawing_types.len(),
        shots_per_class,
        all_samples.len()
    );

    Ok(())
}

#[cfg(test)]
#[cfg(feature = "pytorch")]
mod tests {
    use super::*;

    #[test]
    fn test_training_sample_to_graph() {
        let aug_config = AugmentationConfig::default();
        let sample = generate_training_sample(DrawingType::Architectural, 256, 256, &aug_config);
        let graph_sample = training_sample_to_graph(&sample);

        assert!(graph_sample.graph.nodes().len() >= 2);
        assert!(graph_sample.graph.edges().len() >= 1);
        assert_eq!(
            graph_sample.geometry_labels.len(),
            graph_sample.graph.edges().len()
        );
        assert_eq!(
            graph_sample.semantic_labels.len(),
            graph_sample.graph.edges().len()
        );
    }

    #[test]
    fn test_export_graph_sample_to_temp() {
        let aug_config = AugmentationConfig::default();
        let sample = generate_training_sample(DrawingType::Mechanical, 256, 256, &aug_config);
        let graph_sample = training_sample_to_graph(&sample);

        let temp_dir = std::env::temp_dir().join("cad_graph_test_export");
        let _ = std::fs::remove_dir_all(&temp_dir);

        DatasetExporter::export_sample(&graph_sample, &temp_dir, 0).unwrap();

        assert!(temp_dir.join("sample_000000.graphml").exists());
        assert!(temp_dir.join("sample_000000.json").exists());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }

    #[test]
    fn test_few_shot_benchmark_generation() {
        let temp_dir = std::env::temp_dir().join("cad_few_shot_test");
        let _ = std::fs::remove_dir_all(&temp_dir);

        generate_few_shot_benchmark(2, 256, 256, &temp_dir).unwrap();

        // 4 类 × 2 = 8 样本
        for i in 0..8 {
            assert!(temp_dir.join(format!("sample_{:06}.graphml", i)).exists());
        }
        assert!(temp_dir.join("dataset_stats.json").exists());
        assert!(temp_dir.join("dataset_loader.py").exists());

        let _ = std::fs::remove_dir_all(&temp_dir);
    }
}
