//! 矢量化图模块
//!
//! 将 CAD 矢量化结果转换为图结构：
//! - 节点：端点、交点
//! - 边：线段、连接关系
//! - 特征：几何属性、拓扑属性
//! - GNN: 基于图神经网络的语义识别

pub mod features;
pub mod graph;
pub mod training_data;

#[cfg(feature = "pytorch")]
pub mod gnn;

pub use features::{
    EdgeFeatureExtractor, FeatureExtractor, GeometryExtractor, NodeFeatureExtractor,
};
pub use graph::{CadEdge, CadGraph, CadNode, EdgeType, GraphStatistics, NodeType};

#[cfg(feature = "pytorch")]
pub use gnn::{
    DeviceType, DomainType, GNNTrainer, GatTransformer, GatTransformerConfig, GcnConfig, GcnLayer,
    GeometryType, GraphAttentionHead, GraphConvolution, ModelError, MultiHeadGraphAttention,
    SemanticClassifier, SemanticType, TrainingConfig, TrainingDataGenerator, TrainingMetrics,
    TrainingSample,
};

#[cfg(feature = "pytorch")]
pub use training_data::{
    generate_and_export_graph_dataset, generate_few_shot_benchmark, training_sample_to_graph,
    DatasetExporter, GraphTrainingSample,
};
