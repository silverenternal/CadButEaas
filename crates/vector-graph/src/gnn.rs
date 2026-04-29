//! 图神经网络模块
//!
//! 基于 tch-rs (PyTorch Rust bindings) 实现的图卷积网络
//! - GraphConvolution: 基础图卷积层
//! - GraphAttentionLayer: 图注意力层 (GAT)
//! - GraphAttentionTransformer: VectorGraphNET 风格的注意力 transformer
//! - SemanticClassifier: 语义分类器（识别图纸元素类型）

use crate::features::FeatureExtractor;
use crate::graph::CadGraph;
use std::path::Path;
use tch::nn::Module;
use tch::nn::OptimizerConfig;
use tch::{nn, Device, IndexOp, Kind, Tensor};

/// 图卷积层
///
/// 实现 Kipf & Welling (2016) 的图卷积网络
pub struct GraphConvolution {
    weight: Tensor,
    bias: Option<Tensor>,
    input_dim: usize,
    output_dim: usize,
}

impl GraphConvolution {
    /// 创建新的图卷积层
    pub fn new(var_store: &nn::Path, input_dim: usize, output_dim: usize, use_bias: bool) -> Self {
        // 使用 Xavier 初始化代替 KaimingUniform（tch 0.23 版本 API 差异）
        let weight = var_store.var(
            "weight",
            &[input_dim as i64, output_dim as i64],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
        );

        let bias = if use_bias {
            Some(var_store.var("bias", &[output_dim as i64], nn::Init::Const(0.0)))
        } else {
            None
        };

        Self {
            weight,
            bias,
            input_dim,
            output_dim,
        }
    }

    /// 前向传播
    ///
    /// # 参数
    /// - features: 节点特征矩阵 [num_nodes, input_dim]
    /// - adjacency: 邻接矩阵 COO 格式的边索引 [2, num_edges]
    pub fn forward(&self, features: &Tensor, adjacency: &Tensor) -> Tensor {
        // 特征变换: X @ W
        let output = features.matmul(&self.weight);

        // 消息传递: A @ (X @ W)
        let num_nodes = features.size()[0];
        let output = sparse_matmul(adjacency, &output, num_nodes);

        // 添加偏置
        if let Some(bias) = &self.bias {
            output + bias
        } else {
            output
        }
    }

    /// 输入维度
    pub fn input_dim(&self) -> usize {
        self.input_dim
    }

    /// 输出维度
    pub fn output_dim(&self) -> usize {
        self.output_dim
    }
}

/// 简单的稀疏矩阵乘法（COO 格式）
fn sparse_matmul(edge_index: &Tensor, x: &Tensor, _num_nodes: i64) -> Tensor {
    let src = edge_index.i((0, ..));
    let dst = edge_index.i((1, ..));

    let num_edges = src.size()[0];
    if num_edges == 0 {
        return x.zeros_like();
    }

    // 聚合邻居特征
    let mut output = Tensor::zeros_like(x);
    let src_features = x.index_select(0, &src);

    let dst_expanded = dst.unsqueeze(1).broadcast_to(src_features.size());
    output = output.scatter_add(0, &dst_expanded, &src_features);

    output
}

// ========== 图注意力层 (GAT) ==========

/// 单头图注意力层
pub struct GraphAttentionHead {
    weight: Tensor,
    attention_weights: Tensor,
    input_dim: usize,
    output_dim: usize,
    dropout: f64,
}

impl GraphAttentionHead {
    /// 创建新的注意力头
    pub fn new(var_store: &nn::Path, input_dim: usize, output_dim: usize, dropout: f64) -> Self {
        let weight = var_store.var(
            "weight",
            &[input_dim as i64, output_dim as i64],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
        );

        let attention_weights = var_store.var(
            "attention",
            &[2 * output_dim as i64, 1],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
        );

        Self {
            weight,
            attention_weights,
            input_dim,
            output_dim,
            dropout,
        }
    }

    /// 前向传播
    pub fn forward(&self, features: &Tensor, edge_index: &Tensor, training: bool) -> Tensor {
        let num_nodes = features.size()[0];

        // 特征变换
        let h = features.matmul(&self.weight); // [N, F']

        // 计算注意力分数
        let src = edge_index.i((0, ..)); // [E]
        let dst = edge_index.i((1, ..)); // [E]

        let h_src = h.index_select(0, &src); // [E, F']
        let h_dst = h.index_select(0, &dst); // [E, F']

        // 拼接并计算注意力分数
        let attn_input = Tensor::cat(&[&h_src, &h_dst], 1); // [E, 2F']
        let e = attn_input.matmul(&self.attention_weights).squeeze(); // [E]
        let e = e.leaky_relu();

        // softmax 归一化（每个节点的入边）
        let e_exp = e.exp();

        // 按 dst 分组求和
        let mut sum_exp = Tensor::zeros(&[num_nodes], (e.kind(), e.device()));
        let dst_expanded = dst.to_kind(e.kind());
        sum_exp = sum_exp.scatter_add(0, &dst_expanded, &e_exp);

        // 计算 softmax 注意力权重
        let sum_e_dst = sum_exp.index_select(0, &dst);
        let attention = e_exp / (sum_e_dst + 1e-8);

        // Dropout
        let attention = if training && self.dropout > 0.0 {
            attention.dropout(self.dropout, training)
        } else {
            attention
        };

        // 加权聚合
        let h_attn = h_src * attention.unsqueeze(1);

        let mut output = Tensor::zeros_like(&h);
        let dst_expanded = dst.unsqueeze(1).broadcast_to(h_attn.size());
        output = output.scatter_add(0, &dst_expanded, &h_attn);

        output
    }
}

/// 多头图注意力层
pub struct MultiHeadGraphAttention {
    heads: Vec<GraphAttentionHead>,
    concat: bool,
}

impl MultiHeadGraphAttention {
    /// 创建新的多头注意力层
    pub fn new(
        var_store: &nn::Path,
        input_dim: usize,
        output_dim: usize,
        num_heads: usize,
        dropout: f64,
        concat: bool,
    ) -> Self {
        let mut heads = Vec::new();
        for i in 0..num_heads {
            let head_path = var_store / format!("head_{}", i);
            heads.push(GraphAttentionHead::new(
                &head_path, input_dim, output_dim, dropout,
            ));
        }

        Self { heads, concat }
    }

    /// 前向传播
    pub fn forward(&self, features: &Tensor, edge_index: &Tensor, training: bool) -> Tensor {
        let head_outputs: Vec<Tensor> = self
            .heads
            .iter()
            .map(|head| head.forward(features, edge_index, training))
            .collect();

        if self.concat {
            Tensor::cat(&head_outputs.iter().collect::<Vec<_>>(), 1)
        } else {
            let stack = Tensor::stack(&head_outputs.iter().collect::<Vec<_>>(), 0);
            stack.mean_dim(0, false, Kind::Float)
        }
    }

    /// 输出维度
    pub fn output_dim(&self) -> usize {
        if self.concat {
            self.heads.len() * self.heads[0].output_dim
        } else {
            self.heads[0].output_dim
        }
    }
}

// ========== Graph Attention Transformer (VectorGraphNET 风格) ==========

/// Transformer 前馈网络层
struct FeedForward {
    linear1: Tensor,
    linear2: Tensor,
    dropout: f64,
}

impl FeedForward {
    fn new(var_store: &nn::Path, dim: usize, hidden_dim: usize, dropout: f64) -> Self {
        let linear1 = var_store.var(
            "linear1",
            &[dim as i64, hidden_dim as i64],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
        );
        let linear2 = var_store.var(
            "linear2",
            &[hidden_dim as i64, dim as i64],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
        );

        Self {
            linear1,
            linear2,
            dropout,
        }
    }

    fn forward(&self, x: &Tensor, training: bool) -> Tensor {
        let mut x = x.matmul(&self.linear1);
        x = x.gelu("none");
        if training && self.dropout > 0.0 {
            x = x.dropout(self.dropout, training);
        }
        x = x.matmul(&self.linear2);
        if training && self.dropout > 0.0 {
            x = x.dropout(self.dropout, training);
        }
        x
    }
}

/// Graph Attention Transformer 层配置
#[derive(Debug, Clone)]
pub struct GatTransformerConfig {
    /// 输入特征维度
    pub input_dim: usize,
    /// 隐藏层维度
    pub hidden_dim: usize,
    /// 多头注意力头数
    pub num_heads: usize,
    /// Transformer 层数
    pub num_layers: usize,
    /// 输出维度
    pub output_dim: usize,
    /// Dropout 概率
    pub dropout: f64,
    /// 是否使用残差连接
    pub use_residual: bool,
    /// 是否使用层归一化
    pub use_layer_norm: bool,
}

impl Default for GatTransformerConfig {
    fn default() -> Self {
        Self {
            input_dim: 11,
            hidden_dim: 128,
            num_heads: 4,
            num_layers: 2,
            output_dim: 8,
            dropout: 0.2,
            use_residual: true,
            use_layer_norm: true,
        }
    }
}

/// Graph Attention Transformer 模型 (VectorGraphNET 风格)
pub struct GatTransformer {
    input_projection: Tensor,
    attention_layers: Vec<MultiHeadGraphAttention>,
    feed_forwards: Vec<FeedForward>,
    layer_norms1: Vec<nn::LayerNorm>,
    layer_norms2: Vec<nn::LayerNorm>,
    output_projection: Tensor,
    config: GatTransformerConfig,
}

impl GatTransformer {
    /// 创建新的 Graph Attention Transformer
    pub fn new(var_store: &nn::Path, config: GatTransformerConfig) -> Self {
        // 输入投影
        let input_proj = var_store.var(
            "input_proj",
            &[config.input_dim as i64, config.hidden_dim as i64],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
        );

        // 注意力层和前馈网络
        let mut attention_layers = Vec::new();
        let mut feed_forwards = Vec::new();
        let mut layer_norms1 = Vec::new();
        let mut layer_norms2 = Vec::new();

        let head_dim = config.hidden_dim / config.num_heads;

        for i in 0..config.num_layers {
            // 注意力层
            let attn_path = var_store / format!("layer_{}/attention", i);
            let concat = i < config.num_layers - 1;
            let attn = MultiHeadGraphAttention::new(
                &attn_path,
                config.hidden_dim,
                head_dim,
                config.num_heads,
                config.dropout,
                concat,
            );
            attention_layers.push(attn);

            // 前馈网络
            let ff_path = var_store / format!("layer_{}/ff", i);
            let ff_dim = config.hidden_dim * 4;
            feed_forwards.push(FeedForward::new(
                &ff_path,
                config.hidden_dim,
                ff_dim,
                config.dropout,
            ));

            // 层归一化
            if config.use_layer_norm {
                let ln1_path = var_store / format!("layer_{}/ln1", i);
                let ln2_path = var_store / format!("layer_{}/ln2", i);
                layer_norms1.push(nn::layer_norm(
                    &ln1_path,
                    vec![config.hidden_dim as i64],
                    Default::default(),
                ));
                layer_norms2.push(nn::layer_norm(
                    &ln2_path,
                    vec![config.hidden_dim as i64],
                    Default::default(),
                ));
            }
        }

        // 输出投影
        let output_proj = var_store.var(
            "output_proj",
            &[config.hidden_dim as i64, config.output_dim as i64],
            nn::Init::Randn {
                mean: 0.0,
                stdev: 0.01,
            },
        );

        Self {
            input_projection: input_proj,
            attention_layers,
            feed_forwards,
            layer_norms1,
            layer_norms2,
            output_projection: output_proj,
            config,
        }
    }

    /// 前向传播
    pub fn forward(&self, features: &Tensor, edge_index: &Tensor, training: bool) -> Tensor {
        let mut x = features.matmul(&self.input_projection);

        for i in 0..self.config.num_layers {
            // 注意力子层（带残差）
            let residual = x.shallow_clone();
            x = self.attention_layers[i].forward(&x, edge_index, training);

            if self.config.use_residual {
                x = x + residual;
            }

            if self.config.use_layer_norm {
                x = self.layer_norms1[i].forward(&x);
            }

            // 前馈网络子层（带残差）
            let residual = x.shallow_clone();
            x = self.feed_forwards[i].forward(&x, training);

            if self.config.use_residual {
                x = x + residual;
            }

            if self.config.use_layer_norm {
                x = self.layer_norms2[i].forward(&x);
            }
        }

        x.matmul(&self.output_projection)
    }

    /// 获取模型配置
    pub fn config(&self) -> &GatTransformerConfig {
        &self.config
    }
}

/// GCN 层配置
#[derive(Debug, Clone)]
pub struct GcnConfig {
    /// 输入特征维度
    pub input_dim: usize,
    /// 隐藏层维度
    pub hidden_dims: Vec<usize>,
    /// 输出维度
    pub output_dim: usize,
    /// Dropout 概率
    pub dropout: f64,
    /// 是否使用偏置
    pub use_bias: bool,
}

impl Default for GcnConfig {
    fn default() -> Self {
        Self {
            input_dim: 11, // 节点特征默认维度 (T101 扩展后)
            hidden_dims: vec![64, 32],
            output_dim: 8, // 8 种语义类型
            dropout: 0.5,
            use_bias: true,
        }
    }
}

/// 多层图卷积网络
pub struct GcnLayer {
    layers: Vec<GraphConvolution>,
    dropout: f64,
    config: GcnConfig,
}

impl GcnLayer {
    /// 创建新的 GCN 模型
    pub fn new(var_store: &nn::Path, config: GcnConfig) -> Self {
        let mut layers = Vec::new();
        let mut prev_dim = config.input_dim;

        for (i, &hidden_dim) in config.hidden_dims.iter().enumerate() {
            let layer_path = var_store / format!("conv_{}", i);
            layers.push(GraphConvolution::new(
                &layer_path,
                prev_dim,
                hidden_dim,
                config.use_bias,
            ));
            prev_dim = hidden_dim;
        }

        // 输出层
        let output_path = var_store / "output";
        layers.push(GraphConvolution::new(
            &output_path,
            prev_dim,
            config.output_dim,
            config.use_bias,
        ));

        Self {
            layers,
            dropout: config.dropout,
            config,
        }
    }

    /// 前向传播
    pub fn forward(&self, features: &Tensor, adjacency: &Tensor, training: bool) -> Tensor {
        let mut x = features.shallow_clone();

        for (i, layer) in self.layers.iter().enumerate() {
            x = layer.forward(&x, adjacency);

            // 最后一层之前应用 ReLU 和 Dropout
            if i < self.layers.len() - 1 {
                x = x.relu();
                if training && self.dropout > 0.0 {
                    x = x.dropout(self.dropout, training);
                }
            }
        }

        x
    }

    /// 获取模型配置
    pub fn config(&self) -> &GcnConfig {
        &self.config
    }
}

// ========== 层次化语义标签系统 ==========

/// 语义类别层次
///
/// 采用多层次标签定义：
/// - Level 1: 几何类型 (直线/曲线/圆弧...)
/// - Level 2: 功能类别 (建筑/机械/电路图元)
/// - Level 3: 具体类型 (墙/门/窗/尺寸标注/公差...)

/// Level 1: 几何类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum GeometryType {
    /// 直线段
    Line = 0,
    /// 曲线
    Curve = 1,
    /// 圆弧
    Arc = 2,
    /// 圆
    Circle = 3,
    /// 多段线
    Polyline = 4,
}

/// Level 2: 领域类型
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum DomainType {
    /// 建筑图纸
    Architectural = 0,
    /// 机械图纸
    Mechanical = 1,
    /// 电路图
    Circuit = 2,
    /// 通用
    General = 3,
}

/// Level 3: 具体语义类型 (8+ 种)
#[derive(Debug, Clone, Copy, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub enum SemanticType {
    /// 墙体
    Wall = 0,
    /// 门
    Door = 1,
    /// 窗
    Window = 2,
    /// 标注
    Dimension = 3,
    /// 文字
    Text = 4,
    /// 家具
    Furniture = 5,
    /// 管道
    Pipe = 6,
    /// 轮廓线 (机械)
    Outline = 7,
    /// 虚线
    DashedLine = 8,
    /// 剖面线
    Hatch = 9,
    /// 基准符号
    Datum = 10,
    /// 公差符号
    Tolerance = 11,
    /// 其他
    Other = 12,
}

impl SemanticType {
    /// 从 usize 转换
    pub fn from_usize(value: usize) -> Self {
        match value {
            0 => Self::Wall,
            1 => Self::Door,
            2 => Self::Window,
            3 => Self::Dimension,
            4 => Self::Text,
            5 => Self::Furniture,
            6 => Self::Pipe,
            7 => Self::Outline,
            8 => Self::DashedLine,
            9 => Self::Hatch,
            10 => Self::Datum,
            11 => Self::Tolerance,
            _ => Self::Other,
        }
    }

    /// 获取类型名称
    pub fn name(&self) -> &'static str {
        match self {
            Self::Wall => "Wall",
            Self::Door => "Door",
            Self::Window => "Window",
            Self::Dimension => "Dimension",
            Self::Text => "Text",
            Self::Furniture => "Furniture",
            Self::Pipe => "Pipe",
            Self::Outline => "Outline",
            Self::DashedLine => "DashedLine",
            Self::Hatch => "Hatch",
            Self::Datum => "Datum",
            Self::Tolerance => "Tolerance",
            Self::Other => "Other",
        }
    }

    /// 获取对应的领域类型 (Level 2)
    pub fn domain_type(&self) -> DomainType {
        match self {
            Self::Wall | Self::Door | Self::Window | Self::Furniture => DomainType::Architectural,
            Self::Outline | Self::DashedLine | Self::Hatch | Self::Datum | Self::Tolerance => {
                DomainType::Mechanical
            }
            Self::Dimension | Self::Text | Self::Pipe => DomainType::General,
            Self::Other => DomainType::General,
        }
    }

    /// 获取所有语义类型的总数
    pub fn num_classes() -> usize {
        13
    }
}

impl GeometryType {
    /// 从 usize 转换
    pub fn from_usize(value: usize) -> Self {
        match value {
            0 => Self::Line,
            1 => Self::Curve,
            2 => Self::Arc,
            3 => Self::Circle,
            _ => Self::Polyline,
        }
    }

    /// 获取类型名称
    pub fn name(&self) -> &'static str {
        match self {
            Self::Line => "Line",
            Self::Curve => "Curve",
            Self::Arc => "Arc",
            Self::Circle => "Circle",
            Self::Polyline => "Polyline",
        }
    }

    /// 几何类型总数
    pub fn num_classes() -> usize {
        5
    }
}

impl DomainType {
    /// 从 usize 转换
    pub fn from_usize(value: usize) -> Self {
        match value {
            0 => Self::Architectural,
            1 => Self::Mechanical,
            2 => Self::Circuit,
            _ => Self::General,
        }
    }

    /// 获取类型名称
    pub fn name(&self) -> &'static str {
        match self {
            Self::Architectural => "Architectural",
            Self::Mechanical => "Mechanical",
            Self::Circuit => "Circuit",
            Self::General => "General",
        }
    }

    /// 领域类型总数
    pub fn num_classes() -> usize {
        4
    }
}

/// 设备类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DeviceType {
    /// CPU
    Cpu,
    /// CUDA GPU
    Cuda,
    /// Metal GPU (macOS)
    Metal,
}

impl DeviceType {
    /// 自动检测最佳设备
    pub fn auto() -> Self {
        if tch::Cuda::is_available() {
            DeviceType::Cuda
        } else if tch::utils::has_mps() {
            DeviceType::Metal
        } else {
            DeviceType::Cpu
        }
    }

    /// 转换为 tch Device
    pub fn to_tch_device(self) -> Device {
        match self {
            DeviceType::Cpu => Device::Cpu,
            DeviceType::Cuda => Device::Cuda(0),
            DeviceType::Metal => Device::Mps,
        }
    }
}

/// 模型加载错误
#[derive(Debug, thiserror::Error)]
pub enum ModelError {
    #[error("TorchScript 加载失败: {0}")]
    TorchScriptLoad(String),
    #[error("权重加载失败: {0}")]
    WeightsLoad(String),
    #[error("权重保存失败: {0}")]
    WeightsSave(String),
    #[error("特征维度不匹配: 期望 {expected}, 实际 {actual}")]
    FeatureDimensionMismatch { expected: usize, actual: usize },
    #[error("空图无法进行推理")]
    EmptyGraph,
    #[error("JSON 序列化失败: {0}")]
    JsonSerialization(String),
}

/// 语义分类器
///
/// 基于 GCN 的节点级语义分类器
pub struct SemanticClassifier {
    gcn: GcnLayer,
    var_store: nn::VarStore,
    num_classes: usize,
    /// TorchScript 编译模型（如果加载了预训练模型）
    jit_model: Option<tch::CModule>,
}

impl SemanticClassifier {
    /// 创建新的语义分类器（自动检测设备）
    pub fn new(config: GcnConfig) -> Self {
        let device = DeviceType::auto().to_tch_device();
        Self::with_device(device, config)
    }

    /// 使用指定设备创建语义分类器
    pub fn with_device(device: Device, config: GcnConfig) -> Self {
        let var_store = nn::VarStore::new(device);
        let gcn = GcnLayer::new(&var_store.root(), config.clone());

        Self {
            gcn,
            var_store,
            num_classes: config.output_dim,
            jit_model: None,
        }
    }

    /// 从 TorchScript 模型创建
    pub fn from_torchscript(path: impl AsRef<Path>, device: Device) -> Result<Self, ModelError> {
        let jit_model = tch::CModule::load_on_device(path.as_ref(), device)
            .map_err(|e| ModelError::TorchScriptLoad(e.to_string()))?;

        // 使用默认配置创建基础结构（实际参数将从 JIT 模型加载）
        let config = GcnConfig::default();
        let var_store = nn::VarStore::new(device);

        Ok(Self {
            gcn: GcnLayer::new(&var_store.root(), config.clone()),
            var_store,
            num_classes: config.output_dim,
            jit_model: Some(jit_model),
        })
    }

    /// 导出模型权重（兼容 PyTorch Python 端加载）
    ///
    /// 注意：完整 TorchScript 导出需要在 Python 端使用 `torch.jit.trace` 后导出
    pub fn export_weights(&self, path: impl AsRef<Path>) -> Result<(), ModelError> {
        self.save_weights(path)
    }

    /// 从 CadGraph 进行推理
    pub fn predict(&self, graph: &CadGraph) -> Result<Vec<(SemanticType, f32)>, ModelError> {
        if graph.node_count() == 0 {
            return Err(ModelError::EmptyGraph);
        }

        let device = self.var_store.device();

        // 提取特征
        let extractor = FeatureExtractor::default();
        let node_features = extractor.extract_node_features(graph);

        // 验证特征维度
        let (rows, cols) = node_features.shape();
        if cols != self.gcn.config().input_dim {
            return Err(ModelError::FeatureDimensionMismatch {
                expected: self.gcn.config().input_dim,
                actual: cols,
            });
        }

        // 转换为 PyTorch Tensor
        let data: Vec<f32> = node_features.iter().copied().collect();
        let features = Tensor::from_slice(&data)
            .to_device(device)
            .reshape([rows as i64, cols as i64]);

        // 构建邻接矩阵 COO 格式
        let (src, dst) = extractor.adjacency_matrix_coo(graph);
        let edge_index = if src.is_empty() {
            // 空边矩阵处理
            Tensor::zeros(&[2, 1], (Kind::Int64, device))
        } else {
            Tensor::from_slice(
                &src.iter()
                    .chain(dst.iter())
                    .map(|&x| x as i64)
                    .collect::<Vec<_>>(),
            )
            .to_device(device)
            .reshape([2, src.len() as i64])
        };

        // 前向传播
        let logits = if let Some(jit_model) = &self.jit_model {
            // 使用 TorchScript 模型
            let output = jit_model
                .forward_ts(&[features, edge_index])
                .map_err(|e| ModelError::TorchScriptLoad(e.to_string()))?;
            output.try_into().unwrap_or_else(|_| {
                Tensor::zeros(
                    &[rows as i64, self.num_classes as i64],
                    (Kind::Float, device),
                )
            })
        } else {
            // 使用 Rust 原生模型
            self.gcn.forward(&features, &edge_index, false)
        };

        let probabilities = logits.softmax(1, Kind::Float);

        // 获取预测结果
        let predictions = Vec::<i64>::try_from(probabilities.argmax(1, false)).unwrap();
        let confidences = Vec::<f32>::try_from(probabilities.max_dim(1, false).0).unwrap();

        Ok(predictions
            .into_iter()
            .zip(confidences)
            .map(|(pred, conf)| (SemanticType::from_usize(pred as usize), conf))
            .collect())
    }

    /// 批量推理多张图
    pub fn predict_batch(
        &self,
        graphs: &[CadGraph],
    ) -> Result<Vec<Vec<(SemanticType, f32)>>, ModelError> {
        graphs.iter().map(|g| self.predict(g)).collect()
    }

    /// 保存模型权重（VarStore 格式，Rust 训练用）
    pub fn save_weights(&self, path: impl AsRef<Path>) -> Result<(), ModelError> {
        self.var_store
            .save(path.as_ref())
            .map_err(|e| ModelError::WeightsSave(e.to_string()))?;
        Ok(())
    }

    /// 加载模型权重（VarStore 格式）
    pub fn load_weights(&mut self, path: impl AsRef<Path>) -> Result<(), ModelError> {
        self.var_store
            .load(path.as_ref())
            .map_err(|e| ModelError::WeightsLoad(e.to_string()))?;
        Ok(())
    }

    /// 获取变量存储引用
    pub fn var_store(&self) -> &nn::VarStore {
        &self.var_store
    }

    /// 获取模型设备
    pub fn device(&self) -> Device {
        self.var_store.device()
    }

    /// 获取当前使用的设备类型
    pub fn device_type(&self) -> DeviceType {
        match self.device() {
            Device::Cpu => DeviceType::Cpu,
            Device::Cuda(_) => DeviceType::Cuda,
            Device::Mps => DeviceType::Metal,
            Device::Vulkan => DeviceType::Cpu, // Vulkan 作为 CPU fallback
        }
    }

    /// 获取类别数量
    pub fn num_classes(&self) -> usize {
        self.num_classes
    }

    /// 是否使用 TorchScript 模型
    pub fn is_jit_model(&self) -> bool {
        self.jit_model.is_some()
    }

    /// 获取 GCN 层引用（用于训练期间的前向传播）
    pub fn gcn(&self) -> &GcnLayer {
        &self.gcn
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::CadGraph;

    #[test]
    #[ignore = "需要 PyTorch 环境"]
    fn test_graph_convolution() {
        let device = Device::Cpu;
        let vs = nn::VarStore::new(device);

        let conv = GraphConvolution::new(&vs.root(), 11, 32, true);

        assert_eq!(conv.input_dim(), 11);
        assert_eq!(conv.output_dim(), 32);
    }

    #[test]
    #[ignore = "需要 PyTorch 环境"]
    fn test_semantic_classifier() {
        let config = GcnConfig {
            input_dim: 11,
            hidden_dims: vec![16],
            output_dim: 8,
            dropout: 0.0,
            use_bias: true,
        };

        let classifier = SemanticClassifier::new(config);

        // 简单测试图
        let polyline = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]];
        let graph = CadGraph::from_polylines(&[polyline], 1e-3);

        let predictions = classifier.predict(&graph).unwrap();
        assert_eq!(predictions.len(), 3); // 3 个节点
    }

    #[test]
    #[ignore = "需要 PyTorch 环境"]
    fn test_device_auto_detection() {
        let device = DeviceType::auto();
        println!("Auto-detected device: {:?}", device);
    }

    #[test]
    #[ignore = "需要 PyTorch 环境"]
    fn test_empty_graph_error() {
        let config = GcnConfig::default();
        let classifier = SemanticClassifier::new(config);

        let empty_graph = CadGraph::new(1e-3);
        let result = classifier.predict(&empty_graph);

        assert!(result.is_err());
    }

    #[test]
    #[ignore = "需要 PyTorch 环境"]
    fn test_semantic_type_names() {
        assert_eq!(SemanticType::Wall.name(), "Wall");
        assert_eq!(SemanticType::Door.name(), "Door");
        assert_eq!(SemanticType::Window.name(), "Window");
        assert_eq!(SemanticType::Dimension.name(), "Dimension");
        assert_eq!(SemanticType::Text.name(), "Text");
        assert_eq!(SemanticType::Furniture.name(), "Furniture");
        assert_eq!(SemanticType::Pipe.name(), "Pipe");
        assert_eq!(SemanticType::Other.name(), "Other");
    }

    #[test]
    #[ignore = "需要 PyTorch 环境"]
    fn test_semantic_type_from_usize() {
        assert_eq!(SemanticType::from_usize(0), SemanticType::Wall);
        assert_eq!(SemanticType::from_usize(1), SemanticType::Door);
        assert_eq!(SemanticType::from_usize(7), SemanticType::Other);
        assert_eq!(SemanticType::from_usize(100), SemanticType::Other); // 超出范围返回 Other
    }
}

// ========== 训练器模块 ==========

/// 训练配置
#[derive(Debug, Clone)]
pub struct TrainingConfig {
    /// 学习率
    pub learning_rate: f64,
    /// 权重衰减（L2 正则化）
    pub weight_decay: f64,
    /// 训练轮数
    pub num_epochs: usize,
    /// 每多少轮验证一次
    pub validation_interval: usize,
    /// 每多少轮保存一次检查点
    pub checkpoint_interval: usize,
    /// 早停 patience（连续多少轮无提升则停止）
    pub early_stopping_patience: usize,
    /// 梯度裁剪阈值
    pub grad_clip_norm: f64,
    /// 是否打印训练进度
    pub verbose: bool,
}

impl Default for TrainingConfig {
    fn default() -> Self {
        Self {
            learning_rate: 0.001,
            weight_decay: 1e-4,
            num_epochs: 100,
            validation_interval: 5,
            checkpoint_interval: 20,
            early_stopping_patience: 15,
            grad_clip_norm: 1.0,
            verbose: true,
        }
    }
}

/// 训练指标统计
#[derive(Debug, Clone, Default)]
pub struct TrainingMetrics {
    /// 训练损失
    pub train_loss: Vec<f64>,
    /// 训练准确率
    pub train_accuracy: Vec<f64>,
    /// 验证损失
    pub val_loss: Vec<f64>,
    /// 验证准确率
    pub val_accuracy: Vec<f64>,
    /// 最佳验证准确率
    pub best_val_accuracy: f64,
    /// 最佳验证损失
    pub best_val_loss: f64,
    /// 当前轮数
    pub current_epoch: usize,
}

/// 单图训练样本
#[cfg(feature = "pytorch")]
pub struct TrainingSample {
    /// 节点特征 [num_nodes, num_features]
    pub features: Tensor,
    /// 边索引 COO 格式 [2, num_edges]
    pub edge_index: Tensor,
    /// 节点级别标签 [num_nodes]
    pub labels: Tensor,
    /// 训练掩码 [num_nodes] - 用于半监督学习
    pub train_mask: Option<Tensor>,
    /// 验证掩码 [num_nodes]
    pub val_mask: Option<Tensor>,
}

#[cfg(feature = "pytorch")]
impl TrainingSample {
    /// 从 CadGraph 创建训练样本
    pub fn from_graph(
        graph: &CadGraph,
        labels: &[SemanticType],
        device: Device,
    ) -> Result<Self, ModelError> {
        if graph.node_count() != labels.len() {
            return Err(ModelError::FeatureDimensionMismatch {
                expected: graph.node_count(),
                actual: labels.len(),
            });
        }

        let extractor = FeatureExtractor::default();
        let node_features = extractor.extract_node_features(graph);
        let (rows, cols) = node_features.shape();

        // 转换为 Tensor
        let data: Vec<f32> = node_features.iter().copied().collect();
        let features = Tensor::from_slice(&data)
            .to_device(device)
            .reshape([rows as i64, cols as i64]);

        // 构建邻接矩阵 COO
        let (src, dst) = extractor.adjacency_matrix_coo(graph);
        let edge_index = if src.is_empty() {
            Tensor::zeros(&[2, 1], (Kind::Int64, device))
        } else {
            Tensor::from_slice(
                &src.iter()
                    .chain(dst.iter())
                    .map(|&x| x as i64)
                    .collect::<Vec<_>>(),
            )
            .to_device(device)
            .reshape([2, src.len() as i64])
        };

        // 转换标签
        let labels_i64: Vec<i64> = labels.iter().map(|&l| l as i64).collect();
        let labels = Tensor::from_slice(&labels_i64).to_device(device);

        Ok(Self {
            features,
            edge_index,
            labels,
            train_mask: None,
            val_mask: None,
        })
    }
}

/// GNN 训练器
#[cfg(feature = "pytorch")]
pub struct GNNTrainer {
    classifier: SemanticClassifier,
    optimizer: tch::nn::Optimizer,
    config: TrainingConfig,
    metrics: TrainingMetrics,
    device: Device,
}

#[cfg(feature = "pytorch")]
impl GNNTrainer {
    /// 创建新的训练器
    pub fn new(
        model_config: GcnConfig,
        training_config: TrainingConfig,
        device: Device,
    ) -> Result<Self, ModelError> {
        let classifier = SemanticClassifier::with_device(device, model_config);

        // 创建 Adam 优化器
        let optimizer = tch::nn::adam(0.9, 0.999, training_config.weight_decay)
            .build(classifier.var_store(), training_config.learning_rate)
            .map_err(|e| ModelError::WeightsLoad(e.to_string()))?;

        Ok(Self {
            classifier,
            optimizer,
            config: training_config,
            metrics: TrainingMetrics::default(),
            device,
        })
    }

    /// 训练单轮（单个样本，小样本学习场景）
    pub fn train_step_single(&mut self, sample: &TrainingSample) -> (f64, f64) {
        self.optimizer.zero_grad();

        // 前向传播
        let logits = self
            .classifier
            .gcn
            .forward(&sample.features, &sample.edge_index, true);

        // 计算损失
        let loss = logits.cross_entropy_for_logits(&sample.labels);

        // 反向传播
        loss.backward();

        // 梯度裁剪
        if self.config.grad_clip_norm > 0.0 {
            self.optimizer.clip_grad_norm(self.config.grad_clip_norm);
        }

        self.optimizer.step();

        // 计算准确率
        let predictions = logits.argmax(1, false);
        let correct = predictions
            .eq_tensor(&sample.labels)
            .to_kind(tch::Kind::Float)
            .sum(tch::Kind::Float);
        let accuracy = f64::try_from(correct / sample.labels.size()[0] as f64).unwrap_or(0.0);

        let loss_f64 = f64::try_from(loss).unwrap_or(0.0);
        (loss_f64, accuracy)
    }

    /// 验证单样本
    pub fn val_step_single(&self, sample: &TrainingSample) -> (f64, f64) {
        tch::no_grad(|| {
            let logits = self
                .classifier
                .gcn
                .forward(&sample.features, &sample.edge_index, false);

            let loss = logits.cross_entropy_for_logits(&sample.labels);
            let predictions = logits.argmax(1, false);
            let correct = predictions
                .eq_tensor(&sample.labels)
                .to_kind(tch::Kind::Float)
                .sum(tch::Kind::Float);
            let accuracy = f64::try_from(correct / sample.labels.size()[0] as f64).unwrap_or(0.0);

            let loss_f64 = f64::try_from(loss).unwrap_or(0.0);
            (loss_f64, accuracy)
        })
    }

    /// 小样本少样本学习（Few-shot Learning）
    /// 给定少量标注样本，训练模型进行快速适应
    pub fn train_few_shot(
        &mut self,
        train_samples: &[TrainingSample],
        val_samples: &[TrainingSample],
        output_dir: impl AsRef<Path>,
    ) -> Result<TrainingMetrics, ModelError> {
        let output_path = output_dir.as_ref();
        std::fs::create_dir_all(output_path).map_err(|e| ModelError::WeightsSave(e.to_string()))?;

        let mut patience_counter = 0;
        self.metrics.best_val_loss = f64::MAX;

        if self.config.verbose {
            println!("========== 开始小样本训练 ==========");
            println!(
                "训练样本: {}, 验证样本: {}",
                train_samples.len(),
                val_samples.len()
            );
            println!("训练轮数: {}", self.config.num_epochs);
            println!("学习率: {}", self.config.learning_rate);
            println!("====================================");
        }

        for epoch in 0..self.config.num_epochs {
            // ---------- 训练 ----------
            let mut total_train_loss = 0.0;
            let mut total_train_acc = 0.0;

            for sample in train_samples {
                let (loss, acc) = self.train_step_single(sample);
                total_train_loss += loss;
                total_train_acc += acc;
            }

            let avg_train_loss = total_train_loss / train_samples.len() as f64;
            let avg_train_acc = total_train_acc / train_samples.len() as f64;

            self.metrics.train_loss.push(avg_train_loss);
            self.metrics.train_accuracy.push(avg_train_acc);
            self.metrics.current_epoch = epoch;

            // ---------- 验证 ----------
            if epoch % self.config.validation_interval == 0 && !val_samples.is_empty() {
                let mut total_val_loss = 0.0;
                let mut total_val_acc = 0.0;

                for sample in val_samples {
                    let (loss, acc) = self.val_step_single(sample);
                    total_val_loss += loss;
                    total_val_acc += acc;
                }

                let avg_val_loss = total_val_loss / val_samples.len() as f64;
                let avg_val_acc = total_val_acc / val_samples.len() as f64;

                self.metrics.val_loss.push(avg_val_loss);
                self.metrics.val_accuracy.push(avg_val_acc);

                // 早停检查
                if avg_val_loss < self.metrics.best_val_loss {
                    self.metrics.best_val_loss = avg_val_loss;
                    self.metrics.best_val_accuracy = avg_val_acc;
                    patience_counter = 0;

                    // 保存最佳模型
                    let best_path = output_path.join("model_best.ot");
                    self.classifier.save_weights(&best_path)?;
                } else {
                    patience_counter += 1;
                    if patience_counter >= self.config.early_stopping_patience {
                        if self.config.verbose {
                            println!(
                                "早停触发! 连续 {} 轮无提升",
                                self.config.early_stopping_patience
                            );
                        }
                        break;
                    }
                }

                // 打印进度
                if self.config.verbose {
                    println!(
                        "Epoch [{}/{}] | Train Loss: {:.4} Acc: {:.2}% | Val Loss: {:.4} Acc: {:.2}%{}",
                        epoch + 1,
                        self.config.num_epochs,
                        avg_train_loss,
                        avg_train_acc * 100.0,
                        avg_val_loss,
                        avg_val_acc * 100.0,
                        if patience_counter == 0 { " ← Best" } else { "" }
                    );
                }
            } else if self.config.verbose {
                println!(
                    "Epoch [{}/{}] | Train Loss: {:.4} Acc: {:.2}%",
                    epoch + 1,
                    self.config.num_epochs,
                    avg_train_loss,
                    avg_train_acc * 100.0
                );
            }

            // 保存检查点
            if (epoch + 1) % self.config.checkpoint_interval == 0 {
                let checkpoint_path =
                    output_path.join(format!("checkpoint_epoch_{}.ot", epoch + 1));
                self.classifier.save_weights(&checkpoint_path)?;
            }
        }

        // 保存最终模型
        let final_path = output_path.join("model_final.ot");
        self.classifier.save_weights(&final_path)?;

        // 保存训练指标
        let metrics_json = serde_json::json!({
            "train_loss": self.metrics.train_loss,
            "train_accuracy": self.metrics.train_accuracy,
            "val_loss": self.metrics.val_loss,
            "val_accuracy": self.metrics.val_accuracy,
            "best_val_loss": self.metrics.best_val_loss,
            "best_val_accuracy": self.metrics.best_val_accuracy,
            "total_epochs": self.metrics.current_epoch + 1,
        });

        let metrics_path = output_path.join("training_metrics.json");
        let json_str = serde_json::to_string_pretty(&metrics_json)
            .map_err(|e| ModelError::JsonSerialization(e.to_string()))?;
        std::fs::write(&metrics_path, json_str)
            .map_err(|e| ModelError::WeightsSave(e.to_string()))?;

        if self.config.verbose {
            println!("====================================");
            println!("训练完成!");
            println!("最佳验证损失: {:.4}", self.metrics.best_val_loss);
            println!(
                "最佳验证准确率: {:.2}%",
                self.metrics.best_val_accuracy * 100.0
            );
            println!("模型已保存至: {}", output_path.display());
            println!("====================================");
        }

        Ok(self.metrics.clone())
    }

    /// 获取训练指标
    pub fn metrics(&self) -> &TrainingMetrics {
        &self.metrics
    }

    /// 获取分类器引用
    pub fn classifier(&self) -> &SemanticClassifier {
        &self.classifier
    }

    /// 获取分类器可变引用
    pub fn classifier_mut(&mut self) -> &mut SemanticClassifier {
        &mut self.classifier
    }

    /// 加载最佳模型权重
    pub fn load_best_model(&mut self, checkpoint_dir: impl AsRef<Path>) -> Result<(), ModelError> {
        let best_path = checkpoint_dir.as_ref().join("model_best.ot");
        self.classifier.load_weights(best_path)
    }

    /// 获取训练设备
    pub fn device(&self) -> Device {
        self.device
    }
}

// ========== 训练数据生成器 ==========

#[cfg(feature = "pytorch")]
pub struct TrainingDataGenerator {
    device: Device,
}

#[cfg(feature = "pytorch")]
impl TrainingDataGenerator {
    /// 创建新的数据生成器
    pub fn new(device: Device) -> Self {
        Self { device }
    }

    /// 从合成数据生成训练/验证样本
    pub fn generate_synthetic_dataset(
        &self,
        num_train: usize,
        num_val: usize,
        graph_size: usize,
    ) -> Result<(Vec<TrainingSample>, Vec<TrainingSample>), ModelError> {
        use crate::graph::{CadEdge, CadGraph, EdgeType, NodeType};

        let mut train_samples = Vec::with_capacity(num_train);
        let mut val_samples = Vec::with_capacity(num_val);

        // 生成训练样本
        for i in 0..num_train + num_val {
            // 创建简单的矩形网格图
            let mut graph = CadGraph::new(1e-3);
            let mut nodes = Vec::new();

            for j in 0..graph_size {
                let x = (j % 4) as f64;
                let y = (j / 4) as f64;
                let node_idx = graph.get_or_create_node(x, y, NodeType::Endpoint);
                nodes.push(node_idx);
            }

            // 添加边（相邻连接）
            for j in 0..nodes.len() {
                if j + 1 < nodes.len() && (j + 1) % 4 != 0 {
                    let edge = CadEdge {
                        edge_type: EdgeType::Line,
                        length: 1.0,
                        angle: 0.0,
                        polyline_id: None,
                        parallelism_score: 0.0,
                        collinearity_score: 0.0,
                        avg_neighbor_distance: 0.0,
                    };
                    graph.add_edge(nodes[j], nodes[j + 1], edge);
                }
                if j + 4 < nodes.len() {
                    let edge = CadEdge {
                        edge_type: EdgeType::Line,
                        length: 1.0,
                        angle: std::f64::consts::PI / 2.0,
                        polyline_id: None,
                        parallelism_score: 0.0,
                        collinearity_score: 0.0,
                        avg_neighbor_distance: 0.0,
                    };
                    graph.add_edge(nodes[j], nodes[j + 4], edge);
                }
            }

            // 生成模拟标签（根据节点位置模式）
            let labels: Vec<SemanticType> = (0..graph_size)
                .map(|j| {
                    let x = j % 4;
                    let y = j / 4;
                    if x == 0 || x == 3 || y == 0 || y == graph_size / 4 - 1 {
                        SemanticType::Wall // 边界是墙
                    } else if x == 1 && y == 1 {
                        SemanticType::Door // 门
                    } else if x == 2 && y == 1 {
                        SemanticType::Window // 窗
                    } else {
                        SemanticType::Furniture // 家具
                    }
                })
                .collect();

            let sample = TrainingSample::from_graph(&graph, &labels, self.device)?;

            if i < num_train {
                train_samples.push(sample);
            } else {
                val_samples.push(sample);
            }
        }

        Ok((train_samples, val_samples))
    }
}

#[cfg(test)]
#[cfg(feature = "pytorch")]
mod training_tests {
    use super::*;

    #[test]
    #[ignore = "需要 PyTorch 环境"]
    fn test_training_config_default() {
        let config = TrainingConfig::default();
        assert_eq!(config.learning_rate, 0.001);
        assert_eq!(config.num_epochs, 100);
        assert_eq!(config.early_stopping_patience, 15);
    }

    #[test]
    #[ignore = "需要 PyTorch 环境"]
    fn test_synthetic_data_generation() {
        let generator = TrainingDataGenerator::new(Device::Cpu);
        let (train, val) = generator.generate_synthetic_dataset(5, 2, 8).unwrap();

        assert_eq!(train.len(), 5);
        assert_eq!(val.len(), 2);
    }

    #[test]
    #[ignore = "需要 PyTorch 环境"]
    fn test_trainer_creation() {
        let model_config = GcnConfig {
            input_dim: 11,
            hidden_dims: vec![32, 16],
            output_dim: 13,
            dropout: 0.2,
            use_bias: true,
        };

        let training_config = TrainingConfig {
            num_epochs: 10,
            verbose: false,
            ..TrainingConfig::default()
        };

        let _trainer = GNNTrainer::new(model_config, training_config, Device::Cpu).unwrap();
    }
}
