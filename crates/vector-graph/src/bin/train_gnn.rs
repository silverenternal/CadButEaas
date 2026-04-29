//! GNN 模型训练端到端示例
//!
//! 演示完整流程：
//! 1. 生成合成 CAD 训练数据
//! 2. 矢量化处理
//! 3. 构建图结构
//! 4. GNN 模型训练
//! 5. 评估与导出

#[cfg(feature = "pytorch")]
use vector_graph::*;

#[cfg(feature = "pytorch")]
use std::path::PathBuf;

#[cfg(feature = "pytorch")]
fn print_banner() {
    println!("\n");
    println!("╔══════════════════════════════════════════════════════════════╗");
    println!("║           CadButEaas GNN 训练器 v0.1.0                          ║");
    println!("║     图神经网络 CAD 语义识别端到端训练                            ║");
    println!("╚══════════════════════════════════════════════════════════════╝");
    println!();
}

#[cfg(feature = "pytorch")]
fn print_device_info(device: tch::Device) {
    println!("📱 计算设备: {:?}", device);
    if let tch::Device::Cuda(_) = device {
        println!("   GPU 可用: 是");
    }
    println!();
}

#[cfg(feature = "pytorch")]
fn print_dataset_stats(
    train_count: usize,
    val_count: usize,
    total_nodes: usize,
    total_edges: usize,
) {
    println!("📊 数据集统计:");
    println!("   训练样本: {}", train_count);
    println!("   验证样本: {}", val_count);
    println!("   总节点数: {}", total_nodes);
    println!("   总边数: {}", total_edges);
    println!(
        "   平均节点/图: {:.1}",
        total_nodes as f64 / (train_count + val_count) as f64
    );
    println!(
        "   平均边/图: {:.1}",
        total_edges as f64 / (train_count + val_count) as f64
    );
    println!();
}

#[cfg(feature = "pytorch")]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    print_banner();

    // 解析命令行参数
    let args: Vec<String> = std::env::args().collect();

    let num_train = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let num_val = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(20);
    let graph_size = args.get(3).and_then(|s| s.parse().ok()).unwrap_or(16);
    let num_epochs = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(100);
    let output_dir = args
        .get(5)
        .cloned()
        .unwrap_or_else(|| "./checkpoints".to_string());

    // 自动检测设备
    let device = DeviceType::auto().to_tch_device();
    print_device_info(device);

    // 生成合成训练数据
    println!("🔨 生成合成训练数据...");
    let generator = TrainingDataGenerator::new(device);
    let (train_samples, val_samples) =
        generator.generate_synthetic_dataset(num_train, num_val, graph_size)?;

    // 统计数据信息
    let total_nodes: usize = train_samples
        .iter()
        .chain(val_samples.iter())
        .map(|s| s.features.size()[0] as usize)
        .sum();
    let total_edges: usize = train_samples
        .iter()
        .chain(val_samples.iter())
        .map(|s| s.edge_index.size()[1] as usize)
        .sum();
    print_dataset_stats(
        train_samples.len(),
        val_samples.len(),
        total_nodes,
        total_edges,
    );

    // 模型配置
    println!("🧠 初始化 GNN 模型...");
    let model_config = GcnConfig {
        input_dim: 11, // 节点特征维度
        hidden_dims: vec![128, 64],
        output_dim: 13, // 语义类别数量
        dropout: 0.2,
        use_bias: true,
    };

    println!("   输入维度: {}", model_config.input_dim);
    println!("   隐藏层: {:?}", model_config.hidden_dims);
    println!("   输出维度: {}", model_config.output_dim);
    println!("   Dropout: {}", model_config.dropout);
    println!();

    // 训练配置
    let training_config = TrainingConfig {
        learning_rate: 0.001,
        weight_decay: 1e-4,
        num_epochs,
        validation_interval: 5,
        checkpoint_interval: 20,
        early_stopping_patience: 15,
        grad_clip_norm: 1.0,
        verbose: true,
    };

    // 创建训练器
    println!("🚀 开始训练...");
    let mut trainer = GNNTrainer::new(model_config, training_config, device)?;

    // 执行训练
    let output_path = PathBuf::from(&output_dir);
    let metrics = trainer.train_few_shot(&train_samples, &val_samples, &output_path)?;

    // 加载最佳模型进行推理演示
    println!("\n🔍 加载最佳模型进行推理演示...");
    trainer.load_best_model(&output_path)?;

    // 在验证样本上演示推理
    if !val_samples.is_empty() {
        let test_sample = &val_samples[0];

        println!("\n📋 分类结果示例:");
        println!("   {:<15} {:<15} {:<10}", "真实标签", "预测标签", "置信度");
        println!("   {}", "─".repeat(45));

        // 手动执行推理
        let logits = tch::no_grad(|| {
            trainer.classifier().gcn().forward(
                &test_sample.features,
                &test_sample.edge_index,
                false,
            )
        });

        let probabilities = logits.softmax(1, tch::Kind::Float);
        let predictions = Vec::<i64>::try_from(probabilities.argmax(1, false)).unwrap();
        let confidences = Vec::<f32>::try_from(probabilities.max_dim(1, false).0).unwrap();
        let labels = Vec::<i64>::try_from(test_sample.labels.shallow_clone()).unwrap();

        let mut correct = 0;
        for i in 0..predictions.len().min(8) {
            let pred = SemanticType::from_usize(predictions[i] as usize);
            let true_label = SemanticType::from_usize(labels[i] as usize);
            let is_correct = predictions[i] == labels[i];
            if is_correct {
                correct += 1;
            }

            println!(
                "   {:<15} {:<15} {:.2}% {}",
                format!("{:?}", true_label),
                format!("{:?}", pred),
                confidences[i] * 100.0,
                if is_correct { "✓" } else { "✗" }
            );
        }

        println!(
            "\n✅ 演示样本准确率: {}/{} ({:.1}%)",
            correct,
            predictions.len().min(8),
            correct as f64 / predictions.len().min(8) as f64 * 100.0
        );
    }

    // 输出最终总结
    println!("\n══════════════════════════════════════════════════════════════");
    println!("🎉 训练完成!");
    println!("📈 最佳验证损失: {:.4}", metrics.best_val_loss);
    println!(
        "📊 最佳验证准确率: {:.2}%",
        metrics.best_val_accuracy * 100.0
    );
    println!("📁 模型保存路径: {}", output_path.display());
    println!("══════════════════════════════════════════════════════════════\n");

    Ok(())
}

#[cfg(not(feature = "pytorch"))]
fn main() {
    println!("⚠️  此功能需要启用 'pytorch' feature");
    println!();
    println!("运行方式:");
    println!("  cargo run --bin train_gnn --features pytorch -p vectorize");
    println!();
    println!("命令行参数（可选）:");
    println!("  <num_train>     - 训练样本数量 (默认: 100)");
    println!("  <num_val>       - 验证样本数量 (默认: 20)");
    println!("  <graph_size>    - 每张图的节点数量 (默认: 16)");
    println!("  <num_epochs>    - 训练轮数 (默认: 100)");
    println!("  <output_dir>    - 模型输出目录 (默认: ./checkpoints)");
    println!();
    println!("示例:");
    println!(
        "  cargo run --bin train_gnn --features pytorch -p vectorize -- 50 10 16 50 ./my_models"
    );
}
