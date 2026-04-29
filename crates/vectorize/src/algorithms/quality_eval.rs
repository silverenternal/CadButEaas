//! 矢量化质量评估指标体系
//!
//! 提供多维度的矢量化质量评分：
//! - 几何精度：Chamfer/Hausdorff 距离
//! - 拓扑正确性：同构检查
//! - 语义准确率：混淆矩阵
//! - 综合评分 0-100

use common_types::geometry::Point2;

/// 几何精度评估结果
#[derive(Debug, Clone)]
pub struct GeometryQuality {
    /// Hausdorff 距离（最大偏差）
    pub hausdorff_distance: f64,
    /// Chamfer 距离（平均偏差）
    pub chamfer_distance: f64,
    /// 均方根误差
    pub rmse: f64,
    /// 几何精度评分（0-100，越低越好）
    pub score: f64,
}

/// 拓扑正确性评估结果
#[derive(Debug, Clone)]
pub struct TopologyQuality {
    /// 顶点数匹配率
    pub vertex_match_ratio: f64,
    /// 边数匹配率
    pub edge_match_ratio: f64,
    /// 面数匹配率
    pub face_match_ratio: f64,
    /// 同构相似度（0-1，1=完全同构）
    pub isomorphism_score: f64,
    /// 拓扑正确性评分（0-100）
    pub score: f64,
}

/// 语义准确率评估结果
#[derive(Debug, Clone)]
pub struct SemanticQuality {
    /// 总体准确率
    pub overall_accuracy: f64,
    /// 宏平均 F1 分数
    pub macro_f1: f64,
    /// 微平均 F1 分数
    pub micro_f1: f64,
    /// 各类别的混淆矩阵统计
    pub per_class_stats: Vec<ClassStats>,
    /// 语义准确率评分（0-100）
    pub score: f64,
}

/// 单类别统计
#[derive(Debug, Clone)]
pub struct ClassStats {
    /// 类别名称
    pub class_name: String,
    /// 精确率
    pub precision: f64,
    /// 召回率
    pub recall: f64,
    /// F1 分数
    pub f1_score: f64,
    /// 样本数
    pub samples: usize,
}

/// 总体质量评估结果
#[derive(Debug, Clone)]
pub struct QualityResult {
    /// 几何精度
    pub geometry: GeometryQuality,
    /// 拓扑正确性
    pub topology: TopologyQuality,
    /// 语义准确率
    pub semantic: SemanticQuality,
    /// 综合质量评分（0-100）
    pub overall_score: f64,
    /// 评估时间戳
    pub timestamp: std::time::SystemTime,
}

/// 质量评估器配置
#[derive(Debug, Clone)]
pub struct QualityEvaluatorConfig {
    /// 几何精度权重
    pub geometry_weight: f64,
    /// 拓扑正确性权重
    pub topology_weight: f64,
    /// 语义准确率权重
    pub semantic_weight: f64,
    /// Hausdorff 距离阈值（像素，超过得 0 分）
    pub hausdorff_threshold: f64,
    /// Chamfer 距离阈值（像素，超过得 0 分）
    pub chamfer_threshold: f64,
}

impl Default for QualityEvaluatorConfig {
    fn default() -> Self {
        Self {
            geometry_weight: 0.5,
            topology_weight: 0.3,
            semantic_weight: 0.2,
            hausdorff_threshold: 5.0,
            chamfer_threshold: 2.0,
        }
    }
}

/// 质量评估器
#[derive(Debug, Clone)]
pub struct QualityEvaluator {
    config: QualityEvaluatorConfig,
}

impl Default for QualityEvaluator {
    fn default() -> Self {
        Self::new(QualityEvaluatorConfig::default())
    }
}

impl QualityEvaluator {
    /// 创建新的评估器
    pub fn new(config: QualityEvaluatorConfig) -> Self {
        Self { config }
    }

    /// 计算两点间欧氏距离
    fn point_distance(a: Point2, b: Point2) -> f64 {
        let dx = a[0] - b[0];
        let dy = a[1] - b[1];
        (dx * dx + dy * dy).sqrt()
    }

    /// 找到点集中距离给定点最近的点的距离
    fn find_min_distance(point: Point2, point_set: &[Point2]) -> f64 {
        point_set
            .iter()
            .map(|&p| Self::point_distance(point, p))
            .fold(f64::MAX, f64::min)
    }

    /// 计算单向 Hausdorff 距离
    fn hausdorff_directed(a: &[Point2], b: &[Point2]) -> f64 {
        a.iter()
            .map(|&p| Self::find_min_distance(p, b))
            .fold(0.0, f64::max)
    }

    /// 计算双向 Hausdorff 距离
    pub fn hausdorff_distance(a: &[Point2], b: &[Point2]) -> f64 {
        if a.is_empty() || b.is_empty() {
            return f64::MAX;
        }
        f64::max(
            Self::hausdorff_directed(a, b),
            Self::hausdorff_directed(b, a),
        )
    }

    /// 计算 Chamfer 距离
    pub fn chamfer_distance(a: &[Point2], b: &[Point2]) -> f64 {
        if a.is_empty() || b.is_empty() {
            return f64::MAX;
        }

        let sum_ab: f64 = a.iter().map(|&p| Self::find_min_distance(p, b)).sum();
        let sum_ba: f64 = b.iter().map(|&p| Self::find_min_distance(p, a)).sum();

        sum_ab / a.len() as f64 + sum_ba / b.len() as f64
    }

    /// 计算 RMSE（均方根误差）
    pub fn rmse(a: &[Point2], b: &[Point2]) -> f64 {
        if a.is_empty() || b.is_empty() {
            return f64::MAX;
        }

        let n = a.len().min(b.len());
        let mut sum_sq = 0.0;
        for i in 0..n {
            let dx = a[i][0] - b[i][0];
            let dy = a[i][1] - b[i][1];
            sum_sq += dx * dx + dy * dy;
        }

        (sum_sq / n as f64).sqrt()
    }

    /// 评估几何精度
    pub fn evaluate_geometry(&self, ground_truth: &[Point2], result: &[Point2]) -> GeometryQuality {
        let hausdorff = Self::hausdorff_distance(ground_truth, result);
        let chamfer = Self::chamfer_distance(ground_truth, result);
        let rmse = Self::rmse(ground_truth, result);

        // 计算几何精度评分（越高越好）
        let hausdorff_score =
            (1.0 - (hausdorff / self.config.hausdorff_threshold).min(1.0)) * 100.0;
        let chamfer_score = (1.0 - (chamfer / self.config.chamfer_threshold).min(1.0)) * 100.0;
        let score = (hausdorff_score * 0.6 + chamfer_score * 0.4).max(0.0);

        GeometryQuality {
            hausdorff_distance: hausdorff,
            chamfer_distance: chamfer,
            rmse,
            score,
        }
    }

    /// 评估拓扑正确性（简化版本）
    /// 使用顶点数、边数和面数的匹配程度作为代理指标
    pub fn evaluate_topology_simple(
        &self,
        gt_vertices: usize,
        gt_edges: usize,
        gt_faces: usize,
        result_vertices: usize,
        result_edges: usize,
        result_faces: usize,
    ) -> TopologyQuality {
        let vertex_ratio = if gt_vertices > 0 {
            1.0 - ((result_vertices as isize - gt_vertices as isize).abs() as f64)
                / gt_vertices as f64
        } else {
            1.0
        }
        .max(0.0);

        let edge_ratio = if gt_edges > 0 {
            1.0 - ((result_edges as isize - gt_edges as isize).abs() as f64) / gt_edges as f64
        } else {
            1.0
        }
        .max(0.0);

        let face_ratio = if gt_faces > 0 {
            1.0 - ((result_faces as isize - gt_faces as isize).abs() as f64) / gt_faces as f64
        } else {
            1.0
        }
        .max(0.0);

        // 简单的同构相似度估计
        let isomorphism = (vertex_ratio + edge_ratio + face_ratio) / 3.0;

        // 拓扑评分
        let score = (vertex_ratio * 0.3 + edge_ratio * 0.4 + face_ratio * 0.3) * 100.0;

        TopologyQuality {
            vertex_match_ratio: vertex_ratio,
            edge_match_ratio: edge_ratio,
            face_match_ratio: face_ratio,
            isomorphism_score: isomorphism,
            score,
        }
    }

    /// 从混淆矩阵计算语义质量
    /// 输入：混淆矩阵（行=预测，列=真实），类别名称列表
    pub fn evaluate_semantic(
        &self,
        confusion_matrix: &[Vec<usize>],
        class_names: &[String],
    ) -> SemanticQuality {
        let n_classes = class_names.len();
        if confusion_matrix.len() != n_classes {
            return SemanticQuality {
                overall_accuracy: 0.0,
                macro_f1: 0.0,
                micro_f1: 0.0,
                per_class_stats: Vec::new(),
                score: 0.0,
            };
        }

        let mut per_class_stats = Vec::with_capacity(n_classes);
        let mut total_correct = 0;
        let mut total_samples = 0;
        let mut sum_f1 = 0.0;
        let mut tp_sum = 0;
        let mut fp_sum = 0;
        let mut fn_sum = 0;

        for i in 0..n_classes {
            let tp = confusion_matrix[i][i];
            let fp: usize = confusion_matrix[i].iter().sum::<usize>() - tp;
            let fn_: usize = confusion_matrix.iter().map(|row| row[i]).sum::<usize>() - tp;

            total_correct += tp;
            total_samples += confusion_matrix[i].iter().sum::<usize>();
            tp_sum += tp;
            fp_sum += fp;
            fn_sum += fn_;

            let precision = if tp + fp > 0 {
                tp as f64 / (tp + fp) as f64
            } else {
                0.0
            };
            let recall = if tp + fn_ > 0 {
                tp as f64 / (tp + fn_) as f64
            } else {
                0.0
            };
            let f1 = if precision + recall > 0.0 {
                2.0 * precision * recall / (precision + recall)
            } else {
                0.0
            };

            sum_f1 += f1;

            per_class_stats.push(ClassStats {
                class_name: class_names[i].clone(),
                precision,
                recall,
                f1_score: f1,
                samples: tp + fn_,
            });
        }

        let overall_accuracy = if total_samples > 0 {
            total_correct as f64 / total_samples as f64
        } else {
            0.0
        };

        let macro_f1 = sum_f1 / n_classes as f64;

        let micro_precision = if tp_sum + fp_sum > 0 {
            tp_sum as f64 / (tp_sum + fp_sum) as f64
        } else {
            0.0
        };
        let micro_recall = if tp_sum + fn_sum > 0 {
            tp_sum as f64 / (tp_sum + fn_sum) as f64
        } else {
            0.0
        };
        let micro_f1 = if micro_precision + micro_recall > 0.0 {
            2.0 * micro_precision * micro_recall / (micro_precision + micro_recall)
        } else {
            0.0
        };

        let score = overall_accuracy * 100.0;

        SemanticQuality {
            overall_accuracy,
            macro_f1,
            micro_f1,
            per_class_stats,
            score,
        }
    }

    /// 执行完整的质量评估
    #[allow(clippy::too_many_arguments)]
    pub fn evaluate_full(
        &self,
        gt_points: &[Point2],
        result_points: &[Point2],
        gt_vertices: usize,
        gt_edges: usize,
        gt_faces: usize,
        result_vertices: usize,
        result_edges: usize,
        result_faces: usize,
        confusion_matrix: &[Vec<usize>],
        class_names: &[String],
    ) -> QualityResult {
        let geometry = self.evaluate_geometry(gt_points, result_points);
        let topology = self.evaluate_topology_simple(
            gt_vertices,
            gt_edges,
            gt_faces,
            result_vertices,
            result_edges,
            result_faces,
        );
        let semantic = self.evaluate_semantic(confusion_matrix, class_names);

        let overall_score = geometry.score * self.config.geometry_weight
            + topology.score * self.config.topology_weight
            + semantic.score * self.config.semantic_weight;

        QualityResult {
            geometry,
            topology,
            semantic,
            overall_score,
            timestamp: std::time::SystemTime::now(),
        }
    }

    /// 快速评估（仅几何精度）
    pub fn evaluate_quick(&self, gt_points: &[Point2], result_points: &[Point2]) -> f64 {
        self.evaluate_geometry(gt_points, result_points).score
    }
}

/// 便捷函数：计算两个点集的快速质量评分
pub fn quick_quality_score(ground_truth: &[Point2], result: &[Point2]) -> f64 {
    let evaluator = QualityEvaluator::default();
    evaluator.evaluate_quick(ground_truth, result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_point_distance() {
        let a = [0.0, 0.0];
        let b = [3.0, 4.0];
        assert!((QualityEvaluator::point_distance(a, b) - 5.0).abs() < 1e-6);
    }

    #[test]
    fn test_hausdorff_distance_identical() {
        let points = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let distance = QualityEvaluator::hausdorff_distance(&points, &points);
        assert!(distance.abs() < 1e-6);
    }

    #[test]
    fn test_hausdorff_distance_shifted() {
        let a = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let b = vec![[0.5, 0.0], [1.5, 0.0], [1.5, 1.0], [0.5, 1.0]];
        let distance = QualityEvaluator::hausdorff_distance(&a, &b);
        assert!((distance - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_chamfer_distance_identical() {
        let points = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let distance = QualityEvaluator::chamfer_distance(&points, &points);
        assert!(distance.abs() < 1e-6);
    }

    #[test]
    fn test_rmse_identical() {
        let points = vec![[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]];
        let rmse = QualityEvaluator::rmse(&points, &points);
        assert!(rmse.abs() < 1e-6);
    }

    #[test]
    fn test_evaluate_geometry_perfect() {
        let evaluator = QualityEvaluator::default();
        let points = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let result = evaluator.evaluate_geometry(&points, &points);
        assert!((result.score - 100.0).abs() < 1e-6);
        assert!(result.hausdorff_distance.abs() < 1e-6);
    }

    #[test]
    fn test_evaluate_topology_perfect() {
        let evaluator = QualityEvaluator::default();
        let result = evaluator.evaluate_topology_simple(4, 4, 1, 4, 4, 1);
        assert!((result.score - 100.0).abs() < 1e-6);
        assert!((result.vertex_match_ratio - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_evaluate_topology_mismatch() {
        let evaluator = QualityEvaluator::default();
        let result = evaluator.evaluate_topology_simple(4, 4, 1, 8, 4, 1);
        // 顶点数相差一倍，匹配率 0
        assert!(result.score < 100.0);
    }

    #[test]
    fn test_evaluate_semantic_perfect() {
        let evaluator = QualityEvaluator::default();
        let confusion_matrix = vec![vec![10, 0], vec![0, 15]];
        let class_names = vec!["A".to_string(), "B".to_string()];
        let result = evaluator.evaluate_semantic(&confusion_matrix, &class_names);
        assert!((result.overall_accuracy - 1.0).abs() < 1e-6);
        assert!((result.macro_f1 - 1.0).abs() < 1e-6);
        assert!((result.score - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_quick_quality_score() {
        let points = vec![[0.0, 0.0], [10.0, 0.0], [10.0, 10.0], [0.0, 10.0]];
        let score = quick_quality_score(&points, &points);
        assert!((score - 100.0).abs() < 1e-6);
    }

    #[test]
    fn test_evaluator_creation() {
        let evaluator = QualityEvaluator::new(QualityEvaluatorConfig::default());
        assert!((evaluator.config.geometry_weight - 0.5).abs() < 1e-6);
    }
}
