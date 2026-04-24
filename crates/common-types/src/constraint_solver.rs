//! 几何约束求解器基础框架 (P1-3)
//!
//! # 概述
//!
//! 本模块实现几何约束求解器的基础框架，支持：
//! - 几何约束（平行、垂直、同心、相切等）
//! - 尺寸约束驱动
//! - 欠约束/过约束检测
//! - 约束求解引擎
//!
//! # 架构设计
//!
//! ```text
//! 约束定义 → 约束图构建 → 求解器选择 → 迭代求解 → 结果验证
//!    ↓           ↓           ↓          ↓          ↓
//! Constraint  ConstraintGraph  Solver  IterativeSolver  ValidationResult
//! ```

use crate::geometry::Point2;
use schemars::JsonSchema;
use serde::{Deserialize, Serialize};

// ============================================================================
// 几何约束类型
// ============================================================================

/// 几何约束定义
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum GeometricConstraint {
    /// 重合约束：两个点重合
    Coincident {
        point1: ConstraintPoint,
        point2: ConstraintPoint,
    },
    /// 平行约束：两条线段平行
    Parallel {
        line1: ConstraintLine,
        line2: ConstraintLine,
    },
    /// 垂直约束：两条线段垂直
    Perpendicular {
        line1: ConstraintLine,
        line2: ConstraintLine,
    },
    /// 同心约束：两个圆弧/圆同心
    Concentric {
        circle1: ConstraintCircle,
        circle2: ConstraintCircle,
    },
    /// 相切约束：曲线相切
    Tangent {
        curve1: ConstraintCurve,
        curve2: ConstraintCurve,
    },
    /// 等长约束：两条线段等长
    EqualLength {
        line1: ConstraintLine,
        line2: ConstraintLine,
    },
    /// 等半径约束：两个圆弧/圆等半径
    EqualRadius {
        circle1: ConstraintCircle,
        circle2: ConstraintCircle,
    },
    /// 中点约束：点在直线中点
    Midpoint {
        point: ConstraintPoint,
        line: ConstraintLine,
    },
    /// 点在曲线上约束
    PointOnCurve {
        point: ConstraintPoint,
        curve: ConstraintCurve,
    },
    /// 水平约束：线段水平
    Horizontal { line: ConstraintLine },
    /// 垂直约束：线段垂直
    Vertical { line: ConstraintLine },
}

/// 约束点引用
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ConstraintPoint {
    /// 引用的几何实体 ID
    pub entity_id: String,
    /// 点索引（用于多段线）
    pub point_index: Option<usize>,
}

/// 约束线段引用
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ConstraintLine {
    /// 引用的几何实体 ID
    pub entity_id: String,
    /// 起点索引（用于多段线）
    pub start_index: Option<usize>,
    /// 终点索引（用于多段线）
    pub end_index: Option<usize>,
}

/// 约束圆/圆弧引用
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ConstraintCircle {
    /// 引用的几何实体 ID
    pub entity_id: String,
}

/// 约束曲线引用
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ConstraintCurve {
    /// 引用的几何实体 ID
    pub entity_id: String,
}

// ============================================================================
// 尺寸约束
// ============================================================================

/// 尺寸约束定义
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum DimensionConstraint {
    /// 水平距离约束
    HorizontalDistance {
        target: ConstraintLine,
        distance: f64,
    },
    /// 垂直距离约束
    VerticalDistance {
        target: ConstraintLine,
        distance: f64,
    },
    /// 两点间距离约束
    Distance {
        point1: ConstraintPoint,
        point2: ConstraintPoint,
        distance: f64,
    },
    /// 角度约束
    Angle {
        line1: ConstraintLine,
        line2: ConstraintLine,
        angle: f64, // 单位：度
    },
    /// 半径约束
    Radius {
        circle: ConstraintCircle,
        radius: f64,
    },
    /// 直径约束
    Diameter {
        circle: ConstraintCircle,
        diameter: f64,
    },
}

// ============================================================================
// 约束求解器
// ============================================================================

/// 约束求解器配置
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ConstraintSolverConfig {
    /// 最大迭代次数
    pub max_iterations: usize,
    /// 收敛容差
    pub tolerance: f64,
    /// 是否启用欠约束检测
    pub detect_under_constrained: bool,
    /// 是否启用过约束检测
    pub detect_over_constrained: bool,
    /// 求解超时（毫秒）
    pub timeout_ms: Option<u64>,
}

impl Default for ConstraintSolverConfig {
    fn default() -> Self {
        Self {
            max_iterations: 100,
            tolerance: 1e-6,
            detect_under_constrained: true,
            detect_over_constrained: true,
            timeout_ms: Some(5000),
        }
    }
}

/// 约束求解状态
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize, JsonSchema)]
#[serde(rename_all = "snake_case")]
pub enum SolveStatus {
    /// 求解成功
    Solved,
    /// 求解收敛但未完全满足
    Converged,
    /// 求解失败（未收敛）
    Failed,
    /// 求解超时
    Timeout,
    /// 过约束（无解）
    OverConstrained,
    /// 欠约束（多解）
    UnderConstrained,
}

/// 约束求解结果
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct SolveResult {
    /// 求解状态
    pub status: SolveStatus,
    /// 迭代次数
    pub iterations: usize,
    /// 最终残差
    pub residual: f64,
    /// 求解时间（毫秒）
    pub solve_time_ms: f64,
    /// 约束满足报告
    pub constraint_report: Vec<ConstraintSatisfaction>,
    /// 更新后的几何位置
    pub updated_positions: Vec<(String, Point2)>,
}

/// 单个约束的满足情况
#[derive(Debug, Clone, Serialize, Deserialize, JsonSchema)]
pub struct ConstraintSatisfaction {
    /// 约束 ID
    pub constraint_id: usize,
    /// 是否满足
    pub satisfied: bool,
    /// 残差值
    pub residual: f64,
    /// 约束类型
    pub constraint_type: String,
}

/// 约束求解器
#[derive(Debug)]
pub struct ConstraintSolver {
    /// 求解器配置
    config: ConstraintSolverConfig,
    /// 几何约束列表
    geometric_constraints: Vec<GeometricConstraint>,
    /// 尺寸约束列表
    dimension_constraints: Vec<DimensionConstraint>,
    /// 自由度分析结果
    degrees_of_freedom: Option<usize>,
}

impl ConstraintSolver {
    /// 创建新的求解器
    pub fn new(config: ConstraintSolverConfig) -> Self {
        Self {
            config,
            geometric_constraints: Vec::new(),
            dimension_constraints: Vec::new(),
            degrees_of_freedom: None,
        }
    }

    /// 使用默认配置创建求解器
    pub fn with_default_config() -> Self {
        Self::new(ConstraintSolverConfig::default())
    }

    /// 添加几何约束
    pub fn add_geometric_constraint(&mut self, constraint: GeometricConstraint) {
        self.geometric_constraints.push(constraint);
    }

    /// 添加尺寸约束
    pub fn add_dimension_constraint(&mut self, constraint: DimensionConstraint) {
        self.dimension_constraints.push(constraint);
    }

    /// 清除所有约束
    pub fn clear_constraints(&mut self) {
        self.geometric_constraints.clear();
        self.dimension_constraints.clear();
        self.degrees_of_freedom = None;
    }

    /// 分析自由度
    ///
    /// # 参数
    /// - `num_variables`: 变量数量（通常是点数量的 2 倍）
    /// - `num_constraints`: 约束数量
    ///
    /// # 返回
    /// 自由度数（>0 表示欠约束，=0 表示完全约束，<0 表示过约束）
    pub fn analyze_degrees_of_freedom(&self, num_variables: usize, num_constraints: usize) -> i32 {
        num_variables as i32 - num_constraints as i32
    }

    /// 检测过约束
    pub fn is_over_constrained(&self, num_variables: usize) -> bool {
        self.analyze_degrees_of_freedom(num_variables, self.total_constraints()) < 0
    }

    /// 检测欠约束
    pub fn is_under_constrained(&self, num_variables: usize) -> bool {
        self.analyze_degrees_of_freedom(num_variables, self.total_constraints()) > 0
    }

    /// 获取总约束数
    pub fn total_constraints(&self) -> usize {
        self.geometric_constraints.len() + self.dimension_constraints.len()
    }

    /// 求解约束系统
    ///
    /// # 参数
    /// - `initial_positions`: 初始位置 { entity_id: position }
    ///
    /// # 返回
    /// 求解结果
    pub fn solve(
        &self,
        initial_positions: &std::collections::HashMap<String, Point2>,
    ) -> SolveResult {
        use std::time::Instant;
        let start_time = Instant::now();

        let num_variables = initial_positions.len() * 2;
        let total_constraints = self.total_constraints();
        let dof = self.analyze_degrees_of_freedom(num_variables, total_constraints);

        // 检查过约束
        if dof < 0 && self.config.detect_over_constrained {
            return SolveResult {
                status: SolveStatus::OverConstrained,
                iterations: 0,
                residual: 0.0,
                solve_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                constraint_report: Vec::new(),
                updated_positions: Vec::new(),
            };
        }

        // 检查欠约束
        if dof > 0 && self.config.detect_under_constrained {
            // 欠约束不是错误，继续求解
        }

        // 简化的求解逻辑（实际实现需要更复杂的数值求解）
        // 这里仅演示框架结构
        let mut iterations = 0;
        let mut residual = f64::MAX;

        // 模拟迭代求解
        while iterations < self.config.max_iterations {
            // 计算当前残差
            residual = self.compute_residual(initial_positions);

            // 检查收敛
            if residual < self.config.tolerance {
                break;
            }

            iterations += 1;

            // 检查超时
            if let Some(timeout_ms) = self.config.timeout_ms {
                if start_time.elapsed().as_millis() as u64 >= timeout_ms {
                    return SolveResult {
                        status: SolveStatus::Timeout,
                        iterations,
                        residual,
                        solve_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
                        constraint_report: self.generate_constraint_report(initial_positions),
                        updated_positions: initial_positions
                            .iter()
                            .map(|(k, v)| (k.clone(), *v))
                            .collect(),
                    };
                }
            }
        }

        let status = if residual < self.config.tolerance {
            SolveStatus::Solved
        } else {
            SolveStatus::Converged
        };

        SolveResult {
            status,
            iterations,
            residual,
            solve_time_ms: start_time.elapsed().as_secs_f64() * 1000.0,
            constraint_report: self.generate_constraint_report(initial_positions),
            updated_positions: initial_positions
                .iter()
                .map(|(k, v)| (k.clone(), *v))
                .collect(),
        }
    }

    /// 计算残差
    fn compute_residual(&self, positions: &std::collections::HashMap<String, Point2>) -> f64 {
        let mut total_residual = 0.0;

        // 计算几何约束残差
        for constraint in &self.geometric_constraints {
            total_residual += self.compute_constraint_residual(constraint, positions);
        }

        // 计算尺寸约束残差
        for constraint in &self.dimension_constraints {
            total_residual += self.compute_dimension_residual(constraint, positions);
        }

        total_residual
    }

    /// 计算单个几何约束的残差
    fn compute_constraint_residual(
        &self,
        constraint: &GeometricConstraint,
        positions: &std::collections::HashMap<String, Point2>,
    ) -> f64 {
        match constraint {
            GeometricConstraint::Coincident { point1, point2 } => {
                let p1 = self.get_point_position(point1, positions);
                let p2 = self.get_point_position(point2, positions);
                if let (Some(p1), Some(p2)) = (p1, p2) {
                    ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2)).sqrt()
                } else {
                    f64::MAX
                }
            }
            // 其他约束类型的残差计算...
            _ => 0.0,
        }
    }

    /// 计算尺寸约束残差
    fn compute_dimension_residual(
        &self,
        constraint: &DimensionConstraint,
        positions: &std::collections::HashMap<String, Point2>,
    ) -> f64 {
        match constraint {
            DimensionConstraint::Distance {
                point1,
                point2,
                distance,
            } => {
                let p1 = self.get_point_position(point1, positions);
                let p2 = self.get_point_position(point2, positions);
                if let (Some(p1), Some(p2)) = (p1, p2) {
                    let actual = ((p1[0] - p2[0]).powi(2) + (p1[1] - p2[1]).powi(2)).sqrt();
                    (actual - distance).abs()
                } else {
                    f64::MAX
                }
            }
            // 其他约束类型的残差计算...
            _ => 0.0,
        }
    }

    /// 获取点的位置
    fn get_point_position(
        &self,
        point: &ConstraintPoint,
        positions: &std::collections::HashMap<String, Point2>,
    ) -> Option<Point2> {
        positions.get(&point.entity_id).copied()
    }

    /// 生成约束满足报告
    fn generate_constraint_report(
        &self,
        positions: &std::collections::HashMap<String, Point2>,
    ) -> Vec<ConstraintSatisfaction> {
        let mut report = Vec::new();

        for (i, constraint) in self.geometric_constraints.iter().enumerate() {
            let residual = self.compute_constraint_residual(constraint, positions);
            report.push(ConstraintSatisfaction {
                constraint_id: i,
                satisfied: residual < self.config.tolerance,
                residual,
                constraint_type: self.get_constraint_type_name(constraint),
            });
        }

        report
    }

    /// 获取约束类型名称
    fn get_constraint_type_name(&self, constraint: &GeometricConstraint) -> String {
        match constraint {
            GeometricConstraint::Coincident { .. } => "Coincident".into(),
            GeometricConstraint::Parallel { .. } => "Parallel".into(),
            GeometricConstraint::Perpendicular { .. } => "Perpendicular".into(),
            GeometricConstraint::Concentric { .. } => "Concentric".into(),
            GeometricConstraint::Tangent { .. } => "Tangent".into(),
            GeometricConstraint::EqualLength { .. } => "EqualLength".into(),
            GeometricConstraint::EqualRadius { .. } => "EqualRadius".into(),
            GeometricConstraint::Midpoint { .. } => "Midpoint".into(),
            GeometricConstraint::PointOnCurve { .. } => "PointOnCurve".into(),
            GeometricConstraint::Horizontal { .. } => "Horizontal".into(),
            GeometricConstraint::Vertical { .. } => "Vertical".into(),
        }
    }
}

// ============================================================================
// 测试
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_constraint_solver_creation() {
        let solver = ConstraintSolver::with_default_config();
        assert_eq!(solver.total_constraints(), 0);
    }

    #[test]
    fn test_add_geometric_constraint() {
        let mut solver = ConstraintSolver::with_default_config();

        solver.add_geometric_constraint(GeometricConstraint::Coincident {
            point1: ConstraintPoint {
                entity_id: "P1".into(),
                point_index: None,
            },
            point2: ConstraintPoint {
                entity_id: "P2".into(),
                point_index: None,
            },
        });

        assert_eq!(solver.total_constraints(), 1);
    }

    #[test]
    fn test_add_dimension_constraint() {
        let mut solver = ConstraintSolver::with_default_config();

        solver.add_dimension_constraint(DimensionConstraint::Distance {
            point1: ConstraintPoint {
                entity_id: "P1".into(),
                point_index: None,
            },
            point2: ConstraintPoint {
                entity_id: "P2".into(),
                point_index: None,
            },
            distance: 100.0,
        });

        assert_eq!(solver.total_constraints(), 1);
    }

    #[test]
    fn test_degrees_of_freedom_analysis() {
        let solver = ConstraintSolver::with_default_config();

        // 4 个点（8 个变量），3 个约束
        let dof = solver.analyze_degrees_of_freedom(8, 3);
        assert_eq!(dof, 5); // 8 - 3 = 5

        // 4 个点（8 个变量），8 个约束
        let dof = solver.analyze_degrees_of_freedom(8, 8);
        assert_eq!(dof, 0); // 完全约束

        // 4 个点（8 个变量），10 个约束
        let dof = solver.analyze_degrees_of_freedom(8, 10);
        assert_eq!(dof, -2); // 过约束
    }

    #[test]
    fn test_solve_coincident_constraint() {
        let mut solver = ConstraintSolver::new(ConstraintSolverConfig {
            max_iterations: 100,
            tolerance: 1e-6,
            detect_under_constrained: false,
            detect_over_constrained: false,
            timeout_ms: None,
        });

        solver.add_geometric_constraint(GeometricConstraint::Coincident {
            point1: ConstraintPoint {
                entity_id: "P1".into(),
                point_index: None,
            },
            point2: ConstraintPoint {
                entity_id: "P2".into(),
                point_index: None,
            },
        });

        let mut positions = HashMap::new();
        positions.insert("P1".into(), [0.0, 0.0]);
        positions.insert("P2".into(), [1.0, 1.0]); // 不重合

        let result = solver.solve(&positions);

        // 由于点是分开的，残差应该大于 0
        assert!(result.residual > 0.0);
    }

    #[test]
    fn test_solve_distance_constraint() {
        let mut solver = ConstraintSolver::new(ConstraintSolverConfig {
            max_iterations: 100,
            tolerance: 1e-6,
            detect_under_constrained: false,
            detect_over_constrained: false,
            timeout_ms: None,
        });

        solver.add_dimension_constraint(DimensionConstraint::Distance {
            point1: ConstraintPoint {
                entity_id: "P1".into(),
                point_index: None,
            },
            point2: ConstraintPoint {
                entity_id: "P2".into(),
                point_index: None,
            },
            distance: 100.0,
        });

        let mut positions = HashMap::new();
        positions.insert("P1".into(), [0.0, 0.0]);
        positions.insert("P2".into(), [50.0, 0.0]); // 距离 50，不是 100

        let result = solver.solve(&positions);

        // 残差应该是 50（100 - 50）
        assert!((result.residual - 50.0).abs() < 1e-6);
    }
}
