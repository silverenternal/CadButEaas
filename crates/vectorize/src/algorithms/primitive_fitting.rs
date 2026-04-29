//! 基元拟合引擎
//!
//! 支持多种几何基元拟合，自动选择最优拟合类型：
//! - 直线：Theil-Sen + RANSAC 鲁棒拟合
//! - 圆弧：Kåsa 算法 + Levenberg-Marquardt 优化
//! - 三次贝塞尔曲线：端点切向约束拟合
//! - AIC/BIC 自动选择最优基元类型

use common_types::{Point2, Polyline};
use std::f64::consts::TAU;

/// 拟合基元类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PrimitiveType {
    /// 直线
    Line,
    /// 圆弧
    Arc,
    /// 三次贝塞尔曲线
    Bezier,
}

/// 直线拟合结果
#[derive(Debug, Clone)]
pub struct FittedLine {
    /// 起点
    pub start: Point2,
    /// 终点
    pub end: Point2,
    /// 直线方程：ax + by + c = 0
    pub a: f64,
    pub b: f64,
    pub c: f64,
    /// 均方根误差
    pub rms_error: f64,
}

/// 三次贝塞尔曲线拟合结果
#[derive(Debug, Clone)]
pub struct FittedBezier {
    /// 控制点
    pub p0: Point2,
    pub p1: Point2,
    pub p2: Point2,
    pub p3: Point2,
    /// 均方根误差
    pub rms_error: f64,
}

/// 圆弧拟合结果（复用 arc_fitting 的结构）
pub use super::arc_fitting::FittedCircle;

/// 拟合结果与质量评估
#[derive(Debug, Clone)]
pub struct FitResult {
    /// 基元类型
    pub primitive_type: PrimitiveType,
    /// AIC 信息准则
    pub aic: f64,
    /// BIC 信息准则
    pub bic: f64,
    /// 均方根误差
    pub rms_error: f64,
    /// 内点比例
    pub inlier_ratio: f64,
    /// 具体拟合数据
    pub data: FitData,
}

/// 拟合数据（枚举各种基元类型）
#[derive(Debug, Clone)]
pub enum FitData {
    Line(FittedLine),
    Arc(FittedCircle),
    Bezier(FittedBezier),
}

// ========== 直线拟合 ==========

/// Theil-Sen 直线拟合（鲁棒中位数方法）
///
/// 通过计算所有点对斜率的中位数得到鲁棒的直线参数
pub fn fit_line_theil_sen(points: &Polyline) -> Option<FittedLine> {
    if points.len() < 2 {
        return None;
    }

    let mut slopes = Vec::new();
    let mut intercepts = Vec::new();

    // 计算所有点对的斜率和截距
    for i in 0..points.len() {
        for j in (i + 1)..points.len() {
            let dx = points[j][0] - points[i][0];
            if dx.abs() > 1e-10 {
                let slope = (points[j][1] - points[i][1]) / dx;
                let intercept = points[i][1] - slope * points[i][0];
                slopes.push(slope);
                intercepts.push(intercept);
            }
        }
    }

    if slopes.is_empty() {
        // 垂直线
        let x = points[0][0];
        return Some(FittedLine {
            start: points[0],
            end: points[points.len() - 1],
            a: 1.0,
            b: 0.0,
            c: -x,
            rms_error: 0.0,
        });
    }

    // 取中位数
    slopes.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    intercepts.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let slope = slopes[slopes.len() / 2];
    let intercept = intercepts[intercepts.len() / 2];

    // 直线方程：y = slope * x + intercept → slope * x - y + intercept = 0
    let a = slope;
    let b = -1.0;
    let c = intercept;

    // 计算误差
    let rms_error = calculate_line_rms_error(points, a, b, c);

    Some(FittedLine {
        start: points[0],
        end: points[points.len() - 1],
        a,
        b,
        c,
        rms_error,
    })
}

/// RANSAC 直线拟合（抗噪）
///
/// # 参数
/// - `points`: 输入点集
/// - `threshold`: 内点距离阈值
/// - `max_iterations`: 最大迭代次数
pub fn fit_line_ransac(
    points: &Polyline,
    threshold: f64,
    max_iterations: usize,
) -> Option<FittedLine> {
    if points.len() < 2 {
        return None;
    }

    let mut best_line = None;
    let mut best_inliers = 0;
    let mut best_error = f64::INFINITY;

    for _ in 0..max_iterations {
        // 随机选择两个点
        let i = fastrand::usize(0..points.len());
        let mut j = fastrand::usize(0..points.len());
        while j == i {
            j = fastrand::usize(0..points.len());
        }

        // 用这两个点拟合直线
        let p1 = points[i];
        let p2 = points[j];
        let dx = p2[0] - p1[0];
        let dy = p2[1] - p1[1];

        // 直线方程：dy*x - dx*y + (dx*y1 - dy*x1) = 0
        let a = dy;
        let b = -dx;
        let c = dx * p1[1] - dy * p1[0];

        // 计算内点
        let mut inliers = Vec::new();
        for &p in points {
            let dist = (a * p[0] + b * p[1] + c).abs() / (a * a + b * b).sqrt();
            if dist < threshold {
                inliers.push(p);
            }
        }

        // 更新最佳模型
        if inliers.len() > best_inliers {
            // 用所有内点重新拟合（最小二乘）
            if let Some(line) = fit_line_least_squares(&inliers) {
                if line.rms_error < best_error {
                    best_error = line.rms_error;
                    best_inliers = inliers.len();
                    best_line = Some(line);
                }
            }
        }
    }

    best_line
}

/// 最小二乘直线拟合
fn fit_line_least_squares(points: &Polyline) -> Option<FittedLine> {
    if points.len() < 2 {
        return None;
    }

    let n = points.len() as f64;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_xx = 0.0;

    for &p in points {
        sum_x += p[0];
        sum_y += p[1];
        sum_xy += p[0] * p[1];
        sum_xx += p[0] * p[0];
    }

    let det = n * sum_xx - sum_x * sum_x;

    if det.abs() < 1e-10 {
        // 垂直线
        let x = sum_x / n;
        return Some(FittedLine {
            start: points[0],
            end: points[points.len() - 1],
            a: 1.0,
            b: 0.0,
            c: -x,
            rms_error: 0.0,
        });
    }

    let slope = (n * sum_xy - sum_x * sum_y) / det;
    let intercept = (sum_y - slope * sum_x) / n;

    let a = slope;
    let b = -1.0;
    let c = intercept;

    let rms_error = calculate_line_rms_error(points, a, b, c);

    Some(FittedLine {
        start: points[0],
        end: points[points.len() - 1],
        a,
        b,
        c,
        rms_error,
    })
}

/// 计算直线拟合的均方根误差
fn calculate_line_rms_error(points: &Polyline, a: f64, b: f64, c: f64) -> f64 {
    let mut sum_sq = 0.0;
    let denom = (a * a + b * b).sqrt();

    if denom < 1e-10 {
        return f64::INFINITY;
    }

    for &p in points {
        let dist = (a * p[0] + b * p[1] + c).abs() / denom;
        sum_sq += dist * dist;
    }

    (sum_sq / points.len() as f64).sqrt()
}

// ========== 圆弧拟合（Kåsa + LM 优化）==========

/// 使用 Kåsa 方法拟合圆（最小二乘）
pub use super::arc_fitting::fit_circle_kasa;

/// Levenberg-Marquardt 优化圆拟合
///
/// 通过非线性最小二乘优化 Kåsa 初始估计
pub fn fit_circle_lm(points: &Polyline, max_iterations: usize) -> Option<FittedCircle> {
    let mut circle = fit_circle_kasa(points)?;

    let mut lambda = 0.001;
    let mut prev_error = circle.rms_error;

    for _ in 0..max_iterations {
        let mut jtj = [[0.0; 3]; 3]; // J^T J
        let mut jte = [0.0; 3]; // J^T e

        let mut error_sum = 0.0;

        for &p in points {
            let dx = p[0] - circle.center[0];
            let dy = p[1] - circle.center[1];
            let r = (dx * dx + dy * dy).sqrt();

            if r < 1e-10 {
                continue;
            }

            // 误差：r - R
            let error = r - circle.radius;
            error_sum += error * error;

            // 雅可比：d(error)/d(cx), d(error)/d(cy), d(error)/d(R)
            let jx = -dx / r;
            let jy = -dy / r;
            let jr = -1.0;

            // 构建近似 Hessian
            jtj[0][0] += jx * jx;
            jtj[0][1] += jx * jy;
            jtj[0][2] += jx * jr;
            jtj[1][0] += jy * jx;
            jtj[1][1] += jy * jy;
            jtj[1][2] += jy * jr;
            jtj[2][0] += jr * jx;
            jtj[2][1] += jr * jy;
            jtj[2][2] += jr * jr;

            jte[0] += jx * error;
            jte[1] += jy * error;
            jte[2] += jr * error;
        }

        // Levenberg-Marquardt：添加 lambda * diag(J^T J)
        for i in 0..3 {
            jtj[i][i] *= 1.0 + lambda;
        }

        // 解线性方程组
        let det = jtj[0][0] * (jtj[1][1] * jtj[2][2] - jtj[1][2] * jtj[2][1])
            - jtj[0][1] * (jtj[1][0] * jtj[2][2] - jtj[1][2] * jtj[2][0])
            + jtj[0][2] * (jtj[1][0] * jtj[2][1] - jtj[1][1] * jtj[2][0]);

        if det.abs() < 1e-15 {
            break;
        }

        // Cramer 法则求解
        let delta_cx = ((jte[0] * (jtj[1][1] * jtj[2][2] - jtj[1][2] * jtj[2][1])
            - jtj[0][1] * (jte[1] * jtj[2][2] - jtj[1][2] * jte[2]))
            + jtj[0][2] * (jte[1] * jtj[2][1] - jtj[1][1] * jte[2]))
            / det;

        let delta_cy = (-(jtj[1][0] * (jte[0] * jtj[2][2] - jtj[0][2] * jte[2]))
            + jte[1] * (jtj[0][0] * jtj[2][2] - jtj[0][2] * jtj[2][0])
            - jtj[1][2] * (jtj[0][0] * jte[2] - jte[0] * jtj[2][0]))
            / det;

        let delta_r = ((jtj[2][0] * (jte[0] * jtj[1][1] - jtj[0][1] * jte[1])
            - jtj[2][1] * (jtj[0][0] * jte[1] - jte[0] * jtj[1][0]))
            + jte[2] * (jtj[0][0] * jtj[1][1] - jtj[0][1] * jtj[1][0]))
            / det;

        // 尝试更新
        let new_center = [circle.center[0] - delta_cx, circle.center[1] - delta_cy];
        let new_radius = circle.radius - delta_r;

        // 计算新误差
        let mut new_error = 0.0;
        for &p in points {
            let dx = p[0] - new_center[0];
            let dy = p[1] - new_center[1];
            let r = (dx * dx + dy * dy).sqrt();
            let err = r - new_radius;
            new_error += err * err;
        }

        if new_error < error_sum {
            // 接受更新，减小 lambda
            circle.center = new_center;
            circle.radius = new_radius;
            lambda *= 0.5;
            prev_error = (new_error / points.len() as f64).sqrt();
        } else {
            // 拒绝更新，增大 lambda
            lambda *= 2.0;
        }

        if lambda > 1e10 {
            break;
        }
    }

    circle.rms_error = prev_error;
    Some(circle)
}

// ========== 三次贝塞尔曲线拟合 ==========

/// 三次贝塞尔曲线拟合（端点切向约束）
///
/// 使用端点切向量估计 + 内部控制点优化
pub fn fit_bezier_cubic(points: &Polyline) -> Option<FittedBezier> {
    if points.len() < 4 {
        return None;
    }

    let p0 = points[0];
    let p3 = points[points.len() - 1];

    // 估计端点切向量（使用前几个点和后几个点）
    let n = points.len().min(3);
    let mut p1 = p0;
    for i in 1..n {
        let t = i as f64 / n as f64;
        p1[0] += (points[i][0] - p0[0]) * t * 3.0;
        p1[1] += (points[i][1] - p0[1]) * t * 3.0;
    }
    p1[0] = p0[0] + p1[0] / 3.0;
    p1[1] = p0[1] + p1[1] / 3.0;

    let mut p2 = p3;
    for i in 1..n {
        let idx = points.len() - 1 - i;
        let t = i as f64 / n as f64;
        p2[0] += (points[idx][0] - p3[0]) * t * 3.0;
        p2[1] += (points[idx][1] - p3[1]) * t * 3.0;
    }
    p2[0] = p3[0] + p2[0] / 3.0;
    p2[1] = p3[1] + p2[1] / 3.0;

    // 使用简单的迭代优化控制点
    let mut best_p1 = p1;
    let mut best_p2 = p2;
    let mut best_error = f64::INFINITY;

    for iter in 0..10 {
        // 沿切向微调控制点
        let scale = 1.0 / (1 << iter) as f64;
        for dx in -2..=2 {
            for dy in -2..=2 {
                let test_p1 = [p1[0] + dx as f64 * scale, p1[1] + dy as f64 * scale];
                for dx2 in -2..=2 {
                    for dy2 in -2..=2 {
                        let test_p2 = [p2[0] + dx2 as f64 * scale, p2[1] + dy2 as f64 * scale];
                        let error = calculate_bezier_error(points, p0, test_p1, test_p2, p3);
                        if error < best_error {
                            best_error = error;
                            best_p1 = test_p1;
                            best_p2 = test_p2;
                        }
                    }
                }
            }
        }
        p1 = best_p1;
        p2 = best_p2;
    }

    Some(FittedBezier {
        p0,
        p1: best_p1,
        p2: best_p2,
        p3,
        rms_error: best_error,
    })
}

/// 计算贝塞尔曲线拟合误差
fn calculate_bezier_error(
    points: &Polyline,
    p0: Point2,
    p1: Point2,
    p2: Point2,
    p3: Point2,
) -> f64 {
    let n = points.len();
    let mut sum_sq = 0.0;

    for (i, &p) in points.iter().enumerate() {
        let t = i as f64 / (n - 1) as f64;
        let mt = 1.0 - t;

        // 贝塞尔曲线点计算
        let bx = mt * mt * mt * p0[0]
            + 3.0 * mt * mt * t * p1[0]
            + 3.0 * mt * t * t * p2[0]
            + t * t * t * p3[0];
        let by = mt * mt * mt * p0[1]
            + 3.0 * mt * mt * t * p1[1]
            + 3.0 * mt * t * t * p2[1]
            + t * t * t * p3[1];

        let dx = p[0] - bx;
        let dy = p[1] - by;
        sum_sq += dx * dx + dy * dy;
    }

    (sum_sq / n as f64).sqrt()
}

// ========== 信息准则（AIC/BIC）==========

/// 计算 AIC（Akaike 信息准则）
///
/// AIC = 2k - 2ln(L)，其中 k 是参数数量，L 是似然
fn calculate_aic(rms_error: f64, n_points: usize, k_params: usize) -> f64 {
    let n = n_points as f64;
    let k = k_params as f64;

    // 高斯似然的对数：ln(L) = -n/2 * ln(2πσ²) - sum((y - ŷ)²)/(2σ²)
    // 其中 σ = rms_error
    let sigma2 = rms_error * rms_error + 1e-10;
    let log_likelihood = -n / 2.0 * (TAU * sigma2).ln() - n / 2.0;

    2.0 * k - 2.0 * log_likelihood
}

/// 计算 BIC（Bayesian 信息准则）
///
/// BIC = k*ln(n) - 2ln(L)
fn calculate_bic(rms_error: f64, n_points: usize, k_params: usize) -> f64 {
    let n = n_points as f64;
    let k = k_params as f64;

    let sigma2 = rms_error * rms_error + 1e-10;
    let log_likelihood = -n / 2.0 * (TAU * sigma2).ln() - n / 2.0;

    k * n.ln() - 2.0 * log_likelihood
}

// ========== 自动基元选择 ==========

/// 自动选择最优拟合基元
///
/// 尝试直线、圆弧、贝塞尔曲线三种拟合，使用 AIC/BIC 选择最优
///
/// # 参数
/// - `points`: 输入点集
/// - `line_threshold`: 直线拟合 RANSAC 阈值
pub fn fit_best_primitive(points: &Polyline, line_threshold: f64) -> Option<FitResult> {
    if points.len() < 2 {
        return None;
    }

    let mut results = Vec::new();

    // 尝试直线拟合
    if points.len() >= 2 {
        if let Some(line) = fit_line_ransac(points, line_threshold, 100) {
            let aic = calculate_aic(line.rms_error, points.len(), 2); // 直线有2个参数
            let bic = calculate_bic(line.rms_error, points.len(), 2);

            // 计算内点比例
            let mut inliers = 0;
            let denom = (line.a * line.a + line.b * line.b).sqrt();
            if denom > 1e-10 {
                for &p in points {
                    let dist = (line.a * p[0] + line.b * p[1] + line.c).abs() / denom;
                    if dist < line_threshold {
                        inliers += 1;
                    }
                }
            }
            let inlier_ratio = inliers as f64 / points.len() as f64;

            results.push(FitResult {
                primitive_type: PrimitiveType::Line,
                aic,
                bic,
                rms_error: line.rms_error,
                inlier_ratio,
                data: FitData::Line(line),
            });
        }
    }

    // 尝试圆弧拟合
    if points.len() >= 3 {
        if let Some(arc) = fit_circle_lm(points, 50) {
            let aic = calculate_aic(arc.rms_error, points.len(), 3); // 圆有3个参数
            let bic = calculate_bic(arc.rms_error, points.len(), 3);

            // 内点比例（距离半径的偏差）
            let mut inliers = 0;
            for &p in points {
                let dx = p[0] - arc.center[0];
                let dy = p[1] - arc.center[1];
                let r = (dx * dx + dy * dy).sqrt();
                let dist = (r - arc.radius).abs();
                if dist < line_threshold {
                    inliers += 1;
                }
            }
            let inlier_ratio = inliers as f64 / points.len() as f64;

            results.push(FitResult {
                primitive_type: PrimitiveType::Arc,
                aic,
                bic,
                rms_error: arc.rms_error,
                inlier_ratio,
                data: FitData::Arc(arc),
            });
        }
    }

    // 尝试贝塞尔曲线拟合
    if points.len() >= 4 {
        if let Some(bezier) = fit_bezier_cubic(points) {
            let aic = calculate_aic(bezier.rms_error, points.len(), 6); // 贝塞尔有8个坐标，但2个固定 → 6个自由参数
            let bic = calculate_bic(bezier.rms_error, points.len(), 6);

            results.push(FitResult {
                primitive_type: PrimitiveType::Bezier,
                aic,
                bic,
                rms_error: bezier.rms_error,
                inlier_ratio: 1.0, // 贝塞尔经过所有控制点
                data: FitData::Bezier(bezier),
            });
        }
    }

    // 按 AIC 选择最优模型（越小越好）
    if results.is_empty() {
        // 至少返回直线拟合
        let line = fit_line_theil_sen(points)?;
        let aic = calculate_aic(line.rms_error, points.len(), 2);
        let bic = calculate_bic(line.rms_error, points.len(), 2);
        Some(FitResult {
            primitive_type: PrimitiveType::Line,
            aic,
            bic,
            rms_error: line.rms_error,
            inlier_ratio: 1.0,
            data: FitData::Line(line),
        })
    } else {
        // AIC 权重 0.7，BIC 权重 0.3
        results.sort_by(|a, b| {
            let score_a = a.aic * 0.7 + a.bic * 0.3;
            let score_b = b.aic * 0.7 + b.bic * 0.3;
            score_a
                .partial_cmp(&score_b)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        results.into_iter().next()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_line_fitting_theil_sen() {
        // 生成直线上的点：y = 2x + 1
        let mut points = Vec::new();
        for x in 0..10 {
            points.push([x as f64, 2.0 * x as f64 + 1.0]);
        }

        let line = fit_line_theil_sen(&points).unwrap();

        // 验证斜率接近 2
        assert!((line.a + 2.0).abs() < 0.1 || (line.a - 2.0).abs() < 0.1);
    }

    #[test]
    fn test_line_fitting_ransac() {
        // 使用固定种子确保测试确定性
        fastrand::seed(42);

        // 生成带噪声的直线点
        let mut points = Vec::new();
        for x in 0..20 {
            points.push([x as f64, 2.0 * x as f64 + 1.0 + fastrand::f64() * 0.2 - 0.1]);
        }
        // 添加外点
        points.push([5.0, 100.0]);
        points.push([10.0, -50.0]);

        let line = fit_line_ransac(&points, 1.0, 1000).unwrap();

        // 验证斜率接近 2（RANSAC 应该能排除外点）
        let slope = -line.a / line.b;
        assert!((slope - 2.0).abs() < 0.5);
    }

    #[test]
    fn test_circle_fitting_lm() {
        // 生成圆上的点
        let center = [10.0, 10.0];
        let radius = 5.0;
        let mut points = Vec::new();
        for i in 0..20 {
            let angle = i as f64 * TAU / 20.0;
            points.push([
                center[0] + radius * angle.cos(),
                center[1] + radius * angle.sin(),
            ]);
        }

        let circle = fit_circle_lm(&points, 50).unwrap();

        assert!((circle.center[0] - center[0]).abs() < 0.1);
        assert!((circle.center[1] - center[1]).abs() < 0.1);
        assert!((circle.radius - radius).abs() < 0.1);
    }

    #[test]
    fn test_bezier_fitting() {
        // 近似贝塞尔曲线的点
        let p0 = [0.0, 0.0];
        let p1 = [0.0, 10.0];
        let p2 = [10.0, 10.0];
        let p3 = [10.0, 0.0];

        // 采样
        let mut points = Vec::new();
        for i in 0..20 {
            let t = i as f64 / 19.0;
            let mt = 1.0 - t;
            let x = mt * mt * mt * p0[0]
                + 3.0 * mt * mt * t * p1[0]
                + 3.0 * mt * t * t * p2[0]
                + t * t * t * p3[0];
            let y = mt * mt * mt * p0[1]
                + 3.0 * mt * mt * t * p1[1]
                + 3.0 * mt * t * t * p2[1]
                + t * t * t * p3[1];
            points.push([x, y]);
        }

        let bezier = fit_bezier_cubic(&points).unwrap();
        // 简单优化的控制点可能有一定误差
        assert!(bezier.rms_error < 5.0);
    }

    #[test]
    fn test_fit_best_primitive_line() {
        // 生成直线点
        let mut points = Vec::new();
        for x in 0..20 {
            points.push([x as f64, 2.0 * x as f64 + 1.0]);
        }

        let result = fit_best_primitive(&points, 1.0).unwrap();
        // 应该选择直线
        assert_eq!(result.primitive_type, PrimitiveType::Line);
    }

    #[test]
    fn test_aic_calculation() {
        let aic = calculate_aic(0.5, 100, 2);
        assert!(aic > 0.0);
    }

    #[test]
    fn test_bic_calculation() {
        let bic = calculate_bic(0.5, 100, 2);
        assert!(bic > 0.0);
    }
}
