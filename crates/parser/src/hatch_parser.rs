//! HATCH 填充图案解析器
//!
//! 使用 dxf crate 的低层级组码迭代器解析 DXF 文件中的 HATCH 实体
//!
//! ## 背景
//!
//! dxf 0.6.0 crate 将 HATCH 实体归类为 `ProxyEntity`，无法直接访问其内部数据。
//! 本模块使用 dxf 的低层级迭代器 API，直接读取 DXF 组码来解析 HATCH 实体。
//!
//! ## DXF 组码说明
//!
//! ### HATCH 实体主要组码
//! - 10/20/30: 插入点（标高）
//! - 210/220/230: 法向量
//! - 2: 图案名称（"ANSI31", "ANSI37", "AR-BRSTD" 等）
//! - 70: 填充类型（0 = 图案填充，1 = 实体填充）
//! - 71: 关联标志
//! - 75: 图案类型
//! - 76: 图案样式
//! - 41: 图案比例
//! - 52: 图案角度
//! - 78: 图案线数量
//! - 91: 边界路径数量
//! - 92: 边界路径类型（1 = 多段线，2 = 圆弧，3 = 椭圆，4 = 样条）
//! - 93: 边数量
//! - 97: 源对象数量
//!
//! ### 边界路径组码
//! - 多段线边界：
//!   - 72: 是否有宽度
//!   - 73: 是否闭合
//!   - 93: 顶点数量
//!   - 10/20: 顶点坐标
//!   - 42: Bulge（凸度）
//!
//! - 圆弧边界：
//!   - 10/20: 中心点
//!   - 40: 半径
//!   - 50: 起始角度
//!   - 51: 结束角度
//!   - 73: 逆时针标志
//!
//! - 椭圆弧边界：
//!   - 10/20: 中心点
//!   - 11/21: 长轴端点
//!   - 40: 短轴比率
//!   - 50: 起始角度
//!   - 51: 结束角度
//!   - 73: 逆时针标志
//!
//! - 样条曲线边界：
//!   - 94: 控制点数量
//!   - 95: 拟合点数量
//!   - 96: 节点数量
//!   - 97: 阶数
//!   - 10/20: 控制点坐标
//!   - 40: 节点值
//!
//! ## 使用示例
//!
//! ```no_run
//! use parser::hatch_parser::HatchParser;
//! use std::path::Path;
//!
//! # fn example() -> Result<(), Box<dyn std::error::Error>> {
//! let parser = HatchParser::new();
//! let hatches = parser.parse_hatch_entities(Path::new("floor_plan.dxf"))?;
//! println!("解析到 {} 个 HATCH 实体", hatches.len());
//! # Ok(())
//! # }
//! ```

use common_types::{RawEntity, EntityMetadata};
use common_types::{HatchBoundaryPath, HatchPattern};
use std::path::Path;
use std::fs::File;
use std::io::{BufReader, Read};

/// HATCH 填充图案解析器
#[derive(Clone)]
pub struct HatchParser {
    /// 是否忽略实体填充（Solid Fill）
    ignore_solid: bool,
}

impl HatchParser {
    /// 创建新的 HatchParser
    pub fn new() -> Self {
        Self {
            ignore_solid: false,
        }
    }

    /// 设置是否忽略实体填充
    pub fn with_ignore_solid(mut self, ignore: bool) -> Self {
        self.ignore_solid = ignore;
        self
    }

    /// 检查是否忽略实体填充
    pub fn ignores_solid(&self) -> bool {
        self.ignore_solid
    }

    /// 解析 DXF 文件中的所有 HATCH 实体
    ///
    /// 使用低层级组码迭代器解析 HATCH 实体
    pub fn parse_hatch_entities(&self, file_path: &Path) -> Result<Vec<RawEntity>, String> {
        // 读取 DXF 文件为字节
        let file = File::open(file_path)
            .map_err(|e| format!("打开文件失败：{}", e))?;
        let mut reader = BufReader::new(file);
        let mut contents = String::new();
        reader.read_to_string(&mut contents)
            .map_err(|e| format!("读取文件失败：{}", e))?;

        // 解析组码
        let groups = self.parse_group_codes(&contents)?;
        
        // 提取 HATCH 实体
        let hatches = self.extract_hatches(&groups)?;
        
        Ok(hatches)
    }

    /// 解析 DXF 组码
    fn parse_group_codes(&self, contents: &str) -> Result<Vec<(u16, String)>, String> {
        let mut groups = Vec::new();
        let mut lines = contents.lines();

        while let Some(code_line) = lines.next() {
            if let Ok(code) = code_line.trim().parse::<u16>() {
                if let Some(value_line) = lines.next() {
                    groups.push((code, value_line.trim().to_string()));
                }
            }
        }

        Ok(groups)
    }

    /// 从组码中提取 HATCH 实体
    fn extract_hatches(&self, groups: &[(u16, String)]) -> Result<Vec<RawEntity>, String> {
        let mut hatches = Vec::new();
        let mut i = 0;

        while i < groups.len() {
            // 查找 HATCH 实体起始（组码 0 = "HATCH"）
            if groups[i].0 == 0 && groups[i].1.to_uppercase() == "HATCH" {
                // 解析 HATCH 实体
                if let Some(hatch) = self.parse_single_hatch(groups, &mut i)? {
                    hatches.push(hatch);
                }
            } else {
                i += 1;
            }
        }

        Ok(hatches)
    }

    /// 解析单个 HATCH 实体
    fn parse_single_hatch(
        &self,
        groups: &[(u16, String)],
        i: &mut usize,
    ) -> Result<Option<RawEntity>, String> {
        *i += 1; // 跳过 "HATCH"

        let mut handle: Option<String> = None;
        let mut layer: Option<String> = None;
        let mut color: Option<String> = None;
        let mut _elevation: f64 = 0.0;
        let mut pattern_name = "ANSI31".to_string();
        let mut is_solid = false;
        let mut _pattern_scale = 1.0;
        let mut _pattern_angle = 0.0;
        let mut boundary_paths: Vec<HatchBoundaryPath> = Vec::new();

        // 解析 HATCH 实体数据
        while *i < groups.len() {
            let (code, value) = &groups[*i];

            match code {
                // 组码 0 = 新实体开始，返回
                0 => break,

                // 5 = Handle
                5 => handle = Some(value.clone()),

                // 8 = 图层名
                8 => layer = Some(value.clone()),

                // 62 = 颜色
                62 => color = Some(format!("ACI_{}", value)),

                // 10/20/30 = 插入点（这里只用于获取标高）
                10 => _elevation = value.parse().unwrap_or(0.0),

                // 2 = 图案名称
                2 => pattern_name = value.clone(),

                // 70 = 填充类型（0 = 图案，1 = 实体）
                70 => is_solid = value.parse::<i16>().unwrap_or(0) == 1,

                // 41 = 图案比例
                41 => _pattern_scale = value.parse().unwrap_or(1.0),

                // 52 = 图案角度
                52 => _pattern_angle = value.parse().unwrap_or(0.0),

                // 91 = 边界路径数量
                91 => {
                    let count: i32 = value.parse().unwrap_or(0);
                    for _ in 0..count {
                        if let Some(boundary) = self.parse_boundary_path(groups, i)? {
                            boundary_paths.push(boundary);
                        }
                    }
                }

                _ => {}
            }

            *i += 1;
        }

        // 如果是实体填充且被忽略，返回 None
        if is_solid && self.ignore_solid {
            return Ok(None);
        }

        // 如果没有有效边界，返回 None
        if boundary_paths.is_empty() {
            tracing::warn!("HATCH 实体 handle={:?} 没有有效的边界", handle);
            return Ok(None);
        }

        // 创建元数据
        let metadata = EntityMetadata {
            handle,
            layer,
            color,
            lineweight: None,
            line_type: None,
            material: None,
            width: None,
        };

        // 创建填充图案
        let pattern = if is_solid {
            HatchPattern::Solid {
                color: common_types::Color32::WHITE,
            }
        } else {
            HatchPattern::Predefined {
                name: pattern_name,
            }
        };

        Ok(Some(RawEntity::Hatch {
            boundary_paths,
            pattern,
            solid_fill: is_solid,
            metadata,
            semantic: None,
            scale: _pattern_scale,    // P0-NEW-14 修复：存储 scale
            angle: _pattern_angle,    // P0-NEW-14 修复：存储 angle
        }))
    }

    /// 解析边界路径
    fn parse_boundary_path(
        &self,
        groups: &[(u16, String)],
        i: &mut usize,
    ) -> Result<Option<HatchBoundaryPath>, String> {
        // ✅ P1-NEW-15 修复：验证当前索引有效
        if *i >= groups.len() {
            return Ok(None);
        }

        // ✅ P1-NEW-15 修复：查找 92 组码（边界类型），但限制搜索范围
        let mut found = false;
        while *i < groups.len() {
            let (code, _) = &groups[*i];

            // ✅ 遇到新实体开始，返回
            if *code == 0 {
                return Ok(None);
            }

            // ✅ 找到 92 组码
            if *code == 92 {
                found = true;
                break;
            }

            *i += 1;
        }

        if !found || *i >= groups.len() {
            return Ok(None);
        }

        let boundary_type: i32 = groups[*i].1.parse().unwrap_or(0);
        *i += 1;  // ✅ 跳过 92 组码的值

        match boundary_type {
            1 => self.parse_polyline_boundary(groups, i),
            2 => self.parse_arc_boundary(groups, i),
            3 => self.parse_ellipse_boundary(groups, i),
            4 => self.parse_spline_boundary(groups, i),
            _ => {
                tracing::warn!("未知的边界类型：{}", boundary_type);
                Ok(None)
            }
        }
    }

    /// 解析多段线边界
    fn parse_polyline_boundary(
        &self,
        groups: &[(u16, String)],
        i: &mut usize,
    ) -> Result<Option<HatchBoundaryPath>, String> {
        let mut points = Vec::new();
        let mut is_closed = false;
        let mut bulges: Vec<f64> = Vec::new();  // ✅ P0-4 新增：存储 bulge 信息

        // 72 = 是否有宽度（跳过）
        // 73 = 是否闭合
        while *i < groups.len() {
            let (code, value) = &groups[*i];

            match code {
                0 => break, // 新实体
                73 => is_closed = value.parse::<i16>().unwrap_or(0) == 1,
                10 => {
                    // X 坐标
                    let x: f64 = value.parse().unwrap_or(0.0);
                    *i += 1;
                    // Y 坐标
                    let y: f64 = if *i < groups.len() && groups[*i].0 == 20 {
                        groups[*i].1.parse().unwrap_or(0.0)
                    } else {
                        0.0
                    };
                    points.push([x, y]);

                    // ✅ P0-4 提取 bulge（组码 42）
                    // bulge = tan(θ/4)，用于表示圆弧段
                    if *i + 1 < groups.len() && groups[*i + 1].0 == 42 {
                        let bulge: f64 = groups[*i + 1].1.parse().unwrap_or(0.0);
                        bulges.push(bulge);
                        *i += 1;
                    } else {
                        bulges.push(0.0);
                    }
                }
                _ => {}
            }
            *i += 1;
        }

        if points.is_empty() {
            Ok(None)
        } else {
            // ✅ P0-4 如果有 bulge 信息，存储到边界路径中
            let has_bulge = bulges.iter().any(|&b| b.abs() > 1e-10);
            Ok(Some(HatchBoundaryPath::Polyline {
                points,
                closed: is_closed,
                bulges: if has_bulge { Some(bulges) } else { None },
            }))
        }
    }

    /// 解析圆弧边界
    fn parse_arc_boundary(
        &self,
        groups: &[(u16, String)],
        i: &mut usize,
    ) -> Result<Option<HatchBoundaryPath>, String> {
        let mut center = [0.0, 0.0];
        let mut radius = 1.0;
        let mut start_angle = 0.0;
        let mut end_angle = 0.0;
        let mut ccw = false;

        while *i < groups.len() {
            let (code, value) = &groups[*i];

            match code {
                0 => break,
                10 => center[0] = value.parse().unwrap_or(0.0),
                20 => center[1] = value.parse().unwrap_or(0.0),
                40 => radius = value.parse().unwrap_or(1.0),
                50 => start_angle = value.parse::<f64>().unwrap_or(0.0).to_radians(),
                51 => end_angle = value.parse::<f64>().unwrap_or(0.0).to_radians(),
                73 => ccw = value.parse::<i16>().unwrap_or(0) == 1,
                _ => {}
            }
            *i += 1;
        }

        // ✅ P0-3 修复：保持弧度单位，前端统一使用弧度
        Ok(Some(HatchBoundaryPath::Arc {
            center,
            radius,
            start_angle,  // 弧度
            end_angle,    // 弧度
            ccw,
        }))
    }

    /// 解析椭圆弧边界
    fn parse_ellipse_boundary(
        &self,
        groups: &[(u16, String)],
        i: &mut usize,
    ) -> Result<Option<HatchBoundaryPath>, String> {
        let mut center = [0.0, 0.0];
        let mut major_axis = [1.0, 0.0];
        let mut minor_axis_ratio = 1.0;
        let mut start_angle = 0.0;
        let mut end_angle = 0.0;
        let mut ccw = false;

        while *i < groups.len() {
            let (code, value) = &groups[*i];

            match code {
                0 => break,
                10 => center[0] = value.parse().unwrap_or(0.0),
                20 => center[1] = value.parse().unwrap_or(0.0),
                11 => major_axis[0] = value.parse().unwrap_or(1.0),
                21 => major_axis[1] = value.parse().unwrap_or(0.0),
                40 => minor_axis_ratio = value.parse().unwrap_or(1.0),
                50 => start_angle = value.parse::<f64>().unwrap_or(0.0).to_radians(),
                51 => end_angle = value.parse::<f64>().unwrap_or(0.0).to_radians(),
                73 => ccw = value.parse::<i16>().unwrap_or(0) == 1,
                _ => {}
            }
            *i += 1;
        }

        // ✅ P0-3 修复：保持弧度单位，前端统一使用弧度
        Ok(Some(HatchBoundaryPath::EllipseArc {
            center,
            major_axis,
            minor_axis_ratio,
            start_angle,  // 弧度
            end_angle,    // 弧度
            ccw,
            extrusion_direction: None,  // P2-NEW-29: 默认无法向量
        }))
    }

    /// 解析样条曲线边界
    fn parse_spline_boundary(
        &self,
        groups: &[(u16, String)],
        i: &mut usize,
    ) -> Result<Option<HatchBoundaryPath>, String> {
        let mut control_points = Vec::new();
        let mut knots = Vec::new();
        let mut weights = Vec::new();      // P1-NEW-25 新增：权重
        let mut fit_points = Vec::new();   // P1-NEW-25 新增：拟合点
        let mut degree: i32 = 3;
        let mut num_control_points: i32 = 0;
        let mut num_fit_points: i32 = 0;
        let mut spline_flags: i32 = 0;     // P1-NEW-25 新增：样条标志

        // 先读取数量信息
        while *i < groups.len() {
            let (code, value) = &groups[*i];

            match code {
                0 => break,
                94 => num_control_points = value.parse().unwrap_or(0),
                95 => num_fit_points = value.parse().unwrap_or(0),
                96 => { /* knots count, 不需要显式使用 */ }
                97 => degree = value.parse().unwrap_or(3),
                74 => spline_flags = value.parse().unwrap_or(0),
                _ => {}
            }
            *i += 1;
        }

        // 预分配容量
        control_points.reserve(num_control_points as usize);
        weights.reserve(num_control_points as usize);
        fit_points.reserve(num_fit_points as usize);

        // 解析具体数据
        while *i < groups.len() {
            let (code, value) = &groups[*i];

            match code {
                0 => break,

                // P1-NEW-25: 控制点 (10, 20)
                10 => {
                    let x: f64 = value.parse().unwrap_or(0.0);
                    *i += 1;
                    let y: f64 = if *i < groups.len() && groups[*i].0 == 20 {
                        groups[*i].1.parse().unwrap_or(0.0)
                    } else {
                        0.0
                    };
                    control_points.push([x, y]);
                }

                // P1-NEW-25: 权重 (41) - 紧跟控制点之后
                41 => {
                    if let Ok(weight) = value.parse::<f64>() {
                        weights.push(weight);
                    }
                }

                // P1-NEW-25: 拟合点 (11, 21)
                11 => {
                    let x: f64 = value.parse().unwrap_or(0.0);
                    *i += 1;
                    let y: f64 = if *i < groups.len() && groups[*i].0 == 21 {
                        groups[*i].1.parse().unwrap_or(0.0)
                    } else {
                        0.0
                    };
                    fit_points.push([x, y]);
                }

                // 节点值 (40)
                40 => {
                    if let Ok(knot) = value.parse::<f64>() {
                        knots.push(knot);
                    }
                }

                _ => {}
            }
            *i += 1;
        }

        if control_points.is_empty() {
            Ok(None)
        } else {
            // P1-NEW-25: 如果没有显式权重，默认为 1.0
            if weights.is_empty() {
                weights.resize(control_points.len(), 1.0);
            }

            Ok(Some(HatchBoundaryPath::Spline {
                control_points,
                knots,
                degree: degree as u32,
                weights: Some(weights),    // P1-NEW-25: 存储权重
                fit_points: if fit_points.is_empty() { None } else { Some(fit_points) }, // P1-NEW-25: 存储拟合点
                flags: Some(spline_flags as u32), // P1-NEW-25: 存储标志
            }))
        }
    }
}

impl Default for HatchParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_hatch_parser_creation() {
        let parser = HatchParser::new();
        assert!(!parser.ignore_solid);

        let parser = HatchParser::new().with_ignore_solid(true);
        assert!(parser.ignore_solid);
    }
}
