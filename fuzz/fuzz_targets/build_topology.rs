#![no_main]

//! 拓扑构建模糊测试目标
//!
//! ## 用途
//! 随机生成几何数据，测试拓扑构建的健壮性
//!
//! ## 运行方式
//! ```bash
//! cargo fuzz run build_topology
//! ```

use libfuzzer_sys::fuzz_target;
use common_types::geometry::{Polyline, Point2};
use common_types::adaptive_tolerance::{AdaptiveTolerance, PrecisionLevel};
use common_types::scene::LengthUnit;
use topo::service::TopoService;
use topo::graph_builder::GraphBuilder;

fuzz_target!(|data: &[u8]| {
    // 将随机字节解释为多段线数据
    let polylines = decode_polylines(data);
    
    // 跳过空数据
    if polylines.is_empty() {
        return;
    }

    // 创建自适应容差
    let tolerance = AdaptiveTolerance::new(
        LengthUnit::Mm,
        1000.0,
        PrecisionLevel::Normal,
    );

    // 创建拓扑服务
    let service = TopoService::new(Default::default());

    // 尝试构建拓扑（不应该 panic）
    let _ = service.build_topology(&polylines, &tolerance);
});

/// 将随机字节解码为多段线列表
fn decode_polylines(data: &[u8]) -> Vec<Polyline> {
    if data.len() < 16 {
        return Vec::new();
    }

    let mut polylines = Vec::new();
    let mut i = 0;

    // 第一个字节决定多段线数量（1-10）
    let num_polylines = (data[0] % 10) as usize + 1;

    for _ in 0..num_polylines {
        if i + 4 >= data.len() {
            break;
        }

        // 每个多段线有 2-10 个点
        let num_points = ((data[i] % 9) as usize) + 2;
        i += 1;

        let mut points = Vec::with_capacity(num_points);
        for _ in 0..num_points {
            if i + 16 >= data.len() {
                break;
            }

            // 将 8 个字节解释为两个 f64 坐标
            let x_bytes: [u8; 8] = data[i..i+8].try_into().unwrap_or([0; 8]);
            let y_bytes: [u8; 8] = data[i+8..i+16].try_into().unwrap_or([0; 8]);
            
            let x = f64::from_le_bytes(x_bytes);
            let y = f64::from_le_bytes(y_bytes);

            // 过滤掉 NaN 和无穷大
            if x.is_finite() && y.is_finite() {
                points.push(Point2::from([x, y]));
            }

            i += 16;
        }

        if points.len() >= 2 {
            polylines.push(points);
        }
    }

    polylines
}
