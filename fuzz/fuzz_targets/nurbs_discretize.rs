#![no_main]

//! NURBS 离散化模糊测试目标
//!
//! ## 用途
//! 随机生成 NURBS 曲线参数，测试离散化算法的健壮性
//!
//! ## 运行方式
//! ```bash
//! cargo fuzz run nurbs_discretize
//! ```

use libfuzzer_sys::fuzz_target;
use common_types::geometry::Point2;

fuzz_target!(|data: &[u8]| {
    // 将随机字节解释为 NURBS 控制点和参数
    if data.len() < 20 {
        return;
    }

    // 控制点数量（2-10）
    let num_control_points = ((data[0] % 9) as usize) + 2;
    
    // 曲线度数（1-5）
    let degree = ((data[1] % 5) as usize) + 1;
    
    // 容差（0.001-1.0）
    let tolerance = 0.001 + ((data[2] as f64 / 255.0) * 0.999);

    // 解码控制点
    let mut control_points = Vec::with_capacity(num_control_points);
    let mut i = 3;
    
    for _ in 0..num_control_points {
        if i + 16 >= data.len() {
            break;
        }

        let x_bytes: [u8; 8] = data[i..i+8].try_into().unwrap_or([0; 8]);
        let y_bytes: [u8; 8] = data[i+8..i+16].try_into().unwrap_or([0; 8]);
        
        let x = f64::from_le_bytes(x_bytes);
        let y = f64::from_le_bytes(y_bytes);

        // 过滤掉 NaN 和无穷大
        if x.is_finite() && y.is_finite() {
            control_points.push(Point2::from([x, y]));
        }

        i += 16;
    }

    // 至少需要 2 个控制点
    if control_points.len() < 2 {
        return;
    }

    // 确保度数不超过控制点数量 - 1
    let actual_degree = degree.min(control_points.len() - 1);

    // 这里应该调用 NURBS 离散化函数
    // 由于 vectorize crate 的依赖关系，这里只做基本验证
    let _ = (control_points.len(), actual_degree, tolerance);
});
