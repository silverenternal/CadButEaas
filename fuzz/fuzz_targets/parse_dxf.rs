#![no_main]

//! DXF 解析模糊测试目标
//!
//! ## 用途
//! 随机生成 DXF 文件内容，测试解析器的健壮性
//!
//! ## 运行方式
//! ```bash
//! # 安装 cargo-fuzz
//! cargo install cargo-fuzz
//!
//! # 运行模糊测试
//! cargo fuzz run parse_dxf
//!
//! # 使用多个作业并行运行
//! cargo fuzz run parse_dxf -j 8
//!
//! # 运行直到找到崩溃
//! cargo fuzz run parse_dxf -- -max_total_time=3600
//! ```

use libfuzzer_sys::fuzz_target;
use parser::dxf_parser::DxfParser;
use common_types::adaptive_tolerance::{AdaptiveTolerance, PrecisionLevel};
use common_types::scene::LengthUnit;

fuzz_target!(|data: &[u8]| {
    // 将随机字节转换为字符串（可能包含无效 UTF-8）
    let content = match std::str::from_utf8(data) {
        Ok(s) => s,
        Err(_) => return,  // 跳过无效 UTF-8
    };

    // 跳过空内容
    if content.is_empty() {
        return;
    }

    // 跳过太小的内容
    if content.len() < 10 {
        return;
    }

    // 创建解析器
    let mut parser = DxfParser::new();

    // 创建自适应容差
    let tolerance = AdaptiveTolerance::new(
        LengthUnit::Mm,
        1000.0,  // 默认场景尺度
        PrecisionLevel::Normal,
    );

    // 尝试解析（不应该 panic）
    let _ = parser.parse_string(content, &tolerance);
});
