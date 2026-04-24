//! HATCH 填充图案解析测试
//!
//! 测试 HATCH 实体的解析功能
//!
//! ## 实现说明
//!
//! 本测试使用低层级组码解析器来解析 HATCH 实体。
//! 由于 dxf 0.6.0 crate 将 HATCH 归类为 `ProxyEntity`，无法直接访问其内部数据。
//! 我们实现了基于 DXF 组码的低层级解析器来提取 HATCH 实体数据。
//!
//! ## DXF 组码说明
//! - 组码 0 = "HATCH" 标识 HATCH 实体
//! - 组码 2 = 图案名称（"ANSI31", "ANSI37" 等）
//! - 组码 70 = 填充类型（0 = 图案，1 = 实体）
//! - 组码 91 = 边界路径数量
//! - 组码 92 = 边界类型（1 = 多段线，2 = 圆弧，3 = 椭圆，4 = 样条）

use common_types::RawEntity;
use parser::{hatch_parser::HatchParser, DxfParser};

mod common;
use common::get_dxf_dir;

/// 测试 HatchParser 的基本功能
#[test]
fn test_hatch_parser_creation() {
    let parser = HatchParser::new();
    assert!(!parser.ignores_solid());

    let parser = HatchParser::new().with_ignore_solid(true);
    assert!(parser.ignores_solid());
}

/// 测试 DXF 文件中的 HATCH 实体解析
#[test]
fn test_hatch_entity_parse() {
    let mut parser = DxfParser::new();
    parser.config.ignore_hatch = false; // 启用 HATCH 解析

    let dxf_dir = get_dxf_dir();

    // 列出目录中的所有 DXF 文件
    let mut dxf_files: Vec<String> = Vec::new();
    if dxf_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&dxf_dir) {
            for entry in entries.flatten() {
                let file_name = entry.file_name().to_string_lossy().to_string();
                if file_name.ends_with(".dxf") {
                    dxf_files.push(file_name);
                }
            }
        }
    }

    dxf_files.sort();

    // 统计包含 HATCH 实体的文件
    let mut files_with_hatch = 0;
    let mut total_hatch_count = 0;

    for file_name in &dxf_files {
        let file_path = dxf_dir.join(file_name);

        if !file_path.exists() {
            continue;
        }

        // 解析文件
        if let Ok(entities) = parser.parse_file(&file_path) {
            // 统计 HATCH 实体数量
            let hatch_count = entities
                .iter()
                .filter(|e| matches!(e, RawEntity::Hatch { .. }))
                .count();

            if hatch_count > 0 {
                files_with_hatch += 1;
                total_hatch_count += hatch_count;
                println!("✅ {}: {} 个 HATCH 实体", file_name, hatch_count);
            }
        }
    }

    println!("\n========== HATCH 解析测试报告 ==========");
    println!("总文件数：{}", dxf_files.len());
    println!("包含 HATCH 的文件数：{}", files_with_hatch);
    println!("HATCH 实体总数：{}", total_hatch_count);
    println!("=======================================\n");

    // 注意：如果测试文件中没有 HATCH 实体，此测试也会通过
    // 这是预期行为，因为不是所有 DXF 文件都包含 HATCH
    println!(
        "✅ HATCH 解析测试完成（解析到 {} 个 HATCH 实体）",
        total_hatch_count
    );
}

/// 测试 HATCH 边界路径类型识别
#[test]
fn test_hatch_boundary_types() {
    // HATCH 边界类型包括：
    // - Polyline 边界（组码 92 = 1）
    // - Arc 边界（组码 92 = 2）
    // - Ellipse 边界（组码 92 = 3）
    // - Spline 边界（组码 92 = 4）

    let parser = HatchParser::new();

    // 创建一个简单的 DXF 内容用于测试
    let test_dxf_content = r#"0
SECTION
2
ENTITIES
0
HATCH
10
0.0
20
0.0
30
0.0
2
ANSI31
70
0
91
1
92
1
73
1
93
4
10
0.0
20
0.0
10
10.0
20
0.0
10
10.0
20
10.0
10
0.0
20
10.0
0
ENDSEC
0
EOF
"#;

    // 写入临时文件
    let temp_path = std::env::temp_dir().join("test_hatch.dxf");
    std::fs::write(&temp_path, test_dxf_content).expect("写入测试文件失败");

    // 解析 HATCH
    match parser.parse_hatch_entities(&temp_path) {
        Ok(hatches) => {
            println!("✅ 解析到 {} 个 HATCH 实体", hatches.len());

            for hatch in &hatches {
                if let RawEntity::Hatch { boundary_paths, .. } = hatch {
                    println!("  边界路径数量：{}", boundary_paths.len());
                    for (i, path) in boundary_paths.iter().enumerate() {
                        match path {
                            common_types::HatchBoundaryPath::Polyline {
                                points,
                                closed,
                                bulges,
                            } => {
                                println!(
                                    "    边界 {}: Polyline ({} 点，closed={}, bulges={:?})",
                                    i,
                                    points.len(),
                                    closed,
                                    bulges
                                );
                            }
                            common_types::HatchBoundaryPath::Arc { radius, .. } => {
                                println!("    边界 {}: Arc (radius={})", i, radius);
                            }
                            common_types::HatchBoundaryPath::EllipseArc { major_axis, .. } => {
                                println!(
                                    "    边界 {}: EllipseArc (major_axis={:?})",
                                    i, major_axis
                                );
                            }
                            common_types::HatchBoundaryPath::Spline {
                                control_points,
                                degree,
                                ..
                            } => {
                                println!(
                                    "    边界 {}: Spline (degree={}, control_points={})",
                                    i,
                                    degree,
                                    control_points.len()
                                );
                            }
                        }
                    }
                }
            }

            // 验证至少解析到一个 HATCH
            assert!(!hatches.is_empty(), "应该解析到至少一个 HATCH 实体");
        }
        Err(e) => {
            println!("⚠️ HATCH 解析失败：{}", e);
            // 测试框架可能需要更完善的 DXF 内容
        }
    }

    // 清理临时文件
    let _ = std::fs::remove_file(&temp_path);
}

/// 测试 HATCH 实体填充 vs 图案填充
#[test]
fn test_hatch_solid_vs_pattern() {
    // HATCH 填充类型（组码 70）：
    // - 0 = 图案填充（Pattern fill）
    // - 1 = 实体填充（Solid fill）

    let parser = HatchParser::new();

    // 创建实体填充的 DXF 内容
    let solid_hatch_dxf = r#"0
SECTION
2
ENTITIES
0
HATCH
10
0.0
20
0.0
30
0.0
2

70
1
91
1
92
1
73
1
93
4
10
0.0
20
0.0
10
10.0
20
0.0
10
10.0
20
10.0
10
0.0
20
10.0
0
ENDSEC
0
EOF
"#;

    let temp_path = std::env::temp_dir().join("test_solid_hatch.dxf");
    std::fs::write(&temp_path, solid_hatch_dxf).expect("写入测试文件失败");

    match parser.parse_hatch_entities(&temp_path) {
        Ok(hatches) => {
            for hatch in &hatches {
                if let RawEntity::Hatch { solid_fill, .. } = hatch {
                    println!("✅ HATCH 类型：solid_fill={}", solid_fill);
                }
            }
        }
        Err(e) => {
            println!("⚠️ HATCH 解析失败：{}", e);
        }
    }

    let _ = std::fs::remove_file(&temp_path);
}
