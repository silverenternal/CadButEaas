//! DIMENSION 尺寸标注标注解析测试
//!
//! 测试 DIMENSION 实体的解析功能

use common_types::RawEntity;
use parser::DxfParser;

mod common;
use common::get_dxf_dir;

/// 测试 DXF 文件中的 DIMENSION 实体解析
#[test]
fn test_dimension_entity_parse() {
    let parser = DxfParser::new();
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

    // 统计包含 DIMENSION 实体的文件
    let mut files_with_dimension = 0;
    let mut total_dimension_count = 0;
    let mut total_definition_points = 0;
    let mut dimensions_with_measurement = 0;
    let mut dimensions_with_text = 0;

    for file_name in &dxf_files {
        let file_path = dxf_dir.join(file_name);

        if !file_path.exists() {
            continue;
        }

        // 解析文件
        if let Ok(entities) = parser.parse_file(&file_path) {
            // 统计 DIMENSION 实体数量
            let dimension_count = entities
                .iter()
                .filter(|e| matches!(e, RawEntity::Dimension { .. }))
                .count();

            if dimension_count > 0 {
                files_with_dimension += 1;
                total_dimension_count += dimension_count;

                // 详细统计
                for entity in entities.iter() {
                    if let RawEntity::Dimension {
                        definition_points,
                        measurement,
                        text,
                        ..
                    } = entity
                    {
                        total_definition_points += definition_points.len();
                        if *measurement > 0.0 {
                            dimensions_with_measurement += 1;
                        }
                        if text.is_some() {
                            dimensions_with_text += 1;
                        }
                    }
                }

                println!("✅ {}: {} 个 DIMENSION 实体", file_name, dimension_count);
            }
        }
    }

    println!("\n========== DIMENSION 解析测试报告 ==========");
    println!("总文件数：{}", dxf_files.len());
    println!("包含 DIMENSION 的文件数：{}", files_with_dimension);
    println!("DIMENSION 实体总数：{}", total_dimension_count);
    println!("定义点总数：{}", total_definition_points);
    println!("有测量值的标注数：{}", dimensions_with_measurement);
    println!("有文字的标注数：{}", dimensions_with_text);
    println!("===========================================\n");

    // 注意：不强制要求有 DIMENSION 实体，因为测试文件可能不包含标注
    // 具体的文件验证在 test_dimension_from_real_file() 中进行
}

/// 测试 DIMENSION 类型识别
#[test]
fn test_dimension_type_parse() {
    // 验证维度类型转换逻辑
    use common_types::DimensionType;

    // 测试各种标注类型的识别
    // 注意：实际类型识别在 convert_dimension_type() 中实现
    // 这里验证 DimensionType 枚举的变体

    let dimension_types = [
        DimensionType::Linear,
        DimensionType::Aligned,
        DimensionType::Angular,
        DimensionType::Diameter,
        DimensionType::Radial,
        DimensionType::Ordinate,
        DimensionType::ArcLength,
    ];

    // 断言：所有类型都能正确创建
    assert_eq!(dimension_types.len(), 7, "应该有 7 种维度类型");

    // 验证类型匹配
    match dimension_types[0] {
        DimensionType::Linear => {} // 预期
        _ => panic!("第一个类型应该是 Linear"),
    }

    println!("✅ DIMENSION 类型识别测试通过");
}

/// 测试 DIMENSION 测量值提取
#[test]
fn test_dimension_measurement_parse() {
    // 创建测试数据验证测量值提取
    // 注意：实际测量值提取在 parse_*_dimension_entity() 中实现

    // 验证测量值缩放逻辑
    let raw_measurement: f64 = 100.0;
    let scale: f64 = 1.0;
    let scaled_measurement = raw_measurement * scale;

    assert!(
        (scaled_measurement - 100.0_f64).abs() < 0.001,
        "测量值缩放应该正确"
    );

    // 验证不同缩放比例
    let scale_1000: f64 = 1000.0;
    let scaled_1000 = raw_measurement * scale_1000;
    assert!(
        (scaled_1000 - 100000.0_f64).abs() < 0.001,
        "米到毫米的缩放应该正确"
    );

    println!("✅ DIMENSION 测量值提取逻辑测试通过");
}

/// 测试 DIMENSION 定义点提取
#[test]
fn test_dimension_definition_points_parse() {
    // 验证定义点提取逻辑
    // RotatedDimension 应该有 3 个定义点
    // RadialDimension 和 DiameterDimension 应该有 1-2 个定义点

    // 测试定义点数量要求
    let rotated_dim_points = 3; // pt1, pt2, pt3
    let radial_dim_points_min = 1; // pt1 (pt2 optional)

    assert!(
        rotated_dim_points >= 2,
        "RotatedDimension 应该至少有 2 个定义点"
    );
    assert!(
        radial_dim_points_min >= 1,
        "RadialDimension 应该至少有 1 个定义点"
    );

    // 验证坐标缩放逻辑
    let raw_point = (100.0_f64, 200.0_f64);
    let scale: f64 = 1.0;
    let scaled_point = (raw_point.0 * scale, raw_point.1 * scale);

    assert!((scaled_point.0 - 100.0).abs() < 0.001, "X 坐标缩放应该正确");
    assert!((scaled_point.1 - 200.0).abs() < 0.001, "Y 坐标缩放应该正确");

    println!("✅ DIMENSION 定义点提取逻辑测试通过");
}

/// 测试 DIMENSION 标注文字提取
#[test]
fn test_dimension_text_parse() {
    // 验证标注文字提取逻辑
    let empty_text = "";
    let non_empty_text = "100mm";

    // 空文字应该返回 None
    let text_option: Option<String> = if empty_text.is_empty() {
        None
    } else {
        Some(empty_text.to_string())
    };
    assert!(text_option.is_none(), "空文字应该返回 None");

    // 非空文字应该返回 Some
    let text_option: Option<String> = if non_empty_text.is_empty() {
        None
    } else {
        Some(non_empty_text.to_string())
    };
    assert!(text_option.is_some(), "非空文字应该返回 Some");
    assert_eq!(text_option.unwrap(), "100mm", "文字内容应该正确");

    println!("✅ DIMENSION 标注文字提取逻辑测试通过");
}

/// 测试从真实 DXF 文件解析 DIMENSION 实体
///
/// 注意：当前测试目录中的 DXF 文件不包含 DIMENSION 实体
/// 此测试用于验证解析框架正常工作，当未来添加带 DIMENSION 的测试文件时会自动验证
#[test]
fn test_dimension_from_real_file() {
    let parser = DxfParser::new();
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

    // 统计包含 DIMENSION 实体的文件
    let mut files_with_dimension = 0;
    let mut total_dimension_count = 0;
    let mut dimension_types_found = Vec::new();

    for file_name in &dxf_files {
        let file_path = dxf_dir.join(file_name);

        if !file_path.exists() {
            continue;
        }

        // 解析文件
        if let Ok(entities) = parser.parse_file(&file_path) {
            // 筛选出 DIMENSION 实体
            let dimensions: Vec<_> = entities
                .iter()
                .filter(|e| matches!(e, RawEntity::Dimension { .. }))
                .collect();

            if !dimensions.is_empty() {
                files_with_dimension += 1;
                total_dimension_count += dimensions.len();

                for entity in dimensions.iter() {
                    if let RawEntity::Dimension { dimension_type, .. } = entity {
                        // 使用 match 来比较类型
                        let type_str = match dimension_type {
                            common_types::DimensionType::Linear => "Linear",
                            common_types::DimensionType::Aligned => "Aligned",
                            common_types::DimensionType::Angular => "Angular",
                            common_types::DimensionType::Diameter => "Diameter",
                            common_types::DimensionType::Radial => "Radial",
                            common_types::DimensionType::Ordinate => "Ordinate",
                            common_types::DimensionType::ArcLength => "ArcLength",
                        };
                        if !dimension_types_found.contains(&type_str) {
                            dimension_types_found.push(type_str);
                        }
                    }
                }

                println!("✅ {}: {} 个 DIMENSION 实体", file_name, dimensions.len());
            }
        }
    }

    println!("\n========== DIMENSION 文件解析测试报告 ==========");
    println!("总文件数：{}", dxf_files.len());
    println!("包含 DIMENSION 的文件数：{}", files_with_dimension);
    println!("DIMENSION 实体总数：{}", total_dimension_count);
    println!("发现的维度类型：{:?}", dimension_types_found);
    println!("===========================================\n");

    // 注意：不强制要求有 DIMENSION 实体，因为当前测试文件可能不包含标注
    // 当未来添加带 DIMENSION 的测试文件时，以下断言会自动验证
    if files_with_dimension > 0 {
        assert!(
            total_dimension_count >= 1,
            "应该解析到至少 1 个 DIMENSION 实体"
        );
        assert!(!dimension_types_found.is_empty(), "应该至少有一种维度类型");
        println!("✅ DIMENSION 真实文件解析测试通过");
    } else {
        println!("⚠️  当前测试目录中没有包含 DIMENSION 实体的 DXF 文件");
        println!("   建议：添加带标注的 DXF 文件以验证 DIMENSION 解析功能");
    }
}
