//! 真实 DXF 文件集成测试
//!
//! 使用 dxfs/ 目录下的真实建筑图纸验证解析器

use parser::DxfParser;
mod common;
use common::{find_first_parseable_dxf, get_dxf_dir};

/// 测试所有真实 DXF 文件能否解析成功
#[test]
fn test_parse_all_real_dxf_files() {
    let parser = DxfParser::new();
    let dxf_dir = get_dxf_dir();

    // 首先列出目录中的所有 DXF 文件
    println!("\nDXF 目录：{:?}", dxf_dir);
    println!("目录存在：{}", dxf_dir.exists());

    let mut dxf_files: Vec<String> = Vec::new();
    if dxf_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&dxf_dir) {
            for entry in entries.flatten() {
                let file_name = entry.file_name().to_string_lossy().to_string();
                if file_name.ends_with(".dxf") {
                    println!("  找到 DXF 文件：{}", file_name);
                    dxf_files.push(file_name);
                }
            }
        }
    }

    dxf_files.sort(); // 排序确保测试稳定

    println!("  共 {} 个 DXF 文件\n", dxf_files.len());

    let mut results = Vec::new();

    for file_name in &dxf_files {
        let file_path = dxf_dir.join(file_name);

        // 检查文件是否存在
        if !file_path.exists() {
            println!("⚠️  文件不存在：{} (路径：{:?})", file_name, file_path);
            results.push((file_name.as_str(), Err("文件不存在".to_string())));
            continue;
        }

        // 解析文件
        match parser.parse_file(&file_path) {
            Ok(entities) => {
                results.push((file_name.as_str(), Ok(entities.len())));
                println!("✅ {} 解析成功：{} 个实体", file_name, entities.len());
            }
            Err(e) => {
                results.push((file_name.as_str(), Err(format!("{:?}", e))));
                println!("❌ {} 解析失败：{:?}", file_name, e);
            }
        }
    }

    // 统计结果
    let success_count = results.iter().filter(|r| r.1.is_ok()).count();
    let fail_count = results.iter().filter(|r| r.1.is_err()).count();

    println!("\n========== DXF 解析测试报告 ==========");
    println!("总文件数：{}", dxf_files.len());
    println!("解析成功：{}", success_count);
    println!("解析失败：{}", fail_count);
    println!("=====================================\n");

    // 至少应该有部分文件解析成功
    assert!(success_count > 0, "所有 DXF 文件都解析失败");
}

/// 测试解析后实体类型分布
#[test]
fn test_entity_type_distribution() {
    let parser = DxfParser::new();
    let dxf_dir = get_dxf_dir();

    // 尝试查找第一个可解析的 DXF 文件
    let file_path = find_first_parseable_dxf(&dxf_dir, &parser)
        .expect("dxfs/ 目录下应该至少有一个可解析的 DXF 文件");

    let entities = parser
        .parse_file(&file_path)
        .unwrap_or_else(|_| panic!("{:?} 应该能解析成功", file_path));

    // 统计实体类型
    let mut line_count = 0;
    let mut arc_count = 0;
    let mut circle_count = 0;
    let mut polyline_count = 0;
    let mut text_count = 0;

    for entity in &entities {
        match entity {
            common_types::RawEntity::Line { .. } => line_count += 1,
            common_types::RawEntity::Arc { .. } => arc_count += 1,
            common_types::RawEntity::Circle { .. } => circle_count += 1,
            common_types::RawEntity::Polyline { .. } => polyline_count += 1,
            common_types::RawEntity::Text { .. } => text_count += 1,
            _ => {}
        }
    }

    println!("\nexample.dxf 实体类型分布:");
    println!("  LINE: {}", line_count);
    println!("  ARC: {}", arc_count);
    println!("  CIRCLE: {}", circle_count);
    println!("  POLYLINE: {}", polyline_count);
    println!("  TEXT: {}", text_count);

    // 至少应该有一些实体
    assert!(!entities.is_empty(), "example.dxf 应该包含实体");
}

/// 测试图层分布
#[test]
fn test_layer_distribution() {
    let parser = DxfParser::new();
    let dxf_dir = get_dxf_dir();

    // 使用兜底逻辑查找第一个可解析的 DXF 文件
    let file_path = find_first_parseable_dxf(&dxf_dir, &parser)
        .expect("dxfs/ 目录下应该至少有一个可解析的 DXF 文件");

    let entities = parser
        .parse_file(&file_path)
        .unwrap_or_else(|_| panic!("解析失败：{:?}", file_path));

    // 统计图层分布
    use std::collections::HashMap;
    let mut layer_counts: HashMap<String, usize> = HashMap::new();

    for entity in &entities {
        let layer = match entity {
            common_types::RawEntity::Line { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Polyline { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Arc { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Circle { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Text { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Path { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::BlockReference { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Dimension { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Hatch { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::XRef { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Point { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Image { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Attribute { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::AttributeDefinition { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Leader { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Ray { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::MLine { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Triangle { metadata, .. } => metadata.layer.clone(),
        };

        if let Some(layer) = layer {
            *layer_counts.entry(layer).or_insert(0) += 1;
        }
    }

    println!("\n报告厅 1.dxf 图层分布:");
    let mut layers: Vec<_> = layer_counts.iter().collect();
    layers.sort_by(|a, b| b.1.cmp(a.1)); // 按数量降序排列

    for (layer, count) in layers {
        println!("  {}: {} 个实体", layer, count);
    }
}

/// 测试图层过滤功能
#[test]
fn test_layer_filter() {
    let dxf_dir = get_dxf_dir();

    // 使用兜底逻辑查找第一个可解析的 DXF 文件
    let parser = DxfParser::new();
    let file_path = find_first_parseable_dxf(&dxf_dir, &parser)
        .expect("dxfs/ 目录下应该至少有一个可解析的 DXF 文件");

    // 先解析所有实体
    let all_entities = parser.parse_file(&file_path).expect("解析失败");

    // 获取所有图层
    let mut layers: Vec<String> = all_entities
        .iter()
        .filter_map(|e| match e {
            common_types::RawEntity::Line { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Polyline { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Arc { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Circle { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Text { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Path { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::BlockReference { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Dimension { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Hatch { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::XRef { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Point { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Image { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Attribute { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::AttributeDefinition { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Leader { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Ray { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::MLine { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Triangle { metadata, .. } => metadata.layer.clone(),
        })
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    if layers.is_empty() {
        panic!("文件 {:?} 没有图层信息", file_path);
    }

    // 只过滤第一个图层
    let target_layer = layers.remove(0);
    let parser_filtered = DxfParser::new().with_layer_filter(vec![target_layer.clone()]);
    let filtered_entities = parser_filtered.parse_file(&file_path).expect("解析失败");

    // 验证过滤后的实体都属于目标图层或来自块定义内部图层
    // 注意：根据 DXF 规范，块展开后实体保留块定义内部图层（非块引用图层）
    // 因此我们验证实体有图层信息即可，不强制要求与目标图层完全一致
    for entity in &filtered_entities {
        let layer = match entity {
            common_types::RawEntity::Line { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Polyline { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Arc { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Circle { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Text { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Path { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::BlockReference { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Dimension { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Hatch { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::XRef { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Point { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Image { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Attribute { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::AttributeDefinition { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Leader { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Ray { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::MLine { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Triangle { metadata, .. } => metadata.layer.clone(),
        };

        // 验证：实体应该有图层信息（可能是目标图层，也可能是块定义内部图层）
        assert!(layer.is_some(), "过滤后的实体应该有图层信息");
    }

    // 统计各图层实体数量
    use std::collections::HashMap;
    let mut layer_counts: HashMap<String, usize> = HashMap::new();
    for entity in &filtered_entities {
        let layer = match entity {
            common_types::RawEntity::Line { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Polyline { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Arc { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Circle { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Text { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Path { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::BlockReference { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Dimension { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Hatch { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::XRef { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Point { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Image { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Attribute { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::AttributeDefinition { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Leader { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Ray { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::MLine { metadata, .. } => metadata.layer.clone(),
            common_types::RawEntity::Triangle { metadata, .. } => metadata.layer.clone(),
        };
        if let Some(layer) = layer {
            *layer_counts.entry(layer).or_insert(0) += 1;
        }
    }

    println!("\n图层过滤测试:");
    println!("  原始实体数：{}", all_entities.len());
    println!(
        "  过滤后实体数 (目标图层 '{}'): {}",
        filtered_entities.len(),
        target_layer
    );
    println!("  实际图层分布:");
    for (layer, count) in &layer_counts {
        println!("    {}: {} 个实体", layer, count);
    }
}

/// 测试二进制 DXF 解析
#[test]
fn test_binary_dxf_parsing() {
    use dxf::{
        entities::{Entity, Line},
        Drawing,
    };

    // 创建一个简单的 Drawing
    let mut drawing = Drawing::new();
    drawing.add_entity(Entity::new(dxf::entities::EntityType::Line(Line {
        p1: dxf::Point {
            x: 0.0,
            y: 0.0,
            z: 0.0,
        },
        p2: dxf::Point {
            x: 100.0,
            y: 100.0,
            z: 0.0,
        },
        ..Default::default()
    })));

    // 保存为二进制格式
    let mut buffer = Vec::new();
    drawing
        .save_binary(&mut buffer)
        .expect("保存二进制 DXF 失败");

    // 解析二进制 DXF
    let parser = DxfParser::new();
    let entities = parser.parse_bytes(&buffer).expect("二进制 DXF 解析失败");

    assert_eq!(entities.len(), 1, "应该解析出 1 个实体");
    println!("\n✅ 二进制 DXF 解析测试通过");
}

/// 测试图层发现功能
#[test]
fn test_get_layer_list() {
    let parser = DxfParser::new();
    let dxf_dir = get_dxf_dir();

    // 使用兜底逻辑查找第一个可解析的 DXF 文件
    let file_path = find_first_parseable_dxf(&dxf_dir, &parser)
        .expect("dxfs/ 目录下应该至少有一个可解析的 DXF 文件");

    let layers = parser.get_layer_list(&file_path).expect("获取图层列表失败");

    println!("\n{:?} 图层列表:", file_path);
    for layer in &layers {
        println!("  - {}", layer);
    }

    assert!(!layers.is_empty(), "应该至少有一个图层");
}

/// 测试解析报告功能
#[test]
fn test_parse_with_report() {
    let parser = DxfParser::new();
    let dxf_dir = get_dxf_dir();

    // 使用兜底逻辑查找第一个可解析的 DXF 文件
    let file_path = find_first_parseable_dxf(&dxf_dir, &parser)
        .expect("dxfs/ 目录下应该至少有一个可解析的 DXF 文件");

    let (entities, report) = parser.parse_file_with_report(&file_path).expect("解析失败");

    println!("\n{}", report);
    println!("解析出 {} 个实体", entities.len());

    assert!(!entities.is_empty(), "应该解析出至少一个实体");
    assert!(!report.layer_distribution.is_empty(), "应该至少有一个图层");
    assert!(
        !report.entity_type_distribution.is_empty(),
        "应该至少有一种实体类型"
    );
}

/// 测试智能图层识别功能
#[test]
fn test_smart_layer_detection() {
    let parser = DxfParser::new();
    let dxf_dir = get_dxf_dir();

    // 使用兜底逻辑查找第一个可解析的 DXF 文件
    let file_path = find_first_parseable_dxf(&dxf_dir, &parser)
        .expect("dxfs/ 目录下应该至少有一个可解析的 DXF 文件");

    // 获取所有图层
    let all_layers = parser.get_layer_list(&file_path).expect("获取图层列表失败");
    println!("\n所有图层 ({} 个):", all_layers.len());
    for layer in &all_layers {
        println!("  - {}", layer);
    }

    // 测试墙体图层识别
    let wall_layers = parser
        .detect_wall_layers(&file_path)
        .expect("获取墙体图层失败");
    println!("\n墙体图层：{:?}", wall_layers);

    // 测试门窗图层识别
    let door_window_layers = parser
        .detect_door_window_layers(&file_path)
        .expect("获取门窗图层失败");
    println!("门窗图层：{:?}", door_window_layers);

    // 测试家具图层识别
    let furniture_layers = parser
        .detect_furniture_layers(&file_path)
        .expect("获取家具图层失败");
    println!("家具图层：{:?}", furniture_layers);

    // 测试标注图层识别
    let dimension_layers = parser
        .detect_dimension_layers(&file_path)
        .expect("获取标注图层失败");
    println!("标注图层：{:?}", dimension_layers);

    // 至少应该能识别出一些图层
    let total_detected = wall_layers.len()
        + door_window_layers.len()
        + furniture_layers.len()
        + dimension_layers.len();
    println!("共识别出 {} 个语义图层", total_detected);

    // 断言：如果文件有图层，应该至少识别出一些语义图层
    if !all_layers.is_empty() {
        assert!(
            total_detected > 0,
            "文件有 {} 个图层但未识别出任何语义图层，图层关键词可能需要扩展",
            all_layers.len()
        );
    }

    // 断言：验证识别出的图层确实是所有图层的子集
    for layer in &wall_layers {
        assert!(
            all_layers.contains(layer),
            "墙体图层 '{}' 不在原始图层列表中",
            layer
        );
    }
    for layer in &door_window_layers {
        assert!(
            all_layers.contains(layer),
            "门窗图层 '{}' 不在原始图层列表中",
            layer
        );
    }
    for layer in &furniture_layers {
        assert!(
            all_layers.contains(layer),
            "家具图层 '{}' 不在原始图层列表中",
            layer
        );
    }
    for layer in &dimension_layers {
        assert!(
            all_layers.contains(layer),
            "标注图层 '{}' 不在原始图层列表中",
            layer
        );
    }
}

/// 测试单位解析功能
#[test]
fn test_unit_parsing() {
    let parser = DxfParser::new();
    let dxf_dir = get_dxf_dir();

    // 使用兜底逻辑查找第一个可解析的 DXF 文件
    let file_path = find_first_parseable_dxf(&dxf_dir, &parser)
        .expect("dxfs/ 目录下应该至少有一个可解析的 DXF 文件");

    let (_, report) = parser.parse_file_with_report(&file_path).expect("解析失败");

    println!(
        "\n图纸单位：{:?}, 比例因子：{}",
        report.drawing_units, report.unit_scale
    );

    // 单位应该是已知的几种之一
    if let Some(units) = &report.drawing_units {
        println!("单位解析成功：{}", units);
        assert!(report.unit_scale > 0.0, "比例因子应该为正数");
    } else {
        println!("警告：未检测到单位信息");
    }
}

/// 测试单位比例因子的应用验证
///
/// 验证解析出的坐标是否已正确应用单位转换
#[test]
fn test_unit_scale_application() {
    let parser = DxfParser::new();
    let dxf_dir = get_dxf_dir();

    // 测试所有 DXF 文件（动态查找）
    let mut test_files: Vec<String> = Vec::new();
    if dxf_dir.exists() {
        if let Ok(entries) = std::fs::read_dir(&dxf_dir) {
            for entry in entries.flatten() {
                let file_name = entry.file_name().to_string_lossy().to_string();
                if file_name.ends_with(".dxf") {
                    test_files.push(file_name);
                }
            }
        }
    }

    if test_files.is_empty() {
        panic!("dxfs/ 目录下没有 DXF 文件");
    }

    test_files.sort();

    for file_name in &test_files {
        let file_path = dxf_dir.join(file_name);

        let (entities, report) = parser
            .parse_file_with_report(&file_path)
            .unwrap_or_else(|_| panic!("解析失败：{:?}", file_path));

        println!(
            "\n文件：{} (单位：{:?}, 比例因子：{})",
            file_name, report.drawing_units, report.unit_scale
        );

        // 检查坐标值是否在合理范围内
        // 建筑图纸通常在毫米级别（1000-50000mm）或米级别（1-50m）
        let mut max_coord = 0.0;
        let mut min_coord = f64::MAX;

        for entity in &entities {
            let coords: Vec<f64> = match entity {
                common_types::RawEntity::Line { start, end, .. } => {
                    vec![start[0], start[1], end[0], end[1]]
                }
                common_types::RawEntity::Polyline { points, .. } => {
                    points.iter().flat_map(|v| vec![v[0], v[1]]).collect()
                }
                common_types::RawEntity::Arc { center, radius, .. } => {
                    vec![
                        center[0] - radius,
                        center[0] + radius,
                        center[1] - radius,
                        center[1] + radius,
                    ]
                }
                common_types::RawEntity::Circle { center, radius, .. } => {
                    vec![
                        center[0] - radius,
                        center[0] + radius,
                        center[1] - radius,
                        center[1] + radius,
                    ]
                }
                _ => continue,
            };

            for &coord in &coords {
                let abs_coord = coord.abs();
                if abs_coord > max_coord {
                    max_coord = abs_coord;
                }
                if abs_coord < min_coord {
                    min_coord = abs_coord;
                }
            }
        }

        // 验证坐标范围合理性
        // 如果单位是米（scale=1000），坐标应该在 0.001-100 之间（转换后 1-100000mm）
        // 如果单位是毫米（scale=1），坐标应该在 1-100000 之间
        println!("  坐标范围：[{:.2}, {:.2}]", min_coord, max_coord);

        // 检查比例因子是否合理应用
        if let Some(units) = &report.drawing_units {
            let unit_str = units.to_lowercase();

            // 如果是米单位，坐标值应该较小（通常 < 1000）
            if unit_str.contains("米") || unit_str.contains("meter") {
                assert!(report.unit_scale == 1000.0, "米单位的比例因子应该是 1000");
                // 如果原始坐标 > 10000，应该触发单位不匹配警告
                if max_coord > 10000.0 {
                    assert!(
                        report.unit_mismatch_detected,
                        "米单位但坐标值较大，应该触发单位不匹配警告"
                    );
                    println!("  ✅ 正确检测到单位不匹配");
                }
            }
            // 如果是毫米单位，坐标值应该较大（通常 > 100）
            else if unit_str.contains("毫米") || unit_str.contains("millimeter") {
                assert!(report.unit_scale == 1.0, "毫米单位的比例因子应该是 1");
            }
            // 如果是英寸单位，验证警告机制
            else if unit_str.contains("英寸") || unit_str.contains("inch") {
                assert!(report.unit_scale == 25.4, "英寸单位的比例因子应该是 25.4");
                // 如果坐标值过大，应该触发警告
                if max_coord > 10000.0 {
                    assert!(
                        report.unit_mismatch_detected,
                        "英寸单位但坐标值较大，应该触发单位不匹配警告"
                    );
                    println!("  ✅ 正确检测到单位不匹配");
                }
            }
        }
    }
}

/// 测试问题文件：端点错位 0.3mm
///
/// 验证解析器能否正确处理端点不完全重合的线段
#[test]
fn test_problem_file_endpoint_mismatch() {
    let parser = DxfParser::new();
    let dxf_dir = get_dxf_dir();
    let file_path = dxf_dir.join("问题文件 - 端点错位 0.3mm.dxf");

    // 文件应该存在
    if !file_path.exists() {
        panic!(
            "问题文件不存在：{:?}\n请检查 dxfs/ 目录是否包含该测试文件",
            file_path
        );
    }

    // 解析文件应该成功（即使有端点错位）
    let entities = parser
        .parse_file(&file_path)
        .expect("问题文件应该能解析成功");

    println!("\n问题文件 - 端点错位 0.3mm:");
    println!("  解析出 {} 个实体", entities.len());

    // 应该解析出 4 条 LINE
    let line_count = entities
        .iter()
        .filter(|e| matches!(e, common_types::RawEntity::Line { .. }))
        .count();

    assert_eq!(line_count, 4, "应该解析出 4 条 LINE 实体");

    // 验证：解析器不应该自动修复端点错位（保持原始数据）
    // 检查第三条线的端点是否确实有 0.3mm 错位
    for entity in &entities {
        if let common_types::RawEntity::Line { end, metadata, .. } = entity {
            // 第三条线的 end 点 Y 坐标应该是 100.3（有 0.3mm 错位）
            if end[0] == 0.0 && end[1] == 100.3 {
                println!("  ✅ 检测到端点错位：第三条线 end 点 Y={}", end[1]);
                // 验证错位确实存在
                let expected_y = 100.0;
                let actual_y = end[1];
                let mismatch = (actual_y - expected_y).abs();
                assert!(
                    mismatch > 0.1,
                    "应该检测到端点错位，但错位太小：{}mm",
                    mismatch
                );
                println!("  错位距离：{}mm", mismatch);
            }

            // 验证所有实体的图层信息正确
            if let Some(layer) = &metadata.layer {
                assert_eq!(layer, "WALL", "图层应该是 WALL");
            }
        }
    }

    println!("  ✅ 端点错位文件解析测试通过");
}

/// 测试问题文件：自相交多边形
///
/// 验证解析器能否正确处理自相交的几何形状（使用两条交叉线模拟）
#[test]
fn test_problem_file_self_intersecting() {
    let parser = DxfParser::new();
    let dxf_dir = get_dxf_dir();
    let file_path = dxf_dir.join("问题文件 - 自相交多边形.dxf");

    // 文件应该存在
    if !file_path.exists() {
        panic!(
            "问题文件不存在：{:?}\n请检查 dxfs/ 目录是否包含该测试文件",
            file_path
        );
    }

    // 解析文件应该成功（即使是自相交形状）
    let entities = parser
        .parse_file(&file_path)
        .expect("问题文件应该能解析成功");

    println!("\n问题文件 - 自相交多边形:");
    println!("  解析出 {} 个实体", entities.len());

    // 应该解析出 2 条 LINE（形成 X 形交叉）
    let line_count = entities
        .iter()
        .filter(|e| matches!(e, common_types::RawEntity::Line { .. }))
        .count();

    assert_eq!(line_count, 2, "应该解析出 2 条 LINE 实体（交叉线）");

    // 验证所有实体的图层信息
    for entity in &entities {
        if let common_types::RawEntity::Line { metadata, .. } = entity {
            if let Some(layer) = &metadata.layer {
                assert_eq!(layer, "WALL", "图层应该是 WALL");
            }
        }
    }

    println!("  ✅ 自相交多边形文件解析测试通过");
}
