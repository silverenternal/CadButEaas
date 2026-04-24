//! BLOCK/INSERT 功能测试
//!
//! 测试 DXF 块定义和块引用解析功能

use parser::DxfParser;

/// 创建一个简单的椅子块定义测试 DXF 文件（内存中）
/// 椅子块由 4 条腿（LINE）和 1 个椅面（LWPOLYLINE）组成
fn create_chair_block_dxf() -> Vec<u8> {
    // 使用 ASCII DXF R12 格式
    let mut dxf = String::new();

    // 文件头
    dxf.push_str("0\nSECTION\n2\nHEADER\n9\n$ACADVER\n1\nAC1009\n9\n$INSUNITS\n70\n4\n0\nENDSEC\n");

    // 块定义段
    dxf.push_str("0\nSECTION\n2\nBLOCKS\n");

    // 椅子块定义
    dxf.push_str("0\nBLOCK\n2\nCHAIR_001\n70\n0\n10\n0.0\n20\n0.0\n30\n0.0\n");

    // 椅腿 1
    dxf.push_str(
        "0\nLINE\n8\nFURNITURE\n62\n1\n10\n0.0\n20\n0.0\n30\n0.0\n11\n0.0\n21\n400.0\n31\n0.0\n",
    );
    // 椅腿 2
    dxf.push_str("0\nLINE\n8\nFURNITURE\n62\n1\n10\n400.0\n20\n0.0\n30\n0.0\n11\n400.0\n21\n400.0\n31\n0.0\n");
    // 椅腿 3
    dxf.push_str("0\nLINE\n8\nFURNITURE\n62\n1\n10\n400.0\n20\n400.0\n30\n0.0\n11\n0.0\n21\n400.0\n31\n0.0\n");
    // 椅腿 4
    dxf.push_str(
        "0\nLINE\n8\nFURNITURE\n62\n1\n10\n0.0\n20\n400.0\n30\n0.0\n11\n0.0\n21\n0.0\n31\n0.0\n",
    );

    // 椅面（矩形）
    dxf.push_str("0\nLWPOLYLINE\n8\nFURNITURE\n62\n3\n90\n4\n70\n1\n");
    dxf.push_str("10\n0.0\n20\n0.0\n");
    dxf.push_str("10\n400.0\n20\n0.0\n");
    dxf.push_str("10\n400.0\n20\n400.0\n");
    dxf.push_str("10\n0.0\n20\n400.0\n");

    dxf.push_str("0\nENDBLK\n8\n0\n");

    dxf.push_str("0\nENDSEC\n");

    // 实体段 - 插入 3 把椅子
    dxf.push_str("0\nSECTION\n2\nENTITIES\n");

    // 插入椅子 1：位置 (1000, 1000)，无旋转，比例 1.0
    dxf.push_str("0\nINSERT\n2\nCHAIR_001\n8\nFURNITURE\n62\n1\n10\n1000.0\n20\n1000.0\n30\n0.0\n41\n1.0\n42\n1.0\n43\n1.0\n50\n0.0\n");

    // 插入椅子 2：位置 (2000, 1000)，旋转 90 度，比例 1.0
    dxf.push_str("0\nINSERT\n2\nCHAIR_001\n8\nFURNITURE\n62\n1\n10\n2000.0\n20\n1000.0\n30\n0.0\n41\n1.0\n42\n1.0\n43\n1.0\n50\n90.0\n");

    // 插入椅子 3：位置 (1500, 2000)，旋转 180 度，比例 0.8
    dxf.push_str("0\nINSERT\n2\nCHAIR_001\n8\nFURNITURE\n62\n1\n10\n1500.0\n20\n2000.0\n30\n0.0\n41\n0.8\n42\n0.8\n43\n0.8\n50\n180.0\n");

    dxf.push_str("0\nENDSEC\n");
    dxf.push_str("0\nEOF\n");

    dxf.into_bytes()
}

#[test]
fn test_block_definition_parsing() {
    // 测试块定义解析
    let dxf_bytes = create_chair_block_dxf();

    // 保存到临时文件
    let temp_path = std::env::temp_dir().join("test_chair_block.dxf");
    std::fs::write(&temp_path, &dxf_bytes).expect("无法写入临时文件");

    // 解析文件
    let parser = DxfParser::new();
    let result = parser.parse_file_with_report(&temp_path);

    assert!(result.is_ok(), "解析应该成功：{:?}", result.err());
    let (entities, report) = result.unwrap();

    // 验证报告
    println!("块定义数量：{}", report.block_definitions_count);
    println!("块引用数量：{}", report.block_references_count);
    println!("实体总数：{}", entities.len());

    // 应该有 1 个块定义（CHAIR_001）
    assert_eq!(report.block_definitions_count, 1, "应该有 1 个块定义");

    // 应该有 3 个块引用（3 把椅子）
    assert_eq!(report.block_references_count, 3, "应该有 3 个块引用");

    // 每个椅子块包含 5 个实体（4 条腿 + 1 个椅面）
    // 3 把椅子 = 15 个实体
    assert!(
        entities.len() >= 15,
        "应该至少有 15 个实体（3 把椅子 × 5 个几何元素），实际：{}",
        entities.len()
    );

    // 清理临时文件
    let _ = std::fs::remove_file(&temp_path);
}

#[test]
fn test_block_transformation() {
    // 测试块变换（缩放、旋转、平移）
    let dxf_bytes = create_chair_block_dxf();

    let temp_path = std::env::temp_dir().join("test_chair_transform.dxf");
    std::fs::write(&temp_path, &dxf_bytes).expect("无法写入临时文件");

    let parser = DxfParser::new();
    let (entities, _) = parser
        .parse_file_with_report(&temp_path)
        .expect("解析应该成功");

    // 验证实体坐标范围
    // 椅子 1：位置 (1000, 1000)，无旋转
    // 椅子 2：位置 (2000, 1000)，旋转 90 度
    // 椅子 3：位置 (1500, 2000)，旋转 180 度，缩放 0.8

    let mut x_coords: Vec<f64> = Vec::new();
    let mut y_coords: Vec<f64> = Vec::new();

    for entity in &entities {
        match entity {
            common_types::RawEntity::Line { start, end, .. } => {
                x_coords.extend_from_slice(&[start[0], end[0]]);
                y_coords.extend_from_slice(&[start[1], end[1]]);
            }
            common_types::RawEntity::Polyline { points, .. } => {
                for p in points {
                    x_coords.push(p[0]);
                    y_coords.push(p[1]);
                }
            }
            _ => {}
        }
    }

    // 验证坐标范围合理
    let min_x = x_coords.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_x = x_coords.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
    let min_y = y_coords.iter().cloned().fold(f64::INFINITY, f64::min);
    let max_y = y_coords.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    println!("X 坐标范围：{:.1} - {:.1}", min_x, max_x);
    println!("Y 坐标范围：{:.1} - {:.1}", min_y, max_y);

    // 坐标应该在合理范围内（0 到 3000 左右）
    assert!(min_x >= 0.0, "X 坐标不应该小于 0");
    assert!(max_x <= 3000.0, "X 坐标最大值应该在 3000 以内");
    assert!(min_y >= 0.0, "Y 坐标不应该小于 0");
    assert!(max_y <= 3000.0, "Y 坐标最大值应该在 3000 以内");

    let _ = std::fs::remove_file(&temp_path);
}

#[test]
fn test_block_layer_preservation() {
    // 测试块定义图层保留
    let dxf_bytes = create_chair_block_dxf();

    let temp_path = std::env::temp_dir().join("test_chair_layer.dxf");
    std::fs::write(&temp_path, &dxf_bytes).expect("无法写入临时文件");

    let parser = DxfParser::new();
    let (entities, _) = parser
        .parse_file_with_report(&temp_path)
        .expect("解析应该成功");

    // 验证所有实体都来自 FURNITURE 图层
    for entity in &entities {
        let layer = entity.layer();
        assert_eq!(
            layer,
            Some("FURNITURE"),
            "实体应该来自 FURNITURE 图层，实际：{:?}",
            layer
        );
    }

    let _ = std::fs::remove_file(&temp_path);
}
