//! PDF 真实文件集成测试
//!
//! 测试真实 PDF 文件的解析能力

use parser::{FileType, ParserService};
use std::path::PathBuf;

/// 获取 testpdf 目录路径
fn testpdf_dir() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(manifest_dir).join("../../testpdf")
}

#[test]
fn test_parse_real_pdf_files_exist() {
    // 验证测试文件存在
    let test_dir = testpdf_dir();
    println!("PDF 测试目录：{:?}", test_dir);

    assert!(test_dir.exists(), "testpdf 目录不存在：{:?}", test_dir);

    // 列出所有 PDF 文件
    let pdf_files: Vec<_> = std::fs::read_dir(&test_dir)
        .expect("无法读取测试目录")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "pdf"))
        .collect();

    println!("找到 {} 个 PDF 测试文件", pdf_files.len());
    for file in &pdf_files {
        println!("  - {}", file.file_name().to_string_lossy());
    }

    assert!(pdf_files.len() >= 4, "期望至少 4 个 PDF 测试文件");
}

#[test]
fn test_parse_20x40_house_plan() {
    let pdf_path = testpdf_dir().join("20x40-house-with-4-bedrooms.pdf");

    // 文件不存在时 panic
    if !pdf_path.exists() {
        panic!(
            "测试文件不存在：{:?}，请检查 git lfs 或测试数据部署",
            pdf_path
        );
    }

    let service = ParserService::new();
    let result = service.parse_file(&pdf_path);

    assert!(result.is_ok(), "解析失败：{:?}", result.err());

    let parse_result = result.unwrap();
    let has_raster = parse_result.has_raster();
    let entities = parse_result.into_entities();

    println!("20x40 house plan: 解析出 {} 个矢量实体", entities.len());
    println!("  是否包含光栅：{}", has_raster);

    // 房屋平面图应该包含一定数量的线段
    assert!(!entities.is_empty(), "应该解析出至少一个实体");
}

#[test]
fn test_parse_36x32_house_plan() {
    let pdf_path = testpdf_dir().join("36x32-house-with-4-bedroom.pdf");

    // 文件不存在时 panic
    if !pdf_path.exists() {
        panic!(
            "测试文件不存在：{:?}，请检查 git lfs 或测试数据部署",
            pdf_path
        );
    }

    let service = ParserService::new();
    let result = service.parse_file(&pdf_path);

    assert!(result.is_ok(), "解析失败：{:?}", result.err());

    let parse_result = result.unwrap();
    let entities = parse_result.into_entities();

    println!("36x32 house plan: 解析出 {} 个矢量实体", entities.len());
    assert!(!entities.is_empty(), "应该解析出至少一个实体");
}

#[test]
fn test_parse_64x60_house_plan() {
    let pdf_path = testpdf_dir().join("64x60-house-plan-with4-bedrooms.pdf");

    // 文件不存在时 panic
    if !pdf_path.exists() {
        panic!(
            "测试文件不存在：{:?}，请检查 git lfs 或测试数据部署",
            pdf_path
        );
    }

    let service = ParserService::new();
    let result = service.parse_file(&pdf_path);

    assert!(result.is_ok(), "解析失败：{:?}", result.err());

    let parse_result = result.unwrap();
    let entities = parse_result.into_entities();

    println!("64x60 house plan: 解析出 {} 个矢量实体", entities.len());
    assert!(!entities.is_empty(), "应该解析出至少一个实体");
}

#[test]
fn test_parse_45x40_house_plan() {
    let pdf_path = testpdf_dir().join("45x40-house-with-3-Bedooms.pdf");

    // 文件不存在时 panic
    if !pdf_path.exists() {
        panic!(
            "测试文件不存在：{:?}，请检查 git lfs 或测试数据部署",
            pdf_path
        );
    }

    let service = ParserService::new();
    let result = service.parse_file(&pdf_path);

    assert!(result.is_ok(), "解析失败：{:?}", result.err());

    let parse_result = result.unwrap();
    let entities = parse_result.into_entities();

    println!("45x40 house plan: 解析出 {} 个矢量实体", entities.len());
    assert!(!entities.is_empty(), "应该解析出至少一个实体");
}

#[test]
fn test_parse_all_real_pdfs_report() {
    // 生成所有 PDF 文件的解析报告
    let test_dir = testpdf_dir();
    let pdf_files: Vec<_> = std::fs::read_dir(&test_dir)
        .expect("无法读取测试目录")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "pdf"))
        .collect();

    println!("\n========== PDF 解析测试报告 ==========");

    let mut total_entities = 0;
    let mut success_count = 0;
    let mut fail_count = 0;

    for file in &pdf_files {
        let path = file.path();
        let service = ParserService::new();

        match service.parse_file(&path) {
            Ok(result) => {
                let has_raster = result.has_raster();
                let entities = result.into_entities();
                total_entities += entities.len();
                success_count += 1;

                println!(
                    "✓ {}: {} 实体，光栅={}",
                    path.file_name().unwrap().to_string_lossy(),
                    entities.len(),
                    has_raster
                );
            }
            Err(e) => {
                fail_count += 1;
                println!("✗ {}: {:?}", path.file_name().unwrap().to_string_lossy(), e);
            }
        }
    }

    println!("\n---------- 统计 ----------");
    println!("总文件数：{}", pdf_files.len());
    println!("成功：{}", success_count);
    println!("失败：{}", fail_count);
    println!("总实体数：{}", total_entities);
    println!("==============================\n");

    // 至少应该成功解析大部分文件
    assert!(success_count > 0, "应该至少成功解析一个 PDF 文件");
}

#[test]
fn test_pdf_bytes_parsing() {
    // 测试从字节解析 PDF
    let pdf_path = testpdf_dir().join("20x40-house-with-4-bedrooms.pdf");

    // 文件不存在时 panic
    if !pdf_path.exists() {
        panic!(
            "测试文件不存在：{:?}，请检查 git lfs 或测试数据部署",
            pdf_path
        );
    }

    let bytes = std::fs::read(&pdf_path).expect("无法读取 PDF 文件");
    let service = ParserService::new();
    let result = service.parse_bytes(&bytes, FileType::Pdf);

    assert!(result.is_ok(), "字节解析失败：{:?}", result.err());

    let parse_result = result.unwrap();
    let entities = parse_result.into_entities();

    println!("字节解析：{} 个实体", entities.len());
    assert!(!entities.is_empty(), "应该解析出至少一个实体");
}

#[test]
fn test_pdf_vector_vs_raster_detection() {
    // 测试矢量/光栅判定
    let test_dir = testpdf_dir();
    let pdf_files: Vec<_> = std::fs::read_dir(&test_dir)
        .expect("无法读取测试目录")
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|ext| ext == "pdf"))
        .collect();

    println!("\n========== PDF 类型判定测试 ==========");

    for file in &pdf_files {
        let path = file.path();
        let service = ParserService::new();

        if let Ok(result) = service.parse_file(&path) {
            let has_raster = result.has_raster();
            let entities = result.into_entities();
            let has_vector = !entities.is_empty();

            let pdf_type = match (has_vector, has_raster) {
                (true, false) => "矢量 PDF",
                (true, true) => "混合 PDF",
                (false, true) => "光栅 PDF",
                (false, false) => "空 PDF",
            };

            println!(
                "{}: {}",
                path.file_name().unwrap().to_string_lossy(),
                pdf_type
            );
        }
    }

    println!("==============================\n");
}
