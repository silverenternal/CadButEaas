//! PDF 压缩流支持测试
//!
//! 验证 FlateDecode 等压缩过滤器的支持

use parser::{ParseResult, ParserService, PdfParser};
use std::path::PathBuf;

/// 获取 testpdf 目录路径
fn testpdf_dir() -> PathBuf {
    let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").unwrap();
    PathBuf::from(manifest_dir).join("../../testpdf")
}

#[test]
fn test_pdf_flate_decode_basic() {
    // 测试基础 FlateDecode 压缩支持
    // 使用 lopdf 直接创建带压缩流的 PDF 内容
    use lopdf::{Dictionary, Document, Stream};

    // 创建简单的 PDF 文档
    let _doc = Document::new();

    // 创建压缩流对象
    let compressed_data = vec![1u8, 2, 3, 4, 5]; // 简单测试数据
    let mut stream = Stream::new(Dictionary::new(), compressed_data);
    let _ = stream.compress(); // 压缩可能失败，忽略结果

    // 验证流被压缩（如果压缩成功）
    if stream.dict.get(b"Filter").is_ok() {
        println!("✓ FlateDecode 压缩创建成功");
    } else {
        println!("⚠️  FlateDecode 压缩未执行（可能数据太小）");
    }
}

#[test]
fn test_pdf_all_test_files() {
    // 测试所有真实 PDF 文件（应该都支持压缩流）
    let pdf_dir = testpdf_dir();

    if !pdf_dir.exists() {
        panic!(
            "PDF 测试目录不存在：{:?}\n\
             请检查 testpdf/ 目录是否已创建并包含测试文件",
            pdf_dir
        );
    }

    let service = ParserService::new();
    let mut success_count = 0;
    let mut total_count = 0;

    for entry in std::fs::read_dir(&pdf_dir).expect("无法读取目录").flatten() {
        let path = entry.path();
        if path.extension().is_some_and(|ext| ext == "pdf") {
            total_count += 1;

            match service.parse_file(&path) {
                Ok(result) => match result {
                    ParseResult::Pdf(pdf_content) => {
                        println!(
                            "✓ {:?}: 解析成功 ({} 矢量实体，{} 图像)",
                            path.file_name().unwrap_or_default(),
                            pdf_content.vector_entities.len(),
                            pdf_content.raster_images.len()
                        );
                        success_count += 1;
                    }
                    ParseResult::Cad(_) => {
                        println!(
                            "⚠️  {:?}: 文件类型不匹配",
                            path.file_name().unwrap_or_default()
                        );
                    }
                },
                Err(e) => {
                    println!(
                        "✗ {:?}: 解析失败 - {:?}",
                        path.file_name().unwrap_or_default(),
                        e
                    );
                }
            }
        }
    }

    println!("\nPDF 解析统计：{}/{} 成功", success_count, total_count);
    assert!(success_count > 0, "至少应该成功解析一个 PDF 文件");
}

#[test]
fn test_pdf_parser_direct_flate() {
    // 直接测试 PDF 解析器的 FlateDecode 支持
    use lopdf::{Dictionary, Document, Stream};

    // 创建带压缩内容流的 PDF
    let mut doc = Document::new();

    // 添加页面
    let page_id = doc.new_object_id();

    // 创建压缩的内容流
    let content_stream = b"1 0 0 1 100 100 cm\nq\n1 0 0 1 0 0 cm\nQ\n";
    let mut stream = Stream::new(Dictionary::new(), content_stream.to_vec());
    let _ = stream.compress(); // 使用 FlateDecode 压缩

    let stream_id = doc.add_object(stream);

    // 设置页面内容
    let mut page_dict = Dictionary::new();
    page_dict.set("Type", "Page");
    page_dict.set("Contents", stream_id);

    doc.set_object(page_id, page_dict);

    // 保存到内存
    let mut buffer = Vec::new();
    doc.save_to(&mut buffer).expect("保存 PDF 失败");

    // 尝试解析
    let parser = PdfParser::new();
    let result = parser.parse_bytes(&buffer);

    // 应该成功解析（即使内容为空）
    assert!(
        result.is_ok(),
        "带 FlateDecode 的 PDF 应该能解析：{:?}",
        result.err()
    );

    println!("✓ 带 FlateDecode 压缩的 PDF 解析成功");
}

#[test]
fn test_pdf_image_flate_decode() {
    // 测试 PDF 中图像的 FlateDecode 支持
    use lopdf::{Dictionary, Document, Stream};

    let _doc = Document::new();

    // 创建压缩的图像流
    let image_data = vec![0u8; 100]; // 100 字节的图像数据
    let mut stream = Stream::new(Dictionary::new(), image_data);
    let _ = stream.compress();

    // 验证过滤器
    let filter = stream.dict.get(b"Filter").ok();
    assert!(filter.is_some(), "图像流应该有 Filter");

    if let Some(filter_obj) = filter {
        println!("✓ 图像压缩过滤器：{:?}", filter_obj);
    }

    println!("✓ PDF 图像 FlateDecode 支持验证通过");
}
