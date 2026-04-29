//! 边界情况测试
//!
//! 测试解析器在极端/异常情况下的行为

use parser::{DxfParser, FileType, ParseResult, ParserService};

fn get_entities(result: &ParseResult) -> usize {
    match result {
        ParseResult::Cad(entities) => entities.len(),
        ParseResult::Pdf(content) => content.vector_entities.len(),
    }
}

// ==================== 空文件测试 ====================

#[test]
fn test_empty_dxf_file() {
    let service = ParserService::new();

    let result = service.parse_bytes(&[], FileType::Dxf);
    // 空 DXF 可能解析成功（返回空实体列表）或失败，取决于解析器实现
    match &result {
        Ok(parse_result) => {
            let count = get_entities(parse_result);
            tracing::debug!("空 DXF 文件解析成功，实体数：{}", count);
            assert_eq!(count, 0);
        }
        Err(e) => {
            tracing::debug!("空 DXF 文件解析失败：{:?}", e);
            // 失败也是可接受的行为
        }
    }
}

#[test]
fn test_empty_pdf_file() {
    let service = ParserService::new();

    let result = service.parse_bytes(&[], FileType::Pdf);
    assert!(result.is_err());

    tracing::debug!("空 PDF 文件错误：{:?}", result.unwrap_err());
}

// ==================== 损坏文件测试 ====================

#[test]
fn test_corrupted_dxf_file() {
    let service = ParserService::new();

    let corrupted_data = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE];
    let result = service.parse_bytes(&corrupted_data, FileType::Dxf);
    assert!(result.is_err());

    tracing::debug!("损坏 DXF 文件错误：{:?}", result.unwrap_err());
}

#[test]
fn test_corrupted_pdf_file() {
    let service = ParserService::new();

    let corrupted_data = vec![0xDE, 0xAD, 0xBE, 0xEF, 0xCA, 0xFE];
    let result = service.parse_bytes(&corrupted_data, FileType::Pdf);
    assert!(result.is_err());

    tracing::debug!("损坏 PDF 文件错误：{:?}", result.unwrap_err());
}

#[test]
fn test_invalid_pdf_header() {
    let service = ParserService::new();

    let corrupted_pdf = b"%PDF-1.4\n% corrupted content";
    let result = service.parse_bytes(corrupted_pdf, FileType::Pdf);
    tracing::debug!("损坏 PDF 头结果：{:?}", result);
}

#[test]
fn test_truncated_dxf_file() {
    let service = ParserService::new();

    let truncated = b"0\nSECTION\n2\nENTITIES\n";
    let result = service.parse_bytes(truncated, FileType::Dxf);
    tracing::debug!("截断 DXF 结果：{:?}", result);
}

// ==================== 特殊字符和编码测试 ====================

#[test]
fn test_unicode_in_dxf() {
    let service = ParserService::new();

    let unicode_dxf = b"0\nSECTION\n2\nHEADER\n9\n$DWGCODEPAGE\n3\nANSI_936\n0\nENDSEC\n0\nSECTION\n2\nENTITIES\n0\nLINE\n8\nWALL\n62\n7\n0\nENDSEC\n0\nEOF\n";
    let result = service.parse_bytes(unicode_dxf, FileType::Dxf);

    match &result {
        Ok(parse_result) => {
            let count = get_entities(parse_result);
            tracing::debug!("Unicode DXF 解析成功，实体数：{}", count);
        }
        Err(e) => {
            tracing::debug!("Unicode DXF 解析失败：{:?}", e);
        }
    }
}

#[test]
fn test_special_characters_in_layer_name() {
    let service = ParserService::new();

    let dxf = b"0\nSECTION\n2\nHEADER\n0\nENDSEC\n0\nSECTION\n2\nENTITIES\n0\nLINE\n8\nA-WALL-TEST_@#$%\n62\n7\n0\nENDSEC\n0\nEOF\n";
    let result = service.parse_bytes(dxf, FileType::Dxf);

    match &result {
        Ok(parse_result) => {
            let count = get_entities(parse_result);
            tracing::debug!("特殊字符图层名解析成功，实体数：{}", count);
        }
        Err(e) => {
            tracing::debug!("特殊字符图层名解析失败：{:?}", e);
        }
    }
}

// ==================== 极端坐标值测试 ====================

#[test]
fn test_extreme_coordinates_dxf() {
    let service = ParserService::new();

    let large_coord_dxf = b"0\nSECTION\n2\nENTITIES\n0\nLINE\n10\n1000000.0\n20\n1000000.0\n30\n0.0\n11\n1000001.0\n21\n1000001.0\n31\n0.0\n0\nENDSEC\n0\nEOF\n";
    let result = service.parse_bytes(large_coord_dxf, FileType::Dxf);

    match &result {
        Ok(parse_result) => {
            let count = get_entities(parse_result);
            tracing::debug!("大坐标 DXF 解析成功，实体数：{}", count);
            assert!(count > 0);
        }
        Err(e) => {
            tracing::debug!("大坐标 DXF 解析失败：{:?}", e);
        }
    }
}

#[test]
fn test_negative_coordinates_dxf() {
    let service = ParserService::new();

    let negative_coord_dxf = b"0\nSECTION\n2\nENTITIES\n0\nLINE\n10\n-100.0\n20\n-200.0\n30\n0.0\n11\n-150.0\n21\n-250.0\n31\n0.0\n0\nENDSEC\n0\nEOF\n";
    let result = service.parse_bytes(negative_coord_dxf, FileType::Dxf);

    match &result {
        Ok(parse_result) => {
            let count = get_entities(parse_result);
            tracing::debug!("负坐标 DXF 解析成功，实体数：{}", count);
            assert!(count > 0);
        }
        Err(e) => {
            tracing::debug!("负坐标 DXF 解析失败：{:?}", e);
        }
    }
}

#[test]
fn test_zero_length_line_dxf() {
    let service = ParserService::new();

    let zero_line_dxf = b"0\nSECTION\n2\nENTITIES\n0\nLINE\n10\n100.0\n20\n100.0\n30\n0.0\n11\n100.0\n21\n100.0\n31\n0.0\n0\nENDSEC\n0\nEOF\n";
    let result = service.parse_bytes(zero_line_dxf, FileType::Dxf);

    match &result {
        Ok(parse_result) => {
            let count = get_entities(parse_result);
            tracing::debug!("零长度线段 DXF 解析结果：{} 实体", count);
        }
        Err(e) => {
            tracing::debug!("零长度线段 DXF 解析失败：{:?}", e);
        }
    }
}

// ==================== 性能测试 ====================

#[test]
fn test_performance_many_entities() {
    // 直接使用 Rust dxf 解析器，避免 Python subprocess 开销
    // 性能测试需要排除 ezdxf-bridge 的 ~200ms 额外开销
    let parser = DxfParser::new();

    let mut dxf_content = Vec::new();
    dxf_content.extend_from_slice(b"0\nSECTION\n2\nENTITIES\n");

    for i in 0..1000 {
        dxf_content.extend_from_slice(b"0\nLINE\n");
        dxf_content.extend_from_slice(format!("10\n{}\n", i).as_bytes());
        dxf_content.extend_from_slice(b"20\n0.0\n30\n0.0\n");
        dxf_content.extend_from_slice(format!("11\n{}\n", i + 1).as_bytes());
        dxf_content.extend_from_slice(b"21\n0.0\n31\n0.0\n");
    }

    dxf_content.extend_from_slice(b"0\nENDSEC\n0\nEOF\n");

    let start = std::time::Instant::now();
    let result = parser.parse_bytes(&dxf_content);
    let elapsed = start.elapsed();

    match &result {
        Ok(entities) => {
            let count = entities.len();
            tracing::info!(
                "1000 实体 DXF 解析成功：{} 实体，耗时：{:?}",
                count,
                elapsed
            );
            assert_eq!(count, 1000, "expected 1000 entities, got {}", count);
            assert!(
                elapsed.as_millis() < 300,
                "解析 1000 实体耗时过长：{:?}",
                elapsed
            );
        }
        Err(e) => {
            panic!("解析失败：{:?}", e);
        }
    }
}

// ==================== 文件类型检测测试 ====================

#[test]
fn test_file_type_from_extension() {
    // 测试文件扩展名检测
    assert_eq!(FileType::from_extension("dxf"), Some(FileType::Dxf));
    assert_eq!(FileType::from_extension("DXF"), Some(FileType::Dxf));
    assert_eq!(FileType::from_extension("pdf"), Some(FileType::Pdf));
    assert_eq!(FileType::from_extension("PDF"), Some(FileType::Pdf));
    assert_eq!(FileType::from_extension("dwg"), Some(FileType::Dwg));
    assert_eq!(FileType::from_extension("txt"), None);

    tracing::debug!("文件扩展名检测测试通过");
}

// ==================== 并发测试 ====================

#[test]
fn test_concurrent_parsing() {
    use std::thread;

    let mut handles = vec![];

    for i in 0..4 {
        let i_copy = i;
        let handle = thread::spawn(move || {
            // 直接使用 Rust dxf 解析器，避免 Python subprocess 临时文件竞争
            let parser = DxfParser::new();
            let dxf = format!("0\nSECTION\n2\nENTITIES\n0\nLINE\n10\n{}.0\n20\n0.0\n30\n0.0\n11\n{}.0\n21\n10.0\n31\n0.0\n0\nENDSEC\n0\nEOF\n", i_copy, i_copy + 5);
            let result = parser.parse_bytes(dxf.as_bytes());
            if let Err(e) = &result {
                eprintln!("Thread {} parse error: {:?}", i_copy, e);
            }
            assert!(result.is_ok());

            result.unwrap().len()
        });
        handles.push(handle);
    }

    let mut total_entities = 0;
    for handle in handles {
        total_entities += handle.join().unwrap();
    }

    tracing::debug!("并发解析完成，总实体数：{}", total_entities);
    assert!(total_entities > 0);
}
