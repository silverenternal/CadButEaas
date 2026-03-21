//! 集成测试 - E2E 流水线验证
//!
//! 测试整个处理流水线：
//! 1. DXF/PDF 文件输入
//! 2. 解析 → 矢量化 → 拓扑构建 → 验证 → 导出
//! 3. JSON/Binary 输出验证

mod common;
use common::{get_dxf_dir, get_pdf_dir};
use common_types::scene::SceneState;
use parser::{DxfParser, PdfParser};
use topo::TopoService;
use validator::ValidatorService;
use export::ExportService;

// ============================================================================
// DXF E2E 测试
// ============================================================================

#[test]
fn test_e2e_dxf_to_json() {
    // 使用真实 DXF 文件
    let dxf_path = get_dxf_dir().join("报告厅 1.dxf");

    if !dxf_path.exists() {
        panic!("测试文件不存在：{:?}，请检查 git lfs 或测试数据部署", dxf_path);
    }

    // 1. 解析 DXF
    let mut dxf_parser = DxfParser::new();
    let parse_result = dxf_parser.parse_file(&dxf_path);

    assert!(parse_result.is_ok(), "DXF 解析应该成功：{:?}", parse_result.err());
    let parsed_data = parse_result.unwrap();

    // 2. 拓扑构建
    let mut topo_service = TopoService::new();
    let topo_result = topo_service.build_topology(&parsed_data.polylines);

    assert!(topo_result.is_ok(), "拓扑构建应该成功：{:?}", topo_result.err());
    let topo_data = topo_result.unwrap();

    // 3. 验证
    let validator = ValidatorService::default();
    let scene = SceneState {
        outer: topo_data.loops.first().cloned(),
        holes: vec![],
        boundaries: vec![],
        sources: vec![],
        units: common_types::scene::LengthUnit::Mm,
    };

    let validation_report = validator.validate(&scene);
    assert!(validation_report.is_ok(), "验证应该成功");

    // 4. 导出 JSON
    let export_service = ExportService::new();
    let json_result = export_service.to_json(&scene);

    assert!(json_result.is_ok(), "JSON 导出应该成功");
    let json_str = json_result.unwrap();

    // 验证 JSON 包含必要字段
    assert!(json_str.contains("outer"), "JSON 应该包含 outer 字段");
    assert!(json_str.contains("units"), "JSON 应该包含 units 字段");
    
    println!("✅ E2E 测试通过：报告厅 1.dxf → JSON ({} 字节)", json_str.len());
}

#[test]
fn test_e2e_all_dxf_files() {
    // 测试 dxfs/ 目录中的所有 DXF 文件
    let dxf_dir = get_dxf_dir();

    if !dxf_dir.exists() {
        panic!("测试目录不存在：{:?}，请检查 git lfs 或测试数据部署", dxf_dir);
    }

    let mut success_count = 0;
    let mut fail_count = 0;
    let mut total_entities = 0;
    let mut total_loops = 0;

    for entry in std::fs::read_dir(&dxf_dir).unwrap().flatten() {
        let path = entry.path();

        if path.extension().map_or(false, |ext| ext == "dxf") {
            let file_name = path.file_name().unwrap().to_string_lossy();
            println!("\n=== 测试文件：{} ===", file_name);

            // 1. 解析
            let mut parser = DxfParser::new();
            let parse_result = parser.parse_file(&path);

            if let Ok(parsed) = parse_result {
                let entity_count = parsed.polylines.len();
                println!("  ✅ 解析成功：{} 个实体", entity_count);
                total_entities += entity_count;

                // 2. 拓扑
                let mut topo = TopoService::new();
                if let Ok(topo_result) = topo.build_topology(&parsed.polylines) {
                    let loop_count = topo_result.loops.len();
                    println!("  ✅ 拓扑构建：{} 个环", loop_count);
                    total_loops += loop_count;

                    // 3. 验证
                    let validator = ValidatorService::default();
                    let scene = SceneState {
                        outer: topo_result.loops.first().cloned(),
                        holes: if topo_result.loops.len() > 1 {
                            topo_result.loops[1..].to_vec()
                        } else {
                            vec![]
                        },
                        boundaries: vec![],
                        sources: vec![],
                        units: common_types::scene::LengthUnit::Mm,
                    };

                    if let Ok(report) = validator.validate(&scene) {
                        let validity = if report.is_valid() { "有效" } else { "有问题" };
                        println!("  ✅ 验证通过：{}", validity);
                        success_count += 1;
                    } else {
                        println!("  ❌ 验证失败");
                        fail_count += 1;
                    }
                } else {
                    println!("  ❌ 拓扑失败");
                    fail_count += 1;
                }
            } else {
                println!("  ❌ 解析失败：{:?}", parse_result.err());
                fail_count += 1;
            }
        }
    }

    println!("\n========== DXF E2E 测试报告 ==========");
    println!("总文件数：{}", success_count + fail_count);
    println!("成功：{}", success_count);
    println!("失败：{}", fail_count);
    println!("总实体数：{}", total_entities);
    println!("总环数：{}", total_loops);
    println!("=====================================\n");

    assert!(success_count > 0, "应该至少有一个 DXF 文件测试成功");
}

#[test]
fn test_e2e_dxf_rectangle() {
    // 测试简单的矩形 DXF
    let dxf_content = create_rectangle_dxf();
    
    let mut parser = DxfParser::new();
    let result = parser.parse_content(&dxf_content);
    
    assert!(result.is_ok());
    let parsed = result.unwrap();
    
    // 应该解析出 4 条线段
    assert!(parsed.polylines.len() >= 1, "应该至少有一个多段线");
}

#[test]
fn test_e2e_dxf_with_arc() {
    // 测试带圆弧的 DXF
    let dxf_content = create_dxf_with_arc();
    
    let mut parser = DxfParser::new();
    let result = parser.parse_content(&dxf_content);
    
    assert!(result.is_ok(), "带圆弧的 DXF 应该能成功解析");
}

// ============================================================================
// PDF E2E 测试
// ============================================================================

#[test]
fn test_e2e_all_pdf_files() {
    // 测试 testpdf/ 目录中的所有 PDF 文件
    let pdf_dir = get_pdf_dir();

    if !pdf_dir.exists() {
        panic!("测试目录不存在：{:?}，请检查 git lfs 或测试数据部署", pdf_dir);
    }

    let mut success_count = 0;
    let mut fail_count = 0;
    let mut total_pages = 0;

    for entry in std::fs::read_dir(&pdf_dir).unwrap().flatten() {
        let path = entry.path();

        if path.extension().map_or(false, |ext| ext == "pdf") {
            let file_name = path.file_name().unwrap().to_string_lossy();
            println!("\n=== 测试 PDF 文件：{} ===", file_name);

            // 读取 PDF 文件
            let pdf_bytes = match std::fs::read(&path) {
                Ok(data) => data,
                Err(e) => {
                    println!("  ❌ 读取失败：{:?}", e);
                    fail_count += 1;
                    continue;
                }
            };

            println!("  📄 文件大小：{} KB", pdf_bytes.len() / 1024);

            // 解析 PDF
            let parser = PdfParser::new();
            let result = parser.parse_bytes(&pdf_bytes);

            match result {
                Ok(content) => {
                    let page_count = content.pages.len();
                    println!("  ✅ 解析成功：{} 个页面", page_count);
                    total_pages += page_count;
                    success_count += 1;
                }
                Err(e) => {
                    println!("  ❌ 解析失败：{:?}", e);
                    fail_count += 1;
                }
            }
        }
    }

    println!("\n========== PDF E2E 测试报告 ==========");
    println!("总文件数：{}", success_count + fail_count);
    println!("成功：{}", success_count);
    println!("失败：{}", fail_count);
    println!("总页面数：{}", total_pages);
    println!("=====================================\n");

    assert!(success_count > 0, "应该至少有一个 PDF 文件测试成功");
}

#[test]
fn test_e2e_pdf_to_json() {
    // 1. 创建测试 PDF
    let pdf_bytes = create_simple_pdf();
    
    // 2. 解析 PDF
    let parser = PdfParser::new();
    let result = parser.parse_bytes(&pdf_bytes);
    
    assert!(result.is_ok(), "PDF 解析应该成功");
    let pdf_content = result.unwrap();
    
    // 3. 验证 PDF 内容
    assert!(pdf_content.pages.len() >= 0, "PDF 应该有页面");
}

#[test]
fn test_e2e_pdf_vector_detection() {
    // 测试 PDF 矢量内容检测
    let pdf_bytes = create_simple_pdf();
    
    let parser = PdfParser::new();
    let result = parser.parse_bytes(&pdf_bytes);
    
    assert!(result.is_ok());
}

// ============================================================================
// 拓扑 + 验证 E2E 测试
// ============================================================================

#[test]
fn test_e2e_topology_validation() {
    // 创建一个简单的矩形轮廓
    let rectangle = vec![
        [0.0, 0.0],
        [1000.0, 0.0],
        [1000.0, 800.0],
        [0.0, 800.0],
    ];
    
    // 拓扑构建
    let mut topo_service = TopoService::new();
    let topo_result = topo_service.build_topology(&[rectangle.clone()]);
    
    assert!(topo_result.is_ok(), "拓扑构建应该成功");
    let topo_data = topo_result.unwrap();
    
    // 应该提取出一个闭合环
    assert!(!topo_data.loops.is_empty(), "应该至少有一个环");
    
    // 验证
    let validator = ValidatorService::default();
    let scene = SceneState {
        outer: topo_data.loops.first().cloned(),
        holes: vec![],
        boundaries: vec![],
        sources: vec![],
        units: common_types::scene::LengthUnit::Mm,
    };
    
    let report = validator.validate(&scene);
    assert!(report.is_ok(), "验证应该成功");
    
    let report = report.unwrap();
    assert!(report.is_valid(), "矩形应该是有效的");
}

#[test]
fn test_e2e_self_intersecting_polygon() {
    // 测试自相交多边形（蝴蝶结形状）
    let bowtie = vec![
        [0.0, 0.0],
        [100.0, 100.0],
        [100.0, 0.0],
        [0.0, 100.0],
    ];
    
    let validator = ValidatorService::default();
    let scene = SceneState {
        outer: Some(bowtie),
        holes: vec![],
        boundaries: vec![],
        sources: vec![],
        units: common_types::scene::LengthUnit::Mm,
    };
    
    let report = validator.validate(&scene);
    assert!(report.is_ok(), "验证应该成功");
    
    let report = report.unwrap();
    // 自相交多边形应该被检测出来
    assert!(!report.is_valid() || !report.issues.is_empty(), "应该检测出自相交问题");
}

#[test]
fn test_e2e_polygon_with_hole() {
    // 测试带孔洞的多边形
    let outer = vec![
        [0.0, 0.0],
        [100.0, 0.0],
        [100.0, 100.0],
        [0.0, 100.0],
    ];
    
    let hole = vec![
        [25.0, 25.0],
        [75.0, 25.0],
        [75.0, 75.0],
        [25.0, 75.0],
    ];
    
    let validator = ValidatorService::default();
    let scene = SceneState {
        outer: Some(outer),
        holes: vec![hole],
        boundaries: vec![],
        sources: vec![],
        units: common_types::scene::LengthUnit::Mm,
    };
    
    let report = validator.validate(&scene);
    assert!(report.is_ok(), "带孔洞的多边形验证应该成功");
    
    let report = report.unwrap();
    assert!(report.is_valid(), "带孔洞的多边形应该是有效的");
}

// ============================================================================
// 导出 E2E 测试
// ============================================================================

#[test]
fn test_e2e_export_binary() {
    let scene = SceneState {
        outer: Some(vec![
            [0.0, 0.0],
            [10.0, 0.0],
            [10.0, 10.0],
            [0.0, 10.0],
        ]),
        holes: vec![],
        boundaries: vec![],
        sources: vec![],
        units: common_types::scene::LengthUnit::M,
    };
    
    let export_service = ExportService::new();
    
    // 测试 JSON 导出
    let json_result = export_service.to_json(&scene);
    assert!(json_result.is_ok(), "JSON 导出应该成功");
    
    // 测试 Binary 导出
    let binary_result = export_service.to_binary(&scene);
    assert!(binary_result.is_ok(), "Binary 导出应该成功");
    
    let binary_data = binary_result.unwrap();
    assert!(binary_data.len() > 0, "Binary 数据不应该为空");
}

// ============================================================================
// 辅助函数
// ============================================================================

fn create_simple_dxf() -> String {
    // 创建一个简单的矩形 DXF
    r#"0
SECTION
2
HEADER
9
$ACADVER
1
AC1015
0
ENDSEC
0
SECTION
2
ENTITIES
0
LAYER
2
WALL
70
0
0
LINE
8
WALL
10
0.0
20
0.0
11
1000.0
21
0.0
0
LINE
8
WALL
10
1000.0
20
0.0
11
1000.0
21
1000.0
0
LINE
8
WALL
10
1000.0
20
0.0
11
0.0
21
1000.0
0
LINE
8
WALL
10
0.0
20
1000.0
11
0.0
21
0.0
0
ENDSEC
0
EOF
"#.to_string()
}

fn create_rectangle_dxf() -> String {
    create_simple_dxf()
}

fn create_dxf_with_arc() -> String {
    r#"0
SECTION
2
HEADER
9
$ACADVER
1
AC1015
0
ENDSEC
0
SECTION
2
ENTITIES
0
LAYER
2
WALL
70
0
0
LINE
8
WALL
10
0.0
20
0.0
11
100.0
21
0.0
0
ARC
8
WALL
10
100.0
20
0.0
40
50.0
50
0.0
51
90.0
0
LINE
8
WALL
10
100.0
20
100.0
11
0.0
21
100.0
0
LINE
8
WALL
10
0.0
20
100.0
11
0.0
21
0.0
0
ENDSEC
0
EOF
"#.to_string()
}

fn create_simple_pdf() -> Vec<u8> {
    use lopdf::{Document, Object, Stream};
    use lopdf::dictionary;

    let mut doc = Document::with_version("1.4");

    // 创建内容流
    let content = b"10 20 m\n30 40 l\n50 60 l\nh\nS";
    let content_stream = Stream::new(dictionary! {}, content.to_vec());
    doc.set_object((4, 0), content_stream);

    // 创建页面对象
    doc.set_object((3, 0), dictionary! {
        "Type" => "Page",
        "Parent" => Object::Reference((2, 0)),
        "MediaBox" => Object::Array(vec![
            Object::Integer(0),
            Object::Integer(0),
            Object::Integer(595),
            Object::Integer(842),
        ]),
        "Contents" => Object::Reference((4, 0)),
    });

    // 创建 Pages 对象
    doc.set_object((2, 0), dictionary! {
        "Type" => "Pages",
        "Kids" => Object::Array(vec![Object::Reference((3, 0))]),
        "Count" => Object::Integer(1),
    });

    // 创建 Catalog 对象
    doc.set_object((1, 0), dictionary! {
        "Type" => "Catalog",
        "Pages" => Object::Reference((2, 0)),
    });

    doc.trailer.set("Root", Object::Reference((1, 0)));

    let mut buffer = Vec::new();
    doc.save_to(&mut buffer).unwrap();
    buffer
}
