//! 甲方演示脚本 - E2E 流水线演示
//!
//! 运行方式：cargo run --example e2e_demo --release

use parser::DxfParser;
use topo::TopoService;
use validator::ValidatorService;
use export::ExportService;
use std::path::PathBuf;

fn main() {
    println!("🎯 CAD 几何智能处理系统 - 甲方演示\n");
    println!("═══════════════════════════════════════════════════════\n");

    // 演示 1：DXF 解析 → 拓扑 → 验证 → 导出
    demo_dxf_pipeline();

    // 演示 2：PDF 解析
    demo_pdf_parsing();

    // 演示 3：交互功能演示
    demo_interaction();

    println!("\n═══════════════════════════════════════════════════════");
    println!("✅ 演示完成！\n");
}

fn demo_dxf_pipeline() {
    println!("📁 演示 1: DXF 完整流水线\n");

    let dxf_path = PathBuf::from("dxfs").join("报告厅 1.dxf");
    
    if !dxf_path.exists() {
        println!("  ⚠️  文件不存在：{:?}", dxf_path);
        println!("  跳过演示\n");
        return;
    }

    println!("  输入文件：{:?}", dxf_path);

    // 1. 解析
    print!("  [1/4] 解析 DXF ... ");
    let mut parser = DxfParser::new();
    let result = parser.parse_file(&dxf_path);
    
    match result {
        Ok(entities) => {
            println!("✅ {} 个实体", entities.len());
            
            // 统计实体类型
            let mut line_count = 0;
            let mut arc_count = 0;
            let mut polyline_count = 0;
            
            for entity in &entities {
                match entity {
                    common_types::RawEntity::Line { .. } => line_count += 1,
                    common_types::RawEntity::Arc { .. } => arc_count += 1,
                    common_types::RawEntity::Polyline { .. } => polyline_count += 1,
                    _ => {}
                }
            }
            
            println!("         - LINE: {}", line_count);
            println!("         - ARC: {}", arc_count);
            println!("         - POLYLINE: {}", polyline_count);
        }
        Err(e) => {
            println!("❌ {:?}", e);
            return;
        }
    }

    // 2. 拓扑构建
    print!("  [2/4] 拓扑构建 ... ");
    let parsed = match parser.parse_file(&dxf_path) {
        Ok(p) => p,
        Err(e) => {
            println!("❌ {:?}", e);
            return;
        }
    };
    
    let mut topo_service = TopoService::new();
    let topo_result = topo_service.build_topology(&parsed.polylines);
    
    match topo_result {
        Ok(topo_data) => {
            println!("✅ {} 个环", topo_data.loops.len());
            
            if !topo_data.loops.is_empty() {
                let first_loop = &topo_data.loops[0];
                println!("         - 首环点数：{}", first_loop.len());
            }
        }
        Err(e) => {
            println!("❌ {:?}", e);
            return;
        }
    }

    // 3. 验证
    print!("  [3/4] 几何验证 ... ");
    let topo_data = match topo_service.build_topology(&parsed.polylines) {
        Ok(t) => t,
        Err(e) => {
            println!("❌ {:?}", e);
            return;
        }
    };
    
    let validator = ValidatorService::default();
    let scene = common_types::scene::SceneState {
        outer: topo_data.loops.first().cloned(),
        holes: vec![],
        boundaries: vec![],
        sources: vec![],
        units: common_types::scene::LengthUnit::Mm,
    };

    let report = validator.validate(&scene);
    
    match report {
        Ok(validation_report) => {
            if validation_report.is_valid() {
                println!("✅ 几何有效");
            } else {
                println!("⚠️  发现 {} 个问题", validation_report.issues.len());
                for issue in &validation_report.issues.iter().take(3) {
                    println!("         - {:?}", issue);
                }
            }
        }
        Err(e) => {
            println!("❌ {:?}", e);
            return;
        }
    }

    // 4. 导出
    print!("  [4/4] JSON 导出 ... ");
    let export_service = ExportService::new();
    let json_result = export_service.to_json(&scene);

    match json_result {
        Ok(json_str) => {
            println!("✅ {} 字节", json_str.len());
            
            // 显示 JSON 预览
            let preview = if json_str.len() > 100 {
                &json_str[..100]
            } else {
                &json_str
            };
            println!("         预览：{}...", preview);
        }
        Err(e) => {
            println!("❌ {:?}", e);
            return;
        }
    }

    println!("\n  ✅ DXF 流水线演示完成\n");
}

fn demo_pdf_parsing() {
    println!("\n📁 演示 2: PDF 解析\n");

    let pdf_path = PathBuf::from("testpdf").join("example.pdf");
    
    // 尝试找一个存在的 PDF 文件
    let pdf_dir = PathBuf::from("testpdf");
    let pdf_path = if pdf_dir.exists() {
        std::fs::read_dir(&pdf_dir)
            .ok()
            .and_then(|mut entries| entries.next())
            .and_then(|e| e.ok())
            .map(|e| e.path())
    } else {
        None
    };

    let pdf_path = match pdf_path {
        Some(p) if p.extension().map_or(false, |ext| ext == "pdf") => p,
        _ => {
            println!("  ⚠️  没有找到 PDF 文件");
            println!("  跳过演示\n");
            return;
        }
    };

    println!("  输入文件：{:?}", pdf_path);

    // 读取 PDF
    print!("  [1/2] 读取 PDF ... ");
    let pdf_bytes = match std::fs::read(&pdf_path) {
        Ok(data) => {
            println!("✅ {} KB", data.len() / 1024);
            data
        }
        Err(e) => {
            println!("❌ {:?}", e);
            return;
        }
    };

    // 解析 PDF
    print!("  [2/2] 解析矢量内容 ... ");
    let parser = parser::PdfParser::new();
    let result = parser.parse_bytes(&pdf_bytes);

    match result {
        Ok(content) => {
            println!("✅ {} 个页面", content.pages.len());
            
            // 显示页面信息
            for (i, page) in content.pages.iter().take(3).enumerate() {
                println!("         - 页面 {}: {} 个图元", i + 1, page.primitives.len());
            }
        }
        Err(e) => {
            println!("❌ {:?}", e);
            return;
        }
    }

    println!("\n  ✅ PDF 解析演示完成\n");
}

fn demo_interaction() {
    println!("\n📁 演示 3: 交互功能\n");

    println!("  功能说明:");
    println!("  1. 用户点选一段墙线");
    println!("  2. 系统沿拓扑自动追踪形成闭合环");
    println!("  3. 实时高亮显示追踪路径");
    println!("\n  ⚠️  Web UI 原型开发中...\n");

    // 演示拓扑追踪逻辑
    println!("  演示拓扑追踪算法:");
    
    // 创建一个简单的矩形
    let rectangle = vec![
        [0.0, 0.0],
        [1000.0, 0.0],
        [1000.0, 800.0],
        [0.0, 800.0],
    ];

    print!("    构建拓扑 ... ");
    let mut topo_service = TopoService::new();
    let topo_result = topo_service.build_topology(&[rectangle]);

    match topo_result {
        Ok(topo_data) => {
            println!("✅");
            println!("    环数量：{}", topo_data.loops.len());
            
            if let Some(loop_data) = topo_data.loops.first() {
                println!("    首环点数：{}", loop_data.len());
                println!("    首环坐标:");
                for (i, point) in loop_data.iter().enumerate() {
                    println!("      [{}] = ({:.1}, {:.1})", i, point[0], point[1]);
                }
            }
        }
        Err(e) => {
            println!("❌ {:?}", e);
        }
    }

    println!("\n  ✅ 交互功能演示完成\n");
}
