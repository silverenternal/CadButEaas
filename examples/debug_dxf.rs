use std::fs;
use std::path::Path;

fn main() {
    let dxf_dir = Path::new("dxfs");
    
    // 列出所有 DXF 文件
    println!("DXF 文件列表：");
    if let Ok(entries) = fs::read_dir(dxf_dir) {
        for entry in entries.flatten() {
            let file_name = entry.file_name().to_string_lossy().to_string();
            if file_name.ends_with(".dxf") {
                println!("  - {}", file_name);
            }
        }
    }
    
    // 读取 dimension_test.dxf 并验证
    let dim_test = dxf_dir.join("dimension_test.dxf");
    if dim_test.exists() {
        println!("\ndimension_test.dxf 存在");
        if let Ok(content) = fs::read_to_string(&dim_test) {
            let lines: Vec<&str> = content.lines().take(50).collect();
            println!("前 50 行：");
            for (i, line) in lines.iter().enumerate() {
                println!("{:3}: {}", i + 1, line);
            }
        }
    } else {
        println!("\ndimension_test.dxf 不存在");
    }
    
    // 尝试解析文件
    use parser::DxfParser;
    let parser = DxfParser::new();
    
    println!("\n尝试解析 dimension_test.dxf:");
    if let Ok(entities) = parser.parse_file(&dim_test) {
        println!("解析到 {} 个实体", entities.len());
        for entity in entities.iter().take(10) {
            println!("  - {:?}", entity);
        }
    } else {
        println!("解析失败");
    }
}
