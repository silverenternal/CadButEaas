//! CAD 几何处理系统命令行工具
//!
//! 基于「一切皆服务」(EaaS) 设计哲学的工业级 CAD 几何智能处理系统
//!
//! # 使用示例
//!
//! ```bash
//! # 处理 DXF 文件
//! cad process input.dxf --output scene.json
//!
//! # 使用预设配置
//! cad process input.dxf --profile architectural --output scene.json
//!
//! # 处理 PDF 文件
//! cad process input.pdf --profile scanned --output scene.json
//!
//! # 启动 HTTP 服务
//! cad serve --port 3000
//! ```

use clap::{Parser, Subcommand};
use common_types::{CadError, LengthUnit};
use config::CadConfig;
use orchestrator::service::{OrchestratorService, OrchestratorConfig};
use std::path::PathBuf;
use std::fs;

/// CAD 几何处理系统命令行工具
#[derive(Parser)]
#[command(name = "cad")]
#[command(author = "CAD Team")]
#[command(version = env!("CARGO_PKG_VERSION"))]
#[command(about = "CAD/PDF 图纸识别与边界生成系统", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum Commands {
    /// 处理 CAD/PDF 文件并导出场景
    Process {
        /// 输入文件路径（DXF/PDF）
        #[arg(index = 1)]
        input: PathBuf,

        /// 输出文件路径（JSON/Binary）
        #[arg(short, long, default_value = "scene.json")]
        output: PathBuf,

        /// 使用预设配置（architectural/mechanical/scanned/quick）
        #[arg(short, long)]
        profile: Option<String>,

        /// 配置文件路径（可选，不使用预设时）
        #[arg(short, long)]
        config: Option<PathBuf>,

        /// 端点吸附容差（mm，覆盖配置文件）
        #[arg(long)]
        snap_tolerance: Option<f64>,

        /// 最小线段长度（mm，覆盖配置文件）
        #[arg(long)]
        min_line_length: Option<f64>,

        /// 闭合性检查容差（mm，覆盖配置文件）
        #[arg(long)]
        closure_tolerance: Option<f64>,

        /// 静默模式（减少输出）
        #[arg(short, long)]
        quiet: bool,
    },

    /// 列出可用的预设配置
    ListProfiles,

    /// 显示预设配置详情
    ShowProfile {
        /// 预设配置名称
        #[arg(index = 1)]
        name: String,
    },

    /// 启动 HTTP 服务
    Serve {
        /// 监听端口
        #[arg(short = 'P', long, default_value = "3000")]
        port: u16,

        /// 使用预设配置
        #[arg(short, long)]
        profile: Option<String>,
    },

    /// 验证配置文件
    ValidateConfig {
        /// 配置文件路径
        #[arg(default_value = "cad_config.toml")]
        config: PathBuf,
    },
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Process {
            input,
            output,
            profile,
            config: config_path,
            snap_tolerance,
            min_line_length,
            closure_tolerance,
            quiet,
        } => {
            process_file(
                input,
                output,
                profile,
                config_path,
                snap_tolerance,
                min_line_length,
                closure_tolerance,
                quiet,
            )
            .await?;
        }
        Commands::ListProfiles => {
            list_profiles();
        }
        Commands::ShowProfile { name } => {
            show_profile(&name)?;
        }
        Commands::Serve { port, profile } => {
            serve(port, profile).await?;
        }
        Commands::ValidateConfig { config } => {
            validate_config(&config)?;
        }
    }

    Ok(())
}

/// 处理文件并导出场景
#[allow(clippy::too_many_arguments)]
async fn process_file(
    input: PathBuf,
    output: PathBuf,
    profile: Option<String>,
    config_path: Option<PathBuf>,
    snap_tolerance: Option<f64>,
    min_line_length: Option<f64>,
    closure_tolerance: Option<f64>,
    quiet: bool,
) -> Result<(), Box<dyn std::error::Error>> {
    if !quiet {
        println!("CAD 几何处理系统 v{}", env!("CARGO_PKG_VERSION"));
        println!("==============================");
    }

    // 检查输入文件是否存在
    if !input.exists() {
        return Err(format!("文件不存在：{}", input.display()).into());
    }

    // 加载配置
    let config = load_config(profile, config_path)?;

    // 应用命令行覆盖参数
    let config = apply_overrides(config, snap_tolerance, min_line_length, closure_tolerance)?;

    if !quiet {
        if let Some(profile_name) = &config.profile_name {
            println!("使用预设配置：{}", profile_name);
        } else {
            println!("使用自定义配置");
        }
        println!("输入文件：{}", input.display());
        println!("输出文件：{}", output.display());
        println!();
    }

    // 创建编排服务
    let service = OrchestratorService::new(OrchestratorConfig {
        listen_addr: "127.0.0.1:0".to_string(), // 临时端口
        enable_api: false,
    });

    // 处理文件
    if !quiet {
        println!("正在处理文件...");
    }

    let result = service.process_file(&input).await;

    match result {
        Ok(scene) => {
            if !quiet {
                println!("✅ 处理成功！");
                println!();

                // 显示统计信息
                print_scene_stats(&scene);
            }

            // 导出场景
            let export_config = &config.export;
            let bytes = export_scene(&scene, export_config)?;
            fs::write(&output, &bytes)?;

            if !quiet {
                println!();
                println!("场景已导出至：{}", output.display());
                println!("文件大小：{:.2} KB", bytes.len() as f64 / 1024.0);
            }
        }
        Err(e) => {
            eprintln!("❌ 处理失败：{}", e);

            // 显示恢复建议
            if let CadError::ValidationFailed { issues, .. } = &e {
                eprintln!();
                eprintln!("验证问题：");
                for (i, issue) in issues.iter().enumerate() {
                    eprintln!("  {}. [{}] {}", i + 1, issue.code, issue.message);
                }
            }

            return Err(Box::new(e));
        }
    }

    Ok(())
}

/// 导出场景
fn export_scene(
    scene: &common_types::SceneState,
    export_config: &config::ExportConfig,
) -> Result<Vec<u8>, Box<dyn std::error::Error>> {
    match export_config.format.as_str() {
        "json" => {
            let json = if export_config.json_indent > 0 {
                serde_json::to_string_pretty(scene).map_err(|e| format!("JSON 序列化失败：{}", e))?
            } else {
                serde_json::to_string(scene).map_err(|e| format!("JSON 序列化失败：{}", e))?
            };
            Ok(json.into_bytes())
        }
        "bincode" => {
            let bytes = bincode::serialize(scene).map_err(|e| format!("Bincode 序列化失败：{}", e))?;
            Ok(bytes)
        }
        _ => Err(format!("不支持的导出格式：{}", export_config.format).into()),
    }
}

/// 加载配置（从预设或文件）
fn load_config(
    profile: Option<String>,
    config_path: Option<PathBuf>,
) -> Result<CadConfig, Box<dyn std::error::Error>> {
    // 优先使用预设配置（优先从配置文件读取）
    if let Some(profile_name) = profile {
        // 尝试从配置文件加载预设，失败时回退到硬编码预设
        return CadConfig::from_profile_file(&profile_name)
            .or_else(|_| CadConfig::from_profile(&profile_name))
            .map_err(|e| e.into());
    }

    // 使用配置文件
    if let Some(path) = config_path {
        if !path.exists() {
            return Err(format!("配置文件不存在：{}", path.display()).into());
        }
        let content = fs::read_to_string(&path)?;
        return toml::from_str(&content).map_err(|e| e.into());
    }

    // 使用默认配置
    Ok(CadConfig::default())
}

/// 应用命令行覆盖参数
fn apply_overrides(
    mut config: CadConfig,
    snap_tolerance: Option<f64>,
    min_line_length: Option<f64>,
    closure_tolerance: Option<f64>,
) -> Result<CadConfig, Box<dyn std::error::Error>> {
    if let Some(val) = snap_tolerance {
        if val < 0.0 {
            return Err("snap_tolerance 不能为负数".into());
        }
        config.topology.snap_tolerance_mm = val;
    }

    if let Some(val) = min_line_length {
        if val < 0.0 {
            return Err("min_line_length 不能为负数".into());
        }
        config.topology.min_line_length_mm = val;
    }

    if let Some(val) = closure_tolerance {
        if val < 0.0 {
            return Err("closure_tolerance 不能为负数".into());
        }
        config.validator.closure_tolerance_mm = val;
    }

    Ok(config)
}

/// 列出可用的预设配置
fn list_profiles() {
    println!("可用的预设配置：");
    println!();
    println!("  architectural   - 建筑图纸预设（AutoCAD 导出的建筑平面图）");
    println!("  mechanical      - 机械图纸预设（高精度机械图纸）");
    println!("  scanned         - 扫描图纸预设（线条清晰的扫描版图纸）");
    println!("  quick           - 快速原型预设（低精度要求，快速处理）");
    println!();
    println!("使用方式：cad process input.dxf --profile <预设名称>");
    println!();
    println!("查看详情：cad show-profile <预设名称>");
}

/// 显示预设配置详情
fn show_profile(name: &str) -> Result<(), Box<dyn std::error::Error>> {
    let config = CadConfig::from_profile(name)?;

    println!("预设配置：{}", name);
    println!("==============================");
    println!();

    // 显示拓扑配置
    println!("[topology]");
    println!("  snap_tolerance_mm       = {}", config.topology.snap_tolerance_mm);
    println!("  min_line_length_mm      = {}", config.topology.min_line_length_mm);
    println!("  merge_angle_tolerance_deg = {}", config.topology.merge_angle_tolerance_deg);
    println!("  max_gap_bridge_length_mm = {}", config.topology.max_gap_bridge_length_mm);
    println!();

    // 显示验证器配置
    println!("[validator]");
    println!("  closure_tolerance_mm    = {}", config.validator.closure_tolerance_mm);
    println!("  min_area_m2             = {}", config.validator.min_area_m2);
    println!("  min_edge_length_mm      = {}", config.validator.min_edge_length_mm);
    println!("  min_angle_deg           = {}", config.validator.min_angle_deg);
    println!();

    // 显示导出配置
    println!("[export]");
    println!("  format                  = {}", config.export.format);
    println!("  json_indent             = {}", config.export.json_indent);
    println!("  auto_validate           = {}", config.export.auto_validate);

    Ok(())
}

/// 启动 HTTP 服务
async fn serve(port: u16, profile: Option<String>) -> Result<(), Box<dyn std::error::Error>> {
    println!("CAD 几何处理系统 v{}", env!("CARGO_PKG_VERSION"));
    println!("==============================");

    // 创建编排服务
    let config = OrchestratorConfig {
        listen_addr: format!("0.0.0.0:{}", port),
        enable_api: true,
    };

    if let Some(profile_name) = profile {
        println!("使用预设配置：{}", profile_name);
        // TODO: 将预设配置传递给服务
    }

    let service = OrchestratorService::new(config);

    println!("服务已初始化");
    println!("API 端点：http://localhost:{}/process", port);
    println!("健康检查：http://localhost:{}/health", port);
    println!("==============================");
    println!("按 Ctrl+C 停止服务");

    service.run().await?;

    Ok(())
}

/// 验证配置文件
fn validate_config(path: &PathBuf) -> Result<(), Box<dyn std::error::Error>> {
    if !path.exists() {
        return Err(format!("配置文件不存在：{}", path.display()).into());
    }

    let content = fs::read_to_string(path)?;
    let config: CadConfig = toml::from_str(&content)?;

    // 验证配置合理性
    let mut errors = Vec::new();

    if config.topology.snap_tolerance_mm < 0.0 {
        errors.push("topology.snap_tolerance_mm 不能为负数");
    }

    if config.topology.min_line_length_mm < 0.0 {
        errors.push("topology.min_line_length_mm 不能为负数");
    }

    if config.validator.closure_tolerance_mm < 0.0 {
        errors.push("validator.closure_tolerance_mm 不能为负数");
    }

    if config.validator.min_area_m2 < 0.0 {
        errors.push("validator.min_area_m2 不能为负数");
    }

    if config.validator.min_edge_length_mm < 0.0 {
        errors.push("validator.min_edge_length_mm 不能为负数");
    }

    if config.validator.min_angle_deg < 0.0 || config.validator.min_angle_deg > 180.0 {
        errors.push("validator.min_angle_deg 必须在 0-180 之间");
    }

    if !errors.is_empty() {
        eprintln!("❌ 配置验证失败：");
        for error in errors {
            eprintln!("  - {}", error);
        }
        return Err("配置验证失败".into());
    }

    println!("✅ 配置验证通过！");
    println!();
    println!("配置摘要：");
    println!("  端点吸附容差：{} mm", config.topology.snap_tolerance_mm);
    println!("  最小线段长度：{} mm", config.topology.min_line_length_mm);
    println!("  闭合性检查容差：{} mm", config.validator.closure_tolerance_mm);
    println!("  最小面积：{} m²", config.validator.min_area_m2);

    Ok(())
}

/// 打印场景统计信息
fn print_scene_stats(scene: &common_types::SceneState) {
    println!("场景统计：");

    // 外边界
    if let Some(outer) = &scene.outer {
        println!("  外边界：{} 个点，面积 {:.2} m²",
            outer.points.len(),
            outer.signed_area.abs());
    } else {
        println!("  外边界：无");
    }

    // 孔洞
    println!("  孔洞：{} 个", scene.holes.len());
    for (i, hole) in scene.holes.iter().enumerate() {
        println!("    孔洞 #{}: {} 个点，面积 {:.2} m²",
            i + 1,
            hole.points.len(),
            hole.signed_area.abs());
    }

    // 边界段
    println!("  边界段：{} 个", scene.boundaries.len());

    // 声源
    println!("  声源：{} 个", scene.sources.len());

    // 单位
    println!("  单位：{}", match scene.units {
        LengthUnit::Mm => "毫米 (mm)",
        LengthUnit::Cm => "厘米 (cm)",
        LengthUnit::M => "米 (m)",
        LengthUnit::Inch => "英寸 (inch)",
        LengthUnit::Foot => "英尺 (foot)",
        LengthUnit::Yard => "码 (yard)",
        LengthUnit::Mile => "英里 (mile)",
        LengthUnit::Micron => "微米 (μm)",
        LengthUnit::Kilometer => "千米 (km)",
        LengthUnit::Point => "点 (pt)",
        LengthUnit::Pica => "派卡 (pc)",
        LengthUnit::Unspecified => "未指定",
    });
}
