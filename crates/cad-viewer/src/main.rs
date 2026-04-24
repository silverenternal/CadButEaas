//! CAD Viewer - egui 前端应用
//!
//! # 功能清单
//! - Canvas 渲染（线段绘制/缩放/平移）
//! - 鼠标点选边（射线检测 + 容差）
//! - 实时高亮追踪路径
//! - 语义标注 ComboBox
//! - 文件上传/导出
//! - 缺口可视化
//! - 圈选工具

mod api;
mod app;
mod canvas;
mod components;
pub mod coordinate_compensator; // P0-3 新增：坐标精度补偿器
pub mod gpu_renderer; // P0-6 新增：GPU 渲染器（核显优化）
pub mod gpu_renderer_enhanced;
pub mod lod_selector; // P2-1 新增：LOD 动态选择器
mod panels;
mod render;
mod state;
mod theme; // P11 新增：macOS 风格主题系统
pub mod viewport_culler; // P1-3 新增：视口裁剪器 // P1-1 新增：GPU 渲染器增强版（实例化/选择缓冲/MSAA）

use app::CadApp;

#[tokio::main]
async fn main() -> eframe::Result<()> {
    env_logger::Builder::from_env(env_logger::Env::default().default_filter_or("info")).init();

    let native_options = eframe::NativeOptions {
        viewport: egui::ViewportBuilder::default()
            .with_inner_size([1280.0, 800.0])
            .with_min_inner_size([800.0, 600.0])
            .with_title("CAD Viewer - 几何智能处理系统"),
        ..Default::default()
    };

    eframe::run_native(
        "CAD Viewer",
        native_options,
        Box::new(|cc| {
            // 设置自定义字体和样式
            setup_custom_fonts(&cc.egui_ctx);
            Ok(Box::new(CadApp::new(cc)))
        }),
    )
}

fn setup_custom_fonts(ctx: &egui::Context) {
    let fonts = egui::FontDefinitions::default();

    // 注册中文字体（如果系统有）
    #[cfg(target_os = "windows")]
    {
        if let Ok(font_data) = std::fs::read("C:/Windows/Fonts/simsun.ttc") {
            fonts
                .font_data
                .insert("SimSun".to_owned(), egui::FontData::from_owned(font_data));
            fonts
                .families
                .entry(egui::FontFamily::Proportional)
                .or_default()
                .insert(0, "SimSun".to_owned());
        }
    }

    ctx.set_fonts(fonts);
}
