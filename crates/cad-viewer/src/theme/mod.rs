//! macOS 风格主题系统
//!
//! 设计目标：
//! - 层次感：不同深度的背景色和阴影
//! - 精致感：圆角、细腻的色彩过渡
//! - 专业感：低饱和度、克制的配色

use eframe::egui;
use egui::{Color32, Rounding, Shadow, Stroke};

/// macOS 风格主题
#[derive(Clone, Debug)]
pub struct MacOsTheme {
    /// 窗口背景（纯白/纯黑）
    pub window_bg: Color32,
    /// 面板背景（半透明）
    pub panel_bg: Color32,
    /// 侧边栏背景（稍深）
    pub sidebar_bg: Color32,
    /// 工具栏背景（最浅半透明）
    pub toolbar_bg: Color32,
    /// 强调色（macOS 蓝色）
    pub accent: Color32,
    /// 文本颜色
    pub text: Color32,
    /// 次要文本颜色
    pub text_secondary: Color32,
    /// 边框颜色
    pub border: Color32,
    /// 阴影配置（分层级）
    pub shadow: ShadowConfig,
    /// 圆角配置
    pub rounding: RoundingConfig,
}

/// 阴影配置（分层级）
#[derive(Clone, Debug)]
pub struct ShadowConfig {
    /// 层级 1：轻微阴影（按钮、小卡片）
    pub elevation_1: Shadow,
    /// 层级 2：中等阴影（面板、工具栏）
    pub elevation_2: Shadow,
    /// 层级 3：深度阴影（浮动窗口、模态框）
    pub elevation_3: Shadow,
}

/// 圆角配置
#[derive(Clone, Debug)]
pub struct RoundingConfig {
    /// 小圆角（按钮、输入框）
    pub small: f32,
    /// 中圆角（面板、卡片）
    pub medium: f32,
    /// 大圆角（窗口、模态框）
    pub large: f32,
}

impl MacOsTheme {
    /// 浅色主题
    pub fn light() -> Self {
        Self {
            window_bg: Color32::from_rgb(255, 255, 255),
            panel_bg: Color32::from_rgba_unmultiplied(255, 255, 255, 230),
            sidebar_bg: Color32::from_rgb(245, 245, 247),  // macOS 系统灰
            toolbar_bg: Color32::from_rgba_unmultiplied(255, 255, 255, 200),
            accent: Color32::from_rgb(0, 122, 255),  // macOS 蓝色
            text: Color32::from_rgb(30, 30, 30),
            text_secondary: Color32::from_rgb(142, 142, 147),
            border: Color32::from_rgba_unmultiplied(0, 0, 0, 20),
            shadow: ShadowConfig {
                elevation_1: Shadow {
                    offset: egui::vec2(0.0, 1.0),
                    blur: 3.0,
                    spread: 0.0,
                    color: Color32::from_rgba_unmultiplied(0, 0, 0, 15),
                },
                elevation_2: Shadow {
                    offset: egui::vec2(0.0, 4.0),
                    blur: 12.0,
                    spread: 0.0,
                    color: Color32::from_rgba_unmultiplied(0, 0, 0, 25),
                },
                elevation_3: Shadow {
                    offset: egui::vec2(0.0, 8.0),
                    blur: 24.0,
                    spread: 0.0,
                    color: Color32::from_rgba_unmultiplied(0, 0, 0, 40),
                },
            },
            rounding: RoundingConfig {
                small: 6.0,
                medium: 10.0,
                large: 14.0,
            },
        }
    }

    /// 深色主题
    pub fn dark() -> Self {
        Self {
            window_bg: Color32::from_rgb(30, 30, 30),
            panel_bg: Color32::from_rgba_unmultiplied(44, 44, 46, 230),  // macOS 深灰
            sidebar_bg: Color32::from_rgb(28, 28, 30),
            toolbar_bg: Color32::from_rgba_unmultiplied(44, 44, 46, 200),
            accent: Color32::from_rgb(10, 132, 255),  // macOS 亮蓝
            text: Color32::from_rgb(255, 255, 255),
            text_secondary: Color32::from_rgb(142, 142, 147),
            border: Color32::from_rgba_unmultiplied(255, 255, 255, 30),
            shadow: ShadowConfig {
                elevation_1: Shadow {
                    offset: egui::vec2(0.0, 1.0),
                    blur: 3.0,
                    spread: 0.0,
                    color: Color32::from_rgba_unmultiplied(0, 0, 0, 40),
                },
                elevation_2: Shadow {
                    offset: egui::vec2(0.0, 4.0),
                    blur: 12.0,
                    spread: 0.0,
                    color: Color32::from_rgba_unmultiplied(0, 0, 0, 50),
                },
                elevation_3: Shadow {
                    offset: egui::vec2(0.0, 8.0),
                    blur: 24.0,
                    spread: 0.0,
                    color: Color32::from_rgba_unmultiplied(0, 0, 0, 60),
                },
            },
            rounding: RoundingConfig {
                small: 6.0,
                medium: 10.0,
                large: 14.0,
            },
        }
    }

    /// 应用主题到 egui 上下文
    pub fn apply(&self, ctx: &egui::Context) {
        let mut style = (*ctx.style()).clone();

        // 全局视觉样式
        style.visuals.window_fill = self.window_bg;
        style.visuals.panel_fill = self.panel_bg;
        style.visuals.override_text_color = Some(self.text);
        style.visuals.window_shadow = self.shadow.elevation_3;
        style.visuals.popup_shadow = self.shadow.elevation_2;

        // 交互元素样式
        style.visuals.widgets.noninteractive.bg_fill = self.sidebar_bg;
        style.visuals.widgets.inactive.bg_fill = self.toolbar_bg;
        style.visuals.widgets.hovered.bg_fill = self.accent.linear_multiply(0.15);
        style.visuals.widgets.active.bg_fill = self.accent.linear_multiply(0.3);

        // 圆角 - 全部使用定义的字段
        style.visuals.window_rounding = Rounding::same(self.rounding.large);
        style.visuals.menu_rounding = Rounding::same(self.rounding.medium);
        // P11 新增：按钮和输入框圆角
        style.visuals.widgets.noninteractive.rounding = Rounding::same(self.rounding.small);
        style.visuals.widgets.inactive.rounding = Rounding::same(self.rounding.small);
        style.visuals.widgets.hovered.rounding = Rounding::same(self.rounding.small);
        style.visuals.widgets.active.rounding = Rounding::same(self.rounding.small);

        // 边框颜色 - 使用定义的 border 字段
        style.visuals.widgets.noninteractive.bg_stroke = Stroke::new(1.0, self.border);
        style.visuals.widgets.inactive.bg_stroke = Stroke::new(1.0, self.border);
        style.visuals.widgets.hovered.bg_stroke = Stroke::new(1.0, self.border);

        // 强调色点缀
        style.visuals.selection.bg_fill = self.accent.linear_multiply(0.3);
        style.visuals.selection.stroke = Stroke::new(1.0, self.accent);

        // 超链接颜色
        style.visuals.hyperlink_color = self.accent;

        ctx.set_style(style);
    }
}
