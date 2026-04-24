//! 视觉效果设置面板（P11 新增）
//!
//! 功能：
//! - 一键开启/关闭高级视觉效果
//! - 显示 GPU 等级信息
//! - 支持自定义配置

use crate::components::{Component, ComponentContext};
use eframe::egui;

/// 视觉效果设置面板组件
pub struct VisualSettingsPanel {
    /// 是否展开详细设置
    #[allow(dead_code)]
    expanded: bool,
}

impl Default for VisualSettingsPanel {
    fn default() -> Self {
        Self::new()
    }
}

impl VisualSettingsPanel {
    pub fn new() -> Self {
        Self { expanded: false }
    }
}

#[cfg(feature = "gpu")]
impl Component for VisualSettingsPanel {
    fn name(&self) -> &str {
        "视觉效果设置"
    }

    fn render(&mut self, ctx: &egui::Context, comp_ctx: &mut ComponentContext) {
        use crate::render::{detect_gpu_tier, GpuTier, GpuTierConfig};

        // 克隆主题数据以避免借用冲突
        let theme = comp_ctx.theme.clone();

        // 克隆需要的数据以避免闭包借用冲突
        #[cfg(feature = "gpu")]
        let visual_settings_data = {
            let vs = &comp_ctx.state.ui.visual_settings;
            (
                vs.enable_effects,
                vs.gpu_tier,
                vs.gpu_info.clone(),
                vs.use_custom_config,
                vs.custom_config.clone(),
            )
        };

        #[cfg(not(feature = "gpu"))]
        let visual_settings_data = (
            false,
            crate::render::GpuTier::Unknown,
            crate::render::GpuInfo::default(),
            false,
            crate::render::GpuTierConfig::default(),
        );

        let (mut enable_effects, gpu_tier, mut gpu_info, _use_custom, mut custom_config) =
            visual_settings_data;

        // 如果是首次渲染且 GPU 等级未知，进行检测
        #[cfg(feature = "gpu")]
        if comp_ctx.state.ui.visual_settings.gpu_tier == GpuTier::Unknown {
            let (tier, info) = detect_gpu_tier();
            comp_ctx.state.ui.visual_settings.gpu_tier = tier;
            comp_ctx.state.ui.visual_settings.gpu_info = info.clone();
            comp_ctx.state.ui.visual_settings.enable_effects = tier.enable_glass_effect();
            enable_effects = tier.enable_glass_effect();
            gpu_info = info;
        }

        egui::Window::new("🎨 视觉效果")
            .anchor(egui::Align2::RIGHT_TOP, [-10.0, 40.0])
            .resizable(false)
            .auto_sized()
            .frame(Frame {
                fill: theme.panel_bg,
                rounding: Rounding::same(theme.rounding.medium),
                shadow: theme.shadow.elevation_2,
                inner_margin: Margin::same(12.0),
                stroke: egui::Stroke::new(1.0, theme.border),
                ..Default::default()
            })
            .show(ctx, |ui| {
                ui.set_min_width(280.0);

                // 标题栏
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new("🎨 视觉效果设置")
                            .strong()
                            .color(theme.accent)
                            .size(14.0),
                    );

                    ui.with_layout(egui::Layout::right_to_left(egui::Align::Center), |ui| {
                        if ui.button("✕").clicked() {
                            // 关闭窗口会在外部处理
                        }
                    });
                });

                ui.add_space(8.0);
                ui.separator();
                ui.add_space(8.0);

                // 一键开关
                ui.horizontal(|ui| {
                    ui.label(RichText::new("高级视觉效果:").size(13.0));

                    let checkbox = egui::Checkbox::new(&mut enable_effects, "");
                    ui.add(checkbox);

                    if enable_effects {
                        ui.label(
                            RichText::new("✓ 已启用")
                                .color(Color32::from_rgb(52, 199, 89))
                                .size(12.0),
                        );
                    } else {
                        ui.label(
                            RichText::new("✗ 已关闭")
                                .color(Color32::from_rgb(255, 149, 0))
                                .size(12.0),
                        );
                    }
                });

                ui.add_space(8.0);

                // GPU 信息
                ui.group(|ui| {
                    ui.horizontal(|ui| {
                        ui.label(RichText::new("GPU 信息:").size(12.0));
                        ui.label(
                            RichText::new(format!("{}", gpu_tier))
                                .color(theme.accent)
                                .size(11.0),
                        );
                    });

                    ui.add_space(4.0);

                    if !gpu_info.name.is_empty() {
                        ui.label(
                            RichText::new(format!("  名称：{}", gpu_info.name))
                                .size(10.0)
                                .color(theme.text_secondary),
                        );
                    }
                    ui.label(
                        RichText::new(format!(
                            "  类型：{}",
                            if gpu_info.is_integrated {
                                "集成显卡"
                            } else {
                                "独立显卡"
                            }
                        ))
                        .size(10.0)
                        .color(theme.text_secondary),
                    );
                    ui.label(
                        RichText::new(format!("  后端：{:?}", gpu_info.backend))
                            .size(10.0)
                            .color(theme.text_secondary),
                    );
                });

                ui.add_space(8.0);

                // 当前配置描述
                let config_desc = if _use_custom {
                    format!(
                        "自定义配置：毛玻璃{} | MSAA {}x | 阴影{}",
                        if custom_config.glass_effect {
                            "开"
                        } else {
                            "关"
                        },
                        custom_config.msaa_samples,
                        if custom_config.high_quality_shadows {
                            "高"
                        } else {
                            "标"
                        }
                    )
                } else {
                    format!(
                        "自动检测 ({})：毛玻璃{} | MSAA {}x | 阴影{}",
                        gpu_tier,
                        if gpu_tier.enable_glass_effect() {
                            "开"
                        } else {
                            "关"
                        },
                        gpu_tier.msaa_samples(),
                        if gpu_tier.high_quality_shadows() {
                            "高"
                        } else {
                            "标"
                        }
                    )
                };
                ui.label(
                    RichText::new(config_desc)
                        .size(11.0)
                        .color(theme.text_secondary),
                );

                ui.add_space(8.0);

                // 展开/收起详细设置
                ui.horizontal(|ui| {
                    let button_text = if self.expanded {
                        "▲ 收起详细设置"
                    } else {
                        "▼ 展开详细设置"
                    };
                    if ui.button(button_text).clicked() {
                        self.expanded = !self.expanded;
                    }
                });

                if self.expanded {
                    ui.add_space(8.0);
                    ui.separator();
                    ui.add_space(8.0);

                    // 自定义配置选项
                    let mut use_custom_config = _use_custom;
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.checkbox(&mut use_custom_config, "使用自定义配置");
                        });

                        ui.add_space(8.0);

                        if use_custom_config {
                            // 毛玻璃效果
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("毛玻璃效果:").size(11.0));
                                ui.checkbox(&mut custom_config.glass_effect, "");
                            });

                            // 模糊半径滑块
                            if custom_config.glass_effect {
                                ui.horizontal(|ui| {
                                    ui.label(RichText::new("  模糊半径:").size(10.0));
                                    ui.add(
                                        egui::Slider::new(
                                            &mut custom_config.glass_blur_radius,
                                            0.0..=30.0,
                                        )
                                        .text("px")
                                        .step_by(1.0),
                                    );
                                });
                            }

                            ui.add_space(4.0);

                            // MSAA
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("MSAA:").size(11.0));
                                ui.radio_value(&mut custom_config.msaa_samples, 1, "关闭");
                                ui.radio_value(&mut custom_config.msaa_samples, 2, "2x");
                                ui.radio_value(&mut custom_config.msaa_samples, 4, "4x");
                            });

                            ui.add_space(4.0);

                            // 高质量阴影
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("高质量阴影:").size(11.0));
                                ui.checkbox(&mut custom_config.high_quality_shadows, "");
                            });

                            ui.add_space(4.0);

                            // 面板透明度
                            ui.horizontal(|ui| {
                                ui.label(RichText::new("面板透明度:").size(11.0));
                                ui.add(
                                    egui::Slider::new(
                                        &mut custom_config.panel_transparency,
                                        0.5..=1.0,
                                    )
                                    .text("")
                                    .step_by(0.05),
                                );
                            });
                        } else {
                            // 推荐配置提示
                            ui.label(
                                RichText::new("  使用推荐配置")
                                    .size(11.0)
                                    .color(theme.text_secondary),
                            );
                            ui.label(
                                RichText::new("  根据 GPU 等级自动优化")
                                    .size(10.0)
                                    .color(theme.text_secondary),
                            );
                        }
                    });

                    ui.add_space(8.0);

                    // 预设按钮
                    ui.horizontal(|ui| {
                        if ui.button("🚀 高性能").clicked() {
                            custom_config = GpuTierConfig::low_quality();
                            use_custom_config = true;
                        }
                        if ui.button("⚖️ 平衡").clicked() {
                            custom_config = GpuTierConfig::medium_quality();
                            use_custom_config = true;
                        }
                        if ui.button("🎨 高质量").clicked() {
                            custom_config = GpuTierConfig::high_quality();
                            use_custom_config = true;
                        }
                    });

                    ui.add_space(8.0);

                    // 应用按钮
                    if ui.button("💾 应用配置").clicked() {
                        // 应用配置到毛玻璃渲染器
                        #[cfg(feature = "gpu")]
                        if let Some(glass_renderer) = comp_ctx.glass_renderer_mut() {
                            glass_renderer
                                .set_enabled(enable_effects && custom_config.glass_effect);
                            glass_renderer.set_blur_radius(custom_config.glass_blur_radius);
                        }

                        // 更新状态
                        comp_ctx.state.ui.visual_settings.custom_config = custom_config.clone();
                        comp_ctx.state.ui.visual_settings.use_custom_config = use_custom_config;
                        comp_ctx.state.ui.visual_settings.enable_effects = enable_effects;

                        // 显示提示
                        comp_ctx.state.add_log("视觉效果配置已应用");
                    }
                }

                // 性能提示
                if enable_effects && gpu_tier == GpuTier::Low {
                    ui.add_space(8.0);
                    ui.group(|ui| {
                        ui.horizontal(|ui| {
                            ui.label(
                                RichText::new("⚠️ 性能提示:")
                                    .color(Color32::from_rgb(255, 149, 0))
                                    .size(11.0),
                            );
                        });
                        ui.label(
                            RichText::new("  检测到集成显卡，启用高级效果")
                                .size(10.0)
                                .color(theme.text_secondary),
                        );
                        ui.label(
                            RichText::new("  可能导致帧率下降，建议降低配置")
                                .size(10.0)
                                .color(theme.text_secondary),
                        );
                    });
                }
            });
    }
}

#[cfg(not(feature = "gpu"))]
impl Component for VisualSettingsPanel {
    fn name(&self) -> &str {
        "视觉效果设置"
    }

    fn render(&mut self, ctx: &egui::Context, _comp_ctx: &mut ComponentContext) {
        // 非 GPU 特性版本显示简化信息
        egui::Window::new("🎨 视觉效果")
            .anchor(egui::Align2::RIGHT_TOP, [-10.0, 40.0])
            .resizable(false)
            .auto_sized()
            .show(ctx, |ui| {
                ui.label("GPU 特性未启用");
                ui.label("请使用 --features gpu 编译以启用高级视觉效果");
            });
    }
}
