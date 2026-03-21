//! GPU 渲染器包装器 - 将 GpuRendererEnhanced 集成到 Renderer trait
//!
//! P11 锐评落实：统一 CPU/GPU 渲染接口，让 GPU 渲染也通过 Renderer trait

#[cfg(feature = "gpu")]
use crate::gpu_renderer_enhanced::{GpuRendererEnhanced, RendererConfig, RenderEntity};
use crate::render::{Renderer, RenderContext};
use crate::state::{SceneState, UIState, Camera2D};

/// GPU 渲染器包装器
///
/// 将 GpuRendererEnhanced 包装成 Renderer trait，实现统一的 CPU/GPU 渲染接口
#[cfg(feature = "gpu")]
pub struct GpuRendererWrapper {
    /// 内部 GPU 渲染器
    inner: GpuRendererEnhanced,
    /// 待渲染的实体列表
    entities: Vec<RenderEntity>,
    /// 渲染配置
    config: RendererConfig,
}

#[cfg(feature = "gpu")]
impl GpuRendererWrapper {
    /// 创建新的 GPU 渲染器包装器
    pub fn new(config: RendererConfig) -> Result<Self, String> {
        let inner = GpuRendererEnhanced::new(config.clone())?;
        Ok(Self {
            inner,
            entities: Vec::new(),
            config,
        })
    }

    /// 设置渲染实体
    pub fn set_entities(&mut self, entities: Vec<RenderEntity>) {
        self.entities = entities;
        self.inner.set_entities(self.entities.clone());
    }

    /// 获取内部 GPU 渲染器引用
    pub fn inner(&self) -> &GpuRendererEnhanced {
        &self.inner
    }

    /// 获取内部 GPU 渲染器可变引用
    pub fn inner_mut(&mut self) -> &mut GpuRendererEnhanced {
        &mut self.inner
    }

    /// 提交 GPU 渲染命令
    ///
    /// 这是 GPU 渲染的特殊步骤，需要在 eframe::Frame 的 wgpu_state 中执行
    #[cfg(feature = "gpu")]
    pub fn submit_gpu_commands(
        &mut self,
        device: &wgpu::Device,
        queue: &wgpu::Queue,
        surface_view: &wgpu::TextureView,
        zoom: f32,
    ) -> Result<(), String> {
        let mut encoder = device.create_command_encoder(
            &wgpu::CommandEncoderDescriptor {
                label: Some("CAD GPU Render Encoder"),
            }
        );

        self.inner.render(
            zoom,
            &mut encoder,
            surface_view,
            wgpu::Color::TRANSPARENT,
        )?;

        queue.submit(Some(encoder.finish()));
        Ok(())
    }

    /// 处理窗口大小变化
    pub fn resize(&mut self, width: u32, height: u32) -> Result<(), String> {
        self.inner.resize(width, height).map_err(|e| e.to_string())
    }
}

#[cfg(feature = "gpu")]
impl Renderer for GpuRendererWrapper {
    fn name(&self) -> &str {
        "GPU Renderer"
    }

    fn begin_frame(&mut self) {
        self.entities.clear();
    }

    fn render_scene(&mut self, _ctx: &mut RenderContext, scene: &SceneState, _camera: &Camera2D) {
        // 准备 GPU 实体
        self.entities = scene.edges
            .iter()
            .enumerate()
            .filter(|(_, edge)| {
                // 过滤不可见的边
                if let Some(visible) = edge.visible {
                    if !visible {
                        return false;
                    }
                } else if let Some(layer) = &edge.layer {
                    if !*scene.layers.visibility.get(layer).unwrap_or(&true) {
                        return false;
                    }
                }
                true
            })
            .map(|(idx, edge)| {
                // 使用默认颜色（简化处理，避免 get_layer_color 依赖）
                let color = [1.0, 1.0, 1.0, 1.0];
                RenderEntity::line(edge.start, edge.end, color, idx as u32)
            })
            .collect();

        self.inner.set_entities(self.entities.clone());
    }

    fn render_ui(&mut self, _ctx: &mut RenderContext, _ui: &UIState, _scene: &SceneState, _camera: &Camera2D) {
        // UI 叠加层用 CPU 渲染（egui）
        // 未来可以用 GPU 渲染
    }

    fn end_frame(&mut self) {
        // GPU 帧结束（实际渲染在 submit_gpu_commands 中）
    }

    fn resize(&mut self, width: u32, height: u32) {
        self.resize(width, height).ok();
    }
}

#[cfg(not(feature = "gpu"))]
// 空实现用于编译通过
pub struct GpuRendererWrapper {
    _marker: std::marker::PhantomData<()>,
}

#[cfg(not(feature = "gpu"))]
impl GpuRendererWrapper {
    pub fn new(_config: RendererConfig) -> Result<Self, String> {
        Err("GPU feature not enabled".to_string())
    }
}

#[cfg(not(feature = "gpu"))]
impl Renderer for GpuRendererWrapper {
    fn name(&self) -> &str {
        "GPU Renderer (disabled)"
    }

    fn begin_frame(&mut self) {}
    fn render_scene(&mut self, _ctx: &mut RenderContext, _scene: &SceneState, _camera: &Camera2D) {}
    fn render_ui(&mut self, _ctx: &mut RenderContext, _ui: &UIState, _scene: &SceneState, _camera: &Camera2D) {}
    fn end_frame(&mut self) {}
    fn resize(&mut self, _width: u32, _height: u32) {}
}
