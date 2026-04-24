//! GPU 渲染器 - 核显优化版
//!
//! ## 设计目标
//!
//! 1. **核显友好**：针对 Intel UHD/Iris Xe 和 AMD Radeon 核显优化
//! 2. **轻量级**：简单的 shader，减少显存占用
//! 3. **兼容性**：支持 WebGL2/GLES2 后端（老旧核显也能用）
//! 4. **自动回退**：GPU 不可用时自动切换到 CPU 渲染
//!
//! ## 核显优化策略
//!
//! | 优化项 | 策略 |
//! |--------|------|
//! | 显存占用 | 共享系统内存，使用小 buffer |
//! | 带宽优化 | 批量合并绘制，减少传输次数 |
//! | Shader 简化 | 简单的顶点/片段着色器 |
//! | 实例化渲染 | 相同图层/颜色的实体批量绘制 |
//!
//! ## 使用示例
//!
//! ```rust
//! use cad_viewer::gpu_renderer::{GpuRenderer, RendererConfig};
//!
//! // 创建渲染器（自动检测 GPU 可用性）
//! let config = RendererConfig::default();
//! let renderer = GpuRenderer::new(config);
//!
//! match renderer {
//!     Ok(renderer) => {
//!         // GPU 渲染可用
//!         println!("GPU 渲染已启用");
//!     }
//!     Err(_) => {
//!         // 自动回退到 CPU 渲染
//!         println!("GPU 不可用，使用 CPU 渲染");
//!     }
//! }
//! ```

use common_types::geometry::Point2;

#[cfg(feature = "gpu")]
use wgpu::util::DeviceExt;

// ============================================================================
// WGSL 着色器代码
// ============================================================================

#[cfg(feature = "gpu")]
const SHADER_CODE: &str = r#"
struct VertexInput {
    @location(0) position: vec2<f32>,
    @location(1) color: vec4<f32>,
    @location(2) line_width: f32,
}

// 实例化数据（每个实例一个）
struct InstanceInput {
    @location(3) model_matrix: mat4x4<f32>,
    @location(7) instance_color: vec4<f32>,
    @location(8) instance_id: u32,
}

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) color: vec4<f32>,
    @location(1) instance_id: u32,
}

struct Uniforms {
    view_matrix: mat4x4<f32>,
    zoom: f32,
    padding: vec3<f32>,
}

@group(0) @binding(0)
var<uniform> uniforms: Uniforms;

@vertex
fn vs_main(model: VertexInput, instance: InstanceInput) -> VertexOutput {
    var out: VertexOutput;

    // 应用模型矩阵和视图变换
    let world_pos = instance.model_matrix * vec4<f32>(model.position, 0.0, 1.0);
    let scale = uniforms.zoom;
    let transformed = world_pos.xy * scale;

    out.clip_position = vec4<f32>(transformed, 0.0, 1.0);
    out.color = model.color * instance.instance_color;
    out.instance_id = instance.instance_id;
    return out;
}

@fragment
fn fs_main(in: VertexOutput) -> @location(0) vec4<f32> {
    return in.color;
}

// 选择缓冲输出（用于拾取）
struct PickOutput {
    @location(0) pick_id: vec4<u32>,
}

@fragment
fn fs_pick(in: VertexOutput) -> PickOutput {
    var out: PickOutput;
    // 将 instance_id 编码为 RGBA
    out.pick_id = vec4<u32>(in.instance_id, 0u, 0u, 0u);
    return out;
}
"#;

// ============================================================================
// 渲染配置
// ============================================================================

/// 渲染器配置
#[derive(Debug, Clone)]
pub struct RendererConfig {
    /// 首选后端（用于调试和强制指定）
    pub preferred_backend: Backend,
    /// 是否启用实例化渲染
    pub enable_instancing: bool,
    /// 是否启用批量合并
    pub enable_batching: bool,
    /// 最大批次大小（核显建议 1000-5000）
    pub max_batch_size: usize,
    /// 是否启用垂直同步
    pub vsync: bool,
    /// 目标帧率（0 = 无限制）
    pub target_fps: u32,
}

impl Default for RendererConfig {
    fn default() -> Self {
        Self {
            preferred_backend: Backend::Auto,
            enable_instancing: true,
            enable_batching: true,
            max_batch_size: 2000, // 核显保守值
            vsync: true,
            target_fps: 60,
        }
    }
}

impl RendererConfig {
    /// 创建核显优化配置
    pub fn for_integrated_gpu() -> Self {
        Self {
            preferred_backend: Backend::Auto,
            enable_instancing: true,
            enable_batching: true,
            max_batch_size: 1000, // 更保守的批次大小
            vsync: true,
            target_fps: 60,
        }
    }

    /// 创建高性能配置（仅适用于有独显的情况）
    pub fn for_discrete_gpu() -> Self {
        Self {
            preferred_backend: Backend::Auto,
            enable_instancing: true,
            enable_batching: true,
            max_batch_size: 10000, // 更大的批次
            vsync: false,          // 关闭垂直同步以降低延迟
            target_fps: 144,
        }
    }

    /// 创建 CPU 回退配置
    pub fn for_cpu_fallback() -> Self {
        Self {
            preferred_backend: Backend::Cpu,
            enable_instancing: false,
            enable_batching: false,
            max_batch_size: 100,
            vsync: true,
            target_fps: 30,
        }
    }
}

/// 渲染后端枚举
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Backend {
    /// 自动选择（优先 GPU，不可用时回退到 CPU）
    #[default]
    Auto,
    /// Vulkan（高性能，需要较新核显）
    Vulkan,
    /// DirectX 12（Windows 10+）
    Dx12,
    /// Metal（macOS）
    Metal,
    /// WebGL2（兼容性最好，老旧核显也能用）
    WebGl2,
    /// GLES2（嵌入式 OpenGL）
    Gles2,
    /// CPU 软件渲染（回退方案）
    Cpu,
}

/// 渲染器统计信息
#[derive(Debug, Clone, Default)]
pub struct RenderStats {
    /// 绘制的实体数量
    pub entities_drawn: usize,
    /// 绘制的批次数量
    pub batches_drawn: usize,
    /// 绘制的顶点数量
    pub vertices_drawn: usize,
    /// 渲染时间（毫秒）
    pub render_time_ms: f32,
    /// GPU 显存占用（MB，仅 GPU 模式）
    pub gpu_memory_mb: f32,
    /// 是否使用 GPU 渲染
    pub is_gpu: bool,
    /// 使用的后端
    pub backend: Backend,
}

impl RenderStats {
    /// 创建 CPU 渲染统计
    pub fn cpu_stats(entities: usize, batches: usize, vertices: usize) -> Self {
        Self {
            entities_drawn: entities,
            batches_drawn: batches,
            vertices_drawn: vertices,
            is_gpu: false,
            backend: Backend::Cpu,
            ..Default::default()
        }
    }

    /// 创建 GPU 渲染统计
    pub fn gpu_stats(
        entities: usize,
        batches: usize,
        vertices: usize,
        render_time: f32,
        memory_mb: f32,
        backend: Backend,
    ) -> Self {
        Self {
            entities_drawn: entities,
            batches_drawn: batches,
            vertices_drawn: vertices,
            render_time_ms: render_time,
            gpu_memory_mb: memory_mb,
            is_gpu: true,
            backend,
        }
    }
}

// ============================================================================
// GPU 渲染器（wgpu 实现）
// ============================================================================

/// 顶点数据（GPU 渲染用）- P0-3 修复：添加 line_width 字段
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[cfg(feature = "gpu")]
struct Vertex {
    position: [f32; 2],
    color: [f32; 4],
    line_width: f32, // P0-3 修复：LOD 线宽
}

#[cfg(feature = "gpu")]
impl Vertex {
    fn desc() -> wgpu::VertexBufferLayout<'static> {
        wgpu::VertexBufferLayout {
            array_stride: std::mem::size_of::<Vertex>() as wgpu::BufferAddress,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &[
                wgpu::VertexAttribute {
                    offset: 0,
                    shader_location: 0,
                    format: wgpu::VertexFormat::Float32x2,
                },
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 2]>() as wgpu::BufferAddress,
                    shader_location: 1,
                    format: wgpu::VertexFormat::Float32x4,
                },
                // P0-3 修复：添加 line_width 属性
                wgpu::VertexAttribute {
                    offset: std::mem::size_of::<[f32; 6]>() as wgpu::BufferAddress,
                    shader_location: 2,
                    format: wgpu::VertexFormat::Float32,
                },
            ],
        }
    }
}

/// 均匀缓冲（用于变换矩阵）
#[repr(C)]
#[derive(Debug, Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
#[cfg(feature = "gpu")]
struct Uniforms {
    view_matrix: [[f32; 4]; 4],
    zoom: f32,
    _padding: [f32; 3],
}

#[cfg(feature = "gpu")]
impl Default for Uniforms {
    fn default() -> Self {
        Self {
            view_matrix: [[1.0; 4]; 4],
            zoom: 1.0,
            _padding: [0.0; 3],
        }
    }
}

/// GPU 渲染器
///
/// 使用 wgpu 实现跨平台 GPU 渲染
/// 支持自动回退到 CPU 渲染
pub struct GpuRenderer {
    /// 渲染配置
    config: RendererConfig,
    /// 实际使用的后端
    backend: Backend,
    /// wgpu 实例（GPU 模式下有效）
    #[cfg(feature = "gpu")]
    instance: Option<wgpu::Instance>,
    /// wgpu 适配器（GPU 模式下有效）
    #[cfg(feature = "gpu")]
    adapter: Option<wgpu::Adapter>,
    /// wgpu 设备（GPU 模式下有效）
    #[cfg(feature = "gpu")]
    device: Option<Arc<wgpu::Device>>,
    /// wgpu 队列（GPU 模式下有效）
    #[cfg(feature = "gpu")]
    queue: Option<Arc<wgpu::Queue>>,
    /// 渲染管线（GPU 模式下有效）
    #[cfg(feature = "gpu")]
    render_pipeline: Option<wgpu::RenderPipeline>,
    /// 顶点缓冲区（GPU 模式下有效）
    #[cfg(feature = "gpu")]
    vertex_buffer: Option<wgpu::Buffer>,
    /// 均匀缓冲（GPU 模式下有效）
    #[cfg(feature = "gpu")]
    uniform_buffer: Option<wgpu::Buffer>,
    /// 绑定组（GPU 模式下有效）
    #[cfg(feature = "gpu")]
    bind_group: Option<wgpu::BindGroup>,
    /// 顶点数量
    #[cfg(feature = "gpu")]
    num_vertices: u32,
    /// 实例化渲染：实例缓冲区（GPU 模式下有效）
    #[cfg(feature = "gpu")]
    instance_buffer: Option<wgpu::Buffer>,
    /// 实例化渲染：实例数量
    #[cfg(feature = "gpu")]
    num_instances: u32,
    /// 选择缓冲：拾取纹理（用于 O(1) 点选）
    #[cfg(feature = "gpu")]
    pick_texture: Option<wgpu::Texture>,
    /// 选择缓冲：拾取纹理视图
    #[cfg(feature = "gpu")]
    pick_texture_view: Option<wgpu::TextureView>,
    /// 选择缓冲：拾取结果读取缓冲
    #[cfg(feature = "gpu")]
    pick_read_buffer: Option<wgpu::Buffer>,
    /// MSAA：多重采样纹理
    #[cfg(feature = "gpu")]
    msaa_texture: Option<wgpu::Texture>,
    /// MSAA：多重采样纹理视图
    #[cfg(feature = "gpu")]
    msaa_texture_view: Option<wgpu::TextureView>,
    /// MSAA 采样数（1, 2, 4）
    #[allow(dead_code)] // 预留用于未来 MSAA 配置扩展
    msaa_sample_count: u32,
    /// 拾取渲染管线（用于选择缓冲）
    #[cfg(feature = "gpu")]
    pick_render_pipeline: Option<wgpu::RenderPipeline>,
    /// 渲染统计
    stats: RenderStats,
}

impl GpuRenderer {
    /// 创建新的渲染器（自动检测 GPU 可用性）
    pub fn new(config: RendererConfig) -> Result<Self, String> {
        let backend = config.preferred_backend;

        // 如果明确指定 CPU，直接返回 CPU 模式
        if backend == Backend::Cpu {
            return Ok(Self::cpu_only(config));
        }

        // 尝试初始化 GPU
        #[cfg(feature = "gpu")]
        {
            match Self::init_gpu(config.clone()) {
                Ok(renderer) => Ok(renderer),
                Err(e) => {
                    log::warn!("GPU 初始化失败：{}，回退到 CPU 渲染", e);
                    Ok(Self::cpu_only(config))
                }
            }
        }

        #[cfg(not(feature = "gpu"))]
        {
            log::info!("GPU 功能未启用，使用 CPU 渲染");
            Ok(Self::cpu_only(config))
        }
    }

    /// 创建纯 CPU 渲染器
    fn cpu_only(config: RendererConfig) -> Self {
        Self {
            config,
            backend: Backend::Cpu,
            #[cfg(feature = "gpu")]
            instance: None,
            #[cfg(feature = "gpu")]
            adapter: None,
            #[cfg(feature = "gpu")]
            device: None,
            #[cfg(feature = "gpu")]
            queue: None,
            #[cfg(feature = "gpu")]
            render_pipeline: None,
            #[cfg(feature = "gpu")]
            vertex_buffer: None,
            #[cfg(feature = "gpu")]
            uniform_buffer: None,
            #[cfg(feature = "gpu")]
            bind_group: None,
            #[cfg(feature = "gpu")]
            num_vertices: 0,
            #[cfg(feature = "gpu")]
            instance_buffer: None,
            #[cfg(feature = "gpu")]
            num_instances: 0,
            #[cfg(feature = "gpu")]
            pick_texture: None,
            #[cfg(feature = "gpu")]
            pick_texture_view: None,
            #[cfg(feature = "gpu")]
            pick_read_buffer: None,
            #[cfg(feature = "gpu")]
            msaa_texture: None,
            #[cfg(feature = "gpu")]
            msaa_texture_view: None,
            msaa_sample_count: 1,
            #[cfg(feature = "gpu")]
            pick_render_pipeline: None,
            stats: RenderStats::cpu_stats(0, 0, 0),
        }
    }

    /// 初始化 GPU 渲染器
    #[cfg(feature = "gpu")]
    fn init_gpu(config: RendererConfig) -> Result<Self, String> {
        // 创建 wgpu 实例
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: Self::backend_to_wgpu(config.preferred_backend),
            ..Default::default()
        });

        // 请求适配器
        let adapter = futures::executor::block_on(async {
            instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::LowPower, // 核显优化：低功耗优先
                    force_fallback_adapter: false,
                    compatible_surface: None,
                })
                .await
        })
        .ok_or("无法获取 GPU 适配器")?;

        // 获取适配器信息
        let adapter_info = adapter.get_info();
        log::info!(
            "使用 GPU: {} (类型：{:?}, 后端：{:?})",
            adapter_info.name,
            adapter_info.device_type,
            adapter_info.backend
        );

        // 检查是否为核显
        let is_integrated = adapter_info.device_type == wgpu::DeviceType::IntegratedGpu;
        if is_integrated {
            log::info!("检测到核显，应用核显优化配置");
        }

        // 请求设备和队列
        let (device, queue) = futures::executor::block_on(async {
            adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        required_features: wgpu::Features::empty(), // 核显：不要求特殊功能
                        required_limits: Self::get_limits_for_integrated_gpu(),
                        label: Some("CAD Viewer Device"),
                    },
                    None,
                )
                .await
        })
        .map_err(|e| format!("无法创建 GPU 设备：{}", e))?;

        let backend = match adapter_info.backend {
            wgpu::Backend::Vulkan => Backend::Vulkan,
            wgpu::Backend::Dx12 => Backend::Dx12,
            wgpu::Backend::Metal => Backend::Metal,
            wgpu::Backend::Gl => Backend::WebGl2,
            _ => Backend::Auto,
        };

        // 创建着色器模块
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Line Shader"),
            source: wgpu::ShaderSource::Wgsl(SHADER_CODE.into()),
        });

        // 创建绑定组布局
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Bind Group Layout"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::VERTEX,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            }],
        });

        // 创建管线布局
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        // 创建渲染管线
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("Line Render Pipeline"),
            layout: Some(&pipeline_layout),
            vertex: wgpu::VertexState {
                module: &shader,
                entry_point: "vs_main",
                buffers: &[Vertex::desc()],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            },
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: "fs_main",
                targets: &[Some(wgpu::ColorTargetState {
                    format: wgpu::TextureFormat::Bgra8UnormSrgb,
                    blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                    write_mask: wgpu::ColorWrites::ALL,
                })],
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            }),
            primitive: wgpu::PrimitiveState {
                topology: wgpu::PrimitiveTopology::LineList,
                strip_index_format: None,
                front_face: wgpu::FrontFace::Ccw,
                cull_mode: None,
                polygon_mode: wgpu::PolygonMode::Fill,
                unclipped_depth: false,
                conservative: false,
            },
            depth_stencil: None,
            multisample: wgpu::MultisampleState {
                count: 1,
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            multiview: None,
        });

        // 创建均匀缓冲区
        let uniform_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Uniform Buffer"),
            contents: bytemuck::cast_slice(&[Uniforms::default()]),
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
        });

        // 创建绑定组
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Bind Group"),
            layout: &bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &uniform_buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        });

        Ok(Self {
            config,
            backend,
            instance: Some(instance),
            adapter: Some(adapter),
            device: Some(Arc::new(device)),
            queue: Some(Arc::new(queue)),
            render_pipeline: Some(render_pipeline),
            vertex_buffer: None,
            uniform_buffer: Some(uniform_buffer),
            bind_group: Some(bind_group),
            num_vertices: 0,
            instance_buffer: None,
            num_instances: 0,
            pick_texture: None,
            pick_texture_view: None,
            pick_read_buffer: None,
            msaa_texture: None,
            msaa_texture_view: None,
            msaa_sample_count: 1,
            pick_render_pipeline: None,
            stats: RenderStats::gpu_stats(0, 0, 0, 0.0, 0.0, backend),
        })
    }

    /// 获取核显优化的限制配置
    #[cfg(feature = "gpu")]
    fn get_limits_for_integrated_gpu() -> wgpu::Limits {
        // 核显保守配置
        wgpu::Limits {
            max_texture_dimension_2d: 4096,              // 限制纹理大小
            max_buffer_size: 256 * 1024 * 1024,          // 256MB 最大 buffer
            max_bind_groups: 2,                          // 减少 bind group 数量
            ..wgpu::Limits::downlevel_webgl2_defaults()  // WebGL2 兼容基线
        }
    }

    /// 将 Backend 转换为 wgpu::Backends
    #[cfg(feature = "gpu")]
    fn backend_to_wgpu(backend: Backend) -> wgpu::Backends {
        match backend {
            Backend::Auto => wgpu::Backends::all(),
            Backend::Vulkan => wgpu::Backends::VULKAN,
            Backend::Dx12 => wgpu::Backends::DX12,
            Backend::Metal => wgpu::Backends::METAL,
            Backend::WebGl2 => wgpu::Backends::GL,
            Backend::Gles2 => wgpu::Backends::GL,
            Backend::Cpu => wgpu::Backends::empty(),
        }
    }

    /// 检查是否使用 GPU 渲染
    pub fn is_gpu(&self) -> bool {
        self.backend != Backend::Cpu
    }

    /// 获取使用的后端
    pub fn backend(&self) -> Backend {
        self.backend
    }

    /// 获取渲染统计
    pub fn stats(&self) -> &RenderStats {
        &self.stats
    }

    /// 获取设备（GPU 模式下）
    #[cfg(feature = "gpu")]
    pub fn device(&self) -> Option<Arc<wgpu::Device>> {
        self.device.clone()
    }

    /// 获取队列（GPU 模式下）
    #[cfg(feature = "gpu")]
    pub fn queue(&self) -> Option<Arc<wgpu::Queue>> {
        self.queue.clone()
    }

    /// 渲染线段到 GPU - P0-3 修复：LOD 动态线宽
    ///
    /// # 参数
    /// - `lines`: 线段列表，每条线段为 (起点，终点，颜色，图层名)
    /// - `zoom`: 缩放级别
    /// - `pan`: 平移偏移
    ///
    /// # 返回
    /// 渲染统计信息
    pub fn render_lines(
        &mut self,
        lines: &[(Point2, Point2, [f32; 4], String)], // P0-3 修复：添加图层名
        #[allow(unused_variables)] zoom: f32,
        #[allow(unused_variables)] pan: egui::Vec2,
    ) -> RenderStats {
        #[cfg(feature = "gpu")]
        {
            if let (Some(device), Some(queue)) = (&self.device, &self.queue) {
                // P0-3 修复：计算 LOD 基础线宽
                let base_line_width = Self::calculate_lod_line_width(zoom);

                // 转换为顶点数据 - P0-3 修复：添加 line_width
                let vertices: Vec<Vertex> = lines
                    .iter()
                    .flat_map(|&(start, end, color, ref layer)| {
                        // P0-3 修复：根据图层语义计算线宽乘数
                        let layer_multiplier =
                            Self::get_layer_width_multiplier(Some(layer.as_str()));
                        let line_width = base_line_width * layer_multiplier;

                        [
                            Vertex {
                                position: [start[0] as f32, start[1] as f32],
                                color,
                                line_width,
                            },
                            Vertex {
                                position: [end[0] as f32, end[1] as f32],
                                color,
                                line_width,
                            },
                        ]
                    })
                    .collect();

                let num_vertices = vertices.len() as u32;

                // 创建或更新顶点缓冲区
                let vertex_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Vertex Buffer"),
                    contents: bytemuck::cast_slice(&vertices),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                });

                // 更新均匀缓冲
                let uniforms = Uniforms {
                    view_matrix: create_view_matrix(zoom, pan),
                    zoom,
                    _padding: [0.0; 3],
                };
                if let Some(ref uniform_buffer) = self.uniform_buffer {
                    queue.write_buffer(uniform_buffer, 0, bytemuck::cast_slice(&[uniforms]));
                }

                // 记录渲染命令
                // 注意：实际渲染需要在 eframe 的 render 回调中进行
                // 这里只准备数据
                self.num_vertices = num_vertices;
                self.vertex_buffer = Some(vertex_buffer);

                self.stats = RenderStats::gpu_stats(
                    lines.len(),
                    (lines.len() / self.config.max_batch_size).max(1),
                    num_vertices as usize,
                    0.0, // 渲染时间待实现
                    0.0, // 显存占用待实现
                    self.backend,
                );
                return self.stats.clone();
            }
        }

        // CPU 回退
        RenderStats::cpu_stats(
            lines.len(),
            (lines.len() / self.config.max_batch_size).max(1),
            lines.len() * 2,
        )
    }

    /// P0-3 修复：根据缩放级别计算 LOD 线宽
    ///
    /// ## LOD 策略
    /// - zoom < 0.2: 0.5px（概览模式，线宽最细）
    /// - zoom 0.2-0.5: 1.0px（缩小模式）
    /// - zoom 0.5-1.0: 2.0px（正常模式）
    /// - zoom 1.0-2.0: 2.5px（放大模式）
    /// - zoom > 2.0: 3.0px（细节模式，线宽最粗）
    #[allow(dead_code)]
    fn calculate_lod_line_width(zoom: f32) -> f32 {
        const MIN_WIDTH: f32 = 0.5;
        const MAX_WIDTH: f32 = 3.0;

        // 使用对数缩放，使线宽变化更平滑
        let log_zoom = zoom.max(0.1).ln();
        let normalized = (log_zoom + 1.0) / 3.0; // 归一化到 0-1 范围
        let width = MIN_WIDTH + (MAX_WIDTH - MIN_WIDTH) * normalized.clamp(0.0, 1.0);

        width.clamp(MIN_WIDTH, MAX_WIDTH)
    }

    /// P0-3 修复：根据图层语义获取线宽乘数
    ///
    /// ## 图层线宽优先级
    /// - 墙体：1.5x（最粗，强调结构）
    /// - 门窗：1.2x（较粗）
    /// - 家具：1.0x（正常）
    /// - 标注：0.8x（较细，避免喧宾夺主）
    /// - 其他：1.0x（默认）
    #[allow(dead_code)]
    fn get_layer_width_multiplier(layer: Option<&str>) -> f32 {
        let layer_upper = layer.unwrap_or("").to_uppercase();

        // 墙体图层 - 最粗
        if layer_upper.contains("WALL")
            || layer_upper.contains("墙体")
            || layer_upper.contains("结构")
            || layer_upper.contains("STRUCT")
        {
            return 1.5;
        }

        // 门窗图层 - 较粗
        if layer_upper.contains("DOOR")
            || layer_upper.contains("门")
            || layer_upper.contains("WINDOW")
            || layer_upper.contains("窗")
        {
            return 1.2;
        }

        // 标注图层 - 较细
        if layer_upper.contains("DIM")
            || layer_upper.contains("标注")
            || layer_upper.contains("TEXT")
            || layer_upper.contains("注释")
        {
            return 0.8;
        }

        // 家具和其他 - 默认
        1.0
    }

    /// 执行渲染（需要在 eframe 的 render 回调中调用）
    #[cfg(feature = "gpu")]
    pub fn draw<'a>(
        &'a self,
        encoder: &mut wgpu::CommandEncoder,
        color_attachment: wgpu::TextureView,
        clear_color: wgpu::Color,
    ) -> Result<(), String> {
        if let (Some(ref vertex_buffer), Some(ref bind_group), Some(ref render_pipeline)) =
            (&self.vertex_buffer, &self.bind_group, &self.render_pipeline)
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("Render Pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_attachment,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(clear_color),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: None,
                timestamp_writes: None,
                occlusion_query_set: None,
            });

            render_pass.set_pipeline(render_pipeline);
            render_pass.set_bind_group(0, bind_group, &[]);
            render_pass.set_vertex_buffer(0, vertex_buffer.slice(..));
            render_pass.draw(0..self.num_vertices, 0..1);
        }

        Ok(())
    }

    /// 创建视图矩阵
    #[cfg(feature = "gpu")]
    fn create_view_matrix(_zoom: f32, _pan: egui::Vec2) -> [[f32; 4]; 4] {
        [
            [_zoom, 0.0, 0.0, _pan.x],
            [0.0, _zoom, 0.0, _pan.y],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ]
    }
}

#[cfg(feature = "gpu")]
fn create_view_matrix(zoom: f32, pan: egui::Vec2) -> [[f32; 4]; 4] {
    [
        [zoom, 0.0, 0.0, pan.x],
        [0.0, zoom, 0.0, pan.y],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

impl GpuRenderer {
    /// 清除渲染
    pub fn clear(&mut self) {
        self.stats = RenderStats {
            backend: self.backend,
            is_gpu: self.is_gpu(),
            ..Default::default()
        };
    }

    /// 调整视口大小
    pub fn resize(&mut self, _width: u32, _height: u32) {
        // GPU 模式下需要重新创建 swap chain
        #[cfg(feature = "gpu")]
        {
            // TODO: 实现 resize 逻辑
        }
    }
}

// ============================================================================
// CPU 渲染回退（egui 原生）
// ============================================================================

/// CPU 渲染器（egui 原生实现）
///
/// 当 GPU 不可用时使用，使用 egui 的原生绘制 API
pub struct CpuRenderer {
    config: RendererConfig,
    stats: RenderStats,
}

impl Default for CpuRenderer {
    fn default() -> Self {
        Self::new(RendererConfig::default())
    }
}

impl CpuRenderer {
    pub fn new(config: RendererConfig) -> Self {
        Self {
            config,
            stats: RenderStats::cpu_stats(0, 0, 0),
        }
    }

    /// 渲染线段到 egui painter - P0-3 修复：LOD 动态线宽
    pub fn render_lines_egui(
        &mut self,
        lines: &[(Point2, Point2, [f32; 4], String)], // P0-3 修复：添加图层名
        painter: &egui::Painter,
        rect: egui::Rect,
        zoom: f32,
        pan: egui::Vec2,
        scene_origin: Point2,
    ) -> RenderStats {
        let mut count = 0;
        let center = rect.center();

        // P0-3 修复：计算 LOD 基础线宽
        let base_line_width = Self::calculate_lod_line_width(zoom);

        for &(start, end, color, ref layer) in lines {
            // P0-3 修复：根据图层语义计算线宽
            let layer_multiplier = Self::get_layer_width_multiplier(Some(layer.as_str()));
            let line_width = base_line_width * layer_multiplier;

            // 坐标变换：world -> screen
            let start_pos = self.world_to_screen(start, center, zoom, pan, scene_origin);
            let end_pos = self.world_to_screen(end, center, zoom, pan, scene_origin);

            // 使用 egui 绘制线段 - P0-3 修复：使用 LOD 线宽
            painter.line_segment(
                [start_pos, end_pos],
                egui::Stroke::new(
                    line_width,
                    Color32::from_rgba_unmultiplied(
                        (color[0] * 255.0) as u8,
                        (color[1] * 255.0) as u8,
                        (color[2] * 255.0) as u8,
                        (color[3] * 255.0) as u8,
                    ),
                ),
            );

            count += 1;
        }

        self.stats = RenderStats::cpu_stats(
            count,
            (count / self.config.max_batch_size).max(1),
            count * 2,
        );
        self.stats.clone()
    }

    /// 坐标变换辅助函数
    fn world_to_screen(
        &self,
        world: Point2,
        center: egui::Pos2,
        zoom: f32,
        pan: egui::Vec2,
        scene_origin: Point2,
    ) -> egui::Pos2 {
        let relative = [world[0] - scene_origin[0], world[1] - scene_origin[1]];
        egui::Pos2::new(
            ((relative[0] * zoom as f64) as f32 + pan.x) + center.x,
            ((-relative[1] * zoom as f64) as f32 + pan.y) + center.y,
        )
    }

    /// 获取统计信息
    pub fn stats(&self) -> &RenderStats {
        &self.stats
    }

    /// P0-3 修复：根据缩放级别计算 LOD 线宽（CpuRenderer 版本）
    fn calculate_lod_line_width(zoom: f32) -> f32 {
        const MIN_WIDTH: f32 = 0.5;
        const MAX_WIDTH: f32 = 3.0;

        // 使用对数缩放，使线宽变化更平滑
        let log_zoom = zoom.max(0.1).ln();
        let normalized = (log_zoom + 1.0) / 3.0; // 归一化到 0-1 范围
        let width = MIN_WIDTH + (MAX_WIDTH - MIN_WIDTH) * normalized.clamp(0.0, 1.0);

        width.clamp(MIN_WIDTH, MAX_WIDTH)
    }

    /// P0-3 修复：根据图层语义获取线宽乘数（CpuRenderer 版本）
    fn get_layer_width_multiplier(layer: Option<&str>) -> f32 {
        let layer_upper = layer.unwrap_or("").to_uppercase();

        // 墙体图层 - 最粗
        if layer_upper.contains("WALL")
            || layer_upper.contains("墙体")
            || layer_upper.contains("结构")
            || layer_upper.contains("STRUCT")
        {
            return 1.5;
        }

        // 门窗图层 - 较粗
        if layer_upper.contains("DOOR")
            || layer_upper.contains("门")
            || layer_upper.contains("WINDOW")
            || layer_upper.contains("窗")
        {
            return 1.2;
        }

        // 标注图层 - 较细
        if layer_upper.contains("DIM")
            || layer_upper.contains("标注")
            || layer_upper.contains("TEXT")
            || layer_upper.contains("注释")
        {
            return 0.8;
        }

        // 家具和其他 - 默认
        1.0
    }
}

use egui::Color32;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_renderer_config() {
        let default_config = RendererConfig::default();
        assert!(default_config.enable_instancing);
        assert!(default_config.enable_batching);
        assert_eq!(default_config.max_batch_size, 2000);

        let integrated_config = RendererConfig::for_integrated_gpu();
        assert_eq!(integrated_config.max_batch_size, 1000);

        let discrete_config = RendererConfig::for_discrete_gpu();
        assert_eq!(discrete_config.max_batch_size, 10000);
        assert!(!discrete_config.vsync);

        let cpu_config = RendererConfig::for_cpu_fallback();
        assert_eq!(cpu_config.preferred_backend, Backend::Cpu);
    }

    #[test]
    fn test_render_stats() {
        let cpu_stats = RenderStats::cpu_stats(100, 1, 200);
        assert!(!cpu_stats.is_gpu);
        assert_eq!(cpu_stats.backend, Backend::Cpu);
        assert_eq!(cpu_stats.entities_drawn, 100);
        assert_eq!(cpu_stats.vertices_drawn, 200);

        let gpu_stats = RenderStats::gpu_stats(100, 1, 200, 2.5, 50.0, Backend::Vulkan);
        assert!(gpu_stats.is_gpu);
        assert_eq!(gpu_stats.backend, Backend::Vulkan);
        assert_eq!(gpu_stats.render_time_ms, 2.5);
        assert_eq!(gpu_stats.gpu_memory_mb, 50.0);
    }

    #[test]
    fn test_cpu_renderer() {
        let renderer = CpuRenderer::default();
        let _lines = [
            ([0.0, 0.0], [10.0, 10.0], [1.0, 0.0, 0.0, 1.0]),
            ([10.0, 10.0], [20.0, 0.0], [0.0, 1.0, 0.0, 1.0]),
        ];

        // 注意：这个测试需要 egui 上下文，实际使用时需要在 eframe 环境中测试
        // 这里只是验证 API 设计
        assert!(renderer.stats().entities_drawn == 0);
    }
}
