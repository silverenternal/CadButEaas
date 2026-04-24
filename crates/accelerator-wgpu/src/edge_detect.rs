//! wgpu 边缘检测实现
//!
//! 使用计算着色器实现 Sobel 边缘检测

use accelerator_api::{AcceleratorResult, EdgeDetectConfig, EdgeMap, Image};
use std::sync::OnceLock;
use wgpu::util::DeviceExt;

/// Uniform 缓冲区配置布局
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ConfigUniform {
    low_threshold: f32,
    high_threshold: f32,
    width: u32,
    height: u32,
    padding: u32, // 对齐到 16 字节
}

unsafe impl bytemuck::Pod for ConfigUniform {}
unsafe impl bytemuck::Zeroable for ConfigUniform {}

/// wgpu 边缘检测管线缓存
#[derive(Debug)]
struct EdgeDetectPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl EdgeDetectPipeline {
    fn create(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Edge Detect Shader"),
            source: wgpu::ShaderSource::Wgsl(EDGE_DETECT_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Edge Detect Bind Group Layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: false },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Edge Detect Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Edge Detect Compute Pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: "main",
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        Self {
            pipeline,
            bind_group_layout,
        }
    }
}

/// wgpu 边缘检测着色器 WGSL 代码
pub const EDGE_DETECT_SHADER: &str = r#"
struct Config {
    low_threshold: f32,
    high_threshold: f32,
    width: u32,
    height: u32,
    padding: u32,
};

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage, read> input_image: array<f32>;

@group(0) @binding(2)
var<storage, read_write> output_edges: array<f32>;

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= config.width || y >= config.height) {
        return;
    }

    let idx = y * config.width + x;

    // 初始化输出为非边缘（1.0）
    output_edges[idx] = 1.0;

    // 边界像素直接跳过处理，保持非边缘
    if (x < 1 || x >= config.width - 1 || y < 1 || y >= config.height - 1) {
        return;
    }

    // Sobel 算子
    var gx: f32 = 0.0;
    var gy: f32 = 0.0;

    // 3x3 卷积 Sobel 核
    // X: [-1 0 1] [-1 0 1] [-1 0 1]
    // Y: [-1 -1 -1] [0 0 0] [1 1 1]

    let pixels: [f32; 9] = [
        input_image[(y - 1) * config.width + (x - 1)],
        input_image[(y - 1) * config.width + x],
        input_image[(y - 1) * config.width + (x + 1)],
        input_image[y * config.width + (x - 1)],
        input_image[y * config.width + x],
        input_image[y * config.width + (x + 1)],
        input_image[(y + 1) * config.width + (x - 1)],
        input_image[(y + 1) * config.width + x],
        input_image[(y + 1) * config.width + (x + 1)],
    ];

    // Sobel X 卷积
    gx = -pixels[0] + pixels[2] - pixels[3] + pixels[5] - pixels[6] + pixels[8];
    // Sobel Y 卷积
    gy = -pixels[0] - pixels[1] - pixels[2] + pixels[6] + pixels[7] + pixels[8];

    let magnitude = sqrt(gx * gx + gy * gy);

    // Sobel 梯度幅值大于阈值则标记为边缘
    if (magnitude > config.low_threshold) {
        output_edges[idx] = 0.0; // 边缘
    }
}
"#;

/// 管线缓存（懒加载）
static PIPELINE_CACHE: OnceLock<EdgeDetectPipeline> = OnceLock::new();

/// 获取或创建边缘检测管线
fn get_pipeline(device: &wgpu::Device) -> &'static EdgeDetectPipeline {
    PIPELINE_CACHE.get_or_init(|| EdgeDetectPipeline::create(device))
}

/// wgpu 边缘检测（GPU 加速版本）
pub async fn detect_edges_wgpu(
    context: &crate::WgpuContext,
    image: &Image,
    config: &EdgeDetectConfig,
) -> AcceleratorResult<EdgeMap> {
    let width = image.width;
    let height = image.height;
    let pixel_count = (width * height) as usize;

    // 准备 uniform 配置
    let uniform = ConfigUniform {
        low_threshold: config.low_threshold as f32,
        high_threshold: config.high_threshold as f32,
        width,
        height,
        padding: 0,
    };

    // 创建缓冲区
    let uniform_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Edge Detect Uniform Buffer"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // 准备输入图像数据（转换为 f32 数组）
    let input_data: Vec<f32> = image.data.iter().map(|&p| p as f32 / 255.0).collect();

    let input_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Edge Detect Input Buffer"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

    // 输出缓冲区
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Edge Detect Output Buffer"),
        size: (pixel_count * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // 读取结果回主机的缓冲区
    let read_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Edge Detect Read Buffer"),
        size: (pixel_count * std::mem::size_of::<f32>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // 获取管线
    let pipeline_cache = get_pipeline(&context.device);

    // 创建绑定组
    let bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Edge Detect Bind Group"),
            layout: &pipeline_cache.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: input_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: output_buffer.as_entire_binding(),
                },
            ],
        });

    // 编码计算命令
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Edge Detect Compute Encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Edge Detect Compute Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline_cache.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);

        // 计算工作组数量
        let workgroup_x = width.div_ceil(8);
        let workgroup_y = height.div_ceil(8);
        pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
    }

    // 复制结果到可读缓冲区
    encoder.copy_buffer_to_buffer(&output_buffer, 0, &read_buffer, 0, output_buffer.size());

    // 提交命令
    context.queue.submit(Some(encoder.finish()));

    // 等待结果并映射缓冲区读取
    let buffer_slice = read_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).ok();
    });

    // 轮询设备直到完成
    context.device.poll(wgpu::Maintain::Wait);

    let result = receiver.receive().await.ok_or_else(|| {
        accelerator_api::AcceleratorError::execution_failed("GPU computation cancelled")
    })?;
    result.map_err(|e| accelerator_api::AcceleratorError::execution_failed(format!("{}", e)))?;

    // 读取结果数据
    let data = buffer_slice.get_mapped_range();
    let result_f32: &[f32] = bytemuck::cast_slice(&data);

    // 转换为 EdgeMap 格式（0 = 边缘，255 = 非边缘）
    let mut edge_data = Vec::with_capacity(pixel_count);
    for &val in result_f32 {
        // val: 0.0 = 边缘 → 0，1.0 = 非边缘 → 255
        edge_data.push((val * 255.0).round() as u8);
    }

    drop(data);
    read_buffer.unmap();

    Ok(EdgeMap {
        width,
        height,
        data: edge_data,
    })
}
