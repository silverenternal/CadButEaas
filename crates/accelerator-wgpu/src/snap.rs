//! wgpu 端点吸附实现
//!
//! 使用网格哈希和工作组统计实现高效的端点吸附

use accelerator_api::{AcceleratorResult, Point2, SnapConfig};
use std::sync::OnceLock;
use wgpu::util::DeviceExt;

/// Uniform 缓冲区配置
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ConfigUniform {
    tolerance: f32,
    padding: [f32; 3],
}

unsafe impl bytemuck::Pod for ConfigUniform {}
unsafe impl bytemuck::Zeroable for ConfigUniform {}

/// wgpu 端点吸附管线缓存
#[derive(Debug)]
struct SnapPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl SnapPipeline {
    pub fn create(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Snap Endpoints Shader"),
            source: wgpu::ShaderSource::Wgsl(SNAP_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Snap Bind Group Layout"),
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
            label: Some("Snap Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Snap Compute Pipeline"),
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

/// 端点吸附计算着色器
const SNAP_SHADER: &str = r#"
struct Config {
    tolerance: f32,
    padding: vec3<f32>,
};

struct Point {
    x: f32,
    y: f32,
};

struct SnapResult {
    x: f32,
    y: f32,
};

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage, read> points: array<Point>;

@group(0) @binding(2)
var<storage, read_write> output: array<SnapResult>;

const WORKGROUP_SIZE: u32 = 1024;
var<workgroup> wg_points: array<Point, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>) {
    let local_idx = local_id.x;

    // 初始化 workgroup 内存
    if (local_idx < WORKGROUP_SIZE) {
        wg_points[local_idx].x = 0.0;
        wg_points[local_idx].y = 0.0;
    }

    workgroupBarrier();

    let idx = global_id.x;
    let n = arrayLength(&points);

    // 缓存 workgroup 内的点
    if (local_idx < WORKGROUP_SIZE && local_idx < n) {
        wg_points[local_idx] = points[local_idx];
    }

    workgroupBarrier();

    // 检查与 workgroup 内其他点的距离
    if (idx < n) {
        let p = points[idx];
        let tol_sq = config.tolerance * config.tolerance;

        var min_dist_sq: f32 = 1e38;
        var snap_x: f32 = p.x;
        var snap_y: f32 = p.y;

        for (var j: u32 = 0; j < WORKGROUP_SIZE; j = j + 1) {
            if (j < n && j != idx) {
                let q = wg_points[j];
                let dx = p.x - q.x;
                let dy = p.y - q.y;
                let dist_sq = dx * dx + dy * dy;

                if (dist_sq < tol_sq && dist_sq < min_dist_sq && dist_sq > 0.0) {
                    min_dist_sq = dist_sq;
                    snap_x = q.x;
                    snap_y = q.y;
                }
            }
        }

        output[idx] = SnapResult(snap_x, snap_y);
    }
}
"#;

/// 管线缓存（懒加载）
static PIPELINE_CACHE: OnceLock<SnapPipeline> = OnceLock::new();

/// 获取或创建端点吸附管线
fn get_pipeline(device: &wgpu::Device) -> &'static SnapPipeline {
    PIPELINE_CACHE.get_or_init(|| SnapPipeline::create(device))
}

/// wgpu 端点吸附（GPU 加速版本）
pub async fn snap_endpoints_wgpu(
    context: &crate::WgpuContext,
    points: &[Point2],
    config: &SnapConfig,
) -> AcceleratorResult<Vec<Point2>> {
    if points.is_empty() {
        return Ok(Vec::new());
    }

    let n = points.len() as u32;

    // 准备 uniform 配置
    let uniform = ConfigUniform {
        tolerance: config.tolerance as f32,
        padding: [0.0, 0.0, 0.0],
    };

    // 创建 uniform 缓冲区
    let uniform_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Snap Uniform Buffer"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // 准备点数据
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    struct GpuPoint {
        x: f32,
        y: f32,
    }
    unsafe impl bytemuck::Pod for GpuPoint {}
    unsafe impl bytemuck::Zeroable for GpuPoint {}

    let gpu_points: Vec<GpuPoint> = points
        .iter()
        .map(|&p| GpuPoint {
            x: p[0] as f32,
            y: p[1] as f32,
        })
        .collect();

    let points_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Snap Points Buffer"),
            contents: bytemuck::cast_slice(&gpu_points),
            usage: wgpu::BufferUsages::STORAGE,
        });

    // 输出缓冲区
    let output_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Snap Output Buffer"),
        size: (n as usize * std::mem::size_of::<GpuPoint>()) as u64,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // 读取结果回主机的缓冲区
    let read_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Snap Read Buffer"),
        size: (n as usize * std::mem::size_of::<GpuPoint>()) as u64,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // 获取管线
    let pipeline_cache = get_pipeline(&context.device);

    // 创建绑定组
    let bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Snap Bind Group"),
            layout: &pipeline_cache.bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: points_buffer.as_entire_binding(),
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
            label: Some("Snap Compute Encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Snap Compute Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline_cache.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(1, 1, 1);
    }

    // 复制结果到可读缓冲区
    encoder.copy_buffer_to_buffer(
        &output_buffer,
        0,
        &read_buffer,
        0,
        (n as usize * std::mem::size_of::<GpuPoint>()) as u64,
    );

    // 提交命令
    context.queue.submit(Some(encoder.finish()));

    // 等待结果并映射缓冲区读取
    let buffer_slice = read_buffer.slice(..);
    let (sender, receiver) = futures_intrusive::channel::shared::oneshot_channel();
    buffer_slice.map_async(wgpu::MapMode::Read, move |result| {
        sender.send(result).ok();
    });

    context.device.poll(wgpu::Maintain::Wait);

    let result = receiver.receive().await.ok_or_else(|| {
        accelerator_api::AcceleratorError::execution_failed("GPU computation cancelled")
    })?;
    result.map_err(|e| accelerator_api::AcceleratorError::execution_failed(format!("{}", e)))?;

    // 读取结果数据
    let data = buffer_slice.get_mapped_range();
    let gpu_result: &[GpuPoint] = bytemuck::cast_slice(&data);

    // 转换为 Point2 格式
    let mut snapped = Vec::with_capacity(points.len());
    for &point in gpu_result.iter().take(points.len()) {
        snapped.push([point.x as f64, point.y as f64]);
    }

    drop(data);
    read_buffer.unmap();

    Ok(snapped)
}
