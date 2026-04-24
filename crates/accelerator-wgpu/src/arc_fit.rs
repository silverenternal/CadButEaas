//! wgpu 圆弧拟合实现
//!
//! 使用 Kåsa 算法进行圆弧拟合的 GPU 加速版本
//! GPU 负责计算各项累加和，CPU 负责求解线性方程组

use accelerator_api::{
    AcceleratorError, AcceleratorResult, Arc as AcceleratorArc, ArcFitConfig, Point2,
};
use std::sync::OnceLock;
use wgpu::util::DeviceExt;

/// Uniform 缓冲区配置
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ConfigUniform {
    num_points: u32,
    padding: [u32; 3],
}

unsafe impl bytemuck::Pod for ConfigUniform {}
unsafe impl bytemuck::Zeroable for ConfigUniform {}

/// 输出累加和结构
#[repr(C)]
#[derive(Debug, Clone, Copy, Default)]
struct Accumulators {
    sum_x: f64,
    sum_y: f64,
    sum_xx: f64,
    sum_yy: f64,
    sum_xy: f64,
    sum_xxx: f64,
    sum_yyy: f64,
    sum_xyy: f64,
    sum_xxy: f64,
}

unsafe impl bytemuck::Pod for Accumulators {}
unsafe impl bytemuck::Zeroable for Accumulators {}

/// wgpu 圆弧拟合管线缓存
#[derive(Debug)]
struct ArcFitPipeline {
    pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

impl ArcFitPipeline {
    pub fn create(device: &wgpu::Device) -> Self {
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Arc Fit Shader"),
            source: wgpu::ShaderSource::Wgsl(ARC_FIT_SHADER.into()),
        });

        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Arc Fit Bind Group Layout"),
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
            label: Some("Arc Fit Pipeline Layout"),
            bind_group_layouts: &[&bind_group_layout],
            push_constant_ranges: &[],
        });

        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Arc Fit Compute Pipeline"),
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

/// 圆弧拟合计算着色器
const ARC_FIT_SHADER: &str = r#"
struct Config {
    num_points: u32,
    padding: vec3<u32>,
};

struct Point {
    x: f64,
    y: f64,
};

struct Accumulators {
    sum_x: f64,
    sum_y: f64,
    sum_xx: f64,
    sum_yy: f64,
    sum_xy: f64,
    sum_xxx: f64,
    sum_yyy: f64,
    sum_xyy: f64,
    sum_xxy: f64,
};

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage, read> points: array<Point>;

@group(0) @binding(2)
var<storage, read_write> accumulators: array<Accumulators>;

// 使用 workgroup 内存进行归约
const WORKGROUP_SIZE: u32 = 256;
var<workgroup> wg_sum_x: array<f64, WORKGROUP_SIZE>;
var<workgroup> wg_sum_y: array<f64, WORKGROUP_SIZE>;
var<workgroup> wg_sum_xx: array<f64, WORKGROUP_SIZE>;
var<workgroup> wg_sum_yy: array<f64, WORKGROUP_SIZE>;
var<workgroup> wg_sum_xy: array<f64, WORKGROUP_SIZE>;
var<workgroup> wg_sum_xxx: array<f64, WORKGROUP_SIZE>;
var<workgroup> wg_sum_yyy: array<f64, WORKGROUP_SIZE>;
var<workgroup> wg_sum_xyy: array<f64, WORKGROUP_SIZE>;
var<workgroup> wg_sum_xxy: array<f64, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let local_idx = local_id.x;
    let workgroup_idx = workgroup_id.x;

    // 初始化 workgroup 内存
    wg_sum_x[local_idx] = 0.0;
    wg_sum_y[local_idx] = 0.0;
    wg_sum_xx[local_idx] = 0.0;
    wg_sum_yy[local_idx] = 0.0;
    wg_sum_xy[local_idx] = 0.0;
    wg_sum_xxx[local_idx] = 0.0;
    wg_sum_yyy[local_idx] = 0.0;
    wg_sum_xyy[local_idx] = 0.0;
    wg_sum_xxy[local_idx] = 0.0;

    workgroupBarrier();

    let idx = global_id.x;
    if (idx < config.num_points) {
        let point = points[idx];
        let x = point.x;
        let y = point.y;
        let xx = x * x;
        let yy = y * y;

        wg_sum_x[local_idx] = x;
        wg_sum_y[local_idx] = y;
        wg_sum_xx[local_idx] = xx;
        wg_sum_yy[local_idx] = yy;
        wg_sum_xy[local_idx] = x * y;
        wg_sum_xxx[local_idx] = x * xx;
        wg_sum_yyy[local_idx] = y * yy;
        wg_sum_xyy[local_idx] = x * yy;
        wg_sum_xxy[local_idx] = y * xx;
    }

    workgroupBarrier();

    // workgroup 内归约
    var s = WORKGROUP_SIZE / 2;
    while (s > 0) {
        if (local_idx < s) {
            wg_sum_x[local_idx] = wg_sum_x[local_idx] + wg_sum_x[local_idx + s];
            wg_sum_y[local_idx] = wg_sum_y[local_idx] + wg_sum_y[local_idx + s];
            wg_sum_xx[local_idx] = wg_sum_xx[local_idx] + wg_sum_xx[local_idx + s];
            wg_sum_yy[local_idx] = wg_sum_yy[local_idx] + wg_sum_yy[local_idx + s];
            wg_sum_xy[local_idx] = wg_sum_xy[local_idx] + wg_sum_xy[local_idx + s];
            wg_sum_xxx[local_idx] = wg_sum_xxx[local_idx] + wg_sum_xxx[local_idx + s];
            wg_sum_yyy[local_idx] = wg_sum_yyy[local_idx] + wg_sum_yyy[local_idx + s];
            wg_sum_xyy[local_idx] = wg_sum_xyy[local_idx] + wg_sum_xyy[local_idx + s];
            wg_sum_xxy[local_idx] = wg_sum_xxy[local_idx] + wg_sum_xxy[local_idx + s];
        }
        workgroupBarrier();
        s = s / 2;
    }

    // 写入 workgroup 结果
    if (local_idx == 0) {
        accumulators[workgroup_idx] = Accumulators(
            wg_sum_x[0],
            wg_sum_y[0],
            wg_sum_xx[0],
            wg_sum_yy[0],
            wg_sum_xy[0],
            wg_sum_xxx[0],
            wg_sum_yyy[0],
            wg_sum_xyy[0],
            wg_sum_xxy[0],
        );
    }
}
"#;

/// 管线缓存（懒加载）
static PIPELINE_CACHE: OnceLock<ArcFitPipeline> = OnceLock::new();

/// 获取或创建圆弧拟合管线
fn get_pipeline(device: &wgpu::Device) -> &'static ArcFitPipeline {
    PIPELINE_CACHE.get_or_init(|| ArcFitPipeline::create(device))
}

/// wgpu 圆弧拟合（GPU 计算累加和 + CPU 求解）
pub async fn fit_arc_wgpu(
    context: &crate::WgpuContext,
    points: &[Point2],
    config: &ArcFitConfig,
) -> AcceleratorResult<AcceleratorArc> {
    let _ = config;
    if points.len() < 3 {
        return Err(AcceleratorError::InvalidDataFormat(
            "圆弧拟合至少需要 3 个点".to_string(),
        ));
    }

    let n = points.len() as u32;
    let workgroup_size = 256;
    let num_workgroups = n.div_ceil(workgroup_size);

    // 准备 uniform 配置
    let uniform = ConfigUniform {
        num_points: n,
        padding: [0, 0, 0],
    };

    // 创建 uniform 缓冲区
    let uniform_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Arc Fit Uniform Buffer"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // 准备点数据
    #[repr(C)]
    #[derive(Debug, Clone, Copy)]
    struct GpuPoint {
        x: f64,
        y: f64,
    }
    unsafe impl bytemuck::Pod for GpuPoint {}
    unsafe impl bytemuck::Zeroable for GpuPoint {}

    let gpu_points: Vec<GpuPoint> = points
        .iter()
        .map(|&p| GpuPoint { x: p[0], y: p[1] })
        .collect();

    let points_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Arc Fit Points Buffer"),
            contents: bytemuck::cast_slice(&gpu_points),
            usage: wgpu::BufferUsages::STORAGE,
        });

    // 累加器输出缓冲区
    let num_accumulators = num_workgroups as usize;
    let accumulators_buffer_size = (num_accumulators * std::mem::size_of::<Accumulators>()) as u64;
    let accumulators_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Arc Fit Accumulators Buffer"),
        size: accumulators_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // 读取结果缓冲区
    let read_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Arc Fit Read Buffer"),
        size: accumulators_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // 获取管线
    let pipeline_cache = get_pipeline(&context.device);

    // 创建绑定组
    let bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Arc Fit Bind Group"),
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
                    resource: accumulators_buffer.as_entire_binding(),
                },
            ],
        });

    // 编码计算命令
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Arc Fit Compute Encoder"),
        });

    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Arc Fit Compute Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline_cache.pipeline);
        pass.set_bind_group(0, &bind_group, &[]);
        pass.dispatch_workgroups(num_workgroups, 1, 1);
    }

    // 复制结果到可读缓冲区
    encoder.copy_buffer_to_buffer(
        &accumulators_buffer,
        0,
        &read_buffer,
        0,
        accumulators_buffer_size,
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

    // 读取累加器数据
    let data = buffer_slice.get_mapped_range();
    let gpu_accumulators: &[Accumulators] = bytemuck::cast_slice(&data);

    // CPU 端合并所有 workgroup 的累加和
    let mut total = Accumulators::default();
    for acc in gpu_accumulators.iter().take(num_accumulators) {
        total.sum_x += acc.sum_x;
        total.sum_y += acc.sum_y;
        total.sum_xx += acc.sum_xx;
        total.sum_yy += acc.sum_yy;
        total.sum_xy += acc.sum_xy;
        total.sum_xxx += acc.sum_xxx;
        total.sum_yyy += acc.sum_yyy;
        total.sum_xyy += acc.sum_xyy;
        total.sum_xxy += acc.sum_xxy;
    }

    drop(data);
    read_buffer.unmap();

    // 计算质心
    let n_f64 = points.len() as f64;
    let centroid = [total.sum_x / n_f64, total.sum_y / n_f64];

    // 中心化数据并重新计算累加和（相对于质心）
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_yy = 0.0;
    let mut sum_xy = 0.0;
    let mut sum_xxx = 0.0;
    let mut sum_yyy = 0.0;
    let mut sum_xyy = 0.0;
    let mut sum_xxy = 0.0;

    for &point in points {
        let x = point[0] - centroid[0];
        let y = point[1] - centroid[1];
        let xx = x * x;
        let yy = y * y;
        let xy = x * y;

        sum_x += x;
        sum_y += y;
        sum_xx += xx;
        sum_yy += yy;
        sum_xy += xy;
        sum_xxx += x * xx;
        sum_yyy += y * yy;
        sum_xyy += x * yy;
        sum_xxy += y * xx;
    }

    // 解线性方程组
    let a = n_f64 * sum_xx - sum_x * sum_x;
    let b = n_f64 * sum_xy - sum_x * sum_y;
    let c = n_f64 * sum_yy - sum_y * sum_y;
    let d = n_f64 * sum_xxx + n_f64 * sum_xyy - (sum_x * sum_xx + sum_x * sum_yy);
    let e = n_f64 * sum_xxy + n_f64 * sum_yyy - (sum_y * sum_xx + sum_y * sum_yy);

    let det = a * c - b * b;
    if det.abs() < 1e-10 {
        return Ok(AcceleratorArc::new(
            centroid,
            1e10,
            0.0,
            std::f64::consts::PI * 2.0,
        ));
    }

    let center_x = (c * d - b * e) / (2.0 * det);
    let center_y = (a * e - b * d) / (2.0 * det);

    let center = [center_x + centroid[0], center_y + centroid[1]];

    // 计算半径（平均距离）
    let radius = points
        .iter()
        .map(|p| {
            let dx = p[0] - center[0];
            let dy = p[1] - center[1];
            (dx * dx + dy * dy).sqrt()
        })
        .sum::<f64>()
        / n_f64;

    // 计算起始和终止角度
    let angles: Vec<f64> = points
        .iter()
        .map(|p| (p[1] - center[1]).atan2(p[0] - center[0]))
        .collect();

    let start_angle = angles.iter().cloned().fold(f64::INFINITY, f64::min);
    let end_angle = angles.iter().cloned().fold(f64::NEG_INFINITY, f64::max);

    Ok(AcceleratorArc::new(center, radius, start_angle, end_angle))
}
