//! wgpu 轮廓提取实现
//!
//! 使用计算着色器实现连通分量标记（两步并行算法）

use accelerator_api::{AcceleratorResult, ContourExtractConfig, Contours, EdgeMap};
#[allow(unused_imports)]
use bytemuck::{Pod, Zeroable};
use common_types::{Point2, Polyline};
use std::sync::OnceLock;
use wgpu::util::DeviceExt;

/// Uniform 缓冲区配置
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ConfigUniform {
    width: u32,
    height: u32,
    min_contour_length: u32,
    padding: u32,
}

unsafe impl bytemuck::Pod for ConfigUniform {}
unsafe impl bytemuck::Zeroable for ConfigUniform {}

/// 轮廓提取管线缓存
#[derive(Debug)]
struct ContourExtractPipeline {
    first_pass_pipeline: wgpu::ComputePipeline,
    second_pass_pipeline: wgpu::ComputePipeline,
    first_pass_bind_group_layout: wgpu::BindGroupLayout,
    second_pass_bind_group_layout: wgpu::BindGroupLayout,
}

/// 第一遍连通分量标记计算着色器
const CONTOUR_FIRST_PASS_SHADER: &str = r#"
struct Config {
    width: u32,
    height: u32,
    min_contour_length: u32,
    padding: u32,
};

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage, read> input_edges: array<u32>;  // 0 = edge, 255 = non-edge -> stored as u32

@group(0) @binding(2)
var<storage, read_write> output_labels: array<u32>;

@group(0) @binding(3)
var<storage, read_write> parent: array<u32>;

// Union-Find find with path compression
fn find_root(label: u32, parent: ptr<storage, read_write, array<u32>>) -> u32 {
    var p = parent[label];
    if (p != label) {
        parent[label] = find_root(p, parent);
        return parent[label];
    }
    return p;
}

fn union_root(r1: u32, r2: u32, parent: ptr<storage, read_write, array<u32>>) {
    // Union by root value (smaller becomes parent of larger)
    if (r1 < r2) {
        parent[r2] = r1;
    } else {
        parent[r1] = r2;
    }
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= config.width || y >= config.height) {
        return;
    }

    let idx = y * config.width + x;
    let edge_val = input_edges[idx];

    // 非边缘像素标签为 0
    if (edge_val != 0u) {
        output_labels[idx] = 0u;
        parent[idx] = 0u;
        return;
    }

    // 第一遍：检查左和上邻居
    var label: u32 = 0u;
    let current_eq: u32 = (y * config.width + x) + 1u;

    // 检查左邻居 (x-1, y)
    if (x > 0u) {
        let left_idx = y * config.width + (x - 1u);
        let left_label = output_labels[left_idx];
        if (left_label != 0u) {
            if (label == 0u) {
                label = find_root(left_label, parent);
            } else {
                let r1 = label;
                let r2 = find_root(left_label, parent);
                if (r1 != r2) {
                    union_root(r1, r2, parent);
                }
            }
        }
    }

    // 检查上邻居 (x, y-1)
    if (y > 0u) {
        let up_idx = (y - 1u) * config.width + x;
        let up_label = output_labels[up_idx];
        if (up_label != 0u) {
            if (label == 0u) {
                label = find_root(up_label, parent);
            } else {
                let r1 = label;
                let r2 = find_root(up_label, parent);
                if (r1 != r2) {
                    union_root(r1, r2, parent);
                }
            }
        }
    }

    if (label == 0u) {
        // 新连通分量
        label = current_eq;
    }

    output_labels[idx] = label;
    parent[current_eq] = label;
}
"#;

/// 第二遍连通分量解析着色器（重编号标签）
const CONTOUR_SECOND_PASS_SHADER: &str = r#"
struct Config {
    width: u32,
    height: u32,
    min_contour_length: u32,
    padding: u32,
};

@group(0) @binding(0)
var<uniform> config: Config;

@group(0) @binding(1)
var<storage, read> parent: array<u32>;

@group(0) @binding(2)
var<storage, read_write> output_labels: array<u32>;

fn find_root(label: u32, parent: ptr<storage, read, array<u32>>) -> u32 {
    var p = label;
    while (p != 0u && parent[p] != p) {
        p = parent[p];
    }
    return p;
}

@compute @workgroup_size(8, 8)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= config.width || y >= config.height) {
        return;
    }

    let idx = y * config.width + x;
    var label = output_labels[idx];

    if (label != 0u) {
        // Find root after all unions
        label = find_root(label, parent);
        output_labels[idx] = label;
    }
}
"#;

impl ContourExtractPipeline {
    pub fn create(device: &wgpu::Device) -> Self {
        // First pass: initial labeling and union-find
        let first_pass_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Contour Extract First Pass"),
            source: wgpu::ShaderSource::Wgsl(CONTOUR_FIRST_PASS_SHADER.into()),
        });

        let first_pass_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Contour Extract First Pass Bind Group Layout"),
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
                    wgpu::BindGroupLayoutEntry {
                        binding: 3,
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

        let first_pass_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Contour Extract First Pass Pipeline Layout"),
                bind_group_layouts: &[&first_pass_bind_group_layout],
                push_constant_ranges: &[],
            });

        let first_pass_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Contour Extract First Pass Compute Pipeline"),
                layout: Some(&first_pass_pipeline_layout),
                module: &first_pass_shader,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        // Second pass: resolve roots and compact labels
        let second_pass_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("Contour Extract Second Pass"),
            source: wgpu::ShaderSource::Wgsl(CONTOUR_SECOND_PASS_SHADER.into()),
        });

        let second_pass_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: Some("Contour Extract Second Pass Bind Group Layout"),
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

        let second_pass_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Contour Extract Second Pass Pipeline Layout"),
                bind_group_layouts: &[&second_pass_bind_group_layout],
                push_constant_ranges: &[],
            });

        let second_pass_pipeline =
            device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Contour Extract Second Pass Compute Pipeline"),
                layout: Some(&second_pass_pipeline_layout),
                module: &second_pass_shader,
                entry_point: "main",
                compilation_options: wgpu::PipelineCompilationOptions::default(),
            });

        Self {
            first_pass_pipeline,
            second_pass_pipeline,
            first_pass_bind_group_layout,
            second_pass_bind_group_layout,
        }
    }
}

/// 管线缓存（懒加载）
static PIPELINE_CACHE: OnceLock<ContourExtractPipeline> = OnceLock::new();

/// 获取或创建轮廓提取管线
fn get_pipeline(device: &wgpu::Device) -> &'static ContourExtractPipeline {
    PIPELINE_CACHE.get_or_init(|| ContourExtractPipeline::create(device))
}

/// wgpu 轮廓提取（GPU 加速连通分量标记 + CPU 轮廓跟踪
pub async fn extract_contours_wgpu(
    context: &crate::WgpuContext,
    edges: &EdgeMap,
    config: &ContourExtractConfig,
) -> AcceleratorResult<Contours> {
    let width = edges.width;
    let height = edges.height;
    let pixel_count = (width * height) as usize;

    // 准备 uniform 配置
    let uniform = ConfigUniform {
        width,
        height,
        min_contour_length: config.min_contour_length as u32,
        padding: 0,
    };

    // 创建 uniform 缓冲区
    let uniform_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Contour Extract Uniform Buffer"),
            contents: bytemuck::bytes_of(&uniform),
            usage: wgpu::BufferUsages::UNIFORM,
        });

    // 输入边缘数据转换为 u32 存储（每个像素一个 u32）
    let input_data: Vec<u32> = edges
        .data
        .iter()
        .map(|&p| if p == 0 { 1 } else { 0 })
        .collect();

    let input_buffer = context
        .device
        .create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("Contour Extract Input Buffer"),
            contents: bytemuck::cast_slice(&input_data),
            usage: wgpu::BufferUsages::STORAGE,
        });

    // 输出标签缓冲区
    let labels_buffer_size = (pixel_count * std::mem::size_of::<u32>()) as u64;
    let labels_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Contour Extract Labels Buffer"),
        size: labels_buffer_size,
        usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
        mapped_at_creation: false,
    });

    // parent 缓冲区（union-find）
    let parent_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Contour Extract Parent Buffer"),
        size: labels_buffer_size,
        usage: wgpu::BufferUsages::STORAGE,
        mapped_at_creation: false,
    });

    // 读取结果缓冲区
    let read_buffer_size = labels_buffer_size;
    let read_buffer = context.device.create_buffer(&wgpu::BufferDescriptor {
        label: Some("Contour Extract Read Buffer"),
        size: read_buffer_size,
        usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
        mapped_at_creation: false,
    });

    // 获取管线
    let pipeline_cache = get_pipeline(&context.device);

    // 创建第一遍绑定组
    let first_pass_bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Contour Extract First Pass Bind Group"),
            layout: &pipeline_cache.first_pass_bind_group_layout,
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
                    resource: labels_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 3,
                    resource: parent_buffer.as_entire_binding(),
                },
            ],
        });

    // 创建第二遍绑定组
    let second_pass_bind_group = context
        .device
        .create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Contour Extract Second Pass Bind Group"),
            layout: &pipeline_cache.second_pass_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: uniform_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: parent_buffer.as_entire_binding(),
                },
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: labels_buffer.as_entire_binding(),
                },
            ],
        });

    // 编码计算命令
    let mut encoder = context
        .device
        .create_command_encoder(&wgpu::CommandEncoderDescriptor {
            label: Some("Contour Extract Compute Encoder"),
        });

    // 第一遍：初始标记
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Contour Extract First Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline_cache.first_pass_pipeline);
        pass.set_bind_group(0, &first_pass_bind_group, &[]);

        let workgroup_x = width.div_ceil(8);
        let workgroup_y = height.div_ceil(8);
        pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
    }

    // 第二遍：解析根标签
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("Contour Extract Second Pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(&pipeline_cache.second_pass_pipeline);
        pass.set_bind_group(0, &second_pass_bind_group, &[]);

        let workgroup_x = width.div_ceil(8);
        let workgroup_y = height.div_ceil(8);
        pass.dispatch_workgroups(workgroup_x, workgroup_y, 1);
    }

    // 复制结果到可读缓冲区
    encoder.copy_buffer_to_buffer(&labels_buffer, 0, &read_buffer, 0, labels_buffer_size);

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

    // 读取标签数据
    let data = buffer_slice.get_mapped_range();
    let labels: &[u32] = bytemuck::cast_slice(&data);

    // 在 CPU 上跟踪轮廓
    let contours = trace_contours_from_labels(labels, width, height, config.min_contour_length);

    drop(data);
    read_buffer.unmap();

    // 如果需要简化，使用 Douglas-Peucker
    let result_contours = if config.simplify {
        contours
            .into_iter()
            .filter_map(|contour| {
                if contour.len() >= 2 {
                    let simplified = douglas_peucker(&contour, config.simplify_epsilon);
                    if simplified.len() >= 2 {
                        Some(simplified)
                    } else {
                        None
                    }
                } else {
                    None
                }
            })
            .collect()
    } else {
        contours
            .into_iter()
            .filter(|c| c.len() >= config.min_contour_length)
            .collect()
    };

    // Douglas-Peucker 多边形简化算法（内联实现避免循环依赖）
    fn douglas_peucker(points: &[Point2], epsilon: f64) -> Vec<Point2> {
        if points.len() <= 2 {
            return points.to_vec();
        }

        let mut keep = vec![false; points.len()];
        keep[0] = true;
        keep[points.len() - 1] = true;

        douglas_peucker_recursive(points, 0, points.len() - 1, epsilon, &mut keep);

        points
            .iter()
            .enumerate()
            .filter(|(i, _)| keep[*i])
            .map(|(_, p)| *p)
            .collect()
    }

    fn douglas_peucker_recursive(
        points: &[Point2],
        start: usize,
        end: usize,
        epsilon: f64,
        keep: &mut [bool],
    ) {
        if start >= end {
            return;
        }

        let mut max_dist = 0.0;
        let mut max_idx = start;

        let line_start = points[start];
        let line_end = points[end];

        #[allow(clippy::needless_range_loop)]
        for i in (start + 1)..end {
            let dist = point_to_line_distance(points[i], line_start, line_end);
            if dist > max_dist {
                max_dist = dist;
                max_idx = i;
            }
        }

        if max_dist > epsilon {
            keep[max_idx] = true;
            douglas_peucker_recursive(points, start, max_idx, epsilon, keep);
            douglas_peucker_recursive(points, max_idx, end, epsilon, keep);
        }
    }

    fn point_to_line_distance(point: Point2, line_start: Point2, line_end: Point2) -> f64 {
        let dx = line_end[0] - line_start[0];
        let dy = line_end[1] - line_start[1];

        if dx.abs() < 1e-10 && dy.abs() < 1e-10 {
            return ((point[0] - line_start[0]).powi(2) + (point[1] - line_start[1]).powi(2))
                .sqrt();
        }

        let t = ((point[0] - line_start[0]) * dx + (point[1] - line_start[1]) * dy)
            / (dx * dx + dy * dy);
        let t = t.clamp(0.0, 1.0);

        let proj_x = line_start[0] + t * dx;
        let proj_y = line_start[1] + t * dy;

        ((point[0] - proj_x).powi(2) + (point[1] - proj_y).powi(2)).sqrt()
    }

    Ok(result_contours)
}

/// 从 GPU 计算得到的标签中跟踪轮廓（CPU 侧）
/// 使用 8-邻域跟踪提取每个连通分量的轮廓点
fn trace_contours_from_labels(
    labels: &[u32],
    width: u32,
    height: u32,
    min_length: usize,
) -> Contours {
    let w = width as usize;
    let h = height as usize;
    let mut visited = vec![false; w * h];
    let mut contours = Vec::new();

    for y in 0..h {
        for x in 0..w {
            let idx = y * w + x;
            let label = labels[idx];
            if label != 0 && !visited[idx] {
                // 跟踪这个连通分量
                let contour = trace_single_contour(label, x, y, w, h, labels, &mut visited);

                if contour.len() >= min_length {
                    contours.push(contour);
                }
            }
        }
    }

    contours
}

/// 跟踪单个连通分量（轮廓跟随算法）
fn trace_single_contour(
    target_label: u32,
    start_x: usize,
    start_y: usize,
    width: usize,
    height: usize,
    labels: &[u32],
    visited: &mut [bool],
) -> Polyline {
    let mut contour = Vec::new();
    let mut current_x = start_x as i32;
    let mut current_y = start_y as i32;
    let mut last_dir = 0;

    // 8 邻域方向
    let directions: [(i32, i32); 8] = [
        (1, 0),
        (1, 1),
        (0, 1),
        (-1, 1),
        (-1, 0),
        (-1, -1),
        (0, -1),
        (1, -1),
    ];

    loop {
        let idx = (current_y as usize) * width + (current_x as usize);
        visited[idx] = true;
        contour.push([current_x as f64, current_y as f64]);

        // 从反方向开始搜索下一个点
        let search_start = (last_dir + 4) % 8;

        'search: for i in 0..8 {
            let dir_idx = (search_start + i) % 8;
            let (dx, dy) = directions[dir_idx];
            let nx = current_x + dx;
            let ny = current_y + dy;

            if nx >= 0 && nx < width as i32 && ny >= 0 && ny < height as i32 {
                let nidx = (ny as usize) * width + (nx as usize);
                if labels[nidx] == target_label && !visited[nidx] {
                    current_x = nx;
                    current_y = ny;
                    last_dir = dir_idx;
                    break 'search;
                }
            }
        }

        // 如果回到起点，完成
        if current_x == start_x as i32 && current_y == start_y as i32 && contour.len() > 2 {
            break;
        }

        // 如果没有找到下一个点，退出
        if current_x == start_x as i32 && current_y == start_y as i32 && contour.len() <= 2 {
            break;
        }
    }

    contour
}
