// macOS 风格毛玻璃效果着色器
// 技术实现：
// 1. 高斯模糊（可分离卷积）
// 2. 毛玻璃合成（半透明白色 + 模糊背景）
// 3. 顶部高光（模拟光线反射）

// ==================== 顶点着色器 ====================

@vertex
fn vs_fullscreen_quad(
    @builtin(vertex_index) vertex_index: u32
) -> @builtin(position) vec4<f32> {
    var positions = array<vec2<f32>, 6>(
        vec2<f32>(-1.0, -1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>(-1.0,  1.0),
        vec2<f32>( 1.0, -1.0),
        vec2<f32>( 1.0,  1.0)
    );
    let pos = positions[vertex_index];
    return vec4<f32>(pos, 0.0, 1.0);
}

// ==================== 高斯模糊 - 水平方向 ====================

@group(0) @binding(0)
var t_source: texture_2d<f32>;

@group(0) @binding(1)
var s_sampler: sampler;

@group(0) @binding(2)
var<uniform> blur_params: vec2<f32>;  // x: 方向 (1.0=水平，0.0=垂直), y: 模糊半径

@fragment
fn fs_gaussian_blur(
    @builtin(position) frag_coord: vec4<f32>
) -> @location(0) vec4<f32> {
    let texture_size = vec2<f32>(textureDimensions(t_source));
    let uv = frag_coord.xy / texture_size;
    
    let direction = blur_params.x;
    let blur_radius = blur_params.y;
    
    var color = vec4<f32>(0.0);
    var total_weight = 0.0;
    
    // 高斯分布权重计算
    let sigma = blur_radius / 2.0;
    let sigma_squared = sigma * sigma;
    
    // 采样模糊半径内的像素
    for (var i = -i32(blur_radius); i <= i32(blur_radius); i = i + 1) {
        let offset = vec2<f32>(
            f32(i) * direction,
            f32(i) * (1.0 - direction)
        ) / texture_size;
        
        // 高斯权重：exp(-x² / (2σ²))
        let weight = exp(-(f32(i) * f32(i)) / (2.0 * sigma_squared));
        color += textureSample(t_source, s_sampler, uv + offset) * weight;
        total_weight += weight;
    }
    
    return color / total_weight;
}

// ==================== 毛玻璃合成 ====================

@group(1) @binding(0)
var t_blurred: texture_2d<f32>;

@group(1) @binding(1)
var s_sampler_glass: sampler;

@group(1) @binding(2)
var<uniform> glass_params: vec4<f32>;  // xyz: 玻璃颜色 RGB, w: 不透明度

@fragment
fn fs_glass_composite(
    @builtin(position) frag_coord: vec4<f32>
) -> @location(0) vec4<f32> {
    let texture_size = vec2<f32>(textureDimensions(t_blurred));
    let uv = frag_coord.xy / texture_size;
    
    // 从模糊纹理采样
    let blurred_color = textureSample(t_blurred, s_sampler_glass, uv);
    
    // 毛玻璃颜色（半透明白色）
    let glass_color = glass_params.xyz;
    let glass_alpha = glass_params.w;
    
    // 合成：模糊背景 + 半透明白色
    let result = blurred_color * (1.0 - glass_alpha) + vec4<f32>(glass_color, glass_alpha);
    
    // 添加顶部高光（模拟光线反射）
    let highlight = smoothstep(1.0, 0.7, uv.y) * 0.15;
    let final_color = result + vec4<f32>(highlight, highlight, highlight, 0.0);
    
    return vec4<f32>(final_color.rgb, 1.0);
}
