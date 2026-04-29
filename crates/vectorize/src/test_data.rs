//! 测试数据生成器
//!
//! 生成各类工程图纸的合成测试数据：
//! - 建筑平面图（墙、门、窗、标注）
//! - 机械零件图（轮廓、孔、倒角、螺纹）
//! - 电路图（连线、元件符号）
//! - 低质量扫描图模拟（噪声、模糊、倾斜、阴影）

use image::{GrayImage, Luma};
use std::f64::consts::TAU;

/// 图纸类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DrawingType {
    /// 建筑平面图
    Architectural,
    /// 机械零件图
    Mechanical,
    /// 电路图
    Circuit,
    /// 手绘草图
    HandDrawn,
}

/// 生成图像的质量配置
#[derive(Debug, Clone)]
pub struct QualityConfig {
    /// 高斯模糊半径
    pub blur_radius: f64,
    /// 椒盐噪声比例
    pub salt_pepper_noise: f64,
    /// 倾斜角度（度）
    pub skew_angle: f64,
    /// 对比度（1.0 为正常）
    pub contrast: f64,
    /// 亮度偏移
    pub brightness_offset: i32,
    /// 阴影强度
    pub shadow_intensity: f64,
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            blur_radius: 0.0,
            salt_pepper_noise: 0.0,
            skew_angle: 0.0,
            contrast: 1.0,
            brightness_offset: 0,
            shadow_intensity: 0.0,
        }
    }
}

/// 简单直线绘制（Bresenham）
fn draw_line(image: &mut GrayImage, x1: i32, y1: i32, x2: i32, y2: i32, thickness: u32) {
    let (width, height) = (image.width() as i32, image.height() as i32);

    // 简单粗线：多次绘制
    for t in -(thickness as i32 / 2)..=(thickness as i32 / 2) {
        let mut x = x1;
        let mut y = y1;
        let dx = (x2 - x1).abs();
        let dy = (y2 - y1).abs();
        let sx = if x1 < x2 { 1 } else { -1 };
        let sy = if y1 < y2 { 1 } else { -1 };
        let mut err = dx - dy;

        loop {
            let px = x + t;
            let py = y;
            if px >= 0 && px < width && py >= 0 && py < height {
                image.put_pixel(px as u32, py as u32, Luma([0]));
            }
            let px2 = x;
            let py2 = y + t;
            if px2 >= 0 && px2 < width && py2 >= 0 && py2 < height {
                image.put_pixel(px2 as u32, py2 as u32, Luma([0]));
            }

            if x == x2 && y == y2 {
                break;
            }
            let e2 = 2 * err;
            if e2 > -dy {
                err -= dy;
                x += sx;
            }
            if e2 < dx {
                err += dx;
                y += sy;
            }
        }
    }
}

/// 绘制矩形
fn draw_rect(image: &mut GrayImage, x: i32, y: i32, w: u32, h: u32, thickness: u32) {
    draw_line(image, x, y, x + w as i32, y, thickness);
    draw_line(
        image,
        x + w as i32,
        y,
        x + w as i32,
        y + h as i32,
        thickness,
    );
    draw_line(
        image,
        x + w as i32,
        y + h as i32,
        x,
        y + h as i32,
        thickness,
    );
    draw_line(image, x, y + h as i32, x, y, thickness);
}

/// 绘制圆
fn draw_circle(image: &mut GrayImage, cx: i32, cy: i32, radius: u32, thickness: u32) {
    let (width, height) = (image.width() as i32, image.height() as i32);
    let r = radius as i32;

    for t in -(thickness as i32 / 2)..=(thickness as i32 / 2) {
        let mut x = 0;
        let mut y = r + t;
        let mut d = 1 - r - t;

        while x <= y {
            let points = [
                (cx + x, cy + y),
                (cx - x, cy + y),
                (cx + x, cy - y),
                (cx - x, cy - y),
                (cx + y, cy + x),
                (cx - y, cy + x),
                (cx + y, cy - x),
                (cx - y, cy - x),
            ];

            for (px, py) in points {
                if px >= 0 && px < width && py >= 0 && py < height {
                    image.put_pixel(px as u32, py as u32, Luma([0]));
                }
            }

            if d < 0 {
                d += 2 * x + 3;
            } else {
                d += 2 * (x - y) + 5;
                y -= 1;
            }
            x += 1;
        }
    }
}

/// 绘制门符号（建筑）
fn draw_door_symbol(image: &mut GrayImage, cx: i32, cy: i32, size: u32) {
    // 门框
    draw_line(image, cx - size as i32 / 2, cy, cx + size as i32 / 2, cy, 2);
    // 门弧
    let r = size / 2;
    for angle in 0..=90 {
        let rad = angle as f64 * TAU / 360.0;
        let x = cx + (r as f64 * rad.cos()) as i32;
        let y = cy - (r as f64 * rad.sin()) as i32;
        if x >= 0 && x < image.width() as i32 && y >= 0 && y < image.height() as i32 {
            image.put_pixel(x as u32, y as u32, Luma([0]));
        }
    }
}

/// 绘制窗户符号
fn draw_window_symbol(image: &mut GrayImage, x: i32, y: i32, length: u32, horizontal: bool) {
    if horizontal {
        draw_line(image, x, y, x + length as i32, y, 1);
        draw_line(image, x, y + 5, x + length as i32, y + 5, 1);
    } else {
        draw_line(image, x, y, x, y + length as i32, 1);
        draw_line(image, x + 5, y, x + 5, y + length as i32, 1);
    }
}

/// 绘制尺寸标注
fn draw_dimension(image: &mut GrayImage, x1: i32, y1: i32, x2: i32, y2: i32, offset: i32) {
    // 延伸线
    draw_line(image, x1, y1, x1 + offset, y1, 1);
    draw_line(image, x2, y2, x2 + offset, y2, 1);
    // 尺寸线
    draw_line(image, x1 + offset, y1, x2 + offset, y2, 1);
    // 箭头
    let ax = (x2 - x1) / 10;
    let ay = (y2 - y1) / 10;
    draw_line(image, x1 + offset, y1, x1 + offset + ax, y1 + ay, 1);
    draw_line(image, x2 + offset, y2, x2 + offset - ax, y2 - ay, 1);
}

// ============================================================================
// 建筑平面图生成器
// ============================================================================

/// 生成简单建筑平面图
pub fn generate_architectural_floorplan(width: u32, height: u32) -> GrayImage {
    let mut image = GrayImage::from_pixel(width, height, Luma([255]));

    // 外墙
    let wall_thickness = 4;
    draw_rect(
        &mut image,
        50,
        50,
        width - 100,
        height - 100,
        wall_thickness,
    );

    // 内墙分隔
    let mid_x = (width / 2) as i32;
    let mid_y = (height / 2) as i32;
    draw_line(&mut image, mid_x, 50, mid_x, mid_y, wall_thickness);
    draw_line(&mut image, 50, mid_y, mid_x, mid_y, wall_thickness);

    // 门
    draw_door_symbol(&mut image, mid_x - 40, mid_y, 25);
    draw_door_symbol(&mut image, mid_x + 40, 100, 25);

    // 窗户
    draw_window_symbol(&mut image, 100, 45, 80, true);
    draw_window_symbol(&mut image, (width - 180) as i32, 45, 80, true);
    draw_window_symbol(&mut image, 45, 150, 60, false);

    // 尺寸标注
    draw_dimension(&mut image, 50, 50, (width - 50) as i32, 50, -20);
    draw_dimension(&mut image, 50, 50, 50, (height - 50) as i32, -20);

    image
}

// ============================================================================
// 机械零件图生成器
// ============================================================================

/// 生成法兰盘类零件图
pub fn generate_mechanical_flange(width: u32, height: u32) -> GrayImage {
    let mut image = GrayImage::from_pixel(width, height, Luma([255]));

    let cx = (width / 2) as i32;
    let cy = (height / 2) as i32;

    // 外圆
    draw_circle(&mut image, cx, cy, 80, 2);
    // 内孔
    draw_circle(&mut image, cx, cy, 30, 2);
    // 螺栓孔
    for i in 0..6 {
        let angle = i as f64 * TAU / 6.0;
        let bx = cx + (55.0 * angle.cos()) as i32;
        let by = cy + (55.0 * angle.sin()) as i32;
        draw_circle(&mut image, bx, by, 8, 2);
    }

    // 中心线
    draw_line(&mut image, cx - 90, cy, cx + 90, cy, 1);
    draw_line(&mut image, cx, cy - 90, cx, cy + 90, 1);

    // 尺寸标注
    draw_dimension(&mut image, cx, cy - 80, cx, cy - 30, 30); // 内孔
    draw_dimension(&mut image, cx, cy - 30, cx, cy - 5, 40); // 螺栓孔圆

    image
}

/// 生成轴类零件图
pub fn generate_mechanical_shaft(width: u32, height: u32) -> GrayImage {
    let mut image = GrayImage::from_pixel(width, height, Luma([255]));

    let cy = (height / 2) as i32;

    // 轴轮廓
    draw_line(&mut image, 50, cy - 20, 150, cy - 20, 2); // 第一段上
    draw_line(&mut image, 50, cy + 20, 150, cy + 20, 2); // 第一段下
    draw_line(&mut image, 150, cy - 25, 300, cy - 25, 2); // 第二段上
    draw_line(&mut image, 150, cy + 25, 300, cy + 25, 2); // 第二段下
    draw_line(&mut image, 300, cy - 20, 400, cy - 20, 2); // 第三段上
    draw_line(&mut image, 300, cy + 20, 400, cy + 20, 2); // 第三段下

    // 端面
    draw_line(&mut image, 50, cy - 20, 50, cy + 20, 2);
    draw_line(&mut image, 400, cy - 20, 400, cy + 20, 2);

    // 台阶
    draw_line(&mut image, 150, cy - 25, 150, cy - 20, 2);
    draw_line(&mut image, 150, cy + 20, 150, cy + 25, 2);
    draw_line(&mut image, 300, cy - 25, 300, cy - 20, 2);
    draw_line(&mut image, 300, cy + 20, 300, cy + 25, 2);

    // 键槽
    draw_rect(&mut image, 180, cy - 8, 40, 16, 1);

    // 中心线
    draw_line(&mut image, 40, cy, 410, cy, 1);

    // 尺寸标注
    draw_dimension(&mut image, 50, cy - 30, 400, cy - 30, -15);

    image
}

// ============================================================================
// 电路图生成器
// ============================================================================

/// 绘制电阻符号
fn draw_resistor(image: &mut GrayImage, x: i32, y: i32, length: u32) {
    let l = length as i32;
    draw_line(image, x, y, x + 10, y, 1);
    // 锯齿
    for i in 0..3 {
        let sx = x + 10 + i * 20;
        draw_line(image, sx, y, sx + 10, y - 8, 1);
        draw_line(image, sx + 10, y - 8, sx + 20, y, 1);
    }
    draw_line(image, x + 10 + 60, y, x + l, y, 1);
}

/// 绘制电容符号
fn draw_capacitor(image: &mut GrayImage, x: i32, y: i32, _length: u32) {
    draw_line(image, x, y, x + 20, y, 1);
    draw_line(image, x + 20, y - 15, x + 20, y + 15, 2);
    draw_line(image, x + 30, y - 15, x + 30, y + 15, 2);
    draw_line(image, x + 30, y, x + 50, y, 1);
}

/// 绘制二极管符号
fn draw_diode(image: &mut GrayImage, x: i32, y: i32) {
    draw_line(image, x, y, x + 15, y, 1);
    draw_line(image, x + 15, y - 10, x + 15, y + 10, 1);
    draw_line(image, x + 15, y - 10, x + 30, y, 1);
    draw_line(image, x + 15, y + 10, x + 30, y, 1);
    draw_line(image, x + 30, y, x + 45, y, 1);
}

/// 生成简单电路
pub fn generate_circuit_diagram(width: u32, height: u32) -> GrayImage {
    let mut image = GrayImage::from_pixel(width, height, Luma([255]));

    // 主回路
    draw_line(&mut image, 50, 100, 50, 200, 2);
    draw_line(&mut image, 50, 100, 400, 100, 2);
    draw_line(&mut image, 400, 100, 400, 200, 2);
    draw_line(&mut image, 50, 200, 400, 200, 2);

    // 元件
    draw_resistor(&mut image, 100, 100, 80);
    draw_capacitor(&mut image, 220, 100, 50);
    draw_diode(&mut image, 300, 100);

    // 下支路元件
    draw_resistor(&mut image, 150, 200, 80);
    draw_capacitor(&mut image, 270, 200, 50);

    // 中间连线
    draw_line(&mut image, 140, 100, 140, 150, 1);
    draw_diode(&mut image, 140, 150);
    draw_line(&mut image, 185, 150, 185, 200, 1);

    image
}

// ============================================================================
// 质量退化（模拟扫描效果）
// ============================================================================

/// 应用质量退化效果
pub fn apply_quality_degradation(image: &GrayImage, config: &QualityConfig) -> GrayImage {
    let (width, height) = image.dimensions();
    let mut result = image.clone();

    // 对比度和亮度
    if (config.contrast - 1.0).abs() > 1e-6 || config.brightness_offset != 0 {
        for y in 0..height {
            for x in 0..width {
                let pixel = result.get_pixel(x, y)[0] as f64;
                let adjusted =
                    (pixel - 128.0) * config.contrast + 128.0 + config.brightness_offset as f64;
                let clamped = adjusted.clamp(0.0, 255.0) as u8;
                result.put_pixel(x, y, Luma([clamped]));
            }
        }
    }

    // 简单模糊（盒式滤波）
    if config.blur_radius > 0.5 {
        let blur_r = config.blur_radius as i32;
        let mut blurred = result.clone();
        for y in blur_r..(height as i32 - blur_r) {
            for x in blur_r..(width as i32 - blur_r) {
                let mut sum = 0;
                let mut count = 0;
                for dy in -blur_r..=blur_r {
                    for dx in -blur_r..=blur_r {
                        sum += result.get_pixel((x + dx) as u32, (y + dy) as u32)[0] as u32;
                        count += 1;
                    }
                }
                let avg = (sum / count) as u8;
                blurred.put_pixel(x as u32, y as u32, Luma([avg]));
            }
        }
        result = blurred;
    }

    // 椒盐噪声
    if config.salt_pepper_noise > 0.0 {
        use std::collections::hash_map::DefaultHasher;
        use std::hash::{Hash, Hasher};

        let mut hasher = DefaultHasher::new();
        for y in 0..height {
            for x in 0..width {
                (x, y).hash(&mut hasher);
                let hash = hasher.finish();
                if (hash as f64) / (u64::MAX as f64) < config.salt_pepper_noise {
                    let val = if (hash & 1) == 0 { 0 } else { 255 };
                    result.put_pixel(x, y, Luma([val]));
                }
            }
        }
    }

    // 阴影效果（简单渐晕）
    if config.shadow_intensity > 0.0 {
        let cx = width as f64 / 2.0;
        let cy = height as f64 / 2.0;
        let max_dist = (cx * cx + cy * cy).sqrt();

        for y in 0..height {
            for x in 0..width {
                let dx = x as f64 - cx;
                let dy = y as f64 - cy;
                let dist = (dx * dx + dy * dy).sqrt() / max_dist;
                let factor = 1.0 - config.shadow_intensity * dist * dist;

                let pixel = result.get_pixel(x, y)[0] as f64;
                let adjusted = (pixel * factor).clamp(0.0, 255.0) as u8;
                result.put_pixel(x, y, Luma([adjusted]));
            }
        }
    }

    result
}

// ============================================================================
// 测试集生成器
// ============================================================================

/// 生成完整测试集配置
pub struct TestDatasetConfig {
    pub image_width: u32,
    pub image_height: u32,
    pub num_architectural: usize,
    pub num_mechanical: usize,
    pub num_circuit: usize,
    pub quality_levels: Vec<QualityConfig>,
}

impl Default for TestDatasetConfig {
    fn default() -> Self {
        Self {
            image_width: 512,
            image_height: 512,
            num_architectural: 10,
            num_mechanical: 10,
            num_circuit: 5,
            quality_levels: vec![
                QualityConfig::default(), // 完美质量
                QualityConfig {
                    blur_radius: 1.0,
                    salt_pepper_noise: 0.01,
                    ..Default::default()
                }, // 轻微退化
                QualityConfig {
                    blur_radius: 2.0,
                    salt_pepper_noise: 0.03,
                    contrast: 0.8,
                    brightness_offset: 20,
                    shadow_intensity: 0.3,
                    ..Default::default()
                }, // 中等退化
            ],
        }
    }
}

/// 生成单个测试图像及其 ground truth
pub fn generate_test_image(
    drawing_type: DrawingType,
    quality_config: &QualityConfig,
    width: u32,
    height: u32,
) -> GrayImage {
    let clean = match drawing_type {
        DrawingType::Architectural => generate_architectural_floorplan(width, height),
        DrawingType::Mechanical => {
            if fastrand::bool() {
                generate_mechanical_flange(width, height)
            } else {
                generate_mechanical_shaft(width, height)
            }
        }
        DrawingType::Circuit => generate_circuit_diagram(width, height),
        DrawingType::HandDrawn => {
            // 手绘效果：稍微增加线宽变化
            let mut img = generate_architectural_floorplan(width, height);
            // 简单添加一些随机扰动
            for _ in 0..100 {
                let x = fastrand::u32(0..width);
                let y = fastrand::u32(0..height);
                if img.get_pixel(x, y)[0] < 128 {
                    for dy in -1i32..=1 {
                        for dx in -1i32..=1 {
                            let px = x as i32 + dx;
                            let py = y as i32 + dy;
                            if px >= 0 && px < width as i32 && py >= 0 && py < height as i32 {
                                img.put_pixel(px as u32, py as u32, Luma([0]));
                            }
                        }
                    }
                }
            }
            img
        }
    };

    apply_quality_degradation(&clean, quality_config)
}

// ========== GNN 训练数据生成 ==========

/// 数据增强选项
#[derive(Debug, Clone)]
pub struct AugmentationConfig {
    /// 随机旋转角度范围 (度)
    pub rotation_range: f64,
    /// 随机平移范围 (像素)
    pub translation_range: f64,
    /// 随机缩放范围 (倍数)
    pub scale_min: f64,
    pub scale_max: f64,
    /// 是否添加噪声
    pub add_noise: bool,
    /// 是否随机反转
    pub random_flip: bool,
}

impl Default for AugmentationConfig {
    fn default() -> Self {
        Self {
            rotation_range: 5.0,    // ±5度
            translation_range: 5.0, // ±5像素
            scale_min: 0.95,
            scale_max: 1.05,
            add_noise: true,
            random_flip: true,
        }
    }
}

/// 训练样本
#[derive(Debug, Clone)]
pub struct TrainingSample {
    /// 图像数据
    pub image: GrayImage,
    /// 图纸类型
    pub drawing_type: DrawingType,
    /// 质量等级标签
    pub quality_level: String,
    /// 线坐标 (用于构建图)
    pub line_coords: Vec<((u32, u32), (u32, u32))>,
}

/// 生成单个训练样本（带增强）
pub fn generate_training_sample(
    drawing_type: DrawingType,
    width: u32,
    height: u32,
    augmentation: &AugmentationConfig,
) -> TrainingSample {
    use fastrand::f64 as rand_f64;

    // 基础质量配置
    let quality_config = QualityConfig {
        blur_radius: rand_f64() * 2.0,
        salt_pepper_noise: rand_f64() * 0.03,
        contrast: 0.85 + rand_f64() * 0.3,
        brightness_offset: (rand_f64() * 40.0 - 20.0) as i32,
        shadow_intensity: rand_f64() * 0.3,
        skew_angle: if augmentation.rotation_range > 0.0 {
            (rand_f64() - 0.5) * 2.0 * augmentation.rotation_range
        } else {
            0.0
        },
    };

    // 生成图像
    let image = generate_test_image(drawing_type, &quality_config, width, height);

    // 提取线坐标（简化实现：从图元生成器获取）
    let line_coords = match drawing_type {
        DrawingType::Architectural => generate_architectural_lines(width, height),
        DrawingType::Mechanical => generate_mechanical_lines(width, height),
        DrawingType::Circuit => generate_circuit_lines(width, height),
        DrawingType::HandDrawn => generate_handdrawn_lines(width, height),
    };

    // 确定质量等级标签
    let quality_level =
        if quality_config.blur_radius < 0.5 && quality_config.salt_pepper_noise < 0.005 {
            "perfect".to_string()
        } else if quality_config.blur_radius < 1.0 && quality_config.salt_pepper_noise < 0.02 {
            "light".to_string()
        } else {
            "medium".to_string()
        };

    TrainingSample {
        image,
        drawing_type,
        quality_level,
        line_coords,
    }
}

/// 生成建筑图线坐标
fn generate_architectural_lines(width: u32, height: u32) -> Vec<((u32, u32), (u32, u32))> {
    let mut lines = Vec::new();
    let margin = (width.min(height) / 10) as i32;

    // 外墙 - 矩形
    let x1 = margin as u32;
    let y1 = margin as u32;
    let x2 = (width as i32 - margin) as u32;
    let y2 = (height as i32 - margin) as u32;

    lines.push(((x1, y1), (x2, y1))); // 顶
    lines.push(((x2, y1), (x2, y2))); // 右
    lines.push(((x2, y2), (x1, y2))); // 底
    lines.push(((x1, y2), (x1, y1))); // 左

    // 内墙 - 简单网格
    let mid_x = (x1 + x2) / 2;
    let mid_y = (y1 + y2) / 2;
    lines.push(((mid_x, y1), (mid_x, y2)));
    lines.push(((x1, mid_y), (x2, mid_y)));

    lines
}

/// 生成机械图线坐标
fn generate_mechanical_lines(width: u32, height: u32) -> Vec<((u32, u32), (u32, u32))> {
    let cx = width / 2;
    let cy = height / 2;
    let radius = width.min(height) / 3;

    vec![
        // 外框
        ((cx - radius, cy - radius), (cx + radius, cy - radius)),
        ((cx + radius, cy - radius), (cx + radius, cy + radius)),
        ((cx + radius, cy + radius), (cx - radius, cy + radius)),
        ((cx - radius, cy + radius), (cx - radius, cy - radius)),
        // 十字中心线
        ((cx - radius, cy), (cx + radius, cy)),
        ((cx, cy - radius), (cx, cy + radius)),
    ]
}

/// 生成电路图线坐标
fn generate_circuit_lines(width: u32, height: u32) -> Vec<((u32, u32), (u32, u32))> {
    let margin = width.min(height) / 8;
    let mut lines = vec![
        // 外围走线
        ((margin, margin), (width - margin, margin)),
        ((width - margin, margin), (width - margin, height - margin)),
        ((width - margin, height - margin), (margin, height - margin)),
        ((margin, height - margin), (margin, margin)),
    ];

    // 内部走线
    let h_mid = height / 2;
    let v_mid = width / 2;
    lines.push(((margin, h_mid), (width - margin, h_mid)));
    lines.push(((v_mid, margin), (v_mid, height - margin)));

    lines
}

/// 生成手绘图线坐标
fn generate_handdrawn_lines(width: u32, height: u32) -> Vec<((u32, u32), (u32, u32))> {
    // 简单手绘风格 - 随机线段
    use fastrand::u32 as rand_u32;

    let mut lines = Vec::new();
    let num_lines = 8 + rand_u32(0..5);

    for _ in 0..num_lines {
        let x1 = rand_u32(20..width - 20);
        let y1 = rand_u32(20..height - 20);
        let x2 = rand_u32(20..width - 20);
        let y2 = rand_u32(20..height - 20);
        lines.push(((x1, y1), (x2, y2)));
    }

    lines
}

/// 批量生成训练数据集
pub fn generate_training_dataset(
    num_samples: usize,
    width: u32,
    height: u32,
) -> Vec<TrainingSample> {
    use fastrand::choice;

    let drawing_types = [
        DrawingType::Architectural,
        DrawingType::Mechanical,
        DrawingType::Circuit,
        DrawingType::HandDrawn,
    ];

    let aug_config = AugmentationConfig::default();

    (0..num_samples)
        .map(|_| {
            let drawing_type = choice(&drawing_types)
                .copied()
                .unwrap_or(DrawingType::Architectural);
            generate_training_sample(drawing_type, width, height, &aug_config)
        })
        .collect()
}

/// 数据集统计信息
#[derive(Debug, Clone, serde::Serialize)]
pub struct DatasetStats {
    pub total_samples: usize,
    pub architectural_count: usize,
    pub mechanical_count: usize,
    pub circuit_count: usize,
    pub handdrawn_count: usize,
    pub perfect_count: usize,
    pub light_count: usize,
    pub medium_count: usize,
}

/// 计算数据集统计信息
pub fn dataset_statistics(dataset: &[TrainingSample]) -> DatasetStats {
    let mut stats = DatasetStats {
        total_samples: dataset.len(),
        architectural_count: 0,
        mechanical_count: 0,
        circuit_count: 0,
        handdrawn_count: 0,
        perfect_count: 0,
        light_count: 0,
        medium_count: 0,
    };

    for sample in dataset {
        match sample.drawing_type {
            DrawingType::Architectural => stats.architectural_count += 1,
            DrawingType::Mechanical => stats.mechanical_count += 1,
            DrawingType::Circuit => stats.circuit_count += 1,
            DrawingType::HandDrawn => stats.handdrawn_count += 1,
        }

        match sample.quality_level.as_str() {
            "perfect" => stats.perfect_count += 1,
            "light" => stats.light_count += 1,
            "medium" => stats.medium_count += 1,
            _ => {}
        }
    }

    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_architectural() {
        let img = generate_architectural_floorplan(512, 512);
        assert_eq!(img.width(), 512);
        assert_eq!(img.height(), 512);
    }

    #[test]
    fn test_generate_mechanical_flange() {
        let img = generate_mechanical_flange(400, 400);
        assert_eq!(img.width(), 400);
        assert_eq!(img.height(), 400);
    }

    #[test]
    fn test_generate_mechanical_shaft() {
        let img = generate_mechanical_shaft(512, 512);
        assert_eq!(img.width(), 512);
        assert_eq!(img.height(), 512);
    }

    #[test]
    fn test_generate_circuit() {
        let img = generate_circuit_diagram(512, 512);
        assert_eq!(img.width(), 512);
        assert_eq!(img.height(), 512);
    }

    #[test]
    fn test_quality_degradation() {
        let clean = generate_architectural_floorplan(256, 256);
        let config = QualityConfig {
            blur_radius: 1.0,
            salt_pepper_noise: 0.01,
            contrast: 0.9,
            brightness_offset: 10,
            shadow_intensity: 0.2,
            skew_angle: 0.0,
        };
        let degraded = apply_quality_degradation(&clean, &config);
        assert_eq!(degraded.width(), 256);
        assert_eq!(degraded.height(), 256);
    }

    #[test]
    fn test_generate_test_image() {
        let config = QualityConfig::default();
        let img = generate_test_image(DrawingType::Architectural, &config, 256, 256);
        assert_eq!(img.width(), 256);
        assert_eq!(img.height(), 256);
    }

    #[test]
    fn test_generate_training_dataset() {
        let dataset = generate_training_dataset(5, 256, 256);
        assert_eq!(dataset.len(), 5);

        let stats = dataset_statistics(&dataset);
        assert_eq!(stats.total_samples, 5);
    }
}
