//! Synthetic raster benchmark data for CI.

use common_types::{Point2, Polyline};
use image::{DynamicImage, GrayImage, Luma};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RasterDegradation {
    CleanLineArt,
    ScanNoise,
    JpegCompression,
    PhotoPerspective,
    LowContrast,
    HandSketch,
}

#[derive(Debug, Clone)]
pub struct GroundTruth {
    pub polylines: Vec<Polyline>,
    pub expected_outer_loops: usize,
    pub expected_holes: usize,
    pub dpi: Option<(f64, f64)>,
}

#[derive(Debug, Clone)]
pub struct RasterBenchmarkCase {
    pub name: String,
    pub image: DynamicImage,
    pub ground_truth: GroundTruth,
    pub degradation: RasterDegradation,
}

pub fn small_ci_set() -> Vec<RasterBenchmarkCase> {
    [
        RasterDegradation::CleanLineArt,
        RasterDegradation::ScanNoise,
        RasterDegradation::JpegCompression,
        RasterDegradation::PhotoPerspective,
        RasterDegradation::LowContrast,
        RasterDegradation::HandSketch,
    ]
    .into_iter()
    .map(generate_case)
    .collect()
}

pub fn generate_case(degradation: RasterDegradation) -> RasterBenchmarkCase {
    let mut image = GrayImage::from_pixel(160, 120, Luma([255]));
    let mut truth = vec![
        vec![[20.0, 20.0], [140.0, 20.0]],
        vec![[140.0, 20.0], [140.0, 100.0]],
        vec![[140.0, 100.0], [20.0, 100.0]],
        vec![[20.0, 100.0], [20.0, 20.0]],
        vec![[72.0, 20.0], [72.0, 100.0]],
        vec![[20.0, 62.0], [140.0, 62.0]],
    ];

    for line in &truth {
        draw_line(&mut image, line[0], line[1], 0, 1);
    }

    match degradation {
        RasterDegradation::CleanLineArt => {}
        RasterDegradation::ScanNoise => add_noise(&mut image, 17),
        RasterDegradation::JpegCompression => quantize(&mut image, 32),
        RasterDegradation::PhotoPerspective => {
            truth.push(vec![[28.0, 28.0], [134.0, 18.0]]);
            draw_line(&mut image, [28.0, 28.0], [134.0, 18.0], 0, 1);
            add_noise(&mut image, 29);
        }
        RasterDegradation::LowContrast => {
            for px in image.pixels_mut() {
                px[0] = if px[0] < 128 { 92 } else { 188 };
            }
        }
        RasterDegradation::HandSketch => {
            image = GrayImage::from_pixel(160, 120, Luma([250]));
            for line in &truth {
                draw_line(&mut image, [line[0][0], line[0][1] + 1.0], line[1], 35, 2);
            }
            add_noise(&mut image, 41);
        }
    }

    RasterBenchmarkCase {
        name: format!("{:?}", degradation).to_ascii_lowercase(),
        image: DynamicImage::ImageLuma8(image),
        ground_truth: GroundTruth {
            polylines: truth,
            expected_outer_loops: 1,
            expected_holes: 0,
            dpi: Some((300.0, 300.0)),
        },
        degradation,
    }
}

pub fn geometry_score(extracted: &[Polyline], truth: &GroundTruth) -> (f64, f64) {
    let extracted_segments = extracted.iter().filter(|line| line.len() >= 2).count() as f64;
    let truth_segments = truth.polylines.len().max(1) as f64;
    let recall = (extracted_segments / truth_segments).min(1.0);
    let precision = (truth_segments / extracted_segments.max(1.0)).min(1.0);
    (precision, recall)
}

fn draw_line(image: &mut GrayImage, start: Point2, end: Point2, value: u8, width: i32) {
    let mut x0 = start[0].round() as i32;
    let mut y0 = start[1].round() as i32;
    let x1 = end[0].round() as i32;
    let y1 = end[1].round() as i32;
    let dx = (x1 - x0).abs();
    let sx = if x0 < x1 { 1 } else { -1 };
    let dy = -(y1 - y0).abs();
    let sy = if y0 < y1 { 1 } else { -1 };
    let mut err = dx + dy;

    loop {
        for oy in -width..=width {
            for ox in -width..=width {
                let x = x0 + ox;
                let y = y0 + oy;
                if x >= 0 && y >= 0 && x < image.width() as i32 && y < image.height() as i32 {
                    image.put_pixel(x as u32, y as u32, Luma([value]));
                }
            }
        }
        if x0 == x1 && y0 == y1 {
            break;
        }
        let e2 = 2 * err;
        if e2 >= dy {
            err += dy;
            x0 += sx;
        }
        if e2 <= dx {
            err += dx;
            y0 += sy;
        }
    }
}

fn add_noise(image: &mut GrayImage, seed: u32) {
    for (idx, px) in image.pixels_mut().enumerate() {
        let v = ((idx as u32).wrapping_mul(1103515245).wrapping_add(seed) >> 16) & 0xff;
        if v < 3 {
            px[0] = 0;
        } else if v > 252 {
            px[0] = 255;
        }
    }
}

fn quantize(image: &mut GrayImage, step: u8) {
    for px in image.pixels_mut() {
        px[0] = (px[0] / step) * step;
    }
}
