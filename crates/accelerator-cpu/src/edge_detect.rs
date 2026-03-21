//! CPU 边缘检测实现

use accelerator_api::{EdgeDetectConfig, EdgeMap, Image};
use accelerator_api::{AcceleratorResult, AcceleratorError};
use rayon::prelude::*;

/// CPU 边缘检测（Sobel 算子）
pub fn detect_edges_cpu(image: &Image, config: &EdgeDetectConfig) -> AcceleratorResult<EdgeMap> {
    let width = image.width as usize;
    let height = image.height as usize;
    
    if image.data.len() != width * height {
        return Err(AcceleratorError::InvalidDataFormat(
            format!("图像数据大小不匹配：期望 {}，实际 {}", width * height, image.data.len())
        ));
    }

    let mut edge_data = vec![255u8; width * height];
    let sobel_x = [-1i32, 0, 1, -2, 0, 2, -1, 0, 1];
    let sobel_y = [-1i32, -2, -1, 0, 0, 0, 1, 2, 1];

    // 并行处理每一行
    edge_data
        .par_chunks_mut(width)
        .enumerate()
        .skip(1)
        .take(height - 2)
        .for_each(|(y, row)| {
            for (x, pixel) in row.iter_mut().enumerate().take(width - 1).skip(1) {
                let mut gx = 0i32;
                let mut gy = 0i32;
                let mut idx = 0;

                for dy in -1isize..=1isize {
                    for dx in -1isize..=1isize {
                        let nx = x as isize + dx;
                        let ny = y as isize + dy;
                        let pixel_val = image.data[ny as usize * width + nx as usize] as i32;
                        gx += pixel_val * sobel_x[idx];
                        gy += pixel_val * sobel_y[idx];
                        idx += 1;
                    }
                }

                let magnitude = ((gx * gx + gy * gy) as f32).sqrt() as u8;
                
                // 应用阈值
                let threshold = config.low_threshold as u8;
                *pixel = if magnitude > threshold { 0 } else { 255 };
            }
        });

    Ok(EdgeMap {
        width: image.width,
        height: image.height,
        data: edge_data,
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_edge_detect_simple() {
        let image = Image {
            width: 10,
            height: 10,
            data: vec![128u8; 100],
        };
        
        let config = EdgeDetectConfig::default();
        let result = detect_edges_cpu(&image, &config).unwrap();
        
        assert_eq!(result.width, 10);
        assert_eq!(result.height, 10);
    }
}
