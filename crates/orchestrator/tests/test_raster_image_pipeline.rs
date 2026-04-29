use image::DynamicImage;
use orchestrator::ProcessingPipeline;
use std::io::Cursor;
use vectorize::test_data::{generate_test_image, DrawingType, QualityConfig};

fn make_test_png() -> Vec<u8> {
    let img = generate_test_image(
        DrawingType::Architectural,
        &QualityConfig::default(),
        256,
        256,
    );
    let mut cursor = Cursor::new(Vec::new());
    DynamicImage::ImageLuma8(img)
        .write_to(&mut cursor, image::ImageFormat::Png)
        .unwrap();
    cursor.into_inner()
}

fn raster_test_pipeline() -> ProcessingPipeline {
    ProcessingPipeline::new()
}

#[tokio::test]
async fn process_raster_bytes_png_end_to_end() {
    let png = make_test_png();
    let pipeline = raster_test_pipeline();

    let result = pipeline
        .process_raster_bytes(&png, Some("square.png"))
        .await
        .unwrap();

    assert!(!result.output_bytes.is_empty());
    assert!(!result.scene.edges.is_empty());
}

#[tokio::test]
async fn process_raster_file_png_end_to_end() {
    let png = make_test_png();
    let path = std::env::temp_dir().join(format!(
        "orchestrator_raster_pipeline_{}_square.png",
        std::process::id()
    ));
    std::fs::write(&path, png).unwrap();

    let pipeline = raster_test_pipeline();
    let result = pipeline.process_raster_file(&path).await.unwrap();
    std::fs::remove_file(&path).ok();

    assert!(!result.output_bytes.is_empty());
    assert!(!result.scene.edges.is_empty());
}

#[tokio::test]
async fn process_raster_bytes_rejects_invalid_data() {
    let pipeline = ProcessingPipeline::new();
    let result = pipeline
        .process_raster_bytes(b"not a raster image", Some("txt"))
        .await;

    assert!(result.is_err());
}
