/// 文件类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(super) enum FileType {
    Dxf,
    Pdf,
    Png,
    Jpeg,
    Bmp,
    Tiff,
    WebP,
    Unknown,
}

impl FileType {
    /// 是否为光栅图片格式
    pub(super) fn is_raster(self) -> bool {
        matches!(
            self,
            FileType::Png | FileType::Jpeg | FileType::Bmp | FileType::Tiff | FileType::WebP
        )
    }
}

/// 检测文件类型
pub(super) fn detect_file_type(data: &[u8], file_name: Option<&str>) -> FileType {
    if let Some(name) = file_name {
        let ext = std::path::Path::new(name)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("")
            .to_lowercase();

        match ext.as_str() {
            "dxf" => return FileType::Dxf,
            "pdf" => return FileType::Pdf,
            "png" => return FileType::Png,
            "jpg" | "jpeg" => return FileType::Jpeg,
            "bmp" => return FileType::Bmp,
            "tif" | "tiff" => return FileType::Tiff,
            "webp" => return FileType::WebP,
            _ => {}
        }
    }

    if data.len() >= 4 {
        if data.starts_with(&[0x89, 0x50, 0x4E, 0x47]) {
            return FileType::Png;
        }
        if data.starts_with(&[0xFF, 0xD8, 0xFF]) {
            return FileType::Jpeg;
        }
        if data.starts_with(&[0x42, 0x4D]) {
            return FileType::Bmp;
        }
        if data.starts_with(&[0x49, 0x49, 0x2A, 0x00]) {
            return FileType::Tiff;
        }
        if data.starts_with(&[0x4D, 0x4D, 0x00, 0x2A]) {
            return FileType::Tiff;
        }
        if data.len() >= 12 && data.starts_with(b"RIFF") && &data[8..12] == b"WEBP" {
            return FileType::WebP;
        }
    }

    if data.starts_with(b"%PDF") {
        return FileType::Pdf;
    }

    if data.starts_with(b"AutoCAD")
        || data.starts_with(b"SECTION")
        || data.starts_with(&[0x41, 0x43, 0x31, 0x30])
    {
        return FileType::Dxf;
    }

    if std::str::from_utf8(data).is_ok_and(|s| s.contains("SECTION") && s.contains("ENTITIES")) {
        return FileType::Dxf;
    }

    FileType::Unknown
}

/// 生成简单 UUID（用于 job_id）
pub(super) fn uuid_simple() -> String {
    use uuid::Uuid;
    format!("job-{}", Uuid::new_v4().to_string().replace('-', ""))
}

pub(super) fn parse_bool(value: &str) -> bool {
    matches!(
        value.trim().to_ascii_lowercase().as_str(),
        "1" | "true" | "yes" | "on"
    )
}

pub(super) fn parse_pair(value: &str) -> Option<(f64, f64)> {
    let normalized = value.replace(['x', ';'], ",");
    let mut parts = normalized.split(',').map(str::trim);
    let x = parts.next()?.parse::<f64>().ok()?;
    let y = parts
        .next()
        .and_then(|part| part.parse::<f64>().ok())
        .unwrap_or(x);
    Some((x, y))
}

/// 将 FileType 转换为文件扩展名
pub(super) fn file_type_extension(file_type: &FileType) -> &'static str {
    match file_type {
        FileType::Dxf => "dxf",
        FileType::Pdf => "pdf",
        FileType::Png => "png",
        FileType::Jpeg => "jpg",
        FileType::Bmp => "bmp",
        FileType::Tiff => "tiff",
        FileType::WebP => "webp",
        FileType::Unknown => "unknown",
    }
}
