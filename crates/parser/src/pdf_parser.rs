//! PDF 文件解析器
//!
//! 支持矢量 PDF 和光栅 PDF 的解析
//!
//! ## PDF 图形操作符参考
//! - 路径构造：m (moveto), l (lineto), c/be (curveto), h (closepath)
//! - 路径绘制：S/s (stroke), f/F (fill), B/b (fill+stroke)
//! - 变换矩阵：cm (concat matrix)
//! - 图形状态：q/Q (save/restore)

use common_types::{RawEntity, CadError, Point2, EntityMetadata, PathCommand, PdfParseReason};
use lopdf::{Document, Object};
use std::path::{Path, PathBuf};

pub struct PdfParser {
    /// DPI 阈值，低于此值认为是光栅 PDF
    _raster_dpi_threshold: f64,
}

impl PdfParser {
    pub fn new() -> Self {
        Self {
            _raster_dpi_threshold: 150.0,
        }
    }

    /// 从文件路径解析 PDF
    pub fn parse_file(&self, path: impl AsRef<Path>) -> Result<PdfContent, CadError> {
        let path_buf = path.as_ref().to_path_buf();
        let doc = Document::load(path.as_ref())
            .map_err(|e| CadError::pdf_parse_with_source(
                &path_buf,
                PdfParseReason::FileNotFound,
                e,
            ))?;
        Self::new().parse_document(&doc)
    }

    /// 从字节解析 PDF
    pub fn parse_bytes(&self, bytes: &[u8]) -> Result<PdfContent, CadError> {
        let doc = Document::load_mem(bytes)
            .map_err(|e| CadError::pdf_parse_with_source(
                PathBuf::from("<bytes>"),
                PdfParseReason::ExtractError("PDF 读取失败".to_string()),
                e,
            ))?;
        Self::new().parse_document(&doc)
    }

    fn parse_document(&self, doc: &Document) -> Result<PdfContent, CadError> {
        use rayon::prelude::*;

        let mut raster_images = Vec::new();

        // 并行解析所有页面
        let pages = doc.get_pages();
        let page_results: Vec<_> = pages
            .par_iter()
            .filter_map(|(_page_id, page_id)| {
                let page = doc.get_object(*page_id)
                    .ok()
                    .and_then(|o| o.as_dict().ok())?;

                // 提取页面内容流（支持数组和单个流）
                let entities = self.extract_content_streams(page, doc);

                // 提取页面资源中的图像
                let mut page_images = Vec::new();
                if let Ok(resources) = page.get(b"Resources").and_then(|r| r.as_dict()) {
                    if let Ok(xobjs) = resources.get(b"XObject").and_then(|x| x.as_dict()) {
                        for (name, xobj) in xobjs.iter() {
                            if let Ok(xobj_dict) = xobj.as_dict() {
                                if let Ok(subtype) = xobj_dict.get(b"Subtype").and_then(|s| s.as_name()) {
                                    if subtype == b"Image" {
                                        if let Some(img) = self.extract_image(name, xobj_dict, doc) {
                                            page_images.push(img);
                                        }
                                    }
                                }
                            }
                        }
                    }
                }

                Some((entities, page_images))
            })
            .collect();

        // 合并所有结果
        let mut vector_entities = Vec::new();
        for (entities, images) in page_results {
            vector_entities.extend(entities);
            raster_images.extend(images);
        }

        let is_vector = !vector_entities.is_empty();

        Ok(PdfContent {
            vector_entities,
            raster_images,
            is_vector,
        })
    }

    /// 提取内容流（支持单个流或流数组）
    fn extract_content_streams(&self, page: &lopdf::Dictionary, doc: &Document) -> Vec<RawEntity> {
        let mut entities = Vec::new();
        
        if let Ok(contents) = page.get(b"Contents") {
            match contents {
                Object::Reference(id) => {
                    // 单个内容流
                    if let Ok(obj) = doc.get_object(*id) {
                        entities.extend(self.parse_content_object(obj, doc));
                    }
                }
                Object::Array(arr) => {
                    // 多个内容流数组
                    for item in arr {
                        if let Object::Reference(id) = item {
                            if let Ok(obj) = doc.get_object(*id) {
                                entities.extend(self.parse_content_object(obj, doc));
                            }
                        }
                    }
                }
                Object::Stream(stream) => {
                    // 直接是流对象
                    entities.extend(self.parse_stream(stream));
                }
                _ => {}
            }
        }
        
        entities
    }

    /// 解析内容对象
    fn parse_content_object(&self, obj: &Object, doc: &Document) -> Vec<RawEntity> {
        match obj {
            Object::Stream(stream) => self.parse_stream(stream),
            Object::Array(arr) => {
                arr.iter()
                    .flat_map(|item| self.parse_content_object(item, doc))
                    .collect()
            }
            Object::Reference(id) => {
                if let Ok(obj) = doc.get_object(*id) {
                    self.parse_content_object(obj, doc)
                } else {
                    Vec::new()
                }
            }
            _ => Vec::new(),
        }
    }

    /// 解析流内容
    fn parse_stream(&self, stream: &lopdf::Stream) -> Vec<RawEntity> {
        let mut entities = Vec::new();
        if let Ok(content) = stream.decompressed_content() {
            entities.extend(self.parse_operators(&content));
        }
        entities
    }

    /// 解析 PDF 操作符
    /// 
    /// PDF 图形操作符参考：
    /// - 路径构造：m (moveto), l (lineto), c/be (curveto), h (closepath)  
    /// - 路径绘制：S/s (stroke), f/F (fill), B/b (fill+stroke)
    /// - 图形状态：q/Q (save/restore), cm (transform)
    fn parse_operators(&self, content: &[u8]) -> Vec<RawEntity> {
        let mut entities = Vec::new();
        let content_str = String::from_utf8_lossy(content);

        // PDF 路径绘制状态机
        let mut path_points: Vec<Point2> = Vec::new();
        let mut current_point: Point2 = [0.0, 0.0];
        
        // 使用更健壮的解析方式：按 token 解析而非按行
        let tokens: Vec<&str> = content_str.split_whitespace().collect();
        let mut i = 0;
        
        while i < tokens.len() {
            // 尝试解析数字
            if let Ok(num) = tokens[i].parse::<f64>() {
                // 可能是坐标的开始
                if i + 1 < tokens.len() {
                    if let Ok(num2) = tokens[i + 1].parse::<f64>() {
                        current_point = [num, num2];
                        i += 2;
                        continue;
                    }
                }
            }
            
            // 解析操作符
            let op = tokens[i];
            match op {
                "m" => {
                    // moveto: x y m - 开始新路径
                    path_points = vec![current_point];
                }
                "l" => {
                    // lineto: x y l - 画直线
                    if !path_points.is_empty() {
                        path_points.push(current_point);
                    }
                }
                "c" | "be" => {
                    // curveto: x1 y1 x2 y2 x3 y3 c - 贝塞尔曲线（简化为直线）
                    // 跳过 6 个数字（3 个点）
                    if i + 6 < tokens.len() {
                        if let (Ok(x), Ok(y)) = (
                            tokens[i + 5].parse::<f64>(),
                            tokens[i + 6].parse::<f64>()
                        ) {
                            path_points.push([x, y]);
                        }
                        i += 6;
                    }
                }
                "h" => {
                    // closepath: 闭合路径
                    if path_points.len() >= 2 {
                        entities.push(self.create_path_entity(path_points.clone(), true));
                    }
                    path_points.clear();
                }
                "S" | "s" => {
                    // stroke: 描边当前路径
                    if path_points.len() >= 2 {
                        entities.push(self.create_path_entity(path_points.clone(), false));
                    }
                    path_points.clear();
                }
                "f" | "F" | "f*" => {
                    // fill: 填充当前路径
                    if path_points.len() >= 2 {
                        entities.push(self.create_path_entity(path_points.clone(), false));
                    }
                    path_points.clear();
                }
                "B" | "b" | "B*" => {
                    // fill+stroke: 填充并描边
                    if path_points.len() >= 2 {
                        entities.push(self.create_path_entity(path_points.clone(), false));
                    }
                    path_points.clear();
                }
                "q" => {
                    // save graphics state - 忽略
                }
                "Q" => {
                    // restore graphics state - 忽略
                }
                "cm" => {
                    // transform matrix - 简化处理，暂不应用变换
                    i += 6; // 跳过 6 个矩阵参数
                }
                "re" => {
                    // rectangle: x y w h re - 矩形（简化为 4 个点的闭合路径）
                    if i >= 3 {
                        if let (Ok(x), Ok(y), Ok(w), Ok(h)) = (
                            tokens[i - 4].parse::<f64>(),
                            tokens[i - 3].parse::<f64>(),
                            tokens[i - 2].parse::<f64>(),
                            tokens[i - 1].parse::<f64>()
                        ) {
                            path_points = vec![
                                [x, y],
                                [x + w, y],
                                [x + w, y + h],
                                [x, y + h],
                            ];
                            entities.push(self.create_path_entity(path_points.clone(), true));
                            path_points.clear();
                        }
                    }
                }
                _ => {
                    // 未知操作符，跳过
                }
            }
            
            i += 1;
        }

        // 处理未闭合的路径（如果有 stroke/fill 操作符之前的路径）
        if path_points.len() >= 2 {
            entities.push(self.create_path_entity(path_points, false));
        }

        entities
    }

    /// 创建路径实体
    fn create_path_entity(&self, points: Vec<Point2>, closed: bool) -> RawEntity {
        let mut commands: Vec<PathCommand> = Vec::new();
        
        if points.is_empty() {
            return RawEntity::Path {
                commands,
                metadata: EntityMetadata::new(),
                semantic: None,
            };
        }
        
        // MoveTo 起点
        commands.push(PathCommand::MoveTo { 
            x: points[0][0], 
            y: points[0][1] 
        });
        
        // LineTo 后续点
        for point in &points[1..] {
            commands.push(PathCommand::LineTo { 
                x: point[0], 
                y: point[1] 
            });
        }
        
        // 闭合路径
        if closed {
            commands.push(PathCommand::Close);
        }

        RawEntity::Path {
            commands,
            metadata: EntityMetadata::new(),
            semantic: None,
        }
    }

    /// 提取图像（支持 FlateDecode/DCTDecode 等过滤器）
    fn extract_image(&self, name: &[u8], dict: &lopdf::Dictionary, doc: &Document) -> Option<RasterImage> {
        // 获取图像尺寸
        let width = dict.get(b"Width").ok()?.as_i64().ok()? as u32;
        let height = dict.get(b"Height").ok()?.as_i64().ok()? as u32;

        // 获取图像数据流 - 图像 XObject 本身是流对象
        // 在 doc 中查找对应的流对象（通过遍历所有对象）
        let raw_data = doc
            .objects
            .values()
            .filter_map(|obj| obj.as_stream().ok())
            .find_map(|stream: &lopdf::Stream| {
                // 检查是否是这个图像字典对应的流
                let stream_width = stream.dict.get(b"Width").and_then(|v| v.as_i64()).ok();
                let stream_height = stream.dict.get(b"Height").and_then(|v| v.as_i64()).ok();
                
                if stream_width == Some(width as i64) && stream_height == Some(height as i64) {
                    stream.decompressed_content().ok()
                } else {
                    None
                }
            })
            .unwrap_or_default();

        // 获取过滤器类型并解码
        let filter = dict.get(b"Filter").ok();
        let data = match filter {
            Some(Object::Name(filter_name)) => {
                self.decode_image_data(&raw_data, filter_name.as_slice())
            }
            Some(Object::Array(filters)) => {
                let mut current_data = raw_data;
                for filter in filters {
                    if let Ok(filter_name) = filter.as_name() {
                        current_data = self.decode_image_data(&current_data, filter_name);
                    }
                }
                current_data
            }
            _ => raw_data,
        };

        // 获取 DPI（如果有）
        let mut dpi_x = 72.0;
        let mut dpi_y = 72.0;

        if let Ok(x) = dict.get(b"XResolution").and_then(|v| v.as_i64()) {
            dpi_x = x as f64;
        }
        if let Ok(y) = dict.get(b"YResolution").and_then(|v| v.as_i64()) {
            dpi_y = y as f64;
        }

        Some(RasterImage {
            name: String::from_utf8_lossy(name).to_string(),
            data,
            width,
            height,
            dpi_x,
            dpi_y,
        })
    }

    /// 解码图像数据（支持 FlateDecode/DCTDecode/LZW/RunLength/ASCIIHex/ASCII85）
    fn decode_image_data(&self, data: &[u8], filter_name: &[u8]) -> Vec<u8> {
        match filter_name {
            b"FlateDecode" => {
                // FlateDecode (DEFLATE 压缩，类似 zlib)
                use std::io::Read;
                let mut decoder = flate2::read::ZlibDecoder::new(data);
                let mut decoded = Vec::new();
                if decoder.read_to_end(&mut decoded).is_ok() {
                    return decoded;
                }
                // 解码失败，返回原始数据
                data.to_vec()
            }
            b"DCTDecode" => {
                // DCTDecode (JPEG 压缩)，直接返回 JPEG 数据
                data.to_vec()
            }
            b"CCITTFaxDecode" => {
                // CCITT Fax 压缩，保持原始数据
                data.to_vec()
            }
            b"JBIG2Decode" => {
                // JBIG2 压缩，保持原始数据
                data.to_vec()
            }
            b"JPXDecode" => {
                // JPEG2000 压缩，保持原始数据
                data.to_vec()
            }
            b"LZWDecode" => {
                // LZW 压缩，使用 weezl 库解码
                use weezl::decode::Decoder;
                use weezl::BitOrder;
                
                // PDF LZW 使用 MSB 优先（big-endian）位序，初始码宽 9 位
                let mut decoder = Decoder::new(BitOrder::Msb, 9);

                // weezl 是流式解码器，需要逐步喂数据
                match decoder.decode(data) {
                    Ok(decoded) => decoded,
                    Err(_) => {
                        // 解码失败，返回原始数据
                        data.to_vec()
                    }
                }
            }
            b"RunLengthDecode" => {
                // RunLength 编码
                self.decode_run_length(data)
            }
            b"ASCIIHexDecode" => {
                // ASCII Hex 编码
                self.decode_ascii_hex(data)
            }
            b"ASCII85Decode" => {
                // ASCII85 编码
                self.decode_ascii85(data)
            }
            _ => {
                // 未知过滤器，返回原始数据
                data.to_vec()
            }
        }
    }

    /// 解码 RunLength 编码
    fn decode_run_length(&self, data: &[u8]) -> Vec<u8> {
        let mut decoded = Vec::new();
        let mut i = 0;

        while i < data.len() {
            let count = data[i] as i8;
            i += 1;

            if count >= 0 {
                // 0-127: 复制接下来的 count+1 字节
                let n = (count + 1) as usize;
                if i + n <= data.len() {
                    decoded.extend_from_slice(&data[i..i + n]);
                    i += n;
                } else {
                    break;
                }
            } else if count != -128 {
                // -1 到 -127: 重复下一个字节 (1-count) 次
                if i < data.len() {
                    let byte = data[i];
                    let n = (1 - count) as usize;
                    for _ in 0..n {
                        decoded.push(byte);
                    }
                    i += 1;
                } else {
                    break;
                }
            } else {
                // -128: EOD 标记
                break;
            }
        }

        decoded
    }

    /// 解码 ASCII Hex 编码
    fn decode_ascii_hex(&self, data: &[u8]) -> Vec<u8> {
        let mut decoded = Vec::new();
        let mut hex_pairs = String::new();

        for &byte in data {
            match byte {
                b'0'..=b'9' | b'a'..=b'f' | b'A'..=b'F' => {
                    hex_pairs.push(byte as char);
                }
                b'>' => {
                    // EOD 标记
                    break;
                }
                _ => {
                    // 跳过空白字符
                    continue;
                }
            }
        }

        // 每两个十六进制字符转换为一个字节
        for i in (0..hex_pairs.len()).step_by(2) {
            if i + 1 < hex_pairs.len() {
                if let Ok(byte) = u8::from_str_radix(&hex_pairs[i..i + 2], 16) {
                    decoded.push(byte);
                }
            }
        }

        decoded
    }

    /// 解码 ASCII85 编码
    fn decode_ascii85(&self, data: &[u8]) -> Vec<u8> {
        let mut decoded = Vec::new();
        let mut group = Vec::new();

        for &byte in data {
            match byte {
                b'z' => {
                    // 'z' 表示 4 个零字节
                    decoded.extend_from_slice(&[0, 0, 0, 0]);
                }
                b'!'..=b'u' => {
                    group.push(byte as i32 - 33);
                    if group.len() == 5 {
                        // 解码 5 个字符为 4 个字节
                        let value = group.iter().fold(0i64, |acc, &x| acc * 85 + x as i64);
                        decoded.extend_from_slice(&[
                            ((value >> 24) & 0xFF) as u8,
                            ((value >> 16) & 0xFF) as u8,
                            ((value >> 8) & 0xFF) as u8,
                            (value & 0xFF) as u8,
                        ]);
                        group.clear();
                    }
                }
                b'~' => {
                    // EOD 标记
                    break;
                }
                _ => {
                    // 跳过空白字符
                    continue;
                }
            }
        }

        // 处理最后一组（如果有）
        if !group.is_empty() {
            // 填充到 5 个字符
            while group.len() < 5 {
                group.push(84); // 'u' - 33
            }
            let value = group.iter().fold(0i64, |acc, &x| acc * 85 + x as i64);
            let n = group.len() - 1;
            for i in 0..n {
                decoded.push(((value >> (24 - i * 8)) & 0xFF) as u8);
            }
        }

        decoded
    }
}

impl Default for PdfParser {
    fn default() -> Self {
        Self::new()
    }
}

/// PDF 解析结果
#[derive(Debug, Clone)]
pub struct PdfContent {
    /// 矢量实体
    pub vector_entities: Vec<RawEntity>,
    /// 光栅图像
    pub raster_images: Vec<RasterImage>,
    /// 是否为矢量 PDF
    pub is_vector: bool,
}

impl PdfContent {
    /// 判定 PDF 类型（矢量/光栅/混合）
    pub fn content_type(&self) -> PdfContentType {
        match (self.is_vector, self.raster_images.is_empty()) {
            (true, true) => PdfContentType::Vector,
            (false, false) => PdfContentType::Raster,
            (true, false) => PdfContentType::Mixed,
            (false, true) => PdfContentType::Unknown,
        }
    }

    /// 检查是否为光栅 PDF（需要矢量化处理）
    pub fn needs_vectorization(&self) -> bool {
        !self.is_vector && !self.raster_images.is_empty()
    }

    /// 获取推荐的 DPI（用于光栅 PDF 矢量化）
    pub fn recommended_dpi(&self) -> Option<f64> {
        if self.raster_images.is_empty() {
            return None;
        }
        // 取所有图像 DPI 的平均值
        let sum: f64 = self.raster_images.iter()
            .map(|img| (img.dpi_x + img.dpi_y) / 2.0)
            .sum();
        Some(sum / self.raster_images.len() as f64)
    }

    /// 转换为使用 PdfRasterImage 的格式
    pub fn to_pdf_raster_images(&self) -> Vec<common_types::PdfRasterImage> {
        self.raster_images
            .iter()
            .map(|img| img.to_pdf_raster_image())
            .collect()
    }
}

/// PDF 内容类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PdfContentType {
    /// 纯矢量 PDF
    Vector,
    /// 纯光栅 PDF
    Raster,
    /// 混合类型（既有矢量又有光栅）
    Mixed,
    /// 未知类型
    Unknown,
}

/// 光栅图像信息
#[derive(Debug, Clone)]
pub struct RasterImage {
    pub name: String,
    pub data: Vec<u8>,
    pub width: u32,
    pub height: u32,
    pub dpi_x: f64,
    pub dpi_y: f64,
}

impl RasterImage {
    /// 转换为 PdfRasterImage
    pub fn to_pdf_raster_image(&self) -> common_types::PdfRasterImage {
        common_types::pdf_raster_from_parser_raster(
            self.name.clone(),
            self.width,
            self.height,
            &self.data,
            self.dpi_x,
            self.dpi_y,
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pdf_parser_creation() {
        let _parser = PdfParser::new();
        // PdfParser 现在没有 config 字段，测试基本创建即可
    }

    #[test]
    fn test_parse_bytes_minimal_pdf() {
        // 测试解析器能处理最小 PDF 文件
        let parser = PdfParser::new();
        
        // 创建一个最小的有效 PDF 字节数组
        let pdf_bytes = create_minimal_pdf();
        
        let result = parser.parse_bytes(&pdf_bytes);
        // 解析应该成功
        assert!(result.is_ok());
        
        // PDF 解析器目前功能有限，只要能成功解析即可
        // 主要测试解析器能正常工作
    }

    #[test]
    fn test_parse_bytes_invalid_pdf() {
        let parser = PdfParser::new();
        
        // 无效的 PDF 数据
        let invalid_bytes = b"not a pdf file at all";
        
        let result = parser.parse_bytes(invalid_bytes);
        // 应该返回错误
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_bytes_empty() {
        let parser = PdfParser::new();

        // 空字节
        let result = parser.parse_bytes(&[]);
        // 应该返回错误
        assert!(result.is_err());
    }

    #[test]
    fn test_parse_pdf_with_flate_decode() {
        // 测试解析带 FlateDecode 压缩的 PDF 内容流
        use lopdf::{Document, Object, Stream};
        use lopdf::dictionary;
        use flate2::write::ZlibEncoder;
        use flate2::Compression;
        use std::io::Write;

        let mut doc = Document::with_version("1.4");

        // 创建压缩内容流
        let content = b"10 20 m\n30 40 l\n50 60 l\nh\nS";
        let mut encoder = ZlibEncoder::new(Vec::new(), Compression::default());
        encoder.write_all(content).unwrap();
        let compressed_content = encoder.finish().unwrap();

        // 创建带 FlateDecode 过滤器的流
        let mut content_stream = Stream::new(dictionary! {
            "Filter" => "FlateDecode",
        }, compressed_content);
        content_stream.set_plain_content("10 20 m\n30 40 l\n50 60 l\nh\nS".as_bytes().to_vec());
        doc.set_object((4, 0), content_stream);

        // 创建页面对象
        doc.set_object((3, 0), dictionary! {
            "Type" => "Page",
            "Parent" => Object::Reference((2, 0)),
            "MediaBox" => Object::Array(vec![
                Object::Integer(0),
                Object::Integer(0),
                Object::Integer(595),
                Object::Integer(842),
            ]),
            "Contents" => Object::Reference((4, 0)),
        });

        // 创建 Pages 对象
        doc.set_object((2, 0), dictionary! {
            "Type" => "Pages",
            "Kids" => Object::Array(vec![Object::Reference((3, 0))]),
            "Count" => Object::Integer(1),
        });

        // 创建 Catalog 对象
        doc.set_object((1, 0), dictionary! {
            "Type" => "Catalog",
            "Pages" => Object::Reference((2, 0)),
        });

        doc.trailer.set("Root", Object::Reference((1, 0)));

        let mut buffer = Vec::new();
        doc.save_to(&mut buffer).unwrap();

        // 解析压缩的 PDF
        let parser = PdfParser::new();
        let result = parser.parse_bytes(&buffer);

        // 解析应该成功
        assert!(result.is_ok(), "带 FlateDecode 的 PDF 应该能成功解析");
    }

    /// 创建一个最小的 PDF 文件用于测试
    fn create_minimal_pdf() -> Vec<u8> {
        use lopdf::{Document, Object, Stream, Dictionary};
        use lopdf::dictionary;

        let mut doc = Document::with_version("1.4");

        // 创建内容流
        let content = b"10 20 m\n30 40 l\n50 60 l\nh\nS";
        let content_stream = Stream::new(Dictionary::new(), content.to_vec());
        doc.set_object((4, 0), content_stream);

        // 创建页面对象
        doc.set_object((3, 0), dictionary! {
            "Type" => "Page",
            "Parent" => Object::Reference((2, 0)),
            "MediaBox" => Object::Array(vec![
                Object::Integer(0),
                Object::Integer(0),
                Object::Integer(595),
                Object::Integer(842),
            ]),
            "Contents" => Object::Reference((4, 0)),
        });

        // 创建 Pages 对象
        doc.set_object((2, 0), dictionary! {
            "Type" => "Pages",
            "Kids" => Object::Array(vec![Object::Reference((3, 0))]),
            "Count" => Object::Integer(1),
        });

        // 创建 Catalog 对象
        doc.set_object((1, 0), dictionary! {
            "Type" => "Catalog",
            "Pages" => Object::Reference((2, 0)),
        });

        doc.trailer.set("Root", Object::Reference((1, 0)));

        let mut buffer = Vec::new();
        doc.save_to(&mut buffer).unwrap();
        buffer
    }
}
