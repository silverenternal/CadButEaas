//! 公共类型定义
//!
//! 本 crate 提供整个 CAD 系统的共享类型，确保各服务间数据交换的一致性。

// 注意：scene 必须在 geometry 之前，因为 geometry 依赖 scene::BoundarySemantic
pub mod scene;
pub mod geometry;
pub mod robust_geometry;  // P0-7 新增：稳健几何内核
pub mod adaptive_tolerance;  // P0-1 新增：自适应容差系统
pub mod relative_coords;  // P11- 锐评 P0: 相对坐标系统
pub mod constraint_solver;  // P1-3 新增：约束求解器基础框架
pub mod error;
pub mod service;
pub mod request;
pub mod response;
pub mod circuit_breaker;
pub mod acoustic;

// 显式导出以避免重复导出警告
pub use geometry::*;
// robust_geometry 有 PrecisionLevel 类型，不全局导出，仅导出需要的函数和类型
pub use robust_geometry::{orient2d, orient3d, incircle, Orientation, Orientation3D, ExactF64, PrecisionLevel as GeoPrecisionLevel};
pub use scene::*;
// adaptive_tolerance 也有 PrecisionLevel 类型，不全局导出，仅导出需要的类型
pub use adaptive_tolerance::{AdaptiveTolerance, PrecisionLevel as InteractionPrecisionLevel};
// relative_coords 导出相对坐标系统
pub use relative_coords::{SceneOrigin, RelativePoint, RelativeSceneState, RelativeClosedLoop, RelativeRawEdge};
// 先导出 error，再显式导出 response 中的非冲突类型
pub use error::{CadError, ErrorCode, Result, ErrorLocation, TopoStage, DxfParseReason, PdfParseReason, GeometryConstructionReason, ToleranceErrorReason, TopoErrorReason, LoopExtractReason, GraphBuildReason, ValidationIssue, Severity, IoErrorReason, InternalErrorReason, RecoverySuggestion};
pub use service::*;
// request 中的 ToleranceConfig 与 geometry 中的冲突，使用几何版本
pub use request::{Request, RequestMetadata, RequestId, ParseFileRequest, BuildTopologyRequest, ValidateGeometryRequest, VectorizePdfRequest, ExportRequest, ExportFormat};
// 显式导出 response 中的类型，避免与 error 中的 ValidationIssue/Severity 冲突
pub use response::{Response, ServiceError, ResponseStatus, ResponseTimer};
pub use circuit_breaker::*;

// 导出 PDF 光栅图像相关函数
pub use geometry::decode_image_pixels;
