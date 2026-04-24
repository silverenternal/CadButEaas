//! 加载状态（异步任务）
//!
//! 包含所有异步加载相关的状态

use crate::state::UIState;
use common_types::ParseProgress;
use interact::Edge;

/// 缺口标记数据（用于在异步任务和主线程间传递）
#[derive(Clone)]
pub struct GapMarkerData {
    pub start: [f64; 2],
    pub end: [f64; 2],
    pub length: f64,
}

/// 加载状态
#[derive(Clone)]
pub struct LoadingState {
    /// 是否正在加载
    pub is_loading: bool,
    /// 加载的边数据
    pub edges: Option<Vec<Edge>>,
    /// 缺口标记数据
    pub gap_markers: Option<Vec<GapMarkerData>>,
    /// 错误信息
    pub error: Option<String>,
    /// 解析进度信息
    pub progress: Option<ParseProgress>,
    /// UI 状态（用于更新 WebSocket 连接状态等）
    pub ui: UIState,
}

impl LoadingState {
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self {
            is_loading: false,
            edges: None,
            gap_markers: None,
            error: None,
            progress: None,
            ui: UIState::default(),
        }
    }

    /// 开始加载
    pub fn start(&mut self) {
        self.is_loading = true;
        self.edges = None;
        self.error = None;
    }

    /// 设置缺口标记
    pub fn set_gap_markers(&mut self, gaps: Vec<GapMarkerData>) {
        self.gap_markers = Some(gaps);
        self.ui.show_gaps = true;
    }

    /// 加载成功
    pub fn success(&mut self, edges: Vec<Edge>) {
        self.is_loading = false;
        self.edges = Some(edges);
        self.error = None;
    }

    /// 加载失败
    pub fn error(&mut self, error: String) {
        self.is_loading = false;
        self.edges = None;
        self.error = Some(error);
    }

    /// 更新进度（用于异步任务进度跟踪）
    #[allow(dead_code)]
    pub fn update_progress(&mut self, progress: ParseProgress) {
        self.progress = Some(progress);
    }

    /// 是否有数据（用于检查加载完成状态）
    #[allow(dead_code)]
    pub fn has_data(&self) -> bool {
        self.edges.is_some()
    }

    /// 是否有错误（用于检查加载失败状态）
    #[allow(dead_code)]
    pub fn has_error(&self) -> bool {
        self.error.is_some()
    }
}
