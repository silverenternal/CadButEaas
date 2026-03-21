//! UI 状态（交互状态）
//!
//! 包含所有 UI 相关的状态：选择、工具模式、圈选等

use interact::EdgeId;
use std::collections::HashSet;
use std::time::Instant;
#[cfg(feature = "gpu")]
use crate::render::{GpuTier, GpuTierConfig, GpuInfo};

/// UI 状态（交互状态）
#[derive(Clone)]
pub struct UIState {
    /// 当前选择的边（支持多选）
    pub selected_edges: HashSet<EdgeId>,
    /// 工具模式（用于扩展交互模式，未来用于 Lasso/Trace/GapDetect）
    #[allow(dead_code)]
    pub tool_mode: ToolMode,
    /// 圈选的多边形点
    pub lasso_points: Vec<[f64; 2]>,
    /// 是否正在圈选
    pub is_lassoing: bool,
    /// 显示缺口
    pub show_gaps: bool,
    /// 图层过滤（用于图层过滤功能，未来用于高级筛选）
    #[allow(dead_code)]
    pub layer_filter: LayerFilter,
    /// 自动追踪结果
    pub auto_trace_result: Option<AutoTraceResult>,
    /// 圈选结果
    pub lasso_result: Option<LassoResult>,
    /// P0 改进：当前悬停的边（用于高亮显示）
    pub hovered_edge: Option<EdgeId>,
    /// P11 改进：悬停 Tooltip 文本（用于显示边信息）
    pub hovered_tooltip: Option<String>,
    /// P0 改进：Toast 通知队列
    pub toasts: Vec<ToastNotification>,
    /// 待处理的异步 API 调用
    ///
    /// P11 锐评落实：明确 pending_action 职责
    /// - Command: 同步状态变更（支持撤销/重做）
    /// - pending_action: 异步 API 调用（不可撤销，如文件操作、后端 API 调用）
    ///
    /// 注意：Toolbar 等组件现在通过 Command 触发这些操作，
    /// Command 的 execute() 方法会设置此标志，CadApp 在 process_pending_actions() 中处理
    pub pending_action: Option<String>,
    /// P11 新增：视觉效果设置（GPU 分级降级）
    #[cfg(feature = "gpu")]
    pub visual_settings: VisualSettings,
    /// P11 新增：WebSocket 连接状态
    pub websocket_connected: bool,
}

/// P11 新增：视觉效果设置
#[cfg(feature = "gpu")]
#[derive(Debug, Clone)]
pub struct VisualSettings {
    /// 是否启用高级视觉效果（一键开关）
    pub enable_effects: bool,
    /// GPU 等级（自动检测）
    pub gpu_tier: GpuTier,
    /// GPU 信息
    pub gpu_info: GpuInfo,
    /// 用户自定义配置（可覆盖自动检测）
    pub custom_config: GpuTierConfig,
    /// 是否使用自定义配置（否则使用推荐配置）
    pub use_custom_config: bool,
}

#[cfg(feature = "gpu")]
impl Default for VisualSettings {
    fn default() -> Self {
        Self {
            enable_effects: false,
            gpu_tier: GpuTier::Unknown,
            gpu_info: GpuInfo::default(),
            custom_config: GpuTierConfig::default(),
            use_custom_config: false,
        }
    }
}

#[cfg(feature = "gpu")]
impl VisualSettings {
    /// 获取当前有效配置
    pub fn get_effective_config(&self) -> GpuTierConfig {
        if self.use_custom_config {
            self.custom_config.clone()
        } else {
            self.gpu_tier.get_recommended_config()
        }
    }

    /// 获取当前配置描述
    pub fn get_config_description(&self) -> String {
        if self.use_custom_config {
            format!(
                "自定义配置：毛玻璃{} | MSAA {}x | 阴影{}",
                if self.custom_config.glass_effect { "开" } else { "关" },
                self.custom_config.msaa_samples,
                if self.custom_config.high_quality_shadows { "高" } else { "标" }
            )
        } else {
            format!(
                "自动检测 ({})：毛玻璃{} | MSAA {}x | 阴影{}",
                self.gpu_tier,
                if self.gpu_tier.enable_glass_effect() { "开" } else { "关" },
                self.gpu_tier.msaa_samples(),
                if self.gpu_tier.high_quality_shadows() { "高" } else { "标" }
            )
        }
    }
}

/// 工具模式
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ToolMode {
    #[default]
    Select,
    /// 圈选模式（未来用于批量选择）
    #[allow(dead_code)]
    Lasso,
    /// 追踪模式（未来用于路径追踪）
    #[allow(dead_code)]
    Trace,
    /// 缺口检测模式（未来用于缺口标注）
    #[allow(dead_code)]
    GapDetect,
}

/// 图层过滤
#[derive(Debug, Clone, Default)]
pub struct LayerFilter {
    /// 过滤模式（用于图层过滤功能）
    #[allow(dead_code)]
    pub mode: String,
}

/// 自动追踪结果
#[derive(Clone)]
pub struct AutoTraceResult {
    pub edges: Vec<EdgeId>,
    pub loop_closed: bool,
    pub polygon: Vec<[f64; 2]>,
}

/// 圈选结果
#[derive(Clone)]
pub struct LassoResult {
    /// 选中的边（用于批量操作）
    #[allow(dead_code)]
    pub selected_edges: Vec<EdgeId>,
    /// 闭合环（用于几何计算）
    #[allow(dead_code)]
    pub loops: Vec<Vec<[f64; 2]>>,
}

impl Default for UIState {
    fn default() -> Self {
        Self {
            selected_edges: HashSet::new(),
            tool_mode: ToolMode::Select,
            lasso_points: Vec::new(),
            is_lassoing: false,
            show_gaps: false,
            layer_filter: LayerFilter::default(),
            auto_trace_result: None,
            lasso_result: None,
            hovered_edge: None,
            hovered_tooltip: None,  // P11 改进：悬停 Tooltip 初始化为 None
            toasts: Vec::new(),
            pending_action: None,
            #[cfg(feature = "gpu")]
            visual_settings: VisualSettings::default(),
            websocket_connected: false,  // P11 新增：WebSocket 连接状态
        }
    }
}

impl UIState {
    /// 创建新的 UI 状态
    #[allow(dead_code)]
    pub fn new() -> Self {
        Self::default()
    }

    /// 选择边（用于多选操作）
    #[allow(dead_code)]
    pub fn select_edge(&mut self, edge_id: EdgeId, append: bool) {
        if append {
            // 多选模式
            if self.selected_edges.contains(&edge_id) {
                self.selected_edges.remove(&edge_id);
            } else {
                self.selected_edges.insert(edge_id);
            }
        } else {
            // 单选模式
            self.selected_edges.clear();
            self.selected_edges.insert(edge_id);
        }
    }

    /// 清除选择
    pub fn clear_selection(&mut self) {
        self.selected_edges.clear();
        self.auto_trace_result = None;
        self.lasso_points.clear();
        self.lasso_result = None;
        self.pending_action = None;
        self.hovered_edge = None;  // P0 改进：清除悬停状态
        self.hovered_tooltip = None;  // P11 改进：清除悬停 Tooltip
    }

    /// 获取当前选择的边（单选兼容）
    pub fn selected_edge(&self) -> Option<EdgeId> {
        self.selected_edges.iter().next().copied()
    }

    /// 是否有选择（用于 UI 状态判断）
    #[allow(dead_code)]
    pub fn has_selection(&self) -> bool {
        !self.selected_edges.is_empty()
    }

    // ========================================================================
    // P11 改进：Toast 手动关闭功能
    // ========================================================================

    /// 关闭指定索引的 Toast
    pub fn dismiss_toast(&mut self, index: usize) {
        if index < self.toasts.len() {
            self.toasts.remove(index);
        }
    }

    /// 关闭所有 Toast
    #[allow(dead_code)]
    pub fn dismiss_all_toasts(&mut self) {
        self.toasts.clear();
    }
}

// ============================================================================
// P0 改进：Toast 通知系统
// ============================================================================

/// Toast 通知类型
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToastType {
    Info,
    Success,
    Warning,
    #[allow(dead_code)]  // 保留用于未来错误处理
    Error,
}

/// Toast 通知
#[derive(Clone)]
pub struct ToastNotification {
    pub message: String,
    pub toast_type: ToastType,
    pub created_at: Instant,
    pub duration_secs: f32,
    /// P11 改进：是否需要用户手动关闭
    pub dismissible: bool,
}

impl ToastNotification {
    pub fn info(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            toast_type: ToastType::Info,
            created_at: Instant::now(),
            duration_secs: 3.0,
            dismissible: true,  // P11 改进：支持手动关闭
        }
    }

    pub fn success(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            toast_type: ToastType::Success,
            created_at: Instant::now(),
            duration_secs: 2.0,
            dismissible: true,  // P11 改进：支持手动关闭
        }
    }

    pub fn warning(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            toast_type: ToastType::Warning,
            created_at: Instant::now(),
            duration_secs: 3.0,
            dismissible: true,  // P11 改进：支持手动关闭
        }
    }

    /// 创建错误 Toast（保留用于未来错误处理）
    #[allow(dead_code)]
    pub fn error(message: impl Into<String>) -> Self {
        Self {
            message: message.into(),
            toast_type: ToastType::Error,
            created_at: Instant::now(),
            duration_secs: 0.0,  // P11 改进：0 表示不自动消失
            dismissible: true,    // P11 改进：需要手动关闭
        }
    }

    /// P11 改进：创建持久化 Toast（不自动消失，需要手动关闭）
    #[allow(dead_code)]
    pub fn persistent(message: impl Into<String>, toast_type: ToastType) -> Self {
        Self {
            message: message.into(),
            toast_type,
            created_at: Instant::now(),
            duration_secs: 0.0,  // 0 表示不自动消失
            dismissible: true,
        }
    }
}
