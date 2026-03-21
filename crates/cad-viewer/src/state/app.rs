//! 应用状态（分层管理）
//!
//! P11 锐评落实：将 CadApp 的状态按职责分离

use crate::state::{SceneState, UIState, RenderState, LoadingState};
use eframe::egui;
use parking_lot::RwLock;
use std::sync::Arc;

/// 应用状态（分层管理）
pub struct AppState {
    /// 业务状态（场景数据）
    pub scene: SceneState,
    /// UI 状态（交互状态）
    pub ui: UIState,
    /// 渲染状态（相机、LOD）
    pub render: RenderState,
    /// 加载状态（异步任务）
    pub loading: Arc<RwLock<LoadingState>>,
    /// UI 上下文（用于请求重绘）
    pub ctx: egui::Context,
    /// 日志消息
    pub log_messages: Vec<String>,
    /// 错误消息
    pub error_message: Option<String>,
}

impl AppState {
    /// 创建新的应用状态
    pub fn new(ctx: egui::Context) -> Self {
        Self {
            scene: SceneState::new(),
            ui: UIState::new(),
            render: RenderState::new(),
            loading: Arc::new(RwLock::new(LoadingState::new())),
            ctx,
            log_messages: Vec::new(),
            error_message: None,
        }
    }

    /// 添加日志消息
    pub fn add_log(&mut self, msg: &str) {
        log::info!("{}", msg);
        let timestamp = chrono::Local::now().format("%H:%M:%S");
        self.log_messages.push(format!("[{}] {}", timestamp, msg));
        // 保留最近 100 条
        if self.log_messages.len() > 100 {
            self.log_messages.remove(0);
        }
    }

    /// 设置错误消息
    pub fn set_error(&mut self, msg: String) {
        self.error_message = Some(msg.clone());
        self.add_log(&format!("错误：{}", msg));
    }

    /// 清除错误消息
    pub fn clear_error(&mut self) {
        self.error_message = None;
    }

    /// 获取加载状态（只读）
    pub fn loading_state(&self) -> parking_lot::RwLockReadGuard<'_, LoadingState> {
        self.loading.read()
    }

    /// 获取加载状态（可写）（用于异步任务更新）
    #[allow(dead_code)]
    pub fn loading_state_mut(&self) -> parking_lot::RwLockWriteGuard<'_, LoadingState> {
        self.loading.write()
    }
}
