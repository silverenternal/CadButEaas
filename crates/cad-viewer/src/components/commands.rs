//! 命令系统（支持撤销/重做）
//!
//! P11 锐评落实：使用命令模式封装状态变更，支持操作历史管理

use crate::state::AppState;

/// 命令 trait（所有命令的基接口）
pub trait Command: Send + Sync {
    /// 命令名称（用于 UI 显示）（用于历史面板显示）
    #[allow(dead_code)]
    fn name(&self) -> &str;

    /// 执行命令
    fn execute(&mut self, state: &mut AppState);

    /// 撤销命令
    fn undo(&mut self, state: &mut AppState);

    /// 重做命令（默认实现调用 execute）
    fn redo(&mut self, state: &mut AppState) {
        self.execute(state);
    }
}

/// 命令管理器（支持撤销/重做）
pub struct CommandManager {
    /// 命令历史
    undo_stack: Vec<Box<dyn Command>>,
    /// 重做历史
    redo_stack: Vec<Box<dyn Command>>,
    /// 最大历史深度
    max_history: usize,
}

impl Default for CommandManager {
    fn default() -> Self {
        Self::new(50)
    }
}

impl CommandManager {
    /// 创建新的命令管理器
    pub fn new(max_history: usize) -> Self {
        Self {
            undo_stack: Vec::with_capacity(max_history),
            redo_stack: Vec::new(),
            max_history,
        }
    }

    /// 执行命令
    pub fn execute(&mut self, mut command: Box<dyn Command>, state: &mut AppState) {
        command.execute(state);
        self.undo_stack.push(command);
        self.redo_stack.clear();

        // 限制历史深度
        if self.undo_stack.len() > self.max_history {
            self.undo_stack.remove(0);
        }
    }

    /// 撤销
    pub fn undo(&mut self, state: &mut AppState) -> bool {
        if let Some(mut command) = self.undo_stack.pop() {
            command.undo(state);
            self.redo_stack.push(command);
            true
        } else {
            false
        }
    }

    /// 重做
    pub fn redo(&mut self, state: &mut AppState) -> bool {
        if let Some(mut command) = self.redo_stack.pop() {
            command.redo(state);
            self.undo_stack.push(command);
            true
        } else {
            false
        }
    }

    /// 清空历史（用于场景切换时重置）
    #[allow(dead_code)]
    pub fn clear(&mut self) {
        self.undo_stack.clear();
        self.redo_stack.clear();
    }

    /// 获取撤销历史长度（用于 UI 显示）
    #[allow(dead_code)]
    pub fn undo_depth(&self) -> usize {
        self.undo_stack.len()
    }

    /// 获取重做历史长度（用于 UI 显示）
    #[allow(dead_code)]
    pub fn redo_depth(&self) -> usize {
        self.redo_stack.len()
    }
}

// ============================================================================
// 具体命令实现
// ============================================================================

/// 切换图层可见性命令
pub struct ToggleLayerVisibility {
    layer: String,
    old_visibility: Option<bool>,
}

impl ToggleLayerVisibility {
    pub fn new(layer: String) -> Self {
        Self {
            layer,
            old_visibility: None,
        }
    }
}

impl Command for ToggleLayerVisibility {
    fn name(&self) -> &str {
        "切换图层可见性"
    }

    fn execute(&mut self, state: &mut AppState) {
        // 保存旧状态
        self.old_visibility = Some(*state.scene.layers.visibility.get(&self.layer).unwrap_or(&true));

        // 切换可见性
        let current = *state.scene.layers.visibility.get(&self.layer).unwrap_or(&true);
        state.scene.layers.visibility.insert(self.layer.clone(), !current);
        state.scene.update_visibility_stats();
    }

    fn undo(&mut self, state: &mut AppState) {
        if let Some(old) = self.old_visibility {
            state.scene.layers.visibility.insert(self.layer.clone(), old);
            state.scene.update_visibility_stats();
        }
    }
}

/// 设置图层过滤模式命令
pub struct SetLayerFilter {
    old_mode: String,
    new_mode: String,
}

impl SetLayerFilter {
    pub fn new(new_mode: String) -> Self {
        Self {
            old_mode: String::new(),
            new_mode,
        }
    }
}

impl Command for SetLayerFilter {
    fn name(&self) -> &str {
        "设置图层过滤"
    }

    fn execute(&mut self, state: &mut AppState) {
        self.old_mode = state.scene.layers.filter_mode.clone();
        state.scene.set_layer_filter(&self.new_mode);
    }

    fn undo(&mut self, state: &mut AppState) {
        state.scene.set_layer_filter(&self.old_mode);
    }
}

/// 选择边命令
pub struct SelectEdge {
    edge_id: interact::EdgeId,
    append: bool,
    previous_selection: Vec<interact::EdgeId>,
}

impl SelectEdge {
    pub fn new(edge_id: interact::EdgeId, append: bool) -> Self {
        Self {
            edge_id,
            append,
            previous_selection: Vec::new(),
        }
    }
}

impl Command for SelectEdge {
    fn name(&self) -> &str {
        "选择边"
    }

    fn execute(&mut self, state: &mut AppState) {
        // 保存之前的选择
        self.previous_selection = state.ui.selected_edges.iter().copied().collect();

        // 执行选择
        if self.append {
            if state.ui.selected_edges.contains(&self.edge_id) {
                state.ui.selected_edges.remove(&self.edge_id);
            } else {
                state.ui.selected_edges.insert(self.edge_id);
            }
        } else {
            state.ui.selected_edges.clear();
            state.ui.selected_edges.insert(self.edge_id);
        }
    }

    fn undo(&mut self, state: &mut AppState) {
        state.ui.selected_edges.clear();
        for id in &self.previous_selection {
            state.ui.selected_edges.insert(*id);
        }
    }
}

// ============================================================================
// Toolbar 命令实现（P11 锐评落实：统一命令模式）
// ============================================================================

/// 打开文件命令
/// 
/// P11 锐评落实：Toolbar 不再直接修改 pending_action，而是通过命令
pub struct OpenFileCommand;

impl Command for OpenFileCommand {
    fn name(&self) -> &str {
        "打开文件"
    }

    fn execute(&mut self, state: &mut AppState) {
        // 标记需要打开文件（异步 API 调用）
        state.ui.pending_action = Some("open_file".to_string());
    }

    fn undo(&mut self, _state: &mut AppState) {
        // 异步 API 调用不可撤销
    }
}

/// 导出场景命令
pub struct ExportSceneCommand;

impl Command for ExportSceneCommand {
    fn name(&self) -> &str {
        "导出场景"
    }

    fn execute(&mut self, state: &mut AppState) {
        state.ui.pending_action = Some("export_scene".to_string());
    }

    fn undo(&mut self, _state: &mut AppState) {
        // 异步 API 调用不可撤销
    }
}

/// 自动追踪命令
pub struct AutoTraceCommand;

impl Command for AutoTraceCommand {
    fn name(&self) -> &str {
        "自动追踪"
    }

    fn execute(&mut self, state: &mut AppState) {
        // 触发异步 API 调用
        state.ui.pending_action = Some("auto_trace".to_string());
    }

    fn undo(&mut self, _state: &mut AppState) {
        // 异步 API 调用不可撤销
    }
}

/// 缺口检测命令
pub struct DetectGapsCommand;

impl Command for DetectGapsCommand {
    fn name(&self) -> &str {
        "缺口检测"
    }

    fn execute(&mut self, state: &mut AppState) {
        state.ui.pending_action = Some("detect_gaps".to_string());
    }

    fn undo(&mut self, _state: &mut AppState) {
        // 异步 API 调用不可撤销
    }
}

/// 撤销命令
pub struct UndoCommand;

impl Command for UndoCommand {
    fn name(&self) -> &str {
        "撤销"
    }

    fn execute(&mut self, state: &mut AppState) {
        // 撤销操作由 CommandManager 处理，这里只是标记
        state.ui.pending_action = Some("undo".to_string());
    }

    fn undo(&mut self, _state: &mut AppState) {
        // 无操作
    }
}

/// 重做命令
pub struct RedoCommand;

impl Command for RedoCommand {
    fn name(&self) -> &str {
        "重做"
    }

    fn execute(&mut self, state: &mut AppState) {
        state.ui.pending_action = Some("redo".to_string());
    }

    fn undo(&mut self, _state: &mut AppState) {
        // 无操作
    }
}

/// 清除选择命令
pub struct ClearSelectionCommand;

impl Command for ClearSelectionCommand {
    fn name(&self) -> &str {
        "清除选择"
    }

    fn execute(&mut self, state: &mut AppState) {
        state.ui.clear_selection();
        state.add_log("已清除选择");
    }

    fn undo(&mut self, state: &mut AppState) {
        // 清除选择不可撤销（或可以保存之前的选择状态）
        state.add_log("撤销清除选择（未实现）");
    }
}

/// 切换圈选工具命令
pub struct ToggleLassoToolCommand {
    new_state: bool,
    old_state: Option<bool>,
}

impl ToggleLassoToolCommand {
    pub fn new(new_state: bool) -> Self {
        Self {
            new_state,
            old_state: None,
        }
    }
}

impl Command for ToggleLassoToolCommand {
    fn name(&self) -> &str {
        "切换圈选工具"
    }

    fn execute(&mut self, state: &mut AppState) {
        self.old_state = Some(state.ui.is_lassoing);
        state.ui.is_lassoing = self.new_state;
        state.add_log(if self.new_state {
            "已启用圈选工具"
        } else {
            "已禁用圈选工具"
        });
    }

    fn undo(&mut self, state: &mut AppState) {
        if let Some(old) = self.old_state {
            state.ui.is_lassoing = old;
        }
    }
}
