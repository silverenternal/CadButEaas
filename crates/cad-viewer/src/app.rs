//! 主应用状态和逻辑（P11 落实版）
//!
//! P11 锐评落实：
//! 1. CadApp 作为真正的协调器，而不是「什么都做」
//! 2. 使用 ComponentRegistry 管理组件生命周期
//! 3. 使用 EventCollector 收集和处理事件
//! 4. 组件负责渲染和事件处理，CadApp 负责协调

use crate::api::ApiClient;
use crate::api::{WsTraceResult, WsGapDetectionResult};
use crate::canvas::CanvasWidget;
#[cfg(feature = "gpu")]
use crate::render::GpuRendererWrapper;
#[cfg(feature = "gpu")]
use crate::gpu_renderer_enhanced::RendererConfig;
use crate::panels::{Toolbar, LeftPanel, LayerPanel, BottomPanel, RightPanel};
use crate::state::{AppState, ToastNotification};
use crate::state::AutoTraceResult as UiAutoTraceResult;
use crate::components::{CommandManager, ToggleLayerVisibility, SetLayerFilter};
use crate::components::{ComponentRegistry, EventCollector};
#[cfg(feature = "gpu")]
use crate::render::GlassEffectRenderer;
use crate::theme::MacOsTheme;
use eframe::egui;
use interact::Edge;
use std::sync::Arc;
use tokio::sync::Mutex;

// 导入加速器相关类型
#[cfg(feature = "registry")]
use accelerator_registry::AcceleratorRegistry;
use accelerator_api::Accelerator;
use accelerator_cpu::CpuAccelerator;

/// 主应用状态
///
/// P11 锐评落实：将 CadApp 重构为薄层协调器
/// - 组件注册表：管理所有 UI 组件
/// - 事件收集器：收集和处理输入事件
/// - 命令管理器：支持撤销/重做
/// - 状态分层：Scene/UI/Render/Loading
/// - 主题系统：macOS 风格主题（P11 新增）
pub struct CadApp {
    /// 应用状态（分层管理）
    pub state: AppState,
    /// 命令管理器
    pub command_manager: CommandManager,
    /// API 客户端
    pub api_client: Arc<Mutex<ApiClient>>,
    /// 加速器（trait object，可运行时切换）（用于硬件加速，未来用于高性能计算）
    #[allow(dead_code)]
    pub accelerator: Box<dyn Accelerator>,
    /// 加速器注册表（可选）
    #[cfg(feature = "registry")]
    pub accelerator_registry: Option<AcceleratorRegistry>,
    /// GPU 渲染器（可选，P11 新增：通过 Renderer trait 统一接口）
    #[cfg(feature = "gpu")]
    pub gpu_renderer: Option<GpuRendererWrapper>,
    /// GPU 渲染配置
    #[cfg(feature = "gpu")]
    pub gpu_config: Option<RendererConfig>,
    /// 毛玻璃效果渲染器（P11 新增）
    #[cfg(feature = "gpu")]
    pub glass_renderer: Option<GlassEffectRenderer>,
    /// 缺口标记（向后兼容，未来迁移到 state）
    pub gap_markers: Vec<GapMarker>,
    /// 组件注册表（P11 新增）
    pub components: ComponentRegistry,
    /// 事件收集器（P11 新增）
    pub event_collector: EventCollector,
    /// macOS 主题（P11 新增）
    pub theme: Arc<MacOsTheme>,
}

/// 缺口标记（保持兼容旧代码）
pub struct GapMarker {
    pub start: [f64; 2],
    pub end: [f64; 2],
    pub length: f64,
}

impl CadApp {
    pub fn new(cc: &eframe::CreationContext<'_>) -> Self {
        let api_client = Arc::new(Mutex::new(ApiClient::new("http://localhost:3000")));
        let ctx = cc.egui_ctx.clone();

        // P11 锐评落实：初始化加速器
        #[cfg(feature = "registry")]
        let (accelerator_registry, accelerator) = {
            let registry = AcceleratorRegistry::discover_all();
            let best = registry.select_best(accelerator_api::AcceleratorOp::EdgeDetect)
                .map(|a| Box::new(a.clone()) as Box<dyn Accelerator>)
                .unwrap_or_else(|| Box::new(CpuAccelerator::new()));
            (Some(registry), best)
        };

        #[cfg(not(feature = "registry"))]
        let accelerator = Box::new(CpuAccelerator::new());

        // P11 新增：初始化 macOS 主题（浅色模式）
        let theme = Arc::new(MacOsTheme::light());
        theme.apply(&ctx);

        // P11 新增：初始化毛玻璃渲染器（需要 GPU 支持）
        // 注意：eframe 0.29 的 wgpu 渲染状态访问方式可能不同
        // 这里暂时设为 None，后续通过 Painter 事件初始化
        #[cfg(feature = "gpu")]
        let glass_renderer = None;

        // P11 新增：GPU 分级检测（在后台线程中进行）
        #[cfg(feature = "gpu")]
        let (gpu_tier, gpu_info) = crate::render::detect_gpu_tier();
        
        #[cfg(feature = "gpu")]
        log::info!("GPU 检测完成：{} ({})", gpu_info.name, gpu_tier);

        // 创建组件注册表并注册所有组件
        let mut components = ComponentRegistry::new();
        components.register("toolbar".to_string(), Box::new(Toolbar::new()));
        components.register("left_panel".to_string(), Box::new(LeftPanel::new()));
        components.register("layer_panel".to_string(), Box::new(LayerPanel::new()));
        components.register("bottom_panel".to_string(), Box::new(BottomPanel::new()));
        components.register("right_panel".to_string(), Box::new(RightPanel::new()));
        #[cfg(feature = "gpu")]
        components.register("visual_settings".to_string(), Box::new(crate::panels::VisualSettingsPanel::new()));

        // 创建应用状态
        let mut state = AppState::new(ctx);
        
        // P11 新增：初始化视觉效果设置
        #[cfg(feature = "gpu")]
        {
            state.ui.visual_settings.gpu_tier = gpu_tier;
            state.ui.visual_settings.gpu_info = gpu_info;
            // 根据 GPU 等级自动决定是否启用效果
            state.ui.visual_settings.enable_effects = gpu_tier.enable_glass_effect();
        }

        Self {
            state,
            command_manager: CommandManager::default(),
            api_client,
            accelerator,
            #[cfg(feature = "registry")]
            accelerator_registry,
            #[cfg(feature = "gpu")]
            gpu_renderer: None,
            #[cfg(feature = "gpu")]
            gpu_config: None,
            #[cfg(feature = "gpu")]
            glass_renderer,
            gap_markers: Vec::new(),
            components,
            event_collector: EventCollector::new(),
            theme,
        }
    }

    /// 添加日志消息
    pub fn add_log(&mut self, msg: &str) {
        self.state.add_log(msg);
    }

    // ========================================================================
    // 场景管理方法
    // ========================================================================

    /// 打开文件并加载
    pub fn open_file(&mut self) {
        if let Some(path) = rfd::FileDialog::new()
            .add_filter("DXF 文件", &["dxf"])
            .add_filter("PDF 文件", &["pdf"])
            .pick_file()
        {
            let path_str = path.display().to_string();
            self.add_log(&format!("正在加载文件：{}", path_str));
            self.state.scene.file_path = Some(path.clone());

            // 异步加载文件
            self.load_file_async(path_str);
        }
    }

    /// 异步加载文件
    fn load_file_async(&mut self, path: String) {
        let api_client = self.api_client.clone();
        let ctx = self.state.ctx.clone();
        let loading = self.state.loading.clone();

        // 设置加载状态
        {
            let mut state = loading.write();
            state.start();
        }
        self.add_log("正在处理文件...");

        tokio::spawn(async move {
            let mut client = api_client.lock().await;
            let result = client.load_file(&path).await;

            // 更新共享状态
            {
                let mut state = loading.write();
                match result {
                    Ok(edges) => {
                        log::info!("文件处理成功，获取到 {} 条边", edges.len());
                        state.success(edges);
                    }
                    Err(e) => {
                        log::error!("文件处理失败：{}", e);
                        state.error(e);
                    }
                }
            }

            // P11 落实：文件加载完成后自动连接 WebSocket
            let ws_connected = if let Err(e) = client.connect_websocket().await {
                log::warn!("WebSocket 连接失败（后端可能未运行）: {}", e);
                false
            } else {
                log::info!("WebSocket 连接成功，启用实时交互");
                true
            };

            // 更新 UI 状态中的 WebSocket 连接标志
            {
                let mut state = loading.write();
                state.ui.websocket_connected = ws_connected;
            }

            // 请求重绘
            ctx.request_repaint();
        });
    }

    /// 处理加载完成的数据
    fn process_loaded_data(&mut self, edges: Vec<Edge>) {
        // 清除旧的选择和结果
        self.state.ui.clear_selection();

        // 替换边数据
        self.state.scene.edges = edges;
        self.add_log(&format!("加载完成，共 {} 条边", self.state.scene.edges.len()));

        // 自动适配视图
        self.fit_to_scene();
    }

    /// 自动适配场景边界
    pub fn fit_to_scene(&mut self) {
        if let Some((min, max)) = self.state.scene.calculate_bounds() {
            // 获取画布尺寸（假设默认 800x600）
            let view_width = 800.0;
            let view_height = 600.0;

            self.state.render.camera.fit_to_scene(min, max, view_width, view_height);

            // 设置场景原点为场景边界的最小值
            self.state.scene.scene_origin = [min[0], min[1]];

            // P11 调试：打印详细缩放信息
            let zoom_percent = self.state.render.camera.zoom as f64 * 100.0;
            self.add_log(&format!(
                "已适配视图：缩放 {:.6}% (zoom={:.10}), 场景范围 [{:.1}, {:.1}] × [{:.1}, {:.1}]",
                zoom_percent,
                self.state.render.camera.zoom,
                min[0], max[0],
                min[1], max[1]
            ));
        }
    }

    /// 导出场景
    pub fn export_scene(&mut self) {
        if self.state.scene.edges.is_empty() {
            self.add_log("没有可导出的数据");
            return;
        }

        if let Some(path) = rfd::FileDialog::new()
            .add_filter("JSON 文件", &["json"])
            .save_file()
        {
            let path_str = path.display().to_string();
            self.add_log(&format!("导出场景到：{}", path_str));

            let api_client = self.api_client.clone();
            let ctx = self.state.ctx.clone();
            let path_str_clone = path_str.clone();
            let edges = self.state.scene.edges.clone();

            tokio::spawn(async move {
                let mut client = api_client.lock().await;
                match client.export_scene(&path_str_clone, &edges, "json").await {
                    Ok(_) => log::info!("导出成功"),
                    Err(e) => {
                        log::warn!("后端导出失败 {}，使用本地导出", e);
                        if let Err(local_err) = client.export_scene_local(&path_str_clone, &edges, "json").await {
                            log::error!("本地导出也失败：{}", local_err);
                        }
                    }
                }
                ctx.request_repaint();
            });
        }
    }

    // ========================================================================
    // 交互方法
    // ========================================================================

    /// 自动追踪 - 使用 WebSocket 实时交互（P11 落实）
    pub fn auto_trace(&mut self) {
        if let Some(edge_id) = self.state.ui.selected_edge() {
            self.add_log(&format!("正在从边 {} 开始自动追踪", edge_id));

            let api_client = self.api_client.clone();
            let ctx = self.state.ctx.clone();
            let loading = self.state.loading.clone();

            {
                let mut state = loading.write();
                state.is_loading = true;
            }

            tokio::spawn(async move {
                let mut client = api_client.lock().await;

                // P11 落实：优先使用 WebSocket 实时交互
                let result = if client.is_websocket_connected().await {
                    // 使用 WebSocket
                    match client.ws_select_edge(edge_id).await {
                        Ok(trace_result) => {
                            log::info!("WebSocket 自动追踪成功：{} 条边，闭环={}",
                                trace_result.edges.len(), trace_result.loop_closed);
                            Ok(trace_result)
                        }
                        Err(e) => {
                            log::warn!("WebSocket 追踪失败，回退到 HTTP: {}", e);
                            // 回退到 HTTP
                            client.auto_trace(edge_id).await
                                .map(|r| WsTraceResult {
                                    edges: vec![],
                                    loop_closed: r.success
                                })
                        }
                    }
                } else {
                    // 直接使用 HTTP
                    client.auto_trace(edge_id).await
                        .map(|r| WsTraceResult { 
                            edges: vec![], 
                            loop_closed: r.success 
                        })
                };

                // 处理结果
                match result {
                    Ok(trace_result) => {
                        // 更新 UI 状态 - 使用单独的锁定范围
                        {
                            let mut state = loading.write();
                            state.ui.auto_trace_result = Some(UiAutoTraceResult {
                                edges: trace_result.edges.iter().map(|&id| id).collect(),
                                loop_closed: trace_result.loop_closed,
                                polygon: vec![],
                            });
                        }
                    }
                    Err(e) => {
                        log::error!("自动追踪失败：{}", e);
                        let mut state = loading.write();
                        state.error(e);
                    }
                }
                
                {
                    let mut state = loading.write();
                    state.is_loading = false;
                }
                ctx.request_repaint();
            });
        } else {
            self.add_log("请先选择一条边");
        }
    }

    /// 缺口检测 - 使用 WebSocket 实时交互（P11 落实）
    pub fn detect_gaps(&mut self) {
        self.add_log("开始缺口检测...");

        let api_client = self.api_client.clone();
        let ctx = self.state.ctx.clone();
        let loading = self.state.loading.clone();

        {
            let mut state = loading.write();
            state.is_loading = true;
        }

        tokio::spawn(async move {
            let mut client = api_client.lock().await;
            
            // P11 落实：优先使用 WebSocket 实时交互
            let result = if client.is_websocket_connected().await {
                // 使用 WebSocket
                match client.ws_detect_gaps(0.5).await {
                    Ok(gap_result) => {
                        log::info!("WebSocket 缺口检测成功：{} 个缺口", gap_result.gaps.len());
                        Ok(gap_result)
                    }
                    Err(e) => {
                        log::warn!("WebSocket 缺口检测失败，回退到 HTTP: {}", e);
                        Err(e)
                    }
                }
            } else {
                // 直接使用 HTTP
                client.detect_gaps(0.5).await
                    .map(|r| WsGapDetectionResult {
                        gaps: r.gaps.iter().map(|g| crate::api::WsGapInfoResponse {
                            id: g.id,
                            start: g.start,
                            end: g.end,
                            length: g.length,
                            gap_type: g.gap_type.clone(),
                        }).collect()
                    })
            };

            // 处理结果
            match result {
                Ok(response) => {
                    log::info!("缺口检测完成，检测到 {} 个缺口", response.gaps.len());
                }
                Err(e) => {
                    log::error!("缺口检测失败：{}", e);
                    let mut state = loading.write();
                    state.error(e);
                }
            }
            
            {
                let mut state = loading.write();
                state.is_loading = false;
            }
            ctx.request_repaint();
        });
    }

    // ========================================================================
    // 图层管理方法
    // ========================================================================

    /// 切换图层可见性
    pub fn toggle_layer(&mut self, layer: &str) {
        // 使用命令模式，支持撤销
        let cmd = ToggleLayerVisibility::new(layer.to_string());
        self.command_manager.execute(Box::new(cmd), &mut self.state);
        self.add_log(&format!("图层 '{}' 可见性：{}", layer,
            if !self.state.scene.is_layer_visible(layer) { "开启" } else { "关闭" }));
    }

    /// 设置图层过滤模式
    pub fn set_layer_filter_mode(&mut self, mode: &str) {
        let cmd = SetLayerFilter::new(mode.to_string());
        self.command_manager.execute(Box::new(cmd), &mut self.state);
        self.add_log(&format!("图层过滤模式：{}", mode));
    }

    /// 获取所有唯一图层名称（用于图层面板）
    #[allow(dead_code)]
    pub fn get_unique_layers(&self) -> Vec<String> {
        self.state.scene.get_unique_layers()
    }

    // ========================================================================
    // P0 改进：Toast 通知系统
    // ========================================================================

    /// 显示成功 Toast
    pub fn show_success_toast(&mut self, message: impl Into<String>) {
        self.state.ui.toasts.push(ToastNotification::success(message));
    }

    /// 显示信息 Toast
    pub fn show_info_toast(&mut self, message: impl Into<String>) {
        self.state.ui.toasts.push(ToastNotification::info(message));
    }

    /// 显示警告 Toast
    pub fn show_warning_toast(&mut self, message: impl Into<String>) {
        self.state.ui.toasts.push(ToastNotification::warning(message));
    }

    /// 显示错误 Toast（保留用于未来错误处理）
    #[allow(dead_code)]
    pub fn show_error_toast(&mut self, message: impl Into<String>) {
        self.state.ui.toasts.push(ToastNotification::error(message));
    }

    // ========================================================================
    // GPU 渲染器方法
    // ========================================================================

    /// 初始化 GPU 渲染器（P11 新增：通过 Renderer trait 统一接口）
    #[cfg(feature = "gpu")]
    #[allow(dead_code)]
    pub fn init_gpu_renderer(&mut self, enable_gpu: bool) -> Result<(), String> {
        if !enable_gpu {
            self.gpu_renderer = None;
            self.gpu_config = None;
            self.add_log("GPU 渲染器已禁用");
            return Ok(());
        }

        let config = RendererConfig::default();
        self.gpu_config = Some(config.clone());

        match GpuRendererWrapper::new(config) {
            Ok(renderer) => {
                self.gpu_renderer = Some(renderer);
                self.add_log("GPU 渲染器初始化成功");
                Ok(())
            }
            Err(e) => {
                self.gpu_renderer = None;
                self.gpu_config = None;
                self.add_log(&format!("GPU 渲染器初始化失败：{}，回退到 CPU 渲染", e));
                Err(e)
            }
        }
    }

    /// 更新 GPU 实体（P11 新增：通过 Renderer trait 统一接口）
    #[cfg(feature = "gpu")]
    pub fn update_gpu_entities(&mut self) {
        use crate::gpu_renderer_enhanced::RenderEntity;
        if let Some(ref mut renderer) = self.gpu_renderer {
            let entities: Vec<RenderEntity> = self.state.scene.edges
                .iter()
                .enumerate()
                .map(|(idx, edge)| {
                    // 使用默认颜色（简化处理）
                    let color = [1.0, 1.0, 1.0, 1.0];
                    RenderEntity::line(edge.start, edge.end, color, idx as u32)
                })
                .collect();
            renderer.set_entities(entities);
        }
    }

    // ========================================================================
    // P11 新增：事件处理和组件协调
    // ========================================================================

    /// 处理待处理动作（由组件触发）
    fn process_pending_actions(&mut self) {
        if let Some(action) = self.state.ui.pending_action.take() {
            // 处理图层相关的动作
            if action.starts_with("layer_filter:") {
                let mode = action.strip_prefix("layer_filter:").unwrap();
                self.set_layer_filter_mode(mode);
            } else if action.starts_with("layer_toggle:") {
                let layer = action.strip_prefix("layer_toggle:").unwrap();
                self.toggle_layer(layer);
            } else {
                match action.as_str() {
                    "open_file" => {
                        self.show_info_toast("正在打开文件...");
                        self.open_file();
                    }
                    "export_scene" => {
                        self.show_info_toast("正在导出场景...");
                        self.export_scene();
                    }
                    "auto_trace" => {
                        self.show_info_toast("正在自动追踪...");
                        self.auto_trace();
                    }
                    "detect_gaps" => {
                        self.show_info_toast("正在检测缺口...");
                        self.detect_gaps();
                    }
                    "undo" => {
                        if self.command_manager.undo(&mut self.state) {
                            self.show_success_toast("已撤销");
                        } else {
                            self.show_warning_toast("没有可撤销的操作");
                        }
                    }
                    "redo" => {
                        if self.command_manager.redo(&mut self.state) {
                            self.show_success_toast("已重做");
                        } else {
                            self.show_warning_toast("没有可重做的操作");
                        }
                    }
                    "preset_architectural" => {
                        self.show_success_toast("已应用建筑图纸预设");
                        self.add_log("应用建筑图纸预设");
                    }
                    "preset_mechanical" => {
                        self.show_success_toast("已应用机械图纸预设");
                        self.add_log("应用机械图纸预设");
                    }
                    "preset_scanned" => {
                        self.show_success_toast("已应用扫描图纸预设");
                        self.add_log("应用扫描图纸预设");
                    }
                    "preset_quick" => {
                        self.show_success_toast("已应用快速原型预设");
                        self.add_log("应用快速原型预设");
                    }
                    _ => {
                        self.add_log(&format!("未知动作：{}", action));
                    }
                }
            }
        }
    }

    /// 收集和处理事件
    fn process_events(&mut self, ctx: &egui::Context) {
        // P11 锐评落实：清理双轨制后，EventCollector 只用于全局键盘事件
        // Canvas 交互事件已改为直接命令方案
        self.event_collector.collect_global(ctx);

        // 分发事件到组件，收集产生的命令
        for event in self.event_collector.drain() {
            let commands = self.components.dispatch_event_with_commands(&event, &mut self.state);

            // 执行组件产生的命令
            for cmd in commands {
                self.command_manager.execute(cmd, &mut self.state);
            }
        }
    }
}

impl eframe::App for CadApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        // P11 锐评落实：CadApp 作为协调器的工作流程
        // 1. 检查状态更新
        // 2. 处理事件
        // 3. 渲染组件
        // 4. 执行待处理动作

        // 1. 检查加载状态更新
        let error_msg;
        let edges_data;

        {
            let state = self.state.loading.read();
            error_msg = state.error.clone();
            edges_data = state.edges.clone();
        }

        // 同步边数据
        if let Some(edges) = edges_data {
            self.process_loaded_data(edges);

            // 清除加载状态
            {
                let mut state = self.state.loading.write();
                state.edges = None;
            }
        }

        // 显示错误
        if let Some(error) = error_msg {
            if self.state.error_message.is_none() {
                self.state.set_error(error);
            }
        }

        // 2. 处理事件（键盘快捷键等）
        self.process_events(ctx);

        // 3. 渲染所有组件（P11 锐评落实：使用 ComponentContext 收集命令）
        // 注意：面板组件现在自己负责自己的布局
        #[cfg(not(feature = "gpu"))]
        let commands = self.components.render(ctx, &mut self.state, &self.command_manager, self.theme.clone());
        
        // P11 新增：GPU 特性版本需要传递 glass_renderer 引用
        // 使用分离的借用模式来避免借用检查器冲突
        #[cfg(feature = "gpu")]
        let commands = {
            // 临时变量来分离借用
            let components = &mut self.components;
            let state = &mut self.state;
            let command_manager = &self.command_manager;
            let theme = self.theme.clone();
            let glass_renderer = self.glass_renderer.as_mut();
            components.render(ctx, state, command_manager, theme, glass_renderer)
        };

        // 中央画布（特殊处理，因为需要 egui::Widget 实现）
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.add(CanvasWidget::new(self));
        });

        // 4. 执行组件产生的命令（P11 锐评落实：不再使用 pending_action）
        for cmd in commands {
            self.command_manager.execute(cmd, &mut self.state);
        }

        // 5. 处理 pending_action（P11 锐评落实：修复架构断头路）
        // Command 的 execute() 方法会设置 pending_action，这里统一处理异步 API 调用
        self.process_pending_actions();

        // 错误对话框
        if let Some(error) = self.state.error_message.clone() {
            egui::Window::new("错误")
                .collapsible(false)
                .resizable(false)
                .anchor(egui::Align2::CENTER_CENTER, [0.0, 0.0])
                .show(ctx, |ui| {
                    ui.set_min_width(300.0);
                    ui.set_min_height(100.0);

                    ui.vertical_centered(|ui| {
                        ui.add_space(20.0);
                        ui.label(&error);
                        ui.add_space(20.0);
                    });

                    ui.horizontal_centered(|ui| {
                        if ui.button("确定").clicked() {
                            self.state.clear_error();
                            {
                                let mut state = self.state.loading.write();
                                state.error = None;
                            }
                        }
                    });
                });
        }

        // GPU 渲染集成（P11 锐评落实：通过 Renderer trait 统一接口）
        // 注意：eframe 0.29 中 wgpu_state API 已变更，暂时禁用此功能
        // 未来版本将使用新的 egui_wgpu 集成方式
        #[cfg(all(feature = "gpu", test))]  // 仅在测试时编译，避免错误
        if let Some(ref mut renderer) = self.gpu_renderer {
            // 暂时禁用 GPU 渲染提交
            // if let Some(wgpu_state) = _frame.wgpu_state() {
            //     if let Err(e) = renderer.submit_gpu_commands(
            //         wgpu_state,
            //         self.state.render.camera.zoom as f32,
            //     ) {
            //         log::error!("GPU 渲染失败：{}", e);
            //     }
            // }
        }
    }
}
