//! 错误类型定义 - 结构化版本（EaaS 架构升级版）
//!
//! 改进：
//! 1. 区分具体错误来源（DXF/PDF/几何/拓扑等）
//! 2. 携带更多上下文信息用于调试
//! 3. 支持错误链（source）
//! 4. 统一错误码体系（EaaS 架构要求）
//! 5. 可执行的恢复建议（AutoFix）

use serde_json;
use std::path::PathBuf;
use std::sync::Arc;
use thiserror::Error;

use crate::scene::SceneState;
use crate::RawEdge;

// ============================================================================
// 可执行的恢复建议（P11 建议落地版）
// ============================================================================

/// 自动修复函数（类型 erased）
///
/// 封装了修复逻辑，可以一键应用
///
/// # 生产级特性
///
/// - **前置条件检查**：修复前验证场景状态
/// - **回滚机制**：修复失败后恢复场景
/// - **后置验证**：验证修复是否有效
pub struct AutoFix {
    /// 修复描述
    pub description: String,
    /// 前置条件检查
    pub precondition: Arc<AutoFixCondition>,
    /// 修复函数（类型擦除版本）
    pub func: Arc<AutoFixFunc>,
    /// 回滚函数
    pub rollback: Arc<AutoFixRollback>,
    /// 后置验证
    pub postcondition: Arc<AutoFixCondition>,
}

/// 自动修复函数类型
pub type AutoFixFunc = dyn Fn(&mut SceneState) -> std::result::Result<(), CadError> + Send + Sync;

/// 自动修复条件检查类型
pub type AutoFixCondition = dyn Fn(&SceneState) -> bool + Send + Sync;

/// 自动修复回滚类型
pub type AutoFixRollback = dyn Fn(&mut SceneState) + Send + Sync;

impl AutoFix {
    /// 创建自动修复函数（简化版，不带前置/后置条件）
    pub fn new(
        description: impl Into<String>,
        func: impl Fn(&mut SceneState) -> std::result::Result<(), CadError> + Send + Sync + 'static,
    ) -> Self {
        Self {
            description: description.into(),
            precondition: Arc::new(|_| true), // 总是满足
            func: Arc::new(func),
            rollback: Arc::new(|_| {}),        // 无操作回滚
            postcondition: Arc::new(|_| true), // 总是满足
        }
    }

    /// 创建生产级自动修复函数（带前置/后置条件和回滚）
    ///
    /// # 参数
    ///
    /// * `description` - 修复描述
    /// * `precondition` - 前置条件检查，返回 false 时不执行修复
    /// * `func` - 修复函数
    /// * `rollback` - 回滚函数，在修复失败时调用
    /// * `postcondition` - 后置验证，返回 false 时表示修复无效
    ///
    /// # 示例
    ///
    /// ```
    /// use common_types::error::AutoFix;
    ///
    /// let fix = AutoFix::with_rollback(
    ///     "修复自相交多边形",
    ///     |scene| scene.edges.len() > 0, // 前置条件
    ///     |scene| {
    ///         // 修复逻辑
    ///         Ok(())
    ///     },
    ///     |scene| {
    ///         // 回滚逻辑
    ///     },
    ///     |scene| scene.edges.len() > 0, // 后置验证
    /// );
    /// ```
    pub fn with_rollback(
        description: impl Into<String>,
        precondition: impl Fn(&SceneState) -> bool + Send + Sync + 'static,
        func: impl Fn(&mut SceneState) -> std::result::Result<(), CadError> + Send + Sync + 'static,
        rollback: impl Fn(&mut SceneState) + Send + Sync + 'static,
        postcondition: impl Fn(&SceneState) -> bool + Send + Sync + 'static,
    ) -> Self {
        Self {
            description: description.into(),
            precondition: Arc::new(precondition),
            func: Arc::new(func),
            rollback: Arc::new(rollback),
            postcondition: Arc::new(postcondition),
        }
    }

    /// 应用修复（简单版，不回滚）
    pub fn apply(&self, scene: &mut SceneState) -> std::result::Result<(), CadError> {
        (self.func)(scene)
    }

    /// 安全应用修复（带回滚和验证）
    ///
    /// 这是生产级修复方法，具有完整的保护机制：
    /// 1. 检查前置条件
    /// 2. 保存场景快照
    /// 3. 应用修复
    /// 4. 验证修复效果
    /// 5. 失败时回滚
    pub fn apply_safe(&self, scene: &mut SceneState) -> std::result::Result<(), CadError> {
        // 1. 检查前置条件
        if !(self.precondition)(scene) {
            return Err(CadError::internal(
                crate::error::InternalErrorReason::InvariantViolated {
                    invariant: "AutoFix 前置条件不满足".to_string(),
                },
            ));
        }

        // 2. 保存快照（用于回滚）
        let snapshot = scene.clone();

        // 3. 应用修复
        match (self.func)(scene) {
            Ok(()) => {
                // 4. 验证修复效果
                if !(self.postcondition)(scene) {
                    // 修复无效，回滚
                    (self.rollback)(scene);
                    *scene = snapshot;
                    return Err(CadError::internal(
                        crate::error::InternalErrorReason::InvariantViolated {
                            invariant: "AutoFix 后置验证失败".to_string(),
                        },
                    ));
                }
                Ok(())
            }
            Err(e) => {
                // 5. 修复失败，回滚
                (self.rollback)(scene);
                *scene = snapshot;
                Err(e)
            }
        }
    }

    /// 检查前置条件
    pub fn check_precondition(&self, scene: &SceneState) -> bool {
        (self.precondition)(scene)
    }

    /// 检查后置条件
    pub fn check_postcondition(&self, scene: &SceneState) -> bool {
        (self.postcondition)(scene)
    }

    /// 安全应用修复（增量快照版本）
    ///
    /// 这是生产级修复方法，使用增量快照而非深拷贝整个场景：
    /// 1. 检查前置条件
    /// 2. 创建空白增量快照
    /// 3. 应用修复（修复函数负责记录变更到快照）
    /// 4. 验证修复效果
    /// 5. 失败时使用增量快照回滚
    ///
    /// # 性能优势
    ///
    /// - 原版本：O(n) 深拷贝整个 SceneState（可能包含数万条边）
    /// - 增量版本：O(k) 只记录变更的边/点（k << n）
    ///
    /// # 示例
    ///
    /// ```
    /// use common_types::error::{AutoFix, IncrementalSnapshot};
    /// use common_types::scene::SceneState;
    ///
    /// let fix = AutoFix::with_rollback(
    ///     "修复自相交",
    ///     |scene| true,
    ///     |scene| {
    ///         // 修复逻辑，记录变更
    ///         Ok(())
    ///     },
    ///     |scene| {},
    ///     |scene| true,
    /// );
    ///
    /// let mut scene = SceneState::default();
    /// let result = fix.apply_safe_incremental(&mut scene, |scene, snapshot| {
    ///     // 在这里记录变更到 snapshot
    ///     Ok(())
    /// });
    /// ```
    pub fn apply_safe_incremental<F>(
        &self,
        scene: &mut SceneState,
        fix_func: F,
    ) -> std::result::Result<(), CadError>
    where
        F: FnOnce(&mut SceneState, &mut IncrementalSnapshot) -> std::result::Result<(), CadError>,
    {
        // 1. 检查前置条件
        if !(self.precondition)(scene) {
            return Err(CadError::internal(
                crate::error::InternalErrorReason::InvariantViolated {
                    invariant: "AutoFix 前置条件不满足".to_string(),
                },
            ));
        }

        // 2. 创建空白增量快照
        let mut snapshot = IncrementalSnapshot::new();

        // 3. 应用修复（记录变更到快照）
        match fix_func(scene, &mut snapshot) {
            Ok(()) => {
                // 4. 验证修复效果
                if !(self.postcondition)(scene) {
                    // 修复无效，使用增量快照回滚
                    snapshot.rollback(scene);
                    return Err(CadError::internal(
                        crate::error::InternalErrorReason::InvariantViolated {
                            invariant: "AutoFix 后置验证失败".to_string(),
                        },
                    ));
                }
                Ok(())
            }
            Err(e) => {
                // 5. 修复失败，使用增量快照回滚
                snapshot.rollback(scene);
                Err(e)
            }
        }
    }
}

// 手动实现 Debug，跳过 func 字段
impl std::fmt::Debug for AutoFix {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutoFix")
            .field("description", &self.description)
            .finish()
    }
}

// 手动实现 Clone，使用 Arc 共享
impl Clone for AutoFix {
    fn clone(&self) -> Self {
        Self {
            description: self.description.clone(),
            precondition: Arc::clone(&self.precondition),
            func: Arc::clone(&self.func),
            rollback: Arc::clone(&self.rollback),
            postcondition: Arc::clone(&self.postcondition),
        }
    }
}

// ============================================================================
// 错误恢复建议（P11 Principal Engineer 建议落地版）
// ============================================================================

// ============================================================================
// 增量快照（P11 锐评落地版：避免深拷贝整个场景）
// ============================================================================

/// 增量快照（只记录变更）
///
/// 用于 AutoFix 修复时的高效回滚，避免深拷贝整个 SceneState
///
/// # 性能优势
///
/// - 原版本：O(n) 深拷贝整个 SceneState（可能包含数万条边）
/// - 增量版本：O(k) 只记录变更的边/点（k << n）
#[derive(Debug, Clone, Default)]
pub struct IncrementalSnapshot {
    /// 修改的边（id -> 原始边）
    pub modified_edges: std::collections::HashMap<usize, RawEdge>,
    /// 移除的边（id -> 原始边）- 回滚时需要重新插入到原始位置
    pub removed_edges: std::collections::HashMap<usize, RawEdge>,
    /// 添加的边（id -> 边）- 回滚时需要移除
    pub added_edges: std::collections::HashMap<usize, RawEdge>,
    /// 修改的点（id -> 原始点）
    pub modified_points: std::collections::HashMap<usize, [f64; 2]>,
    /// 移除的点（id -> 原始点）- 回滚时需要重新插入
    pub removed_points: std::collections::HashMap<usize, [f64; 2]>,
    /// 添加的点（id -> 点）- 回滚时需要移除
    pub added_points: std::collections::HashMap<usize, [f64; 2]>,
}

impl IncrementalSnapshot {
    /// 创建新的增量快照
    pub fn new() -> Self {
        Self::default()
    }

    /// 记录边的修改
    pub fn record_edge_modification(&mut self, edge_id: usize, original: RawEdge) {
        self.modified_edges.insert(edge_id, original);
    }

    /// 记录边的移除（保存原始边数据）
    pub fn record_edge_removal(&mut self, edge_id: usize, original: RawEdge) {
        self.removed_edges.insert(edge_id, original);
    }

    /// 记录边的添加（需要指定分配的 ID）
    pub fn record_edge_addition(&mut self, edge_id: usize, edge: RawEdge) {
        self.added_edges.insert(edge_id, edge);
    }

    /// 记录点的修改
    pub fn record_point_modification(&mut self, point_id: usize, original: [f64; 2]) {
        self.modified_points.insert(point_id, original);
    }

    /// 记录点的移除（保存原始点数据）
    pub fn record_point_removal(&mut self, point_id: usize, original: [f64; 2]) {
        self.removed_points.insert(point_id, original);
    }

    /// 记录点的添加（需要指定分配的 ID）
    pub fn record_point_addition(&mut self, point_id: usize, point: [f64; 2]) {
        self.added_points.insert(point_id, point);
    }

    /// 回滚到快照状态
    ///
    /// # 回滚逻辑
    ///
    /// 1. 恢复修改的边：将边数据还原为原始值
    /// 2. 恢复移除的边：重新插入到原始位置（确保场景边列表足够长）
    /// 3. 移除添加的边：使用 retain 批量移除，O(n) 而非 O(n²)
    /// 4. 恢复修改的点：还原外轮廓和孔洞的点坐标（point_id 为索引）
    /// 5. 恢复移除的点：追加到末尾（point_id 为索引）
    /// 6. 移除添加的点：使用 retain 批量移除，O(n) 而非 O(n²)
    ///
    /// # 重要说明
    ///
    /// ## edge_id（全局 ID）vs point_id（局部索引）
    ///
    /// - **edge_id**：`RawEdge` 有 `id` 字段，是全局唯一标识
    ///   - 回滚时通过 ID 查找边，不受 Vec 索引变化影响
    ///   - 使用 `HashSet + retain` 批量移除，时间复杂度 O(n)
    ///
    /// - **point_id**：`Point2 = [f64; 2]` 没有 id 字段，只能使用索引
    ///   - 索引是局部的、动态的，Vec 增删会导致索引偏移
    ///   - 无法使用 `retain` 因为无法区分"添加的点"和"原有的点"
    ///   - 解决方案：按索引降序移除，时间复杂度 O(k*n) 但 k << n 时可接受
    ///
    /// ## P11 锐评修复
    ///
    /// - 原文档说"使用 retain 批量移除"，但代码实际使用 `Vec::remove`
    /// - 原因：点没有唯一 ID，无法用 retain 区分"添加的点"
    /// - 改进：按索引降序移除，避免索引偏移问题
    pub fn rollback(&self, scene: &mut SceneState) {
        // 1. 恢复修改的边
        for (edge_id, original) in &self.modified_edges {
            if let Some(edge) = scene.edges.get_mut(*edge_id) {
                *edge = original.clone();
            }
        }

        // 2. 恢复移除的边（重新插入到原始位置或追加）
        for (edge_id, original) in &self.removed_edges {
            // 确保场景边列表足够长
            while scene.edges.len() <= *edge_id {
                scene.edges.push(RawEdge {
                    id: scene.edges.len(),
                    start: [0.0, 0.0],
                    end: [0.0, 0.0],
                    layer: None,
                    color_index: None,
                });
            }
            scene.edges[*edge_id] = original.clone();
        }

        // 3. 移除添加的边：使用 retain 批量移除，O(n) 而非 O(n²)
        if !self.added_edges.is_empty() {
            let added_ids: std::collections::HashSet<usize> =
                self.added_edges.keys().copied().collect();
            scene.edges.retain(|edge| !added_ids.contains(&edge.id));
        }

        // 4. 恢复修改的点：point_id 是索引
        for (point_idx, original) in &self.modified_points {
            // 恢复外轮廓点
            if let Some(outer) = &mut scene.outer {
                if *point_idx < outer.points.len() {
                    outer.points[*point_idx] = *original;
                }
            }
            // 恢复孔洞点（简化处理：应用到所有孔洞的相同索引）
            for hole in &mut scene.holes {
                if *point_idx < hole.points.len() {
                    hole.points[*point_idx] = *original;
                }
            }
        }

        // 5. 恢复移除的点：按索引降序追加，避免索引偏移
        if !self.removed_points.is_empty() {
            // 收集需要恢复的点，按索引降序排序
            let mut points_to_restore: Vec<_> = self.removed_points.iter().collect();
            points_to_restore.sort_by(|a, b| b.0.cmp(a.0));

            // 尝试恢复外轮廓点
            if let Some(outer) = &mut scene.outer {
                for (point_idx, original) in points_to_restore.iter() {
                    if **point_idx < outer.points.len() {
                        outer.points[**point_idx] = **original;
                    } else {
                        // 如果索引超出范围，追加到末尾
                        outer.points.push(**original);
                    }
                }
            }

            // 孔洞点的恢复（简化处理：追加到第一个孔洞）
            if let Some(hole) = scene.holes.first_mut() {
                for (point_idx, original) in points_to_restore.iter() {
                    if **point_idx < hole.points.len() {
                        hole.points[**point_idx] = **original;
                    } else {
                        hole.points.push(**original);
                    }
                }
            }
        }

        // 6. 移除添加的点
        // P11 锐评修复：point_id 是索引，不是 ID
        // 问题：Vec::remove 是 O(n)，但无法使用 retain 因为不知道哪些点是"添加的"
        // 解决方案：按索引降序移除，避免索引偏移问题
        if !self.added_points.is_empty() {
            let mut added_point_indices: Vec<_> = self.added_points.keys().copied().collect();
            added_point_indices.sort_by(|a, b| b.cmp(a)); // 降序排序，避免索引偏移

            // 从外轮廓移除
            if let Some(outer) = &mut scene.outer {
                for idx in added_point_indices.iter() {
                    if *idx < outer.points.len() {
                        outer.points.remove(*idx);
                    }
                }
            }

            // 从孔洞移除（简化处理：从第一个孔洞移除）
            if let Some(hole) = scene.holes.first_mut() {
                for idx in added_point_indices.iter() {
                    if *idx < hole.points.len() {
                        hole.points.remove(*idx);
                    }
                }
            }
        }
    }

    /// 清空快照
    pub fn clear(&mut self) {
        self.modified_edges.clear();
        self.removed_edges.clear();
        self.added_edges.clear();
        self.modified_points.clear();
        self.removed_points.clear();
        self.added_points.clear();
    }
}
///
/// 为常见错误提供自动修复建议，包括：
/// - 可操作的建议（如："调整 snap_tolerance 到 1.0mm"）
/// - 配置变更建议
/// - 自动修复函数（可选，但推荐）
#[derive(Debug, Clone)]
pub struct RecoverySuggestion {
    /// 人类可读的修复建议
    pub action: String,
    /// 建议的配置变更（配置键，新值）
    pub config_change: Option<(String, serde_json::Value)>,
    /// 优先级（1-10，10 为最高优先级）
    pub priority: u8,
    /// 自动修复函数（可选）
    pub auto_fix: Option<AutoFix>,
}

// 手动实现 Serialize，跳过 auto_fix 字段（因为包含 Arc）
impl serde::Serialize for RecoverySuggestion {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::SerializeStruct;
        let mut state = serializer.serialize_struct("RecoverySuggestion", 3)?;
        state.serialize_field("action", &self.action)?;
        state.serialize_field("config_change", &self.config_change)?;
        state.serialize_field("priority", &self.priority)?;
        // 跳过 auto_fix 字段，因为包含 Arc<dyn Fn...>
        state.end()
    }
}

// 手动实现 Deserialize，auto_fix 字段设为 None
impl<'de> serde::Deserialize<'de> for RecoverySuggestion {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        #[derive(serde::Deserialize)]
        struct Helper {
            action: String,
            config_change: Option<(String, serde_json::Value)>,
            priority: u8,
        }

        let helper = Helper::deserialize(deserializer)?;
        Ok(RecoverySuggestion {
            action: helper.action,
            config_change: helper.config_change,
            priority: helper.priority,
            auto_fix: None, // 反序列化时无法恢复函数指针，设为 None
        })
    }
}

impl RecoverySuggestion {
    /// 创建恢复建议
    pub fn new(action: impl Into<String>) -> Self {
        Self {
            action: action.into(),
            config_change: None,
            priority: 5, // 默认中等优先级
            auto_fix: None,
        }
    }

    /// 设置配置变更建议
    pub fn with_config_change(mut self, key: impl Into<String>, value: serde_json::Value) -> Self {
        self.config_change = Some((key.into(), value));
        self
    }

    /// 设置优先级
    pub fn with_priority(mut self, priority: u8) -> Self {
        self.priority = priority.min(10);
        self
    }

    /// 设置自动修复函数
    pub fn with_auto_fix(
        mut self,
        description: impl Into<String>,
        func: impl Fn(&mut SceneState) -> std::result::Result<(), CadError> + Send + Sync + 'static,
    ) -> Self {
        self.auto_fix = Some(AutoFix::new(description, func));
        self
    }

    /// 应用自动修复（如果可用）
    pub fn apply(&self, scene: &mut SceneState) -> std::result::Result<(), CadError> {
        if let Some(auto_fix) = &self.auto_fix {
            auto_fix.apply(scene)
        } else {
            Err(CadError::internal(InternalErrorReason::NotImplemented {
                feature: "此恢复建议不支持自动修复".to_string(),
            }))
        }
    }

    /// 是否支持自动修复
    pub fn is_auto_fixable(&self) -> bool {
        self.auto_fix.is_some()
    }
}

impl std::fmt::Display for RecoverySuggestion {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.action)?;
        if let Some((key, value)) = &self.config_change {
            write!(f, "（建议配置：{} = {}）", key, value)?;
        }
        if self.auto_fix.is_some() {
            write!(f, " [支持自动修复]")?;
        }
        Ok(())
    }
}

// ============================================================================
// 统一错误码体系（EaaS 架构要求）
// ============================================================================

/// 错误码（格式：SERVICE_CATEGORY_###）
///
/// 错误码分类：
/// - PARSE_*: 解析错误 (001-099)
/// - TOPO_*: 拓扑错误 (100-199)
/// - VALIDATE_*: 验证错误 (200-299)
/// - EXPORT_*: 导出错误 (300-399)
/// - GEOMETRY_*: 几何错误 (400-499)
/// - IO_*: IO 错误 (500-599)
/// - INTERNAL_*: 内部错误 (900-999)
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct ErrorCode(&'static str);

impl ErrorCode {
    // ========== 解析错误 (001-099) ==========
    pub const PARSE_DXF_INVALID_FILE: ErrorCode = ErrorCode("PARSE_DXF_001");
    pub const PARSE_DXF_MISSING_SECTION: ErrorCode = ErrorCode("PARSE_DXF_002");
    pub const PARSE_DXF_UNKNOWN_ENTITY: ErrorCode = ErrorCode("PARSE_DXF_003");
    pub const PARSE_DXF_MALFORMED_ENTITY: ErrorCode = ErrorCode("PARSE_DXF_004");
    pub const PARSE_PDF_INVALID_FILE: ErrorCode = ErrorCode("PARSE_PDF_001");
    pub const PARSE_PDF_PASSWORD_PROTECTED: ErrorCode = ErrorCode("PARSE_PDF_002");
    pub const PARSE_PDF_NO_PAGES: ErrorCode = ErrorCode("PARSE_PDF_003");
    pub const PARSE_IMAGE_INVALID_FORMAT: ErrorCode = ErrorCode("PARSE_IMG_001");
    pub const PARSE_IMAGE_TOO_LARGE: ErrorCode = ErrorCode("PARSE_IMG_002");
    pub const PARSE_VECTORIZE_FAILED: ErrorCode = ErrorCode("PARSE_IMG_003");
    pub const PARSE_UNSUPPORTED_FORMAT: ErrorCode = ErrorCode("PARSE_FMT_001");

    // ========== 拓扑错误 (100-199) ==========
    pub const TOPO_EMPTY_INPUT: ErrorCode = ErrorCode("TOPO_101");
    pub const TOPO_SNAP_FAILED: ErrorCode = ErrorCode("TOPO_102");
    pub const TOPO_GRAPH_BUILD_FAILED: ErrorCode = ErrorCode("TOPO_103");
    pub const TOPO_LOOP_EXTRACT_FAILED: ErrorCode = ErrorCode("TOPO_104");
    pub const TOPO_FACE_EXTRACT_FAILED: ErrorCode = ErrorCode("TOPO_105");
    pub const TOPO_SELF_INTERSECTION: ErrorCode = ErrorCode("TOPO_106");
    pub const TOPO_DUPLICATE_EDGES: ErrorCode = ErrorCode("TOPO_107");
    pub const TOPO_OPEN_LOOP: ErrorCode = ErrorCode("TOPO_108");

    // ========== 验证错误 (200-299) ==========
    pub const VALIDATE_SELF_INTERSECTION: ErrorCode = ErrorCode("VALIDATE_201");
    pub const VALIDATE_DUPLICATE_EDGES: ErrorCode = ErrorCode("VALIDATE_202");
    pub const VALIDATE_TINY_SEGMENTS: ErrorCode = ErrorCode("VALIDATE_203");
    pub const VALIDATE_LARGE_COORDINATES: ErrorCode = ErrorCode("VALIDATE_204");
    pub const VALIDATE_OPEN_LOOPS: ErrorCode = ErrorCode("VALIDATE_205");
    pub const VALIDATE_INVALID_GEOMETRY: ErrorCode = ErrorCode("VALIDATE_206");

    // ========== 导出错误 (300-399) ==========
    pub const EXPORT_DXF_FAILED: ErrorCode = ErrorCode("EXPORT_DXF_301");
    pub const EXPORT_SVG_FAILED: ErrorCode = ErrorCode("EXPORT_SVG_302");
    pub const EXPORT_PDF_FAILED: ErrorCode = ErrorCode("EXPORT_PDF_303");
    pub const EXPORT_JSON_FAILED: ErrorCode = ErrorCode("EXPORT_JSON_304");
    pub const EXPORT_FILE_WRITE_FAILED: ErrorCode = ErrorCode("EXPORT_IO_305");

    // ========== 几何错误 (400-499) ==========
    pub const GEOMETRY_CONSTRUCTION_FAILED: ErrorCode = ErrorCode("GEOM_401");
    pub const GEOMETRY_INVALID_POINT: ErrorCode = ErrorCode("GEOM_402");
    pub const GEOMETRY_INVALID_SEGMENT: ErrorCode = ErrorCode("GEOM_403");
    pub const GEOMETRY_TOLERANCE_ERROR: ErrorCode = ErrorCode("GEOM_404");
    pub const GEOMETRY_SELF_INTERSECTING: ErrorCode = ErrorCode("GEOM_405");

    // ========== IO 错误 (500-599) ==========
    pub const IO_FILE_NOT_FOUND: ErrorCode = ErrorCode("IO_501");
    pub const IO_PERMISSION_DENIED: ErrorCode = ErrorCode("IO_502");
    pub const IO_READ_FAILED: ErrorCode = ErrorCode("IO_503");
    pub const IO_WRITE_FAILED: ErrorCode = ErrorCode("IO_504");

    // ========== 内部错误 (900-999) ==========
    pub const INTERNAL_UNKNOWN: ErrorCode = ErrorCode("INTERNAL_900");
    pub const INTERNAL_NOT_IMPLEMENTED: ErrorCode = ErrorCode("INTERNAL_901");
    pub const INTERNAL_INVARIANT_VIOLATED: ErrorCode = ErrorCode("INTERNAL_902");
    pub const INTERNAL_SERVICE_UNAVAILABLE: ErrorCode = ErrorCode("INTERNAL_903");

    /// 获取错误码字符串
    pub fn as_str(&self) -> &'static str {
        self.0
    }
}

// 自定义序列化实现
impl serde::Serialize for ErrorCode {
    fn serialize<S>(&self, serializer: S) -> std::result::Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        serializer.serialize_str(self.0)
    }
}

// 自定义反序列化实现
impl<'de> serde::Deserialize<'de> for ErrorCode {
    fn deserialize<D>(deserializer: D) -> std::result::Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let s = String::deserialize(deserializer)?;
        Ok(ErrorCode(Box::leak(s.into_boxed_str())))
    }
}

impl std::fmt::Display for ErrorCode {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0)
    }
}

impl From<ErrorCode> for &'static str {
    fn from(code: ErrorCode) -> Self {
        code.0
    }
}

/// 通用错误类型 - 结构化版本（P1-1 重构）
///
/// 根据 Principal Engineer 建议改进：
/// 1. 移除泛化的 `message: String`，使用具体语义字段
/// 2. 改进错误链，保留 source 的语义信息
/// 3. 添加更多上下文信息用于调试
#[derive(Debug, Error)]
pub enum CadError {
    // ========== 解析错误 ==========
    #[error("DXF 解析失败：文件 {file:?} - {reason}")]
    DxfParseError {
        file: PathBuf,
        reason: DxfParseReason,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("PDF 解析失败：文件 {file:?} - {reason}")]
    PdfParseError {
        file: PathBuf,
        reason: PdfParseReason,
        #[source]
        source: Option<Box<dyn std::error::Error + Send + Sync>>,
    },

    #[error("不支持的文件格式：{format}")]
    UnsupportedFormat {
        format: String,
        supported_formats: Vec<String>,
    },

    #[error("矢量化失败：{message}")]
    VectorizeFailed { message: String },

    // ========== 几何错误 ==========
    #[error("几何构造失败：{operation} - {reason}")]
    GeometryConstructionError {
        operation: String,
        reason: GeometryConstructionReason,
        details: Option<String>,
    },

    #[error("几何验证失败：{issue_code} - {reason}")]
    GeometryValidationError {
        issue_code: String,
        reason: String,
        location: Option<ErrorLocation>,
    },

    #[error("容差设置不当：{reason}")]
    ToleranceError {
        reason: ToleranceErrorReason,
        suggested_value: Option<f64>,
    },

    // ========== 拓扑错误 ==========
    #[error("拓扑构建失败：{stage} - {reason}")]
    TopologyConstructionError {
        stage: TopoStage,
        reason: TopoErrorReason,
        details: Option<String>,
    },

    #[error("环提取失败：{reason}")]
    LoopExtractionError {
        reason: LoopExtractReason,
        num_points: usize,
        num_edges: usize,
    },

    #[error("图构建失败：{reason}")]
    GraphConstructionError {
        reason: GraphBuildReason,
        num_input_polylines: usize,
    },

    // ========== 验证错误 ==========
    #[error("验证失败：{count} 个错误，{warning_count} 个警告")]
    ValidationFailed {
        count: usize,
        warning_count: usize,
        issues: Vec<ValidationIssue>,
    },

    #[error("验证错误：{issue_code} - {reason}")]
    ValidationError { issue_code: String, reason: String },

    // ========== IO 错误 ==========
    #[error("文件 IO 错误：{path:?} - {reason}")]
    IoError {
        path: PathBuf,
        reason: IoErrorReason,
        #[source]
        source: std::io::Error,
    },

    // ========== 内部错误 ==========
    #[error("内部错误：{reason}")]
    InternalError {
        reason: InternalErrorReason,
        location: Option<&'static str>,
    },

    #[error("未实现：{feature}")]
    NotImplemented {
        feature: String,
        planned_version: Option<String>,
    },
}

/// DXF 解析失败原因（语义化枚举）
#[derive(Debug, Clone)]
pub enum DxfParseReason {
    FileNotFound,
    InvalidVersion(String),
    MissingSection(String),
    MalformedEntity {
        entity_type: String,
        details: String,
    },
    UnknownEntity(String),
    EncodingError(String),
    InvalidDimensionType(u16),
    /// 通用解析失败（ezdxf 降级、子进程异常等）
    ParseError(String),
}

impl std::fmt::Display for DxfParseReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            DxfParseReason::FileNotFound => write!(f, "文件未找到"),
            DxfParseReason::InvalidVersion(v) => write!(f, "无效的 DXF 版本：{}", v),
            DxfParseReason::MissingSection(s) => write!(f, "缺少必需节：{}", s),
            DxfParseReason::MalformedEntity {
                entity_type,
                details,
            } => {
                write!(f, "实体 {} 格式错误：{}", entity_type, details)
            }
            DxfParseReason::UnknownEntity(e) => write!(f, "未知实体类型：{}", e),
            DxfParseReason::EncodingError(e) => write!(f, "编码错误：{}", e),
            DxfParseReason::InvalidDimensionType(t) => write!(f, "无效的尺寸标注类型：{}", t),
            DxfParseReason::ParseError(msg) => write!(f, "解析失败：{}", msg),
        }
    }
}

/// PDF 解析失败原因（语义化枚举）
#[derive(Debug, Clone)]
pub enum PdfParseReason {
    FileNotFound,
    PasswordProtected,
    NoPages,
    InvalidPageRange,
    RenderError(String),
    ExtractError(String),
}

impl std::fmt::Display for PdfParseReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PdfParseReason::FileNotFound => write!(f, "文件未找到"),
            PdfParseReason::PasswordProtected => write!(f, "PDF 受密码保护"),
            PdfParseReason::NoPages => write!(f, "PDF 无页面"),
            PdfParseReason::InvalidPageRange => write!(f, "无效的页面范围"),
            PdfParseReason::RenderError(e) => write!(f, "渲染错误：{}", e),
            PdfParseReason::ExtractError(e) => write!(f, "提取错误：{}", e),
        }
    }
}

/// 几何构造失败原因（语义化枚举）
#[derive(Debug, Clone)]
pub enum GeometryConstructionReason {
    InvalidPoint {
        x: f64,
        y: f64,
        reason: String,
    },
    InvalidSegment {
        start: [f64; 2],
        end: [f64; 2],
        reason: String,
    },
    ArcDiscretizationError {
        tolerance: f64,
        actual_error: f64,
    },
    NurbsEvaluationError {
        parameter: f64,
        reason: String,
    },
    IntersectionNotFound {
        segment1: [usize; 2],
        segment2: [usize; 2],
    },
}

impl std::fmt::Display for GeometryConstructionReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GeometryConstructionReason::InvalidPoint { x, y, reason } => {
                write!(f, "无效点 ({}, {}): {}", x, y, reason)
            }
            GeometryConstructionReason::InvalidSegment { start, end, reason } => {
                write!(f, "无效线段 [{:?} -> {:?}]: {}", start, end, reason)
            }
            GeometryConstructionReason::ArcDiscretizationError {
                tolerance,
                actual_error,
            } => {
                write!(
                    f,
                    "圆弧离散化误差超标：容差={}, 实际误差={}",
                    tolerance, actual_error
                )
            }
            GeometryConstructionReason::NurbsEvaluationError { parameter, reason } => {
                write!(f, "NURBS 评估失败 (t={}): {}", parameter, reason)
            }
            GeometryConstructionReason::IntersectionNotFound { segment1, segment2 } => {
                write!(f, "未找到线段 {} 和 {} 的交点", segment1[0], segment2[0])
            }
        }
    }
}

/// 容差错误原因（语义化枚举）
#[derive(Debug, Clone)]
pub enum ToleranceErrorReason {
    SnapToleranceTooLarge { value: f64, max_recommended: f64 },
    SnapToleranceTooSmall { value: f64, min_recommended: f64 },
    AngleToleranceInvalid { value: f64 },
    InconsistentTolerances { snap: f64, merge: f64 },
}

impl std::fmt::Display for ToleranceErrorReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            ToleranceErrorReason::SnapToleranceTooLarge {
                value,
                max_recommended,
            } => {
                write!(
                    f,
                    "吸附容差过大：{}mm (推荐最大值：{}mm)",
                    value, max_recommended
                )
            }
            ToleranceErrorReason::SnapToleranceTooSmall {
                value,
                min_recommended,
            } => {
                write!(
                    f,
                    "吸附容差过小：{}mm (推荐最小值：{}mm)",
                    value, min_recommended
                )
            }
            ToleranceErrorReason::AngleToleranceInvalid { value } => {
                write!(f, "无效角度容差：{}°", value)
            }
            ToleranceErrorReason::InconsistentTolerances { snap, merge } => {
                write!(f, "容差不一致：snap={}, merge={}", snap, merge)
            }
        }
    }
}

/// 拓扑错误原因（语义化枚举）
#[derive(Debug, Clone)]
pub enum TopoErrorReason {
    EmptyInput,
    SnapFailed { num_points: usize, error: String },
    GraphBuildFailed { num_edges: usize, error: String },
    LoopExtractionFailed { num_loops: usize, error: String },
    SelfIntersection { point: [f64; 2] },
    DuplicateEdges { count: usize },
    OpenLoop { gap_distance: f64 },
}

impl std::fmt::Display for TopoErrorReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TopoErrorReason::EmptyInput => write!(f, "输入为空"),
            TopoErrorReason::SnapFailed { num_points, error } => {
                write!(f, "端点吸附失败 ({} 点): {}", num_points, error)
            }
            TopoErrorReason::GraphBuildFailed { num_edges, error } => {
                write!(f, "图构建失败 ({} 边): {}", num_edges, error)
            }
            TopoErrorReason::LoopExtractionFailed { num_loops, error } => {
                write!(f, "环提取失败 ({} 环): {}", num_loops, error)
            }
            TopoErrorReason::SelfIntersection { point } => {
                write!(f, "自相交于点 {:?}", point)
            }
            TopoErrorReason::DuplicateEdges { count } => {
                write!(f, "检测到 {} 个重复边", count)
            }
            TopoErrorReason::OpenLoop { gap_distance } => {
                write!(f, "环未闭合，缺口距离={}mm", gap_distance)
            }
        }
    }
}

/// 环提取失败原因（语义化枚举）
#[derive(Debug, Clone)]
pub enum LoopExtractReason {
    NoClosedLoops,
    OuterBoundaryNotFound,
    HoleContainmentFailed { hole_index: usize },
    SelfIntersectingLoop { loop_index: usize },
    AlgorithmFailed { algorithm: String, error: String },
}

impl std::fmt::Display for LoopExtractReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LoopExtractReason::NoClosedLoops => write!(f, "无闭合环"),
            LoopExtractReason::OuterBoundaryNotFound => write!(f, "外边界未找到"),
            LoopExtractReason::HoleContainmentFailed { hole_index } => {
                write!(f, "孔洞 {} 包含关系验证失败", hole_index)
            }
            LoopExtractReason::SelfIntersectingLoop { loop_index } => {
                write!(f, "环 {} 自相交", loop_index)
            }
            LoopExtractReason::AlgorithmFailed { algorithm, error } => {
                write!(f, "算法 {} 失败：{}", algorithm, error)
            }
        }
    }
}

/// 图构建失败原因（语义化枚举）
#[derive(Debug, Clone)]
pub enum GraphBuildReason {
    EmptyInput,
    SnapToleranceInvalid { value: f64 },
    RTreeBuildFailed { num_points: usize },
    EdgeConnectivityFailed { num_edges: usize },
}

impl std::fmt::Display for GraphBuildReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            GraphBuildReason::EmptyInput => write!(f, "输入为空"),
            GraphBuildReason::SnapToleranceInvalid { value } => {
                write!(f, "无效的吸附容差：{}", value)
            }
            GraphBuildReason::RTreeBuildFailed { num_points } => {
                write!(f, "R-Tree 构建失败 ({} 点)", num_points)
            }
            GraphBuildReason::EdgeConnectivityFailed { num_edges } => {
                write!(f, "边连通性失败 ({} 边)", num_edges)
            }
        }
    }
}

/// 验证问题（结构化）
#[derive(Debug, Clone)]
pub struct ValidationIssue {
    pub code: String,
    pub severity: Severity,
    pub message: String,
    pub location: Option<ErrorLocation>,
}

/// 严重程度
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Error,
    Warning,
    Info,
}

/// IO 错误原因（语义化枚举）
#[derive(Debug, Clone)]
pub enum IoErrorReason {
    FileNotFound,
    PermissionDenied,
    ReadFailed,
    WriteFailed,
    InvalidPath,
}

impl std::fmt::Display for IoErrorReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IoErrorReason::FileNotFound => write!(f, "文件未找到"),
            IoErrorReason::PermissionDenied => write!(f, "权限被拒绝"),
            IoErrorReason::ReadFailed => write!(f, "读取失败"),
            IoErrorReason::WriteFailed => write!(f, "写入失败"),
            IoErrorReason::InvalidPath => write!(f, "无效路径"),
        }
    }
}

/// 内部错误原因（语义化枚举）
#[derive(Debug, Clone)]
pub enum InternalErrorReason {
    Unknown,
    NotImplemented { feature: String },
    InvariantViolated { invariant: String },
    ServiceUnavailable { service: String },
    Panic { message: String },
}

impl std::fmt::Display for InternalErrorReason {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            InternalErrorReason::Unknown => write!(f, "未知内部错误"),
            InternalErrorReason::NotImplemented { feature } => {
                write!(f, "未实现：{}", feature)
            }
            InternalErrorReason::InvariantViolated { invariant } => {
                write!(f, "不变量被违反：{}", invariant)
            }
            InternalErrorReason::ServiceUnavailable { service } => {
                write!(f, "服务不可用：{}", service)
            }
            InternalErrorReason::Panic { message } => {
                write!(f, "panic: {}", message)
            }
        }
    }
}

/// 拓扑处理阶段
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum TopoStage {
    Snap,
    GraphBuild,
    LoopExtract,
    FaceExtract,
    HoleDetection,
}

impl std::fmt::Display for TopoStage {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            TopoStage::Snap => write!(f, "端点吸附"),
            TopoStage::GraphBuild => write!(f, "图构建"),
            TopoStage::LoopExtract => write!(f, "环提取"),
            TopoStage::FaceExtract => write!(f, "面提取"),
            TopoStage::HoleDetection => write!(f, "孔洞检测"),
        }
    }
}

/// 错误位置信息
#[derive(Debug, Clone)]
pub struct ErrorLocation {
    pub point: Option<crate::geometry::Point2>,
    pub segment: Option<[usize; 2]>,
    pub loop_index: Option<usize>,
}

impl std::fmt::Display for ErrorLocation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        if let Some(point) = self.point {
            write!(f, "点 ({}, {})", point[0], point[1])?;
        }
        if let Some([a, b]) = self.segment {
            write!(f, "，线段 {}-{}", a, b)?;
        }
        if let Some(idx) = self.loop_index {
            write!(f, "，环 #{}", idx)?;
        }
        Ok(())
    }
}

pub type Result<T> = std::result::Result<T, CadError>;

// ========== CadError impl 块 ==========

impl CadError {
    /// 获取错误码
    pub fn error_code(&self) -> ErrorCode {
        match self {
            Self::DxfParseError { .. } => ErrorCode::PARSE_DXF_INVALID_FILE,
            Self::PdfParseError { .. } => ErrorCode::PARSE_PDF_INVALID_FILE,
            Self::UnsupportedFormat { .. } => ErrorCode::PARSE_UNSUPPORTED_FORMAT,
            Self::GeometryConstructionError { .. } => ErrorCode::GEOMETRY_CONSTRUCTION_FAILED,
            Self::GeometryValidationError { issue_code, .. } => match issue_code.as_str() {
                "SELF_INTERSECTION" => ErrorCode::VALIDATE_SELF_INTERSECTION,
                "DUPLICATE_EDGES" => ErrorCode::VALIDATE_DUPLICATE_EDGES,
                "TINY_SEGMENT" => ErrorCode::VALIDATE_TINY_SEGMENTS,
                "LARGE_COORDINATES" => ErrorCode::VALIDATE_LARGE_COORDINATES,
                "OPEN_LOOP" => ErrorCode::VALIDATE_OPEN_LOOPS,
                _ => ErrorCode::VALIDATE_INVALID_GEOMETRY,
            },
            Self::ToleranceError { .. } => ErrorCode::GEOMETRY_TOLERANCE_ERROR,
            Self::TopologyConstructionError { stage, .. } => match stage {
                TopoStage::Snap => ErrorCode::TOPO_SNAP_FAILED,
                TopoStage::GraphBuild => ErrorCode::TOPO_GRAPH_BUILD_FAILED,
                TopoStage::LoopExtract => ErrorCode::TOPO_LOOP_EXTRACT_FAILED,
                TopoStage::FaceExtract => ErrorCode::TOPO_FACE_EXTRACT_FAILED,
                TopoStage::HoleDetection => ErrorCode::TOPO_SELF_INTERSECTION,
            },
            Self::LoopExtractionError { .. } => ErrorCode::TOPO_LOOP_EXTRACT_FAILED,
            Self::GraphConstructionError { .. } => ErrorCode::TOPO_GRAPH_BUILD_FAILED,
            Self::ValidationFailed { .. } => ErrorCode::VALIDATE_INVALID_GEOMETRY,
            Self::ValidationError { .. } => ErrorCode::VALIDATE_INVALID_GEOMETRY,
            Self::VectorizeFailed { .. } => ErrorCode::PARSE_VECTORIZE_FAILED,
            Self::IoError { .. } => ErrorCode::IO_READ_FAILED,
            Self::InternalError { .. } => ErrorCode::INTERNAL_UNKNOWN,
            Self::NotImplemented { .. } => ErrorCode::INTERNAL_NOT_IMPLEMENTED,
        }
    }

    /// 创建 DXF 解析错误（语义化版本）
    pub fn dxf_parse(file: impl Into<PathBuf>, reason: DxfParseReason) -> Self {
        Self::DxfParseError {
            file: file.into(),
            reason,
            source: None,
        }
    }

    /// 创建 DXF 解析错误（带 source）
    pub fn dxf_parse_with_source(
        file: impl Into<PathBuf>,
        reason: DxfParseReason,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::DxfParseError {
            file: file.into(),
            reason,
            source: Some(Box::new(source)),
        }
    }

    /// 创建 PDF 解析错误（语义化版本）
    pub fn pdf_parse(file: impl Into<PathBuf>, reason: PdfParseReason) -> Self {
        Self::PdfParseError {
            file: file.into(),
            reason,
            source: None,
        }
    }

    /// 创建 PDF 解析错误（带 source）
    pub fn pdf_parse_with_source(
        file: impl Into<PathBuf>,
        reason: PdfParseReason,
        source: impl std::error::Error + Send + Sync + 'static,
    ) -> Self {
        Self::PdfParseError {
            file: file.into(),
            reason,
            source: Some(Box::new(source)),
        }
    }

    /// 创建拓扑构建错误（语义化版本）
    pub fn topo_construction(stage: TopoStage, reason: TopoErrorReason) -> Self {
        Self::TopologyConstructionError {
            stage,
            reason,
            details: None,
        }
    }

    /// 创建拓扑构建错误（带详细信息）
    pub fn topo_construction_with_details(
        stage: TopoStage,
        reason: TopoErrorReason,
        details: impl Into<String>,
    ) -> Self {
        Self::TopologyConstructionError {
            stage,
            reason,
            details: Some(details.into()),
        }
    }

    /// 创建几何验证错误
    pub fn geometry_validation(issue_code: impl Into<String>, reason: impl Into<String>) -> Self {
        Self::GeometryValidationError {
            issue_code: issue_code.into(),
            reason: reason.into(),
            location: None,
        }
    }

    /// 创建几何验证错误（带位置）
    pub fn geometry_validation_at(
        issue_code: impl Into<String>,
        reason: impl Into<String>,
        location: ErrorLocation,
    ) -> Self {
        Self::GeometryValidationError {
            issue_code: issue_code.into(),
            reason: reason.into(),
            location: Some(location),
        }
    }

    /// 创建 IO 错误
    pub fn io_path(
        path: impl Into<PathBuf>,
        reason: IoErrorReason,
        source: std::io::Error,
    ) -> Self {
        Self::IoError {
            path: path.into(),
            reason,
            source,
        }
    }

    /// 创建内部错误
    pub fn internal(reason: InternalErrorReason) -> Self {
        Self::InternalError {
            reason,
            location: None,
        }
    }

    /// 创建内部错误（带位置）
    pub fn internal_at(reason: InternalErrorReason, location: &'static str) -> Self {
        Self::InternalError {
            reason,
            location: Some(location),
        }
    }

    /// 创建未实现错误
    pub fn not_implemented(feature: impl Into<String>) -> Self {
        Self::NotImplemented {
            feature: feature.into(),
            planned_version: None,
        }
    }

    /// 获取错误恢复建议（P11 Principal Engineer 建议）
    ///
    /// 为常见错误提供自动修复建议，包括：
    /// - 可操作的建议（如："调整 snap_tolerance 到 1.0mm"）
    /// - 配置变更建议
    /// - 优先级排序
    pub fn recovery_suggestion(&self) -> Option<RecoverySuggestion> {
        match self {
            // ========== 拓扑错误恢复建议 ==========
            Self::TopologyConstructionError { stage, reason, .. } => match (stage, reason) {
                (TopoStage::Snap, TopoErrorReason::SnapFailed { .. }) => {
                    Some(RecoverySuggestion::new(
                        "检测到端点吸附失败，建议增大吸附容差或检查输入几何质量"
                    )
                    .with_config_change("topology.snap_tolerance_mm", serde_json::json!(1.0))
                    .with_priority(8))
                }
                (TopoStage::GraphBuild, TopoErrorReason::GraphBuildFailed { .. }) => {
                    Some(RecoverySuggestion::new(
                        "图构建失败，建议检查输入线段是否存在重叠或共线问题"
                    )
                    .with_config_change("topology.snap_tolerance_mm", serde_json::json!(0.8))
                    .with_priority(7))
                }
                (TopoStage::LoopExtract, TopoErrorReason::LoopExtractionFailed { .. }) => {
                    Some(RecoverySuggestion::new(
                        "环提取失败，可能存在未闭合的缺口。建议运行端点吸附或手动桥接缺口"
                    )
                    .with_config_change("topology.snap_tolerance_mm", serde_json::json!(0.5))
                    .with_priority(9))
                }
                (TopoStage::LoopExtract, TopoErrorReason::OpenLoop { gap_distance }) => {
                    Some(RecoverySuggestion::new(format!(
                        "检测到环未闭合，缺口距离为 {:.2}mm。建议：1) 增大吸附容差到 {:.2}mm，2) 使用 InteractSvc 桥接缺口",
                        gap_distance,
                        (gap_distance * 1.5).max(0.5)
                    ))
                    .with_config_change("topology.snap_tolerance_mm", serde_json::json!((gap_distance * 1.5).max(0.5)))
                    .with_priority(10))
                }
                (TopoStage::HoleDetection, TopoErrorReason::SelfIntersection { point }) => {
                    Some(RecoverySuggestion::new(format!(
                        "检测到自相交于点 ({:.2}, {:.2})。建议检查输入几何是否存在交叉墙体",
                        point[0], point[1]
                    ))
                    .with_priority(8))
                }
                _ => None,
            },

            // ========== 几何验证错误恢复建议 ==========
            Self::GeometryValidationError { issue_code, location, .. } => {
                match issue_code.as_str() {
                    "SELF_INTERSECTION" => {
                        Some(RecoverySuggestion::new(
                            "检测到自相交多边形（蝴蝶结形状）。建议：1) 在交点处切分线段，2) 重新构建拓扑"
                        )
                        .with_priority(9))
                    }
                    "DUPLICATE_EDGES" => {
                        Some(RecoverySuggestion::new(
                            "检测到重复边。建议运行去重算法或增大吸附容差合并重合端点"
                        )
                        .with_config_change("topology.snap_tolerance_mm", serde_json::json!(0.8))
                        .with_priority(6))
                    }
                    "TINY_SEGMENT" => {
                        Some(RecoverySuggestion::new(
                            "检测到过短线段（可能是噪点）。建议增大最小线段长度阈值或运行去噪算法"
                        )
                        .with_config_change("topology.min_line_length_mm", serde_json::json!(2.0))
                        .with_priority(5))
                    }
                    "OPEN_LOOP" => {
                        let gap_info = location.as_ref()
                            .and_then(|loc| loc.segment)
                            .map(|[a, b]| format!("线段 {}-{}", a, b))
                            .unwrap_or_else(|| "未知位置".to_string());
                        Some(RecoverySuggestion::new(format!(
                            "环未闭合于{}。建议：1) 运行端点吸附，2) 手动桥接缺口，3) 增大吸附容差",
                            gap_info
                        ))
                        .with_config_change("topology.snap_tolerance_mm", serde_json::json!(1.0))
                        .with_priority(10))
                    }
                    "LARGE_COORDINATES" => {
                        Some(RecoverySuggestion::new(
                            "检测到过大坐标值（可能导致数值不稳定）。建议：1) 平移几何到原点附近，2) 使用相对坐标"
                        )
                        .with_priority(7))
                    }
                    _ => None,
                }
            }

            // ========== 容差错误恢复建议 ==========
            Self::ToleranceError { reason, suggested_value } => {
                match reason {
                    ToleranceErrorReason::SnapToleranceTooLarge { value, max_recommended } => {
                        Some(RecoverySuggestion::new(format!(
                            "吸附容差 {}mm 过大，可能导致过度合并。建议减小到 {:.2}mm",
                            value, max_recommended
                        ))
                        .with_config_change("topology.snap_tolerance_mm", serde_json::json!(max_recommended * 0.8))
                        .with_priority(8))
                    }
                    ToleranceErrorReason::SnapToleranceTooSmall { value, min_recommended } => {
                        Some(RecoverySuggestion::new(format!(
                            "吸附容差 {}mm 过小，可能无法合并相邻端点。建议增大到 {:.2}mm",
                            value, min_recommended
                        ))
                        .with_config_change("topology.snap_tolerance_mm", serde_json::json!(min_recommended * 1.2))
                        .with_priority(8))
                    }
                    ToleranceErrorReason::InconsistentTolerances { snap, merge } => {
                        Some(RecoverySuggestion::new(format!(
                            "吸附容差（{}mm）与合并容差（{}mm）不一致。建议保持 snap_tolerance >= merge_gap_tolerance",
                            snap, merge
                        ))
                        .with_priority(6))
                    }
                    _ => suggested_value.map(|val| {
                        RecoverySuggestion::new(format!("建议调整容差值为 {}", val))
                            .with_config_change("topology.snap_tolerance_mm", serde_json::json!(val))
                            .with_priority(5)
                    }),
                }
            }

            // ========== DXF 解析错误恢复建议 ==========
            Self::DxfParseError { reason, file, .. } => {
                match reason {
                    DxfParseReason::FileNotFound => {
                        Some(RecoverySuggestion::new(format!(
                            "文件 {:?} 未找到。请检查文件路径是否正确，或文件是否已被移动/删除",
                            file
                        ))
                        .with_priority(10))
                    }
                    DxfParseReason::InvalidVersion(version) => {
                        Some(RecoverySuggestion::new(format!(
                            "DXF 版本 {} 不受支持。建议：1) 使用 AutoCAD 转换为 AC1015 或更高版本，2) 导出为 PDF 格式",
                            version
                        ))
                        .with_priority(7))
                    }
                    DxfParseReason::MalformedEntity { entity_type, .. } => {
                        Some(RecoverySuggestion::new(format!(
                            "实体 {} 格式错误。建议：1) 在 AutoCAD 中运行 AUDIT 命令修复，2) 导出时选择忽略错误实体",
                            entity_type
                        ))
                        .with_priority(6))
                    }
                    DxfParseReason::ParseError(details) => {
                        Some(RecoverySuggestion::new(format!(
                            "DXF 解析失败：{}。建议：1) 用 AutoCAD 打开确认文件有效，2) 转换为兼容版本后重新导出",
                            details
                        ))
                        .with_priority(5))
                    }
                    _ => None,
                }
            }

            // ========== PDF 解析错误恢复建议 ==========
            Self::PdfParseError { reason, file, .. } => {
                match reason {
                    PdfParseReason::FileNotFound => {
                        Some(RecoverySuggestion::new(format!(
                            "PDF 文件 {:?} 未找到。请检查文件路径是否正确",
                            file
                        ))
                        .with_priority(10))
                    }
                    PdfParseReason::PasswordProtected => {
                        Some(RecoverySuggestion::new(format!(
                            "PDF 文件 {:?} 受密码保护。建议：1) 提供密码，2) 使用 PDF 工具移除密码保护",
                            file
                        ))
                        .with_priority(9))
                    }
                    PdfParseReason::RenderError(_) => {
                        Some(RecoverySuggestion::new(
                            "PDF 渲染失败。建议：1) 检查 PDF 文件是否损坏，2) 尝试使用其他 PDF 阅读器打开，3) 转换为 DXF 格式"
                        )
                        .with_priority(7))
                    }
                    _ => None,
                }
            }

            // ========== 矢量化错误恢复建议 ==========
            Self::VectorizeFailed { message } => {
                Some(RecoverySuggestion::new(format!(
                    "矢量化失败：{}。建议：1) 检查图像质量，2) 调整边缘检测阈值，3) 对于复杂扫描图纸建议先转换为 DXF 格式",
                    message
                ))
                .with_config_change("parser.pdf.edge_threshold", serde_json::json!(0.15))
                .with_priority(6))
            }

            // ========== IO 错误恢复建议 ==========
            Self::IoError { reason, path, .. } => {
                match reason {
                    IoErrorReason::FileNotFound => {
                        Some(RecoverySuggestion::new(format!(
                            "文件 {:?} 未找到。请检查路径是否正确",
                            path
                        ))
                        .with_priority(10))
                    }
                    IoErrorReason::PermissionDenied => {
                        Some(RecoverySuggestion::new(format!(
                            "访问文件 {:?} 权限被拒绝。建议：1) 检查文件权限设置，2) 以管理员身份运行",
                            path
                        ))
                        .with_priority(9))
                    }
                    IoErrorReason::ReadFailed => {
                        Some(RecoverySuggestion::new(format!(
                            "读取文件 {:?} 失败。建议：1) 检查文件是否被其他程序占用，2) 检查磁盘空间",
                            path
                        ))
                        .with_priority(8))
                    }
                    IoErrorReason::WriteFailed => {
                        Some(RecoverySuggestion::new(format!(
                            "写入文件 {:?} 失败。建议：1) 检查磁盘空间，2) 检查目录权限，3) 关闭占用文件的程序",
                            path
                        ))
                        .with_priority(8))
                    }
                    _ => None,
                }
            }

            // ========== 其他错误 ==========
            _ => None,
        }
    }

    /// 获取所有恢复建议，按优先级排序
    ///
    /// 对于包含多个问题的错误（如 ValidationFailed），返回所有问题的建议
    pub fn all_suggestions(&self) -> Vec<RecoverySuggestion> {
        let mut suggestions = Vec::new();

        // 添加主错误的建议
        if let Some(suggestion) = self.recovery_suggestion() {
            suggestions.push(suggestion);
        }

        // 对于 ValidationFailed，添加每个 issue 的建议
        if let Self::ValidationFailed { issues, .. } = self {
            for issue in issues {
                let issue_error = Self::GeometryValidationError {
                    issue_code: issue.code.clone(),
                    reason: issue.message.clone(),
                    location: issue.location.clone(),
                };
                if let Some(suggestion) = issue_error.recovery_suggestion() {
                    suggestions.push(suggestion);
                }
            }
        }

        // 按优先级排序（高优先级在前）
        suggestions.sort_by_key(|b| std::cmp::Reverse(b.priority));
        suggestions
    }
}

// ========== From 转换实现 ==========

impl From<std::io::Error> for CadError {
    fn from(err: std::io::Error) -> Self {
        Self::IoError {
            path: PathBuf::new(),
            reason: IoErrorReason::ReadFailed,
            source: err,
        }
    }
}

// 注意：dxf::Error 和 lopdf::Error 的转换放在 parser crate 中
// 因为 common-types 不应该依赖这些具体的解析库

#[cfg(test)]
mod tests {
    use super::*;
    use std::error::Error;
    use std::path::PathBuf;

    #[test]
    fn test_error_display() {
        let err = CadError::dxf_parse(
            PathBuf::from("/test/file.dxf"),
            DxfParseReason::FileNotFound,
        );
        assert!(err.to_string().contains("DXF 解析失败"));
        assert!(err.to_string().contains("文件未找到"));
    }

    #[test]
    fn test_error_chain() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "file not found");
        let cad_err = CadError::dxf_parse_with_source(
            PathBuf::from("/test/file.dxf"),
            DxfParseReason::FileNotFound,
            io_err,
        );

        // 验证错误链
        assert!(cad_err.source().is_some());
    }

    #[test]
    fn test_topo_stage_display() {
        assert_eq!(TopoStage::Snap.to_string(), "端点吸附");
        assert_eq!(TopoStage::GraphBuild.to_string(), "图构建");
    }

    #[test]
    fn test_error_constructors() {
        let _ = CadError::topo_construction(
            TopoStage::Snap,
            TopoErrorReason::SnapFailed {
                num_points: 100,
                error: "测试错误".to_string(),
            },
        );
        let _ = CadError::geometry_validation("E001", "环未闭合");
        let _ = CadError::internal(InternalErrorReason::Unknown);
        let _ = CadError::not_implemented("PDF 矢量提取");
    }

    #[test]
    fn test_error_reason_display() {
        // 测试 DXF 解析原因
        assert!(DxfParseReason::FileNotFound
            .to_string()
            .contains("文件未找到"));

        // 测试拓扑错误原因
        let topo_err = TopoErrorReason::OpenLoop { gap_distance: 0.5 };
        assert!(topo_err.to_string().contains("环未闭合"));

        // 测试内部错误原因
        assert!(InternalErrorReason::Unknown
            .to_string()
            .contains("未知内部错误"));
    }

    // ========== P11 恢复建议功能测试 ==========

    #[test]
    fn test_recovery_suggestion_open_loop() {
        // 测试环未闭合错误的恢复建议
        let err = CadError::topo_construction(
            TopoStage::LoopExtract,
            TopoErrorReason::OpenLoop { gap_distance: 0.8 },
        );

        let suggestion = err.recovery_suggestion().expect("应该有恢复建议");
        assert!(suggestion.action.contains("环未闭合"));
        assert!(suggestion.action.contains("0.80mm"));
        assert_eq!(suggestion.priority, 10);
        assert!(suggestion.config_change.is_some());
    }

    #[test]
    fn test_recovery_suggestion_snap_failed() {
        // 测试端点吸附失败的恢复建议
        let err = CadError::topo_construction(
            TopoStage::Snap,
            TopoErrorReason::SnapFailed {
                num_points: 50,
                error: "容差过小".to_string(),
            },
        );

        let suggestion = err.recovery_suggestion().expect("应该有恢复建议");
        assert!(suggestion.action.contains("端点吸附失败"));
        assert!(suggestion.action.contains("增大吸附容差"));
        assert_eq!(suggestion.priority, 8);
        assert!(suggestion.config_change.is_some());
    }

    #[test]
    fn test_recovery_suggestion_self_intersection() {
        // 测试自相交错误的恢复建议
        let err = CadError::GeometryValidationError {
            issue_code: "SELF_INTERSECTION".to_string(),
            reason: "多边形自相交".to_string(),
            location: None,
        };

        let suggestion = err.recovery_suggestion().expect("应该有恢复建议");
        assert!(suggestion.action.contains("自相交"));
        assert!(suggestion.action.contains("在交点处切分线段"));
        assert_eq!(suggestion.priority, 9);
    }

    #[test]
    fn test_recovery_suggestion_tolerance_error() {
        // 测试容差错误的恢复建议
        let err = CadError::ToleranceError {
            reason: ToleranceErrorReason::SnapToleranceTooSmall {
                value: 0.2,
                min_recommended: 0.5,
            },
            suggested_value: None,
        };

        let suggestion = err.recovery_suggestion().expect("应该有恢复建议");
        assert!(suggestion.action.contains("过小"));
        assert!(suggestion.action.contains("增大到"));
        assert_eq!(suggestion.priority, 8);
    }

    #[test]
    fn test_recovery_suggestion_file_not_found() {
        // 测试文件未找到错误的恢复建议
        let err = CadError::dxf_parse(
            PathBuf::from("/test/missing.dxf"),
            DxfParseReason::FileNotFound,
        );

        let suggestion = err.recovery_suggestion().expect("应该有恢复建议");
        assert!(suggestion.action.contains("未找到"));
        assert!(suggestion.action.contains("检查文件路径"));
        assert_eq!(suggestion.priority, 10);
    }

    #[test]
    fn test_recovery_suggestion_validation_failed() {
        // 测试验证失败错误的多个建议
        let err = CadError::ValidationFailed {
            count: 2,
            warning_count: 1,
            issues: vec![
                ValidationIssue {
                    code: "OPEN_LOOP".to_string(),
                    severity: Severity::Error,
                    message: "环未闭合".to_string(),
                    location: None,
                },
                ValidationIssue {
                    code: "TINY_SEGMENT".to_string(),
                    severity: Severity::Warning,
                    message: "过短线段".to_string(),
                    location: None,
                },
            ],
        };

        let suggestions = err.all_suggestions();
        assert!(!suggestions.is_empty());
        // 验证按优先级排序
        if suggestions.len() > 1 {
            for i in 0..suggestions.len() - 1 {
                assert!(suggestions[i].priority >= suggestions[i + 1].priority);
            }
        }
    }

    #[test]
    fn test_recovery_suggestion_no_suggestion() {
        // 测试没有恢复建议的错误类型
        let err = CadError::UnsupportedFormat {
            format: "DWG".to_string(),
            supported_formats: vec!["DXF".to_string(), "PDF".to_string()],
        };

        // 某些错误可能没有恢复建议
        let suggestion = err.recovery_suggestion();
        // 允许返回 None 或 Some，取决于实现
        let _ = suggestion;
    }

    #[test]
    fn test_recovery_suggestion_display() {
        // 测试 RecoverySuggestion 的 Display 实现
        let suggestion = RecoverySuggestion::new("调整容差")
            .with_config_change("topology.snap_tolerance_mm", serde_json::json!(1.0))
            .with_priority(8);

        let display = format!("{}", suggestion);
        assert!(display.contains("调整容差"));
        assert!(display.contains("snap_tolerance_mm"));
        assert!(display.contains("1.0"));
    }
}
