//! 调度策略模块

use accelerator_api::Accelerator;
use std::cmp::Ordering;

/// 调度策略
#[derive(Debug, Clone, Default)]
pub enum SchedulingStrategy {
    /// 性能优先：选择性能评分最高的加速器
    #[default]
    PerformanceFirst,
    /// 节能优先：选择功耗最低的加速器
    PowerSaving,
    /// 自定义策略
    Custom(SelectionCriteria),
}

/// 自定义选择条件
#[derive(Clone)]
pub struct SelectionCriteria {
    /// 比较函数：返回 Ordering::Greater 表示 a 优于 b
    pub compare_fn: fn(&dyn Accelerator, &dyn Accelerator) -> Ordering,
}

impl std::fmt::Debug for SelectionCriteria {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("SelectionCriteria")
            .field("compare_fn", &"<closure>")
            .finish()
    }
}

impl SelectionCriteria {
    /// 创建新的选择条件
    pub fn new(compare_fn: fn(&dyn Accelerator, &dyn Accelerator) -> Ordering) -> Self {
        Self { compare_fn }
    }

    /// 比较两个加速器
    pub fn compare(&self, a: &dyn Accelerator, b: &dyn Accelerator) -> Ordering {
        (self.compare_fn)(a, b)
    }

    /// 创建性能优先条件
    pub fn performance_priority() -> Self {
        Self::new(|a, b| {
            a.performance_score()
                .partial_cmp(&b.performance_score())
                .unwrap_or(Ordering::Equal)
        })
    }

    /// 创建内存优先条件（选择可用内存最多的）
    pub fn memory_priority() -> Self {
        Self::new(|a, b| {
            a.capabilities()
                .max_memory_mb
                .cmp(&b.capabilities().max_memory_mb)
        })
    }

    /// 创建兼容性优先条件（选择支持操作最多的）
    pub fn compatibility_priority() -> Self {
        Self::new(|a, b| {
            // 简单计算支持的操作数量
            let ops = [
                accelerator_api::AcceleratorOp::EdgeDetect,
                accelerator_api::AcceleratorOp::ContourExtract,
                accelerator_api::AcceleratorOp::ArcFit,
                accelerator_api::AcceleratorOp::RTreeBuild,
                accelerator_api::AcceleratorOp::EndpointSnap,
                accelerator_api::AcceleratorOp::IntersectionDetect,
                accelerator_api::AcceleratorOp::PolygonSimplify,
                accelerator_api::AcceleratorOp::Threshold,
                accelerator_api::AcceleratorOp::Skeletonize,
            ];

            let a_count = ops.iter().filter(|op| a.supports_op(**op)).count();
            let b_count = ops.iter().filter(|op| b.supports_op(**op)).count();

            a_count.cmp(&b_count)
        })
    }
}
