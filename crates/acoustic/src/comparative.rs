//! 多区域对比分析
//!
//! 实现 ComparativeAnalyzer，提供：
//! - 多区域材料统计对比
//! - 吸声性能对比
//! - 差异分析

use tracing::{debug, instrument};

use crate::acoustic_types::{
    AcousticError, ComparativeAnalysisResult, ComparisonMetric, Frequency, NamedSelection,
    RegionStats, SelectionBoundary, SelectionMode,
};
use common_types::scene::SceneState;

use crate::selection::SelectionCalculator;

/// 多区域对比分析器
pub struct ComparativeAnalyzer {
    selection_calculator: SelectionCalculator,
}

impl ComparativeAnalyzer {
    /// 创建新的 ComparativeAnalyzer
    pub fn new() -> Self {
        Self {
            selection_calculator: SelectionCalculator::new(),
        }
    }

    /// 执行多区域对比分析
    ///
    /// # 性能优化
    ///
    /// 对于多区域对比，先构建 R*-tree 缓存，然后所有区域共享：
    /// - 未优化：O(k × n log n)，k=区域数
    /// - 优化后：O(n log n + k log n)，性能提升 k 倍
    ///
    /// # Arguments
    /// * `scene` - 场景状态
    /// * `selections` - 多个选区
    /// * `metrics` - 对比指标
    ///
    /// # Returns
    /// 对比分析结果
    #[instrument(skip(self, scene), fields(regions = selections.len()))]
    pub fn analyze(
        &mut self,
        scene: &SceneState,
        selections: Vec<NamedSelection>,
        metrics: Vec<ComparisonMetric>,
    ) -> Result<ComparativeAnalysisResult, AcousticError> {
        debug!("执行多区域对比分析，{} 个区域", selections.len());

        if selections.is_empty() {
            return Err(AcousticError::selection("至少需要一个选区"));
        }

        // 对于大场景，先构建 R*-tree 缓存
        if scene.edges.len() >= 100 {
            debug!("为多区域对比构建 R*-tree 缓存，边数：{}", scene.edges.len());
            self.selection_calculator.build_rtree_cache(scene);
        }

        let mut regions = Vec::with_capacity(selections.len());

        for selection in &selections {
            // 计算每个区域的统计
            let mut stats = self.calculate_region_stats(scene, &selection.boundary)?;
            // 设置区域名称
            stats.name = selection.name.clone();
            regions.push(stats);
        }

        // 清除缓存，释放内存
        self.selection_calculator.clear_rtree_cache();

        debug!("完成 {} 个区域的统计", regions.len());

        Ok(ComparativeAnalysisResult { regions })
    }

    /// 计算区域统计
    fn calculate_region_stats(
        &self,
        scene: &SceneState,
        boundary: &SelectionBoundary,
    ) -> Result<RegionStats, AcousticError> {
        // 使用 SelectionCalculator 计算材料统计
        let material_stats =
            self.selection_calculator
                .calculate(scene, boundary.clone(), SelectionMode::Smart)?;

        // 转换为 RegionStats
        Ok(RegionStats {
            name: String::new(), // 名称由调用者设置
            area: material_stats.total_area,
            material_count: material_stats.material_distribution.len(),
            average_absorption: material_stats.average_absorption_coefficient,
            equivalent_absorption_area: material_stats.equivalent_absorption_area,
        })
    }

    /// 计算区域差异
    ///
    /// # Returns
    /// 返回两个区域在各指标上的差异
    pub fn compare_regions(
        &self,
        region1: &RegionStats,
        region2: &RegionStats,
        metric: ComparisonMetric,
    ) -> RegionComparison {
        match metric {
            ComparisonMetric::Area => {
                let diff = region1.area - region2.area;
                let ratio = if region2.area > 0.0 {
                    region1.area / region2.area
                } else {
                    f64::INFINITY
                };
                RegionComparison {
                    metric: "Area".to_string(),
                    value1: region1.area,
                    value2: region2.area,
                    diff,
                    ratio,
                    unit: "m²".to_string(),
                }
            }
            ComparisonMetric::MaterialCount => {
                let diff = region1.material_count as f64 - region2.material_count as f64;
                let ratio = if region2.material_count > 0 {
                    region1.material_count as f64 / region2.material_count as f64
                } else {
                    f64::INFINITY
                };
                RegionComparison {
                    metric: "MaterialCount".to_string(),
                    value1: region1.material_count as f64,
                    value2: region2.material_count as f64,
                    diff,
                    ratio,
                    unit: "count".to_string(),
                }
            }
            ComparisonMetric::AverageAbsorption => {
                // 比较 500Hz 的平均吸声系数
                let val1 = region1
                    .average_absorption
                    .get(&Frequency::Hz500)
                    .copied()
                    .unwrap_or(0.0);
                let val2 = region2
                    .average_absorption
                    .get(&Frequency::Hz500)
                    .copied()
                    .unwrap_or(0.0);
                RegionComparison {
                    metric: "AverageAbsorption@500Hz".to_string(),
                    value1: val1,
                    value2: val2,
                    diff: val1 - val2,
                    ratio: if val2 > 0.0 {
                        val1 / val2
                    } else {
                        f64::INFINITY
                    },
                    unit: "coefficient".to_string(),
                }
            }
            ComparisonMetric::EquivalentAbsorptionArea => {
                // 比较 500Hz 的等效吸声面积
                let val1 = region1
                    .equivalent_absorption_area
                    .get(&Frequency::Hz500)
                    .copied()
                    .unwrap_or(0.0);
                let val2 = region2
                    .equivalent_absorption_area
                    .get(&Frequency::Hz500)
                    .copied()
                    .unwrap_or(0.0);
                RegionComparison {
                    metric: "EquivalentAbsorptionArea@500Hz".to_string(),
                    value1: val1,
                    value2: val2,
                    diff: val1 - val2,
                    ratio: if val2 > 0.0 {
                        val1 / val2
                    } else {
                        f64::INFINITY
                    },
                    unit: "m²".to_string(),
                }
            }
        }
    }
}

impl Default for ComparativeAnalyzer {
    fn default() -> Self {
        Self::new()
    }
}

/// 区域对比结果
#[derive(Debug, Clone)]
pub struct RegionComparison {
    /// 对比指标名称
    pub metric: String,
    /// 区域 1 的值
    pub value1: f64,
    /// 区域 2 的值
    pub value2: f64,
    /// 差值（区域 1 - 区域 2）
    pub diff: f64,
    /// 比率（区域 1 / 区域 2）
    pub ratio: f64,
    /// 单位
    pub unit: String,
}

impl RegionComparison {
    /// 格式化对比结果
    pub fn format(&self) -> String {
        format!(
            "{}: {:.2} vs {:.2} (diff: {:+.2}, ratio: {:.2}x) [{}]",
            self.metric, self.value1, self.value2, self.diff, self.ratio, self.unit
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common_types::scene::RawEdge;
    use std::collections::BTreeMap;

    fn create_test_scene() -> SceneState {
        let mut scene = SceneState::default();

        // 创建两个区域的材料
        for i in 0..10 {
            scene.edges.push(RawEdge {
                id: i,
                start: [i as f64 * 1000.0, 0.0],
                end: [(i + 1) as f64 * 1000.0, 0.0],
                layer: Some(if i < 5 {
                    "concrete".to_string()
                } else {
                    "glass".to_string()
                }),
                color_index: None,
            });
        }

        scene
    }

    #[test]
    fn test_comparative_analyzer_creation() {
        let _analyzer = ComparativeAnalyzer::new();
        let _ = ComparativeAnalyzer::default();
    }

    #[test]
    fn test_analyze_single_region() {
        let scene = create_test_scene();
        let mut analyzer = ComparativeAnalyzer::new();

        let selections = vec![NamedSelection {
            name: "Region 1".to_string(),
            boundary: SelectionBoundary::rect([0.0, 0.0], [5000.0, 1000.0]),
        }];

        let result = analyzer.analyze(&scene, selections, vec![]).unwrap();
        assert_eq!(result.regions.len(), 1);
    }

    #[test]
    fn test_analyze_multiple_regions() {
        let scene = create_test_scene();
        let mut analyzer = ComparativeAnalyzer::new();

        let selections = vec![
            NamedSelection {
                name: "Region 1".to_string(),
                boundary: SelectionBoundary::rect([0.0, 0.0], [3000.0, 1000.0]),
            },
            NamedSelection {
                name: "Region 2".to_string(),
                boundary: SelectionBoundary::rect([5000.0, 0.0], [8000.0, 1000.0]),
            },
        ];

        let result = analyzer.analyze(&scene, selections, vec![]).unwrap();
        assert_eq!(result.regions.len(), 2);
    }

    #[test]
    fn test_empty_selections() {
        let scene = create_test_scene();
        let mut analyzer = ComparativeAnalyzer::new();

        let result = analyzer.analyze(&scene, vec![], vec![]);
        assert!(result.is_err());
    }

    #[test]
    fn test_compare_regions_area() {
        let analyzer = ComparativeAnalyzer::new();

        let region1 = RegionStats {
            name: "Region 1".to_string(),
            area: 100.0,
            material_count: 3,
            average_absorption: BTreeMap::new(),
            equivalent_absorption_area: BTreeMap::new(),
        };

        let region2 = RegionStats {
            name: "Region 2".to_string(),
            area: 50.0,
            material_count: 2,
            average_absorption: BTreeMap::new(),
            equivalent_absorption_area: BTreeMap::new(),
        };

        let comparison = analyzer.compare_regions(&region1, &region2, ComparisonMetric::Area);
        assert_eq!(comparison.metric, "Area");
        assert!((comparison.diff - 50.0).abs() < 0.01);
        assert!((comparison.ratio - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_compare_regions_material_count() {
        let analyzer = ComparativeAnalyzer::new();

        let region1 = RegionStats {
            name: "Region 1".to_string(),
            area: 100.0,
            material_count: 6,
            average_absorption: BTreeMap::new(),
            equivalent_absorption_area: BTreeMap::new(),
        };

        let region2 = RegionStats {
            name: "Region 2".to_string(),
            area: 50.0,
            material_count: 3,
            average_absorption: BTreeMap::new(),
            equivalent_absorption_area: BTreeMap::new(),
        };

        let comparison =
            analyzer.compare_regions(&region1, &region2, ComparisonMetric::MaterialCount);
        assert_eq!(comparison.metric, "MaterialCount");
        assert!((comparison.diff - 3.0).abs() < 0.01);
        assert!((comparison.ratio - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_compare_regions_absorption() {
        let analyzer = ComparativeAnalyzer::new();

        let mut avg1 = BTreeMap::new();
        avg1.insert(Frequency::Hz500, 0.6);

        let mut avg2 = BTreeMap::new();
        avg2.insert(Frequency::Hz500, 0.3);

        let region1 = RegionStats {
            name: "Region 1".to_string(),
            area: 100.0,
            material_count: 3,
            average_absorption: avg1,
            equivalent_absorption_area: BTreeMap::new(),
        };

        let region2 = RegionStats {
            name: "Region 2".to_string(),
            area: 50.0,
            material_count: 2,
            average_absorption: avg2,
            equivalent_absorption_area: BTreeMap::new(),
        };

        let comparison =
            analyzer.compare_regions(&region1, &region2, ComparisonMetric::AverageAbsorption);
        assert!(comparison.metric.contains("AverageAbsorption"));
        assert!((comparison.diff - 0.3).abs() < 0.01);
        assert!((comparison.ratio - 2.0).abs() < 0.01);
    }

    #[test]
    fn test_region_comparison_format() {
        let comparison = RegionComparison {
            metric: "Area".to_string(),
            value1: 100.0,
            value2: 50.0,
            diff: 50.0,
            ratio: 2.0,
            unit: "m²".to_string(),
        };

        let formatted = comparison.format();
        assert!(formatted.contains("100.00"));
        assert!(formatted.contains("50.00"));
        assert!(formatted.contains("+50.00"));
        assert!(formatted.contains("2.00"));
    }
}
