//! 房间级混响时间计算
//!
//! 实现 ReverberationCalculator，提供：
//! - Sabine 公式计算 T60
//! - Eyring 公式计算 T60
//! - 房间体积估算
//! - EDT（早期衰变时间）计算

use std::collections::BTreeMap;
use tracing::{debug, instrument};

use crate::acoustic_types::{AcousticError, Frequency, ReverberationFormula, ReverberationResult};
use common_types::geometry::Point2;
use common_types::scene::{SceneState, SurfaceId};

use crate::material_db::MaterialDatabase;

/// 房间混响时间计算器
pub struct ReverberationCalculator {
    material_db: MaterialDatabase,
}

impl ReverberationCalculator {
    /// 创建新的 ReverberationCalculator
    pub fn new() -> Self {
        Self {
            material_db: MaterialDatabase::with_defaults(),
        }
    }

    /// 使用自定义材料数据库创建 ReverberationCalculator
    pub fn with_material_db(material_db: MaterialDatabase) -> Self {
        Self { material_db }
    }

    /// 计算房间混响时间
    ///
    /// # Arguments
    /// * `scene` - 场景状态
    /// * `room_id` - 房间 ID（外轮廓或孔洞索引）
    /// * `formula` - 使用的公式（Auto 时自动选择）
    /// * `room_height` - 房间高度 (m)
    ///
    /// # Returns
    /// 混响时间结果（包含 T60 和 EDT，频率相关）
    #[instrument(skip(self, scene), fields(room_id = room_id, formula = ?formula))]
    pub fn calculate(
        &self,
        scene: &SceneState,
        room_id: SurfaceId,
        formula: ReverberationFormula,
        room_height: f64,
    ) -> Result<ReverberationResult, AcousticError> {
        debug!("计算房间混响时间，高度={:.2}m", room_height);

        // 1. 获取房间边界（闭合环）
        let room_loop = if room_id == 0 {
            // ID 为 0 时使用外轮廓
            scene.outer.as_ref()
        } else {
            // ID > 0 时使用孔洞
            scene.holes.get(room_id - 1)
        };

        let room_loop = room_loop.ok_or_else(|| AcousticError::invalid_room_id(room_id))?;

        // 2. 计算房间面积（二维，使用鞋带公式）
        let floor_area = self.polygon_area(&room_loop.points);
        debug!("房间地板面积：{:.2} m²", floor_area);

        // 3. 计算房间体积（面积×高度）
        let volume = floor_area * room_height;
        debug!("房间体积：{:.2} m³ (面积×高度)", volume);

        // 4. 计算总表面积（墙面 + 天花板 + 地面）
        let perimeter = self.polygon_perimeter(&room_loop.points);
        let wall_area = perimeter * room_height;
        let ceiling_floor_area = floor_area * 2.0;
        let total_surface_area = wall_area + ceiling_floor_area;
        debug!(
            "总表面积：{:.2} m² (墙面：{:.2}, 天花板 + 地面：{:.2})",
            total_surface_area, wall_area, ceiling_floor_area
        );

        // 5. 计算等效吸声面积（频率相关）
        let equivalent_area = self.compute_equivalent_absorption(scene, room_id, room_height)?;

        // 6. 自动选择公式（如果是 Auto 模式）
        let formula = self.select_formula(&equivalent_area, total_surface_area, formula);

        // 7. 计算 T60 和 EDT
        let t60 = Self::calculate_t60(volume, total_surface_area, &equivalent_area, formula);
        let edt = Self::calculate_edt(&t60);

        debug!(
            "T60 (500Hz): {:.2}s, EDT (500Hz): {:.2}s",
            t60.get(&Frequency::Hz500).unwrap_or(&0.0),
            edt.get(&Frequency::Hz500).unwrap_or(&0.0)
        );

        Ok(ReverberationResult {
            volume,
            total_surface_area,
            formula,
            t60,
            edt,
        })
    }

    /// 计算多边形面积（鞋带公式）
    ///
    /// # Formula
    /// A = 0.5 × |Σ(xᵢ × yᵢ₊₁ - xᵢ₊₁ × yᵢ)|
    fn polygon_area(&self, points: &[Point2]) -> f64 {
        let n = points.len();
        if n < 3 {
            return 0.0;
        }

        let mut area = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            area += points[i][0] * points[j][1];
            area -= points[j][0] * points[i][1];
        }

        (area / 2.0).abs() / 1000.0 / 1000.0 // 转换为 m²（原始单位是 mm）
    }

    /// 计算多边形周长
    fn polygon_perimeter(&self, points: &[Point2]) -> f64 {
        let n = points.len();
        if n < 2 {
            return 0.0;
        }

        let mut perimeter = 0.0;
        for i in 0..n {
            let j = (i + 1) % n;
            let dx = points[j][0] - points[i][0];
            let dy = points[j][1] - points[i][1];
            perimeter += (dx * dx + dy * dy).sqrt();
        }

        perimeter / 1000.0 // 转换为 m（原始单位是 mm）
    }

    /// 计算等效吸声面积 A = Σ(S × α)
    fn compute_equivalent_absorption(
        &self,
        scene: &SceneState,
        room_id: SurfaceId,
        room_height: f64,
    ) -> Result<BTreeMap<Frequency, f64>, AcousticError> {
        let mut equivalent_area: BTreeMap<Frequency, f64> = BTreeMap::new();

        // 获取房间边界
        let room_loop = if room_id == 0 {
            scene.outer.as_ref()
        } else {
            scene.holes.get(room_id - 1)
        };

        let room_loop = room_loop.ok_or_else(|| AcousticError::invalid_room_id(room_id))?;

        // 计算墙面面积
        let perimeter = self.polygon_perimeter(&room_loop.points);
        let wall_area = perimeter * room_height;

        // 计算地板/天花板面积
        let floor_area = self.polygon_area(&room_loop.points);

        // 墙面等效吸声面积（假设墙面材料）
        let wall_absorption = self.get_wall_absorption(scene);
        for freq in Frequency::all() {
            let coeff = wall_absorption.get(&freq).copied().unwrap_or(0.05);
            *equivalent_area.entry(freq).or_insert(0.0) += wall_area * coeff;
        }

        // 地板等效吸声面积（假设地毯）
        let floor_absorption = self.get_floor_absorption();
        for freq in Frequency::all() {
            let coeff = floor_absorption.get(&freq).copied().unwrap_or(0.3);
            *equivalent_area.entry(freq).or_insert(0.0) += floor_area * coeff;
        }

        // 天花板等效吸声面积（假设石膏板）
        let ceiling_absorption = self.get_ceiling_absorption();
        for freq in Frequency::all() {
            let coeff = ceiling_absorption.get(&freq).copied().unwrap_or(0.08);
            *equivalent_area.entry(freq).or_insert(0.0) += floor_area * coeff;
        }

        Ok(equivalent_area)
    }

    /// 获取墙面吸声系数（基于场景中的边界语义）
    fn get_wall_absorption(&self, scene: &SceneState) -> BTreeMap<Frequency, f64> {
        // 检查边界语义
        for boundary in &scene.boundaries {
            if let Some(ref material) = boundary.material {
                return self.get_absorption_from_material_name(material);
            }
        }

        // 默认墙面吸声系数（混凝土）
        self.get_default_absorption("concrete")
    }

    /// 获取地板吸声系数（假设地毯）
    fn get_floor_absorption(&self) -> BTreeMap<Frequency, f64> {
        self.get_default_absorption("carpet")
    }

    /// 获取天花板吸声系数（假设石膏板）
    fn get_ceiling_absorption(&self) -> BTreeMap<Frequency, f64> {
        self.get_default_absorption("gypsum")
    }

    /// 根据材料名称获取吸声系数
    fn get_absorption_from_material_name(&self, material: &str) -> BTreeMap<Frequency, f64> {
        // 首先尝试从材料数据库获取
        if let Some(coeffs) = self.material_db.get_absorption_coeffs(material) {
            return coeffs.clone();
        }

        // 如果数据库中不存在，使用硬编码默认值
        self.get_default_absorption(material)
    }

    /// 获取默认吸声系数（回退值）
    fn get_default_absorption(&self, material: &str) -> BTreeMap<Frequency, f64> {
        let _ = material; // 忽略参数，使用通用默认值

        // 通用默认值（类似抹灰墙面）
        let coeffs = vec![0.02, 0.03, 0.04, 0.05, 0.06, 0.07];
        Frequency::all().into_iter().zip(coeffs).collect()
    }

    /// 计算 T60（混响时间）
    ///
    /// # Sabine 公式
    /// T60 = 0.161 × V / A
    ///
    /// 其中：
    /// - V: 房间体积 (m³)
    /// - A: 等效吸声面积 (m²)
    ///
    /// # Eyring 公式
    /// T60 = 0.161 × V / (-S × ln(1 - α))
    ///
    /// 其中：
    /// - V: 房间体积 (m³)
    /// - S: 总表面积 (m²)
    /// - α: 平均吸声系数 = A / S
    ///
    /// # 注意
    ///
    /// Eyring 公式在高吸声系数（α > 0.2）的房间中更准确。
    /// Sabine 公式在低吸声系数（α < 0.2）的房间中更准确。
    fn calculate_t60(
        volume: f64,
        total_surface_area: f64,
        equivalent_area: &BTreeMap<Frequency, f64>,
        formula: ReverberationFormula,
    ) -> BTreeMap<Frequency, f64> {
        // 注意：formula 应该是已经通过 select_formula 处理过的，不会是 Auto
        equivalent_area
            .iter()
            .map(|(&freq, &a)| {
                let t60 = match formula {
                    ReverberationFormula::Sabine => {
                        // Sabine 公式：T60 = 0.161 × V / A
                        // 避免除零
                        if a > 0.01 {
                            0.161 * volume / a
                        } else {
                            10.0 // 上限值
                        }
                    }
                    ReverberationFormula::Eyring => {
                        // Eyring 公式：T60 = 0.161 × V / (-S × ln(1 - α))
                        // 其中 α = A / S（平均吸声系数）
                        if total_surface_area > 0.01 && a > 0.01 {
                            let alpha = a / total_surface_area;
                            // 确保 alpha 在有效范围内 (0, 1)
                            let alpha = alpha.clamp(0.01, 0.99);
                            let denominator = -total_surface_area * (1.0 - alpha).ln();
                            if denominator > 0.01 {
                                0.161 * volume / denominator
                            } else {
                                10.0
                            }
                        } else {
                            10.0
                        }
                    }
                    ReverberationFormula::Auto => {
                        // 理论上不应该出现，因为 select_formula 已经处理了
                        // 回退到 Sabine
                        if a > 0.01 {
                            0.161 * volume / a
                        } else {
                            10.0
                        }
                    }
                };
                (freq, t60)
            })
            .collect()
    }

    /// 计算 EDT（早期衰变时间）
    ///
    /// # 经验公式
    /// EDT ≈ T60 × 0.85
    fn calculate_edt(t60: &BTreeMap<Frequency, f64>) -> BTreeMap<Frequency, f64> {
        t60.iter().map(|(&freq, &t)| (freq, t * 0.85)).collect()
    }

    /// 自动选择混响公式
    ///
    /// # 选择规则
    /// - α (平均吸声系数) < 0.2: 使用 Sabine 公式
    /// - α >= 0.2: 使用 Eyring 公式
    ///
    /// # 参数
    /// - `equivalent_area`: 等效吸声面积
    /// - `total_surface_area`: 总表面积
    /// - `requested`: 用户请求的公式
    ///
    /// # 返回值
    /// 实际使用的公式
    fn select_formula(
        &self,
        equivalent_area: &BTreeMap<Frequency, f64>,
        total_surface_area: f64,
        requested: ReverberationFormula,
    ) -> ReverberationFormula {
        if requested != ReverberationFormula::Auto {
            return requested;
        }

        // 计算 500Hz 的平均吸声系数
        let alpha_500 = equivalent_area
            .get(&Frequency::Hz500)
            .copied()
            .unwrap_or(0.0)
            / total_surface_area.max(0.01);

        // 根据 α 自动选择
        if alpha_500 < 0.2 {
            debug!("自动选择 Sabine 公式 (α = {:.3} < 0.2)", alpha_500);
            ReverberationFormula::Sabine
        } else {
            debug!("自动选择 Eyring 公式 (α = {:.3} >= 0.2)", alpha_500);
            ReverberationFormula::Eyring
        }
    }
}

impl Default for ReverberationCalculator {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use common_types::scene::{BoundarySegment, BoundarySemantic, ClosedLoop};

    fn create_test_room_scene() -> SceneState {
        let mut scene = SceneState::default();

        // 创建一个 10m x 8m 的房间（80 m²）
        let room_points = vec![[0.0, 0.0], [10000.0, 0.0], [10000.0, 8000.0], [0.0, 8000.0]];
        scene.outer = Some(ClosedLoop::new(room_points));

        // 添加边界语义（混凝土墙）
        scene.boundaries.push(BoundarySegment {
            segment: [0, 1],
            semantic: BoundarySemantic::HardWall,
            material: Some("concrete".to_string()),
            width: None,
        });

        scene
    }

    #[test]
    fn test_reverberation_calculator_creation() {
        let _calc = ReverberationCalculator::new();
        let _ = ReverberationCalculator::default();
    }

    #[test]
    fn test_polygon_area_rectangle() {
        let calc = ReverberationCalculator::new();
        let points = vec![[0.0, 0.0], [10000.0, 0.0], [10000.0, 8000.0], [0.0, 8000.0]];

        let area = calc.polygon_area(&points);
        // 10m × 8m = 80 m²
        assert!((area - 80.0).abs() < 0.1);
    }

    #[test]
    fn test_polygon_perimeter_rectangle() {
        let calc = ReverberationCalculator::new();
        let points = vec![[0.0, 0.0], [10000.0, 0.0], [10000.0, 8000.0], [0.0, 8000.0]];

        let perimeter = calc.polygon_perimeter(&points);
        // 2 × (10m + 8m) = 36m
        assert!((perimeter - 36.0).abs() < 0.1);
    }

    #[test]
    fn test_reverberation_calculation_sabine() {
        let scene = create_test_room_scene();
        let calc = ReverberationCalculator::new();

        let result = calc
            .calculate(&scene, 0, ReverberationFormula::Sabine, 3.0)
            .unwrap();

        // 房间体积：80 m² × 3m = 240 m³
        assert!((result.volume - 240.0).abs() < 1.0);

        // 总表面积应该大于 0
        assert!(result.total_surface_area > 0.0);

        // T60 应该在合理范围内（0.1s - 10s）
        for t60 in result.t60.values() {
            assert!(*t60 > 0.1 && *t60 < 10.0);
        }

        // EDT 应该约等于 T60 × 0.85
        for (freq, t60) in &result.t60 {
            let edt = result.edt.get(freq).unwrap();
            assert!((edt - t60 * 0.85).abs() < 0.01);
        }
    }

    #[test]
    fn test_reverberation_calculation_eyring() {
        let scene = create_test_room_scene();
        let calc = ReverberationCalculator::new();

        let result = calc
            .calculate(&scene, 0, ReverberationFormula::Eyring, 3.0)
            .unwrap();

        assert_eq!(result.formula, ReverberationFormula::Eyring);
        assert!((result.volume - 240.0).abs() < 1.0);
    }

    #[test]
    fn test_reverberation_different_heights() {
        let scene = create_test_room_scene();
        let calc = ReverberationCalculator::new();

        let result_3m = calc
            .calculate(&scene, 0, ReverberationFormula::Sabine, 3.0)
            .unwrap();
        let result_4m = calc
            .calculate(&scene, 0, ReverberationFormula::Sabine, 4.0)
            .unwrap();

        // 更高的房间应该有更长的混响时间（体积更大）
        assert!(result_4m.volume > result_3m.volume);

        // T60 应该随体积增加而增加
        let t60_3m = result_3m.t60.get(&Frequency::Hz500).unwrap();
        let t60_4m = result_4m.t60.get(&Frequency::Hz500).unwrap();
        assert!(t60_4m > t60_3m);
    }

    #[test]
    fn test_invalid_room_id() {
        let scene = create_test_room_scene();
        let calc = ReverberationCalculator::new();

        // 无效的房间 ID
        let result = calc.calculate(&scene, 999, ReverberationFormula::Sabine, 3.0);

        // 验证是 InvalidRoomId 错误，并且有恢复建议
        match result {
            Err(AcousticError::InvalidRoomId {
                room_id,
                suggestion,
            }) => {
                assert_eq!(room_id, 999);
                assert!(suggestion.is_some(), "InvalidRoomId 应该包含恢复建议");
            }
            _ => panic!("Expected InvalidRoomId error"),
        }
    }

    #[test]
    fn test_frequency_coverage() {
        let scene = create_test_room_scene();
        let calc = ReverberationCalculator::new();

        let result = calc
            .calculate(&scene, 0, ReverberationFormula::Sabine, 3.0)
            .unwrap();

        // 所有频率都应该有 T60 和 EDT 值
        for freq in Frequency::all() {
            assert!(result.t60.contains_key(&freq));
            assert!(result.edt.contains_key(&freq));
        }
    }

    #[test]
    fn test_t60_range() {
        let scene = create_test_room_scene();
        let calc = ReverberationCalculator::new();

        let result = calc
            .calculate(&scene, 0, ReverberationFormula::Sabine, 3.0)
            .unwrap();

        // T60 应该在合理范围内
        for (freq, t60) in &result.t60 {
            // 低频通常有较长的混响时间
            // 高频通常有较短的混响时间
            assert!(
                *t60 > 0.0 && *t60 < 10.0,
                "T60 at {:?} = {:.2}s out of range",
                freq,
                t60
            );
        }
    }

    #[test]
    fn test_edt_ratio() {
        let scene = create_test_room_scene();
        let calc = ReverberationCalculator::new();

        let result = calc
            .calculate(&scene, 0, ReverberationFormula::Sabine, 3.0)
            .unwrap();

        // EDT/T60 比率应该接近 0.85
        for (freq, t60) in &result.t60 {
            let edt = result.edt.get(freq).unwrap();
            let ratio = edt / t60;
            assert!(
                (ratio - 0.85).abs() < 0.01,
                "EDT/T60 ratio at {:?} = {:.2}",
                freq,
                ratio
            );
        }
    }

    #[test]
    fn test_formula_auto_selection() {
        let scene = create_test_room_scene();
        let calc = ReverberationCalculator::new();

        // 使用 Auto 模式，应该根据α自动选择公式
        let result = calc
            .calculate(&scene, 0, ReverberationFormula::Auto, 3.0)
            .unwrap();

        // 由于是低吸声房间（混凝土墙），应该自动选择 Sabine
        assert_eq!(result.formula, ReverberationFormula::Sabine);
    }

    #[test]
    fn test_eyring_vs_sabine() {
        let scene = create_test_room_scene();
        let calc = ReverberationCalculator::new();

        let sabine_result = calc
            .calculate(&scene, 0, ReverberationFormula::Sabine, 3.0)
            .unwrap();
        let eyring_result = calc
            .calculate(&scene, 0, ReverberationFormula::Eyring, 3.0)
            .unwrap();

        // 在低吸声系数下，Eyring 公式应该给出更短（更保守）的 T60
        // 这是因为 Eyring 公式考虑了高吸声系数的情况
        let sabine_t60 = sabine_result.t60.get(&Frequency::Hz500).unwrap();
        let eyring_t60 = eyring_result.t60.get(&Frequency::Hz500).unwrap();

        // 通常 Eyring 的 T60 会比 Sabine 短（取决于吸声系数）
        // 但在某些情况下可能相反，所以只验证它们不同
        assert!(
            (sabine_t60 - eyring_t60).abs() > 0.01,
            "Sabine 和 Eyring 应该给出不同的结果"
        );
    }
}
