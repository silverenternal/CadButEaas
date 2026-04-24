#![allow(deprecated)]

//! 声学分析集成测试
//!
//! 测试场景：
//! 1. 从 DXF 解析 → 拓扑构建 → 声学分析全链路
//! 2. 大场景性能基准测试
//! 3. 真实 DXF 文件测试

use acoustic::{
    AcousticError, AcousticInput, AcousticRequest, AcousticResult, AcousticService,
    AcousticServiceConfig, ComparisonMetric, Frequency, NamedSelection, ReverberationFormula,
    SelectionBoundary, SelectionMode,
};
use common_types::scene::{BoundarySegment, BoundarySemantic, ClosedLoop, RawEdge, SceneState};

/// 创建测试场景（简单房间）
fn create_test_room_scene() -> SceneState {
    let mut scene = SceneState::default();

    // 添加墙边（10m x 8m 房间）
    let walls = vec![
        RawEdge {
            id: 0,
            start: [0.0, 0.0],
            end: [10000.0, 0.0],
            layer: Some("WALL".to_string()),
            color_index: None,
        },
        RawEdge {
            id: 1,
            start: [10000.0, 0.0],
            end: [10000.0, 8000.0],
            layer: Some("WALL".to_string()),
            color_index: None,
        },
        RawEdge {
            id: 2,
            start: [10000.0, 8000.0],
            end: [0.0, 8000.0],
            layer: Some("WALL".to_string()),
            color_index: None,
        },
        RawEdge {
            id: 3,
            start: [0.0, 8000.0],
            end: [0.0, 0.0],
            layer: Some("WALL".to_string()),
            color_index: None,
        },
    ];
    scene.edges.extend(walls);

    // 添加外轮廓
    scene.outer = Some(ClosedLoop::new(vec![
        [0.0, 0.0],
        [10000.0, 0.0],
        [10000.0, 8000.0],
        [0.0, 8000.0],
    ]));

    // 添加边界语义
    scene.boundaries.push(BoundarySegment {
        segment: [0, 1],
        semantic: BoundarySemantic::HardWall,
        material: Some("concrete".to_string()),
        width: None,
    });

    scene
}

/// 创建大场景（性能测试用）
fn create_large_scene(edge_count: usize) -> SceneState {
    let mut scene = SceneState::default();

    for i in 0..edge_count {
        let x = (i % 100) as f64 * 1000.0;
        let y = (i / 100) as f64 * 1000.0;
        scene.edges.push(RawEdge {
            id: i,
            start: [x, y],
            end: [x + 500.0, y + 500.0],
            layer: Some("WALL".to_string()),
            color_index: None,
        });
    }

    scene
}

// ============================================================================
// E2E 测试
// ============================================================================

#[test]
fn test_e2e_selection_material_stats() {
    let scene = create_test_room_scene();
    let service = AcousticService::new(AcousticServiceConfig::default());

    let input = AcousticInput {
        scene,
        request: AcousticRequest::SelectionMaterialStats {
            boundary: SelectionBoundary::rect([0.0, 0.0], [10000.0, 8000.0]),
            mode: SelectionMode::Smart,
        },
    };

    let result = service.process_sync(input).expect("计算失败");

    match result.result {
        AcousticResult::SelectionMaterialStats(stats) => {
            assert!(!stats.surface_ids.is_empty());
            assert!(stats.total_area > 0.0);
            assert!(!stats.material_distribution.is_empty());
        }
        _ => panic!("期望 SelectionMaterialStats 结果"),
    }

    println!(
        "E2E 选区材料统计测试通过，耗时：{:.2}ms",
        result.metrics.computation_time_ms
    );
}

#[test]
fn test_e2e_room_reverberation() {
    let scene = create_test_room_scene();
    let service = AcousticService::new(AcousticServiceConfig::default());

    let input = AcousticInput {
        scene,
        request: AcousticRequest::RoomReverberation {
            room_id: 0,
            formula: Some(ReverberationFormula::Sabine),
            room_height: Some(3.0),
        },
    };

    let result = service.process_sync(input).expect("计算失败");

    match result.result {
        AcousticResult::RoomReverberation(rev) => {
            // 房间体积：10m × 8m × 3m = 240 m³
            assert!((rev.volume - 240.0).abs() < 10.0);
            assert!(rev.total_surface_area > 0.0);

            // T60 应该在合理范围内
            for (freq, t60) in &rev.t60 {
                assert!(
                    *t60 > 0.1 && *t60 < 10.0,
                    "T60 at {:?} = {:.2}s out of range",
                    freq,
                    t60
                );
            }

            // EDT 应该约等于 T60 × 0.85
            for (freq, t60) in &rev.t60 {
                let edt = rev.edt.get(freq).unwrap();
                assert!((edt - t60 * 0.85).abs() < 0.1);
            }
        }
        _ => panic!("期望 RoomReverberation 结果"),
    }

    println!(
        "E2E 房间混响时间测试通过，耗时：{:.2}ms",
        result.metrics.computation_time_ms
    );
}

#[test]
fn test_e2e_comparative_analysis() {
    let scene = create_test_room_scene();
    let service = AcousticService::new(AcousticServiceConfig::default());

    let input = AcousticInput {
        scene,
        request: AcousticRequest::ComparativeAnalysis {
            selections: vec![
                NamedSelection {
                    name: "区域 A".to_string(),
                    boundary: SelectionBoundary::rect([0.0, 0.0], [5000.0, 4000.0]),
                },
                NamedSelection {
                    name: "区域 B".to_string(),
                    boundary: SelectionBoundary::rect([5000.0, 4000.0], [10000.0, 8000.0]),
                },
            ],
            metrics: vec![ComparisonMetric::Area, ComparisonMetric::AverageAbsorption],
        },
    };

    let result = service.process_sync(input).expect("计算失败");

    match result.result {
        AcousticResult::ComparativeAnalysis(comp) => {
            assert_eq!(comp.regions.len(), 2);

            // 验证区域名称
            assert!(comp.regions.iter().any(|r| r.name == "区域 A"));
            assert!(comp.regions.iter().any(|r| r.name == "区域 B"));
        }
        _ => panic!("期望 ComparativeAnalysis 结果"),
    }

    println!(
        "E2E 多区域对比测试通过，耗时：{:.2}ms",
        result.metrics.computation_time_ms
    );
}

// ============================================================================
// 性能测试
// ============================================================================

#[test]
fn test_performance_large_scene() {
    let scene = create_large_scene(10000);
    let service = AcousticService::new(AcousticServiceConfig::default());

    let input = AcousticInput {
        scene,
        request: AcousticRequest::SelectionMaterialStats {
            boundary: SelectionBoundary::rect([0.0, 0.0], [50000.0, 50000.0]),
            mode: SelectionMode::Smart,
        },
    };

    let start = std::time::Instant::now();
    let result = service.process_sync(input).expect("计算失败");
    let elapsed = start.elapsed();

    // 10000 条边的场景应该在 100ms 内完成
    assert!(
        elapsed.as_millis() < 100,
        "性能不达标：{:?} > 100ms",
        elapsed
    );

    println!(
        "性能测试通过：10000 条边，耗时：{:.2}ms",
        result.metrics.computation_time_ms
    );
}

// ============================================================================
// 边界情况测试
// ============================================================================

#[test]
fn test_boundary_empty_selection() {
    let scene = create_test_room_scene();
    let service = AcousticService::new(AcousticServiceConfig::default());

    let input = AcousticInput {
        scene,
        request: AcousticRequest::SelectionMaterialStats {
            boundary: SelectionBoundary::rect([100000.0, 100000.0], [110000.0, 110000.0]),
            mode: SelectionMode::Smart,
        },
    };

    let result = service.process_sync(input);
    assert!(result.is_err());

    // 验证错误类型和恢复建议
    match result.unwrap_err() {
        AcousticError::EmptySelection { suggestion } => {
            assert!(suggestion.is_some(), "EmptySelection 应该包含恢复建议");
        }
        _ => panic!("期望 EmptySelection 错误"),
    }
}

#[test]
fn test_boundary_invalid_room_id() {
    let scene = create_test_room_scene();
    let service = AcousticService::new(AcousticServiceConfig::default());

    let input = AcousticInput {
        scene,
        request: AcousticRequest::RoomReverberation {
            room_id: 999,
            formula: Some(ReverberationFormula::Sabine),
            room_height: Some(3.0),
        },
    };

    let result = service.process_sync(input);
    assert!(result.is_err());

    match result.unwrap_err() {
        AcousticError::InvalidRoomId {
            room_id,
            suggestion,
        } => {
            assert_eq!(room_id, 999);
            assert!(suggestion.is_some(), "InvalidRoomId 应该包含恢复建议");
        }
        _ => panic!("期望 InvalidRoomId 错误"),
    }
}

#[test]
fn test_boundary_sabine_vs_eyring() {
    let scene = create_test_room_scene();
    let service = AcousticService::new(AcousticServiceConfig::default());

    // Sabine 公式
    let input_sabine = AcousticInput {
        scene: scene.clone(),
        request: AcousticRequest::RoomReverberation {
            room_id: 0,
            formula: Some(ReverberationFormula::Sabine),
            room_height: Some(3.0),
        },
    };

    // Eyring 公式
    let input_eyring = AcousticInput {
        scene,
        request: AcousticRequest::RoomReverberation {
            room_id: 0,
            formula: Some(ReverberationFormula::Eyring),
            room_height: Some(3.0),
        },
    };

    let result_sabine = service.process_sync(input_sabine).expect("Sabine 计算失败");
    let result_eyring = service.process_sync(input_eyring).expect("Eyring 计算失败");

    match (&result_sabine.result, &result_eyring.result) {
        (AcousticResult::RoomReverberation(sabine), AcousticResult::RoomReverberation(eyring)) => {
            // 两种公式应该给出不同的结果
            let sabine_t60 = sabine.t60.get(&Frequency::Hz500).unwrap();
            let eyring_t60 = eyring.t60.get(&Frequency::Hz500).unwrap();

            // 在低吸声系数下，Eyring 通常给出更短的 T60
            assert!(
                (sabine_t60 - eyring_t60).abs() > 0.01,
                "Sabine 和 Eyring 应该给出不同的结果"
            );
        }
        _ => panic!("期望 RoomReverberation 结果"),
    }
}
