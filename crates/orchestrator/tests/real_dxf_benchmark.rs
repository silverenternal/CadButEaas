//! 真实 DXF 场景性能基准测试
//!
//! 使用真实建筑图纸数据验证 topo 性能，而非合成网格数据。

use common_types::{Point2, Polyline, RawEntity};
use std::path::PathBuf;
use std::time::Instant;
use topo::service::{TopoConfig, TopoService};
use topo::GraphBuilder;

/// 获取 DXF 目录
fn get_dxfs_dir() -> PathBuf {
    let manifest = env!("CARGO_MANIFEST_DIR");
    PathBuf::from(manifest)
        .parent()
        .and_then(|p| p.parent())
        .expect("Invalid manifest dir")
        .join("dxfs")
}

/// 从 DXF 文件提取多段线（复用 pipeline 逻辑）
fn discretize_arc(center: Point2, radius: f64, start_angle: f64, end_angle: f64) -> Polyline {
    let mut points = Vec::new();
    let sweep = if end_angle > start_angle {
        end_angle - start_angle
    } else {
        2.0 * std::f64::consts::PI + end_angle - start_angle
    };
    let segments = (sweep * 180.0 / std::f64::consts::PI * 2.0).max(4.0) as usize;
    for i in 0..=segments {
        let angle = start_angle + sweep * i as f64 / segments as f64;
        points.push([
            center[0] + radius * angle.cos(),
            center[1] + radius * angle.sin(),
        ]);
    }
    points
}

fn discretize_circle(center: Point2, radius: f64) -> Polyline {
    discretize_arc(center, radius, 0.0, 2.0 * std::f64::consts::PI)
}

fn extract_polylines_from_dxf(path: &PathBuf) -> Result<Vec<Polyline>, String> {
    let parser = parser::dxf_parser::DxfParser::new();
    let entities = parser
        .parse_file(path)
        .map_err(|e| format!("解析 DXF 失败: {:?}", e))?;

    let polylines: Vec<Polyline> = entities
        .iter()
        .filter_map(|entity| match entity {
            RawEntity::Line { start, end, .. } => Some(vec![*start, *end]),
            RawEntity::Polyline { points, closed, .. } => {
                let mut pts = points.clone();
                if *closed && pts.first() != pts.last() {
                    if let Some(first) = pts.first() {
                        pts.push(*first);
                    }
                }
                Some(pts)
            }
            RawEntity::Arc {
                center,
                radius,
                start_angle,
                end_angle,
                ..
            } => Some(discretize_arc(*center, *radius, *start_angle, *end_angle)),
            RawEntity::Circle { center, radius, .. } => Some(discretize_circle(*center, *radius)),
            _ => None,
        })
        .filter(|pts| pts.len() >= 2)
        .collect();

    if polylines.is_empty() {
        return Err(format!("未找到多段线，实体数={}", entities.len()));
    }

    Ok(polylines)
}

/// 分步性能诊断 - 使用真实 DXF 数据
#[test]
fn benchmark_real_dxf_step_timing() {
    let dxfs_dir = get_dxfs_dir();

    let test_files = vec![
        ("会议室1.dxf", "小型"),
        ("报告厅1.dxf", "中型"),
        ("报告厅3.dxf", "大型"),
        ("报告厅4.dxf", "超大型"),
    ];

    println!("\n=== 真实 DXF 分步性能诊断 ===");

    for (filename, category) in &test_files {
        let file_path = dxfs_dir.join(filename);
        if !file_path.exists() {
            println!("  SKIP: {} (文件不存在)", filename);
            continue;
        }

        let polylines = match extract_polylines_from_dxf(&file_path) {
            Ok(pts) => pts,
            Err(e) => {
                println!("  SKIP: {} ({})", filename, e);
                continue;
            }
        };

        let total_segments: usize = polylines.iter().map(|p| p.len() - 1).sum();
        let total_points: usize = polylines.iter().map(|p| p.len()).sum();

        println!("\n--- {} ({}) ---", filename, category);
        println!(
            "  多段线: {}, 总点数: {}, 总线段: {}",
            polylines.len(),
            total_points,
            total_segments
        );

        // 超大文件保护：超过 100M 点时跳过（避免 OOM）
        if total_points > 100_000_000 {
            println!("  SKIP: 超过 100M 点上限，跳过处理");
            continue;
        }

        // 重复率评估
        let all_points: Vec<common_types::Point2> = polylines.iter().flatten().copied().collect();
        let mut exact_set = std::collections::HashSet::with_capacity(all_points.len());
        for &pt in &all_points {
            let key = ((pt[0] * 1e9).round() as i64, (pt[1] * 1e9).round() as i64);
            exact_set.insert(key);
        }
        let duplicate_ratio = 1.0 - (exact_set.len() as f64 / all_points.len() as f64);
        println!("  精确重复率: {:.1}%", duplicate_ratio * 100.0);

        // 串行路径分步计时
        let mut gb = GraphBuilder::new(0.5, common_types::LengthUnit::Mm);

        println!("  [串行] 开始 Snap ({} 点)...", all_points.len());
        let t0 = Instant::now();
        gb.snap_and_build(&polylines);
        let snap_time = t0.elapsed();
        println!(
            "  [串行] Snap 完成: {:.1}ms ({} 点 -> {} 点)",
            snap_time.as_secs_f64() * 1000.0,
            all_points.len(),
            gb.points().len()
        );

        let t1 = Instant::now();
        println!("  [串行] 开始 Overlap ({} 边)...", gb.edges().len());
        gb.detect_and_merge_overlapping_segments();
        let overlap_time = t1.elapsed();
        println!(
            "  [串行] Overlap 完成: {:.1}ms",
            overlap_time.as_secs_f64() * 1000.0
        );

        let t2 = Instant::now();
        println!("  [串行] 开始 Intersect ({} 线段)...", gb.edges().len());
        gb.compute_intersections_and_split();
        let intersect_time = t2.elapsed();
        println!(
            "  [串行] Intersect 完成: {:.1}ms",
            intersect_time.as_secs_f64() * 1000.0
        );

        use topo::loop_extractor::LoopExtractor;
        let extractor = LoopExtractor::new(0.5);
        println!("  [串行] 开始 LoopExt ({} 边)...", gb.edges().len());
        let loops = extractor.extract_loops(gb.points(), gb.edges());
        let loop_time = Instant::now();
        println!(
            "  [串行] LoopExt 完成: {:.1}ms ({} 环)",
            loop_time.elapsed().as_secs_f64() * 1000.0,
            loops.len()
        );

        let total = snap_time + overlap_time + intersect_time + loop_time.elapsed();

        println!(
            "  [串行] Snap={:>8.1}ms Overlap={:>7.1}ms Intersect={:>8.1}ms LoopExt={:>6.1}ms Total={:>8.1}ms",
            snap_time.as_secs_f64() * 1000.0,
            overlap_time.as_secs_f64() * 1000.0,
            intersect_time.as_secs_f64() * 1000.0,
            loop_time.elapsed().as_secs_f64() * 1000.0,
            total.as_secs_f64() * 1000.0,
        );

        // 并行路径分步计时
        let psnap_start = Instant::now();
        tracing::info!("开始并行 snap ({} 点)...", all_points.len());
        let (snapped_points, snap_index) =
            topo::parallel::snap_endpoints_parallel(&all_points, 0.5);
        let parallel_snap_elapsed = psnap_start.elapsed();
        println!(
            "  [并行] Snap: {} -> {} 点, {:.1}ms",
            all_points.len(),
            snapped_points.len(),
            parallel_snap_elapsed.as_secs_f64() * 1000.0
        );

        let mut gb2 = GraphBuilder::new(0.5, common_types::LengthUnit::Mm);
        let t_build = Instant::now();
        gb2.set_points_with_mapping(snapped_points, snap_index);
        gb2.build_edges_from_polylines(&polylines);
        let build_elapsed = t_build.elapsed();
        println!(
            "  [并行] BuildEdges: {} 边, {:.1}ms",
            gb2.edges().len(),
            build_elapsed.as_secs_f64() * 1000.0
        );

        let t3 = Instant::now();
        println!("  [并行] 开始 Overlap ({} 边)...", gb2.edges().len());
        gb2.detect_and_merge_overlapping_segments();
        let overlap_time2 = t3.elapsed();
        println!(
            "  [并行] Overlap 完成: {:.1}ms",
            overlap_time2.as_secs_f64() * 1000.0
        );

        let t4 = Instant::now();
        println!("  [并行] 开始 Intersect ({} 线段)...", gb2.edges().len());
        gb2.compute_intersections_and_split();
        let intersect_time2 = t4.elapsed();
        println!(
            "  [并行] Intersect 完成: {:.1}ms",
            intersect_time2.as_secs_f64() * 1000.0
        );

        println!("  [并行] 开始 LoopExt...");
        let loops2 = extractor.extract_loops(gb2.points(), gb2.edges());
        let loop_time2 = Instant::now();
        println!(
            "  [并行] LoopExt 完成: {:.1}ms ({} 环)",
            loop_time2.elapsed().as_secs_f64() * 1000.0,
            loops2.len()
        );

        println!(
            "  [并行] Snap={:>8.1}ms Overlap={:>7.1}ms Intersect={:>8.1}ms LoopExt={:>6.1}ms",
            parallel_snap_elapsed.as_secs_f64() * 1000.0,
            overlap_time2.as_secs_f64() * 1000.0,
            intersect_time2.as_secs_f64() * 1000.0,
            loop_time2.elapsed().as_secs_f64() * 1000.0,
        );

        println!("  结果: 串行环={}, 并行环={}", loops.len(), loops2.len());
        if loops.len() != loops2.len() {
            // 打印环面积对比，识别缺失的环
            let mut areas1: Vec<f64> = loops.iter().map(|l| l.signed_area.abs()).collect();
            let mut areas2: Vec<f64> = loops2.iter().map(|l| l.signed_area.abs()).collect();
            areas1.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            areas2.sort_by(|a, b| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));
            println!(
                "  串行环面积 (top 10): {:?}",
                &areas1[..areas1.len().min(10)]
            );
            println!(
                "  并行环面积 (top 10): {:?}",
                &areas2[..areas2.len().min(10)]
            );
            // 找出串行有但并行没有的面积
            let mut set2 = std::collections::HashSet::new();
            for &a in &areas2 {
                set2.insert((a * 1000.0).round() as i64);
            }
            for &a in &areas1 {
                let key = (a * 1000.0).round() as i64;
                if !set2.contains(&key) {
                    println!("  [差异] 串行环面积={:.3} 在并行中不存在", a);
                }
            }
        }
    }
}

/// 串行 vs 并行 TopoService 端到端对比
#[test]
fn benchmark_serial_vs_parallel_topo() {
    let dxfs_dir = get_dxfs_dir();

    let test_files = vec![
        ("会议室1.dxf", "小型"),
        ("报告厅1.dxf", "中型"),
        ("报告厅3.dxf", "大型"),
    ];

    println!("\n=== 串行 vs 并行 TopoService 端到端对比 ===");
    println!(
        "{:>20} | {:>6} | {:>10} | {:>10} | {:>8}",
        "File", "Segs", "Serial", "Parallel", "Speedup"
    );
    println!("{:-<70}", "");

    for (filename, _category) in &test_files {
        let file_path = dxfs_dir.join(filename);
        if !file_path.exists() {
            println!("{:>20} | {:>6} | {:>10}", filename, "-", "SKIP");
            continue;
        }

        let polylines = match extract_polylines_from_dxf(&file_path) {
            Ok(pts) => pts,
            Err(e) => {
                println!("{:>20} | {:>6} | {:>10}", filename, "-", e);
                continue;
            }
        };

        let total_segments: usize = polylines.iter().map(|p| p.len() - 1).sum();

        // 串行
        let serial_service = TopoService::with_default_config();
        let t0 = Instant::now();
        let serial_result = serial_service.build_topology(&polylines);
        let serial_time = t0.elapsed();

        // 并行
        let parallel_config = TopoConfig::optimized();
        let parallel_service = TopoService::with_config(&parallel_config);
        let t1 = Instant::now();
        let parallel_result = parallel_service.build_topology(&polylines);
        let parallel_time = t1.elapsed();

        let speedup = serial_time.as_secs_f64() / parallel_time.as_secs_f64();

        let serial_ok = serial_result.is_ok();
        let parallel_ok = parallel_result.is_ok();

        let serial_loops = serial_result
            .as_ref()
            .map(|r| r.all_loops.len())
            .unwrap_or(0);
        let parallel_loops = parallel_result
            .as_ref()
            .map(|r| r.all_loops.len())
            .unwrap_or(0);

        println!(
            "{:>20} | {:>6} | {:>7.1}ms {} | {:>7.1}ms {} | {:>6.2}x (环: {} vs {})",
            filename,
            total_segments,
            serial_time.as_secs_f64() * 1000.0,
            if serial_ok { "✓" } else { "✗" },
            parallel_time.as_secs_f64() * 1000.0,
            if parallel_ok { "✓" } else { "✗" },
            speedup,
            serial_loops,
            parallel_loops,
        );
    }
}
