use super::GapInfoResponse;
use common_types::SceneState;
use interact::{Edge, GapInfo};

/// 从实体列表提取边（用于快速渲染）
pub(super) fn entities_to_edges(entities: &[common_types::RawEntity]) -> Vec<Edge> {
    let mut edges = Vec::new();
    let mut edge_id = 0;

    for entity in entities {
        match entity {
            common_types::RawEntity::Line {
                start,
                end,
                metadata,
                ..
            } => {
                push_edge(
                    &mut edges,
                    &mut edge_id,
                    *start,
                    *end,
                    metadata.layer.clone(),
                );
            }
            common_types::RawEntity::Polyline {
                points,
                closed,
                metadata,
                ..
            } => {
                push_polyline_edges(
                    &mut edges,
                    &mut edge_id,
                    points,
                    *closed,
                    metadata.layer.clone(),
                );
            }
            common_types::RawEntity::Arc {
                center,
                radius,
                start_angle,
                end_angle,
                metadata,
                ..
            } => {
                push_arc_edges(
                    &mut edges,
                    &mut edge_id,
                    *center,
                    *radius,
                    *start_angle,
                    *end_angle,
                    true,
                    16,
                    metadata.layer.clone(),
                );
            }
            common_types::RawEntity::Circle {
                center,
                radius,
                metadata,
                ..
            } => {
                let segments = 32;
                for i in 0..segments {
                    let a1 = 2.0 * std::f64::consts::PI * (i as f64) / segments as f64;
                    let a2 = 2.0 * std::f64::consts::PI * ((i + 1) as f64) / segments as f64;
                    let p1 = [center[0] + radius * a1.cos(), center[1] + radius * a1.sin()];
                    let p2 = [center[0] + radius * a2.cos(), center[1] + radius * a2.sin()];
                    push_edge(&mut edges, &mut edge_id, p1, p2, metadata.layer.clone());
                }
            }
            common_types::RawEntity::Text {
                position,
                height,
                content,
                metadata,
                ..
            } => {
                let char_count = content.chars().count() as f64;
                let width = height * char_count * 0.6;
                let text_height = height * 1.2;
                let x = position[0];
                let y = position[1];
                let corners = [
                    [x, y],
                    [x + width, y],
                    [x + width, y + text_height],
                    [x, y + text_height],
                ];

                for i in 0..4 {
                    push_edge(
                        &mut edges,
                        &mut edge_id,
                        corners[i],
                        corners[(i + 1) % 4],
                        metadata.layer.clone(),
                    );
                }
            }
            common_types::RawEntity::BlockReference { block_name, .. } => {
                tracing::debug!("阶段 1 跳过块引用：{} (需要块定义数据)", block_name);
            }
            common_types::RawEntity::Dimension {
                definition_points,
                metadata,
                ..
            } => {
                push_open_polyline_edges(
                    &mut edges,
                    &mut edge_id,
                    definition_points,
                    metadata.layer.clone(),
                );
            }
            common_types::RawEntity::Path {
                commands, metadata, ..
            } => {
                let mut current_point: Option<[f64; 2]> = None;

                for cmd in commands {
                    match cmd {
                        common_types::PathCommand::MoveTo { x, y } => {
                            current_point = Some([*x, *y]);
                        }
                        common_types::PathCommand::LineTo { x, y }
                        | common_types::PathCommand::ArcTo { x, y, .. } => {
                            if let Some(start) = current_point {
                                push_edge(
                                    &mut edges,
                                    &mut edge_id,
                                    start,
                                    [*x, *y],
                                    metadata.layer.clone(),
                                );
                                current_point = Some([*x, *y]);
                            }
                        }
                        common_types::PathCommand::Close => {}
                    }
                }
            }
            common_types::RawEntity::Hatch {
                boundary_paths,
                metadata,
                ..
            } => {
                for boundary in boundary_paths {
                    push_hatch_boundary_edges(
                        &mut edges,
                        &mut edge_id,
                        boundary,
                        metadata.layer.clone(),
                    );
                }
            }
            common_types::RawEntity::XRef { .. } => {
                tracing::warn!("XREF 外部参照支持 - 待完整实现，跳过处理");
            }
            common_types::RawEntity::Leader {
                points, metadata, ..
            }
            | common_types::RawEntity::MLine {
                center_line: points,
                metadata,
                ..
            } => {
                push_open_polyline_edges(&mut edges, &mut edge_id, points, metadata.layer.clone());
            }
            common_types::RawEntity::Ray {
                start,
                direction,
                metadata,
                ..
            } => {
                let ray_end = [
                    start[0] + direction[0] * 10000.0,
                    start[1] + direction[1] * 10000.0,
                ];
                push_edge(
                    &mut edges,
                    &mut edge_id,
                    *start,
                    ray_end,
                    metadata.layer.clone(),
                );
            }
            common_types::RawEntity::Point { .. }
            | common_types::RawEntity::Image { .. }
            | common_types::RawEntity::Attribute { .. }
            | common_types::RawEntity::AttributeDefinition { .. }
            | common_types::RawEntity::Triangle { .. } => {}
        }
    }

    tracing::info!(
        "entities_to_edges: 从 {} 个实体提取 {} 条边",
        entities.len(),
        edges.len()
    );
    edges
}

/// 辅助函数：从 SceneState 创建 Edge 集合
pub(super) fn scene_to_edges(scene: &SceneState) -> Vec<Edge> {
    let mut edges = Vec::new();

    if !scene.edges.is_empty() {
        for raw_edge in &scene.edges {
            let mut edge = Edge::new(raw_edge.id, raw_edge.start, raw_edge.end);
            edge.layer = raw_edge.layer.clone();
            edges.push(edge);
        }
        return edges;
    }

    let mut edge_id = 0;
    if let Some(outer) = &scene.outer {
        push_loop_edges(&mut edges, &mut edge_id, &outer.points);
    }

    for hole in &scene.holes {
        push_loop_edges(&mut edges, &mut edge_id, &hole.points);
    }

    edges
}

/// 辅助函数：转换 GapInfo 为 GapInfoResponse
pub(super) fn gap_info_to_response(gap: &GapInfo) -> GapInfoResponse {
    GapInfoResponse {
        id: gap.id,
        start: [gap.endpoint_a[0], gap.endpoint_a[1]],
        end: [gap.endpoint_b[0], gap.endpoint_b[1]],
        length: gap.length,
        gap_type: format!("{:?}", gap.gap_type),
    }
}

fn push_edge(
    edges: &mut Vec<Edge>,
    edge_id: &mut usize,
    start: [f64; 2],
    end: [f64; 2],
    layer: Option<String>,
) {
    let mut edge = Edge::new(*edge_id, start, end);
    edge.layer = layer;
    edges.push(edge);
    *edge_id += 1;
}

fn push_open_polyline_edges(
    edges: &mut Vec<Edge>,
    edge_id: &mut usize,
    points: &[[f64; 2]],
    layer: Option<String>,
) {
    if points.len() < 2 {
        return;
    }

    for i in 0..points.len() - 1 {
        push_edge(edges, edge_id, points[i], points[i + 1], layer.clone());
    }
}

fn push_polyline_edges(
    edges: &mut Vec<Edge>,
    edge_id: &mut usize,
    points: &[[f64; 2]],
    closed: bool,
    layer: Option<String>,
) {
    push_open_polyline_edges(edges, edge_id, points, layer.clone());

    if closed && points.len() >= 2 {
        push_edge(edges, edge_id, points[points.len() - 1], points[0], layer);
    }
}

fn push_loop_edges(edges: &mut Vec<Edge>, edge_id: &mut usize, points: &[[f64; 2]]) {
    if points.len() < 2 {
        return;
    }

    for i in 0..points.len() {
        push_edge(
            edges,
            edge_id,
            points[i],
            points[(i + 1) % points.len()],
            None,
        );
    }
}

#[allow(clippy::too_many_arguments)]
fn push_arc_edges(
    edges: &mut Vec<Edge>,
    edge_id: &mut usize,
    center: [f64; 2],
    radius: f64,
    start_angle: f64,
    end_angle: f64,
    ccw: bool,
    segments: usize,
    layer: Option<String>,
) {
    let angle_range = if ccw {
        end_angle - start_angle
    } else {
        start_angle - end_angle
    };

    for i in 0..segments {
        let a1 = start_angle + (angle_range * (i as f64) / segments as f64);
        let a2 = start_angle + (angle_range * ((i + 1) as f64) / segments as f64);
        let p1 = [
            center[0] + radius * a1.to_radians().cos(),
            center[1] + radius * a1.to_radians().sin(),
        ];
        let p2 = [
            center[0] + radius * a2.to_radians().cos(),
            center[1] + radius * a2.to_radians().sin(),
        ];
        push_edge(edges, edge_id, p1, p2, layer.clone());
    }
}

fn push_hatch_boundary_edges(
    edges: &mut Vec<Edge>,
    edge_id: &mut usize,
    boundary: &common_types::HatchBoundaryPath,
    layer: Option<String>,
) {
    match boundary {
        common_types::HatchBoundaryPath::Polyline { points, closed, .. } => {
            push_polyline_edges(edges, edge_id, points, *closed, layer);
        }
        common_types::HatchBoundaryPath::Arc {
            center,
            radius,
            start_angle,
            end_angle,
            ccw,
            ..
        } => {
            push_arc_edges(
                edges,
                edge_id,
                *center,
                *radius,
                *start_angle,
                *end_angle,
                *ccw,
                16,
                layer,
            );
        }
        common_types::HatchBoundaryPath::EllipseArc {
            center,
            major_axis,
            minor_axis_ratio,
            start_angle,
            end_angle,
            ccw,
            ..
        } => {
            let segments = 32;
            let angle_range = if *ccw {
                end_angle - start_angle
            } else {
                start_angle - end_angle
            };
            let mut previous: Option<[f64; 2]> = None;

            for i in 0..segments {
                let t = (i as f64) / segments as f64;
                let angle = start_angle + angle_range * t;
                let angle_rad = angle.to_radians();
                let point = [
                    center[0] + major_axis[0] * angle_rad.cos(),
                    center[1] + major_axis[1] * minor_axis_ratio * angle_rad.sin(),
                ];

                if let Some(prev) = previous {
                    push_edge(edges, edge_id, prev, point, layer.clone());
                }
                previous = Some(point);
            }
        }
        common_types::HatchBoundaryPath::Spline { control_points, .. } => {
            push_open_polyline_edges(edges, edge_id, control_points, layer);
        }
    }
}
