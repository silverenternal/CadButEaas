use super::{
    HatchBoundaryPathResponse, HatchEntity, HatchPatternDefinitionResponse,
    HatchPatternLineResponse, HatchPatternResponse,
};

/// 从实体列表提取 HATCH 数据（用于前端渲染）
pub(super) fn entities_to_hatches(entities: &[common_types::RawEntity]) -> Vec<HatchEntity> {
    let mut hatches = Vec::new();
    let mut hatch_id = 0;

    for entity in entities {
        if let common_types::RawEntity::Hatch {
            boundary_paths,
            pattern,
            solid_fill,
            metadata,
            scale,
            angle,
            ..
        } = entity
        {
            let boundary_paths_response = boundary_paths
                .iter()
                .map(hatch_boundary_path_to_response)
                .collect();

            hatches.push(HatchEntity {
                id: hatch_id,
                boundary_paths: boundary_paths_response,
                pattern: hatch_pattern_to_response(pattern, *scale, *angle),
                solid_fill: *solid_fill,
                layer: metadata.layer.clone(),
                scale: *scale,
                angle: *angle,
            });

            hatch_id += 1;
        }
    }

    tracing::info!(
        "entities_to_hatches: 从 {} 个实体提取 {} 个 HATCH",
        entities.len(),
        hatches.len()
    );

    hatches
}

fn hatch_boundary_path_to_response(
    boundary: &common_types::HatchBoundaryPath,
) -> HatchBoundaryPathResponse {
    match boundary {
        common_types::HatchBoundaryPath::Polyline {
            points,
            closed,
            bulges,
        } => HatchBoundaryPathResponse::Polyline {
            points: points.iter().map(|p| [p[0], p[1]]).collect(),
            closed: *closed,
            bulges: bulges.clone(),
        },
        common_types::HatchBoundaryPath::Arc {
            center,
            radius,
            start_angle,
            end_angle,
            ccw,
        } => HatchBoundaryPathResponse::Arc {
            center: [center[0], center[1]],
            radius: *radius,
            start_angle: *start_angle,
            end_angle: *end_angle,
            ccw: *ccw,
        },
        common_types::HatchBoundaryPath::EllipseArc {
            center,
            major_axis,
            minor_axis_ratio,
            start_angle,
            end_angle,
            ccw,
            extrusion_direction: _,
        } => HatchBoundaryPathResponse::EllipseArc {
            center: [center[0], center[1]],
            major_axis: [major_axis[0], major_axis[1]],
            minor_axis_ratio: *minor_axis_ratio,
            start_angle: *start_angle,
            end_angle: *end_angle,
            ccw: *ccw,
        },
        common_types::HatchBoundaryPath::Spline {
            control_points,
            knots,
            degree,
            weights: _,
            fit_points: _,
            flags: _,
        } => HatchBoundaryPathResponse::Spline {
            control_points: control_points.iter().map(|p| [p[0], p[1]]).collect(),
            knots: knots.clone(),
            degree: *degree,
        },
    }
}

fn hatch_pattern_to_response(
    pattern: &common_types::HatchPattern,
    scale: f64,
    angle: f64,
) -> HatchPatternResponse {
    match pattern {
        common_types::HatchPattern::Predefined { name } => HatchPatternResponse::Predefined {
            name: name.clone(),
            scale,
            angle,
        },
        common_types::HatchPattern::Custom { pattern_def } => HatchPatternResponse::Custom {
            pattern_def: HatchPatternDefinitionResponse {
                name: pattern_def.name.clone(),
                description: pattern_def.description.clone(),
                lines: pattern_def
                    .lines
                    .iter()
                    .map(|line| HatchPatternLineResponse {
                        start_point: [line.start_point[0], line.start_point[1]],
                        angle: line.angle,
                        offset: [line.offset[0], line.offset[1]],
                        dash_pattern: line.dash_pattern.clone(),
                    })
                    .collect(),
            },
            scale,
            angle,
        },
        common_types::HatchPattern::Solid { color } => HatchPatternResponse::Solid {
            color: [color.r, color.g, color.b, color.a],
        },
    }
}
