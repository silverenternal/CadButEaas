//! Scene assembly helpers.
//!
//! This crate owns conversions from parsed CAD entities into renderable scene
//! fragments. Keeping these mappings outside `orchestrator` lets the pipeline
//! focus on stage ordering instead of geometry DTO assembly.

use common_types::{
    BoundarySegment, BoundarySemantic, DimensionSummary, DimensionType, HatchBoundaryPath, Point2,
    Polyline, RawEdge, RawEntity, SceneState, TextAnnotation,
};
use rayon::prelude::*;

pub fn polylines_to_raw_edges(polylines: &[Polyline]) -> Vec<RawEdge> {
    let mut edges = Vec::new();
    for polyline in polylines {
        for segment in polyline.windows(2) {
            edges.push(RawEdge {
                id: edges.len(),
                start: segment[0],
                end: segment[1],
                layer: None,
                color_index: None,
            });
        }
    }
    edges
}

pub fn extract_polylines_from_entities(entities: &[RawEntity]) -> Vec<Polyline> {
    entities
        .par_iter()
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
            RawEntity::Hatch { boundary_paths, .. } => {
                let mut all_points = Vec::new();
                for path in boundary_paths {
                    match path {
                        HatchBoundaryPath::Polyline { points, closed, .. } => {
                            all_points.extend(points.clone());
                            if *closed && !points.is_empty() && points.first() != points.last() {
                                all_points.push(points[0]);
                            }
                        }
                        HatchBoundaryPath::Arc {
                            center,
                            radius,
                            start_angle,
                            end_angle,
                            ..
                        } => all_points.extend(discretize_arc(
                            *center,
                            *radius,
                            *start_angle,
                            *end_angle,
                        )),
                        _ => {}
                    }
                }
                (all_points.len() >= 2).then_some(all_points)
            }
            RawEntity::MLine {
                center_line,
                closed,
                ..
            } => {
                let mut pts = center_line.clone();
                if *closed && pts.first() != pts.last() {
                    if let Some(first) = pts.first() {
                        pts.push(*first);
                    }
                }
                (pts.len() >= 2).then_some(pts)
            }
            RawEntity::Leader { points, .. } => (points.len() >= 2).then(|| points.clone()),
            RawEntity::Ray { .. } => None,
            _ => None,
        })
        .collect()
}

pub fn extract_text_annotations(entities: &[RawEntity]) -> Vec<TextAnnotation> {
    entities
        .iter()
        .filter_map(|entity| match entity {
            RawEntity::Text {
                position,
                content,
                height,
                rotation,
                ..
            } => Some(TextAnnotation {
                position: *position,
                content: content.clone(),
                height: *height,
                rotation: *rotation,
            }),
            _ => None,
        })
        .collect()
}

pub fn extract_dimension_summary(entities: &[RawEntity]) -> DimensionSummary {
    let mut summary = DimensionSummary::default();

    for entity in entities {
        if let RawEntity::Dimension {
            dimension_type,
            measurement,
            ..
        } = entity
        {
            summary.total_count += 1;
            match dimension_type {
                DimensionType::Linear => summary.linear_count += 1,
                DimensionType::Aligned => summary.aligned_count += 1,
                DimensionType::Angular => summary.angular_count += 1,
                DimensionType::Radial => summary.radial_count += 1,
                DimensionType::Diameter => summary.diameter_count += 1,
                DimensionType::ArcLength => summary.angular_count += 1,
                DimensionType::Ordinate => summary.ordinate_count += 1,
            }
            if *measurement > 0.0 {
                summary.max_measurement = Some(
                    summary
                        .max_measurement
                        .map_or(*measurement, |max| max.max(*measurement)),
                );
                summary.min_measurement = Some(
                    summary
                        .min_measurement
                        .map_or(*measurement, |min| min.min(*measurement)),
                );
            }
        }
    }

    summary
}

pub fn fill_scene_edges(scene: &mut SceneState, entities: &[RawEntity]) {
    let edges: Vec<RawEdge> = entities.par_iter().flat_map(entity_to_edges).collect();

    scene.edges = edges
        .into_iter()
        .enumerate()
        .map(|(id, mut edge)| {
            edge.id = id;
            edge
        })
        .collect();
}

pub fn auto_infer_boundaries(scene: &mut SceneState, entities: &[&RawEntity]) {
    let entity_info_map: std::collections::HashMap<usize, (Option<String>, Option<String>)> =
        entities
            .iter()
            .enumerate()
            .map(|(idx, entity)| {
                let layer = (*entity).layer().map(String::from);
                let color = (*entity).color().map(String::from);
                (idx, (layer, color))
            })
            .collect();

    if let Some(outer) = &scene.outer {
        let points = &outer.points;
        let n = points.len();

        let boundaries: Vec<BoundarySegment> = (0..n)
            .into_par_iter()
            .map(|i| {
                let start = points[i];
                let end = points[(i + 1) % n];
                infer_boundary_segment(&entity_info_map, entities, start, end, i, (i + 1) % n)
            })
            .collect();

        scene.boundaries.extend(boundaries);
    }

    for hole in scene.holes.iter() {
        let points = &hole.points;
        let n = points.len();

        let boundaries: Vec<BoundarySegment> = (0..n)
            .into_par_iter()
            .map(|i| {
                let start = points[i];
                let end = points[(i + 1) % n];
                infer_boundary_segment(&entity_info_map, entities, start, end, i, (i + 1) % n)
            })
            .collect();

        scene.boundaries.extend(boundaries);
    }
}

fn find_matching_entity_info_from_refs(
    entity_info_map: &std::collections::HashMap<usize, (Option<String>, Option<String>)>,
    start: Point2,
    end: Point2,
    entities: &[&RawEntity],
) -> (Option<String>, Option<String>) {
    const SNAP_TOLERANCE: f64 = 5.0;

    for (idx, entity) in entities.iter().enumerate() {
        match *entity {
            RawEntity::Line {
                start: e_start,
                end: e_end,
                ..
            } => {
                let start_dist = distance_2d(start, *e_start);
                let end_dist = distance_2d(end, *e_end);

                if start_dist < SNAP_TOLERANCE && end_dist < SNAP_TOLERANCE {
                    if let Some((layer, color)) = entity_info_map.get(&idx) {
                        return (layer.clone(), color.clone());
                    }
                }

                let start_dist_rev = distance_2d(start, *e_end);
                let end_dist_rev = distance_2d(end, *e_start);

                if start_dist_rev < SNAP_TOLERANCE && end_dist_rev < SNAP_TOLERANCE {
                    if let Some((layer, color)) = entity_info_map.get(&idx) {
                        return (layer.clone(), color.clone());
                    }
                }
            }
            RawEntity::Polyline { points, closed, .. } => {
                for segment in points.windows(2) {
                    let start_dist = distance_2d(start, segment[0]);
                    let end_dist = distance_2d(end, segment[1]);

                    if start_dist < SNAP_TOLERANCE && end_dist < SNAP_TOLERANCE {
                        if let Some((layer, color)) = entity_info_map.get(&idx) {
                            return (layer.clone(), color.clone());
                        }
                    }
                }

                if *closed && !points.is_empty() {
                    let p1 = points[points.len() - 1];
                    let p2 = points[0];

                    let start_dist = distance_2d(start, p1);
                    let end_dist = distance_2d(end, p2);

                    if start_dist < SNAP_TOLERANCE && end_dist < SNAP_TOLERANCE {
                        if let Some((layer, color)) = entity_info_map.get(&idx) {
                            return (layer.clone(), color.clone());
                        }
                    }
                }
            }
            _ => {}
        }
    }

    (None, None)
}

fn infer_boundary_segment(
    entity_info_map: &std::collections::HashMap<usize, (Option<String>, Option<String>)>,
    entities: &[&RawEntity],
    start: Point2,
    end: Point2,
    seg_start: usize,
    seg_end: usize,
) -> BoundarySegment {
    let (layer, color) = find_matching_entity_info_from_refs(entity_info_map, start, end, entities);

    let semantic = if let Some(ref layer_name) = layer {
        BoundarySegment::infer_semantic_from_layer(layer_name)
    } else {
        BoundarySemantic::HardWall
    };

    let material = if let Some(ref color_name) = color {
        if let Ok(color_idx) = color_name.parse::<u16>() {
            BoundarySegment::infer_material_from_aci_color(color_idx)
        } else {
            None
        }
    } else if let Some(ref layer_name) = layer {
        BoundarySegment::infer_material_from_layer(layer_name)
    } else {
        None
    };

    let width = if matches!(
        semantic,
        BoundarySemantic::Door | BoundarySemantic::Window | BoundarySemantic::Opening
    ) {
        BoundarySegment::calculate_width(start, end)
    } else {
        None
    };

    BoundarySegment {
        segment: [seg_start, seg_end],
        semantic,
        material,
        width,
    }
}

fn entity_to_edges(entity: &RawEntity) -> Vec<RawEdge> {
    match entity {
        RawEntity::Line {
            start,
            end,
            metadata,
            ..
        } => vec![RawEdge {
            id: 0,
            start: *start,
            end: *end,
            layer: metadata.layer.clone(),
            color_index: None,
        }],
        RawEntity::Polyline {
            points,
            closed,
            metadata,
            ..
        } => {
            let mut edges = Vec::new();
            for segment in points.windows(2) {
                edges.push(RawEdge {
                    id: 0,
                    start: segment[0],
                    end: segment[1],
                    layer: metadata.layer.clone(),
                    color_index: None,
                });
            }
            if *closed && points.len() >= 2 {
                edges.push(RawEdge {
                    id: 0,
                    start: points[points.len() - 1],
                    end: points[0],
                    layer: metadata.layer.clone(),
                    color_index: None,
                });
            }
            edges
        }
        RawEntity::Arc {
            center,
            radius,
            start_angle,
            end_angle,
            metadata,
            ..
        } => discretize_arc(*center, *radius, *start_angle, *end_angle)
            .windows(2)
            .map(|segment| RawEdge {
                id: 0,
                start: segment[0],
                end: segment[1],
                layer: metadata.layer.clone(),
                color_index: None,
            })
            .collect(),
        RawEntity::Circle {
            center,
            radius,
            metadata,
            ..
        } => discretize_circle(*center, *radius)
            .windows(2)
            .map(|segment| RawEdge {
                id: 0,
                start: segment[0],
                end: segment[1],
                layer: metadata.layer.clone(),
                color_index: None,
            })
            .collect(),
        RawEntity::MLine {
            center_line,
            metadata,
            ..
        } => center_line
            .windows(2)
            .map(|segment| RawEdge {
                id: 0,
                start: segment[0],
                end: segment[1],
                layer: metadata.layer.clone(),
                color_index: None,
            })
            .collect(),
        RawEntity::Leader {
            points, metadata, ..
        } => points
            .windows(2)
            .map(|segment| RawEdge {
                id: 0,
                start: segment[0],
                end: segment[1],
                layer: metadata.layer.clone(),
                color_index: None,
            })
            .collect(),
        RawEntity::Dimension {
            definition_points,
            metadata,
            ..
        } => definition_points
            .windows(2)
            .map(|segment| RawEdge {
                id: 0,
                start: segment[0],
                end: segment[1],
                layer: metadata.layer.clone(),
                color_index: None,
            })
            .collect(),
        _ => Vec::new(),
    }
}

fn discretize_arc(center: Point2, radius: f64, start_angle: f64, end_angle: f64) -> Polyline {
    let start_rad = start_angle.to_radians();
    let end_rad = end_angle.to_radians();
    let mut angle_diff = end_rad - start_rad;
    if angle_diff < 0.0 {
        angle_diff += 2.0 * std::f64::consts::PI;
    }

    let arc_length = radius * angle_diff;
    let num_segments = (arc_length / 1.0).ceil() as usize;
    let num_segments = num_segments.max(8);

    (0..=num_segments)
        .map(|i| {
            let t = i as f64 / num_segments as f64;
            let angle = start_rad + t * angle_diff;
            [
                center[0] + radius * angle.cos(),
                center[1] + radius * angle.sin(),
            ]
        })
        .collect()
}

fn distance_2d(a: Point2, b: Point2) -> f64 {
    let dx = a[0] - b[0];
    let dy = a[1] - b[1];
    (dx * dx + dy * dy).sqrt()
}

fn discretize_circle(center: Point2, radius: f64) -> Polyline {
    let circumference = 2.0 * std::f64::consts::PI * radius;
    let num_segments = (circumference / 1.0).ceil() as usize;
    let num_segments = num_segments.max(32);

    let mut points: Polyline = (0..num_segments)
        .map(|i| {
            let angle = 2.0 * std::f64::consts::PI * i as f64 / num_segments as f64;
            [
                center[0] + radius * angle.cos(),
                center[1] + radius * angle.sin(),
            ]
        })
        .collect();

    if let Some(first) = points.first().copied() {
        points.push(first);
    }

    points
}
