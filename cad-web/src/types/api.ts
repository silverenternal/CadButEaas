import { z } from 'zod'

// ========== 几何类型 ==========
export const PointSchema = z.tuple([z.number(), z.number()])
export type Point = z.infer<typeof PointSchema>

// ========== 语义标注（在 ProcessResponse 之前定义，因为需要引用） ==========
export const BoundarySemanticSchema = z.enum([
  'hard_wall',
  'absorptive_wall',
  'opening',
  'window',
  'door',
  'custom',
])

export type BoundarySemantic = z.infer<typeof BoundarySemanticSchema>

// ========== 健康检查 ==========
export const HealthResponseSchema = z.object({
  status: z.enum(['healthy', 'unhealthy', 'degraded']),
  version: z.string(),
  api_version: z.string(),
})

export type HealthResponse = z.infer<typeof HealthResponseSchema>

// ========== 文件处理 ==========
export const ProcessResponseSchema = z.object({
  job_id: z.string(),
  status: z.enum(['completed', 'partial', 'failed']),
  message: z.string(),
  result: z.object({
    scene_summary: z.object({
      outer_boundaries: z.number(),
      holes: z.number(),
      total_points: z.number(),
    }),
    validation_summary: z.object({
      error_count: z.number(),
      warning_count: z.number(),
      passed: z.boolean(),
    }),
    output_size: z.number(),
  }),
  errors: z.array(z.string()).default([]),
  edges: z
    .array(
      z.object({
        id: z.number(),
        start: z.tuple([z.number(), z.number()]),
        end: z.tuple([z.number(), z.number()]),
        layer: z.string().optional(),
        is_wall: z.boolean(),
        semantic: BoundarySemanticSchema.optional(),
        // P0-NEW: 弧线支持（可选字段，后端返回时如果没有此字段则为直线）
        arc: z.object({
          center: z.tuple([z.number(), z.number()]),
          radius: z.number(),
          start_angle: z.number(),
          end_angle: z.number(),
          ccw: z.boolean().optional(),
        }).optional(),
      })
    )
    .optional(),
  // P0-4 新增：HATCH 数据
  hatches: z
    .array(
      z.object({
        id: z.number(),
        boundary_paths: z.array(
          z.object({
            type: z.enum(['polyline', 'arc', 'ellipse_arc', 'spline']),
            points: z.array(z.tuple([z.number(), z.number()])).optional(),
            closed: z.boolean().optional(),
            bulges: z.array(z.number()).optional(),  // P0-4 新增：bulge 字段
            center: z.tuple([z.number(), z.number()]).optional(),
            radius: z.number().optional(),
            start_angle: z.number().optional(),
            end_angle: z.number().optional(),
            ccw: z.boolean().optional(),
            major_axis: z.tuple([z.number(), z.number()]).optional(),
            minor_axis_ratio: z.number().optional(),
            control_points: z.array(z.tuple([z.number(), z.number()])).optional(),
            knots: z.array(z.number()).optional(),
            degree: z.number().optional(),
            weights: z.array(z.number()).optional(),        // P1-NEW-32: 样条权重
            fit_points: z.array(z.tuple([z.number(), z.number()])).optional(), // P1-NEW-32: 拟合点
            flags: z.number().optional(),                   // P1-NEW-32: 样条标志
            extrusion_direction: z.tuple([z.number(), z.number(), z.number()]).optional(), // P2-NEW-29: 椭圆法向量
          })
        ),
        pattern: z.object({
          type: z.enum(['predefined', 'custom', 'solid']),
          name: z.string().optional(),
          color: z.tuple([z.number(), z.number(), z.number(), z.number()]).optional(),
          scale: z.number().optional(),  // P0-4 修复：添加 scale 属性
          angle: z.number().optional(),  // P0-4 修复：添加 angle 属性
          pattern_def: z.object({
            name: z.string(),
            description: z.string().optional(),
            lines: z.array(
              z.object({
                start_point: z.tuple([z.number(), z.number()]),
                angle: z.number(),
                offset: z.tuple([z.number(), z.number()]),
                dash_pattern: z.array(z.number()),
              })
            ),
          }).optional(),
        }),
        solid_fill: z.boolean(),
        layer: z.string().optional(),
        scale: z.number().optional(),  // P0-NEW-14 修复：添加 scale 属性
        angle: z.number().optional(),  // P0-NEW-14 修复：添加 angle 属性
      })
    )
    .optional(),
})

export type ProcessResponse = z.infer<typeof ProcessResponseSchema>

// P0-4 新增：从 ProcessResponse 中提取 HATCH 类型
export type HatchEntity = NonNullable<ProcessResponse['hatches']>[number]
export type HatchBoundaryPath = HatchEntity['boundary_paths'][number]
export type HatchPattern = HatchEntity['pattern']

// ========== 配置管理 ==========
export const ProfileSchema = z.object({
  name: z.string(),
  description: z.string(),
})

export const ProfileDetailSchema = z.object({
  name: z.string(),
  topology: z.object({
    snap_tolerance_mm: z.number(),
    min_line_length_mm: z.number(),
    merge_angle_tolerance_deg: z.number(),
    max_gap_bridge_length_mm: z.number(),
  }),
  validator: z.object({
    closure_tolerance_mm: z.number(),
    min_area_m2: z.number(),
    min_edge_length_mm: z.number(),
    min_angle_deg: z.number(),
  }),
  export: z.object({
    format: z.enum(['json', 'binary']),
    json_indent: z.number(),
    auto_validate: z.boolean(),
  }),
})

export type Profile = z.infer<typeof ProfileSchema>
export type ProfileDetail = z.infer<typeof ProfileDetailSchema>

// ========== 交互功能 ==========
export const AutoTraceRequestSchema = z.object({
  edge_id: z.number(),
})

export const AutoTraceResponseSchema = z.object({
  success: z.boolean(),
  loop_points: z.array(z.tuple([z.number(), z.number()])),
  message: z.string(),
})

export const LassoRequestSchema = z.object({
  polygon: z.array(z.tuple([z.number(), z.number()])),
})

export const LassoResponseSchema = z.object({
  selected_edges: z.array(z.number()),
  loops: z.array(z.array(z.tuple([z.number(), z.number()]))),
  connected_components: z.number(),
})

export const GapDetectionRequestSchema = z.object({
  tolerance: z.number(),
})

export const GapInfoSchema = z.object({
  id: z.number(),
  start: z.tuple([z.number(), z.number()]),
  end: z.tuple([z.number(), z.number()]),
  length: z.number(),
  gap_type: z.enum(['collinear', 'orthogonal', 'angled', 'small']),
})

export const GapDetectionResponseSchema = z.object({
  gaps: z.array(GapInfoSchema),
  total_count: z.number(),
})

export type AutoTraceRequest = z.infer<typeof AutoTraceRequestSchema>
export type AutoTraceResponse = z.infer<typeof AutoTraceResponseSchema>
export type LassoRequest = z.infer<typeof LassoRequestSchema>
export type LassoResponse = z.infer<typeof LassoResponseSchema>
export type GapDetectionRequest = z.infer<typeof GapDetectionRequestSchema>
export type GapDetectionResponse = z.infer<typeof GapDetectionResponseSchema>
export type GapInfo = z.infer<typeof GapInfoSchema>

// ========== 缺口桥接 ==========
export const SnapBridgeRequestSchema = z.object({
  gap_id: z.number(),
})

export type SnapBridgeRequest = z.infer<typeof SnapBridgeRequestSchema>

// ========== 语义标注 ==========
// BoundarySemanticSchema 已在文件顶部定义

export const SetSemanticRequestSchema = z.object({
  segment_id: z.number(),
  semantic: BoundarySemanticSchema,
})

export const SetSemanticResponseSchema = z.object({
  success: z.boolean(),
  message: z.string(),
})

export type SetSemanticRequest = z.infer<typeof SetSemanticRequestSchema>
export type SetSemanticResponse = z.infer<typeof SetSemanticResponseSchema>

// ========== 导出功能 ==========
export const ExportRequestSchema = z.object({
  format: z.enum(['json', 'bincode', 'dxf']),
  pretty: z.boolean().optional(),
})

export const ExportResponseSchema = z.object({
  success: z.boolean(),
  message: z.string(),
  download_url: z.string().optional(),
  file_name: z.string().optional(),
  file_size: z.number(),
})

export type ExportRequest = z.infer<typeof ExportRequestSchema>
export type ExportResponse = z.infer<typeof ExportResponseSchema>

// ========== 几何类型 ==========
// PointSchema 已在文件顶部定义

// 弧线类型定义
export const ArcEdgeSchema = z.object({
  center: PointSchema,        // 圆心
  radius: z.number(),         // 半径
  start_angle: z.number(),    // 起始角度（弧度）
  end_angle: z.number(),      // 终止角度（弧度）
  ccw: z.boolean().optional(), // 是否逆时针（默认 false 顺时针）
})
export type ArcEdge = z.infer<typeof ArcEdgeSchema>

export const EdgeSchema = z.object({
  id: z.number(),
  start: PointSchema,
  end: PointSchema,
  layer: z.string().optional(),
  is_wall: z.boolean(),
  semantic: BoundarySemanticSchema.optional(),
  arc: ArcEdgeSchema.optional(), // 如果是弧线，此字段包含弧线参数
})
export type Edge = z.infer<typeof EdgeSchema>

export const CameraStateSchema = z.object({
  zoom: z.number(),
  offsetX: z.number(),
  offsetY: z.number(),
})
export type CameraState = z.infer<typeof CameraStateSchema>

// ========== 错误响应 ==========
export const ErrorResponseSchema = z.object({
  request_id: z.string(),
  status: z.literal('FAILURE'),
  error: z.object({
    code: z.string(),
    message: z.string(),
    details: z.record(z.unknown()).optional(),
    retryable: z.boolean().optional(),
    suggestion: z.string().optional(),
  }),
  latency_ms: z.number(),
})

export type ErrorResponse = z.infer<typeof ErrorResponseSchema>
