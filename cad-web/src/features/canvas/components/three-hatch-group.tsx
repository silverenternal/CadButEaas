import { useMemo, useRef, useEffect } from 'react'
import * as THREE from 'three'
import { mergeGeometries as mergeBufferGeometries } from 'three/examples/jsm/utils/BufferGeometryUtils.js'
import type { HatchEntity } from '@/types/api'

interface HatchGroupProps {
  hatches: HatchEntity[]
}

/**
 * Three.js HATCH 填充渲染组
 * ✅ S014: 按材质分组合并几何体，减少 Draw Call
 */
export function HatchGroup({ hatches }: HatchGroupProps) {
  // 调试：记录 HATCH 数据
  console.log('[HatchGroup] Rendering hatches:', hatches.length)

  // ✅ S014: 按材质属性分组 HATCH
  const groupedHatches = useMemo(() => {
    const groups = new Map<string, HatchEntity[]>()
    
    hatches.forEach((hatch) => {
      // 使用颜色 + 透明度作为分组 key
      const color = hatch.solid_fill
        ? `${hatch.pattern?.color?.[0] || 128}-${hatch.pattern?.color?.[1] || 128}-${hatch.pattern?.color?.[2] || 128}`
        : '99-99-99'
      const opacity = hatch.solid_fill ? 0.5 : 0.3
      const key = `${color}-${opacity}`
      
      const group = groups.get(key) || []
      group.push(hatch)
      groups.set(key, group)
    })
    
    return groups
  }, [hatches])

  return (
    <group name="hatches">
      {Array.from(groupedHatches.entries()).map(([key, groupHatches], index) => (
        <MergedHatchBatch
          key={key}
          hatches={groupHatches}
          batchIndex={index}
        />
      ))}
    </group>
  )
}

interface MergedHatchBatchProps {
  hatches: HatchEntity[]
  batchIndex: number
}

/**
 * 合并的 HATCH 批次渲染
 * ✅ S014: 使用 mergeGeometries 合并同材质的 HATCH
 */
function MergedHatchBatch({ hatches, batchIndex }: MergedHatchBatchProps) {
  const meshRef = useRef<THREE.Mesh>(null)
  
  // 创建合并的几何体和材质
  const { geometry, material } = useMemo(() => {
    try {
      const geometries: THREE.ShapeGeometry[] = []
      
      // 为每个 HATCH 创建 ShapeGeometry
      hatches.forEach((hatch) => {
        try {
          const shape = createHatchShape(hatch)
          if (shape) {
            const geo = new THREE.ShapeGeometry(shape)
            geometries.push(geo)
          }
        } catch (error) {
          console.error('[MergedHatchBatch] Error creating hatch geometry:', error, hatch)
        }
      })
      
      if (geometries.length === 0) {
        return {
          geometry: new THREE.BufferGeometry(),
          material: new THREE.MeshBasicMaterial({ visible: false }),
        }
      }
      
      // ✅ S014: 合并几何体
      const mergedGeometry = mergeGeometries(geometries)
      
      // 创建材质（使用第一个 HATCH 的颜色）
      const firstHatch = hatches[0]
      const color = firstHatch.solid_fill
        ? new THREE.Color().setRGB(
            (firstHatch.pattern?.color?.[0] || 128) / 255,
            (firstHatch.pattern?.color?.[1] || 128) / 255,
            (firstHatch.pattern?.color?.[2] || 128) / 255
          )
        : new THREE.Color(0x999999)
      
      const opacity = firstHatch.solid_fill ? 0.5 : 0.3
      const material = new THREE.MeshBasicMaterial({
        color,
        transparent: true,
        opacity,
        side: THREE.DoubleSide,
        depthWrite: false,
      })
      
      return { geometry: mergedGeometry, material }
    } catch (error) {
      console.error('[MergedHatchBatch] Error:', error)
      return {
        geometry: new THREE.BufferGeometry(),
        material: new THREE.MeshBasicMaterial({ visible: false }),
      }
    }
  }, [hatches])
  
  // ✅ S014: 清理几何体防止内存泄漏
  useEffect(() => {
    return () => {
      geometry.dispose()
      material.dispose()
    }
  }, [geometry, material])
  
  return (
    <mesh
      ref={meshRef}
      geometry={geometry}
      material={material}
      position={[0, 0, -0.5]}  // 渲染在边的下方
      renderOrder={batchIndex}
    />
  )
}

/**
 * 从 HATCH 实体创建 Shape
 */
function createHatchShape(hatch: HatchEntity): THREE.Shape | null {
  const shape = new THREE.Shape()
  const holes: THREE.Path[] = []
  
  // 处理边界路径
  hatch.boundary_paths?.forEach((path, pathIndex) => {
    const points = extractPathPoints(path)
    
    if (points.length < 3) return
    
    // 判断是外边界还是孔洞（通过面积符号）
    const area = calculateSignedArea(points)
    const isOuter = area > 0
    
    if (pathIndex === 0 || isOuter) {
      // 外边界
      moveTo(shape, points[0])
      for (let i = 1; i < points.length; i++) {
        lineTo(shape, points[i])
      }
      shape.closePath()
    } else {
      // 孔洞
      const hole = new THREE.Path()
      moveTo(hole, points[0])
      for (let i = 1; i < points.length; i++) {
        lineTo(hole, points[i])
      }
      hole.closePath()
      holes.push(hole)
    }
  })
  
  // 添加孔洞
  shape.holes = holes
  
  return shape
}

/**
 * 合并多个几何体
 * 使用 THREE.BufferGeometryUtils.mergeGeometries
 */
function mergeGeometries(geometries: THREE.BufferGeometry[]): THREE.BufferGeometry {
  return mergeBufferGeometries(geometries, false)
}

/**
 * 从边界路径提取点
 */
function extractPathPoints(path: HatchEntity['boundary_paths'][0]): [number, number][] {
  const points: [number, number][] = []
  
  if (path.type === 'polyline' && path.points) {
    return path.points
  }
  
  if (path.type === 'arc' && path.center && path.radius) {
    // 离散化圆弧
    const segments = 32
    const startAngle = path.start_angle || 0
    const endAngle = path.end_angle || Math.PI * 2
    const ccw = path.ccw ?? true
    
    for (let i = 0; i <= segments; i++) {
      const t = i / segments
      const angle = ccw
        ? startAngle + t * (endAngle - startAngle)
        : startAngle - t * (startAngle - endAngle)
      
      points.push([
        path.center[0] + path.radius * Math.cos(angle),
        path.center[1] + path.radius * Math.sin(angle),
      ])
    }
    
    return points
  }
  
  if (path.type === 'ellipse_arc' && path.center && path.major_axis) {
    // 离散化椭圆弧
    const segments = 32
    const semiMajor = Math.sqrt(path.major_axis[0] ** 2 + path.major_axis[1] ** 2)
    const semiMinor = semiMajor * (path.minor_axis_ratio ?? 1.0)
    const majorAngle = Math.atan2(path.major_axis[1], path.major_axis[0])
    
    const startAngle = path.start_angle || 0
    const endAngle = path.end_angle || Math.PI * 2
    const ccw = path.ccw ?? true
    
    for (let i = 0; i <= segments; i++) {
      const t = i / segments
      const angle = ccw
        ? startAngle + t * (endAngle - startAngle)
        : startAngle - t * (startAngle - endAngle)
      
      // 椭圆参数方程
      const x = path.center[0] + semiMajor * Math.cos(angle) * Math.cos(majorAngle) -
                                   semiMinor * Math.sin(angle) * Math.sin(majorAngle)
      const y = path.center[1] + semiMajor * Math.cos(angle) * Math.sin(majorAngle) +
                                   semiMinor * Math.sin(angle) * Math.cos(majorAngle)
      
      points.push([x, y])
    }
    
    return points
  }
  
  if (path.type === 'spline' && path.control_points) {
    // 使用控制点近似样条
    return path.control_points
  }
  
  return points
}

/**
 * 移动到新点（Shape API）
 */
function moveTo(target: THREE.Shape | THREE.Path, point: [number, number]) {
  target.moveTo(point[0], point[1])
}

/**
 * 画线到点（Shape API）
 */
function lineTo(target: THREE.Shape | THREE.Path, point: [number, number]) {
  target.lineTo(point[0], point[1])
}

/**
 * 计算多边形有符号面积
 */
function calculateSignedArea(points: [number, number][]): number {
  let area = 0
  for (let i = 0; i < points.length; i++) {
    const j = (i + 1) % points.length
    area += points[i][0] * points[j][1]
    area -= points[j][0] * points[i][1]
  }
  return area / 2
}
