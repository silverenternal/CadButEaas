/**
 * 坐标轴覆盖层组件
 * 
 * 显示 X/Y 坐标轴和原点标记，帮助用户理解场景的：
 * - 尺度（多大？）
 * - 方向（哪是上北？）
 * - 原点（坐标零点在哪？）
 */

import { useMemo } from 'react'
import { Line, Circle, Text } from 'react-konva'
import type { CameraState } from '@/types/api'

interface AxisOverlayProps {
  camera: CameraState
  showOrigin?: boolean
  canvasHeight: number  // ✅ P0-1 新增：画布高度
  canvasWidth?: number  // ✅ P0-1 新增：画布宽度（可选）
}

export function AxisOverlay({
  camera,
  showOrigin = true,
  canvasHeight,  // ✅ P0-1 修复：使用画布高度
}: AxisOverlayProps) {
  // ✅ P0-1 修复：使用画布尺寸计算坐标轴位置（而非 window.innerHeight）
  const padding = 60
  const originX = padding
  const originY = canvasHeight - padding  // ✅ 使用画布高度

  // 坐标轴长度（像素）
  const axisPixelLength = 150

  return (
    <>
      {/* X 轴（红色，指向右） */}
      <Line
        points={[
          originX,
          originY,
          originX + axisPixelLength,
          originY,
        ]}
        stroke="#ef4444"
        strokeWidth={2}
        lineCap="round"
        listening={false}
        perfectDrawEnabled={false}
      />
      {/* X 轴箭头 */}
      <Line
        points={[
          originX + axisPixelLength - 10,
          originY - 5,
          originX + axisPixelLength,
          originY,
          originX + axisPixelLength - 10,
          originY + 5,
        ]}
        stroke="#ef4444"
        strokeWidth={2}
        lineCap="round"
        lineJoin="round"
        listening={false}
        perfectDrawEnabled={false}
      />
      {/* X 轴标签 */}
      <Text
        x={originX + axisPixelLength + 10}
        y={originY - 10}
        text="X"
        fontSize={16}
        fill="#ef4444"
        fontStyle="bold"
        listening={false}
      />

      {/* Y 轴（绿色，指向上） */}
      <Line
        points={[
          originX,
          originY,
          originX,
          originY - axisPixelLength,
        ]}
        stroke="#22c55e"
        strokeWidth={2}
        lineCap="round"
        listening={false}
        perfectDrawEnabled={false}
      />
      {/* Y 轴箭头 */}
      <Line
        points={[
          originX - 5,
          originY - axisPixelLength + 10,
          originX,
          originY - axisPixelLength,
          originX + 5,
          originY - axisPixelLength + 10,
        ]}
        stroke="#22c55e"
        strokeWidth={2}
        lineCap="round"
        lineJoin="round"
        listening={false}
        perfectDrawEnabled={false}
      />
      {/* Y 轴标签 */}
      <Text
        x={originX + 10}
        y={originY - axisPixelLength - 15}
        text="Y"
        fontSize={16}
        fill="#22c55e"
        fontStyle="bold"
        listening={false}
      />

      {/* 原点标记 */}
      {showOrigin && (
        <>
          <Circle
            x={originX}
            y={originY}
            radius={5}
            fill="#3b82f6"
            listening={false}
            perfectDrawEnabled={false}
          />
          <Text
            x={originX - 15}
            y={originY + 10}
            text="(0,0)"
            fontSize={12}
            fill="#64748b"
            listening={false}
          />
        </>
      )}

      {/* 比例尺（可选） */}
      <ScaleBar camera={camera} canvasHeight={canvasHeight} />
    </>
  )
}

/**
 * 比例尺组件
 * 显示当前缩放级别下的参考长度
 */
function ScaleBar({ camera, canvasHeight }: { camera: CameraState; canvasHeight: number }) {
  // ✅ P1-1 优化：计算合适的比例尺长度（世界坐标）
  const scaleBarLength = useMemo(() => {
    // 根据 zoom 级别选择合适的长度
    const baseLength = 1000 // 基础长度（世界坐标，毫米）

    // 调整到合适的显示长度（50-200 像素）
    let multiplier = 1
    if (camera.zoom < 0.05) multiplier = 1000
    else if (camera.zoom < 0.1) multiplier = 100
    else if (camera.zoom < 0.3) multiplier = 50
    else if (camera.zoom < 0.5) multiplier = 10
    else if (camera.zoom < 1) multiplier = 5
    else if (camera.zoom < 2) multiplier = 2
    else if (camera.zoom < 5) multiplier = 1
    else if (camera.zoom < 10) multiplier = 0.5
    else multiplier = 0.2

    return baseLength * multiplier
  }, [camera.zoom])

  // ✅ P1-1 优化：计算屏幕上的长度（像素）
  const screenLength = scaleBarLength * camera.zoom

  // ✅ P1-1 优化：格式化显示文本
  const formatLength = (length: number): string => {
    if (length >= 1000000) {
      return `${(length / 1000000).toFixed(2)}km`
    } else if (length >= 1000) {
      return `${(length / 1000).toFixed(1)}m`
    } else if (length >= 1) {
      return `${length.toFixed(0)}mm`
    } else if (length >= 0.001) {
      return `${(length * 1000).toFixed(1)}μm`
    } else {
      return `${length.toFixed(6)}m`
    }
  }

  const padding = 60
  const barY = canvasHeight - padding - 40  // ✅ P0-1 修复：使用画布高度

  // ✅ P1-1 优化：限制屏幕长度在合理范围内
  const clampedScreenLength = Math.min(Math.max(screenLength, 50), 200)

  return (
    <>
      {/* 比例尺背景 */}
      <Line
        points={[
          padding,
          barY,
          padding + clampedScreenLength,
          barY,
        ]}
        stroke="#94a3b8"
        strokeWidth={4}
        lineCap="round"
        listening={false}
        perfectDrawEnabled={false}
      />
      {/* 比例尺前景 */}
      <Line
        points={[
          padding,
          barY,
          padding + clampedScreenLength,
          barY,
        ]}
        stroke="#f8fafc"
        strokeWidth={2}
        lineCap="round"
        listening={false}
        perfectDrawEnabled={false}
      />
      {/* 比例尺两端标记 */}
      <Line
        points={[
          padding,
          barY - 5,
          padding,
          barY + 5,
        ]}
        stroke="#f8fafc"
        strokeWidth={2}
        lineCap="round"
        listening={false}
        perfectDrawEnabled={false}
      />
      <Line
        points={[
          padding + clampedScreenLength,
          barY - 5,
          padding + clampedScreenLength,
          barY + 5,
        ]}
        stroke="#f8fafc"
        strokeWidth={2}
        lineCap="round"
        listening={false}
        perfectDrawEnabled={false}
      />
      {/* 比例尺文本 */}
      <Text
        x={padding + clampedScreenLength / 2}
        y={barY - 25}
        text={formatLength(scaleBarLength)}
        fontSize={12}
        fill="#64748b"
        fontStyle="bold"
        align="center"
        width={clampedScreenLength}
        offsetX={clampedScreenLength / 2}
        listening={false}
      />
    </>
  )
}
