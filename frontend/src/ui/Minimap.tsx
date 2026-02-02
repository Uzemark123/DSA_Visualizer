import { useMemo } from 'react'
import type { AppState } from '@excalidraw/excalidraw/types'

import { Panel } from './primitives'

type MinimapProps = {
  elements: Array<any>
  appState: Partial<AppState>
  files?: Record<string, { dataURL?: string }>
  width?: number
  height?: number
  padding?: number
}

type ViewRect = {
  left: number
  top: number
  width: number
  height: number
}

type MinimapState = {
  elements: Array<any>
  minX: number
  minY: number
  scale: number
  offsetX: number
  offsetY: number
  viewRect: ViewRect
}

const DEFAULT_WIDTH = 200
const DEFAULT_HEIGHT = 140
const DEFAULT_PADDING = 12

const clamp = (value: number, min: number, max: number) => Math.max(min, Math.min(max, value))

const buildMinimapState = (
  elements: Array<any>,
  appState: Partial<AppState>,
  width: number,
  height: number,
  padding: number,
): MinimapState | null => {
  const visible = elements.filter((el) => el && !el.isDeleted)
  if (!visible.length) {
    return null
  }

  let minX = Infinity
  let minY = Infinity
  let maxX = -Infinity
  let maxY = -Infinity

  for (const el of visible) {
    if (typeof el?.x !== 'number' || typeof el?.y !== 'number') {
      continue
    }
    const widthVal = typeof el.width === 'number' ? el.width : 0
    const heightVal = typeof el.height === 'number' ? el.height : 0
    const x1 = Math.min(el.x, el.x + widthVal)
    const y1 = Math.min(el.y, el.y + heightVal)
    const x2 = Math.max(el.x, el.x + widthVal)
    const y2 = Math.max(el.y, el.y + heightVal)
    minX = Math.min(minX, x1)
    minY = Math.min(minY, y1)
    maxX = Math.max(maxX, x2)
    maxY = Math.max(maxY, y2)
  }

  if (!Number.isFinite(minX) || !Number.isFinite(minY)) {
    return null
  }

  const margin = 120
  minX -= margin
  minY -= margin
  maxX += margin
  maxY += margin

  const boundsWidth = Math.max(1, maxX - minX)
  const boundsHeight = Math.max(1, maxY - minY)
  const innerWidth = width - padding * 2
  const innerHeight = height - padding * 2
  const scale = Math.min(innerWidth / boundsWidth, innerHeight / boundsHeight)
  const contentWidth = boundsWidth * scale
  const contentHeight = boundsHeight * scale
  const offsetX = padding + (innerWidth - contentWidth) / 2
  const offsetY = padding + (innerHeight - contentHeight) / 2

  const zoom = appState?.zoom?.value ?? 1
  const scrollX = appState?.scrollX ?? 0
  const scrollY = appState?.scrollY ?? 0
  const viewWidth = typeof appState.width === 'number' ? appState.width / zoom : 0
  const viewHeight = typeof appState.height === 'number' ? appState.height / zoom : 0
  const viewMinX = -scrollX / zoom
  const viewMinY = -scrollY / zoom

  if (!viewWidth || !viewHeight) {
    return null
  }

  const viewLeftRaw = (viewMinX - minX) * scale + offsetX
  const viewTopRaw = (viewMinY - minY) * scale + offsetY
  const viewWidthRaw = viewWidth * scale
  const viewHeightRaw = viewHeight * scale
  const maxLeft = offsetX + contentWidth
  const maxTop = offsetY + contentHeight
  const clampedWidth = Math.min(viewWidthRaw, contentWidth)
  const clampedHeight = Math.min(viewHeightRaw, contentHeight)
  const viewLeft = clamp(viewLeftRaw, offsetX, maxLeft - clampedWidth)
  const viewTop = clamp(viewTopRaw, offsetY, maxTop - clampedHeight)

  return {
    elements: visible,
    minX,
    minY,
    scale,
    offsetX,
    offsetY,
    viewRect: {
      left: viewLeft,
      top: viewTop,
      width: clampedWidth,
      height: clampedHeight,
    },
  }
}

const Minimap = ({
  elements,
  appState,
  files = {},
  width = DEFAULT_WIDTH,
  height = DEFAULT_HEIGHT,
  padding = DEFAULT_PADDING,
}: MinimapProps) => {
  const minimap = useMemo(
    () => buildMinimapState(elements, appState, width, height, padding),
    [elements, appState, width, height, padding],
  )

  return (
    <Panel
      className="relative rounded-2xl border border-amber-400/70 bg-panel/90 shadow-float"
      style={{ width, height }}
    >
      <div className="absolute inset-3 rounded-xl border border-amber-400/20 bg-ink/50" />
      <div className="absolute left-3 top-3 text-[10px] uppercase tracking-[0.2em] text-amber-200/80">
        Minimap
      </div>
      {minimap && (
        <svg className="absolute inset-0 pointer-events-none" width={width} height={height}>
          {minimap.elements.map((el: any) => {
            const toMiniX = (x: number) => (x - minimap.minX) * minimap.scale + minimap.offsetX
            const toMiniY = (y: number) => (y - minimap.minY) * minimap.scale + minimap.offsetY
            const stroke = 'rgba(255,255,255,0.7)'
            const fill = 'rgba(255,255,255,0.12)'

            if (el.type === 'rectangle') {
              const widthVal = Math.abs(el.width ?? 0)
              const heightVal = Math.abs(el.height ?? 0)
              const x1 = Math.min(el.x, el.x + (el.width ?? 0))
              const y1 = Math.min(el.y, el.y + (el.height ?? 0))
              return (
                <rect
                  key={el.id}
                  x={toMiniX(x1)}
                  y={toMiniY(y1)}
                  width={widthVal * minimap.scale}
                  height={heightVal * minimap.scale}
                  rx={Math.max(1, 4 * minimap.scale)}
                  fill={fill}
                  stroke={stroke}
                  strokeWidth={1}
                />
              )
            }

            if (el.type === 'ellipse') {
              const widthVal = Math.abs(el.width ?? 0)
              const heightVal = Math.abs(el.height ?? 0)
              const x1 = Math.min(el.x, el.x + (el.width ?? 0))
              const y1 = Math.min(el.y, el.y + (el.height ?? 0))
              return (
                <ellipse
                  key={el.id}
                  cx={toMiniX(x1 + widthVal / 2)}
                  cy={toMiniY(y1 + heightVal / 2)}
                  rx={(widthVal / 2) * minimap.scale}
                  ry={(heightVal / 2) * minimap.scale}
                  fill={fill}
                  stroke={stroke}
                  strokeWidth={1}
                />
              )
            }

            if (el.type === 'diamond') {
              const widthVal = Math.abs(el.width ?? 0)
              const heightVal = Math.abs(el.height ?? 0)
              const x1 = Math.min(el.x, el.x + (el.width ?? 0))
              const y1 = Math.min(el.y, el.y + (el.height ?? 0))
              const points = [
                [x1 + widthVal / 2, y1],
                [x1 + widthVal, y1 + heightVal / 2],
                [x1 + widthVal / 2, y1 + heightVal],
                [x1, y1 + heightVal / 2],
              ]
                .map(([x, y]) => `${toMiniX(x)},${toMiniY(y)}`)
                .join(' ')
              return <polygon key={el.id} points={points} fill={fill} stroke={stroke} strokeWidth={1} />
            }

            if (el.type === 'image') {
              const widthVal = Math.abs(el.width ?? 0)
              const heightVal = Math.abs(el.height ?? 0)
              const x1 = Math.min(el.x, el.x + (el.width ?? 0))
              const y1 = Math.min(el.y, el.y + (el.height ?? 0))
              const fileDataUrl = typeof el.fileId === 'string' ? files[el.fileId]?.dataURL : undefined
              if (fileDataUrl) {
                return (
                  <image
                    key={el.id}
                    href={fileDataUrl}
                    x={toMiniX(x1)}
                    y={toMiniY(y1)}
                    width={widthVal * minimap.scale}
                    height={heightVal * minimap.scale}
                    opacity={0.65}
                    preserveAspectRatio="xMidYMid meet"
                  />
                )
              }
              return (
                <rect
                  key={el.id}
                  x={toMiniX(x1)}
                  y={toMiniY(y1)}
                  width={widthVal * minimap.scale}
                  height={heightVal * minimap.scale}
                  rx={Math.max(1, 4 * minimap.scale)}
                  fill="rgba(255,255,255,0.06)"
                  stroke="rgba(255,255,255,0.5)"
                  strokeWidth={1}
                />
              )
            }

            if (el.type === 'text' && !el.containerId) {
              const fontSizeRaw = typeof el.fontSize === 'number' ? el.fontSize : 12
              const fontSize = Math.max(6, Math.min(20, fontSizeRaw * minimap.scale))
              const x = toMiniX(el.x)
              const y = toMiniY(el.y)
              const textValue = typeof el.text === 'string' ? el.text : ''
              if (!/^start here!?$/i.test(textValue.trim()) && !/^dsa visual canvas$/i.test(textValue.trim())) {
                return null
              }
              const lines = textValue.split('\n')
              return (
                <text
                  key={el.id}
                  x={x}
                  y={y}
                  fill="rgba(255,255,255,0.8)"
                  fontSize={fontSize}
                  fontFamily="sans-serif"
                  dominantBaseline="hanging"
                >
                  {lines.map((line: string, index: number) => (
                    <tspan key={`${el.id}-line-${index}`} x={x} dy={index === 0 ? 0 : fontSize * 1.2}>
                      {line}
                    </tspan>
                  ))}
                </text>
              )
            }

            if (el.type === 'line' || el.type === 'arrow' || el.type === 'freedraw') {
              const points = Array.isArray(el.points) ? el.points : []
              if (!points.length) return null
              const d = points
                .map((pt: [number, number], idx: number) => {
                  const px = toMiniX(el.x + pt[0])
                  const py = toMiniY(el.y + pt[1])
                  return `${idx === 0 ? 'M' : 'L'}${px} ${py}`
                })
                .join(' ')
              return <path key={el.id} d={d} fill="none" stroke={stroke} strokeWidth={1} />
            }

            return null
          })}
        </svg>
      )}
      {minimap?.viewRect && (
        <div
          className="absolute rounded-md border border-amber-300/80 bg-amber-400/10"
          style={{
            left: minimap.viewRect.left,
            top: minimap.viewRect.top,
            width: minimap.viewRect.width,
            height: minimap.viewRect.height,
          }}
        />
      )}
    </Panel>
  )
}

export default Minimap
