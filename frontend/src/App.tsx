import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { generateNKeysBetween } from 'fractional-indexing'
import { Excalidraw } from '@excalidraw/excalidraw'
import type { AppState, ExcalidrawImperativeAPI, ExcalidrawInitialDataState } from '@excalidraw/excalidraw/types'
import '@excalidraw/excalidraw/index.css'
import { AppShell, Button, InteractionModeIndicator, LandingOverlay, Panel, RightToolbar, TopBar } from './ui/primitives'
import Minimap from './ui/Minimap'
import { buildGenerateRoadmapRequest, collectPatternsForRequest } from './roadmap/collectPatterns'
import { parseNodeLink } from './roadmap/link_parsing'
import type { PatternNode } from './roadmap/types'
import greenPillIcon from './canvas/images/legend_icons/green_pill.svg'
import bluePillIcon from './canvas/images/legend_icons/blue_pill.svg'
import yellowPillIcon from './canvas/images/legend_icons/yellow_pill.svg'
import purplePillIcon from './canvas/images/legend_icons/purple_pill.svg'
import connectionFrameIcon from './canvas/images/legend_icons/connection_frame.svg'

const ExcalidrawComponent = Excalidraw as any

const initialData: ExcalidrawInitialDataState = {
  elements: [],
  appState: {
    viewBackgroundColor: '#2a2d34',
    currentItemStrokeColor: '#F59E0B',
    currentItemBackgroundColor: '#FB923C',
    currentItemFillStyle: 'solid',
    currentItemStrokeWidth: 2,
  },
}

type UISummary = {
  elementCount: number
  selectionCount: number
  panZoomLabel: string
}

const MINIMAP_WIDTH = 250
const MINIMAP_HEIGHT = 165
const MINIMAP_PADDING = 12
const PROGRESS_STORAGE_KEY = 'dsa_visualizer_progress_v1'
const HAS_SEEN_HELP_KEY = 'hasSeenHelp'
const HAS_VISITED_CANVAS_KEY = 'hasVisitedCanvas'
const VIEW_STATE_STORAGE_KEY = 'dsa_visualizer_view_v1'
const FIRST_VISIT_ZOOM = 0.025

type ProgressState = 'locked' | 'in_progress' | 'completed'
type ProgressMap = Record<string, ProgressState>
type RoadmapDifficulty = 'easy' | 'medium' | 'hard'

type RoadmapProblem = {
  id: string
  title: string
  difficulty?: RoadmapDifficulty
  patternId?: number
  patternLabel?: string
  canvasNodeId?: string
  url?: string
}

type RoadmapFilters = {
  selectionMode: 'in_progress' | 'in_progress_and_completed' | 'custom'
  difficulty: { easy: boolean; medium: boolean; hard: boolean }
  count: number
  customPatternIds: number[]
  priorityPatternIds: number[]
  excludedPatternIds: number[]
}

type RoadmapEntry = {
  id: string
  title: string
  generatedAt: string
  summary: string
  filtersSummary: string
  filters: RoadmapFilters
  problems: RoadmapProblem[]
}

const isNodeElement = (el: any) => ['rectangle', 'ellipse', 'diamond'].includes(el?.type)

type GeometrySnapshot = {
  x: number
  y: number
  width: number
  height: number
  angle: number
  points: Array<[number, number]> | null
}

const normalizeProgressState = (value: unknown): ProgressState | null => {
  if (value === 'locked' || value === 'in_progress' || value === 'completed') {
    return value
  }
  return null
}

const formatRoadmapTimestamp = (date: Date) =>
  date.toLocaleString(undefined, { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' })

const formatDifficultyLabel = (difficulty: RoadmapFilters['difficulty']) => {
  const picks: RoadmapDifficulty[] = []
  if (difficulty.easy) picks.push('easy')
  if (difficulty.medium) picks.push('medium')
  if (difficulty.hard) picks.push('hard')
  if (picks.length === 0) return 'Any'
  return picks.map((item) => item[0].toUpperCase() + item.slice(1)).join('+')
}

const LIGHT_SVG_COLOR = '#ffffff'

const tintSvgMarkup = (svg: string, color: string) => {
  const attrRegex =
    /\b(fill|stroke)=\"(#000000|#000|#1e1e1e|black|rgb\(\s*0\s*,\s*0\s*,\s*0\s*\)|rgba\(\s*0\s*,\s*0\s*,\s*0\s*,\s*1\s*\))\"/gi
  const styleRegex =
    /\b(fill|stroke):\s*(#000000|#000|#1e1e1e|black|rgb\(\s*0\s*,\s*0\s*,\s*0\s*\)|rgba\(\s*0\s*,\s*0\s*,\s*0\s*,\s*1\s*\))/gi
  return svg.replace(attrRegex, (_match, prop) => `${prop}="${color}"`).replace(styleRegex, (_match, prop) => `${prop}:${color}`)
}

const decodeSvgDataUrl = (dataURL: string) => {
  const commaIndex = dataURL.indexOf(',')
  if (commaIndex === -1) return null
  const prefix = dataURL.slice(0, commaIndex)
  const data = dataURL.slice(commaIndex + 1)
  const isBase64 = prefix.includes(';base64')
  if (!prefix.startsWith('data:image/svg+xml')) return null
  try {
    if (isBase64) {
      const decoded = globalThis.atob(data)
      return { prefix, svg: decodeURIComponent(escape(decoded)), isBase64: true }
    }
    return { prefix, svg: decodeURIComponent(data), isBase64: false }
  } catch {
    return { prefix, svg: isBase64 ? globalThis.atob(data) : data, isBase64 }
  }
}

const encodeSvgDataUrl = (prefix: string, svg: string, isBase64: boolean) => {
  if (isBase64) {
    const encoded = globalThis.btoa(unescape(encodeURIComponent(svg)))
    return `${prefix},${encoded}`
  }
  return `${prefix},${encodeURIComponent(svg)}`
}

const tintSvgFiles = (files: Record<string, any>) => {
  let changed = false
  const nextFiles: Record<string, any> = { ...files }
  for (const [fileId, file] of Object.entries(files)) {
    const dataURL = file?.dataURL
    const mimeType = file?.mimeType
    if (typeof dataURL !== 'string') continue
    if (mimeType && mimeType !== 'image/svg+xml' && !dataURL.startsWith('data:image/svg+xml')) {
      continue
    }
    const decoded = decodeSvgDataUrl(dataURL)
    if (!decoded) continue
    const tintedSvg = tintSvgMarkup(decoded.svg, LIGHT_SVG_COLOR)
    const nextDataUrl = encodeSvgDataUrl(decoded.prefix, tintedSvg, decoded.isBase64)
    if (nextDataUrl !== dataURL) {
      nextFiles[fileId] = { ...file, dataURL: nextDataUrl }
      changed = true
    }
  }
  return changed ? nextFiles : files
}


const clonePoints = (points: Array<[number, number]> | null): Array<[number, number]> | null =>
  points ? points.map((point) => [point[0], point[1]]) : null

const clampNumber = (value: number, min: number, max: number) => Math.min(max, Math.max(min, value))

const hasSeenHelp = () => {
  try {
    return localStorage.getItem(HAS_SEEN_HELP_KEY) === 'true'
  } catch {
    return false
  }
}

const hasVisitedCanvas = () => {
  try {
    return localStorage.getItem(HAS_VISITED_CANVAS_KEY) === 'true'
  } catch {
    return false
  }
}

const markVisitedCanvas = () => {
  try {
    localStorage.setItem(HAS_VISITED_CANVAS_KEY, 'true')
  } catch {
    // Ignore storage failures (e.g., private mode).
  }
}

const loadStoredViewState = () => {
  try {
    const raw = localStorage.getItem(VIEW_STATE_STORAGE_KEY)
    if (!raw) return null
    const parsed = JSON.parse(raw)
    const scrollX = Number(parsed?.scrollX)
    const scrollY = Number(parsed?.scrollY)
    const zoom = Number(parsed?.zoom)
    if (!Number.isFinite(scrollX) || !Number.isFinite(scrollY) || !Number.isFinite(zoom)) {
      return null
    }
    return { scrollX, scrollY, zoom }
  } catch {
    return null
  }
}

const storeViewState = (state: { scrollX: number; scrollY: number; zoom: number }) => {
  try {
    localStorage.setItem(VIEW_STATE_STORAGE_KEY, JSON.stringify(state))
  } catch {
    // Ignore storage failures (e.g., private mode).
  }
}

const isSamePoints = (a: Array<[number, number]> | null, b: Array<[number, number]> | null) => {
  if (!a && !b) return true
  if (!a || !b || a.length !== b.length) return false
  for (let index = 0; index < a.length; index += 1) {
    if (a[index][0] !== b[index][0] || a[index][1] !== b[index][1]) {
      return false
    }
  }
  return true
}

const isSameGeometry = (snapshot: GeometrySnapshot | undefined, element: any) => {
  if (!snapshot || !element) return false
  if (
    snapshot.x !== element.x ||
    snapshot.y !== element.y ||
    snapshot.width !== element.width ||
    snapshot.height !== element.height ||
    snapshot.angle !== element.angle
  ) {
    return false
  }
  const elementPoints = Array.isArray(element.points)
    ? element.points.map((point: [number, number]) => [point[0], point[1]]) as Array<[number, number]>
    : null
  return isSamePoints(snapshot.points, elementPoints)
}

const getTextContent = (element: any) => {
  const raw = typeof element?.rawText === 'string' ? element.rawText : element?.text
  return typeof raw === 'string' ? raw.trim() : ''
}

const getAnchorBounds = (elements: Array<any>) => {
  const anchorElement = elements.find((el) => el?.customData?.dsaAnchor === true)
  if (!anchorElement) return null

  const width = Number.isFinite(anchorElement?.width) ? anchorElement.width : 0
  const height = Number.isFinite(anchorElement?.height) ? anchorElement.height : 0
  const x = Number.isFinite(anchorElement?.x) ? anchorElement.x : 0
  const y = Number.isFinite(anchorElement?.y) ? anchorElement.y : 0
  return { x, y, width, height }
}

const buildGeometrySnapshot = (elements: Array<any>) => {
  const ids = new Set<string>()
  const geometryById: Record<string, GeometrySnapshot> = {}

  for (const el of elements) {
    if (!el || el.isDeleted) continue
    ids.add(el.id)
    geometryById[el.id] = {
      x: el.x,
      y: el.y,
      width: el.width,
      height: el.height,
      angle: el.angle,
      points: Array.isArray(el.points)
        ? (el.points.map((point: [number, number]) => [point[0], point[1]]) as Array<[number, number]>)
        : null,
    }
  }

  return { ids, geometryById }
}

const sanitizeElements = (elements: Array<any>) => {
  const filtered = elements.filter((el) => el && !el.isDeleted)
  const byId = new Map<string, any>()
  for (const el of filtered) {
    byId.set(el.id, { ...el })
  }

  const cloned = Array.from(byId.values())

  // Drop invalid text container references.
  for (const el of cloned) {
    if (el?.type === 'text' && el.containerId && !byId.has(el.containerId)) {
      el.containerId = null
    }
  }

  const byIdAfter = new Map(cloned.map((el) => [el.id, el]))
  const textToContainer = new Map<string, string>()
  for (const el of cloned) {
    if (el?.type === 'text' && el.containerId) {
      textToContainer.set(el.id, el.containerId)
    }
  }

  // Clean boundElements to only include valid references.
  for (const el of cloned) {
    if (!Array.isArray(el?.boundElements) || el.boundElements.length === 0) continue
    const nextBound = el.boundElements.filter((ref: any) => {
      const target = byIdAfter.get(ref?.id)
      if (!target) return false
      if (ref?.type === 'text') {
        const containerId = textToContainer.get(ref.id)
        return !containerId || containerId === el.id
      }
      return true
    })
    el.boundElements = nextBound
  }

  // Ensure containers list their bound text.
  for (const [textId, containerId] of textToContainer) {
    const container = byIdAfter.get(containerId)
    if (!container) continue
    const bound = Array.isArray(container.boundElements) ? container.boundElements : []
    const hasText = bound.some((ref: any) => ref?.id === textId && ref?.type === 'text')
    if (!hasText) {
      container.boundElements = [...bound, { id: textId, type: 'text' }]
    }
  }

  // Fix fractional index ordering for bound text (text must come after container).
  const ordered = [...cloned].sort((a, b) => {
    if (a.index < b.index) return -1
    if (a.index > b.index) return 1
    return a.id < b.id ? -1 : a.id > b.id ? 1 : 0
  })

  const orderedIds = new Set(ordered.map((el) => el.id))
  for (const [textId, containerId] of textToContainer) {
    if (!orderedIds.has(textId) || !orderedIds.has(containerId)) continue
    const container = byIdAfter.get(containerId)
    if (!container) continue
    const boundTexts = cloned.filter((el) => el?.type === 'text' && el.containerId === containerId)
    if (boundTexts.length === 0) continue
    const needsReindex = boundTexts.some((text) => text.index <= container.index)
    if (!needsReindex) continue

    const boundTextIds = new Set(boundTexts.map((text) => text.id))
    let successorIndex: string | null = null
    for (const el of ordered) {
      if (el.id === containerId) continue
      if (boundTextIds.has(el.id)) continue
      if (el.index > container.index) {
        successorIndex = el.index
        break
      }
    }

    const keys = generateNKeysBetween(container.index, successorIndex, boundTexts.length)
    const sortedBoundTexts = [...boundTexts].sort((a, b) => {
      if (a.index < b.index) return -1
      if (a.index > b.index) return 1
      return a.id < b.id ? -1 : a.id > b.id ? 1 : 0
    })
    sortedBoundTexts.forEach((text, idx) => {
      text.index = keys[idx]
    })
  }

  return cloned
}

const toRgba = (color: string | undefined, alpha: number, fallback = '#F59E0B') => {
  const normalized = (color ?? '').trim().toLowerCase()
  if (!normalized || normalized === 'transparent') {
    return toRgba(fallback, alpha, fallback)
  }
  if (normalized.startsWith('rgba')) {
    const match = normalized.match(/rgba\(([^)]+)\)/)
    if (match) {
      const parts = match[1].split(',').map((part) => Number(part.trim()))
      if (parts.length >= 3) {
        return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${alpha})`
      }
    }
  }
  if (normalized.startsWith('rgb')) {
    const match = normalized.match(/rgb\(([^)]+)\)/)
    if (match) {
      const parts = match[1].split(',').map((part) => Number(part.trim()))
      if (parts.length >= 3) {
        return `rgba(${parts[0]}, ${parts[1]}, ${parts[2]}, ${alpha})`
      }
    }
  }
  if (normalized.startsWith('#')) {
    const hex = normalized.slice(1)
    if (hex.length === 3) {
      const r = parseInt(hex[0] + hex[0], 16)
      const g = parseInt(hex[1] + hex[1], 16)
      const b = parseInt(hex[2] + hex[2], 16)
      return `rgba(${r}, ${g}, ${b}, ${alpha})`
    }
    if (hex.length === 6) {
      const r = parseInt(hex.slice(0, 2), 16)
      const g = parseInt(hex.slice(2, 4), 16)
      const b = parseInt(hex.slice(4, 6), 16)
      return `rgba(${r}, ${g}, ${b}, ${alpha})`
    }
  }
  return toRgba(fallback, alpha, fallback)
}

const getShapeStyle = (element: any, width: number, height: number) => {
  if (element?.type === 'ellipse') {
    return {
      borderRadius: '9999px',
    }
  }
  if (element?.type === 'diamond') {
    return {
      clipPath: 'polygon(50% 0%, 100% 50%, 50% 100%, 0% 50%)',
    }
  }
  const baseRadius = Math.max(6, Math.min(width, height) * 0.08)
  return {
    borderRadius: `${Math.round(baseRadius)}px`,
  }
}

const getPatternIdFromElement = (element: any): number | null => {
  const raw = element?.customData?.patternId
  if (typeof raw === 'number' && Number.isFinite(raw)) {
    return raw
  }
  if (typeof raw === 'string') {
    const parsed = Number(raw)
    if (Number.isFinite(parsed)) {
      return parsed
    }
  }
  return parseNodeLink(element?.link).patternId
}

const buildOwnershipIndex = (elements: Array<any>) => {
  const elementMap = new Map<string, any>()
  const excludedElementIds = new Set<string>()
  const patternElementIds: string[] = []
  const ownerByElementId = new Map<string, string>()

  for (const el of elements) {
    if (!el || el.isDeleted) continue
    elementMap.set(el.id, el)
    const patternId = getPatternIdFromElement(el)
    if (patternId === 0) {
      excludedElementIds.add(el.id)
    } else if (typeof patternId === 'number' && patternId > 0) {
      patternElementIds.push(el.id)
    }
  }

  const adjacency = new Map<string, Set<string>>()
  for (const id of elementMap.keys()) {
    if (!excludedElementIds.has(id)) {
      adjacency.set(id, new Set())
    }
  }

  for (const el of elements) {
    if (!el || el.isDeleted || el.type !== 'arrow') continue
    const startId = el.startBinding?.elementId
    const endId = el.endBinding?.elementId
    if (!startId || !endId) continue
    if (!elementMap.has(startId) || !elementMap.has(endId)) continue
    if (excludedElementIds.has(startId) || excludedElementIds.has(endId)) continue
    adjacency.get(startId)?.add(endId)
    adjacency.get(endId)?.add(startId)
  }

  const ownershipMap = new Map<string, Set<string>>()
  for (const patternElementId of patternElementIds) {
    if (!ownerByElementId.has(patternElementId)) {
      ownerByElementId.set(patternElementId, patternElementId)
    }
    const owned = new Set<string>()
    const stack: string[] = [patternElementId]
    const visited = new Set<string>([patternElementId])
    while (stack.length > 0) {
      const current = stack.pop()!
      const neighbors = adjacency.get(current)
      if (!neighbors) continue
      for (const neighbor of neighbors) {
        if (visited.has(neighbor)) continue
        visited.add(neighbor)
        stack.push(neighbor)
        if (neighbor !== patternElementId) {
          owned.add(neighbor)
          if (!ownerByElementId.has(neighbor)) {
            ownerByElementId.set(neighbor, patternElementId)
          }
        }
      }
    }
    ownershipMap.set(patternElementId, owned)
  }

  return {
    ownershipMap,
    excludedElementIds,
    patternElementIds,
    ownerByElementId,
  }
}

const applyArrowDimming = (
  elements: Array<any>,
  ownershipIndex: ReturnType<typeof buildOwnershipIndex>,
  progressById: ProgressMap,
  arrowBaseOpacity: Record<string, number>,
) => {
  const elementMap = new Map<string, any>()
  for (const el of elements) {
    if (el && !el.isDeleted) {
      elementMap.set(el.id, el)
    }
  }

  let changed = false
  const nextElements = elements.map((el) => {
    if (!el || el.isDeleted || el.type !== 'arrow') return el
    const baseOpacity =
      arrowBaseOpacity[el.id] ??
      (typeof el.opacity === 'number' ? el.opacity : 100)
    if (!(el.id in arrowBaseOpacity)) {
      arrowBaseOpacity[el.id] = baseOpacity
    }
    const startId = el.startBinding?.elementId
    const endId = el.endBinding?.elementId
    const ownerByElementId = ownershipIndex.ownerByElementId

    const getOwnerProgress = (elementId?: string) => {
      if (!elementId) return null
      const ownerId = ownerByElementId.get(elementId)
      if (!ownerId) return null
      return (
        normalizeProgressState(progressById[ownerId]) ??
        normalizeProgressState(elementMap.get(ownerId)?.customData?.progressState)
      )
    }

    const shouldDim =
      getOwnerProgress(startId) === 'locked' || getOwnerProgress(endId) === 'locked'
    const dimFactor = 0.4
    const downstreamDimFactor = 0.7
    const nextOpacity = shouldDim
      ? Math.max(10, Math.round(baseOpacity * dimFactor * downstreamDimFactor))
      : baseOpacity

    if (el.opacity !== nextOpacity) {
      changed = true
      return {
        ...el,
        opacity: nextOpacity,
      }
    }
    return el
  })

  return { nextElements, changed }
}

const loadProgressMap = (): ProgressMap => {
  try {
    const raw = localStorage.getItem(PROGRESS_STORAGE_KEY)
    if (!raw) return {}
    const parsed = JSON.parse(raw)
    if (parsed && typeof parsed === 'object') {
      const normalized: ProgressMap = {}
      for (const [key, value] of Object.entries(parsed)) {
        const progress = normalizeProgressState(value)
        if (progress) {
          normalized[key] = progress
        }
      }
      return normalized
    }
  } catch {
    return {}
  }
  return {}
}

export default function App() {
  const excalidrawAPI = useRef<ExcalidrawImperativeAPI | null>(null)
  const elementsRef = useRef<readonly unknown[]>([])
  const appStateRef = useRef<Partial<AppState>>({})
  const [showLanding, setShowLanding] = useState(() => !hasSeenHelp())
  const [showLegend, setShowLegend] = useState(false)
  const [showRoadmapData, setShowRoadmapData] = useState(false)
  const [topBarSelection, setTopBarSelection] = useState<'login' | 'help' | 'legend' | 'roadmap' | null>(() =>
    hasSeenHelp() ? null : 'help',
  )
  const [initialScene, setInitialScene] = useState<ExcalidrawInitialDataState | null>(null)
  const [sceneKey, setSceneKey] = useState(0)
  const [minimapElements, setMinimapElements] = useState<Array<any>>([])
  const [minimapAppState, setMinimapAppState] = useState<Partial<AppState>>({})
  const [minimapFiles, setMinimapFiles] = useState<Record<string, any>>({})
  const [interactionMode, setInteractionMode] = useState<'select' | 'pan'>('select')
  const interactionModeRef = useRef<'select' | 'pan'>('select')
  const [progressById, setProgressById] = useState<ProgressMap>({})
  const [roadmaps, setRoadmaps] = useState<RoadmapEntry[]>([])
  const [selectedRoadmapId, setSelectedRoadmapId] = useState<string | null>(null)
  const [selectedPatternElementId, setSelectedPatternElementId] = useState<string | null>(null)
  const lastViewPersistRef = useRef(0)
  const allowViewPersistRef = useRef(false)
  const [contextMenu, setContextMenu] = useState<{ open: boolean; x: number; y: number; nodeId: string | null }>({
    open: false,
    x: 0,
    y: 0,
    nodeId: null,
  })
  const minimapSigRef = useRef('')
  const baseElementsRef = useRef<Array<any>>([])
  const baseIdSetRef = useRef<Set<string>>(new Set())
  const baseGeometryRef = useRef<Record<string, GeometrySnapshot>>({})
  const ownershipIndexRef = useRef<ReturnType<typeof buildOwnershipIndex> | null>(null)
  const arrowBaseOpacityRef = useRef<Record<string, number>>({})
  const roadmapCountRef = useRef(0)
  const isRestoringRef = useRef(false)
  const isClearingEditRef = useRef(false)
  const isForcingToolRef = useRef(false)
  const contextMenuRef = useRef<HTMLDivElement | null>(null)
  const handleCloseLanding = useCallback(() => {
    try {
      localStorage.setItem(HAS_SEEN_HELP_KEY, 'true')
    } catch {
      // best-effort persistence
    }
    setShowLanding(false)
  }, [])

  const getNodeIdFromElement = useCallback((elementId: string, elements: readonly unknown[]) => {
    const element = (elements as Array<any>).find((el) => el?.id === elementId)
    if (!element) return null
    if (isNodeElement(element)) return element.id
    if (element.type === 'text' && element.containerId) {
      const container = (elements as Array<any>).find((el) => el?.id === element.containerId)
      return container && isNodeElement(container) ? container.id : null
    }
    return null
  }, [])

  const resolvePatternElementId = useCallback(
    (nodeElementId: string | null, elements: readonly unknown[]) => {
      if (!nodeElementId) return null
      const element = (elements as Array<any>).find((el) => el?.id === nodeElementId)
      if (!element) return null
      const patternId = getPatternIdFromElement(element)
      if (typeof patternId === 'number' && patternId > 0) {
        return element.id
      }
      const ownershipIndex = ownershipIndexRef.current
      const owner = ownershipIndex?.ownerByElementId.get(nodeElementId)
      return owner ?? null
    },
    [],
  )

  const canTransition = useCallback((from: ProgressState, to: ProgressState) => {
    if (from === to) return false
    if (from === 'locked' && to === 'in_progress') return true
    if (from === 'in_progress' && to === 'completed') return true
    if (from === 'in_progress' && to === 'locked') return true
    if (from === 'completed' && to === 'in_progress') return true
    return false
  }, [])
  const summaryRef = useRef<UISummary>({
    elementCount: 0,
    selectionCount: 0,
    panZoomLabel: 'zoom 100% - pan (0, 0)',
  })
  const [uiVersion, bumpUiVersion] = useState(0)
  const computeSummary = useCallback((elements: readonly unknown[], appState: AppState): UISummary => {
    const selectionCount = Object.keys(appState.selectedElementIds || {}).length
    const zoom = appState.zoom?.value ?? 1
    const x = Math.round(appState.scrollX ?? 0)
    const y = Math.round(appState.scrollY ?? 0)
    return {
      elementCount: elements.length,
      selectionCount,
      panZoomLabel: `zoom ${Math.round(zoom * 100)}% - pan (${x}, ${y})`,
    }
  }, [])

  const handleCanvasChange = useCallback(
    (elements: readonly unknown[], appState: AppState) => {
      if (!isRestoringRef.current) {
        const baseIds = baseIdSetRef.current
        if (baseIds.size > 0) {
          const { ids: currentIds, geometryById: currentGeometry } = buildGeometrySnapshot(elements as Array<any>)
          const missingIds = [...baseIds].filter((id) => !currentIds.has(id))
          const extraIds = [...currentIds].filter((id) => !baseIds.has(id))

          let geometryMismatch = false
          if (missingIds.length === 0 && extraIds.length === 0) {
            for (const id of baseIds) {
              const currentSnapshot = currentGeometry[id]
              const baseSnapshot = baseGeometryRef.current[id]
              if (
                !currentSnapshot ||
                !baseSnapshot ||
                !isSamePoints(baseSnapshot.points, currentSnapshot.points) ||
                baseSnapshot.x !== currentSnapshot.x ||
                baseSnapshot.y !== currentSnapshot.y ||
                baseSnapshot.width !== currentSnapshot.width ||
                baseSnapshot.height !== currentSnapshot.height ||
                baseSnapshot.angle !== currentSnapshot.angle
              ) {
                geometryMismatch = true
                break
              }
            }
          }

          if ((missingIds.length > 0 || extraIds.length > 0 || geometryMismatch) && excalidrawAPI.current) {
            isRestoringRef.current = true
            const sanitizedElements = (elements as Array<any>)
              .filter((el) => el && !el.isDeleted && baseIds.has(el.id))
              .map((el) => {
                const baseGeometry = baseGeometryRef.current[el.id]
                if (!baseGeometry || isSameGeometry(baseGeometry, el)) {
                  return el
                }
                return {
                  ...el,
                  x: baseGeometry.x,
                  y: baseGeometry.y,
                  width: baseGeometry.width,
                  height: baseGeometry.height,
                  angle: baseGeometry.angle,
                  points: clonePoints(baseGeometry.points),
                }
              })

            if (missingIds.length > 0) {
              for (const missingId of missingIds) {
                const baseElement = baseElementsRef.current.find((el) => el?.id === missingId)
                if (baseElement) {
                  sanitizedElements.push(baseElement)
                }
              }
            }

            excalidrawAPI.current.updateScene({
              elements: sanitizedElements,
              appState: {
                selectedElementIds: appState.selectedElementIds,
                editingTextElement: null,
                newElement: null,
                resizingElement: null,
                selectedElementsAreBeingDragged: false,
                isRotating: false,
                selectionElement: null,
                multiElement: null,
                selectedLinearElement: null,
                editingLinearElement: null,
              },
            })
            requestAnimationFrame(() => {
              isRestoringRef.current = false
            })
          }
        }
      }

      if (
        !isForcingToolRef.current &&
        appState.activeTool?.type &&
        appState.activeTool.type !== 'selection' &&
        appState.activeTool.type !== 'hand' &&
        excalidrawAPI.current
      ) {
        isForcingToolRef.current = true
        excalidrawAPI.current.setActiveTool({ type: 'selection', locked: false })
        requestAnimationFrame(() => {
          isForcingToolRef.current = false
        })
      }

      if (!isClearingEditRef.current && appState.editingTextElement && excalidrawAPI.current) {
        isClearingEditRef.current = true
        excalidrawAPI.current.updateScene({
          appState: {
            editingTextElement: null,
          },
        })
        requestAnimationFrame(() => {
          isClearingEditRef.current = false
        })
      }
      elementsRef.current = elements
      appStateRef.current = {
        ...appState,
        width: appState.width,
        height: appState.height,
      }
      const nextMode = appState.activeTool?.type === 'hand' ? 'pan' : 'select'
      if (nextMode !== interactionModeRef.current) {
        interactionModeRef.current = nextMode
        setInteractionMode(nextMode)
      }
      const zoom = appState.zoom?.value ?? 1
      const firstVersion = elements.length ? (elements[0] as any)?.version ?? 0 : 0
      const lastVersion = elements.length ? (elements[elements.length - 1] as any)?.version ?? 0 : 0
      const minimapSig = [
        elements.length,
        firstVersion,
        lastVersion,
        appState.scrollX,
        appState.scrollY,
        zoom,
        appState.width,
        appState.height,
      ].join('|')

      if (minimapSig !== minimapSigRef.current) {
        minimapSigRef.current = minimapSig
        setMinimapElements(elements as Array<any>)
        setMinimapAppState({
          ...appState,
          width: appState.width,
          height: appState.height,
        })
      }
      const zoomValue =
        typeof appState.zoom === 'number'
          ? appState.zoom
          : typeof appState.zoom?.value === 'number'
            ? appState.zoom.value
            : null
      if (
        allowViewPersistRef.current &&
        Number.isFinite(appState.scrollX) &&
        Number.isFinite(appState.scrollY) &&
        Number.isFinite(zoomValue)
      ) {
        const now = Date.now()
        if (now - lastViewPersistRef.current > 300) {
          lastViewPersistRef.current = now
          storeViewState({
            scrollX: appState.scrollX as number,
            scrollY: appState.scrollY as number,
            zoom: zoomValue as number,
          })
        }
      }
      const nextSummary = computeSummary(elements, appState)
      const prev = summaryRef.current
      const changed =
        nextSummary.elementCount !== prev.elementCount ||
        nextSummary.selectionCount !== prev.selectionCount ||
        nextSummary.panZoomLabel !== prev.panZoomLabel
      if (changed) {
        summaryRef.current = nextSummary
        bumpUiVersion((v) => v + 1)
      }

      const selectedIds = Object.keys(appState.selectedElementIds || {})
      const nextNodeElementId =
        selectedIds.length > 0
          ? selectedIds
              .map((id) => getNodeIdFromElement(id, elements))
              .find((id) => id !== null) ?? null
          : null
      const nextPatternElementId = resolvePatternElementId(nextNodeElementId, elements)

      if (nextPatternElementId !== selectedPatternElementId) {
        setSelectedPatternElementId(nextPatternElementId)
        setContextMenu((prev) => (prev.open ? { ...prev, open: false, nodeId: null } : prev))
      }
    },
    [computeSummary, getNodeIdFromElement, resolvePatternElementId, selectedPatternElementId],
  )

  useEffect(() => {
    let cancelled = false

    fetch('/roadmap.excalidraw')
      .then((res) => res.json())
      .then((data) => {
        if (cancelled) return
        const storedProgress = loadProgressMap()
        const hydratedProgress: ProgressMap = { ...storedProgress }
        const rawElements = data?.elements ?? []
        const elementTypes = new Map<string, string>()
        for (const raw of rawElements) {
          if (raw?.id && typeof raw?.type === 'string') {
            elementTypes.set(raw.id, raw.type)
          }
        }
        const elements = rawElements.map((el: any) => {
          if (!el) {
            return el
          }

          const rawLink = el.link
          const parsedLink = parseNodeLink(rawLink)
          const patternId = parsedLink.patternId
          const nextCustomData = { ...(el.customData ?? {}) }
          let hasCustomUpdate = false

          if (patternId !== null) {
            nextCustomData.patternId = patternId
            hasCustomUpdate = true
          }
          if (parsedLink.disambiguation) {
            nextCustomData.disambiguation = parsedLink.disambiguation
            hasCustomUpdate = true
          }
          const sanitizedLink = null

          const isFreeText = el?.type === 'text' && !el?.containerId
          const isArrowLabel =
            el?.type === 'text' && el?.containerId && elementTypes.get(el.containerId) === 'arrow'
          const baseElement =
            el?.type === 'arrow'
              ? {
                  ...el,
                  link: sanitizedLink,
                  strokeColor: '#ffffff',
                  backgroundColor: '#ffffff',
                }
              : isFreeText || isArrowLabel
                ? {
                    ...el,
                    link: sanitizedLink,
                    strokeColor: '#ffffff',
                  }
                : {
                    ...el,
                    link: sanitizedLink,
                  }

          const shouldStoreProgress = typeof patternId === 'number' && patternId > 0

          if (!shouldStoreProgress) {
            return hasCustomUpdate
              ? {
                  ...baseElement,
                  customData: nextCustomData,
                }
              : baseElement
          }

          const progress =
            normalizeProgressState(storedProgress[el.id]) ??
            normalizeProgressState(el?.customData?.progressState) ??
            'locked'
          hydratedProgress[el.id] = progress
          return {
            ...baseElement,
            customData: {
              ...nextCustomData,
              progressState: progress,
            },
          }
        })
        const sanitizedElements = sanitizeElements(elements)
        const appState = {
          ...(data?.appState ?? {}),
          ...initialData.appState,
          viewBackgroundColor: initialData.appState?.viewBackgroundColor ?? '#18181B',
        }
        const storedView = hasVisitedCanvas() ? loadStoredViewState() : null
        if (storedView) {
          appState.scrollX = storedView.scrollX
          appState.scrollY = storedView.scrollY
          appState.zoom = { value: storedView.zoom }
        }
        const files = tintSvgFiles(data?.files ?? {})
        setInitialScene({
          elements: sanitizedElements,
          appState,
          files,
        })
        elementsRef.current = sanitizedElements
        appStateRef.current = {
          ...appState,
          width: appState.width,
          height: appState.height,
        }
        summaryRef.current = computeSummary(sanitizedElements, appState as AppState)
        setMinimapElements(sanitizedElements)
        setMinimapAppState({
          ...appState,
          width: appState.width,
          height: appState.height,
        })
        setMinimapFiles(files)
        setProgressById(hydratedProgress)
        baseElementsRef.current = JSON.parse(JSON.stringify(sanitizedElements))
        const { ids, geometryById } = buildGeometrySnapshot(sanitizedElements)
        baseIdSetRef.current = ids
        baseGeometryRef.current = geometryById
        ownershipIndexRef.current = buildOwnershipIndex(sanitizedElements)
        setSceneKey((prev) => prev + 1)
        bumpUiVersion((v) => v + 1)
      })
      .catch(() => {
        if (!cancelled) {
          setInitialScene(null)
        }
      })

    return () => {
      cancelled = true
    }
  }, [])

  useEffect(() => {
    if (!initialScene || !excalidrawAPI.current) {
      return
    }

    let raf = 0
    const applyInitialCamera = () => {
      const api = excalidrawAPI.current
      if (!api) return
      api.setActiveTool({ type: 'selection', locked: false })

      const storedView = loadStoredViewState()
      if (storedView) {
        // Apply persisted camera state after Excalidraw mounts to avoid snapping back to defaults.
        api.updateScene({
          appState: {
            scrollX: storedView.scrollX,
            scrollY: storedView.scrollY,
            zoom: { value: storedView.zoom },
          },
        })
        markVisitedCanvas()
        allowViewPersistRef.current = true
        return
      }

      const visited = hasVisitedCanvas()
      if (visited) {
        allowViewPersistRef.current = true
        return
      }

      const appState = api.getAppState?.() ?? (appStateRef.current as AppState)
      const viewportWidth = appState?.width ?? 0
      const viewportHeight = appState?.height ?? 0
      if (!viewportWidth || !viewportHeight) {
        raf = requestAnimationFrame(applyInitialCamera)
        return
      }

      const elements = elementsRef.current as Array<any>
      if (!elements.length) {
        raf = requestAnimationFrame(applyInitialCamera)
        return
      }

      const anchorBounds = getAnchorBounds(elements)
      if (!anchorBounds) {
        // TODO: If the anchor label changes, store a stable element id or metadata for precise initial positioning.
        markVisitedCanvas()
        allowViewPersistRef.current = true
        return
      }

      const targetZoom = FIRST_VISIT_ZOOM
      const anchorWidth = Number.isFinite(anchorBounds.width) ? anchorBounds.width : 0
      const anchorHeight = Number.isFinite(anchorBounds.height) ? anchorBounds.height : 0
      const anchorX = (Number.isFinite(anchorBounds.x) ? anchorBounds.x : 0) + anchorWidth / 2
      const anchorY = (Number.isFinite(anchorBounds.y) ? anchorBounds.y : 0) + anchorHeight / 2
      const scrollX = viewportWidth / (2 * targetZoom) - anchorX
      const scrollY = viewportHeight / (2 * targetZoom) - anchorY

      api.updateScene({
        appState: {
          scrollX,
          scrollY,
          zoom: { value: targetZoom },
        },
      })

      markVisitedCanvas()
      allowViewPersistRef.current = true
    }

    raf = requestAnimationFrame(applyInitialCamera)

    return () => cancelAnimationFrame(raf)
  }, [sceneKey, initialScene])

  useEffect(() => {
    try {
      localStorage.setItem(PROGRESS_STORAGE_KEY, JSON.stringify(progressById))
    } catch {
      // Ignore storage failures (e.g., private mode).
    }
  }, [progressById])

  useEffect(() => {
    if (!excalidrawAPI.current) return
    const ownershipIndex = ownershipIndexRef.current
    if (!ownershipIndex) return
    const elements = elementsRef.current as Array<any>
    if (!elements.length) return

    const { nextElements, changed } = applyArrowDimming(
      elements,
      ownershipIndex,
      progressById,
      arrowBaseOpacityRef.current,
    )

    if (!changed) return
    elementsRef.current = nextElements

    const baseMap = new Map<string, any>()
    for (const el of nextElements) {
      if (el && el.id) {
        baseMap.set(el.id, el)
      }
    }
    baseElementsRef.current = (baseElementsRef.current as Array<any>).map((el) => {
      const updated = baseMap.get(el?.id)
      if (!updated || updated.opacity === el.opacity) {
        return el
      }
      return {
        ...el,
        opacity: updated.opacity,
      }
    })

    excalidrawAPI.current.updateScene({
      elements: nextElements,
    })
  }, [progressById, sceneKey])

  const { elementCount, selectionCount, panZoomLabel } = useMemo(() => summaryRef.current, [uiVersion])

  const nodeVisuals = useMemo(() => {
    const appState = appStateRef.current as AppState
    const elements = elementsRef.current as Array<any>
    if (!appState || elements.length === 0) {
      return { glass: [], locks: [], statusIcons: [] }
    }

    const zoom = appState.zoom?.value ?? 1
    const scrollX = appState.scrollX ?? 0
    const scrollY = appState.scrollY ?? 0
    const canvasBackground = appState.viewBackgroundColor ?? '#18181B'

    const ownershipIndex = ownershipIndexRef.current
    const ownerByElementId = ownershipIndex?.ownerByElementId ?? new Map<string, string>()

    const patternProgressByElementId = new Map<string, ProgressState>()
    for (const el of elements) {
      if (!el || el.isDeleted) continue
      const patternId = getPatternIdFromElement(el)
      if (typeof patternId !== 'number' || patternId <= 0) continue
      const progress =
        normalizeProgressState(progressById[el.id]) ??
        normalizeProgressState(el?.customData?.progressState) ??
        'locked'
      patternProgressByElementId.set(el.id, progress)
    }

    const nodeElements = elements.filter((el) => el && !el.isDeleted && isNodeElement(el))
    const glass: Array<any> = []
    const locks: Array<any> = []
    const statusIcons: Array<any> = []

    const glassAlphaPattern = clampNumber((0.18 + zoom * 0.04) * 1.728, 0.27648, 0.44928)
    const glassBorderAlphaPattern = clampNumber((glassAlphaPattern + 0.12) * 1.2, 0.264, 0.42)
    const downstreamDimFactor = 1.3
    const glassAlphaOwned = clampNumber(glassAlphaPattern * downstreamDimFactor, 0.32, 0.6)
    const glassBorderAlphaOwned = clampNumber(glassBorderAlphaPattern * downstreamDimFactor, 0.32, 0.6)
    const baseLockSize = 28
    const baseLockHover = 6
    const baseStatusSize = 22
    const baseStatusPad = 8

    for (const el of nodeElements) {
      const patternId = getPatternIdFromElement(el)
      if (patternId === 0) {
        continue
      }
      const isPatternNode = typeof patternId === 'number' && patternId > 0
      const ownerElementId = ownerByElementId.get(el.id) ?? (isPatternNode ? el.id : null)
      if (!ownerElementId) {
        continue
      }
      const ownerProgress = patternProgressByElementId.get(ownerElementId) ?? 'locked'
      const showGlass = ownerProgress === 'locked'
      const showLock = isPatternNode && ownerProgress === 'locked'
      const showCompleted = isPatternNode && ownerProgress === 'completed'
      const showInProgress = isPatternNode && ownerProgress === 'in_progress'

      const x1 = Math.min(el.x, el.x + el.width)
      const y1 = Math.min(el.y, el.y + el.height)
      const width = Math.abs(el.width)
      const height = Math.abs(el.height)
      const left = (x1 + scrollX) * zoom
      const top = (y1 + scrollY) * zoom
      const screenWidth = width * zoom
      const screenHeight = height * zoom
      const shapeStyle = getShapeStyle(el, screenWidth, screenHeight)

      if (showGlass) {
        const glassAlpha = isPatternNode ? glassAlphaPattern : glassAlphaOwned
        const glassBorderAlpha = isPatternNode ? glassBorderAlphaPattern : glassBorderAlphaOwned
        glass.push({
          id: `${el.id}-glass`,
          style: {
            position: 'absolute',
            left,
            top,
            width: screenWidth,
            height: screenHeight,
            backgroundColor: toRgba(canvasBackground, glassAlpha),
            border: `1px solid ${toRgba(canvasBackground, glassBorderAlpha)}`,
            ...shapeStyle,
          },
        })
      }

      if (showLock) {
        const lockSize = baseLockSize * zoom
        const lockHover = baseLockHover * zoom
        locks.push({
          id: `${el.id}-lock`,
          left: left + (screenWidth - lockSize) / 2,
          top: top - lockSize - lockHover,
          size: lockSize,
        })
      }

      if (showCompleted || showInProgress) {
        const statusSize = baseStatusSize * zoom
        const statusPad = baseStatusPad * zoom
        statusIcons.push({
          id: `${el.id}-status`,
          left: left + screenWidth - statusSize - statusPad,
          top: top + statusPad,
          size: statusSize,
          state: showCompleted ? 'completed' : 'in_progress',
        })
      }
    }

    return { glass, locks, statusIcons }
  }, [progressById, uiVersion])

  const patternNodes = useMemo<PatternNode[]>(() => {
    const elements = elementsRef.current as Array<any>
    if (!elements || elements.length === 0) return []
    const labelsByContainer = new Map<string, string>()
    for (const el of elements) {
      if (!el || el.isDeleted) continue
      if (el.type === 'text' && el.containerId) {
        const label = (el.rawText ?? el.text ?? '').trim()
        if (label) {
          labelsByContainer.set(el.containerId, label)
        }
      }
    }

    return elements
      .filter((el) => {
        if (!el || el.isDeleted) return false
        const patternId = el?.customData?.patternId
        return typeof patternId === 'number' && Number.isFinite(patternId) && patternId > 0
      })
      .map((el) => {
        const patternId = el.customData.patternId as number
        const baseLabel = labelsByContainer.get(el.id) || `Pattern ${patternId}`
        const disambiguation =
          typeof el.customData?.disambiguation === 'string' && el.customData.disambiguation.trim().length > 0
            ? el.customData.disambiguation.trim()
            : undefined
        const label = disambiguation ? `${disambiguation} — ${baseLabel}` : baseLabel

        return {
          elementId: el.id,
          patternId,
          label,
          disambiguation,
        }
      })
  }, [uiVersion])


  const createRoadmapEntryFromBackend = useCallback(
    (
      response: any,
      params: {
        patternSelectionMode: 'in_progress' | 'in_progress_and_completed' | 'custom'
        customPatternIds: number[]
        priorityPatternIds: number[]
        excludedPatternIds: number[]
        difficultyPreference: { easy: boolean; medium: boolean; hard: boolean }
        amountOfProblems: string
      },
    ) => {
      const request = response?.request ?? {}
      const now = new Date()
      const id =
        globalThis.crypto?.randomUUID?.() ??
        `roadmap-${now.getTime()}-${Math.random().toString(36).slice(2, 8)}`

      roadmapCountRef.current += 1

      const problems: RoadmapProblem[] = []
      const roadmapNodes = Array.isArray(response?.roadmap) ? response.roadmap : []
      for (const node of roadmapNodes) {
        const chunk = node?.chunk
        if (chunk && Array.isArray(chunk.problems)) {
          for (const prob of chunk.problems) {
            problems.push({
              id: `${id}-chunk-${chunk.chunkNumber ?? problems.length}-${problems.length + 1}`,
              title: prob?.title ?? 'Untitled',
              url: prob?.url,
            })
          }
        }
        const revision = node?.revisionProblem
        if (revision?.problem) {
          problems.push({
            id: `${id}-revision-${problems.length + 1}`,
            title: revision.problem?.title ?? 'Untitled',
            url: revision.problem?.url,
          })
        }
      }

      const parsedLimit = Number.parseInt(params.amountOfProblems, 10)
      const fallbackCount = Number.isFinite(parsedLimit) && parsedLimit > 0 ? parsedLimit : problems.length
      const requestCount =
        typeof request.user_requested_roadmap_length === 'number'
          ? request.user_requested_roadmap_length
          : fallbackCount

      const filters: RoadmapFilters = {
        selectionMode: params.patternSelectionMode,
        difficulty: params.difficultyPreference,
        count: requestCount,
        customPatternIds: params.customPatternIds,
        priorityPatternIds: params.priorityPatternIds,
        excludedPatternIds: params.excludedPatternIds,
      }

      const difficultyLabel = formatDifficultyLabel(filters.difficulty)
      const filtersSummary = `${formatDifficultyLabel(filters.difficulty)} | ${filters.count} problems`
      const summary = `Generated ${formatRoadmapTimestamp(now)} | ${filters.count} problems | ${difficultyLabel}`

      return {
        id,
        title: `Roadmap #${roadmapCountRef.current}`,
        generatedAt: now.toISOString(),
        summary,
        filtersSummary,
        filters,
        problems,
      }
    },
    [],
  )

  const handleGenerateRoadmap = useCallback(
    (params: {
      patternSelectionMode: 'in_progress' | 'in_progress_and_completed' | 'custom'
      customPatternIds: number[]
      priorityPatternIds: number[]
      excludedPatternIds: number[]
      difficultyPreference: { easy: boolean; medium: boolean; hard: boolean }
      amountOfProblems: string
    }): boolean => {
      const collected = collectPatternsForRequest({
        patternNodes,
        progressById,
        patternSelectionMode: params.patternSelectionMode,
        customPatternIds: params.customPatternIds,
        priorityPatternIds: params.priorityPatternIds,
        excludedPatternIds: params.excludedPatternIds,
      })

      if (collected.include.length === 0) {
        return false
      }

      const payload = buildGenerateRoadmapRequest({
        collectedPatterns: collected,
        difficulty: params.difficultyPreference,
        amountOfProblems: params.amountOfProblems,
      })

      const sendRequest = async () => {
        try {
          const resp = await fetch('/api/generateRoadmapRequest', {
            method: 'POST',
            headers: {
              'Content-Type': 'application/json',
            },
            body: JSON.stringify(payload),
          })
          if (!resp.ok) {
            throw new Error(`Backend responded with ${resp.status}`)
          }
          const data = await resp.json()
          const entry = createRoadmapEntryFromBackend(data, params)
          setRoadmaps((prev) => [entry, ...prev])
          setSelectedRoadmapId(entry.id)
        } catch (error) {
          console.error('Failed to generate roadmap from backend:', error)
        }
      }

      sendRequest()
      return true
    },
    [patternNodes, progressById, createRoadmapEntryFromBackend],
  )

  const handleSelectRoadmap = useCallback((roadmapId: string | null) => {
    setSelectedRoadmapId(roadmapId)
  }, [])

  const handleClearRoadmaps = useCallback(() => {
    setRoadmaps([])
    setSelectedRoadmapId(null)
    roadmapCountRef.current = 0
  }, [])

  const handleSelectProblem = useCallback((problem: RoadmapProblem) => {
    console.log('Selected roadmap problem:', problem)
  }, [])

  const highlights = useMemo(() => {
    const appState = appStateRef.current as AppState
    const zoom = appState?.zoom?.value ?? 1
    const scrollX = appState?.scrollX ?? 0
    const scrollY = appState?.scrollY ?? 0
    const selectedIds = Object.keys(appState?.selectedElementIds || {})
    const elements = elementsRef.current as Array<any>

    return selectedIds
      .map((id) => elements.find((el) => (el as any).id === id))
      .filter(Boolean)
      .map((el) => {
        const x1 = Math.min(el.x, el.x + el.width)
        const y1 = Math.min(el.y, el.y + el.height)
        const width = Math.abs(el.width)
        const height = Math.abs(el.height)
        return {
          id: el.id,
          left: (x1 + scrollX) * zoom,
          top: (y1 + scrollY) * zoom,
          width: width * zoom,
          height: height * zoom,
        }
      })
  }, [uiVersion])

  const uiOptions = useMemo(
    () => ({
      dockedSidebarBreakpoint: 0,
      footer: false,
      canvasActions: {
        changeViewBackgroundColor: false,
        clearCanvas: false,
        export: false,
        loadScene: false,
        saveToActiveFile: false,
        saveAsImage: false,
        toggleTheme: false,
      },
      tools: {
        image: true,
        library: false,
      },
    }),
    [],
  )

  const handleZoomToFit = useCallback(() => {
    excalidrawAPI.current?.scrollToContent(undefined, { fitToContent: true })
  }, [])

  const handleClearCanvas = useCallback(() => {
    excalidrawAPI.current?.resetScene()
  }, [])

  const handleContextMenu = useCallback((event: React.MouseEvent) => {
    event.preventDefault()
    event.stopPropagation()
    if (!selectedPatternElementId) {
      setContextMenu((prev) => (prev.open ? { ...prev, open: false, nodeId: null } : prev))
      return
    }
    setContextMenu({
      open: true,
      x: event.clientX,
      y: event.clientY,
      nodeId: selectedPatternElementId,
    })
  }, [selectedPatternElementId])

  const updateProgressState = useCallback(
    (patternElementId: string, nextState: ProgressState) => {
      setProgressById((prev) => {
        const currentState = prev[patternElementId] ?? 'locked'
        if (!canTransition(currentState, nextState)) {
          return prev
        }
        return {
          ...prev,
          [patternElementId]: nextState,
        }
      })
      elementsRef.current = (elementsRef.current as Array<any>).map((el) => {
        if (el?.id !== patternElementId) return el
        return {
          ...el,
          customData: {
            ...(el.customData ?? {}),
            progressState: nextState,
          },
        }
      })
      baseElementsRef.current = (baseElementsRef.current as Array<any>).map((el) => {
        if (el?.id !== patternElementId) return el
        return {
          ...el,
          customData: {
            ...(el.customData ?? {}),
            progressState: nextState,
          },
        }
      })
    },
    [canTransition],
  )

  useEffect(() => {
    if (!contextMenu.open) return
    const handlePointerDown = (event: PointerEvent) => {
      if (contextMenuRef.current && contextMenuRef.current.contains(event.target as Node)) {
        return
      }
      setContextMenu((prev) => ({ ...prev, open: false, nodeId: null }))
    }
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === 'Escape') {
        setContextMenu((prev) => ({ ...prev, open: false, nodeId: null }))
      }
    }
    window.addEventListener('pointerdown', handlePointerDown)
    window.addEventListener('keydown', handleKeyDown)
    return () => {
      window.removeEventListener('pointerdown', handlePointerDown)
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [contextMenu.open])

  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null
      if (target && (target.tagName === 'INPUT' || target.tagName === 'TEXTAREA' || target.isContentEditable)) {
        return
      }
      const isMod = event.ctrlKey || event.metaKey
      if (event.key === 'Delete' || event.key === 'Backspace') {
        event.preventDefault()
        event.stopPropagation()
        return
      }
      if (isMod && (event.key.toLowerCase() === 'd' || event.key.toLowerCase() === 'v')) {
        event.preventDefault()
        event.stopPropagation()
      }
    }

    window.addEventListener('keydown', handleKeyDown)
    return () => {
      window.removeEventListener('keydown', handleKeyDown)
    }
  }, [])

  return (
    <AppShell>
      <TopBar
        onHelp={() => {
          setTopBarSelection('help')
          setShowLanding(true)
          setShowLegend(false)
          setShowRoadmapData(false)
        }}
        helpActive={topBarSelection === 'help'}
        onLegend={() => {
          setTopBarSelection('legend')
          setShowLanding(false)
          setShowLegend(true)
          setShowRoadmapData(false)
        }}
        legendActive={topBarSelection === 'legend'}
        onRoadmapData={() => {
          setTopBarSelection('roadmap')
          setShowLanding(false)
          setShowLegend(false)
          setShowRoadmapData(true)
        }}
        roadmapDataActive={topBarSelection === 'roadmap'}
        onLogin={() => {
          setTopBarSelection('login')
          setShowLanding(false)
          setShowLegend(false)
          setShowRoadmapData(false)
        }}
        loginActive={topBarSelection === 'login'}
      />
      <RightToolbar
        patternNodes={patternNodes}
        roadmaps={roadmaps}
        selectedRoadmapId={selectedRoadmapId}
        onSelectRoadmap={handleSelectRoadmap}
        onClearRoadmaps={handleClearRoadmaps}
        onSelectProblem={handleSelectProblem}
        onGenerate={handleGenerateRoadmap}
      />
      <LandingOverlay
        open={showLanding}
        onClose={handleCloseLanding}
      >
        <div className="space-y-3">
          <p className="text-sm font-semibold text-slate-100">Welcome to the DSA Visual Canvas</p>
          <p>
            This canvas is designed to help you understand data structures and algorithms as a connected system,
            not a checklist of isolated topics. Each node represents a meaningful concept or pattern, and the links
            between them show how ideas relate, build on one another, or mark conceptual boundaries.
          </p>
          <p>
            The goal is not to follow a single prescribed path, but to explore the landscape. You can move freely,
            zoom into areas of interest, and let relationships guide what you study next.
          </p>
          <p className="text-sm font-semibold text-slate-100">How to Use the Canvas</p>
          <p>You can interact with the canvas directly to explore patterns, mark progress, and understand dependencies:</p>
          <ul className="list-disc pl-5 space-y-2 text-sm text-slate-200">
            <li>Interact directly with the canvas to explore patterns and dependencies</li>
            <li>Select nodes to inspect patterns and examples</li>
            <li>Mark patterns as in progress or completed to track learning</li>
            <li>Follow connections to see how ideas transition or depend on one another</li>
            <li>When in Select mode, you can right-click a node to mark it as in progress or completed.</li>
          </ul>
          <p>
            This tool is designed to support non-linear learning. You don&apos;t need to &ldquo;finish&rdquo; one area before
            touching another.
          </p>
          <p className="text-sm font-semibold text-slate-100">Roadmaps and Progress</p>
          <p>
            From your interaction with the canvas, the system can generate personalized roadmaps. These roadmaps are
            suggestions, not requirements. They are meant to highlight useful next steps based on what you&apos;ve already
            explored or completed.
          </p>
          <ul className="list-disc pl-5 space-y-2 text-sm text-slate-200">
            <li>Think of roadmaps as guidance, not obligation.</li>
            <li>
              You&apos;ll find more detail about how roadmaps are generated, and where the underlying problems come from, in
              the Roadmap &amp; Data overlay.
            </li>
            <li>
              You can also open the Legend at any time to understand what the shapes, colors, and frames on the canvas represent.
            </li>
          </ul>
          <p className="text-sm font-semibold text-slate-100">Controls</p>
          <ul className="list-disc pl-5 space-y-2 text-sm text-slate-200">
            <li>Press 1 to enter Select Mode.</li>
            <li>Press H to enter Pan Mode.</li>
            <li>The currently active mode is always shown on screen.</li>
          </ul>
          <p className="text-sm font-semibold text-slate-100">A Note on Exploration</p>
          <p>There is no single &ldquo;correct&rdquo; order to learn algorithms.</p>
          <p>
            Some concepts are foundations. Others represent shifts in thinking. This canvas is intentionally visual to
            make those differences intuitive rather than abstract. Use it to build understanding, not just coverage.
          </p>
        </div>
      </LandingOverlay>

      <LandingOverlay
        open={showLegend}
        onClose={() => setShowLegend(false)}
        title="Legend"
      >
        <div className="space-y-4">
          <p>
            This legend explains the visual language used on the canvas. Each shape and color represents a different
            kind of idea, from broad domains to specific techniques and implementations.
          </p>
          <div className="space-y-4">
            {[
              {
                icon: greenPillIcon,
                title: 'Green Circle — Data Structure / Algorithm Domain',
                description:
                  'Represents a foundational area such as Arrays & Strings, Trees, Graphs, or Numbers & Math. These are broad domains where many problems live.',
                tooltip: 'Data Structure / Algorithm Domain',
              },
              {
                icon: bluePillIcon,
                title: 'Blue Pill — Pattern / Algorithm',
                description:
                  'Represents a reusable algorithmic pattern or technique, such as Two Pointers, DFS, Binary Search, or Sliding Window.',
                tooltip: 'Pattern / Algorithm',
              },
              {
                icon: yellowPillIcon,
                title: 'Yellow Pill — Subpattern / Delineation',
                description:
                  'Represents a meaningful refinement or variation of a pattern that is worth naming explicitly.',
                tooltip: 'Subpattern / Delineation',
              },
              {
                icon: connectionFrameIcon,
                title: 'Rectangle Frame — Conceptual Grouping',
                description:
                  'Some domains are enclosed within a shared frame to indicate conceptual proximity. These frames suggest ideas that are worth learning together, not hierarchy or prerequisites.',
                tooltip: 'Conceptual Grouping',
              },
              {
                icon: purplePillIcon,
                title: 'Purple Diamond — Algorithmic Framework',
                description:
                  'Represents an entire problem-solving framework rather than a single algorithm (e.g. Backtracking, Dynamic Programming, Greedy). These indicate a conceptual boundary and a shift in how problems are reasoned about.',
                tooltip: 'Algorithmic Framework',
              },
              {
                icon: connectionFrameIcon,
                title: 'Explanation Block',
                description:
                  'Short conceptual descriptions explaining what a pattern is doing and when to recognize it.',
                tooltip: 'Explanation Block',
              },
              {
                icon: bluePillIcon,
                title: 'Code Template',
                description:
                  'A reusable implementation skeleton that captures the essence of a pattern.',
                tooltip: 'Code Template',
              },
            ].map((item) => (
              <div
                key={item.title}
                className="flex items-start gap-4"
                title={item.tooltip}
                aria-label={item.tooltip}
              >
                <img
                  src={item.icon}
                  alt=""
                  className="h-10 w-10 shrink-0"
                  draggable={false}
                />
                <div className="space-y-1">
                  <div className="text-sm font-semibold text-slate-100">{item.title}</div>
                  <div className="text-sm text-slate-300">{item.description}</div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </LandingOverlay>

      <LandingOverlay
        open={showRoadmapData}
        onClose={() => setShowRoadmapData(false)}
        title="Roadmap & Data"
      >
        <div className="space-y-3">
          <p className="text-sm font-semibold text-slate-100">How the Roadmap Works</p>
          <p>
            The roadmap is generated from a curated database of real LeetCode problems, each mapped to a specific pattern on the canvas. When you generate a roadmap, you are not receiving an abstract study plan&mdash;you are receiving a concrete sequence of problems designed to reinforce patterns through structured repetition.
          </p>
          <p>
            The roadmap is built using explicit spaced interleaving. Problems from previously completed or partially completed patterns are deliberately reintroduced later in the roadmap, rather than appearing only once. This creates an undulating path across patterns and tiers, encouraging revision, invariant recognition, and long-term retention instead of short-term completion.
          </p>
          <p>User preferences directly shape the roadmap. You can:</p>
          <ul className="list-disc pl-5 space-y-2 text-sm text-slate-200">
            <li>Treat difficulty as a hard filter or as a weighted preference</li>
            <li>Increase the frequency of specific patterns or subpatterns</li>
            <li>Fully exclude patterns or subpatterns from consideration</li>
            <li>Emphasize structured progression or broader exposure</li>
          </ul>
          <p>
            These preferences are respected while preserving conceptual continuity across the graph. The system balances spacing, priority, and feasibility so that patterns are revisited at meaningful intervals rather than clustered or exhausted early.
          </p>
          <p>
            The roadmap is guidance, not enforcement. You can always diverge, revisit nodes manually, or regenerate with different constraints.
          </p>
          <p className="text-sm font-semibold text-slate-100">Where the Problems Come From</p>
          <p>
            Problems in the roadmap come from a unified, curated database assembled from multiple teaching-focused and industry-relevant sources. These include pedagogical collections such as NeetCode 150, Grind 169, Grokking the Coding Interview, and Sean Prashad&rsquo;s problem list, alongside additional datasets that reflect how frequently problems appear in interviews at large technology companies.
          </p>
          <p>
            Problems are merged, de-duplicated, and reclassified so that each one has a clear conceptual home based on the primary pattern it teaches. Rather than inheriting difficulty labels or list positions verbatim, problems are ranked using multiple signals, including:
          </p>
          <ul className="list-disc pl-5 space-y-2 text-sm text-slate-200">
            <li>Their prominence across teaching-oriented repositories</li>
            <li>Their relative frequency in curated interview-prep datasets</li>
            <li>Their ability to illustrate a pattern&rsquo;s core invariant</li>
          </ul>
          <p>
            This allows the roadmap system to reason about spacing, prioritization, and revision without tying learning progress to any single external list.
          </p>
          <p className="text-sm font-semibold text-slate-100">About the Database</p>
          <p>The database is designed to be:</p>
          <ul className="list-disc pl-5 space-y-2 text-sm text-slate-200">
            <li>Pattern-first rather than platform-first</li>
            <li>Stable enough to support roadmap generation</li>
            <li>Flexible enough to evolve as patterns are refined</li>
            <li>Explicit about conceptual ownership (one clear home per problem)</li>
          </ul>
          <p>
            Each problem is stored with metadata that allows it to be weighted, spaced, and reintroduced in service of learning rather than completion.
          </p>
          <p className="text-sm font-semibold text-slate-100">External Reference</p>
          <p>The underlying problem database and roadmap logic will be documented in a public GitHub repository.</p>
          <p>You&rsquo;ll be able to explore the underlying code, roadmap logic, and database ranking and sorting decisions in detail there.</p>
          <p>GitHub (placeholder):</p>
          <p>[ View the database on GitHub ]</p>
        </div>
      </LandingOverlay>

      <main className="relative flex h-screen w-screen flex-col">
        {/* Screen-space UI: keep overlays fixed to viewport (no scroll/zoom transforms). */}
        <section className="relative flex-1">
          <div className="absolute inset-0">
            <div
              className="excalidraw-host absolute inset-0 z-10 h-full w-full"
              onContextMenuCapture={handleContextMenu}
              onDoubleClickCapture={(event) => {
                event.preventDefault()
                event.stopPropagation()
              }}
            >
              <ExcalidrawComponent
                key={sceneKey}
                excalidrawAPI={(api: ExcalidrawImperativeAPI) => {
                  excalidrawAPI.current = api
                }}
                initialData={initialScene ?? initialData}
                onChange={handleCanvasChange}
                onContextMenu={handleContextMenu}
                UIOptions={uiOptions}
                renderTopRightUI={() => null}
                renderTopLeftUI={() => null}
                renderFooter={() => null}
              />
            </div>

            <div className="pointer-events-none absolute inset-0 z-30">
              {nodeVisuals.glass.map((glass: any) => (
                <div key={glass.id} style={glass.style} />
              ))}
            </div>

            <div className="pointer-events-none absolute inset-0 z-40">
              {nodeVisuals.locks.map((lock: any) => (
                <div
                  key={lock.id}
                  className="absolute flex items-center justify-center rounded-full border border-amber-300/40 bg-ink/70"
                  style={{
                    left: lock.left,
                    top: lock.top,
                    width: lock.size,
                    height: lock.size,
                  }}
                >
                  <svg
                    viewBox="0 0 24 24"
                    width={lock.size - 2}
                    height={lock.size - 2}
                    fill="none"
                    stroke="rgba(245, 158, 11, 0.95)"
                    strokeWidth="2"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                  >
                    <rect x="5" y="11" width="14" height="9" rx="2" />
                    <path d="M8 11V8a4 4 0 0 1 8 0v3" />
                  </svg>
                </div>
              ))}
            </div>

            <div className="pointer-events-none absolute inset-0 z-45">
              {nodeVisuals.statusIcons.map((icon: any) => (
                <div
                  key={icon.id}
                  className="absolute flex items-center justify-center rounded-full border border-amber-300/40 bg-ink/70"
                  style={{
                    left: icon.left,
                    top: icon.top,
                    width: icon.size,
                    height: icon.size,
                  }}
                >
                  {icon.state === 'completed' ? (
                    <svg
                      viewBox="0 0 24 24"
                      width={icon.size - 4}
                      height={icon.size - 4}
                      fill="none"
                      stroke="rgba(245, 158, 11, 0.95)"
                      strokeWidth="2.4"
                      strokeLinecap="round"
                      strokeLinejoin="round"
                    >
                      <path d="M5 13l4 4L19 7" />
                    </svg>
                  ) : (
                    <svg
                      viewBox="0 0 24 24"
                      width={icon.size - 4}
                      height={icon.size - 4}
                      fill="none"
                      stroke="rgba(245, 158, 11, 0.95)"
                      strokeWidth="2.2"
                      strokeLinecap="round"
                      className="animate-spin"
                    >
                      <circle cx="12" cy="12" r="9" strokeDasharray="42 14" />
                    </svg>
                  )}
                </div>
              ))}
            </div>

            <div className="pointer-events-none absolute inset-0 z-50">
              {highlights.map((box) => (
                <div
                  key={box.id}
                  className="absolute rounded-lg ring-4 ring-accent-amber/50 shadow-[0_0_0_2px_rgba(245,158,11,0.6)]"
                  style={{
                    left: box.left,
                    top: box.top,
                    width: box.width,
                    height: box.height,
                  }}
                />
              ))}
            </div>
          </div>

        </section>

        {contextMenu.open && contextMenu.nodeId && (
          <div
            ref={contextMenuRef}
            className="fixed z-50 w-56"
            style={{ left: contextMenu.x, top: contextMenu.y }}
            onPointerDown={(event) => event.stopPropagation()}
          >
            <Panel className="rounded-xl2 border border-white/10 bg-panel/95 px-2 py-2">
              <div className="px-2 pb-2 text-[11px] uppercase tracking-[0.2em] text-slate-400">
                Progress state
              </div>
              <div className="flex flex-col gap-1">
                <button
                  type="button"
                  className="rounded-lg px-3 py-2 text-left text-sm text-slate-100 hover:bg-white/5 disabled:opacity-50"
                  disabled={!canTransition(progressById[contextMenu.nodeId] ?? 'locked', 'in_progress')}
                  onClick={() => {
                    updateProgressState(contextMenu.nodeId!, 'in_progress')
                    setContextMenu((prev) => ({ ...prev, open: false, nodeId: null }))
                  }}
                >
                  Mark In Progress
                </button>
                <button
                  type="button"
                  className="rounded-lg px-3 py-2 text-left text-sm text-slate-100 hover:bg-white/5 disabled:opacity-50"
                  disabled={!canTransition(progressById[contextMenu.nodeId] ?? 'locked', 'completed')}
                  onClick={() => {
                    updateProgressState(contextMenu.nodeId!, 'completed')
                    setContextMenu((prev) => ({ ...prev, open: false, nodeId: null }))
                  }}
                >
                  Mark Completed
                </button>
                <button
                  type="button"
                  className="rounded-lg px-3 py-2 text-left text-sm text-slate-100 hover:bg-white/5 disabled:opacity-50"
                  disabled={!canTransition(progressById[contextMenu.nodeId] ?? 'locked', 'locked')}
                  onClick={() => {
                    updateProgressState(contextMenu.nodeId!, 'locked')
                    setContextMenu((prev) => ({ ...prev, open: false, nodeId: null }))
                  }}
                >
                  Revert to Locked
                </button>
              </div>
            </Panel>
          </div>
        )}

      </main>

      {/* Screen-space UI layer (kept outside the canvas subtree to avoid inheriting transforms). */}
      <div className="pointer-events-none fixed inset-0 z-40">
        <div
          className="pointer-events-none absolute bottom-6 left-6 flex flex-col items-start gap-2"
          style={{ width: MINIMAP_WIDTH }}
        >
          <InteractionModeIndicator mode={interactionMode} className="w-full" />
          <Minimap
            elements={minimapElements}
            appState={minimapAppState}
            files={minimapFiles}
            width={MINIMAP_WIDTH}
            height={MINIMAP_HEIGHT}
            padding={MINIMAP_PADDING}
          />
        </div>
      </div>
    </AppShell>
  )
}
