import { useCallback, useMemo, useRef, useState } from 'react'
import { Excalidraw } from '@excalidraw/excalidraw'
import type { AppState, ExcalidrawImperativeAPI, ExcalidrawInitialDataState } from '@excalidraw/excalidraw/types'
import '@excalidraw/excalidraw/index.css'
import {
  AppShell,
  Button,
  LeftDrawer,
  Panel,
  RightToolbar,
  TopBar,
} from './ui/primitives'

const initialData: ExcalidrawInitialDataState = {
  elements: [],
  appState: {
    viewBackgroundColor: '#0b1224',
  },
}

type UISummary = {
  elementCount: number
  selectionCount: number
  panZoomLabel: string
}

export default function App() {
  const excalidrawAPI = useRef<ExcalidrawImperativeAPI | null>(null)
  const elementsRef = useRef<readonly unknown[]>([])
  const appStateRef = useRef<Partial<AppState>>({})
  const summaryRef = useRef<UISummary>({
    elementCount: 0,
    selectionCount: 0,
    panZoomLabel: 'zoom 100% · pan (0, 0)',
  })
  const [uiVersion, bumpUiVersion] = useState(0)
  const [isLibraryOpen, setIsLibraryOpen] = useState(true)

  const computeSummary = useCallback((elements: readonly unknown[], appState: AppState): UISummary => {
    const selectionCount = Object.keys(appState.selectedElementIds || {}).length
    const zoom = appState.zoom?.value ?? 1
    const x = Math.round(appState.scrollX ?? 0)
    const y = Math.round(appState.scrollY ?? 0)
    return {
      elementCount: elements.length,
      selectionCount,
      panZoomLabel: `zoom ${Math.round(zoom * 100)}% · pan (${x}, ${y})`,
    }
  }, [])

  const handleCanvasChange = useCallback(
    (elements: readonly unknown[], appState: AppState) => {
      elementsRef.current = elements
      appStateRef.current = appState
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
    },
    [computeSummary],
  )

  const { elementCount, selectionCount, panZoomLabel } = useMemo(() => summaryRef.current, [uiVersion])

  const handleZoomToFit = useCallback(() => {
    excalidrawAPI.current?.scrollToContent(undefined, { fitToContent: true })
  }, [])

  const handleClearCanvas = useCallback(() => {
    excalidrawAPI.current?.resetScene()
  }, [])

  return (
    <AppShell>
      <TopBar />
      <RightToolbar />

      <LeftDrawer
        open={isLibraryOpen}
        onToggle={() => setIsLibraryOpen((v) => !v)}
        title="Library"
        subtitle="Saved diagrams / graph snippets"
      >
        <div className="space-y-2 text-sm text-slate-300">
          <p>Use Excalidraw to draw and save scenes.</p>
          <p className="text-xs text-slate-500">Canvas is the source of truth for selection and linking.</p>
        </div>
      </LeftDrawer>

      <main className="relative mx-auto flex max-w-7xl flex-col gap-6 px-6 pb-12 pt-28 lg:flex-row">
        <div className="w-full max-w-xl space-y-4 lg:w-72">
          <Panel className="rounded-2xl p-4">
            <div>
              <p className="text-sm font-semibold text-slate-400">Scene metadata</p>
              <p className="text-2xl font-semibold text-white">{elementCount} elements</p>
              <p className="text-sm text-slate-400">
                {selectionCount} selected · {panZoomLabel}
              </p>
            </div>

            <div className="mt-4 space-y-2 text-sm text-slate-300">
              <div className="flex items-center justify-between rounded-xl border border-white/10 bg-panel/70 px-3 py-2">
                <span>Linking</span>
                <span className="rounded-full bg-emerald-500/10 px-2 py-1 text-xs text-emerald-300">Canvas-driven</span>
              </div>
              <div className="flex items-center justify-between rounded-xl border border-white/10 bg-panel/70 px-3 py-2">
                <span>Selection</span>
                <span className="rounded-full bg-emerald-500/10 px-2 py-1 text-xs text-emerald-300">Canvas-driven</span>
              </div>
              <div className="flex items-center justify-between rounded-xl border border-white/10 bg-panel/70 px-3 py-2">
                <span>Pan &amp; zoom</span>
                <span className="rounded-full bg-emerald-500/10 px-2 py-1 text-xs text-emerald-300">Canvas-driven</span>
              </div>
            </div>

            <div className="mt-4 space-y-2">
              <Button onClick={handleZoomToFit} className="w-full text-sm font-semibold">
                Zoom to fit
              </Button>
              <Button onClick={handleClearCanvas} variant="destructive" className="w-full text-sm font-semibold">
                Reset canvas
              </Button>
            </div>
          </Panel>
        </div>

        <section className="flex-1 space-y-4">
          <Panel className="rounded-2xl border border-white/10 px-4 py-3 flex items-center justify-between">
            <div>
              <p className="text-sm font-semibold text-slate-400">Canvas</p>
              <p className="text-lg font-semibold text-white">Excalidraw embedded</p>
            </div>
            <div className="flex gap-2 text-xs text-slate-300">
              <span className="rounded-full border border-white/10 px-3 py-1">
                Selection: {selectionCount}
              </span>
              <span className="rounded-full border border-white/10 px-3 py-1">{panZoomLabel}</span>
            </div>
          </Panel>

          <Panel className="overflow-hidden rounded-2xl border border-white/10 bg-slate-950/60 p-0">
            <div className="h-[72vh]">
              <Excalidraw
                excalidrawAPI={(api) => {
                  excalidrawAPI.current = api
                }}
                initialData={initialData}
                onChange={handleCanvasChange}
                UIOptions={{ dockedSidebarBreakpoint: 0 }}
              />
            </div>
          </Panel>
        </section>
      </main>
    </AppShell>
  )
}
