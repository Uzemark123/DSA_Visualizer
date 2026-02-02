import React, { useMemo, useState } from 'react'

/* =========================
   UI PRIMITIVES
   ========================= */

export const Panel: React.FC<
  React.PropsWithChildren<
    React.HTMLAttributes<HTMLDivElement> & {
      className?: string
      style?: React.CSSProperties
    }
  >
> = ({ children, className = '', style, ...rest }) => (
  <div
    className={`bg-panel/90 backdrop-blur border border-white/10 shadow-float ${className}`}
    style={style}
    {...rest}
  >
    {children}
  </div>
)

export const Button: React.FC<
  React.ButtonHTMLAttributes<HTMLButtonElement> & {
    variant?: 'default' | 'destructive'
  }
> = ({ className = '', variant = 'default', ...props }) => {
  const base =
    'inline-flex items-center justify-center whitespace-nowrap px-3.5 py-[9px] rounded-full text-[13px] font-semibold transition border shadow-sm'
  const variants = {
    default: 'bg-card/80 text-slate-200 border-white/10 hover:bg-card',
    destructive: 'bg-red-500/10 text-red-100 border-red-400/30 hover:bg-red-500/20',
  }
  return <button {...props} className={`${base} ${variants[variant]} ${className}`} />
}

export const IconButton: React.FC<React.ButtonHTMLAttributes<HTMLButtonElement>> = ({
  className = '',
  ...props
}) => (
  <button
    {...props}
    className={`w-8 h-8 rounded-lg bg-card/80 hover:bg-card transition text-[11px] text-slate-200 border border-white/5 ${className}`}
  />
)

/* =========================
   LAYOUT COMPONENTS
   ========================= */

export const AppShell: React.FC<React.PropsWithChildren> = ({ children }) => (
  <div className="relative min-h-screen bg-ink text-slate-100 overflow-hidden">{children}</div>
)

const selectionPillClass = (selected: boolean) =>
  selected
    ? '!border-amber-300/60 !bg-amber-500/20 !text-accent-amber !hover:bg-amber-500/20 !hover:border-amber-300/60 !hover:text-accent-amber'
    : '!border-white/10 !bg-card/80 !text-slate-200 hover:bg-card hover:border-white/20'

export const TopBar: React.FC<{
  onHelp?: () => void
  helpActive?: boolean
  onLegend?: () => void
  legendActive?: boolean
  onRoadmapData?: () => void
  roadmapDataActive?: boolean
  onLogin?: () => void
  loginActive?: boolean
}> = ({
  onHelp,
  helpActive = false,
  onLegend,
  legendActive = false,
  onRoadmapData,
  roadmapDataActive = false,
  onLogin,
  loginActive = false,
}) => (
  <div className="fixed z-30 top-6 left-1/2 -translate-x-1/2">
    <Panel className="flex items-center gap-2 px-3.5 py-[9px] rounded-full">
      <div className="flex items-center gap-2 text-[13px] text-slate-200">
        <span className="w-2 h-2 rounded-full bg-accent-amber" />
        <span className="font-semibold">DSA Visual Canvas</span>
      </div>
      <div className="flex items-center gap-2 text-slate-300 text-[13px]">
        <Button className={selectionPillClass(loginActive)} onClick={onLogin}>
          Log in
        </Button>
        <Button
          className={selectionPillClass(helpActive)}
          onClick={onHelp}
          title="How to use the canvas"
          aria-label="How to use the canvas"
        >
          Help
        </Button>
        <Button
          className={selectionPillClass(legendActive)}
          onClick={onLegend}
          title="What the shapes and colors mean"
          aria-label="What the shapes and colors mean"
        >
          Legend
        </Button>
        <Button
          className={selectionPillClass(roadmapDataActive)}
          onClick={onRoadmapData}
          title="How roadmaps are generated"
          aria-label="How roadmaps are generated"
        >
          Roadmap &amp; Data
        </Button>
      </div>
    </Panel>
  </div>
)

export const InteractionModeIndicator: React.FC<{ mode: 'select' | 'pan'; className?: string }> = ({
  mode,
  className = '',
}) => {
  const isSelect = mode === 'select'
  const isPan = mode === 'pan'
  return (
    <div className={`pointer-events-none ${className}`}>
      <div className="rounded-2xl border border-white/10 bg-panel/85 px-3.5 py-[9px] text-[13px] uppercase tracking-[0.1em]">
        <div className="grid grid-cols-2 gap-2 text-center">
          <div
            className={`rounded-full border px-3.5 py-1 text-[13px] whitespace-nowrap ${
              isSelect
                ? 'border-amber-300/70 bg-amber-500/20 text-accent-amber'
                : 'border-white/10 bg-ink/40 text-slate-400'
            }`}
          >
            Select (1)
          </div>
          <div
            className={`rounded-full border px-3.5 py-1 text-[13px] whitespace-nowrap ${
              isPan
                ? 'border-amber-300/70 bg-amber-500/20 text-accent-amber'
                : 'border-white/10 bg-ink/40 text-slate-400'
            }`}
          >
            Pan (H)
          </div>
        </div>
      </div>
    </div>
  )
}

export type LandingOverlayProps = {
  open: boolean
  onClose: () => void
  title?: string
  children: React.ReactNode
}

export const LandingOverlay: React.FC<LandingOverlayProps> = ({
  open,
  onClose,
  title,
  children,
}) => {
  const [closing, setClosing] = useState(false)

  React.useEffect(() => {
    if (open) {
      setClosing(false)
    }
  }, [open])

  if (!open) {
    return null
  }

  const handleClose = () => {
    if (closing) {
      return
    }
    setClosing(true)
    window.setTimeout(() => {
      onClose()
    }, 180)
  }

  return (
    <div
      className={`fixed inset-0 z-50 flex items-center justify-center px-6 transition-opacity duration-200 ${
        closing ? 'opacity-0' : 'opacity-100'
      }`}
      onClick={handleClose}
    >
      <div className="absolute inset-0 bg-black/55" />
      <Panel
        className={`relative w-full max-w-3xl max-h-[80vh] rounded-3xl px-6 py-6 transition-all duration-200 flex flex-col ${
          closing ? 'opacity-0 translate-y-2 scale-[0.98]' : 'opacity-100 translate-y-0 scale-100'
        }`}
        onClick={(event) => event.stopPropagation()}
      >
        <div className="flex items-start justify-between gap-4">
          <div>
            {title && <p className="text-xl font-semibold text-slate-100">{title}</p>}
          </div>
          <Button
            onClick={handleClose}
            className="text-[11px] font-semibold border-amber-300/50 bg-amber-500/10 text-accent-amber hover:bg-amber-500/20"
          >
            Take me to the canvas
          </Button>
        </div>
        <div className="mt-4 min-h-0 flex-1 overflow-y-auto text-sm text-slate-200 leading-relaxed">
          {children}
        </div>
        <div className="mt-6">
          <Button
            onClick={handleClose}
            className="w-full text-xs font-semibold border-amber-300/60 bg-amber-500/20 text-accent-amber hover:bg-amber-500/30"
          >
            Take me to the canvas
          </Button>
        </div>
      </Panel>
    </div>
  )
}

type PatternNodeOption = {
  elementId: string
  patternId: number
  label: string
}

type RoadmapProblem = {
  id: string
  title: string
  difficulty?: 'easy' | 'medium' | 'hard'
  patternLabel?: string
  patternId?: number
  canvasNodeId?: string
  url?: string
}

type RoadmapEntry = {
  id: string
  title: string
  generatedAt: string
  summary: string
  filtersSummary: string
  problems: RoadmapProblem[]
}

const PatternDropdown: React.FC<{
  label: string
  selectedIds: number[]
  onChange: (next: number[]) => void
  options: PatternNodeOption[]
  disabled?: boolean
}> = ({ label, selectedIds, onChange, options, disabled = false }) => {
  const [open, setOpen] = useState(false)
  const selectedCount = selectedIds.length

  const togglePatternId = (ids: number[], target: number) =>
    ids.includes(target) ? ids.filter((id) => id !== target) : [...ids, target]

  return (
    <div>
      <label className="block text-sm uppercase tracking-[0.2em] text-slate-400">{label}</label>
      <button
        type="button"
        disabled={disabled}
        onClick={() => setOpen((value) => !value)}
        className={`mt-2 w-full rounded-full border px-3 py-2 text-left text-sm transition ${
          disabled
            ? 'border-white/5 bg-ink/30 text-slate-500'
            : 'border-white/10 bg-ink/40 text-slate-100 hover:border-amber-300/40'
        }`}
      >
        {selectedCount > 0 ? `${selectedCount} selected` : 'Select patterns'}
      </button>
      {open && !disabled && (
        <div className="mt-2 max-h-44 overflow-y-auto rounded-2xl border border-white/10 bg-ink/60 px-2 py-2">
          {options.length === 0 && (
            <div className="px-2 py-2 text-xs text-slate-400">No pattern nodes found.</div>
          )}
          {options.map((pattern) => {
            const id = pattern.patternId
            const checked = selectedIds.includes(id)
            return (
              <label
                key={pattern.elementId}
                className="flex cursor-pointer items-center gap-2 rounded-xl px-2 py-2 text-base text-slate-100 hover:bg-white/5"
              >
                <input
                  type="checkbox"
                  checked={checked}
                  onChange={() => onChange(togglePatternId(selectedIds, id))}
                  className="h-4 w-4 accent-amber-400"
                />
                <span className="truncate">{pattern.label || `Pattern ${pattern.patternId}`}</span>
              </label>
            )
          })}
        </div>
      )}
    </div>
  )
}

export const RightToolbar: React.FC<{
  patternNodes?: PatternNodeOption[]
  roadmaps?: RoadmapEntry[]
  selectedRoadmapId?: string | null
  onSelectRoadmap?: (roadmapId: string | null) => void
  onClearRoadmaps?: () => void
  onSelectProblem?: (problem: RoadmapProblem) => void
  onGenerate?: (params: {
    patternSelectionMode: 'in_progress' | 'in_progress_and_completed' | 'custom'
    customPatternIds: number[]
    priorityPatternIds: number[]
    excludedPatternIds: number[]
    difficultyPreference: { easy: boolean; medium: boolean; hard: boolean }
    amountOfProblems: string
  }) => boolean
}> = ({
  patternNodes = [],
  roadmaps = [],
  selectedRoadmapId = null,
  onSelectRoadmap,
  onClearRoadmaps,
  onSelectProblem,
  onGenerate,
}) => {
  const [open, setOpen] = useState(false)
  const [roadmapsOpen, setRoadmapsOpen] = useState(false)
  const [completedProblems, setCompletedProblems] = useState<Record<string, boolean>>({})
  const [patternSelectionMode, setPatternSelectionMode] = useState<'in_progress' | 'in_progress_and_completed' | 'custom'>(
    'in_progress',
  )
  const [customPatternIds, setCustomPatternIds] = useState<number[]>([])
  const [priorityPatternIds, setPriorityPatternIds] = useState<number[]>([])
  const [excludedPatternIds, setExcludedPatternIds] = useState<number[]>([])
  const [difficultyPreference, setDifficultyPreference] = useState({
    easy: false,
    medium: false,
    hard: false,
  })
  const [amountOfProblems, setAmountOfProblems] = useState('')
  const [generateError, setGenerateError] = useState<string | null>(null)

  const sortedPatterns = useMemo(() => {
    return [...patternNodes].sort((a, b) => a.patternId - b.patternId)
  }, [patternNodes])
  const selectedRoadmap = useMemo(
    () => roadmaps.find((roadmap) => roadmap.id === selectedRoadmapId) ?? null,
    [roadmaps, selectedRoadmapId],
  )
  const hasRoadmaps = roadmaps.length > 0
  const formatDateOnly = (value: string) =>
    new Date(value).toLocaleDateString(undefined, { month: 'short', day: 'numeric' })
  const extractDifficultyLabel = (summary: string) => {
    const segments = summary
      .split('|')
      .map((segment) => segment.trim())
      .filter(Boolean)
    return (
      segments.find((segment) => /easy|medium|hard|any/i.test(segment) && !/problem/i.test(segment)) ??
      ''
    )
  }
  const difficultyStyles: Record<string, string> = {
    easy: 'border-emerald-400/30 bg-emerald-500/10 text-emerald-200',
    medium: 'border-amber-400/30 bg-amber-500/10 text-amber-200',
    hard: 'border-rose-400/30 bg-rose-500/10 text-rose-200',
  }
  const getProblemKey = (roadmapId: string | null, problemId: string) =>
    `${roadmapId ?? 'roadmap'}:${problemId}`
  const toggleProblemCompleted = (roadmapId: string | null, problemId: string) => {
    const key = getProblemKey(roadmapId, problemId)
    setCompletedProblems((prev) => ({
      ...prev,
      [key]: !prev[key],
    }))
  }

  React.useEffect(() => {
    if (generateError) {
      setGenerateError(null)
    }
  }, [
    patternSelectionMode,
    customPatternIds,
    priorityPatternIds,
    excludedPatternIds,
    difficultyPreference,
    amountOfProblems,
  ])

  const roadmapPillPanelClass = 'inline-flex w-40 rounded-xl2 px-3.5 py-[9px] border-amber-300/60 bg-amber-500/15'
  const roadmapPillClass =
    'inline-flex w-full items-center justify-center whitespace-nowrap text-[13px] font-semibold text-accent-amber hover:text-amber-200'

  return (
    <>
      <div className="fixed z-[60] left-6 top-20 flex flex-col items-start gap-3">
        <Panel className={roadmapPillPanelClass}>
          <button
            type="button"
            onClick={() => {
              setRoadmapsOpen((value) => {
                const next = !value
                if (!next) {
                  onSelectRoadmap?.(null)
                }
                return next
              })
              setOpen(false)
            }}
            className={roadmapPillClass}
          >
            My Roadmaps
          </button>
        </Panel>

      <Panel
        className={`w-96 rounded-2xl transition-all duration-300 origin-top bg-panel/95 overflow-hidden flex flex-col min-h-0
          ${roadmapsOpen ? 'opacity-100 translate-y-0 h-[calc(100vh-180px)] max-h-[calc(100vh-180px)]' : 'opacity-0 -translate-y-2 h-0 max-h-0 pointer-events-none'}`}
      >
        <div className="flex h-full min-h-0 flex-col gap-3 px-4 py-4 text-base text-slate-200">
          <div className="flex items-center justify-between">
            <div className="text-[11px] uppercase tracking-[0.2em] text-slate-400">
              Saved roadmaps ({roadmaps.length})
            </div>
            <button
              type="button"
              disabled={!hasRoadmaps}
              onClick={() => onClearRoadmaps?.()}
              className="text-[11px] font-semibold text-slate-400 hover:text-slate-200 disabled:opacity-40"
            >
              Clear all
            </button>
          </div>

          {!hasRoadmaps && (
            <div className="rounded-xl border border-white/10 bg-ink/40 px-3 py-3 text-xs text-slate-400">
              No roadmaps yet - generate one to see it here.
            </div>
          )}

          {hasRoadmaps && (
            <div className="flex min-h-0 flex-initial flex-col gap-2 overflow-y-auto pr-1">
              {roadmaps.map((roadmap, index) => {
                const isActive = roadmap.id === selectedRoadmapId
                const difficultyLabel = extractDifficultyLabel(roadmap.filtersSummary) || 'Any'
                const label = `${index + 1}. Roadmap`
                const metadataLine = [
                  formatDateOnly(roadmap.generatedAt),
                  `${roadmap.problems.length} problems`,
                  difficultyLabel,
                ]
                  .filter(Boolean)
                  .join(' | ')
                return (
                  <button
                    key={roadmap.id}
                    type="button"
                    onClick={() => {
                      onSelectRoadmap?.(roadmap.id)
                    }}
                    className={`w-full rounded-xl border px-3 py-3 text-left transition ${
                      isActive
                        ? 'border-amber-400/40 bg-amber-500/10 text-amber-100'
                        : 'border-white/10 bg-ink/40 text-slate-100 hover:border-amber-300/30 hover:bg-white/5'
                    }`}
                  >
                    <div className="flex items-center justify-between gap-3 whitespace-nowrap text-sm font-semibold text-slate-200">
                      <span className="truncate text-left">{label}</span>
                      <span className="truncate text-right text-slate-300">{metadataLine}</span>
                    </div>
                  </button>
                )
              })}
            </div>
          )}

          {selectedRoadmap && (
            <div className="flex min-h-0 flex-1 flex-col gap-2 border-t border-white/10 pt-3">
              <div className="flex items-center justify-between gap-3 text-sm text-slate-300">
                <span className="min-w-0 truncate text-slate-100">{selectedRoadmap.title}</span>
                <span className="shrink-0 text-slate-300">
                  {[
                    formatDateOnly(selectedRoadmap.generatedAt),
                    `${selectedRoadmap.problems.length} problems`,
                    extractDifficultyLabel(selectedRoadmap.filtersSummary),
                  ]
                    .filter(Boolean)
                    .join(' | ')}
                </span>
              </div>

              <div className="flex min-h-0 flex-1 flex-col gap-2">
                <div className="text-xs uppercase tracking-[0.2em] text-slate-400">Roadmap problems</div>
                <div className="flex min-h-0 flex-1 flex-col gap-2 overflow-y-auto pr-1">
                  {selectedRoadmap.problems.map((problem, index) => (
                    (() => {
                      const key = getProblemKey(selectedRoadmap.id, problem.id)
                      const isCompleted = Boolean(completedProblems[key])
                      return (
                    <button
                      key={problem.id}
                      type="button"
                      onClick={() => onSelectProblem?.(problem)}
                      className={`w-full rounded-xl border px-3 py-2 text-left text-base transition ${
                        isCompleted
                          ? 'border-emerald-400/50 bg-emerald-500/20 text-emerald-50'
                          : 'border-white/10 bg-ink/40 text-slate-100 hover:border-amber-300/30 hover:bg-white/5'
                      }`}
                    >
                      <div className="flex items-start justify-between gap-3">
                        <div className="flex min-w-0 flex-1 items-start gap-3">
                          <span
                            className={`w-8 shrink-0 text-[11px] font-semibold ${
                              isCompleted ? 'text-emerald-50' : 'text-white/85'
                            }`}
                          >
                            #{index + 1}
                          </span>
                          <div className="min-w-0 flex-1">
                            <div className="flex items-start justify-between gap-2">
                              <div className={`text-base font-medium truncate ${isCompleted ? 'text-emerald-50' : ''}`}>
                                {problem.title}
                              </div>
                              {problem.difficulty && (
                                <span
                                  className={`rounded-full border px-2 py-0.5 text-[11px] uppercase ${difficultyStyles[problem.difficulty]}`}
                                >
                                  {problem.difficulty}
                                </span>
                              )}
                            </div>
                            {problem.url && (
                              <a
                                href={problem.url}
                                target="_blank"
                                rel="noreferrer"
                                onClick={(event) => event.stopPropagation()}
                                className={`mt-1 block text-xs truncate ${
                                  isCompleted ? 'text-emerald-100/80 hover:text-emerald-50' : 'text-slate-400 hover:text-amber-200'
                                }`}
                                title={problem.url}
                              >
                                {problem.url}
                              </a>
                            )}
                            {problem.patternLabel && (
                              <div className={`mt-1 text-xs ${isCompleted ? 'text-emerald-100/80' : 'text-slate-400'}`}>
                                Pattern: {problem.patternLabel}
                              </div>
                            )}
                          </div>
                        </div>
                        <input
                          type="checkbox"
                          checked={isCompleted}
                          onChange={() => toggleProblemCompleted(selectedRoadmap.id, problem.id)}
                          onClick={(event) => event.stopPropagation()}
                          className="mt-0.5 h-4 w-4 accent-emerald-400"
                        />
                      </div>
                    </button>
                      )
                    })()
                  ))}
                </div>
                <div className="text-xs text-slate-500">Click to highlight on canvas (coming soon).</div>
              </div>
            </div>
          )}
        </div>
        </Panel>
      </div>

      <div className="fixed z-30 right-6 top-20 flex flex-col items-end gap-3">
        <Panel className={roadmapPillPanelClass}>
          <button
            type="button"
            onClick={() => {
              setOpen((value) => !value)
              setRoadmapsOpen(false)
              setGenerateError(null)
            }}
            className={roadmapPillClass}
          >
            Generate Roadmap
          </button>
        </Panel>

        <Panel
          className={`w-96 rounded-2xl transition-all duration-300 origin-top bg-panel/95 overflow-hidden flex flex-col
            ${open ? 'opacity-100 translate-y-0 max-h-[70vh]' : 'opacity-0 -translate-y-2 max-h-0 pointer-events-none'}`}
        >
        <div className="flex min-h-0 flex-1 flex-col px-4 py-4 text-base text-slate-200">
        <div className="min-h-0 flex-1 space-y-4 overflow-y-auto pr-1">
          <div>
            <label className="block text-[11px] uppercase tracking-[0.2em] text-slate-400">
              Choose your patterns
            </label>
            <div className="mt-2 flex flex-wrap gap-2">
              <Button
                className={selectionPillClass(patternSelectionMode === 'in_progress')}
                onClick={() => setPatternSelectionMode('in_progress')}
              >
                All In Progress
              </Button>
              <Button
                className={selectionPillClass(patternSelectionMode === 'in_progress_and_completed')}
                onClick={() => setPatternSelectionMode('in_progress_and_completed')}
              >
                In Progress + Completed
              </Button>
              <Button
                className={selectionPillClass(patternSelectionMode === 'custom')}
                onClick={() => setPatternSelectionMode('custom')}
              >
                Custom
              </Button>
            </div>
          </div>

          {patternSelectionMode === 'custom' &&
            (
              <PatternDropdown
                label="Custom pattern set"
                selectedIds={customPatternIds}
                onChange={setCustomPatternIds}
                options={sortedPatterns}
              />
            )}

          <PatternDropdown
            label="Priority patterns"
            selectedIds={priorityPatternIds}
            onChange={setPriorityPatternIds}
            options={sortedPatterns}
          />

          <PatternDropdown
            label="Exclude patterns"
            selectedIds={excludedPatternIds}
            onChange={setExcludedPatternIds}
            options={sortedPatterns}
          />

          <div>
            <label className="block text-[11px] uppercase tracking-[0.2em] text-slate-400">
              Difficulty preference
            </label>
            <div className="mt-2 flex flex-wrap gap-2">
              <Button
                className={selectionPillClass(difficultyPreference.easy)}
                onClick={() =>
                  setDifficultyPreference((prev) => ({ ...prev, easy: !prev.easy }))
                }
              >
                Easy
              </Button>
              <Button
                className={selectionPillClass(difficultyPreference.medium)}
                onClick={() =>
                  setDifficultyPreference((prev) => ({ ...prev, medium: !prev.medium }))
                }
              >
                Medium
              </Button>
              <Button
                className={selectionPillClass(difficultyPreference.hard)}
                onClick={() =>
                  setDifficultyPreference((prev) => ({ ...prev, hard: !prev.hard }))
                }
              >
                Hard
              </Button>
            </div>
          </div>

          <div>
            <label className="block text-[11px] uppercase tracking-[0.2em] text-slate-400">
              Amount of problems
            </label>
            <input
              type="text"
              placeholder="e.g. 30"
              value={amountOfProblems}
              onChange={(event) => setAmountOfProblems(event.target.value)}
              className="mt-2 w-full rounded-lg bg-ink/40 border border-white/10 px-3 py-2 text-sm text-slate-100 focus:outline-none focus:ring-2 focus:ring-amber-400/40"
            />
          </div>
        </div>
        <div className="pt-4">
          {generateError && (
            <div className="mb-2 rounded-xl border border-amber-300/40 bg-amber-500/10 px-3 py-2 text-xs text-amber-100">
              {generateError}
            </div>
          )}
          <Button
            className={`w-full ${selectionPillClass(true)}`}
            onClick={() => {
              const didSend = onGenerate?.({
                patternSelectionMode,
                customPatternIds,
                priorityPatternIds,
                excludedPatternIds,
                difficultyPreference,
                amountOfProblems,
              })
              if (!didSend) {
                setGenerateError(
                  "No patterns selected. Mark nodes as ‘In Progress’ or select patterns before generating a roadmap.",
                )
                return
              }
              setGenerateError(null)
              setOpen(false)
            }}
          >
            Give me my roadmap
          </Button>
        </div>
        </div>
        </Panel>
      </div>
    </>
  )
}

/* =========================
   DRAWER
   ========================= */

export const DrawerToggleButton: React.FC<{
  open: boolean
  onToggle: () => void
}> = ({ open, onToggle }) => (
  <Button onClick={onToggle} className="w-full px-4 py-3 text-sm rounded-xl2 flex items-center justify-center gap-2">
    <span className="text-accent-amber">•</span>
    {open ? '(coming soon!) Close shapes and tools' : '(coming soon!) Open shapes and tools'}
  </Button>
)

export const LeftDrawer: React.FC<
  React.PropsWithChildren<{
    open: boolean
    onToggle: () => void
    title?: string
    subtitle?: string
  }>
> = ({ open, onToggle, title, subtitle, children }) => (
  <div className="fixed z-30 top-24 left-6 w-64">
    <DrawerToggleButton open={open} onToggle={onToggle} />

    <Panel
      className={`mt-3 w-full rounded-2xl transition-all duration-300 origin-top
        ${
          open
            ? 'opacity-100 translate-y-0'
            : 'opacity-0 -translate-y-3 pointer-events-none'
        }`}
    >
      {(title || subtitle) && (
        <div className="px-4 py-3 border-b border-white/5">
          {title && <p className="text-sm font-semibold text-slate-100">{title}</p>}
          {subtitle && <p className="text-xs text-slate-400">{subtitle}</p>}
        </div>
      )}
      <div className="px-4 py-3">{children}</div>
    </Panel>
  </div>
)

/* =========================
   CONTEXT PALETTE
   ========================= */

export const PaletteColorSwatch: React.FC<{
  color: string
  onClick?: () => void
}> = ({ color, onClick }) => (
  <button
    onClick={onClick}
    className="w-8 h-8 rounded-full border border-white/15 hover:scale-105 transition"
    style={{ backgroundColor: color }}
  />
)

export const ContextPalette: React.FC<{
  x: number
  y: number
  children: React.ReactNode
}> = ({ x, y, children }) => (
  <div
    className="fixed z-40"
    style={{
      left: x,
      top: y,
      transform: 'translate(-50%, -100%)',
    }}
  >
    <Panel className="rounded-xl2 px-3 py-2 w-64 flex flex-col gap-2">{children}</Panel>
  </div>
)

/* =========================
   EXPORTS
   ========================= */

export default {
  AppShell,
  Panel,
  Button,
  IconButton,
  TopBar,
  RightToolbar,
  LandingOverlay,
  InteractionModeIndicator,
  LeftDrawer,
  DrawerToggleButton,
  ContextPalette,
  PaletteColorSwatch,
}
