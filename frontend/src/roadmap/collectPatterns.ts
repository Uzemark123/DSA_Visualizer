import type {
  PatternNode,
  ProgressState,
  PatternSelectionMode,
  CollectedPatterns,
  GenerateRoadmapRequest,
} from './types'

export function collectPatternsForRequest(params: {
  patternNodes: PatternNode[]
  progressById: Record<string, ProgressState>
  patternSelectionMode: PatternSelectionMode
  customPatternIds: number[]
  priorityPatternIds: number[]
  excludedPatternIds: number[]
}): CollectedPatterns {
  const {
    patternNodes,
    progressById,
    patternSelectionMode,
    customPatternIds,
    priorityPatternIds,
    excludedPatternIds,
  } = params

  const includeSet = new Set<number>()
  const excludeSet = new Set<number>()
  const prioritySet = new Set<number>()

  for (const id of excludedPatternIds) {
    if (Number.isFinite(id)) {
      excludeSet.add(id)
    }
  }

  if (patternSelectionMode === 'custom') {
    for (const id of customPatternIds) {
      if (Number.isFinite(id)) {
        includeSet.add(id)
      }
    }
  } else {
    const includeCompleted = patternSelectionMode === 'in_progress_and_completed'
    for (const node of patternNodes) {
      const progress = progressById[node.elementId]
      if (progress === 'in_progress' || (includeCompleted && progress === 'completed')) {
        includeSet.add(node.patternId)
      }
    }
  }

  for (const id of priorityPatternIds) {
    if (Number.isFinite(id)) {
      prioritySet.add(id)
    }
  }

  for (const excludedId of excludeSet) {
    includeSet.delete(excludedId)
  }

  return {
    include: Array.from(includeSet),
    exclude: Array.from(excludeSet),
    priority: Array.from(prioritySet),
  }
}

export function buildGenerateRoadmapRequest(params: {
  collectedPatterns: CollectedPatterns
  difficulty: {
    easy: boolean
    medium: boolean
    hard: boolean
  }
  amountOfProblems: string
}): GenerateRoadmapRequest {
  const { collectedPatterns, difficulty, amountOfProblems } = params

  const parsedLimit = Number.parseInt(amountOfProblems, 10)
  const limit = Number.isFinite(parsedLimit) && parsedLimit > 0 ? parsedLimit : 20

  const expandLinkedPatterns = (ids: number[]) => {
    const expanded = new Set<number>()
    for (const id of ids) {
      if (id === 44 || id === 45) {
        expanded.add(44)
        expanded.add(45)
      } else {
        expanded.add(id)
      }
    }
    return Array.from(expanded)
  }

  const expandedExclude = expandLinkedPatterns(collectedPatterns.exclude)
  const expandedInclude = expandLinkedPatterns(collectedPatterns.include).filter(
    (id) => !expandedExclude.includes(id),
  )
  const expandedPriority = expandLinkedPatterns(collectedPatterns.priority).filter(
    (id) => !expandedExclude.includes(id),
  )

  return {
    user_requested_roadmap_length: limit,
    user_difficulty_selection: expandedInclude,
    user_priority_patterns: expandedPriority,
    user_excluded_patterns: expandedExclude,
    difficulty,
  }
}
