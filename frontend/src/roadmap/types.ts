// Progress state as used on the frontend
export type ProgressState = 'locked' | 'in_progress' | 'completed'

// Pattern node index entry (frontend-derived)
export type PatternNode = {
  elementId: string
  patternId: number
  label: string
  disambiguation?: string
}

// Selection mode from UI
export type PatternSelectionMode =
  | 'in_progress'
  | 'in_progress_and_completed'
  | 'custom'

// Output of pattern collection
export type CollectedPatterns = {
  include: number[]
  exclude: number[]
  priority: number[]
}

// Full API request payload
export type GenerateRoadmapRequest = {
  user_requested_roadmap_length: number
  user_difficulty_selection: number[]
  user_priority_patterns: number[]
  user_excluded_patterns: number[]
  difficulty: {
    easy: boolean
    medium: boolean
    hard: boolean
  }
}
