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

export const expandPatternIdsTwoWay = expandLinkedPatterns
