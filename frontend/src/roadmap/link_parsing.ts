export type ParsedNodeLink = {
  patternId: number | null
  disambiguation?: string
}

const LINK_PREFIX = 'dsa://node/'

export function parseNodeLink(link: unknown): ParsedNodeLink {
  if (typeof link !== 'string') {
    return { patternId: null }
  }

  if (!link.toLowerCase().startsWith(LINK_PREFIX)) {
    return { patternId: null }
  }

  const [pathPart, queryPart] = link.slice(LINK_PREFIX.length).split('?', 2)
  const idValue = Number(pathPart)
  if (!Number.isFinite(idValue)) {
    return { patternId: null }
  }

  let disambiguation: string | undefined
  if (queryPart) {
    const params = new URLSearchParams(queryPart)
    const dis = params.get('dis')
    if (dis) {
      disambiguation = dis.trim()
    }
  }

  return { patternId: idValue, disambiguation }
}
