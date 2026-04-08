export interface CitationEntry {
  index: number
  witnessId: string
  companyName?: string
  excerptSnippet?: string
  source?: string
}

export interface CitationRegistry {
  register(witnessId: string, meta?: { companyName?: string; excerptSnippet?: string; source?: string }): number
  getAll(): CitationEntry[]
  getIndex(witnessId: string): number | undefined
}

export function createCitationRegistry(): CitationRegistry {
  const entries = new Map<string, CitationEntry>()
  let counter = 0

  return {
    register(witnessId, meta) {
      const existing = entries.get(witnessId)
      if (existing) return existing.index
      counter += 1
      const entry: CitationEntry = {
        index: counter,
        witnessId,
        companyName: meta?.companyName,
        excerptSnippet: meta?.excerptSnippet,
        source: meta?.source,
      }
      entries.set(witnessId, entry)
      return counter
    },
    getAll() {
      return Array.from(entries.values())
    },
    getIndex(witnessId) {
      return entries.get(witnessId)?.index
    },
  }
}
