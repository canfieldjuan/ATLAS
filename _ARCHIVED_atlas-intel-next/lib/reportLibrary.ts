import type { Report, ReportDetail } from '@/lib/types'

type UnknownRecord = Record<string, unknown>

const TIMESTAMP_KEYS = new Set([
  'report_date',
  'created_at',
  'updated_at',
  'generated_at',
  'last_computed_at',
  'data_as_of_date',
  'as_of_date',
  'computed_at',
])

const WITNESS_ARRAY_KEYS = new Set([
  'reasoning_witness_highlights',
  'witness_highlights',
  'witness_pack',
])

const QUOTE_ARRAY_KEYS = new Set([
  'customer_pain_quotes',
  'top_pain_quotes',
  'evidence',
  'quotable_evidence',
])

export interface ReferenceIdsSummary {
  metricIds: string[]
  witnessIds: string[]
}

export interface WitnessCitation {
  key: string
  id?: string
  label?: string
  reviewerCompany?: string
  reviewerTitle?: string
  excerptText?: string
  timeAnchor?: string
  competitor?: string
  witnessType?: string
  selectionReason?: string
  salienceScore?: number | null
  numericTokens: string[]
}

export interface QuoteCitation {
  key: string
  quote: string
  company?: string
  role?: string
  sourceSite?: string
  urgency?: number | null
  painCategory?: string
}

export interface ReportEvidenceSummary {
  witnesses: WitnessCitation[]
  quotes: QuoteCitation[]
  referenceIds: ReferenceIdsSummary
  reasoningSources: string[]
  qualityStatus?: string
  qualityScore?: number | null
  qualityFailedChecks: string[]
  qualityWarnings: string[]
  llmRenderStatus?: string
  dataDensityStatus?: string
}

export interface ReportFreshness {
  state: 'fresh' | 'monitor' | 'stale' | 'unknown'
  label: string
  badgeClass: string
  textClass: string
  detail: string
  ageHours: number | null
  anchor: string | null
}

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function asRecord(value: unknown): UnknownRecord {
  return isRecord(value) ? value : {}
}

function asString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value.trim() : undefined
}

function asNumber(value: unknown): number | null | undefined {
  return typeof value === 'number' && !Number.isNaN(value) ? value : undefined
}

function toStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value
    .map((item) => asString(item))
    .filter((item): item is string => Boolean(item))
}

function parseDateCandidate(value: unknown): string | null {
  if (typeof value !== 'string' || !value.trim()) return null
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? null : date.toISOString()
}

function titleCase(value: string): string {
  return value
    .replace(/_/g, ' ')
    .replace(/\b\w/g, (character) => character.toUpperCase())
}

function formatRelativeAge(ageHours: number): string {
  if (ageHours < 24) return `${Math.max(1, Math.round(ageHours))}h ago`
  if (ageHours < 24 * 7) return `${Math.round(ageHours / 24)}d ago`
  return `${Math.round(ageHours / (24 * 7))}w ago`
}

function collectTimestampCandidates(value: unknown, results: string[], depth = 0, seen = new Set<unknown>()) {
  if (depth > 5) return
  if (value === null || value === undefined) return
  if (typeof value !== 'object') return
  if (seen.has(value)) return
  seen.add(value)

  if (Array.isArray(value)) {
    value.forEach((item) => collectTimestampCandidates(item, results, depth + 1, seen))
    return
  }

  for (const [key, child] of Object.entries(value)) {
    if (TIMESTAMP_KEYS.has(key)) {
      const parsed = parseDateCandidate(child)
      if (parsed) results.push(parsed)
    }
    if (typeof child === 'object' && child !== null) {
      collectTimestampCandidates(child, results, depth + 1, seen)
    }
  }
}

function metricIdsSetToArray(values: Set<string>): string[] {
  return Array.from(values).sort((left, right) => left.localeCompare(right))
}

function sanitizeFilename(value: string): string {
  return value
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, '-')
    .replace(/^-+|-+$/g, '')
    .slice(0, 80) || 'report'
}

function csvCell(value: string | null | undefined): string {
  const text = value ?? ''
  return `"${text.replace(/"/g, '""')}"`
}

function downloadBlob(filename: string, mimeType: string, content: string) {
  const blob = new Blob([content], { type: mimeType })
  const url = window.URL.createObjectURL(blob)
  const anchor = document.createElement('a')
  anchor.href = url
  anchor.download = filename
  anchor.click()
  window.URL.revokeObjectURL(url)
}

function qualityLabel(status: string | undefined): string {
  switch (status) {
    case 'sales_ready':
      return 'Evidence-backed'
    case 'needs_review':
      return 'Needs review'
    case 'thin_evidence':
      return 'Thin evidence'
    case 'deterministic_fallback':
      return 'Fallback render'
    default:
      return 'Persisted artifact'
  }
}

function numericTokens(value: unknown): string[] {
  const tokens: string[] = []
  const record = asRecord(value)
  for (const [key, raw] of Object.entries(record)) {
    const values = Array.isArray(raw) ? raw : [raw]
    const formatted = values
      .map((item) => String(item ?? '').trim())
      .filter(Boolean)
      .join(', ')
    if (formatted) {
      tokens.push(`${titleCase(key)}: ${formatted}`)
    }
  }
  return tokens
}

function witnessLookupKey(id: string | undefined, reviewerCompany: string | undefined, excerptText: string | undefined): string | null {
  if (id) return id
  if (!reviewerCompany && !excerptText) return null
  return `${reviewerCompany ?? ''}:${excerptText ?? ''}`
}

function quoteLookupKey(quote: string, company?: string): string {
  return `${company ?? ''}:${quote}`
}

function looksLikeWitness(record: UnknownRecord): boolean {
  return Boolean(
    asString(record.witness_id)
      || asString(record._sid)
      || (asString(record.excerpt_text) && (asString(record.reviewer_company) || asString(record.selection_reason))),
  )
}

function looksLikeQuote(record: UnknownRecord): boolean {
  return Boolean(asString(record.quote) || asString(record.text))
}

export function formatReportTypeLabel(reportType: string): string {
  return titleCase(reportType)
}

export function reportDisplayTitle(report: Pick<Report, 'report_type' | 'vendor_filter' | 'category_filter'>): string {
  if (['vendor_comparison', 'account_comparison'].includes(report.report_type) && report.vendor_filter && report.category_filter) {
    return `${report.vendor_filter} vs ${report.category_filter}`
  }
  if (report.report_type === 'challenger_brief' && report.vendor_filter && report.category_filter) {
    return `${report.vendor_filter} -> ${report.category_filter}`
  }
  return report.vendor_filter ?? report.category_filter ?? formatReportTypeLabel(report.report_type)
}

export function summarizeReportTrust(report: Pick<Report, 'report_type' | 'quality_status' | 'status'>): {
  label: string
  detail: string
  toneClass: string
} {
  const qualityStatus = (report.quality_status ?? '').toLowerCase()
  if (report.report_type === 'battle_card' && qualityStatus) {
    if (qualityStatus === 'sales_ready') {
      return {
        label: 'Evidence-backed',
        detail: 'Battle card has quality coverage for customer-facing delivery.',
        toneClass: 'bg-emerald-500/15 text-emerald-300',
      }
    }
    if (qualityStatus === 'needs_review') {
      return {
        label: 'Operator review',
        detail: 'Evidence exists, but the artifact still needs human validation.',
        toneClass: 'bg-amber-500/15 text-amber-300',
      }
    }
    if (qualityStatus === 'thin_evidence') {
      return {
        label: 'Thin evidence',
        detail: 'Use with caution until more witness coverage lands.',
        toneClass: 'bg-slate-500/15 text-slate-300',
      }
    }
    return {
      label: 'Fallback render',
      detail: 'Persisted output exists, but trust depends on deterministic coverage.',
      toneClass: 'bg-rose-500/15 text-rose-300',
    }
  }

  if ((report.status ?? '').toLowerCase() === 'completed') {
    return {
      label: 'Persisted artifact',
      detail: 'This deliverable is stored for repeat access and subscription packaging.',
      toneClass: 'bg-cyan-500/15 text-cyan-300',
    }
  }

  return {
    label: 'In workflow',
    detail: 'Artifact exists, but the delivery state is not marked complete yet.',
    toneClass: 'bg-slate-500/15 text-slate-300',
  }
}

export function deriveFreshness(
  report: Pick<Report, 'report_date' | 'created_at'> & {
    intelligence_data?: unknown
    data_density?: Record<string, unknown> | null
  },
): ReportFreshness {
  const candidates = [
    parseDateCandidate(report.report_date),
    parseDateCandidate(report.created_at),
  ].filter((value): value is string => Boolean(value))

  collectTimestampCandidates(report.intelligence_data, candidates)
  collectTimestampCandidates(report.data_density, candidates)

  if (candidates.length === 0) {
    return {
      state: 'unknown',
      label: 'Unknown',
      badgeClass: 'bg-slate-500/15 text-slate-300',
      textClass: 'text-slate-400',
      detail: 'No freshness timestamp was attached to this artifact.',
      ageHours: null,
      anchor: null,
    }
  }

  const newest = candidates.sort((left, right) => new Date(right).getTime() - new Date(left).getTime())[0]
  const ageHours = (Date.now() - new Date(newest).getTime()) / (1000 * 60 * 60)

  if (ageHours <= 72) {
    return {
      state: 'fresh',
      label: 'Fresh',
      badgeClass: 'bg-emerald-500/15 text-emerald-300',
      textClass: 'text-emerald-300',
      detail: `Evidence window looks current (${formatRelativeAge(ageHours)}).`,
      ageHours,
      anchor: newest,
    }
  }

  if (ageHours <= 24 * 7) {
    return {
      state: 'monitor',
      label: 'Monitor',
      badgeClass: 'bg-amber-500/15 text-amber-300',
      textClass: 'text-amber-300',
      detail: `Artifact is aging (${formatRelativeAge(ageHours)}). Recheck before external use.`,
      ageHours,
      anchor: newest,
    }
  }

  return {
    state: 'stale',
    label: 'Stale',
    badgeClass: 'bg-rose-500/15 text-rose-300',
    textClass: 'text-rose-300',
    detail: `Refresh recommended. Latest evidence is ${formatRelativeAge(ageHours)} old.`,
    ageHours,
    anchor: newest,
  }
}

export function extractReportEvidence(
  intelligenceData: unknown,
  dataDensity?: Record<string, unknown> | null,
): ReportEvidenceSummary {
  const witnessMap = new Map<string, WitnessCitation>()
  const witnessKeyById = new Map<string, string>()
  const quoteMap = new Map<string, QuoteCitation>()
  const metricIds = new Set<string>()
  const witnessIds = new Set<string>()
  const reasoningSources = new Set<string>()
  const anchorIdsByLabel = new Map<string, string[]>()

  const root = asRecord(intelligenceData)
  const qualityRecord = asRecord(root.battle_card_quality)
  const qualityStatus = asString(root.quality_status ?? qualityRecord.status)
  const qualityScore = asNumber(root.quality_score ?? qualityRecord.score) ?? null
  const qualityFailedChecks = toStringArray(root.quality_failed_checks ?? qualityRecord.failed_checks)
  const qualityWarnings = toStringArray(root.quality_warnings ?? qualityRecord.warnings)
  const llmRenderStatus = asString(root.llm_render_status)
  const dataDensityStatus = asString(asRecord(dataDensity).status)

  function addReferenceIds(value: unknown) {
    const record = asRecord(value)
    toStringArray(record.metric_ids).forEach((id) => metricIds.add(id))
    toStringArray(record.witness_ids).forEach((id) => witnessIds.add(id))
  }

  function addWitness(value: unknown, label?: string) {
    const record = asRecord(value)
    const id = asString(record.witness_id ?? record._sid)
    const excerptText = asString(record.excerpt_text ?? record.quote ?? record.text)
    const reviewerCompany = asString(record.reviewer_company ?? record.company)
    const reviewerTitle = asString(record.reviewer_title ?? record.title ?? record.role)
    const key = witnessLookupKey(id, reviewerCompany, excerptText)
    if (!key) return

    const nextWitness: WitnessCitation = {
      key,
      id,
      label,
      reviewerCompany,
      reviewerTitle,
      excerptText,
      timeAnchor: asString(record.time_anchor),
      competitor: asString(record.competitor),
      witnessType: asString(record.witness_type),
      selectionReason: asString(record.selection_reason),
      salienceScore: asNumber(record.salience_score) ?? null,
      numericTokens: numericTokens(record.numeric_literals),
    }

    const current = witnessMap.get(key)
    witnessMap.set(key, {
      ...nextWitness,
      label: current?.label ?? nextWitness.label,
      numericTokens: current ? Array.from(new Set([...current.numericTokens, ...nextWitness.numericTokens])) : nextWitness.numericTokens,
    })
    if (id) {
      witnessIds.add(id)
      witnessKeyById.set(id, key)
    }
  }

  function addQuote(value: unknown) {
    if (typeof value === 'string' && value.trim()) {
      const key = quoteLookupKey(value, undefined)
      if (!quoteMap.has(key)) {
        quoteMap.set(key, { key, quote: value.trim() })
      }
      return
    }

    const record = asRecord(value)
    const quote = asString(record.quote ?? record.text)
    if (!quote) return
    const company = asString(record.company ?? record.reviewer_company)
    const key = quoteLookupKey(quote, company)
    if (quoteMap.has(key)) return

    quoteMap.set(key, {
      key,
      quote,
      company,
      role: asString(record.role ?? record.reviewer_title ?? record.title),
      sourceSite: asString(record.source_site ?? record.source),
      urgency: asNumber(record.urgency) ?? null,
      painCategory: asString(record.pain_category),
    })
  }

  function walk(value: unknown, parentKey = '') {
    if (Array.isArray(value)) {
      if (WITNESS_ARRAY_KEYS.has(parentKey)) {
        value.forEach((item) => addWitness(item))
        return
      }
      if (QUOTE_ARRAY_KEYS.has(parentKey)) {
        value.forEach((item) => addQuote(item))
      }
      value.forEach((item) => walk(item, parentKey))
      return
    }

    if (!isRecord(value)) return

    if (looksLikeWitness(value)) {
      addWitness(value)
    }
    if (looksLikeQuote(value)) {
      addQuote(value)
    }

    for (const [key, child] of Object.entries(value)) {
      if (key === 'reasoning_source') {
        const source = asString(child)
        if (source) reasoningSources.add(source)
      }

      if (key === 'reference_ids' || key === 'reasoning_reference_ids') {
        addReferenceIds(child)
      }

      if (key === 'reasoning_anchor_examples' || key === 'anchor_examples') {
        const anchors = asRecord(child)
        for (const [label, rows] of Object.entries(anchors)) {
          if (Array.isArray(rows)) {
            const witnessIdsForLabel = rows
              .map((item) => (typeof item === 'string' ? item.trim() : asString(asRecord(item).witness_id ?? asRecord(item)._sid)))
              .filter((item): item is string => Boolean(item))
            if (witnessIdsForLabel.length > 0) {
              anchorIdsByLabel.set(label, witnessIdsForLabel)
            }
            rows.forEach((item) => {
              if (typeof item !== 'string') addWitness(item, label)
            })
          } else {
            walk(rows, label)
          }
        }
        continue
      }

      if (typeof child === 'object' && child !== null) {
        walk(child, key)
      }
    }
  }

  walk(intelligenceData)

  for (const [label, ids] of anchorIdsByLabel.entries()) {
    for (const id of ids) {
      const witnessKey = witnessKeyById.get(id)
      if (!witnessKey) continue
      const witness = witnessMap.get(witnessKey)
      if (!witness || witness.label) continue
      witnessMap.set(witnessKey, { ...witness, label })
    }
  }

  const witnesses = Array.from(witnessMap.values()).sort((left, right) => {
    if (Boolean(left.label) !== Boolean(right.label)) return left.label ? -1 : 1
    const leftSalience = left.salienceScore ?? -1
    const rightSalience = right.salienceScore ?? -1
    if (leftSalience !== rightSalience) return rightSalience - leftSalience
    return (left.reviewerCompany ?? '').localeCompare(right.reviewerCompany ?? '')
  })

  const quotes = Array.from(quoteMap.values())

  return {
    witnesses,
    quotes,
    referenceIds: {
      metricIds: metricIdsSetToArray(metricIds),
      witnessIds: metricIdsSetToArray(witnessIds),
    },
    reasoningSources: Array.from(reasoningSources).sort((left, right) => left.localeCompare(right)),
    qualityStatus,
    qualityScore,
    qualityFailedChecks,
    qualityWarnings,
    llmRenderStatus,
    dataDensityStatus,
  }
}

export function exportReportsCsv(reports: Report[]) {
  const header = [
    'id',
    'title',
    'report_type',
    'vendor_filter',
    'category_filter',
    'trust_label',
    'freshness_label',
    'status',
    'quality_status',
    'report_date',
    'created_at',
    'executive_summary',
  ]

  const rows = reports.map((report) => {
    const trust = summarizeReportTrust(report)
    const freshness = deriveFreshness(report)
    return [
      csvCell(report.id),
      csvCell(reportDisplayTitle(report)),
      csvCell(formatReportTypeLabel(report.report_type)),
      csvCell(report.vendor_filter),
      csvCell(report.category_filter ?? null),
      csvCell(trust.label),
      csvCell(freshness.label),
      csvCell(report.status),
      csvCell(report.quality_status ?? null),
      csvCell(report.report_date),
      csvCell(report.created_at),
      csvCell(report.executive_summary),
    ].join(',')
  })

  downloadBlob(
    `report-library-${new Date().toISOString().slice(0, 10)}.csv`,
    'text/csv;charset=utf-8',
    `${header.join(',')}\n${rows.join('\n')}`,
  )
}

export function exportReportDetail(report: ReportDetail, format: 'json' | 'markdown') {
  const title = reportDisplayTitle(report)
  const baseName = sanitizeFilename(`${title}-${report.id}`)
  if (format === 'json') {
    downloadBlob(
      `${baseName}.json`,
      'application/json;charset=utf-8',
      JSON.stringify(report, null, 2),
    )
    return
  }

  const trust = summarizeReportTrust(report)
  const freshness = deriveFreshness(report)
  const lines = [
    `# ${title}`,
    '',
    `- Type: ${formatReportTypeLabel(report.report_type)}`,
    `- Trust: ${trust.label}`,
    `- Freshness: ${freshness.label}`,
    `- Status: ${report.status ?? '--'}`,
    `- Generated: ${report.report_date ?? report.created_at ?? '--'}`,
    `- Model: ${report.llm_model ?? '--'}`,
    '',
    '## Executive Summary',
    '',
    report.executive_summary ?? 'No executive summary available.',
    '',
    '## Data Density',
    '',
    '```json',
    JSON.stringify(report.data_density ?? {}, null, 2),
    '```',
    '',
    '## Intelligence Data',
    '',
    '```json',
    JSON.stringify(report.intelligence_data ?? {}, null, 2),
    '```',
    '',
  ]

  downloadBlob(
    `${baseName}.md`,
    'text/markdown;charset=utf-8',
    lines.join('\n'),
  )
}

export function qualityStatusLabel(status: string | undefined): string {
  return qualityLabel(status)
}
