import { clsx } from 'clsx'
import ArchetypeBadge from '../ArchetypeBadge'
import CitationBar from './CitationBar'
import { createCitationRegistry } from './useCitationRegistry'
import type { CitationEntry } from './useCitationRegistry'
import { REPORT_SCALAR_KEYS, humanLabel } from '../../lib/reportConstants'
import {
  isRecord,
  toCrossVendorBattles,
  toKeyInsights,
  toObjectionHandlers,
  toPainQuotes,
  toRecommendedPlays,
  toTalkTrack,
  toWeaknessAnalysis,
} from '../../lib/reportViewModels'
import type {
  CrossVendorBattleViewModel,
  KeyInsightViewModel,
  ObjectionHandlerViewModel,
  PainQuoteViewModel,
  RecommendedPlayViewModel,
  TalkTrackViewModel,
  WeaknessAnalysisItemViewModel,
} from '../../types/reportViewModels'

type AnyObject = Record<string, unknown>
type SectionEvidenceState = 'witness_backed' | 'partial' | 'thin'
type SectionEvidenceSummary = {
  state: SectionEvidenceState
  label: string
  detail: string
  witness_count?: number
  metric_count?: number
}

const QUOTE_KEYS = new Set([
  'anonymized_quotes', 'quotable_evidence', 'quotes', 'top_pain_quotes',
])

const SUMMARY_LABEL_KEYS = [
  'label', 'name', 'title', 'company', 'vendor', 'competitor', 'opponent',
  'category', 'feature', 'role', 'pain_category', 'stage',
]

const SUMMARY_DETAIL_KEYS = [
  'summary', 'description', 'conclusion', 'evidence', 'quote', 'text',
  'key_message', 'timing', 'resource_advantage', 'battle_conclusion',
]

const SUMMARY_META_KEYS = [
  'count', 'mentions', 'mention_count', 'switch_count', 'confidence',
  'urgency', 'score', 'evidence_count',
]

const REFERENCE_KEY_SUFFIXES = ['_reference_ids', '_witness_highlights']

function isEvidenceMetadataKey(key: string): boolean {
  return key === 'reference_ids'
    || key === 'reasoning_reference_ids'
    || key === 'witness_highlights'
    || key === 'reasoning_witness_highlights'
    || REFERENCE_KEY_SUFFIXES.some((suffix) => key.endsWith(suffix))
}

function isScalarValue(value: unknown): boolean {
  return value === null || value === undefined || ['string', 'number', 'boolean'].includes(typeof value)
}

function formatValue(value: unknown): string {
  if (value === null || value === undefined || value === '') return '--'
  if (typeof value === 'number') {
    if (value > 0 && value < 1) return `${(value * 100).toFixed(1)}%`
    if (Number.isInteger(value)) return value.toLocaleString()
    return value.toFixed(1)
  }
  if (typeof value === 'boolean') return value ? 'Yes' : 'No'
  return String(value)
}

function getRenderableColumns(rows: AnyObject[]): string[] {
  return Array.from(new Set(rows.flatMap((row) => Object.keys(row)))).filter((key) =>
    rows.some((row) => isScalarValue(row[key])),
  )
}

function firstScalarText(value: unknown): string | undefined {
  if (typeof value === 'string' && value.trim()) return value.trim()
  if (typeof value === 'number' || typeof value === 'boolean') return String(value)
  if (!isRecord(value)) return undefined
  for (const entry of Object.values(value)) {
    const nested = firstScalarText(entry)
    if (nested) return nested
  }
  return undefined
}

function getSummaryText(obj: AnyObject, keys: string[]): string | undefined {
  for (const key of keys) {
    const value = firstScalarText(obj[key])
    if (value) return value
  }
  return undefined
}

function getSummaryMeta(obj: AnyObject): string[] {
  return SUMMARY_META_KEYS.flatMap((key) => {
    const value = obj[key]
    if (!isScalarValue(value) || value === null || value === undefined || value === '') return []
    return [`${humanLabel(key)}: ${formatValue(value)}`]
  }).slice(0, 3)
}

function isRankedList(value: unknown): value is AnyObject[] {
  if (!Array.isArray(value) || value.length === 0) return false
  return value.every((item) => isRecord(item) && ['count', 'mentions', 'mention_count'].some((key) => key in item))
}

function getRankCount(item: AnyObject): number {
  return Number(item.count ?? item.mentions ?? item.mention_count ?? 0)
}

function extractWitnessIds(value: unknown): string[] {
  if (!isRecord(value) || !Array.isArray(value.witness_ids)) return []
  return value.witness_ids.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
}

function extractWitnessHighlights(value: unknown): AnyObject[] {
  if (!Array.isArray(value)) return []
  return value.filter(isRecord).filter((item) => typeof item.witness_id === 'string' && item.witness_id.trim().length > 0)
}

function extractMetricIds(value: unknown): string[] {
  if (!isRecord(value) || !Array.isArray(value.metric_ids)) return []
  return value.metric_ids.filter((item): item is string => typeof item === 'string' && item.trim().length > 0)
}

function collectSectionEvidence(
  fieldKey: string,
  value: unknown,
  data: AnyObject,
): {
  sectionReferenceSources: unknown[]
  sectionHighlightSources: unknown[]
  witnessIds: string[]
  metricIds: string[]
  highlights: AnyObject[]
} {
  const sectionObject = isRecord(value) ? value : null
  const sectionReferenceSources = [
    sectionObject?.reasoning_reference_ids,
    sectionObject?.reference_ids,
    data[`${fieldKey}_reference_ids`],
  ]
  const sectionHighlightSources = [
    sectionObject?.reasoning_witness_highlights,
    sectionObject?.witness_highlights,
    data[`${fieldKey}_witness_highlights`],
  ]
  const highlights = [
    ...sectionHighlightSources,
    data.reasoning_witness_highlights,
    data.witness_highlights,
  ].flatMap(extractWitnessHighlights)
  const sectionWitnessIds = sectionHighlightSources
    .flatMap(extractWitnessHighlights)
    .map((item) => String(item.witness_id))
  const witnessIds = [
    ...sectionReferenceSources.flatMap(extractWitnessIds),
    ...sectionWitnessIds,
  ]
  const metricIds = sectionReferenceSources.flatMap(extractMetricIds)

  return {
    sectionReferenceSources,
    sectionHighlightSources,
    witnessIds: Array.from(new Set(witnessIds)),
    metricIds: Array.from(new Set(metricIds)),
    highlights,
  }
}

function sectionEvidenceMeta(
  fieldKey: string,
  value: unknown,
  data: AnyObject,
): SectionEvidenceSummary & { className: string } {
  const evidence = collectSectionEvidence(fieldKey, value, data)
  if (evidence.witnessIds.length > 0) {
    const witnessCount = evidence.witnessIds.length
    return {
      state: 'witness_backed',
      label: 'Witness-backed',
      detail: `${witnessCount} linked witness citation${witnessCount === 1 ? '' : 's'}`,
      className: 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30',
    }
  }

  const hasSectionMetadata = evidence.metricIds.length > 0
    || evidence.sectionReferenceSources.some((item) => isRecord(item))
    || evidence.sectionHighlightSources.some((item) => Array.isArray(item))

  if (hasSectionMetadata) {
    return {
      state: 'partial',
      label: 'Partial evidence',
      detail: 'Section has evidence metadata, but no linked witness citations yet.',
      className: 'bg-amber-500/15 text-amber-300 border border-amber-500/30',
    }
  }

  return {
    state: 'thin',
    label: 'Thin evidence',
    detail: 'No linked witness citations for this section yet.',
    className: 'bg-rose-500/15 text-rose-300 border border-rose-500/30',
  }
}

function decorateSectionEvidenceMeta(
  evidence: SectionEvidenceSummary,
): SectionEvidenceSummary & { className: string } {
  if (evidence.state === 'witness_backed') {
    return {
      ...evidence,
      className: 'bg-emerald-500/15 text-emerald-300 border border-emerald-500/30',
    }
  }
  if (evidence.state === 'partial') {
    return {
      ...evidence,
      className: 'bg-amber-500/15 text-amber-300 border border-amber-500/30',
    }
  }
  return {
    ...evidence,
    className: 'bg-rose-500/15 text-rose-300 border border-rose-500/30',
  }
}

function citationEntriesForSection(
  registry: ReturnType<typeof createCitationRegistry>,
  fieldKey: string,
  value: unknown,
  data: AnyObject,
): CitationEntry[] {
  const { witnessIds, highlights } = collectSectionEvidence(fieldKey, value, data)

  if (witnessIds.length === 0) return []

  const highlightMap = new Map<string, AnyObject>()
  for (const highlight of highlights) {
    highlightMap.set(String(highlight.witness_id), highlight)
  }

  for (const witnessId of witnessIds) {
    const highlight = highlightMap.get(witnessId)
    registry.register(witnessId, {
      companyName: typeof highlight?.reviewer_company === 'string' ? highlight.reviewer_company : undefined,
      excerptSnippet: typeof highlight?.excerpt_text === 'string' ? highlight.excerpt_text.slice(0, 80) : undefined,
      source: typeof highlight?.source === 'string' ? highlight.source : undefined,
    })
  }

  return witnessIds
    .map((witnessId) => registry.getAll().find((item) => item.witnessId === witnessId) ?? { index: 0, witnessId })
    .filter((entry) => entry.index > 0)
}

function StructuredTable({ rows }: { rows: AnyObject[] }) {
  if (rows.length === 0) return null
  const columns = getRenderableColumns(rows)
  if (columns.length === 0) return null

  return (
    <div className="overflow-x-auto min-w-0">
      <table className="w-full text-sm table-auto">
        <thead>
          <tr className="border-b border-slate-700/50">
            {columns.map((column) => (
              <th key={column} className="text-left text-xs font-medium text-slate-400 uppercase tracking-wider px-3 py-2 align-top break-words">
                {humanLabel(column)}
              </th>
            ))}
          </tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={index} className="border-b border-slate-800/50 hover:bg-slate-800/30">
              {columns.map((column) => {
                const value = row[column]
                if (column === 'archetype' && typeof value === 'string' && value) {
                  return (
                    <td key={column} className="px-3 py-2 align-top break-words">
                      <ArchetypeBadge archetype={value} confidence={typeof row.archetype_confidence === 'number' ? row.archetype_confidence : undefined} showConfidence />
                    </td>
                  )
                }
                return (
                  <td key={column} className="px-3 py-2 text-slate-300 align-top break-words">
                    {formatValue(value)}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

function ObjectSummaryCard({ obj, index }: { obj: AnyObject; index: number }) {
  const title = getSummaryText(obj, SUMMARY_LABEL_KEYS) ?? `Item ${index + 1}`
  const detail = getSummaryText(obj, SUMMARY_DETAIL_KEYS)
  const meta = getSummaryMeta(obj)
  const remaining = Object.entries(obj)
    .filter(([key, value]) => !SUMMARY_LABEL_KEYS.includes(key) && !SUMMARY_DETAIL_KEYS.includes(key) && isScalarValue(value) && value !== null && value !== undefined && value !== '')
    .slice(0, 3)

  return (
    <div className="bg-slate-800/50 rounded-lg p-3 space-y-2">
      <p className="text-sm text-slate-200 font-medium break-words">{title}</p>
      {detail && detail !== title && <p className="text-xs text-slate-400 break-words">{detail}</p>}
      {meta.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {meta.map((item) => (
            <span key={item} className="px-2 py-0.5 bg-slate-900/70 rounded text-[11px] text-slate-300 max-w-full break-all whitespace-normal">
              {item}
            </span>
          ))}
        </div>
      )}
      {remaining.length > 0 && (
        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-1.5 text-xs">
          {remaining.map(([key, value]) => (
            <div key={key} className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-2">
              <dt className="text-slate-500 min-w-0 break-words">{humanLabel(key)}</dt>
              <dd className="text-slate-300 text-right min-w-0 break-words">{formatValue(value)}</dd>
            </div>
          ))}
        </dl>
      )}
    </div>
  )
}

function MixedArrayList({ items }: { items: unknown[] }) {
  return (
    <div className="space-y-2">
      {items.slice(0, 8).map((item, index) => {
        if (isRecord(item)) return <ObjectSummaryCard key={index} obj={item} index={index} />
        return (
          <div key={index} className="bg-slate-800/50 rounded-lg px-3 py-2 text-sm text-slate-300 break-all">
            {formatValue(item)}
          </div>
        )
      })}
      {items.length > 8 && (
        <p className="text-xs text-slate-500">Showing 8 of {items.length} items</p>
      )}
    </div>
  )
}

function RankedList({ items }: { items: AnyObject[] }) {
  const maxCount = Math.max(...items.map((item) => getRankCount(item)), 1)
  return (
    <div className="space-y-1.5">
      {items.map((item, index) => {
        const label = String(item.category ?? item.name ?? item.feature ?? item.role ?? item.competitor ?? `#${index + 1}`)
        const count = getRankCount(item)
        const pct = Math.max(5, Math.round((count / maxCount) * 100))
        return (
          <div key={index}>
            <div className="flex items-start justify-between gap-2 text-xs mb-0.5">
              <span className="text-slate-300 min-w-0 break-words">{label}</span>
              <span className="text-slate-400 shrink-0">{count}</span>
            </div>
            <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
              <div className="h-full bg-cyan-500/60 rounded-full" style={{ width: `${pct}%` }} />
            </div>
          </div>
        )
      })}
    </div>
  )
}

function StringList({ items, asQuotes }: { items: string[]; asQuotes?: boolean }) {
  if (asQuotes) {
    return (
      <div className="space-y-2">
        {items.map((item, index) => (
          <blockquote key={index} className="text-sm text-slate-300 italic border-l-2 border-cyan-500/50 pl-3">
            {item}
          </blockquote>
        ))}
      </div>
    )
  }
  return (
    <div className="flex flex-wrap gap-1.5">
      {items.map((item, index) => (
        <span key={index} className="px-2 py-0.5 bg-slate-800 rounded text-xs text-slate-300 max-w-full break-all whitespace-normal">
          {item}
        </span>
      ))}
    </div>
  )
}

function InsightList({ items }: { items: KeyInsightViewModel[] }) {
  return (
    <div className="space-y-3">
      {items.map((item, index) => (
        <div key={index} className="bg-slate-800/50 rounded-lg p-3">
          <p className="text-sm text-slate-200 break-words">{item.insight}</p>
          {typeof item.evidence === 'string' && item.evidence.trim() && (
            <p className="text-xs text-slate-500 mt-1 break-all">{item.evidence}</p>
          )}
        </div>
      ))}
    </div>
  )
}

function BattleList({ items }: { items: CrossVendorBattleViewModel[] }) {
  return (
    <div className="space-y-3">
      {items.map((item, index) => (
        <div key={index} className="bg-slate-800/50 rounded-lg p-3 space-y-2">
          <div className="flex flex-wrap items-center gap-2 text-xs">
            {typeof item.opponent === 'string' && <span className="text-white font-medium">vs {item.opponent}</span>}
            {typeof item.winner === 'string' && <span className="px-2 py-0.5 bg-emerald-500/15 text-emerald-300 rounded">winner: {item.winner}</span>}
            {typeof item.loser === 'string' && <span className="px-2 py-0.5 bg-red-500/15 text-red-300 rounded">loser: {item.loser}</span>}
            {typeof item.durability === 'string' && <span className="text-slate-500">{item.durability}</span>}
          </div>
          {typeof item.conclusion === 'string' && (
            <p className="text-sm text-slate-300">{item.conclusion}</p>
          )}
          {Array.isArray(item.key_insights) && item.key_insights.length > 0 && (
            <InsightList items={item.key_insights.filter(isRecord)} />
          )}
        </div>
      ))}
    </div>
  )
}

function PainQuotes({ items }: { items: PainQuoteViewModel[] }) {
  return (
    <div className="space-y-3">
      {items.map((item, index) => (
        <div key={index} className="border-l-2 border-cyan-500/50 pl-3">
          <blockquote className="text-sm text-slate-300 italic break-words whitespace-pre-wrap">"{String(item.quote ?? item.text ?? '')}"</blockquote>
          <div className="flex flex-wrap gap-3 mt-1 text-xs text-slate-500">
            {typeof item.company === 'string' && <span>{item.company}</span>}
            {typeof item.role === 'string' && <span>{item.role}</span>}
            {typeof item.title === 'string' && <span>{item.title}</span>}
            {typeof item.pain_category === 'string' && <span>{item.pain_category}</span>}
            {Number(item.urgency ?? 0) > 0 && <span className="text-amber-400">urgency {String(item.urgency)}/10</span>}
          </div>
        </div>
      ))}
    </div>
  )
}

function ObjectionHandlers({ items }: { items: ObjectionHandlerViewModel[] }) {
  return (
    <div className="space-y-3">
      {items.map((item, index) => (
        <div key={index} className="bg-slate-800/50 rounded-lg p-3 space-y-1.5">
          {typeof item.objection === 'string' && <p className="text-sm text-red-300">{item.objection}</p>}
          {typeof item.acknowledge === 'string' && <p className="text-sm text-slate-300">{item.acknowledge}</p>}
          {typeof item.pivot === 'string' && <p className="text-sm text-green-300">{item.pivot}</p>}
          {typeof item.proof_point === 'string' && <p className="text-xs text-slate-500">{item.proof_point}</p>}
        </div>
      ))}
    </div>
  )
}

function TalkTrack({ obj }: { obj: TalkTrackViewModel }) {
  const sections = [
    ['opening', 'Opening', 'border-cyan-500/50'],
    ['mid_call_pivot', 'Mid-Call Pivot', 'border-amber-500/50'],
    ['proof_points', 'Proof Points', 'border-amber-500/50'],
    ['closing', 'Closing', 'border-green-500/50'],
  ] as const
  return (
    <div className="space-y-3">
      {sections.map(([key, label, color]) => {
        const value = obj[key]
        if (!value) return null
        return (
          <div key={key} className={`border-l-2 ${color} pl-3`}>
            <p className="text-xs text-slate-500 uppercase mb-1">{label}</p>
            {typeof value === 'string' && <p className="text-sm text-slate-300">{value}</p>}
            {Array.isArray(value) && <StringList items={value.map((item) => String(item))} />}
          </div>
        )
      })}
    </div>
  )
}

function WeaknessAnalysis({ items }: { items: WeaknessAnalysisItemViewModel[] }) {
  return (
    <div className="space-y-3">
      {items.map((item, index) => (
        <div key={index} className="bg-slate-800/50 rounded-lg p-3 space-y-1.5">
          <p className="text-sm text-white font-medium">{String(item.weakness ?? item.area ?? '')}</p>
          {typeof item.evidence === 'string' && <p className="text-sm text-slate-400">{item.evidence}</p>}
          {typeof item.customer_quote === 'string' && item.customer_quote.trim() && (
            <blockquote className="text-sm text-slate-300 italic border-l-2 border-cyan-500/50 pl-3 break-words whitespace-pre-wrap">{item.customer_quote}</blockquote>
          )}
          {typeof item.winning_position === 'string' && <p className="text-sm text-cyan-300">{item.winning_position}</p>}
          {typeof item.recommendation === 'string' && <p className="text-sm text-cyan-300">{item.recommendation}</p>}
        </div>
      ))}
    </div>
  )
}

function RecommendedPlays({ items }: { items: RecommendedPlayViewModel[] }) {
  return (
    <div className="space-y-3">
      {items.map((item, index) => (
        <div key={index} className="bg-slate-800/50 rounded-lg p-3 space-y-1">
          <p className="text-sm text-slate-200">{String(item.play ?? item.description ?? '')}</p>
          {typeof item.target_segment === 'string' && <p className="text-xs text-slate-500">Segment: {item.target_segment}</p>}
          {typeof item.key_message === 'string' && <p className="text-xs text-cyan-300">{item.key_message}</p>}
          {typeof item.timing === 'string' && <p className="text-xs text-slate-500">Timing: {item.timing}</p>}
        </div>
      ))}
    </div>
  )
}

function UnknownFallback({ value }: { value: unknown }) {
  const keys = isRecord(value) ? Object.keys(value) : []
  const summary = Array.isArray(value)
    ? `Structured list with ${value.length} item${value.length === 1 ? '' : 's'}`
    : isRecord(value)
      ? `Structured object with ${keys.length} field${keys.length === 1 ? '' : 's'}`
      : 'Structured value'

  return (
    <div className="space-y-2">
      <p className="text-sm text-slate-300">{summary}</p>
      {keys.length > 0 && (
        <div className="flex flex-wrap gap-1.5">
          {keys.slice(0, 8).map((key) => (
            <span key={key} className="px-2 py-0.5 bg-slate-800 rounded text-xs text-slate-300 max-w-full break-all whitespace-normal">
              {humanLabel(key)}
            </span>
          ))}
        </div>
      )}
      <p className="text-xs text-slate-500">
        This field is structured data without a dedicated card yet, but it is no longer rendered as raw JSON.
      </p>
    </div>
  )
}

function MixedObjectCard({ obj, label }: { obj: AnyObject; label?: string }) {
  const scalars = Object.entries(obj).filter(([, value]) => isScalarValue(value) && value !== null && value !== undefined)
  const nested = Object.entries(obj).filter(([, value]) => !isScalarValue(value) && value !== null && value !== undefined)

  return (
    <div className="space-y-3">
      {label && <h5 className="text-xs font-medium text-cyan-400 uppercase tracking-wider">{label}</h5>}
      {scalars.length > 0 && (
        <dl className="grid grid-cols-1 sm:grid-cols-2 gap-2 text-sm">
          {scalars.map(([key, value]) => (
            <div key={key} className="grid grid-cols-[minmax(0,1fr)_minmax(0,1fr)] gap-2">
              <dt className="text-slate-400 min-w-0 break-words">{humanLabel(key)}</dt>
              <dd className="text-white font-medium text-right min-w-0 break-words">{formatValue(value)}</dd>
            </div>
          ))}
        </dl>
      )}
      {nested.map(([key, value]) => (
        <div key={key}>
          {Array.isArray(value)
            ? <StructuredReportValue fieldKey={key} value={value} />
            : isRecord(value)
              ? <MixedObjectCard obj={value} label={humanLabel(key)} />
              : null}
        </div>
      ))}
    </div>
  )
}

export function StructuredReportValue({ fieldKey, value }: { fieldKey: string; value: unknown }) {
  if (value === null || value === undefined) return <span className="text-sm text-slate-500">--</span>
  if (typeof value === 'string') return <p className="text-sm text-slate-300 whitespace-pre-wrap break-words">{value}</p>
  if (typeof value === 'number') return <span className="text-lg font-bold text-white">{formatValue(value)}</span>
  if (typeof value === 'boolean') return <span className="text-sm text-slate-300">{value ? 'Yes' : 'No'}</span>
  if (isRankedList(value)) return <RankedList items={value} />
  if (Array.isArray(value) && value.length > 0 && typeof value[0] === 'string') {
    return <StringList items={value.map((item) => String(item))} asQuotes={QUOTE_KEYS.has(fieldKey)} />
  }
  if (fieldKey === 'key_insights' && Array.isArray(value)) return <InsightList items={toKeyInsights(value)} />
  if (fieldKey === 'cross_vendor_battles' && Array.isArray(value)) return <BattleList items={toCrossVendorBattles(value)} />
  if (fieldKey === 'customer_pain_quotes' && Array.isArray(value)) return <PainQuotes items={toPainQuotes(value)} />
  if (fieldKey === 'objection_handlers' && Array.isArray(value)) return <ObjectionHandlers items={toObjectionHandlers(value)} />
  if ((fieldKey === 'weakness_analysis' || fieldKey === 'vendor_weaknesses') && Array.isArray(value)) {
    return <WeaknessAnalysis items={toWeaknessAnalysis(value)} />
  }
  if (fieldKey === 'recommended_plays' && Array.isArray(value)) return <RecommendedPlays items={toRecommendedPlays(value)} />
  if (fieldKey === 'talk_track' && isRecord(value)) {
    const talkTrack = toTalkTrack(value)
    return talkTrack ? <TalkTrack obj={talkTrack} /> : <span className="text-sm text-slate-500">--</span>
  }
  if (fieldKey === 'competitor_differentiators' && Array.isArray(value)) return <StructuredTable rows={value.filter(isRecord)} />
  if (Array.isArray(value) && value.every(isRecord)) {
    return getRenderableColumns(value).length > 0
      ? <StructuredTable rows={value} />
      : <MixedArrayList items={value} />
  }
  if (Array.isArray(value)) return <MixedArrayList items={value} />
  if (isRecord(value)) return <MixedObjectCard obj={value} />
  return <UnknownFallback value={value} />
}

export function StructuredReportData({
  data,
  skipKeys = [],
  className,
  vendorName,
  onOpenWitness,
  sectionEvidence,
}: {
  data: AnyObject
  skipKeys?: string[]
  className?: string
  vendorName?: string
  onOpenWitness?: (witnessId: string, vendorName: string) => void
  sectionEvidence?: Record<string, SectionEvidenceSummary> | null
}) {
  const registry = createCitationRegistry()
  const entries = Object.entries(data).filter(([key, value]) => {
    if (skipKeys.includes(key) || REPORT_SCALAR_KEYS.has(key) || isEvidenceMetadataKey(key)) return false
    if (key === 'executive_summary') return false
    if (value === null || value === undefined) return false
    if (typeof value === 'string' && value.trim() === '') return false
    if (Array.isArray(value) && value.length === 0) return false
    if (isRecord(value) && Object.keys(value).length === 0) return false
    return true
  })

  if (entries.length === 0) return null

  return (
    <div className={clsx('grid grid-cols-1 xl:grid-cols-2 gap-6 min-w-0', className)}>
      {entries.map(([key, value]) => {
        const evidence = decorateSectionEvidenceMeta(
          sectionEvidence?.[key] ?? sectionEvidenceMeta(key, value, data),
        )
        const citations = vendorName && onOpenWitness
          ? citationEntriesForSection(registry, key, value, data)
          : []
        return (
          <div key={key} className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 min-w-0 overflow-hidden [overflow-wrap:anywhere]">
            <div className="flex flex-wrap items-center gap-2 mb-3">
              <h4 className="text-xs font-medium text-cyan-400 uppercase tracking-wider break-words">
                {humanLabel(key)}
              </h4>
              <span className={clsx('px-2 py-0.5 rounded text-[11px] font-medium', evidence.className)}>
                {evidence.label}
              </span>
            </div>
            <StructuredReportValue fieldKey={key} value={value} />
            <p className="mt-3 text-xs text-slate-500">
              {evidence.detail}
            </p>
            {vendorName && onOpenWitness && citations.length > 0 && (
              <div className="mt-3">
                <CitationBar
                  citations={citations}
                  vendorName={vendorName}
                  onOpenWitness={onOpenWitness}
                />
              </div>
            )}
          </div>
        )
      })}
    </div>
  )
}
