import { clsx } from 'clsx'
import ArchetypeBadge from '../ArchetypeBadge'
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

export const REPORT_SCALAR_KEYS = new Set([
  'vendor_name', 'challenger_name', 'primary_vendor', 'comparison_vendor', 'report_date', 'window_days',
  'signal_count', 'high_urgency_count', 'medium_urgency_count',
  'scope', 'llm_model', 'model_analysis', 'parse_fallback',
])

const QUOTE_KEYS = new Set([
  'anonymized_quotes', 'quotable_evidence', 'quotes', 'top_pain_quotes',
])

const FIELD_LABELS: Record<string, string> = {
  avg_urgency: 'Avg Urgency',
  budget_context: 'Budget Signals',
  churn_signal_density: 'Churn Signal Density',
  customer_pain_quotes: 'Customer Pain Quotes',
  decision_timeline: 'Decision Timeline',
  dm_churn_rate: 'DM Churn Rate',
  ecosystem_context: 'Ecosystem',
  key_insights: 'Key Insights',
  market_structure: 'Market Structure',
  objection_data: 'Objection Data',
  price_complaint_rate: 'Price Complaint Rate',
  sentiment_direction: 'Sentiment Direction',
  source_distribution: 'Source Distribution',
  top_feature_gaps: 'Feature Gaps',
  total_reviews: 'Total Reviews',
  vulnerability_window: 'Vulnerability Window',
  weakness_analysis: 'Weakness Analysis',
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

export function humanLabel(key: string): string {
  return FIELD_LABELS[key] ?? key.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase())
}

function isRankedList(value: unknown): value is AnyObject[] {
  if (!Array.isArray(value) || value.length === 0) return false
  return value.every((item) => isRecord(item) && ['count', 'mentions', 'mention_count'].some((key) => key in item))
}

function getRankCount(item: AnyObject): number {
  return Number(item.count ?? item.mentions ?? item.mention_count ?? 0)
}

function StructuredTable({ rows }: { rows: AnyObject[] }) {
  if (rows.length === 0) return null
  const columns = Array.from(new Set(rows.flatMap((row) => Object.keys(row)))).filter((key) =>
    rows.some((row) => isScalarValue(row[key])),
  )
  if (columns.length === 0) return null

  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-700/50">
            {columns.map((column) => (
              <th key={column} className="text-left text-xs font-medium text-slate-400 uppercase tracking-wider px-3 py-2 whitespace-nowrap">
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
                    <td key={column} className="px-3 py-2 whitespace-nowrap">
                      <ArchetypeBadge archetype={value} confidence={typeof row.archetype_confidence === 'number' ? row.archetype_confidence : undefined} showConfidence />
                    </td>
                  )
                }
                return (
                  <td key={column} className="px-3 py-2 text-slate-300 whitespace-nowrap">
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
            <div className="flex items-center justify-between text-xs mb-0.5">
              <span className="text-slate-300">{label}</span>
              <span className="text-slate-400">{count}</span>
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
        <span key={index} className="px-2 py-0.5 bg-slate-800 rounded text-xs text-slate-300">
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
          <p className="text-sm text-slate-200">{item.insight}</p>
          {typeof item.evidence === 'string' && item.evidence.trim() && (
            <p className="text-xs text-slate-500 mt-1">{item.evidence}</p>
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
          <blockquote className="text-sm text-slate-300 italic">"{String(item.quote ?? item.text ?? '')}"</blockquote>
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
            <blockquote className="text-sm text-slate-300 italic border-l-2 border-cyan-500/50 pl-3">{item.customer_quote}</blockquote>
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
            <span key={key} className="px-2 py-0.5 bg-slate-800 rounded text-xs text-slate-300">
              {humanLabel(key)}
            </span>
          ))}
        </div>
      )}
      <details className="group">
        <summary className="cursor-pointer text-xs text-slate-500 hover:text-slate-300">
          Show raw data
        </summary>
        <pre className="mt-2 text-xs text-slate-400 bg-slate-800/50 rounded p-3 overflow-x-auto">
          {JSON.stringify(value, null, 2)}
        </pre>
      </details>
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
            <div key={key} className="flex justify-between">
              <dt className="text-slate-400">{humanLabel(key)}</dt>
              <dd className="text-white font-medium">{formatValue(value)}</dd>
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
  if (typeof value === 'string') return <p className="text-sm text-slate-300 whitespace-pre-wrap">{value}</p>
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
  if (Array.isArray(value) && value.every(isRecord)) return <StructuredTable rows={value} />
  if (isRecord(value)) return <MixedObjectCard obj={value} />
  return <UnknownFallback value={value} />
}

export function StructuredReportData({
  data,
  skipKeys = [],
  className,
}: {
  data: AnyObject
  skipKeys?: string[]
  className?: string
}) {
  const entries = Object.entries(data).filter(([key, value]) => {
    if (skipKeys.includes(key) || REPORT_SCALAR_KEYS.has(key)) return false
    if (key === 'executive_summary') return false
    if (value === null || value === undefined) return false
    if (typeof value === 'string' && value.trim() === '') return false
    if (Array.isArray(value) && value.length === 0) return false
    if (isRecord(value) && Object.keys(value).length === 0) return false
    return true
  })

  if (entries.length === 0) return null

  return (
    <div className={clsx('grid grid-cols-1 lg:grid-cols-2 gap-6', className)}>
      {entries.map(([key, value]) => (
        <div key={key} className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
          <h4 className="text-xs font-medium text-cyan-400 uppercase tracking-wider mb-3">
            {humanLabel(key)}
          </h4>
          <StructuredReportValue fieldKey={key} value={value} />
        </div>
      ))}
    </div>
  )
}
