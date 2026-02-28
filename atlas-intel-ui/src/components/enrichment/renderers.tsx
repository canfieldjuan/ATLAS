import { ShieldAlert, ShieldCheck, Check, X } from 'lucide-react'
import { colorMaps, mapLookup, type ColorPair } from './colorMaps'
import type { SafetyFlag } from './types'

// --- Helpers ---

function titleCase(s: string): string {
  return s.replace(/_/g, ' ').replace(/\b\w/g, c => c.toUpperCase())
}

function hasContent(v: unknown): boolean {
  if (v === null || v === undefined) return false
  if (Array.isArray(v)) return v.length > 0
  if (typeof v === 'object') return Object.values(v).some(hasContent)
  if (typeof v === 'string') return v.length > 0
  return true
}

// --- 1. EnumBadge ---

interface EnumBadgeProps {
  label: string
  value: string | null | undefined
  colorMap: Record<string, ColorPair>
}

export function EnumBadge({ label, value, colorMap }: EnumBadgeProps) {
  if (!value) return null
  const [bg, text] = mapLookup(colorMap, value)
  return (
    <div className="flex flex-col gap-1">
      <span className="text-[10px] uppercase tracking-wider text-slate-500">{label}</span>
      <span className={`inline-flex px-2.5 py-1 rounded-full text-xs font-medium w-fit ${bg} ${text}`}>
        {titleCase(value)}
      </span>
    </div>
  )
}

// --- 2. StringList ---

interface StringListProps {
  items: string[] | null | undefined
  variant: 'quote' | 'tag' | 'bullet'
}

export function StringList({ items, variant }: StringListProps) {
  if (!items?.length) return null

  if (variant === 'quote') {
    return (
      <div className="space-y-2">
        {items.map((q, i) => (
          <blockquote key={i} className="border-l-2 border-cyan-500/50 pl-3 py-1 text-sm text-slate-300 italic">
            &ldquo;{q}&rdquo;
          </blockquote>
        ))}
      </div>
    )
  }

  if (variant === 'tag') {
    return (
      <div className="flex flex-wrap gap-1.5">
        {items.map((t, i) => (
          <span key={i} className="px-2 py-0.5 rounded-full text-xs bg-green-500/10 text-green-400">
            {t}
          </span>
        ))}
      </div>
    )
  }

  // bullet
  return (
    <ul className="space-y-1 text-sm text-slate-300">
      {items.map((b, i) => (
        <li key={i} className="flex gap-2">
          <span className="text-amber-400 shrink-0">&#8226;</span>
          <span>{b}</span>
        </li>
      ))}
    </ul>
  )
}

// --- 3. ObjectTable ---

interface Column {
  key: string
  label: string
  render?: (val: unknown, row: Record<string, unknown>) => React.ReactNode
}

interface ObjectTableProps {
  rows: Record<string, unknown>[] | null | undefined
  columns: Column[]
}

export function ObjectTable({ rows, columns }: ObjectTableProps) {
  if (!rows?.length) return null
  return (
    <div className="overflow-x-auto">
      <table className="w-full text-sm">
        <thead>
          <tr className="border-b border-slate-700/50">
            {columns.map(c => (
              <th key={c.key} className="text-left text-[10px] uppercase tracking-wider text-slate-500 pb-2 pr-4">
                {c.label}
              </th>
            ))}
          </tr>
        </thead>
        <tbody className="divide-y divide-slate-800/50">
          {rows.map((row, i) => (
            <tr key={i}>
              {columns.map(c => (
                <td key={c.key} className="py-2 pr-4 text-slate-300">
                  {c.render ? c.render(row[c.key], row) : String(row[c.key] ?? '')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  )
}

// Preset column renderers for common patterns
export const columnRenderers = {
  sentimentDot: (val: unknown) => {
    const s = String(val ?? '')
    const dotColor = colorMaps.sentimentDot[s] ?? 'bg-slate-400'
    return (
      <span className="flex items-center gap-1.5">
        <span className={`w-2 h-2 rounded-full shrink-0 ${dotColor}`} />
        <span className="text-xs">{titleCase(s)}</span>
      </span>
    )
  },
  directionBadge: (val: unknown) => {
    const s = String(val ?? '')
    const [bg, text] = mapLookup(colorMaps.comparisonDirection, s)
    return <span className={`px-1.5 py-0.5 rounded text-xs ${bg} ${text}`}>{titleCase(s)}</span>
  },
  platformTag: (val: unknown) => {
    const s = String(val ?? '')
    return <span className="px-1.5 py-0.5 rounded text-xs bg-blue-500/10 text-blue-400">{s}</span>
  },
  bold: (val: unknown) => <span className="font-medium text-white">{String(val ?? '')}</span>,
  text: (val: unknown) => <span className="text-slate-400">{String(val ?? '')}</span>,
}

// --- 4. KeyValueCard ---

interface KVRow {
  label: string
  value: React.ReactNode
}

interface KeyValueCardProps {
  rows: KVRow[]
}

export function KeyValueCard({ rows }: KeyValueCardProps) {
  const visible = rows.filter(r => hasContent(r.value))
  if (!visible.length) return null
  return (
    <dl className="space-y-2">
      {visible.map((r, i) => (
        <div key={i} className="flex flex-col sm:flex-row sm:items-start gap-1">
          <dt className="text-[10px] uppercase tracking-wider text-slate-500 sm:w-28 shrink-0 pt-0.5">{r.label}</dt>
          <dd className="text-sm text-slate-300">{r.value}</dd>
        </div>
      ))}
    </dl>
  )
}

// Helper: render inline badge for KV cards
export function inlineBadge(value: string | null | undefined, colorMap: Record<string, ColorPair>): React.ReactNode {
  if (!value) return <span className="text-slate-500">--</span>
  const [bg, text] = mapLookup(colorMap, value)
  return <span className={`px-2 py-0.5 rounded-full text-xs font-medium ${bg} ${text}`}>{titleCase(value)}</span>
}

// --- 5. FailureCard ---

interface FailureCardProps {
  details: {
    timeline?: string | null
    failed_component?: string | null
    failure_mode?: string | null
    dollar_amount_lost?: number | null
  } | null | undefined
}

export function FailureCard({ details }: FailureCardProps) {
  if (!details) return null
  const hasAny = details.timeline != null || details.failed_component != null || details.failure_mode != null || details.dollar_amount_lost != null
  if (!hasAny) return null

  return (
    <div className="border-l-4 border-red-500/60 bg-red-500/5 rounded-r-lg p-4 space-y-2">
      <div className="flex items-center justify-between">
        <h4 className="text-sm font-medium text-red-400">Failure Details</h4>
        {details.dollar_amount_lost != null && (
          <span className="text-lg font-bold text-red-400">${details.dollar_amount_lost.toLocaleString()}</span>
        )}
      </div>
      <dl className="grid grid-cols-1 sm:grid-cols-3 gap-3 text-sm">
        {details.timeline && (
          <div>
            <dt className="text-[10px] uppercase tracking-wider text-slate-500">Timeline</dt>
            <dd className="text-slate-300 mt-0.5">{details.timeline}</dd>
          </div>
        )}
        {details.failed_component && (
          <div>
            <dt className="text-[10px] uppercase tracking-wider text-slate-500">Component</dt>
            <dd className="text-slate-300 mt-0.5">{details.failed_component}</dd>
          </div>
        )}
        {details.failure_mode && (
          <div>
            <dt className="text-[10px] uppercase tracking-wider text-slate-500">Failure Mode</dt>
            <dd className="text-slate-300 mt-0.5">{details.failure_mode}</dd>
          </div>
        )}
      </dl>
    </div>
  )
}

// --- 6. BooleanIndicator ---

interface BooleanIndicatorProps {
  label: string
  value: boolean | null | undefined
}

export function BooleanIndicator({ label, value }: BooleanIndicatorProps) {
  if (value === null || value === undefined) return null
  return (
    <div className="flex items-center gap-2">
      <span className="text-[10px] uppercase tracking-wider text-slate-500">{label}</span>
      {value === true && (
        <span className="flex items-center gap-1 text-green-400 text-sm">
          <Check className="h-4 w-4" /> Yes
        </span>
      )}
      {value === false && (
        <span className="flex items-center gap-1 text-red-400 text-sm">
          <X className="h-4 w-4" /> No
        </span>
      )}
    </div>
  )
}

// --- 7. TextValue ---

interface TextValueProps {
  value: string | null | undefined
  highlight?: boolean
}

export function TextValue({ value, highlight }: TextValueProps) {
  if (!value) return null
  if (highlight) {
    return (
      <div className="bg-cyan-500/5 border border-cyan-500/20 rounded-lg px-3 py-2">
        <span className="text-sm text-cyan-300">{value}</span>
      </div>
    )
  }
  return <span className="text-sm text-slate-300">{value}</span>
}

// --- Special: SafetyFlagBanner ---

interface SafetyFlagBannerProps {
  flag: SafetyFlag | null | undefined
}

export function SafetyFlagBanner({ flag }: SafetyFlagBannerProps) {
  if (!flag) return null

  if (flag.flagged) {
    return (
      <div className="flex items-start gap-3 bg-red-500/10 border border-red-500/30 rounded-lg p-4">
        <ShieldAlert className="h-5 w-5 text-red-400 shrink-0 mt-0.5" />
        <div>
          <h4 className="text-sm font-medium text-red-400">Safety Concern Flagged</h4>
          {flag.description && (
            <p className="text-sm text-red-300/80 mt-1">{flag.description}</p>
          )}
        </div>
      </div>
    )
  }

  return (
    <div className="flex items-center gap-2 text-sm text-green-400/70">
      <ShieldCheck className="h-4 w-4" />
      <span>No safety concerns</span>
    </div>
  )
}
