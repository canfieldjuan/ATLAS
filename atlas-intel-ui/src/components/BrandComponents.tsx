import { clsx } from 'clsx'
import type { LabelCount } from '../api/client'

export const PIE_COLORS = ['#22d3ee', '#a78bfa', '#f472b6', '#facc15', '#34d399', '#fb923c']

export const TOOLTIP_STYLE = {
  backgroundColor: '#1e293b',
  border: '1px solid #334155',
  borderRadius: 8,
  color: '#e2e8f0',
  fontSize: 13,
}

export const CHURN_COLORS: Record<string, string> = {
  // replacement_behavior
  avoided: 'bg-slate-600', kept_using: 'bg-emerald-600', switched_to: 'bg-red-500',
  returned: 'bg-amber-500', kept_broken: 'bg-red-700', repurchased: 'bg-emerald-500',
  replaced_same: 'bg-cyan-500', switched_brand: 'bg-red-600',
  // trajectory
  always_positive: 'bg-emerald-500', always_negative: 'bg-red-500', degraded: 'bg-amber-500',
  improved: 'bg-cyan-500', mixed_then_negative: 'bg-red-400', mixed_then_positive: 'bg-emerald-400',
  mixed_then_bad: 'bg-red-400', always_bad: 'bg-red-600',
  // switching barrier
  none: 'bg-slate-600', low: 'bg-emerald-600', medium: 'bg-amber-500', high: 'bg-red-500',
  // repurchase
  true: 'bg-emerald-500', false: 'bg-red-500',
  // consequence
  inconvenience: 'bg-slate-500', positive_impact: 'bg-emerald-500',
  financial_loss: 'bg-red-500', workflow_impact: 'bg-amber-500', safety_concern: 'bg-red-700',
}

/** Horizontal bar for distribution data */
export function DistBar({ items, label }: { items: LabelCount[]; label: string }) {
  if (!items.length) return null
  const total = items.reduce((s, i) => s + i.count, 0)
  return (
    <div>
      <h4 className="text-xs text-slate-400 mb-1.5">{label}</h4>
      <div className="flex h-5 rounded-full overflow-hidden">
        {items.map((item) => {
          const pct = (item.count / total) * 100
          const color = CHURN_COLORS[item.label] ?? 'bg-slate-600'
          return (
            <div
              key={item.label}
              className={clsx(color, 'relative group')}
              style={{ width: `${pct}%`, minWidth: pct > 0 ? 4 : 0 }}
              title={`${item.label}: ${item.count} (${pct.toFixed(0)}%)`}
            >
              <div className="absolute bottom-full left-1/2 -translate-x-1/2 mb-1 hidden group-hover:block whitespace-nowrap px-2 py-1 bg-slate-800 text-xs text-white rounded shadow-lg z-10">
                {item.label.replaceAll('_', ' ')}: {item.count} ({pct.toFixed(0)}%)
              </div>
            </div>
          )
        })}
      </div>
      <div className="flex flex-wrap gap-x-3 gap-y-1 mt-1.5">
        {items.map((item) => (
          <span key={item.label} className="flex items-center gap-1 text-xs text-slate-400">
            <span className={clsx('w-2 h-2 rounded-full', CHURN_COLORS[item.label] ?? 'bg-slate-600')} />
            {item.label.replaceAll('_', ' ')} ({item.count})
          </span>
        ))}
      </div>
    </div>
  )
}

/** Pill badges for a distribution */
export function BadgeRow({ items, label, colorFn }: {
  items: LabelCount[]
  label: string
  colorFn?: (val: string) => string
}) {
  if (!items.length) return null
  const defaultColor = (v: string) => CHURN_COLORS[v] ?? 'bg-slate-700'
  const getColor = colorFn ?? defaultColor
  return (
    <div>
      <h4 className="text-xs text-slate-400 mb-1.5">{label}</h4>
      <div className="flex flex-wrap gap-1.5">
        {items.map((item) => (
          <span
            key={item.label}
            className={clsx(
              'px-2.5 py-1 rounded-full text-xs font-medium text-white/90',
              getColor(item.label),
            )}
          >
            {item.label.replaceAll('_', ' ')} ({item.count})
          </span>
        ))}
      </div>
    </div>
  )
}

/** Section card wrapper */
export function Card({ title, children, className }: { title: string; children: React.ReactNode; className?: string }) {
  return (
    <div className={clsx('bg-slate-900/50 border border-slate-700/50 rounded-xl p-5', className)}>
      <h3 className="text-sm font-medium text-slate-300 mb-3">{title}</h3>
      {children}
    </div>
  )
}
