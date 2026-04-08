import { useState } from 'react'
import {
  BarChart3,
  ChevronDown,
  ChevronRight,
  Loader2,
} from 'lucide-react'
import { clsx } from 'clsx'
import useApiData from '../hooks/useApiData'
import { fetchSignalEffectiveness, fetchOutcomeDistribution } from '../api/client'

const GROUP_BY_OPTIONS = [
  { value: 'buying_stage', label: 'Buying Stage' },
  { value: 'role_type', label: 'Role Type' },
  { value: 'urgency_bucket', label: 'Urgency Bucket' },
  { value: 'pain_category', label: 'Pain Category' },
] as const

const OUTCOME_COLORS: Record<string, string> = {
  meeting_booked: 'bg-green-500',
  deal_opened: 'bg-cyan-500',
  deal_won: 'bg-emerald-500',
  deal_lost: 'bg-red-500',
  no_opportunity: 'bg-slate-600',
  disqualified: 'bg-slate-700',
  pending: 'bg-slate-800',
}

export default function SignalEffectivenessPanel() {
  const [expanded, setExpanded] = useState(false)
  const [groupBy, setGroupBy] = useState('buying_stage')

  const { data: effectivenessData, loading: effLoading } = useApiData(
    () => fetchSignalEffectiveness({ group_by: groupBy, min_sequences: 3 }),
    [groupBy],
  )

  const { data: distData, loading: distLoading } = useApiData(
    () => fetchOutcomeDistribution(),
    [],
  )

  const loading = effLoading || distLoading

  return (
    <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl">
      <button
        onClick={() => setExpanded(!expanded)}
        className="w-full flex items-center justify-between px-4 py-3 text-sm font-medium text-slate-300 hover:text-white transition-colors"
      >
        <div className="flex items-center gap-2">
          <BarChart3 className="h-4 w-4 text-cyan-400" />
          <span>Signal Effectiveness</span>
          {distData && distData.total_sequences > 0 && (
            <span className="text-xs text-slate-500">
              ({distData.total_sequences} sequences tracked)
            </span>
          )}
        </div>
        {expanded ? <ChevronDown className="h-4 w-4" /> : <ChevronRight className="h-4 w-4" />}
      </button>

      {expanded && (
        <div className="px-4 pb-4 space-y-4">
          {loading && (
            <div className="flex items-center justify-center py-6">
              <Loader2 className="h-4 w-4 animate-spin text-slate-500" />
            </div>
          )}

          {/* Outcome distribution funnel */}
          {distData && distData.buckets.length > 0 && (
            <div>
              <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Outcome Funnel</h4>
              <div className="flex gap-1 h-6 rounded-lg overflow-hidden">
                {distData.buckets
                  .filter((b) => b.count > 0)
                  .map((b) => (
                    <div
                      key={b.outcome}
                      className={clsx('h-full transition-all', OUTCOME_COLORS[b.outcome] || 'bg-slate-700')}
                      style={{ width: `${b.pct}%`, minWidth: b.pct > 0 ? '4px' : '0' }}
                      title={`${b.outcome.replace(/_/g, ' ')}: ${b.count} (${b.pct.toFixed(1)}%)`}
                    />
                  ))}
              </div>
              <div className="flex flex-wrap gap-3 mt-2">
                {distData.buckets.filter((b) => b.count > 0).map((b) => (
                  <div key={b.outcome} className="flex items-center gap-1.5 text-xs">
                    <div className={clsx('h-2 w-2 rounded-full', OUTCOME_COLORS[b.outcome] || 'bg-slate-700')} />
                    <span className="text-slate-400">{b.outcome.replace(/_/g, ' ')}</span>
                    <span className="text-white font-medium">{b.count}</span>
                    {b.total_revenue > 0 && (
                      <span className="text-green-400">${b.total_revenue.toLocaleString()}</span>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Signal effectiveness by dimension */}
          {effectivenessData && effectivenessData.groups.length > 0 && (
            <div>
              <div className="flex items-center justify-between mb-2">
                <h4 className="text-xs font-medium text-slate-400 uppercase tracking-wider">Conversion by Signal</h4>
                <select
                  value={groupBy}
                  onChange={(e) => setGroupBy(e.target.value)}
                  className="bg-slate-800/50 border border-slate-700/50 rounded px-2 py-0.5 text-xs text-white"
                >
                  {GROUP_BY_OPTIONS.map((opt) => (
                    <option key={opt.value} value={opt.value}>{opt.label}</option>
                  ))}
                </select>
              </div>
              <div className="space-y-2">
                {effectivenessData.groups
                  .sort((a, b) => b.positive_outcome_rate - a.positive_outcome_rate)
                  .map((g) => {
                    const pct = Math.round(g.positive_outcome_rate * 100)
                    return (
                      <div key={g.signal_group} className="flex items-center gap-3">
                        <span className="text-xs text-slate-300 w-32 truncate" title={g.signal_group}>
                          {g.signal_group.replace(/_/g, ' ')}
                        </span>
                        <div className="flex-1 h-2 bg-slate-800 rounded-full overflow-hidden">
                          <div
                            className="h-full bg-cyan-500 rounded-full transition-all"
                            style={{ width: `${pct}%` }}
                          />
                        </div>
                        <span className="text-xs text-white font-medium w-10 text-right">{pct}%</span>
                        <span className="text-[10px] text-slate-500 w-8 text-right">{g.total_sequences}</span>
                        {g.total_revenue > 0 && (
                          <span className="text-[10px] text-green-400 w-16 text-right">
                            ${g.total_revenue.toLocaleString()}
                          </span>
                        )}
                      </div>
                    )
                  })}
              </div>
            </div>
          )}

          {!loading && distData && distData.total_sequences === 0 && (
            <div className="text-sm text-slate-500 text-center py-4">
              No outcome data yet. Record outcomes on sent campaigns to see signal effectiveness.
            </div>
          )}
        </div>
      )}
    </div>
  )
}
