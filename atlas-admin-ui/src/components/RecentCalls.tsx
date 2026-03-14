import type { RecentCall } from '../types'
import { fmtCost, fmtDuration, provColor, timeAgo } from '../utils'
import StatusDot from './StatusDot'

export default function RecentCalls({ data }: { data: RecentCall[] }) {
  return (
    <div className="animate-enter rounded-xl border border-slate-800/80 bg-slate-900/40 p-5" style={{ animationDelay: '500ms' }}>
      <h2 className="mb-4 text-[11px] font-semibold uppercase tracking-widest text-slate-500">
        Recent Calls
      </h2>
      <div className="max-h-[440px] space-y-1 overflow-y-auto pr-1">
        {data.length === 0 && (
          <p className="py-12 text-center text-sm text-slate-600">No calls recorded yet</p>
        )}
        {data.map((call, i) => (
          <div
            key={i}
            className="group flex items-center gap-3 rounded-lg border border-transparent px-3 py-2.5 transition-colors hover:border-slate-800/80 hover:bg-slate-800/20"
          >
            <div className="flex flex-col items-center gap-1">
              <StatusDot status={call.status} />
              <div
                className="h-2 w-2 rounded-full opacity-60"
                style={{ backgroundColor: provColor(call.provider) }}
              />
            </div>

            <div className="min-w-0 flex-1">
              <div className="flex items-baseline gap-2">
                <span className="truncate font-mono text-xs font-medium text-slate-300">
                  {call.span_name}
                </span>
                <span className="shrink-0 text-[10px] text-slate-600">
                  {call.model}
                </span>
              </div>
              <div className="mt-0.5 flex flex-wrap items-center gap-x-3 gap-y-0.5 text-[11px] text-slate-500">
                <span>{(call.input_tokens + call.output_tokens).toLocaleString()} tok</span>
                <span>{fmtDuration(call.duration_ms)}</span>
                {call.tokens_per_second > 0 && (
                  <span>{call.tokens_per_second.toFixed(0)} t/s</span>
                )}
                <span className="font-mono text-cyan-500/80">{fmtCost(call.cost_usd)}</span>
              </div>
            </div>

            <span className="shrink-0 text-[10px] text-slate-600">
              {call.created_at ? timeAgo(call.created_at) : ''}
            </span>
          </div>
        ))}
      </div>
    </div>
  )
}
