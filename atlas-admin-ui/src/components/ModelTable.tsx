import { Cpu } from 'lucide-react'
import type { ModelEntry } from '../types'
import { fmtCost, fmtDuration, fmtTokens, provColor } from '../utils'

export default function ModelTable({ data }: { data: ModelEntry[] }) {
  const topModels = data.slice(0, 8)

  return (
    <div className="animate-enter rounded-xl border border-slate-800/80 bg-slate-900/40 p-5" style={{ animationDelay: '340ms' }}>
      <div className="mb-4 flex items-center justify-between gap-3">
        <h2 className="text-[11px] font-semibold uppercase tracking-widest text-slate-500">
          By Model
        </h2>
        <span className="text-[10px] text-slate-600">Top {topModels.length || 0}</span>
      </div>

      <div className="space-y-2">
        {topModels.length === 0 && (
          <div className="py-8 text-center text-sm text-slate-600">No model data</div>
        )}
        {topModels.map((entry) => (
          <div
            key={`${entry.provider}:${entry.model}`}
            className="rounded-lg border border-slate-800/60 bg-slate-950/50 p-3"
          >
            <div className="flex items-start justify-between gap-3">
              <div className="min-w-0">
                <div className="flex items-center gap-2">
                  <span
                    className="inline-block h-2 w-2 rounded-full"
                    style={{ backgroundColor: provColor(entry.provider) }}
                  />
                  <span className="truncate font-mono text-[11px] text-slate-200">{entry.model}</span>
                </div>
                <div className="mt-1 flex items-center gap-2 text-[10px] text-slate-500">
                  <Cpu className="h-3 w-3" />
                  <span>{entry.provider}</span>
                  <span>{fmtDuration(entry.avg_duration_ms)} avg</span>
                </div>
              </div>
              <div className="text-right">
                <div className="font-mono text-sm text-cyan-400">{fmtCost(entry.cost_usd)}</div>
                <div className="text-[10px] text-slate-500">{entry.calls.toLocaleString()} calls</div>
              </div>
            </div>
            <div className="mt-3 flex items-center justify-between gap-3 text-[10px] text-slate-500">
              <span>{fmtTokens(entry.total_tokens)} tokens</span>
              <span>{entry.avg_tokens_per_second.toFixed(1)} t/s</span>
            </div>
          </div>
        ))}
      </div>
    </div>
  )
}
