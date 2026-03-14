import { Clock } from 'lucide-react'
import type { WorkflowEntry } from '../types'
import { fmtCost, fmtTokens, fmtDuration } from '../utils'

export default function WorkflowTable({ data }: { data: WorkflowEntry[] }) {
  return (
    <div className="animate-enter rounded-xl border border-slate-800/80 bg-slate-900/40 p-5" style={{ animationDelay: '440ms' }}>
      <h2 className="mb-4 text-[11px] font-semibold uppercase tracking-widest text-slate-500">
        Workflow Breakdown
      </h2>
      <div className="overflow-x-auto">
        <table className="w-full text-left text-sm">
          <thead>
            <tr className="border-b border-slate-800 text-[10px] uppercase tracking-widest text-slate-600">
              <th className="pb-3 pr-4 font-semibold">Workflow</th>
              <th className="pb-3 pr-4 text-right font-semibold">Cost</th>
              <th className="pb-3 pr-4 text-right font-semibold">Calls</th>
              <th className="pb-3 pr-4 text-right font-semibold">Tokens</th>
              <th className="pb-3 text-right font-semibold">Latency</th>
            </tr>
          </thead>
          <tbody>
            {data.length === 0 && (
              <tr>
                <td colSpan={5} className="py-12 text-center text-sm text-slate-600">
                  No data yet
                </td>
              </tr>
            )}
            {data.map((w, i) => (
              <tr
                key={i}
                className="border-b border-slate-800/40 transition-colors hover:bg-slate-800/20"
              >
                <td className="py-3 pr-4">
                  <span className="font-mono text-xs text-slate-300">{w.workflow}</span>
                </td>
                <td className="py-3 pr-4 text-right font-mono text-sm text-cyan-400">
                  {fmtCost(w.cost_usd)}
                </td>
                <td className="py-3 pr-4 text-right text-slate-400">{w.calls}</td>
                <td className="py-3 pr-4 text-right text-slate-400">{fmtTokens(w.total_tokens)}</td>
                <td className="py-3 text-right text-slate-500">
                  <span className="inline-flex items-center gap-1">
                    <Clock className="h-3 w-3" />
                    {fmtDuration(w.avg_duration_ms)}
                  </span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  )
}
