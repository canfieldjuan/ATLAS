import {
  BarChart, Bar, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Cell,
} from 'recharts'
import type { ProviderEntry } from '../types'
import { fmtCost, fmtTokens, provColor } from '../utils'

type ProviderTooltipProps = {
  active?: boolean
  payload?: Array<{ payload?: ProviderEntry }>
}

function ProviderTooltip({ active, payload }: ProviderTooltipProps) {
  const entry = payload?.[0]?.payload
  if (!active || !entry) return null

  return (
    <div className="rounded-lg border border-slate-700 bg-slate-800/95 px-3 py-2 text-xs shadow-xl backdrop-blur">
      <p className="mb-1 font-semibold text-slate-200">{entry.provider}</p>
      <p className="text-cyan-400">{fmtTokens(entry.total_tokens)} tokens</p>
      <p className="text-violet-400">{entry.calls.toLocaleString()} calls</p>
      {entry.cost_usd > 0 && <p className="text-emerald-400">{fmtCost(entry.cost_usd)}</p>}
    </div>
  )
}

export default function ProviderChart({ data }: { data: ProviderEntry[] }) {
  return (
    <div className="animate-enter rounded-xl border border-slate-800/80 bg-slate-900/40 p-5" style={{ animationDelay: '320ms' }}>
      <h2 className="mb-4 text-[11px] font-semibold uppercase tracking-widest text-slate-500">
        By Provider
      </h2>
      <div className="h-[300px]">
        {data.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-slate-600">
            No data
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={data} layout="vertical" margin={{ left: 0, right: 8 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" horizontal={false} />
              <XAxis
                type="number"
                tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                tickFormatter={fmtTokens}
                stroke="#334155"
                tickLine={false}
              />
              <YAxis
                type="category"
                dataKey="provider"
                tick={{ fill: '#94a3b8', fontSize: 12, fontFamily: 'DM Sans' }}
                stroke="#334155"
                width={80}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip content={<ProviderTooltip />} />
              <Bar dataKey="total_tokens" name="tokens" radius={[0, 4, 4, 0]} barSize={22}>
                {data.map((entry, i) => (
                  <Cell key={i} fill={provColor(entry.provider)} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  )
}
