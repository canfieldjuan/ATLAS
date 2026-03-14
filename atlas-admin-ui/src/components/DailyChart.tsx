import {
  AreaChart, Area, XAxis, YAxis, CartesianGrid,
  Tooltip, ResponsiveContainer, Line,
} from 'recharts'
import type { MergedDailyEntry } from '../types'
import { fmtDate, fmtTokens } from '../utils'
import CostTooltip from './CostTooltip'

export default function DailyChart({ data }: { data: MergedDailyEntry[] }) {
  return (
    <div className="animate-enter col-span-1 rounded-xl border border-slate-800/80 bg-slate-900/40 p-5 lg:col-span-2" style={{ animationDelay: '260ms' }}>
      <h2 className="mb-4 text-[11px] font-semibold uppercase tracking-widest text-slate-500">
        Daily Usage
      </h2>
      <div className="h-[300px]">
        {data.length === 0 ? (
          <div className="flex h-full items-center justify-center text-sm text-slate-600">
            No data for this period
          </div>
        ) : (
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={data} margin={{ top: 4, right: 4, bottom: 0, left: 0 }}>
              <defs>
                <linearGradient id="costGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#22d3ee" stopOpacity={0.25} />
                  <stop offset="100%" stopColor="#22d3ee" stopOpacity={0} />
                </linearGradient>
                <linearGradient id="callsGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#8b5cf6" stopOpacity={0.2} />
                  <stop offset="100%" stopColor="#8b5cf6" stopOpacity={0} />
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" stroke="#1e293b" />
              <XAxis
                dataKey="date"
                tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'DM Sans' }}
                tickFormatter={fmtDate}
                stroke="#334155"
                tickLine={false}
              />
              <YAxis
                yAxisId="tokens"
                tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                tickFormatter={(v) => fmtTokens(v)}
                stroke="#334155"
                width={48}
                tickLine={false}
                axisLine={false}
              />
              <YAxis
                yAxisId="calls"
                orientation="right"
                tick={{ fill: '#64748b', fontSize: 11, fontFamily: 'JetBrains Mono' }}
                stroke="#334155"
                width={32}
                tickLine={false}
                axisLine={false}
              />
              <Tooltip content={<CostTooltip />} />
              <Area
                yAxisId="tokens"
                type="monotone"
                dataKey="total_tokens"
                name="tokens"
                stroke="#22d3ee"
                strokeWidth={2}
                fill="url(#costGrad)"
                dot={false}
                activeDot={{ r: 4, fill: '#22d3ee', stroke: '#0f172a', strokeWidth: 2 }}
              />
              <Area
                yAxisId="calls"
                type="monotone"
                dataKey="calls"
                name="calls"
                stroke="#8b5cf6"
                strokeWidth={1.5}
                fill="url(#callsGrad)"
                dot={false}
                activeDot={{ r: 3, fill: '#8b5cf6', stroke: '#0f172a', strokeWidth: 2 }}
              />
              <Area
                yAxisId="calls"
                type="monotone"
                dataKey="error_calls"
                name="errors"
                stroke="#f87171"
                strokeWidth={1.5}
                fill="#f8717120"
                dot={false}
                activeDot={{ r: 3, fill: '#f87171', stroke: '#0f172a', strokeWidth: 2 }}
              />
              <Line
                yAxisId="calls"
                type="monotone"
                dataKey="error_calls"
                name="error_line"
                stroke="#f87171"
                strokeWidth={1}
                strokeDasharray="4 4"
                dot={false}
                activeDot={false}
                legendType="none"
              />
            </AreaChart>
          </ResponsiveContainer>
        )}
      </div>
    </div>
  )
}
