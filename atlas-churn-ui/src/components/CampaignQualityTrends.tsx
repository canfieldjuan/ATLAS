import {
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts'
import type { CampaignQualityTrends as CampaignQualityTrendsData } from '../types'

interface CampaignQualityTrendsProps {
  data?: CampaignQualityTrendsData | null
  loading?: boolean
  title?: string
}

function formatDayLabel(value: string) {
  const parts = String(value || '').split('-')
  if (parts.length !== 3) return value
  return `${parts[1]}/${parts[2]}`
}

function formatDelta(delta: number) {
  if (delta > 0) return { label: `up ${delta}`, className: 'text-red-300' }
  if (delta < 0) return { label: `down ${Math.abs(delta)}`, className: 'text-green-300' }
  return { label: 'flat', className: 'text-slate-500' }
}

export default function CampaignQualityTrends({
  data,
  loading = false,
  title = 'Campaign Quality Trends',
}: CampaignQualityTrendsProps) {
  const totalsByDay = data?.totals_by_day ?? []
  const chartData = totalsByDay.map((item) => ({
    day: formatDayLabel(item.day),
    blockerTotal: item.blocker_total,
  }))
  const series = data?.series ?? []
  const orderedDays = totalsByDay.map((item) => item.day)
  const latestDay = orderedDays[orderedDays.length - 1] ?? ''
  const previousDay = orderedDays[orderedDays.length - 2] ?? ''
  const countsByReasonAndDay = new Map<string, Map<string, number>>()

  for (const point of series) {
    if (!countsByReasonAndDay.has(point.reason)) {
      countsByReasonAndDay.set(point.reason, new Map<string, number>())
    }
    countsByReasonAndDay.get(point.reason)!.set(point.day, point.count)
  }

  const trendLeaders = (data?.top_blockers ?? []).slice(0, 5).map((item) => {
    const counts = countsByReasonAndDay.get(item.reason)
    const latest = counts?.get(latestDay) ?? 0
    const previous = counts?.get(previousDay) ?? 0
    return {
      ...item,
      latest,
      delta: latest - previous,
    }
  })

  return (
    <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
      <div className="flex flex-wrap items-end justify-between gap-3 mb-4">
        <div>
          <h3 className="text-sm font-medium text-white">{title}</h3>
          <p className="text-xs text-slate-500 mt-1">
            Last {data?.days ?? '--'} days of blocker volume and reason movement
          </p>
        </div>
        {totalsByDay.length > 0 && (
          <div className="text-right">
            <div className="text-[11px] uppercase tracking-wide text-slate-500">Latest total</div>
            <div className="text-lg font-semibold text-cyan-300">
              {totalsByDay[totalsByDay.length - 1]?.blocker_total ?? 0}
            </div>
          </div>
        )}
      </div>

      {loading ? (
        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,2fr)_minmax(240px,1fr)] gap-4">
          <div className="h-48 rounded-lg bg-slate-800/50 animate-pulse" />
          <div className="space-y-2">
            {[0, 1, 2, 3].map((index) => (
              <div key={index} className="h-10 rounded-lg bg-slate-800/50 animate-pulse" />
            ))}
          </div>
        </div>
      ) : chartData.length === 0 && trendLeaders.length === 0 ? (
        <p className="text-sm text-slate-500">No blocker trend data yet.</p>
      ) : (
        <div className="grid grid-cols-1 xl:grid-cols-[minmax(0,2fr)_minmax(240px,1fr)] gap-4">
          <div className="h-48 rounded-lg border border-slate-800/80 bg-slate-950/30 p-2">
            {chartData.length > 0 ? (
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 8, right: 8, bottom: 4, left: -12 }}>
                  <CartesianGrid stroke="#1e293b" strokeDasharray="3 3" />
                  <XAxis
                    dataKey="day"
                    tick={{ fill: '#94a3b8', fontSize: 11 }}
                    axisLine={{ stroke: '#334155' }}
                    tickLine={false}
                  />
                  <YAxis
                    allowDecimals={false}
                    tick={{ fill: '#94a3b8', fontSize: 11 }}
                    axisLine={{ stroke: '#334155' }}
                    tickLine={false}
                  />
                  <Tooltip
                    formatter={(value: number) => [value, 'Blockers']}
                    contentStyle={{
                      backgroundColor: '#1e293b',
                      border: '1px solid #334155',
                      borderRadius: 8,
                      color: '#e2e8f0',
                      fontSize: 13,
                    }}
                  />
                  <Line
                    type="monotone"
                    dataKey="blockerTotal"
                    stroke="#22d3ee"
                    strokeWidth={2}
                    dot={{ r: 2, fill: '#22d3ee' }}
                    activeDot={{ r: 4 }}
                  />
                </LineChart>
              </ResponsiveContainer>
            ) : (
              <div className="flex h-full items-center justify-center text-sm text-slate-500">
                No daily blocker totals available.
              </div>
            )}
          </div>

          <div>
            <div className="text-[11px] uppercase tracking-wide text-slate-500 mb-2">Reason movement</div>
            <div className="space-y-2">
              {trendLeaders.map((item) => {
                const delta = formatDelta(item.delta)
                return (
                  <div key={item.reason} className="rounded-lg border border-slate-800/80 bg-slate-950/30 px-3 py-2">
                    <div className="flex items-start justify-between gap-3 text-sm">
                      <span className="text-slate-300 break-words">{item.reason}</span>
                      <span className="text-cyan-300 shrink-0">{item.count}</span>
                    </div>
                    <div className="mt-1 flex items-center justify-between text-[11px]">
                      <span className="text-slate-500">latest day: {item.latest}</span>
                      <span className={delta.className}>{delta.label}</span>
                    </div>
                  </div>
                )
              })}
              {trendLeaders.length === 0 && (
                <div className="rounded-lg border border-slate-800/80 bg-slate-950/30 px-3 py-4 text-sm text-slate-500">
                  No blocker reasons available for this window.
                </div>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
