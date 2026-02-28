import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  ResponsiveContainer,
  Cell,
} from 'recharts'
import type { ChurnSignal } from '../types'

interface ChurnChartProps {
  signals: ChurnSignal[]
  maxItems?: number
}

function urgencyColor(score: number) {
  if (score >= 8) return '#ef4444'
  if (score >= 6) return '#f59e0b'
  if (score >= 4) return '#eab308'
  return '#22c55e'
}

export default function ChurnChart({ signals, maxItems = 8 }: ChurnChartProps) {
  const data = signals.slice(0, maxItems).map((s) => ({
    name: s.vendor_name.length > 16 ? s.vendor_name.slice(0, 14) + '...' : s.vendor_name,
    urgency: s.avg_urgency_score,
    reviews: s.total_reviews,
  }))

  return (
    <ResponsiveContainer width="100%" height={300}>
      <BarChart data={data} margin={{ top: 8, right: 8, bottom: 8, left: 0 }}>
        <XAxis
          dataKey="name"
          tick={{ fill: '#94a3b8', fontSize: 11 }}
          axisLine={{ stroke: '#334155' }}
          tickLine={false}
        />
        <YAxis
          domain={[0, 10]}
          tick={{ fill: '#94a3b8', fontSize: 11 }}
          axisLine={{ stroke: '#334155' }}
          tickLine={false}
        />
        <Tooltip
          contentStyle={{
            backgroundColor: '#1e293b',
            border: '1px solid #334155',
            borderRadius: 8,
            color: '#e2e8f0',
            fontSize: 13,
          }}
        />
        <Bar dataKey="urgency" radius={[4, 4, 0, 0]}>
          {data.map((d, i) => (
            <Cell key={i} fill={urgencyColor(d.urgency)} />
          ))}
        </Bar>
      </BarChart>
    </ResponsiveContainer>
  )
}
