import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  ResponsiveContainer,
  RadarChart,
  Radar,
  PolarGrid,
  PolarAngleAxis,
  PolarRadiusAxis,
  LineChart,
  Line,
  Legend,
} from 'recharts'
import type { ChartSpec } from '@/content/blog'

const AXIS_TICK = '#94a3b8'
const GRID_LINE = '#334155'
const FALLBACK_COLORS = ['#22d3ee', '#f472b6', '#a78bfa', '#34d399', '#fbbf24', '#f87171']

function getColor(index: number, explicit?: string): string {
  return explicit || FALLBACK_COLORS[index % FALLBACK_COLORS.length]
}

function ChartBar({ spec }: { spec: ChartSpec }) {
  const bars: { dataKey: string; color?: string }[] = spec.config.bars || []
  return (
    <ResponsiveContainer width="100%" height={320}>
      <BarChart data={spec.data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GRID_LINE} />
        <XAxis
          dataKey={spec.config.x_key || 'name'}
          tick={{ fill: AXIS_TICK, fontSize: 12 }}
          axisLine={{ stroke: GRID_LINE }}
          tickLine={false}
        />
        <YAxis tick={{ fill: AXIS_TICK, fontSize: 12 }} axisLine={false} tickLine={false} />
        <Tooltip
          contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
          labelStyle={{ color: '#e2e8f0' }}
          itemStyle={{ color: '#e2e8f0' }}
        />
        {bars.length > 1 && <Legend wrapperStyle={{ color: AXIS_TICK }} />}
        {bars.map((b, i) => (
          <Bar key={b.dataKey} dataKey={b.dataKey} fill={getColor(i, b.color)} radius={[4, 4, 0, 0]} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}

function ChartHorizontalBar({ spec }: { spec: ChartSpec }) {
  const bars: { dataKey: string; color?: string }[] = spec.config.bars || []
  return (
    <ResponsiveContainer width="100%" height={Math.max(200, spec.data.length * 48)}>
      <BarChart data={spec.data} layout="vertical" margin={{ top: 8, right: 16, bottom: 8, left: 80 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GRID_LINE} horizontal={false} />
        <XAxis type="number" tick={{ fill: AXIS_TICK, fontSize: 12 }} axisLine={false} tickLine={false} />
        <YAxis
          type="category"
          dataKey={spec.config.x_key || 'name'}
          tick={{ fill: AXIS_TICK, fontSize: 12 }}
          axisLine={false}
          tickLine={false}
          width={80}
        />
        <Tooltip
          contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
          labelStyle={{ color: '#e2e8f0' }}
          itemStyle={{ color: '#e2e8f0' }}
        />
        {bars.length > 1 && <Legend wrapperStyle={{ color: AXIS_TICK }} />}
        {bars.map((b, i) => (
          <Bar key={b.dataKey} dataKey={b.dataKey} fill={getColor(i, b.color)} radius={[0, 4, 4, 0]} />
        ))}
      </BarChart>
    </ResponsiveContainer>
  )
}

function ChartRadar({ spec }: { spec: ChartSpec }) {
  const bars: { dataKey: string; color?: string }[] = spec.config.bars || []
  return (
    <ResponsiveContainer width="100%" height={360}>
      <RadarChart data={spec.data}>
        <PolarGrid stroke={GRID_LINE} />
        <PolarAngleAxis dataKey={spec.config.x_key || 'name'} tick={{ fill: AXIS_TICK, fontSize: 12 }} />
        <PolarRadiusAxis tick={{ fill: AXIS_TICK, fontSize: 10 }} axisLine={false} />
        {bars.map((b, i) => (
          <Radar
            key={b.dataKey}
            name={b.dataKey}
            dataKey={b.dataKey}
            stroke={getColor(i, b.color)}
            fill={getColor(i, b.color)}
            fillOpacity={0.2}
          />
        ))}
        <Legend wrapperStyle={{ color: AXIS_TICK }} />
        <Tooltip
          contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
          labelStyle={{ color: '#e2e8f0' }}
          itemStyle={{ color: '#e2e8f0' }}
        />
      </RadarChart>
    </ResponsiveContainer>
  )
}

function ChartLine({ spec }: { spec: ChartSpec }) {
  const bars: { dataKey: string; color?: string }[] = spec.config.bars || []
  return (
    <ResponsiveContainer width="100%" height={320}>
      <LineChart data={spec.data} margin={{ top: 8, right: 16, bottom: 8, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke={GRID_LINE} />
        <XAxis
          dataKey={spec.config.x_key || 'name'}
          tick={{ fill: AXIS_TICK, fontSize: 12 }}
          axisLine={{ stroke: GRID_LINE }}
          tickLine={false}
        />
        <YAxis tick={{ fill: AXIS_TICK, fontSize: 12 }} axisLine={false} tickLine={false} />
        <Tooltip
          contentStyle={{ backgroundColor: '#1e293b', border: '1px solid #334155', borderRadius: 8 }}
          labelStyle={{ color: '#e2e8f0' }}
          itemStyle={{ color: '#e2e8f0' }}
        />
        {bars.length > 1 && <Legend wrapperStyle={{ color: AXIS_TICK }} />}
        {bars.map((b, i) => (
          <Line
            key={b.dataKey}
            type="monotone"
            dataKey={b.dataKey}
            stroke={getColor(i, b.color)}
            strokeWidth={2}
            dot={{ fill: getColor(i, b.color), r: 3 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}

const RENDERERS: Record<string, React.FC<{ spec: ChartSpec }>> = {
  bar: ChartBar,
  horizontal_bar: ChartHorizontalBar,
  radar: ChartRadar,
  line: ChartLine,
}

export default function BlogChart({ spec }: { spec: ChartSpec }) {
  const Renderer = RENDERERS[spec.chart_type]
  if (!Renderer) return null
  return (
    <figure className="my-8 rounded-xl bg-slate-900/50 border border-slate-700/50 p-6">
      {spec.title && (
        <figcaption className="text-sm font-medium text-slate-300 mb-4 text-center">
          {spec.title}
        </figcaption>
      )}
      <Renderer spec={spec} />
    </figure>
  )
}
