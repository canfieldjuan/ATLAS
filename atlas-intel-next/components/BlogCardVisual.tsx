import { Swords, AlertTriangle, ArrowRightLeft, Shield, FileText, BookOpen, type LucideIcon } from 'lucide-react'
import type { BlogPost } from '@/content/blog'

type TopicTheme = {
  gradient: string
  Icon: LucideIcon
  accent: string
  barColor: string
}

const THEMES: Record<string, TopicTheme> = {
  brand_showdown:    { gradient: 'from-purple-900/50 to-slate-800', Icon: Swords,          accent: 'bg-violet-500', barColor: '#8b5cf6' },
  complaint_roundup: { gradient: 'from-red-900/50 to-slate-800',    Icon: AlertTriangle,   accent: 'bg-red-500',    barColor: '#ef4444' },
  migration_report:  { gradient: 'from-emerald-900/50 to-slate-800', Icon: ArrowRightLeft, accent: 'bg-green-500',  barColor: '#10b981' },
  safety_spotlight:  { gradient: 'from-amber-900/50 to-slate-800',  Icon: Shield,          accent: 'bg-amber-500',  barColor: '#f59e0b' },
  guide:             { gradient: 'from-blue-900/50 to-slate-800',   Icon: BookOpen,        accent: 'bg-blue-500',   barColor: '#3b82f6' },
}

const DEFAULT_THEME: TopicTheme = {
  gradient: 'from-cyan-900/40 to-slate-800',
  Icon: FileText,
  accent: 'bg-cyan-500',
  barColor: '#06b6d4',
}

function MiniSparkbar({ data, color }: { data: { name: string; mentions: number }[]; color: string }) {
  const items = data.slice(0, 6)
  if (items.length === 0) return null
  const max = Math.max(...items.map(d => d.mentions))
  if (max === 0) return null
  const barW = 100 / items.length
  return (
    <svg viewBox="0 0 100 20" className="w-full h-5 opacity-60" preserveAspectRatio="none">
      {items.map((d, i) => {
        const h = (d.mentions / max) * 16
        return (
          <rect
            key={i}
            x={i * barW + barW * 0.15}
            y={20 - h}
            width={barW * 0.7}
            height={h}
            rx={1}
            fill={color}
            opacity={0.7 + (d.mentions / max) * 0.3}
          />
        )
      })}
    </svg>
  )
}

export default function BlogCardVisual({ post }: { post: BlogPost }) {
  const theme = THEMES[post.topic_type ?? ''] ?? DEFAULT_THEME
  const { gradient, Icon, accent, barColor } = theme

  // Extract first chart data for sparkbar
  const chartData = post.charts?.[0]?.data as { name: string; mentions: number }[] | undefined

  return (
    <div className={`relative h-40 bg-gradient-to-br ${gradient} overflow-hidden`}>
      {/* Watermark icon */}
      <Icon className="absolute top-3 right-3 h-16 w-16 opacity-[0.15] text-white" strokeWidth={1} />

      {/* Mini sparkbar at bottom */}
      {chartData && chartData.length > 0 && (
        <div className="absolute bottom-3 left-4 right-4">
          <MiniSparkbar data={chartData} color={barColor} />
        </div>
      )}

      {/* Accent line */}
      <div className={`absolute bottom-0 left-0 right-0 h-0.5 ${accent}`} />
    </div>
  )
}
