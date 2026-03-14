const CARD_THEMES = {
  cyan: {
    bg: 'from-cyan-500/10 to-cyan-500/5',
    border: 'border-cyan-500/20 hover:border-cyan-500/40',
    icon: 'text-cyan-400',
    glow: 'hover:shadow-cyan-500/5',
  },
  emerald: {
    bg: 'from-emerald-500/10 to-emerald-500/5',
    border: 'border-emerald-500/20 hover:border-emerald-500/40',
    icon: 'text-emerald-400',
    glow: 'hover:shadow-emerald-500/5',
  },
  orange: {
    bg: 'from-orange-500/10 to-orange-500/5',
    border: 'border-orange-500/20 hover:border-orange-500/40',
    icon: 'text-orange-400',
    glow: 'hover:shadow-orange-500/5',
  },
  violet: {
    bg: 'from-violet-500/10 to-violet-500/5',
    border: 'border-violet-500/20 hover:border-violet-500/40',
    icon: 'text-violet-400',
    glow: 'hover:shadow-violet-500/5',
  },
} as const

export default function StatCard({ icon: Icon, label, value, sub, color = 'cyan', idx = 0 }: {
  icon: React.ComponentType<{ className?: string }>
  label: string
  value: string
  sub?: string
  color?: keyof typeof CARD_THEMES
  idx?: number
}) {
  const t = CARD_THEMES[color]
  return (
    <div
      className={`animate-enter relative overflow-hidden rounded-xl border bg-gradient-to-br p-5 transition-all duration-300 hover:scale-[1.02] hover:shadow-lg ${t.bg} ${t.border} ${t.glow}`}
      style={{ animationDelay: `${idx * 60}ms` }}
    >
      <div className="flex items-start justify-between">
        <div>
          <p className="text-[11px] font-semibold uppercase tracking-widest text-slate-500">
            {label}
          </p>
          <p className="mt-2 font-mono text-2xl font-bold tracking-tight text-slate-100">
            {value}
          </p>
          {sub && <p className="mt-1 text-xs text-slate-500">{sub}</p>}
        </div>
        <div className="rounded-lg bg-slate-800/60 p-2.5">
          <Icon className={`h-5 w-5 ${t.icon}`} />
        </div>
      </div>
    </div>
  )
}
