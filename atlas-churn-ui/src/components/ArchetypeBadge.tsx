import { clsx } from 'clsx'

const ARCHETYPE_COLORS: Record<string, string> = {
  pricing_shock: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
  feature_gap: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
  acquisition_decay: 'bg-purple-500/20 text-purple-300 border-purple-500/30',
  leadership_redesign: 'bg-pink-500/20 text-pink-300 border-pink-500/30',
  integration_break: 'bg-orange-500/20 text-orange-300 border-orange-500/30',
  support_collapse: 'bg-red-500/20 text-red-300 border-red-500/30',
  category_disruption: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30',
  compliance_gap: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
  mixed: 'bg-slate-500/20 text-slate-300 border-slate-500/30',
  stable: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
}

const ARCHETYPE_LABELS: Record<string, string> = {
  pricing_shock: 'Pricing Shock',
  feature_gap: 'Feature Gap',
  acquisition_decay: 'Acquisition Decay',
  leadership_redesign: 'UX Redesign',
  integration_break: 'Integration Break',
  support_collapse: 'Support Collapse',
  category_disruption: 'Category Disruption',
  compliance_gap: 'Compliance Gap',
  mixed: 'Mixed',
  stable: 'Stable',
}

interface ArchetypeBadgeProps {
  archetype: string | null | undefined
  confidence?: number | null
  showConfidence?: boolean
  size?: 'sm' | 'md'
}

export default function ArchetypeBadge({
  archetype,
  confidence,
  showConfidence = false,
  size = 'sm',
}: ArchetypeBadgeProps) {
  if (!archetype) return null

  const colors = ARCHETYPE_COLORS[archetype] || ARCHETYPE_COLORS.mixed
  const label = ARCHETYPE_LABELS[archetype] || archetype.replace(/_/g, ' ')

  return (
    <span
      className={clsx(
        'inline-flex items-center gap-1 border rounded-full font-medium',
        colors,
        size === 'sm' ? 'px-2 py-0.5 text-xs' : 'px-3 py-1 text-sm',
      )}
    >
      {label}
      {showConfidence && confidence != null && (
        <span className="opacity-60">{Math.round(confidence * 100)}%</span>
      )}
    </span>
  )
}
