import { clsx } from 'clsx'

export default function UrgencyBadge({ score }: { score: number | null }) {
  if (score === null || score === undefined) {
    return <span className="text-xs text-slate-500">--</span>
  }

  const rounded = Math.round(score * 10) / 10

  return (
    <span
      className={clsx(
        'inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium',
        score >= 8
          ? 'bg-red-500/20 text-red-400'
          : score >= 6
            ? 'bg-amber-500/20 text-amber-400'
            : score >= 4
              ? 'bg-yellow-500/20 text-yellow-300'
              : 'bg-green-500/20 text-green-400'
      )}
    >
      {rounded}
    </span>
  )
}
