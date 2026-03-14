import { fmtCost, fmtTokens } from '../utils'

// eslint-disable-next-line @typescript-eslint/no-explicit-any
export default function CostTooltip({ active, payload, label }: any) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-lg border border-slate-700/80 bg-slate-800/95 px-3.5 py-2.5 shadow-xl backdrop-blur-sm">
      <p className="mb-1 text-[11px] font-medium text-slate-400">{label}</p>
      {/* eslint-disable-next-line @typescript-eslint/no-explicit-any */}
      {payload.filter((p: any) => p.name !== 'error_line').map((p: any, i: number) => (
        <p key={i} className={`font-mono text-sm ${p.name === 'errors' ? 'text-red-400' : 'text-slate-200'}`}>
          {p.name === 'cost' ? fmtCost(p.value)
            : p.name === 'calls' ? `${p.value} calls`
            : p.name === 'errors' ? `${p.value} errors`
            : fmtTokens(p.value)}
        </p>
      ))}
    </div>
  )
}
