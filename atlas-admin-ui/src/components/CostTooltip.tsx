import { fmtCost, fmtTokens } from '../utils'

type CostTooltipEntry = {
  name?: string
  value?: number
}

type CostTooltipProps = {
  active?: boolean
  payload?: CostTooltipEntry[]
  label?: string
}

export default function CostTooltip({ active, payload, label }: CostTooltipProps) {
  if (!active || !payload?.length) return null
  return (
    <div className="rounded-lg border border-slate-700/80 bg-slate-800/95 px-3.5 py-2.5 shadow-xl backdrop-blur-sm">
      <p className="mb-1 text-[11px] font-medium text-slate-400">{label}</p>
      {payload.filter((item) => item.name !== 'error_line').map((item, index) => (
        <p key={index} className={`font-mono text-sm ${item.name === 'errors' ? 'text-red-400' : 'text-slate-200'}`}>
          {item.name === 'cost' ? fmtCost(item.value ?? 0)
            : item.name === 'calls' ? `${item.value ?? 0} calls`
            : item.name === 'errors' ? `${item.value ?? 0} errors`
            : fmtTokens(item.value ?? 0)}
        </p>
      ))}
    </div>
  )
}
