import {
  DIAGNOSTIC_DAY_OPTIONS,
  DIAGNOSTIC_TOP_N_OPTIONS,
} from '../lib/diagnosticFilters'

interface DiagnosticsQueryControlsProps {
  days: number
  diagnosticsTopN: number
  trendsTopN: number
  onDaysChange: (value: number) => void
  onDiagnosticsTopNChange: (value: number) => void
  onTrendsTopNChange: (value: number) => void
}

function QuerySelect({
  label,
  value,
  options,
  onChange,
}: {
  label: string
  value: number
  options: readonly number[]
  onChange: (value: number) => void
}) {
  return (
    <label className="flex items-center gap-2 text-sm text-slate-400">
      <span>{label}</span>
      <select
        value={value}
        onChange={(event) => onChange(Number(event.target.value))}
        className="bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white px-3 py-2 focus:outline-none focus:border-cyan-500/50"
      >
        {options.map((option) => (
          <option key={option} value={option}>
            {option}
          </option>
        ))}
      </select>
    </label>
  )
}

export default function DiagnosticsQueryControls({
  days,
  diagnosticsTopN,
  trendsTopN,
  onDaysChange,
  onDiagnosticsTopNChange,
  onTrendsTopNChange,
}: DiagnosticsQueryControlsProps) {
  return (
    <div className="flex flex-wrap items-center gap-2">
      <QuerySelect
        label="Window"
        value={days}
        options={DIAGNOSTIC_DAY_OPTIONS}
        onChange={onDaysChange}
      />
      <QuerySelect
        label="Diagnostics"
        value={diagnosticsTopN}
        options={DIAGNOSTIC_TOP_N_OPTIONS}
        onChange={onDiagnosticsTopNChange}
      />
      <QuerySelect
        label="Trends"
        value={trendsTopN}
        options={DIAGNOSTIC_TOP_N_OPTIONS}
        onChange={onTrendsTopNChange}
      />
    </div>
  )
}
