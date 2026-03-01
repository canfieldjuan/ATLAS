import { useState, useEffect, useRef, type ReactNode } from 'react'
import { Search, X, ChevronDown, ChevronUp } from 'lucide-react'

// ---------------------------------------------------------------------------
// FilterChip
// ---------------------------------------------------------------------------

interface FilterChipProps {
  label: string
  onClear: () => void
}

export function FilterChip({ label, onClear }: FilterChipProps) {
  return (
    <span className="inline-flex items-center gap-1 px-2 py-0.5 bg-cyan-500/10 text-cyan-400 text-xs rounded-full">
      {label}
      <button onClick={onClear} className="hover:text-white transition-colors">
        <X className="h-3 w-3" />
      </button>
    </span>
  )
}

// ---------------------------------------------------------------------------
// FilterSearch (debounced text input)
// ---------------------------------------------------------------------------

interface FilterSearchProps {
  label: string
  value: string
  onChange: (value: string) => void
  placeholder?: string
  debounceMs?: number
  icon?: boolean
}

export function FilterSearch({
  label,
  value,
  onChange,
  placeholder,
  debounceMs = 300,
  icon = true,
}: FilterSearchProps) {
  const [display, setDisplay] = useState(value)
  const timerRef = useRef<ReturnType<typeof setTimeout>>(undefined)

  // Sync external value changes (e.g. clear-all) and cancel pending debounce
  useEffect(() => {
    setDisplay(value)
    clearTimeout(timerRef.current)
  }, [value])

  const handleChange = (v: string) => {
    setDisplay(v)
    clearTimeout(timerRef.current)
    timerRef.current = setTimeout(() => onChange(v), debounceMs)
  }

  useEffect(() => () => clearTimeout(timerRef.current), [])

  return (
    <div>
      <label className="block text-xs text-slate-400 mb-1">{label}</label>
      <div className="relative">
        {icon && (
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
        )}
        <input
          type="text"
          value={display}
          onChange={(e) => handleChange(e.target.value)}
          placeholder={placeholder}
          className={`w-full ${icon ? 'pl-9' : 'pl-3'} pr-3 py-1.5 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50`}
        />
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// FilterSelect (dropdown)
// ---------------------------------------------------------------------------

interface FilterSelectProps {
  label: string
  value: string
  onChange: (value: string) => void
  options: { value: string; label: string }[]
  placeholder?: string
  className?: string
}

export function FilterSelect({
  label,
  value,
  onChange,
  options,
  placeholder = 'All',
  className,
}: FilterSelectProps) {
  return (
    <div className={className}>
      {label && <label className="block text-xs text-slate-400 mb-1">{label}</label>}
      <select
        value={value}
        onChange={(e) => onChange(e.target.value)}
        className="w-full px-3 py-1.5 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white focus:outline-none focus:border-cyan-500/50"
      >
        <option value="">{placeholder}</option>
        {options.map((o) => (
          <option key={o.value} value={o.value}>
            {o.label}
          </option>
        ))}
      </select>
    </div>
  )
}

// ---------------------------------------------------------------------------
// FilterBar (container with collapsible secondary row + chips)
// ---------------------------------------------------------------------------

interface ActiveFilter {
  key: string
  label: string
  onClear: () => void
}

interface FilterBarProps {
  children: ReactNode
  expanded?: ReactNode
  activeFilters?: ActiveFilter[]
  onClearAll?: () => void
}

export default function FilterBar({
  children,
  expanded,
  activeFilters,
  onClearAll,
}: FilterBarProps) {
  const [open, setOpen] = useState(false)
  const hasExpanded = !!expanded
  const hasChips = activeFilters && activeFilters.length > 0

  return (
    <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 space-y-3">
      {/* Primary row */}
      <div className="flex items-end gap-3">
        <div className="flex-1 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
          {children}
        </div>
        {hasExpanded && (
          <button
            onClick={() => setOpen(!open)}
            className="flex items-center gap-1 px-2 py-1.5 text-xs text-slate-400 hover:text-white transition-colors shrink-0"
          >
            {open ? <ChevronUp className="h-4 w-4" /> : <ChevronDown className="h-4 w-4" />}
            {open ? 'Less' : 'More'}
          </button>
        )}
      </div>

      {/* Secondary (collapsible) row */}
      {hasExpanded && open && (
        <div className="flex flex-wrap items-end gap-3 pt-2 border-t border-slate-700/30">
          {expanded}
        </div>
      )}

      {/* Active filter chips */}
      {hasChips && (
        <div className="flex flex-wrap items-center gap-2 pt-2 border-t border-slate-700/30">
          {activeFilters.map((f) => (
            <FilterChip key={f.key} label={f.label} onClear={f.onClear} />
          ))}
          {onClearAll && activeFilters.length > 1 && (
            <button
              onClick={onClearAll}
              className="flex items-center gap-1 text-xs text-slate-500 hover:text-white transition-colors"
            >
              <X className="h-3 w-3" />
              Clear all
            </button>
          )}
        </div>
      )}
    </div>
  )
}
