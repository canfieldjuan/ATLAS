import { useMemo, useCallback } from 'react'
import { useSearchParams } from 'react-router-dom'

type FilterType = 'string' | 'number' | 'boolean'

interface FilterFieldConfig {
  type: FilterType
  label: string
  default?: string | number | boolean
}

export type FilterConfig = Record<string, FilterFieldConfig>

function coerce(raw: string | null, cfg: FilterFieldConfig): string | number | boolean | undefined {
  if (raw === null || raw === '') return undefined
  switch (cfg.type) {
    case 'number': {
      const n = Number(raw)
      return isNaN(n) ? undefined : n
    }
    case 'boolean':
      return raw === 'true'
    default:
      return raw
  }
}

function hasValue(v: unknown, cfg: FilterFieldConfig): boolean {
  if (v === undefined || v === null || v === '') return false
  if (cfg.default !== undefined) return v !== cfg.default
  return true
}

export default function useFilterParams<T extends Record<string, unknown>>(config: FilterConfig) {
  const [searchParams, setSearchParams] = useSearchParams()

  const filters = useMemo(() => {
    const result: Record<string, unknown> = {}
    for (const [key, fieldCfg] of Object.entries(config)) {
      const raw = searchParams.get(key)
      const val = coerce(raw, fieldCfg)
      result[key] = val !== undefined ? val : (fieldCfg.default ?? (fieldCfg.type === 'string' ? '' : undefined))
    }
    return result as T
  }, [config, searchParams])

  const setFilter = useCallback(
    (key: keyof T & string, value: unknown) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev)
          const fieldCfg = config[key]
          if (value === undefined || value === null || value === '' || (fieldCfg && value === fieldCfg.default)) {
            next.delete(key)
          } else {
            next.set(key, String(value))
          }
          return next
        },
        { replace: true },
      )
    },
    [config, setSearchParams],
  )

  const clearFilter = useCallback(
    (key: string) => {
      setSearchParams(
        (prev) => {
          const next = new URLSearchParams(prev)
          next.delete(key)
          return next
        },
        { replace: true },
      )
    },
    [setSearchParams],
  )

  const clearAll = useCallback(() => {
    setSearchParams(new URLSearchParams(), { replace: true })
  }, [setSearchParams])

  const activeFilterEntries = useMemo(() => {
    const entries: { key: string; label: string; value: string }[] = []
    for (const [key, fieldCfg] of Object.entries(config)) {
      const v = filters[key as keyof T]
      if (hasValue(v, fieldCfg)) {
        entries.push({ key, label: `${fieldCfg.label}: ${v}`, value: String(v) })
      }
    }
    return entries
  }, [config, filters])

  return { filters, setFilter, clearFilter, clearAll, activeFilterEntries }
}
