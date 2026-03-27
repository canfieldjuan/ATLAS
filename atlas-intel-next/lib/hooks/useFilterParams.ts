import { useMemo, useCallback, useRef } from 'react'
import { useSearchParams, useRouter, usePathname } from 'next/navigation'

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
  const searchParams = useSearchParams()
  const router = useRouter()
  const pathname = usePathname()
  const searchString = searchParams.toString()
  const configRef = useRef(config)
  configRef.current = config

  const filters = useMemo(() => {
    const cfg = configRef.current
    const result: Record<string, unknown> = {}
    for (const [key, fieldCfg] of Object.entries(cfg)) {
      const raw = searchParams.get(key)
      const val = coerce(raw, fieldCfg)
      result[key] = val !== undefined ? val : (fieldCfg.default ?? (fieldCfg.type === 'string' ? '' : undefined))
    }
    return result as T
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchString])

  const updateParams = useCallback(
    (mutate: (prev: URLSearchParams) => URLSearchParams) => {
      const next = mutate(new URLSearchParams(searchParams.toString()))
      const qs = next.toString()
      router.replace(qs ? `${pathname}?${qs}` : pathname)
    },
    [searchParams, router, pathname],
  )

  const setFilter = useCallback(
    (key: keyof T & string, value: unknown) => {
      updateParams((prev) => {
        const next = new URLSearchParams(prev)
        const fieldCfg = configRef.current[key]
        if (value === undefined || value === null || value === '' || (fieldCfg && value === fieldCfg.default)) {
          next.delete(key)
        } else {
          next.set(key, String(value))
        }
        return next
      })
    },
    [updateParams],
  )

  const clearFilter = useCallback(
    (key: string) => {
      updateParams((prev) => {
        const next = new URLSearchParams(prev)
        next.delete(key)
        return next
      })
    },
    [updateParams],
  )

  const clearAll = useCallback(() => {
    router.replace(pathname)
  }, [router, pathname])

  const activeFilterEntries = useMemo(() => {
    const cfg = configRef.current
    const entries: { key: string; label: string; value: string }[] = []
    for (const [key, fieldCfg] of Object.entries(cfg)) {
      const v = filters[key as keyof T]
      if (hasValue(v, fieldCfg)) {
        entries.push({ key, label: `${fieldCfg.label}: ${v}`, value: String(v) })
      }
    }
    return entries
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchString])

  return { filters, setFilter, clearFilter, clearAll, activeFilterEntries }
}
