import { useState, useEffect, useCallback, useRef } from 'react'

interface UseApiDataResult<T> {
  data: T | null
  loading: boolean
  error: Error | null
  refresh: () => void
  refreshing: boolean
}

function dependencyKey(deps: unknown[]): string {
  const seen = new WeakSet<object>()
  try {
    return JSON.stringify(deps, (_key, value: unknown) => {
      if (typeof value === 'bigint') {
        return `${value.toString()}n`
      }
      if (typeof value === 'symbol') {
        return value.toString()
      }
      if (typeof value === 'function') {
        return `[Function:${value.name || 'anonymous'}]`
      }
      if (value && typeof value === 'object') {
        if (seen.has(value)) {
          return '[Circular]'
        }
        seen.add(value)
      }
      return value
    }) ?? 'undefined'
  } catch {
    return deps.map((value) => Object.prototype.toString.call(value)).join('|')
  }
}

export default function useApiData<T>(
  fetcher: () => Promise<T>,
  deps: unknown[] = [],
): UseApiDataResult<T> {
  const [data, setData] = useState<T | null>(null)
  const [loading, setLoading] = useState(true)
  const [error, setError] = useState<Error | null>(null)
  const [refreshing, setRefreshing] = useState(false)
  const requestIdRef = useRef(0)
  const mountedRef = useRef(true)
  const fetcherRef = useRef(fetcher)
  const depsKey = dependencyKey(deps)

  useEffect(() => {
    fetcherRef.current = fetcher
  }, [fetcher])

  const load = useCallback(async (isRefresh: boolean) => {
    const id = ++requestIdRef.current
    if (isRefresh) {
      setRefreshing(true)
    } else {
      setLoading(true)
    }
    setError(null)
    try {
      const result = await fetcherRef.current()
      if (mountedRef.current && id === requestIdRef.current) {
        setData(result)
      }
    } catch (err) {
      if (mountedRef.current && id === requestIdRef.current) {
        setError(err instanceof Error ? err : new Error(String(err)))
      }
    } finally {
      if (mountedRef.current && id === requestIdRef.current) {
        setLoading(false)
        setRefreshing(false)
      }
    }
  }, [])

  useEffect(() => {
    mountedRef.current = true
    requestIdRef.current += 1
    const timeout = window.setTimeout(() => {
      load(false)
    }, 0)
    return () => {
      mountedRef.current = false
      window.clearTimeout(timeout)
    }
  }, [depsKey, load])

  const refresh = useCallback(() => {
    load(true)
  }, [load])

  return { data, loading, error, refresh, refreshing }
}
