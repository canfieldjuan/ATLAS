import { useState, useEffect, useCallback, useRef } from 'react'

interface UseApiDataResult<T> {
  data: T | null
  loading: boolean
  error: Error | null
  refresh: () => void
  refreshing: boolean
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

  const load = useCallback(
    async (isRefresh: boolean) => {
      const id = ++requestIdRef.current
      if (isRefresh) {
        setRefreshing(true)
      } else {
        setLoading(true)
      }
      setError(null)
      try {
        const result = await fetcher()
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
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    deps,
  )

  useEffect(() => {
    mountedRef.current = true
    load(false)
    return () => {
      mountedRef.current = false
    }
  }, [load])

  const refresh = useCallback(() => {
    load(true)
  }, [load])

  return { data, loading, error, refresh, refreshing }
}
