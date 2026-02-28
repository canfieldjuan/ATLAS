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
  // Monotonic counter — each load() call increments; only the latest
  // request is allowed to write state, which eliminates race conditions
  // when deps change rapidly or the user clicks refresh while a fetch
  // is already in-flight.
  const requestRef = useRef(0)
  // Track whether we have successfully loaded data at least once.
  // Used to decide whether a retry should show the full loading skeleton
  // (no data yet) vs. the lighter refreshing indicator (data on screen).
  const hasDataRef = useRef(false)

  const load = useCallback(
    async (isRefresh: boolean) => {
      const id = ++requestRef.current
      if (isRefresh && hasDataRef.current) {
        setRefreshing(true)
      } else {
        setLoading(true)
      }
      setError(null)
      try {
        const result = await fetcher()
        if (requestRef.current === id) {
          hasDataRef.current = true
          setData(result)
        }
      } catch (err) {
        if (requestRef.current === id) {
          setError(err instanceof Error ? err : new Error(String(err)))
        }
      } finally {
        if (requestRef.current === id) {
          setLoading(false)
          setRefreshing(false)
        }
      }
    },
    // eslint-disable-next-line react-hooks/exhaustive-deps
    deps,
  )

  useEffect(() => {
    load(false)
  }, [load])

  const refresh = useCallback(() => {
    load(true)
  }, [load])

  return { data, loading, error, refresh, refreshing }
}
