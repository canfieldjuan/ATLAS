import { useState, useEffect } from 'react'
import { fetchCategories } from '../api/client'

let _cache: string[] | null = null
let _inflight: Promise<{ categories: string[] }> | null = null

export default function useCategories() {
  const [categories, setCategories] = useState<string[]>(_cache ?? [])
  const [loading, setLoading] = useState(_cache === null)

  useEffect(() => {
    if (_cache !== null) return
    let cancelled = false

    if (!_inflight) {
      _inflight = fetchCategories()
    }

    _inflight
      .then((res) => {
        _cache = res.categories
        _inflight = null
        if (!cancelled) {
          setCategories(res.categories)
          setLoading(false)
        }
      })
      .catch(() => {
        _inflight = null
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  return { categories, loading }
}
