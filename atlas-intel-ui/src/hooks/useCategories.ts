import { useState, useEffect } from 'react'
import { fetchCategories } from '../api/client'

let _cache: string[] | null = null

export default function useCategories() {
  const [categories, setCategories] = useState<string[]>(_cache ?? [])
  const [loading, setLoading] = useState(_cache === null)

  useEffect(() => {
    if (_cache !== null) return
    let cancelled = false
    fetchCategories()
      .then((res) => {
        _cache = res.categories
        if (!cancelled) {
          setCategories(res.categories)
          setLoading(false)
        }
      })
      .catch(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  return { categories, loading }
}
