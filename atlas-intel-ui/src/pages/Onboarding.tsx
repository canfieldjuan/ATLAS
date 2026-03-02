import { useState, useCallback } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, Plus, X, Star, Loader2 } from 'lucide-react'
import { useAuth } from '../auth/AuthContext'
import {
  searchAvailableAsins,
  addTrackedAsin,
  type AsinSearchResult,
} from '../api/client'

export default function Onboarding() {
  const { user, refreshUser } = useAuth()
  const navigate = useNavigate()
  const [query, setQuery] = useState('')
  const [results, setResults] = useState<AsinSearchResult[]>([])
  const [added, setAdded] = useState<string[]>([])
  const [searching, setSearching] = useState(false)
  const [adding, setAdding] = useState<string | null>(null)

  const limit = user?.asin_limit ?? 5

  const doSearch = useCallback(async (q: string) => {
    if (q.length < 2) { setResults([]); return }
    setSearching(true)
    try {
      const data = await searchAvailableAsins(q)
      setResults(data.results)
    } catch {
      setResults([])
    } finally {
      setSearching(false)
    }
  }, [])

  async function handleAdd(asin: string) {
    if (added.length >= limit) return
    setAdding(asin)
    try {
      await addTrackedAsin(asin)
      setAdded(prev => [...prev, asin])
    } catch {
      // silent
    } finally {
      setAdding(null)
    }
  }

  function handleRemove(asin: string) {
    setAdded(prev => prev.filter(a => a !== asin))
  }

  async function handleContinue() {
    await refreshUser()
    navigate('/')
  }

  return (
    <div className="max-w-2xl mx-auto py-12 px-4">
      <h1 className="text-2xl font-bold text-white mb-2">Track your products</h1>
      <p className="text-slate-400 mb-6">
        Search for ASINs or product names to add to your dashboard.
        You can track up to <span className="text-cyan-400 font-medium">{limit}</span> products on your current plan.
      </p>

      {/* Search */}
      <div className="relative mb-4">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
        <input
          type="text"
          value={query}
          onChange={e => { setQuery(e.target.value); doSearch(e.target.value) }}
          placeholder="Search by ASIN, product name, or brand..."
          className="w-full pl-10 pr-4 py-2.5 bg-slate-800/60 border border-slate-700/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
        />
        {searching && <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-cyan-400 animate-spin" />}
      </div>

      {/* Results */}
      {results.length > 0 && (
        <div className="bg-slate-800/60 border border-slate-700/50 rounded-lg divide-y divide-slate-700/50 mb-6 max-h-72 overflow-y-auto">
          {results.map(r => {
            const isAdded = added.includes(r.asin)
            return (
              <div key={r.asin} className="flex items-center justify-between px-4 py-3">
                <div className="min-w-0 flex-1">
                  <div className="text-sm text-white truncate">{r.title || r.asin}</div>
                  <div className="text-xs text-slate-500 flex items-center gap-2">
                    <span>{r.asin}</span>
                    {r.brand && <span className="text-slate-400">{r.brand}</span>}
                    {r.average_rating != null && (
                      <span className="flex items-center gap-0.5">
                        <Star className="h-3 w-3 text-yellow-400" />
                        {r.average_rating.toFixed(1)}
                      </span>
                    )}
                  </div>
                </div>
                <button
                  onClick={() => isAdded ? handleRemove(r.asin) : handleAdd(r.asin)}
                  disabled={adding === r.asin || (!isAdded && added.length >= limit)}
                  className={
                    isAdded
                      ? 'ml-3 p-1.5 rounded-lg bg-red-900/30 text-red-400 hover:bg-red-900/50'
                      : 'ml-3 p-1.5 rounded-lg bg-cyan-900/30 text-cyan-400 hover:bg-cyan-900/50 disabled:opacity-30'
                  }
                >
                  {adding === r.asin ? (
                    <Loader2 className="h-4 w-4 animate-spin" />
                  ) : isAdded ? (
                    <X className="h-4 w-4" />
                  ) : (
                    <Plus className="h-4 w-4" />
                  )}
                </button>
              </div>
            )
          })}
        </div>
      )}

      {/* Added ASINs */}
      {added.length > 0 && (
        <div className="mb-6">
          <h3 className="text-sm font-medium text-slate-300 mb-2">
            Tracking ({added.length}/{limit})
          </h3>
          <div className="flex flex-wrap gap-2">
            {added.map(asin => (
              <span
                key={asin}
                className="inline-flex items-center gap-1.5 px-3 py-1 bg-cyan-900/30 text-cyan-300 text-sm rounded-full border border-cyan-700/50"
              >
                {asin}
                <button onClick={() => handleRemove(asin)} className="hover:text-white">
                  <X className="h-3 w-3" />
                </button>
              </span>
            ))}
          </div>
        </div>
      )}

      <button
        onClick={handleContinue}
        className="w-full py-2.5 bg-cyan-600 hover:bg-cyan-500 rounded-lg text-white font-medium transition-colors"
      >
        {added.length > 0 ? 'Continue to dashboard' : 'Skip for now'}
      </button>
    </div>
  )
}
