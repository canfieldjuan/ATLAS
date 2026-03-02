import { useState, useEffect, useCallback, useRef } from 'react'
import { useNavigate } from 'react-router-dom'
import { Search, Plus, X, Loader2, Users } from 'lucide-react'
import { useAuth } from '../../auth/AuthContext'
import {
  fetchTrackedVendors,
  addTrackedVendor,
  removeTrackedVendor,
  searchAvailableVendors,
  type TrackedVendor,
  type VendorSearchResult,
} from '../../api/b2bClient'

export default function B2BOnboarding() {
  const { user } = useAuth()
  const navigate = useNavigate()
  const [tracked, setTracked] = useState<TrackedVendor[]>([])
  const [results, setResults] = useState<VendorSearchResult[]>([])
  const [query, setQuery] = useState('')
  const [searching, setSearching] = useState(false)
  const [adding, setAdding] = useState<string | null>(null)
  const [removing, setRemoving] = useState<string | null>(null)
  const [error, setError] = useState('')
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(undefined)

  const isChallenger = user?.product === 'b2b_challenger'
  const defaultMode = isChallenger ? 'competitor' : 'own'
  const limit = user?.vendor_limit ?? 1

  useEffect(() => {
    fetchTrackedVendors().then(r => setTracked(r.vendors)).catch(() => {})
  }, [])

  const doSearch = useCallback(async (q: string) => {
    if (q.length < 2) { setResults([]); return }
    setSearching(true)
    try {
      const r = await searchAvailableVendors(q)
      setResults(r.vendors)
    } catch {
      setResults([])
    } finally {
      setSearching(false)
    }
  }, [])

  useEffect(() => {
    clearTimeout(debounceRef.current)
    debounceRef.current = setTimeout(() => doSearch(query), 300)
    return () => clearTimeout(debounceRef.current)
  }, [query, doSearch])

  const handleAdd = async (vendorName: string) => {
    setError('')
    setAdding(vendorName)
    try {
      const added = await addTrackedVendor(vendorName, defaultMode)
      setTracked(prev => [...prev, { ...added, avg_urgency: null, churn_intent_count: null, total_reviews: null, nps_proxy: null }])
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to add vendor')
    } finally {
      setAdding(null)
    }
  }

  const handleRemove = async (vendorName: string) => {
    setRemoving(vendorName)
    try {
      await removeTrackedVendor(vendorName)
      setTracked(prev => prev.filter(v => v.vendor_name !== vendorName))
    } catch {
      // ignore
    } finally {
      setRemoving(null)
    }
  }

  const trackedNames = new Set(tracked.map(v => v.vendor_name))
  const atLimit = tracked.length >= limit

  return (
    <div className="max-w-2xl mx-auto py-8 px-4 space-y-6">
      <div className="text-center space-y-2">
        <Users className="h-10 w-10 text-cyan-400 mx-auto" />
        <h1 className="text-2xl font-bold text-white">
          {isChallenger ? 'Track Competitor Vendors' : 'Track Your Vendors'}
        </h1>
        <p className="text-slate-400 text-sm">
          {isChallenger
            ? 'Search for competitors to monitor their churn signals and find high-intent leads.'
            : 'Search for your vendors to monitor churn signals, pain trends, and feature gaps.'
          }
        </p>
      </div>

      {error && (
        <div className="text-sm text-red-400 bg-red-900/20 border border-red-800/50 rounded-lg px-3 py-2">
          {error}
        </div>
      )}

      {/* Search */}
      <div className="relative">
        <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-500" />
        <input
          type="text"
          value={query}
          onChange={e => setQuery(e.target.value)}
          placeholder="Search vendors (e.g. Salesforce, HubSpot)..."
          className="w-full pl-10 pr-4 py-3 bg-slate-800/60 border border-slate-700/50 rounded-xl text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
        />
        {searching && <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-500 animate-spin" />}
      </div>

      {/* Search results */}
      {results.length > 0 && (
        <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl divide-y divide-slate-700/30">
          {results.map(r => {
            const isTracked = trackedNames.has(r.vendor_name)
            return (
              <div key={r.vendor_name} className="flex items-center justify-between px-4 py-3">
                <div>
                  <span className="text-white font-medium">{r.vendor_name}</span>
                  {r.product_category && (
                    <span className="ml-2 text-xs text-slate-500">{r.product_category}</span>
                  )}
                  <div className="text-xs text-slate-500 mt-0.5">
                    {r.total_reviews ?? 0} reviews
                    {r.avg_urgency != null && ` | urgency: ${r.avg_urgency.toFixed(1)}`}
                  </div>
                </div>
                {isTracked ? (
                  <span className="text-xs text-green-400">Tracking</span>
                ) : (
                  <button
                    onClick={() => handleAdd(r.vendor_name)}
                    disabled={atLimit || adding === r.vendor_name}
                    className="flex items-center gap-1 px-2 py-1 text-xs text-cyan-400 hover:text-cyan-300 disabled:text-slate-600 disabled:cursor-not-allowed transition-colors"
                  >
                    {adding === r.vendor_name ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : (
                      <Plus className="h-3 w-3" />
                    )}
                    Add
                  </button>
                )}
              </div>
            )
          })}
        </div>
      )}

      {/* Tracked vendors chips */}
      {tracked.length > 0 && (
        <div className="space-y-2">
          <h3 className="text-sm text-slate-400">
            Tracking ({tracked.length}/{limit})
          </h3>
          <div className="flex flex-wrap gap-2">
            {tracked.map(v => (
              <span
                key={v.vendor_name}
                className="inline-flex items-center gap-1 px-3 py-1.5 bg-cyan-500/10 text-cyan-400 text-sm rounded-full"
              >
                {v.vendor_name}
                <button
                  onClick={() => handleRemove(v.vendor_name)}
                  disabled={removing === v.vendor_name}
                  className="hover:text-white transition-colors disabled:opacity-50"
                >
                  <X className="h-3 w-3" />
                </button>
              </span>
            ))}
          </div>
        </div>
      )}

      {/* Continue */}
      <div className="flex justify-center pt-2">
        <button
          onClick={() => navigate('/b2b')}
          className="px-6 py-2.5 bg-cyan-600 hover:bg-cyan-500 rounded-lg text-white font-medium transition-colors"
        >
          {tracked.length > 0 ? 'Continue to dashboard' : 'Skip for now'}
        </button>
      </div>
    </div>
  )
}
