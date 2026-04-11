import { useState, useCallback, useEffect, useRef } from 'react'
import { Link, useLocation, useNavigate, useSearchParams } from 'react-router-dom'
import { Search, Plus, X, AlertTriangle, Loader2 } from 'lucide-react'
import { useAuth } from '../auth/AuthContext'
import {
  searchAvailableVendors,
  addTrackedVendor,
  removeTrackedVendor,
  listTrackedVendors,
  type VendorSearchResult,
} from '../api/client'

export default function Onboarding() {
  const { user, refreshUser } = useAuth()
  const location = useLocation()
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const returnPath = searchParams.get('back_to')?.trim() || '/'
  const [query, setQuery] = useState(searchParams.get('q')?.trim() || '')
  const [results, setResults] = useState<VendorSearchResult[]>([])
  const [added, setAdded] = useState<string[]>([])
  const [searching, setSearching] = useState(false)
  const [adding, setAdding] = useState<string | null>(null)
  const [removing, setRemoving] = useState<string | null>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  const limit = user?.vendor_limit ?? 5
  const onboardingBackTo = query.trim()
    ? `${location.pathname}?${new URLSearchParams({ q: query.trim() }).toString()}`
    : location.pathname

  useEffect(() => {
    let cancelled = false
    listTrackedVendors()
      .then(data => {
        if (!cancelled) setAdded(data.vendors.map(v => v.vendor_name))
      })
      .catch(() => {})
    return () => { cancelled = true }
  }, [])

  const doSearch = useCallback(async (q: string) => {
    if (q.length < 2) { setResults([]); return }
    setSearching(true)
    try {
      const data = await searchAvailableVendors(q)
      setResults(data.vendors)
    } catch {
      setResults([])
    } finally {
      setSearching(false)
    }
  }, [])

  function handleQueryChange(value: string) {
    setQuery(value)
    if (debounceRef.current) clearTimeout(debounceRef.current)
    if (value.length < 2) {
      setResults([])
      return
    }
    debounceRef.current = setTimeout(() => doSearch(value), 300)
  }

  useEffect(() => {
    const next = new URLSearchParams()
    if (returnPath && returnPath !== '/') next.set('back_to', returnPath)
    if (query.trim()) next.set('q', query.trim())
    if (next.toString() === searchParams.toString()) return
    setSearchParams(next, { replace: true })
  }, [query, returnPath, searchParams, setSearchParams])

  async function handleAdd(vendor: string) {
    if (added.length >= limit) return
    setAdding(vendor)
    try {
      const mode = user?.product === 'b2b_challenger' ? 'competitor' : 'own'
      await addTrackedVendor(vendor, mode)
      setAdded(prev => [...prev, vendor])
    } catch {
      // silent
    } finally {
      setAdding(null)
    }
  }

  async function handleRemove(vendor: string) {
    setRemoving(vendor)
    try {
      await removeTrackedVendor(vendor)
      setAdded(prev => prev.filter(v => v !== vendor))
    } catch {
      // silent
    } finally {
      setRemoving(null)
    }
  }

  async function handleContinue() {
    await refreshUser()
    navigate(returnPath)
  }

  return (
    <div className="min-h-screen bg-slate-900 text-white">
      <div className="max-w-2xl mx-auto py-12 px-4">
        <h1 className="text-2xl font-bold mb-2">Track your vendors</h1>
        <p className="text-slate-400 mb-6">
          Search for SaaS vendors to add to your dashboard.
          You can track up to <span className="text-cyan-400 font-medium">{limit}</span> vendors on your current plan.
        </p>

        {/* Search */}
        <div className="relative mb-4">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
          <input
            type="text"
            value={query}
            onChange={e => handleQueryChange(e.target.value)}
            placeholder="Search by vendor name (e.g. Salesforce, HubSpot, Zendesk)..."
            className="w-full pl-10 pr-4 py-2.5 bg-slate-800/60 border border-slate-700/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
          />
          {searching && <Loader2 className="absolute right-3 top-1/2 -translate-y-1/2 h-4 w-4 text-cyan-400 animate-spin" />}
        </div>

        {/* Results */}
        {results.length > 0 && (
          <div className="bg-slate-800/60 border border-slate-700/50 rounded-lg divide-y divide-slate-700/50 mb-6 max-h-80 overflow-y-auto">
            {results.map(r => {
              const isAdded = added.includes(r.vendor_name)
              return (
                <div key={r.vendor_name} className="flex items-center justify-between px-4 py-3">
                  <div className="min-w-0 flex-1">
                    <div className="text-sm text-white truncate">{r.vendor_name}</div>
                    <div className="text-xs text-slate-500 flex items-center gap-3">
                      {r.product_category && <span>{r.product_category}</span>}
                      {r.total_reviews != null && <span>{r.total_reviews} reviews</span>}
                      {r.avg_urgency != null && (
                        <span className="flex items-center gap-0.5">
                          <AlertTriangle className="h-3 w-3 text-amber-400" />
                          {r.avg_urgency.toFixed(1)} urgency
                        </span>
                      )}
                    </div>
                  </div>
                  <button
                    onClick={() => isAdded ? handleRemove(r.vendor_name) : handleAdd(r.vendor_name)}
                    disabled={adding === r.vendor_name || removing === r.vendor_name || (!isAdded && added.length >= limit)}
                    className={
                      isAdded
                        ? 'ml-3 p-1.5 rounded-lg bg-red-900/30 text-red-400 hover:bg-red-900/50 disabled:opacity-50'
                        : 'ml-3 p-1.5 rounded-lg bg-cyan-900/30 text-cyan-400 hover:bg-cyan-900/50 disabled:opacity-30'
                    }
                  >
                    {adding === r.vendor_name || removing === r.vendor_name ? (
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

        {/* Tracked vendors */}
        {added.length > 0 && (
          <div className="mb-6">
            <h3 className="text-sm font-medium text-slate-300 mb-2">
              Tracking ({added.length}/{limit})
            </h3>
            <div className="flex flex-wrap gap-2">
              {added.map(vendor => (
                <span
                  key={vendor}
                  className="inline-flex items-center gap-1.5 px-3 py-1 bg-cyan-900/30 text-cyan-300 text-sm rounded-full border border-cyan-700/50"
                >
                  <Link
                    to={`/vendors/${encodeURIComponent(vendor)}?${new URLSearchParams({ back_to: onboardingBackTo }).toString()}`}
                    className="hover:text-white"
                  >
                    {vendor}
                  </Link>
                  <button
                    onClick={() => handleRemove(vendor)}
                    disabled={removing === vendor}
                    className="hover:text-white disabled:opacity-50"
                  >
                    {removing === vendor ? (
                      <Loader2 className="h-3 w-3 animate-spin" />
                    ) : (
                      <X className="h-3 w-3" />
                    )}
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
          {added.length > 0 ? 'Continue to watchlists' : 'Skip for now'}
        </button>
      </div>
    </div>
  )
}
