import { useState, useRef, useCallback, useEffect, useMemo } from 'react'
import { Link, useLocation, useSearchParams } from 'react-router-dom'
import { Search, Target, TrendingUp, TrendingDown, Shield, Quote, AlertTriangle, ArrowRight, Loader2, ShieldAlert, CheckCircle2, XCircle, Lightbulb, Clock, Zap, Copy, Download, Check } from 'lucide-react'
import { predictWinLoss, searchAvailableVendors, fetchRecentPredictions, fetchPredictionById, compareWinLoss, downloadPredictionCsv } from '../api/client'
import type { WinLossPrediction, WinLossFactor, WinLossDataGate, RecentPrediction, WinLossCompareResponse } from '../api/client'

function _timeAgo(iso: string): string {
  const diff = Date.now() - new Date(iso).getTime()
  const mins = Math.floor(diff / 60000)
  if (mins < 1) return 'just now'
  if (mins < 60) return `${mins}m ago`
  const hrs = Math.floor(mins / 60)
  if (hrs < 24) return `${hrs}h ago`
  const days = Math.floor(hrs / 24)
  return `${days}d ago`
}

// -- Skeleton loader (matches results layout) --------------------------------

function PredictionSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="bg-slate-800/60 rounded-xl border border-slate-700/50 p-6 flex flex-col items-center">
          <div className="w-[200px] h-[200px] rounded-full bg-slate-700/50" />
          <div className="h-4 w-48 bg-slate-700/50 rounded mt-4" />
          <div className="flex gap-4 mt-3">
            <div className="h-3 w-16 bg-slate-700/50 rounded" />
            <div className="h-3 w-16 bg-slate-700/50 rounded" />
            <div className="h-3 w-16 bg-slate-700/50 rounded" />
          </div>
        </div>
        <div className="lg:col-span-2 space-y-3">
          <div className="h-6 w-40 bg-slate-700/50 rounded" />
          <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
            {[1, 2, 3, 4, 5, 6].map(i => (
              <div key={i} className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
                <div className="flex justify-between mb-2">
                  <div className="h-4 w-32 bg-slate-700/50 rounded" />
                  <div className="h-4 w-10 bg-slate-700/50 rounded" />
                </div>
                <div className="h-2 w-full bg-slate-700/50 rounded-full mb-2" />
                <div className="h-3 w-full bg-slate-700/50 rounded" />
              </div>
            ))}
          </div>
        </div>
      </div>
      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="bg-slate-800/60 rounded-xl border border-slate-700/50 p-6">
          <div className="h-5 w-40 bg-slate-700/50 rounded mb-4" />
          {[1, 2, 3, 4].map(i => (
            <div key={i} className="h-10 bg-slate-900/50 rounded-lg mb-3" />
          ))}
        </div>
        <div className="bg-slate-800/60 rounded-xl border border-slate-700/50 p-6">
          <div className="h-5 w-40 bg-slate-700/50 rounded mb-4" />
          {[1, 2, 3].map(i => (
            <div key={i} className="h-10 bg-slate-900/50 rounded-lg mb-3" />
          ))}
        </div>
      </div>
    </div>
  )
}

// -- Insufficient data panel -------------------------------------------------

function InsufficientDataPanel({ vendor, gates, verdict }: { vendor: string; gates: WinLossDataGate[]; verdict: string }) {
  return (
    <div className="space-y-6">
      <div className="bg-amber-950/30 border border-amber-700/40 rounded-xl p-6">
        <div className="flex items-start gap-3">
          <ShieldAlert className="w-6 h-6 text-amber-400 shrink-0 mt-0.5" />
          <div>
            <h2 className="text-lg font-semibold text-amber-200">Insufficient Data for Reliable Prediction</h2>
            <p className="text-sm text-amber-300/80 mt-1">{verdict}</p>
          </div>
        </div>
      </div>

      <div className="bg-slate-800/60 rounded-xl border border-slate-700/50 p-6">
        <h3 className="text-sm font-medium text-slate-300 mb-4">Data Coverage for {vendor}</h3>
        <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
          {gates.map(g => {
            const pct = Math.min(100, Math.round((g.actual / Math.max(g.required, 1)) * 100))
            return (
              <div key={g.factor} className="bg-slate-900/50 rounded-lg p-4">
                <div className="flex items-center justify-between mb-2">
                  <span className="text-sm text-slate-300">{g.factor}</span>
                  {g.sufficient
                    ? <CheckCircle2 className="w-4 h-4 text-green-400" />
                    : <XCircle className="w-4 h-4 text-red-400" />
                  }
                </div>
                <div className="w-full bg-slate-700 rounded-full h-2 mb-1">
                  <div
                    className={`h-2 rounded-full transition-all ${g.sufficient ? 'bg-green-500' : 'bg-red-500'}`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
                <span className="text-xs text-slate-500">{g.actual} / {g.required} required</span>
              </div>
            )
          })}
        </div>
      </div>
    </div>
  )
}

// -- Probability gauge -------------------------------------------------------

function ProbabilityGauge({ value, confidence }: { value: number; confidence: string }) {
  const pct = Math.round(value * 100)
  const radius = 80
  const circumference = 2 * Math.PI * radius
  const offset = circumference - (value * circumference)

  let color = '#ef4444'
  if (pct >= 70) color = '#22c55e'
  else if (pct >= 50) color = '#f59e0b'
  else if (pct >= 30) color = '#f97316'

  return (
    <div className="relative flex flex-col items-center">
      <svg width="200" height="200" viewBox="0 0 200 200" className="transform -rotate-90">
        <circle cx="100" cy="100" r={radius} fill="none" stroke="#1e293b" strokeWidth="12" />
        <circle
          cx="100" cy="100" r={radius} fill="none"
          stroke={color} strokeWidth="12" strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          className="transition-all duration-1000 ease-out"
        />
      </svg>
      <div className="absolute inset-0 flex flex-col items-center justify-center">
        <span className="text-5xl font-bold text-white">{pct}%</span>
        <span className="text-sm text-slate-400 mt-1 uppercase tracking-wider">{confidence} confidence</span>
      </div>
    </div>
  )
}

// -- Factor bar --------------------------------------------------------------

function FactorBar({ factor }: { factor: WinLossFactor }) {
  if (factor.gated) {
    return (
      <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/30 opacity-60">
        <div className="flex justify-between items-center mb-2">
          <span className="text-sm font-medium text-slate-400">{factor.name}</span>
          <span className="text-xs text-slate-500 bg-slate-700/50 px-2 py-0.5 rounded">No Data</span>
        </div>
        <div className="w-full bg-slate-700 rounded-full h-2 mb-2" />
        <p className="text-xs text-slate-500">{factor.evidence}</p>
      </div>
    )
  }

  const pct = Math.round(factor.score * 100)
  let barColor = 'bg-red-500'
  if (pct >= 70) barColor = 'bg-green-500'
  else if (pct >= 50) barColor = 'bg-amber-500'
  else if (pct >= 30) barColor = 'bg-orange-500'

  return (
    <div className="bg-slate-800/50 rounded-lg p-4 border border-slate-700/50">
      <div className="flex justify-between items-center mb-2">
        <span className="text-sm font-medium text-slate-200">{factor.name}</span>
        <span className="text-sm font-bold text-white">{pct}%</span>
      </div>
      <div className="w-full bg-slate-700 rounded-full h-2 mb-2">
        <div className={`${barColor} h-2 rounded-full transition-all duration-700`} style={{ width: `${pct}%` }} />
      </div>
      <p className="text-xs text-slate-400">{factor.evidence}</p>
      {factor.data_points > 0 && (
        <span className="text-xs text-slate-500 mt-1 inline-block">{factor.data_points.toLocaleString()} data points</span>
      )}
    </div>
  )
}

function buildPredictorSearchParams({
  vendor,
  vendorB,
  companySize,
  industry,
  compareMode,
}: {
  vendor: string
  vendorB: string
  companySize: string
  industry: string
  compareMode: boolean
}) {
  const next = new URLSearchParams()
  if (vendor.trim()) next.set('vendor', vendor.trim())
  if (compareMode) next.set('compare', '1')
  if (compareMode && vendorB.trim()) next.set('vendor_b', vendorB.trim())
  if (companySize.trim()) next.set('company_size', companySize.trim())
  if (industry.trim()) next.set('industry', industry.trim())
  return next
}

function buildBackToPath(pathname: string, search: string) {
  return search ? `${pathname}${search}` : pathname
}

function buildVendorWorkspacePath(vendorName: string, backTo: string) {
  const next = new URLSearchParams()
  next.set('back_to', backTo)
  return `/vendors/${encodeURIComponent(vendorName)}?${next.toString()}`
}

function buildVendorScopedPath(pathname: string, vendorName: string, backTo: string) {
  const next = new URLSearchParams()
  next.set('vendor', vendorName)
  next.set('back_to', backTo)
  return `${pathname}?${next.toString()}`
}

// -- Main page ---------------------------------------------------------------

export default function WinLossPredictor() {
  const location = useLocation()
  const [searchParams, setSearchParams] = useSearchParams()
  const initialVendor = searchParams.get('vendor')?.trim() || ''
  const initialVendorB = searchParams.get('vendor_b')?.trim() || ''
  const initialCompanySize = searchParams.get('company_size')?.trim() || ''
  const initialIndustry = searchParams.get('industry')?.trim() || ''
  const initialCompareMode = searchParams.get('compare') === '1'
  const [vendorInput, setVendorInput] = useState(initialVendor)
  const [companySize, setCompanySize] = useState(initialCompanySize)
  const [industry, setIndustry] = useState(initialIndustry)
  const [prediction, setPrediction] = useState<WinLossPrediction | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [suggestions, setSuggestions] = useState<string[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const [recentPredictions, setRecentPredictions] = useState<RecentPrediction[]>([])
  const [compareMode, setCompareMode] = useState(initialCompareMode)
  const [vendorBInput, setVendorBInput] = useState(initialVendorB)
  const [comparison, setComparison] = useState<WinLossCompareResponse | null>(null)
  const debounceRef = useRef<ReturnType<typeof setTimeout> | null>(null)

  // Load recent predictions on mount
  useEffect(() => {
    fetchRecentPredictions(10).then(res => setRecentPredictions(res.predictions ?? [])).catch(() => {})
  }, [])

  const [suggestionsB, setSuggestionsB] = useState<string[]>([])
  const [showSuggestionsB, setShowSuggestionsB] = useState(false)
  const debounceRefB = useRef<ReturnType<typeof setTimeout> | null>(null)

  const handleSearch = useCallback((query: string) => {
    setVendorInput(query)
    if (debounceRef.current) clearTimeout(debounceRef.current)

    if (query.length < 2) {
      setSuggestions([])
      setShowSuggestions(false)
      return
    }

    debounceRef.current = setTimeout(async () => {
      try {
        const res = await searchAvailableVendors(query)
        const names = (res.vendors || []).map((m) => m.vendor_name)
        setSuggestions(names.slice(0, 6))
        setShowSuggestions(names.length > 0)
      } catch {
        setSuggestions([])
      }
    }, 300)
  }, [])

  const handleSearchB = useCallback((query: string) => {
    setVendorBInput(query)
    if (debounceRefB.current) clearTimeout(debounceRefB.current)

    if (query.length < 2) {
      setSuggestionsB([])
      setShowSuggestionsB(false)
      return
    }

    debounceRefB.current = setTimeout(async () => {
      try {
        const res = await searchAvailableVendors(query)
        const names = (res.vendors || []).map((m) => m.vendor_name)
        setSuggestionsB(names.slice(0, 6))
        setShowSuggestionsB(names.length > 0)
      } catch {
        setSuggestionsB([])
      }
    }, 300)
  }, [])

  useEffect(() => {
    return () => {
      if (debounceRef.current) clearTimeout(debounceRef.current)
      if (debounceRefB.current) clearTimeout(debounceRefB.current)
    }
  }, [])

  const suggestionsContainerRef = useRef<HTMLDivElement>(null)
  const predictorBackTo = useMemo(() => {
    const next = buildPredictorSearchParams({
      vendor: vendorInput,
      vendorB: vendorBInput,
      companySize,
      industry,
      compareMode,
    })
    const query = next.toString()
    return query ? `${location.pathname}?${query}` : buildBackToPath(location.pathname, location.search)
  }, [compareMode, companySize, industry, location.pathname, location.search, vendorBInput, vendorInput])

  useEffect(() => {
    function handleClickOutside(e: MouseEvent) {
      if (suggestionsContainerRef.current && !suggestionsContainerRef.current.contains(e.target as Node)) {
        setShowSuggestions(false)
        setShowSuggestionsB(false)
      }
    }
    document.addEventListener('mousedown', handleClickOutside)
    return () => document.removeEventListener('mousedown', handleClickOutside)
  }, [])

  useEffect(() => {
    const next = buildPredictorSearchParams({
      vendor: vendorInput,
      vendorB: vendorBInput,
      companySize,
      industry,
      compareMode,
    })
    if (next.toString() === searchParams.toString()) return
    setSearchParams(next, { replace: true })
  }, [compareMode, companySize, industry, searchParams, setSearchParams, vendorBInput, vendorInput])

  const handlePredict = async () => {
    if (!vendorInput.trim()) return
    setLoading(true)
    setError('')
    setPrediction(null)
    setComparison(null)
    setShowSuggestions(false)

    try {
      if (compareMode && !vendorBInput.trim()) {
        setError('Enter both vendors for comparison')
        setLoading(false)
        return
      }
      if (compareMode) {
        const result = await compareWinLoss({
          vendor_a: vendorInput.trim(),
          vendor_b: vendorBInput.trim(),
          company_size: companySize || undefined,
          industry: industry || undefined,
        })
        setComparison(result)
      } else {
        const result = await predictWinLoss({
          vendor_name: vendorInput.trim(),
          company_size: companySize || undefined,
          industry: industry || undefined,
        })
        setPrediction(result)
      }
      fetchRecentPredictions(10).then(res => setRecentPredictions(res.predictions ?? [])).catch(() => {})
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Prediction failed')
    } finally {
      setLoading(false)
    }
  }

  const [copied, setCopied] = useState(false)

  const copyPredictionSummary = useCallback((p: WinLossPrediction) => {
    const pct = p.win_probability != null ? Math.round(p.win_probability * 100) : null
    const lines: string[] = [
      `Win/Loss Prediction: ${p.vendor_name}`,
      `Win Probability: ${pct}% (${p.confidence} confidence)`,
      `Verdict: ${p.verdict}`,
      '',
    ]
    if (p.factors.length > 0) {
      lines.push('Scoring Factors:')
      for (const f of p.factors) {
        if (f.gated) {
          lines.push(`  ${f.name}: No Data`)
        } else {
          lines.push(`  ${f.name}: ${Math.round(f.score * 100)}% - ${f.evidence}`)
        }
      }
      lines.push('')
    }
    if (p.switching_triggers.length > 0) {
      lines.push('Top Switching Triggers:')
      for (const t of p.switching_triggers) {
        lines.push(`  - ${t.trigger} (${t.frequency} mentions, urgency ${t.urgency.toFixed(1)})`)
      }
      lines.push('')
    }
    if (p.recommended_approach) {
      lines.push(`Recommended Approach: ${p.recommended_approach}`)
      if (p.lead_with.length > 0) lines.push(`Lead With: ${p.lead_with.join(', ')}`)
      if (p.talking_points.length > 0) {
        lines.push('Talking Points:')
        p.talking_points.forEach((tp, i) => lines.push(`  ${i + 1}. ${tp}`))
      }
      if (p.timing_advice) lines.push(`Timing: ${p.timing_advice}`)
      if (p.risk_factors.length > 0) {
        lines.push('Risks:')
        p.risk_factors.forEach(rf => lines.push(`  - ${rf}`))
      }
    }
    const text = lines.join('\n')
    if (navigator.clipboard) {
      navigator.clipboard.writeText(text).then(() => {
        setCopied(true)
        setTimeout(() => setCopied(false), 2000)
      }).catch(() => {
        setError('Failed to copy to clipboard')
      })
    } else {
      setError('Clipboard not available in this context')
    }
  }, [])

  const loadSavedPrediction = async (id: string) => {
    setLoading(true)
    setError('')
    setPrediction(null)
    try {
      const result = await fetchPredictionById(id)
      setPrediction(result)
      setVendorInput(result.vendor_name)
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to load prediction')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div>
        <h1 className="text-2xl font-bold text-white flex items-center gap-2">
          <Target className="w-7 h-7 text-cyan-400" />
          Win/Loss Predictor
        </h1>
        <p className="text-slate-400 mt-1">
          Predict your win probability against any vendor based on real displacement data, churn signals, and buyer behavior.
        </p>
      </div>

      {/* Input Form */}
      <div className="bg-slate-800/60 rounded-xl border border-slate-700/50 p-6">
        {/* Mode toggle */}
        <div className="flex items-center gap-3 mb-4">
          <button
            onClick={() => { setCompareMode(false); setComparison(null); setShowSuggestions(false); setShowSuggestionsB(false); setError('') }}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${!compareMode ? 'bg-cyan-600 text-white' : 'bg-slate-700 text-slate-400 hover:text-slate-200'}`}
          >
            Single Vendor
          </button>
          <button
            onClick={() => { setCompareMode(true); setPrediction(null); setShowSuggestions(false); setShowSuggestionsB(false); setError('') }}
            className={`px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${compareMode ? 'bg-cyan-600 text-white' : 'bg-slate-700 text-slate-400 hover:text-slate-200'}`}
          >
            Compare Vendors
          </button>
        </div>

        <div ref={suggestionsContainerRef} className="grid grid-cols-1 md:grid-cols-4 gap-4">
          <div className={`${compareMode ? '' : 'md:col-span-2'} relative`}>
            <label className="block text-sm text-slate-400 mb-1">{compareMode ? 'Vendor A' : 'Target Vendor'}</label>
            <div className="relative">
              <Search className="absolute left-3 top-2.5 w-4 h-4 text-slate-500" />
              <input
                type="text"
                value={vendorInput}
                onChange={e => handleSearch(e.target.value)}
                onKeyDown={e => e.key === 'Enter' && handlePredict()}
                placeholder="e.g. Zendesk, Salesforce, HubSpot..."
                className="w-full pl-10 pr-4 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
              />
            </div>
            {showSuggestions && suggestions.length > 0 && (
              <div className="absolute z-10 mt-1 w-full bg-slate-800 border border-slate-600 rounded-lg shadow-lg max-h-48 overflow-y-auto">
                {suggestions.map(name => (
                  <button
                    key={name}
                    onClick={() => { setVendorInput(name); setShowSuggestions(false) }}
                    className="w-full text-left px-4 py-2 text-sm text-slate-200 hover:bg-slate-700 first:rounded-t-lg last:rounded-b-lg"
                  >
                    {name}
                  </button>
                ))}
              </div>
            )}
          </div>

          {compareMode && (
            <div className="relative">
              <label className="block text-sm text-slate-400 mb-1">Vendor B</label>
              <div className="relative">
                <Search className="absolute left-3 top-2.5 w-4 h-4 text-slate-500" />
                <input
                  type="text"
                  value={vendorBInput}
                  onChange={e => handleSearchB(e.target.value)}
                  onKeyDown={e => e.key === 'Enter' && handlePredict()}
                  placeholder="e.g. Freshdesk, Intercom..."
                  className="w-full pl-10 pr-4 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
                />
              </div>
              {showSuggestionsB && suggestionsB.length > 0 && (
                <div className="absolute z-10 mt-1 w-full bg-slate-800 border border-slate-600 rounded-lg shadow-lg max-h-48 overflow-y-auto">
                  {suggestionsB.map(name => (
                    <button
                      key={name}
                      onClick={() => { setVendorBInput(name); setShowSuggestionsB(false) }}
                      className="w-full text-left px-4 py-2 text-sm text-slate-200 hover:bg-slate-700 first:rounded-t-lg last:rounded-b-lg"
                    >
                      {name}
                    </button>
                  ))}
                </div>
              )}
            </div>
          )}

          <div>
            <label className="block text-sm text-slate-400 mb-1">Company Size</label>
            <select
              value={companySize}
              onChange={e => setCompanySize(e.target.value)}
              className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white focus:border-cyan-500 focus:outline-none"
            >
              <option value="">Any size</option>
              <option value="startup">Startup (1-50)</option>
              <option value="smb">SMB (50-200)</option>
              <option value="mid_market">Mid-Market (200-1000)</option>
              <option value="enterprise">Enterprise (1000+)</option>
            </select>
          </div>

          <div>
            <label className="block text-sm text-slate-400 mb-1">Industry</label>
            <input
              type="text"
              value={industry}
              onChange={e => setIndustry(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && handlePredict()}
              placeholder="e.g. fintech, healthcare..."
              className="w-full px-3 py-2 bg-slate-900 border border-slate-600 rounded-lg text-white placeholder-slate-500 focus:border-cyan-500 focus:outline-none"
            />
          </div>
        </div>

        <button
          onClick={handlePredict}
          disabled={loading || !vendorInput.trim() || (compareMode && !vendorBInput.trim())}
          className="mt-4 px-6 py-2.5 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-600 disabled:cursor-not-allowed text-white rounded-lg font-medium transition-colors flex items-center gap-2"
        >
          {loading ? <Loader2 className="w-4 h-4 animate-spin" /> : <Target className="w-4 h-4" />}
          {loading ? 'Analyzing...' : 'Predict Win Probability'}
        </button>

        {error && <p className="mt-3 text-red-400 text-sm">{error}</p>}
      </div>

      {/* Recent Predictions */}
      {recentPredictions.length > 0 && !prediction && !comparison && !loading && (
        <div className="bg-slate-800/60 rounded-xl border border-slate-700/50 p-4">
          <h3 className="text-sm font-medium text-slate-400 mb-3">Recent Predictions</h3>
          <div className="flex flex-wrap gap-2">
            {recentPredictions.map(rp => {
              const pct = rp.win_probability != null ? Math.round(rp.win_probability * 100) : null
              let dotColor = 'bg-red-500'
              if (rp.is_gated) dotColor = 'bg-slate-500'
              else if (pct >= 70) dotColor = 'bg-green-500'
              else if (pct >= 50) dotColor = 'bg-amber-500'
              else if (pct >= 30) dotColor = 'bg-orange-500'
              const ago = _timeAgo(rp.created_at)
              return (
                <button
                  key={rp.prediction_id}
                  onClick={() => loadSavedPrediction(rp.prediction_id)}
                  className="flex items-center gap-2 bg-slate-900/50 hover:bg-slate-700/50 rounded-lg px-3 py-2 text-sm transition-colors"
                >
                  <span className={`w-2 h-2 rounded-full ${dotColor}`} />
                  <span className="text-slate-200">{rp.vendor_name}</span>
                  {!rp.is_gated && <span className="text-slate-500">{pct}%</span>}
                  <span className="text-slate-600 text-xs">{ago}</span>
                </button>
              )
            })}
          </div>
        </div>
      )}

      {/* Loading skeleton */}
      {loading && <PredictionSkeleton />}

      {/* Comparison results */}
      {comparison && (
        <div className="space-y-6">
          {/* Easier target banner */}
          <div className="bg-slate-800/60 rounded-xl border border-cyan-700/40 p-4 flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Target className="w-5 h-5 text-cyan-400" />
              <div>
                {comparison.easier_target === 'tie' ? (
                  <span className="text-lg font-semibold text-slate-300">Both vendors scored equally</span>
                ) : (
                  <>
                    <span className="text-sm text-slate-400">Easier target: </span>
                    <span className="text-lg font-semibold text-white">{comparison.easier_target}</span>
                  </>
                )}
              </div>
            </div>
            {comparison.easier_target !== 'tie' && (
              <div className="text-right">
                <span className="text-2xl font-bold text-cyan-400">+{Math.round(comparison.probability_delta * 100)}%</span>
                <span className="block text-xs text-slate-500">probability advantage</span>
              </div>
            )}
          </div>

          {/* Side-by-side gauges */}
          <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
            {[comparison.vendor_a, comparison.vendor_b].map(pred => (
              <div key={pred.vendor_name} className="bg-slate-800/60 rounded-xl border border-slate-700/50 p-6">
                <h3 className="text-center text-sm font-medium text-slate-400 mb-4">
                  vs. {pred.vendor_name}
                  {pred.vendor_name === comparison.easier_target && (
                    <span className="ml-2 text-xs bg-cyan-900/50 text-cyan-300 px-2 py-0.5 rounded">Easier</span>
                  )}
                </h3>
                {pred.is_gated ? (
                  <div className="flex flex-col items-center">
                    <div className="w-[160px] h-[160px] rounded-full bg-slate-700/30 flex items-center justify-center">
                      <ShieldAlert className="w-10 h-10 text-amber-400" />
                    </div>
                    <p className="text-sm text-amber-300/80 mt-3 text-center">Insufficient data</p>
                  </div>
                ) : (
                  <div className="flex flex-col items-center">
                    <ProbabilityGauge value={pred.win_probability} confidence={pred.confidence} />
                    <p className="text-xs text-slate-500 mt-2">
                      {pred.weights_source === 'calibrated'
                        ? `Calibrated (v${pred.calibration_version ?? '?'})`
                        : 'Default weights'}
                    </p>
                    <div className="mt-4 flex flex-wrap justify-center gap-2 text-xs">
                      <Link
                        to={buildVendorWorkspacePath(pred.vendor_name, predictorBackTo)}
                        className="rounded-lg border border-slate-700 bg-slate-900 px-2.5 py-1.5 text-slate-200 transition-colors hover:bg-slate-800"
                      >
                        Vendor workspace
                      </Link>
                      <Link
                        to={buildVendorScopedPath('/evidence', pred.vendor_name, predictorBackTo)}
                        className="rounded-lg border border-slate-700 bg-slate-900 px-2.5 py-1.5 text-slate-200 transition-colors hover:bg-slate-800"
                      >
                        Evidence
                      </Link>
                      <Link
                        to={buildVendorScopedPath('/reports', pred.vendor_name, predictorBackTo)}
                        className="rounded-lg border border-slate-700 bg-slate-900 px-2.5 py-1.5 text-slate-200 transition-colors hover:bg-slate-800"
                      >
                        Reports
                      </Link>
                      <Link
                        to={buildVendorScopedPath('/opportunities', pred.vendor_name, predictorBackTo)}
                        className="rounded-lg border border-slate-700 bg-slate-900 px-2.5 py-1.5 text-slate-200 transition-colors hover:bg-slate-800"
                      >
                        Opportunities
                      </Link>
                    </div>
                  </div>
                )}
              </div>
            ))}
          </div>

          {/* Factor comparison bars */}
          <div className="bg-slate-800/60 rounded-xl border border-slate-700/50 p-6">
            <h2 className="text-lg font-semibold text-white flex items-center gap-2 mb-4">
              <TrendingUp className="w-5 h-5 text-cyan-400" />
              Factor Comparison
            </h2>
            <div className="space-y-4">
              {(comparison.factor_comparison ?? []).map(fc => {
                const aPct = Math.round(fc.vendor_a_score * 100)
                const bPct = Math.round(fc.vendor_b_score * 100)
                return (
                  <div key={fc.name}>
                    <div className="flex justify-between items-center mb-1">
                      <span className="text-sm text-slate-300">{fc.name}</span>
                      <span className={`text-xs px-2 py-0.5 rounded ${
                        fc.advantage === 'a' ? 'bg-cyan-900/50 text-cyan-300'
                          : fc.advantage === 'b' ? 'bg-purple-900/50 text-purple-300'
                          : 'bg-slate-700 text-slate-400'
                      }`}>
                        {fc.advantage === 'a' ? comparison.vendor_a.vendor_name
                          : fc.advantage === 'b' ? comparison.vendor_b.vendor_name
                          : 'Tied'}
                      </span>
                    </div>
                    <div className="flex gap-2 items-center">
                      <span className="text-xs text-slate-500 w-8 text-right">{aPct}%</span>
                      <div className="flex-1 flex gap-1 h-3">
                        <div className="flex-1 bg-slate-700 rounded-l-full overflow-hidden">
                          <div
                            className="h-full bg-cyan-500 rounded-l-full transition-all duration-700 ml-auto"
                            style={{ width: `${aPct}%` }}
                          />
                        </div>
                        <div className="flex-1 bg-slate-700 rounded-r-full overflow-hidden">
                          <div
                            className="h-full bg-purple-500 rounded-r-full transition-all duration-700"
                            style={{ width: `${bPct}%` }}
                          />
                        </div>
                      </div>
                      <span className="text-xs text-slate-500 w-8">{bPct}%</span>
                    </div>
                    <div className="flex justify-between mt-0.5">
                      <span className="text-xs text-cyan-400/60">{comparison.vendor_a.vendor_name}</span>
                      <span className="text-xs text-purple-400/60">{comparison.vendor_b.vendor_name}</span>
                    </div>
                  </div>
                )
              })}
            </div>
          </div>

          {/* Strategy cards side by side (if either has one) */}
          {(comparison.vendor_a.recommended_approach || comparison.vendor_b.recommended_approach) && (
            <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
              {[comparison.vendor_a, comparison.vendor_b].map(pred => (
                <div key={pred.vendor_name} className="bg-slate-800/60 rounded-xl border border-slate-700/50 p-6">
                  <h3 className="text-sm font-medium text-slate-400 mb-3 flex items-center gap-2">
                    <Lightbulb className="w-4 h-4 text-cyan-400" />
                    Strategy vs. {pred.vendor_name}
                  </h3>
                  {pred.recommended_approach ? (
                    <div>
                      <p className="text-sm text-slate-300 leading-relaxed">{pred.recommended_approach}</p>
                      {(pred.lead_with ?? []).length > 0 && (
                        <div className="flex flex-wrap gap-1.5 mt-3">
                          {pred.lead_with.map((item, i) => (
                            <span key={i} className="text-xs bg-cyan-900/40 text-cyan-300 px-2 py-1 rounded">{item}</span>
                          ))}
                        </div>
                      )}
                      {(pred.talking_points ?? []).length > 0 && (
                        <ol className="mt-3 space-y-1">
                          {pred.talking_points.map((tp, i) => (
                            <li key={i} className="text-xs text-slate-300 flex gap-2">
                              <span className="text-slate-500 shrink-0">{i + 1}.</span>
                              {tp}
                            </li>
                          ))}
                        </ol>
                      )}
                    </div>
                  ) : (
                    <p className="text-sm text-slate-500 italic">Insufficient data for strategy</p>
                  )}
                </div>
              ))}
            </div>
          )}
        </div>
      )}

      {/* Gated state (insufficient data) */}
      {prediction && prediction.is_gated && (
        <InsufficientDataPanel
          vendor={prediction.vendor_name}
          gates={prediction.data_gates}
          verdict={prediction.verdict}
        />
      )}

      {/* Full results */}
      {prediction && !prediction.is_gated && (
        <div className="space-y-6">
          {/* Export toolbar */}
          <div className="flex flex-wrap justify-between gap-3">
            <div className="flex flex-wrap gap-2 text-sm">
              <Link
                to={buildVendorWorkspacePath(prediction.vendor_name, predictorBackTo)}
                className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-slate-200 transition-colors hover:bg-slate-800"
              >
                Vendor workspace
              </Link>
              <Link
                to={buildVendorScopedPath('/evidence', prediction.vendor_name, predictorBackTo)}
                className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-slate-200 transition-colors hover:bg-slate-800"
              >
                Evidence
              </Link>
              <Link
                to={buildVendorScopedPath('/reports', prediction.vendor_name, predictorBackTo)}
                className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-slate-200 transition-colors hover:bg-slate-800"
              >
                Reports
              </Link>
              <Link
                to={buildVendorScopedPath('/opportunities', prediction.vendor_name, predictorBackTo)}
                className="rounded-lg border border-slate-700 bg-slate-900 px-3 py-1.5 text-slate-200 transition-colors hover:bg-slate-800"
              >
                Opportunities
              </Link>
            </div>
            <div className="flex gap-2">
              <button
                onClick={() => copyPredictionSummary(prediction)}
                className="flex items-center gap-2 px-3 py-1.5 text-sm bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg transition-colors"
              >
                {copied ? <Check className="w-4 h-4 text-green-400" /> : <Copy className="w-4 h-4" />}
                {copied ? 'Copied' : 'Copy Summary'}
              </button>
              {prediction.prediction_id != null && (
                <button
                  onClick={() => { if (prediction.prediction_id) downloadPredictionCsv(prediction.prediction_id) }}
                  className="flex items-center gap-2 px-3 py-1.5 text-sm bg-slate-700 hover:bg-slate-600 text-slate-300 rounded-lg transition-colors"
                >
                  <Download className="w-4 h-4" />
                  Export CSV
                </button>
              )}
            </div>
          </div>

          {/* Top: Gauge + Factors */}
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            <div className="bg-slate-800/60 rounded-xl border border-slate-700/50 p-6 flex flex-col items-center justify-center relative">
              <ProbabilityGauge value={prediction.win_probability} confidence={prediction.confidence} />
              <p className="text-center text-slate-300 mt-4 text-sm max-w-xs">{prediction.verdict}</p>
              <div className="flex flex-wrap justify-center gap-3 mt-3 text-xs text-slate-500">
                {Object.entries(prediction.data_coverage).map(([k, v]) => (
                  <span key={k}>{k}: {v.toLocaleString()}</span>
                ))}
              </div>
              {prediction.weights_source && (
                <span className="text-xs text-slate-600 mt-2">
                  {prediction.weights_source === 'calibrated'
                    ? `Calibrated (v${prediction.calibration_version ?? '?'})`
                    : 'Default weights'}
                </span>
              )}
            </div>

            <div className="lg:col-span-2 space-y-3">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2">
                <TrendingUp className="w-5 h-5 text-cyan-400" />
                Scoring Factors
              </h2>
              <div className="grid grid-cols-1 md:grid-cols-2 gap-3">
                {prediction.factors.map(f => <FactorBar key={f.name} factor={f} />)}
              </div>
            </div>
          </div>

          {/* Strategy (LLM-synthesized) */}
          {prediction.recommended_approach && (
            <div className="bg-slate-800/60 rounded-xl border border-cyan-800/30 p-6">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2 mb-3">
                <Lightbulb className="w-5 h-5 text-cyan-400" />
                Recommended Approach
              </h2>
              <p className="text-sm text-slate-300 leading-relaxed">{prediction.recommended_approach}</p>

              <div className="grid grid-cols-1 md:grid-cols-3 gap-4 mt-4">
                {/* Lead With */}
                {prediction.lead_with.length > 0 && (
                  <div>
                    <h3 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-1">
                      <Zap className="w-3 h-3" /> Lead With
                    </h3>
                    <div className="flex flex-wrap gap-1.5">
                      {prediction.lead_with.map((item, i) => (
                        <span key={i} className="text-xs bg-cyan-900/40 text-cyan-300 px-2 py-1 rounded">{item}</span>
                      ))}
                    </div>
                  </div>
                )}

                {/* Talking Points */}
                {prediction.talking_points.length > 0 && (
                  <div>
                    <h3 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2">Talking Points</h3>
                    <ol className="space-y-1">
                      {prediction.talking_points.map((tp, i) => (
                        <li key={i} className="text-xs text-slate-300 flex gap-2">
                          <span className="text-slate-500 shrink-0">{i + 1}.</span>
                          {tp}
                        </li>
                      ))}
                    </ol>
                  </div>
                )}

                {/* Timing + Risks */}
                <div className="space-y-3">
                  {prediction.timing_advice && (
                    <div>
                      <h3 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-1 flex items-center gap-1">
                        <Clock className="w-3 h-3" /> Timing
                      </h3>
                      <p className="text-xs text-slate-300">{prediction.timing_advice}</p>
                    </div>
                  )}
                  {prediction.risk_factors.length > 0 && (
                    <div>
                      <h3 className="text-xs font-medium text-amber-400 uppercase tracking-wider mb-1">Risks</h3>
                      {prediction.risk_factors.map((rf, i) => (
                        <p key={i} className="text-xs text-amber-300/80 flex items-start gap-1">
                          <AlertTriangle className="w-3 h-3 shrink-0 mt-0.5" />
                          {rf}
                        </p>
                      ))}
                    </div>
                  )}
                </div>
              </div>
            </div>
          )}

          {/* Middle: Triggers + Displacement */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-slate-800/60 rounded-xl border border-slate-700/50 p-6">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2 mb-4">
                <TrendingDown className="w-5 h-5 text-red-400" />
                Switching Triggers
              </h2>
              {prediction.switching_triggers.length > 0 ? (
                <div className="space-y-3">
                  {prediction.switching_triggers.map((t, i) => (
                    <div key={i} className="flex items-center justify-between bg-slate-900/50 rounded-lg p-3">
                      <div className="flex items-center gap-3">
                        <span className="text-xs font-mono text-slate-500 w-5">{i + 1}</span>
                        <span className="text-sm text-slate-200">{t.trigger}</span>
                      </div>
                      <div className="flex items-center gap-3 text-xs">
                        {t.frequency > 0 && <span className="text-slate-400">{t.frequency} mentions</span>}
                        {t.urgency > 0 && (
                          <span className={`px-2 py-0.5 rounded-full ${t.urgency >= 7 ? 'bg-red-900/50 text-red-300' : t.urgency >= 4 ? 'bg-amber-900/50 text-amber-300' : 'bg-slate-700 text-slate-400'}`}>
                            {t.urgency.toFixed(1)}
                          </span>
                        )}
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-slate-500">No switching trigger data available.</p>
              )}
            </div>

            <div className="bg-slate-800/60 rounded-xl border border-slate-700/50 p-6">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2 mb-4">
                <ArrowRight className="w-5 h-5 text-cyan-400" />
                Where Defectors Go
              </h2>
              {prediction.displacement_targets.length > 0 ? (
                <div className="space-y-3">
                  {prediction.displacement_targets.map((d, i) => (
                    <div key={i} className="flex items-center justify-between bg-slate-900/50 rounded-lg p-3">
                      <div>
                        <span className="text-sm font-medium text-slate-200">{d.vendor}</span>
                        <span className="text-xs text-slate-500 ml-2">{d.driver}</span>
                      </div>
                      <div className="flex items-center gap-2">
                        <span className="text-xs text-slate-400">{d.mentions} mentions</span>
                        <span className={`text-xs px-2 py-0.5 rounded-full ${
                          d.strength === 'strong' ? 'bg-green-900/50 text-green-300' :
                          d.strength === 'moderate' ? 'bg-amber-900/50 text-amber-300' :
                          'bg-slate-700 text-slate-400'
                        }`}>
                          {d.strength}
                        </span>
                      </div>
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-slate-500">No displacement data available.</p>
              )}
            </div>
          </div>

          {/* Bottom: Quotes + Objections */}
          <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div className="bg-slate-800/60 rounded-xl border border-slate-700/50 p-6">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2 mb-4">
                <Quote className="w-5 h-5 text-emerald-400" />
                Proof Quotes
              </h2>
              <p className="text-xs text-slate-500 mb-3">Real user quotes to use in your pitch</p>
              {prediction.proof_quotes.length > 0 ? (
                <div className="space-y-3">
                  {prediction.proof_quotes.map((q, i) => (
                    <blockquote key={i} className="border-l-2 border-cyan-600 pl-3 py-1">
                      <p className="text-sm text-slate-300 italic">&ldquo;{q.quote}&rdquo;</p>
                      <div className="flex gap-2 mt-1 text-xs text-slate-500">
                        {q.source && <span>{q.source}</span>}
                        {q.role_type && <span>- {q.role_type}</span>}
                      </div>
                    </blockquote>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-slate-500">No quotable evidence available.</p>
              )}
            </div>

            <div className="bg-slate-800/60 rounded-xl border border-slate-700/50 p-6">
              <h2 className="text-lg font-semibold text-white flex items-center gap-2 mb-4">
                <Shield className="w-5 h-5 text-amber-400" />
                Objections to Expect
              </h2>
              <p className="text-xs text-slate-500 mb-3">Why some customers stay - prepare for these</p>
              {prediction.objections.length > 0 ? (
                <div className="space-y-3">
                  {prediction.objections.map((o, i) => (
                    <div key={i} className="bg-slate-900/50 rounded-lg p-3">
                      <div className="flex items-center justify-between">
                        <div className="flex items-center gap-2">
                          <AlertTriangle className="w-3.5 h-3.5 text-amber-500 shrink-0" />
                          <span className="text-sm text-slate-200">{o.objection}</span>
                        </div>
                        {o.frequency > 0 && (
                          <span className="text-xs text-slate-500">{o.frequency} mentions</span>
                        )}
                      </div>
                      {o.counter && (
                        <p className="text-xs text-slate-400 italic mt-2 ml-6 border-l border-slate-700 pl-2">
                          &ldquo;{o.counter}&rdquo;
                        </p>
                      )}
                    </div>
                  ))}
                </div>
              ) : (
                <p className="text-sm text-slate-500">No objection data available.</p>
              )}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}
