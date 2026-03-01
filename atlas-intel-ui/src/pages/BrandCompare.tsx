import { useState, useEffect, useRef, useMemo } from 'react'
import { useSearchParams } from 'react-router-dom'
import { RefreshCw, X, Search, ArrowRight } from 'lucide-react'
import { clsx } from 'clsx'
import { PageError } from '../components/ErrorBoundary'
import { CHURN_COLORS, DistBar, Card } from '../components/BrandComponents'
import useApiData from '../hooks/useApiData'
import {
  fetchBrandComparison,
  fetchBrands,
  type BrandComparison,
  type BrandSummary,
  type LabelCount,
} from '../api/client'

const BRAND_COLORS = ['#22d3ee', '#a78bfa', '#f472b6', '#34d399']

function ComparisonSkeleton() {
  return (
    <div className="space-y-6 animate-pulse">
      <div className="h-7 w-56 bg-slate-700/50 rounded" />
      <div className="h-64 bg-slate-900/50 border border-slate-700/50 rounded-xl" />
      <div className="grid grid-cols-2 gap-6">
        <div className="h-48 bg-slate-900/50 border border-slate-700/50 rounded-xl" />
        <div className="h-48 bg-slate-900/50 border border-slate-700/50 rounded-xl" />
      </div>
    </div>
  )
}

function MetricCell({ value, format, colorFn }: {
  value: number | null | undefined
  format?: 'number' | 'decimal' | 'pct' | 'dollar'
  colorFn?: (v: number) => string
}) {
  if (value == null) return <span className="text-slate-500">--</span>
  let display: string
  switch (format) {
    case 'pct': display = `${value}%`; break
    case 'decimal': display = value.toFixed(1); break
    case 'dollar': display = `$${value.toLocaleString()}`; break
    default: display = value.toLocaleString()
  }
  const color = colorFn ? colorFn(value) : 'text-slate-300'
  return <span className={clsx('font-mono', color)}>{display}</span>
}

export default function BrandCompare() {
  const [searchParams, setSearchParams] = useSearchParams()
  const initialBrands = useMemo(
    () => (searchParams.get('brands') || '').split(',').filter(Boolean).map(decodeURIComponent),
    // eslint-disable-next-line react-hooks/exhaustive-deps
    [],
  )
  const [brands, setBrands] = useState<string[]>(initialBrands)
  const [brandInput, setBrandInput] = useState('')
  const [suggestions, setSuggestions] = useState<BrandSummary[]>([])
  const [showSuggestions, setShowSuggestions] = useState(false)
  const debounceRef = useRef<ReturnType<typeof setTimeout>>(undefined)
  const inputRef = useRef<HTMLInputElement>(null)

  // Sync brands -> URL
  useEffect(() => {
    if (brands.length > 0) {
      setSearchParams({ brands: brands.join(',') }, { replace: true })
    } else {
      setSearchParams({}, { replace: true })
    }
  }, [brands, setSearchParams])

  // Debounced brand search
  useEffect(() => {
    clearTimeout(debounceRef.current)
    if (!brandInput.trim()) {
      setSuggestions([])
      return
    }
    debounceRef.current = setTimeout(async () => {
      try {
        const res = await fetchBrands({ search: brandInput, limit: 8 })
        setSuggestions(res.brands.filter((b) => !brands.includes(b.brand)))
      } catch { /* ignore */ }
    }, 250)
    return () => clearTimeout(debounceRef.current)
  }, [brandInput, brands])

  const addBrand = (name: string) => {
    if (brands.length >= 4 || brands.includes(name)) return
    setBrands((prev) => [...prev, name])
    setBrandInput('')
    setSuggestions([])
    setShowSuggestions(false)
  }

  const removeBrand = (name: string) => {
    setBrands((prev) => prev.filter((b) => b !== name))
  }

  const canFetch = brands.length >= 2

  const { data, loading, error, refresh, refreshing } = useApiData<BrandComparison>(
    () => {
      if (!canFetch) return Promise.reject(new Error('Select at least 2 brands'))
      return fetchBrandComparison(brands)
    },
    [brands.join(',')],
  )

  if (error && canFetch) return <PageError error={error} onRetry={refresh} />

  const brandKeys = data?.brands ?? []

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <h1 className="text-2xl font-bold text-white">Brand Comparison</h1>
        {canFetch && (
          <button
            onClick={refresh}
            disabled={refreshing}
            className="p-2 rounded-lg text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-50"
          >
            <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
          </button>
        )}
      </div>

      {/* Brand Search + Pills */}
      <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4">
        <div className="flex flex-wrap items-center gap-2 mb-3">
          {brands.map((b, i) => (
            <span
              key={b}
              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-full text-sm font-medium text-white"
              style={{ backgroundColor: `${BRAND_COLORS[i % BRAND_COLORS.length]}20`, borderColor: BRAND_COLORS[i % BRAND_COLORS.length], borderWidth: 1 }}
            >
              <span className="w-2 h-2 rounded-full" style={{ backgroundColor: BRAND_COLORS[i % BRAND_COLORS.length] }} />
              {b}
              <button onClick={() => removeBrand(b)} className="ml-0.5 text-slate-400 hover:text-white">
                <X className="h-3.5 w-3.5" />
              </button>
            </span>
          ))}
        </div>
        {brands.length < 4 && (
          <div className="relative">
            <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
            <input
              ref={inputRef}
              type="text"
              value={brandInput}
              onChange={(e) => setBrandInput(e.target.value)}
              onFocus={() => setShowSuggestions(true)}
              onBlur={() => setTimeout(() => setShowSuggestions(false), 200)}
              placeholder={brands.length === 0 ? 'Search brands to compare...' : 'Add another brand...'}
              className="w-full pl-9 pr-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
            />
            {showSuggestions && suggestions.length > 0 && (
              <div className="absolute z-20 top-full mt-1 w-full bg-slate-800 border border-slate-700 rounded-lg shadow-xl max-h-48 overflow-y-auto">
                {suggestions.map((s) => (
                  <button
                    key={s.brand}
                    onMouseDown={() => addBrand(s.brand)}
                    className="w-full text-left px-3 py-2 text-sm text-slate-300 hover:bg-slate-700/50 hover:text-white flex items-center justify-between"
                  >
                    <span>{s.brand}</span>
                    <span className="text-xs text-slate-500">{s.review_count.toLocaleString()} reviews</span>
                  </button>
                ))}
              </div>
            )}
          </div>
        )}
        {!canFetch && (
          <p className="text-xs text-slate-500 mt-2">Select at least 2 brands to compare (max 4)</p>
        )}
      </div>

      {!canFetch && !loading && null}
      {canFetch && loading && <ComparisonSkeleton />}

      {data && brandKeys.length > 0 && (
        <>
          {/* ── Scorecard Table ── */}
          <Card title="Scorecard">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700/50">
                    <th className="text-left text-xs text-slate-500 py-2 pr-4 font-normal">Metric</th>
                    {brandKeys.map((b, i) => (
                      <th key={b} className="text-right py-2 px-3 font-medium" style={{ color: BRAND_COLORS[i] }}>
                        {b}
                      </th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/50">
                  <tr>
                    <td className="py-2 pr-4 text-slate-400">Products</td>
                    {brandKeys.map((b) => (
                      <td key={b} className="text-right px-3 py-2">
                        <MetricCell value={data.per_brand[b]?.product_count} />
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="py-2 pr-4 text-slate-400">Reviews</td>
                    {brandKeys.map((b) => (
                      <td key={b} className="text-right px-3 py-2">
                        <MetricCell value={data.per_brand[b]?.total_reviews} />
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="py-2 pr-4 text-slate-400">Avg Rating</td>
                    {brandKeys.map((b) => (
                      <td key={b} className="text-right px-3 py-2">
                        <MetricCell value={data.per_brand[b]?.avg_rating} format="decimal"
                          colorFn={(v) => v >= 4 ? 'text-emerald-400' : v >= 3 ? 'text-amber-400' : 'text-red-400'} />
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="py-2 pr-4 text-slate-400">Brand Health</td>
                    {brandKeys.map((b) => (
                      <td key={b} className="text-right px-3 py-2">
                        <MetricCell value={data.per_brand[b]?.brand_health}
                          colorFn={(v) => v >= 60 ? 'text-emerald-400' : v >= 40 ? 'text-amber-400' : 'text-red-400'} />
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="py-2 pr-4 text-slate-400">Repurchase %</td>
                    {brandKeys.map((b) => (
                      <td key={b} className="text-right px-3 py-2">
                        <MetricCell value={data.per_brand[b]?.repurchase_pct} format="pct"
                          colorFn={(v) => v >= 60 ? 'text-emerald-400' : v >= 40 ? 'text-amber-400' : 'text-red-400'} />
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="py-2 pr-4 text-slate-400">Safety Flags</td>
                    {brandKeys.map((b) => (
                      <td key={b} className="text-right px-3 py-2">
                        <MetricCell value={data.per_brand[b]?.safety_flagged_count}
                          colorFn={(v) => v === 0 ? 'text-emerald-400' : v <= 3 ? 'text-amber-400' : 'text-red-400'} />
                      </td>
                    ))}
                  </tr>
                </tbody>
              </table>
            </div>
          </Card>

          {/* ── Distributions ── */}
          <Card title="Signal Distributions">
            <div className="space-y-6">
              {(['replacement_breakdown', 'trajectory_breakdown', 'switching_barrier'] as const).map((field) => {
                const label = field === 'replacement_breakdown' ? 'Replacement Behavior'
                  : field === 'trajectory_breakdown' ? 'Sentiment Trajectory'
                  : 'Switching Barrier'
                return (
                  <div key={field}>
                    <h4 className="text-xs text-slate-400 mb-2">{label}</h4>
                    <div className="space-y-2">
                      {brandKeys.map((b, i) => {
                        const items: LabelCount[] = data.per_brand[b]?.[field] ?? []
                        if (!items.length) return null
                        return (
                          <div key={b} className="flex items-center gap-3">
                            <span className="w-24 text-xs font-medium truncate shrink-0" style={{ color: BRAND_COLORS[i] }}>{b}</span>
                            <div className="flex-1">
                              <DistBar items={items} label="" />
                            </div>
                          </div>
                        )
                      })}
                    </div>
                  </div>
                )
              })}
            </div>
          </Card>

          {/* ── Failure Snapshot ── */}
          <Card title="Failure Snapshot">
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700/50">
                    <th className="text-left text-xs text-slate-500 py-2 pr-4 font-normal">Metric</th>
                    {brandKeys.map((b, i) => (
                      <th key={b} className="text-right py-2 px-3 font-medium" style={{ color: BRAND_COLORS[i] }}>{b}</th>
                    ))}
                  </tr>
                </thead>
                <tbody className="divide-y divide-slate-800/50">
                  <tr>
                    <td className="py-2 pr-4 text-slate-400">Failure Count</td>
                    {brandKeys.map((b) => (
                      <td key={b} className="text-right px-3 py-2">
                        <MetricCell value={data.per_brand[b]?.failure_count}
                          colorFn={(v) => v === 0 ? 'text-emerald-400' : v <= 5 ? 'text-amber-400' : 'text-red-400'} />
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="py-2 pr-4 text-slate-400">Avg $ Lost</td>
                    {brandKeys.map((b) => (
                      <td key={b} className="text-right px-3 py-2">
                        <MetricCell value={data.per_brand[b]?.avg_dollar_lost} format="dollar" />
                      </td>
                    ))}
                  </tr>
                  <tr>
                    <td className="py-2 pr-4 text-slate-400">Workaround Rate</td>
                    {brandKeys.map((b) => (
                      <td key={b} className="text-right px-3 py-2">
                        <MetricCell value={data.per_brand[b]?.workaround_rate} format="pct" />
                      </td>
                    ))}
                  </tr>
                </tbody>
              </table>
            </div>
          </Card>

          {/* ── Consequence Breakdown ── */}
          <div className={`grid gap-6 ${brandKeys.length <= 2 ? 'grid-cols-1 lg:grid-cols-2' : brandKeys.length === 3 ? 'grid-cols-1 lg:grid-cols-3' : 'grid-cols-1 sm:grid-cols-2 lg:grid-cols-4'}`}>
            {brandKeys.map((b, i) => {
              const items = data.per_brand[b]?.consequence_breakdown ?? []
              if (!items.length) return null
              const total = items.reduce((s, it) => s + it.count, 0)
              return (
                <Card key={b} title={b}>
                  <div className="space-y-1.5">
                    {items.map((c) => {
                      const pct = total > 0 ? Math.round((c.count / total) * 100) : 0
                      const color = CHURN_COLORS[c.label] ?? 'bg-slate-600'
                      return (
                        <div key={c.label} className="flex items-center gap-2 text-xs">
                          <span className={clsx('w-2 h-2 rounded-full shrink-0', color)} />
                          <span className="text-slate-400 flex-1">{c.label.replaceAll('_', ' ')}</span>
                          <span className="font-mono" style={{ color: BRAND_COLORS[i] }}>{pct}%</span>
                        </div>
                      )
                    })}
                  </div>
                </Card>
              )
            })}
          </div>

          {/* ── Cross-Brand Flows ── */}
          {data.cross_brand.competitive_flows.length > 0 && (
            <Card title="Cross-Brand Competitive Flows">
              <ul className="space-y-2 max-h-64 overflow-y-auto">
                {data.cross_brand.competitive_flows.map((f, i) => (
                  <li key={i} className="flex items-center justify-between text-sm">
                    <span className="text-slate-300 flex items-center gap-2">
                      <span className="text-white font-medium">{f.from_brand}</span>
                      <ArrowRight className="h-3.5 w-3.5 text-slate-500" />
                      <span className="text-white font-medium">{f.to_brand}</span>
                      <span className={clsx(
                        'text-xs px-1.5 py-0.5 rounded',
                        f.direction === 'switched_to' ? 'bg-red-500/10 text-red-400' :
                        f.direction === 'switched_from' ? 'bg-emerald-500/10 text-emerald-400' :
                        'bg-slate-700/50 text-slate-400',
                      )}>
                        {f.direction.replaceAll('_', ' ')}
                      </span>
                    </span>
                    <span className="text-cyan-400 font-mono text-xs shrink-0">{f.count}</span>
                  </li>
                ))}
              </ul>
            </Card>
          )}

          {/* ── Shared Feature Requests ── */}
          {data.cross_brand.shared_feature_requests.length > 0 && (
            <Card title="Shared Feature Requests">
              <ul className="space-y-2 max-h-64 overflow-y-auto">
                {data.cross_brand.shared_feature_requests.map((f, i) => (
                  <li key={i} className="flex items-center justify-between text-sm">
                    <div className="flex-1 min-w-0">
                      <span className="text-slate-300 truncate block">{f.request}</span>
                      <div className="flex gap-1 mt-0.5">
                        {f.brands.map((b) => {
                          const idx = brandKeys.indexOf(b)
                          return (
                            <span key={b} className="text-[10px] px-1.5 py-0.5 rounded"
                              style={{ backgroundColor: `${BRAND_COLORS[idx >= 0 ? idx : 0]}20`, color: BRAND_COLORS[idx >= 0 ? idx : 0] }}>
                              {b}
                            </span>
                          )
                        })}
                      </div>
                    </div>
                    <span className="text-cyan-400 font-mono text-xs shrink-0 ml-2">{f.total_count}</span>
                  </li>
                ))}
              </ul>
            </Card>
          )}

          {/* ── Consideration Overlap ── */}
          {data.cross_brand.consideration_overlap.length > 0 && (
            <Card title="Consideration Overlap">
              <ul className="space-y-2 max-h-64 overflow-y-auto">
                {data.cross_brand.consideration_overlap.map((c, i) => (
                  <li key={i} className="flex items-center justify-between text-sm">
                    <div className="flex-1 min-w-0">
                      <span className="text-white font-medium block truncate">{c.product}</span>
                      <div className="flex gap-1 mt-0.5">
                        {c.mentioned_by_brands.map((b) => {
                          const idx = brandKeys.indexOf(b)
                          return (
                            <span key={b} className="text-[10px] px-1.5 py-0.5 rounded"
                              style={{ backgroundColor: `${BRAND_COLORS[idx >= 0 ? idx : 0]}20`, color: BRAND_COLORS[idx >= 0 ? idx : 0] }}>
                              {b}
                            </span>
                          )
                        })}
                      </div>
                    </div>
                    <span className="text-cyan-400 font-mono text-xs shrink-0 ml-2">{c.total_count}</span>
                  </li>
                ))}
              </ul>
            </Card>
          )}
        </>
      )}
    </div>
  )
}
