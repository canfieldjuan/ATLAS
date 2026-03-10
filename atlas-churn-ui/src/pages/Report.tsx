import { useState, useEffect, type FormEvent } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import {
  FileText, Mail, AlertCircle, Loader2, ShieldCheck,
  TrendingUp, TrendingDown, Minus, Users, BarChart3, ArrowRight,
  X, Check,
} from 'lucide-react'
import PublicLayout from '../components/PublicLayout'
import SeoHead from '../components/SeoHead'
import { useAuth } from '../auth/AuthContext'

const API_BASE = import.meta.env.VITE_API_BASE || ''
const GATE_URL = `${API_BASE}/api/v1/b2b/briefings/gate`
const REPORT_DATA_URL = `${API_BASE}/api/v1/b2b/briefings/report-data`
const CHECKOUT_URL = `${API_BASE}/api/v1/b2b/briefings/checkout`

async function startCheckout(vendorName: string, tier: 'standard' | 'pro') {
  try {
    const res = await fetch(CHECKOUT_URL, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vendor_name: vendorName, tier }),
    })
    if (res.ok) {
      const { url } = await res.json()
      window.location.href = url
    } else {
      const body = await res.json().catch(() => ({ detail: 'Checkout failed' }))
      alert(body.detail || 'Failed to start checkout')
    }
  } catch {
    alert('Network error -- please try again')
  }
}

type Status = 'idle' | 'submitting' | 'loading_report' | 'report' | 'error'

/* eslint-disable @typescript-eslint/no-explicit-any */
type ReportData = {
  vendor_name: string
  briefing: Record<string, any>
  intelligence_reports: Array<{
    report_type: string
    executive_summary: string | null
    data: Record<string, any>
    report_date: string | null
  }>
  product_profile: Record<string, any> | null
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function pressureColor(score: number) {
  if (score >= 70) return 'text-red-400'
  if (score >= 40) return 'text-amber-400'
  return 'text-emerald-400'
}

function pressureBg(score: number) {
  if (score >= 70) return 'bg-red-500/20 border-red-500/30'
  if (score >= 40) return 'bg-amber-500/20 border-amber-500/30'
  return 'bg-emerald-500/20 border-emerald-500/30'
}

function TrendIcon({ trend }: { trend: string | null }) {
  const t = (trend || '').toLowerCase()
  if (['up', 'rising', 'increasing', 'worsening'].includes(t))
    return <TrendingUp className="h-4 w-4 text-red-400" />
  if (['down', 'falling', 'decreasing', 'improving'].includes(t))
    return <TrendingDown className="h-4 w-4 text-emerald-400" />
  return <Minus className="h-4 w-4 text-slate-400" />
}

function fmtPct(v: any) {
  const n = Number(v)
  return isNaN(n) ? 'N/A' : `${n.toFixed(1)}%`
}

function fmtScore(v: any) {
  const n = Number(v)
  return isNaN(n) ? 'N/A' : n.toFixed(1)
}

// ---------------------------------------------------------------------------
// Report View (rich intelligence)
// ---------------------------------------------------------------------------

function RankedBar({ label, count, max }: { label: string; count: number; max: number }) {
  const pct = max > 0 ? Math.round((count / max) * 100) : 10
  return (
    <div>
      <div className="flex items-center justify-between text-xs mb-0.5">
        <span className="text-slate-300">{label}</span>
        <span className="text-slate-400">{count}</span>
      </div>
      <div className="h-1.5 bg-slate-800 rounded-full overflow-hidden">
        <div className="h-full bg-cyan-500/60 rounded-full" style={{ width: `${Math.max(pct, 5)}%` }} />
      </div>
    </div>
  )
}

function PricingModal({ vendorName, onClose }: { vendorName: string; onClose: () => void }) {
  const [loading, setLoading] = useState<'standard' | 'pro' | null>(null)

  const handleSelect = async (tier: 'standard' | 'pro') => {
    setLoading(tier)
    await startCheckout(vendorName, tier)
    setLoading(null)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4" onClick={onClose}>
      <div className="absolute inset-0 bg-black/70 backdrop-blur-sm" />
      <div
        className="relative bg-slate-900 border border-slate-700 rounded-2xl max-w-3xl w-full p-8"
        onClick={e => e.stopPropagation()}
      >
        <button onClick={onClose} className="absolute top-4 right-4 text-slate-400 hover:text-white cursor-pointer">
          <X className="h-5 w-5" />
        </button>

        <div className="text-center mb-8">
          <h2 className="text-2xl font-bold mb-2">Choose Your Plan</h2>
          <p className="text-sm text-slate-400">Weekly churn intelligence for {vendorName}</p>
        </div>

        <div className="grid md:grid-cols-2 gap-6">
          {/* Standard */}
          <div className="border border-slate-700 rounded-xl p-6 flex flex-col">
            <h3 className="text-lg font-semibold mb-1">Standard</h3>
            <div className="mb-4">
              <span className="text-3xl font-bold">$499</span>
              <span className="text-slate-400 text-sm">/mo</span>
            </div>
            <ul className="space-y-2 mb-6 flex-1">
              {[
                'Weekly churn intelligence reports',
                'Pain driver analysis',
                'Competitive displacement tracking',
                'Feature gap identification',
                'Anonymized customer signals',
              ].map(f => (
                <li key={f} className="flex items-start gap-2 text-sm text-slate-300">
                  <Check className="h-4 w-4 text-cyan-400 mt-0.5 shrink-0" />
                  {f}
                </li>
              ))}
            </ul>
            <button
              onClick={() => handleSelect('standard')}
              disabled={loading !== null}
              className="w-full py-3 bg-slate-700 hover:bg-slate-600 rounded-lg font-medium transition-colors cursor-pointer disabled:opacity-50"
            >
              {loading === 'standard' ? 'Redirecting...' : 'Get Started'}
            </button>
          </div>

          {/* Pro */}
          <div className="border-2 border-cyan-500/50 rounded-xl p-6 flex flex-col relative">
            <div className="absolute -top-3 left-1/2 -translate-x-1/2 px-3 py-0.5 bg-cyan-600 rounded-full text-xs font-medium">
              Recommended
            </div>
            <h3 className="text-lg font-semibold mb-1">Pro</h3>
            <div className="mb-4">
              <span className="text-3xl font-bold">$1,499</span>
              <span className="text-slate-400 text-sm">/mo</span>
            </div>
            <ul className="space-y-2 mb-6 flex-1">
              {[
                'Everything in Standard',
                'Named account-level signals',
                'Urgency scoring per account',
                'Real-time churn alerts',
                'Dedicated support',
              ].map(f => (
                <li key={f} className="flex items-start gap-2 text-sm text-slate-300">
                  <Check className="h-4 w-4 text-cyan-400 mt-0.5 shrink-0" />
                  {f}
                </li>
              ))}
            </ul>
            <button
              onClick={() => handleSelect('pro')}
              disabled={loading !== null}
              className="w-full py-3 bg-cyan-600 hover:bg-cyan-500 rounded-lg font-medium transition-colors cursor-pointer disabled:opacity-50"
            >
              {loading === 'pro' ? 'Redirecting...' : 'Get Started'}
            </button>
          </div>
        </div>

        <p className="text-center text-xs text-slate-500 mt-6">Cancel anytime. No long-term contracts.</p>
      </div>
    </div>
  )
}

function ReportView({ data }: { data: ReportData }) {
  const [showPricing, setShowPricing] = useState(false)
  const b = data.briefing
  const score = Number(b.churn_pressure_score) || 0
  const pains: any[] = b.pain_breakdown || []
  const maxPain = Math.max(...pains.map((p: any) => Number(p.count) || 0), 1)
  const displacements: any[] = b.top_displacement_targets || []
  const evidence: any[] = b.evidence || []
  const namedAccountCount: number = b.named_account_count || (b.named_accounts || []).length
  const featureGaps: any[] = b.top_feature_gaps || []
  const painLabels: Record<string, string> = b.pain_labels || {}

  return (
    <PublicLayout variant="report" onCtaClick={() => setShowPricing(true)}>
      <SeoHead
        title={`${data.vendor_name} Churn Intelligence Report | Churn Signals`}
        description={`Full churn intelligence report for ${data.vendor_name}: pain drivers, displacement targets, at-risk accounts, and competitive analysis.`}
        canonical={`https://churnsignals.co/report?vendor=${encodeURIComponent(data.vendor_name)}`}
      />
      <div className="max-w-5xl mx-auto px-6 py-12">
        {/* Context line */}
        <p className="text-sm text-slate-400 mb-6">
          We monitor public customer signals for {data.vendor_name} so your team doesn't have to. Here's what we found this week.
        </p>

        {/* Header */}
        <div className="flex flex-col sm:flex-row sm:items-start sm:justify-between gap-4 mb-8">
          <div>
            <div className="flex items-center gap-3 mb-2">
              <h1 className="text-3xl sm:text-4xl font-bold">{data.vendor_name}</h1>
              <span className="px-3 py-1 bg-slate-700/60 rounded-full text-xs text-slate-300 uppercase tracking-wider">
                {b.category || 'Software'}
              </span>
            </div>
            {b.headline && (
              <p className="text-lg text-slate-300 font-medium">{b.headline}</p>
            )}
          </div>
          <div className={`flex flex-col items-center px-6 py-4 rounded-xl border ${pressureBg(score)}`}>
            <div className={`text-4xl font-bold ${pressureColor(score)}`}>{score.toFixed(0)}</div>
            <div className="text-xs text-slate-400 uppercase tracking-wider">Churn Pressure</div>
            <div className="flex items-center gap-1 mt-1">
              <TrendIcon trend={b.trend} />
              <span className="text-xs text-slate-400 capitalize">{b.trend || 'Stable'}</span>
            </div>
          </div>
        </div>

        {/* Executive Summary */}
        {b.executive_summary && (
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-6 mb-8">
            <h3 className="text-sm font-medium text-cyan-400 uppercase tracking-wider mb-3">Executive Summary</h3>
            <p className="text-sm text-slate-300 leading-relaxed">{b.executive_summary}</p>
          </div>
        )}

        {/* Metrics Row */}
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-4 mb-8">
          {[
            { label: 'Churn Density', value: fmtPct(b.churn_signal_density) },
            { label: 'Avg Urgency', value: `${fmtScore(b.avg_urgency)} / 10` },
            { label: 'Reviews Analyzed', value: String(b.review_count || 0) },
            { label: 'DM Churn Rate', value: fmtPct(b.dm_churn_rate) },
          ].map(m => (
            <div key={m.label} className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-4 text-center">
              <div className="text-xl font-bold text-white">{m.value}</div>
              <div className="text-xs text-slate-400 mt-1">{m.label}</div>
            </div>
          ))}
        </div>

        {/* Two-column grid */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 mb-8">
          {/* Pain Breakdown */}
          {pains.length > 0 && (
            <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
              <h3 className="text-sm font-medium text-cyan-400 uppercase tracking-wider mb-4">
                <BarChart3 className="inline h-4 w-4 mr-1.5 -mt-0.5" />
                What's Driving Churn
              </h3>
              <div className="space-y-2">
                {pains.map((p: any, i: number) => (
                  <RankedBar
                    key={i}
                    label={painLabels[p.category] || p.category || 'Other'}
                    count={Number(p.count) || 0}
                    max={maxPain}
                  />
                ))}
              </div>
            </div>
          )}

          {/* Displacement Targets */}
          {displacements.length > 0 && (
            <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
              <h3 className="text-sm font-medium text-cyan-400 uppercase tracking-wider mb-4">
                <ArrowRight className="inline h-4 w-4 mr-1.5 -mt-0.5" />
                Where They're Going
              </h3>
              <div className="space-y-2">
                {displacements.map((d: any, i: number) => (
                  <div key={i} className="flex items-center justify-between py-2 border-b border-slate-800 last:border-0">
                    <span className="text-sm text-slate-300">{d.competitor || d.name}</span>
                    <span className="text-sm text-slate-400">{d.count || d.mentions} mentions</span>
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Accounts with Friction Signals (count-only teaser) */}
          {namedAccountCount > 0 && (
            <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
              <h3 className="text-sm font-medium text-cyan-400 uppercase tracking-wider mb-4">
                <Users className="inline h-4 w-4 mr-1.5 -mt-0.5" />
                Accounts Showing Friction
              </h3>
              <div className="text-center py-4">
                <div className="text-3xl font-bold text-white mb-1">{namedAccountCount}</div>
                <p className="text-sm text-slate-400 mb-4">
                  accounts with active churn signals detected in the last 30 days
                </p>
                <button
                  onClick={() => setShowPricing(true)}
                  className="inline-block px-4 py-2 bg-cyan-600/20 hover:bg-cyan-600/30 border border-cyan-500/30 rounded-lg text-sm text-cyan-400 font-medium transition-colors cursor-pointer"
                >
                  Unlock Account-Level Details
                </button>
              </div>
            </div>
          )}

          {/* Feature Gaps */}
          {featureGaps.length > 0 && (
            <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
              <h3 className="text-sm font-medium text-cyan-400 uppercase tracking-wider mb-4">Top Feature Gaps</h3>
              <ul className="space-y-2">
                {featureGaps.map((g: any, i: number) => (
                  <li key={i} className="text-sm text-slate-300 flex items-start gap-2">
                    <span className="text-cyan-500 mt-0.5">--</span>
                    {typeof g === 'string' ? g : g.feature || g.name || g.gap || g.area || Object.values(g).find((v: any) => typeof v === 'string') || `#${i + 1}`}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Customer Quotes */}
        {evidence.length > 0 && (
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 mb-8">
            <h3 className="text-sm font-medium text-cyan-400 uppercase tracking-wider mb-4">What Customers Are Saying</h3>
            <div className="space-y-3">
              {evidence.map((q: any, i: number) => {
                const text = typeof q === 'string' ? q : q.quote || q.text || ''
                if (!text) return null
                return (
                  <blockquote key={i} className="text-sm text-slate-300 italic border-l-2 border-red-500/50 pl-3">
                    "{text}"
                  </blockquote>
                )
              })}
            </div>
          </div>
        )}

        {/* Intelligence Reports (vendor comparisons etc.) */}
        {data.intelligence_reports.length > 0 && (
          <div className="mb-8">
            <h2 className="text-lg font-bold mb-4">Deep Analysis</h2>
            <div className="space-y-6">
              {data.intelligence_reports.map((report, idx) => (
                <div key={idx} className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5">
                  <div className="flex items-center gap-3 mb-3">
                    <span className="px-2 py-0.5 bg-cyan-500/20 text-cyan-400 rounded text-xs font-medium">
                      {report.report_type.replace(/_/g, ' ')}
                    </span>
                    {report.report_date && (
                      <span className="text-xs text-slate-500">{report.report_date}</span>
                    )}
                  </div>
                  {report.executive_summary && (
                    <p className="text-sm text-slate-300 mb-4">{report.executive_summary}</p>
                  )}
                  <IntelligenceData data={report.data} />
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Product Profile */}
        {data.product_profile && (
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 mb-8">
            <h3 className="text-sm font-medium text-cyan-400 uppercase tracking-wider mb-3">Product Profile</h3>
            {data.product_profile.profile_summary && (
              <p className="text-sm text-slate-300 mb-4">{data.product_profile.profile_summary}</p>
            )}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {data.product_profile.strengths && (
                <div>
                  <h4 className="text-xs font-medium text-emerald-400 uppercase mb-2">Strengths</h4>
                  <ul className="space-y-1">
                    {(Array.isArray(data.product_profile.strengths) ? data.product_profile.strengths : []).map((s: any, i: number) => (
                      <li key={i} className="text-xs text-slate-400">
                        {typeof s === 'string' ? s : s.area || s.name || s.strength || s.description || Object.values(s).find(v => typeof v === 'string') || `#${i + 1}`}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {data.product_profile.weaknesses && (
                <div>
                  <h4 className="text-xs font-medium text-red-400 uppercase mb-2">Weaknesses</h4>
                  <ul className="space-y-1">
                    {(Array.isArray(data.product_profile.weaknesses) ? data.product_profile.weaknesses : []).map((w: any, i: number) => (
                      <li key={i} className="text-xs text-slate-400">
                        {typeof w === 'string' ? w : w.area || w.name || w.weakness || w.description || Object.values(w).find(v => typeof v === 'string') || `#${i + 1}`}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Footer CTA */}
        <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-6 text-center">
          <h3 className="text-lg font-semibold mb-2">Want this intelligence every week?</h3>
          <p className="text-sm text-slate-400 mb-4">
            Get weekly churn signals, displacement tracking, and at-risk accounts for {data.vendor_name} delivered to your inbox.
          </p>
          <button
            onClick={() => setShowPricing(true)}
            className="inline-block px-6 py-3 bg-cyan-600 hover:bg-cyan-500 rounded-lg text-white font-semibold transition-colors cursor-pointer"
          >
            Get Weekly Reports
          </button>
          <p className="text-xs text-slate-500 mt-2">Starting at $499/mo</p>
        </div>
      </div>

      {showPricing && (
        <PricingModal vendorName={data.vendor_name} onClose={() => setShowPricing(false)} />
      )}
    </PublicLayout>
  )
}

/** Render intelligence data fields as key-value cards */
function IntelligenceData({ data }: { data: Record<string, any> }) {
  const SKIP_KEYS = new Set([
    'report_date', 'window_days', 'primary_vendor', 'comparison_vendor',
  ])

  const entries = Object.entries(data).filter(([k]) => !SKIP_KEYS.has(k))
  if (entries.length === 0) return null

  return (
    <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
      {entries.map(([key, value]) => (
        <div key={key} className="bg-slate-800/50 rounded-lg p-3">
          <h5 className="text-xs text-slate-400 mb-1.5">{key.replace(/_/g, ' ')}</h5>
          <IntelValue value={value} />
        </div>
      ))}
    </div>
  )
}

function IntelValue({ value }: { value: any }) {
  if (typeof value === 'string') return <p className="text-xs text-slate-300">{value}</p>
  if (typeof value === 'number') return <span className="text-sm font-bold text-white">{value}</span>
  if (typeof value === 'boolean') return <span className="text-xs text-slate-300">{value ? 'Yes' : 'No'}</span>

  // Ranked list [{category/name/competitor: str, count: n}]
  if (Array.isArray(value) && value.length > 0 && typeof value[0] === 'object' && 'count' in value[0]) {
    const max = Math.max(...value.map((v: any) => Number(v.count) || 0), 1)
    return (
      <div className="space-y-1">
        {value.slice(0, 8).map((item: any, i: number) => {
          const label = item.category || item.name || item.competitor || item.feature || `#${i + 1}`
          return <RankedBar key={i} label={label} count={Number(item.count) || 0} max={max} />
        })}
      </div>
    )
  }

  // String array
  if (Array.isArray(value) && value.length > 0 && typeof value[0] === 'string') {
    return (
      <div className="space-y-1">
        {value.slice(0, 5).map((s: string, i: number) => (
          <p key={i} className="text-xs text-slate-300">{s}</p>
        ))}
      </div>
    )
  }

  // Object array (e.g. displacement flows with companies)
  if (Array.isArray(value) && value.length > 0) {
    return (
      <div className="space-y-1">
        {value.slice(0, 5).map((item: any, i: number) => {
          const label = item.name || item.company || item.competitor || item.label || item.category || Object.values(item).find((v: any) => typeof v === 'string') || `#${i + 1}`
          const companies = item.companies
          return (
            <div key={i}>
              <span className="text-xs text-slate-300 font-medium">{label}</span>
              {item.count != null && <span className="text-xs text-slate-500 ml-1">({item.count})</span>}
              {Array.isArray(companies) && companies.length > 0 && (
                <div className="text-xs text-slate-500 ml-2">{companies.join(', ')}</div>
              )}
            </div>
          )
        })}
      </div>
    )
  }

  // Flat object
  if (typeof value === 'object' && value !== null && !Array.isArray(value)) {
    return (
      <div className="space-y-0.5">
        {Object.entries(value).slice(0, 8).map(([k, v]) => (
          <div key={k} className="flex justify-between text-xs">
            <span className="text-slate-400">{k.replace(/_/g, ' ')}</span>
            <span className="text-white">{typeof v === 'number' ? v.toLocaleString() : String(v)}</span>
          </div>
        ))}
      </div>
    )
  }

  return <span className="text-xs text-slate-500">--</span>
}

// ---------------------------------------------------------------------------
// Post-Checkout Success + Account Creation
// ---------------------------------------------------------------------------

const CHECKOUT_SESSION_URL = `${API_BASE}/api/v1/b2b/briefings/checkout-session`

function CheckoutSuccess({ vendor, sessionId }: { vendor: string; sessionId: string }) {
  const { signup, login } = useAuth()
  const navigate = useNavigate()
  const [sessionEmail, setSessionEmail] = useState('')
  const [tier, setTier] = useState('')
  const [fullName, setFullName] = useState('')
  const [companyName, setCompanyName] = useState('')
  const [password, setPassword] = useState('')
  const [signupError, setSignupError] = useState('')
  const [signupLoading, setSignupLoading] = useState(false)
  const [accountCreated, setAccountCreated] = useState(false)
  const [mode, setMode] = useState<'signup' | 'login'>('signup')

  useEffect(() => {
    if (!sessionId) return
    fetch(`${CHECKOUT_SESSION_URL}?session_id=${encodeURIComponent(sessionId)}`)
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (data?.email) setSessionEmail(data.email)
        if (data?.tier) setTier(data.tier)
      })
      .catch(() => {})
  }, [sessionId])

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    setSignupError('')
    if (password.length < 8) {
      setSignupError('Password must be at least 8 characters')
      return
    }
    setSignupLoading(true)
    try {
      if (mode === 'signup') {
        await signup(sessionEmail, password, fullName, companyName, 'b2b_retention')
      } else {
        await login(sessionEmail, password)
      }
      setAccountCreated(true)
      setTimeout(() => navigate('/landing'), 2000)
    } catch (err) {
      const msg = err instanceof Error ? err.message : 'Something went wrong'
      if (msg.toLowerCase().includes('already registered')) {
        setMode('login')
        setSignupError('Account exists -- sign in to link your subscription.')
      } else {
        setSignupError(msg)
      }
    } finally {
      setSignupLoading(false)
    }
  }

  return (
    <PublicLayout variant="report">
      <SeoHead
        title="Subscription Confirmed | Churn Signals"
        description={`Your ${vendor} churn intelligence subscription is confirmed.`}
        canonical="https://churnsignals.co/report"
      />
      <div className="min-h-[60vh] flex items-center justify-center px-6 py-16">
        <div className="max-w-lg w-full">
          {/* Confirmation header */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-16 h-16 rounded-full bg-green-500/10 border border-green-500/20 mb-6">
              <Check className="h-8 w-8 text-green-400" />
            </div>
            <h1 className="text-3xl font-bold mb-3">You're all set</h1>
            <p className="text-slate-400">
              Your {vendor} {tier === 'pro' ? 'Pro' : 'Standard'} subscription is confirmed.
            </p>
          </div>

          {/* What happens next */}
          <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-6 text-left space-y-4 mb-8">
            <h2 className="text-sm font-medium text-cyan-400 uppercase tracking-wider">What happens next</h2>
            <div className="space-y-3">
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 rounded-full bg-cyan-600/20 text-cyan-400 text-xs font-bold flex items-center justify-center shrink-0 mt-0.5">1</div>
                <div>
                  <p className="text-sm font-medium text-white">First report within 24 hours</p>
                  <p className="text-xs text-slate-400">We'll compile the latest churn signals for {vendor} and deliver your first full report.</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 rounded-full bg-cyan-600/20 text-cyan-400 text-xs font-bold flex items-center justify-center shrink-0 mt-0.5">2</div>
                <div>
                  <p className="text-sm font-medium text-white">Weekly delivery every Monday</p>
                  <p className="text-xs text-slate-400">Fresh intelligence lands in your inbox at the start of each week -- pain drivers, displacement data, and competitive signals.</p>
                </div>
              </div>
              <div className="flex items-start gap-3">
                <div className="w-6 h-6 rounded-full bg-cyan-600/20 text-cyan-400 text-xs font-bold flex items-center justify-center shrink-0 mt-0.5">3</div>
                <div>
                  <p className="text-sm font-medium text-white">Act on the signals</p>
                  <p className="text-xs text-slate-400">Use the intelligence to get ahead of churn -- engage at-risk accounts, address pain points, and track competitive threats.</p>
                </div>
              </div>
            </div>
          </div>

          {/* Account creation form */}
          {!accountCreated ? (
            <div className="bg-slate-800/60 border border-slate-700/50 rounded-xl p-6">
              <h2 className="text-lg font-semibold text-white mb-1">
                {mode === 'signup' ? 'Create your account' : 'Sign in to your account'}
              </h2>
              <p className="text-sm text-slate-400 mb-5">
                {mode === 'signup'
                  ? 'Set up your login to manage your subscription and access reports.'
                  : 'Sign in to link this subscription to your existing account.'}
              </p>

              {/* Purchased product card */}
              <div
                className="flex items-start gap-3 p-3 rounded-lg border mb-5"
                style={{ borderColor: tier === 'pro' ? 'rgb(6 182 212 / 0.5)' : 'rgb(139 92 246 / 0.5)', backgroundColor: tier === 'pro' ? 'rgb(8 51 68 / 0.3)' : 'rgb(76 29 149 / 0.2)' }}
              >
                <ShieldCheck className="h-5 w-5 mt-0.5 shrink-0 text-white" />
                <div>
                  <div className="text-sm font-medium text-white">
                    {vendor} -- {tier === 'pro' ? 'Pro' : 'Standard'}
                  </div>
                  <div className="text-xs text-slate-400">
                    {tier === 'pro'
                      ? 'Weekly reports + named account signals + urgency scoring'
                      : 'Weekly churn intelligence reports + competitive analysis'}
                  </div>
                </div>
              </div>

              {signupError && (
                <div className="flex items-center gap-2 text-sm text-red-400 bg-red-900/20 border border-red-800/50 rounded-lg px-3 py-2 mb-4">
                  <AlertCircle className="h-4 w-4 shrink-0" />
                  {signupError}
                </div>
              )}

              <form onSubmit={handleSubmit} className="space-y-4">
                {mode === 'signup' && (
                  <>
                    <div>
                      <label className="block text-sm text-slate-400 mb-1">Full name</label>
                      <input
                        type="text"
                        value={fullName}
                        onChange={e => setFullName(e.target.value)}
                        required
                        className="w-full px-3 py-2 bg-slate-900/60 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                        placeholder="Jane Smith"
                      />
                    </div>
                    <div>
                      <label className="block text-sm text-slate-400 mb-1">Company name</label>
                      <input
                        type="text"
                        value={companyName}
                        onChange={e => setCompanyName(e.target.value)}
                        required
                        className="w-full px-3 py-2 bg-slate-900/60 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                        placeholder="Acme Inc."
                      />
                    </div>
                  </>
                )}
                <div>
                  <label className="block text-sm text-slate-400 mb-1">Email</label>
                  <input
                    type="email"
                    value={sessionEmail}
                    onChange={e => setSessionEmail(e.target.value)}
                    required
                    className="w-full px-3 py-2 bg-slate-900/60 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                    placeholder="you@company.com"
                  />
                </div>
                <div>
                  <label className="block text-sm text-slate-400 mb-1">Password</label>
                  <input
                    type="password"
                    value={password}
                    onChange={e => setPassword(e.target.value)}
                    required
                    minLength={8}
                    className="w-full px-3 py-2 bg-slate-900/60 border border-slate-600/50 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500"
                    placeholder="Min. 8 characters"
                  />
                </div>
                <button
                  type="submit"
                  disabled={signupLoading}
                  className="w-full py-2.5 bg-cyan-600 hover:bg-cyan-500 disabled:opacity-50 rounded-lg text-white font-medium transition-colors cursor-pointer"
                >
                  {signupLoading
                    ? (mode === 'signup' ? 'Creating account...' : 'Signing in...')
                    : (mode === 'signup' ? 'Create Account' : 'Sign In')}
                </button>
              </form>
            </div>
          ) : (
            <div className="bg-green-900/20 border border-green-800/50 rounded-xl p-6 text-center">
              <Check className="h-6 w-6 text-green-400 mx-auto mb-2" />
              <p className="text-sm text-green-400 font-medium">Account created! Redirecting to your dashboard...</p>
            </div>
          )}

          <p className="text-sm text-slate-400 text-center mt-6">
            Questions? Reach us at{' '}
            <a href="mailto:outreach@churnsignals.co" className="text-cyan-400 hover:text-cyan-300 transition-colors">
              outreach@churnsignals.co
            </a>
          </p>
        </div>
      </div>
    </PublicLayout>
  )
}

// ---------------------------------------------------------------------------
// Main Component
// ---------------------------------------------------------------------------

export default function Report() {
  const [params] = useSearchParams()
  const vendor = params.get('vendor') || ''
  const token = params.get('ref') || ''
  const checkoutStatus = params.get('checkout') || ''
  // If mode=view, skip the gate and load report directly (post-gate redirect)
  const mode = params.get('mode') || ''

  const [email, setEmail] = useState('')
  const [status, setStatus] = useState<Status>(mode === 'view' ? 'loading_report' : 'idle')
  const [errorMsg, setErrorMsg] = useState('')
  const [reportData, setReportData] = useState<ReportData | null>(null)

  useEffect(() => {
    document.title = vendor
      ? `${vendor} Churn Intelligence Report -- Churn Signals`
      : 'Churn Intelligence Report -- Churn Signals'
  }, [vendor])

  // Auto-load report if mode=view (post-gate redirect)
  useEffect(() => {
    if (mode === 'view' && token && !reportData) {
      loadReport(token)
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [mode, token])

  async function loadReport(reportToken: string) {
    setStatus('loading_report')
    try {
      const res = await fetch(`${REPORT_DATA_URL}?token=${encodeURIComponent(reportToken)}`)
      if (res.ok) {
        const data = await res.json()
        setReportData(data)
        setStatus('report')
      } else {
        const body = await res.json().catch(() => ({ detail: 'Failed to load report' }))
        const detail = body.detail
        // Pydantic returns detail as array of objects -- flatten to string
        const msg = typeof detail === 'string'
          ? detail
          : Array.isArray(detail)
            ? detail.map((d: any) => d.msg || JSON.stringify(d)).join('; ')
            : `Error ${res.status}`
        setErrorMsg(msg)
        setStatus('error')
      }
    } catch {
      setErrorMsg('Network error loading report')
      setStatus('error')
    }
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!email.trim()) return

    setStatus('submitting')
    setErrorMsg('')

    try {
      const res = await fetch(GATE_URL, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email.trim(), token }),
      })

      if (res.ok) {
        const body = await res.json()
        // Gate passed -- load the full report inline
        const reportToken = body.report_token || token
        await loadReport(reportToken)
      } else {
        const body = await res.json().catch(() => ({ detail: 'Something went wrong' }))
        const detail = body.detail
        const msg = typeof detail === 'string'
          ? detail
          : Array.isArray(detail)
            ? detail.map((d: any) => d.msg || JSON.stringify(d)).join('; ')
            : `Error ${res.status}`
        setErrorMsg(msg)
        setStatus('error')
      }
    } catch {
      setErrorMsg('Network error -- please try again')
      setStatus('error')
    }
  }

  // Post-checkout success page
  if (checkoutStatus === 'success' && vendor) {
    return <CheckoutSuccess vendor={vendor} sessionId={params.get('session_id') || ''} />
  }

  if (!vendor || !token) {
    return (
      <PublicLayout variant="report">
        <div className="min-h-[60vh] flex items-center justify-center px-6">
          <div className="max-w-md text-center">
            <AlertCircle className="h-12 w-12 text-red-400 mx-auto mb-4" />
            <h1 className="text-2xl font-bold mb-2">Invalid Link</h1>
            <p className="text-slate-400">
              This report link is missing required parameters. Check the link from your email and try again.
            </p>
          </div>
        </div>
      </PublicLayout>
    )
  }

  // Loading report data
  if (status === 'loading_report') {
    return (
      <PublicLayout variant="report">
        <div className="min-h-[60vh] flex items-center justify-center px-6">
          <div className="text-center">
            <Loader2 className="h-10 w-10 text-cyan-400 animate-spin mx-auto mb-4" />
            <p className="text-slate-400">Loading {vendor} intelligence report...</p>
          </div>
        </div>
      </PublicLayout>
    )
  }

  // Full report view
  if (status === 'report' && reportData) {
    return <ReportView data={reportData} />
  }

  // Gate form
  return (
    <PublicLayout variant="report">
      <SeoHead
        title={`${vendor} Churn Intelligence Report | Churn Signals`}
        description={`Access the full ${vendor} churn intelligence report: account-level signals, displacement data, and risk scores.`}
        canonical={`https://churnsignals.co/report?vendor=${encodeURIComponent(vendor)}`}
      />
      <div className="min-h-[60vh] flex items-center justify-center px-6 py-16">
        <div className="w-full max-w-md">
          {/* Header */}
          <div className="text-center mb-8">
            <div className="inline-flex items-center justify-center w-14 h-14 rounded-full bg-cyan-500/10 border border-cyan-500/20 mb-5">
              <FileText className="h-7 w-7 text-cyan-400" />
            </div>
            <h1 className="text-2xl sm:text-3xl font-bold mb-3">
              <span className="text-cyan-400">{vendor}</span> Churn Intelligence
            </h1>
            <p className="text-slate-400 leading-relaxed">
              Enter your work email to access the full report -- account-level churn signals, displacement data, and risk scores.
            </p>
          </div>

          {/* Form */}
          <form onSubmit={handleSubmit} className="space-y-4">
            <div className="relative">
              <Mail className="absolute left-3.5 top-1/2 -translate-y-1/2 h-5 w-5 text-slate-500" />
              <input
                type="email"
                required
                value={email}
                onChange={e => setEmail(e.target.value)}
                placeholder="you@company.com"
                autoComplete="email"
                className="w-full pl-11 pr-4 py-3 bg-slate-800/80 border border-slate-700 rounded-lg text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500 focus:ring-1 focus:ring-cyan-500/30 transition-colors"
              />
            </div>

            <button
              type="submit"
              disabled={status === 'submitting' || !email.trim()}
              className="w-full py-3 bg-cyan-600 hover:bg-cyan-500 disabled:bg-slate-700 disabled:text-slate-500 rounded-lg text-white font-semibold transition-colors flex items-center justify-center gap-2"
            >
              {status === 'submitting' ? (
                <>
                  <Loader2 className="h-5 w-5 animate-spin" />
                  Loading report...
                </>
              ) : (
                'Access the Full Report'
              )}
            </button>

            {status === 'error' && (
              <div className="flex items-start gap-2 p-3 bg-red-500/10 border border-red-500/20 rounded-lg">
                <AlertCircle className="h-5 w-5 text-red-400 shrink-0 mt-0.5" />
                <p className="text-sm text-red-300">{errorMsg}</p>
              </div>
            )}
          </form>

          {/* Trust signals */}
          <div className="mt-8 flex items-center justify-center gap-2 text-xs text-slate-500">
            <ShieldCheck className="h-4 w-4" />
            <span>No spam. Unsubscribe anytime. Data sourced from public reviews.</span>
          </div>
        </div>
      </div>
    </PublicLayout>
  )
}
