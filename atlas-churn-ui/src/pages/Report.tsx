import { useState, useEffect, useRef, type FormEvent } from 'react'
import { useSearchParams, useNavigate } from 'react-router-dom'
import {
  FileText, Mail, AlertCircle, Loader2, ShieldCheck,
  TrendingUp, TrendingDown, Minus, Users, BarChart3, ArrowRight,
  X, Check,
} from 'lucide-react'
import PublicLayout from '../components/PublicLayout'
import SeoHead from '../components/SeoHead'
import { useAuth } from '../auth/AuthContext'
import { buildLoginRedirectPath, buildSignupRedirectPath } from '../auth/redirects'
import { StructuredReportData } from '../components/report-renderers/StructuredReportData'
import { isSpecializedReportType } from '../lib/reportConstants'
import { SpecializedReportData } from '../components/report-renderers/SpecializedReportData'
import { normalizeReportObject } from '../lib/reportNormalization'

const API_BASE = import.meta.env.VITE_API_BASE || ''
const GATE_URL = `${API_BASE}/api/v1/b2b/briefings/gate`
const REPORT_DATA_URL = `${API_BASE}/api/v1/b2b/briefings/report-data`
const CHECKOUT_URL = `${API_BASE}/api/v1/b2b/briefings/checkout`
const CACHE_BUSTER_PARAM = '_ts'
const WATCHLISTS_LOGIN_URL = buildLoginRedirectPath('/watchlists', 'b2b_retention')
const CHALLENGERS_LOGIN_URL = buildLoginRedirectPath('/challengers', 'b2b_challenger')
const WATCHLISTS_SIGNUP_URL = buildSignupRedirectPath('/watchlists', 'b2b_retention')
const CHALLENGERS_SIGNUP_URL = buildSignupRedirectPath('/challengers', 'b2b_challenger')

function addFreshParam(url: string, params: Record<string, string>): string {
  const next = new URL(url, window.location.origin)
  for (const [key, value] of Object.entries(params)) {
    if (value) next.searchParams.set(key, value)
  }
  next.searchParams.set(CACHE_BUSTER_PARAM, String(Date.now()))
  return next.toString()
}

function noStoreInit(init: RequestInit = {}): RequestInit {
  const headers = init.headers ? (init.headers as Record<string, string>) : {}
  return {
    ...init,
    cache: 'no-store',
    headers: {
      ...headers,
      'Cache-Control': 'no-cache',
      Pragma: 'no-cache',
    },
  }
}

type CheckoutResult = { ok: true } | { ok: false; error: string }

async function startCheckout(vendorName: string, tier: 'standard' | 'pro'): Promise<CheckoutResult> {
  try {
    const res = await fetch(CHECKOUT_URL, noStoreInit({
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ vendor_name: vendorName, tier }),
    }))
    if (!res.ok) {
      const body = await res.json().catch(() => ({ detail: 'Checkout failed' }))
      const detail = isRecord(body) ? body.detail : undefined
      return { ok: false, error: formatApiDetail(detail, res.status) }
    }

    const body = await res.json().catch(() => ({}))
    const url = isRecord(body) ? body.url : undefined
    if (typeof url !== 'string' || !url.trim()) {
      return { ok: false, error: 'Checkout session unavailable -- please try again.' }
    }

    window.location.assign(url)
    return { ok: true }
  } catch {
    return { ok: false, error: 'Network error -- please try again' }
  }
}

type Status = 'idle' | 'submitting' | 'loading_report' | 'report' | 'error'

type UnknownRecord = Record<string, unknown>

type ReportData = {
  vendor_name: string
  briefing: UnknownRecord
  intelligence_reports: Array<{
    report_type: string
    executive_summary: string | null
    data: UnknownRecord
    report_date: string | null
  }>
  product_profile: UnknownRecord | null
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

function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function asRecord(value: unknown): UnknownRecord {
  return isRecord(value) ? value : {}
}

function asArray(value: unknown): unknown[] {
  return Array.isArray(value) ? value : []
}

function firstStringValue(value: unknown): string | undefined {
  if (typeof value === 'string' && value.trim()) return value
  if (!isRecord(value)) return undefined
  return Object.values(value).find(
    (entry): entry is string => typeof entry === 'string' && entry.trim().length > 0,
  )
}

function formatApiDetail(detail: unknown, statusCode: number): string {
  if (typeof detail === 'string' && detail.trim()) return detail
  if (Array.isArray(detail)) {
    return detail.map((item) => {
      if (isRecord(item) && typeof item.msg === 'string') return item.msg
      return JSON.stringify(item)
    }).join('; ')
  }
  return `Error ${statusCode}`
}

function fmtPct(v: unknown) {
  const n = Number(v)
  return isNaN(n) ? 'N/A' : `${n.toFixed(1)}%`
}

function fmtScore(v: unknown) {
  const n = Number(v)
  return isNaN(n) ? 'N/A' : n.toFixed(1)
}

function normalizePublicReportData(data: ReportData): ReportData {
  return {
    ...data,
    briefing: normalizeReportObject(data.briefing),
    intelligence_reports: data.intelligence_reports.map((report) => ({
      ...report,
      data: normalizeReportObject(report.data),
    })),
    product_profile: data.product_profile ? normalizeReportObject(data.product_profile) : null,
  }
}

function positiveInteger(value: unknown): number | null {
  if (typeof value === 'number' && Number.isInteger(value) && value > 0) return value
  if (typeof value === 'string') {
    const trimmed = value.trim()
    if (!trimmed) return null
    const parsed = Number.parseInt(trimmed, 10)
    return Number.isInteger(parsed) && parsed > 0 ? parsed : null
  }
  return null
}

function reportEvidenceSnapshot(report: { report_date: string | null; data: unknown }) {
  const reportData = isRecord(report.data) ? report.data : {}
  const asOfDate = firstStringValue(reportData.data_as_of_date)
    || firstStringValue(reportData.as_of_date)
    || firstStringValue(reportData.report_date)
    || report.report_date
    || null
  const windowDays = positiveInteger(reportData.analysis_window_days)
    ?? positiveInteger(reportData.window_days)
    ?? positiveInteger(reportData.evidence_window_days)
  return {
    asOfDate,
    windowDays,
  }
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
  const [checkoutError, setCheckoutError] = useState('')

  const handleSelect = async (tier: 'standard' | 'pro') => {
    setCheckoutError('')
    setLoading(tier)
    const result = await startCheckout(vendorName, tier)
    setLoading(null)
    if (!result.ok) {
      setCheckoutError(result.error)
    }
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

        {checkoutError && (
          <div
            role="alert"
            className="mt-6 flex items-center gap-2 rounded-lg border border-red-800/50 bg-red-900/20 px-3 py-2 text-sm text-red-300"
          >
            <AlertCircle className="h-4 w-4 shrink-0" />
            <span>{checkoutError}</span>
          </div>
        )}

        <p className="text-center text-xs text-slate-500 mt-6">Cancel anytime. No long-term contracts.</p>
      </div>
    </div>
  )
}

function ReportView({ data }: { data: ReportData }) {
  const [searchParams] = useSearchParams()
  const reportBackTo = (() => {
    const query = searchParams.toString()
    return query ? `/report?${query}` : '/report'
  })()
  const [showPricing, setShowPricing] = useState(false)
  const b = asRecord(data.briefing)
  const score = Number(b.churn_pressure_score) || 0
  const pains = asArray(b.pain_breakdown).filter(isRecord)
  const maxPain = Math.max(...pains.map((pain) => Number(pain.count) || 0), 1)
  const displacements = asArray(b.top_displacement_targets).filter(isRecord)
  const evidence = asArray(b.evidence)
  const namedAccountCount = Number(b.named_account_count) || asArray(b.named_accounts).length
  const featureGaps = asArray(b.top_feature_gaps)
  const painLabels = asRecord(b.pain_labels) as Record<string, string>
  const timingSummary = firstStringValue(b.timing_summary) || ''
  const priorityTriggers = asArray(b.priority_timing_triggers)
  const buyerProfiles = asArray(b.buyer_profiles).filter(isRecord)
  const crossVendorConclusions = asArray(b.cross_vendor_conclusions).filter(isRecord)
  const segIntel = asRecord(b.segment_intelligence)
  const topSegments = asArray(segIntel.top_segments).filter(isRecord)

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
                {firstStringValue(b.category) || 'Software'}
              </span>
            </div>
            {firstStringValue(b.headline) && (
              <p className="text-lg text-slate-300 font-medium">{firstStringValue(b.headline)}</p>
            )}
          </div>
          <div className={`flex flex-col items-center px-6 py-4 rounded-xl border ${pressureBg(score)}`}>
            <div className={`text-4xl font-bold ${pressureColor(score)}`}>{score.toFixed(0)}</div>
            <div className="text-xs text-slate-400 uppercase tracking-wider">Churn Pressure</div>
            <div className="flex items-center gap-1 mt-1">
              <TrendIcon trend={firstStringValue(b.trend) ?? null} />
              <span className="text-xs text-slate-400 capitalize">{firstStringValue(b.trend) || 'Stable'}</span>
            </div>
          </div>
        </div>

        {/* Executive Summary */}
        {firstStringValue(b.executive_summary) && (
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-6 mb-8">
            <h3 className="text-sm font-medium text-cyan-400 uppercase tracking-wider mb-3">Executive Summary</h3>
            <p className="text-sm text-slate-300 leading-relaxed">{firstStringValue(b.executive_summary)}</p>
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
                {pains.map((p, i: number) => (
                  <RankedBar
                    key={i}
                    label={painLabels[String(p.category)] || String(p.category || 'Other')}
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
                {displacements.map((d, i: number) => (
                  <div key={i} className="flex items-center justify-between py-2 border-b border-slate-800 last:border-0">
                    <span className="text-sm text-slate-300">{String(d.competitor || d.name || '')}</span>
                    <span className="text-sm text-slate-400">{String(d.count || d.mentions || 0)} mentions</span>
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
                {featureGaps.map((g, i: number) => (
                  <li key={i} className="text-sm text-slate-300 flex items-start gap-2">
                    <span className="text-cyan-500 mt-0.5">--</span>
                    {typeof g === 'string' ? g : firstStringValue(g) || `#${i + 1}`}
                  </li>
                ))}
              </ul>
            </div>
          )}
        </div>

        {/* Segment Intelligence */}
        {topSegments.length > 0 && (
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 mb-8">
            <h3 className="text-sm font-medium text-cyan-400 uppercase tracking-wider mb-4">Segments at Risk</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700">
                    <th className="text-left text-xs text-slate-400 pb-2 pr-4">Role</th>
                    <th className="text-left text-xs text-slate-400 pb-2 pr-4">Stage</th>
                    <th className="text-left text-xs text-slate-400 pb-2 pr-4">Top Pain</th>
                    <th className="text-right text-xs text-slate-400 pb-2">Signals</th>
                  </tr>
                </thead>
                <tbody>
                  {topSegments.map((seg, i: number) => (
                    <tr key={i} className="border-b border-slate-800 last:border-0">
                      <td className="py-2 pr-4 text-slate-300 capitalize">{String(seg.role_type || '').replace(/_/g, ' ')}</td>
                      <td className="py-2 pr-4 text-slate-400 capitalize">{String(seg.top_buying_stage || '').replace(/_/g, ' ')}</td>
                      <td className="py-2 pr-4 text-amber-400 capitalize">{String(seg.top_pain || '').replace(/_/g, ' ')}</td>
                      <td className="py-2 text-right text-slate-500">{Number(seg.review_count) || 0}</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Timing Window */}
        {(timingSummary || priorityTriggers.length > 0) && (
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 mb-8">
            <h3 className="text-sm font-medium text-cyan-400 uppercase tracking-wider mb-3">Timing Window</h3>
            {timingSummary && <p className="text-sm text-slate-300 mb-3 leading-relaxed">{timingSummary}</p>}
            {priorityTriggers.length > 0 && (
              <ul className="space-y-1">
                {priorityTriggers.slice(0, 3).map((t, i: number) => (
                  <li key={i} className="text-sm text-slate-400 flex items-start gap-2">
                    <span className="text-cyan-500 mt-0.5">--</span>
                    {typeof t === 'string' ? t : firstStringValue(t) || ''}
                  </li>
                ))}
              </ul>
            )}
          </div>
        )}

        {/* Competitive Intel */}
        {crossVendorConclusions.filter(c => String(c.analysis_type) === 'pairwise_battle' && firstStringValue(c.summary)).length > 0 && (
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 mb-8">
            <h3 className="text-sm font-medium text-cyan-400 uppercase tracking-wider mb-4">Competitive Intel</h3>
            <div className="space-y-3">
              {crossVendorConclusions
                .filter(c => String(c.analysis_type) === 'pairwise_battle' && firstStringValue(c.summary))
                .slice(0, 2)
                .map((c, i: number) => {
                  const vendors = asArray(c.vendors).map(v => String(v))
                  const conf = Math.round((Number(c.confidence) || 0) * 100)
                  return (
                    <div key={i} className="border-l-2 border-blue-500/40 pl-3">
                      {vendors.length > 0 && (
                        <div className="text-xs text-slate-500 mb-1">{vendors.join(' vs ')}</div>
                      )}
                      <p className="text-sm text-slate-300 leading-relaxed">{firstStringValue(c.summary)}</p>
                      {conf > 0 && <div className="text-xs text-slate-500 mt-1">{conf}% confidence</div>}
                    </div>
                  )
                })}
            </div>
          </div>
        )}

        {/* Who Is Evaluating */}
        {buyerProfiles.length > 0 && (
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 mb-8">
            <h3 className="text-sm font-medium text-cyan-400 uppercase tracking-wider mb-4">Who Is Evaluating</h3>
            <div className="overflow-x-auto">
              <table className="w-full text-sm">
                <thead>
                  <tr className="border-b border-slate-700">
                    <th className="text-left text-xs text-slate-400 pb-2 pr-4">Role</th>
                    <th className="text-left text-xs text-slate-400 pb-2 pr-4">Stage</th>
                    <th className="text-right text-xs text-slate-400 pb-2 pr-4">Urgency</th>
                    <th className="text-right text-xs text-slate-400 pb-2">Signals</th>
                  </tr>
                </thead>
                <tbody>
                  {buyerProfiles.slice(0, 4).map((bp, i: number) => {
                    const urgency = Number(bp.avg_urgency) || 0
                    const urgencyColor = urgency >= 7 ? 'text-red-400' : urgency >= 4 ? 'text-amber-400' : 'text-emerald-400'
                    return (
                      <tr key={i} className="border-b border-slate-800 last:border-0">
                        <td className="py-2 pr-4 text-slate-300 capitalize">{String(bp.role_type || '').replace(/_/g, ' ')}</td>
                        <td className="py-2 pr-4 text-slate-400 capitalize">{String(bp.buying_stage || '').replace(/_/g, ' ')}</td>
                        <td className={`py-2 pr-4 text-right font-semibold ${urgencyColor}`}>{urgency.toFixed(1)}</td>
                        <td className="py-2 text-right text-slate-500">{Number(bp.review_count) || 0}</td>
                      </tr>
                    )
                  })}
                </tbody>
              </table>
            </div>
          </div>
        )}

        {/* Customer Quotes */}
        {evidence.length > 0 && (
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 mb-8">
            <h3 className="text-sm font-medium text-cyan-400 uppercase tracking-wider mb-4">What Customers Are Saying</h3>
            <div className="space-y-3">
              {evidence.map((q, i: number) => {
                const text = typeof q === 'string' ? q : String(asRecord(q).quote || asRecord(q).text || '')
                if (!text) return null
                return (
                  <blockquote key={i} className="text-sm text-slate-300 italic border-l-2 border-red-500/50 pl-3 break-words whitespace-pre-wrap">
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
                (() => {
                  const evidenceSnapshot = reportEvidenceSnapshot(report)
                  return (
                    <div key={idx} className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 min-w-0 overflow-hidden">
                      <div className="flex flex-wrap items-center gap-3 mb-3">
                        <span className="px-2 py-0.5 bg-cyan-500/20 text-cyan-400 rounded text-xs font-medium break-all">
                          {report.report_type.replace(/_/g, ' ')}
                        </span>
                        {report.report_date && (
                          <span className="text-xs text-slate-500 break-all">{report.report_date}</span>
                        )}
                      </div>
                      {report.executive_summary && (
                        <p className="text-sm text-slate-300 mb-4 break-words whitespace-pre-wrap">{report.executive_summary}</p>
                      )}
                      <IntelligenceData
                        reportType={report.report_type}
                        data={report.data}
                        vendorName={data.vendor_name}
                        backTo={reportBackTo}
                        asOfDate={evidenceSnapshot.asOfDate}
                        windowDays={evidenceSnapshot.windowDays}
                      />
                    </div>
                  )
                })()
              ))}
            </div>
          </div>
        )}

        {/* Product Profile */}
        {data.product_profile && (
          <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 mb-8">
            <h3 className="text-sm font-medium text-cyan-400 uppercase tracking-wider mb-3">Product Profile</h3>
            {firstStringValue(data.product_profile.profile_summary) && (
              <p className="text-sm text-slate-300 mb-4">{firstStringValue(data.product_profile.profile_summary)}</p>
            )}
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {asArray(data.product_profile.strengths).length > 0 && (
                <div>
                  <h4 className="text-xs font-medium text-emerald-400 uppercase mb-2">Strengths</h4>
                  <ul className="space-y-1">
                    {asArray(data.product_profile.strengths).map((s, i: number) => (
                      <li key={i} className="text-xs text-slate-400">
                        {typeof s === 'string' ? s : firstStringValue(s) || `#${i + 1}`}
                      </li>
                    ))}
                  </ul>
                </div>
              )}
              {asArray(data.product_profile.weaknesses).length > 0 && (
                <div>
                  <h4 className="text-xs font-medium text-red-400 uppercase mb-2">Weaknesses</h4>
                  <ul className="space-y-1">
                    {asArray(data.product_profile.weaknesses).map((w, i: number) => (
                      <li key={i} className="text-xs text-slate-400">
                        {typeof w === 'string' ? w : firstStringValue(w) || `#${i + 1}`}
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
            Get Weekly Intelligence
          </button>
          <div className="mt-4 flex flex-wrap items-center justify-center gap-4 text-sm text-slate-400">
            <a href={WATCHLISTS_LOGIN_URL} className="hover:text-white transition-colors">
              Sign in to Watchlists
            </a>
            <a href={CHALLENGERS_LOGIN_URL} className="hover:text-white transition-colors">
              Sign in to Challengers
            </a>
          </div>
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
function IntelligenceData({
  reportType,
  data,
  vendorName,
  backTo,
  asOfDate,
  windowDays,
}: {
  reportType: string
  data: unknown
  vendorName: string
  backTo: string
  asOfDate?: string | null
  windowDays?: number | null
}) {
  if (isSpecializedReportType(reportType)) {
    return <SpecializedReportData reportType={reportType} data={data} vendorName={vendorName} backTo={backTo} asOfDate={asOfDate} windowDays={windowDays} />
  }
  return (
    <StructuredReportData
      data={normalizeReportObject(isRecord(data) ? data : {})}
      skipKeys={[
        'report_date',
        'data_as_of_date',
        'as_of_date',
        'window_days',
        'analysis_window_days',
        'evidence_window_days',
        'primary_vendor',
        'comparison_vendor',
      ]}
      vendorName={vendorName}
      backTo={backTo}
      asOfDate={asOfDate}
      windowDays={windowDays}
    />
  )
}

// ---------------------------------------------------------------------------
// Post-Checkout Success + Account Creation
// ---------------------------------------------------------------------------

const CHECKOUT_SESSION_URL = `${API_BASE}/api/v1/b2b/briefings/checkout-session`

function CheckoutSuccess({ vendor, sessionId }: { vendor: string; sessionId: string }) {
  const { signup, login } = useAuth()
  const navigate = useNavigate()
  const sessionRequestIdRef = useRef(0)
  const redirectTimerRef = useRef<number | null>(null)
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
    sessionRequestIdRef.current += 1
    if (redirectTimerRef.current != null) {
      window.clearTimeout(redirectTimerRef.current)
      redirectTimerRef.current = null
    }
    setSessionEmail('')
    setTier('')
    setFullName('')
    setCompanyName('')
    setPassword('')
    setSignupError('')
    setSignupLoading(false)
    setAccountCreated(false)
    setMode('signup')
    if (!sessionId) return
    const requestId = sessionRequestIdRef.current
    fetch(addFreshParam(CHECKOUT_SESSION_URL, { session_id: sessionId }), noStoreInit())
      .then(r => r.ok ? r.json() : null)
      .then(data => {
        if (sessionRequestIdRef.current !== requestId) return
        if (data?.email) setSessionEmail(data.email)
        if (data?.tier) setTier(data.tier)
      })
      .catch(() => {})
  }, [sessionId, vendor])

  useEffect(() => () => {
    sessionRequestIdRef.current += 1
    if (redirectTimerRef.current != null) {
      window.clearTimeout(redirectTimerRef.current)
      redirectTimerRef.current = null
    }
  }, [])

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    const requestId = sessionRequestIdRef.current
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
      if (sessionRequestIdRef.current !== requestId) return
      setAccountCreated(true)
      if (redirectTimerRef.current != null) {
        window.clearTimeout(redirectTimerRef.current)
      }
      redirectTimerRef.current = window.setTimeout(() => {
        if (sessionRequestIdRef.current !== requestId) return
        navigate('/watchlists')
      }, 2000)
    } catch (err) {
      if (sessionRequestIdRef.current !== requestId) return
      const msg = err instanceof Error ? err.message : 'Something went wrong'
      if (msg.toLowerCase().includes('already registered')) {
        setMode('login')
        setSignupError('Account exists -- sign in to link your subscription.')
      } else {
        setSignupError(msg)
      }
    } finally {
      if (sessionRequestIdRef.current !== requestId) return
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
              <p className="text-sm text-green-400 font-medium">Account ready. Redirecting to Watchlists...</p>
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
  const routeVersionRef = useRef(0)
  const loadRequestIdRef = useRef(0)

  useEffect(() => {
    document.title = vendor
      ? `${vendor} Churn Intelligence Report -- Churn Signals`
      : 'Churn Intelligence Report -- Churn Signals'
  }, [vendor])

  useEffect(() => {
    routeVersionRef.current += 1
    loadRequestIdRef.current += 1
    setEmail('')
    setErrorMsg('')
    setReportData(null)
    setStatus(mode === 'view' && token ? 'loading_report' : 'idle')
  }, [vendor, token, mode, checkoutStatus])

  useEffect(() => {
    if (checkoutStatus === 'success' || mode !== 'view' || !token) return
    void loadReport(token, routeVersionRef.current)
  }, [checkoutStatus, mode, token, vendor])

  async function loadReport(reportToken: string, routeVersion = routeVersionRef.current) {
    const requestId = loadRequestIdRef.current + 1
    loadRequestIdRef.current = requestId
    setStatus('loading_report')
    try {
      const res = await fetch(
        addFreshParam(REPORT_DATA_URL, { token: reportToken }),
        noStoreInit(),
      )
      if (routeVersionRef.current !== routeVersion || loadRequestIdRef.current !== requestId) return
      if (res.ok) {
        const data = normalizePublicReportData(await res.json() as ReportData)
        setReportData(data)
        setStatus('report')
      } else {
        const body = await res.json().catch(() => ({ detail: 'Failed to load report' }))
        setErrorMsg(formatApiDetail(body.detail, res.status))
        setStatus('error')
      }
    } catch {
      if (routeVersionRef.current !== routeVersion || loadRequestIdRef.current !== requestId) return
      setErrorMsg('Network error loading report')
      setStatus('error')
    }
  }

  async function handleSubmit(e: FormEvent) {
    e.preventDefault()
    if (!email.trim()) return

    const routeVersion = routeVersionRef.current
    const requestId = loadRequestIdRef.current + 1
    loadRequestIdRef.current = requestId
    setStatus('submitting')
    setErrorMsg('')

    try {
      const res = await fetch(GATE_URL, noStoreInit({
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ email: email.trim(), token }),
      }))
      if (routeVersionRef.current !== routeVersion || loadRequestIdRef.current !== requestId) return

      if (res.ok) {
        const body = await res.json()
        // Gate passed -- load the full report inline
        const reportToken = body.report_token || token
        await loadReport(reportToken, routeVersion)
      } else {
        const body = await res.json().catch(() => ({ detail: 'Something went wrong' }))
        setErrorMsg(formatApiDetail(body.detail, res.status))
        setStatus('error')
      }
    } catch {
      if (routeVersionRef.current !== routeVersion || loadRequestIdRef.current !== requestId) return
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
          <div className="mt-5 flex flex-wrap items-center justify-center gap-3 text-sm">
            <a href={WATCHLISTS_LOGIN_URL} className="text-cyan-400 hover:text-cyan-300 transition-colors">
              Sign in to Watchlists
            </a>
            <a href={CHALLENGERS_LOGIN_URL} className="text-cyan-400 hover:text-cyan-300 transition-colors">
              Sign in to Challengers
            </a>
            <a href={WATCHLISTS_SIGNUP_URL} className="text-slate-400 hover:text-white transition-colors">
              Start Vendor Retention
            </a>
            <a href={CHALLENGERS_SIGNUP_URL} className="text-slate-400 hover:text-white transition-colors">
              Start Challenger Lead Gen
            </a>
          </div>
        </div>
      </div>
    </PublicLayout>
  )
}
