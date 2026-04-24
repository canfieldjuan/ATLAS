import { useState, useEffect, useCallback, useRef, type ReactNode } from 'react'
import { Link } from 'react-router-dom'
import {
  X, ExternalLink, Quote, User, Building2, Calendar,
  Star, Tag, Fingerprint, FileText, ChevronRight, Loader2,
  Pin, Flag, EyeOff, RotateCcw, Copy,
} from 'lucide-react'
import { clsx } from 'clsx'
import { fetchWitness, fetchAnnotations, setAnnotation, removeAnnotations, fetchAccountsInMotionFeed } from '../api/client'
import type { EvidenceWitnessDetail, EvidenceAnnotation, AccountsInMotionFeedItem } from '../api/client'

// Inject keyframes once at module load (not per-render)
if (typeof document !== 'undefined' && !document.getElementById('evidence-drawer-keyframes')) {
  const style = document.createElement('style')
  style.id = 'evidence-drawer-keyframes'
  style.textContent = `
    @keyframes slideInRight {
      from { transform: translateX(100%); opacity: 0.8; }
      to { transform: translateX(0); opacity: 1; }
    }
  `
  document.head.appendChild(style)
}

const SOURCE_COLORS: Record<string, string> = {
  g2: 'bg-orange-500/20 text-orange-300 border-orange-500/30',
  capterra: 'bg-blue-500/20 text-blue-300 border-blue-500/30',
  trustradius: 'bg-violet-500/20 text-violet-300 border-violet-500/30',
  reddit: 'bg-red-500/20 text-red-300 border-red-500/30',
  gartner: 'bg-emerald-500/20 text-emerald-300 border-emerald-500/30',
  peerspot: 'bg-teal-500/20 text-teal-300 border-teal-500/30',
  trustpilot: 'bg-green-500/20 text-green-300 border-green-500/30',
  getapp: 'bg-sky-500/20 text-sky-300 border-sky-500/30',
  hackernews: 'bg-amber-500/20 text-amber-300 border-amber-500/30',
  twitter: 'bg-cyan-500/20 text-cyan-300 border-cyan-500/30',
  youtube: 'bg-rose-500/20 text-rose-300 border-rose-500/30',
  stackoverflow: 'bg-yellow-500/20 text-yellow-300 border-yellow-500/30',
  producthunt: 'bg-pink-500/20 text-pink-300 border-pink-500/30',
  github: 'bg-slate-400/20 text-slate-300 border-slate-400/30',
  quora: 'bg-red-400/20 text-red-300 border-red-400/30',
  rss: 'bg-amber-400/20 text-amber-300 border-amber-400/30',
}

const SIGNAL_COLORS: Record<string, string> = {
  pricing_backlash: 'bg-red-900/40 text-red-300',
  complaint: 'bg-orange-900/40 text-orange-300',
  feature_gap: 'bg-amber-900/40 text-amber-300',
  competitor_pressure: 'bg-violet-900/40 text-violet-300',
  recommendation: 'bg-emerald-900/40 text-emerald-300',
  positive_anchor: 'bg-green-900/40 text-green-300',
  event: 'bg-cyan-900/40 text-cyan-300',
}

function SourceBadge({ source }: { source: string }) {
  const colors = SOURCE_COLORS[source.toLowerCase()] || 'bg-slate-600/30 text-slate-300 border-slate-500/30'
  return (
    <span className={clsx('inline-flex items-center px-2 py-0.5 rounded text-xs font-medium border', colors)}>
      {source}
    </span>
  )
}

function highlightExcerpt(
  fullText: string,
  excerptText: string,
  highlightStart?: number | null,
  highlightEnd?: number | null,
): ReactNode {
  if (!fullText) return <></>

  // Prefer offset-based highlighting when the API provides adjusted positions
  if (highlightStart != null && highlightEnd != null
      && highlightStart >= 0 && highlightEnd > highlightStart
      && highlightEnd <= fullText.length) {
    const before = fullText.slice(0, highlightStart)
    const match = fullText.slice(highlightStart, highlightEnd)
    const after = fullText.slice(highlightEnd)
    return (
      <>
        {before}
        <mark className="bg-cyan-500/20 text-cyan-200 rounded px-0.5">{match}</mark>
        {after}
      </>
    )
  }

  // Fall back to substring search
  if (!excerptText) return <>{fullText}</>
  const lower = fullText.toLowerCase()
  const excerptLower = excerptText.toLowerCase()
  const idx = lower.indexOf(excerptLower)

  if (idx === -1) return <>{fullText}</>

  const before = fullText.slice(0, idx)
  const match = fullText.slice(idx, idx + excerptText.length)
  const after = fullText.slice(idx + excerptText.length)

  return (
    <>
      {before}
      <mark className="bg-cyan-500/20 text-cyan-200 rounded px-0.5">{match}</mark>
      {after}
    </>
  )
}

interface EvidenceDrawerProps {
  vendorName: string
  witnessId: string | null
  asOfDate?: string | null
  windowDays?: number
  open: boolean
  onClose: () => void
  explorerUrl?: string | null
  backToPath?: string | null
}

function reviewDetailPath(reviewId: string, backToPath?: string | null) {
  const params = new URLSearchParams()
  if (backToPath) params.set('back_to', backToPath)
  const query = params.toString()
  return query ? `/reviews/${reviewId}?${query}` : `/reviews/${reviewId}`
}

function accountReviewPath(row: AccountsInMotionFeedItem, backToPath?: string | null) {
  const params = new URLSearchParams()
  params.set('account_vendor', row.vendor || '')
  params.set('account_company', row.company || '')
  params.set('account_report_date', row.report_date || '')
  params.set('account_watch_vendor', row.watch_vendor || '')
  params.set('account_category', row.category || '')
  params.set('account_track_mode', row.track_mode || '')
  if (backToPath) params.set('back_to', backToPath)
  return `/watchlists?${params.toString()}`
}

function watchlistsPath(vendorName: string, backToPath?: string | null) {
  const params = new URLSearchParams()
  params.set('vendor_name', vendorName)
  if (backToPath) params.set('back_to', backToPath)
  return `/watchlists?${params.toString()}`
}

function alertsPath(vendorName: string, backToPath?: string | null, companyName?: string | null) {
  const params = new URLSearchParams()
  params.set('vendor', vendorName)
  if (companyName?.trim()) {
    params.set('company', companyName.trim())
  }
  if (backToPath) params.set('back_to', backToPath)
  return `/alerts?${params.toString()}`
}

function opportunitiesPath(vendorName: string, backToPath?: string | null) {
  const params = new URLSearchParams()
  params.set('vendor', vendorName)
  if (backToPath) params.set('back_to', backToPath)
  return `/opportunities?${params.toString()}`
}

function vendorWorkspacePath(vendorName: string, backToPath?: string | null) {
  const params = new URLSearchParams()
  if (backToPath) params.set('back_to', backToPath)
  const query = params.toString()
  return query ? `/vendors/${encodeURIComponent(vendorName)}?${query}` : `/vendors/${encodeURIComponent(vendorName)}`
}

function parseBackTarget(value: string | null | undefined) {
  if (!value) return null
  if (value.startsWith('/')) return value
  try {
    const url = new URL(value, window.location.origin)
    if (url.origin !== window.location.origin) return null
    return `${url.pathname}${url.search}`
  } catch {
    return null
  }
}

function parseSnapshotDate(value: string | null | undefined) {
  const text = value?.trim() || ''
  return /^\d{4}-\d{2}-\d{2}$/.test(text) ? text : null
}

function upstreamSnapshotDate(value: string | null | undefined) {
  let current = parseBackTarget(value)
  while (current) {
    try {
      const url = new URL(current, window.location.origin)
      const accountReportDate = parseSnapshotDate(url.searchParams.get('account_report_date'))
      if (accountReportDate) return accountReportDate
      const asOfDate = parseSnapshotDate(url.searchParams.get('as_of_date'))
      if (asOfDate) return asOfDate
      current = parseBackTarget(url.searchParams.get('back_to'))
    } catch {
      return null
    }
  }
  return null
}

function upstreamNestedPath(
  value: string | null | undefined,
  prefix: '/alerts' | '/vendors/' | '/watchlists' | '/reports' | '/opportunities',
) {
  let current = parseBackTarget(value)
  while (current) {
    if (current.startsWith(prefix)) return current
    try {
      const url = new URL(current, window.location.origin)
      current = parseBackTarget(url.searchParams.get('back_to'))
    } catch {
      return null
    }
  }
  return null
}

function upstreamReviewPath(value: string | null | undefined, reviewId: string) {
  let current = parseBackTarget(value)
  while (current) {
    try {
      const url = new URL(current, window.location.origin)
      if (url.pathname === `/reviews/${encodeURIComponent(reviewId)}`) return current
      current = parseBackTarget(url.searchParams.get('back_to'))
    } catch {
      return null
    }
  }
  return null
}

function upstreamAccountReviewPath(
  value: string | null | undefined,
  vendorName: string,
  reviewerCompany: string,
) {
  const normalizedVendor = vendorName.trim().toLowerCase()
  const normalizedCompany = reviewerCompany.trim().toLowerCase()
  let current = parseBackTarget(value)
  while (current) {
    try {
      const url = new URL(current, window.location.origin)
      if (url.pathname === '/watchlists') {
        const accountCompany = url.searchParams.get('account_company')?.trim().toLowerCase() || ''
        const accountVendor = url.searchParams.get('account_vendor')?.trim().toLowerCase() || ''
        if (accountCompany === normalizedCompany && (!accountVendor || accountVendor === normalizedVendor)) {
          return current
        }
      }
      current = parseBackTarget(url.searchParams.get('back_to'))
    } catch {
      return null
    }
  }
  return null
}

function toAbsoluteUrl(pathOrUrl: string) {
  if (typeof window === 'undefined') return pathOrUrl
  return new URL(pathOrUrl, window.location.origin).toString()
}

function AnnotationRemovalModal({
  annotationType,
  confirming,
  error,
  onCancel,
  onConfirm,
}: {
  annotationType: string
  confirming: boolean
  error: string | null
  onCancel: () => void
  onConfirm: () => void
}) {
  return (
    <div className="fixed inset-0 z-[60] flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-slate-950/80"
        onClick={() => {
          if (!confirming) onCancel()
        }}
      />
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="evidence-annotation-removal-title"
        className="relative z-10 w-full max-w-md rounded-xl border border-amber-500/20 bg-slate-950 p-6 shadow-2xl"
      >
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-start gap-3">
            <div className="rounded-lg bg-amber-500/10 p-2 text-amber-300">
              <RotateCcw className="h-4 w-4" />
            </div>
            <div>
              <h2 id="evidence-annotation-removal-title" className="text-base font-semibold text-white">
                Remove {annotationType}
              </h2>
              <p className="mt-1 text-sm text-slate-400">
                Remove {annotationType} annotation from this witness? This clears the saved analyst override.
              </p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => {
              if (!confirming) onCancel()
            }}
            className="text-slate-500 hover:text-slate-300 disabled:cursor-not-allowed disabled:opacity-50"
            disabled={confirming}
            aria-label="Close annotation removal dialog"
          >
            <X className="h-4 w-4" />
          </button>
        </div>
        {error ? (
          <div role="alert" className="mt-4 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
            {error}
          </div>
        ) : null}
        <div className="mt-6 flex justify-end gap-3">
          <button
            type="button"
            onClick={onCancel}
            disabled={confirming}
            className="rounded-md bg-slate-800 px-3 py-2 text-sm font-medium text-slate-300 hover:bg-slate-700 disabled:cursor-not-allowed disabled:opacity-50"
          >
            Cancel
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={confirming}
            className="inline-flex items-center gap-2 rounded-md bg-amber-500 px-3 py-2 text-sm font-medium text-slate-950 hover:bg-amber-400 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {confirming ? 'Removing...' : `Remove ${annotationType}`}
          </button>
        </div>
      </div>
    </div>
  )
}

export default function EvidenceDrawer({
  vendorName,
  witnessId,
  asOfDate,
  windowDays,
  open,
  onClose,
  explorerUrl,
  backToPath,
}: EvidenceDrawerProps) {
  const [witness, setWitness] = useState<EvidenceWitnessDetail | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState('')
  const [annotation, setAnnotationState] = useState<EvidenceAnnotation | null>(null)
  const [matchedAccountReviewPath, setMatchedAccountReviewPath] = useState<string | null>(null)
  const [annotating, setAnnotating] = useState(false)
  const [annotationActionError, setAnnotationActionError] = useState<string | null>(null)
  const [copiedLinkKey, setCopiedLinkKey] = useState<string | null>(null)
  const [linkCopyError, setLinkCopyError] = useState<string | null>(null)
  const [pendingRemoveAnnotation, setPendingRemoveAnnotation] = useState(false)
  const [removeAnnotationError, setRemoveAnnotationError] = useState<string | null>(null)
  const panelRef = useRef<HTMLDivElement>(null)
  const effectiveAsOfDate = parseSnapshotDate(asOfDate) ?? upstreamSnapshotDate(backToPath)

  useEffect(() => {
    if (!open || !witnessId || !vendorName) return
    setLoading(true)
    setError('')
    setAnnotationState(null)
    setMatchedAccountReviewPath(null)
    setAnnotationActionError(null)
    setCopiedLinkKey(null)
    setLinkCopyError(null)
    setPendingRemoveAnnotation(false)
    setRemoveAnnotationError(null)
    Promise.all([
      fetchWitness(witnessId, vendorName, {
        as_of_date: effectiveAsOfDate || undefined,
        window_days: windowDays,
      }),
      fetchAnnotations({ vendor_name: vendorName }).catch(() => ({ annotations: [] })),
    ])
      .then(([witnessRes, annotRes]) => {
        setWitness(witnessRes.witness)
        const match = annotRes.annotations.find((a: EvidenceAnnotation) => a.witness_id === witnessId)
        setAnnotationState(match || null)
      })
      .catch(err => setError(err instanceof Error ? err.message : 'Failed to load witness'))
      .finally(() => setLoading(false))
  }, [open, witnessId, vendorName, effectiveAsOfDate, windowDays])

  useEffect(() => {
    if (!open || !witness?.reviewer_company || !vendorName) {
      setMatchedAccountReviewPath(null)
      return
    }
    const exactAccountReviewPath = upstreamAccountReviewPath(backToPath, vendorName, witness.reviewer_company)
    if (exactAccountReviewPath) {
      setMatchedAccountReviewPath(exactAccountReviewPath)
      return
    }
    if (backToPath?.startsWith('/watchlists')) {
      setMatchedAccountReviewPath(null)
      return
    }
    let cancelled = false
    fetchAccountsInMotionFeed({
      vendor_name: vendorName,
      include_stale: true,
    })
      .then((res) => {
        if (cancelled) return
        const reviewerCompany = witness.reviewer_company?.trim().toLowerCase()
        const match = (res.accounts || []).find((row) => row.company?.trim().toLowerCase() === reviewerCompany)
        setMatchedAccountReviewPath(match ? accountReviewPath(match, backToPath) : null)
      })
      .catch(() => {
        if (cancelled) return
        setMatchedAccountReviewPath(null)
      })
    return () => {
      cancelled = true
    }
  }, [backToPath, open, vendorName, witness?.reviewer_company])

  async function handleAnnotate(type: 'pin' | 'flag' | 'suppress') {
    if (!witnessId || !vendorName) return
    setAnnotating(true)
    setAnnotationActionError(null)
    try {
      const result = await setAnnotation({
        witness_id: witnessId,
        vendor_name: vendorName,
        annotation_type: type,
      })
      setAnnotationState(result)
    } catch (err) {
      setAnnotationActionError(err instanceof Error ? err.message : 'Failed to update annotation')
    } finally {
      setAnnotating(false)
    }
  }

  function requestRemoveAnnotation() {
    setAnnotationActionError(null)
    setPendingRemoveAnnotation(true)
    setRemoveAnnotationError(null)
  }

  async function handleRemoveAnnotation() {
    if (!witnessId) return
    setAnnotating(true)
    setRemoveAnnotationError(null)
    try {
      await removeAnnotations({ witness_ids: [witnessId] })
      setAnnotationState(null)
      setAnnotationActionError(null)
      setPendingRemoveAnnotation(false)
    } catch (err) {
      setRemoveAnnotationError(err instanceof Error ? err.message : 'Failed to remove annotation')
    } finally {
      setAnnotating(false)
    }
  }

  async function handleCopyLink(copyKey: string, target: string) {
    if (!navigator.clipboard?.writeText) {
      setCopiedLinkKey(null)
      setLinkCopyError('Copy is unavailable in this browser')
      return
    }
    try {
      await navigator.clipboard.writeText(toAbsoluteUrl(target))
      setCopiedLinkKey(copyKey)
      setLinkCopyError(null)
    } catch (err) {
      setCopiedLinkKey(null)
      setLinkCopyError(err instanceof Error ? err.message : 'Failed to copy link')
    }
  }

  const handleKeyDown = useCallback((e: KeyboardEvent) => {
    if (e.key === 'Escape') onClose()
  }, [onClose])

  useEffect(() => {
    if (open) document.addEventListener('keydown', handleKeyDown)
    return () => document.removeEventListener('keydown', handleKeyDown)
  }, [open, handleKeyDown])

  if (!open) return null

  const libraryPath = vendorName
    ? upstreamNestedPath(backToPath, '/reports') ?? (() => {
        const params = new URLSearchParams()
        params.set('vendor_filter', vendorName)
        if (backToPath) params.set('back_to', backToPath)
        return `/reports?${params.toString()}`
      })()
    : null
  const libraryTargetsReportDetail = Boolean(libraryPath?.startsWith('/reports/'))
  const libraryLinkText = libraryTargetsReportDetail ? 'Open report detail' : 'View library'
  const libraryCopyLabel = libraryTargetsReportDetail ? 'Copy report' : 'Copy library'
  const libraryCopyAriaLabel = libraryTargetsReportDetail ? 'Copy report detail link' : 'Copy report library link'
  const watchlistsWorkspacePath = vendorName
    ? upstreamNestedPath(backToPath, '/watchlists') ?? watchlistsPath(vendorName, backToPath)
    : null
  const visibleWatchlistsPath = watchlistsWorkspacePath && watchlistsWorkspacePath !== matchedAccountReviewPath
    ? watchlistsWorkspacePath
    : null
  const alertsWorkspacePath = vendorName
    ? upstreamNestedPath(backToPath, '/alerts') ?? alertsPath(vendorName, backToPath, witness?.reviewer_company)
    : null
  const opportunitiesWorkspacePath = vendorName
    ? upstreamNestedPath(backToPath, '/opportunities') ?? opportunitiesPath(vendorName, backToPath)
    : null
  const vendorPath = vendorName
    ? upstreamNestedPath(backToPath, '/vendors/') ?? vendorWorkspacePath(vendorName, backToPath)
    : null
  const reviewPath = witness?.review_id
    ? upstreamReviewPath(backToPath, witness.review_id) ?? reviewDetailPath(witness.review_id, backToPath)
    : null

  return (
    <div className="fixed inset-0 z-50 flex justify-end">
      {/* Backdrop */}
      <div className="absolute inset-0 bg-black/60 backdrop-blur-sm" onClick={onClose} />

      {/* Panel */}
      <div
        ref={panelRef}
        className="relative w-full max-w-2xl bg-slate-900 border-l border-slate-700/50 shadow-2xl overflow-y-auto"
        style={{ animation: 'slideInRight 0.2s ease-out' }}
      >
        {/* Header */}
        <div className="sticky top-0 z-10 bg-slate-900/95 backdrop-blur-sm border-b border-slate-700/50 px-6 py-4">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <Fingerprint className="w-5 h-5 text-cyan-400" />
              <h2 className="text-lg font-semibold text-white">Witness Detail</h2>
              {annotation && (
                <span className={clsx(
                  'inline-flex items-center gap-1 px-2 py-0.5 rounded text-xs font-medium',
                  annotation.annotation_type === 'pin' && 'bg-amber-500/20 text-amber-300',
                  annotation.annotation_type === 'flag' && 'bg-red-500/20 text-red-300',
                  annotation.annotation_type === 'suppress' && 'bg-slate-500/20 text-slate-400',
                )}>
                  {annotation.annotation_type === 'pin' && <Pin className="h-3 w-3" />}
                  {annotation.annotation_type === 'flag' && <Flag className="h-3 w-3" />}
                  {annotation.annotation_type === 'suppress' && <EyeOff className="h-3 w-3" />}
                  {annotation.annotation_type}
                </span>
              )}
            </div>
            <button onClick={onClose} className="p-1 rounded hover:bg-slate-700/50 text-slate-400 hover:text-white transition-colors">
              <X className="w-5 h-5" />
            </button>
          </div>
          {witness && (
            <div className="mt-3 flex items-center gap-2">
              {explorerUrl && (
                <>
                  <a
                    href={explorerUrl}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-cyan-500/10 text-cyan-300 hover:bg-cyan-500/20 transition-colors"
                  >
                    <Fingerprint className="h-3 w-3" />
                    Open in Evidence Explorer
                  </a>
                  <button
                    type="button"
                    onClick={() => void handleCopyLink('explorer', explorerUrl)}
                    aria-label="Copy evidence explorer link"
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-slate-700/50 text-slate-300 hover:bg-slate-600/50 transition-colors"
                  >
                    <Copy className="h-3 w-3" />
                    {copiedLinkKey === 'explorer' ? 'Copied' : 'Copy explorer'}
                  </button>
                </>
              )}
              {matchedAccountReviewPath ? (
                <>
                  <Link
                    to={matchedAccountReviewPath}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-emerald-500/10 text-emerald-300 hover:bg-emerald-500/20 transition-colors"
                  >
                    <Building2 className="h-3 w-3" />
                    Open account review
                  </Link>
                  <button
                    type="button"
                    onClick={() => void handleCopyLink('account-review', matchedAccountReviewPath)}
                    aria-label="Copy account review link"
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-slate-700/50 text-slate-300 hover:bg-slate-600/50 transition-colors"
                  >
                    <Copy className="h-3 w-3" />
                    {copiedLinkKey === 'account-review' ? 'Copied' : 'Copy account review'}
                  </button>
                </>
              ) : null}
              {visibleWatchlistsPath ? (
                <>
                  <Link
                    to={visibleWatchlistsPath}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-indigo-500/10 text-indigo-300 hover:bg-indigo-500/20 transition-colors"
                  >
                    <Building2 className="h-3 w-3" />
                    Watchlists
                  </Link>
                  <button
                    type="button"
                    onClick={() => void handleCopyLink('watchlists', visibleWatchlistsPath)}
                    aria-label="Copy watchlists link"
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-slate-700/50 text-slate-300 hover:bg-slate-600/50 transition-colors"
                  >
                    <Copy className="h-3 w-3" />
                    {copiedLinkKey === 'watchlists' ? 'Copied' : 'Copy watchlists'}
                  </button>
                </>
              ) : null}
              {alertsWorkspacePath ? (
                <>
                  <Link
                    to={alertsWorkspacePath}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-violet-500/10 text-violet-300 hover:bg-violet-500/20 transition-colors"
                  >
                    <ExternalLink className="h-3 w-3" />
                    Alerts API
                  </Link>
                  <button
                    type="button"
                    onClick={() => void handleCopyLink('alerts', alertsWorkspacePath)}
                    aria-label="Copy alerts link"
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-slate-700/50 text-slate-300 hover:bg-slate-600/50 transition-colors"
                  >
                    <Copy className="h-3 w-3" />
                    {copiedLinkKey === 'alerts' ? 'Copied' : 'Copy alerts'}
                  </button>
                </>
              ) : null}
              {opportunitiesWorkspacePath ? (
                <>
                  <Link
                    to={opportunitiesWorkspacePath}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-amber-500/10 text-amber-300 hover:bg-amber-500/20 transition-colors"
                  >
                    <ChevronRight className="h-3 w-3" />
                    Opportunities
                  </Link>
                  <button
                    type="button"
                    onClick={() => void handleCopyLink('opportunities', opportunitiesWorkspacePath)}
                    aria-label="Copy opportunities link"
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-slate-700/50 text-slate-300 hover:bg-slate-600/50 transition-colors"
                  >
                    <Copy className="h-3 w-3" />
                    {copiedLinkKey === 'opportunities' ? 'Copied' : 'Copy opportunities'}
                  </button>
                </>
              ) : null}
              {vendorPath ? (
                <>
                  <Link
                    to={vendorPath}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-sky-500/10 text-sky-300 hover:bg-sky-500/20 transition-colors"
                  >
                    <Building2 className="h-3 w-3" />
                    Vendor workspace
                  </Link>
                  <button
                    type="button"
                    onClick={() => void handleCopyLink('vendor', vendorPath)}
                    aria-label="Copy vendor workspace link"
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-slate-700/50 text-slate-300 hover:bg-slate-600/50 transition-colors"
                  >
                    <Copy className="h-3 w-3" />
                    {copiedLinkKey === 'vendor' ? 'Copied' : 'Copy vendor'}
                  </button>
                </>
              ) : null}
              {annotation ? (
                <button
                  onClick={requestRemoveAnnotation}
                  disabled={annotating}
                  className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-slate-700/50 text-slate-300 hover:bg-slate-600/50 disabled:opacity-50 transition-colors"
                >
                  <RotateCcw className="h-3 w-3" />
                  Remove {annotation.annotation_type}
                </button>
              ) : (
                <>
                  <button
                    onClick={() => handleAnnotate('pin')}
                    disabled={annotating}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-amber-500/10 text-amber-300 hover:bg-amber-500/20 disabled:opacity-50 transition-colors"
                    title="Pin: prioritize in campaign generation"
                  >
                    <Pin className="h-3 w-3" />
                    Pin
                  </button>
                  <button
                    onClick={() => handleAnnotate('flag')}
                    disabled={annotating}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-red-500/10 text-red-300 hover:bg-red-500/20 disabled:opacity-50 transition-colors"
                    title="Flag: mark for quality review"
                  >
                    <Flag className="h-3 w-3" />
                    Flag
                  </button>
                  <button
                    onClick={() => handleAnnotate('suppress')}
                    disabled={annotating}
                    className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-md text-xs font-medium bg-slate-500/10 text-slate-400 hover:bg-slate-500/20 disabled:opacity-50 transition-colors"
                    title="Suppress: exclude from campaign generation"
                  >
                    <EyeOff className="h-3 w-3" />
                    Suppress
                  </button>
                </>
              )}
            </div>
          )}
          {annotationActionError ? (
            <div role="alert" className="mt-3 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
              {annotationActionError}
            </div>
          ) : null}
          {linkCopyError ? (
            <div role="alert" className="mt-3 rounded-lg border border-rose-500/30 bg-rose-500/10 px-3 py-2 text-sm text-rose-200">
              {linkCopyError}
            </div>
          ) : null}
        </div>

        {loading && (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-6 h-6 text-cyan-400 animate-spin" />
          </div>
        )}

        {error && (
          <div className="mx-6 mt-6 p-4 bg-red-900/20 border border-red-800/50 rounded-lg text-red-300 text-sm">{error}</div>
        )}

        {witness && !loading && (
          <div className="p-6 space-y-6">
            {/* Excerpt card */}
            <div className="bg-slate-800/60 rounded-xl border border-cyan-500/20 p-5">
              <div className="flex items-start gap-3 mb-3">
                <Quote className="w-4 h-4 text-cyan-400 mt-1 shrink-0" />
                <p className="text-slate-200 text-sm leading-relaxed italic">
                  &ldquo;{witness.excerpt_text}&rdquo;
                </p>
              </div>
              <div className="flex items-center gap-3 mt-3 flex-wrap">
                <SourceBadge source={witness.source || 'unknown'} />
                {witness.witness_type && (
                  <span className="text-xs px-2 py-0.5 rounded bg-cyan-900/30 text-cyan-300 border border-cyan-800/40 font-medium">
                    {witness.witness_type.replace(/_/g, ' ')}
                  </span>
                )}
                {witness.pain_category && (
                  <span className="text-xs px-2 py-0.5 rounded bg-red-900/30 text-red-300 border border-red-800/40">
                    {witness.pain_category}
                  </span>
                )}
                {witness.competitor && (
                  <span className="text-xs px-2 py-0.5 rounded bg-violet-900/30 text-violet-300 border border-violet-800/40">
                    vs {witness.competitor}
                  </span>
                )}
                {witness.salience_score != null && (
                  <span className="text-xs text-slate-500 font-mono">
                    salience: {witness.salience_score.toFixed(2)}
                  </span>
                )}
              </div>
            </div>

            {/* Reviewer context */}
            <div className="grid grid-cols-2 gap-3">
              {witness.reviewer_company && (
                <div className="bg-slate-800/40 rounded-lg p-3 border border-slate-700/30">
                  <div className="flex items-center gap-2 text-slate-500 text-xs mb-1">
                    <Building2 className="w-3 h-3" /> Company
                  </div>
                  <span className="text-sm text-slate-200">{witness.reviewer_company}</span>
                </div>
              )}
              {witness.reviewer_title && (
                <div className="bg-slate-800/40 rounded-lg p-3 border border-slate-700/30">
                  <div className="flex items-center gap-2 text-slate-500 text-xs mb-1">
                    <User className="w-3 h-3" /> Title
                  </div>
                  <span className="text-sm text-slate-200">{witness.reviewer_title}</span>
                </div>
              )}
              {witness.reviewed_at && (
                <div className="bg-slate-800/40 rounded-lg p-3 border border-slate-700/30">
                  <div className="flex items-center gap-2 text-slate-500 text-xs mb-1">
                    <Calendar className="w-3 h-3" /> Reviewed
                  </div>
                  <span className="text-sm text-slate-200">
                    {new Date(witness.reviewed_at).toLocaleDateString()}
                  </span>
                </div>
              )}
              {witness.rating != null && (
                <div className="bg-slate-800/40 rounded-lg p-3 border border-slate-700/30">
                  <div className="flex items-center gap-2 text-slate-500 text-xs mb-1">
                    <Star className="w-3 h-3" /> Rating
                  </div>
                  <span className="text-sm text-slate-200">{witness.rating}/5</span>
                </div>
              )}
            </div>

            {/* Signal tags */}
            {Array.isArray(witness.signal_tags) && witness.signal_tags.length > 0 && (
              <div>
                <h3 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                  <Tag className="w-3 h-3" /> Signal Tags
                </h3>
                <div className="flex flex-wrap gap-1.5">
                  {witness.signal_tags.map((tag, i) => (
                    <span key={i} className={clsx(
                      'text-xs px-2 py-0.5 rounded',
                      SIGNAL_COLORS[tag] || 'bg-slate-700/50 text-slate-300',
                    )}>
                      {tag.replace(/_/g, ' ')}
                    </span>
                  ))}
                </div>
              </div>
            )}

            {/* Witness metadata */}
            <div className="bg-slate-800/30 rounded-lg p-3 border border-slate-700/20">
              <h3 className="text-xs font-medium text-slate-500 uppercase tracking-wider mb-2 flex items-center gap-2">
                <Fingerprint className="w-3 h-3" /> Witness Metadata
              </h3>
              <div className="grid grid-cols-2 gap-x-4 gap-y-1 text-xs">
                <span className="text-slate-500">ID</span>
                <span className="text-slate-300 font-mono truncate">{witness.witness_id}</span>
                <span className="text-slate-500">Type</span>
                <span className="text-slate-300">{witness.witness_type || 'unknown'}</span>
                {witness.specificity_score != null && (
                  <>
                    <span className="text-slate-500">Specificity</span>
                    <span className="text-slate-300">{witness.specificity_score.toFixed(2)}</span>
                  </>
                )}
                {witness.selection_reason && (
                  <>
                    <span className="text-slate-500">Selection</span>
                    <span className="text-slate-300">{witness.selection_reason}</span>
                  </>
                )}
              </div>
            </div>

            {/* Full review text with highlight */}
            {witness.review_text && (
              <div>
                <h3 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                  <FileText className="w-3 h-3" /> Source Review
                  {libraryPath ? (
                    <>
                      <a
                        href={libraryPath}
                        className="inline-flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300 mt-1"
                      >
                        {libraryLinkText} <ExternalLink className="h-3 w-3" />
                      </a>
                      <button
                        type="button"
                        onClick={() => void handleCopyLink('library', libraryPath)}
                        aria-label={libraryCopyAriaLabel}
                        className="inline-flex items-center gap-1 text-xs text-slate-300 hover:text-slate-200 mt-1"
                      >
                        <Copy className="h-3 w-3" />
                        {copiedLinkKey === 'library' ? 'Copied' : libraryCopyLabel}
                      </button>
                    </>
                  ) : null}
                  {reviewPath ? (
                    <>
                      <Link
                        to={reviewPath}
                        className="inline-flex items-center gap-1 text-xs text-cyan-400 hover:text-cyan-300 mt-1"
                      >
                        Open review detail <ExternalLink className="h-3 w-3" />
                      </Link>
                      <button
                        type="button"
                        onClick={() => void handleCopyLink('review-detail', reviewPath)}
                        aria-label="Copy review detail link"
                        className="inline-flex items-center gap-1 text-xs text-slate-300 hover:text-slate-200 mt-1"
                      >
                        <Copy className="h-3 w-3" />
                        {copiedLinkKey === 'review-detail' ? 'Copied' : 'Copy review'}
                      </button>
                    </>
                  ) : null}
                  {witness.source_url && (
                    <a href={witness.source_url} target="_blank" rel="noopener noreferrer"
                       className="ml-auto text-cyan-400 hover:text-cyan-300 flex items-center gap-1">
                      <ExternalLink className="w-3 h-3" /> View original
                    </a>
                  )}
                </h3>
                {witness.highlight_source === 'match_summary' && (
                  <div className="mb-2 text-xs text-amber-400/80">
                    Excerpt is drawn from the review title or summary, not the body shown below.
                  </div>
                )}
                {witness.highlight_source === 'inferred' && (
                  <div className="mb-2 text-xs text-amber-400/80">
                    Excerpt was synthesized from the review content and is not a verbatim quote.
                  </div>
                )}
                <div className="bg-slate-950/60 rounded-lg p-4 border border-slate-700/30 text-sm text-slate-300 leading-relaxed max-h-80 overflow-y-auto">
                  {highlightExcerpt(witness.review_text, witness.excerpt_text, witness.highlight_start, witness.highlight_end)}
                </div>
              </div>
            )}

            {/* Evidence spans from this review */}
            {witness.evidence_spans && witness.evidence_spans.length > 0 && (
              <div>
                <h3 className="text-xs font-medium text-slate-400 uppercase tracking-wider mb-2 flex items-center gap-2">
                  <ChevronRight className="w-3 h-3" />
                  Related Evidence Spans ({witness.evidence_spans.length} of {witness.all_evidence_span_count})
                </h3>
                <div className="space-y-2">
                  {witness.evidence_spans.map((span, i) => (
                    <div key={i} className="bg-slate-800/40 rounded-lg p-3 border border-slate-700/20">
                      <div className="flex items-center gap-2 mb-1">
                        <span className={clsx(
                          'text-xs px-1.5 py-0.5 rounded',
                          SIGNAL_COLORS[span.signal_type] || 'bg-slate-700/50 text-slate-300',
                        )}>
                          {span.signal_type?.replace(/_/g, ' ')}
                        </span>
                        {span.pain_category && (
                          <span className="text-xs text-slate-500">{span.pain_category}</span>
                        )}
                      </div>
                      <p className="text-xs text-slate-400 italic">
                        &ldquo;{span.text}&rdquo;
                      </p>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
      {annotation && pendingRemoveAnnotation ? (
        <AnnotationRemovalModal
          annotationType={annotation.annotation_type}
          confirming={annotating}
          error={removeAnnotationError}
          onCancel={() => {
            if (!annotating) {
              setPendingRemoveAnnotation(false)
              setRemoveAnnotationError(null)
            }
          }}
          onConfirm={() => void handleRemoveAnnotation()}
        />
      ) : null}
    </div>
  )
}

export { SourceBadge, SOURCE_COLORS, SIGNAL_COLORS }
