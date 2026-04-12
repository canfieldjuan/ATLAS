import { useNavigate, Link, useLocation, useSearchParams } from 'react-router-dom'
import { FileBarChart, RefreshCw, Search, X, Loader2, ChevronLeft, ChevronRight, Bell, Pencil, Copy, Check } from 'lucide-react'
import { clsx } from 'clsx'
import { PageError } from '../components/ErrorBoundary'
import UpgradeGate from '../components/UpgradeGate'
import ReportTrustPanel from '../components/ReportTrustPanel'
import useApiData from '../hooks/useApiData'
import { usePlanGate } from '../hooks/usePlanGate'
import {
  buildReportLibraryViewScopeKey,
  downloadReportPdf,
  fetchReport,
  fetchReports,
  generateAccountComparisonReport,
  generateAccountDeepDiveReport,
  generateVendorComparisonReport,
  listReportSubscriptions,
  requestBattleCardReport,
  type ReportLibraryViewFilters,
  type ReportSubscription,
  type ReportSubscriptionScopeType,
} from '../api/client'
import { useState, useEffect, useMemo, useCallback } from 'react'
import { REPORT_TYPE_COLORS } from '../lib/reportConstants'
import SubscriptionModal from '../components/SubscriptionModal'
import type { Report } from '../types'

function CardSkeleton() {
  return (
    <div className="bg-slate-900/50 border border-slate-700/50 rounded-xl p-5 animate-pulse">
      <div className="flex items-start justify-between mb-3">
        <div className="h-5 w-28 bg-slate-700/50 rounded" />
        <div className="h-4 w-4 bg-slate-700/50 rounded" />
      </div>
      <div className="h-4 w-36 bg-slate-700/50 rounded mb-2" />
      <div className="h-3 w-full bg-slate-700/50 rounded mb-1" />
      <div className="h-3 w-3/4 bg-slate-700/50 rounded" />
      <div className="h-3 w-20 bg-slate-700/50 rounded mt-4" />
    </div>
  )
}

function toIso(value: string | null | undefined): string | null {
  if (!value) return null
  const date = new Date(value)
  return Number.isNaN(date.getTime()) ? null : date.toISOString()
}

function titleizeFilterValue(value: string) {
  return value.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase())
}

function reportFreshnessState(report: Report): string {
  return report.freshness_state || report.trust?.freshness_state || 'unknown'
}

function reportReviewState(report: Report): string {
  return report.review_state || report.trust?.review_state || 'clean'
}

function reportTitle(report: Report): string {
  if (['vendor_comparison', 'account_comparison'].includes(report.report_type) && report.vendor_filter && report.category_filter) {
    return `${report.vendor_filter} vs ${report.category_filter}`
  }
  if (report.report_type === 'challenger_brief' && report.vendor_filter && report.category_filter) {
    return `${report.vendor_filter} → ${report.category_filter}`
  }
  return report.vendor_filter ?? report.report_type.replace(/_/g, ' ')
}

function reportSubscriptionLabel(report: Report): string {
  return `${report.report_type.replace(/_/g, ' ')} - ${report.vendor_filter ?? 'all'}`
}

function reportExportPresentation(report: Report) {
  if (report.has_pdf_export) {
    return {
      label: 'PDF ready',
      className: 'bg-emerald-500/10 text-emerald-300 border border-emerald-500/20',
    }
  }

  const artifactState = report.artifact_state || report.trust?.artifact_state
  if (artifactState === 'processing') {
    return {
      label: 'Generating PDF',
      className: 'bg-amber-500/10 text-amber-300 border border-amber-500/20',
    }
  }

  if (artifactState === 'failed') {
    return {
      label: 'PDF failed',
      className: 'bg-rose-500/10 text-rose-300 border border-rose-500/20',
    }
  }

  return {
    label: 'PDF unavailable',
    className: 'bg-slate-800/70 text-slate-400 border border-slate-700/60',
  }
}

function reportSubscriptionActionPresentation(reportSubscription: NonNullable<Report['report_subscription']> | null) {
  if (!reportSubscription) {
    return {
      label: 'Subscribe',
      className: 'bg-slate-800 border border-slate-700 text-slate-300 hover:text-white hover:bg-slate-700',
    }
  }

  if (reportSubscription.enabled) {
    return {
      label: 'Manage Subscription',
      className: 'bg-cyan-900/20 border border-cyan-700/40 text-cyan-200 hover:bg-cyan-900/35',
    }
  }

  return {
    label: 'Resume Subscription',
    className: 'bg-amber-500/10 border border-amber-500/25 text-amber-300 hover:bg-amber-500/20',
  }
}

function describeSubscriptionFilters(filterPayload?: ReportLibraryViewFilters | null) {
  const normalized = {
    report_type: (filterPayload?.report_type || '').trim(),
    vendor_filter: (filterPayload?.vendor_filter || '').trim(),
    quality_status: (filterPayload?.quality_status || '').trim(),
    freshness_state: (filterPayload?.freshness_state || '').trim(),
    review_state: (filterPayload?.review_state || '').trim(),
  }
  const lines: string[] = []
  if (normalized.report_type) lines.push(`Type: ${titleizeFilterValue(normalized.report_type)}`)
  if (normalized.vendor_filter) lines.push(`Vendor: ${normalized.vendor_filter}`)
  if (normalized.quality_status) lines.push(`Quality: ${titleizeFilterValue(normalized.quality_status)}`)
  if (normalized.freshness_state) lines.push(`Freshness: ${titleizeFilterValue(normalized.freshness_state)}`)
  if (normalized.review_state) lines.push(`Review: ${titleizeFilterValue(normalized.review_state)}`)
  return lines
}

function reportDetailLocation(reportId: string, backTo: string) {
  const next = new URLSearchParams()
  if (backTo !== '/reports') {
    next.set('back_to', backTo)
  }

  return {
    pathname: `/reports/${reportId}`,
    search: next.toString() ? `?${next.toString()}` : '',
  }
}

function reportExternalBackTarget(searchParams: URLSearchParams) {
  const value = searchParams.get('back_to')
  if (!value) return null
  if (
    value.startsWith('/vendors/')
    || value.startsWith('/watchlists')
    || value.startsWith('/reviews')
    || value.startsWith('/evidence')
    || value.startsWith('/opportunities')
  ) return value
  return null
}

const REPORT_BACK_TARGET_KEYS = [
  'report_type',
  'vendor_filter',
  'quality_status',
  'freshness_state',
  'review_state',
  'composer',
  'primary_vendor',
  'comparison_vendor',
  'primary_company',
  'comparison_company',
  'deep_dive_company',
  'battle_card_vendor',
  'tab',
  'subscription_id',
] as const

function buildReportsBackTarget(searchParams: URLSearchParams) {
  const externalBackTarget = reportExternalBackTarget(searchParams)
  if (externalBackTarget) return externalBackTarget

  const next = new URLSearchParams()
  for (const key of REPORT_BACK_TARGET_KEYS) {
    const value = searchParams.get(key)
    if (value) next.set(key, value)
  }
  const qs = next.toString()
  return qs ? `/reports?${qs}` : '/reports'
}

function buildReportSubscriptionLocation(searchParams: URLSearchParams, report: Report) {
  const backTarget = buildReportsBackTarget(searchParams)
  const next = new URLSearchParams(
    backTarget.startsWith('/reports?') ? backTarget.slice('/reports?'.length) : '',
  )
  if (!backTarget.startsWith('/reports')) {
    next.set('back_to', backTarget)
  }
  next.set('report_subscription', report.id)
  next.set('report_focus_type', report.report_type)
  if (report.vendor_filter) next.set('report_focus_vendor', report.vendor_filter)
  else next.delete('report_focus_vendor')
  next.set('report_focus_label', report.report_subscription?.scope_label ?? reportSubscriptionLabel(report))
  const qs = next.toString()
  return qs ? `/reports?${qs}` : '/reports'
}

function buildSubscriptionManageLocation(subscriptionId: string) {
  const next = new URLSearchParams()
  next.set('tab', 'subscriptions')
  next.set('subscription_id', subscriptionId)
  return `/reports?${next.toString()}`
}

function vendorDetailPath(vendorName: string, backTo?: string) {
  const base = `/vendors/${encodeURIComponent(vendorName)}`
  if (!backTo) return base
  const next = new URLSearchParams()
  next.set('back_to', backTo)
  return `${base}?${next.toString()}`
}

function evidencePath(vendorName: string, backTo: string) {
  const next = new URLSearchParams()
  next.set('vendor', vendorName)
  next.set('tab', 'witnesses')
  next.set('back_to', backTo)
  return `/evidence?${next.toString()}`
}

function opportunitiesPath(vendorName: string, backTo: string) {
  const next = new URLSearchParams()
  next.set('vendor', vendorName)
  next.set('back_to', backTo)
  return `/opportunities?${next.toString()}`
}

type ReportComposer =
  | 'vendor_comparison'
  | 'account_deep_dive'
  | 'account_comparison'
  | 'battle_card'

const REPORT_COMPOSERS = new Set<ReportComposer>([
  'vendor_comparison',
  'account_deep_dive',
  'account_comparison',
  'battle_card',
])

export default function Reports() {
  const navigate = useNavigate()
  const location = useLocation()
  const [searchParams, setSearchParams] = useSearchParams()
  const { canAccessReports } = usePlanGate()
  const [typeFilter, setTypeFilter] = useState(() => searchParams.get('report_type') ?? '')
  const [qualityFilter, setQualityFilter] = useState(() => searchParams.get('quality_status') ?? '')
  const [freshnessFilter, setFreshnessFilter] = useState(() => searchParams.get('freshness_state') ?? '')
  const [reviewFilter, setReviewFilter] = useState(() => searchParams.get('review_state') ?? '')
  const [vendorSearch, setVendorSearch] = useState(() => searchParams.get('vendor_filter') ?? '')
  const [debouncedVendor, setDebouncedVendor] = useState('')
  const [primaryVendor, setPrimaryVendor] = useState('')
  const [comparisonVendor, setComparisonVendor] = useState('')
  const [primaryCompany, setPrimaryCompany] = useState('')
  const [comparisonCompany, setComparisonCompany] = useState('')
  const [deepDiveCompany, setDeepDiveCompany] = useState('')
  const [creatingComparison, setCreatingComparison] = useState(false)
  const [creatingAccountComparison, setCreatingAccountComparison] = useState(false)
  const [creatingAccountDeepDive, setCreatingAccountDeepDive] = useState(false)
  const [battleCardVendor, setBattleCardVendor] = useState('')
  const [creatingBattleCard, setCreatingBattleCard] = useState(false)
  const [libSubOpen, setLibSubOpen] = useState(false)
  const [activeComposer, setActiveComposer] = useState<ReportComposer | null>(() => {
    const composer = searchParams.get('composer')
    return composer && REPORT_COMPOSERS.has(composer as ReportComposer)
      ? composer as ReportComposer
      : null
  })
  const [activeTab, setActiveTab] = useState<'library' | 'subscriptions'>(
    () => searchParams.get('tab') === 'subscriptions' ? 'subscriptions' : 'library',
  )
  const [subscriptions, setSubscriptions] = useState<ReportSubscription[]>([])
  const [subLoading, setSubLoading] = useState(false)
  const [editingSub, setEditingSub] = useState<ReportSubscription | null>(null)
  const [cardSubTarget, setCardSubTarget] = useState<{ scopeType: 'report'; scopeKey: string; scopeLabel: string } | null>(null)
  const [savedReportSubscriptions, setSavedReportSubscriptions] = useState<Record<string, NonNullable<Report['report_subscription']>>>({})
  const [requestedReportFallback, setRequestedReportFallback] = useState<Report | null>(null)
  const [copiedReportLinkId, setCopiedReportLinkId] = useState<string | null>(null)
  const [copiedSubscriptionLinkId, setCopiedSubscriptionLinkId] = useState<string | null>(null)
  const activeVendorFilter = debouncedVendor.trim()
  const requestedReportSubscriptionId = searchParams.get('report_subscription') ?? ''
  const requestedReportFocusType = searchParams.get('report_focus_type') ?? ''
  const requestedReportFocusVendor = searchParams.get('report_focus_vendor') ?? ''
  const requestedReportFocusLabel = searchParams.get('report_focus_label') ?? ''
  const effectiveTypeFilter = typeFilter || (requestedReportSubscriptionId ? requestedReportFocusType : '')
  const effectiveVendorFilter = activeVendorFilter || (requestedReportSubscriptionId ? requestedReportFocusVendor : '')
  const reportsBackTarget = useMemo(() => buildReportsBackTarget(searchParams), [searchParams])
  const currentLibraryPath = useMemo(
    () => `${location.pathname}${location.search}`,
    [location.pathname, location.search],
  )
  const backButtonLabel = reportsBackTarget.startsWith('/vendors/')
    ? 'Back to Vendor'
    : reportsBackTarget.startsWith('/watchlists')
      ? 'Back to Watchlists'
      : reportsBackTarget.startsWith('/reviews')
        ? 'Back to Review'
        : reportsBackTarget.startsWith('/evidence')
          ? 'Back to Evidence'
          : reportsBackTarget.startsWith('/opportunities')
            ? 'Back to Opportunities'
      : 'Back to Library'

  useEffect(() => {
    const composer = searchParams.get('composer')
    if (composer && REPORT_COMPOSERS.has(composer as ReportComposer)) {
      setActiveComposer(composer as ReportComposer)
    }
    setPrimaryVendor(searchParams.get('primary_vendor') ?? '')
    setComparisonVendor(searchParams.get('comparison_vendor') ?? '')
    setPrimaryCompany(searchParams.get('primary_company') ?? '')
    setComparisonCompany(searchParams.get('comparison_company') ?? '')
    setDeepDiveCompany(searchParams.get('deep_dive_company') ?? '')
    setBattleCardVendor(searchParams.get('battle_card_vendor') ?? '')
  }, [searchParams])

  useEffect(() => {
    const timer = setTimeout(() => setDebouncedVendor(vendorSearch), 300)
    return () => clearTimeout(timer)
  }, [vendorSearch])

  const updateReportsQuery = useCallback((updates: Record<string, string | null | undefined>) => {
    const next = new URLSearchParams(searchParams)
    Object.entries(updates).forEach(([key, value]) => {
      const trimmed = typeof value === 'string' ? value.trim() : value
      if (trimmed) next.set(key, trimmed)
      else next.delete(key)
    })
    setSearchParams(next, { replace: true })
  }, [searchParams, setSearchParams])

  const updateComposerParams = useCallback((composer: ReportComposer | null, updates: Record<string, string | null | undefined>) => {
    updateReportsQuery({
      composer: composer ?? undefined,
      ...updates,
    })
  }, [updateReportsQuery])

  const updateSearchFilterParams = useCallback((nextFilters: {
    report_type?: string
    vendor_filter?: string
    quality_status?: string
    freshness_state?: string
    review_state?: string
  }) => {
    updateReportsQuery({
      report_type: nextFilters.report_type ?? typeFilter,
      vendor_filter: nextFilters.vendor_filter ?? vendorSearch,
      quality_status: nextFilters.quality_status ?? qualityFilter,
      freshness_state: nextFilters.freshness_state ?? freshnessFilter,
      review_state: nextFilters.review_state ?? reviewFilter,
    })
  }, [
    typeFilter,
    vendorSearch,
    qualityFilter,
    freshnessFilter,
    reviewFilter,
    updateReportsQuery,
  ])

  const { data, loading, error, refresh, refreshing } = useApiData(
    () => fetchReports({
      report_type: effectiveTypeFilter || undefined,
      vendor_filter: effectiveVendorFilter || undefined,
      quality_status: qualityFilter || undefined,
      freshness_state: freshnessFilter || undefined,
      review_state: reviewFilter || undefined,
      include_stale: freshnessFilter === 'stale',
      limit: 100,
    }),
    [effectiveTypeFilter, effectiveVendorFilter, qualityFilter, freshnessFilter, reviewFilter],
  )

  const PAGE_SIZES = [25, 50, 100] as const
  const [page, setPage] = useState(0)
  const [perPage, setPerPage] = useState<number>(PAGE_SIZES[0])

  const reports = useMemo(() => data?.reports ?? [], [data])
  const filteredReports = useMemo(
    () => reports.filter((report) => {
      if (qualityFilter) {
        if (report.report_type !== 'battle_card') return false
        if ((report.quality_status || '').toLowerCase() !== qualityFilter) return false
      }
      if (freshnessFilter && reportFreshnessState(report) !== freshnessFilter) return false
      if (reviewFilter && reportReviewState(report) !== reviewFilter) return false
      return true
    }),
    [reports, qualityFilter, freshnessFilter, reviewFilter],
  )

  // Reset page when data or filters change
  const reportCount = filteredReports.length
  useEffect(() => { setPage(0) }, [reportCount, typeFilter, activeVendorFilter, qualityFilter, freshnessFilter, reviewFilter])

  const totalPages = Math.max(1, Math.ceil(filteredReports.length / perPage))
  const safePage = Math.min(page, totalPages - 1)
  const pagedReports = useMemo(
    () => filteredReports.slice(safePage * perPage, (safePage + 1) * perPage),
    [filteredReports, safePage, perPage],
  )
  const showPagination = filteredReports.length > PAGE_SIZES[0]

  const newestReportAt = filteredReports
    .map((report) => toIso(report.created_at ?? report.report_date))
    .filter((value): value is string => Boolean(value))
    .sort((a, b) => b.localeCompare(a))[0] ?? null
  const newestReportAgeHours = newestReportAt
    ? (Date.now() - new Date(newestReportAt).getTime()) / (1000 * 60 * 60)
    : null
  const reportsStale = newestReportAgeHours !== null && newestReportAgeHours > 24
  const hasFilters = typeFilter !== '' || activeVendorFilter !== '' || qualityFilter !== '' || freshnessFilter !== '' || reviewFilter !== ''
  const debouncePending = vendorSearch !== debouncedVendor
  const libraryViewFilters = useMemo<ReportLibraryViewFilters>(
    () => ({
      report_type: typeFilter || undefined,
      vendor_filter: activeVendorFilter || undefined,
      quality_status: qualityFilter || undefined,
      freshness_state: freshnessFilter || undefined,
      review_state: reviewFilter || undefined,
    }),
    [typeFilter, activeVendorFilter, qualityFilter, freshnessFilter, reviewFilter],
  )
  const libraryScopeType: ReportSubscriptionScopeType = hasFilters ? 'library_view' : 'library'
  const libraryScopeKey = hasFilters ? buildReportLibraryViewScopeKey(libraryViewFilters) : 'library'
  const libraryScopeLabel = useMemo(() => {
    const parts: string[] = []
    if (typeFilter) parts.push(titleizeFilterValue(typeFilter))
    if (activeVendorFilter) parts.push(activeVendorFilter)
    if (qualityFilter) parts.push(titleizeFilterValue(qualityFilter))
    if (freshnessFilter) parts.push(titleizeFilterValue(freshnessFilter))
    if (reviewFilter) parts.push(titleizeFilterValue(reviewFilter))
    if (parts.length === 0) return 'Full Intelligence Library'
    return `${parts.join(' • ')} Library`
  }, [typeFilter, activeVendorFilter, qualityFilter, freshnessFilter, reviewFilter])

  const refreshSubs = useCallback(async () => {
    setSubLoading(true)
    try {
      const res = await listReportSubscriptions()
      setSubscriptions(res.subscriptions ?? [])
    } catch (err) {
      console.error('Failed to load report subscriptions', err)
      setSubscriptions([])
    } finally {
      setSubLoading(false)
    }
  }, [])

  useEffect(() => {
    void refreshSubs()
  }, [refreshSubs])

  useEffect(() => {
    const requestedSubscriptionId = searchParams.get('subscription_id') ?? ''
    if (!requestedSubscriptionId || subLoading) return
    const match = subscriptions.find((sub) => sub.id === requestedSubscriptionId)
    if (!match) return
    if (activeTab !== 'subscriptions') setActiveTab('subscriptions')
    if (!editingSub || editingSub.id !== match.id) {
      setEditingSub(match)
    }
  }, [subscriptions, subLoading, searchParams, activeTab, editingSub])

  const handleTabChange = useCallback((tab: 'library' | 'subscriptions') => {
    setActiveTab(tab)
    updateReportsQuery({
      tab: tab === 'subscriptions' ? 'subscriptions' : undefined,
      subscription_id: tab === 'subscriptions' ? searchParams.get('subscription_id') : undefined,
      report_subscription: tab === 'library' ? searchParams.get('report_subscription') : undefined,
      report_focus_type: tab === 'library' ? searchParams.get('report_focus_type') : undefined,
      report_focus_vendor: tab === 'library' ? searchParams.get('report_focus_vendor') : undefined,
      report_focus_label: tab === 'library' ? searchParams.get('report_focus_label') : undefined,
    })
  }, [searchParams, updateReportsQuery])

  const handleEditSubscription = useCallback((subscription: ReportSubscription) => {
    setActiveTab('subscriptions')
    setEditingSub(subscription)
    updateReportsQuery({
      tab: 'subscriptions',
      subscription_id: subscription.id,
    })
  }, [updateReportsQuery])

  const handleCloseEditingSubscription = useCallback(() => {
    setEditingSub(null)
    updateReportsQuery({ subscription_id: undefined })
  }, [updateReportsQuery])

  const resolveReportSubscription = useCallback(
    (report: Report) => savedReportSubscriptions[report.id] ?? report.report_subscription ?? null,
    [savedReportSubscriptions],
  )

  useEffect(() => {
    setRequestedReportFallback(null)
  }, [requestedReportSubscriptionId])

  useEffect(() => {
    if (activeTab !== 'library') return
    if (!requestedReportSubscriptionId) return
    if (reports.some((candidate) => candidate.id === requestedReportSubscriptionId)) return
    if (requestedReportFallback?.id === requestedReportSubscriptionId) return

    let cancelled = false
    void fetchReport(requestedReportSubscriptionId)
      .then((report) => {
        if (cancelled) return
        setRequestedReportFallback(report)
        if (!requestedReportFocusType || !requestedReportFocusVendor || !requestedReportFocusLabel) {
          updateReportsQuery({
            report_focus_type: requestedReportFocusType || report.report_type,
            report_focus_vendor: requestedReportFocusVendor || report.vendor_filter || undefined,
            report_focus_label: requestedReportFocusLabel || report.report_subscription?.scope_label || reportSubscriptionLabel(report),
          })
        }
      })
      .catch((err) => {
        if (!cancelled) {
          console.error('Failed to hydrate requested report subscription target', err)
          setRequestedReportFallback(null)
        }
      })

    return () => {
      cancelled = true
    }
  }, [
    activeTab,
    requestedReportSubscriptionId,
    reports,
    requestedReportFallback,
    requestedReportFocusType,
    requestedReportFocusVendor,
    requestedReportFocusLabel,
    updateReportsQuery,
  ])

  useEffect(() => {
    if (activeTab !== 'library') return
    if (!requestedReportSubscriptionId) return
    const report = reports.find((candidate) => candidate.id === requestedReportSubscriptionId) ?? requestedReportFallback
    if (!report && requestedReportFocusLabel) {
      const nextTarget = {
        scopeType: 'report' as const,
        scopeKey: requestedReportSubscriptionId,
        scopeLabel: requestedReportFocusLabel,
      }
      if (
        cardSubTarget?.scopeKey === nextTarget.scopeKey
        && cardSubTarget.scopeLabel === nextTarget.scopeLabel
      ) {
        return
      }
      setCardSubTarget(nextTarget)
      return
    }
    if (!report) return
    const reportSubscription = resolveReportSubscription(report)
    const nextTarget = {
      scopeType: 'report' as const,
      scopeKey: report.id,
      scopeLabel: requestedReportFocusLabel || reportSubscription?.scope_label || reportSubscriptionLabel(report),
    }
    if (
      cardSubTarget?.scopeKey === nextTarget.scopeKey
      && cardSubTarget.scopeLabel === nextTarget.scopeLabel
    ) {
      return
    }
    setCardSubTarget(nextTarget)
  }, [activeTab, requestedReportSubscriptionId, requestedReportFocusLabel, reports, requestedReportFallback, resolveReportSubscription, cardSubTarget])

  const handleOpenReportSubscription = useCallback((report: Report) => {
    const reportSubscription = resolveReportSubscription(report)
    setCardSubTarget({
      scopeType: 'report',
      scopeKey: report.id,
      scopeLabel: reportSubscription?.scope_label || reportSubscriptionLabel(report),
    })
    const location = buildReportSubscriptionLocation(searchParams, {
      ...report,
      report_subscription: reportSubscription,
    })
    setSearchParams(new URLSearchParams(location.slice('/reports?'.length)), { replace: true })
  }, [resolveReportSubscription, searchParams, setSearchParams])

  const handleCopyReportSubscriptionLink = useCallback((report: Report) => {
    const reportSubscription = resolveReportSubscription(report)
    const location = buildReportSubscriptionLocation(searchParams, {
      ...report,
      report_subscription: reportSubscription,
    })
    void navigator.clipboard.writeText(`${window.location.origin}${location}`).then(() => {
      setCopiedReportLinkId(report.id)
      window.setTimeout(() => {
        setCopiedReportLinkId((current) => (current === report.id ? null : current))
      }, 2000)
    })
  }, [resolveReportSubscription, searchParams])

  const handleCopySubscriptionLink = useCallback((subscription: ReportSubscription) => {
    const location = buildSubscriptionManageLocation(subscription.id)
    void navigator.clipboard.writeText(`${window.location.origin}${location}`).then(() => {
      setCopiedSubscriptionLinkId(subscription.id)
      window.setTimeout(() => {
        setCopiedSubscriptionLinkId((current) => (current === subscription.id ? null : current))
      }, 2000)
    })
  }, [])

  function clearFilters() {
    setTypeFilter('')
    setQualityFilter('')
    setFreshnessFilter('')
    setReviewFilter('')
    setVendorSearch('')
    setDebouncedVendor('')
    updateReportsQuery({
      report_type: undefined,
      vendor_filter: undefined,
      quality_status: undefined,
      freshness_state: undefined,
      review_state: undefined,
    })
  }

  function composerPanelClass(composer: ReportComposer) {
    return clsx(
      'bg-slate-900/50 border rounded-xl p-4 transition-colors',
      activeComposer === composer
        ? 'border-cyan-500/50 bg-cyan-950/20'
        : 'border-slate-700/50',
    )
  }

  async function handleCreateComparison() {
    if (!primaryVendor.trim() || !comparisonVendor.trim()) {
      alert('Enter both vendors to compare')
      return
    }
    setCreatingComparison(true)
    try {
      const result = await generateVendorComparisonReport({
        primary_vendor: primaryVendor.trim(),
        comparison_vendor: comparisonVendor.trim(),
        persist: true,
      })
      const reportId = typeof result.report_id === 'string' ? result.report_id : ''
      setPrimaryVendor('')
      setComparisonVendor('')
      setActiveComposer(null)
      updateComposerParams(null, {
        primary_vendor: undefined,
        comparison_vendor: undefined,
        composer: undefined,
      })
      refresh()
      if (reportId) {
        navigate(reportDetailLocation(reportId, reportsBackTarget), {
          state: { backTo: reportsBackTarget },
        })
      }
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Comparison generation failed')
    } finally {
      setCreatingComparison(false)
    }
  }

  async function handleCreateAccountComparison() {
    if (!primaryCompany.trim() || !comparisonCompany.trim()) {
      alert('Enter both companies to compare')
      return
    }
    setCreatingAccountComparison(true)
    try {
      const result = await generateAccountComparisonReport({
        primary_company: primaryCompany.trim(),
        comparison_company: comparisonCompany.trim(),
        persist: true,
      })
      const reportId = typeof result.report_id === 'string' ? result.report_id : ''
      setPrimaryCompany('')
      setComparisonCompany('')
      setActiveComposer(null)
      updateComposerParams(null, {
        primary_company: undefined,
        comparison_company: undefined,
        composer: undefined,
      })
      refresh()
      if (reportId) {
        navigate(reportDetailLocation(reportId, reportsBackTarget), {
          state: { backTo: reportsBackTarget },
        })
      }
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Account comparison generation failed')
    } finally {
      setCreatingAccountComparison(false)
    }
  }

  async function handleCreateBattleCard() {
    if (!battleCardVendor.trim()) {
      alert('Enter a vendor name')
      return
    }
    setCreatingBattleCard(true)
    try {
      const result = await requestBattleCardReport({ vendor_name: battleCardVendor.trim() })
      const reportId = typeof result.report_id === 'string' ? result.report_id : ''
      setBattleCardVendor('')
      setActiveComposer(null)
      updateComposerParams(null, {
        battle_card_vendor: undefined,
        composer: undefined,
      })
      refresh()
      if (reportId) {
        navigate(reportDetailLocation(reportId, reportsBackTarget), {
          state: { backTo: reportsBackTarget },
        })
      }
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Battle card generation failed')
    } finally {
      setCreatingBattleCard(false)
    }
  }

  async function handleCreateAccountDeepDive() {
    if (!deepDiveCompany.trim()) {
      alert('Enter a company for the account deep dive')
      return
    }
    setCreatingAccountDeepDive(true)
    try {
      const result = await generateAccountDeepDiveReport({
        company_name: deepDiveCompany.trim(),
        persist: true,
      })
      const reportId = typeof result.report_id === 'string' ? result.report_id : ''
      setDeepDiveCompany('')
      setActiveComposer(null)
      updateComposerParams(null, {
        deep_dive_company: undefined,
        composer: undefined,
      })
      refresh()
      if (reportId) {
        navigate(reportDetailLocation(reportId, reportsBackTarget), {
          state: { backTo: reportsBackTarget },
        })
      }
    } catch (err) {
      alert(err instanceof Error ? err.message : 'Account deep dive generation failed')
    } finally {
      setCreatingAccountDeepDive(false)
    }
  }

  if (!canAccessReports) {
    return (
      <UpgradeGate allowed={false} feature="Intelligence Library" requiredPlan="Starter">
        <div />
      </UpgradeGate>
    )
  }

  if (error) return <PageError error={error} onRetry={refresh} />

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-4">
        <div className="flex items-center justify-between">
          <div className="flex items-center gap-3">
            {reportsBackTarget !== '/reports' && (
              <button
                onClick={() => navigate(reportsBackTarget)}
                className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors"
              >
                <ChevronLeft className="h-4 w-4" />
                {backButtonLabel}
              </button>
            )}
            <div>
              <h1 className="text-2xl font-bold text-white">Intelligence Library</h1>
              {activeVendorFilter ? (
                <div className="mt-2 flex flex-wrap items-center gap-3 text-sm">
                  <span className="text-slate-500">
                    Filtered to <span className="text-slate-300">{activeVendorFilter}</span>
                  </span>
                  <Link
                    to={vendorDetailPath(activeVendorFilter, currentLibraryPath)}
                    className="text-cyan-400 hover:text-cyan-300 transition-colors"
                  >
                    Vendor workspace
                  </Link>
                  <Link
                    to={evidencePath(activeVendorFilter, currentLibraryPath)}
                    className="text-violet-300 hover:text-violet-200 transition-colors"
                  >
                    Evidence
                  </Link>
                  <Link
                    to={opportunitiesPath(activeVendorFilter, currentLibraryPath)}
                    className="text-emerald-300 hover:text-emerald-200 transition-colors"
                  >
                    Opportunities
                  </Link>
                </div>
              ) : null}
            </div>
          </div>
          <div className="flex items-center gap-2">
            <Link
              to="/briefing-review"
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors"
            >
              Briefings
            </Link>
            <button
              onClick={() => setLibSubOpen(true)}
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm border border-cyan-700/40 bg-cyan-900/20 text-cyan-300 hover:bg-cyan-900/40 transition-colors"
            >
              <Bell className="h-4 w-4" />
              {hasFilters ? 'Subscribe to View' : 'Subscribe to Library'}
            </button>
            <button
              onClick={refresh}
              disabled={refreshing}
              className="inline-flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-50"
            >
              <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
              Refresh
            </button>
          </div>
        </div>

        {/* Tab bar */}
        <div className="flex items-center gap-1 border-b border-slate-700/50 pb-px">
          {([
            { key: 'library' as const, label: 'Library' },
            { key: 'subscriptions' as const, label: 'Subscriptions', count: subscriptions.length },
          ]).map((tab) => (
            <button
              key={tab.key}
              onClick={() => handleTabChange(tab.key)}
              className={clsx(
                'px-4 py-2 text-sm font-medium transition-colors border-b-2',
                activeTab === tab.key
                  ? 'text-cyan-400 border-cyan-400'
                  : 'text-slate-400 border-transparent hover:text-white',
              )}
            >
              {tab.label}{tab.count != null && tab.count > 0 ? ` (${tab.count})` : ''}
            </button>
          ))}
        </div>

        {/* Subscriptions tab */}
        {activeTab === 'subscriptions' && (
          <div className="space-y-4">
            {subLoading ? (
              <div className="flex items-center gap-2 text-sm text-slate-400">
                <Loader2 className="h-4 w-4 animate-spin" /> Loading subscriptions...
              </div>
            ) : subscriptions.length === 0 ? (
              <div className="text-sm text-slate-500 py-8 text-center">
                No subscriptions yet. Subscribe from a report or the library view.
              </div>
            ) : (
              <div className="space-y-2">
                {subscriptions.map((sub) => {
                  const filterSummary = sub.scope_type === 'library_view'
                    ? describeSubscriptionFilters(sub.filter_payload)
                    : []
                  return (
                  <div key={sub.id} className="bg-slate-800/50 border border-slate-700/30 rounded-lg p-4 flex items-center justify-between">
                    <div className="min-w-0 flex-1">
                      <div className="flex items-center gap-2 flex-wrap">
                        <span className="text-white font-medium">{sub.scope_label || sub.scope_key}</span>
                        <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-slate-700/50 text-slate-300">{sub.scope_type}</span>
                        <span className="px-2 py-0.5 rounded text-[10px] font-medium bg-cyan-500/20 text-cyan-400">{sub.delivery_frequency}</span>
                        <span className={clsx('px-2 py-0.5 rounded text-[10px] font-medium', sub.enabled ? 'bg-green-500/20 text-green-400' : 'bg-red-500/20 text-red-400')}>
                          {sub.enabled ? 'Active' : 'Paused'}
                        </span>
                      </div>
                      <div className="mt-1 flex items-center gap-3 text-xs text-slate-500 flex-wrap">
                        <span>Focus: {sub.deliverable_focus}</span>
                        <span>Freshness: {sub.freshness_policy.replace(/_/g, ' ')}</span>
                        {sub.next_delivery_at && <span>Next: {new Date(sub.next_delivery_at).toLocaleDateString()}</span>}
                        {sub.last_delivery_status && (
                          <span className={sub.last_delivery_status === 'delivered' ? 'text-green-400' : 'text-red-400'}>
                            Last: {sub.last_delivery_status} ({sub.last_delivery_report_count ?? 0} reports)
                          </span>
                        )}
                      </div>
                      {sub.scope_type === 'library_view' && filterSummary.length > 0 && (
                        <div className="mt-2 flex flex-wrap gap-2">
                          {filterSummary.map((line) => (
                            <span
                              key={line}
                              className="rounded-full border border-cyan-700/30 bg-cyan-950/20 px-2.5 py-1 text-[11px] text-cyan-100"
                            >
                              {line}
                            </span>
                          ))}
                        </div>
                      )}
                      {sub.recipient_emails.length > 0 && (
                        <div className="mt-1 text-xs text-slate-500">To: {sub.recipient_emails.join(', ')}</div>
                      )}
                    </div>
                    <div className="ml-4 flex items-center gap-2">
                      <button
                        onClick={() => handleCopySubscriptionLink(sub)}
                        className="inline-flex items-center gap-1 px-2 py-1.5 text-xs text-slate-400 hover:text-white hover:bg-slate-700/50 rounded-lg transition-colors"
                        title="Copy subscription link"
                      >
                        {copiedSubscriptionLinkId === sub.id ? (
                          <>
                            <Check className="h-3.5 w-3.5" />
                            Copied
                          </>
                        ) : (
                          <>
                            <Copy className="h-3.5 w-3.5" />
                            Copy Link
                          </>
                        )}
                      </button>
                      <button
                        onClick={() => handleEditSubscription(sub)}
                        className="p-2 text-slate-400 hover:text-cyan-400 hover:bg-slate-700/50 rounded-lg transition-colors"
                        title="Edit subscription"
                      >
                        <Pencil className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                  )
                })}
              </div>
            )}
          </div>
        )}

        {activeTab === 'library' && (
          <div className="space-y-6">
            <div className="flex flex-col sm:flex-row items-start sm:items-center gap-4 flex-wrap">
              <div className="relative flex-1 max-w-xs w-full">
                <Search className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-slate-500" />
                <input
                  type="text"
                  placeholder="Filter by vendor..."
                  value={vendorSearch}
                  onChange={(e) => {
                    const nextValue = e.target.value
                    setVendorSearch(nextValue)
                    updateSearchFilterParams({ vendor_filter: nextValue })
                  }}
                  className="w-full pl-9 pr-3 py-2 bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white placeholder-slate-500 focus:outline-none focus:border-cyan-500/50"
                />
              </div>
              <select
                value={typeFilter}
                onChange={(e) => {
                  const nextValue = e.target.value
                  setTypeFilter(nextValue)
                  updateSearchFilterParams({ report_type: nextValue })
                }}
                className="bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white px-3 py-2 focus:outline-none focus:border-cyan-500/50"
              >
                <option value="">All Types</option>
                <option value="weekly_churn_feed">Weekly Churn Feed</option>
                <option value="vendor_scorecard">Vendor Scorecard</option>
                <option value="displacement_report">Displacement Report</option>
                <option value="category_overview">Category Overview</option>
                <option value="exploratory_overview">Exploratory Overview</option>
                <option value="vendor_comparison">Vendor Comparison</option>
                <option value="account_comparison">Account Comparison</option>
                <option value="account_deep_dive">Account Deep Dive</option>
                <option value="vendor_retention">Vendor Retention</option>
                <option value="challenger_intel">Challenger Intel</option>
                <option value="battle_card">Battle Card</option>
                <option value="vendor_deep_dive">Vendor Deep Dive</option>
                <option value="challenger_brief">Challenger Brief</option>
              </select>
              <select
                value={qualityFilter}
                onChange={(e) => {
                  const nextValue = e.target.value
                  setQualityFilter(nextValue)
                  updateSearchFilterParams({ quality_status: nextValue })
                }}
                className="bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white px-3 py-2 focus:outline-none focus:border-cyan-500/50"
              >
                <option value="">All Quality</option>
                <option value="sales_ready">Sales Ready</option>
                <option value="needs_review">Needs Review</option>
                <option value="thin_evidence">Thin Evidence</option>
                <option value="deterministic_fallback">Fallback</option>
              </select>
              <select
                value={freshnessFilter}
                onChange={(e) => {
                  const nextValue = e.target.value
                  setFreshnessFilter(nextValue)
                  updateSearchFilterParams({ freshness_state: nextValue })
                }}
                className="bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white px-3 py-2 focus:outline-none focus:border-cyan-500/50"
              >
                <option value="">All Freshness</option>
                <option value="fresh">Fresh</option>
                <option value="monitor">Monitor</option>
                <option value="stale">Stale</option>
              </select>
              <select
                value={reviewFilter}
                onChange={(e) => {
                  const nextValue = e.target.value
                  setReviewFilter(nextValue)
                  updateSearchFilterParams({ review_state: nextValue })
                }}
                className="bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white px-3 py-2 focus:outline-none focus:border-cyan-500/50"
              >
                <option value="">All Review States</option>
                <option value="clean">Clean</option>
                <option value="warnings">Warnings</option>
                <option value="open_review">Open Review</option>
                <option value="blocked">Blocked</option>
              </select>
              {hasFilters && (
                <button
                  onClick={clearFilters}
                  className="inline-flex items-center gap-1 px-2.5 py-1.5 rounded-lg text-xs text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors"
                >
                  <X className="h-3 w-3" />
                  Clear filters
                </button>
              )}
            </div>

            <div className="flex items-center gap-2 text-sm text-slate-400">
              {debouncePending || loading ? (
                <>
                  <Loader2 className="h-3.5 w-3.5 animate-spin" />
                  Searching...
                </>
              ) : (
                <span>{filteredReports.length} report{filteredReports.length !== 1 ? 's' : ''} found{reports.length >= 200 ? ' (showing max 200)' : ''}</span>
              )}
            </div>
            <div className="text-xs">
              {newestReportAt ? (
                <span className={reportsStale ? 'text-amber-400' : 'text-slate-500'}>
                  Latest report generated {new Date(newestReportAt).toLocaleString()}
                  {reportsStale ? ' (older than 24h)' : ''}
                </span>
              ) : (
                <span className="text-slate-500">No report freshness timestamp available</span>
              )}
            </div>

            <div className={composerPanelClass('vendor_comparison')}>
              <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
                <div className="flex-1">
                  <label className="block text-xs font-medium text-slate-400 mb-1">Primary vendor</label>
                  <input
                    value={primaryVendor}
                    onChange={(e) => {
                      const nextValue = e.target.value
                      setActiveComposer('vendor_comparison')
                      setPrimaryVendor(nextValue)
                      updateComposerParams('vendor_comparison', { primary_vendor: nextValue })
                    }}
                    placeholder="Example: Salesforce"
                    className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white px-3 py-2 focus:outline-none focus:border-cyan-500/50"
                  />
                </div>
                <div className="flex-1">
                  <label className="block text-xs font-medium text-slate-400 mb-1">Comparison vendor</label>
                  <input
                    value={comparisonVendor}
                    onChange={(e) => {
                      const nextValue = e.target.value
                      setActiveComposer('vendor_comparison')
                      setComparisonVendor(nextValue)
                      updateComposerParams('vendor_comparison', { comparison_vendor: nextValue })
                    }}
                    placeholder="Example: HubSpot"
                    className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white px-3 py-2 focus:outline-none focus:border-cyan-500/50"
                  />
                </div>
                <button
                  onClick={handleCreateComparison}
                  disabled={creatingComparison}
                  className="inline-flex items-center justify-center px-4 py-2 rounded-lg bg-fuchsia-500/15 text-fuchsia-300 text-sm font-medium hover:bg-fuchsia-500/25 transition-colors disabled:opacity-50"
                >
                  {creatingComparison ? 'Generating...' : 'Create Comparison'}
                </button>
              </div>
            </div>

            <div className={composerPanelClass('account_deep_dive')}>
              <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
                <div className="flex-1">
                  <label className="block text-xs font-medium text-slate-400 mb-1">Account deep dive company</label>
                  <input
                    value={deepDiveCompany}
                    onChange={(e) => {
                      const nextValue = e.target.value
                      setActiveComposer('account_deep_dive')
                      setDeepDiveCompany(nextValue)
                      updateComposerParams('account_deep_dive', { deep_dive_company: nextValue })
                    }}
                    placeholder="Example: DataPulse Analytics"
                    className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white px-3 py-2 focus:outline-none focus:border-cyan-500/50"
                  />
                </div>
                <button
                  onClick={handleCreateAccountDeepDive}
                  disabled={creatingAccountDeepDive}
                  className="inline-flex items-center justify-center px-4 py-2 rounded-lg bg-pink-500/15 text-pink-300 text-sm font-medium hover:bg-pink-500/25 transition-colors disabled:opacity-50"
                >
                  {creatingAccountDeepDive ? 'Generating...' : 'Create Account Deep Dive'}
                </button>
              </div>
            </div>

            <div className={composerPanelClass('account_comparison')}>
              <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
                <div className="flex-1">
                  <label className="block text-xs font-medium text-slate-400 mb-1">Primary company</label>
                  <input
                    value={primaryCompany}
                    onChange={(e) => {
                      const nextValue = e.target.value
                      setActiveComposer('account_comparison')
                      setPrimaryCompany(nextValue)
                      updateComposerParams('account_comparison', { primary_company: nextValue })
                    }}
                    placeholder="Example: DataPulse Analytics"
                    className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white px-3 py-2 focus:outline-none focus:border-cyan-500/50"
                  />
                </div>
                <div className="flex-1">
                  <label className="block text-xs font-medium text-slate-400 mb-1">Comparison company</label>
                  <input
                    value={comparisonCompany}
                    onChange={(e) => {
                      const nextValue = e.target.value
                      setActiveComposer('account_comparison')
                      setComparisonCompany(nextValue)
                      updateComposerParams('account_comparison', { comparison_company: nextValue })
                    }}
                    placeholder="Example: FinEdge Capital"
                    className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white px-3 py-2 focus:outline-none focus:border-cyan-500/50"
                  />
                </div>
                <button
                  onClick={handleCreateAccountComparison}
                  disabled={creatingAccountComparison}
                  className="inline-flex items-center justify-center px-4 py-2 rounded-lg bg-rose-500/15 text-rose-300 text-sm font-medium hover:bg-rose-500/25 transition-colors disabled:opacity-50"
                >
                  {creatingAccountComparison ? 'Generating...' : 'Create Account Comparison'}
                </button>
              </div>
            </div>

            <div className={composerPanelClass('battle_card')}>
              <div className="flex flex-col gap-3 lg:flex-row lg:items-end">
                <div className="flex-1">
                  <label className="block text-xs font-medium text-slate-400 mb-1">Battle card vendor</label>
                  <input
                    value={battleCardVendor}
                    onChange={(e) => {
                      const nextValue = e.target.value
                      setActiveComposer('battle_card')
                      setBattleCardVendor(nextValue)
                      updateComposerParams('battle_card', { battle_card_vendor: nextValue })
                    }}
                    placeholder="Example: Zendesk"
                    className="w-full bg-slate-800/50 border border-slate-700/50 rounded-lg text-sm text-white px-3 py-2 focus:outline-none focus:border-cyan-500/50"
                  />
                </div>
                <button
                  onClick={handleCreateBattleCard}
                  disabled={creatingBattleCard}
                  className="inline-flex items-center justify-center px-4 py-2 rounded-lg bg-red-500/15 text-red-300 text-sm font-medium hover:bg-red-500/25 transition-colors disabled:opacity-50"
                >
                  {creatingBattleCard ? 'Generating...' : 'Create Battle Card'}
                </button>
              </div>
            </div>

            {loading ? (
              <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                {Array.from({ length: 4 }, (_, i) => (
                  <CardSkeleton key={i} />
                ))}
              </div>
            ) : filteredReports.length === 0 ? (
              <div className="flex flex-col items-center justify-center py-24 text-center">
                <FileBarChart className="h-10 w-10 text-slate-600 mb-4" />
                <p className="text-slate-500 mb-4">No reports found</p>
                {hasFilters && (
                  <button
                    onClick={clearFilters}
                    className="px-3 py-1.5 rounded-lg bg-cyan-500/10 text-cyan-400 text-sm font-medium hover:bg-cyan-500/20 transition-colors"
                  >
                    Clear filters
                  </button>
                )}
              </div>
            ) : (
              <div className="space-y-4">
                <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                  {pagedReports.map((r: Report) => (
                    (() => {
                      const exportState = reportExportPresentation(r)
                      const title = reportTitle(r)
                      const reportSubscription = resolveReportSubscription(r)
                      const subscriptionAction = reportSubscriptionActionPresentation(reportSubscription)
                      return (
                        <div
                          key={r.id}
                          className="bg-slate-900/50 border border-slate-700/50 backdrop-blur rounded-xl p-5 text-left hover:border-cyan-500/30 transition-colors"
                          data-testid={`report-card-${r.id}`}
                        >
                          <button
                            onClick={() => navigate(reportDetailLocation(r.id, reportsBackTarget), {
                              state: { backTo: reportsBackTarget },
                            })}
                            className="w-full text-left"
                          >
                            <div className="flex items-start justify-between mb-3">
                              <div className="flex items-center gap-2">
                                <span
                                  className={`inline-flex items-center px-2 py-0.5 rounded text-xs font-medium ${
                                    REPORT_TYPE_COLORS[r.report_type] ?? 'bg-slate-500/20 text-slate-400'
                                  }`}
                                >
                                  {r.report_type.replace(/_/g, ' ')}
                                </span>
                              </div>
                              <FileBarChart className="h-4 w-4 text-slate-500" />
                            </div>
                            {(r.vendor_filter || r.category_filter) && (
                              <p className="text-sm text-white font-medium mb-1">
                                {title}
                              </p>
                            )}
                            <p className="text-sm text-slate-400 line-clamp-2">
                              {r.executive_summary ?? 'No summary available'}
                            </p>
                            <div className="flex flex-wrap items-center gap-2 mt-3 mb-3">
                              <p className="text-xs text-slate-500">
                                {r.report_date ?? r.created_at ?? '--'}
                              </p>
                              <span className={clsx('inline-flex items-center px-2 py-0.5 rounded text-[11px] font-medium', exportState.className)}>
                                {exportState.label}
                              </span>
                            </div>
                            <ReportTrustPanel
                              compact
                              status={r.status}
                              artifactState={r.artifact_state ?? r.trust?.artifact_state}
                              artifactLabel={r.artifact_label ?? r.trust?.artifact_label}
                              blockerCount={r.blocker_count}
                              warningCount={r.warning_count}
                              unresolvedIssueCount={r.unresolved_issue_count}
                              qualityStatus={r.quality_status}
                              latestFailureStep={r.latest_failure_step}
                              latestErrorSummary={r.latest_error_summary}
                              freshnessState={r.freshness_state ?? r.trust?.freshness_state}
                              freshnessLabel={r.freshness_label ?? r.trust?.freshness_label}
                              reviewState={r.review_state ?? r.trust?.review_state}
                              reviewLabel={r.review_label ?? r.trust?.review_label}
                              freshnessTimestamp={r.created_at ?? r.report_date}
                            />
                          </button>
                          <div className="mt-4 pt-3 border-t border-slate-800 flex items-center justify-between gap-2">
                            <button
                              onClick={() => handleCopyReportSubscriptionLink(r)}
                              className="inline-flex items-center gap-1.5 px-3 py-1.5 rounded-lg bg-slate-800 border border-slate-700 text-slate-300 text-xs font-medium hover:text-white hover:bg-slate-700 transition-colors"
                            >
                              {copiedReportLinkId === r.id ? (
                                <>
                                  <Check className="h-3.5 w-3.5 text-emerald-400" />
                                  Copied
                                </>
                              ) : (
                                <>
                                  <Copy className="h-3.5 w-3.5" />
                                  Copy Link
                                </>
                              )}
                            </button>
                            {r.has_pdf_export ? (
                              <button
                                onClick={() => downloadReportPdf(r.id)}
                                className="inline-flex items-center px-3 py-1.5 rounded-lg bg-emerald-500/10 text-emerald-300 text-xs font-medium hover:bg-emerald-500/20 transition-colors"
                              >
                                Export PDF
                              </button>
                            ) : (
                              <button
                                onClick={() => handleOpenReportSubscription(r)}
                                className={clsx(
                                  'inline-flex items-center px-3 py-1.5 rounded-lg text-xs font-medium transition-colors',
                                  subscriptionAction.className,
                                )}
                              >
                                {subscriptionAction.label}
                              </button>
                            )}
                          </div>
                        </div>
                      )
                    })()
                  ))}
                </div>

                {showPagination && (
                  <div className="flex items-center justify-between px-4 py-3 bg-slate-900/50 border border-slate-700/50 rounded-xl">
                    <div className="flex items-center gap-2 text-xs text-slate-400">
                      <span>
                        {safePage * perPage + 1}-{Math.min((safePage + 1) * perPage, filteredReports.length)} of {filteredReports.length}
                      </span>
                      <select
                        value={perPage}
                        onChange={(e) => { setPerPage(Number(e.target.value)); setPage(0) }}
                        className="bg-slate-800/50 border border-slate-700/50 rounded px-1.5 py-0.5 text-xs text-white focus:outline-none"
                      >
                        {PAGE_SIZES.map((s) => (
                          <option key={s} value={s}>{s} / page</option>
                        ))}
                      </select>
                    </div>
                    <div className="flex items-center gap-1">
                      <button
                        onClick={() => setPage((p) => Math.max(0, p - 1))}
                        disabled={safePage === 0}
                        className="p-1 rounded text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                      >
                        <ChevronLeft className="h-4 w-4" />
                      </button>
                      <span className="text-xs text-slate-400 px-2">
                        {safePage + 1} / {totalPages}
                      </span>
                      <button
                        onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                        disabled={safePage >= totalPages - 1}
                        className="p-1 rounded text-slate-400 hover:text-white hover:bg-slate-800/50 transition-colors disabled:opacity-30 disabled:cursor-not-allowed"
                      >
                        <ChevronRight className="h-4 w-4" />
                      </button>
                    </div>
                  </div>
                )}
              </div>
            )}
          </div>
        )}

        {/* Edit subscription modal */}
        {editingSub && (
          <SubscriptionModal
            open={true}
            onClose={handleCloseEditingSubscription}
            scopeType={editingSub.scope_type}
            scopeKey={editingSub.scope_key}
            scopeLabel={editingSub.scope_label}
            filterPayload={editingSub.filter_payload}
            onSaved={() => { void refreshSubs(); handleCloseEditingSubscription() }}
          />
        )}
      </div>

      <SubscriptionModal
        open={libSubOpen}
        onClose={() => setLibSubOpen(false)}
        scopeType={libraryScopeType}
        scopeKey={libraryScopeKey}
        scopeLabel={libraryScopeLabel}
        filterPayload={libraryScopeType === 'library_view' ? libraryViewFilters : undefined}
        onSaved={() => { void refreshSubs(); setLibSubOpen(false) }}
      />

      <SubscriptionModal
        open={Boolean(cardSubTarget)}
        onClose={() => {
          setCardSubTarget(null)
          updateReportsQuery({
            report_subscription: undefined,
            report_focus_type: undefined,
            report_focus_vendor: undefined,
            report_focus_label: undefined,
          })
        }}
        scopeType={cardSubTarget?.scopeType ?? 'report'}
        scopeKey={cardSubTarget?.scopeKey ?? ''}
        scopeLabel={cardSubTarget?.scopeLabel ?? ''}
        onSaved={(subscription) => {
          if (subscription.scope_type === 'report') {
            setSavedReportSubscriptions((current) => ({
              ...current,
              [subscription.scope_key]: {
                id: subscription.id,
                scope_type: 'report',
                scope_key: subscription.scope_key,
                scope_label: subscription.scope_label,
                enabled: subscription.enabled,
              },
            }))
          }
          void refreshSubs()
          setCardSubTarget(null)
          updateReportsQuery({
            report_subscription: subscription.scope_key,
            report_focus_type: reports.find((report) => report.id === subscription.scope_key)?.report_type
              ?? (requestedReportFallback?.id === subscription.scope_key ? requestedReportFallback.report_type : undefined),
            report_focus_vendor: reports.find((report) => report.id === subscription.scope_key)?.vendor_filter
              ?? (requestedReportFallback?.id === subscription.scope_key ? requestedReportFallback.vendor_filter : undefined)
              ?? undefined,
            report_focus_label: subscription.scope_label,
          })
        }}
      />
    </div>
  )
}
