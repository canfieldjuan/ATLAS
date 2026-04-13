import { useCallback, useEffect, useMemo, useRef, useState } from 'react'
import { Link, useNavigate, useSearchParams } from 'react-router-dom'
import {
  Activity,
  ArrowLeft,
  BellRing,
  Building2,
  Download,
  Fingerprint,
  Plus,
  RefreshCw,
  Search,
  Telescope,
  Trash2,
  X,
  Zap,
} from 'lucide-react'
import { clsx } from 'clsx'
import StatCard from '../components/StatCard'
import DataTable, { type Column } from '../components/DataTable'
import UrgencyBadge from '../components/UrgencyBadge'
import ArchetypeBadge from '../components/ArchetypeBadge'
import AccountMovementDrawer from '../components/AccountMovementDrawer'
import EvidenceDrawer from '../components/EvidenceDrawer'
import { PageError } from '../components/ErrorBoundary'
import useApiData from '../hooks/useApiData'
import {
  addTrackedVendor,
  createCompetitiveSet,
  deliverWatchlistAlertEmail,
  evaluateWatchlistAlertEvents,
  createWatchlistView,
  deleteCompetitiveSet,
  deleteWatchlistView,
  fetchCompetitiveSetPlan,
  fetchAccountsInMotionFeed,
  fetchSlowBurnWatchlist,
  listWatchlistAlertEmailLog,
  listWatchlistAlertEvents,
  listTrackedVendors,
  listCompetitiveSets,
  listWatchlistViews,
  removeTrackedVendor,
  runCompetitiveSetNow,
  searchAvailableVendors,
  type AccountsInMotionFeedItem,
  type CompetitiveSet,
  type CompetitiveSetDefaults,
  type CompetitiveSetPlan,
  type CompetitiveSetRun,
  type TrackedVendor,
  type WatchlistAlertEvent,
  type WatchlistView,
  updateWatchlistView,
  updateCompetitiveSet,
  type VendorSearchResult,
  downloadCsv,
  generateCampaigns,
} from '../api/client'
import type { ChurnSignal } from '../types'

interface WatchlistsData {
  vendors: TrackedVendor[]
  watchlistViews: WatchlistView[]
  competitiveSets: CompetitiveSet[]
  competitiveSetDefaults: CompetitiveSetDefaults | null
  feed: ChurnSignal[]
  vendorAlertHitCount: number
  feedStaleThresholdHitCount: number
  accounts: AccountsInMotionFeedItem[]
  accountAlertHitCount: number
  accountStaleThresholdHitCount: number
  vendorsWithAccounts: number
  freshestAccountsReportDate: string | null
  loadWarnings: string[]
}

interface CompetitiveSetLastRunOverride {
  last_run_status: CompetitiveSet['last_run_status']
  last_run_summary: Record<string, unknown>
  last_run_at: string
}

const SEARCH_DEBOUNCE_MS = 250
const MIN_VENDOR_SEARCH_CHARS = 2
const SEARCH_RESULTS_PREVIEW_LIMIT = 8
const ACCOUNT_URGENCY_FILTER_OPTIONS = [7, 8, 9] as const

type AccountPresentationTier = 'named_high' | 'named_medium' | 'review'

function toLoadWarning(section: string, err: unknown): string {
  const detail = err instanceof Error && err.message.trim()
    ? ` ${err.message.trim()}`
    : ''
  return `${section} is temporarily unavailable.${detail}`
}

function isSameAccountMovementRow(
  left: AccountsInMotionFeedItem,
  right: AccountsInMotionFeedItem,
) {
  return (
    left.vendor === right.vendor
    && (left.company || '') === (right.company || '')
    && (left.report_date || '') === (right.report_date || '')
    && (left.watch_vendor || '') === (right.watch_vendor || '')
    && (left.category || '') === (right.category || '')
    && (left.track_mode || '') === (right.track_mode || '')
  )
}

function formatTs(value: string | null | undefined) {
  if (!value) return '--'
  const date = new Date(value)
  if (Number.isNaN(date.getTime())) return '--'
  return date.toLocaleString()
}

function toTimestamp(value: string | null | undefined) {
  if (!value) return null
  const ts = new Date(value).getTime()
  return Number.isNaN(ts) ? null : ts
}

function parseOptionalNumber(value: string) {
  if (!value.trim()) return null
  const parsed = Number(value)
  return Number.isFinite(parsed) ? parsed : null
}

function parseBooleanSearchParam(value: string | null | undefined) {
  if (!value) return false
  const normalized = value.trim().toLowerCase()
  return normalized === 'true' || normalized === '1'
}

function ageInDays(value: string | null | undefined) {
  const ts = toTimestamp(value)
  if (ts == null) return null
  const diff = Date.now() - ts
  if (!Number.isFinite(diff) || diff < 0) return 0
  return diff / (1000 * 60 * 60 * 24)
}

function freshnessTone(status: string | null | undefined, fallbackTimestamp?: string | null | undefined) {
  if (status === 'fresh') return 'text-emerald-400'
  if (status === 'stale') return 'text-amber-400'
  if (status === 'synthesis_pending' || status === 'artifact_missing') return 'text-slate-400'
  const ts = toTimestamp(fallbackTimestamp)
  if (ts == null) return 'text-slate-500'
  return 'text-slate-400'
}

function freshnessLabel(status: string | null | undefined, fallbackTimestamp?: string | null | undefined) {
  if (status === 'fresh') return 'fresh'
  if (status === 'stale') return 'stale'
  if (status === 'synthesis_pending') return 'synthesis pending'
  if (status === 'artifact_missing') return 'artifact missing'
  return fallbackTimestamp ? 'timestamp only' : '--'
}

function formatCostUsd(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return '--'
  return new Intl.NumberFormat('en-US', {
    style: 'currency',
    currency: 'USD',
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  }).format(value)
}

function formatWholeNumber(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return '--'
  return new Intl.NumberFormat('en-US', {
    maximumFractionDigits: 0,
  }).format(value)
}

function asString(value: unknown): string {
  return typeof value === 'string' ? value.trim() : ''
}

function toStringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.map((item) => asString(item)).filter(Boolean)
    : []
}

function summarizeReasoningDelta(signal: ChurnSignal): string[] {
  const delta = signal.reasoning_delta
  if (!delta) return []
  const items: string[] = []
  if (delta.wedge_changed) items.push('Wedge changed')
  if (delta.confidence_changed) items.push('Confidence shifted')
  if (delta.top_destination_changed) {
    const destination = asString(delta.current_top_destination)
    items.push(destination ? `Destination: ${destination}` : 'Destination shifted')
  }
  const timingWindows = toStringArray(delta.new_timing_windows)
  if (timingWindows.length > 0) items.push(`Timing +${timingWindows.length}`)
  const accounts = toStringArray(delta.new_account_signals)
  if (accounts.length > 0) items.push(`Accounts +${accounts.length}`)
  return items
}

function confidenceBand(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) {
    return {
      label: 'unscored',
      tone: 'bg-slate-700/50 text-slate-300 border-slate-600/50',
    }
  }
  if (value >= 7) {
    return {
      label: 'higher confidence',
      tone: 'bg-emerald-500/10 text-emerald-300 border-emerald-500/30',
    }
  }
  if (value >= 4) {
    return {
      label: 'moderate confidence',
      tone: 'bg-amber-500/10 text-amber-300 border-amber-500/30',
    }
  }
  return {
    label: 'low confidence',
    tone: 'bg-rose-500/10 text-rose-300 border-rose-500/30',
  }
}

function accountPresentationTier(row: AccountsInMotionFeedItem): AccountPresentationTier {
  const hasNamedAccount = asString(row.company).length > 0
  const confidence = typeof row.confidence === 'number' && !Number.isNaN(row.confidence)
    ? row.confidence
    : null
  if (!hasNamedAccount) return 'review'
  if (row.quality_flags.length > 0) return 'review'
  if (confidence != null && confidence >= 7) return 'named_high'
  if (confidence != null && confidence >= 4) return 'named_medium'
  return 'review'
}

function vendorAlertTriggered(row: ChurnSignal, threshold: number | null) {
  return threshold != null && row.avg_urgency_score >= threshold
}

function accountAlertTriggered(row: AccountsInMotionFeedItem, threshold: number | null) {
  return threshold != null && row.urgency >= threshold
}

function staleThresholdTriggered(
  threshold: number | null,
  options: {
    staleDays?: number | null
    freshnessTimestamp?: string | null
    fallbackTimestamp?: string | null
  },
) {
  if (threshold == null) return false
  if (options.staleDays != null) return options.staleDays > threshold
  const age = ageInDays(options.freshnessTimestamp || options.fallbackTimestamp)
  return age != null && age > threshold
}

function watchlistViewVendorNames(view: WatchlistView) {
  if (view.vendor_names?.length) {
    return view.vendor_names
  }
  return view.vendor_name ? [view.vendor_name] : []
}

function watchlistViewMatchesState(
  view: WatchlistView,
  filters: {
    vendor_name: string
    vendor_names: string[]
    category: string
    source: string
    min_urgency: string
    fresh_only: boolean
    named_accounts_only: boolean
    changed_wedges_only: boolean
    vendor_alert_threshold: string
    account_alert_threshold: string
    stale_days_threshold: string
    alert_email_enabled: boolean
    alert_delivery_frequency: 'daily' | 'weekly'
  },
) {
  return JSON.stringify([...watchlistViewVendorNames(view)].sort()) === JSON.stringify([...(filters.vendor_names || [])].sort())
    && (view.category || '') === filters.category
    && (view.source || '') === filters.source
    && String(view.min_urgency ?? '') === filters.min_urgency
    && view.include_stale === !filters.fresh_only
    && view.named_accounts_only === filters.named_accounts_only
    && view.changed_wedges_only === filters.changed_wedges_only
    && String(view.vendor_alert_threshold ?? '') === filters.vendor_alert_threshold
    && String(view.account_alert_threshold ?? '') === filters.account_alert_threshold
    && String(view.stale_days_threshold ?? '') === filters.stale_days_threshold
    && view.alert_email_enabled === filters.alert_email_enabled
    && view.alert_delivery_frequency === filters.alert_delivery_frequency
}

function summarizeWatchlistView(view: WatchlistView) {
  const parts: string[] = []
  const vendorNames = watchlistViewVendorNames(view)
  if (vendorNames.length) parts.push(vendorNames.join(', '))
  if (view.category) parts.push(view.category)
  if (view.source) parts.push(view.source)
  if (view.min_urgency != null) parts.push(`urgency ${view.min_urgency}+`)
  if (!view.include_stale) parts.push('fresh only')
  if (view.named_accounts_only) parts.push('named accounts only')
  if (view.changed_wedges_only) parts.push('changed wedges only')
  if (view.vendor_alert_threshold != null) parts.push(`vendor alert ${view.vendor_alert_threshold}+`)
  if (view.account_alert_threshold != null) parts.push(`account alert ${view.account_alert_threshold}+`)
  if (view.stale_days_threshold != null) parts.push(`stale after ${view.stale_days_threshold}d`)
  if (view.alert_email_enabled) parts.push(`email ${view.alert_delivery_frequency}`)
  return parts.length > 0 ? parts.join(' - ') : 'All signals'
}

function summarizeLocalWatchlistSuppression(view: WatchlistView) {
  const parts: string[] = []
  if (view.named_accounts_only) parts.push('named accounts only')
  if (view.changed_wedges_only) parts.push('changed wedges only')
  return parts.join(' + ')
}

function summarizeWatchlistDeliverySummary(
  base: string,
  status: string | null | undefined,
  view: WatchlistView,
) {
  if (status === 'no_events') {
    const localSuppression = summarizeLocalWatchlistSuppression(view)
    if (localSuppression) return `${base} · local filters: ${localSuppression}`
  }
  return base
}

function summarizeSavedViewDelivery(view: WatchlistView) {
  if (!view.last_alert_delivery_status) return 'Last --'
  const base = view.last_alert_delivery_summary || `Last ${view.last_alert_delivery_status.replace(/_/g, ' ')}`
  return summarizeWatchlistDeliverySummary(base, view.last_alert_delivery_status, view)
}

function watchlistViewUrl(viewId: string) {
  const params = new URLSearchParams()
  params.set('view', viewId)
  return `${window.location.origin}/watchlists?${params.toString()}`
}

function parseBackTo(value: string | null) {
  if (!value) return null
  if (
    value.startsWith('/alerts')
    || value.startsWith('/evidence')
    || value.startsWith('/vendors/')
    || value.startsWith('/reports')
    || value.startsWith('/opportunities')
  ) return value
  try {
    const url = new URL(value, window.location.origin)
    if (
      url.origin === window.location.origin
      && (
        url.pathname === '/alerts'
        || url.pathname === '/evidence'
        || url.pathname.startsWith('/vendors/')
        || url.pathname === '/reports'
        || url.pathname === '/opportunities'
      )
    ) {
      return `${url.pathname}${url.search}`
    }
  } catch {
    return null
  }
  return null
}

function backToLabel(value: string | null) {
  if (!value) return 'Back'
  if (value.startsWith('/alerts')) return 'Back to Alerts'
  if (value.startsWith('/evidence')) return 'Back to Evidence'
  if (value.startsWith('/vendors/')) return 'Back to Vendor'
  if (value.startsWith('/reports')) return 'Back to Reports'
  if (value.startsWith('/opportunities')) return 'Back to Opportunities'
  return 'Back'
}

function watchlistPath(searchParams: URLSearchParams) {
  const query = searchParams.toString()
  return `/watchlists${query ? `?${query}` : ''}`
}

function watchlistAlertsPath(searchParams: URLSearchParams) {
  const params = new URLSearchParams()
  params.set('back_to', watchlistPath(searchParams))
  return `/alerts?${params.toString()}`
}

function watchlistVendorAlertsPath(searchParams: URLSearchParams, vendorName: string, companyName?: string | null) {
  const params = new URLSearchParams()
  params.set('vendor', vendorName)
  if (companyName?.trim()) {
    params.set('company', companyName.trim())
  }
  params.set('back_to', watchlistPath(searchParams))
  return `/alerts?${params.toString()}`
}

function watchlistVendorAlertsUrl(searchParams: URLSearchParams, vendorName: string, companyName?: string | null) {
  return `${window.location.origin}${watchlistVendorAlertsPath(searchParams, vendorName, companyName)}`
}

function watchlistVendorPath(searchParams: URLSearchParams, vendorName: string) {
  const params = new URLSearchParams()
  params.set('back_to', watchlistPath(searchParams))
  return `/vendors/${encodeURIComponent(vendorName)}?${params.toString()}`
}

function watchlistOpportunitiesPath(searchParams: URLSearchParams, vendorName: string) {
  const params = new URLSearchParams()
  params.set('vendor', vendorName)
  params.set('back_to', watchlistPath(searchParams))
  return `/opportunities?${params.toString()}`
}

function watchlistReportsPath(searchParams: URLSearchParams, vendorName: string) {
  const params = new URLSearchParams()
  params.set('vendor_filter', vendorName)
  params.set('back_to', watchlistPath(searchParams))
  return `/reports?${params.toString()}`
}

function watchlistReviewDetailPath(searchParams: URLSearchParams, row: AccountsInMotionFeedItem, reviewId: string) {
  const params = new URLSearchParams()
  params.set('back_to', watchlistPath(accountFocusParams(searchParams, row)))
  return `/reviews/${encodeURIComponent(reviewId)}?${params.toString()}`
}

function accountFocusParamsFromFocus(
  searchParams: URLSearchParams,
  focus: {
    vendor: string
    company: string
    report_date: string
    watch_vendor: string
    category: string
    track_mode: string
  },
) {
  const next = new URLSearchParams(searchParams)
  next.set('account_vendor', focus.vendor)
  next.set('account_company', focus.company)
  next.set('account_report_date', focus.report_date)
  next.set('account_watch_vendor', focus.watch_vendor)
  next.set('account_category', focus.category)
  next.set('account_track_mode', focus.track_mode)
  return next
}

function accountFocusParams(searchParams: URLSearchParams, row: AccountsInMotionFeedItem) {
  return accountFocusParamsFromFocus(searchParams, accountFocusFromRow(row))
}

function vendorFocusParams(searchParams: URLSearchParams, vendorName: string) {
  const next = new URLSearchParams(searchParams)
  next.delete('view')
  next.delete('account_vendor')
  next.delete('account_company')
  next.delete('account_report_date')
  next.delete('account_watch_vendor')
  next.delete('account_category')
  next.delete('account_track_mode')
  next.delete('witness_id')
  next.delete('witness_vendor')
  next.set('vendor_name', vendorName)
  return next
}

function watchlistAccountUrl(searchParams: URLSearchParams, row: AccountsInMotionFeedItem) {
  return `${window.location.origin}${watchlistPath(accountFocusParams(searchParams, row))}`
}

function watchlistReviewUrl(searchParams: URLSearchParams, row: AccountsInMotionFeedItem, reviewId: string) {
  return `${window.location.origin}${watchlistReviewDetailPath(searchParams, row, reviewId)}`
}

function watchlistVendorUrl(searchParams: URLSearchParams, vendorName: string) {
  return `${window.location.origin}${watchlistPath(vendorFocusParams(searchParams, vendorName))}`
}

function watchlistVendorWorkspaceUrl(searchParams: URLSearchParams, vendorName: string) {
  return `${window.location.origin}${watchlistVendorPath(searchParams, vendorName)}`
}

function watchlistOpportunitiesUrl(searchParams: URLSearchParams, vendorName: string) {
  return `${window.location.origin}${watchlistOpportunitiesPath(searchParams, vendorName)}`
}

function watchlistReportsUrl(searchParams: URLSearchParams, vendorName: string) {
  return `${window.location.origin}${watchlistReportsPath(searchParams, vendorName)}`
}

function watchlistSnapshotDate(searchParams: URLSearchParams) {
  const text = searchParams.get('account_report_date')?.trim() || ''
  return /^\d{4}-\d{2}-\d{2}$/.test(text) ? text : ''
}

function watchlistEvidenceExplorerPath(
  searchParams: URLSearchParams,
  vendorName: string,
  witnessId?: string | null,
  source?: string | null,
) {
  const params = new URLSearchParams()
  params.set('vendor', vendorName)
  params.set('tab', 'witnesses')
  const snapshotDate = watchlistSnapshotDate(searchParams)
  if (snapshotDate) params.set('as_of_date', snapshotDate)
  if (witnessId) params.set('witness_id', witnessId)
  if (source) params.set('source', source)
  params.set('back_to', watchlistPath(searchParams))
  return `/evidence?${params.toString()}`
}

function watchlistAccountEvidenceExplorerPath(
  searchParams: URLSearchParams,
  row: AccountsInMotionFeedItem,
  witnessId?: string | null,
  source?: string | null,
) {
  return watchlistEvidenceExplorerPath(accountFocusParams(searchParams, row), row.vendor, witnessId, source)
}

function watchlistAlertEventVendorName(event: WatchlistAlertEvent) {
  return event.vendor_name?.trim() || event.account_review_focus?.vendor || ''
}

function watchlistAlertEventPrimaryWitnessId(event: WatchlistAlertEvent) {
  const witnessIds = event.reasoning_reference_ids?.witness_ids
  if (!Array.isArray(witnessIds) || witnessIds.length === 0) return ''
  const firstWitnessId = witnessIds[0]
  return typeof firstWitnessId === 'string' ? firstWitnessId : String(firstWitnessId || '')
}

function watchlistAlertEventPrimaryReviewId(event: WatchlistAlertEvent) {
  const reviewIds = event.source_review_ids
  if (!Array.isArray(reviewIds) || reviewIds.length === 0) return ''
  const firstReviewId = reviewIds[0]
  return typeof firstReviewId === 'string' ? firstReviewId : String(firstReviewId || '')
}

function watchlistAlertEventContextParams(searchParams: URLSearchParams, event: WatchlistAlertEvent) {
  return event.account_review_focus
    ? accountFocusParamsFromFocus(searchParams, event.account_review_focus)
    : searchParams
}

function watchlistAlertEventAccountPath(searchParams: URLSearchParams, event: WatchlistAlertEvent) {
  if (!event.account_review_focus) return null
  return watchlistPath(watchlistAlertEventContextParams(searchParams, event))
}

function watchlistAlertEventReviewDetailPath(
  searchParams: URLSearchParams,
  event: WatchlistAlertEvent,
  reviewId: string,
) {
  if (!event.account_review_focus) return null
  const params = new URLSearchParams()
  params.set('back_to', watchlistPath(accountFocusParamsFromFocus(searchParams, event.account_review_focus)))
  return `/reviews/${encodeURIComponent(reviewId)}?${params.toString()}`
}

function watchlistAlertEventEvidencePath(
  searchParams: URLSearchParams,
  event: WatchlistAlertEvent,
  witnessId?: string | null,
  source?: string | null,
) {
  const vendorName = watchlistAlertEventVendorName(event)
  if (!vendorName) return null
  return watchlistEvidenceExplorerPath(watchlistAlertEventContextParams(searchParams, event), vendorName, witnessId, source)
}

function watchlistAlertEventVendorPath(searchParams: URLSearchParams, event: WatchlistAlertEvent) {
  const vendorName = watchlistAlertEventVendorName(event)
  if (!vendorName) return null
  return watchlistVendorPath(watchlistAlertEventContextParams(searchParams, event), vendorName)
}

function watchlistAlertEventAlertsPath(searchParams: URLSearchParams, event: WatchlistAlertEvent) {
  const vendorName = watchlistAlertEventVendorName(event)
  if (!vendorName) return null
  return watchlistVendorAlertsPath(
    watchlistAlertEventContextParams(searchParams, event),
    vendorName,
    event.account_review_focus?.company || event.company_name,
  )
}

function watchlistAlertEventReportsPath(searchParams: URLSearchParams, event: WatchlistAlertEvent) {
  const vendorName = watchlistAlertEventVendorName(event)
  if (!vendorName) return null
  return watchlistReportsPath(watchlistAlertEventContextParams(searchParams, event), vendorName)
}

function watchlistAlertEventOpportunitiesPath(searchParams: URLSearchParams, event: WatchlistAlertEvent) {
  const vendorName = watchlistAlertEventVendorName(event)
  if (!vendorName) return null
  return watchlistOpportunitiesPath(watchlistAlertEventContextParams(searchParams, event), vendorName)
}

function watchlistAlertEventAccountUrl(searchParams: URLSearchParams, event: WatchlistAlertEvent) {
  const path = watchlistAlertEventAccountPath(searchParams, event)
  return path ? `${window.location.origin}${path}` : null
}

function watchlistAlertEventReviewUrl(
  searchParams: URLSearchParams,
  event: WatchlistAlertEvent,
  reviewId: string,
) {
  const path = watchlistAlertEventReviewDetailPath(searchParams, event, reviewId)
  return path ? `${window.location.origin}${path}` : null
}

function watchlistAlertEventEvidenceUrl(
  searchParams: URLSearchParams,
  event: WatchlistAlertEvent,
  witnessId?: string | null,
  source?: string | null,
) {
  const path = watchlistAlertEventEvidencePath(searchParams, event, witnessId, source)
  return path ? `${window.location.origin}${path}` : null
}

function watchlistAlertEventVendorUrl(searchParams: URLSearchParams, event: WatchlistAlertEvent) {
  const path = watchlistAlertEventVendorPath(searchParams, event)
  return path ? `${window.location.origin}${path}` : null
}

function watchlistAlertEventAlertsUrl(searchParams: URLSearchParams, event: WatchlistAlertEvent) {
  const path = watchlistAlertEventAlertsPath(searchParams, event)
  return path ? `${window.location.origin}${path}` : null
}

function watchlistAlertEventReportsUrl(searchParams: URLSearchParams, event: WatchlistAlertEvent) {
  const path = watchlistAlertEventReportsPath(searchParams, event)
  return path ? `${window.location.origin}${path}` : null
}

function watchlistAlertEventOpportunitiesUrl(searchParams: URLSearchParams, event: WatchlistAlertEvent) {
  const path = watchlistAlertEventOpportunitiesPath(searchParams, event)
  return path ? `${window.location.origin}${path}` : null
}

function accountFocusFromRow(row: AccountsInMotionFeedItem) {
  return {
    vendor: row.vendor || '',
    company: row.company || '',
    report_date: row.report_date || '',
    watch_vendor: row.watch_vendor || '',
    category: row.category || '',
    track_mode: row.track_mode || '',
  }
}

function accountFocusMatchesRow(
  row: AccountsInMotionFeedItem,
  focus: {
    vendor: string
    company: string
    report_date: string
    watch_vendor: string
    category: string
    track_mode: string
  },
) {
  return (row.vendor || '') === focus.vendor
    && (row.company || '') === focus.company
    && (row.report_date || '') === focus.report_date
    && (row.watch_vendor || '') === focus.watch_vendor
    && (row.category || '') === focus.category
    && (row.track_mode || '') === focus.track_mode
}

function alertEventTone(event: WatchlistAlertEvent) {
  if (event.event_type === 'stale_data') {
    return 'border-amber-500/30 bg-amber-500/10 text-amber-200'
  }
  if (event.event_type === 'account_alert') {
    return 'border-emerald-500/30 bg-emerald-500/10 text-emerald-200'
  }
  return 'border-cyan-500/30 bg-cyan-500/10 text-cyan-200'
}

function alertEventLabel(event: WatchlistAlertEvent) {
  if (event.event_type === 'stale_data') return 'stale policy'
  if (event.event_type === 'account_alert') return 'account alert'
  return 'vendor alert'
}

function watchlistAlertScoreSourceLabel(source: string | null | undefined) {
  if (source === 'preview_signal_score') return 'preview signal'
  if (source === 'urgency') return 'urgency'
  return source || null
}

type PendingWatchlistDestructiveAction =
  | { kind: 'delete_watchlist_view'; view: WatchlistView }
  | { kind: 'remove_vendor'; vendorName: string }
  | { kind: 'delete_competitive_set'; item: CompetitiveSet }

function describePendingWatchlistDestructiveAction(action: PendingWatchlistDestructiveAction) {
  switch (action.kind) {
    case 'delete_watchlist_view':
      return {
        title: 'Delete saved view',
        message: `Delete saved view ${action.view.name}? This removes the saved monitoring slice and its delivery settings.`,
        confirmLabel: 'Delete saved view',
        confirmingLabel: 'Deleting...',
      }
    case 'remove_vendor':
      return {
        title: 'Remove tracked vendor',
        message: `Remove ${action.vendorName} from your watchlists? This stops it from appearing in tracked monitoring and vendor-scoped shortcuts.`,
        confirmLabel: 'Remove vendor',
        confirmingLabel: 'Removing...',
      }
    case 'delete_competitive_set':
      return {
        title: 'Delete competitive set',
        message: `Delete competitive set ${action.item.name}? This removes the saved comparison scope and its preview history.`,
        confirmLabel: 'Delete competitive set',
        confirmingLabel: 'Deleting...',
      }
  }
}

function DestructiveActionModal({
  title,
  message,
  confirmLabel,
  confirmingLabel,
  confirming,
  error,
  onCancel,
  onConfirm,
}: {
  title: string
  message: string
  confirmLabel: string
  confirmingLabel: string
  confirming: boolean
  error: string | null
  onCancel: () => void
  onConfirm: () => void
}) {
  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4">
      <div
        className="absolute inset-0 bg-slate-950/80"
        onClick={() => {
          if (!confirming) onCancel()
        }}
      />
      <div
        role="alertdialog"
        aria-modal="true"
        aria-labelledby="watchlists-destructive-action-title"
        className="relative z-10 w-full max-w-md rounded-xl border border-rose-500/20 bg-slate-950 p-6 shadow-2xl"
      >
        <div className="flex items-start justify-between gap-3">
          <div className="flex items-start gap-3">
            <div className="rounded-lg bg-rose-500/10 p-2 text-rose-300">
              <Trash2 className="h-4 w-4" />
            </div>
            <div>
              <h2 id="watchlists-destructive-action-title" className="text-base font-semibold text-white">
                {title}
              </h2>
              <p className="mt-1 text-sm text-slate-400">{message}</p>
            </div>
          </div>
          <button
            type="button"
            onClick={() => {
              if (!confirming) onCancel()
            }}
            className="text-slate-500 hover:text-slate-300 disabled:cursor-not-allowed disabled:opacity-50"
            disabled={confirming}
            aria-label="Close destructive action dialog"
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
            className="rounded-md bg-rose-500/10 px-3 py-2 text-sm font-medium text-rose-300 hover:bg-rose-500/20 disabled:cursor-not-allowed disabled:opacity-50"
          >
            {confirming ? confirmingLabel : confirmLabel}
          </button>
        </div>
      </div>
    </div>
  )
}

export default function Watchlists() {
  const navigate = useNavigate()
  const [searchParams, setSearchParams] = useSearchParams()
  const requestedVendorName = searchParams.get('vendor_name')?.trim() || ''
  const requestedBackTo = parseBackTo(searchParams.get('back_to'))
  const [searchInput, setSearchInput] = useState('')
  const [debouncedSearch, setDebouncedSearch] = useState('')
  const [savedViewName, setSavedViewName] = useState('')
  const [selectedVendorFilter, setSelectedVendorFilter] = useState('')
  const [selectedVendorFilters, setSelectedVendorFilters] = useState<string[]>([])
  const [evidenceDrawerOpen, setEvidenceDrawerOpen] = useState(false)
  const [evidenceDrawerWitnessId, setEvidenceDrawerWitnessId] = useState<string | null>(null)
  const [evidenceDrawerVendor, setEvidenceDrawerVendor] = useState('')
  const handleOpenWitness = useCallback((witnessId: string, vendorName: string) => {
    setEvidenceDrawerWitnessId(witnessId)
    setEvidenceDrawerVendor(vendorName)
    setEvidenceDrawerOpen(true)
  }, [])
  const [selectedCategoryFilter, setSelectedCategoryFilter] = useState(searchParams.get('category')?.trim() || '')
  const [selectedSourceFilter, setSelectedSourceFilter] = useState(searchParams.get('source')?.trim() || '')
  const [selectedMinUrgency, setSelectedMinUrgency] = useState(searchParams.get('min_urgency')?.trim() || '')
  const [freshOnly, setFreshOnly] = useState(() => parseBooleanSearchParam(searchParams.get('fresh_only')))
  const [namedAccountsOnly, setNamedAccountsOnly] = useState(() => parseBooleanSearchParam(searchParams.get('named_accounts_only')))
  const [changedWedgesOnly, setChangedWedgesOnly] = useState(() => parseBooleanSearchParam(searchParams.get('changed_wedges_only')))
  const [vendorAlertThreshold, setVendorAlertThreshold] = useState(searchParams.get('vendor_alert_threshold')?.trim() || '')
  const [accountAlertThreshold, setAccountAlertThreshold] = useState(searchParams.get('account_alert_threshold')?.trim() || '')
  const [staleDaysThreshold, setStaleDaysThreshold] = useState(searchParams.get('stale_days_threshold')?.trim() || '')
  const [alertEmailEnabled, setAlertEmailEnabled] = useState(false)
  const [alertDeliveryFrequency, setAlertDeliveryFrequency] = useState<'daily' | 'weekly'>('daily')
  const [trackMode, setTrackMode] = useState<'own' | 'competitor'>('competitor')
  const [label, setLabel] = useState('')
  const [submittingVendor, setSubmittingVendor] = useState<string | null>(null)
  const [removingVendor, setRemovingVendor] = useState<string | null>(null)
  const [savingWatchlistView, setSavingWatchlistView] = useState(false)
  const [deletingWatchlistViewId, setDeletingWatchlistViewId] = useState<string | null>(null)
  const [evaluatingWatchlistViewId, setEvaluatingWatchlistViewId] = useState<string | null>(null)
  const [emailingWatchlistViewId, setEmailingWatchlistViewId] = useState<string | null>(null)
  const [savingCompetitiveSet, setSavingCompetitiveSet] = useState(false)
  const [runningCompetitiveSetId, setRunningCompetitiveSetId] = useState<string | null>(null)
  const [previewingCompetitiveSetId, setPreviewingCompetitiveSetId] = useState<string | null>(null)
  const [openCompetitiveSetPreviewId, setOpenCompetitiveSetPreviewId] = useState<string | null>(null)
  const [deletingCompetitiveSetId, setDeletingCompetitiveSetId] = useState<string | null>(null)
  const [pendingDestructiveAction, setPendingDestructiveAction] = useState<PendingWatchlistDestructiveAction | null>(null)
  const [pendingDestructiveError, setPendingDestructiveError] = useState<string | null>(null)
  const [editingCompetitiveSetId, setEditingCompetitiveSetId] = useState<string | null>(null)
  const [competitiveSetPreviews, setCompetitiveSetPreviews] = useState<Record<string, CompetitiveSetPlan>>({})
  const [competitiveSetRuns, setCompetitiveSetRuns] = useState<Record<string, CompetitiveSetRun[]>>({})
  const [competitiveSetLastRunOverrides, setCompetitiveSetLastRunOverrides] = useState<Record<string, CompetitiveSetLastRunOverride>>({})
  const [competitiveSetChangedOnly, setCompetitiveSetChangedOnly] = useState<Record<string, boolean>>({})
  const [competitiveSetForceRun, setCompetitiveSetForceRun] = useState<Record<string, boolean>>({})
  const [competitiveSetForceCrossVendor, setCompetitiveSetForceCrossVendor] = useState<Record<string, boolean>>({})
  const [actionMessage, setActionMessage] = useState<string | null>(null)
  const [actionError, setActionError] = useState<string | null>(null)
  const [selectedAccount, setSelectedAccount] = useState<AccountsInMotionFeedItem | null>(null)
  const [showReviewAccounts, setShowReviewAccounts] = useState(false)
  const [generatingCampaignFor, setGeneratingCampaignFor] = useState<string | null>(null)

  async function handleGenerateCampaign(item: AccountsInMotionFeedItem) {
    const key = `${item.company}::${item.vendor}`
    setGeneratingCampaignFor(key)
    setActionMessage(null)
    setActionError(null)
    try {
      const result = await generateCampaigns({
        company_name: item.company ?? undefined,
        vendor_name: item.vendor,
        limit: 5,
      })
      setActionMessage(`Generated ${result.generated ?? 0} campaign(s) for ${item.company ?? item.vendor}`)
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Campaign generation failed')
    } finally {
      setGeneratingCampaignFor(null)
    }
  }

  function openDestructiveAction(action: PendingWatchlistDestructiveAction) {
    setPendingDestructiveAction(action)
    setPendingDestructiveError(null)
    setActionMessage(null)
    setActionError(null)
  }

  function closeDestructiveAction() {
    if (deletingWatchlistViewId || removingVendor || deletingCompetitiveSetId) return
    setPendingDestructiveAction(null)
    setPendingDestructiveError(null)
  }

  const [competitiveSetName, setCompetitiveSetName] = useState('')
  const [competitiveSetFocal, setCompetitiveSetFocal] = useState('')
  const [competitiveSetCompetitors, setCompetitiveSetCompetitors] = useState<string[]>([])
  const [competitiveSetRefreshMode, setCompetitiveSetRefreshMode] = useState<'manual' | 'scheduled'>('manual')
  const [competitiveSetRefreshHours, setCompetitiveSetRefreshHours] = useState('')
  const [competitiveSetActive, setCompetitiveSetActive] = useState(true)
  const [competitiveSetVendorEnabled, setCompetitiveSetVendorEnabled] = useState(true)
  const [competitiveSetPairwiseEnabled, setCompetitiveSetPairwiseEnabled] = useState(true)
  const [competitiveSetCategoryEnabled, setCompetitiveSetCategoryEnabled] = useState(false)
  const [competitiveSetAsymmetryEnabled, setCompetitiveSetAsymmetryEnabled] = useState(false)

  useEffect(() => {
    const timer = setTimeout(() => {
      setDebouncedSearch(searchInput.trim())
    }, SEARCH_DEBOUNCE_MS)
    return () => clearTimeout(timer)
  }, [searchInput])

  const { data, loading, error, refresh, refreshing } = useApiData<WatchlistsData>(
    async () => {
      const slowBurnParams = {
        vendor_names: selectedVendorFilters.length ? selectedVendorFilters : undefined,
        category: selectedCategoryFilter || undefined,
        vendor_alert_threshold: vendorAlertThreshold ? Number(vendorAlertThreshold) : undefined,
        stale_days_threshold: staleDaysThreshold ? Number(staleDaysThreshold) : undefined,
      }
      const accountsParams = {
        vendor_names: selectedVendorFilters.length ? selectedVendorFilters : undefined,
        category: selectedCategoryFilter || undefined,
        source: selectedSourceFilter || undefined,
        min_urgency: selectedMinUrgency ? Number(selectedMinUrgency) : undefined,
        include_stale: freshOnly ? false : undefined,
        account_alert_threshold: accountAlertThreshold ? Number(accountAlertThreshold) : undefined,
        stale_days_threshold: staleDaysThreshold ? Number(staleDaysThreshold) : undefined,
      }
      const [tracked, trackedSets, savedViews, feedResult, accountsResult] = await Promise.all([
        listTrackedVendors(),
        listCompetitiveSets(true),
        listWatchlistViews(),
        fetchSlowBurnWatchlist(
          Object.values(slowBurnParams).some((value) => value !== undefined)
            ? slowBurnParams
            : undefined,
        ).then(
          (value) => ({ ok: true as const, value }),
          (error) => ({ ok: false as const, error }),
        ),
        fetchAccountsInMotionFeed(
          Object.values(accountsParams).some((value) => value !== undefined)
            ? accountsParams
            : undefined,
        ).then(
          (value) => ({ ok: true as const, value }),
          (error) => ({ ok: false as const, error }),
        ),
      ])
      const loadWarnings: string[] = []
      const feed = feedResult.ok
        ? feedResult.value
        : (
            loadWarnings.push(toLoadWarning('Vendor movement feed', feedResult.error)),
            {
              signals: [],
              count: 0,
              vendor_alert_threshold: null,
              vendor_alert_hit_count: 0,
              stale_days_threshold: null,
              stale_threshold_hit_count: 0,
            }
          )
      const accounts = accountsResult.ok
        ? accountsResult.value
        : (
            loadWarnings.push(toLoadWarning('Accounts in motion', accountsResult.error)),
            {
              accounts: [],
              count: 0,
              tracked_vendor_count: 0,
              vendors_with_accounts: 0,
              min_urgency: 7,
              per_vendor_limit: 10,
              freshest_report_date: null,
              account_alert_hit_count: 0,
              stale_threshold_hit_count: 0,
            }
          )
      return {
        vendors: tracked.vendors,
        watchlistViews: savedViews.views,
        competitiveSets: trackedSets.competitive_sets,
        competitiveSetDefaults: trackedSets.defaults ?? null,
        feed: feed.signals,
        vendorAlertHitCount: feed.vendor_alert_hit_count ?? 0,
        feedStaleThresholdHitCount: feed.stale_threshold_hit_count ?? 0,
        accounts: accounts.accounts,
        accountAlertHitCount: accounts.account_alert_hit_count ?? 0,
        accountStaleThresholdHitCount: accounts.stale_threshold_hit_count ?? 0,
        vendorsWithAccounts: accounts.vendors_with_accounts,
        freshestAccountsReportDate: accounts.freshest_report_date,
        loadWarnings,
      }
    },
    [
      selectedVendorFilter,
      selectedVendorFilters,
      selectedCategoryFilter,
      selectedSourceFilter,
      selectedMinUrgency,
      freshOnly,
      vendorAlertThreshold,
      accountAlertThreshold,
      staleDaysThreshold,
    ],
  )

  const {
    data: searchResults,
    loading: searchLoading,
  } = useApiData<{ vendors: VendorSearchResult[]; count: number }>(
    async () => {
      if (debouncedSearch.length < MIN_VENDOR_SEARCH_CHARS) return { vendors: [], count: 0 }
      return searchAvailableVendors(debouncedSearch)
    },
    [debouncedSearch],
    { refreshOnFocus: false, refreshOnReconnect: false },
  )

  const trackedVendors = useMemo(() => data?.vendors ?? [], [data?.vendors])
  const watchlistViews = useMemo(() => data?.watchlistViews ?? [], [data?.watchlistViews])
  const competitiveSets = data?.competitiveSets ?? []
  const competitiveSetDefaults = data?.competitiveSetDefaults ?? null
  const feed = useMemo(() => data?.feed ?? [], [data?.feed])
  const vendorAlertHitCountFromApi = data?.vendorAlertHitCount ?? 0
  const feedStaleThresholdHitCountFromApi = data?.feedStaleThresholdHitCount ?? 0
  const accounts = useMemo(() => data?.accounts ?? [], [data?.accounts])
  const accountAlertHitCountFromApi = data?.accountAlertHitCount ?? 0
  const accountStaleThresholdHitCountFromApi = data?.accountStaleThresholdHitCount ?? 0
  const vendorsWithAccounts = data?.vendorsWithAccounts ?? 0
  useEffect(() => {
    setCompetitiveSetLastRunOverrides((current) => {
      let changed = false
      const next = { ...current }
      for (const [competitiveSetId, override] of Object.entries(current)) {
        const backendSet = competitiveSets.find((item) => item.id === competitiveSetId)
        const backendTs = toTimestamp(backendSet?.last_run_at ?? null)
        const overrideTs = toTimestamp(override.last_run_at)
        if (!backendSet || backendTs == null || overrideTs == null) continue
        if (backendTs >= overrideTs && backendSet.last_run_status) {
          delete next[competitiveSetId]
          changed = true
        }
      }
      return changed ? next : current
    })
  }, [competitiveSets])
  const trackedNames = useMemo(
    () => new Set(trackedVendors.map((vendor) => vendor.vendor_name.toLowerCase())),
    [trackedVendors],
  )
  const candidateResults = (searchResults?.vendors ?? []).filter(
    (vendor) => !trackedNames.has(vendor.vendor_name.toLowerCase()),
  )
  const freshestAccountsReportDate = data?.freshestAccountsReportDate ?? null
  const focalOptions = trackedVendors
  const competitorOptions = trackedVendors.filter(
    (vendor) => vendor.vendor_name !== competitiveSetFocal,
  )
  const vendorFilterOptions = useMemo(
    () => Array.from(new Set([
      ...trackedVendors.map((vendor) => vendor.vendor_name),
      selectedVendorFilter,
    ].filter(Boolean))).sort((a, b) => a.localeCompare(b)),
    [selectedVendorFilter, trackedVendors],
  )
  const categoryFilterOptions = useMemo(
    () => Array.from(new Set([
      ...feed.map((row) => row.product_category || ''),
      ...accounts.map((row) => row.category || ''),
      selectedCategoryFilter,
    ].filter(Boolean))).sort((a, b) => a.localeCompare(b)),
    [accounts, feed, selectedCategoryFilter],
  )
  const sourceFilterOptions = useMemo(
    () => Array.from(new Set([
      ...accounts.flatMap((row) => Object.keys(row.source_distribution || {})),
      selectedSourceFilter,
    ].filter(Boolean))).sort((a, b) => a.localeCompare(b)),
    [accounts, selectedSourceFilter],
  )
  const currentEvidenceVendor = useMemo(() => {
    if (selectedVendorFilters.length === 1) return selectedVendorFilters[0]
    if (selectedVendorFilter) return selectedVendorFilter
    return ''
  }, [selectedVendorFilter, selectedVendorFilters])
  const currentViewFilters = useMemo(
    () => ({
      vendor_name: selectedVendorFilter,
      vendor_names: selectedVendorFilters,
      category: selectedCategoryFilter,
      source: selectedSourceFilter,
      min_urgency: selectedMinUrgency,
      fresh_only: freshOnly,
      named_accounts_only: namedAccountsOnly,
      changed_wedges_only: changedWedgesOnly,
      vendor_alert_threshold: vendorAlertThreshold,
      account_alert_threshold: accountAlertThreshold,
      stale_days_threshold: staleDaysThreshold,
      alert_email_enabled: alertEmailEnabled,
      alert_delivery_frequency: alertDeliveryFrequency,
    }),
    [
      selectedVendorFilter,
      selectedVendorFilters,
      selectedCategoryFilter,
      selectedSourceFilter,
      selectedMinUrgency,
      freshOnly,
      namedAccountsOnly,
      changedWedgesOnly,
      vendorAlertThreshold,
      accountAlertThreshold,
      staleDaysThreshold,
      alertEmailEnabled,
      alertDeliveryFrequency,
    ],
  )
  const activeVendorAlertThreshold = parseOptionalNumber(vendorAlertThreshold)
  const activeAccountAlertThreshold = parseOptionalNumber(accountAlertThreshold)
  const activeStaleDaysThreshold = parseOptionalNumber(staleDaysThreshold)
  const activeWatchlistView = useMemo(
    () => watchlistViews.find((view) => watchlistViewMatchesState(view, currentViewFilters)) ?? null,
    [currentViewFilters, watchlistViews],
  )
  const requestedWatchlistViewId = searchParams.get('view')?.trim() || ''
  const requestedWatchlistView = useMemo(
    () => (requestedWatchlistViewId
      ? watchlistViews.find((view) => view.id === requestedWatchlistViewId) ?? null
      : null),
    [requestedWatchlistViewId, watchlistViews],
  )
  const initialRequestedWatchlistViewIdRef = useRef(requestedWatchlistViewId)
  const outboundWatchlistSearchParams = useMemo(() => {
    const next = new URLSearchParams(searchParams)
    const durableViewId = initialRequestedWatchlistViewIdRef.current || requestedWatchlistViewId || activeWatchlistView?.id || ''
    if (durableViewId) {
      next.set('view', durableViewId)
    } else {
      next.delete('view')
    }
    if (selectedVendorFilter.trim()) {
      next.set('vendor_name', selectedVendorFilter.trim())
    } else {
      next.delete('vendor_name')
    }
    if (selectedSourceFilter.trim()) {
      next.set('source', selectedSourceFilter.trim())
    } else {
      next.delete('source')
    }
    if (selectedCategoryFilter.trim()) {
      next.set('category', selectedCategoryFilter.trim())
    } else {
      next.delete('category')
    }
    if (selectedMinUrgency.trim()) {
      next.set('min_urgency', selectedMinUrgency.trim())
    } else {
      next.delete('min_urgency')
    }
    if (freshOnly) {
      next.set('fresh_only', 'true')
    } else {
      next.delete('fresh_only')
    }
    if (namedAccountsOnly) {
      next.set('named_accounts_only', 'true')
    } else {
      next.delete('named_accounts_only')
    }
    if (changedWedgesOnly) {
      next.set('changed_wedges_only', 'true')
    } else {
      next.delete('changed_wedges_only')
    }
    if (vendorAlertThreshold.trim()) {
      next.set('vendor_alert_threshold', vendorAlertThreshold.trim())
    } else {
      next.delete('vendor_alert_threshold')
    }
    if (accountAlertThreshold.trim()) {
      next.set('account_alert_threshold', accountAlertThreshold.trim())
    } else {
      next.delete('account_alert_threshold')
    }
    if (staleDaysThreshold.trim()) {
      next.set('stale_days_threshold', staleDaysThreshold.trim())
    } else {
      next.delete('stale_days_threshold')
    }
    return next
  }, [
    accountAlertThreshold,
    activeWatchlistView?.id,
    changedWedgesOnly,
    freshOnly,
    namedAccountsOnly,
    requestedWatchlistViewId,
    searchParams,
    selectedCategoryFilter,
    selectedMinUrgency,
    selectedSourceFilter,
    selectedVendorFilter,
    staleDaysThreshold,
    vendorAlertThreshold,
  ])
  const currentAlertsPath = currentEvidenceVendor
    ? watchlistVendorAlertsPath(outboundWatchlistSearchParams, currentEvidenceVendor)
    : watchlistAlertsPath(outboundWatchlistSearchParams)
  const selectedAccountSearchParams = useMemo(
    () => (selectedAccount
      ? accountFocusParams(outboundWatchlistSearchParams, selectedAccount)
      : outboundWatchlistSearchParams),
    [outboundWatchlistSearchParams, selectedAccount],
  )
  const {
    data: activeAlertEventsData,
    refresh: refreshAlertEvents,
    refreshing: refreshingAlertEvents,
  } = useApiData(
    async () => {
      if (!activeWatchlistView) {
        return {
          watchlist_view_id: '',
          watchlist_view_name: '',
          status: 'open' as const,
          events: [] as WatchlistAlertEvent[],
          count: 0,
        }
      }
      return listWatchlistAlertEvents(activeWatchlistView.id, {
        status: 'open',
        limit: 25,
      })
    },
    [activeWatchlistView?.id],
    { refreshOnFocus: false, refreshOnReconnect: false },
  )
  const {
    data: activeAlertEmailLogData,
    refresh: refreshAlertEmailLog,
    refreshing: refreshingAlertEmailLog,
  } = useApiData(
    async () => {
      if (!activeWatchlistView) {
        return {
          watchlist_view_id: '',
          watchlist_view_name: '',
          deliveries: [],
          count: 0,
        }
      }
      return listWatchlistAlertEmailLog(activeWatchlistView.id, { limit: 5 })
    },
    [activeWatchlistView?.id],
    { refreshOnFocus: false, refreshOnReconnect: false },
  )
  const filteredFeed = useMemo(
    () => (changedWedgesOnly ? feed.filter((row) => Boolean(row.reasoning_delta?.wedge_changed)) : feed),
    [changedWedgesOnly, feed],
  )
  const hiddenFeedCount = Math.max(feed.length - filteredFeed.length, 0)
  const vendorAlertHitCount = useMemo(
    () => filteredFeed.some((row) => row.vendor_alert_hit !== undefined)
      ? filteredFeed.filter((row) => Boolean(row.vendor_alert_hit)).length
      : vendorAlertHitCountFromApi || filteredFeed.filter((row) => vendorAlertTriggered(row, activeVendorAlertThreshold)).length,
    [activeVendorAlertThreshold, filteredFeed, vendorAlertHitCountFromApi],
  )
  const accountBuckets = useMemo(() => {
    const high: AccountsInMotionFeedItem[] = []
    const medium: AccountsInMotionFeedItem[] = []
    const review: AccountsInMotionFeedItem[] = []
    for (const account of accounts) {
      const tier = accountPresentationTier(account)
      if (tier === 'named_high') {
        high.push(account)
      } else if (tier === 'named_medium') {
        medium.push(account)
      } else {
        review.push(account)
      }
    }
    return {
      high,
      medium,
      review,
      primary: [...high, ...medium],
    }
  }, [accounts])
  const visibleReviewAccounts = useMemo(
    () => (namedAccountsOnly ? [] : accountBuckets.review),
    [accountBuckets.review, namedAccountsOnly],
  )
  const hiddenReviewAccountCount = Math.max(accountBuckets.review.length - visibleReviewAccounts.length, 0)
  const visibleAccounts = useMemo(
    () => [...accountBuckets.primary, ...visibleReviewAccounts],
    [accountBuckets.primary, visibleReviewAccounts],
  )
  const accountAlertHitCount = useMemo(
    () => visibleAccounts.some((row) => row.account_alert_hit !== undefined)
      ? visibleAccounts.filter((row) => Boolean(row.account_alert_hit)).length
      : accountAlertHitCountFromApi || visibleAccounts.filter((row) => accountAlertTriggered(row, activeAccountAlertThreshold)).length,
    [activeAccountAlertThreshold, accountAlertHitCountFromApi, visibleAccounts],
  )
  const vendorStaleThresholdHitCount = useMemo(
    () => filteredFeed.some((row) => row.stale_threshold_hit !== undefined)
      ? filteredFeed.filter((row) => Boolean(row.stale_threshold_hit)).length
      : feedStaleThresholdHitCountFromApi || filteredFeed.filter((row) => staleThresholdTriggered(activeStaleDaysThreshold, {
        freshnessTimestamp: row.freshness_timestamp,
        fallbackTimestamp: row.last_computed_at,
      })).length,
    [activeStaleDaysThreshold, feedStaleThresholdHitCountFromApi, filteredFeed],
  )
  const accountStaleThresholdHitCount = useMemo(
    () => visibleAccounts.some((row) => row.stale_threshold_hit !== undefined)
      ? visibleAccounts.filter((row) => Boolean(row.stale_threshold_hit)).length
      : accountStaleThresholdHitCountFromApi || visibleAccounts.filter((row) => staleThresholdTriggered(activeStaleDaysThreshold, {
        staleDays: row.stale_days,
        freshnessTimestamp: row.freshness_timestamp,
        fallbackTimestamp: row.report_date,
      })).length,
    [accountStaleThresholdHitCountFromApi, activeStaleDaysThreshold, visibleAccounts],
  )
  const staleThresholdHitCount = vendorStaleThresholdHitCount + accountStaleThresholdHitCount
  const activeAlertEvents = activeAlertEventsData?.events ?? []
  const activeAlertEmailDeliveries = activeAlertEmailLogData?.deliveries ?? []
  const activeAlertEventsEmptyState = useMemo(() => {
    if (!activeWatchlistView) return 'No open persisted alert events for this saved view yet.'
    const localSuppression = summarizeLocalWatchlistSuppression(activeWatchlistView)
    return localSuppression
      ? `No open persisted alert events for this saved view yet. Local filters: ${localSuppression}.`
      : 'No open persisted alert events for this saved view yet.'
  }, [activeWatchlistView])
  const requestedAccountFocus = useMemo(
    () => ({
      vendor: searchParams.get('account_vendor')?.trim() || '',
      company: searchParams.get('account_company')?.trim() || '',
      report_date: searchParams.get('account_report_date')?.trim() || '',
      watch_vendor: searchParams.get('account_watch_vendor')?.trim() || '',
      category: searchParams.get('account_category')?.trim() || '',
      track_mode: searchParams.get('account_track_mode')?.trim() || '',
    }),
    [searchParams],
  )
  const hasRequestedAccountFocus = Object.values(requestedAccountFocus).some(Boolean)
  const requestedAccount = useMemo(
    () => (hasRequestedAccountFocus
      ? visibleAccounts.find((account) => accountFocusMatchesRow(account, requestedAccountFocus)) ?? null
      : null),
    [hasRequestedAccountFocus, requestedAccountFocus, visibleAccounts],
  )
  const requestedWitnessId = searchParams.get('witness_id')?.trim() || ''
  const requestedWitnessVendor = searchParams.get('witness_vendor')?.trim() || ''
  const hasRequestedWitnessFocus = Boolean(requestedWitnessId && requestedWitnessVendor)
  const hasActiveAlertPolicy = (
    activeVendorAlertThreshold != null
    || activeAccountAlertThreshold != null
    || activeStaleDaysThreshold != null
  )
  const hasActiveFeedFilters = [
    selectedVendorFilter,
    selectedCategoryFilter,
    selectedSourceFilter,
    selectedMinUrgency,
    freshOnly,
    namedAccountsOnly,
    changedWedgesOnly,
    vendorAlertThreshold,
    accountAlertThreshold,
    staleDaysThreshold,
  ].some(Boolean)

  useEffect(() => {
    if (!selectedAccount) return
    const currentAccount = visibleAccounts.find((account) => isSameAccountMovementRow(account, selectedAccount))
    if (!currentAccount) {
      setSelectedAccount(null)
      return
    }
    if (currentAccount !== selectedAccount) {
      setSelectedAccount(currentAccount)
    }
  }, [selectedAccount, visibleAccounts])

  useEffect(() => {
    if (!hasRequestedAccountFocus || !requestedAccount) return
    if (selectedAccount && isSameAccountMovementRow(selectedAccount, requestedAccount)) return
    setSelectedAccount(requestedAccount)
  }, [hasRequestedAccountFocus, requestedAccount, selectedAccount])

  useEffect(() => {
    if (!hasRequestedWitnessFocus) return
    if (
      evidenceDrawerOpen
      && evidenceDrawerWitnessId === requestedWitnessId
      && evidenceDrawerVendor === requestedWitnessVendor
    ) {
      return
    }
    setEvidenceDrawerWitnessId(requestedWitnessId)
    setEvidenceDrawerVendor(requestedWitnessVendor)
    setEvidenceDrawerOpen(true)
  }, [
    evidenceDrawerOpen,
    evidenceDrawerVendor,
    evidenceDrawerWitnessId,
    hasRequestedWitnessFocus,
    requestedWitnessId,
    requestedWitnessVendor,
  ])

  useEffect(() => {
    if (loading) return
    if (requestedAccount && !selectedAccount) return
    const nextFocus = selectedAccount ? accountFocusFromRow(selectedAccount) : null
    const currentFocus = {
      vendor: searchParams.get('account_vendor')?.trim() || '',
      company: searchParams.get('account_company')?.trim() || '',
      report_date: searchParams.get('account_report_date')?.trim() || '',
      watch_vendor: searchParams.get('account_watch_vendor')?.trim() || '',
      category: searchParams.get('account_category')?.trim() || '',
      track_mode: searchParams.get('account_track_mode')?.trim() || '',
    }
    if (
      (nextFocus?.vendor || '') === currentFocus.vendor
      && (nextFocus?.company || '') === currentFocus.company
      && (nextFocus?.report_date || '') === currentFocus.report_date
      && (nextFocus?.watch_vendor || '') === currentFocus.watch_vendor
      && (nextFocus?.category || '') === currentFocus.category
      && (nextFocus?.track_mode || '') === currentFocus.track_mode
    ) {
      return
    }
    setSearchParams((current) => {
      const next = new URLSearchParams(current)
      const accountKeys = [
        'account_vendor',
        'account_company',
        'account_report_date',
        'account_watch_vendor',
        'account_category',
        'account_track_mode',
      ]
      for (const key of accountKeys) next.delete(key)
      if (nextFocus) {
        next.set('account_vendor', nextFocus.vendor)
        next.set('account_company', nextFocus.company)
        next.set('account_report_date', nextFocus.report_date)
        next.set('account_watch_vendor', nextFocus.watch_vendor)
        next.set('account_category', nextFocus.category)
        next.set('account_track_mode', nextFocus.track_mode)
      }
      return next
    }, { replace: true })
  }, [
    hasRequestedAccountFocus,
    loading,
    requestedAccount,
    searchParams,
    selectedAccount,
    setSearchParams,
  ])

  useEffect(() => {
    const currentWitnessId = searchParams.get('witness_id')?.trim() || ''
    const currentWitnessVendor = searchParams.get('witness_vendor')?.trim() || ''
    const nextWitnessId = evidenceDrawerOpen ? (evidenceDrawerWitnessId || '') : ''
    const nextWitnessVendor = evidenceDrawerOpen ? evidenceDrawerVendor.trim() : ''
    if (currentWitnessId === nextWitnessId && currentWitnessVendor === nextWitnessVendor) return
    setSearchParams((current) => {
      const next = new URLSearchParams(current)
      next.delete('witness_id')
      next.delete('witness_vendor')
      if (nextWitnessId && nextWitnessVendor) {
        next.set('witness_id', nextWitnessId)
        next.set('witness_vendor', nextWitnessVendor)
      }
      return next
    }, { replace: true })
  }, [
    evidenceDrawerOpen,
    evidenceDrawerVendor,
    evidenceDrawerWitnessId,
    searchParams,
    setSearchParams,
  ])

  useEffect(() => {
    if (editingCompetitiveSetId) return
    if (competitiveSetRefreshHours !== '') return
    const defaultHours = competitiveSetDefaults?.default_refresh_interval_hours
    if (!defaultHours) return
    setCompetitiveSetRefreshHours(String(defaultHours))
  }, [
    competitiveSetDefaults?.default_refresh_interval_hours,
    competitiveSetRefreshHours,
    editingCompetitiveSetId,
  ])

  function resetCompetitiveSetForm() {
    setEditingCompetitiveSetId(null)
    setCompetitiveSetName('')
    setCompetitiveSetFocal('')
    setCompetitiveSetCompetitors([])
    setCompetitiveSetRefreshMode('manual')
    setCompetitiveSetRefreshHours(String(competitiveSetDefaults?.default_refresh_interval_hours ?? ''))
    setCompetitiveSetActive(true)
    setCompetitiveSetVendorEnabled(true)
    setCompetitiveSetPairwiseEnabled(true)
    setCompetitiveSetCategoryEnabled(false)
    setCompetitiveSetAsymmetryEnabled(false)
  }

  const applyWatchlistView = useCallback((view: WatchlistView) => {
    const vendorNames = watchlistViewVendorNames(view)
    setSelectedVendorFilter(vendorNames[0] || '')
    setSelectedVendorFilters(vendorNames)
    setSelectedCategoryFilter(view.category || '')
    setSelectedSourceFilter(view.source || '')
    setSelectedMinUrgency(view.min_urgency != null ? String(view.min_urgency) : '')
    setFreshOnly(!view.include_stale)
    setNamedAccountsOnly(view.named_accounts_only)
    setChangedWedgesOnly(view.changed_wedges_only)
    setVendorAlertThreshold(view.vendor_alert_threshold != null ? String(view.vendor_alert_threshold) : '')
    setAccountAlertThreshold(view.account_alert_threshold != null ? String(view.account_alert_threshold) : '')
    setStaleDaysThreshold(view.stale_days_threshold != null ? String(view.stale_days_threshold) : '')
    setAlertEmailEnabled(view.alert_email_enabled)
    setAlertDeliveryFrequency(view.alert_delivery_frequency || 'daily')
    setSavedViewName(view.name)
  }, [])

  useEffect(() => {
    if (!requestedWatchlistView) return
    if (activeWatchlistView?.id === requestedWatchlistView.id) return
    applyWatchlistView(requestedWatchlistView)
  }, [activeWatchlistView?.id, applyWatchlistView, requestedWatchlistView])

  useEffect(() => {
    if (!requestedVendorName || requestedWatchlistView) return
    if (
      selectedVendorFilters.length === 1
      && selectedVendorFilters[0] === requestedVendorName
      && selectedVendorFilter === requestedVendorName
    ) {
      return
    }
    setSelectedVendorFilter(requestedVendorName)
    setSelectedVendorFilters([requestedVendorName])
  }, [requestedVendorName, requestedWatchlistView, selectedVendorFilter, selectedVendorFilters])

  useEffect(() => {
    if (loading) return
    if (requestedWatchlistView && requestedWatchlistView.id !== activeWatchlistView?.id) return
    const currentViewId = searchParams.get('view')?.trim() || ''
    const nextViewId = activeWatchlistView?.id || ''
    if (currentViewId === nextViewId) return
    setSearchParams((current) => {
      const next = new URLSearchParams(current)
      if (nextViewId) {
        next.set('view', nextViewId)
      } else {
        next.delete('view')
      }
      return next
    }, { replace: true })
  }, [
    activeWatchlistView?.id,
    loading,
    requestedWatchlistView,
    searchParams,
    setSearchParams,
  ])

  useEffect(() => {
    if (loading) return
    if (requestedWatchlistView && requestedWatchlistView.id !== activeWatchlistView?.id) return
    const currentVendorName = searchParams.get('vendor_name')?.trim() || ''
    const nextVendorName = activeWatchlistView
      ? ''
      : (selectedVendorFilters.length === 1 ? selectedVendorFilters[0] : selectedVendorFilter).trim()
    if (currentVendorName === nextVendorName) return
    setSearchParams((current) => {
      const next = new URLSearchParams(current)
      if (nextVendorName) {
        next.set('vendor_name', nextVendorName)
      } else {
        next.delete('vendor_name')
      }
      return next
    }, { replace: true })
  }, [
    activeWatchlistView,
    loading,
    requestedWatchlistView,
    searchParams,
    selectedVendorFilter,
    selectedVendorFilters,
    setSearchParams,
  ])

  useEffect(() => {
    if (loading) return
    if (requestedWatchlistView && requestedWatchlistView.id !== activeWatchlistView?.id) return
    const currentSource = searchParams.get('source')?.trim() || ''
    const nextSource = selectedSourceFilter.trim()
    if (currentSource === nextSource) return
    setSearchParams((current) => {
      const next = new URLSearchParams(current)
      if (nextSource) {
        next.set('source', nextSource)
      } else {
        next.delete('source')
      }
      return next
    }, { replace: true })
  }, [
    activeWatchlistView?.id,
    loading,
    requestedWatchlistView,
    searchParams,
    selectedSourceFilter,
    setSearchParams,
  ])

  useEffect(() => {
    if (loading) return
    if (requestedWatchlistView && requestedWatchlistView.id !== activeWatchlistView?.id) return
    const currentCategory = searchParams.get('category')?.trim() || ''
    const nextCategory = selectedCategoryFilter.trim()
    if (currentCategory === nextCategory) return
    setSearchParams((current) => {
      const next = new URLSearchParams(current)
      if (nextCategory) {
        next.set('category', nextCategory)
      } else {
        next.delete('category')
      }
      return next
    }, { replace: true })
  }, [
    activeWatchlistView?.id,
    loading,
    requestedWatchlistView,
    searchParams,
    selectedCategoryFilter,
    setSearchParams,
  ])


  useEffect(() => {
    if (loading) return
    if (requestedWatchlistView && requestedWatchlistView.id !== activeWatchlistView?.id) return
    const currentMinUrgency = searchParams.get('min_urgency')?.trim() || ''
    const nextMinUrgency = selectedMinUrgency.trim()
    if (currentMinUrgency === nextMinUrgency) return
    setSearchParams((current) => {
      const next = new URLSearchParams(current)
      if (nextMinUrgency) {
        next.set('min_urgency', nextMinUrgency)
      } else {
        next.delete('min_urgency')
      }
      return next
    }, { replace: true })
  }, [
    activeWatchlistView?.id,
    loading,
    requestedWatchlistView,
    searchParams,
    selectedMinUrgency,
    setSearchParams,
  ])

  useEffect(() => {
    if (loading) return
    if (requestedWatchlistView && requestedWatchlistView.id !== activeWatchlistView?.id) return
    const currentFreshOnly = parseBooleanSearchParam(searchParams.get('fresh_only'))
    if (currentFreshOnly === freshOnly) return
    setSearchParams((current) => {
      const next = new URLSearchParams(current)
      if (freshOnly) {
        next.set('fresh_only', 'true')
      } else {
        next.delete('fresh_only')
      }
      return next
    }, { replace: true })
  }, [
    activeWatchlistView?.id,
    freshOnly,
    loading,
    requestedWatchlistView,
    searchParams,
    setSearchParams,
  ])


  useEffect(() => {
    if (loading) return
    if (requestedWatchlistView && requestedWatchlistView.id !== activeWatchlistView?.id) return
    const currentNamedAccountsOnly = parseBooleanSearchParam(searchParams.get('named_accounts_only'))
    if (currentNamedAccountsOnly === namedAccountsOnly) return
    setSearchParams((current) => {
      const next = new URLSearchParams(current)
      if (namedAccountsOnly) {
        next.set('named_accounts_only', 'true')
      } else {
        next.delete('named_accounts_only')
      }
      return next
    }, { replace: true })
  }, [
    activeWatchlistView?.id,
    loading,
    namedAccountsOnly,
    requestedWatchlistView,
    searchParams,
    setSearchParams,
  ])

  useEffect(() => {
    if (loading) return
    if (requestedWatchlistView && requestedWatchlistView.id !== activeWatchlistView?.id) return
    const currentChangedWedgesOnly = parseBooleanSearchParam(searchParams.get('changed_wedges_only'))
    if (currentChangedWedgesOnly === changedWedgesOnly) return
    setSearchParams((current) => {
      const next = new URLSearchParams(current)
      if (changedWedgesOnly) {
        next.set('changed_wedges_only', 'true')
      } else {
        next.delete('changed_wedges_only')
      }
      return next
    }, { replace: true })
  }, [
    activeWatchlistView?.id,
    changedWedgesOnly,
    loading,
    requestedWatchlistView,
    searchParams,
    setSearchParams,
  ])


  useEffect(() => {
    if (loading) return
    if (requestedWatchlistView && requestedWatchlistView.id !== activeWatchlistView?.id) return
    const currentVendorAlertThreshold = searchParams.get('vendor_alert_threshold')?.trim() || ''
    const nextVendorAlertThreshold = vendorAlertThreshold.trim()
    if (currentVendorAlertThreshold === nextVendorAlertThreshold) return
    setSearchParams((current) => {
      const next = new URLSearchParams(current)
      if (nextVendorAlertThreshold) {
        next.set('vendor_alert_threshold', nextVendorAlertThreshold)
      } else {
        next.delete('vendor_alert_threshold')
      }
      return next
    }, { replace: true })
  }, [
    activeWatchlistView?.id,
    loading,
    requestedWatchlistView,
    searchParams,
    setSearchParams,
    vendorAlertThreshold,
  ])

  useEffect(() => {
    if (loading) return
    if (requestedWatchlistView && requestedWatchlistView.id !== activeWatchlistView?.id) return
    const currentAccountAlertThreshold = searchParams.get('account_alert_threshold')?.trim() || ''
    const nextAccountAlertThreshold = accountAlertThreshold.trim()
    if (currentAccountAlertThreshold === nextAccountAlertThreshold) return
    setSearchParams((current) => {
      const next = new URLSearchParams(current)
      if (nextAccountAlertThreshold) {
        next.set('account_alert_threshold', nextAccountAlertThreshold)
      } else {
        next.delete('account_alert_threshold')
      }
      return next
    }, { replace: true })
  }, [
    accountAlertThreshold,
    activeWatchlistView?.id,
    loading,
    requestedWatchlistView,
    searchParams,
    setSearchParams,
  ])

  useEffect(() => {
    if (loading) return
    if (requestedWatchlistView && requestedWatchlistView.id !== activeWatchlistView?.id) return
    const currentStaleDaysThreshold = searchParams.get('stale_days_threshold')?.trim() || ''
    const nextStaleDaysThreshold = staleDaysThreshold.trim()
    if (currentStaleDaysThreshold === nextStaleDaysThreshold) return
    setSearchParams((current) => {
      const next = new URLSearchParams(current)
      if (nextStaleDaysThreshold) {
        next.set('stale_days_threshold', nextStaleDaysThreshold)
      } else {
        next.delete('stale_days_threshold')
      }
      return next
    }, { replace: true })
  }, [
    activeWatchlistView?.id,
    loading,
    requestedWatchlistView,
    searchParams,
    setSearchParams,
    staleDaysThreshold,
  ])

  async function handleCopyCurrentViewLink() {
    try {
      await navigator.clipboard.writeText(`${window.location.origin}${watchlistPath(searchParams)}`)
      setActionError(null)
      setActionMessage('Copied current view link')
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy current view link')
    }
  }

  async function handleCopyAlertsLink() {
    try {
      await navigator.clipboard.writeText(`${window.location.origin}${currentAlertsPath}`)
      setActionError(null)
      setActionMessage('Copied Alerts API link')
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy Alerts API link')
    }
  }

  async function handleCopyWatchlistViewLink(view: WatchlistView) {
    try {
      await navigator.clipboard.writeText(watchlistViewUrl(view.id))
      setActionError(null)
      setActionMessage(`Copied link for ${view.name}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : `Failed to copy link for ${view.name}`)
    }
  }

  async function handleCopyAlertEventAccountReviewLink(event: WatchlistAlertEvent) {
    const targetUrl = watchlistAlertEventAccountUrl(outboundWatchlistSearchParams, event)
    const eventLabel = event.company_name || event.vendor_name || 'alert event'
    if (!targetUrl) return
    try {
      await navigator.clipboard.writeText(targetUrl)
      setActionError(null)
      setActionMessage(`Copied account review link for ${eventLabel}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy account review link')
    }
  }

  async function handleCopyAlertEventReviewLink(event: WatchlistAlertEvent, reviewId: string) {
    const targetUrl = watchlistAlertEventReviewUrl(outboundWatchlistSearchParams, event, reviewId)
    const eventLabel = event.company_name || event.vendor_name || 'alert event'
    if (!targetUrl) return
    try {
      await navigator.clipboard.writeText(targetUrl)
      setActionError(null)
      setActionMessage(`Copied review link for ${eventLabel}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy review link')
    }
  }

  async function handleCopyAlertEventWitnessLink(
    event: WatchlistAlertEvent,
    witnessId: string,
    source?: string | null,
  ) {
    const targetUrl = watchlistAlertEventEvidenceUrl(outboundWatchlistSearchParams, event, witnessId, source)
    const eventLabel = event.company_name || watchlistAlertEventVendorName(event) || 'alert event'
    if (!targetUrl) return
    try {
      await navigator.clipboard.writeText(targetUrl)
      setActionError(null)
      setActionMessage(`Copied witness link for ${eventLabel}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy witness link')
    }
  }

  async function handleCopyAlertEventEvidenceLink(event: WatchlistAlertEvent, source?: string | null) {
    const targetUrl = watchlistAlertEventEvidenceUrl(outboundWatchlistSearchParams, event, null, source)
    const vendorName = watchlistAlertEventVendorName(event)
    if (!targetUrl || !vendorName) return
    try {
      await navigator.clipboard.writeText(targetUrl)
      setActionError(null)
      setActionMessage(`Copied evidence link for ${vendorName}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy evidence link')
    }
  }

  async function handleCopyAlertEventVendorLink(event: WatchlistAlertEvent) {
    const targetUrl = watchlistAlertEventVendorUrl(outboundWatchlistSearchParams, event)
    const vendorName = watchlistAlertEventVendorName(event)
    if (!targetUrl || !vendorName) return
    try {
      await navigator.clipboard.writeText(targetUrl)
      setActionError(null)
      setActionMessage(`Copied vendor workspace link for ${vendorName}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy vendor workspace link')
    }
  }

  async function handleCopyAlertEventAlertsLink(event: WatchlistAlertEvent) {
    const targetUrl = watchlistAlertEventAlertsUrl(outboundWatchlistSearchParams, event)
    const vendorName = watchlistAlertEventVendorName(event)
    if (!targetUrl || !vendorName) return
    try {
      await navigator.clipboard.writeText(targetUrl)
      setActionError(null)
      setActionMessage(`Copied Alerts API link for ${vendorName}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy Alerts API link')
    }
  }

  async function handleCopyAlertEventReportsLink(event: WatchlistAlertEvent) {
    const targetUrl = watchlistAlertEventReportsUrl(outboundWatchlistSearchParams, event)
    const vendorName = watchlistAlertEventVendorName(event)
    if (!targetUrl || !vendorName) return
    try {
      await navigator.clipboard.writeText(targetUrl)
      setActionError(null)
      setActionMessage(`Copied reports link for ${vendorName}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy reports link')
    }
  }

  async function handleCopyAlertEventOpportunitiesLink(event: WatchlistAlertEvent) {
    const targetUrl = watchlistAlertEventOpportunitiesUrl(outboundWatchlistSearchParams, event)
    const vendorName = watchlistAlertEventVendorName(event)
    if (!targetUrl || !vendorName) return
    try {
      await navigator.clipboard.writeText(targetUrl)
      setActionError(null)
      setActionMessage(`Copied opportunities link for ${vendorName}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy opportunities link')
    }
  }

  async function handleCopySelectedAccountLink() {
    if (!selectedAccount) return
    try {
      await navigator.clipboard.writeText(`${window.location.origin}${watchlistPath(selectedAccountSearchParams)}`)
      setActionError(null)
      setActionMessage(`Copied account link for ${selectedAccount.company || selectedAccount.vendor}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy account link')
    }
  }

  async function handleCopySelectedAlertsLink() {
    if (!selectedAccount) return
    try {
      await navigator.clipboard.writeText(
        watchlistVendorAlertsUrl(selectedAccountSearchParams, selectedAccount.vendor, selectedAccount.company),
      )
      setActionError(null)
      setActionMessage(`Copied Alerts API link for ${selectedAccount.company || selectedAccount.vendor}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy Alerts API link')
    }
  }

  async function handleCopyAccountRowLink(row: AccountsInMotionFeedItem) {
    try {
      await navigator.clipboard.writeText(watchlistAccountUrl(searchParams, row))
      setActionError(null)
      setActionMessage(`Copied account link for ${row.company || row.vendor}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy account link')
    }
  }

  async function handleCopyAccountRowVendorLink(row: AccountsInMotionFeedItem) {
    try {
      await navigator.clipboard.writeText(watchlistVendorWorkspaceUrl(outboundWatchlistSearchParams, row.vendor))
      setActionError(null)
      setActionMessage(`Copied vendor link for ${row.company || row.vendor}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy vendor link')
    }
  }

  async function handleCopyAccountRowAlertsLink(row: AccountsInMotionFeedItem) {
    try {
      await navigator.clipboard.writeText(
        watchlistVendorAlertsUrl(accountFocusParams(outboundWatchlistSearchParams, row), row.vendor, row.company),
      )
      setActionError(null)
      setActionMessage(`Copied Alerts API link for ${row.company || row.vendor}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy Alerts API link')
    }
  }

  async function handleCopyAccountRowReviewLink(row: AccountsInMotionFeedItem, reviewId: string) {
    try {
      await navigator.clipboard.writeText(watchlistReviewUrl(searchParams, row, reviewId))
      setActionError(null)
      setActionMessage(`Copied review link for ${row.company || row.vendor}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy review link')
    }
  }

  async function handleCopyAccountRowWitnessLink(
    row: AccountsInMotionFeedItem,
    witnessId: string,
    source?: string | null,
  ) {
    try {
      await navigator.clipboard.writeText(
        `${window.location.origin}${watchlistAccountEvidenceExplorerPath(searchParams, row, witnessId, source)}`,
      )
      setActionError(null)
      setActionMessage(`Copied witness link for ${row.company || row.vendor}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy witness link')
    }
  }

  async function handleCopyAccountRowOpportunityLink(row: AccountsInMotionFeedItem) {
    try {
      await navigator.clipboard.writeText(watchlistOpportunitiesUrl(outboundWatchlistSearchParams, row.vendor))
      setActionError(null)
      setActionMessage(`Copied opportunity link for ${row.company || row.vendor}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy opportunity link')
    }
  }

  async function handleCopyAccountRowReportsLink(row: AccountsInMotionFeedItem) {
    try {
      await navigator.clipboard.writeText(watchlistReportsUrl(outboundWatchlistSearchParams, row.vendor))
      setActionError(null)
      setActionMessage(`Copied reports link for ${row.company || row.vendor}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy reports link')
    }
  }

  async function handleCopySelectedReportsLink() {
    if (!selectedAccount) return
    try {
      await navigator.clipboard.writeText(watchlistReportsUrl(selectedAccountSearchParams, selectedAccount.vendor))
      setActionError(null)
      setActionMessage(`Copied reports link for ${selectedAccount.company || selectedAccount.vendor}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy reports link')
    }
  }

  async function handleCopySelectedOpportunitiesLink() {
    if (!selectedAccount) return
    try {
      await navigator.clipboard.writeText(watchlistOpportunitiesUrl(selectedAccountSearchParams, selectedAccount.vendor))
      setActionError(null)
      setActionMessage(`Copied opportunities link for ${selectedAccount.company || selectedAccount.vendor}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy opportunities link')
    }
  }

  async function handleCopySelectedEvidenceLink() {
    if (!selectedAccount) return
    try {
      await navigator.clipboard.writeText(
        `${window.location.origin}${watchlistEvidenceExplorerPath(selectedAccountSearchParams, selectedAccount.vendor, null, selectedSourceFilter)}`,
      )
      setActionError(null)
      setActionMessage(`Copied evidence link for ${selectedAccount.company || selectedAccount.vendor}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy evidence link')
    }
  }

  async function handleCopySelectedVendorLink() {
    if (!selectedAccount) return
    try {
      await navigator.clipboard.writeText(watchlistVendorWorkspaceUrl(selectedAccountSearchParams, selectedAccount.vendor))
      setActionError(null)
      setActionMessage(`Copied vendor link for ${selectedAccount.company || selectedAccount.vendor}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy vendor link')
    }
  }

  async function handleCopySelectedWitnessLink(witnessId: string) {
    if (!selectedAccount) return
    const witnessSource = selectedSourceFilter || selectedAccount.source_reviews[0]?.source || null
    try {
      await navigator.clipboard.writeText(
        `${window.location.origin}${watchlistAccountEvidenceExplorerPath(selectedAccountSearchParams, selectedAccount, witnessId, witnessSource)}`,
      )
      setActionError(null)
      setActionMessage(`Copied witness link for ${selectedAccount.company || selectedAccount.vendor}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy witness link')
    }
  }

  async function handleCopyVendorRowLink(vendorName: string) {
    try {
      await navigator.clipboard.writeText(watchlistVendorUrl(searchParams, vendorName))
      setActionError(null)
      setActionMessage(`Copied vendor link for ${vendorName}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy vendor link')
    }
  }

  async function handleCopyVendorWorkspaceLink(vendorName: string) {
    try {
      await navigator.clipboard.writeText(watchlistVendorWorkspaceUrl(searchParams, vendorName))
      setActionError(null)
      setActionMessage(`Copied vendor workspace link for ${vendorName}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy vendor workspace link')
    }
  }

  async function handleCopyVendorEvidenceLink(vendorName: string, source?: string | null) {
    try {
      await navigator.clipboard.writeText(
        `${window.location.origin}${watchlistEvidenceExplorerPath(searchParams, vendorName, null, source)}`,
      )
      setActionError(null)
      setActionMessage(`Copied evidence link for ${vendorName}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy evidence link')
    }
  }

  async function handleCopyVendorAlertsLink(vendorName: string) {
    try {
      await navigator.clipboard.writeText(watchlistVendorAlertsUrl(searchParams, vendorName))
      setActionError(null)
      setActionMessage(`Copied Alerts API link for ${vendorName}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy Alerts API link')
    }
  }

  async function handleCopyVendorOpportunitiesLink(vendorName: string) {
    try {
      await navigator.clipboard.writeText(watchlistOpportunitiesUrl(searchParams, vendorName))
      setActionError(null)
      setActionMessage(`Copied opportunities link for ${vendorName}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy opportunities link')
    }
  }

  async function handleCopyVendorReportsLink(vendorName: string) {
    try {
      await navigator.clipboard.writeText(watchlistReportsUrl(searchParams, vendorName))
      setActionError(null)
      setActionMessage(`Copied reports link for ${vendorName}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy reports link')
    }
  }

  async function handleCopyVendorWitnessLink(
    vendorName: string,
    witnessId: string,
    source?: string | null,
  ) {
    try {
      await navigator.clipboard.writeText(
        `${window.location.origin}${watchlistEvidenceExplorerPath(searchParams, vendorName, witnessId, source)}`,
      )
      setActionError(null)
      setActionMessage(`Copied witness link for ${vendorName}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy witness link')
    }
  }

  async function handleCopyVendorAccountReviewLink(vendorName: string, row: AccountsInMotionFeedItem) {
    try {
      await navigator.clipboard.writeText(watchlistAccountUrl(searchParams, row))
      setActionError(null)
      setActionMessage(`Copied account review link for ${vendorName}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy account review link')
    }
  }

  async function handleCopySelectedReviewLink(reviewId: string) {
    if (!selectedAccount) return
    try {
      await navigator.clipboard.writeText(watchlistReviewUrl(selectedAccountSearchParams, selectedAccount, reviewId))
      setActionError(null)
      setActionMessage(`Copied review link for ${selectedAccount.company || selectedAccount.vendor}`)
    } catch (err) {
      setActionMessage(null)
      setActionError(err instanceof Error ? err.message : 'Failed to copy review link')
    }
  }

  function handleCloseSelectedAccount() {
    setSearchParams((current) => {
      const next = new URLSearchParams(current)
      next.delete('account_vendor')
      next.delete('account_company')
      next.delete('account_report_date')
      next.delete('account_watch_vendor')
      next.delete('account_category')
      next.delete('account_track_mode')
      return next
    }, { replace: true })
    setSelectedAccount(null)
  }

  function handleCloseWitnessDrawer() {
    setSearchParams((current) => {
      const next = new URLSearchParams(current)
      next.delete('witness_id')
      next.delete('witness_vendor')
      return next
    }, { replace: true })
    setEvidenceDrawerOpen(false)
    setEvidenceDrawerWitnessId(null)
  }

  async function handleSaveWatchlistView() {
    const nextName = savedViewName.trim() || activeWatchlistView?.name || ''
    if (!nextName) {
      setActionError('Saved view name is required')
      setActionMessage(null)
      return
    }
    setSavingWatchlistView(true)
    setActionError(null)
    setActionMessage(null)
    const payload = {
      name: nextName,
      vendor_names: selectedVendorFilters.length ? selectedVendorFilters : undefined,
      category: selectedCategoryFilter || undefined,
      source: selectedSourceFilter || undefined,
      min_urgency: selectedMinUrgency ? Number(selectedMinUrgency) : undefined,
      include_stale: !freshOnly,
      named_accounts_only: namedAccountsOnly,
      changed_wedges_only: changedWedgesOnly,
      vendor_alert_threshold: vendorAlertThreshold ? Number(vendorAlertThreshold) : undefined,
      account_alert_threshold: accountAlertThreshold ? Number(accountAlertThreshold) : undefined,
      stale_days_threshold: staleDaysThreshold ? Number(staleDaysThreshold) : undefined,
      alert_email_enabled: alertEmailEnabled,
      alert_delivery_frequency: alertDeliveryFrequency,
    }
    try {
      if (activeWatchlistView) {
        await updateWatchlistView(activeWatchlistView.id, payload)
        setActionMessage(`Updated saved view ${nextName}`)
      } else {
        await createWatchlistView(payload)
        setActionMessage(`Saved view ${nextName} created`)
      }
      refresh()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to save watchlist view')
    } finally {
      setSavingWatchlistView(false)
    }
  }

  function handleDeleteWatchlistView(view: WatchlistView) {
    openDestructiveAction({ kind: 'delete_watchlist_view', view })
  }

  async function confirmDeleteWatchlistView(view: WatchlistView) {
    setDeletingWatchlistViewId(view.id)
    try {
      await deleteWatchlistView(view.id)
      if (activeWatchlistView?.id === view.id) {
        setSavedViewName('')
      }
      setPendingDestructiveAction(null)
      setPendingDestructiveError(null)
      setActionMessage(`Deleted saved view ${view.name}`)
      refresh()
    } catch (err) {
      setPendingDestructiveError(err instanceof Error ? err.message : 'Failed to delete saved view')
    } finally {
      setDeletingWatchlistViewId(null)
    }
  }

  async function handleEvaluateWatchlistAlerts() {
    if (!activeWatchlistView) {
      setActionError('Save or apply a watchlist view before evaluating alerts')
      setActionMessage(null)
      return
    }
    setEvaluatingWatchlistViewId(activeWatchlistView.id)
    setActionError(null)
    setActionMessage(null)
    try {
      const result = await evaluateWatchlistAlertEvents(activeWatchlistView.id)
      setActionMessage(
        `Evaluated ${activeWatchlistView.name}: ${result.new_open_event_count} new open, ${result.resolved_event_count} resolved`,
      )
      refresh()
      refreshAlertEvents()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to evaluate watchlist alerts')
    } finally {
      setEvaluatingWatchlistViewId(null)
    }
  }

  async function handleEmailWatchlistAlerts() {
    if (!activeWatchlistView) {
      setActionError('Save or apply a watchlist view before emailing alerts')
      setActionMessage(null)
      return
    }
    setEmailingWatchlistViewId(activeWatchlistView.id)
    setActionError(null)
    setActionMessage(null)
    try {
      const result = await deliverWatchlistAlertEmail(activeWatchlistView.id, {
        evaluate_before_send: true,
      })
      setActionMessage(result.summary)
      refresh()
      refreshAlertEvents()
      refreshAlertEmailLog()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to email watchlist alerts')
    } finally {
      setEmailingWatchlistViewId(null)
    }
  }

  async function handleAddVendor(vendorName: string) {
    setSubmittingVendor(vendorName)
    setActionError(null)
    setActionMessage(null)
    try {
      await addTrackedVendor(vendorName, trackMode, label.trim())
      setActionMessage(`${vendorName} added to watchlists`)
      setSearchInput('')
      setDebouncedSearch('')
      setLabel('')
      refresh()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to add vendor')
    } finally {
      setSubmittingVendor(null)
    }
  }

  function handleRemoveVendor(vendorName: string) {
    openDestructiveAction({ kind: 'remove_vendor', vendorName })
  }

  async function confirmRemoveVendor(vendorName: string) {
    setRemovingVendor(vendorName)
    try {
      await removeTrackedVendor(vendorName)
      setPendingDestructiveAction(null)
      setPendingDestructiveError(null)
      setActionMessage(`${vendorName} removed from watchlists`)
      refresh()
    } catch (err) {
      setPendingDestructiveError(err instanceof Error ? err.message : 'Failed to remove vendor')
    } finally {
      setRemovingVendor(null)
    }
  }

  function loadCompetitiveSetForEdit(item: CompetitiveSet) {
    setEditingCompetitiveSetId(item.id)
    setCompetitiveSetName(item.name)
    setCompetitiveSetFocal(item.focal_vendor_name)
    setCompetitiveSetCompetitors(item.competitor_vendor_names)
    setCompetitiveSetRefreshMode(item.refresh_mode)
    setCompetitiveSetRefreshHours(
      String(item.refresh_interval_hours ?? competitiveSetDefaults?.default_refresh_interval_hours ?? ''),
    )
    setCompetitiveSetActive(item.active)
    setCompetitiveSetVendorEnabled(item.vendor_synthesis_enabled)
    setCompetitiveSetPairwiseEnabled(item.pairwise_enabled)
    setCompetitiveSetCategoryEnabled(item.category_council_enabled)
    setCompetitiveSetAsymmetryEnabled(item.asymmetry_enabled)
  }

  async function handleSaveCompetitiveSet() {
    setSavingCompetitiveSet(true)
    setActionError(null)
    setActionMessage(null)
    const payload = {
      name: competitiveSetName.trim(),
      focal_vendor_name: competitiveSetFocal,
      competitor_vendor_names: competitiveSetCompetitors,
      active: competitiveSetActive,
      refresh_mode: competitiveSetRefreshMode,
      refresh_interval_hours: competitiveSetRefreshMode === 'scheduled'
        ? Number.parseInt(competitiveSetRefreshHours, 10)
          || competitiveSetDefaults?.default_refresh_interval_hours
          || null
        : null,
      vendor_synthesis_enabled: competitiveSetVendorEnabled,
      pairwise_enabled: competitiveSetPairwiseEnabled,
      category_council_enabled: competitiveSetCategoryEnabled,
      asymmetry_enabled: competitiveSetAsymmetryEnabled,
    }
    try {
      if (editingCompetitiveSetId) {
        await updateCompetitiveSet(editingCompetitiveSetId, payload)
        setActionMessage(`Updated competitive set ${payload.name}`)
      } else {
        await createCompetitiveSet(payload)
        setActionMessage(`Created competitive set ${payload.name}`)
      }
      resetCompetitiveSetForm()
      refresh()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to save competitive set')
    } finally {
      setSavingCompetitiveSet(false)
    }
  }

  function handleDeleteCompetitiveSet(item: CompetitiveSet) {
    openDestructiveAction({ kind: 'delete_competitive_set', item })
  }

  async function confirmDeleteCompetitiveSet(item: CompetitiveSet) {
    setDeletingCompetitiveSetId(item.id)
    try {
      await deleteCompetitiveSet(item.id)
      if (editingCompetitiveSetId === item.id) resetCompetitiveSetForm()
      setPendingDestructiveAction(null)
      setPendingDestructiveError(null)
      setActionMessage(`Deleted competitive set ${item.name}`)
      refresh()
    } catch (err) {
      setPendingDestructiveError(err instanceof Error ? err.message : 'Failed to delete competitive set')
    } finally {
      setDeletingCompetitiveSetId(null)
    }
  }

  function applyCompetitiveSetPreviewState(
    competitiveSetId: string,
    plan: CompetitiveSetPlan,
    recentRuns: CompetitiveSetRun[],
  ) {
    setCompetitiveSetPreviews((current) => ({
      ...current,
      [competitiveSetId]: plan,
    }))
    setCompetitiveSetRuns((current) => ({
      ...current,
      [competitiveSetId]: recentRuns,
    }))
    setCompetitiveSetChangedOnly((current) => ({
      ...current,
      [competitiveSetId]: current[competitiveSetId] ?? (competitiveSetDefaults?.default_changed_vendors_only ?? true),
    }))
    setCompetitiveSetForceRun((current) => ({
      ...current,
      [competitiveSetId]: current[competitiveSetId] ?? false,
    }))
    setCompetitiveSetForceCrossVendor((current) => ({
      ...current,
      [competitiveSetId]: current[competitiveSetId] ?? false,
    }))
  }

  async function handlePreviewCompetitiveSet(item: CompetitiveSet) {
    setPreviewingCompetitiveSetId(item.id)
    setActionError(null)
    setActionMessage(null)
    try {
      const result = await fetchCompetitiveSetPlan(item.id)
      applyCompetitiveSetPreviewState(item.id, result.plan, result.recent_runs)
      setOpenCompetitiveSetPreviewId(item.id)
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to load competitive-set preview')
    } finally {
      setPreviewingCompetitiveSetId(null)
    }
  }

  async function handleRunCompetitiveSet(item: CompetitiveSet) {
    setRunningCompetitiveSetId(item.id)
    setActionError(null)
    setActionMessage(null)
    try {
      const changedOnly = competitiveSetChangedOnly[item.id] ?? true
      const forceRun = competitiveSetForceRun[item.id] ?? false
      const forceCrossVendor = competitiveSetForceCrossVendor[item.id] ?? false
      const result = await runCompetitiveSetNow(item.id, {
        changed_vendors_only: changedOnly,
        force: forceRun,
        force_cross_vendor: forceCrossVendor,
      })
      if (result.already_running) {
        try {
          const preview = await fetchCompetitiveSetPlan(item.id)
          applyCompetitiveSetPreviewState(item.id, preview.plan, preview.recent_runs)
          const currentRun = preview.recent_runs.find((run) => (
            result.execution_id
              ? run.execution_id === result.execution_id || run.run_id === result.execution_id
              : run.status === 'running'
          )) ?? preview.recent_runs[0]
          if (currentRun) {
            setCompetitiveSetLastRunOverrides((current) => ({
              ...current,
              [item.id]: {
                last_run_status: currentRun.status,
                last_run_summary: currentRun.summary ?? {},
                last_run_at: currentRun.started_at,
              },
            }))
          }
        } catch {
          // Keep the stale preview if the follow-up refresh fails.
        }
        setActionMessage(result.message || `${item.name} is already running`)
        refresh()
        return
      }
      const startedAt = new Date().toISOString()
      const optimisticSummary = {
        changed_vendors_only: changedOnly,
        force: forceRun,
        force_cross_vendor: forceCrossVendor,
      }
      const optimisticRunId = result.execution_id
        ? `optimistic:${result.execution_id}`
        : `optimistic:${startedAt}`
      setCompetitiveSetLastRunOverrides((current) => ({
        ...current,
        [item.id]: {
          last_run_status: 'running',
          last_run_summary: optimisticSummary,
          last_run_at: startedAt,
        },
      }))
      setCompetitiveSetRuns((current) => ({
        ...current,
        [item.id]: [
          {
            id: optimisticRunId,
            competitive_set_id: item.id,
            account_id: '',
            run_id: result.execution_id ?? optimisticRunId,
            trigger: 'manual',
            status: 'running',
            execution_id: result.execution_id ?? null,
            summary: optimisticSummary,
            started_at: startedAt,
            completed_at: null,
            created_at: startedAt,
          },
          ...(current[item.id] ?? []).filter((run) => run.id !== optimisticRunId),
        ],
      }))
      setActionMessage(
        `${item.name} refresh started${result.execution_id ? ` (${result.execution_id})` : ''}`,
      )
      refresh()
    } catch (err) {
      setActionError(err instanceof Error ? err.message : 'Failed to start competitive-set refresh')
    } finally {
      setRunningCompetitiveSetId(null)
    }
  }

  const trackedColumns: Column<TrackedVendor>[] = [
    {
      key: 'vendor',
      header: 'Vendor',
      render: (row) => (
        <div>
          <div className="font-medium text-white">{row.vendor_name}</div>
          <div className="text-xs text-slate-500">{row.label || row.track_mode.replace(/_/g, ' ')}</div>
        </div>
      ),
      sortable: true,
      sortValue: (row) => row.vendor_name,
    },
    {
      key: 'mode',
      header: 'Mode',
      render: (row) => (
        <span className="inline-flex rounded-full bg-slate-800 px-2 py-0.5 text-xs text-slate-300">
          {row.track_mode.replace(/_/g, ' ')}
        </span>
      ),
      sortable: true,
      sortValue: (row) => row.track_mode,
    },
    {
      key: 'urgency',
      header: 'Avg Urgency',
      render: (row) => <UrgencyBadge score={row.avg_urgency ?? 0} />,
      sortable: true,
      sortValue: (row) => row.avg_urgency ?? 0,
    },
    {
      key: 'reviews',
      header: 'Reviews',
      render: (row) => <span className="text-slate-300">{row.total_reviews ?? '--'}</span>,
      sortable: true,
      sortValue: (row) => row.total_reviews ?? 0,
    },
    {
      key: 'nps',
      header: 'NPS Proxy',
      render: (row) => <span className="text-slate-300">{row.nps_proxy?.toFixed(1) ?? '--'}</span>,
      sortable: true,
      sortValue: (row) => row.nps_proxy ?? -999,
    },
    {
      key: 'freshness',
      header: 'Freshness',
      render: (row) => (
        <div className="text-xs">
          <div className={clsx(freshnessTone(row.freshness_status, row.freshness_timestamp))}>
            {formatTs(row.freshness_timestamp || row.last_computed_at || row.latest_snapshot_date || row.latest_accounts_report_date)}
          </div>
          <div className="text-slate-500" title={row.freshness_reason || undefined}>
            {freshnessLabel(row.freshness_status, row.freshness_timestamp)}
          </div>
        </div>
      ),
      sortable: true,
      sortValue: (row) => toTimestamp(row.freshness_timestamp || row.last_computed_at || row.latest_snapshot_date || row.latest_accounts_report_date) ?? 0,
    },
    {
      key: 'actions',
      header: 'Actions',
      render: (row) => (
        <div className="flex items-center justify-end gap-2">
          <button
            onClick={(event) => {
              event.stopPropagation()
              navigate(watchlistVendorPath(searchParams, row.vendor_name))
            }}
            className="rounded-md bg-cyan-500/10 px-2.5 py-1 text-xs font-medium text-cyan-300 hover:bg-cyan-500/20"
          >
            View
          </button>
          <Link
            to={watchlistEvidenceExplorerPath(searchParams, row.vendor_name, null, selectedSourceFilter)}
            onClick={(event) => event.stopPropagation()}
            aria-label={`Open evidence for ${row.vendor_name}`}
            className="rounded-md bg-fuchsia-500/10 px-2.5 py-1 text-xs font-medium text-fuchsia-300 hover:bg-fuchsia-500/20"
          >
            Evidence
          </Link>
          <Link
            to={watchlistVendorAlertsPath(searchParams, row.vendor_name)}
            onClick={(event) => event.stopPropagation()}
            aria-label={`Open alerts for ${row.vendor_name}`}
            className="rounded-md bg-violet-500/10 px-2.5 py-1 text-xs font-medium text-violet-200 hover:bg-violet-500/20"
          >
            Alerts
          </Link>
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation()
              void handleCopyVendorAlertsLink(row.vendor_name)
            }}
            aria-label={`Copy alerts link for ${row.vendor_name}`}
            className="rounded-md bg-slate-800 px-2.5 py-1 text-xs font-medium text-slate-300 hover:bg-slate-700"
          >
            Copy Alerts
          </button>
          <Link
            to={watchlistReportsPath(searchParams, row.vendor_name)}
            onClick={(event) => event.stopPropagation()}
            aria-label={`Open reports for ${row.vendor_name}`}
            className="rounded-md bg-violet-500/10 px-2.5 py-1 text-xs font-medium text-violet-300 hover:bg-violet-500/20"
          >
            Reports
          </Link>
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation()
              void handleCopyVendorReportsLink(row.vendor_name)
            }}
            aria-label={`Copy reports link for ${row.vendor_name}`}
            className="rounded-md bg-slate-800 px-2.5 py-1 text-xs font-medium text-slate-300 hover:bg-slate-700"
          >
            Copy Reports
          </button>
          <Link
            to={watchlistOpportunitiesPath(searchParams, row.vendor_name)}
            onClick={(event) => event.stopPropagation()}
            aria-label={`Open opportunities for ${row.vendor_name}`}
            className="rounded-md bg-amber-500/10 px-2.5 py-1 text-xs font-medium text-amber-300 hover:bg-amber-500/20"
          >
            Opportunities
          </Link>
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation()
              void handleCopyVendorOpportunitiesLink(row.vendor_name)
            }}
            aria-label={`Copy opportunities link for ${row.vendor_name}`}
            className="rounded-md bg-slate-800 px-2.5 py-1 text-xs font-medium text-slate-300 hover:bg-slate-700"
          >
            Copy Opportunities
          </button>
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation()
              void handleCopyVendorEvidenceLink(row.vendor_name, selectedSourceFilter)
            }}
            aria-label={`Copy evidence link for ${row.vendor_name}`}
            className="rounded-md bg-slate-800 px-2.5 py-1 text-xs font-medium text-slate-300 hover:bg-slate-700"
          >
            Copy Evidence
          </button>
          <button
            type="button"
            onClick={(event) => {
              event.stopPropagation()
              void handleCopyVendorRowLink(row.vendor_name)
            }}
            aria-label={`Copy vendor link for ${row.vendor_name}`}
            className="rounded-md bg-slate-800 px-2.5 py-1 text-xs font-medium text-slate-300 hover:bg-slate-700"
          >
            Copy Link
          </button>
          <button
            onClick={(event) => {
              event.stopPropagation()
              handleRemoveVendor(row.vendor_name)
            }}
            disabled={removingVendor === row.vendor_name}
            className="rounded-md bg-rose-500/10 px-2.5 py-1 text-xs font-medium text-rose-300 hover:bg-rose-500/20 disabled:opacity-50"
          >
            <span className="inline-flex items-center gap-1">
              <Trash2 className="h-3 w-3" />
              Remove
            </span>
          </button>
        </div>
      ),
    },
  ]

  const feedColumns: Column<ChurnSignal>[] = [
    {
      key: 'vendor',
      header: 'Vendor Movement',
      render: (row) => {
        const primaryWitnessId = row.reasoning_reference_ids?.witness_ids?.[0] || ''
        const accountSignals = Array.isArray(row.reasoning_delta?.new_account_signals)
          ? row.reasoning_delta.new_account_signals.map((value) => String(value).trim().toLowerCase()).filter(Boolean)
          : []
        const matchedAccountReview = accountSignals.length > 0
          ? visibleAccounts.find((account) => (
            account.vendor === row.vendor_name
            && account.company
            && accountSignals.includes(account.company.trim().toLowerCase())
          )) ?? null
          : null
        return (
          <div>
          <div className="font-medium text-white">{row.vendor_name}</div>
          <div className="text-xs text-slate-500">{row.product_category ?? 'Uncategorized'}</div>
          <div className="mt-1 flex flex-wrap gap-1 text-[11px]">
            {(row.vendor_alert_hit ?? vendorAlertTriggered(row, activeVendorAlertThreshold)) ? (
              <span className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-2 py-0.5 text-cyan-300">
                vendor alert hit
              </span>
            ) : null}
            {(row.stale_threshold_hit ?? staleThresholdTriggered(activeStaleDaysThreshold, {
              freshnessTimestamp: row.freshness_timestamp,
              fallbackTimestamp: row.last_computed_at,
            })) ? (
              <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-amber-300">
                stale policy hit
              </span>
            ) : null}
          </div>
          {row.synthesis_wedge_label && (
            <div className="mt-1 text-[11px] text-cyan-300">{row.synthesis_wedge_label}</div>
          )}
          <div className="mt-2 flex flex-wrap items-center gap-3 text-[11px]">
            <Link
              to={watchlistEvidenceExplorerPath(searchParams, row.vendor_name, null, selectedSourceFilter)}
              onClick={(event) => event.stopPropagation()}
              aria-label={`Open vendor evidence for ${row.vendor_name}`}
              className="inline-flex items-center gap-1 text-violet-300 hover:text-violet-200"
            >
              <Fingerprint className="h-3 w-3" />
              Evidence Explorer
            </Link>
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation()
                void handleCopyVendorEvidenceLink(row.vendor_name, selectedSourceFilter)
              }}
              aria-label={`Copy vendor evidence link for ${row.vendor_name}`}
              className="text-slate-300 hover:text-slate-200"
            >
              Copy evidence
            </button>
            <Link
              to={watchlistVendorAlertsPath(searchParams, row.vendor_name)}
              onClick={(event) => event.stopPropagation()}
              aria-label={`Open vendor alerts for ${row.vendor_name}`}
              className="text-violet-200 hover:text-violet-100"
            >
              Alerts
            </Link>
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation()
                void handleCopyVendorAlertsLink(row.vendor_name)
              }}
              aria-label={`Copy vendor alerts link for ${row.vendor_name}`}
              className="text-slate-300 hover:text-slate-200"
            >
              Copy alerts
            </button>
          </div>
          <div className="mt-1 flex flex-wrap gap-3 text-[11px]">
            {primaryWitnessId ? (
              <Link
                to={watchlistEvidenceExplorerPath(searchParams, row.vendor_name, primaryWitnessId, selectedSourceFilter)}
                onClick={(event) => event.stopPropagation()}
                aria-label={`Open vendor witness for ${row.vendor_name}`}
                className="text-cyan-300 hover:text-cyan-200"
              >
                Witness
              </Link>
            ) : null}
            {primaryWitnessId ? (
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation()
                  void handleCopyVendorWitnessLink(row.vendor_name, primaryWitnessId, selectedSourceFilter)
                }}
                aria-label={`Copy vendor witness link for ${row.vendor_name}`}
                className="text-slate-300 hover:text-slate-200"
              >
                Copy witness
              </button>
            ) : null}
            {matchedAccountReview ? (
              <Link
                to={watchlistPath(accountFocusParams(searchParams, matchedAccountReview))}
                onClick={(event) => event.stopPropagation()}
                aria-label={`Open account review for ${row.vendor_name}`}
                className="text-emerald-300 hover:text-emerald-200"
              >
                Account Review
              </Link>
            ) : null}
            {matchedAccountReview ? (
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation()
                  void handleCopyVendorAccountReviewLink(row.vendor_name, matchedAccountReview)
                }}
                aria-label={`Copy vendor account review link for ${row.vendor_name}`}
                className="text-slate-300 hover:text-slate-200"
              >
                Copy account review
              </button>
            ) : null}
            <Link
              to={watchlistReportsPath(searchParams, row.vendor_name)}
              onClick={(event) => event.stopPropagation()}
              aria-label={`Open vendor reports for ${row.vendor_name}`}
              className="text-fuchsia-300 hover:text-fuchsia-200"
            >
              Reports
            </Link>
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation()
                void handleCopyVendorReportsLink(row.vendor_name)
              }}
              aria-label={`Copy vendor reports link for ${row.vendor_name}`}
              className="text-slate-300 hover:text-slate-200"
            >
              Copy reports
            </button>
            <Link
              to={watchlistOpportunitiesPath(searchParams, row.vendor_name)}
              onClick={(event) => event.stopPropagation()}
              aria-label={`Open vendor opportunities for ${row.vendor_name}`}
              className="text-amber-300 hover:text-amber-200"
            >
              Opportunities
            </Link>
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation()
                void handleCopyVendorOpportunitiesLink(row.vendor_name)
              }}
              aria-label={`Copy vendor opportunities link for ${row.vendor_name}`}
              className="text-slate-300 hover:text-slate-200"
            >
              Copy opportunities
            </button>
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation()
                void handleCopyVendorRowLink(row.vendor_name)
              }}
              aria-label={`Copy vendor link for ${row.vendor_name}`}
              className="text-slate-300 hover:text-slate-200"
            >
              Copy Link
            </button>
          </div>
        </div>
      )},
      sortable: true,
      sortValue: (row) => row.vendor_name,
    },
    {
      key: 'urgency',
      header: 'Urgency',
      render: (row) => <UrgencyBadge score={row.avg_urgency_score} />,
      sortable: true,
      sortValue: (row) => row.avg_urgency_score,
    },
    {
      key: 'archetype',
      header: 'Wedge',
      render: (row) => (
        <ArchetypeBadge
          archetype={row.archetype}
          confidence={row.archetype_confidence}
          showConfidence
        />
      ),
      sortable: true,
      sortValue: (row) => row.archetype ?? '',
    },
    {
      key: 'delta',
      header: 'Change Signal',
      render: (row) => {
        const items = summarizeReasoningDelta(row)
        if (items.length === 0) {
          return <span className="text-xs text-slate-500">Stable</span>
        }
        return (
          <div className="space-y-1">
            {items.slice(0, 2).map((item, index) => (
              <div key={`${row.vendor_name}-${index}`} className="text-xs text-slate-300">
                {item}
              </div>
            ))}
          </div>
        )
      },
      sortable: true,
      sortValue: (row) => summarizeReasoningDelta(row).join(' ') || row.synthesis_wedge_label || '',
    },
    {
      key: 'support',
      header: 'Support',
      render: (row) => <span className="text-slate-300">{row.support_sentiment?.toFixed(1) ?? '--'}</span>,
      sortable: true,
      sortValue: (row) => row.support_sentiment ?? -999,
    },
    {
      key: 'growth',
      header: 'Growth',
      render: (row) => (
        <span className="text-slate-300">
          {row.employee_growth_rate != null ? `${row.employee_growth_rate >= 0 ? '+' : ''}${row.employee_growth_rate.toFixed(1)}%` : '--'}
        </span>
      ),
      sortable: true,
      sortValue: (row) => row.employee_growth_rate ?? -999,
    },
    {
      key: 'freshness',
      header: 'Last Updated',
      render: (row) => (
        <div className="text-xs">
          <div className={clsx(freshnessTone(row.freshness_status, row.freshness_timestamp || row.last_computed_at))}>
            {formatTs(row.freshness_timestamp || row.last_computed_at)}
          </div>
          <div className="text-slate-500" title={row.freshness_reason || undefined}>
            {freshnessLabel(row.freshness_status, row.freshness_timestamp || row.last_computed_at)}
          </div>
        </div>
      ),
      sortable: true,
      sortValue: (row) => toTimestamp(row.freshness_timestamp || row.last_computed_at) ?? 0,
    },
  ]

  const accountColumns: Column<AccountsInMotionFeedItem>[] = [
    {
      key: 'company',
      header: 'Account Movement',
      render: (row) => {
        const confidence = confidenceBand(row.confidence)
        const label = row.company || 'Anonymous signal cluster'
        return (
          <div>
            <div className="font-medium text-white">{label}</div>
            <div className="mt-1 flex flex-wrap items-center gap-2 text-xs">
              <span className="text-slate-500">{row.vendor}</span>
              <span className={clsx('rounded-full border px-2 py-0.5', confidence.tone)}>
                {confidence.label}
              </span>
              {(row.account_alert_hit ?? accountAlertTriggered(row, activeAccountAlertThreshold)) ? (
                <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-emerald-300">
                  account alert hit
                </span>
              ) : null}
              {(row.stale_threshold_hit ?? staleThresholdTriggered(activeStaleDaysThreshold, {
                staleDays: row.stale_days,
                freshnessTimestamp: row.freshness_timestamp,
                fallbackTimestamp: row.report_date,
              })) ? (
                <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-amber-300">
                  stale policy hit
                </span>
              ) : null}
            </div>
          </div>
        )
      },
      sortable: true,
      sortValue: (row) => row.company || '',
    },
    {
      key: 'urgency',
      header: 'Urgency',
      render: (row) => <UrgencyBadge score={row.urgency} />,
      sortable: true,
      sortValue: (row) => row.urgency,
    },
    {
      key: 'timing',
      header: 'Timing',
      render: (row) => <span className="text-slate-300">{row.contract_signal || '--'}</span>,
      sortable: true,
      sortValue: (row) => row.contract_signal || '',
    },
    {
      key: 'pain',
      header: 'Pain',
      render: (row) => <span className="text-slate-300">{row.pain_categories[0]?.category || '--'}</span>,
      sortable: true,
      sortValue: (row) => row.pain_categories[0]?.category || '',
    },
    {
      key: 'quote',
      header: 'Evidence',
      render: (row) => (
        <div className="max-w-[320px]">
          <div className="truncate text-slate-300">{row.evidence[0] || '--'}</div>
          <div className="mt-1 text-xs text-slate-500">
            {row.evidence_count} span{row.evidence_count === 1 ? '' : 's'} - {row.source_reviews.length} review{row.source_reviews.length === 1 ? '' : 's'}
          </div>
        </div>
      ),
    },
    {
      key: 'report_date',
      header: 'Freshness',
      render: (row) => (
        <div className="text-xs">
          <div className={clsx(freshnessTone(row.freshness_status ?? null, row.freshness_timestamp || row.report_date))}>
            {formatTs(row.freshness_timestamp || row.report_date)}
          </div>
          <div className="text-slate-500" title={row.freshness_reason || undefined}>
            {freshnessLabel(row.freshness_status ?? (row.is_stale ? 'stale' : 'fresh'), row.freshness_timestamp || row.report_date)}
          </div>
        </div>
      ),
      sortable: true,
      sortValue: (row) => toTimestamp(row.freshness_timestamp || row.report_date) ?? 0,
    },
    {
      key: 'actions',
      header: 'Actions',
      render: (row) => {
        const genKey = `${row.company}::${row.vendor}`
        const isGenerating = generatingCampaignFor === genKey
        const primaryReviewId = row.source_reviews[0]?.id || ''
        const primaryWitnessId = row.reasoning_reference_ids?.witness_ids?.[0] || ''
        const primaryWitnessSource = selectedSourceFilter || row.source_reviews[0]?.source || null
        return (
          <div className="flex items-center gap-1">
            <button
              className="rounded-md bg-cyan-500/10 px-2 py-1 text-xs font-medium text-cyan-300 hover:bg-cyan-500/20"
              onClick={(event) => {
                event.stopPropagation()
                navigate(watchlistVendorPath(outboundWatchlistSearchParams, row.vendor))
              }}
            >
              View vendor
            </button>
            <button
              type="button"
              onClick={(event) => {
                event.stopPropagation()
                void handleCopyAccountRowVendorLink(row)
              }}
              aria-label={`Copy account vendor link for ${row.vendor}`}
              className="rounded-md bg-slate-800 px-2 py-1 text-xs font-medium text-slate-300 hover:bg-slate-700"
              title="Copy vendor link"
            >
              Copy vendor
            </button>
            {row.company && (
              <Link
                to={watchlistAccountEvidenceExplorerPath(searchParams, row, null, selectedSourceFilter)}
                onClick={(event) => event.stopPropagation()}
                aria-label={`Open account evidence for ${row.vendor}`}
                className="rounded-md bg-violet-500/10 px-2 py-1 text-xs font-medium text-violet-300 hover:bg-violet-500/20"
                title="Open in Evidence Explorer"
              >
                <Fingerprint className="inline h-3 w-3" />
              </Link>
            )}
            {primaryWitnessId && (
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation()
                  void handleCopyAccountRowWitnessLink(row, primaryWitnessId, primaryWitnessSource)
                }}
                aria-label={`Copy account witness link for ${row.vendor}`}
                className="rounded-md bg-slate-800 px-2 py-1 text-xs font-medium text-slate-300 hover:bg-slate-700"
                title="Copy witness link"
              >
                Copy witness
              </button>
            )}
            {primaryWitnessId && (
              <Link
                to={watchlistAccountEvidenceExplorerPath(searchParams, row, primaryWitnessId, primaryWitnessSource)}
                onClick={(event) => event.stopPropagation()}
                aria-label={`Open primary witness for ${row.vendor}`}
                className="rounded-md bg-fuchsia-500/10 px-2 py-1 text-xs font-medium text-fuchsia-300 hover:bg-fuchsia-500/20"
                title="Open witness drilldown"
              >
                Witness
              </Link>
            )}
            {primaryReviewId && (
              <Link
                to={watchlistReviewDetailPath(searchParams, row, primaryReviewId)}
                onClick={(event) => event.stopPropagation()}
                aria-label={`Open review detail for ${row.vendor}`}
                className="rounded-md bg-sky-500/10 px-2 py-1 text-xs font-medium text-sky-300 hover:bg-sky-500/20"
                title="Open review detail"
              >
                Review
              </Link>
            )}
            {primaryReviewId && (
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation()
                  void handleCopyAccountRowReviewLink(row, primaryReviewId)
                }}
                aria-label={`Copy review link for ${row.vendor}`}
                className="rounded-md bg-slate-800 px-2 py-1 text-xs font-medium text-slate-300 hover:bg-slate-700"
                title="Copy review link"
              >
                Copy review
              </button>
            )}
            {row.company && (
              <button
                type="button"
                onClick={(event) => {
                  event.stopPropagation()
                  void handleCopyAccountRowLink(row)
                }}
                aria-label={`Copy account link for ${row.company || row.vendor}`}
                className="rounded-md bg-emerald-500/10 px-2 py-1 text-xs font-medium text-emerald-300 hover:bg-emerald-500/20"
                title="Copy account link"
              >
                Copy Link
              </button>
            )}
            {row.company && (
              <Link
                to={watchlistVendorAlertsPath(accountFocusParams(outboundWatchlistSearchParams, row), row.vendor, row.company)}
                onClick={(e) => e.stopPropagation()}
                aria-label={`View alerts for ${row.vendor}`}
                className="rounded-md bg-violet-500/10 px-2 py-1 text-xs font-medium text-violet-200 hover:bg-violet-500/20"
                title="View alerts"
              >
                Alerts
              </Link>
            )}
            {row.company && (
              <button
                type="button"
                className="rounded-md bg-slate-800 px-2 py-1 text-xs font-medium text-slate-300 hover:bg-slate-700"
                onClick={(event) => {
                  event.stopPropagation()
                  void handleCopyAccountRowAlertsLink(row)
                }}
                aria-label={`Copy account alerts link for ${row.vendor}`}
                title="Copy alerts link"
              >
                Copy alerts
              </button>
            )}
            {row.company && (
              <Link
                to={watchlistReportsPath(outboundWatchlistSearchParams, row.vendor)}
                onClick={(e) => e.stopPropagation()}
                aria-label={`View reports for ${row.vendor}`}
                className="rounded-md bg-violet-500/10 px-2 py-1 text-xs font-medium text-violet-300 hover:bg-violet-500/20"
                title="View reports"
              >
                Reports
              </Link>
            )}
            {row.company && (
              <button
                type="button"
                className="rounded-md bg-slate-800 px-2 py-1 text-xs font-medium text-slate-300 hover:bg-slate-700"
                onClick={(event) => {
                  event.stopPropagation()
                  void handleCopyAccountRowReportsLink(row)
                }}
                aria-label={`Copy account reports link for ${row.vendor}`}
                title="Copy reports link"
              >
                Copy reports
              </button>
            )}
            {row.company && (
              <Link
                to={watchlistOpportunitiesPath(outboundWatchlistSearchParams, row.vendor)}
                onClick={(e) => e.stopPropagation()}
                aria-label={`View opportunities for ${row.vendor}`}
                className="rounded-md bg-amber-500/10 px-2 py-1 text-xs font-medium text-amber-300 hover:bg-amber-500/20"
                title="View opportunities"
              >
                <Telescope className="inline h-3 w-3" />
              </Link>
            )}
            {row.company && (
              <button
                type="button"
                className="rounded-md bg-slate-800 px-2 py-1 text-xs font-medium text-slate-300 hover:bg-slate-700"
                onClick={(event) => {
                  event.stopPropagation()
                  void handleCopyAccountRowOpportunityLink(row)
                }}
                aria-label={`Copy account opportunity link for ${row.vendor}`}
                title="Copy opportunity link"
              >
                Copy opportunity
              </button>
            )}
            {row.company && (
              <button
                className="rounded-md bg-green-500/10 px-2 py-1 text-xs font-medium text-green-300 hover:bg-green-500/20 disabled:opacity-50"
                onClick={(event) => {
                  event.stopPropagation()
                  handleGenerateCampaign(row)
                }}
                disabled={isGenerating}
                title="Generate campaigns"
              >
                <Zap className={`inline h-3 w-3 ${isGenerating ? 'animate-pulse' : ''}`} />
              </button>
            )}
          </div>
        )
      },
    },
  ]

  async function handleConfirmDestructiveAction() {
    if (!pendingDestructiveAction) return
    switch (pendingDestructiveAction.kind) {
      case 'delete_watchlist_view':
        await confirmDeleteWatchlistView(pendingDestructiveAction.view)
        return
      case 'remove_vendor':
        await confirmRemoveVendor(pendingDestructiveAction.vendorName)
        return
      case 'delete_competitive_set':
        await confirmDeleteCompetitiveSet(pendingDestructiveAction.item)
        return
    }
  }

  const pendingDestructiveActionConfig = pendingDestructiveAction
    ? describePendingWatchlistDestructiveAction(pendingDestructiveAction)
    : null
  const pendingDestructiveActionBusy = pendingDestructiveAction
    ? (
        pendingDestructiveAction.kind === 'delete_watchlist_view'
          ? deletingWatchlistViewId === pendingDestructiveAction.view.id
          : pendingDestructiveAction.kind === 'remove_vendor'
            ? removingVendor === pendingDestructiveAction.vendorName
            : deletingCompetitiveSetId === pendingDestructiveAction.item.id
      )
    : false

  if (error) return <PageError error={error} onRetry={refresh} />

  return (
    <div className="space-y-6">
      <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
        <div>
          {requestedBackTo ? (
            <Link
              to={requestedBackTo}
              className="mb-3 inline-flex items-center gap-2 text-sm text-slate-400 hover:text-white"
            >
              <ArrowLeft className="h-4 w-4" />
              {backToLabel(requestedBackTo)}
            </Link>
          ) : null}
          <h1 className="text-2xl font-bold text-white">Watchlists</h1>
          <p className="mt-1 max-w-3xl text-sm text-slate-400">
            Track the vendors that matter, monitor movement across the slow-burn feed, and jump directly into vendor detail for evidence-backed review.
          </p>
        </div>
        <div className="flex items-center gap-2">
          {currentEvidenceVendor && (
            <div className="flex items-center gap-2">
              <Link
                to={watchlistVendorPath(searchParams, currentEvidenceVendor)}
                className="inline-flex items-center gap-2 self-start rounded-lg px-3 py-1.5 text-sm text-cyan-300 transition-colors hover:bg-cyan-500/10 hover:text-cyan-200"
              >
                <Building2 className="h-4 w-4" />
                Open Current View in Vendor Workspace
              </Link>
              <button
                type="button"
                onClick={() => void handleCopyVendorWorkspaceLink(currentEvidenceVendor)}
                aria-label="Copy current vendor link"
                className="inline-flex items-center gap-2 self-start rounded-lg px-3 py-1.5 text-sm text-slate-400 transition-colors hover:bg-slate-800/50 hover:text-white"
              >
                Copy Vendor
              </button>
              <Link
                to={watchlistEvidenceExplorerPath(searchParams, currentEvidenceVendor, null, selectedSourceFilter)}
                className="inline-flex items-center gap-2 self-start rounded-lg px-3 py-1.5 text-sm text-violet-300 transition-colors hover:bg-violet-500/10 hover:text-violet-200"
              >
                <Fingerprint className="h-4 w-4" />
                Open Current View in Evidence Explorer
              </Link>
              <button
                type="button"
                onClick={() => void handleCopyVendorEvidenceLink(currentEvidenceVendor, selectedSourceFilter)}
                aria-label="Copy current evidence link"
                className="inline-flex items-center gap-2 self-start rounded-lg px-3 py-1.5 text-sm text-slate-400 transition-colors hover:bg-slate-800/50 hover:text-white"
              >
                Copy Evidence
              </button>
            </div>
          )}
          <button
            type="button"
            onClick={() => void handleCopyCurrentViewLink()}
            aria-label="Copy current view link"
            className="inline-flex items-center gap-2 self-start rounded-lg px-3 py-1.5 text-sm text-slate-400 transition-colors hover:bg-slate-800/50 hover:text-white"
          >
            Copy View
          </button>
          <button
            onClick={() => downloadCsv('/export/signals')}
            className="inline-flex items-center gap-2 self-start rounded-lg px-3 py-1.5 text-sm text-slate-400 transition-colors hover:bg-slate-800/50 hover:text-white"
          >
            <Download className="h-4 w-4" />
            Export
          </button>
          <button
            onClick={refresh}
            disabled={refreshing}
            className="inline-flex items-center gap-2 self-start rounded-lg px-3 py-1.5 text-sm text-slate-400 transition-colors hover:bg-slate-800/50 hover:text-white disabled:opacity-50"
          >
            <RefreshCw className={clsx('h-4 w-4', refreshing && 'animate-spin')} />
            Refresh
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 xl:grid-cols-4">
        <StatCard
          label="Tracked Vendors"
          value={trackedVendors.length}
          icon={<Building2 className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="Competitors"
          value={trackedVendors.filter((vendor) => vendor.track_mode === 'competitor').length}
          icon={<BellRing className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="Feed Rows"
          value={filteredFeed.length}
          sub={hasActiveAlertPolicy
            ? `${vendorAlertHitCount} alert hit${vendorAlertHitCount === 1 ? '' : 's'}` +
              (hiddenFeedCount > 0 ? ` (${hiddenFeedCount} hidden by changed wedges only)` : '')
            : hiddenFeedCount > 0
              ? `${hiddenFeedCount} hidden by changed wedges only`
              : undefined}
          icon={<Activity className="h-5 w-5" />}
          skeleton={loading}
        />
        <StatCard
          label="Accounts In Motion"
          value={accountBuckets.primary.length}
          sub={vendorsWithAccounts > 0
            ? hasActiveAlertPolicy
              ? `${accountAlertHitCount} alert hit${accountAlertHitCount === 1 ? '' : 's'}` +
                (staleThresholdHitCount > 0 ? ` - ${staleThresholdHitCount} stale` : '') +
                (accountBuckets.review.length > 0 ? ` - ${accountBuckets.review.length} below threshold` : '') +
                (hiddenReviewAccountCount > 0
                  ? ` (${hiddenReviewAccountCount} hidden by named accounts only)`
                  : '')
              : `${visibleReviewAccounts.length} review-needed cluster${visibleReviewAccounts.length === 1 ? '' : 's'}` +
                (hiddenReviewAccountCount > 0
                  ? ` (${hiddenReviewAccountCount} hidden by named accounts only)`
                  : '')
            : 'No persisted account movement yet'}
          icon={<RefreshCw className="h-5 w-5" />}
          skeleton={loading}
        />
      </div>

      <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 p-4">
        <div className="flex flex-col gap-3 xl:flex-row xl:items-start xl:justify-between">
          <div>
            <h2 className="text-sm font-medium text-white">Saved Views</h2>
            <p className="mt-1 text-xs text-slate-500">
              Persist your current filters, review rules, and alert policy so the watchlists surface supports repeat monitoring instead of one transient default.
            </p>
          </div>
          <div className="flex w-full max-w-xl flex-col gap-2 sm:flex-row">
            <input
              aria-label="Saved view name"
              type="text"
              value={savedViewName}
              onChange={(event) => setSavedViewName(event.target.value)}
              placeholder={activeWatchlistView ? activeWatchlistView.name : 'Name this view'}
              className="w-full rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-cyan-500/50 focus:outline-none"
            />
            <button
              className="rounded-md bg-cyan-500/10 px-3 py-2 text-sm font-medium text-cyan-300 hover:bg-cyan-500/20 disabled:opacity-50"
              onClick={handleSaveWatchlistView}
              disabled={savingWatchlistView}
            >
              {savingWatchlistView ? 'Saving...' : activeWatchlistView ? 'Update active view' : 'Save current view'}
            </button>
          </div>
        </div>
        <div className="mt-4 grid gap-3 sm:grid-cols-3">
          <label className="space-y-1">
            <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Vendor Alert Threshold</span>
            <input
              aria-label="Vendor alert threshold"
              type="number"
              min="0"
              max="10"
              step="0.1"
              value={vendorAlertThreshold}
              onChange={(event) => setVendorAlertThreshold(event.target.value)}
              placeholder="Optional"
              className="w-full rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-cyan-500/50 focus:outline-none"
            />
          </label>
          <label className="space-y-1">
            <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Account Alert Threshold</span>
            <input
              aria-label="Account alert threshold"
              type="number"
              min="0"
              max="10"
              step="0.1"
              value={accountAlertThreshold}
              onChange={(event) => setAccountAlertThreshold(event.target.value)}
              placeholder="Optional"
              className="w-full rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-cyan-500/50 focus:outline-none"
            />
          </label>
          <label className="space-y-1">
            <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Stale Days Threshold</span>
            <input
              aria-label="Stale days threshold"
              type="number"
              min="0"
              max="365"
              step="1"
              value={staleDaysThreshold}
              onChange={(event) => setStaleDaysThreshold(event.target.value)}
              placeholder="Optional"
              className="w-full rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-cyan-500/50 focus:outline-none"
            />
          </label>
        </div>
        <p className="mt-2 text-[11px] text-slate-500">
          These values are persisted with the saved view and evaluated in this monitoring surface now. Scheduled email delivery uses the same stored policy.
        </p>
        <div className="mt-4 grid gap-3 sm:grid-cols-2">
          <label className="flex items-center gap-2 rounded-lg border border-slate-700/50 bg-slate-950/40 px-3 py-2 text-sm text-slate-200">
            <input
              aria-label="Enable alert email delivery"
              type="checkbox"
              checked={alertEmailEnabled}
              onChange={(event) => setAlertEmailEnabled(event.target.checked)}
              className="h-4 w-4 rounded border-slate-600 bg-slate-800 text-cyan-500 focus:ring-cyan-500/40"
            />
            <span>Enable scheduled alert emails</span>
          </label>
          <label className="space-y-1">
            <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Email cadence</span>
            <select
              aria-label="Alert delivery frequency"
              value={alertDeliveryFrequency}
              onChange={(event) => setAlertDeliveryFrequency(event.target.value as 'daily' | 'weekly')}
              disabled={!alertEmailEnabled}
              className="w-full rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2 text-sm text-white focus:border-cyan-500/50 focus:outline-none disabled:cursor-not-allowed disabled:opacity-50"
            >
              <option value="daily">Daily</option>
              <option value="weekly">Weekly</option>
            </select>
          </label>
        </div>
        {hasActiveAlertPolicy ? (
          <div className="mt-3 flex flex-wrap gap-2 text-[11px]">
            {activeVendorAlertThreshold != null ? (
              <span className="rounded-full border border-cyan-500/30 bg-cyan-500/10 px-2 py-0.5 text-cyan-300">
                Vendor alerts at {activeVendorAlertThreshold}+ urgency: {vendorAlertHitCount} hit{vendorAlertHitCount === 1 ? '' : 's'}
                {hiddenFeedCount > 0 ? ` (${hiddenFeedCount} hidden by changed wedges only)` : ''}
              </span>
            ) : null}
            {activeAccountAlertThreshold != null ? (
              <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-emerald-300">
                Account alerts at {activeAccountAlertThreshold}+ urgency: {accountAlertHitCount} hit{accountAlertHitCount === 1 ? '' : 's'}
                {hiddenReviewAccountCount > 0 ? ` (${hiddenReviewAccountCount} hidden by named accounts only)` : ''}
              </span>
            ) : null}
            {activeStaleDaysThreshold != null ? (
              <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-amber-300">
                Stale policy after {activeStaleDaysThreshold} day{activeStaleDaysThreshold === 1 ? '' : 's'}: {staleThresholdHitCount} hit{staleThresholdHitCount === 1 ? '' : 's'}
              </span>
            ) : null}
          </div>
        ) : null}
        <div className="mt-4 flex flex-wrap gap-2">
          {watchlistViews.length === 0 ? (
            <div className="rounded-lg border border-dashed border-slate-700/50 px-3 py-4 text-sm text-slate-500">
              No saved views yet. Save the current controls once you have a monitoring slice worth revisiting.
            </div>
          ) : watchlistViews.map((view) => {
            const isActive = activeWatchlistView?.id === view.id
            return (
              <div
                key={view.id}
                className={clsx(
                  'flex items-center gap-2 rounded-lg border px-3 py-2',
                  isActive
                    ? 'border-cyan-500/40 bg-cyan-500/10'
                    : 'border-slate-700/50 bg-slate-950/40',
                )}
              >
                <button
                  className="text-left"
                  onClick={() => applyWatchlistView(view)}
                >
                  <div className={clsx('text-sm font-medium', isActive ? 'text-cyan-200' : 'text-white')}>
                    {view.name}
                  </div>
                  <div className="mt-0.5 text-[11px] text-slate-400">
                    {summarizeWatchlistView(view)}
                  </div>
                  {view.alert_email_enabled ? (
                    <div className="mt-1 text-[11px] text-slate-500">
                      Next email {formatTs(view.next_alert_delivery_at)} · {summarizeSavedViewDelivery(view)}
                    </div>
                  ) : null}
                </button>
                <button
                  aria-label={`Copy link for saved view ${view.name}`}
                  className="rounded-md bg-slate-800 px-2 py-1 text-xs font-medium text-slate-300 hover:bg-slate-700"
                  onClick={() => void handleCopyWatchlistViewLink(view)}
                  type="button"
                >
                  Copy Link
                </button>
                <button
                  aria-label={`Delete saved view ${view.name}`}
                  className="rounded-md bg-rose-500/10 px-2 py-1 text-xs font-medium text-rose-300 hover:bg-rose-500/20 disabled:opacity-50"
                  onClick={() => handleDeleteWatchlistView(view)}
                  type="button"
                  disabled={deletingWatchlistViewId === view.id}
                >
                  Delete
                </button>
              </div>
            )
          })}
        </div>
      </div>

      <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 p-4">
        <div className="flex flex-col gap-3 sm:flex-row sm:items-start sm:justify-between">
          <div>
            <h2 className="text-sm font-medium text-white">Saved View Alert Events</h2>
            <p className="mt-1 text-xs text-slate-500">
              Persisted alert hits are tied to saved views so the same policy drives UI review and scheduled delivery.
            </p>
          </div>
          <div className="flex flex-col gap-2 sm:flex-row">
            <Link
              to={currentAlertsPath}
              className="rounded-md bg-violet-500/10 px-3 py-2 text-center text-sm font-medium text-violet-300 hover:bg-violet-500/20"
            >
              Alerts API
            </Link>
            <button
              className="rounded-md bg-slate-800 px-3 py-2 text-sm font-medium text-slate-200 hover:bg-slate-700"
              onClick={() => void handleCopyAlertsLink()}
              type="button"
            >
              Copy Alerts Link
            </button>
            <button
              className="rounded-md bg-emerald-500/10 px-3 py-2 text-sm font-medium text-emerald-300 hover:bg-emerald-500/20 disabled:opacity-50"
              onClick={handleEvaluateWatchlistAlerts}
              disabled={!activeWatchlistView || evaluatingWatchlistViewId === activeWatchlistView?.id}
            >
              {evaluatingWatchlistViewId === activeWatchlistView?.id ? 'Evaluating...' : 'Evaluate alerts'}
            </button>
            <button
              className="rounded-md bg-cyan-500/10 px-3 py-2 text-sm font-medium text-cyan-300 hover:bg-cyan-500/20 disabled:opacity-50"
              onClick={handleEmailWatchlistAlerts}
              disabled={!activeWatchlistView || emailingWatchlistViewId === activeWatchlistView?.id}
            >
              {emailingWatchlistViewId === activeWatchlistView?.id ? 'Emailing...' : 'Email open alerts'}
            </button>
          </div>
        </div>
        {!activeWatchlistView ? (
          <div className="mt-4 rounded-lg border border-dashed border-slate-700/50 px-3 py-4 text-sm text-slate-500">
            Apply a saved view to evaluate and review persisted alert events.
          </div>
        ) : (
          <div className="mt-4 space-y-3">
            <div className="flex flex-wrap items-center gap-2 text-xs">
              <span className="rounded-full border border-slate-700/60 bg-slate-800/60 px-2 py-0.5 text-slate-300">
                Active view: {activeWatchlistView.name}
              </span>
              <span className="rounded-full border border-slate-700/60 bg-slate-800/60 px-2 py-0.5 text-slate-300">
                {activeAlertEvents.length} open event{activeAlertEvents.length === 1 ? '' : 's'}
              </span>
              {activeWatchlistView.alert_email_enabled ? (
                <span className="rounded-full border border-cyan-500/20 bg-cyan-500/10 px-2 py-0.5 text-cyan-300">
                  email {activeWatchlistView.alert_delivery_frequency} · next {formatTs(activeWatchlistView.next_alert_delivery_at)}
                </span>
              ) : null}
              {refreshingAlertEvents ? (
                <span className="rounded-full border border-slate-700/60 bg-slate-800/60 px-2 py-0.5 text-slate-400">
                  refreshing
                </span>
              ) : null}
            </div>
            {activeAlertEvents.length === 0 ? (
              <div className="rounded-lg border border-dashed border-slate-700/50 px-3 py-4 text-sm text-slate-500">
                {activeAlertEventsEmptyState}
              </div>
            ) : (
              <div className="space-y-2">
                {activeAlertEvents.map((event) => {
                  const vendorName = watchlistAlertEventVendorName(event)
                  const primaryReviewId = watchlistAlertEventPrimaryReviewId(event)
                  const primaryWitnessId = watchlistAlertEventPrimaryWitnessId(event)
                  const eventSource = event.source || selectedSourceFilter || null
                  const accountReviewPath = watchlistAlertEventAccountPath(outboundWatchlistSearchParams, event)
                  const reviewPath = primaryReviewId
                    ? watchlistAlertEventReviewDetailPath(outboundWatchlistSearchParams, event, primaryReviewId)
                    : null
                  const evidencePath = watchlistAlertEventEvidencePath(outboundWatchlistSearchParams, event, null, eventSource)
                  const witnessPath = primaryWitnessId
                    ? watchlistAlertEventEvidencePath(outboundWatchlistSearchParams, event, primaryWitnessId, eventSource)
                    : null
                  const vendorWorkspacePath = watchlistAlertEventVendorPath(outboundWatchlistSearchParams, event)
                  const alertsPath = watchlistAlertEventAlertsPath(outboundWatchlistSearchParams, event)
                  const reportsPath = watchlistAlertEventReportsPath(outboundWatchlistSearchParams, event)
                  const opportunitiesPath = watchlistAlertEventOpportunitiesPath(outboundWatchlistSearchParams, event)
                  const eventLabel = event.company_name || vendorName
                  const alertScoreSourceLabel = watchlistAlertScoreSourceLabel(event.account_alert_score_source)
                  return (
                    <div
                      key={event.id}
                      className={clsx('rounded-lg border px-3 py-3', alertEventTone(event))}
                    >
                      <div className="flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between">
                        <div>
                          <div className="flex flex-wrap items-center gap-2 text-xs">
                            <span className="rounded-full border border-current/30 px-2 py-0.5">
                              {alertEventLabel(event)}
                            </span>
                            {event.vendor_name ? (
                              <span className="text-slate-100">{event.vendor_name}</span>
                            ) : null}
                            {event.company_name ? (
                              <span className="text-slate-300">- {event.company_name}</span>
                            ) : null}
                          </div>
                          <div className="mt-1 text-sm font-medium text-white">{event.summary}</div>
                          {event.event_type === 'account_alert' && (
                            <div className="mt-1 flex flex-wrap gap-3 text-[11px] text-slate-300">
                              {event.account_reasoning_preview_only ? (
                                <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-amber-200">
                                  Early account signal
                                </span>
                              ) : null}
                              {typeof event.account_alert_score === 'number' ? (
                                <span>
                                  Alert score: {event.account_alert_score.toFixed(1)}
                                  {alertScoreSourceLabel ? ` via ${alertScoreSourceLabel}` : ''}
                                </span>
                              ) : null}
                            </div>
                          )}
                          <div className="mt-1 flex flex-wrap gap-3 text-[11px] text-slate-400">
                            {event.category ? <span>Category: {event.category}</span> : null}
                            {event.source ? <span>Source: {event.source}</span> : null}
                            {event.threshold_value != null ? <span>Threshold: {event.threshold_value}</span> : null}
                          </div>
                          <div className="mt-2 flex flex-wrap gap-3 text-[11px]">
                            {accountReviewPath ? (
                              <Link
                                to={accountReviewPath}
                                aria-label={`Open alert account review for ${eventLabel}`}
                                className="text-emerald-300 hover:text-emerald-200"
                              >
                                Account Review
                              </Link>
                            ) : null}
                            {accountReviewPath ? (
                              <button
                                type="button"
                                onClick={() => void handleCopyAlertEventAccountReviewLink(event)}
                                aria-label={`Copy alert account review link for ${eventLabel}`}
                                className="text-slate-300 hover:text-slate-200"
                              >
                                Copy account review
                              </button>
                            ) : null}
                            {reviewPath ? (
                              <Link
                                to={reviewPath}
                                aria-label={`Open alert review detail for ${eventLabel}`}
                                className="text-sky-300 hover:text-sky-200"
                              >
                                Review
                              </Link>
                            ) : null}
                            {reviewPath ? (
                              <button
                                type="button"
                                onClick={() => void handleCopyAlertEventReviewLink(event, primaryReviewId)}
                                aria-label={`Copy alert review link for ${eventLabel}`}
                                className="text-slate-300 hover:text-slate-200"
                              >
                                Copy review
                              </button>
                            ) : null}
                            {witnessPath ? (
                              <Link
                                to={witnessPath}
                                aria-label={`Open alert witness for ${eventLabel}`}
                                className="text-cyan-300 hover:text-cyan-200"
                              >
                                Witness
                              </Link>
                            ) : null}
                            {witnessPath ? (
                              <button
                                type="button"
                                onClick={() => void handleCopyAlertEventWitnessLink(event, primaryWitnessId, eventSource)}
                                aria-label={`Copy alert witness link for ${eventLabel}`}
                                className="text-slate-300 hover:text-slate-200"
                              >
                                Copy witness
                              </button>
                            ) : null}
                            {alertsPath ? (
                              <Link
                                to={alertsPath}
                                aria-label={`Open alert delivery activity for ${vendorName}`}
                                className="text-violet-200 hover:text-violet-100"
                              >
                                Alerts API
                              </Link>
                            ) : null}
                            {vendorName ? (
                              <button
                                type="button"
                                onClick={() => void handleCopyAlertEventAlertsLink(event)}
                                aria-label={`Copy alert delivery activity link for ${vendorName}`}
                                className="text-slate-300 hover:text-slate-200"
                              >
                                Copy alerts
                              </button>
                            ) : null}
                            {evidencePath ? (
                              <Link
                                to={evidencePath}
                                aria-label={`Open alert evidence for ${vendorName}`}
                                className="text-violet-300 hover:text-violet-200"
                              >
                                Evidence
                              </Link>
                            ) : null}
                            {evidencePath ? (
                              <button
                                type="button"
                                onClick={() => void handleCopyAlertEventEvidenceLink(event, eventSource)}
                                aria-label={`Copy alert evidence link for ${vendorName}`}
                                className="text-slate-300 hover:text-slate-200"
                              >
                                Copy evidence
                              </button>
                            ) : null}
                            {vendorWorkspacePath ? (
                              <Link
                                to={vendorWorkspacePath}
                                aria-label={`Open alert vendor workspace for ${vendorName}`}
                                className="text-amber-300 hover:text-amber-200"
                              >
                                Vendor workspace
                              </Link>
                            ) : null}
                            {vendorName ? (
                              <button
                                type="button"
                                onClick={() => void handleCopyAlertEventVendorLink(event)}
                                aria-label={`Copy alert vendor workspace link for ${vendorName}`}
                                className="text-slate-300 hover:text-slate-200"
                              >
                                Copy vendor
                              </button>
                            ) : null}
                            {reportsPath ? (
                              <Link
                                to={reportsPath}
                                aria-label={`Open alert reports for ${vendorName}`}
                                className="text-fuchsia-300 hover:text-fuchsia-200"
                              >
                                Reports
                              </Link>
                            ) : null}
                            {vendorName ? (
                              <button
                                type="button"
                                onClick={() => void handleCopyAlertEventReportsLink(event)}
                                aria-label={`Copy alert reports link for ${vendorName}`}
                                className="text-slate-300 hover:text-slate-200"
                              >
                                Copy reports
                              </button>
                            ) : null}
                            {opportunitiesPath ? (
                              <Link
                                to={opportunitiesPath}
                                aria-label={`Open alert opportunities for ${vendorName}`}
                                className="text-orange-300 hover:text-orange-200"
                              >
                                Opportunities
                              </Link>
                            ) : null}
                            {vendorName ? (
                              <button
                                type="button"
                                onClick={() => void handleCopyAlertEventOpportunitiesLink(event)}
                                aria-label={`Copy alert opportunities link for ${vendorName}`}
                                className="text-slate-300 hover:text-slate-200"
                              >
                                Copy opportunities
                              </button>
                            ) : null}
                          </div>
                        </div>
                        <div className="text-[11px] text-slate-400">
                          Last seen {formatTs(event.last_seen_at)}
                        </div>
                      </div>
                    </div>
                  )
                })}
              </div>
            )}
            <div className="rounded-lg border border-slate-800/60 bg-slate-950/40 p-3">
              <div className="flex flex-wrap items-center gap-2 text-xs">
                <span className="font-medium text-slate-200">Email delivery log</span>
                {refreshingAlertEmailLog ? (
                  <span className="rounded-full border border-slate-700/60 bg-slate-800/60 px-2 py-0.5 text-slate-400">
                    refreshing
                  </span>
                ) : null}
              </div>
              {activeAlertEmailDeliveries.length === 0 ? (
                <div className="mt-2 text-sm text-slate-500">No watchlist alert emails have been sent for this saved view yet.</div>
              ) : (
                <div className="mt-3 space-y-2">
                  {activeAlertEmailDeliveries.map((delivery) => (
                    <div key={delivery.id} className="rounded-md border border-slate-800/60 bg-slate-900/50 px-3 py-2">
                      <div className="flex flex-wrap items-center gap-2 text-xs">
                        <span className="rounded-full border border-slate-700/60 bg-slate-800/60 px-2 py-0.5 text-slate-200">
                          {delivery.status}
                        </span>
                        <span className="text-slate-400">{delivery.event_count} event{delivery.event_count === 1 ? '' : 's'}</span>
                        <span className="text-slate-500">{formatTs(delivery.delivered_at || delivery.created_at)}</span>
                      </div>
                      <div className="mt-1 text-sm text-slate-200">
                        {summarizeWatchlistDeliverySummary(delivery.summary, delivery.status, activeWatchlistView)}
                      </div>
                      <div className="mt-1 text-xs text-slate-500">
                        {delivery.recipient_emails.join(', ') || '--'}
                      </div>
                      {delivery.error ? (
                        <div className="mt-1 text-xs text-rose-300">{delivery.error}</div>
                      ) : null}
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>
        )}
      </div>

      <div
        className="rounded-xl border border-slate-700/50 bg-slate-900/50 p-4"
        role="group"
        aria-label="Feed controls"
      >
        <div className="flex flex-col gap-3 xl:flex-row xl:items-end xl:justify-between">
          <div>
            <h2 className="text-sm font-medium text-white">Feed Controls</h2>
            <p className="mt-1 text-xs text-slate-500">
              Vendor and category apply to both feeds. Source, urgency floor, freshness, and named-account review rules apply to Accounts In Motion. Changed-wedge review applies to the vendor feed.
            </p>
          </div>
          {hasActiveFeedFilters ? (
            <button
              className="rounded-md bg-slate-800 px-2.5 py-1.5 text-xs font-medium text-slate-300 hover:bg-slate-700"
              onClick={() => {
                setSelectedVendorFilter('')
                setSelectedVendorFilters([])
                setSelectedCategoryFilter('')
                setSelectedSourceFilter('')
                setSelectedMinUrgency('')
                setFreshOnly(false)
                setNamedAccountsOnly(false)
                setChangedWedgesOnly(false)
                setVendorAlertThreshold('')
                setAccountAlertThreshold('')
                setStaleDaysThreshold('')
                setSavedViewName('')
              }}
            >
              Clear filters
            </button>
          ) : null}
        </div>
        <div className="mt-4 grid gap-3 sm:grid-cols-2 xl:grid-cols-7">
          <label className="space-y-1">
            <span className="text-xs font-medium uppercase tracking-wide text-slate-500">
              Vendors {selectedVendorFilters.length > 0 && `(${selectedVendorFilters.length})`}
            </span>
            <select
              className="w-full rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2 text-sm text-white focus:border-cyan-500/50 focus:outline-none"
              multiple
              size={Math.min(vendorFilterOptions.length + 1, 6)}
              value={selectedVendorFilters}
              onChange={(event) => {
                const selected = Array.from(event.target.selectedOptions, (o) => o.value)
                setSelectedVendorFilters(selected)
                setSelectedVendorFilter(selected[0] || '')
              }}
            >
              {vendorFilterOptions.map((vendorName) => (
                <option key={vendorName} value={vendorName}>
                  {vendorName}
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-1">
            <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Category</span>
            <select
              className="w-full rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2 text-sm text-white focus:border-cyan-500/50 focus:outline-none"
              value={selectedCategoryFilter}
              onChange={(event) => setSelectedCategoryFilter(event.target.value)}
            >
              <option value="">All categories</option>
              {categoryFilterOptions.map((categoryName) => (
                <option key={categoryName} value={categoryName}>
                  {categoryName}
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-1">
            <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Source</span>
            <select
              className="w-full rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2 text-sm text-white focus:border-cyan-500/50 focus:outline-none"
              value={selectedSourceFilter}
              onChange={(event) => setSelectedSourceFilter(event.target.value)}
            >
              <option value="">All sources</option>
              {sourceFilterOptions.map((sourceName) => (
                <option key={sourceName} value={sourceName}>
                  {sourceName}
                </option>
              ))}
            </select>
          </label>
          <label className="space-y-1">
            <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Min Urgency</span>
            <select
              className="w-full rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2 text-sm text-white focus:border-cyan-500/50 focus:outline-none"
              value={selectedMinUrgency}
              onChange={(event) => setSelectedMinUrgency(event.target.value)}
            >
              <option value="">Default urgency floor</option>
              {ACCOUNT_URGENCY_FILTER_OPTIONS.map((score) => (
                <option key={score} value={String(score)}>
                  {score}+
                </option>
              ))}
            </select>
          </label>
          <label className="flex items-center gap-2 rounded-lg border border-slate-700/50 bg-slate-800/30 px-3 py-2 text-sm text-slate-300 xl:mt-6">
            <input
              checked={freshOnly}
              className="rounded border-slate-600 bg-slate-900 text-cyan-400 focus:ring-cyan-500"
              onChange={(event) => setFreshOnly(event.target.checked)}
              type="checkbox"
            />
            Fresh only
          </label>
          <label className="flex items-center gap-2 rounded-lg border border-slate-700/50 bg-slate-800/30 px-3 py-2 text-sm text-slate-300 xl:mt-6">
            <input
              checked={namedAccountsOnly}
              className="rounded border-slate-600 bg-slate-900 text-cyan-400 focus:ring-cyan-500"
              onChange={(event) => setNamedAccountsOnly(event.target.checked)}
              type="checkbox"
            />
            Named accounts only
          </label>
          <label className="flex items-center gap-2 rounded-lg border border-slate-700/50 bg-slate-800/30 px-3 py-2 text-sm text-slate-300 xl:mt-6">
            <input
              checked={changedWedgesOnly}
              className="rounded border-slate-600 bg-slate-900 text-cyan-400 focus:ring-cyan-500"
              onChange={(event) => setChangedWedgesOnly(event.target.checked)}
              type="checkbox"
            />
            Changed wedges only
          </label>
        </div>
      </div>

      {(data?.loadWarnings?.length || actionMessage || actionError) && (
        <div className="space-y-3">
          {data?.loadWarnings?.map((warning) => (
            <div
              key={warning}
              className="rounded-xl border border-amber-500/30 bg-amber-500/10 px-4 py-3 text-sm text-amber-100"
            >
              {warning}
            </div>
          ))}
          {(actionMessage || actionError) && (
            <div
              className={clsx(
                'flex items-start justify-between gap-3 rounded-xl border px-4 py-3 text-sm',
                actionError
                  ? 'border-rose-500/30 bg-rose-500/10 text-rose-200'
                  : 'border-emerald-500/30 bg-emerald-500/10 text-emerald-200',
              )}
            >
              <span>{actionError || actionMessage}</span>
              <button
                onClick={() => {
                  setActionError(null)
                  setActionMessage(null)
                }}
                className="text-current/70 hover:text-current"
              >
                <X className="h-4 w-4" />
              </button>
            </div>
          )}
        </div>
      )}

      <div className="grid gap-6 xl:grid-cols-[1.1fr,0.9fr]">
        <div className="space-y-6">
          <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 overflow-hidden">
            <div className="border-b border-slate-700/50 px-4 py-3">
              <h2 className="text-sm font-medium text-white">Vendor Movement Feed</h2>
              <p className="text-xs text-slate-500">
                Existing slow-burn watchlist data for your tracked vendors. Click a row to inspect the vendor surface.
              </p>
            </div>
            <DataTable
              columns={feedColumns}
              data={filteredFeed}
              onRowClick={(row) => navigate(watchlistVendorPath(searchParams, row.vendor_name))}
              emptyMessage={hasActiveFeedFilters
                ? 'No vendor movement matches the current filters.'
                : 'No watchlist movement yet. Add tracked vendors to start monitoring.'}
              skeletonRows={loading ? 6 : undefined}
            />
          </div>

          <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 overflow-hidden">
            <div className="border-b border-slate-700/50 px-4 py-3">
              <h2 className="text-sm font-medium text-white">Tracked Vendors</h2>
              <p className="text-xs text-slate-500">
                Your current monitored portfolio with direct links into the existing vendor detail surface.
              </p>
            </div>
            <DataTable
              columns={trackedColumns}
              data={trackedVendors}
              onRowClick={(row) => navigate(watchlistVendorPath(searchParams, row.vendor_name))}
              emptyMessage="No tracked vendors yet."
              emptyAction={{ label: 'Add your first vendor', onClick: () => document.getElementById('watchlist-search')?.focus() }}
              skeletonRows={loading ? 5 : undefined}
            />
          </div>

          <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 overflow-hidden">
            <div className="border-b border-slate-700/50 px-4 py-3">
              <h2 className="text-sm font-medium text-white">Accounts In Motion</h2>
              <p className="text-xs text-slate-500">
                Tenant-wide account movement aggregated from persisted accounts-in-motion reports across your tracked vendors. Click a row to inspect source-review lineage and reasoning references.
              </p>
              <div className="mt-2 flex flex-wrap gap-2 text-[11px]">
                <span className="rounded-full border border-emerald-500/30 bg-emerald-500/10 px-2 py-0.5 text-emerald-300">
                  {accountBuckets.high.length} named high confidence
                </span>
                <span className="rounded-full border border-amber-500/30 bg-amber-500/10 px-2 py-0.5 text-amber-300">
                  {accountBuckets.medium.length} named moderate confidence
                </span>
                {visibleReviewAccounts.length > 0 ? (
                  <span className="rounded-full border border-rose-500/30 bg-rose-500/10 px-2 py-0.5 text-rose-300">
                    {visibleReviewAccounts.length} review-needed cluster{visibleReviewAccounts.length === 1 ? '' : 's'}
                  </span>
                ) : null}
              </div>
              {freshestAccountsReportDate ? (
                <p className={clsx('mt-1 text-[11px]', freshnessTone('fresh', freshestAccountsReportDate))}>
                  Freshest report {formatTs(freshestAccountsReportDate)}
                </p>
              ) : null}
            </div>
            <DataTable
              columns={accountColumns}
              data={accountBuckets.primary}
              onRowClick={(row) => setSelectedAccount(row)}
              emptyMessage={visibleReviewAccounts.length > 0
                ? 'No named higher-confidence account rows yet. Review the lower-confidence cluster section below.'
                : hasActiveFeedFilters
                  ? 'No accounts-in-motion rows match the current filters.'
                  : 'No persisted accounts-in-motion rows yet for your tracked vendors.'}
              skeletonRows={loading ? 5 : undefined}
            />
          </div>

          {visibleReviewAccounts.length > 0 && (
            <div className="rounded-xl border border-amber-500/20 bg-amber-500/5 overflow-hidden">
              <div className="border-b border-amber-500/20 px-4 py-3">
                <div className="flex items-start justify-between gap-3">
                  <div>
                    <h2 className="text-sm font-medium text-amber-100">Review-Needed Signal Clusters</h2>
                    <p className="mt-1 text-xs text-amber-200/80">
                      These rows are still evidence-backed, but account identity or confidence is weaker. Treat them as signal clusters until the evidence drawer supports the account claim.
                    </p>
                  </div>
                  <button
                    className="rounded-md bg-amber-500/10 px-2.5 py-1 text-xs font-medium text-amber-200 hover:bg-amber-500/20"
                    onClick={() => setShowReviewAccounts((current) => !current)}
                  >
                    {showReviewAccounts ? 'Hide clusters' : `Show ${visibleReviewAccounts.length} cluster${visibleReviewAccounts.length === 1 ? '' : 's'}`}
                  </button>
                </div>
              </div>
              {showReviewAccounts ? (
                <DataTable
                  columns={accountColumns}
                  data={visibleReviewAccounts}
                  onRowClick={(row) => setSelectedAccount(row)}
                  emptyMessage="No review-needed signal clusters."
                />
              ) : null}
            </div>
          )}
        </div>

        <div className="space-y-6">
          <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 p-4">
            <div className="mb-3">
              <h2 className="text-sm font-medium text-white">Add Vendor To Watchlists</h2>
              <p className="mt-1 text-xs text-slate-500">
                Search existing vendors, choose whether you track them as your own footprint or a competitor, then add them to your monitored set.
              </p>
            </div>

            <div className="space-y-3">
              <div className="relative">
                <Search className="absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-slate-500" />
                <input
                  id="watchlist-search"
                  type="text"
                  placeholder="Search vendors..."
                  value={searchInput}
                  onChange={(event) => setSearchInput(event.target.value)}
                  className="w-full rounded-lg border border-slate-700/50 bg-slate-800/50 py-2 pl-9 pr-3 text-sm text-white placeholder-slate-500 focus:border-cyan-500/50 focus:outline-none"
                />
              </div>

              <div className="grid gap-3 sm:grid-cols-2">
                <label className="space-y-1">
                  <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Track Mode</span>
                  <select
                    value={trackMode}
                    onChange={(event) => setTrackMode(event.target.value as 'own' | 'competitor')}
                    className="w-full rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2 text-sm text-white focus:border-cyan-500/50 focus:outline-none"
                  >
                    <option value="competitor">Competitor</option>
                    <option value="own">Own vendor</option>
                  </select>
                </label>
                <label className="space-y-1">
                  <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Label</span>
                  <input
                    type="text"
                    value={label}
                    onChange={(event) => setLabel(event.target.value)}
                    placeholder="Optional team label"
                    className="w-full rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-cyan-500/50 focus:outline-none"
                  />
                </label>
              </div>
            </div>

            <div className="mt-4 space-y-2">
              {debouncedSearch.length < MIN_VENDOR_SEARCH_CHARS ? (
                <div className="rounded-lg border border-dashed border-slate-700/50 px-3 py-4 text-sm text-slate-500">
                  Type at least {MIN_VENDOR_SEARCH_CHARS} characters to search available vendors.
                </div>
              ) : searchLoading ? (
                <div className="rounded-lg border border-slate-700/50 bg-slate-950/40 px-3 py-4 text-sm text-slate-400">
                  Searching vendors...
                </div>
              ) : candidateResults.length === 0 ? (
                <div className="rounded-lg border border-slate-700/50 bg-slate-950/40 px-3 py-4 text-sm text-slate-400">
                  No untracked vendor matches found for this query.
                </div>
              ) : (
                <div className="space-y-2">
                  {candidateResults.slice(0, SEARCH_RESULTS_PREVIEW_LIMIT).map((vendor) => (
                    <div
                      key={vendor.vendor_name}
                      className="flex items-center justify-between gap-3 rounded-lg border border-slate-700/50 bg-slate-950/40 px-3 py-3"
                    >
                      <div className="min-w-0">
                        <div className="truncate font-medium text-white">{vendor.vendor_name}</div>
                        <div className="mt-1 flex flex-wrap gap-x-3 gap-y-1 text-xs text-slate-500">
                          <span>{vendor.product_category ?? 'Uncategorized'}</span>
                          <span>{vendor.total_reviews ?? 0} reviews</span>
                          <span>avg urgency {vendor.avg_urgency?.toFixed(1) ?? '--'}</span>
                        </div>
                      </div>
                      <button
                        onClick={() => handleAddVendor(vendor.vendor_name)}
                        disabled={submittingVendor === vendor.vendor_name}
                        className="inline-flex items-center gap-1 rounded-md bg-cyan-500/10 px-2.5 py-1.5 text-xs font-medium text-cyan-300 hover:bg-cyan-500/20 disabled:opacity-50"
                      >
                        <Plus className="h-3 w-3" />
                        Add
                      </button>
                    </div>
                  ))}
                </div>
              )}
            </div>
          </div>

          <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 p-4">
            <div className="mb-3 flex items-start justify-between gap-3">
              <div>
                <h2 className="text-sm font-medium text-white">Competitive Sets</h2>
                <p className="mt-1 text-xs text-slate-500">
                  Define a focal vendor and exact competitors, then refresh synthesis only for that scope instead of the full vendor universe.
                </p>
              </div>
              {editingCompetitiveSetId ? (
                <button
                  onClick={resetCompetitiveSetForm}
                  className="rounded-md bg-slate-800 px-2 py-1 text-xs text-slate-300 hover:bg-slate-700"
                >
                  Cancel edit
                </button>
              ) : null}
            </div>

            <div className="space-y-3">
              <label className="space-y-1">
                <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Set Name</span>
                <input
                  type="text"
                  value={competitiveSetName}
                  onChange={(event) => setCompetitiveSetName(event.target.value)}
                  placeholder="Salesforce core competitors"
                  className="w-full rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2 text-sm text-white placeholder-slate-500 focus:border-cyan-500/50 focus:outline-none"
                />
              </label>

              <div className="grid gap-3 sm:grid-cols-2">
                <label className="space-y-1">
                  <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Focal Vendor</span>
                  <select
                    value={competitiveSetFocal}
                    onChange={(event) => {
                      setCompetitiveSetFocal(event.target.value)
                      setCompetitiveSetCompetitors((current) => current.filter((item) => item !== event.target.value))
                    }}
                    className="w-full rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2 text-sm text-white focus:border-cyan-500/50 focus:outline-none"
                  >
                    <option value="">Select tracked vendor</option>
                    {focalOptions.map((vendor) => (
                      <option key={vendor.id} value={vendor.vendor_name}>
                        {vendor.vendor_name} - {vendor.track_mode}
                      </option>
                    ))}
                  </select>
                </label>
                <label className="space-y-1">
                  <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Refresh Mode</span>
                  <select
                    value={competitiveSetRefreshMode}
                    onChange={(event) => setCompetitiveSetRefreshMode(event.target.value as 'manual' | 'scheduled')}
                    className="w-full rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2 text-sm text-white focus:border-cyan-500/50 focus:outline-none"
                  >
                    <option value="manual">Manual</option>
                    <option value="scheduled">Scheduled</option>
                  </select>
                </label>
              </div>

              {competitiveSetRefreshMode === 'scheduled' ? (
                <label className="space-y-1">
                  <span className="text-xs font-medium uppercase tracking-wide text-slate-500">Refresh Every (Hours)</span>
                  <input
                    type="number"
                    min={1}
                    max={720}
                    value={competitiveSetRefreshHours}
                    onChange={(event) => setCompetitiveSetRefreshHours(event.target.value)}
                    className="w-full rounded-lg border border-slate-700/50 bg-slate-800/50 px-3 py-2 text-sm text-white focus:border-cyan-500/50 focus:outline-none"
                  />
                </label>
              ) : null}

              <div className="space-y-2 rounded-lg border border-slate-700/50 bg-slate-950/40 p-3">
                <div className="text-xs font-medium uppercase tracking-wide text-slate-500">Tracked Competitors</div>
                <div className="grid gap-2">
                  {competitorOptions.length === 0 ? (
                    <div className="text-sm text-slate-500">Add tracked vendors first, then choose the ones to compare against the focal vendor.</div>
                  ) : competitorOptions.map((vendor) => (
                    <label key={vendor.id} className="flex items-center gap-2 text-sm text-slate-300">
                      <input
                        type="checkbox"
                        checked={competitiveSetCompetitors.includes(vendor.vendor_name)}
                        onChange={(event) => {
                          setCompetitiveSetCompetitors((current) => {
                            if (event.target.checked) return [...current, vendor.vendor_name]
                            return current.filter((item) => item !== vendor.vendor_name)
                          })
                        }}
                        className="rounded border-slate-600 bg-slate-900 text-cyan-400 focus:ring-cyan-500"
                      />
                      <span>{vendor.vendor_name}</span>
                      <span className="text-xs text-slate-500">{vendor.track_mode}</span>
                    </label>
                  ))}
                </div>
              </div>

              <div className="grid gap-2 sm:grid-cols-2">
                <label className="flex items-center gap-2 rounded-lg border border-slate-700/50 bg-slate-950/40 px-3 py-2 text-sm text-slate-300">
                  <input type="checkbox" checked={competitiveSetActive} onChange={(event) => setCompetitiveSetActive(event.target.checked)} />
                  Active
                </label>
                <label className="flex items-center gap-2 rounded-lg border border-slate-700/50 bg-slate-950/40 px-3 py-2 text-sm text-slate-300">
                  <input type="checkbox" checked={competitiveSetVendorEnabled} onChange={(event) => setCompetitiveSetVendorEnabled(event.target.checked)} />
                  Vendor synthesis
                </label>
                <label className="flex items-center gap-2 rounded-lg border border-slate-700/50 bg-slate-950/40 px-3 py-2 text-sm text-slate-300">
                  <input type="checkbox" checked={competitiveSetPairwiseEnabled} onChange={(event) => setCompetitiveSetPairwiseEnabled(event.target.checked)} />
                  Pairwise
                </label>
                <label className="flex items-center gap-2 rounded-lg border border-slate-700/50 bg-slate-950/40 px-3 py-2 text-sm text-slate-300">
                  <input type="checkbox" checked={competitiveSetCategoryEnabled} onChange={(event) => setCompetitiveSetCategoryEnabled(event.target.checked)} />
                  Category council
                </label>
                <label className="flex items-center gap-2 rounded-lg border border-slate-700/50 bg-slate-950/40 px-3 py-2 text-sm text-slate-300 sm:col-span-2">
                  <input type="checkbox" checked={competitiveSetAsymmetryEnabled} onChange={(event) => setCompetitiveSetAsymmetryEnabled(event.target.checked)} />
                  Resource asymmetry
                </label>
              </div>

              <button
                onClick={handleSaveCompetitiveSet}
                disabled={
                  savingCompetitiveSet
                  || !competitiveSetName.trim()
                  || !competitiveSetFocal
                  || competitiveSetCompetitors.length === 0
                }
                className="inline-flex items-center gap-2 rounded-lg bg-cyan-500/10 px-3 py-2 text-sm font-medium text-cyan-300 hover:bg-cyan-500/20 disabled:opacity-50"
              >
                <Plus className="h-4 w-4" />
                {editingCompetitiveSetId ? 'Update competitive set' : 'Create competitive set'}
              </button>
            </div>

            <div className="mt-5 space-y-3">
              {competitiveSets.length === 0 ? (
                <div className="rounded-lg border border-dashed border-slate-700/50 px-3 py-4 text-sm text-slate-500">
                  No competitive sets yet. Build one from your tracked vendors to keep synthesis scoped to the exact vendors you care about.
                </div>
              ) : competitiveSets.map((item) => {
                const competitorCount = item.competitor_vendor_names.length
                const vendorJobCount = item.vendor_synthesis_enabled ? competitorCount + 1 : 0
                const pairwiseJobCount = item.pairwise_enabled ? competitorCount : 0
                const asymmetryJobCount = item.asymmetry_enabled ? competitorCount : 0
                const preview = competitiveSetPreviews[item.id]
                const estimate = preview?.estimate
                const recentRuns = competitiveSetRuns[item.id] ?? []
                const likelyRerunVendors = (estimate?.likely_rerun_vendors ?? []).slice(0, 6)
                const likelyReuseVendors = (estimate?.likely_reuse_vendors ?? []).slice(0, 6)
                const previewOpen = openCompetitiveSetPreviewId === item.id
                const changedOnly = competitiveSetChangedOnly[item.id] ?? true
                const forceRun = competitiveSetForceRun[item.id] ?? false
                const forceCrossVendor = competitiveSetForceCrossVendor[item.id] ?? false
                const lastRunOverride = competitiveSetLastRunOverrides[item.id]
                const lastRunStatus = lastRunOverride?.last_run_status ?? item.last_run_status
                const lastRunSummary = lastRunOverride?.last_run_summary ?? item.last_run_summary ?? {}
                const lastRunSkipReason = typeof lastRunSummary._skip_synthesis === 'string'
                  ? lastRunSummary._skip_synthesis
                  : null
                return (
                  <div key={item.id} className="rounded-lg border border-slate-700/50 bg-slate-950/40 p-3">
                    <div className="flex items-start justify-between gap-3">
                      <div>
                        <div className="font-medium text-white">{item.name}</div>
                        <div className="mt-1 text-xs text-slate-500">
                          Focal: {item.focal_vendor_name} - {competitorCount} competitors - {item.refresh_mode}
                          {item.refresh_mode === 'scheduled' && item.refresh_interval_hours
                            ? ` every ${item.refresh_interval_hours}h`
                            : ''}
                        </div>
                      </div>
                      <span
                        className={clsx(
                          'rounded-full px-2 py-0.5 text-[11px] font-medium',
                          lastRunStatus === 'succeeded' && 'bg-emerald-500/10 text-emerald-300',
                          lastRunStatus === 'partial' && 'bg-amber-500/10 text-amber-300',
                          lastRunStatus === 'failed' && 'bg-rose-500/10 text-rose-300',
                          lastRunStatus === 'running' && 'bg-cyan-500/10 text-cyan-300',
                          !lastRunStatus && 'bg-slate-800 text-slate-300',
                        )}
                      >
                        {lastRunStatus ?? 'never run'}
                      </span>
                    </div>
                    <div className="mt-2 flex flex-wrap gap-2 text-xs text-slate-400">
                      <span>{vendorJobCount} vendor jobs</span>
                      <span>{pairwiseJobCount} pairwise jobs</span>
                      {item.category_council_enabled ? <span>category council enabled</span> : null}
                      {item.asymmetry_enabled ? <span>{asymmetryJobCount} asymmetry pairs</span> : null}
                    </div>
                    {(lastRunSummary.changed_vendors_only === true
                      || lastRunSummary.force === true
                      || lastRunSummary.force_cross_vendor === true
                      || lastRunSkipReason) ? (
                      <div className="mt-2 flex flex-wrap gap-2 text-[11px] text-slate-400">
                        {lastRunSummary.changed_vendors_only === true ? <span>changed only</span> : null}
                        {lastRunSummary.force === true ? <span>vendor forced</span> : null}
                        {lastRunSummary.force_cross_vendor === true ? <span>cross-vendor forced</span> : null}
                        {lastRunSkipReason ? <span>{lastRunSkipReason}</span> : null}
                      </div>
                    ) : null}
                    <div className="mt-2 flex flex-wrap gap-1">
                      {item.competitor_vendor_names.map((vendorName) => (
                        <span key={vendorName} className="rounded-full bg-slate-800 px-2 py-0.5 text-[11px] text-slate-300">
                          {vendorName}
                        </span>
                      ))}
                    </div>
                    <div className="mt-3 flex flex-wrap gap-2">
                      <button
                        onClick={() => handlePreviewCompetitiveSet(item)}
                        disabled={previewingCompetitiveSetId === item.id}
                        className="rounded-md bg-cyan-500/10 px-2.5 py-1 text-xs font-medium text-cyan-300 hover:bg-cyan-500/20 disabled:opacity-50"
                      >
                        {previewingCompetitiveSetId === item.id ? 'Loading preview...' : 'Preview cost'}
                      </button>
                      <button
                        onClick={() => loadCompetitiveSetForEdit(item)}
                        className="rounded-md bg-slate-800 px-2.5 py-1 text-xs font-medium text-slate-300 hover:bg-slate-700"
                      >
                        Edit
                      </button>
                      <button
                        onClick={() => handleDeleteCompetitiveSet(item)}
                        disabled={deletingCompetitiveSetId === item.id}
                        className="rounded-md bg-rose-500/10 px-2.5 py-1 text-xs font-medium text-rose-300 hover:bg-rose-500/20 disabled:opacity-50"
                      >
                        Delete
                      </button>
                    </div>
                    {previewOpen && preview ? (
                      <div className="mt-3 rounded-lg border border-cyan-500/20 bg-slate-900/70 p-3">
                        <div className="flex items-start justify-between gap-3">
                          <div>
                            <div className="text-sm font-medium text-white">Run Preview</div>
                            <div className="mt-1 text-xs text-slate-400">
                              Estimated upper bound for a non-forced run over the next {preview.estimate?.lookback_days ?? '--'} days of history.
                            </div>
                          </div>
                          <button
                            onClick={() => setOpenCompetitiveSetPreviewId((current) => current === item.id ? null : current)}
                            className="rounded-md bg-slate-800 px-2 py-1 text-[11px] font-medium text-slate-300 hover:bg-slate-700"
                          >
                            Close
                          </button>
                        </div>
                        <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-4">
                          <div className="rounded-md border border-slate-700/50 bg-slate-950/50 p-2">
                            <div className="text-[11px] uppercase tracking-wide text-slate-500">Planned Jobs</div>
                            <div className="mt-1 text-sm font-medium text-white">{formatWholeNumber(preview.estimated_total_jobs)}</div>
                            <div className="mt-1 text-[11px] text-slate-400">
                              {formatWholeNumber(preview.vendor_job_count)} vendor - {formatWholeNumber(preview.pairwise_job_count + preview.category_job_count + preview.asymmetry_job_count)} cross-vendor
                            </div>
                          </div>
                          <div className="rounded-md border border-slate-700/50 bg-slate-950/50 p-2">
                            <div className="text-[11px] uppercase tracking-wide text-slate-500">Token Estimate</div>
                            <div className="mt-1 text-sm font-medium text-white">{formatWholeNumber(estimate?.estimated_total_tokens)}</div>
                            <div className="mt-1 text-[11px] text-slate-400">
                              {formatWholeNumber(estimate?.estimated_vendor_tokens)} vendor - {formatWholeNumber(estimate?.estimated_cross_vendor_tokens)} cross-vendor
                            </div>
                          </div>
                          <div className="rounded-md border border-slate-700/50 bg-slate-950/50 p-2">
                            <div className="text-[11px] uppercase tracking-wide text-slate-500">Cost Upper Bound</div>
                            <div className="mt-1 text-sm font-medium text-white">{formatCostUsd(estimate?.estimated_total_cost_usd)}</div>
                            <div className="mt-1 text-[11px] text-slate-400">
                              {formatCostUsd(estimate?.estimated_vendor_cost_usd)} vendor - {formatCostUsd(estimate?.estimated_cross_vendor_cost_usd)} cross-vendor
                            </div>
                          </div>
                          <div className="rounded-md border border-slate-700/50 bg-slate-950/50 p-2">
                            <div className="text-[11px] uppercase tracking-wide text-slate-500">History Coverage</div>
                            <div className="mt-1 text-sm font-medium text-white">
                              {formatWholeNumber(estimate?.vendor_jobs_with_history)} vendor - {formatWholeNumber(estimate?.cross_vendor_jobs_with_history)} cross-vendor
                            </div>
                            <div className="mt-1 text-[11px] text-slate-400">
                              {formatWholeNumber(estimate?.vendor_jobs_using_fallback)} vendor fallback - {formatWholeNumber(estimate?.cross_vendor_jobs_using_fallback)} cross fallback
                            </div>
                          </div>
                        </div>
                        <div className="mt-3 grid gap-2 sm:grid-cols-2 xl:grid-cols-3">
                          <div className="rounded-md border border-slate-700/50 bg-slate-950/50 p-2">
                            <div className="text-[11px] uppercase tracking-wide text-slate-500">Likely Vendor Outcome</div>
                            <div className="mt-1 text-sm font-medium text-white">
                              {formatWholeNumber(estimate?.vendor_jobs_likely_to_reason)} rerun - {formatWholeNumber(estimate?.vendor_jobs_likely_hash_reuse)} reuse
                            </div>
                            <div className="mt-1 text-[11px] text-slate-400">
                              {formatWholeNumber(estimate?.vendor_jobs_missing_pools)} no pool - {formatWholeNumber(estimate?.vendor_jobs_likely_stale_reuse)} stale reuse
                            </div>
                          </div>
                          <div className="rounded-md border border-slate-700/50 bg-slate-950/50 p-2">
                            <div className="text-[11px] uppercase tracking-wide text-slate-500">Likely Vendor Spend</div>
                            <div className="mt-1 text-sm font-medium text-white">{formatCostUsd(estimate?.estimated_vendor_cost_usd_likely_to_reason)}</div>
                            <div className="mt-1 text-[11px] text-slate-400">
                              {formatWholeNumber(estimate?.estimated_vendor_tokens_likely_to_reason)} vendor tokens if changed-only
                            </div>
                          </div>
                          <div className="rounded-md border border-slate-700/50 bg-slate-950/50 p-2">
                            <div className="text-[11px] uppercase tracking-wide text-slate-500">Rerun Reasons</div>
                            <div className="mt-1 text-sm font-medium text-white">
                              {formatWholeNumber(estimate?.vendor_jobs_likely_hash_changed)} changed - {formatWholeNumber(estimate?.vendor_jobs_likely_prior_quality_weak)} weak
                            </div>
                            <div className="mt-1 text-[11px] text-slate-400">
                              {formatWholeNumber(estimate?.vendor_jobs_likely_missing_prior)} missing prior - {formatWholeNumber(estimate?.vendor_jobs_likely_missing_reference_ids)} missing refs
                            </div>
                          </div>
                        </div>
                        <div className="mt-3 text-xs text-slate-400">
                          {estimate?.note ?? 'Estimate unavailable.'}
                        </div>
                        {likelyRerunVendors.length > 0 ? (
                          <div className="mt-3">
                            <div className="text-[11px] uppercase tracking-wide text-slate-500">Likely Rerun Vendors</div>
                            <div className="mt-2 flex flex-wrap gap-1">
                              {likelyRerunVendors.map((entry) => (
                                <span key={entry} className="rounded-full bg-amber-500/10 px-2 py-0.5 text-[11px] text-amber-300">
                                  {entry.replace(':', ' - ')}
                                </span>
                              ))}
                            </div>
                          </div>
                        ) : null}
                        {likelyReuseVendors.length > 0 ? (
                          <div className="mt-3">
                            <div className="text-[11px] uppercase tracking-wide text-slate-500">Likely Reuse Vendors</div>
                            <div className="mt-2 flex flex-wrap gap-1">
                              {likelyReuseVendors.map((entry) => (
                                <span key={entry} className="rounded-full bg-emerald-500/10 px-2 py-0.5 text-[11px] text-emerald-300">
                                  {entry.replace(':', ' - ')}
                                </span>
                              ))}
                            </div>
                          </div>
                        ) : null}
                        <div className="mt-4">
                          <div className="text-[11px] uppercase tracking-wide text-slate-500">Recent Runs</div>
                          {recentRuns.length === 0 ? (
                            <div className="mt-2 text-xs text-slate-400">No recorded runs yet.</div>
                          ) : (
                            <div className="mt-2 space-y-2">
                              {recentRuns.map((run) => {
                                const summary = run.summary ?? {}
                                const totalTokens = typeof summary.total_tokens === 'number'
                                  ? summary.total_tokens
                                  : null
                                const reused = typeof summary.vendors_skipped_hash_reuse === 'number'
                                  ? summary.vendors_skipped_hash_reuse
                                  : null
                                const vendorReasoned = typeof summary.vendors_reasoned === 'number'
                                  ? summary.vendors_reasoned
                                  : 0
                                const crossVendorReasoned = typeof summary.cross_vendor_succeeded === 'number'
                                  ? summary.cross_vendor_succeeded
                                  : 0
                                const reasoned = vendorReasoned + crossVendorReasoned
                                const vendorFailed = typeof summary.vendors_failed === 'number'
                                  ? summary.vendors_failed
                                  : 0
                                const crossVendorFailed = typeof summary.cross_vendor_failed === 'number'
                                  ? summary.cross_vendor_failed
                                  : 0
                                const failed = vendorFailed + crossVendorFailed
                                const skipReason = typeof summary._skip_synthesis === 'string'
                                  ? summary._skip_synthesis
                                  : null
                                return (
                                  <div key={run.id} className="rounded-md border border-slate-700/50 bg-slate-950/50 p-2">
                                    <div className="flex items-center justify-between gap-2">
                                      <div className="text-xs text-slate-300">
                                        {formatTs(run.started_at)}
                                      </div>
                                      <span
                                        className={clsx(
                                          'rounded-full px-2 py-0.5 text-[11px] font-medium',
                                          run.status === 'succeeded' && 'bg-emerald-500/10 text-emerald-300',
                                          run.status === 'partial' && 'bg-amber-500/10 text-amber-300',
                                          run.status === 'failed' && 'bg-rose-500/10 text-rose-300',
                                          run.status === 'running' && 'bg-cyan-500/10 text-cyan-300',
                                        )}
                                      >
                                        {run.status}
                                      </span>
                                    </div>
                                      <div className="mt-2 flex flex-wrap gap-3 text-[11px] text-slate-400">
                                        <span>{formatWholeNumber(reasoned)} reasoned</span>
                                        <span>{formatWholeNumber(reused)} reused</span>
                                        <span>{formatWholeNumber(failed)} failed</span>
                                        <span>{formatWholeNumber(totalTokens)} tokens</span>
                                        {summary.changed_vendors_only === true ? <span>changed only</span> : null}
                                        {summary.force === true ? <span>vendor forced</span> : null}
                                        {summary.force_cross_vendor === true ? <span>cross-vendor forced</span> : null}
                                      </div>
                                      {skipReason ? (
                                        <div className="mt-2 text-[11px] text-slate-500">{skipReason}</div>
                                      ) : null}
                                    </div>
                                  )
                              })}
                            </div>
                          )}
                        </div>
                        <label className="mt-3 flex items-center gap-2 rounded-md border border-slate-700/50 bg-slate-950/50 px-3 py-2 text-xs text-slate-300">
                          <input
                            type="checkbox"
                            checked={changedOnly}
                            onChange={(event) => {
                              setCompetitiveSetChangedOnly((current) => ({
                                ...current,
                                [item.id]: event.target.checked,
                              }))
                            }}
                          />
                          Run changed vendors only
                        </label>
                        <label className="mt-2 flex items-center gap-2 rounded-md border border-slate-700/50 bg-slate-950/50 px-3 py-2 text-xs text-slate-300">
                          <input
                            type="checkbox"
                            checked={forceRun}
                            onChange={(event) => {
                              setCompetitiveSetForceRun((current) => ({
                                ...current,
                                [item.id]: event.target.checked,
                              }))
                            }}
                          />
                          Force vendor rerun
                        </label>
                        <label className="mt-2 flex items-center gap-2 rounded-md border border-slate-700/50 bg-slate-950/50 px-3 py-2 text-xs text-slate-300">
                          <input
                            type="checkbox"
                            checked={forceCrossVendor}
                            onChange={(event) => {
                              setCompetitiveSetForceCrossVendor((current) => ({
                                ...current,
                                [item.id]: event.target.checked,
                              }))
                            }}
                          />
                          Force cross-vendor synthesis
                        </label>
                        <div className="mt-3 flex flex-wrap gap-2">
                          <button
                            onClick={() => handleRunCompetitiveSet(item)}
                            disabled={runningCompetitiveSetId === item.id}
                            className="rounded-md bg-cyan-500/10 px-2.5 py-1 text-xs font-medium text-cyan-300 hover:bg-cyan-500/20 disabled:opacity-50"
                          >
                            {runningCompetitiveSetId === item.id ? 'Starting run...' : 'Run now'}
                          </button>
                          <button
                            onClick={() => handlePreviewCompetitiveSet(item)}
                            disabled={previewingCompetitiveSetId === item.id}
                            className="rounded-md bg-slate-800 px-2.5 py-1 text-xs font-medium text-slate-300 hover:bg-slate-700 disabled:opacity-50"
                          >
                            Refresh preview
                          </button>
                        </div>
                      </div>
                    ) : null}
                  </div>
                )
              })}
            </div>
          </div>

          <div className="rounded-xl border border-slate-700/50 bg-slate-900/50 p-4">
            <h2 className="text-sm font-medium text-white">Phase 1 Boundary</h2>
            <p className="mt-2 text-sm text-slate-400">
              This slice now includes tenant-wide persisted accounts-in-motion aggregation, inline account evidence drilldown, saved monitoring views, and confidence-tiered suppression for weaker rows.
            </p>
          </div>
        </div>
      </div>
      {pendingDestructiveAction && pendingDestructiveActionConfig ? (
        <DestructiveActionModal
          title={pendingDestructiveActionConfig.title}
          message={pendingDestructiveActionConfig.message}
          confirmLabel={pendingDestructiveActionConfig.confirmLabel}
          confirmingLabel={pendingDestructiveActionConfig.confirmingLabel}
          confirming={pendingDestructiveActionBusy}
          error={pendingDestructiveError}
          onCancel={closeDestructiveAction}
          onConfirm={() => {
            void handleConfirmDestructiveAction()
          }}
        />
      ) : null}

      <AccountMovementDrawer
        item={selectedAccount}
        open={selectedAccount != null}
        onClose={handleCloseSelectedAccount}
        onViewVendor={(vendorName) => navigate(watchlistVendorPath(selectedAccountSearchParams, vendorName))}
        alertsApiUrl={selectedAccount ? watchlistVendorAlertsPath(selectedAccountSearchParams, selectedAccount.vendor, selectedAccount.company) : null}
        onCopyAlertsLink={() => void handleCopySelectedAlertsLink()}
        onCopyVendorLink={() => void handleCopySelectedVendorLink()}
        onCopyWitnessLink={(witnessId) => void handleCopySelectedWitnessLink(witnessId)}
        onCopyLink={() => void handleCopySelectedAccountLink()}
        onCopyEvidenceLink={() => void handleCopySelectedEvidenceLink()}
        evidenceExplorerUrl={selectedAccount
          ? watchlistEvidenceExplorerPath(selectedAccountSearchParams, selectedAccount.vendor, null, selectedSourceFilter)
          : null}
        onOpenWitness={handleOpenWitness}
        onGenerateCampaign={handleGenerateCampaign}
        onViewReport={(item) => navigate(watchlistReportsPath(selectedAccountSearchParams, item.vendor))}
        onCopyReportLink={() => void handleCopySelectedReportsLink()}
        onViewOpportunity={(item) => navigate(watchlistOpportunitiesPath(selectedAccountSearchParams, item.vendor))}
        onCopyOpportunityLink={() => void handleCopySelectedOpportunitiesLink()}
        onViewReview={(reviewId) => selectedAccount
          ? navigate(watchlistReviewDetailPath(selectedAccountSearchParams, selectedAccount, reviewId))
          : null}
        onCopyReviewLink={(reviewId) => void handleCopySelectedReviewLink(reviewId)}
        generating={selectedAccount ? generatingCampaignFor === `${selectedAccount.company}::${selectedAccount.vendor}` : false}
      />

      <EvidenceDrawer
        vendorName={evidenceDrawerVendor}
        witnessId={evidenceDrawerWitnessId}
        open={evidenceDrawerOpen}
        onClose={handleCloseWitnessDrawer}
        backToPath={watchlistPath(searchParams)}
        explorerUrl={
          evidenceDrawerWitnessId && evidenceDrawerVendor
            ? watchlistEvidenceExplorerPath(searchParams, evidenceDrawerVendor, evidenceDrawerWitnessId, selectedSourceFilter)
            : null
        }
      />
    </div>
  )
}
