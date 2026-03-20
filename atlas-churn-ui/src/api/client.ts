import { tryRefreshToken } from '../auth/AuthContext'
import type {
  ChurnSignal,
  ChurnSignalDetail,
  HighIntentCompany,
  VendorProfile,
  VendorHistoryResponse,
  VendorPeriodComparisonResponse,
  Report,
  ReportDetail,
  ReviewSummary,
  ReviewDetail,
  PipelineStatus,
  AffiliateOpportunity,
  AffiliatePartner,
  ClickSummary,
  Campaign,
  CampaignStats,
  VendorTarget,
  BlogDraftSummary,
  BlogDraft,
  BlogEvidence,
  Prospect,
  ProspectStats,
  ReviewQueueDraft,
  AuditEvent,
  BriefingDraft,
} from '../types'
import { normalizeReportDetail, normalizeVendorProfile } from '../lib/reportNormalization'

const API_BASE = import.meta.env.VITE_API_BASE || ''
const BASE = `${API_BASE}/api/v1/b2b/dashboard`
const CAMPAIGNS_BASE = `${API_BASE}/api/v1/b2b/campaigns`
const TARGETS_BASE = `${API_BASE}/api/v1/b2b/vendor-targets`
const BLOG_ADMIN_BASE = `${API_BASE}/api/v1/admin/blog`
const PROSPECTS_BASE = `${API_BASE}/api/v1/b2b/prospects`
const BRIEFINGS_BASE = `${API_BASE}/api/v1/b2b/briefings`

function authHeaders(): Record<string, string> {
  const token = localStorage.getItem('atlas_token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

function forceLogout() {
  localStorage.removeItem('atlas_token')
  localStorage.removeItem('atlas_refresh_token')
  window.location.href = '/landing'
}

async function handleResponse<T>(res: Response, retryFetch: () => Promise<Response>): Promise<T> {
  if (res.status === 401) {
    const newToken = await tryRefreshToken()
    if (!newToken) {
      forceLogout()
      throw new Error('Session expired')
    }
    const retry = await retryFetch()
    if (retry.status === 401) {
      forceLogout()
      throw new Error('Session expired')
    }
    if (!retry.ok) {
      const body = await retry.text().catch(() => '')
      throw new Error(`API ${retry.status}: ${body || retry.statusText}`)
    }
    return retry.json()
  }
  if (res.status === 402) {
    window.location.href = '/account'
    throw new Error('Payment required')
  }
  if (res.status === 403) {
    const body = await res.json().catch(() => ({ detail: '' }))
    if (body.detail?.includes('Trial expired')) {
      window.location.href = '/account'
    }
    throw new Error(body.detail || 'Forbidden')
  }
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`API ${res.status}: ${body}`)
  }
  return res.json()
}

// ---------------------------------------------------------------------------
// Generic fetchers
// ---------------------------------------------------------------------------

async function get<T>(base: string, path: string, params?: Record<string, string | number | boolean | undefined>): Promise<T> {
  const url = new URL(base + path, window.location.origin)
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null && v !== '') {
        url.searchParams.set(k, String(v))
      }
    }
  }
  const doFetch = () => fetch(url.toString(), { headers: authHeaders() })
  const res = await doFetch()
  return handleResponse<T>(res, doFetch)
}

async function post<T>(base: string, path: string, body?: unknown): Promise<T> {
  const url = base + path
  const doFetch = () => fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  const res = await doFetch()
  return handleResponse<T>(res, doFetch)
}

async function patch<T>(base: string, path: string, body: unknown): Promise<T> {
  const url = base + path
  const doFetch = () => fetch(url, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(body),
  })
  const res = await doFetch()
  return handleResponse<T>(res, doFetch)
}

async function put<T>(base: string, path: string, body: unknown): Promise<T> {
  const url = base + path
  const doFetch = () => fetch(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: JSON.stringify(body),
  })
  const res = await doFetch()
  return handleResponse<T>(res, doFetch)
}

async function del<T>(base: string, path: string): Promise<T> {
  const url = base + path
  const doFetch = () => fetch(url, { method: 'DELETE', headers: authHeaders() })
  const res = await doFetch()
  return handleResponse<T>(res, doFetch)
}

// ---------------------------------------------------------------------------
// Signals
// ---------------------------------------------------------------------------

export async function fetchSignals(params?: {
  vendor_name?: string
  min_urgency?: number
  category?: string
  limit?: number
}) {
  return get<{
    signals: ChurnSignal[]
    count: number
    total_vendors?: number
    high_urgency_count?: number
    total_signal_reviews?: number
  }>(BASE, '/signals', params)
}

export async function fetchSlowBurnWatchlist(params?: {
  vendor_name?: string
  category?: string
  limit?: number
}) {
  return get<{ signals: ChurnSignal[]; count: number }>(BASE, '/slow-burn-watchlist', params)
}

export async function fetchSignal(vendorName: string, productCategory?: string) {
  return get<ChurnSignalDetail>(BASE, `/signals/${encodeURIComponent(vendorName)}`, {
    product_category: productCategory,
  })
}

export async function fetchHighIntent(params?: {
  vendor_name?: string
  min_urgency?: number
  window_days?: number
  limit?: number
}) {
  return get<{ companies: HighIntentCompany[]; count: number }>(BASE, '/high-intent', params)
}

export async function fetchVendorProfile(vendorName: string) {
  const profile = await get<VendorProfile>(BASE, `/vendors/${encodeURIComponent(vendorName)}`)
  return normalizeVendorProfile(profile)
}

export async function fetchVendorHistory(vendorName: string, params?: {
  days?: number
  limit?: number
}) {
  return get<VendorHistoryResponse>(BASE, '/vendor-history', {
    vendor_name: vendorName,
    ...params,
  })
}

export async function compareVendorPeriods(vendorName: string, params?: {
  period_a_days_ago?: number
  period_b_days_ago?: number
}) {
  return get<VendorPeriodComparisonResponse>(BASE, '/compare-vendor-periods', {
    vendor_name: vendorName,
    ...params,
  })
}

export async function fetchReports(params?: {
  report_type?: string
  vendor_filter?: string
  limit?: number
}) {
  return get<{ reports: Report[]; count: number }>(BASE, '/reports', params)
}

export async function generateVendorComparisonReport(body: {
  primary_vendor: string
  comparison_vendor: string
  window_days?: number
  persist?: boolean
}) {
  return post<Record<string, unknown>>(BASE, '/reports/compare', body)
}

export async function generateAccountComparisonReport(body: {
  primary_company: string
  comparison_company: string
  window_days?: number
  persist?: boolean
}) {
  return post<Record<string, unknown>>(BASE, '/reports/compare-companies', body)
}

export async function generateAccountDeepDiveReport(body: {
  company_name: string
  window_days?: number
  persist?: boolean
}) {
  return post<Record<string, unknown>>(BASE, '/reports/company-deep-dive', body)
}

export async function fetchReport(reportId: string) {
  const report = await get<ReportDetail>(BASE, `/reports/${reportId}`)
  return normalizeReportDetail(report)
}

export async function fetchReviews(params?: {
  vendor_name?: string
  pain_category?: string
  min_urgency?: number
  company?: string
  has_churn_intent?: boolean
  window_days?: number
  limit?: number
}) {
  return get<{ reviews: ReviewSummary[]; count: number }>(BASE, '/reviews', params)
}

export async function fetchReview(reviewId: string) {
  return get<ReviewDetail>(BASE, `/reviews/${reviewId}`)
}

export async function fetchPipeline() {
  return get<PipelineStatus>(BASE, '/pipeline')
}

// ---------------------------------------------------------------------------
// Affiliates
// ---------------------------------------------------------------------------

export async function fetchAffiliateOpportunities(params?: {
  min_urgency?: number
  min_score?: number
  window_days?: number
  limit?: number
  vendor_name?: string
  dm_only?: boolean
}) {
  return get<{ opportunities: AffiliateOpportunity[]; count: number }>(BASE, '/affiliates/opportunities', params)
}

export async function fetchAffiliatePartners() {
  return get<{ partners: AffiliatePartner[]; count: number }>(BASE, '/affiliates/partners')
}

export async function fetchClickSummary() {
  return get<{ clicks: ClickSummary[] }>(BASE, '/affiliates/clicks/summary')
}

export async function createAffiliatePartner(body: Omit<AffiliatePartner, 'id' | 'created_at' | 'updated_at'>) {
  return post<AffiliatePartner>(BASE, '/affiliates/partners', body)
}

export async function updateAffiliatePartner(id: string, body: Partial<AffiliatePartner>) {
  return patch<AffiliatePartner>(BASE, `/affiliates/partners/${id}`, body)
}

export async function deleteAffiliatePartner(id: string) {
  return del<{ status: string }>(BASE, `/affiliates/partners/${id}`)
}

export async function recordAffiliateClick(partnerId: string, reviewId?: string) {
  return post<{ status: string }>(BASE, '/affiliates/clicks', { partner_id: partnerId, review_id: reviewId, referrer: 'dashboard' })
}

// ---------------------------------------------------------------------------
// Campaigns
// ---------------------------------------------------------------------------

export async function fetchCampaigns(params?: {
  status?: string
  company?: string
  vendor?: string
  channel?: string
  limit?: number
}) {
  return get<{ campaigns: Campaign[]; count: number }>(CAMPAIGNS_BASE, '', params)
}

export async function fetchCampaign(id: string) {
  return get<Campaign>(CAMPAIGNS_BASE, `/${id}`)
}

export async function fetchCampaignStats() {
  return get<CampaignStats>(CAMPAIGNS_BASE, '/stats')
}

export async function generateCampaigns(body?: {
  vendor_name?: string
  company_name?: string
  min_score?: number
  limit?: number
  target_mode?: string
}) {
  return post<{ generated: number; companies?: number }>(CAMPAIGNS_BASE, '/generate', body ?? {})
}

export async function approveCampaign(id: string) {
  return post<{ status: string }>(CAMPAIGNS_BASE, `/${id}/approve`)
}

export async function updateCampaign(id: string, body: Partial<Pick<Campaign, 'subject' | 'body' | 'cta' | 'status'>>) {
  return patch<Campaign>(CAMPAIGNS_BASE, `/${id}`, body)
}

// ---------------------------------------------------------------------------
// Vendor Targets
// ---------------------------------------------------------------------------

export async function fetchVendorTargets(params?: {
  target_mode?: string
  status?: string
  search?: string
  limit?: number
}) {
  return get<{ targets: VendorTarget[]; count: number }>(TARGETS_BASE, '', params)
}

export async function fetchVendorTarget(id: string) {
  return get<VendorTarget>(TARGETS_BASE, `/${id}`)
}

export async function createVendorTarget(body: Partial<VendorTarget>) {
  return post<VendorTarget>(TARGETS_BASE, '', body)
}

export async function updateVendorTarget(id: string, body: Partial<VendorTarget>) {
  return put<VendorTarget>(TARGETS_BASE, `/${id}`, body)
}

export async function claimVendorTarget(id: string) {
  return post<VendorTarget & { already_claimed?: boolean }>(TARGETS_BASE, `/${id}/claim`)
}

export async function deleteVendorTarget(id: string) {
  return del<{ status: string }>(TARGETS_BASE, `/${id}`)
}

// ---------------------------------------------------------------------------
// CSV Export
// ---------------------------------------------------------------------------

export function downloadCsv(
  path: string,
  params?: Record<string, string | number | boolean | undefined>,
) {
  const url = new URL(BASE + path, window.location.origin)
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null && v !== '') {
        url.searchParams.set(k, String(v))
      }
    }
  }
  const token = localStorage.getItem('atlas_token')
  if (token) url.searchParams.set('token', token)
  window.open(url.toString(), '_blank')
}

export async function generateVendorReport(id: string) {
  return post<{ status: string; signal_count?: number; high_urgency_count?: number }>(TARGETS_BASE, `/${id}/generate-report`)
}

// ---------------------------------------------------------------------------
// Blog Admin (drafts, evidence, publish)
// ---------------------------------------------------------------------------

export async function fetchBlogDrafts(status?: string) {
  return get<BlogDraftSummary[]>(BLOG_ADMIN_BASE, '/drafts', { status })
}

export async function fetchBlogDraft(id: string) {
  return get<BlogDraft>(BLOG_ADMIN_BASE, `/drafts/${id}`)
}

export async function fetchBlogEvidence(id: string) {
  return get<{ reviews: BlogEvidence[]; count: number }>(BLOG_ADMIN_BASE, `/drafts/${id}/evidence`)
}

export async function publishBlogDraft(id: string) {
  return post<{ ok: boolean; id: string; slug: string; published_at: string }>(BLOG_ADMIN_BASE, `/drafts/${id}/publish`)
}

export async function updateBlogDraft(id: string, body: Partial<Pick<BlogDraft, 'title' | 'content' | 'status' | 'reviewer_notes'>>) {
  return patch<{ ok: boolean; id: string }>(BLOG_ADMIN_BASE, `/drafts/${id}`, body)
}

// ---------------------------------------------------------------------------
// Prospects
// ---------------------------------------------------------------------------

export async function fetchProspects(params?: {
  company?: string
  status?: string
  seniority?: string
  limit?: number
  offset?: number
}) {
  return get<{ prospects: Prospect[]; count: number }>(PROSPECTS_BASE, '', params)
}

export async function fetchProspectStats() {
  return get<ProspectStats>(PROSPECTS_BASE, '/stats')
}

// ---------------------------------------------------------------------------
// Campaign Review Queue (enhanced)
// ---------------------------------------------------------------------------

export async function fetchReviewQueue(params?: {
  status?: string
  include_prospects?: boolean
  limit?: number
  offset?: number
}) {
  return get<{ drafts: ReviewQueueDraft[]; count: number }>(CAMPAIGNS_BASE, '/review-queue', params)
}

export async function fetchReviewQueueSummary() {
  return get<{
    pending_review: number
    pending_recipient: number
    ready_to_send: number
    suppressed: number
    oldest_draft_age_hours: number | null
    by_partner: { partner_name: string; count: number }[]
  }>(CAMPAIGNS_BASE, '/review-queue/summary')
}

export async function fetchCampaignAuditLog(campaignId: string) {
  return get<{ count: number; audit_log: AuditEvent[] }>(CAMPAIGNS_BASE, `/${campaignId}/audit-log`)
}

export async function bulkApproveCampaigns(ids: string[], action: 'approve' | 'queue-send' | 'reject') {
  return post<{ processed: number; failed: { id: string; reason: string }[] }>(CAMPAIGNS_BASE, '/bulk-approve', { campaign_ids: ids, action })
}

export async function bulkRejectCampaigns(ids: string[], reason?: string) {
  return post<{ rejected: number; failed: { id: string; reason: string }[] }>(CAMPAIGNS_BASE, '/bulk-reject', { campaign_ids: ids, reason })
}

// ---------------------------------------------------------------------------
// Tenant Vendor Tracking (onboarding + scoped data)
// ---------------------------------------------------------------------------

const TENANT_BASE = `${API_BASE}/api/v1/b2b/tenant`

export interface TrackedVendor {
  id: string
  vendor_name: string
  track_mode: string
  label: string | null
  added_at: string | null
  avg_urgency: number | null
  churn_intent_count: number | null
  total_reviews: number | null
  nps_proxy: number | null
}

export interface VendorSearchResult {
  vendor_name: string
  product_category: string | null
  total_reviews: number | null
  avg_urgency: number | null
}

export async function searchAvailableVendors(q: string) {
  return get<{ vendors: VendorSearchResult[]; count: number }>(TENANT_BASE, '/vendors/search', { q })
}

export async function addTrackedVendor(vendor_name: string, track_mode: string = 'own', label: string = '') {
  return post<TrackedVendor>(TENANT_BASE, '/vendors', { vendor_name, track_mode, label })
}

export async function removeTrackedVendor(vendor_name: string) {
  return del<{ status: string }>(TENANT_BASE, `/vendors/${encodeURIComponent(vendor_name)}`)
}

export async function listTrackedVendors() {
  return get<{ vendors: TrackedVendor[]; count: number }>(TENANT_BASE, '/vendors')
}

// ---------------------------------------------------------------------------
// Briefing Review Queue (HITL)
// ---------------------------------------------------------------------------

export async function fetchBriefingReviewQueue(params?: {
  status?: string
  limit?: number
  offset?: number
}) {
  return get<{ briefings: BriefingDraft[]; count: number }>(BRIEFINGS_BASE, '/review-queue', params)
}

export async function fetchBriefingReviewSummary() {
  return get<{
    pending_approval: number
    sent: number
    rejected: number
    failed: number
    oldest_pending_hours: number | null
  }>(BRIEFINGS_BASE, '/review-queue/summary')
}

export async function bulkApproveBriefings(ids: string[]) {
  return post<{ processed: number; failed: { id: string; reason: string }[] }>(BRIEFINGS_BASE, '/bulk-approve', { briefing_ids: ids, action: 'approve' })
}

export async function bulkRejectBriefings(ids: string[], reason?: string) {
  return post<{ rejected: number; failed: { id: string; reason: string }[] }>(BRIEFINGS_BASE, '/bulk-reject', { briefing_ids: ids, reason })
}
