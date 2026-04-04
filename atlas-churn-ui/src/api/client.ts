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
  CampaignQualityDiagnostics,
  CampaignStats,
  CampaignQualityTrends,
  VendorTarget,
  BlogDraftSummary,
  BlogDraftSummaryRollup,
  BlogDraft,
  BlogEvidence,
  BlogQualityDiagnostics,
  BlogQualityTrends,
  Prospect,
  ProspectStats,
  ReviewQueueDraft,
  AuditEvent,
  BriefingDraft,
  VisibilityQueueItem,
  VisibilityEvent,
  ArtifactAttempt,
  EnrichmentQuarantine,
  ExtractionHealthAudit,
  SynthesisValidationResult,
  AdminCostSummary,
  AdminCostOperation,
  AdminCostVendor,
  AdminCostB2bEfficiency,
  AdminCostBurnDashboard,
  AdminCostGenericReasoning,
  AdminCostReconciliation,
  AdminCostRecentCall,
  AdminCostCacheHealth,
  AdminCostReasoningActivity,
  AdminCostRunDetail,
  AdminTaskHealthRow,
  DedupDecision,
  PipelineReviewAction,
} from '../types'
import { normalizeReportDetail, normalizeVendorProfile } from '../lib/reportNormalization'

const API_BASE = import.meta.env.VITE_API_BASE || ''
const TENANT_BASE = `${API_BASE}/api/v1/b2b/tenant`
const AFFILIATES_BASE = `${TENANT_BASE}/affiliates`
const CAMPAIGNS_BASE = `${API_BASE}/api/v1/b2b/campaigns`
const TARGETS_BASE = `${API_BASE}/api/v1/b2b/vendor-targets`
const PREDICT_BASE = `${API_BASE}/api/v1/b2b/predict`
const BLOG_ADMIN_BASE = `${API_BASE}/api/v1/admin/blog`
const PROSPECTS_BASE = `${API_BASE}/api/v1/b2b/prospects`
const BRIEFINGS_BASE = `${API_BASE}/api/v1/b2b/briefings`
const AUTONOMOUS_BASE = `${API_BASE}/api/v1/autonomous`
const CACHE_BUSTER_PARAM = '_ts'

function maybeFallbackApiPath(url: string): string | null {
  if (!url.includes('/api/v1/')) return null
  return url.replace('/api/v1/', '/api/')
}

async function fetchWithApiFallback(url: string, init?: RequestInit): Promise<Response> {
  const res = await fetch(url, init)
  if (res.status !== 404) return res
  const fallbackUrl = maybeFallbackApiPath(url)
  if (!fallbackUrl) return res
  return fetch(fallbackUrl, init)
}

function freshHeaders(): Record<string, string> {
  return {
    'Cache-Control': 'no-cache',
    Pragma: 'no-cache',
  }
}

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
  url.searchParams.set(CACHE_BUSTER_PARAM, String(Date.now()))
  const doFetch = () => fetchWithApiFallback(url.toString(), {
    headers: { ...freshHeaders(), ...authHeaders() },
    cache: 'no-store',
  })
  const res = await doFetch()
  return handleResponse<T>(res, doFetch)
}

async function post<T>(base: string, path: string, body?: unknown): Promise<T> {
  const url = base + path
  const doFetch = () => fetchWithApiFallback(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...freshHeaders(), ...authHeaders() },
    cache: 'no-store',
    body: body !== undefined ? JSON.stringify(body) : undefined,
  })
  const res = await doFetch()
  return handleResponse<T>(res, doFetch)
}

async function patch<T>(base: string, path: string, body: unknown): Promise<T> {
  const url = base + path
  const doFetch = () => fetchWithApiFallback(url, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json', ...freshHeaders(), ...authHeaders() },
    cache: 'no-store',
    body: JSON.stringify(body),
  })
  const res = await doFetch()
  return handleResponse<T>(res, doFetch)
}

async function put<T>(base: string, path: string, body: unknown): Promise<T> {
  const url = base + path
  const doFetch = () => fetchWithApiFallback(url, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json', ...freshHeaders(), ...authHeaders() },
    cache: 'no-store',
    body: JSON.stringify(body),
  })
  const res = await doFetch()
  return handleResponse<T>(res, doFetch)
}

async function del<T>(base: string, path: string): Promise<T> {
  const url = base + path
  const doFetch = () => fetchWithApiFallback(url, {
    method: 'DELETE',
    headers: { ...freshHeaders(), ...authHeaders() },
    cache: 'no-store',
  })
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
  }>(TENANT_BASE, '/signals', params)
}

export async function fetchSlowBurnWatchlist(params?: {
  vendor_name?: string
  category?: string
  limit?: number
}) {
  return get<{ signals: ChurnSignal[]; count: number }>(TENANT_BASE, '/slow-burn-watchlist', params)
}

export async function fetchSignal(vendorName: string, productCategory?: string) {
  return get<ChurnSignalDetail>(TENANT_BASE, `/signals/${encodeURIComponent(vendorName)}`, {
    product_category: productCategory,
  })
}

export async function fetchHighIntent(params?: {
  vendor_name?: string
  min_urgency?: number
  window_days?: number
  limit?: number
}) {
  return get<{ companies: HighIntentCompany[]; count: number }>(TENANT_BASE, '/high-intent', params)
}

export async function fetchVendorProfile(vendorName: string) {
  const profile = await get<VendorProfile>(TENANT_BASE, `/signals/${encodeURIComponent(vendorName)}`)
  return normalizeVendorProfile(profile)
}

export async function fetchVendorHistory(vendorName: string, params?: {
  days?: number
  limit?: number
}) {
  return get<VendorHistoryResponse>(TENANT_BASE, '/vendor-history', {
    vendor_name: vendorName,
    ...params,
  })
}

export async function compareVendorPeriods(vendorName: string, params?: {
  period_a_days_ago?: number
  period_b_days_ago?: number
}) {
  return get<VendorPeriodComparisonResponse>(TENANT_BASE, '/compare-vendor-periods', {
    vendor_name: vendorName,
    ...params,
  })
}

export async function fetchReports(params?: {
  report_type?: string
  vendor_filter?: string
  limit?: number
}) {
  return get<{ reports: Report[]; count: number }>(TENANT_BASE, '/reports', params)
}

export async function generateVendorComparisonReport(body: {
  primary_vendor: string
  comparison_vendor: string
  window_days?: number
  persist?: boolean
}) {
  return post<Record<string, unknown>>(TENANT_BASE, '/reports/compare', body)
}

export async function generateAccountComparisonReport(body: {
  primary_company: string
  comparison_company: string
  window_days?: number
  persist?: boolean
}) {
  return post<Record<string, unknown>>(TENANT_BASE, '/reports/compare-companies', body)
}

export async function generateAccountDeepDiveReport(body: {
  company_name: string
  window_days?: number
  persist?: boolean
}) {
  return post<Record<string, unknown>>(TENANT_BASE, '/reports/company-deep-dive', body)
}

export async function fetchReport(reportId: string) {
  const report = await get<ReportDetail>(TENANT_BASE, `/reports/${reportId}`)
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
  return get<{ reviews: ReviewSummary[]; count: number }>(TENANT_BASE, '/reviews', params)
}

export async function fetchReview(reviewId: string) {
  return get<ReviewDetail>(TENANT_BASE, `/reviews/${reviewId}`)
}

export async function fetchPipeline() {
  return get<PipelineStatus>(TENANT_BASE, '/pipeline')
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
  return get<{ opportunities: AffiliateOpportunity[]; count: number }>(AFFILIATES_BASE, '/opportunities', params)
}

export async function fetchAffiliatePartners() {
  return get<{ partners: AffiliatePartner[]; count: number }>(AFFILIATES_BASE, '/partners')
}

export async function fetchClickSummary() {
  return get<{ clicks: ClickSummary[] }>(AFFILIATES_BASE, '/clicks/summary')
}

export async function createAffiliatePartner(body: Omit<AffiliatePartner, 'id' | 'created_at' | 'updated_at'>) {
  return post<AffiliatePartner>(AFFILIATES_BASE, '/partners', body)
}

export async function updateAffiliatePartner(id: string, body: Partial<AffiliatePartner>) {
  return patch<AffiliatePartner>(AFFILIATES_BASE, `/partners/${id}`, body)
}

export async function deleteAffiliatePartner(id: string) {
  return del<{ status: string }>(AFFILIATES_BASE, `/partners/${id}`)
}

export async function recordAffiliateClick(partnerId: string, reviewId?: string) {
  return post<{ status: string }>(AFFILIATES_BASE, '/clicks', { partner_id: partnerId, review_id: reviewId, referrer: 'dashboard' })
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

export async function fetchCampaignQualityTrends(params?: {
  days?: number
  top_n?: number
}) {
  return get<CampaignQualityTrends>(CAMPAIGNS_BASE, '/quality-trends', params)
}

export async function fetchCampaignQualityDiagnostics(params?: {
  days?: number
  top_n?: number
}) {
  return get<CampaignQualityDiagnostics>(CAMPAIGNS_BASE, '/quality-diagnostics', params)
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
  const url = new URL(TENANT_BASE + path, window.location.origin)
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

export async function fetchBlogDraftSummary() {
  return get<BlogDraftSummaryRollup>(BLOG_ADMIN_BASE, '/drafts/summary')
}

export async function fetchBlogQualityTrends(params?: {
  days?: number
  top_n?: number
}) {
  return get<BlogQualityTrends>(BLOG_ADMIN_BASE, '/quality-trends', params)
}

export async function fetchBlogQualityDiagnostics(params?: {
  days?: number
  top_n?: number
}) {
  return get<BlogQualityDiagnostics>(BLOG_ADMIN_BASE, '/quality-diagnostics', params)
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
    quality_pass: number
    quality_fail: number
    quality_missing: number
    blocker_total: number
    by_boundary: { boundary: string; count: number }[]
    top_blockers: { reason: string; count: number }[]
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

export interface CompetitiveSet {
  id: string
  account_id: string
  name: string
  focal_vendor_name: string
  competitor_vendor_names: string[]
  active: boolean
  refresh_mode: 'manual' | 'scheduled'
  refresh_interval_hours: number | null
  vendor_synthesis_enabled: boolean
  pairwise_enabled: boolean
  category_council_enabled: boolean
  asymmetry_enabled: boolean
  last_run_at: string | null
  last_success_at: string | null
  last_run_status: 'running' | 'succeeded' | 'partial' | 'failed' | null
  last_run_summary: Record<string, unknown>
  created_at: string
  updated_at: string
}

export interface CompetitiveSetPlan {
  competitive_set_id: string
  focal_vendor_name: string
  vendor_names: string[]
  pairwise_pairs: string[][]
  category_names: string[]
  asymmetry_pairs: string[][]
  vendor_synthesis_enabled: boolean
  pairwise_enabled: boolean
  category_council_enabled: boolean
  asymmetry_enabled: boolean
  vendor_job_count: number
  pairwise_job_count: number
  category_job_count: number
  asymmetry_job_count: number
  estimated_total_jobs: number
  category_by_vendor?: Record<string, string | null>
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

export async function listCompetitiveSets(include_inactive: boolean = false) {
  return get<{ competitive_sets: CompetitiveSet[]; count: number }>(
    TENANT_BASE,
    '/competitive-sets',
    { include_inactive },
  )
}

export async function createCompetitiveSet(body: {
  name: string
  focal_vendor_name: string
  competitor_vendor_names: string[]
  active?: boolean
  refresh_mode?: 'manual' | 'scheduled'
  refresh_interval_hours?: number | null
  vendor_synthesis_enabled?: boolean
  pairwise_enabled?: boolean
  category_council_enabled?: boolean
  asymmetry_enabled?: boolean
}) {
  return post<CompetitiveSet>(TENANT_BASE, '/competitive-sets', body)
}

export async function updateCompetitiveSet(
  competitiveSetId: string,
  body: Record<string, unknown>,
) {
  return put<CompetitiveSet>(TENANT_BASE, `/competitive-sets/${encodeURIComponent(competitiveSetId)}`, body)
}

export async function deleteCompetitiveSet(competitiveSetId: string) {
  return del<{ deleted: boolean; competitive_set_id: string }>(
    TENANT_BASE,
    `/competitive-sets/${encodeURIComponent(competitiveSetId)}`,
  )
}

export async function fetchCompetitiveSetPlan(competitiveSetId: string) {
  return get<{ competitive_set: CompetitiveSet; plan: CompetitiveSetPlan }>(
    TENANT_BASE,
    `/competitive-sets/${encodeURIComponent(competitiveSetId)}/plan`,
  )
}

export async function runCompetitiveSetNow(
  competitiveSetId: string,
  body?: { force?: boolean; force_cross_vendor?: boolean },
) {
  return post<{
    execution_id: string | null
    status: string
    message: string
    competitive_set_id: string
    plan: CompetitiveSetPlan
  }>(TENANT_BASE, `/competitive-sets/${encodeURIComponent(competitiveSetId)}/run`, body)
}

export interface AccountsInMotionFeedItem {
  company: string | null
  vendor: string
  watch_vendor: string
  track_mode: string
  watchlist_label: string | null
  category: string | null
  urgency: number
  role_type: string | null
  buying_stage: string | null
  budget_authority: boolean | null
  pain_categories: { category: string; severity: string }[]
  evidence: string[]
  alternatives_considering: { name: string; reason?: string }[]
  contract_signal: string | null
  reviewer_title: string | null
  company_size_raw: string | null
  quality_flags: string[]
  opportunity_score: number | null
  quote_match_type: string | null
  confidence: number | null
  source_distribution: Record<string, number>
  enriched_at: string | null
  employee_count: number | null
  industry: string | null
  annual_revenue: string | null
  domain: string | null
  contacts: Array<Record<string, unknown>>
  contact_count: number
  report_date: string | null
  stale_days: number | null
  is_stale: boolean
  data_source: string | null
}

export async function fetchAccountsInMotionFeed(params?: {
  min_urgency?: number
  per_vendor_limit?: number
  limit?: number
}) {
  return get<{
    accounts: AccountsInMotionFeedItem[]
    count: number
    tracked_vendor_count: number
    vendors_with_accounts: number
    min_urgency: number
    per_vendor_limit: number
    freshest_report_date: string | null
  }>(TENANT_BASE, '/accounts-in-motion-feed', params)
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

// ---------------------------------------------------------------------------
// Pipeline visibility
// ---------------------------------------------------------------------------
const VISIBILITY_BASE = `${API_BASE}/api/v1/pipeline/visibility`
const ADMIN_COSTS_BASE = `${API_BASE}/api/v1/admin/costs`

export async function fetchVisibilitySummary(hours = 24) {
  return get<{
    period_hours: number
    open_actionable: number
    open_total: number
    failures_period: number
    quarantines_period: number
    rejections_period: number
    recovered_validation_retries_period: number
  }>(VISIBILITY_BASE, '/summary', { hours })
}

export async function fetchVisibilityQueue(params?: {
  limit?: number; offset?: number; stage?: string; severity?: string
}) {
  return get<{ items: VisibilityQueueItem[]; limit: number; offset: number }>(
    VISIBILITY_BASE, '/queue', params as Record<string, string | number>
  )
}

export async function fetchVisibilityEvents(params?: {
  limit?: number; offset?: number; stage?: string; event_type?: string
  severity?: string; entity_type?: string; reason_code?: string; hours?: number
}) {
  return get<{ events: VisibilityEvent[]; limit: number; offset: number }>(
    VISIBILITY_BASE, '/events', params as Record<string, string | number>
  )
}

export async function fetchArtifactAttempts(params?: {
  artifact_type?: string; status?: string; limit?: number; offset?: number; hours?: number
}) {
  return get<{ attempts: ArtifactAttempt[]; limit: number; offset: number }>(
    VISIBILITY_BASE, '/attempts', params as Record<string, string | number>
  )
}

export async function fetchEnrichmentQuarantines(params?: {
  reason_code?: string; vendor_name?: string; unreleased_only?: boolean; limit?: number
}) {
  return get<{ quarantines: EnrichmentQuarantine[]; limit: number; offset: number }>(
    VISIBILITY_BASE, '/quarantines', params as Record<string, string | number | boolean>
  )
}

export async function fetchExtractionHealth(params?: {
  days?: number
  top_n?: number
}) {
  return get<ExtractionHealthAudit>(
    VISIBILITY_BASE,
    '/extraction-health',
    params as Record<string, string | number>,
  )
}

export async function fetchSynthesisValidationResults(params?: {
  vendor_name?: string; rule_code?: string; severity?: string; passed?: boolean
  run_id?: string; retry_only?: boolean; limit?: number; offset?: number
}) {
  return get<{ results: SynthesisValidationResult[]; limit: number; offset: number }>(
    VISIBILITY_BASE, '/synthesis-validation', params as Record<string, string | number | boolean>
  )
}

export async function fetchDedupDecisions(params?: {
  stage?: string; entity_type?: string; reason_code?: string; run_id?: string
  limit?: number; offset?: number
}) {
  return get<{ decisions: DedupDecision[]; limit: number; offset: number }>(
    VISIBILITY_BASE, '/dedup-decisions', params as Record<string, string | number>
  )
}

export async function fetchVisibilityReviewActions(params?: {
  review_id?: string; target_entity_type?: string; target_entity_id?: string
  limit?: number; offset?: number
}) {
  return get<{ actions: PipelineReviewAction[]; limit: number; offset: number }>(
    VISIBILITY_BASE, '/review-actions', params as Record<string, string | number>
  )
}

export async function resolveVisibilityReview(reviewId: string, action: string, note?: string) {
  return post<{ status: string }>(VISIBILITY_BASE, `/reviews/${reviewId}/resolve?action=${action}${note ? '&note=' + encodeURIComponent(note) : ''}`)
}

// ---------------------------------------------------------------------------
// Admin cost analytics
// ---------------------------------------------------------------------------

export async function fetchAdminCostSummary(days = 30) {
  return get<AdminCostSummary>(ADMIN_COSTS_BASE, '/summary', { days })
}

export async function fetchAdminCostByOperation(params?: {
  days?: number
  limit?: number
  provider?: string
  model?: string
  span_name?: string
  operation_type?: string
  status?: string
  cache_only?: boolean
}) {
  return get<{ period_days: number; operations: AdminCostOperation[] }>(
    ADMIN_COSTS_BASE,
    '/by-operation',
    params as Record<string, string | number | boolean>,
  )
}

export async function fetchAdminCostByVendor(params?: {
  days?: number
  limit?: number
}) {
  return get<{ period_days: number; vendors: AdminCostVendor[] }>(
    ADMIN_COSTS_BASE,
    '/by-vendor',
    params as Record<string, string | number | boolean>,
  )
}

export async function fetchAdminCostB2bEfficiency(params?: {
  days?: number
  top_n?: number
  run_limit?: number
}) {
  return get<AdminCostB2bEfficiency>(
    ADMIN_COSTS_BASE,
    '/b2b-efficiency',
    params as Record<string, string | number | boolean>,
  )
}

export async function fetchAdminCostBurnDashboard(params?: {
  days?: number
  top_n?: number
}) {
  return get<AdminCostBurnDashboard>(
    ADMIN_COSTS_BASE,
    '/burn-dashboard',
    params as Record<string, string | number | boolean>,
  )
}

export async function fetchAdminCostGenericReasoning(params?: {
  days?: number
  top_n?: number
}) {
  return get<AdminCostGenericReasoning>(
    ADMIN_COSTS_BASE,
    '/generic-reasoning',
    params as Record<string, string | number | boolean>,
  )
}

export async function fetchAdminCostReconciliation(days = 30) {
  return get<AdminCostReconciliation>(ADMIN_COSTS_BASE, '/reconciliation', { days })
}

export async function fetchAdminCostRecent(params?: {
  limit?: number
  days?: number
  provider?: string
  model?: string
  span_name?: string
  operation_type?: string
  status?: string
  cache_only?: boolean
}) {
  return get<{ calls: AdminCostRecentCall[] }>(
    ADMIN_COSTS_BASE,
    '/recent',
    params as Record<string, string | number | boolean>,
  )
}

export async function fetchAdminCostCacheHealth(days = 30, top_n = 8) {
  return get<AdminCostCacheHealth>(ADMIN_COSTS_BASE, '/cache-health', { days, top_n })
}

export async function fetchAdminCostReasoningActivity(days = 30) {
  return get<AdminCostReasoningActivity>(ADMIN_COSTS_BASE, '/reasoning-activity', { days })
}

export async function fetchAdminTaskHealth(days = 30) {
  return get<{ tasks: AdminTaskHealthRow[] }>(ADMIN_COSTS_BASE, '/task-health', { days })
}

export async function fetchAdminCostRun(
  run_id: string,
  params?: {
    call_limit?: number
    event_limit?: number
    attempt_limit?: number
    batch_item_limit?: number
  },
) {
  return get<AdminCostRunDetail>(
    ADMIN_COSTS_BASE,
    `/runs/${encodeURIComponent(run_id)}`,
    params as Record<string, string | number | boolean>,
  )
}

export async function runAutonomousTask(taskId: string, body?: Record<string, unknown>) {
  return post<{ execution_id: string | null; status: string; message: string; already_running?: boolean }>(
    AUTONOMOUS_BASE,
    `/${encodeURIComponent(taskId)}/run`,
    body,
  )
}

// ── Win/Loss Predictor ───────────────────────────────────────────────────

export interface WinLossDataGate {
  factor: string
  required: number
  actual: number
  sufficient: boolean
}

export interface WinLossFactor {
  name: string
  score: number
  weight: number
  evidence: string
  data_points: number
  gated: boolean
}

export interface WinLossTrigger {
  trigger: string
  frequency: number
  urgency: number
  source: string
}

export interface WinLossQuote {
  quote: string
  source: string
  role_type: string
  urgency: number
}

export interface WinLossObjection {
  objection: string
  frequency: number
  counter: string
}

export interface WinLossPrediction {
  vendor_name: string
  win_probability: number
  confidence: string
  verdict: string
  is_gated: boolean
  data_gates: WinLossDataGate[]
  factors: WinLossFactor[]
  switching_triggers: WinLossTrigger[]
  proof_quotes: WinLossQuote[]
  objections: WinLossObjection[]
  displacement_targets: { vendor: string; mentions: number; driver: string; strength: string }[]
  segment_match: { typical_sizes: unknown; typical_industries: unknown; size_match: number; industry_match: number } | null
  data_coverage: Record<string, number>
  weights_source: string
  calibration_version: number | null
  recommended_approach: string | null
  lead_with: string[]
  talking_points: string[]
  timing_advice: string | null
  risk_factors: string[]
  prediction_id: string | null
}

export interface RecentPrediction {
  prediction_id: string
  vendor_name: string
  win_probability: number
  confidence: string
  is_gated: boolean
  created_at: string
}

export async function predictWinLoss(params: {
  vendor_name: string
  company_size?: string
  industry?: string
}) {
  return post<WinLossPrediction>(PREDICT_BASE, '/win-loss', params)
}

export async function fetchRecentPredictions(limit: number = 10) {
  return get<{ predictions: RecentPrediction[]; count: number }>(PREDICT_BASE, '/win-loss/recent', { limit })
}

export async function fetchPredictionById(predictionId: string) {
  return get<WinLossPrediction>(PREDICT_BASE, `/win-loss/${encodeURIComponent(predictionId)}`)
}

export interface FactorComparison {
  name: string
  vendor_a_score: number
  vendor_b_score: number
  advantage: 'a' | 'b' | 'tie'
}

export interface WinLossCompareResponse {
  vendor_a: WinLossPrediction
  vendor_b: WinLossPrediction
  easier_target: string
  probability_delta: number
  factor_comparison: FactorComparison[]
}

export async function compareWinLoss(params: {
  vendor_a: string
  vendor_b: string
  company_size?: string
  industry?: string
}) {
  return post<WinLossCompareResponse>(PREDICT_BASE, '/win-loss/compare', params)
}

export function downloadPredictionCsv(predictionId: string) {
  const url = new URL(PREDICT_BASE + `/win-loss/${encodeURIComponent(predictionId)}/csv`, window.location.origin)
  const token = localStorage.getItem('atlas_token')
  if (token) url.searchParams.set('token', token)
  window.open(url.toString(), '_blank')
}
