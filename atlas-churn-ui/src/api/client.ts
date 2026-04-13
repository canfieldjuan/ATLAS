import { tryRefreshToken } from '../auth/AuthContext'
import { buildCurrentRedirectTarget, buildLoginRedirectPath } from '../auth/redirects'
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
  ManualQueueEntry,
  CompanyOverride,
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
  ReasoningReferenceIds,
  CompanySignalCandidateGroupSummary,
  CompanySignalCandidateGroupListResponse,
  CompanySignalReviewImpactSummary,
  WatchlistDeliveryOpsSummary,
  WatchlistDeliveryOpsDetail,
} from '../types'
import { normalizeReportDetail, normalizeVendorProfile } from '../lib/reportNormalization'

const API_BASE = import.meta.env.VITE_API_BASE || ''
const TENANT_BASE = `${API_BASE}/api/v1/b2b/tenant`
const AFFILIATES_BASE = `${TENANT_BASE}/affiliates`
const CAMPAIGNS_BASE = `${API_BASE}/api/v1/b2b/campaigns`
const TARGETS_BASE = `${API_BASE}/api/v1/b2b/vendor-targets`
const PREDICT_BASE = `${API_BASE}/api/v1/b2b/predict`
const EVIDENCE_BASE = `${API_BASE}/api/v1/b2b/evidence`
const BLOG_ADMIN_BASE = `${API_BASE}/api/v1/admin/blog`
const PROSPECTS_BASE = `${API_BASE}/api/v1/b2b/prospects`
const BRIEFINGS_BASE = `${API_BASE}/api/v1/b2b/briefings`
const WEBHOOKS_BASE = `${API_BASE}/api/v1/b2b/dashboard`
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
  const redirectTarget = buildCurrentRedirectTarget(window.location)
  window.location.replace(buildLoginRedirectPath(redirectTarget))
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
  vendor_alert_threshold?: number
  stale_days_threshold?: number
  limit?: number
}) {
  return get<{
    signals: ChurnSignal[]
    count: number
    vendor_alert_threshold?: number | null
    vendor_alert_hit_count?: number
    stale_days_threshold?: number | null
    stale_threshold_hit_count?: number
  }>(TENANT_BASE, '/slow-burn-watchlist', params)
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

export async function fetchLeadDetail(company: string) {
  return get<{ company: string; reviews: ReviewSummary[]; count: number }>(
    TENANT_BASE,
    `/leads/${encodeURIComponent(company)}`,
  )
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
  quality_status?: string
  freshness_state?: string
  review_state?: string
  include_stale?: boolean
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

export async function requestBattleCardReport(body: {
  vendor_name: string
  refresh?: boolean
}) {
  return post<Record<string, unknown>>(TENANT_BASE, '/reports/battle-card', body)
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

export async function pushToCrm(
  opportunities: {
    company: string
    vendor: string
    urgency: number
    pain?: string
    role_type?: string
    buying_stage?: string
    contract_end?: string
    decision_timeline?: string
    decision_maker?: boolean | null
    competitor_context?: string
    primary_quote?: string
    trust_tier?: string
    source?: string | null
    review_id?: string | null
    seat_count?: number | null
    industry?: string | null
    company_size?: string | null
    company_domain?: string | null
    company_country?: string | null
    revenue_range?: string | null
    alternatives?: string[]
  }[],
) {
  return post<{ pushed: number; failed: { company: string; vendor: string; reason: string }[] }>(
    TENANT_BASE, '/push-to-crm', { opportunities },
  )
}

export type WebhookEventType = 'change_event' | 'churn_alert' | 'report_generated' | 'signal_update' | 'high_intent_push'
export type WebhookChannel = 'generic' | 'slack' | 'teams' | 'crm_hubspot' | 'crm_salesforce' | 'crm_pipedrive'

export interface AlertAccountReviewFocus {
  vendor: string
  company: string
  report_date: string
  watch_vendor: string
  category: string
  track_mode: string
}

export interface WebhookLatestCrmPushSummary {
  id: string
  signal_type: string
  signal_id: string | null
  review_id: string | null
  report_id: string | null
  report_type: string | null
  report_title: string | null
  vendor_name: string | null
  company_name: string | null
  crm_record_id: string | null
  crm_record_type: string | null
  status: string
  error: string | null
  pushed_at: string | null
  account_review_focus?: AlertAccountReviewFocus | null
}

export interface WebhookSubscription {
  id: string
  url: string
  event_types: WebhookEventType[]
  channel: WebhookChannel
  enabled: boolean
  description: string | null
  created_at: string
  updated_at?: string
  recent_deliveries_7d?: number
  recent_success_rate_7d?: number | null
  latest_failure_event_type?: string | null
  latest_failure_status_code?: number | null
  latest_failure_error?: string | null
  latest_failure_at?: string | null
  latest_failure_signal_id?: string | null
  latest_failure_review_id?: string | null
  latest_failure_report_id?: string | null
  latest_failure_report_type?: string | null
  latest_failure_report_title?: string | null
  latest_failure_vendor_name?: string | null
  latest_failure_company_name?: string | null
  latest_failure_account_review_focus?: AlertAccountReviewFocus | null
  latest_test_success?: boolean | null
  latest_test_status_code?: number | null
  latest_test_error?: string | null
  latest_test_at?: string | null
  latest_test_signal_id?: string | null
  latest_test_review_id?: string | null
  latest_test_report_id?: string | null
  latest_test_report_type?: string | null
  latest_test_report_title?: string | null
  latest_test_vendor_name?: string | null
  latest_test_company_name?: string | null
  latest_test_account_review_focus?: AlertAccountReviewFocus | null
  latest_crm_push?: WebhookLatestCrmPushSummary | null
}

export interface WebhookDeliverySummary {
  window_days: number
  active_subscriptions: number
  total_deliveries: number
  successful: number
  failed: number
  success_rate: number | null
  avg_success_duration_ms: number | null
  p95_success_duration_ms: number | null
  last_delivery_at: string | null
}

export interface WebhookDelivery {
  id: string
  event_type: string
  status_code: number | null
  duration_ms: number | null
  attempt: number
  success: boolean
  error: string | null
  delivered_at: string
  vendor_name?: string | null
  company_name?: string | null
  signal_id?: string | null
  signal_type?: string | null
  review_id?: string | null
  report_id?: string | null
  report_type?: string | null
  report_title?: string | null
  account_review_focus?: AlertAccountReviewFocus | null
}

export interface WebhookCrmPushLogEntry {
  id: string
  signal_type: string
  signal_id: string | null
  review_id?: string | null
  report_id?: string | null
  report_type?: string | null
  report_title?: string | null
  vendor_name: string | null
  company_name: string | null
  crm_record_id: string | null
  crm_record_type: string | null
  status: string
  error: string | null
  pushed_at: string
  account_review_focus?: AlertAccountReviewFocus | null
}

export interface WebhookCreateBody {
  url: string
  secret: string
  event_types: WebhookEventType[]
  channel: WebhookChannel
  auth_header?: string
  description?: string
}

export interface WebhookUpdateBody {
  url?: string
  event_types?: WebhookEventType[]
  enabled?: boolean
  description?: string
}

export interface WebhookTestResult {
  success: boolean
  subscription_id: string
  channel: WebhookChannel
  error?: string
}

export async function listWebhooks(params?: {
  vendor_name?: string
  company_name?: string
}) {
  return get<{ webhooks: WebhookSubscription[]; count: number }>(WEBHOOKS_BASE, '/webhooks', params)
}

export async function fetchWebhookDeliverySummary(days = 7, params?: {
  vendor_name?: string
  company_name?: string
}) {
  return get<WebhookDeliverySummary>(WEBHOOKS_BASE, '/webhooks/delivery-summary', { days, ...params })
}

export async function createWebhook(body: WebhookCreateBody) {
  return post<WebhookSubscription>(WEBHOOKS_BASE, '/webhooks', body)
}

export async function updateWebhookSubscription(webhookId: string, body: WebhookUpdateBody) {
  return patch<WebhookSubscription>(WEBHOOKS_BASE, `/webhooks/${encodeURIComponent(webhookId)}`, body)
}

export async function deleteWebhookSubscription(webhookId: string) {
  return del<{ deleted: boolean; id: string }>(WEBHOOKS_BASE, `/webhooks/${encodeURIComponent(webhookId)}`)
}

export async function testWebhookSubscription(webhookId: string) {
  return post<WebhookTestResult>(WEBHOOKS_BASE, `/webhooks/${encodeURIComponent(webhookId)}/test`)
}

export async function listWebhookDeliveries(webhookId: string, params?: {
  success?: boolean
  event_type?: string
  limit?: number
  vendor_name?: string
  company_name?: string
}) {
  return get<{ deliveries: WebhookDelivery[]; count: number }>(
    WEBHOOKS_BASE,
    `/webhooks/${encodeURIComponent(webhookId)}/deliveries`,
    params,
  )
}

export async function listWebhookCrmPushLog(webhookId: string, params: number | {
  limit?: number
  status?: 'success' | 'failed'
  vendor_name?: string
  company_name?: string
} = 20) {
  const normalizedParams = typeof params === 'number' ? { limit: params } : params
  return get<{ pushes: WebhookCrmPushLogEntry[]; count: number }>(
    WEBHOOKS_BASE,
    `/webhooks/${encodeURIComponent(webhookId)}/crm-push-log`,
    normalizedParams,
  )
}

export async function fetchCompanySignalReviewImpactSummary(params?: {
  vendor_name?: string
  review_scope?: string
  review_action?: string
  company_signal_action?: string
  canonical_gap_reason?: string
  review_priority_band?: string
  review_priority_reason?: string
  review_unlock_path?: string
  review_unlock_reason?: string
  candidate_source?: string
  rebuild_outcome?: string
  rebuild_reason?: string
  window_days?: number
  top_n?: number
}) {
  return get<CompanySignalReviewImpactSummary>(
    TENANT_BASE,
    '/company-signal-review-impact-summary',
    params,
  )
}

export async function fetchCompanySignalCandidateGroupSummary(params?: {
  vendor_name?: string
  company_name?: string
  source_name?: string
  candidate_bucket?: string
  review_status?: string
  canonical_gap_reason?: string
  review_priority_band?: string
  review_priority_reason?: string
  min_urgency?: number
  min_confidence?: number
  min_reviews?: number
  decision_makers_only?: boolean
  signal_evidence_present?: boolean
  window_days?: number
  top_n?: number
}) {
  return get<CompanySignalCandidateGroupSummary>(
    TENANT_BASE,
    '/company-signal-candidate-group-summary',
    params,
  )
}

export async function fetchCompanySignalCandidateGroups(params?: {
  vendor_name?: string
  company_name?: string
  source_name?: string
  candidate_bucket?: string
  review_status?: string
  canonical_gap_reason?: string
  review_priority_band?: string
  review_priority_reason?: string
  min_urgency?: number
  min_confidence?: number
  min_reviews?: number
  decision_makers_only?: boolean
  signal_evidence_present?: boolean
  window_days?: number
  limit?: number
}) {
  return get<CompanySignalCandidateGroupListResponse>(
    TENANT_BASE,
    '/company-signal-candidate-groups',
    params,
  )
}

export async function approveCompanySignalCandidateGroup(groupId: string, body?: {
  notes?: string
  trigger_rebuild?: boolean
}) {
  const payload = {
    trigger_rebuild: body?.trigger_rebuild ?? true,
    ...(body?.notes !== undefined ? { notes: body.notes } : {}),
  }
  return post<Record<string, unknown>>(
    TENANT_BASE,
    `/company-signal-candidate-groups/${encodeURIComponent(groupId)}/approve`,
    payload,
  )
}

export async function suppressCompanySignalCandidateGroup(groupId: string, body?: {
  notes?: string
  trigger_rebuild?: boolean
}) {
  const payload = {
    trigger_rebuild: body?.trigger_rebuild ?? true,
    ...(body?.notes !== undefined ? { notes: body.notes } : {}),
  }
  return post<Record<string, unknown>>(
    TENANT_BASE,
    `/company-signal-candidate-groups/${encodeURIComponent(groupId)}/suppress`,
    payload,
  )
}

export async function approveCompanySignalCandidateGroups(body: {
  group_ids: string[]
  notes?: string
  trigger_rebuild?: boolean
}) {
  const payload = {
    group_ids: body.group_ids,
    trigger_rebuild: body.trigger_rebuild ?? true,
    ...(body.notes !== undefined ? { notes: body.notes } : {}),
  }
  return post<Record<string, unknown>>(
    TENANT_BASE,
    '/company-signal-candidate-groups/approve',
    payload,
  )
}

export async function suppressCompanySignalCandidateGroups(body: {
  group_ids: string[]
  notes?: string
  trigger_rebuild?: boolean
}) {
  const payload = {
    group_ids: body.group_ids,
    trigger_rebuild: body.trigger_rebuild ?? true,
    ...(body.notes !== undefined ? { notes: body.notes } : {}),
  }
  return post<Record<string, unknown>>(
    TENANT_BASE,
    '/company-signal-candidate-groups/suppress',
    payload,
  )
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
  base: string = TENANT_BASE,
) {
  const url = new URL(base + path, window.location.origin)
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

export function downloadProspectsCsv(params?: Record<string, string | number | boolean | undefined>) {
  downloadCsv('/export', params, PROSPECTS_BASE)
}

export function downloadCampaignsCsv(params?: Record<string, string | number | boolean | undefined>) {
  downloadCsv('/export', params, CAMPAIGNS_BASE)
}

export function downloadBriefingsCsv(params?: Record<string, string | number | boolean | undefined>) {
  downloadCsv('/export', params, BRIEFINGS_BASE)
}

export async function generateVendorReport(id: string) {
  return post<{ status: string; signal_count?: number; high_urgency_count?: number }>(TARGETS_BASE, `/${id}/generate-report`)
}

// ---------------------------------------------------------------------------
// Opportunity Dispositions
// ---------------------------------------------------------------------------

export interface OpportunityDisposition {
  id: string
  opportunity_key: string
  company: string
  vendor: string
  review_id: string | null
  disposition: 'snoozed' | 'dismissed' | 'saved'
  snoozed_until: string | null
  created_at: string
  updated_at: string
}

export async function fetchDispositions(params?: { disposition?: string }) {
  return get<{ dispositions: OpportunityDisposition[]; count: number }>(
    TENANT_BASE,
    '/opportunity-dispositions',
    params,
  )
}

export async function setDisposition(body: {
  opportunity_key: string
  company: string
  vendor: string
  review_id?: string | null
  disposition: 'snoozed' | 'dismissed' | 'saved'
  snoozed_until?: string | null
}) {
  return post<OpportunityDisposition>(TENANT_BASE, '/opportunity-dispositions', body)
}

export async function bulkSetDisposition(body: {
  items: {
    opportunity_key: string
    company: string
    vendor: string
    review_id?: string | null
  }[]
  disposition: 'snoozed' | 'dismissed' | 'saved'
  snoozed_until?: string | null
}) {
  return post<{ updated: number }>(TENANT_BASE, '/opportunity-dispositions/bulk', body)
}

export async function removeDispositions(body: { opportunity_keys: string[] }) {
  return post<{ removed: number }>(TENANT_BASE, '/opportunity-dispositions/remove', body)
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

export async function fetchManualQueue(params?: {
  company?: string
  limit?: number
  offset?: number
}) {
  return get<{ queue: ManualQueueEntry[]; count: number }>(PROSPECTS_BASE, '/manual-queue', params)
}

export async function resolveManualQueueEntry(
  id: string,
  body: { action: 'retry' | 'dismiss'; domain?: string },
) {
  return post<{ entry: ManualQueueEntry }>(PROSPECTS_BASE, `/manual-queue/${id}/resolve`, body)
}

export async function fetchCompanyOverrides(params?: {
  company?: string
}) {
  return get<{ overrides: CompanyOverride[]; count: number }>(PROSPECTS_BASE, '/company-overrides', params)
}

export async function upsertCompanyOverride(body: {
  company_name_raw: string
  search_names?: string[]
  domains?: string[]
}) {
  return post<{ override: CompanyOverride }>(PROSPECTS_BASE, '/company-overrides', body)
}

export async function deleteCompanyOverride(id: string) {
  return del<{ deleted: boolean }>(PROSPECTS_BASE, `/company-overrides/${id}`)
}

export async function bootstrapCompanyOverrides() {
  return post<{ imported: number }>(PROSPECTS_BASE, '/company-overrides/bootstrap')
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

export interface TimelineEvent {
  type: 'campaign_event' | 'sequence_state' | 'signal_detected'
  timestamp: string | null
  vendor?: string
  event?: string
  channel?: string
  step?: number | null
  max_steps?: number | null
  subject?: string | null
  recipient?: string | null
  source?: string | null
  error?: string | null
  opportunity_score?: number | null
  urgency_score?: number | null
  sequence_id?: string
  status?: string
  open_count?: number
  click_count?: number
  last_sent_at?: string | null
  last_opened_at?: string | null
  last_clicked_at?: string | null
  reply_received_at?: string | null
  bounced_at?: string | null
  bounce_type?: string | null
  outcome?: string | null
  outcome_recorded_at?: string | null
  outcome_notes?: string | null
  outcome_revenue?: number | null
  buying_stage?: string | null
  role_type?: string | null
  pain_categories?: string[]
  seat_count?: number | null
  contract_end?: string | null
}

export async function fetchCompanyTimeline(params: {
  company: string
  vendor?: string
}) {
  return get<{ company: string; vendor: string | null; events: TimelineEvent[]; count: number }>(
    CAMPAIGNS_BASE,
    '/company-timeline',
    params,
  )
}

export async function bulkApproveCampaigns(ids: string[], action: 'approve' | 'queue-send' | 'reject') {
  return post<{ processed: number; failed: { id: string; reason: string }[] }>(CAMPAIGNS_BASE, '/bulk-approve', { campaign_ids: ids, action })
}

export async function bulkRejectCampaigns(ids: string[], reason?: string) {
  return post<{ rejected: number; failed: { id: string; reason: string }[] }>(CAMPAIGNS_BASE, '/bulk-reject', { campaign_ids: ids, reason })
}

// ---------------------------------------------------------------------------
// Outcome Recording & Calibration
// ---------------------------------------------------------------------------

export async function setSequenceRecipient(sequenceId: string, recipientEmail: string) {
  return post<{ status: string; recipient_email: string }>(
    CAMPAIGNS_BASE,
    `/sequences/${sequenceId}/set-recipient`,
    { recipient_email: recipientEmail },
  )
}

export async function recordSequenceOutcome(sequenceId: string, body: {
  outcome: string
  notes?: string
  revenue?: number
}) {
  return post<{
    status: string
    sequence_id: string
    outcome: string
    previous_outcome: string
    recorded_at: string
  }>(CAMPAIGNS_BASE, `/sequences/${sequenceId}/outcome`, body)
}

export async function fetchSequenceOutcome(sequenceId: string) {
  return get<{
    sequence_id: string
    company_name: string
    outcome: string
    outcome_recorded_at: string | null
    outcome_recorded_by: string | null
    outcome_notes: string | null
    outcome_revenue: number | null
    outcome_history: { stage: string; recorded_at: string; previous: string; revenue: number | null; recorded_by: string }[]
  }>(CAMPAIGNS_BASE, `/sequences/${sequenceId}/outcome`)
}

export interface SignalEffectivenessGroup {
  signal_group: string
  total_sequences: number
  meetings: number
  deals_opened: number
  deals_won: number
  deals_lost: number
  no_opportunity: number
  disqualified: number
  positive_outcome_rate: number
  total_revenue: number
}

export async function fetchSignalEffectiveness(params?: {
  vendor_name?: string
  min_sequences?: number
  group_by?: string
}) {
  return get<{
    group_by: string
    vendor_filter: string | null
    min_sequences: number
    groups: SignalEffectivenessGroup[]
    total_groups: number
  }>(TENANT_BASE, '/signal-effectiveness', params)
}

export async function fetchOutcomeDistribution(params?: {
  vendor_name?: string
}) {
  return get<{
    total_sequences: number
    vendor_filter: string | null
    buckets: {
      outcome: string
      count: number
      pct: number
      total_revenue: number
      first_recorded: string | null
      last_recorded: string | null
    }[]
  }>(TENANT_BASE, '/outcome-distribution', params)
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
  last_computed_at: string | null
  latest_snapshot_date: string | null
  latest_accounts_report_date: string | null
  freshness_status: string | null
  freshness_reason: string | null
  freshness_timestamp: string | null
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

export interface CompetitiveSetRun {
  id: string
  competitive_set_id: string
  account_id: string
  run_id: string
  trigger: string
  status: 'running' | 'succeeded' | 'partial' | 'failed'
  execution_id: string | null
  summary: Record<string, unknown>
  started_at: string
  completed_at: string | null
  created_at: string
}

export interface CompetitiveSetDefaults {
  default_refresh_interval_hours: number
  max_competitors: number
  default_changed_vendors_only: boolean
}

export interface CompetitiveSetEstimate {
  lookback_days: number
  vendor_jobs_planned: number
  pairwise_jobs_planned: number
  category_jobs_planned: number
  asymmetry_jobs_planned: number
  estimated_vendor_tokens: number
  estimated_cross_vendor_tokens: number
  estimated_total_tokens: number
  estimated_vendor_cost_usd: number | null
  estimated_cross_vendor_cost_usd: number | null
  estimated_total_cost_usd: number | null
  estimated_vendor_tokens_likely_to_reason: number
  estimated_vendor_cost_usd_likely_to_reason: number | null
  vendor_jobs_with_history: number
  vendor_jobs_using_fallback: number
  cross_vendor_jobs_with_history: number
  cross_vendor_jobs_using_fallback: number
  vendor_jobs_with_matching_pools: number
  vendor_jobs_missing_pools: number
  vendor_jobs_likely_to_reason: number
  vendor_jobs_likely_hash_reuse: number
  vendor_jobs_likely_stale_reuse: number
  vendor_jobs_likely_missing_prior: number
  vendor_jobs_likely_hash_changed: number
  vendor_jobs_likely_prior_quality_weak: number
  vendor_jobs_likely_missing_packet_artifacts: number
  vendor_jobs_likely_missing_reference_ids: number
  likely_rerun_vendors: string[]
  likely_reuse_vendors: string[]
  recent_vendor_sample_count: number
  recent_cross_vendor_sample_count: number
  note: string
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
  estimate?: CompetitiveSetEstimate
}

export interface VendorSearchResult {
  vendor_name: string
  product_category: string | null
  total_reviews: number | null
  avg_urgency: number | null
}

export interface WatchlistView {
  id: string
  name: string
  vendor_name: string | null
  vendor_names: string[]
  category: string | null
  source: string | null
  min_urgency: number | null
  include_stale: boolean
  named_accounts_only: boolean
  changed_wedges_only: boolean
  vendor_alert_threshold: number | null
  account_alert_threshold: number | null
  stale_days_threshold: number | null
  alert_email_enabled: boolean
  alert_delivery_frequency: 'daily' | 'weekly'
  next_alert_delivery_at: string | null
  last_alert_delivery_at: string | null
  last_alert_delivery_status: string | null
  last_alert_delivery_summary: string | null
  last_alert_delivery_suppressed_preview_summary?: {
    count: number
    reasons: Record<string, number>
    reason_details?: Record<string, { summary?: string | null; short_summary?: string | null }> | null
  } | null
  preview_alerts_enabled?: boolean
  preview_account_alert_policy?: {
    applies_to_preview_only: boolean
    enabled: boolean
    enabled_source?: string | null
    min_confidence: number | null
    min_confidence_source?: string | null
    require_budget_authority: boolean
    require_budget_authority_source?: string | null
    override_min_confidence: number | null
    override_require_budget_authority: boolean | null
  } | null
  created_at: string | null
  updated_at: string | null
}

export interface WatchlistAlertEvent {
  id: string
  watchlist_view_id: string
  event_type: 'vendor_alert' | 'account_alert' | 'stale_data'
  threshold_field: 'vendor_alert_threshold' | 'account_alert_threshold' | 'stale_days_threshold'
  entity_type: 'vendor' | 'account' | 'signal_cluster'
  entity_key: string
  vendor_name: string | null
  company_name: string | null
  category: string | null
  source: string | null
  threshold_value: number | null
  summary: string
  payload: Record<string, unknown>
  account_alert_score?: number | null
  account_alert_score_source?: string | null
  account_alert_policy_reason?: string | null
  account_reasoning_preview_only?: boolean | null
  reasoning_reference_ids?: ReasoningReferenceIds | null
  source_review_ids?: string[] | null
  account_review_focus?: AlertAccountReviewFocus | null
  status: 'open' | 'resolved'
  first_seen_at: string | null
  last_seen_at: string | null
  resolved_at: string | null
  created_at: string | null
  updated_at: string | null
}

export interface WatchlistAlertEmailDelivery {
  id: string
  recipient_emails: string[]
  message_ids: string[]
  event_count: number
  status: 'processing' | 'sent' | 'partial' | 'failed' | 'no_events' | 'skipped'
  summary: string
  error: string | null
  delivered_at: string | null
  created_at: string | null
  updated_at: string | null
  scheduled_for?: string | null
  delivery_frequency?: 'daily' | 'weekly' | null
  delivery_mode?: 'live' | 'scheduled' | null
  suppressed_preview_summary?: {
    count: number
    reasons: Record<string, number>
    reason_details?: Record<string, { summary?: string | null; short_summary?: string | null }> | null
  } | null
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

export async function listWatchlistViews() {
  return get<{ views: WatchlistView[]; count: number }>(TENANT_BASE, '/watchlist-views')
}

export async function createWatchlistView(body: {
  name: string
  vendor_name?: string
  category?: string
  source?: string
  min_urgency?: number
  include_stale?: boolean
  named_accounts_only?: boolean
  changed_wedges_only?: boolean
  vendor_alert_threshold?: number
  account_alert_threshold?: number
  stale_days_threshold?: number
  alert_email_enabled?: boolean
  alert_delivery_frequency?: 'daily' | 'weekly'
}) {
  return post<WatchlistView>(TENANT_BASE, '/watchlist-views', body)
}

export async function updateWatchlistView(
  watchlistViewId: string,
  body: {
    name: string
    vendor_name?: string
    category?: string
    source?: string
    min_urgency?: number
    include_stale?: boolean
    named_accounts_only?: boolean
    changed_wedges_only?: boolean
    vendor_alert_threshold?: number
    account_alert_threshold?: number
    stale_days_threshold?: number
    alert_email_enabled?: boolean
    alert_delivery_frequency?: 'daily' | 'weekly'
  },
) {
  return put<WatchlistView>(TENANT_BASE, `/watchlist-views/${encodeURIComponent(watchlistViewId)}`, body)
}

export async function deleteWatchlistView(watchlistViewId: string) {
  return del<{ deleted: boolean; watchlist_view_id: string }>(
    TENANT_BASE,
    `/watchlist-views/${encodeURIComponent(watchlistViewId)}`,
  )
}

export async function listWatchlistAlertEvents(
  watchlistViewId: string,
  params?: { status?: 'open' | 'resolved' | 'all'; limit?: number },
) {
  return get<{
    watchlist_view_id: string
    watchlist_view_name: string
    status: 'open' | 'resolved' | 'all'
    events: WatchlistAlertEvent[]
    count: number
  }>(
    TENANT_BASE,
    `/watchlist-views/${encodeURIComponent(watchlistViewId)}/alert-events`,
    params,
  )
}

export async function evaluateWatchlistAlertEvents(watchlistViewId: string) {
  return post<{
    watchlist_view_id: string
    watchlist_view_name: string
    evaluated_at: string
    events: WatchlistAlertEvent[]
    count: number
    new_open_event_count: number
    resolved_event_count: number
  }>(
    TENANT_BASE,
    `/watchlist-views/${encodeURIComponent(watchlistViewId)}/alert-events/evaluate`,
  )
}

export async function listWatchlistAlertEmailLog(
  watchlistViewId: string,
  params?: { limit?: number },
) {
  return get<{
    watchlist_view_id: string
    watchlist_view_name: string
    deliveries: WatchlistAlertEmailDelivery[]
    count: number
  }>(
    TENANT_BASE,
    `/watchlist-views/${encodeURIComponent(watchlistViewId)}/alert-email-log`,
    params,
  )
}

export async function deliverWatchlistAlertEmail(
  watchlistViewId: string,
  body?: { evaluate_before_send?: boolean },
) {
  return post<{
    watchlist_view_id: string
    watchlist_view_name: string
    status: 'sent' | 'partial' | 'failed' | 'no_events'
    recipient_emails: string[]
    event_count: number
    message_ids: string[]
    summary: string
    error: string | null
  }>(
    TENANT_BASE,
    `/watchlist-views/${encodeURIComponent(watchlistViewId)}/alert-events/deliver-email`,
    body,
  )
}

export async function listCompetitiveSets(include_inactive: boolean = false) {
  return get<{ competitive_sets: CompetitiveSet[]; count: number; defaults?: CompetitiveSetDefaults }>(
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
  return get<{ competitive_set: CompetitiveSet; plan: CompetitiveSetPlan; recent_runs: CompetitiveSetRun[] }>(
    TENANT_BASE,
    `/competitive-sets/${encodeURIComponent(competitiveSetId)}/plan`,
  )
}

export async function runCompetitiveSetNow(
  competitiveSetId: string,
  body?: { force?: boolean; force_cross_vendor?: boolean; changed_vendors_only?: boolean },
) {
  return post<{
    execution_id: string | null
    status: string
    message: string
    already_running?: boolean
    competitive_set_id: string
    plan: CompetitiveSetPlan
  }>(TENANT_BASE, `/competitive-sets/${encodeURIComponent(competitiveSetId)}/run`, body)
}

export interface AccountsInMotionFeedItem {
  source_reviews: Array<{
    id: string
    source: string | null
    source_url: string | null
    vendor_name: string
    rating: number | null
    summary: string | null
    review_excerpt: string | null
    reviewer_name: string | null
    reviewer_title: string | null
    reviewer_company: string | null
    reviewed_at: string | null
  }>
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
  reasoning_reference_ids: ReasoningReferenceIds | null
  source_distribution: Record<string, number>
  source_review_ids: string[]
  evidence_count: number
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
  freshness_status?: string | null
  freshness_reason?: string | null
  freshness_timestamp?: string | null
  account_alert_hit?: boolean
  stale_threshold_hit?: boolean
}

export async function fetchAccountsInMotionFeed(params?: {
  vendor_name?: string
  category?: string
  source?: string
  min_urgency?: number
  include_stale?: boolean
  account_alert_threshold?: number
  stale_days_threshold?: number
  per_vendor_limit?: number
  limit?: number
}) {
  return get<{
    accounts: AccountsInMotionFeedItem[]
    count: number
    tracked_vendor_count: number
    vendors_with_accounts: number
    min_urgency: number
    account_alert_threshold?: number | null
    account_alert_hit_count?: number
    stale_days_threshold?: number | null
    stale_threshold_hit_count?: number
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

export async function fetchWatchlistDeliveryOps(params?: {
  days?: number
  limit?: number
}) {
  return get<WatchlistDeliveryOpsSummary>(
    VISIBILITY_BASE,
    '/watchlist-delivery',
    params as Record<string, string | number>,
  )
}

export async function fetchWatchlistDeliveryViewDetail(
  watchlistViewId: string,
  params?: { event_status?: 'open' | 'resolved' | 'all'; event_limit?: number; log_limit?: number },
) {
  return get<WatchlistDeliveryOpsDetail>(
    VISIBILITY_BASE,
    `/watchlist-delivery/views/${encodeURIComponent(watchlistViewId)}`,
    params as Record<string, string | number>,
  )
}

export async function runWatchlistDeliveryForView(watchlistViewId: string) {
  return post<{
    watchlist_view_id: string
    watchlist_view_name: string
    status: string
    recipient_emails: string[]
    event_count: number
    message_ids: string[]
    summary: string
    error: string | null
  }>(
    VISIBILITY_BASE,
    `/watchlist-delivery/views/${encodeURIComponent(watchlistViewId)}/deliver-now`,
  )
}

export async function disableWatchlistDeliveryForView(watchlistViewId: string) {
  return post<{
    disabled: boolean
    view: {
      id: string
      alert_email_enabled: boolean
      next_alert_delivery_at: string | null
      last_alert_delivery_status: string | null
      last_alert_delivery_summary: string | null
    }
  }>(
    VISIBILITY_BASE,
    `/watchlist-delivery/views/${encodeURIComponent(watchlistViewId)}/disable-email`,
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
  source_name?: string
  event_type?: string
  entity_type?: string
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
  source_name?: string
  event_type?: string
  entity_type?: string
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
  win_probability: number | null
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
  win_probability: number | null
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

// -- Evidence Explorer --------------------------------------------------------

export interface EvidenceWitness {
  witness_id: string
  review_id: string
  witness_type: string
  excerpt_text: string
  source: string
  reviewed_at: string | null
  reviewer_company: string | null
  reviewer_title: string | null
  pain_category: string | null
  competitor: string | null
  salience_score: number | null
  specificity_score: number | null
  selection_reason: string | null
  signal_tags: string[] | null
  as_of_date: string | null
}

export interface EvidenceWitnessDetail extends EvidenceWitness {
  review_text: string | null
  summary: string | null
  pros: string | null
  cons: string | null
  rating: number | null
  review_source: string | null
  source_url: string | null
  enrichment_status: string | null
  evidence_spans: Array<{
    signal_type: string
    raw_text: string
    pain_category: string | null
    excerpt_text: string | null
    start_char: number | null
    end_char: number | null
  }>
  all_evidence_span_count: number
}

export interface EvidenceFacets {
  pain_categories: string[]
  sources: string[]
  witness_types: string[]
}

export interface EvidenceVault {
  vendor_name: string
  as_of_date: string | null
  analysis_window_days: number | null
  schema_version: string | null
  created_at: string | null
  weakness_evidence: Array<Record<string, unknown>>
  strength_evidence: Array<Record<string, unknown>>
  company_signals: Array<Record<string, unknown>>
  metric_snapshot: Record<string, unknown>
  provenance: Record<string, unknown>
  witness_count: number
}

export interface EvidenceTrace {
  vendor_name: string
  trace: {
    synthesis: {
      as_of_date: string
      schema_version: string
      evidence_hash: string
      sections: Record<string, unknown>
      llm_model: string | null
      tokens_used: number | null
    } | null
    reasoning_packet: {
      as_of_date: string
      evidence_hash: string
      section_count: number
      witness_pack_size: number
    } | null
    witnesses: EvidenceWitness[]
    source_reviews: Array<{
      id: string
      source: string
      source_url: string | null
      vendor_name: string
      rating: number | null
      summary: string | null
      review_excerpt: string | null
      reviewer_name: string | null
      reviewer_title: string | null
      reviewer_company: string | null
      reviewed_at: string | null
    }>
    evidence_diff: {
      computed_date: string
      confirmed_count: number
      contradicted_count: number
      novel_count: number
      missing_count: number
      diff_ratio: number
      decision: string
      has_core_contradiction: boolean
    } | null
  }
  stats: {
    witness_count: number
    unique_reviews: number
    has_synthesis: boolean
    has_packet: boolean
    has_diff: boolean
  }
}

export async function fetchWitnesses(params: {
  vendor_name: string
  as_of_date?: string
  window_days?: number
  pain_category?: string
  source?: string
  competitor?: string
  witness_type?: string
  min_salience?: number
  limit?: number
  offset?: number
}) {
  return get<{
    vendor_name: string
    as_of_date: string | null
    analysis_window_days: number
    witnesses: EvidenceWitness[]
    total: number
    limit: number
    offset: number
    facets: EvidenceFacets
  }>(EVIDENCE_BASE, '/witnesses', params as Record<string, string | number | boolean>)
}

export async function fetchWitness(
  witnessId: string,
  vendorName: string,
  params?: { as_of_date?: string; window_days?: number },
) {
  return get<{ witness: EvidenceWitnessDetail }>(
    EVIDENCE_BASE,
    `/witnesses/${encodeURIComponent(witnessId)}`,
    { vendor_name: vendorName, ...(params || {}) },
  )
}

export async function fetchEvidenceVault(params: {
  vendor_name: string
  as_of_date?: string
  window_days?: number
}) {
  return get<EvidenceVault | { vendor_name: string; vault: null; message: string }>(
    EVIDENCE_BASE, '/vault', params as Record<string, string | number | boolean>,
  )
}

export async function fetchEvidenceTrace(params: {
  vendor_name: string
  as_of_date?: string
  window_days?: number
}) {
  return get<EvidenceTrace>(
    EVIDENCE_BASE, '/trace', params as Record<string, string | number | boolean>,
  )
}

// -- Evidence Annotations -----------------------------------------------------

export interface EvidenceAnnotation {
  id: string
  witness_id: string
  vendor_name: string
  annotation_type: 'pin' | 'flag' | 'suppress'
  note_text: string | null
  created_at: string
  updated_at: string
}

export async function fetchAnnotations(params?: {
  vendor_name?: string
  annotation_type?: string
}) {
  return get<{ annotations: EvidenceAnnotation[]; count: number }>(
    EVIDENCE_BASE,
    '/annotations',
    params,
  )
}

export async function setAnnotation(body: {
  witness_id: string
  vendor_name: string
  annotation_type: 'pin' | 'flag' | 'suppress'
  note_text?: string | null
}) {
  return post<EvidenceAnnotation>(EVIDENCE_BASE, '/annotations', body)
}

export async function removeAnnotations(body: { witness_ids: string[] }) {
  return post<{ removed: number }>(EVIDENCE_BASE, '/annotations/remove', body)
}

// -- Report PDF Export --------------------------------------------------------

export function downloadReportPdf(reportId: string) {
  const url = new URL(`${TENANT_BASE}/reports/${encodeURIComponent(reportId)}/pdf`, window.location.origin)
  const token = localStorage.getItem('atlas_token')
  if (token) url.searchParams.set('token', token)
  window.open(url.toString(), '_blank')
}

// -- Report Subscriptions -----------------------------------------------------

export type ReportSubscriptionScopeType = 'library' | 'library_view' | 'report'

export interface ReportLibraryViewFilters {
  report_type?: string
  vendor_filter?: string
  quality_status?: string
  freshness_state?: string
  review_state?: string
}

function normalizeFilterPart(value?: string) {
  const normalized = (value || '').trim().toLowerCase()
  return normalized || 'all'
}

function slugifyFilterPart(value: string) {
  return value.replace(/[^a-z0-9_-]+/g, '-').replace(/-+/g, '-').replace(/^-|-$/g, '') || 'all'
}

export function normalizeReportLibraryViewFilters(filters?: ReportLibraryViewFilters | null): ReportLibraryViewFilters {
  const normalized: ReportLibraryViewFilters = {}
  const reportType = (filters?.report_type || '').trim()
  const vendorFilter = (filters?.vendor_filter || '').trim()
  const qualityStatus = (filters?.quality_status || '').trim()
  const freshnessState = (filters?.freshness_state || '').trim()
  const reviewState = (filters?.review_state || '').trim()
  if (reportType) normalized.report_type = reportType
  if (vendorFilter) normalized.vendor_filter = vendorFilter
  if (qualityStatus) normalized.quality_status = qualityStatus
  if (freshnessState) normalized.freshness_state = freshnessState
  if (reviewState) normalized.review_state = reviewState
  return normalized
}

export function buildReportLibraryViewScopeKey(filters?: ReportLibraryViewFilters | null) {
  const normalized = normalizeReportLibraryViewFilters(filters)
  const typePart = slugifyFilterPart(normalizeFilterPart(normalized.report_type))
  const vendorPart = slugifyFilterPart(normalizeFilterPart(normalized.vendor_filter))
  const qualityPart = slugifyFilterPart(normalizeFilterPart(normalized.quality_status))
  const freshnessPart = slugifyFilterPart(normalizeFilterPart(normalized.freshness_state))
  const reviewPart = slugifyFilterPart(normalizeFilterPart(normalized.review_state))
  return `library-view--type-${typePart}--vendor-${vendorPart}--quality-${qualityPart}--freshness-${freshnessPart}--review-${reviewPart}`
}

export interface ReportSubscription {
  id: string
  account_id: string
  scope_type: ReportSubscriptionScopeType
  scope_key: string
  scope_label: string
  filter_payload: ReportLibraryViewFilters
  delivery_frequency: 'weekly' | 'monthly' | 'quarterly'
  deliverable_focus: 'all' | 'battle_cards' | 'executive_reports' | 'comparison_packs'
  freshness_policy: 'fresh_only' | 'fresh_or_monitor' | 'any'
  recipient_emails: string[]
  delivery_note: string
  enabled: boolean
  next_delivery_at: string | null
  last_delivery_at?: string | null
  last_delivery_status?: string | null
  last_delivery_summary?: string | null
  last_delivery_report_count?: number | null
  created_at: string
  updated_at: string
}

export interface ReportSubscriptionUpsert {
  scope_label: string
  filter_payload?: ReportLibraryViewFilters
  delivery_frequency: 'weekly' | 'monthly' | 'quarterly'
  deliverable_focus: 'all' | 'battle_cards' | 'executive_reports' | 'comparison_packs'
  freshness_policy: 'fresh_only' | 'fresh_or_monitor' | 'any'
  recipients: string[]
  delivery_note?: string
  enabled: boolean
}

export async function fetchReportSubscription(scopeType: ReportSubscriptionScopeType, scopeKey: string) {
  return get<{ subscription: ReportSubscription | null }>(
    TENANT_BASE, `/report-subscriptions/${encodeURIComponent(scopeType)}/${encodeURIComponent(scopeKey)}`,
  )
}

export async function listReportSubscriptions() {
  return get<{ subscriptions: ReportSubscription[] }>(TENANT_BASE, '/report-subscriptions')
}

export async function upsertReportSubscription(scopeType: ReportSubscriptionScopeType, scopeKey: string, body: ReportSubscriptionUpsert) {
  return put<{ subscription: ReportSubscription }>(
    TENANT_BASE, `/report-subscriptions/${encodeURIComponent(scopeType)}/${encodeURIComponent(scopeKey)}`, body,
  )
}
