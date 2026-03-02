import { tryRefreshToken } from '../auth/AuthContext'

const BASE = '/api/v1/b2b/tenant'

function authHeaders(): Record<string, string> {
  const token = localStorage.getItem('atlas_token')
  return token ? { Authorization: `Bearer ${token}` } : {}
}

function forceLogout(): never {
  localStorage.removeItem('atlas_token')
  localStorage.removeItem('atlas_refresh_token')
  window.location.href = '/login'
  throw new Error('Session expired')
}

async function handleResponse<T>(res: Response, retry: () => Promise<Response>): Promise<T> {
  if (res.status === 401) {
    const newToken = await tryRefreshToken()
    if (!newToken) forceLogout()
    const retryRes = await retry()
    if (retryRes.status === 401) forceLogout()
    if (!retryRes.ok) {
      const body = await retryRes.text().catch(() => '')
      throw new Error(`API ${retryRes.status}: ${body || retryRes.statusText}`)
    }
    return retryRes.json()
  }
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`API ${res.status}: ${body || res.statusText}`)
  }
  return res.json()
}

async function get<T>(path: string, params?: Record<string, string | number | boolean | undefined>): Promise<T> {
  const url = new URL(`${BASE}${path}`, window.location.origin)
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

async function post<T>(path: string, body?: unknown): Promise<T> {
  const doFetch = () =>
    fetch(`${BASE}${path}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: body ? JSON.stringify(body) : undefined,
    })
  const res = await doFetch()
  return handleResponse<T>(res, doFetch)
}

async function del<T>(path: string): Promise<T> {
  const doFetch = () =>
    fetch(`${BASE}${path}`, {
      method: 'DELETE',
      headers: authHeaders(),
    })
  const res = await doFetch()
  return handleResponse<T>(res, doFetch)
}

async function patch<T>(path: string, body?: unknown): Promise<T> {
  const doFetch = () =>
    fetch(`${BASE}${path}`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json', ...authHeaders() },
      body: body ? JSON.stringify(body) : undefined,
    })
  const res = await doFetch()
  return handleResponse<T>(res, doFetch)
}

// -- Types --

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

export interface ChurnSignal {
  vendor_name: string
  product_category: string | null
  total_reviews: number | null
  churn_intent_count: number | null
  avg_urgency_score: number
  avg_rating_normalized: number | null
  nps_proxy: number | null
  price_complaint_rate: number | null
  decision_maker_churn_rate: number | null
  last_computed_at: string | null
}

export interface VendorDetail {
  vendor_name: string
  churn_signal: {
    avg_urgency_score: number
    churn_intent_count: number
    total_reviews: number
    nps_proxy: number | null
    price_complaint_rate: number | null
    decision_maker_churn_rate: number | null
    top_pain_categories: unknown
    top_competitors: unknown
    top_feature_gaps: unknown
    quotable_evidence: unknown
    top_use_cases: unknown
    top_integration_stacks: unknown
    budget_signal_summary: unknown
    sentiment_distribution: unknown
    buyer_authority_summary: unknown
    timeline_summary: unknown
    last_computed_at: string | null
  } | null
  review_counts: { total: number; enriched: number }
  high_intent_companies: { company: string; urgency: number; pain: string | null }[]
  pain_distribution: { pain_category: string; count: number }[]
}

export interface HighIntentLead {
  company: string
  vendor: string
  category: string | null
  role_level: string | null
  decision_maker: boolean | null
  urgency: number
  pain: string | null
  alternatives: unknown
  contract_signal: string | null
  seat_count: number | null
  contract_end: string | null
  buying_stage: string | null
}

export interface LeadDetail {
  company: string
  reviews: {
    id: string
    vendor_name: string
    category: string | null
    rating: number | null
    urgency: number
    pain: string | null
    intent_to_leave: boolean | null
    decision_maker: boolean | null
    role_level: string | null
    buying_stage: string | null
    alternatives: unknown
    contract_end: string | null
    enriched_at: string | null
  }[]
  count: number
}

export interface B2BReview {
  id: string
  vendor_name: string
  product_category: string | null
  reviewer_company: string | null
  rating: number | null
  urgency_score: number | null
  pain_category: string | null
  intent_to_leave: boolean | null
  decision_maker: boolean | null
  enriched_at: string | null
}

export interface B2BReviewDetail {
  id: string
  source: string | null
  source_url: string | null
  vendor_name: string
  product_name: string | null
  product_category: string | null
  rating: number | null
  summary: string | null
  review_text: string | null
  pros: string | null
  cons: string | null
  reviewer_name: string | null
  reviewer_title: string | null
  reviewer_company: string | null
  company_size_raw: string | null
  reviewer_industry: string | null
  reviewed_at: string | null
  enrichment: Record<string, unknown> | null
  enrichment_status: string | null
  enriched_at: string | null
}

export interface B2BReport {
  id: string
  report_date: string | null
  report_type: string
  executive_summary: string | null
  vendor_filter: string | null
  status: string | null
  created_at: string | null
}

export interface B2BReportDetail extends B2BReport {
  category_filter: string | null
  intelligence_data: unknown
  data_density: unknown
  llm_model: string | null
}

export interface B2BCampaign {
  id: string
  company_name: string
  vendor_name: string
  channel: string
  subject: string | null
  status: string
  approved_at: string | null
  sent_at: string | null
  created_at: string | null
}

export interface DashboardOverview {
  tracked_vendors: number
  avg_urgency: number
  total_churn_signals: number
  total_reviews: number
  recent_leads: { company: string; vendor: string; urgency: number; pain: string | null }[]
}

export interface PainTrend {
  week: string
  pain_category: string
  count: number
}

export interface DisplacementFlow {
  vendor_name: string
  competitors: unknown
  leaving: boolean | null
  mention_count: number
}

// -- Fetchers --

export function fetchOverview(): Promise<DashboardOverview> {
  return get('/overview')
}

export function fetchTrackedVendors(): Promise<{ vendors: TrackedVendor[]; count: number }> {
  return get('/vendors')
}

export function addTrackedVendor(vendor_name: string, track_mode: string = 'own', label: string = ''): Promise<TrackedVendor> {
  return post('/vendors', { vendor_name, track_mode, label })
}

export function removeTrackedVendor(vendor_name: string): Promise<{ status: string }> {
  return del(`/vendors/${encodeURIComponent(vendor_name)}`)
}

export function searchAvailableVendors(q: string): Promise<{ vendors: VendorSearchResult[]; count: number }> {
  return get('/vendors/search', { q })
}

export function fetchSignals(params?: {
  min_urgency?: number
  category?: string
  limit?: number
}): Promise<{ signals: ChurnSignal[]; count: number }> {
  return get('/signals', params)
}

export function fetchVendorDetail(vendor_name: string): Promise<VendorDetail> {
  return get(`/signals/${encodeURIComponent(vendor_name)}`)
}

export function fetchPainTrends(window_days?: number): Promise<{ trends: PainTrend[]; count: number }> {
  return get('/pain-trends', { window_days })
}

export function fetchDisplacement(limit?: number): Promise<{ displacement: DisplacementFlow[]; count: number }> {
  return get('/displacement', { limit })
}

export function fetchLeads(params?: {
  min_urgency?: number
  window_days?: number
  limit?: number
}): Promise<{ leads: HighIntentLead[]; count: number }> {
  return get('/leads', params)
}

export function fetchLeadDetail(company: string): Promise<LeadDetail> {
  return get(`/leads/${encodeURIComponent(company)}`)
}

export function fetchReports(params?: {
  report_type?: string
  limit?: number
}): Promise<{ reports: B2BReport[]; count: number }> {
  return get('/reports', params)
}

export function fetchReportDetail(id: string): Promise<B2BReportDetail> {
  return get(`/reports/${id}`)
}

export function fetchReviews(params?: {
  pain_category?: string
  min_urgency?: number
  company?: string
  has_churn_intent?: boolean
  window_days?: number
  limit?: number
}): Promise<{ reviews: B2BReview[]; count: number }> {
  return get('/reviews', params)
}

export function fetchReviewDetail(id: string): Promise<B2BReviewDetail> {
  return get(`/reviews/${id}`)
}

export function fetchCampaigns(params?: {
  status?: string
  limit?: number
}): Promise<{ campaigns: B2BCampaign[]; count: number }> {
  return get('/campaigns', params)
}

export function generateCampaigns(vendor_name: string, company_filter?: string): Promise<{ campaigns_created: number; message?: string }> {
  return post('/campaigns/generate', { vendor_name, company_filter })
}

export function updateCampaign(id: string, status: string): Promise<{ status: string; campaign_id: string; new_status: string }> {
  return patch(`/campaigns/${id}`, { status })
}
