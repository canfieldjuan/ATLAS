import type {
  ChurnSignal,
  ChurnSignalDetail,
  HighIntentCompany,
  VendorProfile,
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
} from '../types'

const BASE = '/api/v1/b2b/dashboard'

async function get<T>(path: string, params?: Record<string, string | number | boolean | undefined>): Promise<T> {
  const url = new URL(BASE + path, window.location.origin)
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null && v !== '') {
        url.searchParams.set(k, String(v))
      }
    }
  }
  const res = await fetch(url.toString())
  if (!res.ok) {
    const body = await res.text()
    throw new Error(`API ${res.status}: ${body}`)
  }
  return res.json()
}

export async function fetchSignals(params?: {
  vendor_name?: string
  min_urgency?: number
  category?: string
  limit?: number
}) {
  return get<{ signals: ChurnSignal[]; count: number }>('/signals', params)
}

export async function fetchSignal(vendorName: string, productCategory?: string) {
  return get<ChurnSignalDetail>(`/signals/${encodeURIComponent(vendorName)}`, {
    product_category: productCategory,
  })
}

export async function fetchHighIntent(params?: {
  vendor_name?: string
  min_urgency?: number
  window_days?: number
  limit?: number
}) {
  return get<{ companies: HighIntentCompany[]; count: number }>('/high-intent', params)
}

export async function fetchVendorProfile(vendorName: string) {
  return get<VendorProfile>(`/vendors/${encodeURIComponent(vendorName)}`)
}

export async function fetchReports(params?: {
  report_type?: string
  vendor_filter?: string
  limit?: number
}) {
  return get<{ reports: Report[]; count: number }>('/reports', params)
}

export async function fetchReport(reportId: string) {
  return get<ReportDetail>(`/reports/${reportId}`)
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
  return get<{ reviews: ReviewSummary[]; count: number }>('/reviews', params)
}

export async function fetchReview(reviewId: string) {
  return get<ReviewDetail>(`/reviews/${reviewId}`)
}

export async function fetchPipeline() {
  return get<PipelineStatus>('/pipeline')
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
  return get<{ opportunities: AffiliateOpportunity[]; count: number }>('/affiliates/opportunities', params)
}

export async function fetchAffiliatePartners() {
  return get<{ partners: AffiliatePartner[]; count: number }>('/affiliates/partners')
}

export async function fetchClickSummary() {
  return get<{ clicks: ClickSummary[] }>('/affiliates/clicks/summary')
}

export async function createAffiliatePartner(body: Omit<AffiliatePartner, 'id' | 'created_at' | 'updated_at'>) {
  const res = await fetch(BASE + '/affiliates/partners', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  return res.json()
}

export async function updateAffiliatePartner(id: string, body: Partial<AffiliatePartner>) {
  const res = await fetch(BASE + `/affiliates/partners/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  return res.json()
}

export async function deleteAffiliatePartner(id: string) {
  const res = await fetch(BASE + `/affiliates/partners/${id}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  return res.json()
}

export async function recordAffiliateClick(partnerId: string, reviewId?: string) {
  const res = await fetch(BASE + '/affiliates/clicks', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ partner_id: partnerId, review_id: reviewId, referrer: 'dashboard' }),
  })
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  return res.json()
}

// ---------------------------------------------------------------------------
// Campaigns
// ---------------------------------------------------------------------------

const CAMPAIGNS_BASE = '/api/v1/b2b/campaigns'

async function campaignGet<T>(path: string, params?: Record<string, string | number | boolean | undefined>): Promise<T> {
  const url = new URL(CAMPAIGNS_BASE + path, window.location.origin)
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null && v !== '') {
        url.searchParams.set(k, String(v))
      }
    }
  }
  const res = await fetch(url.toString())
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  return res.json()
}

export async function fetchCampaigns(params?: {
  status?: string
  company?: string
  vendor?: string
  channel?: string
  limit?: number
}) {
  return campaignGet<{ campaigns: Campaign[]; count: number }>('', params)
}

export async function fetchCampaign(id: string) {
  return campaignGet<Campaign>(`/${id}`)
}

export async function fetchCampaignStats() {
  return campaignGet<CampaignStats>('/stats')
}

export async function generateCampaigns(body?: {
  vendor_name?: string
  company_name?: string
  min_score?: number
  limit?: number
  target_mode?: string
}) {
  const res = await fetch(CAMPAIGNS_BASE + '/generate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body ?? {}),
  })
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  return res.json()
}

export async function approveCampaign(id: string) {
  const res = await fetch(CAMPAIGNS_BASE + `/${id}/approve`, { method: 'POST' })
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  return res.json()
}

export async function updateCampaign(id: string, body: Partial<Pick<Campaign, 'subject' | 'body' | 'cta' | 'status'>>) {
  const res = await fetch(CAMPAIGNS_BASE + `/${id}`, {
    method: 'PATCH',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  return res.json()
}

// ---------------------------------------------------------------------------
// Vendor Targets
// ---------------------------------------------------------------------------

const TARGETS_BASE = '/api/v1/b2b/vendor-targets'

async function targetGet<T>(path: string, params?: Record<string, string | number | boolean | undefined>): Promise<T> {
  const url = new URL(TARGETS_BASE + path, window.location.origin)
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null && v !== '') {
        url.searchParams.set(k, String(v))
      }
    }
  }
  const res = await fetch(url.toString())
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  return res.json()
}

export async function fetchVendorTargets(params?: {
  target_mode?: string
  status?: string
  search?: string
  limit?: number
}) {
  return targetGet<{ targets: VendorTarget[]; count: number }>('', params)
}

export async function fetchVendorTarget(id: string) {
  return targetGet<VendorTarget>(`/${id}`)
}

export async function createVendorTarget(body: Partial<VendorTarget>) {
  const res = await fetch(TARGETS_BASE, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  return res.json()
}

export async function updateVendorTarget(id: string, body: Partial<VendorTarget>) {
  const res = await fetch(TARGETS_BASE + `/${id}`, {
    method: 'PUT',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  return res.json()
}

export async function deleteVendorTarget(id: string) {
  const res = await fetch(TARGETS_BASE + `/${id}`, { method: 'DELETE' })
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  return res.json()
}

export async function generateVendorReport(id: string) {
  const res = await fetch(TARGETS_BASE + `/${id}/generate-report`, { method: 'POST' })
  if (!res.ok) throw new Error(`API ${res.status}: ${await res.text()}`)
  return res.json()
}
