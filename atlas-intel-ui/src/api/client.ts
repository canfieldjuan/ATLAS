const BASE = '/api/v1/consumer/dashboard'

function authHeaders(): Record<string, string> {
  const token = localStorage.getItem('atlas_token')
  return token ? { Authorization: `Bearer ${token}` } : {}
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
  const res = await fetch(url.toString(), { headers: authHeaders() })
  if (res.status === 401) {
    localStorage.removeItem('atlas_token')
    localStorage.removeItem('atlas_refresh_token')
    window.location.href = '/login'
    throw new Error('Session expired')
  }
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`API ${res.status}: ${body || res.statusText}`)
  }
  return res.json()
}

async function post<T>(path: string, body?: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json', ...authHeaders() },
    body: body ? JSON.stringify(body) : undefined,
  })
  if (res.status === 401) {
    localStorage.removeItem('atlas_token')
    localStorage.removeItem('atlas_refresh_token')
    window.location.href = '/login'
    throw new Error('Session expired')
  }
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`API ${res.status}: ${body || res.statusText}`)
  }
  return res.json()
}

async function del<T>(path: string): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'DELETE',
    headers: authHeaders(),
  })
  if (!res.ok) {
    const body = await res.text().catch(() => '')
    throw new Error(`API ${res.status}: ${body || res.statusText}`)
  }
  return res.json()
}

// -- Types --

export interface PipelineStatus {
  enrichment_counts: Record<string, number>
  deep_enrichment_counts: Record<string, number>
  category_counts: Record<string, number>
  total_reviews: number
  enriched: number
  deep_enriched: number
  targeted_for_deep: number
  total_brands: number
  total_asins: number
  last_enrichment_at: string | null
  last_deep_enrichment_at: string | null
}

export interface BrandSummary {
  brand: string
  product_count: number
  review_count: number
  avg_rating: number | null
  total_ratings: number | null
  avg_complaint_score: number | null
  avg_praise_score: number | null
  complaint_count: number
  praise_count: number
  safety_count: number
  brand_health: number | null
}

export interface BrandProduct {
  asin: string
  title: string
  average_rating: number | null
  rating_number: number
  price: string | null
  review_count: number
  avg_complaint_score: number | null
  avg_praise_score: number | null
  complaint_count: number
  praise_count: number
}

export interface SentimentAspect {
  aspect: string
  positive: number
  negative: number
  mixed: number
  neutral: number
}

export interface FeatureRequest {
  request: string
  count: number
}

export interface CompetitiveFlow {
  brand: string
  direction: string
  count: number
  avg_rating: number | null
}

export interface LabelCount {
  label: string
  count: number
}

export interface FailureAnalysis {
  failure_count: number
  top_failure_modes: { mode: string; count: number }[]
  top_failed_components: { component: string; count: number }[]
  avg_dollar_lost: number | null
  total_dollar_lost: number | null
}

export interface ConsiderationEntry {
  product: string
  count: number
  top_reason: string | null
}

export interface PositiveAspect {
  aspect: string
  count: number
}

export interface BrandDetail {
  brand: string
  product_count: number
  total_reviews: number
  deep_review_count: number
  avg_rating: number | null
  brand_health: number | null
  products: BrandProduct[]
  sentiment_aspects: SentimentAspect[]
  top_features: FeatureRequest[]
  competitive_flows: CompetitiveFlow[]
  top_positives: PositiveAspect[]
  consideration_set: ConsiderationEntry[]
  // Churn signals
  loyalty_breakdown: LabelCount[]
  repurchase_breakdown: LabelCount[]
  replacement_breakdown: LabelCount[]
  trajectory_breakdown: LabelCount[]
  switching_barrier: LabelCount[]
  // Failure analysis
  failure_analysis: FailureAnalysis
  // Buyer psychology
  buyer_profile: {
    expertise: LabelCount[]
    budget: LabelCount[]
    discovery_channel: LabelCount[]
    frustration: LabelCount[]
    intensity: LabelCount[]
    research_depth: LabelCount[]
    occasion: LabelCount[]
    household: LabelCount[]
    buyer_type: LabelCount[]
    price_sentiment: LabelCount[]
    professions: { profession: string; count: number }[]
  }
  // Engagement signals
  consequence_breakdown: LabelCount[]
  delay_breakdown: LabelCount[]
  ecosystem_lock_in: LabelCount[]
  amplification_intent: LabelCount[]
  openness_breakdown: LabelCount[]
  safety_flagged_count: number
  // First-pass enrichment
  first_pass: {
    enriched_count: number
    severity_breakdown: LabelCount[]
    time_to_failure: LabelCount[]
    workaround_rate: number | null
    workaround_count: number
    top_root_causes: { cause: string; count: number }[]
    top_manufacturing_suggestions: { suggestion: string; count: number }[]
    top_alternatives_mentioned: { product: string; count: number }[]
  }
}

// -- Brand Comparison Types --

export interface BrandCompareMetrics {
  product_count: number
  total_reviews: number
  deep_review_count: number
  avg_rating: number | null
  brand_health: number | null
  repurchase_pct: number | null
  safety_flagged_count: number
  failure_count: number
  avg_dollar_lost: number | null
  replacement_breakdown: LabelCount[]
  trajectory_breakdown: LabelCount[]
  switching_barrier: LabelCount[]
  consequence_breakdown: LabelCount[]
  severity_breakdown: LabelCount[]
  workaround_rate: number | null
}

export interface CrossBrandFlow {
  from_brand: string
  to_brand: string
  direction: string
  count: number
  avg_rating: number | null
}

export interface SharedFeatureRequest {
  request: string
  brands: string[]
  total_count: number
}

export interface ConsiderationOverlap {
  product: string
  mentioned_by_brands: string[]
  total_count: number
}

export interface BrandComparison {
  brands: string[]
  per_brand: Record<string, BrandCompareMetrics>
  cross_brand: {
    competitive_flows: CrossBrandFlow[]
    shared_feature_requests: SharedFeatureRequest[]
    consideration_overlap: ConsiderationOverlap[]
  }
}

export interface FlowEntry {
  from_brand: string
  to_brand: string
  direction: string
  count: number
  avg_rating: number | null
}

export interface FeatureGapEntry {
  request: string
  count: number
  brands_affected: number
  brand_list: string[]
  avg_rating: number | null
}

export interface NegativeAspect {
  aspect: string
  negative: number
  total: number
  pct_negative: number
  top_brands: string[]
}

export interface SafetySignal {
  id: string
  asin: string
  rating: number | null
  summary: string | null
  review_excerpt: string | null
  safety_flag: Record<string, unknown> | null
  brand: string
  title: string
  imported_at: string | null
}

export interface ReviewSummary {
  id: string
  asin: string
  brand: string | null
  title: string | null
  rating: number | null
  root_cause: string | null
  pain_score: number | null
  severity: string | null
  summary: string | null
  source_category: string | null
  enrichment_status: string | null
  deep_enrichment_status: string | null
  imported_at: string | null
}

export interface ReviewDetail {
  id: string
  asin: string
  source_category: string | null
  rating: number | null
  summary: string | null
  review_text: string | null
  reviewer_id: string | null
  imported_at: string | null
  enrichment_status: string | null
  root_cause: string | null
  severity: string | null
  pain_score: number | null
  time_to_failure: string | null
  workaround_found: boolean | null
  workaround_text: string | null
  alternative_mentioned: boolean | null
  alternative_name: string | null
  alternative_asin: string | null
  deep_enrichment_status: string | null
  deep_enrichment: Record<string, unknown> | null
  deep_enriched_at: string | null
  brand: string | null
  product_title: string | null
  product_avg_rating: number | null
  product_total_ratings: number | null
  product_price: string | null
}

// -- Fetchers --

export function fetchCategories() {
  return get<{ categories: string[] }>('/categories')
}

export function fetchPipeline(params?: { source_category?: string }) {
  return get<PipelineStatus>('/pipeline', params as Record<string, string>)
}

export function fetchBrands(params?: {
  source_category?: string
  min_reviews?: number
  search?: string
  sort_by?: string
  limit?: number
  offset?: number
}) {
  return get<{ brands: BrandSummary[]; count: number; total_count: number }>('/brands', params as Record<string, string | number>)
}

export function fetchBrandDetail(name: string) {
  return get<BrandDetail>(`/brands/${encodeURIComponent(name)}`)
}

export function fetchBrandComparison(brands: string[]) {
  return get<BrandComparison>('/brands/compare', { brands: brands.join(',') })
}

export function fetchFlows(params?: {
  source_category?: string
  brand?: string
  direction?: string
  min_count?: number
  limit?: number
}) {
  return get<{ flows: FlowEntry[]; count: number }>('/flows', params as Record<string, string | number>)
}

export function fetchFeatures(params?: {
  source_category?: string
  brand?: string
  min_count?: number
  limit?: number
}) {
  return get<{ feature_requests: FeatureGapEntry[]; negative_aspects: NegativeAspect[] }>('/features', params as Record<string, string | number>)
}

export function fetchSafety(params?: {
  source_category?: string
  brand?: string
  min_rating?: number
  max_rating?: number
  limit?: number
}) {
  return get<{ signals: SafetySignal[]; count: number; total_flagged: number }>('/safety', params as Record<string, string | number>)
}

export function fetchReviews(params?: {
  source_category?: string
  brand?: string
  asin?: string
  min_rating?: number
  max_rating?: number
  root_cause?: string
  has_comparisons?: boolean
  has_feature_requests?: boolean
  search?: string
  severity?: string
  enrichment_status?: string
  imported_after?: string
  imported_before?: string
  sort_by?: string
  limit?: number
  offset?: number
}) {
  return get<{ reviews: ReviewSummary[]; count: number; total_count: number }>('/reviews', params as Record<string, string | number | boolean>)
}

export function fetchReview(id: string) {
  return get<ReviewDetail>(`/reviews/${encodeURIComponent(id)}`)
}

// -- ASIN Tracking --

export interface TrackedAsin {
  asin: string
  label: string | null
  added_at: string | null
  title: string | null
  brand: string | null
  average_rating: number | null
  rating_number: number | null
  price: string | null
}

export interface AsinSearchResult {
  asin: string
  title: string | null
  brand: string | null
  average_rating: number | null
  rating_number: number | null
  price: string | null
}

export function fetchTrackedAsins() {
  return get<{ asins: TrackedAsin[]; count: number }>('/asins')
}

export function addTrackedAsin(asin: string, label?: string) {
  return post<{ status: string; asin: string }>('/asins', { asin, label })
}

export function removeTrackedAsin(asin: string) {
  return del<{ status: string }>(`/asins/${encodeURIComponent(asin)}`)
}

export function searchAvailableAsins(q: string, limit = 20) {
  return get<{ results: AsinSearchResult[] }>('/asins/search', { q, limit })
}
