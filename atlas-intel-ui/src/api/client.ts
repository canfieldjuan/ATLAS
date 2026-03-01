const BASE = '/api/v1/consumer/dashboard'

async function get<T>(path: string, params?: Record<string, string | number | boolean | undefined>): Promise<T> {
  const url = new URL(`${BASE}${path}`, window.location.origin)
  if (params) {
    for (const [k, v] of Object.entries(params)) {
      if (v !== undefined && v !== null && v !== '') {
        url.searchParams.set(k, String(v))
      }
    }
  }
  const res = await fetch(url.toString())
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

export interface LoyaltyBreakdown {
  level: string
  count: number
}

export interface DistributionItem {
  level?: string
  type?: string
  channel?: string
  count: number
}

export interface BrandDetail {
  brand: string
  product_count: number
  total_reviews: number
  avg_rating: number | null
  products: BrandProduct[]
  sentiment_aspects: SentimentAspect[]
  top_features: FeatureRequest[]
  competitive_flows: CompetitiveFlow[]
  loyalty_breakdown: LoyaltyBreakdown[]
  buyer_profile: {
    expertise: DistributionItem[]
    budget: DistributionItem[]
    discovery_channel: DistributionItem[]
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

export function fetchPipeline() {
  return get<PipelineStatus>('/pipeline')
}

export function fetchBrands(params?: {
  source_category?: string
  min_reviews?: number
  search?: string
  sort_by?: string
  limit?: number
  offset?: number
}) {
  return get<{ brands: BrandSummary[]; count: number }>('/brands', params as Record<string, string | number>)
}

export function fetchBrandDetail(name: string) {
  return get<BrandDetail>(`/brands/${encodeURIComponent(name)}`)
}

export function fetchFlows(params?: {
  source_category?: string
  brand?: string
  min_count?: number
  limit?: number
}) {
  return get<{ flows: FlowEntry[]; count: number }>('/flows', params as Record<string, string | number>)
}

export function fetchFeatures(params?: {
  source_category?: string
  brand?: string
  limit?: number
}) {
  return get<{ feature_requests: FeatureGapEntry[]; negative_aspects: NegativeAspect[] }>('/features', params as Record<string, string | number>)
}

export function fetchSafety(params?: {
  source_category?: string
  brand?: string
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
  sort_by?: string
  limit?: number
  offset?: number
}) {
  return get<{ reviews: ReviewSummary[]; count: number }>('/reviews', params as Record<string, string | number | boolean>)
}

export function fetchReview(id: string) {
  return get<ReviewDetail>(`/reviews/${encodeURIComponent(id)}`)
}
