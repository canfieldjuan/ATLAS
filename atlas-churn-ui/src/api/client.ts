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
