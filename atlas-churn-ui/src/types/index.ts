export interface ChurnSignal {
  vendor_name: string
  product_category: string | null
  total_reviews: number
  churn_intent_count: number
  avg_urgency_score: number
  avg_rating_normalized: number | null
  nps_proxy: number | null
  price_complaint_rate: number | null
  decision_maker_churn_rate: number | null
  last_computed_at: string | null
}

export interface ChurnSignalDetail extends ChurnSignal {
  negative_reviews: number
  top_pain_categories: string[] | null
  top_competitors: string[] | null
  top_feature_gaps: string[] | null
  company_churn_list: string[] | null
  quotable_evidence: string[] | null
  top_use_cases: { module: string; mentions: number }[] | null
  top_integration_stacks: { tool: string; mentions: number }[] | null
  budget_signal_summary: Record<string, unknown> | null
  sentiment_distribution: Record<string, number> | null
  buyer_authority_summary: Record<string, unknown> | null
  timeline_summary: { company: string | null; contract_end: string | null; evaluation_deadline: string | null; decision_timeline: string | null; urgency: number }[] | null
  created_at: string | null
}

export interface HighIntentCompany {
  company: string
  vendor: string
  category: string | null
  role_level: string | null
  decision_maker: boolean | null
  urgency: number
  pain: string | null
  alternatives: string[] | null
  contract_signal: string | null
  seat_count: number | null
  lock_in_level: string | null
  contract_end: string | null
  buying_stage: string | null
}

export interface VendorProfile {
  vendor_name: string
  churn_signal: {
    avg_urgency_score: number
    churn_intent_count: number
    total_reviews: number
    nps_proxy: number | null
    price_complaint_rate: number | null
    decision_maker_churn_rate: number | null
    top_pain_categories: string[] | null
    top_competitors: string[] | null
    top_feature_gaps: string[] | null
    quotable_evidence: string[] | null
    top_use_cases: { module: string; mentions: number }[] | null
    top_integration_stacks: { tool: string; mentions: number }[] | null
    budget_signal_summary: Record<string, unknown> | null
    sentiment_distribution: Record<string, number> | null
    buyer_authority_summary: Record<string, unknown> | null
    timeline_summary: { company: string | null; contract_end: string | null; evaluation_deadline: string | null; decision_timeline: string | null; urgency: number }[] | null
    last_computed_at: string | null
  } | null
  review_counts: {
    total: number
    pending_enrichment: number
    enriched: number
  }
  high_intent_companies: {
    company: string
    urgency: number
    pain: string | null
  }[]
  pain_distribution: {
    pain_category: string
    count: number
  }[]
}

export interface Report {
  id: string
  report_date: string | null
  report_type: string
  executive_summary: string | null
  vendor_filter: string | null
  status: string | null
  created_at: string | null
}

export interface ReportDetail extends Report {
  category_filter: string | null
  intelligence_data: Record<string, unknown> | null
  data_density: Record<string, unknown> | null
  llm_model: string | null
}

export interface ReviewSummary {
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

export interface ReviewDetail {
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
  imported_at: string | null
  enrichment: Record<string, unknown> | null
  enrichment_status: string | null
  enriched_at: string | null
}

export interface PipelineStatus {
  enrichment_counts: Record<string, number>
  recent_imports_24h: number
  last_enrichment_at: string | null
  active_scrape_targets: number
  last_scrape_at: string | null
}
