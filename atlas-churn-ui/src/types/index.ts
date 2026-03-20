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
  support_sentiment: number | null
  legacy_support_score: number | null
  new_feature_velocity: number | null
  employee_growth_rate: number | null
  archetype: string | null
  archetype_confidence: number | null
  reasoning_mode: string | null
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
  falsification_conditions: string[] | null
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
  alternatives: { name: string; context?: string; reason?: string }[] | null
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
    archetype: string | null
    archetype_confidence: number | null
    reasoning_mode: string | null
    falsification_conditions: string[] | null
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

export interface VendorHistorySnapshot {
  snapshot_date: string
  total_reviews: number
  churn_intent: number
  churn_density: number
  avg_urgency: number
  positive_review_pct: number | null
  recommend_ratio: number | null
  support_sentiment: number | null
  legacy_support_score: number | null
  new_feature_velocity: number | null
  employee_growth_rate: number | null
  top_pain: string | null
  top_competitor: string | null
  pain_count: number | null
  competitor_count: number | null
  displacement_edge_count: number | null
  high_intent_company_count: number | null
}

export interface VendorHistoryResponse {
  vendor_name: string
  snapshots: VendorHistorySnapshot[]
  count: number
}

export interface VendorPeriodComparisonResponse {
  vendor_name: string
  period_a: VendorHistorySnapshot | null
  period_b: VendorHistorySnapshot | null
  deltas: Record<string, number>
}

export interface Report {
  id: string
  report_date: string | null
  report_type: string
  executive_summary: string | null
  vendor_filter: string | null
  category_filter?: string | null
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
  source: string | null
  reviewed_at: string | null
  role_level: string | null
  buying_stage: string | null
  sentiment_direction: string | null
  competitors_mentioned: (string | { name: string; context?: string; reason?: string })[]
  quotable_phrases: string[]
  positive_aspects: string[]
  specific_complaints: string[]
  enriched_at: string | null
  reviewer_title: string | null
  company_size: string | null
  industry: string | null
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

export interface AffiliatePartner {
  id: string
  name: string
  product_name: string
  product_aliases: string[]
  category: string | null
  affiliate_url: string
  commission_type: string
  commission_value: string | null
  notes: string | null
  enabled: boolean
  created_at: string
  updated_at: string
}

export interface AffiliateOpportunity {
  review_id: string
  vendor_name: string
  reviewer_company: string | null
  product_category: string | null
  urgency: number
  is_dm: boolean | null
  role_type: string | null
  buying_stage: string | null
  seat_count: number | null
  contract_end: string | null
  decision_timeline: string | null
  competitor_name: string
  mention_context: string | null
  mention_reason: string | null
  partner_id: string
  partner_name: string
  affiliate_url: string
  commission_type: string
  commission_value: string | null
  partner_category: string | null
  opportunity_score: number
}

export interface ClickSummary {
  id: string
  name: string
  product_name: string
  click_count: number
}

export type CampaignStatus = 'draft' | 'approved' | 'sent' | 'expired'

export interface Campaign {
  id: string
  company_name: string
  vendor_name: string
  product_category: string | null
  opportunity_score: number | null
  urgency_score: number | null
  channel: string
  subject: string | null
  body: string | null
  cta: string | null
  status: CampaignStatus
  batch_id: string | null
  llm_model: string | null
  created_at: string | null
  approved_at: string | null
  sent_at: string | null
}

export interface CampaignStats {
  by_status: Record<string, number>
  by_channel: Record<string, number>
  top_vendors: { vendor_name: string; count: number }[]
  total: number
}

// ---------------------------------------------------------------------------
// Blog Drafts
// ---------------------------------------------------------------------------

export interface BlogDraftSummary {
  id: string
  slug: string
  title: string
  topic_type: string
  status: string
  llm_model: string | null
  created_at: string | null
  published_at: string | null
}

export interface BlogDraft extends BlogDraftSummary {
  description?: string | null
  tags: string[]
  content: string
  charts: unknown[]
  data_context: Record<string, unknown> | null
  reviewer_notes: string | null
  source_report_date: string | null
}

export interface BlogEvidence {
  id: string
  vendor_name: string
  reviewer_company: string | null
  headline: string | null
  full_text: string | null
  pain_categories: string[]
  urgency_score: number | null
  source_site: string | null
  review_date: string | null
}

// ---------------------------------------------------------------------------
// Prospects
// ---------------------------------------------------------------------------

export interface Prospect {
  id: string
  first_name: string | null
  last_name: string | null
  email: string | null
  email_status: string | null
  title: string | null
  seniority: string | null
  department: string | null
  company_name: string | null
  company_domain: string | null
  linkedin_url: string | null
  city: string | null
  state: string | null
  country: string | null
  status: string
  created_at: string
  updated_at: string
}

export interface ProspectStats {
  total: number
  active: number
  contacted: number
  this_month: number
}

// ---------------------------------------------------------------------------
// Campaign Review (enhanced)
// ---------------------------------------------------------------------------

export interface ReviewQueueDraft {
  id: string
  company_name: string
  vendor_name: string
  channel: string
  subject: string | null
  body: string | null
  cta: string | null
  status: string
  step_number: number | null
  recipient_email: string | null
  seq_recipient: string | null
  partner_name: string | null
  product_name: string | null
  is_suppressed: number
  seq_status: string | null
  current_step: number | null
  max_steps: number | null
  open_count: number | null
  click_count: number | null
  target_persona: string | null
  prospect_first_name: string | null
  prospect_last_name: string | null
  prospect_title: string | null
  prospect_seniority: string | null
  prospect_email_status: string | null
  created_at: string | null
}

export interface AuditEvent {
  id: string
  event_type: string
  source: string | null
  campaign_id: string | null
  sequence_id: string | null
  step_number: number | null
  recipient_email: string | null
  subject: string | null
  metadata: Record<string, unknown> | null
  created_at: string | null
}

// ---------------------------------------------------------------------------
// Briefing Review (HITL)
// ---------------------------------------------------------------------------

export type BriefingStatus = 'pending_approval' | 'sent' | 'opened' | 'clicked' | 'bounced' | 'failed' | 'suppressed' | 'rejected'

export interface BriefingDraft {
  id: string
  vendor_name: string
  recipient_email: string
  subject: string | null
  briefing_html: string | null
  status: BriefingStatus
  target_mode: string | null
  created_at: string | null
  approved_at: string | null
  rejected_at: string | null
  reject_reason: string | null
}

export type TargetMode = 'vendor_retention' | 'challenger_intel'
export type TargetTier = 'report' | 'dashboard' | 'api'
export type VendorTargetOwnershipScope = 'owned' | 'legacy_global' | 'account_owned'

export interface VendorTarget {
  id: string
  company_name: string
  target_mode: TargetMode
  contact_name: string | null
  contact_email: string | null
  contact_role: string | null
  products_tracked: string[] | null
  competitors_tracked: string[] | null
  tier: TargetTier
  status: string
  notes: string | null
  account_id?: string | null
  ownership_scope?: VendorTargetOwnershipScope
  created_at: string
  updated_at: string
  campaign_stats?: {
    total_campaigns: number
    drafts: number
    sent: number
    approved: number
    last_campaign_at: string | null
  }
  recent_reports?: {
    id: string
    report_date: string | null
    report_type: string
    executive_summary: string | null
    created_at: string | null
  }[]
}
