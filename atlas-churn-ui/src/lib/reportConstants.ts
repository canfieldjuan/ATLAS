export const SPECIALIZED_REPORT_TYPES = [
  'challenger_brief',
  'accounts_in_motion',
  'battle_card',
  'vendor_comparison',
  'account_comparison',
  'weekly_churn_feed',
  'vendor_scorecard',
  'displacement_report',
  'category_overview',
  'vendor_deep_dive',
] as const

export function isSpecializedReportType(reportType: string): boolean {
  return SPECIALIZED_REPORT_TYPES.includes(reportType as (typeof SPECIALIZED_REPORT_TYPES)[number])
}

export const REPORT_SCALAR_KEYS = new Set([
  'vendor_name', 'challenger_name', 'primary_vendor', 'comparison_vendor', 'report_date', 'window_days',
  'signal_count', 'high_urgency_count', 'medium_urgency_count',
  'scope', 'llm_model', 'model_analysis', 'parse_fallback',
])

const FIELD_LABELS: Record<string, string> = {
  avg_urgency: 'Avg Urgency',
  budget_context: 'Budget Signals',
  churn_signal_density: 'Churn Signal Density',
  customer_pain_quotes: 'Customer Pain Quotes',
  decision_timeline: 'Decision Timeline',
  dm_churn_rate: 'DM Churn Rate',
  ecosystem_context: 'Ecosystem',
  key_insights: 'Key Insights',
  market_structure: 'Market Structure',
  objection_data: 'Objection Data',
  price_complaint_rate: 'Price Complaint Rate',
  sentiment_direction: 'Sentiment Direction',
  source_distribution: 'Source Distribution',
  top_feature_gaps: 'Feature Gaps',
  total_reviews: 'Total Reviews',
  vulnerability_window: 'Vulnerability Window',
  weakness_analysis: 'Weakness Analysis',
}

export function humanLabel(key: string): string {
  return FIELD_LABELS[key] ?? key.replace(/_/g, ' ').replace(/\b\w/g, (char) => char.toUpperCase())
}

export const REPORT_TYPE_COLORS: Record<string, string> = {
  weekly_churn_feed: 'bg-cyan-500/20 text-cyan-400',
  vendor_scorecard: 'bg-violet-500/20 text-violet-400',
  displacement_report: 'bg-amber-500/20 text-amber-400',
  category_overview: 'bg-emerald-500/20 text-emerald-400',
  exploratory_overview: 'bg-slate-500/20 text-slate-300',
  vendor_comparison: 'bg-fuchsia-500/20 text-fuchsia-300',
  account_comparison: 'bg-rose-500/20 text-rose-300',
  account_deep_dive: 'bg-pink-500/20 text-pink-300',
  vendor_retention: 'bg-orange-500/20 text-orange-400',
  challenger_intel: 'bg-purple-500/20 text-purple-400',
  challenger_brief: 'bg-purple-500/20 text-purple-400',
  battle_card: 'bg-red-500/20 text-red-400',
  vendor_deep_dive: 'bg-sky-500/20 text-sky-400',
}
