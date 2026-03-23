export interface KeyInsightViewModel {
  insight: string
  evidence: string
}

export interface PainQuoteViewModel {
  quote: string
  text?: string
  company?: string
  role?: string
  title?: string
  industry?: string
  source_site?: string
  urgency?: number | null
  pain_category?: string
}

export interface ObjectionHandlerViewModel {
  objection?: string
  acknowledge?: string
  pivot?: string
  proof_point?: string
}

export interface TalkTrackViewModel {
  opening?: string
  mid_call_pivot?: string
  proof_points?: string[]
  closing?: string
}

export interface WeaknessAnalysisItemViewModel {
  weakness?: string
  area?: string
  name?: string
  evidence?: string
  customer_quote?: string
  winning_position?: string
  recommendation?: string
  count?: number | null
  evidence_count?: number | null
}

export interface RecommendedPlayViewModel {
  play?: string
  name?: string
  description?: string
  target_segment?: string
  key_message?: string
  timing?: string
}

export interface CompetitorDifferentiatorViewModel {
  competitor?: string
  mentions?: number | null
  count?: number | null
  primary_driver?: string
  solves_weakness?: string
  switch_count?: number | null
}

export interface CrossVendorBattleViewModel {
  opponent?: string
  winner?: string
  loser?: string
  durability?: string
  conclusion?: string
  confidence?: number | null
  key_insights: KeyInsightViewModel[]
}

export interface ChallengerBriefDisplacementViewModel {
  total_mentions?: number | null
  signal_strength?: string
  confidence_score?: number | null
  primary_driver?: string
  source_distribution: Record<string, number>
  key_quote?: string
}

export interface ChallengerIncumbentProfileViewModel {
  archetype?: string
  archetype_confidence?: number | null
  risk_level?: string
  churn_pressure_score?: number | null
  price_complaint_rate?: number | null
  dm_churn_rate?: number | null
  sentiment_direction?: string
  key_signals: string[]
  top_weaknesses: WeaknessAnalysisItemViewModel[]
  top_pain_quotes: PainQuoteViewModel[]
}

export interface ChallengerStrengthViewModel {
  area?: string
  name?: string
  evidence_count?: number | null
  mentions?: number | null
}

export interface ChallengerWeaknessCoverageViewModel {
  incumbent_weakness?: string
  match_quality?: string
}

export interface ChallengerAdvantageViewModel {
  profile_summary?: string
  strengths: ChallengerStrengthViewModel[]
  weakness_coverage: ChallengerWeaknessCoverageViewModel[]
  commonly_switched_from: string[]
}

export interface ChallengerTargetAccountViewModel {
  company?: string
  opportunity_score?: number | null
  buying_stage?: string
  urgency?: number | null
  industry?: string
  considers_challenger?: boolean
}

export interface ChallengerSalesPlaybookViewModel {
  discovery_questions: string[]
  landmine_questions: string[]
  objection_handlers: ObjectionHandlerViewModel[]
  talk_track: TalkTrackViewModel | null
  recommended_plays: RecommendedPlayViewModel[]
}

export interface IntegrationComparisonViewModel {
  shared: string[]
  challenger_exclusive: string[]
  incumbent_exclusive: string[]
}

export interface ChallengerBriefViewModel {
  incumbent?: string
  challenger?: string
  total_target_accounts?: number | null
  accounts_considering_challenger?: number | null
  displacement_summary: ChallengerBriefDisplacementViewModel
  incumbent_profile: ChallengerIncumbentProfileViewModel
  challenger_advantage: ChallengerAdvantageViewModel
  head_to_head: CrossVendorBattleViewModel & { synthesized?: boolean }
  target_accounts: ChallengerTargetAccountViewModel[]
  sales_playbook: ChallengerSalesPlaybookViewModel
  integration_comparison: IntegrationComparisonViewModel
  data_sources: Record<string, boolean>
}

export interface PricingPressureViewModel {
  price_complaint_rate?: number | null
  price_increase_rate?: number | null
  avg_seat_count?: number | null
}

export interface FeatureGapViewModel {
  feature?: string
  category?: string
  mentions?: number | null
  count?: number | null
}

export interface CrossVendorContextViewModel {
  top_destination?: string
  battle_conclusion?: string
  market_regime?: string
}

export interface CompetitiveLandscapeViewModel {
  vulnerability_window?: string
  top_alternatives: string[]
  displacement_triggers: string[]
}

export interface ResourceAsymmetryViewModel {
  opponent?: string
  conclusion?: string
  resource_advantage?: string
  confidence?: number | null
}

export interface CategoryCouncilViewModel {
  category?: string
  conclusion?: string
  confidence?: number | null
  key_insights: KeyInsightViewModel[]
}

export interface ActiveEvaluationDeadlineViewModel {
  company?: string
  decision_timeline?: string
  contract_end?: string
  evaluation_deadline?: string
  urgency?: number | null
}

export interface BattleCardViewModel {
  vendor?: string
  category?: string
  churn_pressure_score?: number | null
  total_reviews?: number | null
  confidence?: string
  archetype?: string
  archetype_risk_level?: string
  archetype_key_signals: string[]
  vendor_weaknesses: WeaknessAnalysisItemViewModel[]
  weakness_analysis: WeaknessAnalysisItemViewModel[]
  customer_pain_quotes: PainQuoteViewModel[]
  competitor_differentiators: CompetitorDifferentiatorViewModel[]
  cross_vendor_battles: CrossVendorBattleViewModel[]
  competitive_landscape: CompetitiveLandscapeViewModel | null
  resource_asymmetry: ResourceAsymmetryViewModel | null
  category_council: CategoryCouncilViewModel | null
  objection_handlers: ObjectionHandlerViewModel[]
  talk_track: TalkTrackViewModel | null
  recommended_plays: RecommendedPlayViewModel[]
  active_evaluation_deadlines: ActiveEvaluationDeadlineViewModel[]
  source_distribution: Record<string, number>
  llm_render_status?: string
  quality_status?: string
  quality_score?: number | null
}

export interface ComparisonMetricSnapshotViewModel {
  vendor_name?: string
  signal_count?: number | null
  churn_signal_density?: number | null
  avg_urgency_score?: number | null
  positive_review_pct?: number | null
  recommend_ratio?: number | null
  churn_intent_count?: number | null
}

export interface TrendAnalysisViewModel {
  prior_report_date?: string
  primary_churn_density_change?: number | null
  comparison_churn_density_change?: number | null
  primary_urgency_change?: number | null
  comparison_urgency_change?: number | null
}

export interface SwitchingTriggerViewModel {
  competitor?: string
  primary_reason?: string
  mention_count?: number | null
}

export interface ComparisonFlowViewModel {
  name?: string
  count?: number | null
  companies: string[]
}

export interface ComparisonReportViewModel {
  primary_name?: string
  comparison_name?: string
  primary_metrics: ComparisonMetricSnapshotViewModel
  comparison_metrics: ComparisonMetricSnapshotViewModel
  primary_top_pains: FeatureGapViewModel[]
  comparison_top_pains: FeatureGapViewModel[]
  primary_quote_highlights: string[]
  comparison_quote_highlights: string[]
  primary_strengths: WeaknessAnalysisItemViewModel[]
  primary_weaknesses: WeaknessAnalysisItemViewModel[]
  comparison_strengths: WeaknessAnalysisItemViewModel[]
  comparison_weaknesses: WeaknessAnalysisItemViewModel[]
  primary_switching_triggers: SwitchingTriggerViewModel[]
  comparison_switching_triggers: SwitchingTriggerViewModel[]
  shared_pain_categories: string[]
  shared_alternatives: string[]
  shared_vendors: string[]
  direct_displacement: ComparisonFlowViewModel[]
  trend_analysis: TrendAnalysisViewModel | null
  urgency_gap?: number | null
  vendor_archetypes: Record<string, { archetype?: string; confidence?: number | null }>
}

export interface FeedNamedAccountViewModel {
  company?: string
  urgency?: number | null
}

export interface WeeklyChurnFeedItemViewModel {
  vendor?: string
  category?: string
  total_reviews?: number | null
  churn_signal_density?: number | null
  avg_urgency?: number | null
  sample_size_confidence?: string
  churn_pressure_score?: number | null
  top_pain?: string
  pain_breakdown: FeatureGapViewModel[]
  top_feature_gaps: string[]
  dm_churn_rate?: number | null
  price_complaint_rate?: number | null
  dominant_buyer_role?: string
  top_displacement_targets: CompetitorDifferentiatorViewModel[]
  key_quote?: string
  evidence: string[]
  sentiment_direction?: string
  trend?: string
  action_recommendation?: string
  named_accounts: FeedNamedAccountViewModel[]
}

export interface VendorScorecardViewModel {
  vendor?: string
  total_reviews?: number | null
  churn_signal_density?: number | null
  positive_review_pct?: number | null
  avg_urgency?: number | null
  recommend_ratio?: number | null
  sample_size_confidence?: string
  top_pain?: string
  top_competitor_threat?: string
  trend?: string
  sentiment_direction?: string
}

export interface AccountsInMotionAccountViewModel {
  company?: string
  opportunity_score?: number | null
  buying_stage?: string
  urgency?: number | null
  pain_category?: string
  industry?: string
  domain?: string
  alternatives_considering: string[]
  top_quote?: string
}

export interface AccountsInMotionViewModel {
  total_accounts_in_motion?: number | null
  archetype?: string
  archetype_confidence?: number | null
  pricing_pressure: PricingPressureViewModel
  feature_gaps: FeatureGapViewModel[]
  cross_vendor_context: CrossVendorContextViewModel
  accounts: AccountsInMotionAccountViewModel[]
}
