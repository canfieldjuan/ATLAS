import type { SuppressionReason, VendorClaim } from '../api/client'

export type HeadToHeadReadinessState =
  | 'report_safe'
  | 'monitor_only'
  | 'suppressed'
  | 'validation_unavailable'

export type BattleCardDisplacementReadinessState =
  | 'report_safe'
  | 'monitor_only'
  | 'suppressed'
  | 'validation_unavailable'

// Backend produces canonical SuppressionReason values plus two non-canonical
// strings: 'validation_unavailable' (when claim_rows is None) and
// 'not_report_safe' (when a render-safe claim has no canonical reason).
// See atlas_brain/autonomous/tasks/b2b_battle_cards.py:_battle_card_displacement_gate_payload.
export type BattleCardDisplacementSuppressionReason =
  | SuppressionReason
  | 'validation_unavailable'
  | 'not_report_safe'

export interface BattleCardDisplacementSwitchVolumeViewModel {
  value?: number | null
}

export interface BattleCardDisplacementMigrationProofViewModel {
  readiness_state: BattleCardDisplacementReadinessState
  render_allowed: boolean
  report_allowed: boolean
  suppression_reason: BattleCardDisplacementSuppressionReason | null
  gate_message?: string
  confidence?: string
  switching_is_real?: boolean
  top_destination?: string
  switch_volume?: BattleCardDisplacementSwitchVolumeViewModel | null
  product_claims: VendorClaim[]
}

export interface BattleCardDisplacementCustomerWinningPatternViewModel {
  readiness_state: BattleCardDisplacementReadinessState
  render_allowed: boolean
  report_allowed: boolean
  suppression_reason: BattleCardDisplacementSuppressionReason | null
  gate_message?: string
  confidence?: string
  summary?: string
  product_claims: VendorClaim[]
}

export interface BattleCardDisplacementReasoningSectionGateViewModel {
  readiness_state: BattleCardDisplacementReadinessState
  render_allowed: boolean
  report_allowed: boolean
  suppression_reason: BattleCardDisplacementSuppressionReason | null
  product_claims: VendorClaim[]
}

export interface BattleCardDisplacementReasoningViewModel {
  product_claim_gate: BattleCardDisplacementReasoningSectionGateViewModel
  migration_proof: BattleCardDisplacementMigrationProofViewModel | null
  customer_winning_pattern: BattleCardDisplacementCustomerWinningPatternViewModel | null
}

export interface KeyInsightViewModel {
  insight: string
  evidence: string
}

export interface ReasoningWitnessViewModel {
  witness_id?: string
  _sid?: string
  reviewer_company?: string
  reviewer_title?: string
  excerpt_text?: string
  time_anchor?: string
  competitor?: string
  witness_type?: string
  selection_reason?: string
  salience_score?: number | null
  numeric_literals?: Record<string, unknown>
  grounding_status?: string
  phrase_polarity?: string
  phrase_subject?: string
  phrase_role?: string
  phrase_verbatim?: boolean
  pain_confidence?: string
}

export type ReasoningAnchorExamplesViewModel = Record<string, ReasoningWitnessViewModel[]>

export interface ReasoningReferenceIdsViewModel {
  metric_ids: string[]
  witness_ids: string[]
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
  product_claim?: VendorClaim | null
  claim_validation_unavailable?: boolean
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
  product_claim?: VendorClaim | null
  claim_validation_unavailable?: boolean
}

export interface RecommendedPlayViewModel {
  play?: string
  name?: string
  description?: string
  target_segment?: string
  key_message?: string
  timing?: string
  product_claim?: VendorClaim | null
  claim_validation_unavailable?: boolean
}

export interface CompetitorDifferentiatorViewModel {
  competitor?: string
  mentions?: number | null
  count?: number | null
  primary_driver?: string
  solves_weakness?: string
  switch_count?: number | null
  product_claim?: VendorClaim | null
  claim_validation_unavailable?: boolean
}

export interface CrossVendorBattleViewModel {
  opponent?: string
  winner?: string
  loser?: string
  durability?: string
  conclusion?: string
  confidence?: number | null
  key_insights: KeyInsightViewModel[]
  reference_ids?: ReasoningReferenceIdsViewModel
  product_claim?: VendorClaim | null
  claim_validation_unavailable?: boolean
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
  reasoning_anchor_examples?: ReasoningAnchorExamplesViewModel
  reasoning_witness_highlights?: ReasoningWitnessViewModel[]
  reasoning_reference_ids?: ReasoningReferenceIdsViewModel
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
  product_claim?: VendorClaim | null
  claim_validation_unavailable?: boolean
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
  head_to_head: CrossVendorBattleViewModel & {
    synthesized?: boolean
    product_claim?: VendorClaim | null
    readiness_state?: HeadToHeadReadinessState
    claim_validation_unavailable?: boolean
    suppression_reason?: SuppressionReason | null
  }
  target_accounts: ChallengerTargetAccountViewModel[]
  sales_playbook: ChallengerSalesPlaybookViewModel
  integration_comparison: IntegrationComparisonViewModel
  data_sources: Record<string, boolean>
  reasoning_anchor_examples?: ReasoningAnchorExamplesViewModel
  reasoning_witness_highlights?: ReasoningWitnessViewModel[]
  reasoning_source?: string
  reasoning_reference_ids?: ReasoningReferenceIdsViewModel
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
  market_regime?: string
  winner?: string
  loser?: string
  durability?: string
  key_insights: KeyInsightViewModel[]
  reference_ids?: ReasoningReferenceIdsViewModel
  product_claim?: VendorClaim | null
  claim_validation_unavailable?: boolean
}

export interface ActiveEvaluationDeadlineViewModel {
  company?: string
  decision_timeline?: string
  contract_end?: string
  evaluation_deadline?: string
  urgency?: number | null
  buying_stage?: string
  role?: string
  pain?: string
  source?: string
}

export interface SegmentPlaybookItemViewModel {
  segment?: string
  why_vulnerable?: string
  best_opening_angle?: string
  disqualifier?: string
  estimated_reach?: string
}

export interface TimingTriggerViewModel {
  trigger?: string
  action?: string
  urgency?: string
}

export interface RetentionSignalViewModel {
  aspect?: string
  mentions?: number | null
}

export interface IncumbentStrengthViewModel {
  area?: string
  source?: string
  mention_count?: number | null
}

export interface ObjectionMetricsViewModel {
  avg_urgency?: number | null
  dm_churn_rate?: number | null
  budget_context?: string
  price_complaint_rate?: number | null
  churn_signal_density?: number | null
  recommend_ratio?: number | null
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
  executive_summary?: string
  timing_summary?: string
  account_pressure_summary?: string
  account_pressure_disclaimer?: string
  account_actionability_tier?: string
  vendor_weaknesses: WeaknessAnalysisItemViewModel[]
  weakness_analysis: WeaknessAnalysisItemViewModel[]
  customer_pain_quotes: PainQuoteViewModel[]
  competitor_differentiators: CompetitorDifferentiatorViewModel[]
  cross_vendor_battles: CrossVendorBattleViewModel[]
  competitive_landscape: CompetitiveLandscapeViewModel | null
  resource_asymmetry: ResourceAsymmetryViewModel | null
  category_council: CategoryCouncilViewModel | null
  objection_handlers: ObjectionHandlerViewModel[]
  objection_metrics: ObjectionMetricsViewModel | null
  talk_track: TalkTrackViewModel | null
  recommended_plays: RecommendedPlayViewModel[]
  active_evaluation_deadlines: ActiveEvaluationDeadlineViewModel[]
  segment_targets: SegmentPlaybookItemViewModel[]
  timing_window?: string
  timing_triggers: TimingTriggerViewModel[]
  account_market_summary?: string
  landmine_questions: string[]
  discovery_questions: string[]
  retention_signals: RetentionSignalViewModel[]
  incumbent_strengths: IncumbentStrengthViewModel[]
  // Trust/calibration
  evidence_depth_warning?: string
  evidence_conclusions: string[]
  low_confidence_sections: string[]
  reasoning_section_disclaimers?: Record<string, string>
  falsification_conditions: string[]
  uncertainty_sources: string[]
  // Operational signals
  account_pressure_metrics?: AccountPressureMetricsViewModel
  buyer_authority?: Record<string, unknown>
  integration_stack: string[]
  keyword_spikes?: KeywordSpikesViewModel
  source_distribution: Record<string, number>
  llm_render_status?: string
  quality_status?: string
  quality_score?: number | null
  quality_failed_checks: string[]
  quality_warnings: string[]
  reasoning_source?: string
  reasoning_reference_ids?: ReasoningReferenceIdsViewModel
  reasoning_witness_highlights?: ReasoningWitnessViewModel[]
  displacement_reasoning?: BattleCardDisplacementReasoningViewModel | null
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
  title?: string
  buying_stage?: string
  company_size?: string
  source?: string
  decision_maker?: boolean
  confidence_score?: number | null
}

export interface FeedRetentionStrengthViewModel {
  area?: string
  mention_count?: number | null
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
  category_council?: CategoryCouncilViewModel | null
  retention_strengths: FeedRetentionStrengthViewModel[]
  account_pressure_summary?: string
  account_pressure_disclaimer?: string
  account_actionability_tier?: string
  timing_summary?: string
  priority_timing_triggers: string[]
  reasoning_source?: string
  reasoning_reference_ids?: ReasoningReferenceIdsViewModel
}

export interface DeepDivePainBreakdownViewModel {
  category?: string
  count?: number | null
  pct?: number | null
}

export interface DeepDiveDisplacementTargetViewModel {
  vendor?: string
  mention_count?: number | null
  primary_driver?: string
}

export interface DeepDiveFeatureGapViewModel {
  feature?: string
  mentions?: number | null
}

export interface DeepDiveCaseStudyViewModel {
  quote?: string
  company?: string
  urgency?: number | null
  title?: string
}

export interface VendorDeepDiveViewModel {
  vendor?: string
  category?: string
  total_reviews?: number | null
  churn_signal_density?: number | null
  churn_pressure_score?: number | null
  avg_urgency?: number | null
  risk_level?: string
  sentiment_direction?: string
  trend?: string
  archetype?: string
  archetype_confidence?: number | null
  dm_churn_rate?: number | null
  price_complaint_rate?: number | null
  dominant_buyer_role?: string
  pain_breakdown: DeepDivePainBreakdownViewModel[]
  displacement_targets: DeepDiveDisplacementTargetViewModel[]
  feature_gaps: DeepDiveFeatureGapViewModel[]
  industry_distribution: Array<{ industry?: string; count?: number | null }>
  company_size_distribution: Array<{ size?: string; count?: number | null }>
  case_studies: DeepDiveCaseStudyViewModel[]
  sentiment_breakdown?: { positive?: number | null; negative?: number | null; neutral?: number | null }
  retention_strengths?: Array<{ area?: string; mention_count?: number | null }>
  category_council?: CategoryCouncilViewModel | null
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
  decision_maker?: boolean
  confidence?: number | null
  quality_flags?: string[]
  contact_name?: string
  contact_title?: string
}

export interface AccountPressureMetricsViewModel {
  total_accounts?: number | null
  high_intent_count?: number | null
  active_eval_count?: number | null
}

export interface KeywordSpikesViewModel {
  spike_count?: number | null
  keywords: string[]
}

export interface TimingMetricsViewModel {
  immediate_trigger_count?: number | null
  active_eval_signals?: number | null
  sentiment_direction?: string
  [key: string]: unknown
}

export interface AccountsInMotionViewModel {
  total_accounts_in_motion?: number | null
  archetype?: string
  archetype_confidence?: number | null
  pricing_pressure: PricingPressureViewModel
  feature_gaps: FeatureGapViewModel[]
  cross_vendor_context: CrossVendorContextViewModel
  accounts: AccountsInMotionAccountViewModel[]
  account_pressure_summary?: string
  account_pressure_disclaimer?: string
  account_actionability_tier?: string
  account_pressure_metrics?: AccountPressureMetricsViewModel
  priority_account_names?: string[]
  timing_summary?: string
  timing_metrics?: TimingMetricsViewModel
  priority_timing_triggers?: string[]
  segment_targeting_summary?: string
  category_council?: CategoryCouncilViewModel
  reasoning_source?: string
  reasoning_reference_ids?: ReasoningReferenceIdsViewModel
}
