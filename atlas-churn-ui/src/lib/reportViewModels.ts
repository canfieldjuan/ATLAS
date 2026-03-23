import type {
  ActiveEvaluationDeadlineViewModel,
  AccountsInMotionAccountViewModel,
  AccountsInMotionViewModel,
  BattleCardViewModel,
  CategoryCouncilViewModel,
  ChallengerAdvantageViewModel,
  ChallengerBriefViewModel,
  ChallengerBriefDisplacementViewModel,
  ChallengerIncumbentProfileViewModel,
  ChallengerSalesPlaybookViewModel,
  ChallengerTargetAccountViewModel,
  ComparisonFlowViewModel,
  ComparisonMetricSnapshotViewModel,
  ComparisonReportViewModel,
  CompetitorDifferentiatorViewModel,
  CrossVendorBattleViewModel,
  CrossVendorContextViewModel,
  CompetitiveLandscapeViewModel,
  FeatureGapViewModel,
  IntegrationComparisonViewModel,
  KeyInsightViewModel,
  ObjectionHandlerViewModel,
  PainQuoteViewModel,
  PricingPressureViewModel,
  RecommendedPlayViewModel,
  ResourceAsymmetryViewModel,
  SwitchingTriggerViewModel,
  TalkTrackViewModel,
  TrendAnalysisViewModel,
  VendorScorecardViewModel,
  WeeklyChurnFeedItemViewModel,
  WeaknessAnalysisItemViewModel,
} from '../types/reportViewModels'

type UnknownRecord = Record<string, unknown>

export function isRecord(value: unknown): value is UnknownRecord {
  return typeof value === 'object' && value !== null && !Array.isArray(value)
}

function asRecord(value: unknown): UnknownRecord {
  return isRecord(value) ? value : {}
}

function asString(value: unknown): string | undefined {
  return typeof value === 'string' && value.trim() ? value : undefined
}

function asNumber(value: unknown): number | null | undefined {
  return typeof value === 'number' && !Number.isNaN(value) ? value : undefined
}

function asBoolean(value: unknown): boolean | undefined {
  return typeof value === 'boolean' ? value : undefined
}

export function toStringArray(value: unknown): string[] {
  if (!Array.isArray(value)) return []
  return value.map((item) => asString(item)).filter((item): item is string => Boolean(item))
}

function toRecordArray(value: unknown): UnknownRecord[] {
  if (!Array.isArray(value)) return []
  return value.filter(isRecord)
}

export function toKeyInsights(value: unknown): KeyInsightViewModel[] {
  if (!Array.isArray(value)) return []
  return value.map((item) => {
    if (typeof item === 'string') return { insight: item, evidence: '' }
    const obj = asRecord(item)
    return {
      insight: asString(obj.insight ?? obj.summary ?? obj.name) ?? '',
      evidence: asString(obj.evidence ?? obj.metric ?? obj.detail) ?? '',
    }
  }).filter((item) => item.insight)
}

export function toPainQuotes(value: unknown): PainQuoteViewModel[] {
  return toRecordArray(value).map((item) => ({
    quote: asString(item.quote ?? item.text) ?? '',
    text: asString(item.text),
    company: asString(item.company),
    role: asString(item.role),
    title: asString(item.title),
    industry: asString(item.industry),
    source_site: asString(item.source_site),
    urgency: asNumber(item.urgency) ?? null,
    pain_category: asString(item.pain_category),
  })).filter((item) => item.quote)
}

export function toWeaknessAnalysis(value: unknown): WeaknessAnalysisItemViewModel[] {
  return toRecordArray(value).map((item) => ({
    weakness: asString(item.weakness),
    area: asString(item.area),
    name: asString(item.name),
    evidence: asString(item.evidence),
    customer_quote: asString(item.customer_quote),
    winning_position: asString(item.winning_position),
    recommendation: asString(item.recommendation),
    count: asNumber(item.count) ?? null,
    evidence_count: asNumber(item.evidence_count) ?? null,
  }))
}

export function toCompetitorDifferentiators(value: unknown): CompetitorDifferentiatorViewModel[] {
  return toRecordArray(value).map((item) => ({
    competitor: asString(item.competitor ?? item.name),
    mentions: asNumber(item.mentions) ?? null,
    count: asNumber(item.count) ?? null,
    primary_driver: asString(item.primary_driver),
    solves_weakness: asString(item.solves_weakness),
    switch_count: asNumber(item.switch_count) ?? null,
  }))
}

export function toObjectionHandlers(value: unknown): ObjectionHandlerViewModel[] {
  return toRecordArray(value).map((item) => ({
    objection: asString(item.objection),
    acknowledge: asString(item.acknowledge),
    pivot: asString(item.pivot),
    proof_point: asString(item.proof_point),
  }))
}

export function toRecommendedPlays(value: unknown): RecommendedPlayViewModel[] {
  return toRecordArray(value).map((item) => ({
    play: asString(item.play),
    name: asString(item.name),
    description: asString(item.description),
    target_segment: asString(item.target_segment),
    key_message: asString(item.key_message),
    timing: asString(item.timing),
  }))
}

export function toTalkTrack(value: unknown): TalkTrackViewModel | null {
  const obj = asRecord(value)
  if (Object.keys(obj).length === 0) return null
  return {
    opening: asString(obj.opening),
    mid_call_pivot: asString(obj.mid_call_pivot),
    proof_points: toStringArray(obj.proof_points),
    closing: asString(obj.closing),
  }
}

export function toCrossVendorBattles(value: unknown): CrossVendorBattleViewModel[] {
  return toRecordArray(value).map((item) => ({
    opponent: asString(item.opponent),
    winner: asString(item.winner),
    loser: asString(item.loser),
    durability: asString(item.durability),
    conclusion: asString(item.conclusion),
    confidence: asNumber(item.confidence) ?? null,
    key_insights: toKeyInsights(item.key_insights),
  }))
}

function toCompetitiveLandscape(value: unknown): CompetitiveLandscapeViewModel | null {
  const obj = asRecord(value)
  if (Object.keys(obj).length === 0) return null
  const topAlternatives = obj.top_alternatives
  return {
    vulnerability_window: asString(obj.vulnerability_window),
    top_alternatives: Array.isArray(topAlternatives)
      ? toStringArray(topAlternatives)
      : asString(topAlternatives)
        ? [asString(topAlternatives) as string]
        : [],
    displacement_triggers: toStringArray(obj.displacement_triggers),
  }
}

function toResourceAsymmetry(value: unknown): ResourceAsymmetryViewModel | null {
  const obj = asRecord(value)
  if (Object.keys(obj).length === 0) return null
  return {
    opponent: asString(obj.opponent),
    conclusion: asString(obj.conclusion),
    resource_advantage: asString(obj.resource_advantage),
    confidence: asNumber(obj.confidence) ?? null,
  }
}

function toCategoryCouncil(value: unknown): CategoryCouncilViewModel | null {
  const obj = asRecord(value)
  if (Object.keys(obj).length === 0) return null
  return {
    category: asString(obj.category),
    conclusion: asString(obj.conclusion),
    confidence: asNumber(obj.confidence) ?? null,
    key_insights: toKeyInsights(obj.key_insights),
  }
}

function toActiveEvaluationDeadlines(value: unknown): ActiveEvaluationDeadlineViewModel[] {
  return toRecordArray(value).map((item) => ({
    company: asString(item.company),
    decision_timeline: asString(item.decision_timeline),
    contract_end: asString(item.contract_end),
    evaluation_deadline: asString(item.evaluation_deadline),
    urgency: asNumber(item.urgency) ?? null,
  }))
}

function toDisplacementSummary(value: unknown): ChallengerBriefDisplacementViewModel {
  const obj = asRecord(value)
  return {
    total_mentions: asNumber(obj.total_mentions) ?? null,
    signal_strength: asString(obj.signal_strength),
    confidence_score: asNumber(obj.confidence_score) ?? null,
    primary_driver: asString(obj.primary_driver),
    source_distribution: Object.fromEntries(
      Object.entries(asRecord(obj.source_distribution)).filter(([, count]) => typeof count === 'number'),
    ) as Record<string, number>,
    key_quote: asString(obj.key_quote),
  }
}

function toIncumbentProfile(value: unknown): ChallengerIncumbentProfileViewModel {
  const obj = asRecord(value)
  return {
    archetype: asString(obj.archetype),
    archetype_confidence: asNumber(obj.archetype_confidence) ?? null,
    risk_level: asString(obj.risk_level),
    churn_pressure_score: asNumber(obj.churn_pressure_score) ?? null,
    price_complaint_rate: asNumber(obj.price_complaint_rate) ?? null,
    dm_churn_rate: asNumber(obj.dm_churn_rate) ?? null,
    sentiment_direction: asString(obj.sentiment_direction),
    key_signals: toStringArray(obj.key_signals),
    top_weaknesses: toWeaknessAnalysis(obj.top_weaknesses),
    top_pain_quotes: toPainQuotes(obj.top_pain_quotes),
  }
}

function toChallengerAdvantage(value: unknown): ChallengerAdvantageViewModel {
  const obj = asRecord(value)
  return {
    profile_summary: asString(obj.profile_summary),
    strengths: toRecordArray(obj.strengths).map((item) => ({
      area: asString(item.area),
      name: asString(item.name),
      evidence_count: asNumber(item.evidence_count) ?? null,
      mentions: asNumber(item.mentions) ?? null,
    })),
    weakness_coverage: toRecordArray(obj.weakness_coverage).map((item) => ({
      incumbent_weakness: asString(item.incumbent_weakness),
      match_quality: asString(item.match_quality),
    })),
    commonly_switched_from: toRecordArray(obj.commonly_switched_from).map((item) => asString(item.vendor)).filter((item): item is string => Boolean(item)).concat(
      toStringArray(obj.commonly_switched_from),
    ),
  }
}

function toTargetAccounts(value: unknown): ChallengerTargetAccountViewModel[] {
  return toRecordArray(value).map((item) => ({
    company: asString(item.company),
    opportunity_score: asNumber(item.opportunity_score) ?? null,
    buying_stage: asString(item.buying_stage),
    urgency: asNumber(item.urgency) ?? null,
    industry: asString(item.industry),
    considers_challenger: asBoolean(item.considers_challenger),
  }))
}

function toSalesPlaybook(value: unknown): ChallengerSalesPlaybookViewModel {
  const obj = asRecord(value)
  return {
    discovery_questions: toStringArray(obj.discovery_questions),
    landmine_questions: toStringArray(obj.landmine_questions),
    objection_handlers: toObjectionHandlers(obj.objection_handlers),
    talk_track: toTalkTrack(obj.talk_track),
    recommended_plays: toRecommendedPlays(obj.recommended_plays),
  }
}

function toIntegrationComparison(value: unknown): IntegrationComparisonViewModel {
  const obj = asRecord(value)
  return {
    shared: toStringArray(obj.shared),
    challenger_exclusive: toStringArray(obj.challenger_exclusive),
    incumbent_exclusive: toStringArray(obj.incumbent_exclusive),
  }
}

export function toChallengerBriefViewModel(value: UnknownRecord): ChallengerBriefViewModel {
  const headToHead = toCrossVendorBattles([value.head_to_head])[0]
  return {
    incumbent: asString(value.incumbent),
    challenger: asString(value.challenger),
    total_target_accounts: asNumber(value.total_target_accounts) ?? null,
    accounts_considering_challenger: asNumber(value.accounts_considering_challenger) ?? null,
    displacement_summary: toDisplacementSummary(value.displacement_summary),
    incumbent_profile: toIncumbentProfile(value.incumbent_profile),
    challenger_advantage: toChallengerAdvantage(value.challenger_advantage),
    head_to_head: {
      ...(headToHead ?? { key_insights: [] }),
      synthesized: asBoolean(asRecord(value.head_to_head).synthesized),
    },
    target_accounts: toTargetAccounts(value.target_accounts),
    sales_playbook: toSalesPlaybook(value.sales_playbook),
    integration_comparison: toIntegrationComparison(value.integration_comparison),
    data_sources: Object.fromEntries(
      Object.entries(asRecord(value.data_sources)).filter(([, flag]) => typeof flag === 'boolean'),
    ) as Record<string, boolean>,
  }
}

function toPricingPressure(value: unknown): PricingPressureViewModel {
  const obj = asRecord(value)
  return {
    price_complaint_rate: asNumber(obj.price_complaint_rate) ?? null,
    price_increase_rate: asNumber(obj.price_increase_rate) ?? null,
    avg_seat_count: asNumber(obj.avg_seat_count) ?? null,
  }
}

function toFeatureGaps(value: unknown): FeatureGapViewModel[] {
  return toRecordArray(value).map((item) => ({
    feature: asString(item.feature),
    category: asString(item.category),
    mentions: asNumber(item.mentions) ?? null,
    count: asNumber(item.count) ?? null,
  }))
}

function toCrossVendorContext(value: unknown): CrossVendorContextViewModel {
  const obj = asRecord(value)
  return {
    top_destination: asString(obj.top_destination),
    battle_conclusion: asString(obj.battle_conclusion),
    market_regime: asString(obj.market_regime),
  }
}

function toMotionAccounts(value: unknown): AccountsInMotionAccountViewModel[] {
  return toRecordArray(value).map((item) => ({
    company: asString(item.company),
    opportunity_score: asNumber(item.opportunity_score) ?? null,
    buying_stage: asString(item.buying_stage),
    urgency: asNumber(item.urgency) ?? null,
    pain_category: asString(item.pain_category),
    industry: asString(item.industry),
    domain: asString(item.domain),
    alternatives_considering: toStringArray(item.alternatives_considering),
    top_quote: asString(item.top_quote),
  }))
}

function toComparisonMetricSnapshot(value: unknown): ComparisonMetricSnapshotViewModel {
  const obj = asRecord(value)
  return {
    vendor_name: asString(obj.vendor_name),
    signal_count: asNumber(obj.signal_count) ?? null,
    churn_signal_density: asNumber(obj.churn_signal_density) ?? null,
    avg_urgency_score: asNumber(obj.avg_urgency_score) ?? null,
    positive_review_pct: asNumber(obj.positive_review_pct) ?? null,
    recommend_ratio: asNumber(obj.recommend_ratio) ?? null,
    churn_intent_count: asNumber(obj.churn_intent_count) ?? null,
  }
}

function toSwitchingTriggers(value: unknown): SwitchingTriggerViewModel[] {
  return toRecordArray(value).map((item) => ({
    competitor: asString(item.competitor),
    primary_reason: asString(item.primary_reason),
    mention_count: asNumber(item.mention_count) ?? null,
  }))
}

function toTrendAnalysis(value: unknown): TrendAnalysisViewModel | null {
  const obj = asRecord(value)
  if (Object.keys(obj).length === 0) return null
  return {
    prior_report_date: asString(obj.prior_report_date),
    primary_churn_density_change: asNumber(obj.primary_churn_density_change) ?? null,
    comparison_churn_density_change: asNumber(obj.comparison_churn_density_change) ?? null,
    primary_urgency_change: asNumber(obj.primary_urgency_change) ?? null,
    comparison_urgency_change: asNumber(obj.comparison_urgency_change) ?? null,
  }
}

export function toWeeklyChurnFeedItems(value: unknown): WeeklyChurnFeedItemViewModel[] {
  const items = Array.isArray(value)
    ? value
    : asRecord(value).weekly_churn_feed
  return toRecordArray(items).map((item) => ({
    vendor: asString(item.vendor),
    category: asString(item.category),
    total_reviews: asNumber(item.total_reviews) ?? null,
    churn_signal_density: asNumber(item.churn_signal_density) ?? null,
    avg_urgency: asNumber(item.avg_urgency) ?? null,
    sample_size_confidence: asString(item.sample_size_confidence),
    churn_pressure_score: asNumber(item.churn_pressure_score) ?? null,
    top_pain: asString(item.top_pain),
    pain_breakdown: toFeatureGaps(item.pain_breakdown),
    top_feature_gaps: toStringArray(item.top_feature_gaps),
    dm_churn_rate: asNumber(item.dm_churn_rate) ?? null,
    price_complaint_rate: asNumber(item.price_complaint_rate) ?? null,
    dominant_buyer_role: asString(item.dominant_buyer_role),
    top_displacement_targets: toCompetitorDifferentiators(item.top_displacement_targets),
    key_quote: asString(item.key_quote),
    evidence: toStringArray(item.evidence),
    sentiment_direction: asString(item.sentiment_direction),
    trend: asString(item.trend),
    action_recommendation: asString(item.action_recommendation),
    named_accounts: toRecordArray(item.named_accounts).map((account) => ({
      company: asString(account.company),
      urgency: asNumber(account.urgency) ?? null,
    })),
  }))
}

export function toVendorScorecards(value: unknown): VendorScorecardViewModel[] {
  const obj = asRecord(value)
  const items = Array.isArray(value)
    ? value
    : Array.isArray(obj.vendor_scorecards)
      ? obj.vendor_scorecards
      : Object.keys(obj).length > 0
        ? [obj]
        : []
  return toRecordArray(items).map((item) => ({
    vendor: asString(item.vendor),
    total_reviews: asNumber(item.total_reviews ?? item.review_count) ?? null,
    churn_signal_density: asNumber(item.churn_signal_density) ?? null,
    positive_review_pct: asNumber(item.positive_review_pct) ?? null,
    avg_urgency: asNumber(item.avg_urgency) ?? null,
    recommend_ratio: asNumber(item.recommend_ratio) ?? null,
    sample_size_confidence: asString(item.sample_size_confidence),
    top_pain: asString(item.top_pain),
    top_competitor_threat: asString(item.top_competitor_threat),
    trend: asString(item.trend),
    sentiment_direction: asString(item.sentiment_direction),
  }))
}

function toComparisonFlows(value: unknown): ComparisonFlowViewModel[] {
  return toRecordArray(value).map((item) => ({
    name: asString(item.name),
    count: asNumber(item.count) ?? null,
    companies: toStringArray(item.companies),
  }))
}

export function toBattleCardViewModel(value: UnknownRecord): BattleCardViewModel {
  return {
    vendor: asString(value.vendor),
    category: asString(value.category),
    churn_pressure_score: asNumber(value.churn_pressure_score) ?? null,
    total_reviews: asNumber(value.total_reviews) ?? null,
    confidence: asString(value.confidence),
    archetype: asString(value.archetype),
    archetype_risk_level: asString(value.archetype_risk_level),
    archetype_key_signals: toStringArray(value.archetype_key_signals),
    vendor_weaknesses: toWeaknessAnalysis(value.vendor_weaknesses),
    weakness_analysis: toWeaknessAnalysis(value.weakness_analysis),
    customer_pain_quotes: toPainQuotes(value.customer_pain_quotes),
    competitor_differentiators: toCompetitorDifferentiators(value.competitor_differentiators),
    cross_vendor_battles: toCrossVendorBattles(value.cross_vendor_battles),
    competitive_landscape: toCompetitiveLandscape(value.competitive_landscape),
    resource_asymmetry: toResourceAsymmetry(value.resource_asymmetry),
    category_council: toCategoryCouncil(value.category_council),
    objection_handlers: toObjectionHandlers(value.objection_handlers),
    talk_track: toTalkTrack(value.talk_track),
    recommended_plays: toRecommendedPlays(value.recommended_plays),
    active_evaluation_deadlines: toActiveEvaluationDeadlines(value.active_evaluation_deadlines),
    source_distribution: Object.fromEntries(
      Object.entries(asRecord(value.source_distribution)).filter(([, count]) => typeof count === 'number'),
    ) as Record<string, number>,
    llm_render_status: asString(value.llm_render_status),
    quality_status: asString(value.quality_status ?? asRecord(value.battle_card_quality).status),
    quality_score: asNumber(asRecord(value.battle_card_quality).score) ?? null,
  }
}

export function toComparisonReportViewModel(value: UnknownRecord): ComparisonReportViewModel {
  const primaryName = asString(value.primary_vendor ?? value.primary_company)
  const comparisonName = asString(value.comparison_vendor ?? value.comparison_company)
  const vendorArchetypes = Object.fromEntries(
    Object.entries(asRecord(value.vendor_archetypes)).map(([vendor, details]) => {
      const item = asRecord(details)
      return [vendor, {
        archetype: asString(item.archetype),
        confidence: asNumber(item.confidence) ?? null,
      }]
    }),
  ) as Record<string, { archetype?: string; confidence?: number | null }>

  return {
    primary_name: primaryName,
    comparison_name: comparisonName,
    primary_metrics: toComparisonMetricSnapshot(
      value.primary_metrics ?? value.primary_company_metrics ?? value.primary_vendor,
    ),
    comparison_metrics: toComparisonMetricSnapshot(
      value.comparison_metrics ?? value.comparison_company_metrics ?? value.comparison_vendor,
    ),
    primary_top_pains: toFeatureGaps(value.primary_top_pains),
    comparison_top_pains: toFeatureGaps(value.comparison_top_pains),
    primary_quote_highlights: toStringArray(value.primary_quote_highlights),
    comparison_quote_highlights: toStringArray(value.comparison_quote_highlights),
    primary_strengths: toWeaknessAnalysis(value.primary_strengths),
    primary_weaknesses: toWeaknessAnalysis(value.primary_weaknesses),
    comparison_strengths: toWeaknessAnalysis(value.comparison_strengths),
    comparison_weaknesses: toWeaknessAnalysis(value.comparison_weaknesses),
    primary_switching_triggers: toSwitchingTriggers(value.primary_switching_triggers),
    comparison_switching_triggers: toSwitchingTriggers(value.comparison_switching_triggers),
    shared_pain_categories: toStringArray(value.shared_pain_categories),
    shared_alternatives: toStringArray(value.shared_alternatives),
    shared_vendors: toStringArray(value.shared_vendors),
    direct_displacement: toComparisonFlows(value.direct_displacement),
    trend_analysis: toTrendAnalysis(value.trend_analysis),
    urgency_gap: asNumber(value.urgency_gap) ?? null,
    vendor_archetypes: vendorArchetypes,
  }
}

export function toAccountsInMotionViewModel(value: UnknownRecord): AccountsInMotionViewModel {
  return {
    total_accounts_in_motion: asNumber(value.total_accounts_in_motion) ?? null,
    archetype: asString(value.archetype),
    archetype_confidence: asNumber(value.archetype_confidence) ?? null,
    pricing_pressure: toPricingPressure(value.pricing_pressure),
    feature_gaps: toFeatureGaps(value.feature_gaps),
    cross_vendor_context: toCrossVendorContext(value.cross_vendor_context),
    accounts: toMotionAccounts(value.accounts),
  }
}
