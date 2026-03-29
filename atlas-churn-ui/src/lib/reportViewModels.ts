import type {
  AccountPressureMetricsViewModel,
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
  TimingMetricsViewModel,
  TrendAnalysisViewModel,
  VendorDeepDiveViewModel,
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
    market_regime: asString(obj.market_regime),
    winner: asString(obj.winner),
    loser: asString(obj.loser),
    durability: asString(obj.durability),
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
    buying_stage: asString(item.buying_stage),
    role: asString(item.role),
    pain: asString(item.pain),
    source: asString(item.source),
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
    decision_maker: item.decision_maker === true,
    confidence: asNumber(item.confidence) ?? null,
    quality_flags: toStringArray(item.quality_flags),
    contact_name: asString(item.contact_name),
    contact_title: asString(item.contact_title) || asString(item.title),
  }))
}

function toAccountPressureMetrics(value: unknown): AccountPressureMetricsViewModel | undefined {
  const obj = asRecord(value)
  if (Object.keys(obj).length === 0) return undefined
  return {
    total_accounts: asNumber(obj.total_accounts) ?? null,
    high_intent_count: asNumber(obj.high_intent_count) ?? null,
    active_eval_count: asNumber(obj.active_eval_count) ?? null,
  }
}

function toTimingMetrics(value: unknown): TimingMetricsViewModel | undefined {
  const obj = asRecord(value)
  if (Object.keys(obj).length === 0) return undefined
  return {
    immediate_trigger_count: asNumber(obj.immediate_trigger_count) ?? null,
    active_eval_signals: asNumber(obj.active_eval_signals) ?? null,
    sentiment_direction: asString(obj.sentiment_direction),
  }
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
      title: asString(account.title),
      buying_stage: asString(account.buying_stage),
      company_size: asString(account.company_size),
      source: asString(account.source),
      decision_maker: account.decision_maker === true,
      confidence_score: asNumber(account.confidence_score) ?? null,
    })),
    category_council: toCategoryCouncil(item.category_council),
    retention_strengths: toRecordArray(item.retention_strengths).map((s) => ({
      area: asString(s.area),
      mention_count: asNumber(s.mention_count) ?? null,
    })),
    account_pressure_summary: asString(item.account_pressure_summary),
    timing_summary: asString(item.timing_summary),
    priority_timing_triggers: toStringArray(item.priority_timing_triggers),
  }))
}

export function toVendorDeepDives(value: unknown): VendorDeepDiveViewModel[] {
  return toRecordArray(value).map((item) => ({
    vendor: asString(item.vendor),
    category: asString(item.category),
    total_reviews: asNumber(item.total_reviews) ?? null,
    churn_signal_density: asNumber(item.churn_signal_density) ?? null,
    churn_pressure_score: asNumber(item.churn_pressure_score) ?? null,
    avg_urgency: asNumber(item.avg_urgency) ?? null,
    risk_level: asString(item.risk_level),
    sentiment_direction: asString(item.sentiment_direction),
    trend: asString(item.trend),
    archetype: asString(item.archetype),
    archetype_confidence: asNumber(item.archetype_confidence) ?? null,
    dm_churn_rate: asNumber(item.dm_churn_rate) ?? null,
    price_complaint_rate: asNumber(item.price_complaint_rate) ?? null,
    dominant_buyer_role: asString(item.dominant_buyer_role),
    pain_breakdown: toRecordArray(item.pain_breakdown).map((p) => ({
      category: asString(p.category),
      count: asNumber(p.count) ?? null,
      pct: asNumber(p.pct) ?? null,
    })),
    displacement_targets: toRecordArray(item.displacement_targets).map((d) => ({
      vendor: asString(d.vendor),
      mention_count: asNumber(d.mention_count) ?? null,
      primary_driver: asString(d.primary_driver),
    })),
    feature_gaps: toRecordArray(item.feature_gaps).map((f) => ({
      feature: asString(f.feature),
      mentions: asNumber(f.mentions) ?? null,
    })),
    industry_distribution: toRecordArray(item.industry_distribution).map((i) => ({
      industry: asString(i.industry),
      count: asNumber(i.count) ?? null,
    })),
    company_size_distribution: toRecordArray(item.company_size_distribution).map((s) => ({
      size: asString(s.size),
      count: asNumber(s.count) ?? null,
    })),
    case_studies: toRecordArray(item.case_studies).map((c) => ({
      quote: asString(c.quote),
      company: asString(c.company),
      urgency: asNumber(c.urgency) ?? null,
      title: asString(c.title),
    })),
    sentiment_breakdown: (() => {
      const sb = asRecord(item.sentiment_breakdown)
      return Object.keys(sb).length > 0 ? {
        positive: asNumber(sb.positive) ?? null,
        negative: asNumber(sb.negative) ?? null,
        neutral: asNumber(sb.neutral) ?? null,
      } : undefined
    })(),
    retention_strengths: toRecordArray(item.retention_strengths).map((s) => ({
      area: asString(s.area),
      mention_count: asNumber(s.mention_count) ?? null,
    })),
    category_council: toCategoryCouncil(item.category_council),
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
  const timingIntel = asRecord(value.timing_intelligence)
  const accountReasoning = asRecord(value.account_reasoning)
  const segmentPlaybook = asRecord(value.segment_playbook)
  const objectionData = asRecord(value.objection_data)

  // active_evaluation_deadlines: prefer dedicated field, fall back to high_intent_companies
  const evalDeadlines = Array.isArray(value.active_evaluation_deadlines) && (value.active_evaluation_deadlines as unknown[]).length > 0
    ? toActiveEvaluationDeadlines(value.active_evaluation_deadlines)
    : toActiveEvaluationDeadlines(value.high_intent_companies)

  return {
    vendor: asString(value.vendor),
    category: asString(value.category),
    churn_pressure_score: asNumber(value.churn_pressure_score) ?? null,
    total_reviews: asNumber(value.total_reviews) ?? null,
    confidence: asString(value.confidence),
    archetype: asString(value.archetype),
    archetype_risk_level: asString(value.archetype_risk_level),
    archetype_key_signals: toStringArray(value.archetype_key_signals),
    executive_summary: asString(value.executive_summary),
    timing_summary: asString(value.timing_summary),
    account_pressure_summary: asString(value.account_pressure_summary),
    vendor_weaknesses: toWeaknessAnalysis(value.vendor_weaknesses),
    weakness_analysis: toWeaknessAnalysis(value.weakness_analysis),
    customer_pain_quotes: toPainQuotes(value.customer_pain_quotes),
    competitor_differentiators: toCompetitorDifferentiators(value.competitor_differentiators),
    cross_vendor_battles: toCrossVendorBattles(value.cross_vendor_battles),
    competitive_landscape: toCompetitiveLandscape(value.competitive_landscape),
    resource_asymmetry: toResourceAsymmetry(value.resource_asymmetry),
    category_council: toCategoryCouncil(value.category_council),
    objection_handlers: toObjectionHandlers(value.objection_handlers),
    objection_metrics: Object.keys(objectionData).length > 0 ? {
      avg_urgency: asNumber(objectionData.avg_urgency) ?? null,
      dm_churn_rate: asNumber(objectionData.dm_churn_rate) ?? null,
      budget_context: asString(objectionData.budget_context),
      price_complaint_rate: asNumber(objectionData.price_complaint_rate) ?? null,
      churn_signal_density: asNumber(objectionData.churn_signal_density) ?? null,
      recommend_ratio: asNumber(objectionData.recommend_ratio) ?? null,
    } : null,
    talk_track: toTalkTrack(value.talk_track),
    recommended_plays: toRecommendedPlays(value.recommended_plays),
    active_evaluation_deadlines: evalDeadlines,
    segment_targets: toRecordArray(segmentPlaybook.priority_segments ?? value.segment_playbook).map((item) => ({
      segment: asString(item.segment),
      why_vulnerable: asString(item.why_vulnerable),
      best_opening_angle: asString(item.best_opening_angle),
      disqualifier: asString(item.disqualifier),
      estimated_reach: asString(item.estimated_reach),
    })),
    timing_window: asString(timingIntel.best_timing_window),
    timing_triggers: toRecordArray(timingIntel.immediate_triggers).map((item) => ({
      trigger: asString(item.trigger),
      action: asString(item.action),
      urgency: asString(item.urgency),
    })),
    account_market_summary: asString(accountReasoning.market_summary),
    landmine_questions: toStringArray(value.landmine_questions),
    discovery_questions: toStringArray(value.discovery_questions),
    retention_signals: toRecordArray(value.retention_signals).map((item) => ({
      aspect: asString(item.aspect),
      mentions: asNumber(item.mentions) ?? null,
    })),
    incumbent_strengths: toRecordArray(value.incumbent_strengths).map((item) => ({
      area: asString(item.area),
      source: asString(item.source),
      mention_count: asNumber(item.mention_count) ?? null,
    })),
    evidence_depth_warning: asString(value.evidence_depth_warning),
    evidence_conclusions: toStringArray(value.evidence_conclusions),
    low_confidence_sections: toStringArray(value.low_confidence_sections),
    falsification_conditions: toStringArray(value.falsification_conditions),
    uncertainty_sources: toStringArray(value.uncertainty_sources),
    account_pressure_metrics: (() => {
      const m = asRecord(value.account_pressure_metrics)
      return Object.keys(m).length > 0 ? {
        total_accounts: asNumber(m.total_accounts) ?? null,
        high_intent_count: asNumber(m.high_intent_count) ?? null,
        active_eval_count: asNumber(m.active_eval_count) ?? null,
      } : undefined
    })(),
    buyer_authority: (() => {
      const ba = asRecord(value.buyer_authority)
      return Object.keys(ba).length > 0 ? ba : undefined
    })(),
    integration_stack: toStringArray(value.integration_stack),
    keyword_spikes: (() => {
      const ks = asRecord(value.keyword_spikes)
      return Object.keys(ks).length > 0 ? {
        spike_count: asNumber(ks.spike_count) ?? null,
        keywords: toStringArray(ks.keywords),
      } : undefined
    })(),
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
    account_pressure_summary: asString(value.account_pressure_summary),
    account_pressure_metrics: toAccountPressureMetrics(value.account_pressure_metrics),
    priority_account_names: toStringArray(value.priority_account_names),
    timing_summary: asString(value.timing_summary),
    timing_metrics: toTimingMetrics(value.timing_metrics),
    priority_timing_triggers: toStringArray(value.priority_timing_triggers),
    segment_targeting_summary: asString(value.segment_targeting_summary),
    category_council: toCategoryCouncil(value.category_council) ?? undefined,
  }
}
