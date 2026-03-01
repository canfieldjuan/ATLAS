// Deep enrichment field types for all 32 extraction fields

// --- Section A: Product Analysis ---

export interface SentimentAspect {
  aspect: string
  sentiment: 'positive' | 'negative' | 'mixed' | 'neutral'
  detail: string
}

export interface FailureDetails {
  timeline: string | null
  failed_component: string | null
  failure_mode: string | null
  dollar_amount_lost: number | null
}

export interface ProductComparison {
  product_name: string
  direction: string
  context: string
}

export interface BuyerContext {
  use_case: string | null
  buyer_type: string | null
  price_sentiment: 'expensive' | 'fair' | 'cheap' | 'not_mentioned' | null
}

export interface ExternalReference {
  source: string
  context: string
}

// --- Section B: Buyer Psychology ---

export type ExpertiseLevel = 'novice' | 'intermediate' | 'expert' | 'professional'
export type FrustrationThreshold = 'low' | 'medium' | 'high'
export type DiscoveryChannel = 'amazon_organic' | 'youtube' | 'reddit' | 'friend' | 'amazon_choice' | 'unknown'
export type BuyerHousehold = 'single' | 'family' | 'professional' | 'gift' | 'bulk'
export type BudgetType = 'budget_constrained' | 'value_seeker' | 'premium_willing' | 'unknown'
export type UseIntensity = 'light' | 'moderate' | 'heavy'
export type ResearchDepth = 'impulse' | 'light' | 'moderate' | 'deep'
export type ConsequenceSeverity = 'none' | 'positive_impact' | 'inconvenience' | 'workflow_impact' | 'financial_loss' | 'safety_concern'
export type ReplacementBehavior = 'returned' | 'replaced_same' | 'switched_brand' | 'switched_to' | 'avoided' | 'kept_broken' | 'kept_using' | 'repurchased' | 'unknown'

export interface ConsiderationSetItem {
  product: string
  why_not: string
}

export interface CommunityMention {
  platform: string
  context: string
}

// --- Section C: Extended Context ---

export type BrandLoyaltyDepth = 'first_time' | 'occasional' | 'loyal' | 'long_term_loyal'
export type ReviewDelaySignal = 'immediate' | 'days' | 'weeks' | 'months' | 'unknown'
export type SentimentTrajectory = 'always_negative' | 'degraded' | 'mixed_then_negative' | 'mixed_then_positive' | 'improved' | 'always_positive' | 'unknown'
export type OccasionContext = 'none' | 'gift' | 'replacement' | 'upgrade' | 'first_in_category' | 'seasonal' | 'event' | 'professional_use'

export interface EcosystemLockIn {
  level: 'free' | 'partially' | 'fully'
  ecosystem: string | null
}

export interface SafetyFlag {
  flagged: boolean
  description: string | null
}

export interface BulkPurchaseSignal {
  type: 'single' | 'multi'
  estimated_qty: number | null
}

export interface SwitchingBarrier {
  level: 'none' | 'low' | 'medium' | 'high'
  reason: string | null
}

export interface AmplificationIntent {
  intent: 'quiet' | 'private' | 'social'
  context: string | null
}

export interface ReviewSentimentOpenness {
  open: boolean
  condition: string | null
}

// --- Full deep enrichment object ---

export interface DeepEnrichment {
  // Section A: Product Analysis
  sentiment_aspects: SentimentAspect[] | null
  feature_requests: string[] | null
  failure_details: FailureDetails | null
  product_comparisons: ProductComparison[] | null
  product_name_mentioned: string | null
  buyer_context: BuyerContext | null
  quotable_phrases: string[] | null
  would_repurchase: boolean | null
  external_references: ExternalReference[] | null
  positive_aspects: string[] | null

  // Section B: Buyer Psychology
  expertise_level: ExpertiseLevel | null
  frustration_threshold: FrustrationThreshold | null
  discovery_channel: DiscoveryChannel | null
  consideration_set: ConsiderationSetItem[] | null
  buyer_household: BuyerHousehold | null
  profession_hint: string | null
  budget_type: BudgetType | null
  use_intensity: UseIntensity | null
  research_depth: ResearchDepth | null
  community_mentions: CommunityMention[] | null
  consequence_severity: ConsequenceSeverity | null
  replacement_behavior: ReplacementBehavior | null

  // Section C: Extended Context
  brand_loyalty_depth: BrandLoyaltyDepth | null
  ecosystem_lock_in: EcosystemLockIn | null
  safety_flag: SafetyFlag | null
  bulk_purchase_signal: BulkPurchaseSignal | null
  review_delay_signal: ReviewDelaySignal | null
  sentiment_trajectory: SentimentTrajectory | null
  occasion_context: OccasionContext | null
  switching_barrier: SwitchingBarrier | null
  amplification_intent: AmplificationIntent | null
  review_sentiment_openness: ReviewSentimentOpenness | null
}
