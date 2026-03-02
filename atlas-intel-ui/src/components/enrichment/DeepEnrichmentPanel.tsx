import { colorMaps } from './colorMaps'
import type { DeepEnrichment } from './types'
import {
  EnumBadge,
  StringList,
  ObjectTable,
  columnRenderers,
  KeyValueCard,
  inlineBadge,
  FailureCard,
  BooleanIndicator,
  TextValue,
  SafetyFlagBanner,
} from './renderers'

type Section = 'product_analysis' | 'buyer_psychology' | 'extended_context'

interface Props {
  section: Section
  data: DeepEnrichment
}

// --- Label for field groups ---
function SectionLabel({ children }: { children: React.ReactNode }) {
  return <h4 className="text-[10px] uppercase tracking-wider text-slate-500 mb-2">{children}</h4>
}

// --- Product Analysis ---

function ProductAnalysis({ d }: { d: DeepEnrichment }) {
  return (
    <div className="space-y-5">
      {/* Sentiment aspects table */}
      <ObjectTable
        rows={d.sentiment_aspects as Record<string, unknown>[] | null}
        columns={[
          { key: 'aspect', label: 'Aspect', render: columnRenderers.bold },
          { key: 'sentiment', label: 'Sentiment', render: columnRenderers.sentimentDot },
          { key: 'detail', label: 'Detail', render: columnRenderers.text },
        ]}
      />

      {/* Product name + buyer context + repurchase */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        {d.product_name_mentioned && (
          <div>
            <SectionLabel>Product Name Mentioned</SectionLabel>
            <TextValue value={d.product_name_mentioned} highlight />
          </div>
        )}
        {d.buyer_context && (
          <div>
            <SectionLabel>Buyer Context</SectionLabel>
            <KeyValueCard rows={[
              { label: 'Use Case', value: d.buyer_context.use_case },
              { label: 'Buyer Type', value: inlineBadge(d.buyer_context.buyer_type, colorMaps.buyerType) },
              { label: 'Price Feel', value: inlineBadge(d.buyer_context.price_sentiment, colorMaps.priceSentiment) },
            ]} />
          </div>
        )}
        {d.would_repurchase !== null && d.would_repurchase !== undefined && (
          <div className="flex items-end">
            <BooleanIndicator label="Would Repurchase" value={d.would_repurchase} />
          </div>
        )}
      </div>

      {/* Failure card */}
      <FailureCard details={d.failure_details} />

      {/* Product comparisons */}
      {d.product_comparisons?.length ? (
        <div>
          <SectionLabel>Product Comparisons</SectionLabel>
          <ObjectTable
            rows={d.product_comparisons as unknown as Record<string, unknown>[]}
            columns={[
              { key: 'product_name', label: 'Product', render: columnRenderers.bold },
              { key: 'direction', label: 'Direction', render: columnRenderers.directionBadge },
              { key: 'context', label: 'Context', render: columnRenderers.text },
            ]}
          />
        </div>
      ) : null}

      {/* Positive aspects + feature requests side by side */}
      {(d.positive_aspects?.length || d.feature_requests?.length) ? (
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
          {d.positive_aspects?.length ? (
            <div>
              <SectionLabel>Positive Aspects</SectionLabel>
              <StringList items={d.positive_aspects} variant="tag" />
            </div>
          ) : null}
          {d.feature_requests?.length ? (
            <div>
              <SectionLabel>Feature Requests</SectionLabel>
              <StringList items={d.feature_requests} variant="bullet" />
            </div>
          ) : null}
        </div>
      ) : null}

      {/* Quotable phrases */}
      {d.quotable_phrases?.length ? (
        <div>
          <SectionLabel>Quotable Phrases</SectionLabel>
          <StringList items={d.quotable_phrases} variant="quote" />
        </div>
      ) : null}

      {/* External references */}
      {d.external_references?.length ? (
        <div>
          <SectionLabel>External References</SectionLabel>
          <ObjectTable
            rows={d.external_references as unknown as Record<string, unknown>[]}
            columns={[
              { key: 'source', label: 'Source', render: columnRenderers.bold },
              { key: 'context', label: 'Context', render: columnRenderers.text },
            ]}
          />
        </div>
      ) : null}
    </div>
  )
}

// --- Buyer Psychology ---

function BuyerPsychology({ d }: { d: DeepEnrichment }) {
  return (
    <div className="space-y-5">
      {/* Enum badge strip */}
      <div className="flex flex-wrap gap-4">
        <EnumBadge label="Expertise" value={d.expertise_level} colorMap={colorMaps.expertiseLevel} />
        <EnumBadge label="Frustration" value={d.frustration_threshold} colorMap={colorMaps.frustrationThreshold} />
        <EnumBadge label="Budget" value={d.budget_type} colorMap={colorMaps.budgetType} />
        <EnumBadge label="Intensity" value={d.use_intensity} colorMap={colorMaps.useIntensity} />
        <EnumBadge label="Research" value={d.research_depth} colorMap={colorMaps.researchDepth} />
        <EnumBadge label="Consequence" value={d.consequence_severity} colorMap={colorMaps.consequenceSeverity} />
        <EnumBadge label="Replacement" value={d.replacement_behavior} colorMap={colorMaps.replacementBehavior} />
      </div>

      {/* Household + discovery + profession */}
      <div className="flex flex-wrap items-start gap-4">
        <EnumBadge label="Household" value={d.buyer_household} colorMap={colorMaps.buyerHousehold} />
        <EnumBadge label="Discovery" value={d.discovery_channel} colorMap={colorMaps.discoveryChannel} />
        {d.profession_hint && (
          <div className="flex flex-col gap-1">
            <span className="text-[10px] uppercase tracking-wider text-slate-500">Profession</span>
            <TextValue value={d.profession_hint} />
          </div>
        )}
      </div>

      {/* Consideration set table */}
      {d.consideration_set?.length ? (
        <div>
          <SectionLabel>Consideration Set</SectionLabel>
          <ObjectTable
            rows={d.consideration_set as unknown as Record<string, unknown>[]}
            columns={[
              { key: 'product', label: 'Product', render: columnRenderers.bold },
              { key: 'why_not', label: 'Why Not', render: columnRenderers.text },
            ]}
          />
        </div>
      ) : null}

      {/* Community mentions table */}
      {d.community_mentions?.length ? (
        <div>
          <SectionLabel>Community Mentions</SectionLabel>
          <ObjectTable
            rows={d.community_mentions as unknown as Record<string, unknown>[]}
            columns={[
              { key: 'platform', label: 'Platform', render: columnRenderers.platformTag },
              { key: 'context', label: 'Context', render: columnRenderers.text },
            ]}
          />
        </div>
      ) : null}
    </div>
  )
}

// --- Extended Context ---

function ExtendedContext({ d }: { d: DeepEnrichment }) {
  return (
    <div className="space-y-5">
      {/* Enum badge strip */}
      <div className="flex flex-wrap gap-4">
        <EnumBadge label="Brand Loyalty" value={d.brand_loyalty_depth} colorMap={colorMaps.brandLoyaltyDepth} />
        <EnumBadge label="Review Delay" value={d.review_delay_signal} colorMap={colorMaps.reviewDelaySignal} />
        <EnumBadge label="Sentiment Arc" value={d.sentiment_trajectory} colorMap={colorMaps.sentimentTrajectory} />
        <EnumBadge label="Occasion" value={d.occasion_context} colorMap={colorMaps.occasionContext} />
      </div>

      {/* Ecosystem lock-in + switching barrier */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {d.ecosystem_lock_in && (
          <div className="bg-slate-800/20 rounded-lg p-3">
            <SectionLabel>Ecosystem Lock-in</SectionLabel>
            <KeyValueCard rows={[
              { label: 'Level', value: inlineBadge(d.ecosystem_lock_in.level, colorMaps.ecosystemLevel) },
              { label: 'Ecosystem', value: d.ecosystem_lock_in.ecosystem },
            ]} />
          </div>
        )}
        {d.switching_barrier && (
          <div className="bg-slate-800/20 rounded-lg p-3">
            <SectionLabel>Switching Barrier</SectionLabel>
            <KeyValueCard rows={[
              { label: 'Level', value: inlineBadge(d.switching_barrier.level, colorMaps.switchingBarrierLevel) },
              { label: 'Reason', value: d.switching_barrier.reason },
            ]} />
          </div>
        )}
      </div>

      {/* Amplification + sentiment openness */}
      <div className="grid grid-cols-1 sm:grid-cols-2 gap-4">
        {d.amplification_intent && (
          <div className="bg-slate-800/20 rounded-lg p-3">
            <SectionLabel>Amplification Intent</SectionLabel>
            <KeyValueCard rows={[
              { label: 'Intent', value: inlineBadge(d.amplification_intent.intent, colorMaps.amplificationIntent) },
              { label: 'Context', value: d.amplification_intent.context },
            ]} />
          </div>
        )}
        {d.review_sentiment_openness && (
          <div className="bg-slate-800/20 rounded-lg p-3">
            <SectionLabel>Openness to Change</SectionLabel>
            <KeyValueCard rows={[
              {
                label: 'Stance',
                value: d.review_sentiment_openness.open
                  ? <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-green-500/10 text-green-400">Open</span>
                  : <span className="px-2 py-0.5 rounded-full text-xs font-medium bg-red-500/10 text-red-400">Closed</span>,
              },
              { label: 'Condition', value: d.review_sentiment_openness.condition },
            ]} />
          </div>
        )}
      </div>

      {/* Safety flag */}
      <SafetyFlagBanner flag={d.safety_flag} />

      {/* Bulk purchase signal */}
      {d.bulk_purchase_signal && (
        <div>
          <SectionLabel>Bulk Purchase Signal</SectionLabel>
          <KeyValueCard rows={[
            { label: 'Type', value: inlineBadge(d.bulk_purchase_signal.type, colorMaps.bulkPurchaseType) },
            { label: 'Est. Qty', value: d.bulk_purchase_signal.estimated_qty != null ? String(d.bulk_purchase_signal.estimated_qty) : null },
          ]} />
        </div>
      )}
    </div>
  )
}

// --- Main Panel ---

export default function DeepEnrichmentPanel({ section, data }: Props) {
  switch (section) {
    case 'product_analysis':
      return <ProductAnalysis d={data} />
    case 'buyer_psychology':
      return <BuyerPsychology d={data} />
    case 'extended_context':
      return <ExtendedContext d={data} />
  }
}

// Count non-null/non-empty fields for a section
export function countFields(section: Section, data: DeepEnrichment): number {
  const keys = sectionKeys[section]
  return keys.filter(k => {
    const v = data[k as keyof DeepEnrichment]
    if (v === null || v === undefined) return false
    if (Array.isArray(v)) return v.length > 0
    if (typeof v === 'object') return Object.values(v as unknown as Record<string, unknown>).some(x => x !== null && x !== undefined)
    if (typeof v === 'string') return v.length > 0
    return true
  }).length
}

const sectionKeys: Record<Section, string[]> = {
  product_analysis: [
    'sentiment_aspects', 'feature_requests', 'failure_details', 'product_comparisons',
    'product_name_mentioned', 'buyer_context', 'quotable_phrases', 'would_repurchase',
    'external_references', 'positive_aspects',
  ],
  buyer_psychology: [
    'expertise_level', 'frustration_threshold', 'discovery_channel', 'consideration_set',
    'buyer_household', 'profession_hint', 'budget_type', 'use_intensity', 'research_depth',
    'community_mentions', 'consequence_severity', 'replacement_behavior',
  ],
  extended_context: [
    'brand_loyalty_depth', 'ecosystem_lock_in', 'safety_flag', 'bulk_purchase_signal',
    'review_delay_signal', 'sentiment_trajectory', 'occasion_context', 'switching_barrier',
    'amplification_intent', 'review_sentiment_openness',
  ],
}
