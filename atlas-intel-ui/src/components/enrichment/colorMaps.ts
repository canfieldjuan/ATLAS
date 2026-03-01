// Enum value -> Tailwind class mappings for colored badges
// Each map returns [bgClass, textClass] pairs

type ColorPair = [bg: string, text: string]

const fallback: ColorPair = ['bg-slate-500/10', 'text-slate-400']

function mapLookup(map: Record<string, ColorPair>, value: string | null | undefined): ColorPair {
  if (!value) return fallback
  return map[value] ?? fallback
}

// --- Section A ---

const priceSentiment: Record<string, ColorPair> = {
  expensive:     ['bg-red-500/10',    'text-red-400'],
  fair:          ['bg-green-500/10',  'text-green-400'],
  cheap:         ['bg-amber-500/10',  'text-amber-400'],
  not_mentioned: ['bg-slate-500/10',  'text-slate-400'],
}

const comparisonDirection: Record<string, ColorPair> = {
  switched_to:   ['bg-green-500/10',  'text-green-400'],
  switched_from: ['bg-red-500/10',    'text-red-400'],
  considered:    ['bg-amber-500/10',  'text-amber-400'],
  compared:      ['bg-blue-500/10',   'text-blue-400'],
  recommended:   ['bg-cyan-500/10',   'text-cyan-400'],
  avoided:       ['bg-orange-500/10', 'text-orange-400'],
  used_with:     ['bg-purple-500/10', 'text-purple-400'],
  relied_on:     ['bg-emerald-500/10','text-emerald-400'],
}

const sentimentDot: Record<string, string> = {
  positive: 'bg-green-400',
  negative: 'bg-red-400',
  mixed:    'bg-amber-400',
  neutral:  'bg-slate-400',
}

// --- Section B ---

const expertiseLevel: Record<string, ColorPair> = {
  novice:        ['bg-slate-500/10',  'text-slate-400'],
  intermediate:  ['bg-blue-500/10',   'text-blue-400'],
  expert:        ['bg-purple-500/10', 'text-purple-400'],
  professional:  ['bg-cyan-500/10',   'text-cyan-400'],
}

const frustrationThreshold: Record<string, ColorPair> = {
  low:    ['bg-red-500/10',   'text-red-400'],
  medium: ['bg-amber-500/10', 'text-amber-400'],
  high:   ['bg-green-500/10', 'text-green-400'],
}

const discoveryChannel: Record<string, ColorPair> = {
  amazon_organic: ['bg-amber-500/10',  'text-amber-400'],
  youtube:        ['bg-red-500/10',    'text-red-400'],
  reddit:         ['bg-orange-500/10', 'text-orange-400'],
  friend:         ['bg-green-500/10',  'text-green-400'],
  amazon_choice:  ['bg-yellow-500/10', 'text-yellow-400'],
  unknown:        ['bg-slate-500/10',  'text-slate-400'],
}

const buyerHousehold: Record<string, ColorPair> = {
  single:       ['bg-blue-500/10',   'text-blue-400'],
  family:       ['bg-green-500/10',  'text-green-400'],
  professional: ['bg-purple-500/10', 'text-purple-400'],
  gift:         ['bg-pink-500/10',   'text-pink-400'],
  bulk:         ['bg-amber-500/10',  'text-amber-400'],
}

const budgetType: Record<string, ColorPair> = {
  budget_constrained: ['bg-red-500/10',    'text-red-400'],
  value_seeker:       ['bg-amber-500/10',  'text-amber-400'],
  premium_willing:    ['bg-green-500/10',  'text-green-400'],
  unknown:            ['bg-slate-500/10',  'text-slate-400'],
}

const useIntensity: Record<string, ColorPair> = {
  light:    ['bg-green-500/10',  'text-green-400'],
  moderate: ['bg-amber-500/10',  'text-amber-400'],
  heavy:    ['bg-red-500/10',    'text-red-400'],
}

const researchDepth: Record<string, ColorPair> = {
  impulse:  ['bg-red-500/10',    'text-red-400'],
  light:    ['bg-amber-500/10',  'text-amber-400'],
  moderate: ['bg-blue-500/10',   'text-blue-400'],
  deep:     ['bg-purple-500/10', 'text-purple-400'],
}

const consequenceSeverity: Record<string, ColorPair> = {
  inconvenience:   ['bg-slate-500/10',  'text-slate-400'],
  workflow_impact: ['bg-amber-500/10',  'text-amber-400'],
  financial_loss:  ['bg-orange-500/10', 'text-orange-400'],
  safety_concern:  ['bg-red-500/10',    'text-red-400'],
}

const replacementBehavior: Record<string, ColorPair> = {
  returned:       ['bg-amber-500/10',  'text-amber-400'],
  replaced_same:  ['bg-blue-500/10',   'text-blue-400'],
  switched_brand: ['bg-red-500/10',    'text-red-400'],
  switched_to:    ['bg-red-500/10',    'text-red-400'],
  avoided:        ['bg-orange-500/10', 'text-orange-400'],
  kept_broken:    ['bg-slate-500/10',  'text-slate-400'],
  unknown:        ['bg-slate-500/10',  'text-slate-400'],
}

// --- Section C ---

const brandLoyaltyDepth: Record<string, ColorPair> = {
  first_time:      ['bg-slate-500/10',   'text-slate-400'],
  occasional:      ['bg-blue-500/10',    'text-blue-400'],
  loyal:           ['bg-green-500/10',   'text-green-400'],
  long_term_loyal: ['bg-emerald-500/10', 'text-emerald-400'],
}

const reviewDelaySignal: Record<string, ColorPair> = {
  immediate: ['bg-green-500/10',  'text-green-400'],
  days:      ['bg-blue-500/10',   'text-blue-400'],
  weeks:     ['bg-amber-500/10',  'text-amber-400'],
  months:    ['bg-purple-500/10', 'text-purple-400'],
  unknown:   ['bg-slate-500/10',  'text-slate-400'],
}

const sentimentTrajectory: Record<string, ColorPair> = {
  always_bad:         ['bg-red-500/10',    'text-red-400'],
  degraded:           ['bg-orange-500/10', 'text-orange-400'],
  mixed_then_bad:     ['bg-amber-500/10',  'text-amber-400'],
  initially_positive: ['bg-blue-500/10',   'text-blue-400'],
  unknown:            ['bg-slate-500/10',  'text-slate-400'],
}

const occasionContext: Record<string, ColorPair> = {
  none:              ['bg-slate-500/10',  'text-slate-400'],
  gift:              ['bg-pink-500/10',   'text-pink-400'],
  replacement:       ['bg-amber-500/10',  'text-amber-400'],
  upgrade:           ['bg-blue-500/10',   'text-blue-400'],
  first_in_category: ['bg-green-500/10',  'text-green-400'],
  seasonal:          ['bg-purple-500/10', 'text-purple-400'],
  event:             ['bg-cyan-500/10',   'text-cyan-400'],
  professional_use:  ['bg-indigo-500/10', 'text-indigo-400'],
}

const ecosystemLevel: Record<string, ColorPair> = {
  free:      ['bg-green-500/10',  'text-green-400'],
  partially: ['bg-amber-500/10',  'text-amber-400'],
  fully:     ['bg-red-500/10',    'text-red-400'],
}

const switchingBarrierLevel: Record<string, ColorPair> = {
  none:   ['bg-green-500/10',  'text-green-400'],
  low:    ['bg-blue-500/10',   'text-blue-400'],
  medium: ['bg-amber-500/10',  'text-amber-400'],
  high:   ['bg-red-500/10',    'text-red-400'],
}

const amplificationIntentMap: Record<string, ColorPair> = {
  quiet:   ['bg-slate-500/10',  'text-slate-400'],
  private: ['bg-blue-500/10',   'text-blue-400'],
  social:  ['bg-red-500/10',    'text-red-400'],
}

const buyerType: Record<string, ColorPair> = {
  casual:       ['bg-slate-500/10',  'text-slate-400'],
  power_user:   ['bg-blue-500/10',   'text-blue-400'],
  professional: ['bg-purple-500/10', 'text-purple-400'],
  first_time:   ['bg-green-500/10',  'text-green-400'],
  repeat_buyer: ['bg-cyan-500/10',   'text-cyan-400'],
  gift_buyer:   ['bg-pink-500/10',   'text-pink-400'],
  unknown:      ['bg-slate-500/10',  'text-slate-400'],
}

const bulkPurchaseType: Record<string, ColorPair> = {
  single: ['bg-slate-500/10', 'text-slate-400'],
  multi:  ['bg-amber-500/10', 'text-amber-400'],
}

// Public API
export const colorMaps = {
  priceSentiment,
  comparisonDirection,
  sentimentDot,
  expertiseLevel,
  frustrationThreshold,
  discoveryChannel,
  buyerHousehold,
  budgetType,
  useIntensity,
  researchDepth,
  consequenceSeverity,
  replacementBehavior,
  brandLoyaltyDepth,
  reviewDelaySignal,
  sentimentTrajectory,
  occasionContext,
  ecosystemLevel,
  switchingBarrierLevel,
  amplificationIntent: amplificationIntentMap,
  buyerType,
  bulkPurchaseType,
} as const

export { mapLookup, fallback }
export type { ColorPair }
