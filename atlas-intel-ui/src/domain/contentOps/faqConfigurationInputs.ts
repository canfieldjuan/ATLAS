export const FAQ_MARKDOWN_OUTPUT = 'faq_markdown'
export const FAQ_DEFLECTION_REPORT_CONFIGURATION_OUTPUT = 'faq_deflection_report'
export const FAQ_INTENT_RULES_INPUT = 'faq_intent_rules'
export const FAQ_DOCUMENTATION_TERMS_INPUT = 'faq_documentation_terms'
export const FAQ_REPRESENTATIVE_TAXONOMY_TERMS_INPUT =
  'faq_representative_taxonomy_terms'
export const FAQ_VOCABULARY_GAP_RULES_INPUT = 'faq_vocabulary_gap_rules'

export const FAQ_CONFIGURATION_OUTPUTS = [
  FAQ_MARKDOWN_OUTPUT,
  FAQ_DEFLECTION_REPORT_CONFIGURATION_OUTPUT,
] as const

export function faqConfigurationInputsSelected(
  outputs: readonly string[],
): boolean {
  return outputs.some((output) =>
    FAQ_CONFIGURATION_OUTPUTS.includes(
      output as (typeof FAQ_CONFIGURATION_OUTPUTS)[number],
    ),
  )
}

export interface FAQConfigurationInputContracts {
  intentRules?: unknown
  documentationTerms?: unknown
  representativeTaxonomyTerms?: unknown
  vocabularyGapRules?: unknown
}

export function faqConfigurationControlsVisible(
  outputs: readonly string[],
  contracts: FAQConfigurationInputContracts,
): boolean {
  return (
    faqConfigurationInputsSelected(outputs) &&
    Boolean(
      contracts.intentRules ||
        contracts.documentationTerms ||
        contracts.representativeTaxonomyTerms ||
        contracts.vocabularyGapRules,
    )
  )
}

export function faqIntentRulesDraftValue(value: unknown): string {
  if (!Array.isArray(value)) {
    return typeof value === 'string' ? value : ''
  }
  return value
    .filter((rule) => typeof rule !== 'undefined' && rule !== null)
    .map((rule) => {
      if (typeof rule === 'string') return rule
      if (isIntentRuleObject(rule)) {
        return `${rule.topic}=${rule.keywords.join(',')}`
      }
      return String(rule)
    })
    .join('\n')
}

export function faqIntentRulesFromDraft(value: string): string[] {
  const seen = new Set<string>()
  const rules: string[] = []
  for (const line of value.split(/\r?\n/)) {
    const rule = line.trim()
    const key = rule.toLowerCase()
    if (!rule || seen.has(key)) continue
    seen.add(key)
    rules.push(rule)
  }
  return rules
}

function isIntentRuleObject(
  value: unknown,
): value is { topic: string; keywords: string[] } {
  return (
    typeof value === 'object' &&
    value !== null &&
    'topic' in value &&
    typeof value.topic === 'string' &&
    'keywords' in value &&
    Array.isArray(value.keywords) &&
    value.keywords.every((keyword) => typeof keyword === 'string')
  )
}
