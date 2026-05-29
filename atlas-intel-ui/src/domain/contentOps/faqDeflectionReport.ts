export const FAQ_DEFLECTION_REPORT_OUTPUT = 'faq_deflection_report'
export const FAQ_RESOLUTION_EVIDENCE_STATUS = 'resolution_evidence'

export interface FAQDeflectionReportView {
  markdown: string
  summary: FAQDeflectionReportSummaryView
  items: FAQDeflectionReportItemView[]
  provenItems: FAQDeflectionReportItemView[]
  noProvenItems: FAQDeflectionReportItemView[]
  outputChecks: Record<string, boolean>
}

export interface FAQDeflectionReportSummaryView {
  generated: number | null
  sourceCount: number | null
  ticketSourceCount: number | null
  draftedAnswerCount: number | null
  noProvenAnswerCount: number | null
  topQuestion: string
  topOpportunityScore: number | null
}

export interface FAQDeflectionReportItemView {
  question: string
  topic: string
  summary: string
  answerEvidenceStatus: string
  ticketCount: number | null
  opportunityScore: number | null
  weightedFrequency: number | null
  steps: string[]
  sourceIds: string[]
  evidenceQuotes: string[]
  termMappings: FAQTermMappingView[]
}

export interface FAQTermMappingView {
  customerTerm: string
  documentationTerm: string
  suggestion: string
  sourceIdCount: number | null
}

export type FAQDeflectionReportAnswerTone = 'proven' | 'unproven'

export function faqDeflectionReportView(
  result: Record<string, unknown>,
): FAQDeflectionReportView | null {
  const summary = recordValue(result.summary)
  const faqResult = recordValue(result.faq_result)
  if (!summary || !faqResult || typeof result.markdown !== 'string') {
    return null
  }

  const items = recordArray(faqResult.items).map(faqDeflectionReportItem)
  const provenItems = items.filter(isProvenFAQDeflectionReportItem)
  const noProvenItems = items.filter((item) => !isProvenFAQDeflectionReportItem(item))

  return {
    markdown: result.markdown,
    summary: {
      generated: numberField(summary, 'generated'),
      sourceCount: numberField(summary, 'source_count'),
      ticketSourceCount: numberField(summary, 'ticket_source_count'),
      draftedAnswerCount: numberField(summary, 'drafted_answer_count'),
      noProvenAnswerCount: numberField(summary, 'no_proven_answer_count'),
      topQuestion: stringField(summary, 'top_question') ?? '',
      topOpportunityScore: numberField(summary, 'top_opportunity_score'),
    },
    items,
    provenItems,
    noProvenItems,
    outputChecks: booleanRecord(summary.output_checks),
  }
}

export function isProvenFAQDeflectionReportItem(
  item: FAQDeflectionReportItemView,
): boolean {
  return item.answerEvidenceStatus === FAQ_RESOLUTION_EVIDENCE_STATUS
}

export function faqDeflectionReportAnswerSteps(
  item: FAQDeflectionReportItemView,
  tone: FAQDeflectionReportAnswerTone,
): string[] {
  if (tone !== 'proven' || !isProvenFAQDeflectionReportItem(item)) {
    return []
  }
  return item.steps
}

function faqDeflectionReportItem(
  item: Record<string, unknown>,
): FAQDeflectionReportItemView {
  return {
    question: stringField(item, 'question') ?? 'Untitled FAQ opportunity',
    topic: stringField(item, 'topic') ?? '',
    summary: stringField(item, 'summary') ?? '',
    answerEvidenceStatus: stringField(item, 'answer_evidence_status') ?? '',
    ticketCount: numberField(item, 'ticket_count'),
    opportunityScore: numberField(item, 'opportunity_score'),
    weightedFrequency: numberField(item, 'weighted_frequency'),
    steps: stringArray(item.steps),
    sourceIds: stringArray(item.source_ids),
    evidenceQuotes: stringArray(item.evidence_quotes),
    termMappings: recordArray(item.term_mappings).map((mapping) => ({
      customerTerm: stringField(mapping, 'customer_term') ?? '',
      documentationTerm: stringField(mapping, 'documentation_term') ?? '',
      suggestion: stringField(mapping, 'suggestion') ?? '',
      sourceIdCount: numberField(mapping, 'source_id_count'),
    })),
  }
}

function booleanRecord(value: unknown): Record<string, boolean> {
  const record = recordValue(value)
  if (!record) return {}
  return Object.fromEntries(
    Object.entries(record)
      .filter((entry): entry is [string, boolean] => typeof entry[1] === 'boolean'),
  )
}

function recordValue(value: unknown): Record<string, unknown> | null {
  return value && typeof value === 'object' && !Array.isArray(value)
    ? value as Record<string, unknown>
    : null
}

function recordArray(value: unknown): Record<string, unknown>[] {
  return Array.isArray(value) ? value.filter(isRecord) : []
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return Boolean(value && typeof value === 'object' && !Array.isArray(value))
}

function stringField(
  record: Record<string, unknown>,
  key: string,
): string | null {
  const value = record[key]
  return typeof value === 'string' && value.trim() ? value.trim() : null
}

function numberField(
  record: Record<string, unknown>,
  key: string,
): number | null {
  const value = record[key]
  return typeof value === 'number' && Number.isFinite(value) ? value : null
}

function stringArray(value: unknown): string[] {
  return Array.isArray(value)
    ? value.filter(
        (item): item is string => typeof item === 'string' && item.trim() !== '',
      )
    : []
}
