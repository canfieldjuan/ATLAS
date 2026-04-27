import type { SuppressionReason } from '../types'

export const SUPPRESSION_LABELS: Record<SuppressionReason, string> = {
  insufficient_supporting_count: 'Insufficient supporting evidence',
  contradictory_evidence: 'Contradictory evidence',
  unverified_evidence: 'Unverified evidence',
  denominator_unknown: 'Unknown denominator',
  sample_size_below_threshold: 'Sample too small',
  weak_evidence_only: 'Weak evidence only',
  passing_mention_only: 'Passing mention only',
  subject_not_subject_vendor: 'Wrong subject',
  polarity_not_renderable: 'Unsupported polarity',
  role_not_renderable: 'Unsupported role',
  low_confidence: 'Low confidence',
  consumer_filter_applied: 'Filtered for this use',
}

export function suppressionLabel(reason: SuppressionReason | null | undefined): string | null {
  return reason ? SUPPRESSION_LABELS[reason] : null
}
