import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import CampaignFailureExplanation from './CampaignFailureExplanation'

describe('CampaignFailureExplanation', () => {
  it('renders the failure cause, missing inputs, and unused proof terms', () => {
    render(
      <CampaignFailureExplanation
        explanation={{
          boundary: 'manual_approval',
          primary_blocker: 'missing_exact_proof_term',
          cause_type: 'content_ignored_available_evidence',
          blocking_issues: ['missing_exact_proof_term'],
          warnings: [],
          matched_groups: ['pain_terms'],
          available_groups: ['numeric_terms', 'pain_terms'],
          missing_groups: ['numeric_terms'],
          required_proof_terms: ['$200k/year', 'Q2 renewal'],
          used_proof_terms: [],
          unused_proof_terms: ['$200k/year', 'Q2 renewal'],
          missing_inputs: [],
          missing_primary_inputs: ['reasoning_reference_ids'],
          context_sources: ['metadata', 'reasoning_fallback'],
          fallback_used: true,
          reasoning_view_found: true,
          anchor_count: 2,
          highlight_count: 1,
          reference_id_counts: { witness_ids: 2 },
          anchor_labels: ['outlier_or_named_account'],
          context_has_anchor_examples: true,
          context_has_witness_highlights: true,
          context_has_reference_ids: true,
        }}
      />,
    )

    expect(screen.getByText('Why This Failed')).toBeInTheDocument()
    expect(screen.getByText('missing_exact_proof_term')).toBeInTheDocument()
    expect(screen.getByText('Content Ignored Available Evidence')).toBeInTheDocument()
    expect(screen.getByText('Unavailable Before Fallback')).toBeInTheDocument()
    expect(screen.getByText('reasoning_reference_ids')).toBeInTheDocument()
    expect(screen.getByText('$200k/year')).toBeInTheDocument()
    expect(screen.getByText('manual_approval')).toBeInTheDocument()
  })

  it('renders nothing when there is no actionable explanation data', () => {
    const { container } = render(
      <CampaignFailureExplanation
        explanation={{
          blocking_issues: [],
          warnings: [],
          matched_groups: [],
          available_groups: [],
          missing_groups: [],
          required_proof_terms: [],
          used_proof_terms: [],
          unused_proof_terms: [],
          missing_inputs: [],
          missing_primary_inputs: [],
          context_sources: [],
          reference_id_counts: {},
        }}
      />,
    )

    expect(container).toBeEmptyDOMElement()
  })
})
