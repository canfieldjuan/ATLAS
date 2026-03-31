import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import BlogFailureExplanation from './BlogFailureExplanation'

describe('BlogFailureExplanation', () => {
  it('renders the shared failure explanation content', () => {
    render(
      <BlogFailureExplanation
        explanation={{
          boundary: 'publish',
          primary_blocker: 'unsupported_data_claim:Magento',
          cause_type: 'unsupported_claim',
          blocking_issues: ['unsupported_data_claim:Magento'],
          warnings: [],
          matched_groups: [],
          available_groups: [],
          missing_groups: [],
          required_proof_terms: [],
          used_proof_terms: [],
          unused_proof_terms: [],
          missing_inputs: [],
          missing_primary_inputs: [],
          context_sources: ['data_context'],
        }}
      />,
    )

    expect(screen.getByText('Why This Failed')).toBeInTheDocument()
    expect(screen.getByText('unsupported_data_claim:Magento')).toBeInTheDocument()
  })
})
