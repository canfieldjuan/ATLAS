import { render, screen } from '@testing-library/react'
import { describe, expect, it } from 'vitest'
import BlogQualityDiagnosticsPanel from './BlogQualityDiagnosticsPanel'

describe('BlogQualityDiagnosticsPanel', () => {
  it('renders grouped diagnostics data', () => {
    render(
      <BlogQualityDiagnosticsPanel
        data={{
          days: 14,
          top_n: 10,
          by_boundary: [{ boundary: 'publish', count: 2 }],
          by_cause_type: [{ cause_type: 'unsupported_claim', count: 2 }],
          top_primary_blockers: [{ reason: 'unsupported_data_claim:Magento', count: 2 }],
          top_missing_inputs: [{ input: 'reasoning_anchor_examples', count: 1 }],
          by_topic_type: [{ topic_type: 'migration_guide', count: 2 }],
          top_subjects: [{ subject: 'Shopify', count: 2 }],
        }}
      />,
    )

    expect(screen.getByText('Blog Failure Diagnostics')).toBeInTheDocument()
    expect(screen.getAllByText('unsupported_claim').length).toBeGreaterThan(0)
    expect(screen.getAllByText('unsupported_data_claim:Magento').length).toBeGreaterThan(0)
    expect(screen.getByText('migration_guide')).toBeInTheDocument()
    expect(screen.getAllByText('Shopify').length).toBeGreaterThan(0)
  })

  it('renders an empty state when no diagnostics exist', () => {
    render(<BlogQualityDiagnosticsPanel data={null} />)

    expect(screen.getByText('No blog failure diagnostics available yet.')).toBeInTheDocument()
  })
})
