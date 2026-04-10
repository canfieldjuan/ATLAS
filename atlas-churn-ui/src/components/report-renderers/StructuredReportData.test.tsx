import { fireEvent, render, screen } from '@testing-library/react'
import { describe, expect, it, vi } from 'vitest'
import { StructuredReportData } from './StructuredReportData'

describe('StructuredReportData', () => {
  it('renders section-level citations from sibling metadata and hides raw metadata-only sections', () => {
    const onOpenWitness = vi.fn()

    render(
      <StructuredReportData
        data={{
          key_insights: [
            { label: 'Pricing friction', summary: 'Pricing created churn risk' },
          ],
          key_insights_reference_ids: {
            witness_ids: ['w1', 'w2'],
          },
          key_insights_witness_highlights: [
            { witness_id: 'w1', reviewer_company: 'Acme', excerpt_text: 'Pricing changed overnight' },
            { witness_id: 'w2', reviewer_company: 'Bravo', excerpt_text: 'Costs increased too fast' },
          ],
          reasoning_reference_ids: { witness_ids: ['top-level-only'] },
        }}
        vendorName="Zendesk"
        onOpenWitness={onOpenWitness}
      />,
    )

    expect(screen.getByText('Key Insights')).toBeInTheDocument()
    expect(screen.queryByText('Key Insights Reference Ids')).not.toBeInTheDocument()
    expect(screen.queryByText('Reasoning Reference Ids')).not.toBeInTheDocument()

    const citationButtons = screen.getAllByRole('button', { name: /\[\d+\]/ })
    expect(citationButtons).toHaveLength(2)

    fireEvent.click(citationButtons[0])
    expect(onOpenWitness).toHaveBeenCalledWith('w1', 'Zendesk')
  })
})
