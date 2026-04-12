import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { MemoryRouter } from 'react-router-dom'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import { StructuredReportData } from './StructuredReportData'

describe('StructuredReportData', () => {
  beforeEach(() => {
    cleanup()
  })

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
    expect(screen.getByText('Witness-backed')).toBeInTheDocument()
    expect(screen.getByText('2 linked witness citations')).toBeInTheDocument()
    expect(screen.queryByText('Key Insights Reference Ids')).not.toBeInTheDocument()
    expect(screen.queryByText('Reasoning Reference Ids')).not.toBeInTheDocument()

    const citationButtons = screen.getAllByRole('button', { name: /\[\d+\]/ })
    expect(citationButtons).toHaveLength(2)

    fireEvent.click(citationButtons[0])
    expect(onOpenWitness).toHaveBeenCalledWith('w1', 'Zendesk')
  })

  it('falls back to the evidence explorer when witness-backed sections do not have an open handler', () => {
    render(
      <MemoryRouter>
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
          }}
          vendorName="Zendesk"
          backTo="/report?vendor=Zendesk&ref=test-token&mode=view"
          asOfDate="2026-04-08"
          windowDays={45}
        />
      </MemoryRouter>,
    )

    expect(screen.getByRole('link', { name: 'View 2 witness citations' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&as_of_date=2026-04-08&window_days=45&back_to=%2Freport%3Fvendor%3DZendesk%26ref%3Dtest-token%26mode%3Dview',
    )
  })

  it('shows explicit partial and thin evidence states when sections lack witness citations', () => {
    render(
      <StructuredReportData
        data={{
          objection_handlers: {
            summary: 'Pricing objections are rising',
            reference_ids: {
              metric_ids: ['m1'],
            },
          },
          recommended_plays: {
            summary: 'Lead with migration support',
          },
        }}
      />,
    )

    expect(screen.getByText('Partial evidence')).toBeInTheDocument()
    expect(screen.getByText('Section has evidence metadata, but no linked witness citations yet.')).toBeInTheDocument()
    expect(screen.getByText('Thin evidence')).toBeInTheDocument()
    expect(screen.getByText('No linked witness citations for this section yet.')).toBeInTheDocument()
  })

  it('prefers backend-provided section evidence over local inference', () => {
    render(
      <StructuredReportData
        data={{
          recommended_plays: {
            summary: 'Lead with migration support',
          },
        }}
        sectionEvidence={{
          recommended_plays: {
            state: 'partial',
            label: 'Partial evidence',
            detail: 'Backend flagged this section for operator review.',
          },
        }}
      />,
    )

    expect(screen.getByText('Partial evidence')).toBeInTheDocument()
    expect(screen.getByText('Backend flagged this section for operator review.')).toBeInTheDocument()
    expect(screen.queryByText('Thin evidence')).not.toBeInTheDocument()
  })
})
