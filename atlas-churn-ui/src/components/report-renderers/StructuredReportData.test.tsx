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

  it('fails closed for dangerous structured report fields without ProductClaim context', () => {
    render(
      <StructuredReportData
        data={{
          cross_vendor_battles: [
            {
              opponent: 'Google Workspace',
              winner: 'Microsoft 365',
              loser: 'Google Workspace',
              conclusion: 'Microsoft 365 is winning enterprise evaluations.',
            },
          ],
        }}
      />,
    )

    expect(screen.getByText('Cross Vendor Battles')).toBeInTheDocument()
    expect(screen.getByTestId('dangerous-cross_vendor_battles-gate')).toHaveTextContent(
      'Legacy/unvalidated report field',
    )
    expect(screen.queryByText('winner: Microsoft 365')).not.toBeInTheDocument()
    expect(screen.queryByText('Microsoft 365 is winning enterprise evaluations.')).not.toBeInTheDocument()
  })

  it('renders dangerous structured report fields when every item is report-safe', () => {
    render(
      <StructuredReportData
        data={{
          cross_vendor_battles: [
            {
              opponent: 'Google Workspace',
              winner: 'Microsoft 365',
              conclusion: 'Microsoft 365 is winning enterprise evaluations.',
              product_claim: {
                render_allowed: true,
                report_allowed: true,
                supporting_count: 3,
                direct_evidence_count: 3,
                witness_count: 2,
              },
            },
          ],
        }}
      />,
    )

    expect(screen.queryByTestId('dangerous-cross_vendor_battles-gate')).not.toBeInTheDocument()
    expect(screen.getByText('winner: Microsoft 365')).toBeInTheDocument()
    expect(screen.getByText('Microsoft 365 is winning enterprise evaluations.')).toBeInTheDocument()
  })

  it('fails closed for mixed dangerous sections when any item lacks ProductClaim context', () => {
    render(
      <StructuredReportData
        data={{
          recommended_plays: [
            {
              play: 'Lead with migration support',
              product_claim: {
                render_allowed: true,
                report_allowed: true,
                supporting_count: 2,
                direct_evidence_count: 2,
                witness_count: 2,
              },
            },
            {
              play: 'Claim the competitor is losing renewals',
            },
          ],
        }}
      />,
    )

    expect(screen.getByTestId('dangerous-recommended_plays-gate')).toHaveTextContent(
      'Legacy/unvalidated report field',
    )
    expect(screen.queryByText('Lead with migration support')).not.toBeInTheDocument()
    expect(screen.queryByText('Claim the competitor is losing renewals')).not.toBeInTheDocument()
  })

  it('fails closed for nested dangerous report fields inside mixed objects', () => {
    render(
      <StructuredReportData
        data={{
          competitive_analysis: {
            summary: 'Nested competitive section.',
            cross_vendor_battles: [
              {
                opponent: 'Google Workspace',
                winner: 'Microsoft 365',
                loser: 'Google Workspace',
                conclusion: 'Microsoft 365 is winning enterprise evaluations.',
              },
            ],
            talk_track: {
              opening: 'Tell buyers Microsoft 365 is winning the category.',
            },
          },
        }}
      />,
    )

    expect(screen.getByText('Competitive Analysis')).toBeInTheDocument()
    expect(screen.getByTestId('dangerous-cross_vendor_battles-gate')).toHaveTextContent(
      'Legacy/unvalidated report field',
    )
    expect(screen.getByTestId('dangerous-talk_track-gate')).toHaveTextContent(
      'Legacy/unvalidated report field',
    )
    expect(screen.queryByText('winner: Microsoft 365')).not.toBeInTheDocument()
    expect(
      screen.queryByText('Microsoft 365 is winning enterprise evaluations.'),
    ).not.toBeInTheDocument()
    expect(
      screen.queryByText('Tell buyers Microsoft 365 is winning the category.'),
    ).not.toBeInTheDocument()
  })
})
