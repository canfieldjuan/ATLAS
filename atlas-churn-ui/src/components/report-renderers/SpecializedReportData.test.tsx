import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import { SpecializedReportData } from './SpecializedReportData'

describe('SpecializedReportData', () => {
  it('renders thin-evidence battle cards without a headline using executive summary, talk track, and plays', () => {
    render(
      <MemoryRouter>
        <SpecializedReportData
          reportType="battle_card"
          data={{
            vendor: 'Zendesk',
            quality_status: 'thin_evidence',
            executive_summary:
              'Zendesk customers are trapped in multi-month support failures that cascade into billing disputes and email delivery breakdowns.',
            talk_track: {
              opening: 'Buyers are actively pressure-testing Zendesk because contract lock in concerns keep resurfacing.',
              mid_call_pivot: 'This is not just a support problem. It is a trust problem.',
              closing: 'Let us run a quick audit of support reliability and billing predictability.',
            },
            recommended_plays: [
              {
                play: 'Target evaluators with a side-by-side evaluation on fit and switching friction.',
                key_message: 'Lead with faster evaluation clarity and fewer edge-case surprises.',
              },
            ],
          }}
        />
      </MemoryRouter>,
    )

    expect(screen.getByText('thin evidence')).toBeInTheDocument()
    expect(
      screen.getByText('This battle card is usable for directional validation, but the evidence base is still thin.'),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/Zendesk customers are trapped in multi-month support failures/i),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/Buyers are actively pressure-testing Zendesk/i),
    ).toBeInTheDocument()
    expect(
      screen.getByText(/Target evaluators with a side-by-side evaluation/i),
    ).toBeInTheDocument()
  })

  it('renders accounts-in-motion reports with witness actions', async () => {
    const user = userEvent.setup()
    const onOpenWitness = vi.fn()

    render(
      <MemoryRouter>
        <SpecializedReportData
          reportType="accounts_in_motion"
          vendorName="Zendesk"
          onOpenWitness={onOpenWitness}
          data={{
            total_accounts_in_motion: 3,
            account_pressure_summary: 'A single named account is showing early evaluation pressure.',
            account_pressure_disclaimer: 'Early account signal only.',
            account_actionability_tier: 'low',
            pricing_pressure: {
              price_complaint_rate: 0.25,
            },
            cross_vendor_context: {
              top_destination: 'Freshdesk',
            },
            reference_ids: {
              witness_ids: ['witness-1'],
            },
            accounts: [
              {
                company: 'Acme Corp',
                opportunity_score: 72,
                top_quote: 'Pricing pressure is accelerating.',
              },
            ],
          }}
        />
      </MemoryRouter>,
    )

    expect(screen.getByText('Total Accounts')).toBeInTheDocument()
    expect(screen.getByText('3')).toBeInTheDocument()
    expect(screen.getAllByText('Freshdesk')).toHaveLength(2)
    expect(screen.getByText('Early account signal only.')).toBeInTheDocument()
    expect(screen.getByText('Confidence tier: low')).toBeInTheDocument()

    await user.click(screen.getByRole('button', { name: '1 witnesses' }))
    await user.click(screen.getAllByRole('button', { name: '[1]' })[0])

    expect(onOpenWitness).toHaveBeenNthCalledWith(1, 'witness-1', 'Zendesk')
    expect(onOpenWitness).toHaveBeenNthCalledWith(2, 'witness-1', 'Zendesk')
  })


  it('links witness fallbacks to the evidence explorer with the active vendor filter', () => {
    render(
      <MemoryRouter>
        <SpecializedReportData
          reportType="accounts_in_motion"
          vendorName="Zendesk"
          backTo="/report?vendor=Zendesk&ref=test-token&mode=view"
          asOfDate="2026-04-08"
          windowDays={45}
          data={{
            reference_ids: {
              witness_ids: ['witness-1'],
            },
            accounts: [
              {
                company: 'Acme Corp',
                opportunity_score: 72,
                top_quote: 'Pricing pressure is accelerating.',
              },
            ],
          }}
        />
      </MemoryRouter>,
    )

    expect(screen.getByRole('link', { name: '1 witnesses' })).toHaveAttribute(
      'href',
      '/evidence?vendor=Zendesk&tab=witnesses&as_of_date=2026-04-08&window_days=45&back_to=%2Freport%3Fvendor%3DZendesk%26ref%3Dtest-token%26mode%3Dview',
    )
  })

  it('renders category overview arrays directly', () => {
    render(
      <MemoryRouter>
        <SpecializedReportData
          reportType="category_overview"
          data={[
            {
              category: 'Helpdesk',
              dominant_pain: 'pricing',
              highest_churn_risk: 'Zendesk',
              emerging_challenger: 'Freshdesk',
              market_regime: 'replacement_heavy',
              market_shift_signal: 'Mid-market teams are actively comparing alternatives.',
            },
          ]}
        />
      </MemoryRouter>,
    )

    expect(screen.getAllByText('Helpdesk')).toHaveLength(2)
    expect(screen.getAllByText('Freshdesk')).toHaveLength(3)
    expect(screen.getByText('Mid-market teams are actively comparing alternatives.')).toBeInTheDocument()
  })
})
