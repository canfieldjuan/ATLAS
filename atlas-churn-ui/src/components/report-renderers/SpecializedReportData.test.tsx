import { render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { MemoryRouter } from 'react-router-dom'
import { describe, expect, it, vi } from 'vitest'
import { SpecializedReportData } from './SpecializedReportData'

describe('SpecializedReportData', () => {
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

    await user.click(screen.getByRole('button', { name: '1 witnesses' }))
    await user.click(screen.getAllByRole('button', { name: '[1]' })[0])

    expect(onOpenWitness).toHaveBeenNthCalledWith(1, 'witness-1', 'Zendesk')
    expect(onOpenWitness).toHaveBeenNthCalledWith(2, 'witness-1', 'Zendesk')
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
