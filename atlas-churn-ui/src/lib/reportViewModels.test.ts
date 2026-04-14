import { describe, expect, it } from 'vitest'

import { toAccountsInMotionViewModel, toBattleCardViewModel, toWeeklyChurnFeedItems } from './reportViewModels'

describe('toBattleCardViewModel', () => {
  it('falls back to reasoning_witness_highlights for older battle-card rows', () => {
    const viewModel = toBattleCardViewModel({
      vendor: 'Shopify',
      reference_ids: {
        witness_ids: ['witness:shopify:1'],
        metric_ids: ['metric:1'],
      },
      reasoning_witness_highlights: [
        {
          witness_id: 'witness:shopify:1',
          reviewer_company: 'Acme Co',
          excerpt_text: 'Pricing got out of hand at renewal.',
        },
      ],
    })

    expect(viewModel.reasoning_reference_ids?.witness_ids).toEqual(['witness:shopify:1'])
    expect(viewModel.reasoning_witness_highlights?.[0]?.witness_id).toBe('witness:shopify:1')
    expect(viewModel.reasoning_witness_highlights?.[0]?.reviewer_company).toBe('Acme Co')
  })

  it('preserves account-pressure quality fields', () => {
    const viewModel = toBattleCardViewModel({
      vendor: 'Shopify',
      account_pressure_summary: 'A small set of named accounts is showing early churn pressure.',
      account_pressure_disclaimer: 'Early account signal only.',
      account_actionability_tier: 'low',
    })

    expect(viewModel.account_pressure_summary).toBe(
      'A small set of named accounts is showing early churn pressure.',
    )
    expect(viewModel.account_pressure_disclaimer).toBe('Early account signal only.')
    expect(viewModel.account_actionability_tier).toBe('low')
  })
})

describe('reportViewModels account-pressure quality fields', () => {
  it('preserves weekly churn feed account-pressure quality fields', () => {
    const items = toWeeklyChurnFeedItems([
      {
        vendor: 'Zendesk',
        account_pressure_summary: 'Two accounts are under pressure.',
        account_pressure_disclaimer: 'Mixed confidence.',
        account_actionability_tier: 'mixed',
      },
    ])

    expect(items[0]?.account_pressure_disclaimer).toBe('Mixed confidence.')
    expect(items[0]?.account_actionability_tier).toBe('mixed')
  })

  it('preserves accounts-in-motion account-pressure quality fields', () => {
    const viewModel = toAccountsInMotionViewModel({
      account_pressure_summary: 'A single named account is showing early evaluation pressure.',
      account_pressure_disclaimer: 'Early account signal only.',
      account_actionability_tier: 'low',
      accounts: [],
      pricing_pressure: {},
      feature_gaps: [],
      cross_vendor_context: {},
    })

    expect(viewModel.account_pressure_disclaimer).toBe('Early account signal only.')
    expect(viewModel.account_actionability_tier).toBe('low')
  })
})
