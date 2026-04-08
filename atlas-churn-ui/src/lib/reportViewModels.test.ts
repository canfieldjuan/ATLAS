import { describe, expect, it } from 'vitest'

import { toBattleCardViewModel } from './reportViewModels'

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
})
