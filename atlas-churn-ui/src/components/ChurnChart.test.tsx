import { cleanup, render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import ChurnChart from './ChurnChart'
import type { ChurnSignal } from '../types'

vi.mock('recharts', () => ({
  ResponsiveContainer: ({ children }: { children: ReactNode }) => (
    <div data-testid="responsive-container">{children}</div>
  ),
  BarChart: ({ data, children }: { data: unknown; children: ReactNode }) => (
    <div data-testid="bar-chart" data-chart={JSON.stringify(data)}>{children}</div>
  ),
  XAxis: () => <div data-testid="x-axis" />,
  YAxis: () => <div data-testid="y-axis" />,
  Tooltip: () => <div data-testid="tooltip" />,
  Bar: ({ children }: { children: ReactNode }) => <div data-testid="bar">{children}</div>,
  Cell: ({ fill }: { fill: string }) => <span data-testid="cell" data-fill={fill} />,
}))

const baseSignal = (overrides: Partial<ChurnSignal>): ChurnSignal => ({
  vendor_name: 'Zendesk',
  product_category: 'Helpdesk',
  total_reviews: 100,
  churn_intent_count: 20,
  avg_urgency_score: 7.2,
  avg_rating_normalized: 0.4,
  nps_proxy: 10,
  price_complaint_rate: 0.2,
  decision_maker_churn_rate: 0.1,
  support_sentiment: 0.3,
  legacy_support_score: 0.4,
  new_feature_velocity: 0.5,
  employee_growth_rate: 0.1,
  archetype: 'pricing_shock',
  archetype_confidence: 0.8,
  reasoning_mode: 'full',
  last_computed_at: '2026-04-10T00:00:00Z',
  ...overrides,
})

describe('ChurnChart', () => {
  beforeEach(() => {
    cleanup()
  })

  it('truncates vendor names, limits rows, and colors bars by urgency', () => {
    const signals: ChurnSignal[] = [
      baseSignal({ vendor_name: 'VeryLongVendorName123', avg_urgency_score: 8.4 }),
      baseSignal({ vendor_name: 'Intercom', avg_urgency_score: 6.1 }),
      baseSignal({ vendor_name: 'HubSpot', avg_urgency_score: 4.5 }),
      baseSignal({ vendor_name: 'Freshdesk', avg_urgency_score: 2.9 }),
    ]

    render(<ChurnChart signals={signals} maxItems={3} />)

    const chart = screen.getByTestId('bar-chart')
    const data = JSON.parse(chart.getAttribute('data-chart') ?? '[]')

    expect(data).toEqual([
      { name: 'VeryLongVendor...', urgency: 8.4, reviews: 100 },
      { name: 'Intercom', urgency: 6.1, reviews: 100 },
      { name: 'HubSpot', urgency: 4.5, reviews: 100 },
    ])

    const fills = screen.getAllByTestId('cell').map((cell) => cell.getAttribute('data-fill'))
    expect(fills).toEqual(['#ef4444', '#f59e0b', '#eab308'])
  })

  it('uses the low-urgency color for calmer vendors', () => {
    render(
      <ChurnChart
        signals={[baseSignal({ vendor_name: 'Basecamp', avg_urgency_score: 3.2 })]}
      />,
    )

    expect(screen.getByTestId('cell')).toHaveAttribute('data-fill', '#22c55e')
  })
})
