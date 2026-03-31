import { render, screen } from '@testing-library/react'
import type { ReactNode } from 'react'
import { beforeAll, describe, expect, it, vi } from 'vitest'
import CampaignQualityTrends from './CampaignQualityTrends'

vi.mock('recharts', async () => {
  const actual = await vi.importActual<typeof import('recharts')>('recharts')
  return {
    ...actual,
    ResponsiveContainer: ({ children }: { children: ReactNode }) => (
      <div style={{ width: 640, height: 320 }}>{children}</div>
    ),
  }
})

describe('CampaignQualityTrends', () => {
  beforeAll(() => {
    class ResizeObserverMock {
      observe() {}
      unobserve() {}
      disconnect() {}
    }
    globalThis.ResizeObserver = ResizeObserverMock as typeof ResizeObserver
    Object.defineProperty(HTMLElement.prototype, 'offsetWidth', {
      configurable: true,
      value: 640,
    })
    Object.defineProperty(HTMLElement.prototype, 'offsetHeight', {
      configurable: true,
      value: 320,
    })
    Object.defineProperty(HTMLElement.prototype, 'clientWidth', {
      configurable: true,
      value: 640,
    })
    Object.defineProperty(HTMLElement.prototype, 'clientHeight', {
      configurable: true,
      value: 320,
    })
  })

  it('renders blocker movement and daily totals', () => {
    render(
      <CampaignQualityTrends
        data={{
          days: 14,
          top_n: 5,
          top_blockers: [
            { reason: 'missing_exact_proof_term', count: 3 },
            { reason: 'report_tier_language:dashboard', count: 1 },
          ],
          series: [
            { day: '2026-03-28', reason: 'missing_exact_proof_term', count: 1 },
            { day: '2026-03-29', reason: 'missing_exact_proof_term', count: 2 },
            { day: '2026-03-29', reason: 'report_tier_language:dashboard', count: 1 },
          ],
          totals_by_day: [
            { day: '2026-03-28', blocker_total: 1 },
            { day: '2026-03-29', blocker_total: 3 },
          ],
        }}
      />,
    )

    expect(screen.getByText('Campaign Quality Trends')).toBeInTheDocument()
    expect(screen.getByText('Last 14 days of blocker volume and reason movement')).toBeInTheDocument()
    expect(screen.getByText('missing_exact_proof_term')).toBeInTheDocument()
    expect(screen.getAllByText('up 1')).toHaveLength(2)
    expect(screen.getByText('latest day: 2')).toBeInTheDocument()
  })

  it('renders an empty state when no trend data exists', () => {
    render(
      <CampaignQualityTrends
        data={{
          days: 14,
          top_n: 5,
          top_blockers: [],
          series: [],
          totals_by_day: [],
        }}
      />,
    )

    expect(screen.getByText('No blocker trend data yet.')).toBeInTheDocument()
  })
})
