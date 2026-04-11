import { cleanup, render, screen } from '@testing-library/react'
import userEvent from '@testing-library/user-event'
import { beforeEach, describe, expect, it, vi } from 'vitest'
import SignalEffectivenessPanel from './SignalEffectivenessPanel'

const useApiData = vi.hoisted(() => vi.fn())

vi.mock('../hooks/useApiData', () => ({
  default: useApiData,
}))

type HookResult = {
  data: unknown
  loading: boolean
  error: Error | null
  refresh: () => void
  refreshing: boolean
}

let effectivenessByGroup: Record<string, HookResult>
let distributionResult: HookResult

describe('SignalEffectivenessPanel', () => {
  beforeEach(() => {
    cleanup()
    vi.clearAllMocks()

    effectivenessByGroup = {
      buying_stage: {
        data: {
          group_by: 'buying_stage',
          vendor_filter: null,
          min_sequences: 3,
          total_groups: 2,
          groups: [
            {
              signal_group: 'evaluation',
              total_sequences: 6,
              meetings: 3,
              deals_opened: 2,
              deals_won: 1,
              deals_lost: 0,
              no_opportunity: 1,
              disqualified: 0,
              positive_outcome_rate: 0.5,
              total_revenue: 50000,
            },
            {
              signal_group: 'consideration',
              total_sequences: 4,
              meetings: 1,
              deals_opened: 1,
              deals_won: 0,
              deals_lost: 1,
              no_opportunity: 1,
              disqualified: 0,
              positive_outcome_rate: 0.25,
              total_revenue: 0,
            },
          ],
        },
        loading: false,
        error: null,
        refresh: vi.fn(),
        refreshing: false,
      },
      role_type: {
        data: {
          group_by: 'role_type',
          vendor_filter: null,
          min_sequences: 3,
          total_groups: 1,
          groups: [
            {
              signal_group: 'executive_buyer',
              total_sequences: 5,
              meetings: 3,
              deals_opened: 2,
              deals_won: 1,
              deals_lost: 0,
              no_opportunity: 0,
              disqualified: 0,
              positive_outcome_rate: 0.6,
              total_revenue: 80000,
            },
          ],
        },
        loading: false,
        error: null,
        refresh: vi.fn(),
        refreshing: false,
      },
    }

    distributionResult = {
      data: {
        total_sequences: 7,
        vendor_filter: null,
        buckets: [
          { outcome: 'meeting_booked', count: 4, pct: 57.1, total_revenue: 50000, first_recorded: null, last_recorded: null },
          { outcome: 'deal_opened', count: 2, pct: 28.6, total_revenue: 80000, first_recorded: null, last_recorded: null },
          { outcome: 'pending', count: 1, pct: 14.3, total_revenue: 0, first_recorded: null, last_recorded: null },
        ],
      },
      loading: false,
      error: null,
      refresh: vi.fn(),
      refreshing: false,
    }

    useApiData.mockImplementation((_fetcher: unknown, deps: unknown[] = []) => {
      if (deps.length === 1) {
        return effectivenessByGroup[String(deps[0])] ?? effectivenessByGroup.buying_stage
      }
      return distributionResult
    })
  })

  it('starts collapsed and shows the tracked sequence count in the header', () => {
    render(<SignalEffectivenessPanel />)

    expect(screen.getByRole('button', { name: /Signal Effectiveness/i })).toBeInTheDocument()
    expect(screen.getByText('(7 sequences tracked)')).toBeInTheDocument()
    expect(screen.queryByText('Outcome Funnel')).not.toBeInTheDocument()
  })

  it('shows the empty-state guidance when no outcome data exists', async () => {
    const user = userEvent.setup()
    distributionResult = {
      data: {
        total_sequences: 0,
        vendor_filter: null,
        buckets: [],
      },
      loading: false,
      error: null,
      refresh: vi.fn(),
      refreshing: false,
    }
    effectivenessByGroup.buying_stage = {
      data: {
        group_by: 'buying_stage',
        vendor_filter: null,
        min_sequences: 3,
        total_groups: 0,
        groups: [],
      },
      loading: false,
      error: null,
      refresh: vi.fn(),
      refreshing: false,
    }

    render(<SignalEffectivenessPanel />)

    await user.click(screen.getByRole('button', { name: /Signal Effectiveness/i }))

    expect(
      screen.getByText('No outcome data yet. Record outcomes on sent campaigns to see signal effectiveness.'),
    ).toBeInTheDocument()
  })

  it('renders the funnel and switches grouping dimensions', async () => {
    const user = userEvent.setup()

    render(<SignalEffectivenessPanel />)

    await user.click(screen.getByRole('button', { name: /Signal Effectiveness/i }))

    expect(screen.getByText('Outcome Funnel')).toBeInTheDocument()
    expect(screen.getByText('meeting booked')).toBeInTheDocument()
    expect(screen.getAllByText('$50,000')).toHaveLength(2)
    expect(screen.getByText('evaluation')).toBeInTheDocument()
    expect(screen.getByText('50%')).toBeInTheDocument()

    await user.selectOptions(screen.getByRole('combobox'), 'role_type')

    expect(screen.queryByText('evaluation')).not.toBeInTheDocument()
    expect(screen.getByText('executive buyer')).toBeInTheDocument()
    expect(screen.getByText('60%')).toBeInTheDocument()
    expect(screen.getAllByText('$80,000')).toHaveLength(2)
  })
})
